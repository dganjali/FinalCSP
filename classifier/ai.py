import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import pydicom
from numba import njit, prange

# Set random seed for reproducibility
np.random.seed(19)

# Paths to the dataset
base_dir = "rsna-pneumonia-detection-challenge"
images_dir = os.path.join(base_dir, "stage_2_train_images")
labels_csv = os.path.join(base_dir, "stage_2_train_labels.csv")

# Read CSV
df = pd.read_csv(labels_csv)

# Debug: print available columns
print("CSV columns:", df.columns.tolist())

# Use the appropriate column for image file names.
# Since CSV columns are: ['patientId', 'x', 'y', 'width', 'height', 'Target'],
# we'll use 'patientId' as the image identifier.
img_col = "patientId"

if img_col not in df.columns:
    raise KeyError(f"Column '{img_col}' not found in CSV. Available columns: {df.columns.tolist()}")

# Append file extension for DICOM images (assuming they all have '.dcm' extension)
df[img_col] = df[img_col].astype(str) + ".dcm"

# Debug: print first few rows to verify
print(df.head())

# Parameters for image processing
IMG_SIZE = 64  # resize images to 64x64
input_dim = IMG_SIZE * IMG_SIZE

# Load images and labels into arrays
data = []
labels = []
print("Loading images...")
for idx, row in df.iterrows():
    img_path = os.path.join(images_dir, row[img_col])
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        continue
    try:
        dcm = pydicom.dcmread(img_path)
        img = dcm.pixel_array
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        continue
    # Convert to uint8 if necessary (scaling may be required depending on DICOM)
    img = img.astype(np.uint8)
    # Resize image (cv2.resize expects a 2D array for grayscale)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # Normalize pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0
    data.append(img.flatten())
    labels.append(row["Target"])

if len(data) == 0:
    raise ValueError("No images were loaded. Check your dataset paths and DICOM files.")

# Convert list to numpy arrays; X shape: (input_dim, m)
X = np.array(data).T  # each column is one flattened image
Y = np.array(labels).reshape(1, -1)  # shape: (1, m)
m = X.shape[1]
print(f"Loaded {m} images.")

# Define network architecture (for binary classification)
layer_dims = [input_dim, 512, 128, 1]

# He initialization for weights and zeros for biases
W1 = np.random.randn(layer_dims[1], layer_dims[0]) * np.sqrt(2.0 / layer_dims[0])
W2 = np.random.randn(layer_dims[2], layer_dims[1]) * np.sqrt(2.0 / layer_dims[1])
W3 = np.random.randn(layer_dims[3], layer_dims[2]) * np.sqrt(2.0 / layer_dims[2])
b1 = np.zeros((layer_dims[1], 1))
b2 = np.zeros((layer_dims[2], 1))
b3 = np.zeros((layer_dims[3], 1))

parameters = {"W1": W1,
              "W2": W2,
              "W3": W3,
              "b1": b1,
              "b2": b2,
              "b3": b3}

# Ensure arrays and parameters are float32
X = X.astype(np.float32)
Y = Y.astype(np.float32)
for key in parameters:
    parameters[key] = parameters[key].astype(np.float32)

# Use Numba parallelization in our forward and backward functions.

@njit
def tile_bias(b, m):
    # b shape: (n, 1) -> output: (n, m)
    n = b.shape[0]
    out = np.empty((n, m), dtype=b.dtype)
    for i in range(m):
        for j in range(n):
            out[j, i] = b[j, 0]
    return out

@njit(parallel=True, fastmath=True)
def fwd_prop_numba(W1, b1, W2, b2, W3, b3, X):
    m = X.shape[1]
    # Use custom tiling for biases
    Z1 = np.dot(W1, X) + tile_bias(b1, m)
    A1 = np.maximum(Z1, np.float32(0.0))
    Z2 = np.dot(W2, A1) + tile_bias(b2, m)
    A2 = np.maximum(Z2, np.float32(0.0))
    Z3 = np.dot(W3, A2) + tile_bias(b3, m)
    np.clip(Z3, -np.float32(50.0), np.float32(50.0), out=Z3)
    A3 = np.float32(1.0) / (np.float32(1.0) + np.exp(-Z3))
    return A3, Z1, A1, Z2, A2, Z3

@njit(parallel=True, fastmath=True)
def backward_prop_numba(W1, W2, W3, X, Y, Z1, A1, Z2, A2, A3):
    m = X.shape[1]
    dZ3 = A3 - Y
    dW3 = (np.float32(1.0)/m) * np.dot(dZ3, A2.T)
    db3 = (np.float32(1.0)/m) * np.sum(dZ3, axis=1).reshape(dZ3.shape[0], 1)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = dA2 * (Z2 > np.float32(0.0))
    dW2 = (np.float32(1.0)/m) * np.dot(dZ2, A1.T)
    db2 = (np.float32(1.0)/m) * np.sum(dZ2, axis=1).reshape(dZ2.shape[0], 1)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * (Z1 > np.float32(0.0))
    dW1 = (np.float32(1.0)/m) * np.dot(dZ1, X.T)
    db1 = (np.float32(1.0)/m) * np.sum(dZ1, axis=1).reshape(dZ1.shape[0], 1)
    
    return dW1, db1, dW2, db2, dW3, db3

def update_parameters(parameters, grads, learning_rate):
    parameters["W1"] -= learning_rate * grads[0]
    parameters["b1"] -= learning_rate * grads[1]
    parameters["W2"] -= learning_rate * grads[2]
    parameters["b2"] -= learning_rate * grads[3]
    parameters["W3"] -= learning_rate * grads[4]
    parameters["b3"] -= learning_rate * grads[5]
    return parameters

# Training loop with mini-batching using improved, pre-compiled functions
num_epochs = 400
learning_rate = 0.01
batch_size = 64
m = X.shape[1]
num_batches = m // batch_size

for epoch in range(num_epochs):
    permutation = np.random.permutation(m)
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation]
  
    epoch_loss = 0.0
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        X_batch = X_shuffled[:, start:end]
        Y_batch = Y_shuffled[:, start:end]
  
        A3, Z1, A1, Z2, A2, Z3 = fwd_prop_numba(parameters["W1"], parameters["b1"],
                                                  parameters["W2"], parameters["b2"],
                                                  parameters["W3"], parameters["b3"],
                                                  X_batch)
  
        # Compute loss in vectorized form
        loss = -np.sum(Y_batch * np.log(A3 + 1e-8) + (1 - Y_batch) * np.log(1 - A3 + 1e-8)) / X_batch.shape[1]
        epoch_loss += loss
  
        grads = backward_prop_numba(parameters["W1"], parameters["W2"], parameters["W3"],
                                    X_batch, Y_batch, Z1, A1, Z2, A2, A3)
        parameters = update_parameters(parameters, grads, learning_rate)
  
    # Optionally display accuracy every few epochs
    if epoch % 100 == 0:
        A3_full, _, _, _, _, _ = fwd_prop_numba(parameters["W1"], parameters["b1"],
                                                parameters["W2"], parameters["b2"],
                                                parameters["W3"], parameters["b3"],
                                                X)
        predictions = (A3_full > 0.5).astype(np.float32)
        accuracy = np.mean(predictions == Y)
        print(f"Epoch {epoch} - Avg Loss: {epoch_loss/num_batches:.4f} - Accuracy: {accuracy:.4f}")
  
    # Save parameters after each epoch for checkpointing
    with open("parameters.pkl", "wb") as f:
        pickle.dump(parameters, f)

# Final predictions and display examples
A3, _ = fwd_prop_numba(parameters["W1"], parameters["b1"],
                       parameters["W2"], parameters["b2"],
                       parameters["W3"], parameters["b3"],
                       X)
predictions = (A3 > 0.5).astype(int)
print("Train set:")
print("Final Accuracy:", np.mean(predictions == Y))
print("Test set:")

for i in range(5):
    plt.imshow(X[:, i].reshape(IMG_SIZE, IMG_SIZE), cmap=plt.get_cmap('gray'))
    plt.text(1, 2, "Pred: " + str(predictions[0, i]) + "  Actual: " + str(Y[0, i]),
             fontsize=15, color='white')
    plt.show()

# Save trained parameters
with open("parameters.pkl", "wb") as f:
    pickle.dump(parameters, f)