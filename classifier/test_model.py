import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Define image size (should match training)
IMG_SIZE = 64

# Activation functions
def relu(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

# Forward propagation (using the same network architecture as in training)
def fwd_prop(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    return A3

# Load trained parameters
with open("parameters.pkl", "rb") as f:
    parameters = pickle.load(f)

# Define paths to the test data (here we reuse the pneumonia CSV & training images for testing)
base_dir = os.path.join("rsna-pneumonia-detection-challenge")
images_dir = os.path.join(base_dir, "stage_2_train_images")
labels_csv = os.path.join(base_dir, "stage_2_train_labels.csv")

# Read CSV of labels. (Assuming columns 'patientId' and 'Target' exist.)
df = pd.read_csv(labels_csv)

# Construct image filenames. (Assume images are stored as PNG files.)
df["image_id"] = df["patientId"].astype(str) + ".png"

# Build test set lists
X_list = []
Y_list = []

print("Building test set...")
for idx, row in df.iterrows():
    label = row["Target"]
    image_file = row["image_id"]
    img_path = os.path.join(images_dir, image_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    X_list.append(img.flatten())
    Y_list.append(label)

X_test = np.array(X_list).T  # Shape: (IMG_SIZE*IMG_SIZE, num_samples)
Y_test = np.array(Y_list).reshape(1, -1)  # Shape: (1, num_samples)

print(f"Test set size: {X_test.shape[1]} images")

# Run forward propagation to get predictions
A3 = fwd_prop(X_test, parameters)
predictions = (A3 > 0.5).astype(int)
accuracy = np.mean(predictions == Y_test)
print("Test Accuracy: {:.2f}%".format(accuracy * 100))

# Optionally, display a few examples and their predictions.
for i in range(min(5, X_test.shape[1])):
    plt.imshow(X_test[:, i].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.text(1, 2, "Pred: " + str(predictions[0, i]) + " | Actual: " + str(Y_test[0, i]),
             fontsize=15, color='white')
    plt.title("Example " + str(i))
    plt.axis("off")
    plt.show()