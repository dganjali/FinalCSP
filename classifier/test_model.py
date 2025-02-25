import os
import cv2
import numpy as np
import pickle
from numba import njit
import pydicom  # Add this import at the top
IMG_SIZE = 64

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
    # Increase numerical stability with stronger clipping
    np.clip(Z3, -np.float32(20.0), np.float32(20.0), out=Z3)
    # Add epsilon to prevent division by zero
    A3 = np.float32(1.0) / (np.float32(1.0) + np.exp(-Z3) + np.float32(1e-7))
    return A3, Z1, A1, Z2, A2, Z3

def load_parameters():
    with open("parameters.pkl", "rb") as f:
        return pickle.load(f)

def predict_image(image_path, parameters):
    # Load DICOM image using pydicom instead of cv2
    try:
        dcm = pydicom.dcmread(image_path)
        img = dcm.pixel_array
        img = img.astype(np.uint8)  # Convert to uint8 if necessary
    except Exception as e:
        print(f"Could not load image: {image_path}")
        print(f"Error: {e}")
        return None
    
    # Resize and normalize
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Reshape for network input
    X = img_normalized.reshape(-1, 1)
    
    # Forward propagation
    A3, _, _, _, _, _ = fwd_prop_numba(
        parameters["W1"], parameters["b1"],
        parameters["W2"], parameters["b2"],
        parameters["W3"], parameters["b3"],
        X
    )
    
    return float(A3[0])

def main():
    # Load model parameters
    parameters = load_parameters()
    if parameters is None:
        print("Could not load model parameters")
        return
    
    # Test directory path
    test_dir = "rsna-pneumonia-detection-challenge/stage_2_test_images"
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return
    
    # Process each image in test directory
    for filename in os.listdir(test_dir):
        if filename.endswith(('.dcm')):
            image_path = os.path.join(test_dir, filename)
            probability = predict_image(image_path, parameters)
            
            if probability is not None:
                prediction = "Pneumonia" if probability > 0.5 else "Normal"
                print(f"Image: {filename}")
                print(f"Prediction: {prediction} (probability: {probability:.2%})")
                print("-" * 50)

if __name__ == "__main__":
    print("Starting test...")
    main()