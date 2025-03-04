import os
import cv2
import numpy as np
import pickle
import pydicom
from numba import njit
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

IMG_SIZE = 64

@njit
def tile_bias(b, m):
    n = b.shape[0]
    out = np.empty((n, m), dtype=b.dtype)
    for i in range(m):
        for j in range(n):
            out[j, i] = b[j, 0]
    return out

@njit(parallel=True, fastmath=True)
def fwd_prop_numba(W1, b1, W2, b2, W3, b3, X):
    m = X.shape[1]
    Z1 = np.dot(W1, X) + tile_bias(b1, m)
    A1 = np.maximum(Z1, np.float32(0.0))
    Z2 = np.dot(W2, A1) + tile_bias(b2, m)
    A2 = np.maximum(Z2, np.float32(0.0))
    Z3 = np.dot(W3, A2) + tile_bias(b3, m)
    np.clip(Z3, -np.float32(20.0), np.float32(20.0), out=Z3)
    A3 = np.float32(1.0) / (np.float32(1.0) + np.exp(-Z3) + np.float32(1e-7))
    return A3, Z1, A1, Z2, A2, Z3

def load_parameters():
    try:
        with open("parameters.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Error: parameters.pkl not found. Make sure the file exists and the path is correct.")
        return None
    except Exception as e:
        print(f"Error loading parameters.pkl: {e}")
        return None

def predict_image(image_path, parameters):
    try:
        dcm = pydicom.dcmread(image_path)
        img = dcm.pixel_array
        img = img.astype(np.uint8)
    except Exception as e:
        print(f"Could not load image: {image_path}")
        print(f"Error: {e}")
        return None
    
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized.astype(np.float32) / 255.0
    X = img_normalized.reshape(-1, 1)
    
    A3, _, _, _, _, _ = fwd_prop_numba(
        parameters["W1"], parameters["b1"],
        parameters["W2"], parameters["b2"],
        parameters["W3"], parameters["b3"],
        X
    )
    
    return float(A3[0])

def upload_image():
    global img_path
    img_path = filedialog.askopenfilename(filetypes=[("DICOM files", "*.dcm")])  # Specify DICOM file type
    if img_path:
        try:
            dcm = pydicom.dcmread(img_path)
            img = dcm.pixel_array
            img = img.astype(np.uint8)

            # Convert to PIL Image
            img = Image.fromarray(img)
            img = img.resize((256, 256))  # Resize for display
            photo = ImageTk.PhotoImage(img)  # Create PhotoImage object

            # Configure image_label to display the image
            image_label.config(image=photo)
            image_label.image = photo  # Keep a reference!
            result_label.config(text="")  # Clear previous result

        except Exception as e:
            print(f"Error loading DICOM image: {e}")
            result_label.config(text=f"Error loading DICOM image: {e}")

def predict():
    if not hasattr(globals(), 'img_path'):
        result_label.config(text="Please upload an image first.")
        return

    parameters = load_parameters()
    if parameters is None:
        result_label.config(text="Failed to load model parameters.")
        return

    try:
        probability = predict_image(img_path, parameters)
        if probability is not None:
            if probability > 0.5:
                result_label.config(text=f"Pneumonia detected (Probability: {probability:.4f})")
            else:
                result_label.config(text=f"No pneumonia detected (Probability: {probability:.4f})")
        else:
            result_label.config(text="Prediction failed.")
    except Exception as e:
        result_label.config(text=f"An error occurred: {e}")

# --- UI setup ---
root = tk.Tk()
root.title("Pneumonia Detection")

upload_button = tk.Button(root, text="Upload DICOM Image", command=upload_image)
upload_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()