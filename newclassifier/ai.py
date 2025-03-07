'''
Python script to train a convolutional neural network to classify images of rock, paper, and scissors.
The dataset is taken from kaggle: https://www.kaggle.com/datasets/alexandredj/rock-paper-scissors-dataset.
The images are preprocessed and augmented using the ImageDataGenerator class from keras.
The model is trained using the fit method and the training history is plotted.
The model is evaluated on the validation set and saved to a file (rps_model.keras) for later testing.
Training ended with a validation accuracy of 0.98 after 30 epochs.
'''
# important libraries 
# os allows us to interact with the operating system in order to load the dataset.
# opencv lets us process the images.
# numpy is used for numerical operations on the images.
# matplotlib is used to plot the training metrics to give a visual representation of the model's performance.
# tensorflow is the deep learning framework used to build and train the convolutional neural network.
# scikit provides the train_test_split function to split the dataset into training and validation sets.
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# snippet to configure tensorflow to use a physical GPU if available
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU detected. Running on CPU.")
except Exception as e:
    print(f"An error occurred while configuring GPU: {e}")

# paths to the dataset
train_dir = "archive"

# parameters used for image processing
IMG_SIZE = 150
BATCH_SIZE = 32
NUM_CLASSES = 3

# load images as well as labels into an array
data = []
labels = []
categories = ['rock', 'paper', 'scissors']

# load images and labels
print("Loading images...")
for category_index, category in enumerate(categories):
    category_dir = os.path.join(train_dir, category)
    for img_name in os.listdir(category_dir):
        img_path = os.path.join(category_dir, img_name)
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0
            data.append(img)
            labels.append(category_index)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

data = np.array(data)
labels = np.array(labels)

# split data into different sets (training and validation)
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# augment the data
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# create the actual model
model = Sequential([
    InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                    epochs=30,
                    validation_data=(X_val, y_val))

# evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {accuracy}")

# plot the metris of the model
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# save the model for future testing and live prediction
model.save("rps_model.keras")