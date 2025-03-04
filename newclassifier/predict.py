import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('rock_paper_scissors_model.h5')

# Compile the model (important to do this after loading)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Function to prepare the image for prediction
def prepare(frame):
    IMG_SIZE = 200  # Ensure this matches the training IMG_SIZE
    img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # Resize image to IMG_SIZE x IMG_SIZE
    img_array = np.array(img_array).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function for live prediction
def live_predict():
    # Start webcam capture
    cap = cv2.VideoCapture(0) # Try different indices if 0 doesn't work
    # cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION) # macOS specific

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Prepare the frame for prediction
        img = prepare(frame)
        
        # Get prediction
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)  # Get the index of the class with the highest probability
        
        # Define the classes
        classes = ['rock', 'paper', 'scissors']
        
        # Display the prediction on the frame
        label = f"Prediction: {classes[predicted_class]}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the frame with the prediction
        cv2.imshow('Rock-Paper-Scissors Prediction', frame)
        
        # Break the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Run the live prediction
live_predict()