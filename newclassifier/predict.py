'''
This file is used to make live predictions using the trained model. 
Using the openCV library, the webcam feed is captured and the frames are passed to the model for prediction. 
The prediction is displayed at the top of the frame in real-time.
The parameters are loaded as a keras model from the file rps_model.keras.
'''
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# load previously trained model
model = load_model('rps_model.keras')

# compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# prepare a frame for prediction
def prepare(frame):
    IMG_SIZE = 150
    img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_array).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# live prediction function using opencv to capture webcam feed
def live_predict():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # call prepare function to prepare the frame for prediction
        img = prepare(frame)

        # actually make that prediction
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        class_names = ['rock', 'paper', 'scissors']
        label = class_names[predicted_class]

        # display the frame with the prediction
        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Webcam', frame)

        # break out of program if key q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

live_predict()