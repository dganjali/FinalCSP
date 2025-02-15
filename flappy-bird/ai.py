import numpy as np
from tensorflow import keras
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class FlappyBirdAI:
    def __init__(self):
        self.model = keras.Sequential([
            keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(50, 50, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        
    def predict(self, game_state):
        state = self.preprocess_state(game_state)
        return float(self.model.predict(state)[0, 0]) > 0.5
        
    def preprocess_state(self, game_state):
        grid = np.zeros((50, 50, 1))
        bird_y = int((game_state['birdY'] / game_state['windowHeight']) * 50)
        
        # Place bird
        for i in range(-1, 2):
            for j in range(-1, 2):
                if 0 <= bird_y + i < 50:
                    if 0 <= 15 + j < 50:
                        grid[bird_y + i, 15 + j, 0] = 1
                        
        # Place pipe if exists
        if game_state['nearestPipe']:
            pipe_x = int((game_state['nearestPipe']['left'] / game_state['windowWidth']) * 50)
            pipe_top = int((game_state['nearestPipe']['top'] / game_state['windowHeight']) * 50)
            
            if 0 <= pipe_x < 50:
                grid[:pipe_top, pipe_x, 0] = 1
                grid[pipe_top+10:, pipe_x, 0] = 1
                
        return np.expand_dims(grid, axis=0)

ai = FlappyBirdAI()

@app.route('/predict', methods=['POST'])
def predict():
    game_state = request.json
    should_jump = ai.predict(game_state)
    return jsonify({'should_jump': bool(should_jump)})

@app.route('/train', methods=['POST'])
def train():
    training_data = request.json
    states = [d['state'] for d in training_data]
    actions = np.array([d['action'] for d in training_data])
    
    processed_states = np.stack([ai.preprocess_state(state) for state in states])
    ai.model.fit(processed_states, actions, epochs=10, batch_size=32)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(port=5000)