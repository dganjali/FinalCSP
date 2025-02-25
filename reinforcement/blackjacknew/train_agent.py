import gym
from gym.wrappers import FlattenObservation
from stable_baselines3 import DQN

# Create the environment and flatten the observation space
env = gym.make("Blackjack-v1", render_mode=None)
env = FlattenObservation(env)

# Define the model
model = DQN("MlpPolicy", 
           env, 
           verbose=1, 
           learning_rate=0.001, 
           buffer_size=10000, 
           batch_size=32, 
           gamma=0.99)

# Train the model
model.learn(total_timesteps=50000)

# Save the trained model
model.save("dqn_blackjack")

env.close()