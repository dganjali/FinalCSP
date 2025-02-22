import gym
from stable_baselines3 import DQN
from custom_env import CatchBallEnv

# Create the environment
env = CatchBallEnv(render_mode=None)

# Define the model
model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001, buffer_size=10000, batch_size=32, gamma=0.99)

# Train the model
model.learn(total_timesteps=50000)

# Save the trained model
model.save("dqn_catch_ball")

env.close()