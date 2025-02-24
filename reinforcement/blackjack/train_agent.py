import gym
from stable_baselines3 import DQN
from blackjackenv import BlackjackEnv

# Create the environment
env = BlackjackEnv(render_mode=None)

# Define the model
model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001, buffer_size=10000, batch_size=32, gamma=0.99)

# Train the model and print info
total_timesteps = 50000
for i in range(1, 6):
    print(f"Starting training run {i}")
    model.learn(total_timesteps=total_timesteps // 5)
    print(f"Completed training run {i}")

# Save the trained model
model.save("dqn_blackjack")

env.close()
