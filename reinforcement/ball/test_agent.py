import gym
from stable_baselines3 import DQN
from custom_env import CatchBallEnv

# Load the environment
env = CatchBallEnv(render_mode="human")

# Load trained model
model = DQN.load("dqn_catch_ball")

# Test the trained agent
for episode in range(1000):
    state, _ = env.reset()
    done = False

    while not done:
        env.render()
        action, _ = model.predict(state)
        state, reward, done, _, _ = env.step(action)

env.close()
