import gym

# Create the environment with render_mode
env = gym.make("CartPole-v1", render_mode="human")

# Reset the environment to start
state, _ = env.reset()

# Run a test episode
for _ in range(1000):
    env.render()  # Display the environment
    action = env.action_space.sample()  # Take a random action
    state, reward, done, truncated, info = env.step(action)  # Apply action
    if done or truncated:
        state, _ = env.reset()  # Reset if the episode ends

env.close()
