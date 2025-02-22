from custom_env import CatchBallEnv

env = CatchBallEnv(render_mode="human")  # Enable rendering
state, _ = env.reset()

for _ in range(20):  # Run for 20 episodes
    done = False
    state, _ = env.reset()

    while not done:
        env.render()
        action = env.action_space.sample()  # Random action
        state, reward, done, _, _ = env.step(action)
        print(f"State: {state}, Reward: {reward}")

env.close()
