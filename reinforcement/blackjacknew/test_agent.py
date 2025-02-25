import gym
from gym.wrappers import FlattenObservation
from stable_baselines3 import DQN
import time

# Load the environment with the same wrapper as used in training
env = gym.make("Blackjack-v1", render_mode="human")
env = FlattenObservation(env)

# Load trained model
model = DQN.load("dqn_blackjack")

# Test the trained agent
for episode in range(10):  # Reduced from 1000 for demonstration
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    print(f"\n=== Episode {episode+1} ===")
    
    while not done:
        env.render()
        time.sleep(1)  # Add delay to make rendering visible
        
        action, _ = model.predict(state)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        print(f"Action: {'Hit' if action == 1 else 'Stand'}, Reward: {reward}")
    
    print(f"Episode {episode+1} finished with total reward: {total_reward}")

env.close()