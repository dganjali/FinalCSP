import gymnasium as gym
from stable_baselines3 import DQN
from blackjackenv import BlackjackEnv
from discretize_action_wrapper import DiscretizeActionWrapper

def main():
    env = BlackjackEnv(render_mode=None, training=True)
    env = DiscretizeActionWrapper(env)
    
    model = DQN(
    "MlpPolicy",
    env,
    learning_rate=5e-4,
    buffer_size=100000,
    learning_starts=20000,
    batch_size=128,
    target_update_interval=500,
    exploration_final_eps=0.02,
    gamma=0.99,  # Long-term strategy focus
    verbose=1
    )

    
    model.learn(total_timesteps=200000)
    model.save("dqn_blackjack")
    env.close()

if __name__ == "__main__":
    main()
