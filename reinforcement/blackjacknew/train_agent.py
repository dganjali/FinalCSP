from stable_baselines3 import DQN
from blackjack_env import BlackjackEnv
from discretize_action_wrapper import DiscretizeActionWrapper

def main():
    env = BlackjackEnv(render_mode=None)
    env = DiscretizeActionWrapper(env)
    
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        buffer_size=100000,
        learning_starts=10000,
        batch_size=128,
        gamma=0.95,
        exploration_final_eps=0.02,
        target_update_interval=1000,
        verbose=1
    )
    
    model.learn(total_timesteps=500000)
    model.save("blackjack_dqn")
    env.close()

if __name__ == "__main__":
    main()
