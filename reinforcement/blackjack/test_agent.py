import time
from blackjackenv import BlackjackEnv
from discretize_action_wrapper import DiscretizeActionWrapper
from stable_baselines3 import DQN

env = BlackjackEnv(render_mode="human", training=False)
env = DiscretizeActionWrapper(env)
model = DQN.load("dqn_blackjack")

for episode in range(10):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    print(f"\n=== Episode {episode+1} ===")
    print(f"Starting Bankroll: ${env.unwrapped.bankroll}")
    
    while not done:
        env.render()
        time.sleep(1)
        
        action_int, _ = model.predict(obs)
        # Decode for logging only; the wrapper also decodes internally for env.step
        move = action_int // 101
        bet = action_int % 101
        
        obs, reward, done, _, _ = env.step(action_int)
        total_reward += reward
        
        print(f"Action: {'Hit' if move == 1 else 'Stand'} | Bet: ${bet}")
        print(f"Reward: {reward:.1f} | New Balance: ${env.unwrapped.bankroll}")
    
    print(f"Total Reward: {total_reward:.1f}")

env.close()
