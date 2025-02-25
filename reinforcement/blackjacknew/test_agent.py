import time
from stable_baselines3 import DQN
from blackjack_env import BlackjackEnv
from discretize_action_wrapper import DiscretizeActionWrapper

env = BlackjackEnv(render_mode="human")
env = DiscretizeActionWrapper(env)
model = DQN.load("blackjack_dqn")

for episode in range(3):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    print(f"\nEpisode {episode+1}")
    print(f"Starting Bankroll: ${env.unwrapped.bankroll}")
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        env.render()
        
        # Print detailed game information
        move = "Hit" if action % 2 == 1 else "Stand"
        bet = (action // 2 + 1) * 5
        print(f"Action: {move}, Bet: ${bet}")
        
        if move == "Stand" or terminated:
            print(f"Dealer's hand: {env.unwrapped.dealer_hand}")
            print(f"Dealer's total: {env.unwrapped._hand_value(env.unwrapped.dealer_hand)}")
            
            if env.unwrapped._hand_value(env.unwrapped.player_hand) > 21:
                print("Player busts!")
            elif env.unwrapped._hand_value(env.unwrapped.dealer_hand) > 21:
                print("Dealer busts! Player wins!")
            elif env.unwrapped._hand_value(env.unwrapped.player_hand) > env.unwrapped._hand_value(env.unwrapped.dealer_hand):
                print("Player wins!")
            elif env.unwrapped._hand_value(env.unwrapped.player_hand) < env.unwrapped._hand_value(env.unwrapped.dealer_hand):
                print("Dealer wins!")
            else:
                print("It's a tie!")
        
        time.sleep(1)
        
        if terminated or truncated:
            done = True
    
    print(f"Episode {episode+1} Results:")
    print(f"Final Bankroll: ${env.unwrapped.bankroll}")
    print(f"Total Reward: {total_reward:.2f}")

env.close()
