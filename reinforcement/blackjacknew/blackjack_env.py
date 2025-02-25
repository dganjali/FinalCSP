# blackjack_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random

class BlackjackEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # Action space: (bet_amount, move) 
        self.action_space = spaces.Tuple((
            spaces.Discrete(96),  # $5 increments from $5-$500 (5*(1-96))
            spaces.Discrete(2)    # 0=stand, 1=hit
        ))
        
        # Updated observation space: player_sum, dealer_card, usable_ace, bankroll
        self.observation_space = spaces.Box(
            low=np.array([4, 1, 0, 0]),
            high=np.array([21, 10, 1, 1000]),
            dtype=np.float32
        )
        
        # Game parameters
        self.initial_bankroll = 1000
        self.min_bet = 5
        self.max_games = 20
        self.deck = []
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.bankroll = self.initial_bankroll
        self.game_count = 0
        self.current_bet = 0
        self.player_hand = []
        self.dealer_hand = []
        self.shuffle_deck()
        return self._get_obs(), {}

    def shuffle_deck(self):
        self.deck = [1,2,3,4,5,6,7,8,9,10,10,10,10] * 4
        random.shuffle(self.deck)

    def step(self, action):
        terminated = False
        truncated = False
        reward = 0
        
        # Handle betting phase
        if not self.player_hand:
            bet_amount = (action[0] + 1) * 5  # Convert to $5-$500
            self.current_bet = min(max(bet_amount, self.min_bet), self.bankroll)
            self.bankroll -= self.current_bet
            self.player_hand = [self.draw_card(), self.draw_card()]
            self.dealer_hand = [self.draw_card()]
            return self._get_obs(), 0.0, False, False, {}

        # Handle game action
        move = action[1]
        if move == 1:  # Hit
            self.player_hand.append(self.draw_card())
            player_value = self._hand_value(self.player_hand)
            
            if player_value > 21:  # Bust
                terminated = True
                reward = -self.current_bet
            else:
                reward = 0.1 * (player_value / 21)  # Encourage approaching 21
        else:  # Stand
            terminated = True
            dealer_value = self._dealer_play()
            player_value = self._hand_value(self.player_hand)
            
            if player_value > 21:
                reward = -self.current_bet
            elif dealer_value > 21 or player_value > dealer_value:
                reward = self.current_bet
                self.bankroll += self.current_bet * 2
            elif player_value == dealer_value:
                reward = 0
                self.bankroll += self.current_bet
            else:
                reward = -self.current_bet

        if terminated:
            self.game_count += 1
            self.current_bet = 0
            self.player_hand = []
            self.dealer_hand = []
            
            if self.game_count >= self.max_games or self.bankroll < self.min_bet:
                truncated = True
                # Final reward based on total performance
                final_reward = (self.bankroll - self.initial_bankroll) / self.initial_bankroll
                reward += final_reward * 2  # Amplify final outcome

        return self._get_obs(), float(reward), terminated, truncated, {}

    def _get_obs(self):
        player_value = self._hand_value(self.player_hand) if self.player_hand else 0
        dealer_card = self.dealer_hand[0] if self.dealer_hand else 1
        usable_ace = int(1 in (self.player_hand or []) and player_value + 10 <= 21)
        
        return np.array([
            player_value,
            dealer_card,
            usable_ace,
            self.bankroll  # Return actual bankroll value
        ], dtype=np.float32)

    def _hand_value(self, hand):
        value = sum(hand)
        aces = hand.count(1)
        while value <= 21 and aces > 0:
            value += 10
            aces -= 1
            if value > 21:
                value -= 10
                break
        return value

    def _dealer_play(self):
        while self._hand_value(self.dealer_hand) < 17:
            self.dealer_hand.append(self.draw_card())
        return self._hand_value(self.dealer_hand)

    def draw_card(self):
        if len(self.deck) < 15:
            self.shuffle_deck()
        return self.deck.pop()

    def render(self):
        if self.render_mode == "human":
            print(f"\nGame {self.game_count + 1}/{self.max_games}")
            print(f"Bankroll: ${self.bankroll}")
            if self.player_hand:
                print(f"Player: {self.player_hand} (Total: {self._hand_value(self.player_hand)})")
                print(f"Dealer: [{self.dealer_hand[0]}, ?]")
                print(f"Current Bet: ${self.current_bet}")

