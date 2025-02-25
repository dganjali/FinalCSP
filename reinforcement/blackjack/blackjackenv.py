import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random

class BlackjackEnv(gym.Env):
    def __init__(self, render_mode=None, training=True):
        super().__init__()
        self.render_mode = render_mode
        self.training = training
        self.action_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(11)))  # 0-10% of bankroll
        self.observation_space = spaces.Box(
            low=np.array([4, 1, 0, 0]),  # player_sum, dealer_card, usable_ace, bet_ratio
            high=np.array([21, 10, 1, 1]),
            dtype=np.float32
        )

        self.initial_bankroll = 1000
        self.bankroll = self.initial_bankroll
        self.current_bet = 0
        self.max_games = 20
        self.game_count = 0
        self.bet_placed = False
        
        self.reset_deck()

    def reset_deck(self):
        self.deck = [1,2,3,4,5,6,7,8,9,10,10,10,10] * 4
        random.shuffle(self.deck)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if len(self.deck) < 15:
            self.reset_deck()
            
        self.player_hand = [self.draw_card(), self.draw_card()]
        self.dealer_hand = [self.draw_card()]
        self.current_bet = 0
        self.bet_placed = False
        self.game_count = 0
        
        return self._get_obs(), {}

    def step(self, action):
        move, bet_pct = action
        terminated = False
        truncated = False
        old_bankroll = self.bankroll
        shaped_reward = 0

        # Betting phase
        if not self.bet_placed:
            # Convert percentage to actual bet amount (1-10% of bankroll)
            self.current_bet = min(self.bankroll * (bet_pct/10), self.bankroll)
            self.bankroll -= self.current_bet
            self.bet_placed = True
            shaped_reward += 0.01 * (bet_pct/10)  # Encourage proportional betting

        # Game logic with intermediate rewards
        if move == 1:  # Hit
            self.player_hand.append(self.draw_card())
            player_value = self.hand_value(self.player_hand)
            
            # Shaped rewards for strategic positions
            if 17 <= player_value <= 21:
                shaped_reward += 0.1  # Encourage stopping near 17
            elif player_value > 21:
                shaped_reward -= 0.5  # Strong penalty for busting
            else:
                shaped_reward += 0.05 * (player_value/21)  # Progress reward

        if terminated := (move == 0 or self.hand_value(self.player_hand) > 21):
            # Final outcome rewards
            dealer_value = self.hand_value(self.dealer_hand)
            while dealer_value < 17:
                self.dealer_hand.append(self.draw_card())
                dealer_value = self.hand_value(self.dealer_hand)

            player_value = self.hand_value(self.player_hand)
            outcome = self._determine_outcome(player_value, dealer_value)
            
            # Outcome-based rewards scaled by bet percentage
            if outcome == "win":
                self.bankroll += self.current_bet * 2
                shaped_reward += 1.0 * (self.current_bet/self.initial_bankroll)
            elif outcome == "push":
                self.bankroll += self.current_bet
                shaped_reward += 0.2
            else:
                shaped_reward -= 0.8 * (self.current_bet/self.initial_bankroll)

            # Natural blackjack bonus
            if len(self.player_hand) == 2 and player_value == 21:
                shaped_reward += 0.5

            # Update game state
            self.game_count += 1
            self.bet_placed = False
            self.player_hand = [self.draw_card(), self.draw_card()]
            self.dealer_hand = [self.draw_card()]

            if self.game_count >= self.max_games:
                truncated = True

        # Combine shaped and monetary rewards
        reward = shaped_reward + (self.bankroll - old_bankroll)/self.initial_bankroll
        return self._get_obs(), float(reward), terminated, truncated, {}

    def _get_obs(self):
        return np.array([
            self.hand_value(self.player_hand),
            self.dealer_hand[0],
            int(1 in self.player_hand and self.hand_value(self.player_hand) + 10 <= 21),
            self.current_bet / self.initial_bankroll  # Normalized bet ratio
        ], dtype=np.float32)

    def _determine_outcome(self, player, dealer):
        if player > 21: return "loss"
        if dealer > 21: return "win"
        if player > dealer: return "win"
        if player == dealer: return "push"
        return "loss"

    # Keep other methods same as previous version
    def draw_card(self):
        return self.deck.pop() # pop from the shuffled deck

    def hand_value(self, hand):
        value = sum(hand)
        num_aces = hand.count(1)
        while value + 10 <= 21 and num_aces:
            value += 10
            num_aces -= 1
        return value

    def render(self):
        if self.render_mode == "human":
            print(f"Player Hand: {self.player_hand} (Value: {self.hand_value(self.player_hand)})")
            print(f"Dealer Hand: {self.dealer_hand} (Upcard: {self.dealer_hand[0]})")
            print(f"Player Balance: ${self.bankroll}")
            print(f"Current Bet: ${self.current_bet}")
