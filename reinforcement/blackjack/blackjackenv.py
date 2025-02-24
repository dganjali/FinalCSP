import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BlackjackEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(BlackjackEnv, self).__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(2)  # 0: Stick, 1: Hit
        self.observation_space = spaces.Box(low=np.array([1, 1, 0]), high=np.array([31, 11, 1]), dtype=np.int32)
        self.deck = [1,2,3,4,5,6,7,8,9,10,10,10,10] * 4
        self.reset()
    
    def draw_card(self):
        return np.random.choice(self.deck)
    
    def hand_value(self, hand):
        value = sum(hand)
        num_aces = hand.count(1)
        while value + 10 <= 21 and num_aces:
            value += 10
            num_aces -= 1
        return value
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_hand = [self.draw_card(), self.draw_card()]
        self.dealer_hand = [self.draw_card()]
        return np.array([self.hand_value(self.player_hand), self.dealer_hand[0], int(1 in self.player_hand)]), {}
    
    def step(self, action):
        if action == 1:  # Hit
            self.player_hand.append(self.draw_card())
            if self.hand_value(self.player_hand) > 21:
                return np.array([self.hand_value(self.player_hand), self.dealer_hand[0], int(1 in self.player_hand)]), -1, True, False, {}
        else:  # Stick
            while self.hand_value(self.dealer_hand) < 17:
                self.dealer_hand.append(self.draw_card())
            player_score = self.hand_value(self.player_hand)
            dealer_score = self.hand_value(self.dealer_hand)
            reward = 1 if player_score > dealer_score or dealer_score > 21 else -1 if player_score < dealer_score else 0
            return np.array([player_score, self.dealer_hand[0], int(1 in self.player_hand)]), reward, True, False, {}
        return np.array([self.hand_value(self.player_hand), self.dealer_hand[0], int(1 in self.player_hand)]), 0, False, False, {}
    
    def render(self):
        if self.render_mode == "human":
            print(f"Player: {self.player_hand} ({self.hand_value(self.player_hand)})")
            print(f"Dealer: {self.dealer_hand} ({self.hand_value(self.dealer_hand)})")
