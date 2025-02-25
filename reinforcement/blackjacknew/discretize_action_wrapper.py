# discretize_action_wrapper.py
import gymnasium as gym
from gymnasium import spaces

class DiscretizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(96 * 2)  # 96 bets × 2 moves
    
    def action(self, action):
        bet = action // 2
        move = action % 2
        return (bet, move)
