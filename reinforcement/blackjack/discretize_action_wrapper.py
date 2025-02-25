import gymnasium as gym
from gymnasium import spaces

class DiscretizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(2 * 101)
    
    def action(self, action):
        move = action // 101
        bet = action % 101
        return (move, bet)
