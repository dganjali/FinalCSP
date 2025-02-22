import gym
import numpy as np
import pygame
from gym import spaces

class CatchBallEnv(gym.Env):
    """Custom Gym environment where the agent must catch a falling ball."""
    
    def __init__(self, render_mode=None):
        super().__init__()

        # Environment settings
        self.grid_size = 10  # 10x10 grid
        self.paddle_size = 2  # Paddle is 2 blocks wide
        self.max_speed = 3  # Ball can fall up to 3 cells at a time

        # Action space: 0 = Left, 1 = Stay, 2 = Right
        self.action_space = spaces.Discrete(3)

        # Observation space: Ball (row, col) + Paddle position
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(3,), dtype=np.int32)

        # Rendering settings
        self.render_mode = render_mode
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((300, 300))
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self, seed=None, options=None):
        """Resets the environment to start a new episode."""
        super().reset(seed=seed)

        self.ball_x = np.random.randint(0, self.grid_size)  # Random start column
        self.ball_y = 0  # Start at the top
        self.paddle_x = self.grid_size // 2  # Paddle starts at center
        self.speed = 1  # Ball starts slow

        return np.array([self.ball_x, self.ball_y, self.paddle_x], dtype=np.int32), {}

    def step(self, action):
        """Applies an action and moves the environment forward."""
        # Move paddle
        if action == 0:  # Left
            self.paddle_x = max(0, self.paddle_x - 1)
        elif action == 2:  # Right
            self.paddle_x = min(self.grid_size - self.paddle_size, self.paddle_x + 1)

        # Move ball down (increase speed over time)
        self.ball_y += min(self.speed, self.grid_size - 1 - self.ball_y)

        # Increase difficulty: Ball speed increases after 5 catches
        self.speed = min(self.speed + 0.1, self.max_speed)

        # Check if ball reaches bottom row
        done = False
        reward = 0
        if self.ball_y == self.grid_size - 1:
            done = True  # Episode ends
            if self.paddle_x <= self.ball_x < self.paddle_x + self.paddle_size:
                reward = 1  # Caught the ball!
            else:
                reward = -1  # Missed the ball, game over

        return np.array([self.ball_x, self.ball_y, self.paddle_x], dtype=np.int32), reward, done, False, {}

    def render(self):
        """Renders the environment using pygame."""
        if self.render_mode != "human":
            return
        
        self.screen.fill((0, 0, 0))

        cell_size = 30
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pygame.draw.rect(self.screen, (50, 50, 50), (j * cell_size, i * cell_size, cell_size, cell_size), 1)

        # Draw ball
        pygame.draw.circle(self.screen, (255, 0, 0), (self.ball_x * cell_size + cell_size // 2, self.ball_y * cell_size + cell_size // 2), cell_size // 3)

        # Draw paddle
        for i in range(self.paddle_size):
            pygame.draw.rect(self.screen, (0, 255, 0), ((self.paddle_x + i) * cell_size, (self.grid_size - 1) * cell_size, cell_size, cell_size))

        pygame.display.flip()
        self.clock.tick(10)  # Control game speed

    def close(self):
        """Closes the environment."""
        if self.render_mode == "human":
            pygame.quit()
