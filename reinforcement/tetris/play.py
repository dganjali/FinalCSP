import sys
import cv2
import gymnasium as gym
from tetris_gymnasium.envs import Tetris

if __name__ == "__main__":
    # Create an instance of Tetris with human rendering
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    env.reset(seed=42)

    terminated = False
    while not terminated:
        # Render the current state (assumed to show on screen)
        env.render()

        # Wait for user input action.
        action = None
        while action is None:
            key = cv2.waitKey(1)
            if key == ord("a"):
                action = env.unwrapped.actions.move_left
            elif key == ord("d"):
                action = env.unwrapped.actions.move_right
            elif key == ord("s"):
                action = env.unwrapped.actions.move_down
            elif key == ord("w"):
                action = env.unwrapped.actions.rotate_counterclockwise
            elif key == ord("e"):
                action = env.unwrapped.actions.rotate_clockwise
            elif key == ord(" "):  # space key for hard drop
                action = env.unwrapped.actions.hard_drop
            elif key == ord("q"):
                action = env.unwrapped.actions.swap
            elif key == ord("r"):
                # Reset environment when 'r' is pressed
                env.reset(seed=42)
                break

            # Check if the display window has been closed.
            if cv2.getWindowProperty(env.unwrapped.window_name, cv2.WND_PROP_VISIBLE) < 1:
                sys.exit()

        # If an action was chosen, step the env.
        if action is not None:
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print("Game Over!")
                terminated = True

    env.close()