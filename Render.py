from Lawn_Mower_Environment import LawnMowerEnv
from DQN_Agent import DQNAgent
import matplotlib.pyplot as plt
import numpy as np

def visualize_model(model_path, grid_size=15, num_obstacles=10):
    """
    Visualize the trained DQN agent in the Lawn Mower Environment.

    @param model_path: Path to the trained model file.
    @param grid_size: Size of the environment grid (default: 15x15).
    @param num_obstacles: Number of obstacles in the grid (default: 10).
    """
    # Initialize the environment
    env = LawnMowerEnv(grid_size=grid_size, num_obstacles=num_obstacles)

    # Initialize the agent and load the trained model
    agent = DQNAgent(state_shape=env.observation_space.shape, action_size=env.action_space.n)
    agent.load(model_path)

    # Reset the environment
    state = env.reset()
    done = False
    steps = 0
    total_reward = 0

    print("Starting Visualization...")

    while not done:
        # Use the trained model to select an action
        action = agent.act(state)

        # Execute action and observe the result
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1

        # Render the grid
        plt.imshow(state, cmap='Greens', vmin=-1, vmax=1)

        # Add obstacles
        for pos in env.obstacles:
            plt.scatter(pos[1], pos[0], c='red', s=100, label="Obstacle" if steps == 1 else "")

        # Add the agent's current position
        plt.scatter(env.agent_pos[1], env.agent_pos[0], c='blue', s=100, label="Agent" if steps == 1 else "")

        # Add title and legend
        if steps == 1:
            plt.legend(loc="upper right")
        plt.title(f"Step: {steps}, Reward: {total_reward:.2f}")
        plt.pause(0.1)  # Pause for visualization
        plt.clf()  # Clear the plot for the next frame

        # Update state
        state = next_state

    # Final visualization
    print(f"Visualization Complete: Steps = {steps}, Total Reward = {total_reward:.2f}")
    plt.close()

if __name__ == "__main__":
    """
    Main function to visualize the trained DQN model.

    Usage:
        - Update the `model_path` with the path to the saved DQN model.
    """
    model_path = "lawn_mower_dqn_G15_O10_log_1.h5"  # Replace with the correct model path
    try:
        visualize_model(model_path=model_path, grid_size=15, num_obstacles=10)
    except FileNotFoundError as e:
        print(f"Error: {e}")
