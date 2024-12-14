import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class LawnMowerEnv(gym.Env):
    """
    Lawn Mower Environment for Reinforcement Learning.
    The agent navigates a grid to mow grass, avoid obstacles, and optimize efficiency.
    """

    def __init__(self, grid_size=15, num_obstacles=10):
        """
        Initialize the Lawn Mower environment.

        @param grid_size: Size of the grid (default: 15x15).
        @param num_obstacles: Number of obstacles in the environment (default: 10).
        """
        super(LawnMowerEnv, self).__init__()
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.state = None
        self.agent_pos = None
        self.obstacles = None

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.observation_space = spaces.Box(low=-1, high=1, shape=(grid_size, grid_size), dtype=np.float32)

    def reset(self):
        """
        Reset the environment to its initial state.

        @return: The initial state of the grid.
        """
        # Initialize grid
        self.state = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Place obstacles
        self.obstacles = set()
        while len(self.obstacles) < self.num_obstacles:
            obstacle_pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            if obstacle_pos != (0, 0):  # Avoid starting position
                self.obstacles.add(obstacle_pos)

        for obstacle in self.obstacles:
            self.state[obstacle] = -1  # Mark obstacle cells

        # Initialize agent position
        self.agent_pos = [0, 0]
        self.state[tuple(self.agent_pos)] = 1  # Mark starting position as mowed
        return self.state.copy()

    def step(self, action: int):
        """
        Execute an action and update the environment.

        @param action: The action to perform (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT).
        @return: A tuple of (state, reward, done, info).
        """
        row, col = self.agent_pos

        # Determine new position based on action
        if action == 0: new_pos = [max(0, row - 1), col]  # UP
        elif action == 1: new_pos = [min(self.grid_size - 1, row + 1), col]  # DOWN
        elif action == 2: new_pos = [row, max(0, col - 1)]  # LEFT
        elif action == 3: new_pos = [row, min(self.grid_size - 1, col + 1)]  # RIGHT
        else: raise ValueError("Invalid action")

        reward = 0

        # Penalty for hitting obstacles
        if tuple(new_pos) in self.obstacles:
            return self.state.copy(), -10, False, {}

        # Update position
        self.agent_pos = new_pos

        # Reward shaping
        if self.state[tuple(new_pos)] == 0:  # Mowing new grass
            reward += 5
            self.state[tuple(new_pos)] = 1  # Mark as mowed
        else:  # Penalty for revisiting mowed cells
            reward -= 1

        # Small step penalty to encourage efficiency
        reward -= 0.01

        # Check for grid completion
        done = self.is_grid_fully_mowed()
        return self.state.copy(), reward, done, {}

    def is_grid_fully_mowed(self):
        """
        Check if all mowable cells have been mowed.

        @return: True if grid is fully mowed, otherwise False.
        """
        return np.all((self.state == 1) | (self.state == -1))

    def render(self):
        """
        Visualize the grid and agent's position.
        """
        plt.imshow(self.state, cmap="Greens")
        for pos in self.obstacles:
            plt.scatter(pos[1], pos[0], c='red', s=100, label="Obstacle")  # Obstacles
        plt.scatter(self.agent_pos[1], self.agent_pos[0], c='blue', s=100, label="Agent")  # Agent
        plt.title("Lawn Mower Environment")
        plt.legend()
        plt.pause(0.1)
        plt.clf()
