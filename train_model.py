from Lawn_Mower_Environment import LawnMowerEnv
from DQN_Agent import DQNAgent
import numpy as np
import tensorflow as tf
from collections import deque

# Initialize environment and agent
grid_size = 15  # Grid size: 15x15
num_obstacles = 10  # Number of obstacles: 10
env = LawnMowerEnv(grid_size=grid_size, num_obstacles=num_obstacles)
agent = DQNAgent(state_shape=env.observation_space.shape, action_size=env.action_space.n)

# Metrics for logging
reward_history = deque(maxlen=5)  # Calculate average reward over the last 5 episodes
episodes = 2000  # Number of training episodes
steps_per_episode = []

# TensorBoard setup
log_dir = "logs/Grid_15_Obstacles_10_log"
writer = tf.summary.create_file_writer(log_dir)

def train():
    """
    Train the DQN agent in the Lawn Mower Environment.

    The agent navigates a 15x15 grid while mowing grass, avoiding obstacles, 
    and learning efficient paths. Reward shaping is applied:
    - +5 for mowing a new cell.
    - -2 for revisiting an already mowed cell.
    - -10 for hitting an obstacle.
    - Small step penalty (-0.01) to encourage efficiency.

    Training metrics (reward, steps, loss, and max Q-value) are logged to TensorBoard.
    """
    for e in range(episodes):
        state = env.reset()  # Reset environment at the start of the episode
        total_reward = 0  # Total reward for the episode
        total_steps = 0  # Steps taken in the episode
        total_loss = 0  # Track total loss for logging
        max_q_value = float('-inf')  # Track maximum Q-value for logging

        for time in range(1000):  # Limit steps per episode
            action = agent.act(state)  # Select action using epsilon-greedy policy
            next_state, reward, done, _ = env.step(action)  # Perform action and observe environment response

            # Reward shaping
            if reward == 5:
                reward += 5  # Bonus for mowing a new cell
            elif reward == -1:
                reward -= 1  # Penalty for revisiting mowed cells
            elif reward == -10:
                reward -= 10  # Penalty for hitting obstacles
            reward -= 0.01  # Small step penalty

            # Store experience in replay memory
            agent.remember(state, action, reward, next_state, done)

            # Update total reward, steps, and Q-value
            state = next_state
            total_reward += reward
            total_steps += 1
            q_value = np.amax(agent.model.predict(state[np.newaxis, :], verbose=0))
            max_q_value = max(max_q_value, q_value)

            if done:
                break  # End the episode if the grid is fully mowed

        # Train the agent with experience replay and collect the loss
        total_loss = agent.replay(total_reward=total_reward, steps_to_complete=total_steps)

        # Calculate average reward for last 5 episodes
        reward_history.append(total_reward)
        avg_reward_last_5 = np.mean(reward_history)

        # Log metrics to TensorBoard
        with writer.as_default():
            tf.summary.scalar("Total Reward", total_reward, step=e)
            tf.summary.scalar("Steps to Complete Grid", total_steps, step=e)
            tf.summary.scalar("Loss", total_loss, step=e)
            tf.summary.scalar("Max Q-Value", max_q_value, step=e)
            tf.summary.scalar("Epsilon", agent.epsilon, step=e)

        # Print episode summary
        print(f"Episode {e+1}/{episodes}: Total Reward: {total_reward:.2f}, Steps: {total_steps}, "
              f"Avg Reward (Last 5): {avg_reward_last_5:.2f}, Loss: {total_loss:.4f}, Max Q-Value: {max_q_value:.2f}")

        # Save the model at intervals
        if e % 1 == 0 and e > 0:
            model_name = f"lawn_mower_dqn_G15_O10_log_{e}.h5"
            agent.save(model_name)
            print(f"Model saved at episode {e} as {model_name}")

    # Save the final model
    final_model_name = "lawn_mower_dqn_G15_O10_final.h5"
    agent.save(final_model_name)
    print(f"Final model saved as {final_model_name}")

if __name__ == "__main__":
    train()
