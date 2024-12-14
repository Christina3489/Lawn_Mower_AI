import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from collections import deque
from tensorflow.keras.losses import MeanSquaredError  # Explicitly import the loss function
import numpy as np
import random

class DQNAgent:
    """
    Deep Q-Network (DQN) Agent for Reinforcement Learning.

    The agent interacts with an environment to learn optimal actions using the DQN algorithm,
    leveraging experience replay and epsilon-greedy exploration.
    """
    def __init__(self, state_shape, action_size, log_dir="logs/Grid_15_Obstacles_10_log"):
        """
        Initialize the DQN agent with necessary parameters and components.

        @param state_shape: Shape of the state space.
        @param action_size: Number of possible actions.
        @param log_dir: Directory for TensorBoard logs (default: "logs/Grid_15_Obstacles_10_log").
        """
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=5000)  # Experience replay memory
        self.gamma = 0.990  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_decay = 0.998  # Decay rate for exploration
        self.epsilon_min = 0.1  # Minimum exploration rate
        self.batch_size = 32  # Mini-batch size for experience replay
        self.log_dir = log_dir  # Directory for logging TensorBoard metrics
        self.writer = tf.summary.create_file_writer(log_dir)  # TensorBoard writer
        self.episode_count = 0  # Counter for training episodes

        # Build the neural network model
        self.model = self._build_model(state_shape, action_size)

    def _build_model(self, state_shape, action_size):
        """
        Build a neural network to approximate the Q-value function.

        @param state_shape: Shape of the input state space.
        @param action_size: Number of possible actions.
        @return: Compiled Keras model for Q-value estimation.
        """
        model = Sequential([
            Flatten(input_shape=state_shape),
            Dense(24, activation='relu'),
            Dense(24, activation='relu'),
            Dense(action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in the replay memory.

        @param state: Current state.
        @param action: Action taken.
        @param reward: Reward received.
        @param next_state: Resulting state after the action.
        @param done: Whether the episode has ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Select an action using the epsilon-greedy policy.

        @param state: Current state of the environment.
        @return: Chosen action.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore: Random action
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(q_values[0])  # Exploit: Action with highest Q-value

    def replay(self, total_reward=0, steps_to_complete=0):
        """
        Train the neural network using mini-batches sampled from experience replay memory.
        Log metrics for TensorBoard.

        @param total_reward: Total reward for the episode.
        @param steps_to_complete: Steps taken to complete the grid.
        """
        if len(self.memory) < self.batch_size:
            return 0  # Return zero loss if there's not enough data to train

        minibatch = random.sample(self.memory, self.batch_size)
        total_loss = 0
        max_q_value = float('-inf')

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state[np.newaxis, :], verbose=0)[0])
            target_f = self.model.predict(state[np.newaxis, :], verbose=0)
            max_q_value = max(max_q_value, np.amax(target_f))  # Ensure Q-value is valid
            target_f[0][action] = target
            history = self.model.fit(state[np.newaxis, :], target_f, epochs=1, verbose=0)
            total_loss += history.history['loss'][0]

        # Safety checks to avoid logging None
        total_loss = total_loss / self.batch_size if total_loss is not None else 0
        max_q_value = max_q_value if max_q_value != float('-inf') else 0

        # Log metrics to TensorBoard
        with self.writer.as_default():
            tf.summary.scalar("Total Reward", total_reward, step=self.episode_count)
            tf.summary.scalar("Steps to Complete Grid", steps_to_complete, step=self.episode_count)
            tf.summary.scalar("Loss", total_loss, step=self.episode_count)
            tf.summary.scalar("Max Q-Value", max_q_value, step=self.episode_count)
            tf.summary.scalar("Epsilon", self.epsilon, step=self.episode_count)

        self.episode_count += 1
        return total_loss

    def save(self, name):
        """
        Save the trained model to a file.

        @param name: Path to save the model.
        """
        self.model.save(name)

    # def load(self, name):
    #     """
    #     Load a pre-trained model from a file.

    #     @param name: Path to the saved model.
    #     """
    #     self.model = tf.keras.models.load_model(name)
    def load(self, name):
        """
        Load a pre-trained model.

        @param name: Path to the saved model file.
        """
        
        self.model = tf.keras.models.load_model(name, custom_objects={'mse': MeanSquaredError()})
