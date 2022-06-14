from __future__ import division
from pathlib import Path
from tqdm import tqdm

import os
import random
import numpy as np
import tensorflow as tf

WEIGHT_PATH = os.path.join(os.path.dirname(__file__), 'weights.h5')
WEIGHT_PATH_HOLD = os.path.join(os.path.dirname(__file__), 'weights_hold.h5')
IMAGE_PATH = os.path.join(os.path.dirname(__file__), 'model.png')
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.memory = []
        self.buffer_size = capacity
    
    def insert(self, experience):
        if len(self.memory) + len(experience) >= self.buffer_size:
            self.memory[0:(len(experience) + len(self.memory)) - self.buffer_size] = []
        self.memory.append(experience)

    def sample(self, size):
        return random.sample(self.memory, size)

class DQN:
    def __init__(self, state_size=9, gamma=0.99, epsilon=1, epsilon_min=0.0001, epsilon_decay=0.9995, hold_mode=0):
        
        self.state_size = state_size
        self.model = self._create_model()
        self.buffer = ReplayBuffer()

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.hold_mode = hold_mode  #whether considering hold when trainning, playing ai.
        
        self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR,
                                                          histogram_freq=1000,
                                                          write_graph=True,
                                                          write_images=True)

    def _create_model(self):
        """
        construct a new model with tensorflow and return it.
        
        Returns: 
            A tensorflow model.
        """

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear'),
        ])

        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=['mean_squared_error'])

        model.summary()

        #Show tensor flow Layers
        tf.keras.utils.plot_model(model, IMAGE_PATH, show_shapes=True)

        return model
    
    def choose_action(self, possible_states):
        """
        - Implement the action-choosing function.
        - Choose the best action with given state and epsilon
        
        Parameters:
            possible_states: A list of all the possible next state. With form--(action, state)
        
        Returns:
            best_state: the chosen state.
        """

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(possible_states)

        max_q = None
        best_state = None
        q_values = self._predict_ratings([state for action, state in possible_states])

        for i, (action, state) in enumerate(possible_states):
            q_value = q_values[i]
            if not max_q or q_value > max_q:
                max_q = q_value
                best_state = (action, state)

        return best_state

    def _predict_ratings(self, states):
        """
        -- Get the outputs of the Neural Network for multiple states.
        
        """
        y = np.array(states)
        values = self.model.predict(y)
        return [value[0] for value in values]
    
    def learn(self, batch_size=512, epochs=1):
        """
        -- Learn experience and adjust the weights(value).

        After leaning it will adjust the weight, the epsilon will decay since we
        want the probability of choosing random action will be decreasing with the 
        trainning process.

        """
        if len(self.buffer.memory) < batch_size:
            return

        batch = self.buffer.sample(batch_size)
        train_x = []
        train_y = []

        q_values = self._predict_ratings([x[2] for x in batch])

        for i, (previous_state, reward, next_state, done) in enumerate(batch):
            if not done:
                q_value = q_values[i]
                q = reward + self.gamma * q_value
            else:
                q = reward
            train_x.append(previous_state)
            train_y.append(q)

        self.model.fit(np.array(train_x), np.array(train_y), batch_size=len(train_x), verbose=0,
                       epochs=epochs, callbacks=[self.tensorboard])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def train(self, env, episodes=1):
        """
        -- Trains the Neural Network for n episodes.
        -- In every episode, it trains the model with the 20000 most recent experiences.
        Returns: 
            [steps, rewards, scores]: for monitoring trainning process            
        """
        rewards = []
        scores = []
        steps = 0

        for _ in tqdm(range(episodes)):
            obs = env.reset()
            previous_state = env.game.board.get_info([])
            done = False
            total_reward = 0
            count = 0
            while not done:
                action, state = self.choose_action(obs)
                obs, reward, done, info = env.step(action)
                self.buffer.insert((previous_state, reward, state, done))
                previous_state = state
                steps += 1
                count += 1
                total_reward += reward

            rewards.append(total_reward)
            scores.append(env.game.score)

            self.learn()

        return [steps, rewards, scores]

    def load(self):
        """Load the weights."""
        if self.hold_mode == 0:
            if Path(WEIGHT_PATH).is_file():
                self.model.load_weights(WEIGHT_PATH)
        else:
            if Path(WEIGHT_PATH_HOLD).is_file():
                self.model.load_weights(WEIGHT_PATH_HOLD)

    def save(self):
        """Save the weights."""
        if self.hold_mode == 0:
            if not os.path.exists(os.path.dirname(WEIGHT_PATH)):
                os.makedirs(os.path.dirname(WEIGHT_PATH))

            self.model.save_weights(WEIGHT_PATH)
        else:
            if not os.path.exists(os.path.dirname(WEIGHT_PATH_HOLD)):
                os.makedirs(os.path.dirname(WEIGHT_PATH_HOLD))

            self.model.save_weights(WEIGHT_PATH_HOLD)
    
    def seed(self, seed=20):
        '''
        It is very IMPORTENT to set random seed for reproducibility of your result!
        '''
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)  