from turtle import shape
import numpy as np
import gym
import os
from tqdm import tqdm
import pygame
import gym_tetris_6state
import random

total_reward = []
total_score = []
episode = 5000
decay = 0
best_qtable = np.zeros((11, 4, 21, 16, 180, 190, 200, 36))

class Agent():
    def __init__(self, env, epsilon=1, epsilon_min=0.0001, epsilon_decay=0.9995, learning_rate=0.5, GAMMA=0.97):
        """
        The agent learning how to control the action of tetris.

        Hyperparameters:
            epsilon: Determines the explore/expliot rate of the agent
            learning_rate: Determines the step size while moving toward a minimum of a loss function
            GAMMA: The discount factor (tradeoff between immediate rewards and future rewards)
        """
        self.env = env

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.gamma = GAMMA

        # initialize qtable
        # state = (Rows cleared, Bumpiness, Holes, Landing height, Row transition, Column transition, Cumulative wells, Eroded piece cells, Aggregate height)
        # action = (column, rotation)
        n_action_space = []
        n_observation_space = []
        for dimenstion in self.env.action_space:
            n_action_space.append(dimenstion.n)
        for dimenstion in self.env.observation_space:
            n_observation_space.append(dimenstion.n)
        n_action_space[0] += 1
        n_observation_space[0] += 1
        # print(n_action_space + n_observation_space)
        # 11, 4, 21, 80, 180, 190, 200, 36
        self.qtable = np.zeros((n_action_space + n_observation_space))

    def choose_best_state(self, possible_states):
        max_rating = None
        best_state = None
        for action, state in possible_states:
            rating = self.qtable[tuple(list(action)+state)]
            if max_rating is None or rating > max_rating:
                max_rating = rating
                best_state = (action, state)
        return best_state

    def choose_action(self, possible_states):
        """
        Choose the best action with given state and epsilon

        Parameters:
            state: A representation of the current state of the enviornment.
            epsilon: Determines the explore/expliot rate of the agent.
        
        Returns:
            action: The action to be evaluated.
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(possible_states)
        else:
            return self.choose_best_state(possible_states)
    
    def learn(self, state, action, reward, next_state, obs, done):
        """
        Calculate the new q-value base on the reward and state transformation observered after taking the action.

        Parameters:
            state: The state of the enviornment before taking the action.
            action: The exacuted action.
            reward: Obtained from the enviornment after taking the action.
            next_state: The state of the enviornment after taking the action.
            done: A boolean indicates whether the episode is done.

        Returns:
            None (Don't need to return anything)
        """
        
        # Q-learning Algorithm:
        # qtable[state, action] = (1-l_rate)*qtable[state, action] + l_rate*(reward + gamma*max(qtable[next_state, actions]))
        """
        max_next = None
        for column in range(11):
            for rotation in range(4):
                qvalue = self.qtable[tuple([column]+[rotation]+next_state)]
                if max_next is None or max_next < qvalue:
                    max_next = qvalue
        """
        if not done:
            max_next = None
            for next_action, _ in obs:
                qvalue = self.qtable[tuple(list(next_action)+next_state)]
                if max_next is None or max_next < qvalue:
                    max_next = qvalue
            current_value = self.qtable[tuple(list(action)+state)]
            self.qtable[tuple(list(action)+state)] = (1 - self.learning_rate) * current_value + self.learning_rate * (reward + self.gamma * max_next)
            # print(self.qtable[tuple(list(action)+state)])
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # You can add some conditions to decide when to save your table
        if done:
            global best_qtable
            if(len(total_reward) > 0):
                testing_agent = Agent(env)
                testing_agent.qtable = best_qtable
                r = [0, 0] # [old, new]
                for i in range(2):
                    t_obs = testing_agent.env.reset()
                    t_state = testing_agent.env.game.board.get_info([])
                    while True:
                        t_action, t_next_state = testing_agent.choose_best_state(t_obs)
                        t_obs, t_reward, t_done, _ = testing_agent.env.step(t_action)
                        r[i] += t_reward
                        if t_done:
                            break
                        t_state = t_next_state
                    testing_agent.qtable = self.qtable
                if r[1] > r[0]:
                    best_qtable = self.qtable

            else:
                best_qtable = self.qtable

def train(env):
    """
    Train the agent on the given environment

    Paramenters:
        env: the given environment.

    Returns:
        None (Don't need to return anything)
    """
    training_agent = Agent(env)
    rewards = []
    scores = []
    for ep in tqdm(range(episode)):
        obs = env.reset()
        state = env.game.board.get_info([])
        done = False
        count_reward = 0
        while True:
            action, next_state = training_agent.choose_action(obs)
            obs, reward, done, _ = env.step(action)
            #print(done, next_state)

            training_agent.learn(state, action, reward, next_state, obs, done)
            count_reward += reward

            if done:
                rewards.append(count_reward)
                scores.append(env.game.score)
                break

            state = next_state

        if (ep + 1) % 500 == 0:
            training_agent.learning_rate -= decay
    total_reward.append(rewards)
    total_score.append(scores)

def test(env):
    """
    Test the agent on the given environment.

    Paramenters:
        env: the given environment.

    Returns:
        None (Don't need to return anything)
    """
    testing_agent = Agent(env)
    #testing_agent.qtable = np.load("./Tables/tetris_table.npy")
    #print(best_qtable)
    testing_agent.qtable = best_qtable
    rewards = []
    scores = []

    for _ in range(100):
        obs = env.reset()
        state = env.game.board.get_info([])
        count_reward = 0
        while True:
            action, next_state = testing_agent.choose_best_state(obs)
            obs, reward, done, _ = testing_agent.env.step(action)
            count_reward += reward
            if done:
                rewards.append(count_reward)
                scores.append(env.game.score)
                break
            state = next_state
    print(f"average reward: {np.mean(rewards)}")
    print(rewards)
    print(f"average score: {np.mean(scores)}")

def seed(seed=20):
    '''
    It is very IMPORTENT to set random seed for reproducibility of your result!
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    '''
    The main funtion
    '''
    #SEED = 71

    env = gym.make("tetris-v1", action_mode = 1)
    env.seed()
    #env.action_space.seed(SEED)

   # training section:
    for i in range(2):
        print(f"#{i + 1} training progress")
        train(env)
    # testing section:
    test(env)

    if not os.path.exists("./Rewards"):
        os.mkdir("./Rewards")
    if not os.path.exists("./Scores"):
        os.mkdir("./Scores")

    np.save("./Rewards/Q_learning_rewards.npy", np.array(total_reward))
    np.save("./Scores/Q_learning_scores.npy", np.array(total_score))

    env.close()