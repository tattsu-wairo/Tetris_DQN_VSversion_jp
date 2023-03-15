from turtle import shape
import numpy as np
import gym
import os
from tqdm import tqdm
import pygame
import gym_tetris_6state
import random

class Agent():
    def __init__(self, env):
        self.env = env
    def choose_action(self, obs):
        """
        Parameters:
            obs:
            1. landing height
            2. eroded piece cells
            3. row transitions
            4. column transitions
            5. holes
            6. wells
        """
        max_value = None
        best_action = None
        for action, state in obs:
            value = -45 * state[0] + 34 * state[1] - 32 * state[2] - 93 * state[3] - (79 * state[4]) - 34 * state[5]
            if max_value is None or max_value < value:
                max_value = value
                best_action = action
        return best_action


def test(env):
    """
    Test the agent on the given environment.

    Parameters:
        env: the given environment.

    Returns:
        None (Don't need to return anything)
    """
    testing_agent = Agent(env)
    rewards = []
    scores = []

    for _ in range(100):
        obs = env.reset()
        count_reward = 0
        while True:
            action = testing_agent.choose_action(obs)
            obs, reward, done, _ = testing_agent.env.step(action)
            count_reward += reward
            print(reward)
            if done:
                rewards.append(count_reward)
                scores.append(env.game.score)
                break
    print(f"average reward: {np.mean(rewards)}")
    print(f"average score: {np.mean(scores)}")

if __name__ == "__main__":
    '''
    The main funtion
    '''
    #SEED = 71

    env = gym.make("tetris-v1", action_mode = 1)
    env.seed()
    #env.action_space.seed(SEED)

    # testing section:
    test(env)

    """
    if not os.path.exists("./Rewards"):
        os.mkdir("./Rewards")
    if not os.path.exists("./Scores"):
        os.mkdir("./Scores")

    np.save("./Rewards/Q_learning_rewards.npy", np.array(total_reward))
    np.save("./Scores/Q_learning_scores.npy", np.array(total_score))
    """

    env.close()