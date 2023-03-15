import gym
import gym_tetris
import os
import numpy as np

from pathlib import Path
from statistics import mean, median
from gym_tetris.ai.DQN import DQN

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def load_reward():
    if not os.path.exists("./Rewards"):
        os.mkdir("./Rewards")

    if Path("./Rewards/DQN_rewards.npy").is_file():
        return np.load("./Rewards/DQN_rewards.npy").tolist()
    else:
        return []


def load_score():
    if not os.path.exists("./Scores"):
        os.mkdir("./Scores")

    if Path("./Scores/DQN_scores.npy").is_file():
        return np.load("./Scores/DQN_scores.npy").tolist()
    else:
        return []


def main():
    env = gym.make("tetris-v1", action_mode=1)
    network = DQN()
    network.load()

    SEED = 50
    env.seed(SEED)
    network.seed(SEED)
    env.action_space.seed(SEED)

    total_rewards = load_reward()
    total_scores = load_score()

    running = True
    total_games = 0
    total_steps = 0

    while total_games < 5000:
        steps, rewards, scores = network.train(env, episodes=25)
        total_rewards.append(rewards)
        total_scores.append(scores)

        total_games += len(scores)
        total_steps += steps
        network.save()
        print("==================")
        print("* Total Games: ", total_games)
        print("* Total Steps: ", total_steps)
        print("* Epsilon: ", network.epsilon)
        print("*")
        print("* Average: ", sum(rewards) / len(rewards),
              "/", sum(scores) / len(scores))
        print("* Median: ", median(rewards), "/", median(scores))
        print("* Mean: ", mean(rewards), "/", mean(scores))
        print("* Min: ", min(rewards), "/", min(scores))
        print("* Max: ", max(rewards), "/", max(scores))
        print("==================")

    np.save("./Rewards/DQN_rewards.npy", np.array(total_rewards))
    np.save("./Scores/DQN_scores.npy", np.array(total_scores))

    env.close()


if __name__ == '__main__':
    main()
