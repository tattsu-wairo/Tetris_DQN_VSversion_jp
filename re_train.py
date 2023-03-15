import gym
import gym_tetris
import os
import numpy as np

from pathlib import Path
from statistics import mean, median
from gym_tetris.ai.DQN import DQN
import tensorflow as tf
import time
import datetime
import json
import requests
from gym_tetris.ai.DQN import WEIGHT_PATH_HOLD
import pygame

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# gpu = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu[0], True)

webhook_url = 'https://discord.com/api/webhooks/1049978183665066064/gN0LPx_z9Ile4Q_Y3FkkWMobDnS58OS8bFwplSq_u0pZIZrmJYFlv32JMCyj3ZscwYhY'
main_content = {'content': '送るテキスト'}
headers = {'Content-Type': 'application/json'}

LEARN_GAME = 4500


def load_reward():
    if not os.path.exists("./Rewards"):
        os.mkdir("./Rewards")

    if Path("./Rewards/DQN_rewards_hold.npy").is_file():
        return np.load("./Rewards/DQN_rewards_hold.npy").tolist()
    else:
        return []


def load_score():
    if not os.path.exists("./Scores"):
        os.mkdir("./Scores")

    if Path("./Scores/DQN_scores_hold.npy").is_file():
        return np.load("./Scores/DQN_scores_hold.npy").tolist()
    else:
        return []


def main():
    env = gym.make("tetris-v1", action_mode=1, hold_mode=1)
    network = DQN(hold_mode=1)
    network.load()

    SEED = 50
    env.seed(SEED)
    network.seed(SEED)
    env.action_space.seed(SEED)

    total_rewards = load_reward()
    total_scores = load_score()

    total_games = 0
    total_steps = 0

    # Start
    start_time = time.perf_counter()
    while total_games < LEARN_GAME:
        steps, rewards, scores = network.train(env, episodes=25)
        total_rewards.append(rewards)
        total_scores.append(scores)

        total_games += len(scores)
        total_steps += steps
        network.save()

    np.save("./Rewards/DQN_rewards_hold.npy", np.array(total_rewards))
    np.save("./Scores/DQN_scores_hold.npy", np.array(total_scores))

    # End
    end_time = time.perf_counter()
    # PElapsedTime
    elapsed_time = end_time - start_time
    td = datetime.timedelta(seconds=elapsed_time)
    print(td)
    content_text = f"""
* File Name: {WEIGHT_PATH_HOLD}
* Total Games: {total_games}
* Total Steps: {total_steps}
* Total Times: {td}
"""
    main_content = {'content': content_text}
    requests.post(webhook_url, json.dumps(main_content), headers=headers)

    env.close()


if __name__ == '__main__':
    main()