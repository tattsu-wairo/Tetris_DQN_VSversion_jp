import gym

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
from gym_tetris.envs.tetris_env import WEIGHT_PATH_HOLD_ENEMY

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

webhook_url = 'https://discord.com/api/webhooks/1049978183665066064/gN0LPx_z9Ile4Q_Y3FkkWMobDnS58OS8bFwplSq_u0pZIZrmJYFlv32JMCyj3ZscwYhY'
main_content = {'content': '送るテキスト'}
headers = {'Content-Type': 'application/json'}

LEARN_GAME = 4750
def load_reward():
    if not os.path.exists("./Rewards"):
        os.mkdir("./Rewards")

    if Path("./Rewards/DQN_rewards_hold_vs.npy").is_file():
        return np.load("./Rewards/DQN_rewards_hold_vs.npy").tolist()
    else:
        return []


def load_score():
    if not os.path.exists("./Scores"):
        os.mkdir("./Scores")

    if Path("./Scores/DQN_scores_hold_vs.npy").is_file():
        return np.load("./Scores/DQN_scores_hold_vs.npy").tolist()
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

    running = True
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
        content_text = f"""
        ==================
        * {WEIGHT_PATH_HOLD} vs {WEIGHT_PATH_HOLD_ENEMY}
        * Total Games: {total_games}
        * Total Steps: {total_steps}
        * Epsilon: {network.epsilon}
        *
        * Average: {sum(rewards) / len(rewards)} / {sum(scores) / len(scores)}
        * Median: {median(rewards)} / {median(scores)}
        * Mean: {mean(rewards)} / {mean(scores)}
        * Min: {min(rewards)} / {min(scores)}
        * Max: {max(rewards)} / {max(scores)}
        =================="""
        main_content = {'content': content_text}
        # if total_games % 100 == 0:
        #     try:
        #         requests.post(webhook_url, json.dumps(main_content), headers=headers)
        #     except:
        #         pass

    np.save("./Rewards/DQN_rewards_hold_vs.npy", np.array(total_rewards))
    np.save("./Scores/DQN_scores_hold_vs.npy", np.array(total_scores))

    # End
    end_time = time.perf_counter()
    # PElapsedTime
    elapsed_time = end_time - start_time
    td = datetime.timedelta(seconds=elapsed_time)
    print(td)
    content_text = f"""
* Player1(training): {WEIGHT_PATH_HOLD}
* vs
* Player2(enemy): {WEIGHT_PATH_HOLD_ENEMY}
* Total Games: {total_games}
* Total Steps: {total_steps}
* Total Times: {td}
"""
    main_content = {'content': content_text}
    requests.post(webhook_url, json.dumps(main_content), headers=headers)

    env.close()


if __name__ == '__main__':
    main()
