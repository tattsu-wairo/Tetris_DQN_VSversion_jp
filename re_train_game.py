import gym
import gym_tetris
import os
import numpy as np

from pathlib import Path
from statistics import mean, median
from gym_tetris.ai.DQN import DQN
from keras.callbacks import ModelCheckpoint

import time
import datetime
import tensorflow as tf

import json
import requests

import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

'''Discordに進捗を送るためのやつ'''
# Discordでサーバー設定→連携サービス→ウェブフックを作成して、ウェブフックURLを取得
webhook_url = '' # 取得したウェブフックURLを''に貼り付け
headers = {'Content-Type': 'application/json'} # 特にいじらなくてOK
content_text = '' # Discordのチャットとして送られる文
main_content = {'content': content_text} # 文章が更新されたらチャットを送る前にこのコードを走るようにする

# チャットを送りたいタイミングで以下のコードを挿入
# requests.post(webhook_url, json.dumps(main_content), headers=headers)

LEARN_GAME_STEP = 100000 # 学習ステップ数
MODEL_A = os.path.join(os.path.dirname(__file__), 'MODEL_A_50000_capacity_50000_BatchNormarization_True')
MODEL_B = os.path.join(os.path.dirname(__file__), 'MODEL_B_50000_capacity_50000_BatchNormarization_True')
MODEL_C = os.path.join(os.path.dirname(__file__), 'MODEL_C_vs_A_50000_capacity_50000_BatchNormarization_True')
MODEL_D = os.path.join(os.path.dirname(__file__), 'MODEL_D_vs_A_50000_capacity_50000_BatchNormarization_True')

REWARDS_PATH = "./Rewards/DQN_rewards_hold_50000_test.npy"
SCORES_PATH = "./Scores/DQN_rewards_hold_50000_test.npy"

def load_reward():
    if not os.path.exists("./Rewards"):
        os.mkdir("./Rewards")

    if Path(REWARDS_PATH).is_file():
        return np.load(REWARDS_PATH).tolist()
    else:
        return []


def load_score():
    if not os.path.exists("./Scores"):
        os.mkdir("./Scores")

    if Path(SCORES_PATH).is_file():
        return np.load(SCORES_PATH).tolist()
    else:
        return []


def main():
    env = gym.make("tetris-v1", action_mode=1, hold_mode=1)
    network = DQN(hold_mode=1)

    SEED = 50
    env.seed(SEED)
    network.seed(SEED)
    env.action_space.seed(SEED)

    total_rewards = load_reward()
    total_scores = load_score()

    total_games = 0
    total_steps = 0

    running = True

    is_save_modelA = False

    # Start
    start_time = time.perf_counter()
    obs = env.reset()
    previous_state = env.game.board.get_info([],0)
    done = False
    while running:
        steps, rewards, scores = network.train(env)
        total_games += 1
        total_steps += steps
        if not is_save_modelA:
            if total_steps >= int(LEARN_GAME_STEP/2):
                # 1人プレイの環境で学習させる場合はモデルA
                # 対戦プレイの環境で学習させる場合はモデルD
                tf.keras.models.save_model(network.model, MODEL_A)
                is_save_modelA = not is_save_modelA

        print("total_steps : " + str(total_steps) + "\n" +
              "total_games : " + str(total_games) + "\n" + "\033[2A", end="")
        if total_steps >= LEARN_GAME_STEP:
            running = False

    print("\033[1B")
    print(f"total_games : {total_games}")
    print(f"total_steps : {total_steps}")

    # 1人プレイの環境で学習させる場合はモデルB
    # 対戦プレイの環境で学習させる場合はモデルC
    tf.keras.models.save_model(network.model,MODEL_B)

    # print("==================")
    # print("* Total Games: ", total_games)
    # print("* Total Steps: ", i+1)
    # print("* Epsilon: ", network.epsilon)
    # print("*")
    # print("Total Time: ", td)
    # print("==================")

    np.save(REWARDS_PATH, np.array(total_rewards))
    np.save(SCORES_PATH, np.array(total_scores))
    # End
    end_time = time.perf_counter()
    # PElapsedTime
    elapsed_time = end_time - start_time
    td = datetime.timedelta(seconds=elapsed_time)
    print("学習時間: ",td)
    env.close()


if __name__ == '__main__':
    main()
