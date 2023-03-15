import gym
import pygame
import gym_tetris
import numpy as np
import os
from gym_tetris.ai.DQN import DQN
import tensorflow as tf
import time
import sys
from pathlib import Path

# モデルのファイルを指定
MODEL_A = os.path.join(os.path.dirname(__file__), 'MODEL_A_50000_03')
MODEL_B = os.path.join(os.path.dirname(__file__), 'MODEL_B_50000_03')
MODEL_C = os.path.join(os.path.dirname(__file__), 'MODEL_C_vs_A_50000_03')
MODEL_D = os.path.join(os.path.dirname(__file__), 'MODEL_D_vs_A_50000_03')

'''
MODEL_A_50000_03, MODEL_B_50000_03, MODEL_C_vs_A_50000_03, MODEL_D_vs_A_50000_03
上記のものが実際に論文で使用したモデル
'''

# Player1として動かすモデルを指定
MODEL_PLAYER = MODEL_A
# 動かすゲーム数を指定
PLAY_GAME_EPISODES = 10
# 動かすステップ数を指定
PLAY_GAME_STEPS = 100000


def main():
    env = gym.make("tetris-v1", action_mode=1, hold_mode=1)
    network = DQN(gamma=1, epsilon=0, epsilon_min=0,
                  epsilon_decay=0, hold_mode=1)
    if Path(MODEL_PLAYER).is_dir():
        network.model = tf.keras.models.load_model(MODEL_PLAYER)
    else:
        sys.exit()
    print("\n")
    print(network.model)

    obs = env.reset()
    running = True
    display = True

    win_count_1 = 0  # Player1
    win_count_2 = 0  # Player2
    game_count = 0  # ゲーム数
    total_clear_line = 0  # Player1の総合消去ライン数
    total_clear_line_enemy = 0  # Player2の総合消去ライン数
    total_steps = 0  # 総合ステップ数
    attack1 = 0  # Player1が送る火力の数
    attack2 = 0  # Player2が送る火力の数
    total_attack_player1 = 0  # Player1が送った火力の合計数
    total_attack_player2 = 0  # Player2が送った火力の合計数
    max_clear_line = 0  # Player1の最大消去ライン数
    clear_line = 0  # Player1の1ゲーム中のクリアライン数
    clear_line_enemy = 0  # Player2の1ゲーム中のクリアライン数

    rewards = []
    # 各同時消しの回数リスト[0LINE,Single,Double,Triple,Tetris]
    clear_line_count = [0, 0, 0, 0, 0]
    while running:
        action, state = network.choose_action(obs)
        obs, reward, done, info = env.step(action)
        total_steps += 1
        attack1 = sum(info["attack1"])
        attack2 = sum(info["attack2"])
        total_attack_player1 += attack1
        total_attack_player2 += attack2
        clear_line += info["clear_line"]
        clear_line_enemy += info["clear_line_enemy"]
        total_clear_line += info["clear_line"]
        total_clear_line_enemy += info["clear_line_enemy"]
        clear_line_count[info["clear_line"]] += 1
        if display:
            env.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        display = not display
        if done:
            if info["winner"] == "プレイヤー1":
                win_count_1 += 1
            else:
                win_count_2 += 1
            if max_clear_line < clear_line:
                max_clear_line = clear_line
            clear_line = 0
            clear_line_enemy = 0
            game_count += 1
            running = False
            obs = env.reset()
        if total_steps % 1000 == 0:
            print(clear_line)
        if game_count == PLAY_GAME_EPISODES:
            running = False
        # if total_steps >= PLAY_GAME_STEPS:
        #     running = False

    # 1ゲームが終わる前に終了した場合の処理
    if game_count == 0:
        # ゲームカウントを1にする
        game_count = 1

    print("=====result=====")
    print(f"win_count_1 : {win_count_1}")
    print(f"win_count_2 : {win_count_2}")
    print("==========")
    print(f"total_clear_line_1: {total_clear_line}")
    print(f"average_clear_line_1: {total_clear_line/game_count}")
    print(
        f"Clear_per_{total_steps}_piece_1: {total_clear_line / total_steps*100}")
    print("==========")
    print(f"total_clear_line_2: {total_clear_line_enemy}")
    print(f"average_clear_line_2: {total_clear_line_enemy/game_count}")
    print(
        f"Clear_per_{total_steps}_piece_2: {total_clear_line_enemy / total_steps*100}")
    print("==========")

    print(f"single: {clear_line_count[1]}")
    print(f"double: {clear_line_count[2]}")
    print(f"triple: {clear_line_count[3]}")
    print(f"tetris: {clear_line_count[4]}")
    print(f"MAX_CLEAR_LINE: {max_clear_line}")
    print(f"total_attack1: {total_attack_player1}")
    print(f"total_attack2: {total_attack_player2}")
    print(
        f"Attack_per_{total_steps}_piece1: {(total_attack_player1/total_steps*100)}")
    print(
        f"Attack_per_{total_steps}_piece2: {(total_attack_player2/total_steps*100)}")
    print("==========")
    env.close()


if __name__ == '__main__':
    main()
