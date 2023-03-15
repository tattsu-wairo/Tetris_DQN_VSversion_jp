import gym
import pygame
import gym_tetris
import numpy as np
import os
from gym_tetris.ai.DQN import DQN
import tensorflow as tf
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
def main():
    env = gym.make("tetris-v1", action_mode=1, hold_mode=1)
    network = DQN(gamma=1, epsilon=0, epsilon_min=0,
                  epsilon_decay=0, hold_mode=1)
    network.load()

    obs = env.reset()
    running = True
    display = True

    win_count_1 = 0
    win_count_2 = 0
    game_count = 0
    total_clear_line = 0

    rewards = []
    clear_line_count = [0,0,0,0,0]
    while running:
        action, state = network.choose_action(obs)
        obs, reward, done, info = env.step(action)
        # rewards.append(reward)
        total_clear_line += info["clear_line"]
        clear_line_count[info["clear_line"]]+=1
        if display:
            env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    display = not display
        if done:
            # if info["winner"]=="プレイヤー1":
            #     win_count_1+=1
            # else:
            #     win_count_2+=1
            game_count+=1
            obs = env.reset()
            if game_count == 100:
                running = False
            # print(f"reward: {np.mean(rewards)}")
    # print(f"win_count_1 : {win_count_1}")
    # print(f"win_count_2 : {win_count_2}")
    # win_rate = int(win_count_1/(win_count_1+win_count_2)*100)
    # print(win_rate)
    print(f"total_clear_line: {total_clear_line}")
    print(f"average_clear_line: {total_clear_line / game_count}")
    env.close()


if __name__ == '__main__':
    main()
