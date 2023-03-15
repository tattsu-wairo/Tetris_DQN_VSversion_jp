import gym
import pygame
import gym_tetris
import numpy as np
import os

from gym_tetris.ai.DQN import DQN

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def main():
    env = gym.make("tetris-v1", action_mode=1)
    network = DQN(gamma=1, epsilon=0, epsilon_min=0, epsilon_decay=0)
    network.load()

    obs = env.reset()
    running = True
    display = True

    rewards = []
    while running:
        action, state = network.choose_action(obs)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        if display:
            env.render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    display = not display
        if done:
            obs = env.reset()
            print(f"reward: {np.mean(rewards)}")

    env.close()


if __name__ == '__main__':
    main()
