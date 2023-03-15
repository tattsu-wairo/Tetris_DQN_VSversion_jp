import gym
import pygame
import gym_tetris_6state

from Pierre_Dellacherie import Agent


def main():
    env = gym.make("tetris-v1", action_mode=1)
    testing_agent = Agent(env)

    obs = env.reset()
    running = True
    display = True
    while running:
        action = testing_agent.choose_action(obs)
        obs, reward, done, _ = testing_agent.env.step(action)

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

    env.close()


if __name__ == '__main__':
    main()