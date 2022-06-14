import gym
import gym_tetris
import os
import numpy as np

from pathlib import Path
from statistics import mean, median
from gym_tetris.ai.DQN import DQN

def load_reward():
    if not os.path.exists("./Rewards"):
	    os.mkdir("./Rewards")
    else:
        if Path("./Rewards/DQN_rewards_hold.npy").is_file():
            return np.load("./Rewards/DQN_rewards_hold.npy").tolist()
        else:
            return []


def main():
    env = gym.make("tetris-v1", action_mode=1, hold_mode=1)
    network = DQN(epsilon=0.2259657422954923, hold_mode=1)
    network.load()

    SEED = 50
    env.seed(SEED)
    network.seed(SEED)
    env.action_space.seed(SEED)
    
    total_rewards = load_reward()

    running = True
    total_games = 3000
    total_steps = 85199
    
    while total_games < 5000:
        steps, rewards, scores = network.train(env, episodes=25)
        total_rewards.append(rewards)
        
        total_games += len(scores)
        total_steps += steps
        network.save()
        print("==================")
        print("* Total Games: ", total_games)
        print("* Total Steps: ", total_steps)
        print("* Epsilon: ", network.epsilon)
        print("*")
        print("* Average: ", sum(rewards) / len(rewards), "/", sum(scores) / len(scores))
        print("* Median: ", median(rewards), "/", median(scores))
        print("* Mean: ", mean(rewards), "/", mean(scores))
        print("* Min: ", min(rewards), "/", min(scores))
        print("* Max: ", max(rewards), "/", max(scores))
        print("==================")

    np.save("./Rewards/DQN_rewards_hold.npy", np.array(total_rewards))

    env.close()


if __name__ == '__main__':
    main()
