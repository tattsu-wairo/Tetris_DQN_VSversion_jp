import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def initialize_plot(mode):
    plt.figure(figsize=(10, 5))
    plt.title('Tetris-Q-learning')
    plt.xlabel('total games')
    if mode==0:
        plt.ylabel('rewards')
    else:
        plt.ylabel('scores')

def DQN_Reward():
    DQN_Rewards = np.load("./Rewards/DQN_rewards.npy")

    DQN_avg = np.mean(DQN_Rewards, axis=1)
    DQN_std = np.std(DQN_Rewards, axis=1)

    initialize_plot(0)

    plt.plot([i * 25 for i in range(200)], DQN_avg,
             label='DQN_Rewards', color='blue')
    plt.fill_between([i * 25 for i in range(200)],
                     DQN_avg+DQN_std, DQN_avg-DQN_std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Graphs/DQN_Reward.png")
    plt.show()
    plt.close()

def DQN_Score():
    DQN_Scores = np.load("./Scores/DQN_scores.npy")

    DQN_avg = np.mean(DQN_Scores, axis=1)
    DQN_std = np.std(DQN_Scores, axis=1)

    initialize_plot(1)

    plt.plot([i * 25 for i in range(200)], DQN_avg,
             label='DQN_Scores', color='green')
    plt.fill_between([i * 25 for i in range(200)],
                     DQN_avg+DQN_std, DQN_avg-DQN_std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Graphs/DQN_Score.png")
    plt.show()
    plt.close()

def DQN_Reward_Hold():
    DQN_Rewards = np.load("./Rewards/DQN_rewards_hold.npy")

    DQN_avg = np.mean(DQN_Rewards, axis=1)
    DQN_std = np.std(DQN_Rewards, axis=1)

    initialize_plot(0)

    plt.plot([i * 25 for i in range(200)], DQN_avg,
             label='DQN_Reward_Hold', color='orange')
    plt.fill_between([i * 25 for i in range(200)],
                     DQN_avg+DQN_std, DQN_avg-DQN_std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Graphs/DQN_Reward_hold.png")
    plt.show()
    plt.close()

def DQN_Score_Hold():
    DQN_Rewards = np.load("./Scores/DQN_scores_hold.npy")

    DQN_avg = np.mean(DQN_Rewards, axis=1)
    DQN_std = np.std(DQN_Rewards, axis=1)

    initialize_plot(1)

    plt.plot([i * 25 for i in range(200)], DQN_avg,
             label='DQN_Score_Hold', color='red')
    plt.fill_between([i * 25 for i in range(200)],
                     DQN_avg+DQN_std, DQN_avg-DQN_std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Graphs/DQN_Score_hold.png")
    plt.show()
    plt.close()

def Q_learning_Reward():
    Q_learning_Rewards = np.load("./Rewards/Q_learning_rewards.npy").transpose()

    Q_learning_avg = np.mean(Q_learning_Rewards, axis=1)
    Q_learning_std = np.std(Q_learning_Rewards, axis=1)

    initialize_plot()
    
    plt.plot([i for i in range(5000)], Q_learning_avg,
             label='Q_learning', color='green')
    plt.fill_between([i for i in range(5000)],
                     Q_learning_avg+Q_learning_std, Q_learning_avg-Q_learning_std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Graphs/Q_learning_Reward.png")
    plt.show()
    plt.close()

def Q_learning_Score():
    Q_learning_Scores = np.load("./Scores/Q_learning_scores.npy").transpose()

    Q_learning_avg = np.mean(Q_learning_Scores, axis=1)
    Q_learning_std = np.std(Q_learning_Scores, axis=1)

    initialize_plot()
    
    plt.plot([i for i in range(5000)], Q_learning_avg,
             label='Q_learning', color='green')
    plt.fill_between([i for i in range(5000)],
                     Q_learning_avg+Q_learning_std, Q_learning_avg-Q_learning_std, facecolor='lightblue')
    plt.legend(loc="best")
    plt.savefig("./Graphs/Q_learning_Score.png")
    plt.show()
    plt.close()

def compare_reward():
    DQN_Rewards = np.load("./Rewards/DQN_rewards.npy")
    DQN_avg = np.mean(DQN_Rewards, axis=1)
    Hold_DQN_Rewards = np.load("./Rewards/DQN_rewards_hold.npy")
    Hold_DQN_avg = np.mean(Hold_DQN_Rewards, axis=1)
    initialize_plot(0)
    plt.plot([i * 25 for i in range(200)], DQN_avg, label='DQN', color='blue')
    plt.plot([i * 25 for i in range(200)],
             Hold_DQN_avg, label='DQN_Hold', color='orange')
    plt.legend(loc="best")
    plt.savefig("./Graphs/compare_reward.png")
    plt.show()
    plt.close()

def compare_score():
    DQN_Rewards = np.load("./Scores/DQN_scores.npy")
    DQN_avg = np.mean(DQN_Rewards, axis=1)
    Hold_DQN_Rewards = np.load("./Scores/DQN_scores_hold.npy")
    Hold_DQN_avg = np.mean(Hold_DQN_Rewards, axis=1)
    initialize_plot(1)
    plt.plot([i * 25 for i in range(200)], DQN_avg, label='DQN', color='green')
    plt.plot([i * 25 for i in range(200)],
             Hold_DQN_avg, label='DQN_Hold', color='red')
    plt.legend(loc="best")
    plt.savefig("./Graphs/compare_score.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    '''
    Plot the trend of Rewards
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--DQN_reward", action="store_true")
    parser.add_argument("--DQN_score", action="store_true")
    parser.add_argument("--DQN_reward_hold", action="store_true")
    parser.add_argument("--DQN_score_hold", action="store_true")
    parser.add_argument("--Q_learning_reward", action="store_true")
    parser.add_argument("--Q_learning_score", action="store_true")
    parser.add_argument("--compare_reward", action="store_true")
    parser.add_argument("--compare_score", action="store_true")
    args = parser.parse_args()

        
    if not os.path.exists("./Graphs"):
        os.mkdir("./Graphs")

    if args.DQN_reward:
        DQN_Reward()
    elif args.DQN_score:
        DQN_Score()
    elif args.DQN_reward_hold:
        DQN_Reward_Hold()
    elif args.DQN_score_hold:
        DQN_Score_Hold()
    elif args.Q_learning_reward:
        Q_learning_Reward()
    elif args.Q_learning_score:
        Q_learning_Score()
    elif args.compare_reward:
        compare_reward()
    elif args.compare_score:
        compare_score()