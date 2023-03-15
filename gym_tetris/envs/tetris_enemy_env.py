import os
import random

import gym
import numpy as np
import pygame

from gym_tetris.board import Board
from gym_tetris.game import Game
from gym_tetris.view import View
from gym_tetris.ai.DQN import DQN

from pathlib import Path

# WIN_WIDTH = 480
WIN_WIDTH = 1000
WIN_HEIGHT = 526


class TetrisEnemyEnv(gym.Env):
    """プレイヤー2(敵)のクラス"""
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, action_mode=1, hold_mode=0):
        self.view = None
        self.game = None
        self.action_mode = action_mode
        self.hold_mode = hold_mode
        if action_mode == 0:
            # Nothing, Left, Right, Rotate left, Rotate right, Drop, Full Drop, Hold
            self.action_space = gym.spaces.Discrete(8)
        elif action_mode == 1:
            self.action_space = gym.spaces.Tuple((
                gym.spaces.Discrete(10),  # X
                gym.spaces.Discrete(4),  # Rotation
            ))

    def step(self, action):
        """Performs one step/frame in the game and returns the observation, reward and if the game is over."""
        if self.action_mode == 0:
            if action == 1:  # Left
                self.game.board.move_piece(-1)
            elif action == 2:  # Right
                self.game.board.move_piece(1)
            elif action == 3:  # Rotate left
                self.game.board.rotate_piece(-1)
            elif action == 4:  # Rotate right
                self.game.board.rotate_piece(1)
            elif action == 5:  # Drop
                self.game.board.drop_piece()
                self.game.drop_time = self.game.get_drop_speed()
            elif action == 6:  # Full drop
                self.game.board.drop_piece_fully()
            elif action == 7:  # Hold
                self.game.board.hold_piece()
        elif self.action_mode == 1:
            x, rotation, change = action
            if change == 1:  # choose to change function
                self.game.board.hold_piece()
            self.game.board.move_and_drop(x, rotation)

        rows = self.game.tick()
        rows_count = len(rows)
        done = self.game.board.is_game_over()

        reward = 1

        if rows_count == 1:
            reward += 40
        elif rows_count == 2:
            reward += 100
        elif rows_count == 3:
            reward += 300
        elif rows_count == 4:
            reward += 1200

        if done:
            reward -= 5

        return np.array(self.game.board.get_possible_states(self.game.combo)), reward, done, {"clear_line":rows_count}

    def reset(self):
        """Starts a new game."""
        if self.hold_mode == 0:
            self.game = Game(Board(10, 20))
        else:
            self.game = Game(Board(10, 20, 1))
        return np.array(self.game.board.get_possible_states(self.game.combo))

    def close(self):
        """Closes the window."""
        if self.view is not None:
            self.view = None
            pygame.quit()

    def render(self, mode='human', close=False, width=WIN_WIDTH, height=WIN_HEIGHT):
        """Renders the game."""
        if self.view is None:
            pygame.init()
            win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
            font = pygame.font.Font(os.path.join(
                os.path.dirname(__file__), '..', 'assets', 'font.ttf'), 20)
            pygame.display.set_caption("Tetris")
            self.view = View(win, font)

        self.view.draw(self.game,self.enemy_env.game)

    def seed(self, seed=None):
        """Set the random seed for the game."""
        random.seed(seed)
        return [seed]
