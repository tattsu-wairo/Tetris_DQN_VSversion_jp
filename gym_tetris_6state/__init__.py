from gym.envs.registration import register

register(
    id='tetris-v1',
    entry_point='gym_tetris_6state.envs:TetrisEnv',
)
