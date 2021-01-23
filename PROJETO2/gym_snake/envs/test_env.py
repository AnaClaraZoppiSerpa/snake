from gym_snake.envs.SnakeEnv import SnakeEnv
from stable_baselines.common.env_checker import check_env

env = SnakeEnv(440, 440)

# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)