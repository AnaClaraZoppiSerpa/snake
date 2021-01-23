from stable_baselines3 import DQN, A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

from gym_snake.envs.SnakeEnv import SnakeEnv

# enable_render=True turns on the display
snake_env = SnakeEnv(440, 440, enable_render=False)
env = make_vec_env(lambda: snake_env, n_envs=1)

model = A2C(MlpPolicy, env, verbose=1, learning_rate=1e-3)
model.learn(total_timesteps=20000, log_interval=200)

# Print rewards and scores for each episode
print(snake_env.results)

snake_env = SnakeEnv(440, 440, enable_render=True)
env = make_vec_env(lambda: snake_env, n_envs=1)
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()