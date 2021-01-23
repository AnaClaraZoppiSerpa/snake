#from stable_baselines import DQN, PPO2, A2C, ACKTR
#from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy

from stable_baselines3 import DQN, A2C
#from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.cmd_util import make_vec_env

from gym_snake.envs.GoLeft import GoLeftEnv
from gym_snake.envs.SnakeEnv import SnakeEnv

# Instantiate the env
env = SnakeEnv(440, 440, enable_render=True)
env = make_vec_env(lambda: env, n_envs=1)

model = A2C(MlpLstmPolicy, env, verbose=1, learning_rate=1e-3)
model.learn(total_timesteps=20000, log_interval=200)
# model.save("deepq_breakout")
#
# del model # remove to demonstrate saving and loading
#
# model = DQN.load("deepq_breakout")
print("Teste")
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()