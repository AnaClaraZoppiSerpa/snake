#from stable_baselines import DQN, PPO2, A2C, ACKTR
#from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy #not yet implemented in the new version

from stable_baselines3 import A2C
#from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.a2c import MlpPolicy
#from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env

from gym_snake.envs.GoLeft import GoLeftEnv
from gym_snake.envs.SnakeEnv import SnakeEnv

from stable_baselines3.common.evaluation import evaluate_policy
import time

# Instantiate the env
env = SnakeEnv(440, 440, enable_render=True)
env = make_vec_env(lambda: env, n_envs=1)

model = A2C(MlpPolicy, env, verbose=1, learning_rate=1e-3)
model.learn(total_timesteps=20000, log_interval=200)
# model.save("deepq_breakout")
#
# del model # remove to demonstrate saving and loading
#
# model = DQN.load("deepq_breakout")
print("Teste")
obs = env.reset()
#while True:

#qua sarebbe da fare la parte di run_q_learning (delle metriche e dei plot)
for i in range(1000):
    print(i)
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

print("Evaluating...")
start=time.time()
mean_reward, std_reward  = evaluate_policy(model, env, n_eval_episodes=1000)
end=time.time()
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
print("evaluation duration in s: ", end-start)