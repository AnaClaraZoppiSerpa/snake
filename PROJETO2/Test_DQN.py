from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from gym_snake.envs.SnakeEnv import SnakeEnv
from stable_baselines3.common.evaluation import evaluate_policy
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# auxiliary function to evaluate model
def evaluate(model,env, num_steps=1000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)
        # here, action, rewards and dones are arrays
        # because we are using vectorized env
        obs, rewards, dones, info = env.step(action)
      
        # Stats
        episode_rewards[-1] += rewards[0]
        if dones[0]:
            obs = env.reset()
            episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
  
    return mean_100ep_reward

    ## Compute mean reward for the last 50 episodes
    #mean_50ep_reward = round(np.mean(episode_rewards[-50:]), 1)
    #print("Mean reward:", mean_50ep_reward, "Num episodes:", len(episode_rewards))
  
    #return mean_50ep_reward

def plot_metrics(metrics, filepath=None):
    formatted_dict = {'episodes': [],
                      'metrics': [],
                      'results': []}

    n = len(metrics['episodes'])
    for i in range(n):
        episode = metrics['episodes'][i]
        score = metrics['scores'][i]
        reward = metrics['rewards'][i]

        formatted_dict['episodes'].append(episode)
        formatted_dict['metrics'].append('score')
        formatted_dict['results'].append(score)

        formatted_dict['episodes'].append(episode)
        formatted_dict['metrics'].append('reward')
        formatted_dict['results'].append(reward)

    df_metrics = pd.DataFrame(formatted_dict)
    sns.lineplot(data=df_metrics, x='episodes', y='results', hue='metrics')
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath)

# Instantiate the env
env = SnakeEnv(440, 440, enable_render=True)
env = make_vec_env(lambda: env, n_envs=1)

model = DQN(MlpPolicy, env, verbose=1, learning_rate=1e-3)
model.learn(total_timesteps=20000, log_interval=200,n_eval_episodes=1000)
# model.save("deepq_breakout")
#
# del model # remove to demonstrate saving and loading
#
# model = DQN.load("deepq_breakout")
print("Teste")
obs = env.reset()
#while True:

#Test1
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

#Test2
print("Evaluating...")
start=time.time()
mean_reward, std_reward  = evaluate_policy(model, env, n_eval_episodes=1000)
end=time.time()
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
print("evaluation duration in s: ", end-start)

#Test3
print("Evaluating v2...")
start=time.time()
mean_reward = evaluate(model, env)
end=time.time()
print(f"mean_reward:{mean_reward:.2f}")
print("evaluation v2 duration in s: ", end-start)


#Test4
#metrics = {'episodes': [], 'scores': [], 'rewards': []}

#for j in range(1000):
#    ep_reward=0
#    for i in range(200):
#        action, _states = model.predict(obs)
#        obs, reward, dones, info = env.step(action)
#        env.render()
#        ep_reward+=reward

#    print(f'Game {episode}      Score: {env.game.score}')
#    mean_reward = episode_reward/j
#    metrics['episodes'].append(j)
#    metrics['rewards'].append(mean_reward)
#    metrics['scores'].append(env.game.score)

#plot_metrics(metrics, filepath=None)