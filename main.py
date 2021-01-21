import argparse
import distutils.util
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygame
import seaborn as sns

from agent import Agent, QLearningAgent
from environment import Environment
from screen import Screen


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


def decode_state(encoded_state):
    """
    encoded_state: an array of 0s and 1s representing a binary value

    return: decimal value
    """
    decoded = ''
    for s in encoded_state:
        decoded += str(s)

    return int(decoded, 2)


def decode_action(encoded_action):
    if isinstance(encoded_action, np.ndarray):
        return encoded_action.argmax()
    return encoded_action


def run(agent: Agent, episodes, display, speed): #it is used from any agent, so i must insert an option about the agent is DQN, at least for the Inizialize_game part (should be inserted inside the init of the agent?)
    pygame.init()

    env = Environment(440, 440)
    screen = Screen(env)

    episode = 0
    metrics = {'episodes': [],
               'scores': [],
               'rewards': []}

    while episode < episodes:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        if display:
            screen.display()

        # the net is not trained, then we should do something "special" (initialize_game of the base repository) before to act following DQN
        state1, done = env.reset()
        state1 = decode_state(state1)
        action1 = agent.choose_action(state1) 
        episode_reward = 0
        while not done:
            # Getting the next state, reward
            state2, reward, done = env.step(action1)
            state2 = decode_state(state2)
            # Choosing the next action
            action2 = agent.choose_action(state2)

            # Learning the Q-value
            #decoded_action1 = decode_action(action1)
            #decoded_action2 = decode_action(action2)
            #agent.update(state1, state2, reward, decoded_action1, decoded_action2)

            state1 = state2
            action1 = action2
            episode_reward += reward

            if display:
                screen.display()
                pygame.time.wait(speed)

        episode += 1
        print(f'Game {episode}      Score: {env.game.score}')

        mean_reward = episode_reward/episodes
        metrics['episodes'].append(episode)
        metrics['rewards'].append(mean_reward)
        metrics['scores'].append(env.game.score)

        #probabilmente è paragonabile all'agent.update degli altri metodi, quindi bisognerebbe riadattare questa cosa
        #model_weights = agent.state_dict()
        #torch.save(model_weights, params["weights_path"]) #should be added beacuse it save the weights during the train

    return metrics

if __name__ == '__main__':
    # Set options to activate or deactivate the game view, and its speed
    pygame.font.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--display", nargs='?', type=distutils.util.strtobool, default=True)
    parser.add_argument("--speed", nargs='?', type=int, default=50)
    parser.add_argument("--episodes", nargs='?', type=int, default=150)
    parser.add_argument("--figure", nargs='?', type=str, default=None)

    args = parser.parse_args()
    print("Args", args)

    # Defining all the required parameters
    N0 = 1
    gamma = 1

    action_space = np.eye(3)
    num_actions = 3
    num_state = 2 ** 11
    qLearningAgent = QLearningAgent(N0, gamma, num_state, num_actions, action_space)

    ''' #new
    agent = DQNAgent(params)
    agent = agent.to(DEVICE)
    agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=params['learning_rate'])
    '''

    metrics = run(qLearningAgent, episodes=args.episodes, speed=args.speed, display=args.display)
    plot_metrics(metrics, filepath=args.figure)
    time.time()


