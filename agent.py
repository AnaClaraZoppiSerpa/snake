from abc import abstractmethod
from random import randint
from collections import deque

import numpy as np

class Agent:
    def __init__(self, N0, gamma, num_state, num_actions, action_space):
        """
        Contructor
        Args:
            N0: Initial degree of exploration
            gamma: The discount factor
            num_state: The number of states
            num_actions: The number of actions
            action_space: To call the random action
        """
        self.epsilon_0 = N0
        self.gamma = gamma
        self.num_state = num_state
        self.num_actions = num_actions

        self.Q = np.zeros((self.num_state, self.num_actions))
        self.action_space = action_space

        # N(S_t): number of times that state s has been visited
        self.state_counter = [0] * self.num_state

        # N(S, a):  number of times that action a has been selected from state s
        self.state_action_counter = np.zeros((self.num_state, self.num_actions))

        # Usados só no SARSA lambda
        self.E = np.zeros((self.num_state, self.num_actions))
        self.lambda_value = 0

        self.action_history = deque([0] * 10, 10)

    def decode_action(self, encoded_action):
        if isinstance(encoded_action, np.ndarray):
            return encoded_action.argmax()
        return encoded_action

    """
    The Base class that is implemented by
    other classes to avoid the duplicate 'choose_action'
    method
    """
    def choose_action(self, state):
        # epsilon_t = N0/(N0 + N(S_t))
        epsilon = self.epsilon_0 / (self.epsilon_0 + self.state_counter[state])
        if np.random.uniform(0, 1) < epsilon:
            action_index = randint(0, self.num_actions-1)
        else:
            action_index = np.argmax(self.Q[state, :])

        action = self.action_space[action_index]
        self.state_counter[state] += 1
        self.state_action_counter[state, action_index] += 1

        return action

    @abstractmethod
    def update(self, prev_state, next_state, reward, prev_action, next_action):
        pass


class QLearningAgent(Agent):
    def update(self, prev_state, next_state, reward, prev_action, next_action):
        """
        Update the action value function using the Q-Learning update.
        Q(S_t, A_t) = Q(S_t, A_t) + alpha(reward + (gamma * Max Q(S_t+1, *) - Q(S_t, A_t))
        Args:
            prev_state: The previous state
            next_state: The next state
            reward: The reward for taking the respective action
            prev_action: The previous action
            next_action: The next action
        Returns:
            None
        """
        alpha = 1 / self.state_action_counter[prev_state, prev_action]
        predict = self.Q[prev_state, prev_action]
        target = reward + self.gamma * np.max(self.Q[next_state, :])
        self.Q[prev_state, prev_action] += alpha * (target - predict)

class SARSAAgent(Agent):
    def update(self, prev_state, next_state, reward, prev_action, next_action):
        """
        Update the action value function using the SARSA update.
        Q(S, A) = Q(S, A) + alpha(reward + (gamma * Q(S_, A_) - Q(S, A))
        Args:
            prev_state: The previous state
            next_state: The next state
            reward: The reward for taking the respective action
            prev_action: The previous action
            next_action: The next action
        Returns:
            None
        """
        alpha = 1 / self.state_action_counter[prev_state, prev_action]
        predict = self.Q[prev_state, prev_action]
        target = reward + self.gamma * self.Q[next_state, next_action]
        self.Q[prev_state, prev_action] += alpha * (target - predict)

class SARSALambdaAgent(Agent):
    def reset_E(self):
        self.E = np.zeros((self.num_state, self.num_actions))

    def update(self, prev_state, next_state, reward, prev_action, next_action):
        delta = reward + self.gamma*self.Q[next_state, next_action] - self.Q[prev_state, prev_action]
        # self.E[prev_state, prev_action] = self.gamma * self.lambda_value * self.E[prev_state, prev_action] + 1
        # alpha = 1 / self.state_action_counter[prev_state, prev_action]
        # self.Q[prev_state, prev_action] = self.Q[prev_state, prev_action] + alpha*delta*self.E[prev_state, prev_action]

        self.E[prev_state, prev_action] += 1

        #alpha = 1 / self.state_action_counter[prev_state, prev_action]
        alpha = 0.01

        for s in range(self.num_state):
            for a in range(self.num_actions):
                self.Q[prev_state, prev_action] += alpha * delta * self.E[s, a]
                self.E[s, a] = self.gamma * self.lambda_value * self.E[s, a]

class MonteCarloAgent(QLearningAgent):
    def choose_action(self, state):
        # epsilon_t = N0/(N0 + N(S_t))
        epsilon = self.epsilon_0 / (self.epsilon_0 + self.state_counter[state])
        if np.random.uniform(0, 1) < epsilon:
            action_index = randint(0, self.num_actions - 1)
        else:
            action_index = np.argmax(self.Q[state, :])

        action = self.action_space[action_index]
        self.state_counter[state] += 1
        self.state_action_counter[state, action_index] += 1

        if self.decode_action(action) != 0 and len(set(self.action_history)) < 3 and np.sum(
                np.array(self.action_history)) != 0:
            action_index = randint(0, self.num_actions - 1)
            action = self.action_space[action_index]

        self.action_history.append(self.decode_action(action))
        return action
