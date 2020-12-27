from abc import abstractmethod
from random import randint
from collections import deque

import numpy as np


class AgentLFA:
    def __init__(self, N0, gamma, num_state, num_actions, action_space, alpha=0.01):
        """
        Contructor
        Args:
            N0: Initial degree of exploration
            gamma: The discount factor
            num_state: The number of states
            num_actions: The number of actions
        """
        self.epsilon_0 = N0
        self.gamma = gamma
        self.num_state = num_state
        self.num_actions = num_actions

        # self.Q = np.zeros((self.num_state, self.num_actions))
        self.action_space = action_space

        # N(S_t): number of times that state s has been visited
        self.state_counter = [0] * self.num_state

        # N(S, a):  number of times that action a has been selected from state s
        self.state_action_counter = np.zeros((self.num_state, self.num_actions))

        # Usados só no SARSA lambda
        self.E = np.zeros((self.num_state, self.num_actions))
        self.lambda_value = 0

        self.W = {}
        for a in range(num_actions):
            self.W[a] = np.ones(20)
        self.alpha = alpha

        self.action_history = deque([0] * 10, 10)

    def decode_state(self, state):
        """
        Decode a binary representation of a state into its decimal base;

        encoded_state: an array of 0s and 1s representing a binary value

        return: decimal value
        """
        decoded = ''
        for s in state:
            decoded += str(s)

        return int(decoded, 2)

    def decode_action(self, encoded_action):
        if isinstance(encoded_action, np.ndarray):
            return encoded_action.argmax()
        return encoded_action

    """
    Feature Vector
    """

    def feature_vector(self, state):
        # Our state vector is already 11 dimensions, and we are adding 9 more
        feature_list = []
        feature_list.extend(state)

        danger_moving_left = state[2] * state[3]
        danger_moving_right = state[1] * state[4]
        danger_moving_ahead = state[0] * (state[5] + state[6])
        any_danger = state[0] * state[1] * state[2]
        feature_list.extend([danger_moving_left, danger_moving_right, danger_moving_ahead, any_danger])

        moving_food_left = state[3] * state[7]
        moving_food_right = state[4] * state[8]
        moving_food_up = state[5] * state[9]
        moving_food_down = state[6] * state[10]
        moving_to_food = moving_food_left or moving_food_right or moving_food_up or moving_food_down
        feature_list.extend([moving_food_left, moving_food_right, moving_food_up, moving_food_down, moving_to_food])

        return np.array(feature_list)

    """
    State Value Function
    """

    def state_value_function(self, state, a):
        return self.W[a].dot(self.feature_vector(state))

    """
    The Base class that is implemented by
    other classes to avoid the duplicate 'choose_action'
    method
    """

    def choose_action(self, state):
        decoded_state = self.decode_state(state)
        # epsilon_t = N0/(N0 + N(S_t))
        epsilon = self.epsilon_0 / (self.epsilon_0 + self.state_counter[decoded_state])
        if np.random.uniform(0, 1) < epsilon:
            action_index = randint(0, self.num_actions - 1)
        else:
            # action_index = np.argmax(self.Q[state, :])
            actions_values = []
            for a in range(self.num_actions):
                actions_values.append(self.state_value_function(state, a))
            action_index = np.argmax(actions_values)

        action = self.action_space[action_index]
        self.state_counter[decoded_state] += 1
        self.state_action_counter[decoded_state, action_index] += 1

        if self.decode_action(action) != 0 and len(set(self.action_history)) < 3 and np.sum(np.array(self.action_history)) != 0:
            action_index = randint(0, self.num_actions - 1)
            action = self.action_space[action_index]

        self.action_history.append(self.decode_action(action))
        return action

    def update(self, target, state, action):
        self.W[action] = self.W[action] + self.alpha * (
                    target - self.state_value_function(state, action)) * self.feature_vector(state)


class QLearningAgentLFA(AgentLFA):
    pass
    # def update(self, prev_state, next_state, reward, prev_action, next_action):
    #     """
    #     Update the action value function using the Q-Learning update.
    #     Q(S_t, A_t) = Q(S_t, A_t) + alpha(reward + (gamma * Max Q(S_t+1, *) - Q(S_t, A_t))
    #     Args:
    #         prev_state: The previous state
    #         next_state: The next state
    #         reward: The reward for taking the respective action
    #         prev_action: The previous action
    #         next_action: The next action
    #     Returns:
    #         None
    #     """
    #     alpha = 1 / self.state_action_counter[prev_state, prev_action]
    #     predict = self.Q[prev_state, prev_action]
    #     target = reward + self.gamma * np.max(self.Q[next_state, :])
    #     self.Q[prev_state, prev_action] += alpha * (target - predict)


class SARSAAgentLFA(AgentLFA):
    pass
    # def update(self, prev_state, next_state, reward, prev_action, next_action):
    #     """
    #     Update the action value function using the SARSA update.
    #     Q(S, A) = Q(S, A) + alpha(reward + (gamma * Q(S_, A_) - Q(S, A))
    #     Args:
    #         prev_state: The previous state
    #         next_state: The next state
    #         reward: The reward for taking the respective action
    #         prev_action: The previous action
    #         next_action: The next action
    #     Returns:
    #         None
    #     """
    #     alpha = 1 / self.state_action_counter[prev_state, prev_action]
    #     predict = self.Q[prev_state, prev_action]
    #     target = reward + self.gamma * self.Q[next_state, next_action]
    #     self.Q[prev_state, prev_action] += alpha * (target - predict)


class SARSALambdaAgentLFA(AgentLFA):
    def reset_E(self):
        self.E = np.zeros((self.num_state, self.num_actions))

    def update(self, target, state, action):
        #backward view linear TD(lambda)
        delta = target - self.state_value_function(state, action)
        Et = self.gamma * self.lambda_value * self.E[self.decode_state(state), action] + self.feature_vector(state)
        deltaW = self.alpha * delta * Et
        self.W[action] = self.W[action] + deltaW


class MonteCarloAgentLFA(QLearningAgentLFA):
    pass
