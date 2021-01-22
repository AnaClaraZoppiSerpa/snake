import gym
import pygame
from gym import spaces
from gym.utils import seeding
from operator import add
from base_classes import Game
from screen_gym import Screen
import numpy as np

def default_reward(env):
    """
    Return the reward.
    The reward is:
        -10 when Snake crashes.
        +10 when Snake eats food
        0 otherwise
    """
    reward = 0
    if env.game.crash:
        reward = -10
    elif env.player.eaten:
        reward = 10

    return reward

class SnakeEnv(gym.Env):

    def __init__(self, game_width, game_height, reward_function=default_reward, enable_render=True):
        self.game_width = game_width
        self.game_height = game_height
        self.game = Game(game_width, game_height)
        self.player = self.game.player
        self.food = self.game.food
        self.get_reward = reward_function
        self.enable_render = enable_render
        self.screen = Screen(self.game, self.player, self.food)

        num_state = 2 ** 11
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(num_state)
        self.count = 0

        #self.observation_space = spaces.Box(low=0, high=1, shape=(11,)) #this is a try, i don't know if the agruments are correct
        #self.action_space = spaces.Discrete(3) #number of action available - left,right, contiue forward
        ##self.action_space = spaces.Box(np.array([0,0,0]), np.array([+1,+1,+1])) #should be better?


        # Simulation related variables.
        # self.seed()
        # self.reset()


    # def __del__(self):
    #     if self.enable_render:
    #         self.screen.quit_game()
    #         # quit()

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def step(self, action):
        action_space = np.eye(3)
        action = action_space[action]
        self.player.do_move(action, self.player.x, self.player.y, self.game, self.food)
        state = self.__get_state()
        reward = self.get_reward(self)
        done = self.game.crash

        info = {}

        # else:
        #     print("State", state, "Reward", reward, "Action", action)

        self.count += 1
        if self.count == 1000:
            done = True

        # if done:
        #     print("Fim")
        if self.enable_render:
            self.screen.display()
            pygame.time.wait(1)
        return state, reward, done, info

    def reset(self):
        self.game = Game(self.game_width, self.game_height)
        self.player = self.game.player
        self.food = self.game.food

        self.screen.game = self.game
        self.screen.player = self.player
        self.screen.food = self.food
        self.count = 0
        return self.__get_state()

    def render(self, mode="human", close=False):
        if self.enable_render:
            self.screen.display()
            pygame.time.wait(1)

    def decode_state(self, encoded_state):
        """
        encoded_state: an array of 0s and 1s representing a binary value

        return: decimal value
        """
        decoded = ''
        for s in encoded_state:
            decoded += str(s)

        return int(decoded, 2)

    def __get_state(self):
        """
        Return the state.
        The state is a numpy array of 11 values, representing:
            - Danger 1 OR 2 steps ahead
            - Danger 1 OR 2 steps on the right
            - Danger 1 OR 2 steps on the left
            - Snake is moving left
            - Snake is moving right
            - Snake is moving up
            - Snake is moving down
            - The food is on the left
            - The food is on the right
            - The food is on the upper side
            - The food is on the lower side
        """
        state = [
            (self.player.x_change == 20 and self.player.y_change == 0 and (
                    (list(map(add, self.player.position[-1], [20, 0])) in self.player.position) or
                    self.player.position[-1][0] + 20 >= (self.game.game_width - 20))) or (
                    self.player.x_change == -20 and self.player.y_change == 0 and (
                    (list(map(add, self.player.position[-1], [-20, 0])) in self.player.position) or
                    self.player.position[-1][0] - 20 < 20)) or (
                    self.player.x_change == 0 and self.player.y_change == -20 and (
                    (list(map(add, self.player.position[-1], [0, -20])) in self.player.position) or
                    self.player.position[-1][-1] - 20 < 20)) or (
                    self.player.x_change == 0 and self.player.y_change == 20 and (
                    (list(map(add, self.player.position[-1], [0, 20])) in self.player.position) or
                    self.player.position[-1][-1] + 20 >= (self.game.game_height - 20))),  # danger straight

            (self.player.x_change == 0 and self.player.y_change == -20 and (
                    (list(map(add, self.player.position[-1], [20, 0])) in self.player.position) or
                    self.player.position[-1][0] + 20 > (self.game.game_width - 20))) or (
                    self.player.x_change == 0 and self.player.y_change == 20 and (
                        (list(map(add, self.player.position[-1],
                                  [-20,
                                   0])) in self.player.position) or
                        self.player.position[-1][0] - 20 < 20)) or (
                    self.player.x_change == -20 and self.player.y_change == 0 and ((list(map(
                add, self.player.position[-1], [0, -20])) in self.player.position) or self.player.position[-1][
                                                                                       -1] - 20 < 20)) or (
                    self.player.x_change == 20 and self.player.y_change == 0 and (
                    (list(map(add, self.player.position[-1], [0, 20])) in self.player.position) or
                    self.player.position[-1][
                        -1] + 20 >= (self.game.game_height - 20))),  # danger right

            (self.player.x_change == 0 and self.player.y_change == 20 and (
                    (list(map(add, self.player.position[-1], [20, 0])) in self.player.position) or
                    self.player.position[-1][0] + 20 > (self.game.game_width - 20))) or (
                    self.player.x_change == 0 and self.player.y_change == -20 and ((list(map(
                add, self.player.position[-1], [-20, 0])) in self.player.position) or self.player.position[-1][
                                                                                       0] - 20 < 20)) or (
                    self.player.x_change == 20 and self.player.y_change == 0 and (
                    (list(map(add, self.player.position[-1], [0, -20])) in self.player.position) or
                    self.player.position[-1][
                        -1] - 20 < 20)) or (
                    self.player.x_change == -20 and self.player.y_change == 0 and (
                    (list(map(add, self.player.position[-1], [0, 20])) in self.player.position) or
                    self.player.position[-1][-1] + 20 >= (self.game.game_height - 20))),  # danger left

            self.player.x_change == -20,  # move left
            self.player.x_change == 20,  # move right
            self.player.y_change == -20,  # move up
            self.player.y_change == 20,  # move down
            self.food.x_food < self.player.x,  # food left
            self.food.x_food > self.player.x,  # food right
            self.food.y_food < self.player.y,  # food up
            self.food.y_food > self.player.y  # food down
        ]

        for i in range(len(state)):
            if state[i]:
                state[i] = 1
            else:
                state[i] = 0

        return self.decode_state(state)
        #return np.asarray(state)
