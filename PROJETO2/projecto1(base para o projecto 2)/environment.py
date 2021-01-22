from operator import add
from base_classes import Game
import gym
from gym import error, spaces, utils
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

class Environment(gym.Env):
    def __init__(self, game_width, game_height, reward_function=default_reward):
        super(Environment, self).__init__()
        self.game_width = game_width
        self.game_height = game_height
        self.game = Game(game_width, game_height)
        self.player = self.game.player
        self.food = self.game.food
        self.get_reward = reward_function
        #we need something like this
        #self.observation_space = spaces.Box(low= np.zeros((s, prop)), high = np.full((s, prop), float('inf')), shape = (s, prop), dtype = np.float32)
        #self.action_space = spaces.Box(low = np.zeros((s+1, ), dtype = int), high = np.array([100]*(s+1)), shape = (s + 1, ), dtype = np.uint8)
        # Example for using image as input:
        #self.observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)#type and shape of the observation (the matrix/screen of tha playing snake)
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,)) #this is a try, i don't know if the agruments are correct
        self.action_space = spaces.Discrete(3) #number of action available - left,right, contiue forward
        #self.action_space = spaces.Box(np.array([0,0,0]), np.array([+1,+1,+1])) #should be better?
    def reset(self):
        self.game = Game(self.game_width, self.game_height)
        self.player = self.game.player
        self.food = self.game.food
        return self.__get_state() #, self.game.crash #shoul return only an array

    def step(self, action):
        self.player.do_move(action, self.player.x, self.player.y, self.game, self.food)
        state = self.__get_state()
        reward = self.get_reward(self)
        done = self.game.crash
        return state, reward, done, {} #the last one are the info, can be left empty as in the library documentation example (gym-soccer)
 
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
                        self.player.x_change == 0 and self.player.y_change == 20 and ((list(map(add, self.player.position[-1],
                                                                                      [-20,
                                                                                       0])) in self.player.position) or
                                                                            self.player.position[-1][0] - 20 < 20)) or (
                        self.player.x_change == -20 and self.player.y_change == 0 and ((list(map(
                    add, self.player.position[-1], [0, -20])) in self.player.position) or self.player.position[-1][
                                                                                 -1] - 20 < 20)) or (
                        self.player.x_change == 20 and self.player.y_change == 0 and (
                        (list(map(add, self.player.position[-1], [0, 20])) in self.player.position) or self.player.position[-1][
                    -1] + 20 >= (self.game.game_height - 20))),  # danger right

            (self.player.x_change == 0 and self.player.y_change == 20 and (
                        (list(map(add, self.player.position[-1], [20, 0])) in self.player.position) or
                        self.player.position[-1][0] + 20 > (self.game.game_width - 20))) or (
                        self.player.x_change == 0 and self.player.y_change == -20 and ((list(map(
                    add, self.player.position[-1], [-20, 0])) in self.player.position) or self.player.position[-1][
                                                                                 0] - 20 < 20)) or (
                        self.player.x_change == 20 and self.player.y_change == 0 and (
                        (list(map(add, self.player.position[-1], [0, -20])) in self.player.position) or self.player.position[-1][
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

        return np.asarray(state)#gym wants np.arrays

    #teh following is an optional method to do
    def render(self, mode='console'):#should remove ! but for now i don't want to use this function
        if mode != 'console':
          raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        #info for debug (i think)