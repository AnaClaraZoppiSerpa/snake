from operator import add
from base_classes import Game

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

class Environment:
    def __init__(self, game_width, game_height, reward_function=default_reward):
        self.game_width = game_width
        self.game_height = game_height
        self.game = Game(game_width, game_height)
        self.player = self.game.player
        self.food = self.game.food
        self.get_reward = reward_function

    def reset(self):
        self.game = Game(self.game_width, self.game_height)
        self.player = self.game.player
        self.food = self.game.food
        return self.__get_state(), self.game.crash

    def step(self, action):
        self.player.do_move(action, self.player.x, self.player.y, self.game, self.food)
        state = self.__get_state()
        reward = self.get_reward(self)
        done = self.game.crash
        return state, reward, done

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

        return state