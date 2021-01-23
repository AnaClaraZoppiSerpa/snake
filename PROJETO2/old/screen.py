import pygame

from environment import Environment


class Screen(object):
    def __init__(self, env:Environment):
        self.env = env
        self.record = 0

    def __display_ui(self):
        score = self.env.game.score
        record = self.record

        myfont = pygame.font.SysFont('Segoe UI', 20)
        myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
        text_score = myfont.render('SCORE: ', True, (0, 0, 0))
        text_score_number = myfont.render(str(score), True, (0, 0, 0))
        text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
        text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
        self.env.game.gameDisplay.blit(text_score, (45, 440))
        self.env.game.gameDisplay.blit(text_score_number, (120, 440))
        self.env.game.gameDisplay.blit(text_highest, (190, 440))
        self.env.game.gameDisplay.blit(text_highest_number, (360, 440))
        self.env.game.gameDisplay.blit(self.env.game.bg, (10, 10))

    def display(self):
        if self.env.game.score > self.record:
            self.record = self.env.game.score

        self.env.game.gameDisplay.fill((255, 255, 255))
        self.__display_ui()
        self.env.player.display_player(self.env.player.position[-1][0], self.env.player.position[-1][1],
                                       self.env.player.food, self.env.game)
        self.env.food.display_food(self.env.food.x_food, self.env.food.y_food, self.env.game)
