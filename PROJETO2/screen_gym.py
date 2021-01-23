import pygame
import time


class Screen(object):
    def __init__(self, game, player, food):
        check_errors = pygame.init()
        #if check_errors[1] > 0:
        #    print("(!) Had {0} initializing errors, exiting...".format(check_errors[1]))
        #    sys.exit(-1)
        #else:
        #    print("(+) PyGame successfully initialized!")

        pygame.font.init()

        self.game = game
        self.player = player
        self.food = food
        self.record = 0

    def __display_ui(self):
        score = self.game.score
        record = self.record

        myfont = pygame.font.SysFont('Segoe UI', 20)
        myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
        text_score = myfont.render('SCORE: ', True, (0, 0, 0))
        text_score_number = myfont.render(str(score), True, (0, 0, 0))
        text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
        text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
        self.game.gameDisplay.blit(text_score, (45, 440))
        self.game.gameDisplay.blit(text_score_number, (120, 440))
        self.game.gameDisplay.blit(text_highest, (190, 440))
        self.game.gameDisplay.blit(text_highest_number, (360, 440))
        self.game.gameDisplay.blit(self.game.bg, (10, 10))

    def display(self):
        if self.game.score > self.record:
            self.record = self.game.score

        self.game.gameDisplay.fill((255, 255, 255))
        self.__display_ui()
        self.player.display_player(self.player.position[-1][0], self.player.position[-1][1],
                                       self.player.food, self.game)
        self.food.display_food(self.food.x_food, self.food.y_food, self.game)

    def quit_game(self):
        pygame.quit()
