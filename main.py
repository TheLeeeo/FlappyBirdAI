import neat
import pygame
import os
import random
import visualize

SCREEN_SIZE = (800, 600)
screen = pygame.display.set_mode(SCREEN_SIZE)

BackgroundImage = pygame.image.load("BackgroundImage.png").convert()
BackgroundImage = pygame.transform.scale(BackgroundImage, SCREEN_SIZE)

BIRD_SIZE = (50, 50)
BirdImage = pygame.image.load("BirdImage.png").convert().convert_alpha()
#BirdImage.set_colorkey((255, 0, 255))
BirdImage = pygame.transform.scale(BirdImage, BIRD_SIZE)

PIPE_SIZE = (100, 600)
PipeImage = pygame.image.load("PipeImage.png").convert().convert_alpha()
#BirdImage.set_colorkey((0, 0, 0))
PipeImage = pygame.transform.scale(PipeImage, PIPE_SIZE)

FlippedPipeImage = pygame.transform.flip(PipeImage, False, True)

NUMBER_OF_PIPES = 3
distance_between_pipes = (SCREEN_SIZE[0] + PIPE_SIZE[0]) / NUMBER_OF_PIPES

pygame.init()
pygame.display.set_caption("Burd game")

clock = pygame.time.Clock()

font = pygame.font.SysFont('comicsans', 64)
smallFont = pygame.font.SysFont('comicsans', 32)

SCORE_TARGET = 20


def play_winning_net(net):
    bird = Bird(net)

    active_pipes = []
    for i in range(NUMBER_OF_PIPES):
        active_pipes.append(Pipe(SCREEN_SIZE[0] + (SCREEN_SIZE[0] + PIPE_SIZE[0]) * i / NUMBER_OF_PIPES))

    next_pipe = 0

    run = True
    while run:  # frame loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        clock.tick(30)
        screen.blit(BackgroundImage, (0, 0))

        for pipe in active_pipes:
            pipe.move()

            if pipe.lowerRect.topright[0] < 50 and pipe.passed is False:  # pipe is passed
                next_pipe += 1
                if next_pipe == NUMBER_OF_PIPES:
                    next_pipe = 0
                pipe.passed = True

            if pipe.lowerRect.topright[0] < 0:
                pipe.reset()

            pipe.draw()

        bird.process_1(active_pipes[next_pipe])  # distance to top and bottom
        # bird.process_2(active_pipes[next_pipe])  # bird y and pipe y

        bird.move()

        if bird.rect.topleft[1] < 0 or bird.rect.bottomleft[1] > SCREEN_SIZE[1]:  # Bird out of bounds
            run = False

        elif bird.rect.colliderect(active_pipes[next_pipe].upperRect) or \
                bird.rect.colliderect(active_pipes[next_pipe].lowerRect):
            run = False
        else:
            bird.draw()

        pygame.display.flip()


def eval_genomes(genome_iterable, config):
    birds = []

    for genome_id, genome in genome_iterable:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        birds.append(Bird(net, genome))

    active_pipes = []
    for i in range(NUMBER_OF_PIPES):
        active_pipes.append(Pipe(SCREEN_SIZE[0] + (SCREEN_SIZE[0] + PIPE_SIZE[0]) * i / NUMBER_OF_PIPES))
        
    next_pipe = 0
        
    run = True
    while run and len(birds) > 1:  # frame loop
        for event in pygame.event.get():  #
            if event.type == pygame.QUIT:  #
                run = False  #

        clock.tick(60)  #
        screen.blit(BackgroundImage, (0, 0))  #

        for pipe in active_pipes:
            pipe.move()

            if pipe.lowerRect.topright[0] < 50 and pipe.passed is False:  # pipe is passed
                next_pipe += 1
                if next_pipe == NUMBER_OF_PIPES:
                    next_pipe = 0
                pipe.passed = True

                for bird in birds:
                    bird.genome.fitness += 1

                if birds[0].genome.fitness >= SCORE_TARGET:
                    run = False

            if pipe.lowerRect.topright[0] < 0:
                pipe.reset()

            pipe.draw()  #

        for i, bird in enumerate(birds):  # every single bird

            bird.process_1(active_pipes[next_pipe])  # distance to top and bottom
            # bird.process_2(active_pipes[next_pipe])  # bird y and pipe y

            bird.move()

            if bird.rect.topleft[1] < 0 or bird.rect.bottomleft[1] > SCREEN_SIZE[1]:  # Bird out of bounds
                birds.pop(i)

            elif bird.rect.colliderect(active_pipes[next_pipe].upperRect) or \
                    bird.rect.colliderect(active_pipes[next_pipe].lowerRect):
                birds.pop(i)
            else:
                bird.draw()  #
                bird.genome.fitness += 0.01

        pygame.display.flip()  #


def start(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    p = neat.Population(config)

    # p.add_reporter(neat.StdOutReporter(True))
    # stats = neat.StatisticsReporter()
    # p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes, 200)

    print('\nBest genome:\n{!s}'.format(winner))

    winning_net = neat.nn.FeedForwardNetwork.create(winner, config)
    play_winning_net(winning_net)

    node_names = {-1: 'UpperDist', -2: 'LowerDist', 0: 'Jump'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    # visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)


class Bird:
    GRAVITY = 3
    JUMP_VELOCITY = -20

    def __init__(self, net=None, genome=None):
        self.rect = pygame.Rect(50, SCREEN_SIZE[1] / 2, 0, 0).inflate(BIRD_SIZE)
        self.velocity = 0
        self.net = net
        self.genome = genome

    def jump(self):
        self.velocity = self.JUMP_VELOCITY

    def move(self):
        self.velocity += self.GRAVITY
        if self.velocity > 20:
            self.velocity = 20
        self.rect.y += self.velocity

    # Determines if to jump based on the distances Bird -> TopPipe and Bird -> Bottom pipe
    def process_1(self, active_pipe):
        dist_to_top = active_pipe.upperRect.bottomleft[1] - self.rect.center[1]
        dist_to_bottom = active_pipe.lowerRect.topleft[1] - self.rect.center[1]
        if self.net.activate((dist_to_top, dist_to_bottom))[0] > 0.75:
            self.jump()

    # Determines if to jump based on Bird y and Pipe y
    def process_2(self, active_pipe):
        pipe_y = active_pipe.upperRect.bottomleft[1] + Pipe.GAP / 2

        if self.net.activate((pipe_y, self.rect.center[1]))[0] > 0.75:
            self.jump()

    def draw(self):
        screen.blit(BirdImage, self.rect.topleft)


class Pipe:
    VELOCITY = 10
    GAP = 150

    MAX_JUMP_DIST = distance_between_pipes / VELOCITY * -Bird.JUMP_VELOCITY

    def __init__(self, x):
        y = random.randrange(50, SCREEN_SIZE[1] - self.GAP - 50)
        self.upperRect = pygame.Rect(x, y - PipeImage.get_height(), PIPE_SIZE[0], PIPE_SIZE[1])
        self.lowerRect = pygame.Rect(x, y + self.GAP, PIPE_SIZE[0], PIPE_SIZE[1])
        self.passed = False

    def move(self):
        self.upperRect.x -= self.VELOCITY
        self.lowerRect.x -= self.VELOCITY

    def reset(self):
        y = random.randrange(50, SCREEN_SIZE[1] - self.GAP - 50)
        if abs(y - self.upperRect.bottomleft[1]) > self.MAX_JUMP_DIST + Bird.JUMP_VELOCITY:
            self.reset()
            return

        self.upperRect.bottomleft = (SCREEN_SIZE[0], y)
        self.lowerRect.topleft = (SCREEN_SIZE[0], y + self.GAP)
        self.passed = False

    def draw(self):
        screen.blit(FlippedPipeImage, self.upperRect.topleft)
        screen.blit(PipeImage, self.lowerRect.topleft)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    start(config_path)
