"""
https://github.com/Yaga987/Games to reach original game

"""
# Game
import os
import sys
import time
import pygame
import random
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

# initialize pygame
pygame.init()

# set the window size
window_width = 800
window_height = 600
window = pygame.display.set_mode((window_width, window_height))

# set the game title
pygame.display.set_caption("Maze Runner")

# define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
PLAYER_COLOR = ((random.uniform(0, 255)), (random.uniform(0, 255)), (random.uniform(0, 255)))
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# set the game clock
clock = pygame.time.Clock()

# define the player class
class Player(pygame.sprite.Sprite):
    def __init__(self, game_over):
        super().__init__()
        self.image = pygame.Surface([50, 50])
        self.image.fill(PLAYER_COLOR)
        self.rect = self.image.get_rect()
        self.rect.x = window_width // 2
        self.rect.y = window_height // 2
        self.health = 3
        self.game_over = game_over

    def move(self, dx, dy):
        self.rect.x += dx
        self.rect.y += dy

    def hit(self):
        self.health -= 1
        if self.health <= 0:
            self.game_over = True

# define the coin class
class Coin(pygame.sprite.Sprite):
    def __init__(self, all_sprites):
        super().__init__()
        self.image = pygame.Surface([25, 25])
        self.image.fill(YELLOW)
        self.rect = self.image.get_rect()
        self.all_sprites = all_sprites

        # generate a new position that is not too close to any existing sprite
        while True:
            self.rect.x = random.randint(0, window_width - 25)
            self.rect.y = random.randint(0, window_height - 25)
            if not any(sprite.rect.colliderect(self.rect.inflate(50, 50)) for sprite in all_sprites):
                break

    def update(self):
        self.all_sprites.add(self)

# define the Gem class
class Gem(pygame.sprite.Sprite):
    def __init__(self, all_sprites):
        super().__init__()
        self.image = pygame.Surface([10, 10])
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.all_sprites = all_sprites

        # generate a new position that is not too close to any existing sprite
        while True:
            self.rect.x = random.randint(0, window_width - 10)
            self.rect.y = random.randint(0, window_height - 10)
            if not any(sprite.rect.colliderect(self.rect.inflate(25, 25)) for sprite in all_sprites):
                break

    def update(self):
        self.all_sprites.add(self)

# define the obstacle class
class Obstacle(pygame.sprite.Sprite):
    def __init__(self, all_sprites):
        super().__init__()
        self.image = pygame.Surface([50, 50])
        self.image.fill(RED)
        self.rect = self.image.get_rect()

        # generate a new position that is not too close to any existing sprite
        while True:
            self.rect.x = random.randint(0, window_width - 50)
            self.rect.y = random.randint(0, window_height - 50)
            if not any(sprite.rect.colliderect(self.rect.inflate(100, 100)) for sprite in all_sprites):
                break
        self.all_sprites = all_sprites

# define the CustomEnv class
class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(63,), dtype=np.float32)

    def step(self, action):
        # handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.stepcounter += 1

        self.max_coins = 20
        self.max_gems = 5
        self.max_obstacles = 5

        # Change the head position based on the button direction
        if action == 0:
            self.player.move(-10, 0)
        elif action == 1:
            self.player.move(10, 0)
        elif action == 2:
            self.player.move(0, -10)
        elif action == 3:
            self.player.move(0, 10)

        # check if player is outside of the screen
        if self.player.rect.left < -25 or self.player.rect.right > window_width + 25 or \
                self.player.rect.bottom < -25 or self.player.rect.top > window_height + 25:
            self.reward = -100
            self.player.game_over = True

        # check collisions between the player and coins
        coins_hit = pygame.sprite.spritecollide(
            self.player, self.coins, True)
        for coin in coins_hit:
            self.reward += 10

        # check collisions between the player and gems
        gems_hit = pygame.sprite.spritecollide(
            self.player, self.gems, True)
        for gem in gems_hit:
            self.reward += 100

        # check collisions between the player and obstacles
        obstacles_hit = pygame.sprite.spritecollide(
            self.player, self.obstacles, True)
        for obstacle in obstacles_hit:
            self.reward = -10
            self.player.hit()

        # spawn new obstacle if needed
        if len(self.obstacles) == 0 or len(self.obstacles) < self.max_obstacles:
            self.obstacle = Obstacle(self.all_sprites)
            self.obstacles.add(self.obstacle)

        # spawn new obstacle if needed
        if len(self.coins) == 0 or len(self.coins) < self.max_coins:
            self.coin = Coin(self.all_sprites)
            self.coins.add(self.coin)

        # spawn new obstacle if needed
        if len(self.gems) == 0 or len(self.gems) < self.max_gems:
            self.gem = Gem(self.all_sprites)
            self.gems.add(self.gem)

        # spawn new obstacle after 3 seconds
        for event in pygame.event.get():
            if event.type == pygame.USEREVENT:
                self.obstacle = Obstacle(self.all_sprites)
                self.obstacles.add(self.obstacle)

        # update the screen
        window.fill(BLACK)
        self.coins.draw(window)
        self.gems.draw(window)
        self.obstacles.draw(window)
        window.blit(self.player.image, self.player.rect)
        pygame.display.flip()

        # set the game speed
        clock.tick(60)

        # check if game over
        if self.player.game_over:
            self.done = True
            self.reward -= 1000

        if (self.start_time - time.time() > 100):
            if (self.reward // 100) >= 1:
                self.gift_rate = 10
            self.start_time = time.time()
            self.gift_reward = int(((4 * ((self.reward % 10) + 1) * (((self.reward // 10) + 1) ** 2)) // 3) * self.gift_rate * (random.random() + random.randint(1,3)))
            self.reward += self.gift_reward

        if self.stepcounter >= 10000:
            self.reward = -1
            self.stepcounter = 0

        info = {}

        # Location of player, coins and obstacles also player health
        player_x = self.player.rect.x
        player_y = self.player.rect.y
        player_health = self.player.health

        # get coin locations
        loc_coins = [(0, 0)] * self.max_coins
        for i, coin in enumerate(self.coins):
            loc_coins[i] = (coin.rect.x, coin.rect.y)

        # get gem locations
        loc_gems = [(0, 0)] * self.max_gems
        for i, gem in enumerate(self.gems):
            loc_gems[i] = (gem.rect.x, gem.rect.y)

        # get obstacle locations
        loc_obstacles = [(0, 0)] * self.max_obstacles
        for i, obstacle in enumerate(self.obstacles):
            loc_obstacles[i] = (obstacle.rect.x, obstacle.rect.y)

        # create 1D array
        self.observation = np.zeros((2*self.max_coins + 2*self.max_obstacles + 2*self.max_gems + 3,))
        self.observation[0] = player_health
        self.observation[1] = player_x
        self.observation[2] = player_y
        for i in range(self.max_coins):
            self.observation[3 + 2*i] = loc_coins[i][0]
            self.observation[4 + 2*i] = loc_coins[i][1]
        for i in range(self.max_obstacles):
            self.observation[3 + 2*self.max_coins + 2*i] = loc_obstacles[i][0]
            self.observation[4 + 2*self.max_coins + 2*i] = loc_obstacles[i][1]
        for i in range(self.max_gems):
            self.observation[3 + 2*self.max_coins + 2*self.max_obstacles +2*i] = loc_gems[i][0]
            self.observation[4 + 2*self.max_coins + 2*self.max_obstacles +2*i] = loc_gems[i][1]

        return self.observation, self.reward, self.done, info

    def reset(self):
        self.start_time = time.time()
        self.gift_rate = 2.5
        self.done = False
        self.game_over = False
        self.reward = 0
        self.stepcounter = 0

        self.max_coins = 20
        self.max_gems = 5
        self.max_obstacles = 5

        # create the player object
        self.player = Player(self.game_over)
        self.player.health = 3

        # create the coins, gems and obstacles groups
        self.coins = pygame.sprite.Group()
        self.gems = pygame.sprite.Group()
        self.obstacles = pygame.sprite.Group()
        self.all_sprites = pygame.sprite.Group()

        # add coins to the coins group
        for i in range(self.max_coins):
            self.coin = Coin(self.all_sprites)
            self.coins.add(self.coin)
            self.all_sprites.add(self.coin)

        # add gems to the gems group
        for i in range(self.max_gems):
            self.gem = Gem(self.all_sprites)
            self.gems.add(self.gem)
            self.all_sprites.add(self.gem)

        # add the player to all_sprites
        self.all_sprites.add(self.player)

        # add obstacles to the obstacles group
        for i in range(self.max_obstacles):
            self.obstacle = Obstacle(self.all_sprites)
            self.obstacles.add(self.obstacle)
            self.all_sprites.add(self.obstacle)

        # Location of player, coins and obstacles also player health
        player_x = self.player.rect.x
        player_y = self.player.rect.y
        player_health = self.player.health

        # get coin locations
        loc_coins = [(0, 0)] * self.max_coins
        for i, coin in enumerate(self.coins):
            loc_coins[i] = (coin.rect.x, coin.rect.y)

        # get gem locations
        loc_gems = [(0, 0)] * self.max_gems
        for i, gem in enumerate(self.gems):
            loc_gems[i] = (gem.rect.x, gem.rect.y)

        # get obstacle locations
        loc_obstacles = [(0, 0)] * self.max_obstacles
        for i, obstacle in enumerate(self.obstacles):
            loc_obstacles[i] = (obstacle.rect.x, obstacle.rect.y)

        # create 1D array
        self.observation = np.zeros((2*self.max_coins + 2*self.max_obstacles + 2*self.max_gems + 3,))
        self.observation[0] = player_health
        self.observation[1] = player_x
        self.observation[2] = player_y
        for i in range(self.max_coins):
            self.observation[3 + 2*i] = loc_coins[i][0]
            self.observation[4 + 2*i] = loc_coins[i][1]
        for i in range(self.max_obstacles):
            self.observation[3 + 2*self.max_coins + 2*i] = loc_obstacles[i][0]
            self.observation[4 + 2*self.max_coins + 2*i] = loc_obstacles[i][1]
        for i in range(self.max_gems):
            self.observation[3 + 2*self.max_coins + 2*self.max_obstacles +2*i] = loc_gems[i][0]
            self.observation[4 + 2*self.max_coins + 2*self.max_obstacles +2*i] = loc_gems[i][1]

        return self.observation  # reward, done, info can't be included

env = CustomEnv()
env.reset()
models_dir = '?'
logdir = '?'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

Time_Step = 10000
episodes = 100

# model = DQN('MlpPolicy', env, verbose=1 ,tensorboard_log=logdir)

# for i in range(episodes):
#     model.learn(total_timesteps=Time_Step ,reset_num_timesteps=False, tb_log_name='CustomENV')
#     model.save(f"{models_dir}\{Time_Step*i}")

# for episode in range(episodes):
#     done = False
#     obs = env.reset()
#     while True:
#         if env.player.game_over:
#             done = True
#         if done:
#             break
#         random_action = env.action_space.sample()
#         print("action", random_action)
#         obs, reward, done, info = env.step(random_action)
#         print('reward', reward)

model_path = f'{models_dir}/50000'

model = DQN.load(model_path, env=env)

reward_lst = []

best_rew = -float('inf')
worst_rew = float('inf')

for ep in range(episodes):
    obs = env.reset()
    done = False
    ep_reward = 0
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        ep_reward += rewards
    print(f"Episode {ep+1} reward: {ep_reward}")
    reward_lst.append(ep_reward)
    best_rew = max(best_rew, ep_reward)
    worst_rew = min(worst_rew, ep_reward)

print(f'Best reward: {best_rew:.2f}')
print(f'Worst reward: {worst_rew:.2f}')

max_idx = reward_lst.index(best_rew)
min_idx = reward_lst.index(worst_rew)
mean_rew = np.mean(reward_lst)

plt.plot(reward_lst)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.plot(max_idx, max(reward_lst), 'ro')
plt.plot(min_idx, min(reward_lst), 'go')
plt.text(max_idx, max(reward_lst), f'Max Reward: {max(reward_lst):.2f}', ha='left', va='bottom')
plt.text(min_idx, min(reward_lst), f'Min Reward: {min(reward_lst):.2f}', ha='left', va='top')

plt.axhline(mean_rew, color='orange', label='Mean Reward')
plt.legend()
plt.savefig(f'?')
plt.show()