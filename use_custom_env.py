"""
The game code:
https://github.com/Yaga987/Games/blob/main/Game-2.py
"""
import pygame
import random
import gym
import math
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_checker import check_env

# Initialize Pygame
pygame.init()

# Set up the window
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 400
WINDOW_TITLE = "Pygame Game"
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption(WINDOW_TITLE)
font = pygame.font.Font(None, 36)
# Set up the game clock
clock = pygame.time.Clock()

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Discrete(4) # What is the action size for this game we have up, down, left, right
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(4,), dtype=np.float32)
        
    def step(self, action):
        # handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Move the player
        if action == 0:
            self.player_x -= self.player_speed
        if action == 1:
            self.player_x += self.player_speed
        if action == 2:
            self.player_y -= self.player_speed
        if action == 3:
            self.player_y += self.player_speed
        
        # check if player is outside of the screen
        if self.player_x < -25 or self.player_x > WINDOW_WIDTH + 25 or \
                self.player_y < -25 or self.player_y > WINDOW_HEIGHT + 25:
            self.done = True

        # Move the enemy
        if self.enemy_x < self.player_x:
            self.enemy_x += self.enemy_speed
        elif self.enemy_x > self.player_x:
            self.enemy_x -= self.enemy_speed
        if self.enemy_y < self.player_y:
            self.enemy_y += self.enemy_speed
        elif self.enemy_y > self.player_y:
            self.enemy_y -= self.enemy_speed

        # Check for collision
        if (self.player_x < self.enemy_x + self.enemy_size and
                self.player_x + self.player_size > self.enemy_x and
                self.player_y < self.enemy_y + self.enemy_size and
                self.player_y + self.player_size > self.enemy_y):
            self.done = True

        if self.done == True:
            self.reward -= 1000

        dx = self.enemy_x - self.player_x
        dy = self.enemy_y - self.player_y
        dist = math.sqrt(dx ** 2 + dy ** 2)

        if dist >= 10:
            self.reward += 25
        elif dist >= 5:
            self.reward += 5
        else:
            self.reward -= 1

        # Draw the game
        window.fill((0, 0, 0))
        pygame.draw.rect(window, self.player_color, (self.player_x, self.player_y, self.player_size, self.player_size))
        pygame.draw.rect(window, self.enemy_color, (self.enemy_x, self.enemy_y, self.enemy_size, self.enemy_size))
        score_text = font.render("Reward: " + str(self.reward), True, (255, 255, 255))
        window.blit(score_text, (10, 10))
        pygame.display.flip()

        # Limit the frame rate
        clock.tick(60)

        self.observation  = [self.player_x, self.player_y, self.enemy_x, self.enemy_y]

        return self.observation, self.reward, self.done, {}

    def reset(self):
        # Set up env
        self.reward = 0
        self.done = False

        # Set up the player
        self.player_size = 50
        self.player_x = WINDOW_WIDTH//2
        self.player_y = WINDOW_HEIGHT//2
        self.player_speed = 5
        self.player_color = (0, 0, random.uniform(0,255))

        # Set up the enemy
        self.enemy_x = random.randint(0, WINDOW_WIDTH - self.player_size)
        self.enemy_y = random.randint(0, WINDOW_HEIGHT - self.player_size)
        self.enemy_speed = 3
        self.enemy_size = 50
        self.enemy_color = (random.uniform(0,255), 0, 0)

        self.observation  = [self.player_x, self.player_y, self.enemy_x, self.enemy_y]

        return self.observation  # reward, done, info can't be included
    
env = CustomEnv()
env.reset()

Time_Step = 10000
episodes = 10000

# First test env 

# env = CustomEnv()
# # It will check your custom environment and output additional warnings if needed
# check_env(env)

# Double check with random actions

# for episode in range(episodes):
#     obs = env.reset()
#     while True:
#         if env.done:
#             break
#         random_action = env.action_space.sample()
#         print("action", random_action)
#         obs, reward, done, info = env.step(random_action)
#         print('reward', reward)


# Train model

models_dir = '?'
logdir = '?'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

model = A2C('MlpPolicy', env, verbose=1 ,tensorboard_log=logdir)

for i in range(episodes):
    model.learn(total_timesteps=Time_Step ,reset_num_timesteps=False, tb_log_name='CustomENV-1')
    model.save(f"{models_dir}\{Time_Step*i}")

# See results

# model_path = f'{models_dir}/50000'

# model = DQN.load(model_path, env=env)

# reward_lst = []

# best_rew = -float('inf')
# worst_rew = float('inf')

# for ep in range(episodes):
#     obs = env.reset()
#     done = False
#     ep_reward = 0
#     while not done:
#         action, _states = model.predict(obs)
#         obs, rewards, done, info = env.step(action)
#         ep_reward += rewards
#     print(f"Episode {ep+1} reward: {ep_reward}")
#     reward_lst.append(ep_reward)
#     best_rew = max(best_rew, ep_reward)
#     worst_rew = min(worst_rew, ep_reward)

# print(f'Best reward: {best_rew:.2f}')
# print(f'Worst reward: {worst_rew:.2f}')

# max_idx = reward_lst.index(best_rew)
# min_idx = reward_lst.index(worst_rew)
# mean_rew = np.mean(reward_lst)

# plt.plot(reward_lst)
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.plot(max_idx, max(reward_lst), 'ro')
# plt.plot(min_idx, min(reward_lst), 'go')
# plt.text(max_idx, max(reward_lst), f'Max Reward: {max(reward_lst):.2f}', ha='left', va='bottom')
# plt.text(min_idx, min(reward_lst), f'Min Reward: {min(reward_lst):.2f}', ha='left', va='top')

# plt.axhline(mean_rew, color='orange', label='Mean Reward')
# plt.legend()
# plt.savefig(f'?')
# plt.show()