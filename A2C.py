import gym
import os
from stable_baselines3 import A2C, PPO

env = gym.make('LunarLander-v2')  # continuous: LunarLanderContinuous-v2
env.reset()

models_dir = 'Python/PPO/models/A2C'
logdir = 'Python/PPO/logs'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

Time_Step = 1000000
episodes = 10

model = A2C('MlpPolicy', env, verbose=1 ,tensorboard_log=logdir)
# model = A2C('MlpPolicy', env, verbose=1)

for i in range(episodes):
    model.learn(total_timesteps=Time_Step ,reset_num_timesteps=False, tb_log_name='A2C')
    model.save(f'{models_dir}/{Time_Step*i}')

# best_rew = -float('inf')

# for i in range(episodes):
#     obs = env.reset()
#     done = False
#     episode_rew = 0
#     while not done:
#         action, _ = model.predict(obs)
#         obs, rew, done, _ = env.step(action)
#         episode_rew += rew
#     best_rew = max(best_rew, episode_rew)

# print(f'Best reward: {best_rew:.2f}')
