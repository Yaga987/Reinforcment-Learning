import gym
import os
from stable_baselines3 import A2C, PPO

env = gym.make('LunarLander-v2')  # continuous: LunarLanderContinuous-v2
env.reset()

models_dir = 'Python/PPO/models/PPO'
logdir = 'Python/PPO/logs'

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

Time_Step = 10000
episodes = 10

# model = PPO('MlpPolicy', env, verbose=1 ,tensorboard_log=logdir)
# # model = A2C('MlpPolicy', env, verbose=1)

# for i in range(episodes):
#     model.learn(total_timesteps=Time_Step ,reset_num_timesteps=False, tb_log_name='PPO')
#     model.save(f"{models_dir}/{Time_Step*i}")

model_path = f'{models_dir}/170000.zip'

model = PPO.load(model_path, env=env)

best_rew = -float('inf')

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        # print(rewards)
    print(ep)
    best_rew = max(best_rew, rewards)

print(f'Best reward: {best_rew:.2f}')
