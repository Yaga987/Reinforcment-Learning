import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

# Create the CartPole environment
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Define the DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95   # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural network with 2 hidden layers of x units each
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# Initialize the agent
agent = DQNAgent(state_size, action_size)

# Train the agent
episodes = 100
batch_size = 32
scores = []
for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    score = 0
    while not done:
        # env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            agent.replay(batch_size)
    scores.append(score)
    print("episode: {}/{}, score: {}".format(e+1, episodes, score))

# Plot the results
mean_score = np.mean(scores)
max_score = np.max(scores)
min_score = np.min(scores)

max_idx = scores.index(max_score)
min_idx = scores.index(min_score)

plt.plot(scores)
plt.title('DQN Performance on CartPole')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.figtext(0.6, 0.7, f"Mean Score: {mean_score:.2f}")
plt.figtext(0.6, 0.65, f"Max Score: {max_score:.2f}")
plt.figtext(0.6, 0.6, f"Min Score: {min_score:.2f}")
plt.plot(max_idx, max(scores), 'ro')
plt.plot(min_idx, min(scores), 'go')
plt.axhline(mean_score, color='orange', label='Mean Reward')
plt.legend()
plt.savefig(f'D:\Code\Python\RL\DQN-Performance.png')
plt.show()