import gym
import gym_climb
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense
from collections import deque
import random
import numpy as np
from keras.optimizers import Adam

class DQAgent:
    def __init__(self):
        self.env = gym.make('Pyclimb-v0')
        self.env.set_view(True)
        self.input_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.nets = []
        for i in range(self.action_size):
            self.nets.append(self.qnetwork(self.input_size))

        self.memory = deque(maxlen=1000000)
        self.batch_size = 32
        self.epsilon = 1.0
        self.gamma = 0.95
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def qnetwork(self, input_size):
        model = Sequential()
        model.add(Dense(24, input_dim=input_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def remember(self, s, a, r, s_, d):
        self.memory.append((s, a, r, s_, d))

    def get_action(self, state):
        if self.epsilon > np.random.rand():
            return random.randrange(self.action_size)
        val = []
        for i in self.nets:
            val.append(i.predict(state)[0])
        return np.argmax(val)

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for s, a, r, s_, d in minibatch:
            if d:
                target = r
            else:
                val = []
                for i in self.nets:
                    val.append(i.predict(s_)[0])
                target = r + self.gamma * np.amax(val)
            self.nets[a].fit(s, [[target]], epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    agent = DQAgent()

    NUM_EPISODES = 999999
    MAX_T = 10000

    for episode in range(NUM_EPISODES):
        t = 0
        total_reward = 0
        state = agent.env.reset()
        state = np.reshape(state, [1, agent.input_size])
        for t in range(MAX_T):
            agent.env.render()
            action = agent.get_action(state)
            next_state, reward, done, _ = agent.env.step(action)
            next_state = np.reshape(next_state, [1, agent.input_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            t += 1
            if done:
                print("Episode %d finished after %i time steps with total reward = %f."
                      % (episode, t, total_reward))
                break
            if len(agent.memory) > agent.batch_size:
                agent.replay()