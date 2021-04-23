import numpy as np
import math
import random
import gym
import gym_climb

def simulate():
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    total_reward = 0
    total_rewards = []
    training_done = False
    threshold = 1000
    env.set_view(True)

    show_cnt = 0
    for episode in range(NUM_EPISODES):

        total_rewards.append(total_reward)
        obv = env.reset()
        state_0 = state_to_bucket(obv)
        total_reward = 0

        if episode >= threshold:
            explore_rate = 0.01

        for t in range(MAX_T):
            action = select_action(state_0, explore_rate)
            obv, reward, done, _ = env.step(action)
            state = state_to_bucket(obv)
            total_reward += reward

            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate * (reward + DISCOUNT_FACTOR * (best_q) - q_table[state_0 + (action,)])

            state_0 = state
            env.render()
            if done or t >= MAX_T - 1:
                print("Episode %d finished after %i time steps with total reward = %f."
                      % (episode, t, total_reward))
                break

        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)

def select_action(state, explore_rate):
    if random.random() < explore_rate:
        action = env.action_space.sample()
    else:
        action = int(np.argmax(q_table[state]))
    return action

def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))

def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

if __name__ == "__main__":

    env = gym.make("Pyclimb-v0")
    NUM_BUCKETS = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_ACTIONS = env.action_space.n
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

    MIN_EXPLORE_RATE = 0.001
    MIN_LEARNING_RATE = 0.2
    DECAY_FACTOR = np.prod(NUM_BUCKETS, dtype=float) / 10.0
    NUM_EPISODES = 999999
    MAX_T = 10000
    DISCOUNT_FACTOR = 0.99
    q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)
    simulate()