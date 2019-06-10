import gym
from PIL import Image
from lshash.lshash import LSHash
from collections import deque
from random import random
from diskcache import FanoutCache, Cache
qtable = Cache('cache')
qtable.clear()
table = dict()
from time import sleep
import numpy as np
env = gym.make('Breakout-v0')
lshs = LSHash(500, 8192)


LEARNING_RATE = 0.15
DISCOUNT = 0.95
EPISODES = 25000

def preprocess(obs):
    image = Image.fromarray(observation)
    image = image.resize((64, 64))
    image = image.convert(mode='1')
    array = np.array(image, dtype=np.uint8).flatten()
    return array

def get_action(obs_seq):
    query = lshs.query(obs_seq, num_results=1)
    if len(query) <= 0:
        lshs.index(obs_seq)
        actions = np.ones(env.action_space.n)
        qtable[obs_seq] = actions
    elif query[0][1] >= 10:
        lshs.index(obs_seq)
        actions = np.ones(env.action_space.n)
        qtable[obs_seq] = actions
    else:
        print(f'Query Value: {query[0][1]}')
        query_obs_seq = np.array(query[0][0], dtype=np.uint8)
        # print(f'query obs: {query_obs_seq}')
        actions = qtable.get(query_obs_seq)
        if actions is None:
            raise Exception("Error actions is None, observations sequence not found in qtable!")
    if np.sum(actions) <= 0.00001:
        action = None
        return action
    print(f'\n\t\t\t\t\tQvalues: {actions}')
    p = (1 / np.sum(actions)) * actions
    print(p)
    action = np.random.choice(range(env.action_space.n), p=p)
    # print(action)
    return action
    #return np.argmax(actions)

def learn(action, obs_seq, reward, next_obs_seq):
    query = lshs.query(obs_seq, num_results=1)
    query_obs_seq = np.array(query[0][0], dtype=np.uint8)
    q_actions = qtable[query_obs_seq]


    next_query = lshs.query(next_obs_seq, num_results=1)
    if len(next_query) <= 0:
        lshs.index(next_obs_seq)
        nq_actions = np.ones(env.action_space.n)
        qtable[next_obs_seq] = nq_actions
    elif query[0][1] >= 10:
        lshs.index(next_obs_seq)
        nq_actions = np.ones(env.action_space.n)
        qtable[next_obs_seq] = nq_actions
    else:
        next_query_obs_seq = np.array(next_query[0][0], dtype=np.uint8)
        nq_actions = qtable.get(next_query_obs_seq)
        if nq_actions is None:
            raise Exception("Learning failed, ")
    max_future_q = np.max(nq_actions)
    current_q = q_actions[action]
    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
    if reward > 0:
        print('############################################')
        print(f'reward" {reward}')
        print(f'NEW Q: {new_q}')
    q_actions[action] = new_q
    qtable[query_obs_seq] = q_actions

def done_learn(action, obs_seq, reward):
    query = lshs.query(obs_seq, num_results=1)
    query_obs_seq = np.array(query[0][0], dtype=np.uint8)
    q_actions = qtable[query_obs_seq]

    new_q = reward
    q_actions[action] = new_q
    qtable[query_obs_seq] = q_actions

for i_episode in range(20000000):
    observation = env.reset()
    observation = preprocess(observation)
    sequence = deque([observation, observation], maxlen=2)
    for t in range(10000):
        env.render()
        obs_seq = np.array(sequence).flatten()
        if random() >= 0.0:
            action = get_action(obs_seq)
        else:
            action = env.action_space.sample()
        if action is None:
            break
        else:
            observation, reward, done, info = env.step(action)
        observation = preprocess(observation)
        sequence.append(observation)
        next_obs_seq = np.array(sequence).flatten()
        if t == 10000-1:
            done = True
        if not done:
            learn(action, obs_seq, reward, next_obs_seq)
        else:
            done_learn(action, obs_seq, reward)
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()
