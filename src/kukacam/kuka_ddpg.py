'''
Applying DDPG algorithm to KukaCamGym Environment

'''
import gym
from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv
import numpy as np
import time
from itertools import count
import matplotlib.pyplot as plt
env = KukaCamGymEnv(renders=True, isDiscrete=False)
print('shape of Observation space: ', env.observation_space.shape)
print('shape of Action space: ', env.action_space.shape)
print('Reward Range: ', env.reward_range)
print('Action High value: ', env.action_space.high)
print('Action Low Value: ', env.action_space.low)

################
# Hyper-parameters
######################
MAX_EPISODES = 1000
STACK_SIZE = 5 # number of frames stacked together

LR_A = 0.001
LR_C = 0.002
GAMMA = 0.99

replacement = [
    dict(name='soft', tau=0.005),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]  # you can try different target replacement strategies

MEMORY_CAPACITY = 50000
BATCH_SIZE = 64

for episode in range(MAX_EPISODES):
    obsv = env.reset()

    episodic_reward = 0
    frames = []
    while True:
        if episode > MAX_EPISODES - 3:
            frames.append(env.render(mode='rgb_array'))
        agent.policy(state)


for
