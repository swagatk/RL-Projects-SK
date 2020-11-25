import gym
from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv
import numpy as np
import time
from itertools import count
import matplotlib.pyplot as plt
#env = gym.make('KukaCamBulletEnv-v0')
env = KukaCamGymEnv(renders=True, isDiscrete=False)
print('shape of Observation space: ', env.observation_space.shape)
print('shape of Action space: ', env.action_space.shape)
print('Reward Range: ', env.reward_range)
print('Action High value: ', env.action_space.high)
print('Action Low Value: ', env.action_space.low)
for episode in range(2):
    print('Episode: ', episode)
    obsv = env.reset()
    for t in count():
        plt.imshow(obsv)
        plt.show()
        #env.render()
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        if done:
            break
        obs = next_obs
        #print('t =', t)

env.close()