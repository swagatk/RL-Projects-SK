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

step_cnt = []
for episode in range(2):
    print('Episode: ', episode)
    obsv = env.reset()
    t = 0
    ep_reward = 0
    while True:
        plt.imshow(obsv)
        #plt.show()
        #env.render()
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        ep_reward += reward
        t += 1
        if done:
            step_cnt.append(t)
            print('Episode: {}, Steps: {}, Score: {}'.format(episode, t, reward))
            break
        obs = next_obs

env.close()
plt.plot(step_cnt)