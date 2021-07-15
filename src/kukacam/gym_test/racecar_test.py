import gym
import numpy as np
# from gym import envs
# print(envs.registry.all())
import matplotlib.pyplot as plt

from pybullet_envs.bullet.racecarZEDGymEnv import RacecarZEDGymEnv
from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv

# observation is an image
#env = RacecarZEDGymEnv(isDiscrete=False, renders=False)
env = RacecarGymEnv(isDiscrete=False, renders=True)


# observation is a two dimensional vector
#env = RacecarGymEnv(isDiscrete=False, renders=True)

print('shape of Observation space: ', env.observation_space.shape)
print('shape of Action space: ', env.action_space.shape)
print('Reward Range: ', env.reward_range)
print('Action High value: ', env.action_space.high)
print('Action Low Value: ', env.action_space.low)


for ep in range(5):

    obsv = env.reset()
    print('Observation:', obsv)

    ep_reward = 0
    t = 0
    while True:
        env.render()
        action = env.action_space.sample()
        #print('action: ', action)
        next_obsv, reward, done, _ = env.step(action)
        #print('reward: ', reward)
        ep_reward += reward
        t += 1
        if done:
            print('Episode: {}, Steps: {}, Reward: {}'.format(ep, t, ep_reward))
            break
env.close()
