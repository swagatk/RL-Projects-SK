#import roboschool
import gym
import numpy as np
# from gym import envs
# print(envs.registry.all())
import matplotlib.pyplot as plt

import pybullet_envs.bullet.racecarZEDGymEnv as e
env = e.RacecarZEDGymEnv(isDiscrete=False, renders=True)

print('shape of Observation space: ', env.observation_space.shape)
print('shape of Action space: ', env.action_space.shape)
print('Reward Range: ', env.reward_range)
print('Action High value: ', env.action_space.high)
print('Action Low Value: ', env.action_space.low)


for ep in range(3):

    obsv = env.reset()
    obsv2 = env.getExtendedObservation()
    print(np.shape(obsv2))
    plt.imshow(obsv2)
    plt.show()
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
