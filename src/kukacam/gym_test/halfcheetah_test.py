"""
Example program to run PyBullet's 'HalfCheetahBulletEnv-v0'

"""
import gym
import imageio
import numpy as np
import pybullet_envs as pe

env = gym.make('HalfCheetahBulletEnv-v0')
env._max_episode_steps = 100
print('Observation Space: ', env.observation_space.shape)
print('Action Space:', env.action_space.shape)
print('Reward Range:', env.reward_range)
print('Action High & Low values:',
      env.action_space.high, env.action_space.low)

images = []
for ep in range(3):
    t = 0
    ep_reward = 0
    obsv = env.reset()
    while True:
        img = env.render(mode='rgb_array')
        action = env.action_space.sample()
        next_obsv, reward, done, _ = env.step(action)
        obsv = next_obsv
        t += 1
        ep_reward += reward
        images.append(img)
        if done:
            print('Episode:{}, Steps:{}, Score:{}'\
                  .format(ep, t, ep_reward))
            break
imageio.mimsave('cheetah.gif', np.array(images), fps=29)
env.close()

