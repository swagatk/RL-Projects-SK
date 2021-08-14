import gym
import numpy as np
import imageio
# from gym import envs
# print(envs.registry.all())
import matplotlib.pyplot as plt

from pybullet_envs.bullet.racecarZEDGymEnv import RacecarZEDGymEnv
from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv

SAVE_IMG = False
SAVE_ANIM = False
VERBOSE = False

# observation is an image
env = RacecarZEDGymEnv(isDiscrete=False, renders=True)
#env = RacecarGymEnv(isDiscrete=False, renders=True)


# observation is a two dimensional vector
#env = RacecarGymEnv(isDiscrete=False, renders=True)

print('shape of Observation space: ', env.observation_space.shape)
print('shape of Action space: ', env.action_space.shape)
print('Reward Range: ', env.reward_range)
print('Action High value: ', env.action_space.high)
print('Action Low Value: ', env.action_space.low)

images = []
for ep in range(3):

    obsv = env.reset()
    #print('Observation:', obsv)

    ep_reward = 0
    t = 0
    while True:
        img = env.render(mode='rgb_array')
        plt.imshow(obsv)
        plt.axis('off')
        #plt.show()

        images.append(img)
        action = env.action_space.sample()
        next_obsv, reward, done, _ = env.step(action)

        if SAVE_IMG:
            plt.savefig('image_{}_{}.png'.format(ep, t))

        if VERBOSE:
            print('action: ', action)
            print('reward: ', reward)

        ep_reward += reward
        t += 1
        obsv = next_obsv
        if done:
            print('Episode: {}, Steps: {}, Reward: {}'.format(ep, t, ep_reward))
            break
env.close()

if SAVE_ANIM:
    imageio.mimsave('racecar.gif', np.array(images), fps=1)