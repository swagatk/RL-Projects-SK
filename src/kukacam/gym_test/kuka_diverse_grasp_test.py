import numpy as np
import imageio
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv

import matplotlib.pyplot as plt
env = KukaDiverseObjectEnv(renders=False,  # True to see the simulation environment
                           isDiscrete=False,
                           removeHeightHack=False,
                           maxSteps=20)

print('shape of Observation space: ', env.observation_space.shape)
print('shape of Action space: ', env.action_space.shape)
print('Reward Range: ', env.reward_range)
print('Action High value: ', env.action_space.high)
print('Action Low Value: ', env.action_space.low)

SAVE_IMG = False
SAVE_ANIM = False
VERBOSE = False

images = []
for ep in range(5):
    obsv = env.reset()
    t = 0
    score = 0
    img = env.render(mode='rgb_array')
    images.append(img)
    while True:
        plt.imshow(obsv)
        plt.axis('off')
        #plt.show()

        if SAVE_IMG:
            plt.savefig('image_{}_{}.png'.format(ep,t))

        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        img = env.render(mode='rgb_array')
        images.append(img)

        if VERBOSE:
            print('action: ', action)
            print('reward: ', reward)

        t += 1
        score += reward
        obsv = next_obs   # not used
        if done:
            print('Episode:{}, Steps: {}, Score: {}'.format(ep, t, score))
            break
env.close()

if SAVE_ANIM:
    # Save image files
    imageio.mimsave('kukadiverse.gif', np.array(images), fps=1)
