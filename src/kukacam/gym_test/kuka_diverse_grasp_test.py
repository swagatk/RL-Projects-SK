import numpy as np
import imageio
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv

import matplotlib.pyplot as plt
env = KukaDiverseObjectEnv(renders=True,  # True to see the simulation environment
                           isDiscrete=False,
                           removeHeightHack=False,
                           maxSteps=20)

print('shape of Observation space: ', env.observation_space.shape)
print('shape of Action space: ', env.action_space.shape)
print('Reward Range: ', env.reward_range)
print('Action High value: ', env.action_space.high)
print('Action Low Value: ', env.action_space.low)

images = []
for ep in range(10):
    obsv = env.reset()
    t = 0
    score = 0
    img = env.render(mode='rgb_array')
    images.append(img)
    while True:
        #plt.imshow(obsv)
        #plt.show()
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        img = env.render(mode='rgb_array')
        images.append(img)
        t += 1
        score += reward
        # obsv = next_obs   # not used
        if done:
            print('Episode:{}, Steps: {}, Score: {}'.format(ep, t, score))
            break
env.close()
imageio.mimsave('kukadiverse.gif', np.array(images), fps=1)
