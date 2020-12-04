from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces
from itertools import count
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


for episode in range(10):
    print('Episode: ', episode)
    obsv = env.reset()
    for t in count():
        #plt.imshow(obsv)
        #plt.show()
        #env.render()
        action = env.action_space.sample()
        print(action)
        next_obs, reward, done, info = env.step(action)
        if done:
            break
        obs = next_obs
        #print('t =', t)

env.close()
