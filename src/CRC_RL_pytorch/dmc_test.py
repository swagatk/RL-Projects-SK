'''
Make sure that the following packages are
installed:
- mujoco 2.1.0
- mujoco-py
- dmc2gym
'''

import dmc2gym
import numpy as np

env = dmc2gym.make(domain_name='point_mass', 
              task_name='easy', 
              seed=1)

done = False 
obs = env.reset()
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    print(np.shape(obs))
    print(reward)