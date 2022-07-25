import gym
import numpy as np
import mujoco_py
# from gym import envs
# print(envs.registry.all())

env = gym.make('FetchReach-v0')
env.reset()

print('Observation space: ', env.observation_space)
print('shape of Action space: ', env.action_space.shape)
print('Reward Range: ', env.reward_range)
print('Action High value: ', env.action_space.high)
print('Action Low Value: ', env.action_space.low)
for ep in range(100):
    state = env.reset()
    print(state)
    ep_reward = 0
    t = 0
    while True:
        #env.render()
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        ep_reward += reward
        t += 1
        if done:
            print('Episode: {}, Steps: {}, Score: {}'.format(ep, t, ep_reward))
            break

env.close()

