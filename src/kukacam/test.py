import gym
import matplotlib.pyplot as plt

#env = gym.make('CarRacing-v0')
env = gym.make('BipedalWalkerHardcore-v2')
print('Observation shape:', env.observation_space.shape)
print('Action space shape:', env.action_space.shape)
print('Reward Range:', env.reward_range)
print('Action space High:', env.action_space.high)
for ep in range(2):
    ep_reward = 0
    obsv = env.reset()
    #plt.imshow(obsv)
    #plt.show()
    while True:
        env.render()
        action = env.action_space.sample()
        next_obsv,reward, done,_ = env.step(action)
        ep_reward += reward
        if done:
            print('Episode: {}, Reward: {}'.format(ep, ep_reward))
            break
env.close()
