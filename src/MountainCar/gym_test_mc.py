import gym


env = gym.make('MountainCarContinuous-v0')

print('Shape of Observation Space:', env.observation_space.shape)
print('Shape of Action Space:', env.action_space.shape)
print('Action upper bound:', env.action_space.high)
print('Reward Range:', env.reward_range)



for ep in range(10):
    obs = env.reset()

    t = 0
    ep_reward = 0
    while True:
        #env.render()
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)

        ep_reward += reward
        t += 1
        obs = next_obs

        if done:
            print('Episode: {}, Reward:{}, Steps:{}'.format(ep, ep_reward, t))
            break

env.close()
    