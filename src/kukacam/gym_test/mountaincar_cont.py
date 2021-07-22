import gym

env = gym.make('MountainCarContinuous-v0')

print('Shape of Observation Space: ', env.observation_space.shape)
print('Shape of Action Space: ', env.action_space.shape)
print('Action Limits: ', env.action_space.high, env.action_space.low)


for ep in range(5):
    obs = env.reset()
    ep_reward = 0
    t = 0
    while True:
        # env.render()
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        ep_reward += reward 
        t += 1
        obs = next_obs
        if done:
            print(f'Episode: {ep}, Steps: {t}, Score: {ep_reward}')
            break
env.close()