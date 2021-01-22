"""
Testing Pendulum environment
"""
import gym
env = gym.make("Pendulum-v0")
state_size = env.observation_space.shape
print("Shape of State Space ->  {}".format(state_size))
action_size = env.action_space.shape
print("Size of Action Space ->  {}".format(action_size))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))
print('Reward Range:', env.reward_range)

for ep in range(20):
    state = env.reset()
    done = False
    ep_reward = 0
    while not done:
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        ep_reward += reward
        if done:
            print('Episode: {}, Reward: {}'.format(ep, ep_reward))
            break
env.close()