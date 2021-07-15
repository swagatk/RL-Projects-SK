import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
import imageio
from _datetime import datetime

from pybullet_envs.bullet.racecarZEDGymEnv import RacecarZEDGymEnv
env = RacecarZEDGymEnv(isDiscrete=False, renders=True)


#env = DummyVecEnv([lambda: RacecarZEDGymEnv(isDiscrete=False, renders=False)])
# Automatically normalize the input features and reward
#env = VecNormalize(env, norm_obs=True, norm_reward=True)

print('shape of Observation space: ', env.observation_space.shape)
print('shape of Action space: ', env.action_space.shape)
print('Reward Range: ', env.reward_range)
print('Action High value: ', env.action_space.high)
print('Action Low Value: ', env.action_space.low)

# create a RL model
model = SAC('CnnPolicy', env, buffer_size=20000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f'mean_reward before training: {mean_reward:.2f} +/- {std_reward:.2f}')

# Train the model
start = datetime.now()
model.learn(total_timesteps=10000)
model.save('sac_racecar')
end = datetime.now()
print('Time for training: ', end-start)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f'mean_reward after training: {mean_reward:.2f} +/- {std_reward:.2f}')

for ep in range(2):
    obsv = env.reset()
    #obsv2 = env.getExtendedObservation()
    #print(np.shape(obsv2))
    #plt.imshow(obsv2)
    #plt.show()
    ep_reward = 0
    t = 0
    while True:
        env.render()
        #action = env.action_space.sample()
        action = model.predict(obsv, deterministic=True)
        next_obsv, reward, done, _ = env.step(action)
        #print('reward: ', reward)
        ep_reward += reward
        t += 1
        if done:
            print('Episode: {}, Steps: {}, Reward: {}'.format(ep, t, ep_reward))
            break
env.close()


