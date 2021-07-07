import os
import gym
import pybullet_envs as pe
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2
import imageio
import numpy as np
import datetime

env = DummyVecEnv([lambda: gym.make("HalfCheetahBulletEnv-v0")])
# Automatically normalize the input features and reward
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
model = PPO2('MlpPolicy', env)
start = datetime.datetime.now()
model.learn(total_timesteps=2000)
end = datetime.datetime.now()
print('Time for training: ', end-start)
# Don't forget to save the VecNormalize statistics when saving the agent
log_dir = "/tmp/"
model.save(log_dir+"ppo_halfcheetah")
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
env.save(stats_path)

# Test
images = []
obs = model.env.reset()
img = model.env.render(mode='rgb_array')
for i in range(100):
    images.append(img)
    action, _ = model.predict(obs)
    obs, _, _, _ = model.env.step(action)
    img = model.env.render(mode='rgb_array')
imageio.mimsave('cheetah_ppo2.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)

# To demonstrate loading
del model, env
# load the agent
model = PPO2.load(log_dir + "ppo_halfcheetah")

# load saved statistics
env = DummyVecEnv([lambda: gym.make("HalfCheetahBulletEnv-v0")])
env = VecNormalize.load(stats_path, env)

# do not update them at test time
env.training = False

# reward Normalization is not needed at test time
env.norm_reward = False


