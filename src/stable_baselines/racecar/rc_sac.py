
"""
Applying SB3.SAC to RaceCar problem
"""

import os
import pybullet as p
import gym
from pybullet_envs.bullet.racecarZEDGymEnv import RacecarZEDGymEnv
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from datetime import datetime

# Local Imports
from rc_gym_wrapper import RaceCarCustomWrapper4pt
from rc_custom_cnn import CustomCNN

# Directories for storing logs & results
root_path = './'
model_path = root_path + 'best_model/'
result_path = root_path + 'results/'
monitor_path = root_path + 'monitor/'
tb_log_path = root_path + 'tb_log/'
checkpt_path = root_path + 'checkpoints/'
video_path = root_path + 'animation/'

os.makedirs(model_path, exist_ok=True)
os.makedirs(result_path, exist_ok=True)
os.makedirs(tb_log_path, exist_ok=True)
os.makedirs(monitor_path, exist_ok=True)
os.makedirs(checkpt_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)


# create the environment
env = RaceCarCustomWrapper4pt(RacecarZEDGymEnv(isDiscrete=False,
                                                 renders=False), shape=40, max_steps=20)
env = Monitor(env, monitor_path)

# environment for periodic evaluation
eval_env = RaceCarCustomWrapper4pt(RacecarZEDGymEnv(isDiscrete=False,
                                                      renders=False), shape=40, max_steps=20)
eval_env = Monitor(eval_env, monitor_path)

#callbacks
eval_callback = EvalCallback(eval_env, best_model_save_path=root_path+'best_model/',
                             log_path=root_path+'results/', eval_freq=500,
                             deterministic=True, render=False)
checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=checkpt_path,
                                         name_prefix='rc_sac_checkpt')
callback_list = CallbackList([checkpoint_callback, eval_callback])

# custom policy arguments
policy_kwargs = dict(
    features_extractor_class = CustomCNN,
    features_extractor_kwargs = dict(features_dim=64),
    net_arch = dict(qf=[128, 64, 32], pi=[128, 64, 64])
)

# create SAC model
model = SAC('CnnPolicy', env, buffer_size=100000, batch_size=256,
            policy_kwargs=policy_kwargs, tensorboard_log=tb_log_path)

# train the model: 50K time steps is adequate
start_time = datetime.now()
model.learn(total_timesteps=250000, log_interval=4, tb_log_name='rc_sac', callback=callback_list)
end_time = datetime.now()
print('Training time: ', end_time - start_time)

# Evaluate the trained model
mean, std = evaluate_policy(model, env, n_eval_episodes=50, deterministic=True)
print('Evaluate the model after training: {} +/- {}'.format(mean, std))

