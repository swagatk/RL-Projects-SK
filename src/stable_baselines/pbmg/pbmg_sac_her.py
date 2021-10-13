"""
Applying SB3.SAC.HER to pybullet multigoal environment
"""
import gym
import os
import numpy as np
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import pybullet_multigoal_gym as pmg 

#from ..racecar.custom_cnn import CustomCNN


class NormalizeObsvnWrapper(gym.Wrapper):
  """
  :param env: (gym.Env)   Gym environment that will be wrapped
  """
  def __init__(self, env):
    #assert isinstance(env.observation_space, gym.spaces.Box),\
    # "Valid for continuous observation spaces of type gym.spaces.Box"

    self._height = env.observation_space.shape[0]
    self._width = env.observation_space.shape[1]
    self._channels = env.observation_space.shape[2]

    env.observation_space = gym.spaces.Box(low=0, high=255,
                                            shape=(self._channels, 
                                                   self._height,
                                                   self._width))
    env.reward_range = (-np.inf, np.inf)
    # call the parent constructor so that we can access self.env
    super(NormalizeObsvnWrapper, self).__init__(env)
    self.env.reward_range = (-np.inf, np.inf)


  def _modify_obsvn(self, obs):
    new_obs = np.transpose(obs, (2, 0, 1))
    new_obs = np.asarray(new_obs, dtype=np.float32) / 255.0
    return new_obs

  def reset(self):
    """
    Convert Images from HxWxC format to CxHxW
    Normalize the pixels between 0 and 1.0
    """
    return self._modify_obsvn(self.env.reset())
  
  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    new_obs = self._modify_obsvn(obs)
    info['channel_first'] = True
    info['nomalize pixel'] = True
    return new_obs, reward, done, info

######################################33333333

class FlattenObservation(gym.ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)
        self.observation_space = spaces.flatten_space(env.observation_space)

    def observation(self, observation):
        return gym.spaces.flatten(self.env.observation_space, observation)

class ResizeObservation(gym.ObservationWrapper):
    r"""Downsample the image observation to a square image."""

    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape

        self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        import cv2

        observation = cv2.resize(
            observation, self.shape[::-1], interpolation=cv2.INTER_AREA
        )
        if observation.ndim == 2:
            observation = np.expand_dims(observation, -1)
        return observation



###############################33
## initiate PMG environment
camera_setup = [
    {
        'cameraEyePosition': [-1.0, 0.25, 0.6],
        'cameraTargetPosition': [-0.6, 0.05, 0.2],
        'cameraUpVector': [0, 0, 1],
        'render_width': 128,
        'render_height': 128
    },
    {
        'cameraEyePosition': [-1.0, -0.25, 0.6],
        'cameraTargetPosition': [-0.6, -0.05, 0.2],
        'cameraUpVector': [0, 0, 1],
        'render_width': 128,
        'render_height': 128
    }
]

env = pmg.make_env(
    # task args ['reach', 'push', 'slide', 'pick_and_place', 
    #            'block_stack', 'block_rearrange', 'chest_pick_and_place', 'chest_push']
    task='reach',
    gripper='parallel_jaw',
    num_block=4,  # only meaningful for multi-block tasks
    render=False,
    binary_reward=True,
    max_episode_steps=20,
    # image observation args
    image_observation=False,
    depth_image=False,
    goal_image=True,
    visualize_target=True,
    camera_setup=camera_setup,
    observation_cam_id=0,
    goal_cam_id=1,
    # curriculum args
    use_curriculum=True,
    num_goals_to_generate=90)



# custom policy arguments
# policy_kwargs = dict(
#     features_extractor_class = CustomCNN,
#     features_extractor_kwargs = dict(features_dim=64),
#     net_arch = dict(qf=[128, 64, 32], pi=[128, 64, 64])
# )


# Note the wrapper classes work with Box observation space and not with gym.spaces.Dict
# add wrappers to the env
env = NormalizeObsvnWrapper(env)
#env = FlattenObservation(env)
env = ResizeObservation(env, 32)

tb_log_path = './tb_log/'
monitor_path = './monitor/'
os.makedirs(tb_log_path, exist_ok=True)
os.makedirs(monitor_path, exist_ok=True)

model = SAC( #'CnnPolicy', 
                'MultiInputPolicy',
                env, 
                buffer_size=100000, 
                batch_size=256,
                #policy_kwargs=policy_kwargs, 
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(
                    n_sampled_goal=4,
                    goal_selection_strategy='future',
                    online_sampling=True,
                    max_episode_length=5
                ),
                tensorboard_log=tb_log_path,
                verbose=1,
)

env = Monitor(env, monitor_path)

print('Training the model')
# Train the model
model.learn(total_timesteps=200000, log_interval=4, tb_log_name='pbmg_sac_her')


# Evaluate the trained model
mean, std = evaluate_policy(model, env, n_eval_episodes=50, deterministic=True)
print('Evaluate the model after training: {} +/- {}'.format(mean, std))

# save trained model
model.save('./pbmg_sac_her')

# load the model
model = SAC.load('./pbmg_sac_her', env=env)
# Test
obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    if done:
        obs = env.reset()