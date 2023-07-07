from gym.core import Env
import numpy as np
from numpy.random import randint
import os 
import gym
import torch
import torch.nn.functional as F
import dmc2gym
from collections import deque 

def make_env(
        domain_name,
        task_name,
        seed=0,
        episode_length=1000,
        frame_stack=3,
        action_repeat=4,
        image_size=100,
        mode='train',
    ):
    """ Make environment for experiment """
    assert mode in {'train', 'color_easy', 'color_hard', 'video_easy', 'video_hard'},\
            f'specified mode "{mode}" is not supported'
    
    
    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
        visualize_reward=False,
        from_pixels=True,
        height=image_size,
        width=image_size,
        episode_length=episode_length,
        frame_skip=action_repeat
    )

    env = FrameStack(env, frame_stack)

    return env


class FrameStack(gym.Wrapper):
    def __init__(self, env: Env, k):
        super().__init__(env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shape[0]*k,) + shape[1:]),
            dtype=env.observation_space.dtype,
        )
        self._max_episode_steps = env._max_episode_steps 

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()


    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info 
    
    def _get_obs(self):
        assert len(self._frames) == self._k 
        return np.concatenate(list(self._frames), axis=0)