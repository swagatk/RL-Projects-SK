"""
This will be used for DMC environments
"""
from turtle import shape
import dmc2gym
import gym
from collections import deque
import numpy as np
import json

###############################
# FrameStackWrapper

class FrameStackWrapper4DMC(gym.Wrapper):
    """
    This generates stacked frames for DMC environments.
    By default, the frames are in channel first format.
    It needs to be converted to channel last format for
    use with Tensorflow.
    Returns: stacked frames in channel last format: (H, W, K)
    """
    def __init__(self, env, k, 
                    channel_first=False) -> None:
        super().__init__(env)

        self._k = k # number of frames to stack
        self._frames = deque([], maxlen=self._k)
        self.channel_first=channel_first

        # in DMC, it is in channel first format
        obs_shape = env.observation_space.shape

        if self.channel_first:   
            self.observation_space = gym.spaces.Box(
                low=0, high=255, 
                shape=(obs_shape[0]*self._k, obs_shape[1], obs_shape[2]), 
                dtype=env.observation_space.dtype
                )
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, 
                shape=(obs_shape[1], obs_shape[2], obs_shape[0]*self._k), 
                dtype=env.observation_space.dtype
                )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        if self.channel_first == False:
            obs = np.transpose(obs, (1, 2, 0))
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.channel_first == False:
            obs = np.transpose(obs, (1, 2, 0))
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k 
        if self.channel_first:
            stacked_frame = np.concatenate(self._frames, axis=0) # k*C, H, W
        else:
            stacked_frame = np.concatenate(self._frames, axis=2) # H, W, C*k
        return stacked_frame

###############################
if __name__ == '__main__':


    env = dmc2gym.make(
        domain_name='cheetah',
        task_name='run',
        seed=42,
        visualize_reward=False,
        from_pixels=True,
        height=64,
        width=64,
        frame_skip=10
    )

    env = FrameStackWrapper4DMC(env, k=3, channel_first=False)

    print('\nObservation shape:', env.observation_space.shape)
    print('\nAction shape:', env.action_space.shape)
    print('\nAction upper bound:', env.action_space.high)
    print('\nObservation space dtype:', env.observation_space.dtype)

    state = env.reset()
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    print('\nshape of state: ', np.shape(state) )
    print('shape of next_state:', np.shape(next_state))

