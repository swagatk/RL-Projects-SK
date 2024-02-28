"""
Replay buffer for storing experiences
"""
import numpy as np
from collections import deque
import random
import pickle
import tensorflow as tf
import sys
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))


#########################################
# REPLAY BUFFER
####################################
class Buffer:
    def __init__(self, buffer_capacity, batch_size):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=self.buffer_capacity)

    def __len__(self):
        return len(self.buffer)

    def record(self, experience: list):
        self.buffer.append(experience)

    def sample(self):
        valid_batch_size = min(len(self.buffer), self.batch_size)
        mini_batch = random.sample(self.buffer, valid_batch_size)

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        for i in range(valid_batch_size):
            state_batch.append(mini_batch[i][0])
            action_batch.append(mini_batch[i][1])
            reward_batch.append(mini_batch[i][2])
            next_state_batch.append(mini_batch[i][3])
            done_batch.append(mini_batch[i][4])
        # for-loop ends
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def save_data(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.buffer, file)

    def load_data(self, filename):
        with open(filename, 'rb') as file:
            self.buffer = pickle.load(file)

    def get_samples(self, n_samples=None):
        if n_samples is None or n_samples > len(self.buffer):
            n_samples = len(self.buffer) # return all samples if nothing is specified

        s_batch = []
        a_batch = []
        r_batch = []
        ns_batch = []
        d_batch = []
        for i in range(n_samples):
            s_batch.append(self.buffer[i][0])
            a_batch.append(self.buffer[i][1])
            r_batch.append(self.buffer[i][2])
            ns_batch.append(self.buffer[i][3])
            d_batch.append(self.buffer[i][4])

        return s_batch, a_batch, r_batch, ns_batch, d_batch

    def clear(self):
        self.buffer.clear()


############################
# HER BUFFER
##########################
class HERBuffer:
    def __init__(self, buffer_capacity, batch_size):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=self.buffer_capacity)

    def __len__(self):
        return len(self.buffer)

    def record(self, experience: list):
        self.buffer.append(experience)

    def sample(self):
        valid_batch_size = min(len(self.buffer), self.batch_size)
        mini_batch = random.sample(self.buffer, valid_batch_size)

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        goal_batch = []
        for i in range(valid_batch_size):
            state_batch.append(mini_batch[i][0])
            action_batch.append(mini_batch[i][1])
            reward_batch.append(mini_batch[i][2])
            next_state_batch.append(mini_batch[i][3])
            done_batch.append(mini_batch[i][4])
            goal_batch.append(mini_batch[i][5])
        # for-loop ends
        return state_batch, action_batch, reward_batch,\
               next_state_batch, done_batch, goal_batch

    def get_all_samples(self):
        s_batch, a_batch, r_batch, ns_batch, d_batch, g_batch\
            = [], [], [], [], [], []
        for i in range(len(self.buffer)):
            s_batch.append(self.buffer[i][0])
            a_batch.append(self.buffer[i][1])
            r_batch.append(self.buffer[i][2])
            ns_batch.append(self.buffer[i][3])
            d_batch.append(self.buffer[i][4])
            g_batch.append(self.buffer[i][5])
        # outside for-loop
        return s_batch, a_batch, r_batch, ns_batch, d_batch, g_batch


##################################
## Replay Buffer - memory efficient
# does not make use of collections.deque()
#####################################

class ReplayBuffer():
    "Buffer to store environment transitions"
    def __init__(self,
                 obs_shape,
                 action_shape,
                 capacity) -> None:

        self.capacity = capacity
        self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False


    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.dones[self.idx], done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0


    def sample(self, batch_size=24):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        dones = self.dones[idxs]

        return obses, actions, rewards, next_obses, dones


    def __getitem__(self, index):
      if index >= 0 and index < self.capacity if self.full else self.idx:
        obs = self.obses[index]
        action = self.actions[index]
        reward = self.rewards[index]
        next_obs = self.next_obses[index]
        done = self.dones[index]
        return obs, action, reward, next_obs, done
      else:
        raise ValueError('Index is out of range')

    def __len__(self):
        return self.capacity if self.full else self.idx



