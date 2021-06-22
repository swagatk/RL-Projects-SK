"""
Replay buffer for storing experiences
"""
import numpy as np
from collections import deque
import random
import pickle
import tensorflow as tf


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

    def record(self, experience: tuple):
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

    def record(self, state, action, reward, next_state, done, goal):
        self.buffer.append([state, action, reward, next_state, done, goal])

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





