"""
Replay buffer for storing experiences
"""
import numpy as np
from collections import deque
import random
import pickle


#########################################
# REPLAY BUFFER
####################################
class KukaBuffer:
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

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def save_data(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.buffer, file)

    def load_data(self, filename):
        with open(filename, 'rb') as file:
            self.buffer = pickle.load(file)

    def get_all_samples(self):
        s_batch = []
        a_batch = []
        r_batch = []
        ns_batch = []
        d_batch = []
        for i in range(len(self.buffer)):
            s_batch.append(self.buffer[i][0])
            a_batch.append(self.buffer[i][1])
            r_batch.append(self.buffer[i][2])
            ns_batch.append(self.buffer[i][3])
            d_batch.append(self.buffer[i][4])

        return s_batch, a_batch, r_batch, ns_batch, d_batch




