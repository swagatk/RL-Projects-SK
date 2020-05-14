import random
import numpy as np
from collections import deque
from .sumtree import SumTree

class MemoryBuffer:
    def __init__(self, buffer_size, with_per=False):
        
        self.buffer_size = buffer_size
        self.with_per = with_per
        self.count = 0

        if(self.with_per):
            self.alpha = 0.5
            self.epsilon = 0.01
            self.buffer = SumTree(self.buffer_size)
        else:
            self.buffer = deque(maxlen=self.buffer_size)

    def memorize(self, state, action, reward, next_state, done,
                 error=None):
        experience = (state, action, reward, next_state, done)

        if self.with_per:
            priority = self.priority(error[0])
            self.buffer.add(priority, experience)
            self.count += 1
        else:
            if self.count < self.buffer_size:
                self.buffer.append(experience)
                self.count += 1
            else:
                self.buffer.append(experience)


    def priority(self, error):
        return (error + self.epsilon) ** self.alpha

    def size(self):
        if self.with_per:
            return self.count
        else:
            return len(self.buffer)

    def sample_batch(self, batch_size):
        batch = []

        if self.with_per:
            T = self.buffer.total() // batch_size
            for i in range(batch_size):
                a = T * i
                b = T * (i+1)
                s = random.uniform(a, b)
                idx, error, data = self.buffer.get(s)
                batch.append((data,idx))
            idx = np.array([i[5] for i in batch])

        elif self.count < batch_size:
            idx = None
            batch = random.sample(self.buffer, self.count)
        else:
            idx = None
            batch = random.sample(self.buffer, batch_size)


        # return a batch of experience
        s_batch = np.array([i[0] for i in batch])
        a_batch = np.array([i[1] for i in batch])
        r_batch = np.array([i[2] for i in batch])
        new_s_batch = np.array([i[3] for i in batch])
        d_batch = np.array([i[4] for i in batch])
        return s_batch, a_batch, r_batch, new_s_batch, d_batch, idx

    def update(self, idx, new_error):
        self.buffer.update(idx, self.priority(new_error))

    def clear(self):
        if(self.with_per): self.buffer=SumTree(buffer_size)
        else: self.buffer = deque(maxlen=buffer_size)
        self.count = 0




