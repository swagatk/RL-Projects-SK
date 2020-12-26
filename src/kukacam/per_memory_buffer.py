"""
Memory Buffer for Priority Experience Replay
"""
from sumtree import SumTree
import numpy as np
import pickle


class Memory:
    def __init__(self, max_capacity: int, batch_size: int,
                 state_size: tuple, action_size: tuple):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = max_capacity

        # buffer to store (s,a,r,s',d) tuples
        self.buffer = [(np.zeros(shape=self.state_size),
                        np.zeros(shape=self.action_size),
                        0.0,
                        np.zeros(shape=self.state_size),
                        0.0) for i in range(self.buffer_size)]

        # Initially all priorities are set to zero
        self.sum_tree = SumTree([0 for i in range(self.buffer_size)])

        self.curr_write_idx = 0
        self.available_samples = 0

        self.beta = 0.4  # Importance sampling factor
        self.alpha = 0.6  # priority factor
        self.min_priority = 0.01
        self.batch_size = batch_size

    def __len__(self):
        return self.available_samples

    def record(self, experience: tuple, priority: float):
        # add the experience to the buffer
        self.buffer[self.curr_write_idx] = experience

        # update the priority of this experience in the sum tree
        self.update(self.curr_write_idx, priority)

        self.curr_write_idx += 1

        if self.curr_write_idx >= self.buffer_size:
            self.curr_write_idx = 0

        if self.available_samples < self.buffer_size:
            self.available_samples += 1  # max value = self.buffer_size

    def adjust_priority(self, priority: float):
        return np.power(priority + self.min_priority, self.alpha)

    def update(self, idx: int, priority: float):
        self.sum_tree.update(self.sum_tree.leaf_nodes[idx],
                             self.adjust_priority(priority))

    def sample(self):
        sampled_idxs = []
        is_weights = []  # importance sampling weights
        sample_no = 0
        while sample_no < self.batch_size:
            sample_val = np.random.uniform(0, self.sum_tree.root_node.value)
            sample_node = self.sum_tree.retrieve(sample_val, self.sum_tree.root_node)

            # check if this is a valid idx
            if sample_node.idx < self.available_samples:
                sampled_idxs.append(sample_node.idx)
                p = sample_node.value / (self.sum_tree.root_node.value + 1e-3)  # avoid singularity
                is_weights.append(self.available_samples * p)  # give equal weights
            sample_no += 1
        # while loop ends here
        # apply beta factor and normalise so that maximum is_weight < 1
        is_weights = np.array(is_weights)
        is_weights = np.power(is_weights, -self.beta)
        is_weights = is_weights / np.max(is_weights)  # normalize to (0, 1)

        # load states and next_states
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        for i in range(len(sampled_idxs)):
            state_batch.append(self.buffer[sampled_idxs[i]][0])
            action_batch.append(self.buffer[sampled_idxs[i]][1])
            reward_batch.append(self.buffer[sampled_idxs[i]][2])
            next_state_batch.append(self.buffer[sampled_idxs[i]][3])
            done_batch.append(self.buffer[sampled_idxs[i]][4])
        return state_batch, action_batch, reward_batch,\
               next_state_batch, done_batch, sampled_idxs, is_weights

    def save_priorities_txt(self, filename):
        priorities = self.sum_tree.get_priorities()

        with open(filename, 'w') as file:
            for i in range(self.buffer_size):
                file.write('{}\t{}\n'.format(i, priorities[i]))

    def save_data(self, buffer_filename):
        # get priorities
        priorities = self.sum_tree.get_priorities()
        parameters = (
            self.buffer,
            self.curr_write_idx,
            self.available_samples,
            priorities
        )
        with open(buffer_filename, 'wb') as file:
            pickle.dump(parameters, file)

    def load_data(self, buffer_filename):
        with open(buffer_filename, 'rb') as file:
            parameters = pickle.load(file)

        self.buffer, self.curr_write_idx, \
            self.available_samples, priorities = parameters
        self.sum_tree = SumTree(priorities)









