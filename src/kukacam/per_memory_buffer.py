"""
Memory Buffer for Priority Experience Replay
"""
from sumtree import SumTree
import numpy as np
from collections import deque


class Memory:
    def __init__(self, max_capacity: int, batch_size: int):
        self.size = max_capacity
        self.curr_write_idx = 0
        self.buffer = deque(maxlen=self.size)
        self.sum_tree = SumTree([0 for i in range(self.size)])

        self.beta = 0.4  # Importance sampling factor
        self.alpha = 0.6  # priority factor
        self.min_priority = 0.01
        self.batch_size = batch_size

    def record(self, experience: tuple, priority: float):
        self.buffer.append(experience)

        # update the priority of this experience in the sum tree
        self.update(self.curr_write_idx, priority)

        self.curr_write_idx += 1

        # reset the current writer position if it exceeds the buffer size
        if self.curr_write_idx >= self.size:
            self.curr_write_idx = 0

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
            if sample_node.idx < len(self.buffer):
                sampled_idxs.append(sample_node.idx)
                p = sample_node.value / self.sum_tree.root_node.value
                is_weights.append(len(self.buffer) * p)
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
        for i in range(len(sampled_idxs)):
            state_batch.append(self.buffer[sampled_idxs[i]][0])
            action_batch.append(self.buffer[sampled_idxs[i]][1])
            reward_batch.append(self.buffer[sampled_idxs[i]][2])
            next_state_batch.append(self.buffer[sampled_idxs[i]][3])

        return state_batch, action_batch, reward_batch, next_state_batch, sampled_idxs, is_weights




