"""
Sum Tree Implementation

"""
import numpy as np
import random


class Node:
    def __init__(self, left, right, is_leaf: bool = False, idx=None):
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        if not self.is_leaf:
            self.value = self.left.value + self.right.value
        self.parent = None
        self.idx = idx
        if left is not None:
            left.parent = self
        if right is not None:
            right.parent = self

    @classmethod
    def create_leaf(cls, value, idx):
        leaf = cls(None, None, is_leaf=True, idx=idx)
        leaf.value = value
        return leaf


class SumTree:
    def __init__(self, input_list: list):
        nodes = [Node.create_leaf(v, i) for i, v in enumerate(input_list)]
        self.leaf_nodes = nodes
        while len(nodes) > 1:
            inodes = iter(nodes)
            nodes = [Node(*pair) for pair in zip(inodes, inodes)]
        self.root_node = nodes[0]

    def retrieve(self, value: float, node: Node):
        if node.is_leaf:
            return node
        if node.left.value >= value:
            return self.retrieve(value, node.left)
        else:
            return self.retrieve(value - node.left.value, node.right)

    def update(self, node: Node, new_value: float):
        change = new_value - node.value
        node.value = new_value
        self.propagate_changes(change, node.parent)

    def propagate_changes(self, change: float, node: Node):
        node.value += change
        if node.parent is not None:
            self.propagate_changes(change, node.parent)

    def get_priorities(self):
        priorities = []
        for i in range(len(self.leaf_nodes)):
            priorities.append(self.leaf_nodes[i].value)
        return priorities

    def demonstrate_sampling(self):
        tree_total = self.root_node.value
        iterations = 100000
        selected_vals = []
        for i in range(iterations):
            rand_val = np.random.uniform(0, tree_total)
            selected_node = self.retrieve(rand_val, self.root_node)
            selected_val = selected_node.value
            selected_vals.append(selected_val)
        return selected_vals


if __name__ == '__main__':
    input = [1, 4, 2, 3]
    s = SumTree(input)
    selected_values = s.demonstrate_sampling()

    print('should be ~4:', sum([1 for x in selected_values if x == 4]) /
          sum([1 for y in selected_values if y == 1]))