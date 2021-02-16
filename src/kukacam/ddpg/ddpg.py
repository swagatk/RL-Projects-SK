"""
Implementing DDPG for Kuka Diverse Object Environment

- Here the Actor and Critic  share common Convolution Feature Network Layer to extract features from RGB images
- Both Actor and Critic update the Feature Network parameters during training. Weights of feature network are
updated twice in each iteration.
- Status: No success. The average reward for last 100 episodes after 4K episodes is about 0.2.
- Todo:
    - Display the Conv Layer output on Tensorboard.
    - Fill the replay buffer with random experiences in the beginning
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
import pickle

from OUActionNoise import OUActionNoise
from FeatureNet import FeatureNetwork
from actor import KukaActor
from critic import KukaCritic
from buffer import KukaBuffer
########################################
# check tensorflow version
from packaging import version
print("Tensorflow Version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This program requires Tensorflow 2.0 or above"
#######################################

#######################################
# avoid CUDNN_STATUS_INTERNAL_ERROR
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

####################################
# ACTOR-CRITIC AGENT
##################################
class DDPG_Agent:
    def __init__(self, state_size, action_size,
                 replacement,
                 lr_a, lr_c,
                 batch_size,
                 memory_capacity,
                 gamma,
                 upper_bound, lower_bound):
        self.state_size = state_size
        self.action_size = action_size
        self.replacement = replacement
        self.actor_lr = lr_a
        self.critic_lr = lr_c
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.gamma = gamma
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self.feature = FeatureNetwork(self.state_size, self.actor_lr)

        self.actor = KukaActor(self.state_size, self.action_size, self.replacement,
                               self.actor_lr, self.upper_bound, self.feature)
        self.critic = KukaCritic(self.state_size, self.action_size, self.replacement,
                                 self.critic_lr, self.gamma, self.feature)
        self.buffer = KukaBuffer(self.memory_capacity, self.batch_size)

        std_dev = 0.2
        self.noise_object = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

        # Initially make weights for target and model equal
        self.actor.target.set_weights(self.actor.model.get_weights())
        self.critic.target.set_weights(self.critic.model.get_weights())

    def record(self, experience: tuple):
        self.buffer.record(experience)

    def policy(self, state):
        # Check the size of state: (w, h, c)
        # convert the numpy array state into a tensor of size (1, w, h, c)
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

        sampled_action = tf.squeeze(self.actor.model(tf_state))
        noise = self.noise_object()  # scalar value

        # convert into the same shape as that of the action vector
        noise_vec = noise * np.ones(self.action_size)

        # Add noise to the action
        sampled_action = sampled_action.numpy() + noise_vec

        # Make sure that the action is within bounds
        valid_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        return np.squeeze(valid_action)

    def experience_replay(self):
        # sample from stored memory
        state_batch, action_batch, reward_batch,\
                            next_state_batch, done_batch = self.buffer.sample()

        # convert to tensors
        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)
        done_batch = tf.convert_to_tensor(done_batch, dtype=tf.float32)

        actor_loss = self.actor.train(state_batch, self.critic)
        critic_loss = self.critic.train(state_batch, action_batch, reward_batch,
                                        next_state_batch, done_batch, self.actor)

        return actor_loss, critic_loss

    def update_targets(self):
        self.actor.update_target()
        self.critic.update_target()

    def save_model(self, path, actor_filename, critic_filename,
                   replay_filename):
        actor_file = path + actor_filename
        critic_file = path + critic_filename
        replay_file = path + replay_filename

        self.actor.save_weights(actor_file)
        self.critic.save_weights(critic_file)
        self.buffer.save_data(replay_file)

    def load_model(self, path, actor_filename, critic_filename,
                   replay_filename):
        actor_file = path + actor_filename
        critic_file = path + critic_filename
        replay_file = path + replay_filename

        self.actor.model.load_weights(actor_file)
        self.actor.target.load_weights(actor_file)
        self.critic.model.load_weights(critic_file)
        self.critic.target.load_weights(critic_file)
        self.buffer.load_data(replay_file)

