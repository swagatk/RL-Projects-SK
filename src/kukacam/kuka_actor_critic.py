import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from actor_critic import Actor, Critic, AC_Agent, OUActionNoise
from collections import deque
import random

#######################################
# avoid CUDNN_STATUS_INTERNAL_ERROR
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
################################################
# required for reproducing the result
np.random.seed(1)
tf.random.set_seed(1)


############################################
# ACTOR
##############################
class KukaActor(Actor):
    def __init__(self, state_size, action_size,
                 replacement, learning_rate,
                 upper_bound):
        super().__init__(state_size, action_size, replacement,
                         learning_rate, upper_bound)

    def _build_net(self):
        # input is a stack of 1-channel YUV images
        last_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03)
        state_input = layers.Input(shape=self.state_size)
        conv_l1 = layers.Conv2D(15, kernel_size=6, strides=2,
                                padding="SAME", activation="relu")(state_input)
        max_pool1 = layers.MaxPool2D(pool_size=3, strides=2)(conv_l1)
        f1 = layers.Flatten()(max_pool1)
        l1 = layers.Dense(256, activation='relu')(f1)
        l2 = layers.Dense(256, activation='relu')(l1)
        net_out = layers.Dense(self.action_size, activation='tanh',
                               kernel_initializer=last_init)(l2)

        net_out = net_out * self.upper_bound
        model = keras.Model(state_input, net_out)
        model.summary()
        return model


###################################
class KukaCritic(Critic):
    def __init__(self, state_size, action_size, replacement,
                 learning_rate, gamma):
        super().__init__(state_size, action_size,  replacement,
                         learning_rate, gamma)

    def _build_net(self):
        # state input is a stack of 1-D YUV images
        state_input = layers.Input(shape=self.state_size)
        conv_l1 = layers.Conv2D(15, kernel_size=6, strides=2,
                                padding="SAME", activation="relu")(state_input)
        max_pool1 = layers.MaxPool2D(pool_size=3, strides=2)(conv_l1)
        f1 = layers.Flatten()(max_pool1)

        state_out = layers.Dense(16, activation="relu")(f1)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(self.action_size,))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through separate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        net_out = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model(inputs=[state_input, action_input], outputs=net_out)
        model.summary()
        return model

#########################################
class KukaBuffer():
    def __init__(self, buffer_capacity, batch_size):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=self.buffer_capacity)

    def record(self, state, action, reward, next_state):
        self.buffer.append([state, action, reward, next_state])

    def sample(self):
        valid_batch_size = min(len(self.buffer), self.batch_size)
        mini_batch = random.sample(self.buffer, valid_batch_size)

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        for i in range(valid_batch_size):
            state_batch.append(mini_batch[i][0])
            action_batch.append(mini_batch[i][1])
            reward_batch.append(mini_batch[i][2])
            next_state_batch.append(mini_batch[i][3])

        # convert to tensors
        state_batch = tf.convert_to_tensor(state_batch)
        action_batch = tf.convert_to_tensor(action_batch)
        reward_batch = tf.convert_to_tensor(reward_batch)
        next_state_batch = tf.convert_to_tensor(next_state_batch)

        return state_batch, action_batch, reward_batch, next_state_batch



class KukaACAgent(AC_Agent):
    def __init__(self, state_size, action_size,
                 replacement, lr_a, lr_c,
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

        self.actor = KukaActor(self.state_size, self.action_size, self.replacement,
                           self.actor_lr, self.upper_bound)
        self.critic = KukaCritic(self.state_size, self.action_size, self.replacement,
                             self.critic_lr, self.gamma)
        self.buffer = KukaBuffer(self.memory_capacity, self.batch_size)

        std_dev = 0.2
        self.noise_object = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

        # Initially make weights for target and model equal
        self.actor.target.set_weights(self.actor.model.get_weights())
        self.critic.target.set_weights(self.critic.model.get_weights())


    def policy(self, state):
        # Check the size of state: 1, 256, 341, 1
        sampled_action = tf.squeeze(self.actor.model(state))
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
        state_batch, action_batch, reward_batch, \
        next_state_batch = self.buffer.sample()

        self.actor.train(state_batch, self.critic)
        self.critic.train(state_batch, action_batch, reward_batch,
                          next_state_batch, self.actor)

    def update_targets(self):
        self.actor.update_target()
        self.critic.update_target()

