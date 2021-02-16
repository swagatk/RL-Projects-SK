"""
Implementing DDPG for Kuka Diverse Object Environment

Here the Actor and Critic do not share common Conv Layer to extract features.
Each one of the actor and critic network have their own Conv Layers to extract features from the input images.

Status: No success. The average reward for last 100 episodes after 4K episodes is about 0.2.
Todo: Fill the replay buffer with random experiences in the beginning.
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random

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
################################################

######################
# Seed Initialization
# required for reproducing the result
#np.random.seed(1)
#tf.random.set_seed(1)
#####################################


##########################
# NOISE MODEL
#################################
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


############################################
# ACTOR
##############################
class KukaActor:
    def __init__(self, state_size, action_size,
                 replacement, learning_rate,
                 upper_bound):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.replacement = replacement
        self.upper_bound = upper_bound
        self.train_step_count = 0

        # create NN models
        self.model = self._build_net()
        self.target = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        # input is a stack of 1-channel YUV images
        last_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03)
        state_input = layers.Input(shape=self.state_size)
        conv1 = layers.Conv2D(15, kernel_size=6, strides=2,
                                padding="SAME", activation="relu")(state_input)
        bn1 = layers.BatchNormalization()(conv1)
        conv2 = layers.Conv2D(32, kernel_size=6, strides=2,
                              padding="SAME", activation="relu")(bn1)
        bn2 = layers.BatchNormalization()(conv2)
        conv3 = layers.Conv2D(32, kernel_size=6, strides=2,
                              padding="SAME", activation="relu")(bn2)
        bn3 = layers.BatchNormalization()(conv3)

        #max_pool1 = layers.MaxPool2D(pool_size=3, strides=2)(conv_l1)
        f1 = layers.Flatten()(bn3)

        l1 = layers.Dense(128, activation='relu')(f1)
        l2 = layers.Dense(64, activation='relu')(l1)
        net_out = layers.Dense(self.action_size[0], activation='tanh',
                               kernel_initializer=last_init)(l2)

        net_out = net_out * self.upper_bound
        model = keras.Model(state_input, net_out)
        model.summary()
        return model

    def update_target(self):

        if self.replacement['name'] == 'hard':
            if self.train_step_count % \
                    self.replacement['rep_iter_a'] == 0:
                self.target.set_weights(self.model.get_weights())
        else:
            w = np.array(self.model.get_weights())
            w_dash = np.array(self.target.get_weights())
            new_wts = self.replacement['tau'] * w + \
                      (1 - self.replacement['tau']) * w_dash
            self.target.set_weights(new_wts)

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def train(self, state_batch, critic):
        self.train_step_count += 1
        with tf.GradientTape() as tape:
            actor_weights = self.model.trainable_variables
            actions = self.model(state_batch)
            critic_value = critic.model([state_batch, actions])
            # -ve value is used to maximize value function
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_weights)
        self.optimizer.apply_gradients(zip(actor_grad, actor_weights))
        return actor_loss


###################################
# CRITIC
#####################
class KukaCritic:
    def __init__(self, state_size, action_size,
                 replacement,
                 learning_rate=1e-3,
                 gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.replacement = replacement
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.model = self._build_net()
        self.target = self._build_net()
        self.gamma = gamma
        self.train_step_count = 0

    def _build_net(self):
        # state input is a stack of 1-D YUV images
        state_input = layers.Input(shape=self.state_size)
        conv1 = layers.Conv2D(15, kernel_size=6, strides=2,
                                padding="SAME", activation="relu")(state_input)
        bn1 = layers.BatchNormalization()(conv1)
        conv2 = layers.Conv2D(32, kernel_size=6, strides=2,
                              padding="SAME", activation="relu")(bn1)
        bn2 = layers.BatchNormalization()(conv2)

        conv3 = layers.Conv2D(32, kernel_size=6, strides=2,
                              padding="SAME", activation="relu")(bn2)
        bn3 = layers.BatchNormalization()(conv3)

        #max_pool1 = layers.MaxPool2D(pool_size=3, strides=2)(conv_l1)
        f1 = layers.Flatten()(bn3)

        state_out = layers.Dense(32, activation="relu")(f1)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=self.action_size)
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through separate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(128, activation="relu")(concat)
        out = layers.Dense(64, activation="relu")(out)
        net_out = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model(inputs=[state_input, action_input], outputs=net_out)
        model.summary()
        return model

    def update_target(self):
        if self.replacement['name'] == 'hard':
            if self.train_step_count % \
                    self.replacement['rep_iter_a'] == 0:
                self.target.set_weights(self.model.get_weights())
        else:
            w = np.array(self.model.get_weights())
            w_dash = np.array(self.target.get_weights())
            new_wts = self.replacement['tau'] * w + \
                      (1 - self.replacement['tau']) * w_dash
            self.target.set_weights(new_wts)

    def train(self, state_batch, action_batch, reward_batch,
              next_state_batch, actor):
        self.train_step_count += 1
        with tf.GradientTape() as tape:
            critic_weights = self.model.trainable_variables
            target_actions = actor.target(next_state_batch)
            target_critic = self.target([next_state_batch, target_actions])
            y = reward_batch + self.gamma * target_critic
            critic_value = self.model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_weights)
        self.optimizer.apply_gradients(zip(critic_grad, critic_weights))
        return critic_loss

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


#########################################
# REPLAY BUFFER
####################################
class KukaBuffer:
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
        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)

        return state_batch, action_batch, reward_batch, next_state_batch


#######################################
# Actor-Critic Agent for Kuka Environment
##################################
class KukaActorCriticAgent:
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

        self.noise_object = OUActionNoise(mean=np.zeros(1), std_deviation=0.2 * np.ones(1))

        # Initially make weights for target and model equal
        self.actor.target.set_weights(self.actor.model.get_weights())
        self.critic.target.set_weights(self.critic.model.get_weights())

    def policy(self, state):
        # Check the size of state: (w,h,c) - its a numpy array

        # convert the numpy array state into a tensor of size (1, w, h, c)
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

        sampled_action = tf.squeeze(self.actor.model(tf_state))
        noise = self.noise_object()  # scalar value

        # convert into the same shape as that of the action vector
        noise_vec = noise * np.ones(shape=self.action_size)

        # Add noise to the action
        sampled_action = sampled_action.numpy() + noise_vec # check if we need to add noise

        # Make sure that the action is within bounds
        valid_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        return np.squeeze(valid_action)

    def experience_replay(self):
        # sample from stored memory
        state_batch, action_batch, reward_batch,\
                    next_state_batch = self.buffer.sample()

        actor_loss = self.actor.train(state_batch, self.critic)
        critic_loss = self.critic.train(state_batch, action_batch, reward_batch,
                          next_state_batch, self.actor)

        return actor_loss, critic_loss

    def update_targets(self):
        self.actor.update_target()
        self.critic.update_target()

