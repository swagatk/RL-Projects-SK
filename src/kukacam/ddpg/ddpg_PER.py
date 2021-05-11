"""
DDPG + PER (PRIORITY EXPERIENCE REPLAY)
Status: Not working. The average reward over 100 episodes remains below 0.3 even after 12K episodes.
Buffer size is limited to 30K only. It gives error on increasing the buffer size
Todo: Focus on code reuse
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
from common.per_memory_buffer import Memory
from tensorflow.keras import layers
import random
from common.FeatureNet import FeatureNetwork
from common.OUActionNoise import OUActionNoise
########################################
# check tensorflow version
from packaging import version
print("Tensorflow Version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This program requires Tensorflow 2.0 or above"
##############################
# For reproducibility
random.seed(2212)
np.random.seed(2212)
tf.random.set_seed(2212)

#######################################
# Flags
_DEBUG = False
#######################################
# avoid CUDNN_STATUS_INTERNAL_ERROR
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


#########################################
# Actor Network
class KukaActor:
    def __init__(self, state_size, action_size,
                 replacement, learning_rate,
                 upper_bound, feature_model):
        self.state_size = state_size   # shape: (w, h, c)
        self.action_size = action_size  # shape: (n, )
        self.lr = learning_rate
        self.replacement = replacement
        self.upper_bound = upper_bound
        self.train_step_count = 0

        # create NN models
        self.feature_model = feature_model
        self.model = self._build_net()
        self.target = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        # Initially both models share same weights
        self.target.set_weights(self.model.get_weights())

    def _build_net(self):
        # input is a stack of 1-channel YUV images
        last_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03)

        state_input = layers.Input(shape=self.state_size)
        feature = self.feature_model(state_input)

        x = layers.Dense(128, activation="relu")(feature)
        x = layers.Dense(64, activation="relu")(x)
        net_out = layers.Dense(self.action_size[0], activation='tanh',
                               kernel_initializer=last_init)(x)

        net_out = net_out * self.upper_bound  # element-wise product
        model = keras.Model(state_input, net_out)
        model.summary()
        keras.utils.plot_model(model, to_file='actor_net.png',
                               show_shapes=True, show_layer_names=True)
        return model

    def __call__(self, state):
        # input is a tensor
        action = tf.squeeze(self.model(state))
        return action

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
        #actor_grad = [tf.clip_by_norm(g, 1e-3) for g in actor_grad]     # gradient clipping
        self.optimizer.apply_gradients(zip(actor_grad, actor_weights))
        return actor_loss


#######################################
# Critic Network
class KukaCritic:
    def __init__(self, state_size, action_size,
                 replacement,
                 learning_rate,
                 gamma, feature_model):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.replacement = replacement
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.gamma = gamma
        self.train_step_count = 0

        # create two models
        self.feature_model = feature_model
        self.model = self._build_net()
        self.target = self._build_net()

        # Initially both models share same weights
        self.target.set_weights(self.model.get_weights())

    def _build_net(self):
        # state input is a stack of 1-D YUV images
        state_input = layers.Input(shape=self.state_size)

        feature = self.feature_model(state_input)
        state_out = layers.Dense(32, activation="relu")(feature)

        # Action as input
        action_input = layers.Input(shape=self.action_size)
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through separate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(128, activation="relu")(concat)
        out = layers.Dense(64, activation="relu")(out)
        out = layers.Dense(32, activation="relu")(out)
        net_out = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model(inputs=[state_input, action_input], outputs=net_out)
        model.summary()
        keras.utils.plot_model(model, to_file='critic_net.png',
                               show_shapes=True, show_layer_names=True)
        return model

    def __call__(self, state, action):
        # inputs are tensors
        value = tf.squeeze(self.model([state, action]))
        return value

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
              next_state_batch, done_batch, actor):
        self.train_step_count += 1
        with tf.GradientTape() as tape:
            critic_weights = self.model.trainable_variables
            target_actions = actor.target(next_state_batch)
            target_critic = self.target([next_state_batch, target_actions])
            y = reward_batch + self.gamma * (1 - done_batch) * target_critic
            critic_value = self.model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_weights)
        #critic_grad = [tf.clip_by_value(g, 1e-3) for g in critic_grad]      # Gradient Clipping
        self.optimizer.apply_gradients(zip(critic_grad, critic_weights))
        return critic_loss

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


####################################
# ACTOR-CRITIC AGENT
##################################
class DDPGPERAgent:
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

        self.start_episode = 0
        self.ep_reward_list = []
        self.avg_reward_list = []

        self.feature = FeatureNetwork(self.state_size, self.actor_lr)

        self.actor = KukaActor(self.state_size, self.action_size, self.replacement,
                               self.actor_lr, self.upper_bound, self.feature)
        self.critic = KukaCritic(self.state_size, self.action_size, self.replacement,
                                 self.critic_lr, self.gamma, self.feature)
        self.buffer = Memory(self.memory_capacity, self.batch_size,
                             self.state_size, self.action_size)

        std_dev = 0.2
        self.noise_object = OUActionNoise(mean=np.zeros(1),
                                          std_deviation=float(std_dev) * np.ones(1))

        # Initially make weights for target and model equal
        self.actor.target.set_weights(self.actor.model.get_weights())
        self.critic.target.set_weights(self.critic.model.get_weights())

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

    def record(self, experience: tuple):
        priority = self.get_per_error(experience)
        self.buffer.record(experience, priority)

    def experience_replay(self):
        # sample from stored memory
        state_batch, action_batch, reward_batch, \
                next_state_batch, done_batch,\
                 sampled_idxs, is_weights = self.buffer.sample()

        # convert to tensors
        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)
        done_batch = tf.convert_to_tensor(done_batch, dtype=tf.float32)

        # train actor & critic networks
        actor_loss = self.actor.train(state_batch, self.critic)
        critic_loss = self.critic.train(state_batch, action_batch, reward_batch,
                            next_state_batch, done_batch, self.actor)

        # update the priorities in the replay buffer
        for i in range(len(sampled_idxs)):
            self.buffer.update(sampled_idxs[i], is_weights[i])

        if _DEBUG:
            with open('./root_value.txt', 'a+') as file:
                file.write('{}\t{:0.3f}\n'.format(self.buffer.available_samples,
                                                self.buffer.sum_tree.root_node.value))

        return actor_loss, critic_loss

    def get_per_error(self, experience: tuple):
        '''
        Computes the time-delay error which is used to determine the relative priority
        of the experience. This will be used for PER-based training
        delta[i] = r_t + gamma * QT(s', a_t) - Q(s, a)
        :param experience:
        :return: error
        '''

        state, action, reward, next_state, done = experience

        # convert the numpy array state into a tensor of size (1, w, h, c)
        tf_state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        tf_next_state = tf.expand_dims(tf.convert_to_tensor(next_state, dtype=tf.float32), 0)
        tf_action = tf.expand_dims(tf.convert_to_tensor(action, dtype=tf.float32), 0)
        tf_done = tf.convert_to_tensor(done, dtype=tf.float32)  # scalar

        # predict Q(s,a) for a given batch of states and action
        q_value = tf.squeeze(self.critic.model([tf_state, tf_action]))

        # target_action
        t_action = tf.squeeze(self.actor.target(tf_next_state))
        t_action = tf.expand_dims(t_action, 0)

        # target Qt(s',at)
        qt_next = tf.squeeze(self.critic.target([tf_next_state, t_action]))

        target_qt = reward + self.gamma * (1 - tf_done) * qt_next

        # time-delay error
        error = tf.squeeze(target_qt - q_value).numpy()
        error = DDPGPERAgent.huber_loss(error)

        if _DEBUG:
            with open('./td_error.txt', 'a+') as file:
                file.write('{:0.2f}\t{:0.2f}\t{:0.2f}\n'.format(error, target_qt.numpy(), qt_next.numpy()))
        return error

    @staticmethod
    def huber_loss(error):
        return 0.5 * error ** 2 if abs(error) < 1.0 else abs(error) - 0.5

    def update_targets(self):
        self.actor.update_target()
        self.critic.update_target()

    def save_model(self, path, actor_filename,
                   critic_filename, buffer_filename):
        actor_file = path + actor_filename
        critic_file = path + critic_filename
        buffer_file = path + buffer_filename

        self.actor.save_weights(actor_file)
        self.critic.save_weights(critic_file)
        self.buffer.save_data(buffer_file)

    def load_model(self, path, actor_filename,
                   critic_filename, buffer_filename):
        actor_file = path + actor_filename
        critic_file = path + critic_filename
        buffer_file = path + buffer_filename

        self.actor.model.load_weights(actor_file)
        self.actor.target.load_weights(actor_file)
        self.critic.model.load_weights(critic_file)
        self.critic.target.load_weights(critic_file)
        self.buffer.load_data(buffer_file)


