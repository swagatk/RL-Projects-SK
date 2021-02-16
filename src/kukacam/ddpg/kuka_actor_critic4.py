"""
Combine Twin Delayed DDPG with PER

"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pickle
from FeatureNet import FeatureNetwork
from OUActionNoise import OUActionNoise
from per_memory_buffer import Memory
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


############################################
# ACTOR
##############################
class KukaTD3Actor:
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

    def _build_net(self):
        # input is a stack of 1-channel YUV images
        last_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03)

        state_input = layers.Input(shape=self.state_size)
        feature = self.feature_model(state_input)

        l2 = layers.Dense(64, activation="relu")(feature)
        net_out = layers.Dense(self.action_size[0], activation='tanh',
                               kernel_initializer=last_init)(l2)

        net_out = net_out * self.upper_bound  # element-wise product
        model = keras.Model(state_input, net_out)
        model.summary()
        keras.utils.plot_model(model, to_file='actor_net.png',
                               show_shapes=True, show_layer_names=True)
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
class KukaTD3Critic:
    def __init__(self, state_size, action_size,
                 replacement,
                 learning_rate,
                 gamma, feature_model):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.replacement = replacement
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.feature_model = feature_model
        self.model = self._build_net()
        self.target = self._build_net()
        self.gamma = gamma
        self.train_step_count = 0

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
        net_out = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model(inputs=[state_input, action_input], outputs=net_out)
        model.summary()
        keras.utils.plot_model(model, to_file='critic_net.png',
                               show_shapes=True, show_layer_names=True)
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

    def train(self, state_batch, action_batch, y):
        self.train_step_count += 1
        with tf.GradientTape() as tape:
            critic_weights = self.model.trainable_variables
            critic_value = self.model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_weights)
        self.optimizer.apply_gradients(zip(critic_grad, critic_weights))
        return critic_loss

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


####################################
# ACTOR-CRITIC AGENT
##################################
class KukaTD3Agent:
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
        self.policy_decay = 2   # new
        self.train_step = 0  # new
        self.actor_loss = 0  # new

        # shared feature model
        self.feature = FeatureNetwork(self.state_size, self.actor_lr)

        # Actor model
        self.actor = KukaTD3Actor(self.state_size, self.action_size, self.replacement,
                               self.actor_lr, self.upper_bound, self.feature)

        # Two critic models
        self.critic_one = KukaTD3Critic(self.state_size, self.action_size, self.replacement,
                                 self.critic_lr, self.gamma, self.feature)

        self.critic_two = KukaTD3Critic(self.state_size, self.action_size, self.replacement,
                                    self.critic_lr, self.gamma, self.feature)

        self.buffer = Memory(self.memory_capacity, self.batch_size,
                             self.state_size, self.action_size)

        std_dev = 0.2
        self.noise = OUActionNoise(mean=np.zeros(1),
                                          std_deviation=float(std_dev) * np.ones(1))

        # Initially make weights for target and model equal
        self.actor.target.set_weights(self.actor.model.get_weights())
        self.critic_one.target.set_weights(self.critic_one.model.get_weights())
        self.critic_two.target.set_weights(self.critic_two.model.get_weights())

    def policy(self, state):
        # Check the size of state: (w, h, c)

        # convert the numpy array state into a tensor of size (1, w, h, c)
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

        # get action
        sampled_action = tf.squeeze(self.actor.model(tf_state))
        noise = self.noise()  # scalar value

        # convert into the same shape as that of the action vector
        noise_vec = noise * np.ones(self.action_size)

        # Add noise to the action
        sampled_action = sampled_action.numpy() + noise_vec

        # Make sure that the action is within bounds
        valid_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        return np.squeeze(valid_action)

    def experience_replay(self):
        # implement priority experience replay

        # sample data from the replay buffer
        state_batch, action_batch, reward_batch, \
        next_state_batch, \
        done_batch, sampled_idxs, is_weights = self.buffer.sample()

        # convert to tensors
        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)
        done_batch = tf.convert_to_tensor(done_batch, dtype=tf.float32)

        # train the critic
        y = self.compute_critic_target(reward_batch, next_state_batch, done_batch)
        critic_one_loss = self.critic_one.train(state_batch, action_batch, y)
        critic_two_loss = self.critic_two.train(state_batch, action_batch, y)

        # update the Actor less frequently
        actor_loss = self.actor_loss
        if self.train_step % self.policy_decay == 0:
            actor_loss = self.actor.train(state_batch, self.critic_one)
            self.actor_loss = actor_loss
        critic_loss = np.minimum(critic_one_loss, critic_two_loss)

        # update the priorities in the replay buffer
        for i in range(len(sampled_idxs)):
            self.buffer.update(sampled_idxs[i], is_weights[i])

        with open('./root_value.txt', 'a+') as file:
            file.write('{}\t{:0.3f}\n'.format(self.buffer.available_samples,
                                              self.buffer.sum_tree.root_node.value))

        self.train_step += 1
        return actor_loss, critic_loss

    def get_per_error(self, experience: tuple):
        '''
        Computes the time-delay error which is used to determine the relative priority
        of the experience. This will be used for PER-based training
        delta[i] = r_t + gamma * QT(s', a) - Q(s, a)
        :param experience:
        :return: error
        '''

        state, action, reward, next_state, done = experience

        # convert the numpy array state into a tensor of size (1, w, h, c)
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        tf_next_state = tf.expand_dims(tf.convert_to_tensor(next_state), 0)
        tf_action = tf.expand_dims(tf.convert_to_tensor(action), 0)

        # predict Q(s,a) for a given batch of states and action
        prim_qt = tf.squeeze(self.critic_one.model([tf_state, tf_action]))

        # target Qtarget(s',a)
        target_qtp1 = tf.squeeze(self.critic_one.target([tf_next_state, tf_action]))

        target_qt = reward + self.gamma * (1 - done) * target_qtp1

        # time-delay error
        error = tf.squeeze(target_qt - prim_qt).numpy()
        error = KukaTD3Agent.huber_loss(error)

        with open('./td_error.txt', 'a+') as file:
            file.write('{:0.2f}\t{:0.2f}\t{:0.2f}\n'.format(error, target_qt.numpy(), target_qtp1.numpy()))
        return error

    @staticmethod
    def huber_loss(error):
        return 0.5 * error ** 2 if abs(error) < 1.0 else abs(error) - 0.5

    def compute_critic_target(self, r_batch, ns_batch, d_batch):
        # target smoothing
        target_actions = self.compute_target_actions(ns_batch)
        target_critic_one = self.critic_one.target([ns_batch, target_actions])
        target_critic_two = self.critic_two.target([ns_batch, target_actions])
        target_critic = np.minimum(target_critic_one, target_critic_two)
        y = r_batch + self.gamma * (1 - d_batch) * target_critic
        return y

    def compute_target_actions(self, ns_batch):
        target_actions = self.actor.target(ns_batch)

        # create a noise vector
        noise_vec = self.noise() * np.ones(self.action_size)

        # add noise to action
        target_actions = target_actions.numpy() + noise_vec

        clipped_target_actions = np.clip(target_actions, self.lower_bound, self.upper_bound)
        return clipped_target_actions

    def update_targets(self):
        self.actor.update_target()
        self.critic_one.update_target()
        self.critic_two.update_target()

    def save_model(self, path, actor_filename, critic_filename,
                   replay_filename):
        actor_file = path + actor_filename
        critic_one_file = path + 'one_' + critic_filename
        critic_two_file = path + 'two_' + critic_filename
        replay_file = path + replay_filename

        self.actor.save_weights(actor_file)
        self.critic_one.save_weights(critic_one_file)
        self.critic_two.save_weights(critic_two_file)

        with open(replay_file, 'wb') as file:
            pickle.dump(self.buffer, file)

    def load_model(self, path, actor_filename, critic_filename,
                   replay_filename):
        actor_file = path + actor_filename
        critic_one_file = path + 'one_' + critic_filename
        critic_two_file = path + 'two_' + critic_filename
        replay_file = path + replay_filename

        self.actor.model.load_weights(actor_file)
        self.actor.target.load_weights(actor_file)
        self.critic_one.model.load_weights(critic_one_file)
        self.critic_one.target.load_weights(critic_one_file)
        self.critic_two.model.load_weights(critic_two_file)
        self.critic_two.target.load_weights(critic_two_file)

        with open(replay_file, 'rb') as file:
            self.buffer = pickle.load(file)

