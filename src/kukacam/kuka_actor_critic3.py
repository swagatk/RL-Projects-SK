"""
Implementing DDPG for Kuka Diverse Object Environment

- Here the Actor and Critic  share common Convolution Feature Network Layer to extract features from RGB images
- Both Actor and Critic update the Feature Network parameters during training. Weights of feature network are
updated twice in each iteration.
- We use Priority Experience Replay where each experience is selected based on some priority which is updated during
training.
- Status: No success. The average reward for last 100 episodes after 4K episodes is about 0.2.
- Todo: Display the Conv Layer output on Tensorboard.
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from per_memory_buffer import Memory
import pickle

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


###########################
# feature network
########################
class FeatureNetwork:
    def __init__(self, state_size, learning_rate=1e-3):
        self.state_size = state_size
        self.lr = learning_rate
        # create NN models
        self.model = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        img_input = layers.Input(shape=self.state_size)

        # shared convolutional layers
        conv1 = layers.Conv2D(16, kernel_size=5, strides=2,
                              padding="SAME", activation="relu")(img_input)
        bn1 = layers.BatchNormalization()(conv1)
        conv2 = layers.Conv2D(32, kernel_size=5, strides=2,
                              padding="SAME", activation="relu")(bn1)
        bn2 = layers.BatchNormalization()(conv2)
        conv3 = layers.Conv2D(32, kernel_size=5, strides=2,
                              padding="SAME", activation="relu")(bn2)
        bn3 = layers.BatchNormalization()(conv3)
        f = layers.Flatten()(bn3)
        f = layers.Dense(128, activation="relu")(f)
        f = layers.Dense(128, activation="relu")(f)
        f = layers.Dense(64, activation="relu")(f)
        model = tf.keras.Model(inputs=img_input, outputs=f)
        print('shared feature network')
        model.summary()
        keras.utils.plot_model(model, to_file='feature_net.png',
                               show_shapes=True, show_layer_names=True)
        return model

    def __call__(self, state):
        return self.model(state)


############################################
# ACTOR
##############################
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

####################################
# ACTOR-CRITIC AGENT
##################################
class KukaActorCriticAgent:
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
        self.buffer = Memory(self.memory_capacity, self.batch_size)

        std_dev = 0.2
        self.noise_object = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

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
        #sampled_action = sampled_action.numpy() + noise_vec
        sampled_action = sampled_action.numpy()

        # Make sure that the action is within bounds
        valid_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        return np.squeeze(valid_action)

    def experience_replay(self):
        # sample from stored memory
        state_batch, action_batch, reward_batch, \
                next_state_batch, sampled_idxs, is_weights = self.buffer.sample()

        # convert to tensors
        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)

        # train actor & critic networks
        actor_loss = self.actor.train(state_batch, self.critic)
        critic_loss = self.critic.train(state_batch, action_batch, reward_batch,
                                        next_state_batch, self.actor)

        # update the priorities in the replay buffer
        for i in range(len(sampled_idxs)):
            self.buffer.update(sampled_idxs[i], is_weights[i])

        return actor_loss, critic_loss

    def get_per_error(self, experience: tuple):
        '''
        Computes the time-delay error which is used to determine the relative priority
        of the experience. This will be used for PER-based training
        delta[i] = r_t + gamma * QT(s', a) - Q(s, a)
        :param experience:
        :return: error
        '''

        state, action, reward, next_state = experience

        # convert the numpy array state into a tensor of size (1, w, h, c)
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        tf_next_state = tf.expand_dims(tf.convert_to_tensor(next_state), 0)
        tf_action = tf.expand_dims(tf.convert_to_tensor(action), 0)

        # predict Q(s,a) for a given batch of states and action
        prim_qt = tf.squeeze(self.critic.model([tf_state, tf_action]))

        # target Qtarget(s',a)
        target_qtp1 = tf.squeeze(self.critic.target([tf_next_state, tf_action]))

        target_qt = reward + self.gamma * target_qtp1

        error = target_qt - prim_qt
        return self.huber_loss(error.numpy())

    @staticmethod
    def huber_loss(error):
        return 0.5 * error ** 2 if abs(error) < 1.0 else abs(error) - 0.5

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
        with open(replay_file, 'wb') as file:
            pickle.dump(self.buffer, file)

    def load_model(self, path, actor_filename, critic_filename,
                   replay_filename):
        actor_file = path + actor_filename
        critic_file = path + critic_filename
        replay_file = path + replay_filename

        self.actor.model.load_weights(actor_file)
        self.actor.target.load_weights(actor_file)
        self.critic.model.load_weights(critic_file)
        self.critic.target.load_weights(critic_file)

        with open(replay_file, 'rb') as file:
            self.buffer = pickle.load(file)

