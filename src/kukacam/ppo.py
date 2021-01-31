"""
Implementing Proximal Policy Optimization (PPO) for Kuka Environment

PPO_CLIP Algorithm
"""
import tensorflow as tf
import numpy as np
from FeatureNet import FeatureNetwork
from buffer import KukaBuffer
import tensorflow_probability as tfp
from scipy import signal

###########################
## TENSORFLOW Related Logistics
################################
# check tensorflow version
from packaging import version

#######################
print("Tensorflow Version: ", tf.__version__)
print("Keras Version: ", tf.keras.__version__)
print('Tensorflow Probability Version: ', tfp.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This program requires Tensorflow 2.0 or above"

# avoid CUDNN_STATUS_INTERNAL_ERROR
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
####################################

################
## ACTOR NETWORK
################
class PPOActor:
    def __init__(self, state_size, action_size,
                 learning_rate, epsilon, lmbda, kl_target,
                 upper_bound, feature_model, method='clip'):
        self.state_size = state_size  # shape: (w, h, c)
        self.action_size = action_size  # shape: (n, )
        self.lr = learning_rate
        self.epsilon = epsilon          # required for PPO-clip
        self.upper_bound = upper_bound
        self.epsilon = epsilon          # required for clip method
        self.lam = lmbda        # required for KL-penalty method
        self.kl_target = kl_target
        self.kl_value = 0           # most recent kl_divergence
        self.method = method        # 'clip' or 'penalty'

        # create NN models
        self.feature_model = feature_model
        self.model = self._build_net(trainable=True)
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        # additions
        logstd = tf.Variable(np.zeros(shape=self.action_size, dtype=np.float32))
        self.model.logstd = logstd
        self.model.trainable_variables.append(logstd)

    def _build_net(self, trainable=True):
        # input is a stack of 1-channel YUV images
        last_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03)
        state_input = tf.keras.layers.Input(shape=self.state_size)
        f = self.feature_model(state_input)
        f = tf.keras.layers.Dense(128, activation='relu', trainable=trainable)(f)
        f = tf.keras.layers.Dense(64, activation="relu", trainable=trainable)(f)
        net_out = tf.keras.layers.Dense(self.action_size[0], activation='tanh',
                                        kernel_initializer=last_init, trainable=trainable)(f)
        net_out = net_out * self.upper_bound  # element-wise product
        model = tf.keras.Model(state_input, net_out, name='actor')
        model.summary()
        tf.keras.utils.plot_model(model, to_file='actor_net.png',
                                  show_shapes=True, show_layer_names=True)
        return model

    def __call__(self, state):
        # input is a tensor
        mean = tf.squeeze(self.model(state))
        std = tf.squeeze(tf.exp(self.model.logstd))
        return mean, std

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def train(self, state_batch, action_batch, advantages, old_pi):
        with tf.GradientTape() as tape:
            mean = tf.squeeze(self.model(state_batch))
            std = tf.squeeze(tf.exp(self.model.logstd))     # check the size of std here
            pi = tfp.distributions.Normal(mean, std)
            ratio = tf.exp(pi.log_prob(tf.squeeze(action_batch)) -
                           old_pi.log_prob(tf.squeeze(action_batch)))       # shape = (50,3)
            adv_stack = tf.stack([advantages, advantages, advantages], axis=1)  # shape = (50,3)
            surr = ratio * adv_stack   # surrogate function
            kl = tfp.distributions.kl_divergence(old_pi, pi)
            self.kl_value = tf.reduce_mean(kl)
            if self.method == 'penalty':    # KL-penalty method
                actor_loss = -(tf.reduce_mean(surr - self.lam * kl))
                # if self.kl_value < self.kl_target / 1.5:
                #     self.lam /= 2
                # elif self.kl_value > self.kl_target * 1.5:
                #     self.lam *= 2
            elif self.method == 'clip':
                actor_loss = -tf.reduce_mean(
                    tf.minimum(surr, tf.clip_by_value(ratio, 1. - self.epsilon,
                                                      1. + self.epsilon) * adv_stack))
            actor_weights = self.model.trainable_variables
        actor_grad = tape.gradient(actor_loss, actor_weights)
        self.optimizer.apply_gradients(zip(actor_grad, actor_weights))
        return actor_loss.numpy()

    def update_lambda(self):
        if self.kl_value < self.kl_target / 1.5:
            self.lam /= 2
        elif self.kl_value > self.kl_target * 1.5:
            self.lam *= 2


####################
# CRITIC NETWORK
##################
class PPOCritic:
    def __init__(self, state_size, action_size,
                 learning_rate, feature_model):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.feature_model = feature_model
        self.model = self._build_net(trainable=True)

    def _build_net(self, trainable=True):
        # state input is a stack of 1-D YUV images
        state_input = tf.keras.layers.Input(shape=self.state_size)
        feature = self.feature_model(state_input)
        out = tf.keras.layers.Dense(128, activation="relu", trainable=trainable)(feature)
        out = tf.keras.layers.Dense(64, activation="relu", trainable=trainable)(out)
        out = tf.keras.layers.Dense(32, activation="relu", trainable=trainable)(out)
        net_out = tf.keras.layers.Dense(1, trainable=trainable)(out)

        # Outputs single value for a given state = V(s)
        model = tf.keras.Model(inputs=state_input, outputs=net_out, name='critic')
        model.summary()
        tf.keras.utils.plot_model(model, to_file='critic_net.png',
                                  show_shapes=True, show_layer_names=True)
        return model

    def train(self, state_batch, disc_rewards):
        with tf.GradientTape() as tape:
            critic_weights = self.model.trainable_variables
            critic_value = tf.squeeze(self.model(state_batch))
            critic_loss = tf.math.reduce_mean(tf.square(disc_rewards - critic_value))
        critic_grad = tape.gradient(critic_loss, critic_weights)
        self.optimizer.apply_gradients(zip(critic_grad, critic_weights))
        return critic_loss.numpy()

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


########################
## PPO AGENT
#########################
class KukaPPOAgent:
    def __init__(self, state_size, action_size, batch_size,
                 memory_capacity, upper_bound,
                 lr_a=1e-3, lr_c=1e-3, gamma=0.99,  lmbda=0.5,
                 epsilon=0.2, kl_target=0.01, method='clip'):
        self.state_size = state_size
        self.action_size = action_size
        self.actor_lr = lr_a
        self.critic_lr = lr_c
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.gamma = gamma  # discount factor
        self.lam = lmbda  # required for estimating advantage (GAE)
        self.epsilon = epsilon  # clip_factor
        self.upper_bound = upper_bound      # action upper bound
        self.kl_target = kl_target
        self.method = method

        self.feature = FeatureNetwork(self.state_size)
        self.buffer = KukaBuffer(self.memory_capacity, self.batch_size)
        self.actor = PPOActor(self.state_size, self.action_size, self.actor_lr,
                              self.epsilon, self.lam, self.kl_target, self.upper_bound,
                              self.feature, self.method)

        # critic estimates the advantage
        self.critic = PPOCritic(self.state_size, self.action_size,
                                self.critic_lr, self.feature)

    def policy(self, state, greedy=False):
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        mean, std = self.actor(tf_state)

        if greedy:
            action = mean
        else:
            pi = tfp.distributions.Normal(mean, std)
            action = pi.sample()
        valid_action = tf.clip_by_value(action, -self.upper_bound, self.upper_bound)
        return valid_action.numpy()

    def record(self, experience: tuple):
        self.buffer.record(experience)

    def train(self, training_epochs=20):
        n_split = len(self.buffer) // self.batch_size
        n_samples = n_split * self.batch_size

        s_batch, a_batch, r_batch, ns_batch, d_batch = \
            self.buffer.get_samples(n_samples)

        s_batch = tf.convert_to_tensor(s_batch, dtype=tf.float32)
        a_batch = tf.convert_to_tensor(a_batch, dtype=tf.float32)
        r_batch = tf.convert_to_tensor(r_batch, dtype=tf.float32)
        ns_batch = tf.convert_to_tensor(ns_batch, dtype=tf.float32)
        d_batch = tf.convert_to_tensor(d_batch, dtype=tf.float32)

        disc_cum_rewards = KukaPPOAgent.discount(r_batch.numpy(), self.gamma)
        advantages = self.compute_advantages(s_batch, ns_batch, r_batch, d_batch)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        disc_cum_rewards = tf.convert_to_tensor(disc_cum_rewards, dtype=tf.float32)

        # current policy
        mean, std = self.actor(s_batch)
        pi = tfp.distributions.Normal(mean, std)

        # create splits
        s_split = tf.split(s_batch, n_split)
        a_split = tf.split(a_batch, n_split)
        dr_split = tf.split(disc_cum_rewards, n_split)
        adv_split = tf.split(advantages, n_split)
        indexes = np.arange(n_split, dtype=int)

        # training
        a_loss_list = []
        c_loss_list = []
        kl_list = []
        np.random.shuffle(indexes)
        for _ in range(training_epochs):
            for i in indexes:
                old_pi = pi[i * self.batch_size: (i+1) * self.batch_size]

                # update actor
                a_loss_list.append(self.actor.train(s_split[i], a_split[i],
                                                    adv_split[i], old_pi))
                kl_list.append(self.actor.kl_value)

                # update critic
                c_loss_list.append(self.critic.train(s_split[i], dr_split[i]))

            # update lambda once in each epoch
            #self.actor.update_lambda()

        actor_loss = np.mean(a_loss_list)
        critic_loss = np.mean(c_loss_list)
        kld_mean = np.mean(kl_list)

        # clear the buffer
        self.buffer.clear()
        return actor_loss, critic_loss, kld_mean

    @staticmethod
    def discount(x, gamma):
        return signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

    # Generalized Advantage Estimate (GAE)
    def compute_advantages(self, s_batch, ns_batch, r_batch, d_batch):
        s_values = tf.squeeze(self.critic.model(s_batch))
        ns_values = tf.squeeze(self.critic.model(ns_batch))

        # time-delay error
        tds = r_batch + self.gamma * ns_values * (1. - d_batch) - s_values
        adv = KukaPPOAgent.discount(tds.numpy(), self.gamma * self.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)       # sometimes helpful
        return adv

    def save_model(self, path, actor_filename,
                   critic_filename, buffer_filename=None):
        actor_file = path + actor_filename
        critic_file = path + critic_filename

        self.actor.save_weights(actor_file)
        self.critic.save_weights(critic_file)

        if buffer_filename is not None:
            buffer_file = path + buffer_filename
            self.buffer.save_data(buffer_file)

    def load_model(self, path, actor_filename,
                   critic_filename, buffer_filename=None):
        actor_file = path + actor_filename
        critic_file = path + critic_filename

        self.actor.model.load_weights(actor_file)
        self.critic.model.load_weights(critic_file)

        if buffer_filename is not None:
            buffer_file = path + buffer_filename
            self.buffer.load_data(buffer_file)
