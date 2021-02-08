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
                 learning_rate, epsilon, beta, ent_coeff, kl_target,
                 upper_bound, feature_model, method='clip'):
        self.state_size = state_size  # shape: (w, h, c)
        self.action_size = action_size  # shape: (n, )
        self.lr = learning_rate
        self.epsilon = epsilon          # required for PPO-clip
        self.upper_bound = upper_bound
        self.epsilon = epsilon          # required for clip method
        self.beta = beta        # required for KL-penalty method
        self.entropy_coeff = ent_coeff           # entropy coefficient
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

    def train(self, state_batch, action_batch, advantages, old_pi, c_loss):
        with tf.GradientTape() as tape:
            mean = tf.squeeze(self.model(state_batch))
            std = tf.squeeze(tf.exp(self.model.logstd))     # check the size of std here
            pi = tfp.distributions.Normal(mean, std)
            ratio = tf.exp(pi.log_prob(tf.squeeze(action_batch)) -
                           old_pi.log_prob(tf.squeeze(action_batch)))       # shape = (-1,3)
            adv_stack = tf.stack([advantages for i in range(self.action_size[0])], axis=1) # shape(-1,3)
            surr = ratio * adv_stack   # surrogate function
            kl = tfp.distributions.kl_divergence(old_pi, pi)    # kl divergence
            entropy = tf.reduce_mean(pi.entropy())      # entropy
            self.kl_value = tf.reduce_mean(kl)
            if self.method == 'penalty':    # KL-penalty method
                actor_loss = -(tf.reduce_mean(surr - self.beta * kl))   # beta
                self.update_beta()
            elif self.method == 'clip':
                l_clip = tf.reduce_mean(
                    tf.minimum(surr, tf.clip_by_value(ratio, 1. - self.epsilon,
                                                      1. + self.epsilon) * adv_stack))
                actor_loss = - (l_clip - c_loss + self.entropy_coeff * entropy)
            actor_weights = self.model.trainable_variables
        actor_grad = tape.gradient(actor_loss, actor_weights)
        self.optimizer.apply_gradients(zip(actor_grad, actor_weights))
        return actor_loss.numpy()

    def update_beta(self):
        if self.kl_value < self.kl_target / 1.5:
            self.beta /= 2
        elif self.kl_value > self.kl_target * 1.5:
            self.beta *= 2


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

    def __call__(self, state):  # state is a numpy array
        tf_state = tf.convert_to_tensor(state, dtype=tf.float32)
        value = tf.squeeze(self.model(tf_state))
        return value

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
    def __init__(self, state_size, action_size,
                 upper_bound,
                 lr_a=1e-3, lr_c=1e-3, gamma=0.99,  lmbda=0.9, beta=0.5, ent_coeff=0.01,
                 epsilon=0.07, kl_target=0.01, method='clip'):
        self.state_size = state_size
        self.action_size = action_size
        self.actor_lr = lr_a
        self.critic_lr = lr_c
        self.gamma = gamma  # discount factor
        self.lam = lmbda  # required for Generalized Advantage Estimator (GAE)
        self.beta = beta    # required for KL-Penalty method
        self.entropy_coeff = ent_coeff      # entropy coeff
        self.epsilon = epsilon  # clip_factor
        self.upper_bound = upper_bound      # action upper bound
        self.kl_target = kl_target
        self.method = method

        self.feature = FeatureNetwork(self.state_size)
        self.actor = PPOActor(self.state_size, self.action_size, self.actor_lr,
                              self.epsilon, self.beta, self.entropy_coeff, self.kl_target, self.upper_bound,
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

    def train(self, states, actions, rewards, dones):
        # note that state has one extra row
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        disc_cum_rewards, advantages = self.compute_advantages(states, rewards, dones)   # outputs are tensors

        # current policy
        mean, std = self.actor(states[:-1])
        old_pi = tfp.distributions.Normal(mean, std)

        c_loss = self.critic.train(states[:-1], disc_cum_rewards)
        a_loss = self.actor.train(states[:-1], actions, advantages, old_pi, c_loss)

        # update lambda once in each epoch
        # if self.method == 'penalty':
        #     self.actor.update_beta()

        return a_loss, c_loss

    def compute_advantages(self, s_batch, r_batch, d_batch):

        # make sure that len(s_batch) = len(r_batch) + 1
        # all inputs are tensors
        values = self.critic(s_batch)
        g = 0
        returns = []
        for i in reversed(range(len(r_batch))):
            delta = r_batch[i] + self.gamma * values[i + 1] * d_batch[i] - values[i]
            g = delta + self.gamma * self.lam * d_batch[i] * g
            returns.append(g + values[i])

        returns.reverse()  # check the type of returns - is it a tensor?
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        adv = returns - values[:-1]  # omits the last item
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        return returns, adv

    def save_model(self, path, actor_filename,
                   critic_filename):
        actor_file = path + actor_filename
        critic_file = path + critic_filename
        self.actor.save_weights(actor_file)
        self.critic.save_weights(critic_file)

    def load_model(self, path, actor_filename,
                   critic_filename):
        actor_file = path + actor_filename
        critic_file = path + critic_filename
        self.actor.model.load_weights(actor_file)
        self.critic.model.load_weights(critic_file)

