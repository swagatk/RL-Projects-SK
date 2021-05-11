"""
PPO Algorithm for Pendulum Gym Environment
Tensorflow 2.x compatible

- It seems to work properly with best episodic score reaching -200 within 1000 episodes or around 10-12 seasons
- Implements both 'KL-Penalty' method as well as 'PPO-Clip' method
- makes use of tensorflow probability
- The program terminates when the season score over 50 episodes > -200
"""
import pickle
import gym
import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from collections import deque
from tensorflow.keras import layers, Model
from tensorflow import keras
from scipy import signal

############################

print('TFP Version:', tfp.__version__)
print('Tensorflow version:', tf.__version__)
print('Keras Version:', tf.keras.__version__)

# set random seed for reproducibility
tf.random.set_seed(20)
np.random.seed(20)

############### hyper parameters

MAX_SEASONS = 5000      # total number of training seasons
TRAIN_EPISODES = 50     # total number of episodes in each season
TEST_EPISODES = 10      # total number of episodes for testing
TRAIN_EPOCHS = 20       # training epochs in each season
GAMMA = 0.9     # reward discount
LR_A = 0.0001    # learning rate for actor
LR_C = 0.0002    # learning rate for critic
BATCH_SIZE = 50     # minimum batch size for updating PPO
MAX_BUFFER_SIZE = 20000     # maximum buffer capacity > TRAIN_EPISODES * 200
METHOD = 'clip'          # 'clip' or 'penalty'

##################
KL_TARGET = 0.01
LAM = 0.5
EPSILON = 0.2


#####################
# ACTOR NETWORK
####################
class Actor:
    def __init__(self, state_size, action_size,
                 learning_rate, epsilon, lmbda, kl_target,
                 upper_bound, method='clip'):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.upper_bound = upper_bound
        self.epsilon = epsilon  # required for 'clip' method
        self.lam = lmbda  # required for 'penalty' method
        self.method = method
        self.kl_target = kl_target  # required for 'penalty' method
        self.kl_value = 0       # most recent kld value

        # create NN models
        self.model = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        # additions
        logstd = tf.Variable(np.zeros(shape=self.action_size, dtype=np.float32))
        self.model.logstd = logstd
        self.model.trainable_variables.append(logstd)

    def _build_net(self):
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        state_input = layers.Input(shape=self.state_size)
        l = layers.Dense(128, activation='relu')(state_input)
        l = layers.Dense(64, activation='relu')(l)
        l = layers.Dense(64, activation='relu')(l)
        net_out = layers.Dense(self.action_size[0], activation='tanh',
                               kernel_initializer=last_init)(l)
        net_out = net_out * self.upper_bound
        model = keras.Model(state_input, net_out)
        model.summary()
        return model

    def __call__(self, state):
        # input state is a tensor
        mean = tf.squeeze(self.model(state))
        std = tf.squeeze(tf.exp(self.model.logstd))
        return mean, std  # returns tensors

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def train(self, state_batch, action_batch, advantages, old_pi):

        with tf.GradientTape() as tape:
            mean = tf.squeeze(self.model(state_batch))
            std = tf.squeeze(tf.exp(self.model.logstd))
            pi = tfp.distributions.Normal(mean, std)
            ratio = tf.exp(pi.log_prob(tf.squeeze(action_batch)) -
                           old_pi.log_prob(tf.squeeze(action_batch)))
            surr = ratio * advantages  # surrogate function
            kl = tfp.distributions.kl_divergence(old_pi, pi)
            self.kl_value = tf.reduce_mean(kl)
            if self.method == 'penalty':  # ppo-penalty method
                actor_loss = -(tf.reduce_mean(surr - self.lam * kl))
                # # update the lambda value after each epoch
                # if kl_mean < self.kl_target / 1.5:
                #   self.lam /= 2
                # elif kl_mean > self.kl_target * 1.5:
                #   self.lam *= 2
            elif self.method == 'clip':  # ppo-clip method
                actor_loss = - tf.reduce_mean(
                    tf.minimum(surr, tf.clip_by_value(ratio,
                                                      1. - self.epsilon, 1. + self.epsilon) * advantages))
            actor_weights = self.model.trainable_variables

        # outside gradient tape
        actor_grad = tape.gradient(actor_loss, actor_weights)
        self.optimizer.apply_gradients(zip(actor_grad, actor_weights))

        return actor_loss.numpy(), self.kl_value.numpy()

    def update_lambda(self):
        # update the lambda value after each epoch
        if self.kl_value < self.kl_target / 1.5:
            self.lam /= 2
        elif self.kl_value > self.kl_target * 1.5:
            self.lam *= 2


####################################
# CRITIC NETWORK
################################
class Critic:
    def __init__(self, state_size, action_size,
                 learning_rate=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.train_step_count = 0
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.model = self._build_net()

    def _build_net(self):
        state_input = layers.Input(shape=self.state_size)
        out = layers.Dense(64, activation="relu")(state_input)
        out = layers.Dense(64, activation="relu")(out)
        out = layers.Dense(64, activation="relu")(out)
        net_out = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model(inputs=state_input, outputs=net_out)
        model.summary()
        return model

    def train(self, state_batch, disc_rewards):
        self.train_step_count += 1
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


######################
# BUFFER
######################
class Buffer:
    def __init__(self, buffer_capacity, batch_size):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=self.buffer_capacity)

    def __len__(self):
        return len(self.buffer)

    def record(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        valid_batch_size = min(len(self.buffer), self.batch_size)
        mini_batch = random.sample(self.buffer, valid_batch_size)

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        for i in range(valid_batch_size):
            state_batch.append(mini_batch[i][0])
            action_batch.append(mini_batch[i][1])
            reward_batch.append(mini_batch[i][2])
            next_state_batch.append(mini_batch[i][3])
            done_batch.append(mini_batch[i][4])

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def save_data(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.buffer, file)

    def load_data(self, filename):
        with open(filename, 'rb') as file:
            self.buffer = pickle.load(file)

    def get_samples(self, n_samples=None):

        if n_samples is None or n_samples > len(self.buffer):
            n_samples = len(self.buffer)

        s_batch = []
        a_batch = []
        r_batch = []
        ns_batch = []
        d_batch = []
        for i in range(n_samples):
            s_batch.append(self.buffer[i][0])
            a_batch.append(self.buffer[i][1])
            r_batch.append(self.buffer[i][2])
            ns_batch.append(self.buffer[i][3])
            d_batch.append(self.buffer[i][4])

        return s_batch, a_batch, r_batch, ns_batch, d_batch

    def clear(self):
        # empty the buffer
        self.buffer.clear()


#########################################
## PPO AGENT
########################################
class PPOAgent:
    def __init__(self, state_size, action_size, batch_size,
                 memory_capacity, upper_bound,
                 lr_a=1e-3, lr_c=1e-3,
                 gamma=0.99, lmbda=0.5, epsilon=0.2, kl_target=0.01,
                 method='clip'):
        self.state_size = state_size
        self.action_size = action_size
        self.actor_lr = lr_a
        self.critic_lr = lr_c
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.gamma = gamma  # discount factor
        self.upper_bound = upper_bound
        self.lmbda = lmbda  # required for GAE
        self.epsilon = epsilon  # required for PPO-CLIP
        self.kl_target = kl_target
        self.method = method
        self.best_ep_reward = -np.inf

        self.actor = Actor(self.state_size, self.action_size,
                           self.actor_lr, self.epsilon, self.lmbda,
                           self.kl_target, self.upper_bound, self.method)
        self.critic = Critic(self.state_size, self.action_size, self.critic_lr)
        self.buffer = Buffer(self.memory_capacity, self.batch_size)

    def policy(self, state, greedy=False):
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        mean, std = self.actor(tf_state)

        if greedy:
            action = mean
        else:
            pi = tfp.distributions.Normal(mean, std)
            action = pi.sample(sample_shape=self.action_size)
        valid_action = tf.clip_by_value(action, -self.upper_bound, self.upper_bound)
        return valid_action.numpy()

    def train(self, training_epochs=20, tmax=None):
        if tmax is not None and len(self.buffer) < tmax:
            return 0, 0, 0

        n_split = len(self.buffer) // self.batch_size
        n_samples = n_split * self.batch_size

        s_batch, a_batch, r_batch, ns_batch, d_batch = \
            self.buffer.get_samples(n_samples)

        s_batch = tf.convert_to_tensor(s_batch, dtype=tf.float32)
        a_batch = tf.convert_to_tensor(a_batch, dtype=tf.float32)
        r_batch = tf.convert_to_tensor(r_batch, dtype=tf.float32)
        ns_batch = tf.convert_to_tensor(ns_batch, dtype=tf.float32)
        d_batch = tf.convert_to_tensor(d_batch, dtype=tf.float32)

        disc_sum_reward = PPOAgent.discount(r_batch.numpy(), self.gamma)
        advantages = self.compute_advantages(r_batch, s_batch,
                                             ns_batch, d_batch)  # returns a numpy array
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        disc_sum_reward = tf.convert_to_tensor(disc_sum_reward, dtype=tf.float32)

        # current policy
        mean, std = self.actor(s_batch)
        pi = tfp.distributions.Normal(mean, std)

        s_split = tf.split(s_batch, n_split)
        a_split = tf.split(a_batch, n_split)
        dr_split = tf.split(disc_sum_reward, n_split)
        adv_split = tf.split(advantages, n_split)
        indexes = np.arange(n_split, dtype=int)

        a_loss_list = []
        c_loss_list = []
        kld_list = []
        np.random.shuffle(indexes)
        for _ in range(training_epochs):
            for i in indexes:
                old_pi = pi[i*self.batch_size: (i+1)*self.batch_size]

                # update actor
                a_loss, kld = self.actor.train(s_split[i], a_split[i], adv_split[i], old_pi)
                a_loss_list.append(a_loss)
                kld_list.append(kld)
                #a_loss.append(self.actor.train(s_split[i], a_split[i], adv_split[i], old_pi))

                # update critic
                c_loss_list.append(self.critic.train(s_split[i], dr_split[i]))

            # update lambda after each epoch
            if self.method == 'penalty':
                self.actor.update_lambda()

        actor_loss = np.mean(a_loss_list)
        critic_loss = np.mean(c_loss_list)
        mean_kld = np.mean(kld_list)

        # clear the buffer  -- this is important
        self.buffer.clear()

        return actor_loss, critic_loss, mean_kld

    @staticmethod
    def discount(x, gamma):
        return signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

    def compute_advantages(self, r_batch, s_batch, ns_batch, d_batch):
        s_values = tf.squeeze(self.critic.model(s_batch))
        ns_values = tf.squeeze(self.critic.model(ns_batch))

        tds = r_batch + self.gamma * ns_values * (1. - d_batch) - s_values
        adv = PPOAgent.discount(tds.numpy(), self.gamma * self.lmbda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)   #sometimes helpful
        return adv

    def save_model(self, path, actorfile, criticfile, bufferfile=None):
        actor_fname = path + actorfile
        critic_fname = path + criticfile

        self.actor.save_weights(actor_fname)
        self.critic.save_weights(critic_fname)

        if bufferfile is not None:
            buffer_fname = path + bufferfile
            self.buffer.save_data(buffer_fname)

    def load_model(self, path, actorfile, criticfile, bufferfile=None):

        actor_fname = path + actorfile
        critic_fname = path + criticfile

        self.actor.load_weights(actor_fname)
        self.critic.load_weights(critic_fname)

        if bufferfile is not None:
            buffer_fname = path + bufferfile
            self.buffer.load_data(buffer_fname)

        print('Model Parameters are loaded ...')


##################
def collect_trajectories(env, agent, max_episodes):
    ep_reward_list = []
    steps = 0
    for ep in range(max_episodes):
        state = env.reset()
        t = 0
        ep_reward = 0
        while True:
            action = agent.policy(state)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.record(state, action, reward, next_state, done)
            ep_reward += reward
            state = next_state
            t += 1
            if done:
                ep_reward_list.append(ep_reward)
                steps += t
                break

    mean_ep_reward = np.mean(ep_reward_list)
    return steps, mean_ep_reward

# This includes seasons for training
def main1(env, agent):

    path = './'
    if agent.method == 'clip':
        outfile = open(path + 'result_'+'clip_1'+'.txt', 'w')
    else:
        outfile = open(path + 'result_'+'klp_1'+'.txt', 'w')

    # training

    total_steps = 0
    best_score = -np.inf
    for s in range(MAX_SEASONS):
        t, s_reward = collect_trajectories(env, agent, TRAIN_EPISODES)

        a_loss, c_loss, kld_value = agent.train(training_epochs=TRAIN_EPOCHS)

        total_steps += t

        print('Season:{}, Episodes:{}, Training_steps:{}, mean_ep_reward:{:.2f}' \
              .format(s, (s + 1) * TRAIN_EPISODES, total_steps, s_reward))

        # if best_score < s_reward:
        #     best_score = s_reward
        #     agent.save_model(path, 'actor_weights.h5', 'critic_weights.h5')
        #     print('*** Season: {}, best score:{} Model Saved ***'.format(s, best_score))

        if agent.method == 'penalty':
            outfile.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(s, s_reward,
                                            a_loss, c_loss, kld_value, agent.actor.lam))
        else:
            outfile.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(s, s_reward,
                                                                a_loss, c_loss, kld_value))

        if s_reward > -200:
            print('Problem is solved in {} seasons involving {} steps'.format(s, total_steps))
            agent.save_model(path, 'actor_weights.h5', 'critic_weights.h5')
            break

    env.close()
    outfile.close()


# this is standard approach where the model goes through training over episodes
def main2(env, agent):

    path = './'
    if agent.method == 'clip':
        outfile = open(path + 'result_'+'clip_2'+'.txt', 'w')
    else:
        outfile = open(path + 'result_'+'klp_2'+'.txt', 'w')

    # training
    max_episodes = 10000
    total_steps = 0
    best_score = -np.inf
    ep_reward_list = deque(maxlen=40)
    for ep in range(max_episodes):
        state = env.reset()
        ep_reward = 0
        t = 0
        mean_a_loss = 0
        mean_c_loss = 0
        mean_kl_value = 0
        while True:
            action = agent.policy(state)
            next_state, reward, done, info = env.step(action)
            agent.buffer.record(state, action, reward, next_state, done)

            # train
            a_loss, c_loss, kld_value = agent.train(training_epochs=TRAIN_EPOCHS, tmax=10000)

            ep_reward += reward
            mean_a_loss += a_loss
            mean_c_loss += c_loss
            mean_kl_value += kld_value

            state = next_state
            t += 1

            if done:
                ep_reward_list.append(ep_reward)
                mean_a_loss /= t
                mean_c_loss /= t
                mean_kl_value /= t
                total_steps += t
                break

        if ep > 100 and ep % 20 == 0:
            test_score = validate(env, agent)
            if best_score < test_score:
                best_score = test_score
                agent.best_ep_reward = best_score
                agent.save_model(path, 'actor_weights.h5', 'critic_weights.h5')
                print('*** Episode: {}, validation_score: {}. Model saved. ***'.format(ep, best_score))

        # if ep % 100 == 0:
        #     print('Episode:{}, ep_reward:{:.2f}, avg_reward:{:.2f} \n'
        #           .format(ep, ep_reward, np.mean(ep_reward_list)))

        if agent.method == 'penalty':
            outfile.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(ep, ep_reward,
                                    np.mean(ep_reward_list), mean_a_loss, mean_c_loss,
                                                            mean_kl_value, agent.actor.lam))
        else:
            outfile.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(ep, ep_reward,
                                            np.mean(ep_reward_list), mean_a_loss,
                                                        mean_c_loss, mean_kl_value))

        if ep > 100 and best_score > -200:
            print('Problem is solved in {} seasons involving {} steps with avg reward {}'
                  .format(ep, total_steps, np.mean(ep_reward_list)))
            break

    env.close()
    outfile.close()

# test a model
def test(env, agent):
    path = './'
    agent.load_model(path, 'actor_weights.h5', 'critic_weights.h5')
    ep_reward_list = []
    for ep in range(10):
        state = env.reset()
        ep_reward = 0
        t = 0
        while True:
            env.render()
            action = agent.policy(state)
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            t += 1
            state = next_state
            if done:
                ep_reward_list.append(ep_reward)
                print('Episode: {}, Reward: {}'.format(ep, ep_reward))
                break

    print('Avg episodic reward: ', np.mean(ep_reward_list))
    env.close()

# used for validating
def validate(env, agent, ep_max=50):
    ep_reward_list = []
    for ep in range(ep_max):
        state = env.reset()
        ep_reward = 0
        t = 0
        while True:
            #env.render()
            action = agent.policy(state)
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            t += 1
            state = next_state
            if done:
                ep_reward_list.append(ep_reward)
                break

    return np.mean(ep_reward_list)



####################################
### MAIN FUNCTION
################################

if __name__ == '__main__':

    # Gym Environment
    env = gym.make('Pendulum-v0')
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape
    action_bound = env.action_space.high

    # create an agent
    agent = PPOAgent(state_dim, action_dim, BATCH_SIZE, MAX_BUFFER_SIZE,
                     action_bound,
                     LR_A, LR_C, GAMMA, LAM, EPSILON, KL_TARGET, METHOD)

    # training with seasons
    main1(env, agent)

    # training with episodes
    #main2(env, agent)

    # test
    # test(env, agent)

