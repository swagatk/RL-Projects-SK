'''
PPO Algorithm for Pendulum-v0 environment
Tensorflow 2.0

Changes compared to 'ppo1.py'
- We use a different version for compute_advantage function that also computes the discounted returns
- It does not require the BUFFER class. Experiences are stored in a list which is discarded after each iteration
- The actor loss function also include critic loss term as well as entropy term

Effect:
- The problem get solved within 15 seasons. The problem is considered solved when season score > -200.
- including critic_loss and entropy improves the convergence speed. The problem gets solved within 13 seasons.
'''
import random
import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from collections import deque
from tensorflow.keras import layers, Model
from tensorflow import keras
import os
############################

print('TFP Version:', tfp.__version__)
print('Tensorflow version:', tf.__version__)
print('Keras Version:', tf.keras.__version__)

# set random seed for reproducibility
random.seed(20)
tf.random.set_seed(20)
np.random.seed(20)


#####################
# ACTOR NETWORK
####################
class Actor:
    def __init__(self, state_size, action_size,
                 learning_rate, epsilon, beta, c_loss_coeff, ent_coeff, kl_target,
                 upper_bound, method='clip'):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.upper_bound = upper_bound
        self.epsilon = epsilon  # required for 'clip' method
        self.beta = beta  # required for 'penalty' method
        self.entropy_coeff = ent_coeff      # weighting factor for entropy
        self.c_loss_coeff = c_loss_coeff    # weighting factor critic loss
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

    def train(self, state_batch, action_batch, advantages, old_pi, c_loss):
        with tf.GradientTape() as tape:
            mean = tf.squeeze(self.model(state_batch))
            std = tf.squeeze(tf.exp(self.model.logstd))     # check the size of std here
            pi = tfp.distributions.Normal(mean, std)
            ratio = tf.exp(pi.log_prob(tf.squeeze(action_batch)) -
                           old_pi.log_prob(tf.squeeze(action_batch)))       # shape = (-1,3)
            surr = ratio * advantages # surrogate function
            kl = tfp.distributions.kl_divergence(old_pi, pi)    # kl divergence
            entropy = tf.reduce_mean(pi.entropy())      # entropy
            self.kl_value = tf.reduce_mean(kl)
            if self.method == 'penalty':    # KL-penalty method
                actor_loss = -(tf.reduce_mean(surr - self.beta * kl))   # beta
                # self.update_beta()
            elif self.method == 'clip':
                l_clip = tf.reduce_mean(
                    tf.minimum(surr, tf.clip_by_value(ratio, 1. - self.epsilon,
                                                      1. + self.epsilon) * advantages))
                actor_loss = -(l_clip - self.c_loss_coeff * c_loss +
                                                self.entropy_coeff * entropy)
            actor_weights = self.model.trainable_variables
        actor_grad = tape.gradient(actor_loss, actor_weights)
        self.optimizer.apply_gradients(zip(actor_grad, actor_weights))
        return actor_loss.numpy()

    def update_beta(self):
        # update the lambda value after each epoch
        if self.kl_value < self.kl_target / 1.5:
            self.beta /= 2
        elif self.kl_value > self.kl_target * 1.5:
            self.beta *= 2

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

    def __call__(self, state):
        # input is a tensor
        value = tf.squeeze(self.model(state))
        return value

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


#########################################
## PPO AGENT
########################################
class PPOAgent:
    def __init__(self, state_size, action_size,
             batch_size, upper_bound,
             lr_a=1e-4,             # Actor learning rate
             lr_c=2e-4,             # Critic Learning Rate
             gamma=0.9,             # Discount factor
             lmbda=0.95,            # Required for GAE
             beta=0.5,              # Required for KL-Penalty Method
             ent_coeff=0.01,        # Weighting factor for Entropy term
             c_loss_coeff=0.5,      # Weighting factor Critic Loss component
             epsilon=0.2,           # Required for PPO-CLIP
             kl_target=0.01,        # Required for KL-Penalty method
             method='penalty'):
        self.state_size = state_size
        self.action_size = action_size
        self.actor_lr = lr_a
        self.critic_lr = lr_c
        self.batch_size = batch_size
        self.gamma = gamma  # discount factor
        self.upper_bound = upper_bound
        self.beta = beta        # required for KL-Penalty method
        self.entropy_coeff = ent_coeff      # Weightage for Entropy component
        self.c_loss_coeff = c_loss_coeff    # Weightage for loss component
        self.lmbda = lmbda  # required for GAE
        self.epsilon = epsilon  # required for PPO-CLIP
        self.kl_target = kl_target      # Required for KL-Penalty method
        self.method = method

        self.actor = Actor(self.state_size, self.action_size,
                           self.actor_lr, self.epsilon, self.beta, self.c_loss_coeff,
                           self.entropy_coeff, self.kl_target, self.upper_bound, self.method)
        self.critic = Critic(self.state_size, self.action_size, self.critic_lr)

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

    def train(self, states, actions, rewards, next_states, dones, epochs=20):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # compute advantages and discounted cumulative rewards
        target_values, advantages = self.compute_advantages(states, next_states, rewards, dones)
        target_values = tf.convert_to_tensor(target_values, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

        # current action probability distribution
        mean, std = self.actor(states)
        pi = tfp.distributions.Normal(mean, std)

        n_split = len(rewards) // self.batch_size
        assert n_split > 0, 'there should be at least one split'

        indexes = np.arange(n_split, dtype=int)
        np.random.shuffle(indexes)

        # training
        a_loss_list = []
        c_loss_list = []
        kl_list = []
        for _ in range(epochs):
            for i in indexes:
                old_pi = pi[i * self.batch_size: (i + 1) * self.batch_size]

                s_split = tf.gather(states, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)
                a_split = tf.gather(actions, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)
                tv_split = tf.gather(target_values, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)
                adv_split = tf.gather(advantages, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)

                # update critic
                cl = self.critic.train(s_split, tv_split)
                c_loss_list.append(cl)

                # update actor
                a_loss_list.append(self.actor.train(s_split, a_split,
                                                    adv_split, old_pi, cl))
                kl_list.append(self.actor.kl_value)

            # update lambda once in each epoch
            if self.method == 'penalty':
                self.actor.update_beta()

        actor_loss = np.mean(a_loss_list)
        critic_loss = np.mean(c_loss_list)
        kld_mean = np.mean(kl_list)

        return actor_loss, critic_loss, kld_mean

    # Generalized Advantage Estimate (GAE)
    def compute_advantages(self, states, next_states, rewards, dones):
        # inputs are tensors
        # outputs are tensors
        s_values = self.critic(states)
        ns_values = self.critic(next_states)

        adv = np.zeros(shape=(len(rewards), ))
        returns = np.zeros(shape=(len(rewards), ))

        discount = self.gamma
        lmbda = self.lmbda
        g = 0
        returns_current = ns_values[-1]
        for i in reversed(range(len(rewards))):
            gamma = discount * (1. - dones[i])
            td_error = rewards[i] + gamma * ns_values[i] - s_values[i]
            g = td_error + gamma * lmbda * g
            returns_current = rewards[i] + gamma * returns_current
            adv[i] = g
            returns[i] = returns_current
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        return returns, adv

    def save_model(self, path, actorfile, criticfile):
        actor_fname = path + actorfile
        critic_fname = path + criticfile
        self.actor.save_weights(actor_fname)
        self.critic.save_weights(critic_fname)

    def load_model(self, path, actorfile, criticfile):
        actor_fname = path + actorfile
        critic_fname = path + criticfile
        self.actor.load_weights(actor_fname)
        self.critic.load_weights(critic_fname)
        print('Model Parameters are loaded ...')


#####################################
# Function Definitions

# collect trajectories for a fixed number of time steps
def collect_trajectories(env, agent, tmax=1000):
    states = []
    next_states = []
    actions = []
    rewards = []
    dones = []
    ep_count = 0        # episode count
    state = env.reset()
    for t in range(tmax):
        action = agent.policy(state)
        next_state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        dones.append(done)
        state = next_state

        if done:
            ep_count += 1
            state = env.reset()

    return states, actions, rewards, next_states, dones, ep_count


# main routine for PPO training
def main(env, agent, path='./'):

    if agent.method == 'clip':
        filename = path + 'result_ppo_clip.txt'
    else:
        filename = path + 'result_ppo_klp.txt'

    if os.path.exists(filename):
        print('Deleting existing file. A new one will be created.')
        os.remove(filename)
    else:
        print('The file does not exist. It will be created.')

    #training
    max_seasons = 1000
    best_score = -np.inf
    # best_valid_score = 0
    scores_window = deque(maxlen=100)
    save_scores = []
    for s in range(max_seasons):
        # collect trajectories
        states, actions, rewards, next_states, dones, ep_count = \
            collect_trajectories(env, agent, tmax=10000)

        # train the agent
        a_loss, c_loss, kld_value = agent.train(states, actions, rewards,
                                                next_states, dones, epochs=20)

        # decay the clipping parameter over time
        # agent.actor.epsilon *= 0.999
        # agent.entropy_coeff *= 0.998

        season_score = np.sum(rewards, axis=0) / ep_count
        scores_window.append(season_score)
        save_scores.append(season_score)
        mean_reward = np.mean(scores_window)

        print('Season: {}, season_score: {}, # episodes:{}, mean score:{:.2f}'\
              .format(s, season_score, ep_count, mean_reward))

        if best_score < mean_reward:
            best_score = mean_reward
            agent.save_model(path, 'actor_weights_best.h5', 'critic_weights_best.h5')
            #print('*** Season:{}, best score: {}. Model Saved ***'.format(s, best_score))

        # book keeping
        if agent.method == 'penalty':
            with open(filename, 'a') as file:
                file.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(s,
                                    season_score, mean_reward, a_loss, c_loss, kld_value, agent.actor.beta))
        else:
            with open(filename, 'a') as file:
                file.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(s,
                                                        season_score, mean_reward, a_loss, c_loss, kld_value))

        if season_score > -200:
            print('Problem is solved in {} seasons.'.format(s))
            agent.save_model(path, 'actor_weights.h5', 'critic_weights.h5')
            break

    env.close()


if __name__ == '__main__':

    # Gym Environment
    env = gym.make('Pendulum-v0')
    state_size = env.observation_space.shape
    action_size = env.action_space.shape
    action_bound = env.action_space.high

    # create an agent
    agent = PPOAgent(state_size, action_size,
                         batch_size=200,
                         upper_bound=action_bound)

    # training with seasons
    main(env, agent)

    # test
    # test(env, agent)


