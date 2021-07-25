'''
IPG + HER
Originally contributed by Mr. Hayden Sampson
URL: https://github.com/hayden750/DeepHEC

- I use a slightly different reward function
- function1: add_her_experience(): adds HER experience to the buffer once at the end of each season with K=1.
- function2: add_her_experience_with_terminal_goal() adds HER experience to the buffer after the end of each episode
        with terminal state as the goal.
- both of these functions have same performance. function2 is relatively better with lower variance.

'''

import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
import os
import datetime
import random
from collections import deque

# Absolute path of modules
sys.path.append("/home/swagat/GIT/RL-Projects-SK/src/kukacam/common")
sys.path.append("/home/swagat/GIT/RL-Projects-SK/src/kukacam")
sys.path.append("/home/swagat/GIT/RL-Projects-SK/src/kukacam/IPG")
print(sys.path)

# Local imports
from common.FeatureNet import FeatureNetwork, AttentionFeatureNetwork
# from common.AttnFeatureNet import AttentionFeatureNetwork
from common.buffer import HERBuffer
from common.utils import uniquify


class IPGActor:
    def __init__(self, state_size, goal_size, action_size, upper_bound,
                 lr, epsilon, feature):
        self.state_size = state_size  # shape: (w, h, c)
        self.goal_size = goal_size    # shape: (w, h, c)
        self.action_size = action_size  # shape: (n, )
        self.lr = lr
        self.epsilon = epsilon              # Clipping value
        self.upper_bound = upper_bound

        # create NN models
        self.feature_model = feature
        self.model = self._build_net(trainable=True)
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        # additions
        logstd = tf.Variable(np.zeros(shape=self.action_size, dtype=np.float32))
        self.model.logstd = logstd
        self.model.trainable_variables.append(logstd)

    def _build_net(self, trainable=True):
        # input: 49x49x3 RGB image
        last_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03)
        state_input = tf.keras.layers.Input(shape=self.state_size)
        goal_input = tf.keras.layers.Input(shape=self.goal_size)

        if self.feature_model is None:
            f_state = tf.keras.layers.Dense(128, activation="relu", trainable=trainable)(state_input)
            f_goal = tf.keras.layers.Dense(128, activation="relu", trainable=trainable)(goal_input)
        else:
            f_state = self.feature_model(state_input)
            f_goal = self.feature_model(goal_input)

        f = tf.keras.layers.Concatenate()([f_state, f_goal])
        f = tf.keras.layers.Dense(128, activation='relu', trainable=trainable)(f)
        f = tf.keras.layers.Dense(64, activation="relu", trainable=trainable)(f)
        net_out = tf.keras.layers.Dense(self.action_size[0], activation='tanh',
                                        kernel_initializer=last_init, trainable=trainable)(f)
        net_out = net_out * self.upper_bound  # element-wise product
        model = tf.keras.Model(inputs=[state_input, goal_input], outputs=net_out, name='actor')
        model.summary()
        tf.keras.utils.plot_model(model, to_file='ipg_actor_net.png',
                                  show_shapes=True, show_layer_names=True)
        return model

    def __call__(self, state, goal):
        # input is a tensor
        mean = tf.squeeze(self.model([state, goal]))
        std = tf.squeeze(tf.exp(self.model.logstd))
        return mean, std    # returns tensors

    def train(self, states, actions, advantages, old_pi, critic, b, s_batch, goals, g_batch):
        with tf.GradientTape() as tape:
            # on-policy ppo loss
            mean = tf.squeeze(self.model([states, goals]))
            std = tf.squeeze(tf.exp(self.model.logstd))
            pi = tfp.distributions.Normal(mean, std)
            ratio = tf.exp(pi.log_prob(tf.squeeze(actions)) -
                           old_pi.log_prob(tf.squeeze(actions)))
            adv_stack = tf.stack([advantages for _ in range(self.action_size[0])], axis=1)  # shape(-1,3) check ..
            surr = ratio * adv_stack         # surrogate function
            clipped_surr = tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * adv_stack
            ppo_loss = tf.reduce_mean(tf.minimum(surr, clipped_surr))

            # off-policy ppo loss
            action_est = tf.squeeze(self.model([s_batch, g_batch]))       # action estimate
            q_values = critic.model([s_batch, action_est])     # q estimates
            # sum_q_values = K.sum(K.mean(q_values))          # check the dimensions
            sum_q_values = tf.reduce_sum(tf.reduce_mean(q_values))
            off_loss = (b / len(s_batch)) * sum_q_values
            actor_loss = -tf.reduce_sum(ppo_loss + off_loss)
            actor_wts = self.model.trainable_variables

        # outside gradient tape
        actor_grad = tape.gradient(actor_loss, actor_wts)
        self.optimizer.apply_gradients(zip(actor_grad, actor_wts))
        return actor_loss.numpy()

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


class DDPGCritic:
    def __init__(self, state_size, action_size, learning_rate,
                 gamma, feature_model):
        self.state_size = state_size    # shape: (w, h, c)
        self.action_size = action_size  # shape: (n, )
        self.lr = learning_rate
        self.gamma = gamma          # discount factor

        # create NN model
        self.feature = feature_model
        self.model = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        state_input = layers.Input(shape=self.state_size)

        if self.feature is None:
            f = layers.Dense(128, activation="relu")(state_input)
        else:
            f = self.feature(state_input)

        state_out = layers.Dense(32, activation='relu')(f)
        state_out = layers.Dense(32, activation='relu')(state_out)

        # action as input
        action_input = layers.Input(shape=self.action_size)
        action_out = layers.Dense(32, activation='relu')(action_input)

        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(128, activation='relu')(concat)
        out = layers.Dense(64, activation='relu')(out)
        net_out = layers.Dense(1)(out)
        model = tf.keras.Model(inputs=[state_input, action_input], outputs=net_out)
        model.summary()
        tf.keras.utils.plot_model(model, to_file='critic_net.png',
                                  show_shapes=True, show_layer_names=True)
        return model

    def train(self, state_batch, action_batch, y):
        with tf.GradientTape() as tape:
            critic_wts = self.model.trainable_variables
            critic_value = self.model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.square(y - critic_value))
        # outside gradient tape
        critic_grad = tape.gradient(critic_loss, critic_wts)
        self.optimizer.apply_gradients(zip(critic_grad, critic_wts))
        return critic_loss.numpy()

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


class Baseline:
    def __init__(self, state_size, action_size, lr, feature):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.feature = feature
        self.model = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        state_input = tf.keras.layers.Input(shape=self.state_size)

        if self.feature is None:
            f = tf.keras.layers.Dense(128, activation="relu")(state_input)
        else:
            f = self.feature(state_input)

        out = tf.keras.layers.Dense(128, activation='relu')(f)
        out = tf.keras.layers.Dense(64, activation='relu')(out)
        out = tf.keras.layers.Dense(32, activation='relu')(out)
        net_out = tf.keras.layers.Dense(1)(out)
        model = tf.keras.Model(inputs=state_input, outputs=net_out)
        model.summary()
        return model

    def train(self, states, returns):
        with tf.GradientTape() as tape:
            critic_wts = self.model.trainable_variables
            critic_values = self.model(states)
            critic_loss = tf.math.reduce_mean(tf.square(returns - critic_values))

        critic_grad = tape.gradient(critic_loss, critic_wts)
        self.optimizer.apply_gradients(zip(critic_grad, critic_wts))
        return critic_loss.numpy()

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


class IPGHERAgent:
    def __init__(self, env, SEASONS, success_value, lr_a, lr_c,
                 epochs, training_batch, batch_size, buffer_capacity, epsilon,
                 gamma, lmbda, use_attention=False, use_mujoco=False,
                 filename=None, tb_log=False, val_freq=50, path='./'):
        self.env = env
        self.action_size = self.env.action_space.shape

        self.use_mujoco = use_mujoco
        if self.use_mujoco:
            self.state_size = self.env.observation_space["observation"].shape
            self.goal_size = self.env.observation_space["desired_goal"].shape
        else:
            self.state_size = self.env.observation_space.shape
            self.goal_size = self.state_size

        self.upper_bound = self.env.action_space.high
        self.SEASONS = SEASONS
        self.episode = 0
        self.replay_count = 0
        self.success_value = success_value
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.epochs = epochs
        self.training_batch = training_batch
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.epsilon = epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.use_attention = use_attention
        self.thr = 0.3          # threshold similarity between state & goal
        self.filename = filename
        self.TB_LOG = tb_log
        self.val_freq = val_freq
        self.path = path            # location to store results

        if len(self.state_size) == 3:
            self.image_input = True     # image input
        elif len(self.state_size) == 1:
            self.image_input = False        # vector input
        else:
            raise ValueError("Input can be a vector or an image")

        # create HER buffer to store experiences
        self.buffer = HERBuffer(self.buffer_capacity, self.batch_size)

        # extract features from input images
        if self.use_mujoco:
            self.feature = None     # assuming non-image mujoco environment
        elif self.use_attention and self.image_input:   # attention + image input
            print('Currently Attention handles only image input')
            self.feature = AttentionFeatureNetwork(self.state_size, lr_a)
        elif self.use_attention is False and self.image_input is True:  # image input
            print('You have selected an image input')
            self.feature = FeatureNetwork(self.state_size, lr_a)
        else:       # non-image input
            print('You have selected a non-image input.')
            self.feature = None

        # Actor Model
        self.actor = IPGActor(state_size=self.state_size, goal_size=self.goal_size,
                              action_size=self.action_size, upper_bound=self.upper_bound,
                              lr=self.lr_a, epsilon=self.epsilon, feature=self.feature)
        # Critic Model
        self.critic = DDPGCritic(state_size=self.state_size, action_size=self.action_size,
                                 learning_rate=self.lr_c, gamma=self.gamma, feature_model=self.feature)
        # Baseline Model
        self.baseline = Baseline(state_size=self.state_size, action_size=self.action_size,
                                                lr=self.lr_c, feature=self.feature)

    def policy(self, state, goal, deterministic=False):
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        tf_goal = tf.expand_dims(tf.convert_to_tensor(goal), 0)
        mean, std = self.actor(tf_state, tf_goal)
        if deterministic:
            action = mean
        else:
            pi = tfp.distributions.Normal(mean, std)
            action = pi.sample()
            #action = mean + tf.random.uniform(-self.upper_bound, self.upper_bound, size=mean.shape) * std

        action = tf.clip_by_value(action, -self.upper_bound, self.upper_bound)
        return action.numpy()

    def compute_advantage(self, r_batch, s_batch, ns_batch, d_batch):
        # input: tensors
        gamma = self.gamma
        lmbda = self.lmbda
        s_values = tf.squeeze(self.baseline.model(s_batch)) # input: tensor
        ns_values = tf.squeeze(self.baseline.model(ns_batch))
        returns = []
        gae = 0     # generalized advantage estimate
        for i in reversed(range(len(r_batch))):
            delta = r_batch[i] + gamma * ns_values[i] * (1 - d_batch[i]) - s_values[i]
            gae = delta + gamma * lmbda * (1 - d_batch[i]) * gae
            returns. insert(0, gae + s_values[i])

        returns = np.array(returns)
        adv = returns - s_values.numpy()
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10) # output: numpy array
        return adv, returns

    def compute_targets(self, r_batch, ns_batch, d_batch, g_batch):
        mean = self.actor.model([ns_batch, g_batch])
        target_critic = self.critic.model([ns_batch, mean])
        y = r_batch + self.gamma * (1 - d_batch) * target_critic
        return y

    def compute_adv_bar(self, s_batch, a_batch, g_batch):
        mean = self.actor.model([s_batch, g_batch])
        x = tf.squeeze(a_batch) - tf.squeeze(mean)
        y = tf.squeeze(self.critic.model([s_batch, mean]))
        adv_bar = y * x         # check this
        return adv_bar

    def reward_func(self, state, goal):
        good_done = np.linalg.norm(state - goal) <= self.thr
        reward = 1 if good_done else 0
        return good_done, reward

    # implements on-policy & off-policy training
    def replay(self, states, actions, rewards, next_states, dones, goals):
        n_split = len(rewards) // self.batch_size

        # use this if you are using tf.split
        # valid_length = n_split * self.batch_size
        # states = states[0:valid_length]
        # actions = actions[0:valid_length]
        # rewards = rewards[0:valid_length]
        # dones = dones[0:valid_length]
        # next_states = next_states[0:valid_length]
        # goals = goals[0:valid_length]

        # convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        goals = tf.convert_to_tensor(goals, dtype=tf.float32)

        # IPG paper: https: // arxiv.org / abs / 1706.00387
        # Fit baseline using collected experience and compute advantages
        advantages, returns = self.compute_advantage(rewards, states, next_states, dones) # returns np.arrays

        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        # if use control variate: Compute critic-based advantages
        # and compute learning signal
        use_CV = False
        v = 0.2
        if use_CV:
            # compute critic-based advantage
            adv_bar = self.compute_adv_bar(states, actions)
            ls = advantages - adv_bar
            b = 1
        else:
            ls = advantages
            b = v

        ls *= (1 - v)

        # s_split = tf.split(states, n_split)
        # a_split = tf.split(actions, n_split)
        # t_split = tf.split(returns, n_split)
        # ls_split = tf.split(ls, n_split)
        # g_split = tf.split(goals, n_split)

        indexes = np.arange(n_split, dtype=int)

        # current policy
        mean, std = self.actor(states, goals)
        pi = tfp.distributions.Normal(mean, std)

        a_loss_list = []
        c_loss_list = []
        np.random.shuffle(indexes)
        for _ in range(self.epochs):
            s_batch, a_batch, r_batch, ns_batch, d_batch, g_batch = self.buffer.sample()

            # convert to tensors
            s_batch = tf.convert_to_tensor(s_batch, dtype=tf.float32)
            a_batch = tf.convert_to_tensor(a_batch, dtype=tf.float32)
            r_batch = tf.convert_to_tensor(r_batch, dtype=tf.float32)
            ns_batch = tf.convert_to_tensor(ns_batch, dtype=tf.float32)
            d_batch = tf.convert_to_tensor(np.asarray(d_batch), dtype=tf.float32)
            g_batch = tf.convert_to_tensor(g_batch, dtype=tf.float32)

            for i in indexes:
                old_pi = pi[i * self.batch_size: (i+1) * self.batch_size]
                s_split = tf.gather(states, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)
                a_split = tf.gather(actions, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)
                ls_split = tf.gather(ls, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)
                g_split = tf.gather(goals, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)
                t_split = tf.gather(returns, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)

                # update actor
                # a_loss = self.actor.train(s_split[i], a_split[i], ls_split[i],
                #                           old_pi, self.critic, b, s_batch,
                #                           g_split[i], g_batch)
                a_loss = self.actor.train(s_split, a_split, ls_split,
                                          old_pi, self.critic, b, s_batch,
                                          g_split, g_batch)
                a_loss_list.append(a_loss)
                # update baseline
                # v_loss = self.baseline.train(s_split[i], t_split[i])
                v_loss = self.baseline.train(s_split, t_split)

            # update critic
            y = self.compute_targets(r_batch, ns_batch, d_batch, g_batch)
            c_loss = self.critic.train(s_batch, a_batch, y)
            c_loss_list.append(c_loss)

        self.replay_count += 1      # Why do you need this?
        return np.mean(a_loss_list), np.mean(c_loss_list)

    # Validation routine
    def validate(self, env, max_eps=50):
        ep_reward_list = []
        for ep in range(max_eps):

            if self.use_mujoco:
                env_init = self.env.reset()
                goal = env_init["desired_goal"]
                state = env_init["observation"]
            else:
                state = self.env.reset()
                state = np.asarray(state, dtype=np.float32) / 255.0  # convert into float array
                goal = self.env.reset()  # take random state as goal
                goal = np.asarray(goal, dtype=np.float32) / 255.0

            t = 0
            ep_reward = 0
            while True:
                action = self.policy(state, goal, deterministic=True)
                next_obsv, reward, done, _ = env.step(action)

                if self.use_mujoco:
                    reward = 1 if reward == 0 else 0
                    next_state = next_obsv["observation"]
                else:
                    next_state = np.asarray(next_obsv, dtype=np.float32) / 255.0

                state = next_state
                ep_reward += reward
                t += 1
                if done:
                    ep_reward_list.append(ep_reward)
                    break
        # outside for loop
        mean_ep_reward = np.mean(ep_reward_list)
        return mean_ep_reward

    def add_her_experience2(self, ep_experience, K=1):
        # add additional experience to the replay buffer
        for t in range(len(ep_experience)):
            # get K future steps per time-step
            for _ in range(K):
                # get a random future time instant
                future = np.random.randint(t, len(ep_experience))
                # get a new goal at t_future
                goal_ = ep_experience[future][3]
                state_ = ep_experience[t][0]
                action_ = ep_experience[t][1]
                next_state_ = ep_experience[t][3]
                done_, reward_ = self.reward_func(next_state_, goal_)
                # add new experience to HER buffer
                self.buffer.record(state_, action_, reward_, next_state_, done_, goal_)

    def add_her_experience(self, ep_experience, hind_goal):
        for i in range(len(ep_experience)):
            if hind_goal is None:
                future = np.random.randint(i, len(ep_experience))
                goal_ = ep_experience[future][3]
            else:
                goal_ = hind_goal
            state_ = ep_experience[i][0]
            action_ = ep_experience[i][1]
            next_state_ = ep_experience[i][3]
            current_goal = ep_experience[i][5]
            if self.use_mujoco:
                done_ = np.array_equal(current_goal, goal_)
                reward_ = 1 if done_ else 0
            else:
                done_, reward_ = self.reward_func(next_state_, goal_)
            # add new experience to the main buffer
            self.buffer.record(state_, action_, reward_, next_state_, done_, goal_)

    def run(self):
        if self.path is None:
            self.path = './'
        #######################
        # TENSORBOARD SETTINGS
        if self.TB_LOG:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = self.path + 'tb_logs/train/' + current_time
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        ########################################
        if self.filename is None:
            self.filename = 'ipg_her_output.txt'
        self.filename = uniquify(self.path + self.filename)

        if self.val_freq is not None:
            val_scores = deque(maxlen=50)
            val_score = 0

        # initial state and goal
        if self.use_mujoco:
            env_init = self.env.reset()
            goal = env_init["desired_goal"]
            state = env_init["observation"]
        else:
            state = self.env.reset()
            state = np.asarray(state, dtype=np.float32) / 255.0  # convert into float array
            goal = self.env.reset()  # take random state as goal
            goal = np.asarray(goal, dtype=np.float32) / 255.0

        start = datetime.datetime.now()
        best_score = -np.inf
        s_scores = []             # all season scores
        s_ep_lens = []            # average episode length for each season
        for s in range(self.SEASONS):
            # discard trajectories from previous season
            states, next_states, actions, rewards, dones, goals = [], [], [], [], [], []
            ep_experience = []     # episodic experience buffer
            s_score = 0
            done, score = False, 0
            self.episode = 0  # initialize the episode_count for each season
            for _ in range(self.training_batch):    # time steps
                action = self.policy(state, goal)
                next_state, reward, done, _ = self.env.step(action)

                if self.use_mujoco:
                    reward = 1 if reward == 0 else 0
                    achieved_goal = next_state["achieved_goal"]
                    next_state = next_state["observation"]
                else:
                    next_state = np.asarray(next_state, dtype=np.float32) / 255.0

                # this is used for on-policy training
                states.append(state)
                next_states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                goals.append(goal)

                # store in replay buffer for off-policy training
                self.buffer.record(state, action, reward, next_state, done, goal)
                # Also store in a separate buffer
                ep_experience.append([state, action, reward, next_state, done, goal])

                state = next_state
                score += reward

                if done:
                    if self.use_mujoco:
                        hind_goal = achieved_goal
                    else:
                        hind_goal = ep_experience[-1][3]  # Final state strategy

                    # Add hindsight experience to the buffer
                    # in this case, use last state as the goal_state
                    # here ep_exp is cleared at the end of each episode.
                    # which makes sense
                    self.add_her_experience(ep_experience, hind_goal)
                    # clear the local experience buffer
                    ep_experience = []

                    self.episode += 1
                    s_score += score        # season score

                    if self.use_mujoco:
                        env_init = self.env.reset()
                        goal = env_init["desired_goal"]
                        state = env_init["observation"]
                        done, score = False, 0
                    else:
                        state = self.env.reset()
                        state = np.asarray(state, dtype=np.float32) / 255.0
                        goal = self.env.reset()
                        goal = np.asarray(goal, dtype=np.float32) / 255.0
                        done, score = False, 0

            # end of for training_batch loop

            # Add hindsight experience to the buffer
            # here we are using random goal states
            # self.add_her_experience(ep_experience)
            # clear the local experience buffer
            # ep_experience = []          # not required as we are clearing it for every season.

            # on-policy & off-policy training
            a_loss, c_loss = self.replay(states, actions, rewards, next_states, dones, goals)

            avg_episodic_score = s_score / sum(dones)
            s_scores.append(avg_episodic_score)
            mean_s_score = np.mean(s_scores)
            s_ep_lens.append(self.training_batch / sum(dones))
            mean_ep_len = np.mean(s_ep_lens)

            if mean_s_score > best_score:
                self.save_model('actor_wts.h5', 'critic_wts.h5', 'baseline_wts.h5')
                print('Season: {}, Update best score: {}-->{}, Model saved!'.format(s, best_score, mean_s_score))
                best_score = mean_s_score

            if self.val_freq is not None and s % self.val_freq == 0:
                print('Season: {}, Score: {}, Mean score: {}'.format(s, s_score, mean_s_score))
                val_score = self.validate(self.env)
                val_scores.append(val_score)
                mean_val_score = np.mean(val_scores)
                print('Season: {}, Validation Score: {}, Mean Validation Score: {}' \
                      .format(s, val_score, mean_val_score))

            if self.TB_LOG:
                with train_summary_writer.as_default():
                    tf.summary.scalar('1. Season Score', s_score, step=s)
                    tf.summary.scalar('2. Mean Season Score', mean_s_score, step=s)
                    tf.summary.scalar('3. Success rate', avg_episodic_score, step=s)
                    tf.summary.scalar('4. Validation Score', val_score, step=s)
                    tf.summary.scalar('5. Actor Loss', a_loss, step=s)
                    tf.summary.scalar('6. Critic Loss', c_loss, step=s)
                    tf.summary.scalar('7. Mean Episode Length', mean_ep_len, step=s)

            with open(self.filename, 'a') as file:
                file.write('{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'
                           .format(s, self.episode, mean_ep_len,
                                   avg_episodic_score, mean_s_score, a_loss, c_loss))

            if self.success_value is not None:
                if best_score > self.success_value:
                    print('Problem is solved in {} episodes with score {}'.format(self.episode, best_score))
                    break

        # end of season loop
        end = datetime.datetime.now()
        print('Time to completion: {}'.format(end-start))
        self.env.close()

    def save_model(self, actor_filename, critic_filename, baseline_filename):
        actor_file = self.path + actor_filename
        critic_file = self.path + critic_filename
        baseline_file = self.path + baseline_filename
        self.actor.save_weights(actor_file)
        self.critic.save_weights(critic_file)
        self.baseline.save_weights(baseline_file)

    def load_model(self, actor_filename, critic_filename, baseline_filename):
        actor_file = self.path + actor_filename
        critic_file = self.path + critic_filename
        baseline_file = self.path + baseline_filename
        self.actor.load_weights(actor_file)
        self.critic.load_weights(critic_file)
        self.baseline.load_weights(baseline_file)



class IPGHERAgent2:
    def __init__(self, state_size, action_size, action_upper_bound, lr_a, lr_c,
                 epochs, batch_size, buffer_capacity, epsilon,
                 gamma, lmbda, use_attention=False):
        self.state_size = state_size
        self.action_size = action_size 
        self.goal_size = state_size
        self.upper_bound = action_upper_bound
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.epochs = epochs
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.epsilon = epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.use_attention = use_attention
        self.thr = 0.3          # threshold similarity between state & goal

        if len(self.state_size) == 3:
            self.image_input = True     # image input
        elif len(self.state_size) == 1:
            self.image_input = False        # vector input
        else:
            raise ValueError("Input can be a vector or an image")

        # create HER buffer to store experiences
        self.buffer = HERBuffer(self.buffer_capacity, self.batch_size)

        # extract features from input images
        if self.use_attention and self.image_input:   # attention + image input
            print('Currently Attention handles only image input')
            self.feature = AttentionFeatureNetwork(self.state_size, lr_a)
        elif self.use_attention is False and self.image_input is True:  # image input
            print('You have selected an image input')
            self.feature = FeatureNetwork(self.state_size, lr_a)
        else:       # non-image input
            print('You have selected a non-image input.')
            self.feature = None

        # Actor Model
        self.actor = IPGActor(state_size=self.state_size, goal_size=self.goal_size,
                              action_size=self.action_size, upper_bound=self.upper_bound,
                              lr=self.lr_a, epsilon=self.epsilon, feature=self.feature)
        # Critic Model
        self.critic = DDPGCritic(state_size=self.state_size, action_size=self.action_size,
                                 learning_rate=self.lr_c, gamma=self.gamma, feature_model=self.feature)
        # Baseline Model
        self.baseline = Baseline(state_size=self.state_size, action_size=self.action_size,
                                                lr=self.lr_c, feature=self.feature)

    def policy(self, state, goal, deterministic=False):
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        tf_goal = tf.expand_dims(tf.convert_to_tensor(goal), 0)
        mean, std = self.actor(tf_state, tf_goal)
        if deterministic:
            action = mean
        else:
            pi = tfp.distributions.Normal(mean, std)
            action = pi.sample()
            #action = mean + tf.random.uniform(-self.upper_bound, self.upper_bound, size=mean.shape) * std

        action = tf.clip_by_value(action, -self.upper_bound, self.upper_bound)
        return action.numpy()

    def compute_advantage(self, r_batch, s_batch, ns_batch, d_batch):
        # input: tensors
        gamma = self.gamma
        lmbda = self.lmbda
        s_values = tf.squeeze(self.baseline.model(s_batch)) # input: tensor
        ns_values = tf.squeeze(self.baseline.model(ns_batch))
        returns = []
        gae = 0     # generalized advantage estimate
        for i in reversed(range(len(r_batch))):
            delta = r_batch[i] + gamma * ns_values[i] * (1 - d_batch[i]) - s_values[i]
            gae = delta + gamma * lmbda * (1 - d_batch[i]) * gae
            returns. insert(0, gae + s_values[i])

        returns = np.array(returns)
        adv = returns - s_values.numpy()
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10) # output: numpy array
        return adv, returns

    def compute_targets(self, r_batch, ns_batch, d_batch, g_batch):
        mean = self.actor.model([ns_batch, g_batch])
        target_critic = self.critic.model([ns_batch, mean])
        y = r_batch + self.gamma * (1 - d_batch) * target_critic
        return y

    def compute_adv_bar(self, s_batch, a_batch, g_batch):
        mean = self.actor.model([s_batch, g_batch])
        x = tf.squeeze(a_batch) - tf.squeeze(mean)
        y = tf.squeeze(self.critic.model([s_batch, mean]))
        adv_bar = y * x         # check this
        return adv_bar

    def reward_func(self, state, goal):
        good_done = np.linalg.norm(state - goal) <= self.thr
        reward = 1 if good_done else 0
        return good_done, reward

    # implements on-policy & off-policy training
    def replay(self, states, actions, rewards, next_states, dones, goals):
        n_split = len(rewards) // self.batch_size

        # use this if you are using tf.split
        # valid_length = n_split * self.batch_size
        # states = states[0:valid_length]
        # actions = actions[0:valid_length]
        # rewards = rewards[0:valid_length]
        # dones = dones[0:valid_length]
        # next_states = next_states[0:valid_length]
        # goals = goals[0:valid_length]

        # convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        goals = tf.convert_to_tensor(goals, dtype=tf.float32)

        # IPG paper: https: // arxiv.org / abs / 1706.00387
        # Fit baseline using collected experience and compute advantages
        advantages, returns = self.compute_advantage(rewards, states, next_states, dones) # returns np.arrays

        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        # if use control variate: Compute critic-based advantages
        # and compute learning signal
        use_CV = False
        v = 0.2
        if use_CV:
            # compute critic-based advantage
            adv_bar = self.compute_adv_bar(states, actions)
            ls = advantages - adv_bar
            b = 1
        else:
            ls = advantages
            b = v

        ls *= (1 - v)

        # s_split = tf.split(states, n_split)
        # a_split = tf.split(actions, n_split)
        # t_split = tf.split(returns, n_split)
        # ls_split = tf.split(ls, n_split)
        # g_split = tf.split(goals, n_split)

        indexes = np.arange(n_split, dtype=int)

        # current policy
        mean, std = self.actor(states, goals)
        pi = tfp.distributions.Normal(mean, std)

        a_loss_list = []
        c_loss_list = []
        np.random.shuffle(indexes)
        for _ in range(self.epochs):
            s_batch, a_batch, r_batch, ns_batch, d_batch, g_batch = self.buffer.sample()

            # convert to tensors
            s_batch = tf.convert_to_tensor(s_batch, dtype=tf.float32)
            a_batch = tf.convert_to_tensor(a_batch, dtype=tf.float32)
            r_batch = tf.convert_to_tensor(r_batch, dtype=tf.float32)
            ns_batch = tf.convert_to_tensor(ns_batch, dtype=tf.float32)
            d_batch = tf.convert_to_tensor(np.asarray(d_batch), dtype=tf.float32)
            g_batch = tf.convert_to_tensor(g_batch, dtype=tf.float32)

            for i in indexes:
                old_pi = pi[i * self.batch_size: (i+1) * self.batch_size]
                s_split = tf.gather(states, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)
                a_split = tf.gather(actions, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)
                ls_split = tf.gather(ls, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)
                g_split = tf.gather(goals, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)
                t_split = tf.gather(returns, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)

                # update actor
                # a_loss = self.actor.train(s_split[i], a_split[i], ls_split[i],
                #                           old_pi, self.critic, b, s_batch,
                #                           g_split[i], g_batch)
                a_loss = self.actor.train(s_split, a_split, ls_split,
                                          old_pi, self.critic, b, s_batch,
                                          g_split, g_batch)
                a_loss_list.append(a_loss)
                # update baseline
                # v_loss = self.baseline.train(s_split[i], t_split[i])
                v_loss = self.baseline.train(s_split, t_split)

            # update critic
            y = self.compute_targets(r_batch, ns_batch, d_batch, g_batch)
            c_loss = self.critic.train(s_batch, a_batch, y)
            c_loss_list.append(c_loss)

        self.replay_count += 1      # Why do you need this?
        return np.mean(a_loss_list), np.mean(c_loss_list)

    def add_her_experience2(self, ep_experience, K=1):
        # add additional experience to the replay buffer
        for t in range(len(ep_experience)):
            # get K future steps per time-step
            for _ in range(K):
                # get a random future time instant
                future = np.random.randint(t, len(ep_experience))
                # get a new goal at t_future
                goal_ = ep_experience[future][3]
                state_ = ep_experience[t][0]
                action_ = ep_experience[t][1]
                next_state_ = ep_experience[t][3]
                done_, reward_ = self.reward_func(next_state_, goal_)
                # add new experience to HER buffer
                self.buffer.record(state_, action_, reward_, next_state_, done_, goal_)

    def add_her_experience(self, ep_experience, hind_goal):
        for i in range(len(ep_experience)):
            if hind_goal is None:
                future = np.random.randint(i, len(ep_experience))
                goal_ = ep_experience[future][3]
            else:
                goal_ = hind_goal
            state_ = ep_experience[i][0]
            action_ = ep_experience[i][1]
            next_state_ = ep_experience[i][3]
            current_goal = ep_experience[i][5]
            if self.use_mujoco:
                done_ = np.array_equal(current_goal, goal_)
                reward_ = 1 if done_ else 0
            else:
                done_, reward_ = self.reward_func(next_state_, goal_)
            # add new experience to the main buffer
            self.buffer.record(state_, action_, reward_, next_state_, done_, goal_)

    def save_model(self, save_path):
        actor_file = save_path + 'ipg_actor_wts.h5'
        critic_file = save_path + 'ipg_critic_wts.h5'
        baseline_file = save_path + 'ipg_bl_wts.h5'
        self.actor.save_weights(actor_file)
        self.critic.save_weights(critic_file)
        self.baseline.save_weights(baseline_file)

    def load_model(self, load_path):
        actor_file = load_path + 'ipg_actor_wts.h5'
        critic_file = load_path + 'ipg_critic_wts.h5'
        baseline_file = load_path + 'ipg_bl_wts.h5'
        self.actor.load_weights(actor_file)
        self.critic.load_weights(critic_file)
        self.baseline.load_weights(baseline_file)
