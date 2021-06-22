"""
Implementing Proximal Policy Optimization (PPO) for Kuka Environment
PPO_CLIP Algorithm

This program is same as 'ppo.py' with following changes:
- compute_advantage function is written differently. Look into 'ipg.py' file inside IPG folder.
- PPOAgent class contains a 'run' function which is similar to the main file
- Also computes total computation time
- It is created to compare results with 'ipg.py' and 'ipg_her.py'
- It is used inside the file 'main.py' in the parent folder.
"""

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import datetime
import os
from collections import deque

# local imports
from common.FeatureNet import FeatureNetwork, AttentionFeatureNetwork
from common.utils import uniquify

###########################
## TENSORFLOW Related Logistics
################################
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
                 learning_rate, epsilon, beta, c_loss_coeff,
                 entropy_coeff, kl_target,
                 upper_bound, feature_model, method='clip'):
        self.state_size = state_size  # shape: (w, h, c)
        self.action_size = action_size  # shape: (n, )
        self.lr = learning_rate
        self.upper_bound = upper_bound
        self.epsilon = epsilon          # required for clip method
        self.beta = beta        # required for KL-penalty method
        self.c1 = c_loss_coeff      # critic loss coefficient
        self.c2 = entropy_coeff     # entropy coefficient
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

        if self.feature_model is None:
            f = tf.keras.layers.Dense(128, activation='relu', trainable=trainable)(state_input)
        else:
            f = self.feature_model(state_input)

        f = tf.keras.layers.Dense(128, activation='relu', trainable=trainable)(f)
        f = tf.keras.layers.Dense(64, activation="relu", trainable=trainable)(f)
        net_out = tf.keras.layers.Dense(self.action_size[0], activation='tanh',
                                        kernel_initializer=last_init, trainable=trainable)(f)
        net_out = net_out * self.upper_bound  # element-wise product
        model = tf.keras.Model(state_input, net_out, name='actor')
        model.summary()
        tf.keras.utils.plot_model(model, to_file='ppo_actor_net.png',
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
                # self.update_beta()    # used in agent.train()
            elif self.method == 'clip':
                clipped_surr = tf.clip_by_value(ratio, 1. - self.epsilon,
                                                1. + self.epsilon) * adv_stack
                l_clip = tf.reduce_mean(tf.minimum(surr, clipped_surr))
                actor_loss = - (l_clip - self.c1 * c_loss + self.c2 * entropy)
            actor_weights = self.model.trainable_variables
        # outside gradient tape
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

        if self.feature_model is None:
            feature = tf.keras.layers.Dense(128, activation='relu', trainable=trainable)(state_input)
        else:
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

    def __call__(self, state):
        # input is a tensor
        value = tf.squeeze(self.model(state))
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
# PPO AGENT
#########################
class PPOAgent:
    def __init__(self, env, SEASONS, success_value, lr_a, lr_c,
                 epochs, training_steps, batch_size,
                 epsilon, gamma, lmbda, beta=0.5, c_loss_coeff=0.0,
                 entropy_coeff=0.0, kl_target=0.01,
                 use_attention=False, use_mujoco=False, method='clip'):
        self.env = env
        self.action_size = self.env.action_space.shape

        self.use_mujoco = use_mujoco
        if self.use_mujoco:
            self.state_size = self.env.observation_space["observation"].shape
        else:
            self.state_size = self.env.observation_space.shape

        self.upper_bound = self.env.action_space.high
        self.SEASONS = SEASONS
        self.success_value = success_value
        self.lr_a = lr_a    # actor learning rate
        self.lr_c = lr_c    # critic learning rate
        self.epochs = epochs
        self.episode = 0
        self.training_steps = training_steps    # no. of steps in each season
        self.batch_size = batch_size
        self.gamma = gamma  # discount factor
        self.lmbda = lmbda  # required for Generalized Advantage Estimator (GAE)
        self.beta = beta    # required for KL-Penalty method
        self.c1 = c_loss_coeff      # c_loss coefficient
        self.c2 = entropy_coeff     # entropy coefficient
        self.epsilon = epsilon  # clip_factor
        self.kl_target = kl_target      # target value of KL divergence
        self.use_attention = use_attention
        self.method = method    # chose 'clip' or 'penalty'

        # create actor/critic models
        if self.use_mujoco:
            self.feature = None
        elif self.use_attention:
            self.feature = AttentionFeatureNetwork(self.state_size, lr_a)
        else:
            self.feature = FeatureNetwork(self.state_size, lr_a)

        self.actor = PPOActor(self.state_size, self.action_size, self.lr_a,
                              self.epsilon, self.beta, self.c1, self.c2, self.kl_target,
                              self.upper_bound, self.feature, self.method)
        self.critic = PPOCritic(self.state_size, self.action_size,
                                self.lr_c, self.feature)       # estimates state value

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

    def get_value(self, state):
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        value = self.critic(tf_state)
        return value.numpy()

    def train(self, states, actions, rewards, next_states, dones):

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # compute advantages and discounted cumulative rewards
        target_values, advantages = self.compute_advantage(rewards, states, next_states, dones)

        target_values = tf.convert_to_tensor(target_values, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

        # current action probability distribution
        mean, std = self.actor(states)
        pi = tfp.distributions.Normal(mean, std)

        n_split = len(rewards) // self.batch_size
        assert n_split > 0, 'there should be at least one split'

        indexes = np.arange(n_split, dtype=int)

        # training
        a_loss_list = []
        c_loss_list = []
        kl_list = []
        np.random.shuffle(indexes)
        for _ in range(self.epochs):
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

    # Generalized Advantage Estimator (GAE)
    def compute_advantage(self, r_batch, s_batch, ns_batch, d_batch):
        # input: tensors
        gamma = self.gamma
        lmbda = self.lmbda
        s_values = tf.squeeze(self.critic(s_batch)) # input: tensor
        ns_values = tf.squeeze(self.critic(ns_batch))
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

    def save_model(self, path, actor_filename, critic_filename):
        actor_file = path + actor_filename
        critic_file = path + critic_filename
        self.actor.save_weights(actor_file)
        self.critic.save_weights(critic_file)

    def load_model(self, path, actor_filename, critic_filename):
        actor_file = path + actor_filename
        critic_file = path + critic_filename
        self.actor.model.load_weights(actor_file)
        self.critic.model.load_weights(critic_file)

    # Validation routine
    def validate(self, env, max_eps=50):
        ep_reward_list = []
        for ep in range(max_eps):
            if self.use_mujoco:
                state = env.reset()["observation"]
            else:
                state = env.reset()
                state = np.asarray(state, dtype=np.float32) / 255.0

            t = 0
            ep_reward = 0
            while True:
                action = self.policy(state)
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

    # agent training routine
    def run(self):
        ##################################
        # TENSORBOARD SETTINGS
        TB_LOG = False # enable / disable tensorboard logging
        if TB_LOG:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'logs/train/' + current_time
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        ########################################
        PATH_FLAG = True
        path = './'
        if self.use_attention:
            filename = path + 'result_ppo_attn.txt'
        else:
            filename = path + 'result_ppo.txt'

        if PATH_FLAG:   # create unique filenames
            filename = uniquify(filename)
        else:   # delete existing files
            if os.path.exists(filename):
                os.remove(filename)
        ###################################
        VALIDATION = False
        if VALIDATION:
            val_scores = deque(maxlen=50)
            val_freq = 10
        ##################################
        start = datetime.datetime.now()
        best_score = -np.inf
        s_scores = deque(maxlen=50)
        for s in range(self.SEASONS):
            # discard trajectories from previous season
            states, next_states, actions, rewards, dones = [], [], [], [], []
            s_score = 0

            # initial state
            if self.use_mujoco:
                state = self.env.reset()["observation"]
            else:
                state = self.env.reset()
                state = np.asarray(state, dtype=np.float32) / 255.0

            done, score = False, 0
            for t in range(self.training_steps):
                action = self.policy(state)
                next_state, reward, done, _ = self.env.step(action)

                if self.use_mujoco:
                    reward = 1 if reward == 0 else 0
                    next_state = next_state["observation"]
                else:
                    next_state = np.asarray(next_state, dtype=np.float32) / 255.0

                states.append(state)
                next_states.append(next_state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

                state = next_state
                score += reward

                if done:
                    self.episode += 1
                    s_score += score    # season score

                    if self.use_mujoco:
                        state = self.env.reset()["observation"]
                    else:
                        state = self.env.reset()
                        state = np.asarray(state, dtype=np.float32)

                    done, score = False, 0

            # end of season
            # train the agent
            a_loss, c_loss, kld = self.train(states, actions, rewards,
                                             next_states, dones)

            success_rate = s_score / sum(dones)
            s_scores.append(s_score)
            mean_s_score = np.mean(s_scores)
            if mean_s_score > best_score:
                self.save_model(path, 'actor_wts.h5', 'critic_wts.h5')
                print('Season: {}, Update best score: {} --> {}, Model Saved!'\
                      .format(s, best_score, mean_s_score))
                best_score = mean_s_score

            if s % 10 == 0:
                print('Season: {}, success_rate: {}, mean_success_rate: {}'\
                      .format(s, success_rate, mean_s_score/sum(dones)))

            if VALIDATION:
                if s % val_freq == 0:
                    val_score = self.validate(self.env)
                    val_scores.append(val_score)
                    mean_val_score = np.mean(val_scores)
                    print('Season: {}, Validation Score: {}, Mean Validation Score: {}'\
                          .format(s, val_score, mean_val_score))

            if TB_LOG:
                with train_summary_writer.as_default():
                    tf.summary.scalar('1. Season Score', s_score, step=s)
                    tf.summary.scalar('2. Mean Season Score', mean_s_score, step=s)
                    tf.summary.scalar('3. Success rate', success_rate, step=s)
                    if VALIDATION:
                        tf.summary.scalar('4. Validation Score', val_score, step=s)
                    tf.summary.scalar('5. Actor Loss', a_loss, step=s)
                    tf.summary.scalar('6. Critic Loss', c_loss, step=s)

            with open(filename, 'a') as file:
                file.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n' \
                           .format(s, s_score, mean_s_score, a_loss, c_loss))

            # if best_score > self.success_value:
            #     print('Problem is solved in {} episodes with score {}'.format(self.episode, best_score))
            #     break

        # season-loop ends here
        self.env.close()
        end = datetime.datetime.now()
        print('Time to completion: {}'.format(end-start))


