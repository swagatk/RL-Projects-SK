'''
Soft Actor Critic Algorithm
Original Source: https://github.com/shakti365/soft-actor-critic

- policy function is now a part of the actor network. 
- tf.GradientTape() part of the actor/critic networks are modified to ensure proper flow of gradients. This is something that should be
    replicated to other algorithms such as PPO, IPG, DDPG, TD3 etc. (To do)
- Two versions of SACAgent class definitions are provided. The first version takes 'env' as an input argument which makes it difficult to account
    for changes in environment specifications. The second version SACAgent2 keeps the environment separate from agent definition making it easier
    for the user to pre-process environment input/output signals before passing them to the agent. This is something we will replicate
    in future algorithms. 
- SACAgent2 will supersede SACAgent in the future. 

'''
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
import os
import datetime
import random
from collections import deque
import sys 

sys.path.append(r'/content/gdrive/MyDrive/Colab/RL-Projects-SK/src/kukacam/')

# Local imports
from common.FeatureNet import FeatureNetwork, AttentionFeatureNetwork
from common.buffer import Buffer
from common.utils import uniquify


###############
# ACTOR NETWORK
###############

class SACActor:
    def __init__(self, state_size, action_size, upper_bound,
                 learning_rate, feature):
        self.state_size = state_size  # shape: (w, h, c)
        self.action_size = action_size  # shape: (n, )
        self.lr = learning_rate
        self.upper_bound = upper_bound

        # create NN models
        self.feature_model = feature
        self.model = self._build_net(trainable=True)    # returns mean action
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

        # additional output
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

        f = tf.keras.layers.Dense(256, activation='relu', trainable=trainable)(f)
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

    def policy(self, state):
        # input: tensor
        # output: tensor
        mean = tf.squeeze(self.model(state))
        std = tf.squeeze(tf.exp(self.model.logstd))

        pi = tfp.distributions.Normal(mean, std)
        action_ = pi.sample()
        log_pi_ = pi.log_prob(action_)
        action = tf.clip_by_value(action_, -self.upper_bound, self.upper_bound)

        if tf.rank(action) < 1:     # scalar
            log_pi_a = log_pi_ - tf.math.log(1 - action ** 2 + 1e-16)
        elif 1 <= tf.rank(action) < 2:  # vector
            log_pi_a = tf.reduce_sum((log_pi_ - tf.math.log(1 - action ** 2 + 1e-16)), axis=0, keepdims=True)
        else:   # matrix
            log_pi_a = tf.reduce_sum((log_pi_ - tf.math.log(1 - action ** 2 + 1e-16)), axis=1, keepdims=True)

        return action, log_pi_a

    def train(self, states, alpha, critic1, critic2):
        with tf.GradientTape() as tape:
            actions, log_pi_a = self.policy(states)
            q1 = critic1(states, actions)
            q2 = critic2(states, actions)
            min_q = tf.minimum(q1, q2)
            soft_q = min_q - alpha * log_pi_a
            actor_loss = -tf.reduce_mean(soft_q)
        # outside gradient tape block
        actor_wts = self.model.trainable_variables
        actor_grads = tape.gradient(actor_loss, actor_wts)
        self.optimizer.apply_gradients(zip(actor_grads, actor_wts))
        return actor_loss.numpy()

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


class SACCritic:
    """
    Approximates Q(s,a) function
    """
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
            f = layers.Dense(128, activation='relu')(state_input)
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

    def __call__(self, state, action):
        # input: tensors
        q_value = tf.squeeze(self.model([state, action]))
        return q_value

    def train(self, states, actions, rewards, next_states, dones, actor,
              target_critic1, target_critic2, alpha):
        with tf.GradientTape() as tape:
            # Get Q estimates using actions from replay buffer
            q_values = tf.squeeze(self.model([states, actions]))

            # Sample actions from the policy network for next states
            a_next, log_pi_a = actor.policy(next_states)

            # Get Q value estimates from target Q networks
            q1_target = target_critic1(next_states, a_next)
            q2_target = target_critic2(next_states, a_next)

            # Apply the clipped double Q trick
            # Get the minimum Q value of the 2 target networks
            min_q_target = tf.minimum(q1_target, q2_target)

            # Add the entropy term to get soft Q target
            soft_q_target = min_q_target - alpha * log_pi_a

            y = tf.stop_gradient(rewards * self.gamma * (1 - dones) * soft_q_target)
            critic_loss = tf.math.reduce_mean(tf.square(y - q_values))
            critic_wts = self.model.trainable_variables
        # outside gradient tape
        critic_grad = tape.gradient(critic_loss, critic_wts)
        self.optimizer.apply_gradients(zip(critic_grad, critic_wts))
        return critic_loss.numpy()

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


class SACAgent:
    def __init__(self, env, success_value, epochs,
                 training_episodes, batch_size, buffer_capacity, lr_a=0.0003, lr_c=0.0003, alpha=0.2,
                 gamma=0.99, tau=0.995, use_attention=False, use_mujoco=False,
                 filename=None, tb_log=False, val_freq=50, path='./'):
        self.env = env
        self.action_size = self.env.action_space.shape

        self.use_mujoco = use_mujoco
        if self.use_mujoco:
            self.state_size = self.env.observation_space["observation"].shape
        else:
            self.state_size = self.env.observation_space.shape

        self.upper_bound = np.squeeze(self.env.action_space.high)
        self.time_steps = 0
        self.success_value = success_value
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.epochs = epochs
        self.training_episodes = training_episodes
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.target_entropy = -tf.constant(np.prod(self.action_size), dtype=tf.float32)
        self.gamma = gamma                  # discount factor
        self.tau = tau                      # polyak averaging factor
        self.use_attention = use_attention
        self.filename = filename
        self.TB_LOG = tb_log
        self.val_freq = val_freq
        self.path = path

        if len(self.state_size) == 3:
            self.image_input = True     # image input
        elif len(self.state_size) == 1:
            self.image_input = False        # vector input
        else:
            raise ValueError("Input can be a vector or an image")

        # Select a suitable feature extractor
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

        self.actor = SACActor(self.state_size, self.action_size, self.upper_bound,
                              self.lr_a, self.feature)

        # create two critics
        self.critic1 = SACCritic(self.state_size, self.action_size,
                                 self.lr_c, self.gamma, self.feature)
        self.critic2 = SACCritic(self.state_size, self.action_size,
                                 self.lr_c, self.gamma, self.feature)

        # create two target critics
        self.target_critic1 = SACCritic(self.state_size, self.action_size,
                                 self.lr_c, self.gamma, self.feature)
        self.target_critic2 = SACCritic(self.state_size, self.action_size,
                                 self.lr_c, self.gamma, self.feature)

        # create alpha as a trainable variable
        self.alpha = tf.Variable(alpha, dtype=tf.float32)
        self.alpha_optimizer = tf.keras.optimizers.Adam(lr_a)

        # Buffer for off-policy training
        self.buffer = Buffer(self.buffer_capacity, self.batch_size)

    def policy(self, state):
        # input: numpy array
        # output: numpy array
        if state.ndim < len(self.state_size) + 1:      # single sample
            tf_state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        else:
            tf_state = tf.convert_to_tensor(state, dtype=tf.float32)

        action, log_pi = self.actor.policy(tf_state)   # returns tensors
        return action.numpy(), log_pi.numpy()

    def update_alpha(self, states):
        # input: tensor
        with tf.GradientTape() as tape:
            # sample actions from the policy for the current states
            _, log_pi_a = self.actor.policy(states)
            alpha_loss = - tf.reduce_mean(self.alpha * (log_pi_a + self.target_entropy))
        # outside gradient tape block
        variables = [self.alpha]
        grads = tape.gradient(alpha_loss, variables)
        self.alpha_optimizer.apply_gradients(zip(grads, variables))
        return alpha_loss.numpy()

    def update_target_networks(self):
        target_wts1 = np.array(self.target_critic1.model.get_weights())
        wts1 = np.array(self.critic1.model.get_weights())
        target_wts2 = np.array(self.target_critic2.model.get_weights())
        wts2 = np.array(self.critic2.model.get_weights())

        target_wts1 = self.tau * target_wts1 + (1 - self.tau) * wts1
        target_wts2 = self.tau * target_wts2 + (1 - self.tau) * wts2

        self.target_critic1.model.set_weights(target_wts1)
        self.target_critic2.model.set_weights(target_wts2)

    def replay(self):
        c1_losses, c2_losses, actor_losses, alpha_losses = [], [], [], []
        for epoch in range(self.epochs):
            # sample a minibatch from the replay buffer
            states, actions, rewards, next_states, dones = self.buffer.sample()

            # convert to tensors
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)

            # update Q network weights

            # update Q (Critic) network weights
            c1_loss = self.critic1.train(states, actions, rewards, next_states, dones,
                                         self.actor, self.target_critic1, self.target_critic2,
                                         self.alpha)

            c2_loss = self.critic2.train(states, actions, rewards, next_states, dones,
                                         self.actor, self.target_critic1, self.target_critic2,
                                         self.alpha)
            # update policy networks
            actor_loss = self.actor.train(states, self.alpha, self.critic1, self.critic2)

            # update entropy coefficient
            alpha_loss = self.update_alpha(states)

            # update target network weights
            self.update_target_networks()

            c1_losses.append(c1_loss)
            c2_losses.append(c2_loss)
            actor_losses.append(actor_loss)
            alpha_losses.append(alpha_loss)
        # epoch loop ends here
        mean_c1_loss = np.mean(c1_losses)
        mean_c2_loss = np.mean(c2_losses)
        mean_actor_loss = np.mean(actor_losses)
        mean_alpha_loss = np.mean(alpha_losses)

        return mean_c1_loss, mean_c2_loss, mean_actor_loss, mean_alpha_loss

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
                action, _ = self.policy(state)
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

    def run(self):
        #######################
        # TENSORBOARD SETTINGS
        if self.TB_LOG:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = self.path + 'logs/train/' + current_time
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        ########################################

        if self.filename is None:
            self.filename = 'sac_output.txt'
        self.filename = uniquify(self.path + self.filename)

        if self.val_freq is not None:
            val_scores = deque(maxlen=50)
            val_score = 0

        start = datetime.datetime.now()
        best_score = -np.inf
        ep_lens = []        # episodic length
        ep_scores = []      # All episodic scores
        self.time_steps = 0
        for ep in range(self.training_episodes):

            # initial state
            if self.use_mujoco:
                state = self.env.reset()["observation"]
            else:
                state = self.env.reset()
                state = np.asarray(state, dtype=np.float32) / 255.0

            ep_score = 0       # score for each episode
            t = 0          # length of each episode
            while True:
                action, _ = self.policy(state)
                action2 = np.reshape(action, self.action_size)
                next_state, reward, done, _ = self.env.step(action2)
                if self.use_mujoco:
                    reward = 1 if reward == 0 else 0
                    next_state = next_state["observation"]
                else:
                    next_state = np.asarray(next_state, dtype=np.float32) / 255.0

                # store in replay buffer for off-policy training
                self.buffer.record((state, action, reward, next_state, done))

                state = next_state
                ep_score += reward
                t += 1

                if done:
                    self.time_steps += t
                    break
                # done block ends here
            # end of one episode
            # off-policy training after each season
            c1_loss, c2_loss, actor_loss, alpha_loss = self.replay()
            ep_scores.append(ep_score)
            ep_lens.append(t)
            mean_ep_score = np.mean(ep_scores)
            mean_ep_len = np.mean(ep_lens)

            if ep > 100 and mean_ep_score > best_score:
                self.save_model('actor_wts.h5', 'c1_wts.h5', 'c2_wts.h5', 'c1t_wts.h5', 'c2t_wts.h5')
                print('Episode: {}, Update best score: {}-->{}, Model saved!'.format(ep, best_score, mean_ep_score))
                best_score = mean_ep_score

            if self.val_freq is not None:
                if ep % self.val_freq == 0:
                    print('Episode: {}, Score: {}, Mean score: {}'.format(ep, ep_score, mean_ep_score))
                    val_score = self.validate(self.env)
                    val_scores.append(val_score)
                    mean_val_score = np.mean(val_scores)
                    print('Episode: {}, Validation Score: {}, Mean Validation Score: {}' \
                          .format(ep, val_score, mean_val_score))

            if self.TB_LOG:
                with train_summary_writer.as_default():
                    tf.summary.scalar('1. Episodic Score', ep_score, step=ep)
                    tf.summary.scalar('2. Mean Season Score', mean_ep_score, step=ep)
                    if self.val_freq is not None:
                        tf.summary.scalar('3. Validation Score', val_score, step=ep)
                    tf.summary.scalar('4. Actor Loss', actor_loss, step=ep)
                    tf.summary.scalar('5. Critic Loss', c1_loss, step=ep)
                    tf.summary.scalar('6. Critic Loss', c2_loss, step=ep)
                    tf.summary.scalar('7. Mean Episode Length', mean_ep_len, step=ep)
                    tf.summary.scalar('8. Alpha Loss', alpha_loss, step=ep)

            with open(self.filename, 'a') as file:
                file.write('{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n'
                           .format(ep, self.time_steps, mean_ep_len,
                                   ep_score, mean_ep_score, actor_loss, c1_loss, c2_loss, alpha_loss))

            if self.success_value is not None:
                if best_score > self.success_value:
                    print('Problem is solved in {} episodes with score {}'.format(ep, best_score))
                    print('Mean Episodic score: {}'.format(mean_ep_score))
                    break
        # end of season-loop
        end = datetime.datetime.now()
        print('Time to Completion: {}'.format(end - start))
        self.env.close()
        print('Mean episodic score over {} episodes: {:.2f}'.format(self.training_episodes, np.mean(ep_scores)))

    def save_model(self, path, actor_file, c1_file, c2_file, c1t_file, c2t_file):
        actor_file = path + actor_file
        critic1_file = path + c1_file
        critic2_file = path + c2_file
        target_c1_file = path + c1t_file
        target_c2_file = path + c2t_file
        self.actor.save_weights(actor_file)
        self.critic1.save_weights(critic1_file)
        self.critic2.save_weights(critic2_file)
        self.target_critic1.save_weights(target_c1_file)
        self.target_critic2.save_weights(target_c2_file)

    def load_model(self, path, actor_file, c1_file, c2_file, c1t_file, c2t_file):
        actor_file = path + actor_file
        critic1_file = path + c1_file
        critic2_file = path + c2_file
        target_c1_file = path + c1t_file
        target_c2_file = path + c2t_file
        self.actor.load_weights(actor_file)
        self.critic1.load_weights(critic1_file)
        self.critic2.load_weights(critic2_file)
        self.target_critic1.load_weights(target_c1_file)
        self.target_critic2.load_weights(target_c2_file)

    

class SACAgent2:
    # the environment variable is not a part of this class
    def __init__(self, state_size, action_size, action_upper_bound, epochs,
                  batch_size, buffer_capacity, lr_a=0.0003, lr_c=0.0003,
                 gamma=0.99, tau=0.995, alpha=0.2, use_attention=False,
                  path='./'):
        self.action_size = action_size
        self.state_size = state_size
        self.upper_bound = action_upper_bound
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.epochs = epochs
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.target_entropy = -tf.constant(np.prod(self.action_size), dtype=tf.float32)
        self.gamma = gamma                  # discount factor
        self.tau = tau                      # polyak averaging factor
        self.use_attention = use_attention
        self.path = path

        if len(self.state_size) == 3:
            self.image_input = True     # image input
        elif len(self.state_size) == 1:
            self.image_input = False        # vector input
        else:
            raise ValueError("Input can be a vector or an image")

        # Select a suitable feature extractor
        if self.use_attention and self.image_input:   # attention + image input
            print('Currently Attention handles only image input')
            self.feature = AttentionFeatureNetwork(self.state_size, self.lr_a)
        elif self.use_attention is False and self.image_input is True:  # image input
            print('You have selected an image input')
            self.feature = FeatureNetwork(self.state_size, self.lr_a)
        else:       # non-image input
            print('You have selected a non-image input.')
            self.feature = None

        # create actor for learning policy
        self.actor = SACActor(self.state_size, self.action_size, self.upper_bound,
                              self.lr_a, self.feature)

        # create two critics
        self.critic1 = SACCritic(self.state_size, self.action_size,
                                 self.lr_c, self.gamma, self.feature)
        self.critic2 = SACCritic(self.state_size, self.action_size,
                                 self.lr_c, self.gamma, self.feature)

        # create two target critics
        self.target_critic1 = SACCritic(self.state_size, self.action_size,
                                        self.lr_c, self.gamma, self.feature)
        self.target_critic2 = SACCritic(self.state_size, self.action_size,
                                        self.lr_c, self.gamma, self.feature)

        # create alpha as a trainable variable
        # This is the entropy coefficient required for soft target
        self.alpha = tf.Variable(alpha, dtype=tf.float32)
        self.alpha_optimizer = tf.keras.optimizers.Adam(self.lr_a)

        # Buffer for off-policy training
        self.buffer = Buffer(self.buffer_capacity, self.batch_size)

    def policy(self, state):
        if state.ndim < len(self.state_size) + 1:      # single instance
            tf_state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        else:       # for a batch of samples
            tf_state = tf.convert_to_tensor(state, dtype=tf.float32)

        action, log_pi = self.actor.policy(tf_state) # returns tensors
        return action.numpy(), log_pi.numpy()

    def update_alpha(self, states):
        # input: tensor
        with tf.GradientTape() as tape:
            # sample actions from the policy for the current states
            _, log_pi_a = self.actor.policy(states)
            alpha_loss = tf.reduce_mean(-self.alpha * (log_pi_a + self.target_entropy))
        # outside gradient tape block
        variables = [self.alpha]
        grads = tape.gradient(alpha_loss, variables)
        self.alpha_optimizer.apply_gradients(zip(grads, variables))
        return alpha_loss.numpy()

    def update_target_networks(self):
        target_wts1 = np.array(self.target_critic1.model.get_weights())
        wts1 = np.array(self.critic1.model.get_weights())
        target_wts2 = np.array(self.target_critic2.model.get_weights())
        wts2 = np.array(self.critic2.model.get_weights())

        target_wts1 = self.tau * target_wts1 + (1 - self.tau) * wts1
        target_wts2 = self.tau * target_wts2 + (1 - self.tau) * wts2

        self.target_critic1.model.set_weights(target_wts1)
        self.target_critic2.model.set_weights(target_wts2)

    def replay(self):
        c1_losses, c2_losses, actor_losses, alpha_losses = [], [], [], []
        for epoch in range(self.epochs):
            # sample a minibatch from the replay buffer
            states, actions, rewards, next_states, dones = self.buffer.sample()

            # convert to tensors
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)

            # update Q (Critic) network weights
            c1_loss = self.critic1.train(states, actions, rewards, next_states, dones,
                                         self.actor, self.target_critic1, self.target_critic2,
                                         self.alpha)

            c2_loss = self.critic2.train(states, actions, rewards, next_states, dones,
                                         self.actor, self.target_critic1, self.target_critic2,
                                         self.alpha)

            # update (actor) policy networks
            actor_loss = self.actor.train(states, self.alpha, self.critic1, self.critic2)

            # update entropy coefficient
            alpha_loss = self.update_alpha(states)

            # update target network weights
            self.update_target_networks()

            c1_losses.append(c1_loss)
            c2_losses.append(c2_loss)
            actor_losses.append(actor_loss)
            alpha_losses.append(alpha_loss)
        # epoch loop ends here
        mean_c1_loss = np.mean(c1_losses)
        mean_c2_loss = np.mean(c2_losses)
        mean_actor_loss = np.mean(actor_losses)
        mean_alpha_loss = np.mean(alpha_losses)

        return mean_c1_loss, mean_c2_loss, mean_actor_loss, mean_alpha_loss

    
    
    def save_model(self, save_path):
        actor_file = save_path + 'sac_actor_wts.h5'
        critic1_file = save_path + 'sac_c1_wts.h5'
        critic2_file = save_path + 'sac_c2_wts.h5'
        target_c1_file = save_path + 'sac_c1t_wts.h5'
        target_c2_file = save_path + 'sac_c2t_wts.h5'
        self.actor.save_weights(actor_file)
        self.critic1.save_weights(critic1_file)
        self.critic2.save_weights(critic2_file)
        self.target_critic1.save_weights(target_c1_file)
        self.target_critic2.save_weights(target_c2_file)

    def load_model(self, load_path):
        actor_file = load_path + 'sac_actor_wts.h5'
        critic1_file = load_path + 'sac_c1_wts.h5'
        critic2_file = load_path + 'sac_c2_wts.h5'
        target_c1_file = load_path + 'sac_c1t_wts.h5'
        target_c2_file = load_path + 'sac_c2t_wts.h5'
        self.actor.load_weights(actor_file)
        self.critic1.load_weights(critic1_file)
        self.critic2.load_weights(critic2_file)
        self.target_critic1.load_weights(target_c1_file)
        self.target_critic2.load_weights(target_c2_file)



