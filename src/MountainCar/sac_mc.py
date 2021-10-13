import tensorflow as tf
from tensorflow.python.keras.backend import learning_phase
import tensorflow_probability as tfp
import numpy as np
import sys
import os
import gym
import matplotlib.pyplot as plt
from datetime import datetime

# Add the current folder to python's import path
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))

from common.buffer import Buffer


class SACActorMC():
    def __init__(self, state_size, action_size, 
                        upper_bound, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.upper_bound = upper_bound
        self.lr = learning_rate
        self.model = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        s_inp = tf.keras.layers.Input(shape=self.state_size)
        f = tf.keras.layers.Dense(32, activation='relu')(s_inp)
        f = tf.keras.layers.Dense(32, activation='relu')(f)
        mu =  tf.keras.layers.Dense(self.action_size[0])(f)
        log_sig = tf.keras.layers.Dense(self.action_size[0])(f)

        model = tf.keras.Model(s_inp, [mu, log_sig], name='actor')
        model.summary()
        return model

    def __call__(self, state):
        # input: tensor, output: tensor
        mean, log_sigma = self.model(state)
        std = tf.exp(log_sigma)
        return mean, std

    def policy(self, state):
        # input: tensor, output: tensor
        mean, std = self.__call__(state)

        pi = tfp.distributions.Normal(mean, std)
        action_ = pi.sample()
        log_pi_ = pi.log_prob(action_)

        action = tf.clip_by_value(action_, -self.upper_bound, self.upper_bound)

        log_pi_a = log_pi_ - tf.reduce_sum(tf.math.log(1 - action ** 2 + 1e-16), axis=1, keepdims=True)

        return action, log_pi_a

    def train(self, states, alpha, critic1, critic2):
        with tf.GradientTape() as tape:
            actions, log_pi_a = self.policy(states)
            q1 = critic1(states, actions)
            q2 = critic2(states, actions)
            min_q = tf.minimum(q1, q2)
            soft_q = min_q - alpha * log_pi_a
            actor_loss = -tf.reduce_mean(soft_q)
            actor_wts = self.model.trainable_variables
        # outside gradient tape block
        actor_grads = tape.gradient(actor_loss, actor_wts)
        self.optimizer.apply_gradients(zip(actor_grads, actor_wts))
        return actor_loss.numpy()

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


class SACCriticMC():
    def __init__(self, state_size, action_size,
                learning_rate, gamma):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma

        self.model = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        s_inp = tf.keras.layers.Input(shape=self.state_size)
        a_inp = tf.keras.layers.Input(shape=self.action_size)
        c_inp = tf.keras.layers.Concatenate()([s_inp, a_inp])
        f = tf.keras.layers.Dense(32, activation='relu')(c_inp)
        f = tf.keras.layers.Dense(32, activation='relu')(f)
        q = tf.keras.layers.Dense(1)(f)
        model = tf.keras.Model([s_inp, a_inp], q, name='critic')
        model.summary()
        return model

    def __call__(self, state, action):
        # input: tensors, output: tensors
        q_value = self.model([state, action])
        return q_value

    def train(self, states, actions, rewards, next_states, dones, actor,
              target_critic1, target_critic2, alpha):
        with tf.GradientTape() as tape:
            # Get Q estimates using actions from replay buffer
            q_values = self.model([states, actions])

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

            y = tf.stop_gradient(rewards + self.gamma * (1 - dones) * soft_q_target)
            critic_loss = tf.math.reduce_mean(tf.square(y - q_values))
            critic_wts = self.model.trainable_variables
        # outside gradient tape
        critic_grad = tape.gradient(critic_loss, critic_wts)
        self.optimizer.apply_gradients(zip(critic_grad, critic_wts))
        return critic_loss.numpy()

    def train2(self, state_batch, action_batch, y):
        with tf.GradientTape() as tape:
            q_values = self.model([state_batch, action_batch])
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
    def __init__(self, state_size, action_size, upper_bound,
                buffer_capacity=100000, batch_size=128,
                epochs=50,
                learning_rate=0.0003, alpha=0.2, gamma=0.99,
                polyak=0.995):

        self.state_size = state_size
        self.action_size = action_size
        self.upper_bound = upper_bound
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.polyak = polyak 
        self.lr = learning_rate
        self.epochs = epochs
        self.target_entropy = -tf.constant(np.prod(self.action_size), dtype=tf.float32)
        

        self.actor = SACActorMC(self.state_size, self.action_size,
                    self.upper_bound, self.lr)

        self.critic1 = SACCriticMC(self.state_size, self.action_size,
                    self.lr, self.gamma)
        self.critic2 = SACCriticMC(self.state_size, self.action_size,
                    self.lr, self.gamma)

        self.target_critic1 = SACCriticMC(self.state_size, self.action_size,
                    self.lr, self.gamma)
        self.target_critic2 = SACCriticMC(self.state_size, self.action_size,
                    self.lr, self.gamma)

        self.alpha = tf.Variable(alpha, dtype=tf.float32)
        self.alpha_optimizer = tf.keras.optimizers.Adam(self.lr)

        self.buffer = Buffer(self.buffer_capacity, self.batch_size)

    def sample_action(self, state):
        # input: numpy array, output: ndarray
        if state.ndim < len(self.state_size) + 1:      # single sample
            tf_state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), axis=0)
        else:
            tf_state = tf.convert_to_tensor(state, dtype=tf.float32)

        action, log_pi = self.actor.policy(tf_state)   # returns tensors
        return action[0].numpy(), log_pi[0].numpy()

    def update_alpha(self, states):
        # input: tensor
        with tf.GradientTape() as tape:
            # sample actions from the policy for the current states
            _, log_pi_a = self.actor.policy(states)
            alpha_loss = tf.reduce_mean(-self.alpha * (log_pi_a + self.target_entropy))
            variables = [self.alpha]
        # outside gradient tape block
        grads = tape.gradient(alpha_loss, variables)
        self.alpha_optimizer.apply_gradients(zip(grads, variables))
        return alpha_loss.numpy()

    def update_target_networks(self):
        for theta_target, theta in zip(self.target_critic1.model.trainable_variables,
                    self.critic1.model.trainable_variables):
            theta_target = self.polyak * theta_target + (1 - self.polyak) * theta
        
        for theta_target, theta in zip(self.target_critic2.model.trainable_variables,
                    self.critic2.model.trainable_variables):
            theta_target = self.polyak * theta_target + (1 - self.polyak) * theta

    def update_q_networks(self, states, actions, rewards, next_states, dones):

        pi_a, log_pi_a = self.actor.policy(next_states) # output: tensor

        q1_target = self.target_critic1(next_states, pi_a) # input: tensor
        q2_target = self.critic2(next_states, pi_a)

        min_q_target = tf.minimum(q1_target, q2_target)

        soft_q_target = min_q_target  - self.alpha * log_pi_a  

        y = rewards + self.gamma * (1 - dones) * soft_q_target

        c1_loss = self.critic1.train2(states, actions, y)
        c2_loss = self.critic2.train2(states, actions, y)

        mean_c_loss = np.mean([c1_loss, c2_loss])
        return mean_c_loss 

    def train(self, CRIT_T2=True):
        critic_losses, actor_losses, alpha_losses = [], [], []
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
            if CRIT_T2:
                critic_loss = self.update_q_networks(states, actions, rewards, next_states, dones)
            else:
                c1_loss = self.critic1.train(states, actions, rewards, next_states, dones,
                                            self.actor, self.target_critic1, self.target_critic2,
                                            self.alpha)

                c2_loss = self.critic2.train(states, actions, rewards, next_states, dones,
                                            self.actor, self.target_critic1, self.target_critic2,
                                            self.alpha)

                critic_loss = np.mean([c1_loss, c2_loss])
                                    
            # update policy networks
            actor_loss = self.actor.train(states, self.alpha, self.critic1, self.critic2)

            # update entropy coefficient
            alpha_loss = self.update_alpha(states)

            # update target network weights
            self.update_target_networks()

            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)
            alpha_losses.append(alpha_loss)
        # epoch loop ends here
        mean_critic_loss = np.mean(critic_losses)
        mean_actor_loss = np.mean(actor_losses)
        mean_alpha_loss = np.mean(alpha_losses)

        return mean_actor_loss, mean_critic_loss, mean_alpha_loss

if __name__ == '__main__':


    env = gym.make('MountainCarContinuous-v0')
    env.seed(42)
    action_size = env.action_space.shape
    state_size = env.observation_space.shape
    action_upper_bound = env.action_space.high

    agent = SACAgent(state_size, action_size, 
            action_upper_bound, learning_rate=0.001)

    
    # repeat until convergence
    max_episodes = 600
    global_step = 0
    episode = 0
    ep_rewards = []
    avg_ep_rewards = []
    critic_losses, actor_losses, alpha_losses = [], [], []
    avg_c_loss, avg_a_loss, avg_alpha_loss = [], [], []
    start = datetime.now()
    for ep in range(max_episodes):
        state = env.reset()
        step = 0
        ep_reward = 0
        done = False
        while not done:
            action, _ = agent.sample_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.buffer.record([state, action, reward, next_state, done])

            ep_reward += reward
            state = next_state
            step += 1
            global_step += 1

        # train after each episode
        ep_rewards.append(ep_reward)
        avg_ep_rewards.append(np.mean(ep_rewards[-100:]))
        a_loss, c_loss, alpha_loss = agent.train()

        actor_losses.append(a_loss)
        critic_losses.append(c_loss)
        alpha_losses.append(alpha_loss)
        avg_a_loss.append(np.mean(actor_losses))
        avg_c_loss.append(np.mean(critic_losses))
        avg_alpha_loss.append(np.mean(alpha_losses))

        episode += 1
        with open('sac_output.txt', 'a') as file:
            file.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(episode,
                    ep_reward, np.mean(ep_rewards),
                    np.mean(actor_losses), np.mean(critic_losses),
                    np.mean(alpha_losses)))

        print('Episode: {}, reward: {}, mean_reward:{}'.format(episode,
                        ep_reward, np.mean(ep_rewards[-100:])))

        if np.mean(ep_rewards[-100:]) > -10:
            print('Problem is solved in {} episodes'.format(episode))
            break
    
    end = datetime.now()
    print('Global steps:', global_step)
    print('Average Episodic Score:', np.mean(ep_rewards))
    print('Training time:', (end-start))
    
    # plot
    fig, axes = plt.subplots(2,2)
    axes[0, 0].plot(avg_ep_rewards) 
    axes[0, 0].set_ylabel('Avg Episodic Rewards')
    axes[0, 1].plot(avg_alpha_loss)
    axes[0, 1].set_ylabel('Avg Alpha Losses')
    axes[1, 0].plot(avg_a_loss)
    axes[1, 0].set_ylabel('Avg Actor Losses')
    axes[1, 1].plot(avg_c_loss)
    axes[1, 1].set_ylabel('Avg Critic Losses')
    axes[1, 0].set_xlabel('Episodes')
    axes[1, 1].set_xlabel('Episodes')
    fig.tight_layout()
    plt.savefig('sac_mc_response.png')
    plt.show()

        
