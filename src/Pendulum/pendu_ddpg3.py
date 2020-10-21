"""
Solving Pendulum using DDPG
Uses TF-Keras

source:
    https://github.com/piotrplata/keras-ddpg/blob/master/ac_pendulum.py

- Average of last 100 scores is around -160
"""
import os
import gym
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Concatenate
from keras.optimizers import Adam
import keras.backend as K
import random
from collections import deque
import matplotlib.pyplot as plt

random.seed(2212)
np.random.seed(2212)
tf.set_random_seed(2212)

def stack_samples(samples):

    array = np.array(samples)

    states = np.stack(array[:,0]).reshape((array.shape[0], -1))
    actions = np.stack(array[:,1]).reshape((array.shape[0], -1))
    rewards = np.stack(array[:,2]).reshape((array.shape[0], -1))
    next_states = np.stack(array[:,3]).reshape((array.shape[0], -1))
    dones = np.stack(array[:,4]).reshape((array.shape[0], -1))

    return states, actions, rewards, next_states, dones 


class ActorCritic:
    def __init__(self, sess, state_size, action_size, action_bound):
        self.sess = sess

        self.learning_rate = 0.0001
        self.epsilon = .9
        self.epsilon_decay = .99995
        self.gamma = 0.9
        self.tau = 0.01

        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound 


        #########################################
        # Actor Model
        # Chain Rule: 
        # de/dA = de/dC * dC/dA where e is error
        ########################################

        self.memory = deque(maxlen = 4000)
        self.actor_state_input, self.actor_model = \
                                    self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()


        # dC/dA will be fed from the critic
        self.actor_critic_grad = tf.placeholder(tf.float32, 
                                               [None, self.action_size])

        actor_model_weights = self.actor_model.trainable_weights

        # total grad = dA/dW * (-dC/dA)
        self.actor_grads = tf.gradients(self.actor_model.output,
                                        actor_model_weights,
                                        -self.actor_critic_grad)

        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = \
            tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        ###############################################
        # Critic Model
        ##########################

        # computes the Q(s,a)
        self.critic_state_input, self.critic_action_input, \
                self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        # dC/dA - This will be fed to the actor
        self.critic_grads = tf.gradients(self.critic_model.output,
                                         self.critic_action_input)

        # Initialize tf gradient calculations
        self.sess.run(tf.initialize_all_variables())

        ############
        # Model Definitions
        ###################

    def create_actor_model(self):
        state_input = Input(shape=(self.state_size,))
        h1 = Dense(500, activation='relu')(state_input)
        h2 = Dense(1000, activation='relu')(h1)
        h3 = Dense(500, activation='relu')(h2)
        output = Dense(self.action_size, activation='tanh')(h3)

        model = Model(input=state_input, output=output)
        model.compile(loss='mse', optimizer=Adam(lr=0.0001))
        model.summary()
        return state_input, model


    def create_critic_model(self):
        state_input = Input(shape=(self.state_size, ))
        state_h1 = Dense(500, activation='relu')(state_input)
        state_h2 = Dense(1000)(state_h1)

        action_input = Input(shape=(self.action_size,))
        action_h1 = Dense(500)(action_input)

        merged = Concatenate()([state_h2, action_h1])
        merged_h1 = Dense(500, activation='relu')(merged)
        output = Dense(1, activation='linear')(merged_h1)
        model = Model(input=[state_input, action_input], output=output)
        model.compile(loss='mse', optimizer=Adam(lr=0.0001))
        model.summary()
        return state_input, action_input, model

    ###############
    # Model Training
    ############################

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])


    def _train_actor(self, samples):

        states, actions, rewards, new_states, _ = \
                                        stack_samples(samples)
        predicted_actions = self.actor_model.predict(states)
        grads = self.sess.run(self.critic_grads, feed_dict={
            self.critic_state_input: states,
            self.critic_action_input: predicted_actions 
        })[0]

        self.sess.run(self.optimize, feed_dict={
            self.actor_state_input: states,
            self.actor_critic_grad: grads
        })


    def _train_critic(self, samples):

        states, actions, rewards, new_states, dones = \
                stack_samples(samples)
        target_actions = self.target_actor_model.predict(new_states)
        future_rewards = self.target_critic_model.predict([new_states,
                                                          target_actions])

        rewards += self.gamma * future_rewards * (1 - dones)

        evaluation = self.critic_model.fit([states, actions], rewards,
                                          verbose=0)

    def train(self, batch_size=256):
        if len(self.memory) < batch_size:
            return
        rewards = []
        samples = random.sample(self.memory, batch_size)

        self.samples = samples
        self._train_critic(samples)
        self._train_actor(samples)

    ##########################
    # Update Target Models
    ####################

    def _update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i] * \
                    self.tau + actor_target_weights[i] * (1 - self.tau)
        self.target_actor_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights() 

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i] * \
                    self.tau + critic_target_weights[i] * (1 - self.tau)
        self.target_critic_model.set_weights(critic_target_weights)


    def update_target(self):
        self._update_critic_target()
        self._update_actor_target()

    ###################################
    # Model Predictions
    ###########

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.actor_model.predict(state) * 2 + \
                                np.random.normal() 
        else:
            return self.actor_model.predict(state) * 2


    def save_model(self, path):
        self.actor_model.save_weights(path + '_actor.h5')
        self.critic_model.save_weights(path + '_critic.h5')

    def load_model(self, actor_path, critic_path):
        self.actor_model.load_weights(actor_path)
        self.critic_model.load_weights(critic_path)

###########################
if __name__ == '__main__':
    sess = tf.Session()
    K.set_session(sess)

    env = gym.make('Pendulum-v0')
    env.seed(0)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_bound = env.action_space.high 

    # Create Actor-Critic Agent
    agent = ActorCritic(sess, state_size, action_size, action_bound)

    num_trials = 3000
    trial_len = 200
    print('Maximum Steps per episode:{}'\
          .format(env.spec.max_episode_steps))

    file = open('data_log.txt', 'w')

    max_reward = -1000

    last100scores = deque(maxlen=100)
    scores = []
    avgscores = []
    avg100scores = []
    for i in range(num_trials):
        state = env.reset().reshape((1, state_size))

        reward_sum = 0
        for j in range(trial_len):
            action = agent.act(state)
            action = action.reshape((1, action_size))
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape((1, state_size))

            agent.remember(state, action, reward, next_state, done)
            
            if (j % 5 == 0):
                agent.train()
                agent.update_target()

            state = next_state
            reward_sum += reward[0]

            if j == (trial_len -1):
                done = True
                scores.append(reward_sum)
                avgscores.append(np.mean(scores))
                last100scores.append(reward_sum)
                avg100scores.append(np.mean(last100scores))
                file.write('{}\t{}\t{}\t{}\n'.format(i, 
                                                     reward_sum,
                                                    np.mean(scores),
                                                    np.mean(last100scores)))

        if (i % 100 == 0):
            print('trial: {}, scores: {}, avgscores: {}, \
                  avg100scores:{}'.format(i, reward_sum,
                                         np.mean(scores),
                                         np.mean(last100scores)))
        if reward_sum > max_reward:
            max_reward = reward_sum
            exp_dir = './ddpg/models/'

            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)

            export_path = '{}_episode_{}'.format(exp_dir, i)
            agent.save_model(export_path)

env.close()

##############
## Plot
plt.plot(scores, 'r-', label='Scores')
plt.plot(avgscores, 'g-', label='Avg Scores')
plt.plot(avg100scores, 'b-', label='Avg of last 100 Scores')
plt.xlabel('Trials')
plt.ylabel('Scores')
plt.title('DDPG-Pendulum-v0')
plt.grid()
plt.legend(loc='lower right')
plt.savefig('./pendu_ddpg3.png')
plt.show()


        



            


        
        



        


        







    





