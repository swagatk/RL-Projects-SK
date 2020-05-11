import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Add, Input
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import gym
from collections import deque
import numpy as np
import random  
import matplotlib.pyplot as plt

#import pdb # for debugging

# setting seed for reproducibility of results
random.seed(2212)
np.random.seed(2212)
tf.set_random_seed(2212)
   
##########################
# Actor Model
######################
class Actor:
  def __init__(self, sess, state_size, action_size):

    self.sess = sess
    K.set_session(sess)
    self.action_size = action_size
    self.state_size = state_size
    self.state_input, self.output, self.model = self.create_model()
    model_weights = self.model.trainable_weights

    # placeholder for critic gradients wrt action inputs: dC/dA
    self.actor_critic_grads = tf.placeholder(tf.float32, [None, action_size])

    # add small constants inside log to avoid log(0) = -infinity
    log_prob = tf.math.log(self.output + 10e-10)

    # Multiply log by -1 to convert minimization problem into a maximization problem
    neg_log_prob = tf.multiply(log_prob, -1) 

    # Calculate and update the weights of the model to optimize the actor
    self.actor_grads = tf.gradients(neg_log_prob, model_weights, \
                                    self.actor_critic_grads)
    grads = zip(self.actor_grads, model_weights)
    self.optimize = tf.train.AdamOptimizer(0.001).apply_gradients(grads)

  def create_model(self):
    state_input = Input(shape=self.state_size)
    state_h1 = Dense(24, activation='relu')(state_input)
    state_h2 = Dense(24, activation='relu')(state_h1)
    output = Dense(self.action_size, activation='softmax')(state_h2)
    model = Model(inputs=state_input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001))
    return state_input, output, model
  
  def train(self, critic_gradients_val, x_states):
    self.sess.run(self.optimize, feed_dict={
        self.state_input: x_states,
        self.actor_critic_grads: critic_gradients_val
    })

###################
# Critic Model
#####################
class Critic:
  def __init__(self, sess, state_size, action_size):
    K.set_session(sess)
    self.sess = sess
    self.action_size = action_size
    self.state_size = state_size
    self.state_input, self.action_input, self.output, self.model = \
                          self.create_model()

    # dC/dA                      
    self.critic_gradients = tf.gradients(self.output, self.action_input)

  def create_model(self):
    state_input = Input(shape=self.state_size)
    state_h1 = Dense(24, activation='relu')(state_input)
    state_h2 = Dense(24, activation='relu')(state_h1)

    action_input = Input(shape=(self.action_size, ))
    action_h1 = Dense(24, activation='relu')(action_input)
    action_h2 = Dense(24, activation='relu')(action_h1)

    state_action = Add()([state_h2, action_h2])
    state_action_h1 = Dense(24, activation='relu')(state_action)
    output = Dense(1, activation='linear')(state_action_h1)

    model = Model(inputs=[state_input,action_input], outputs=output)
    model.compile(loss='mse', optimizer=Adam(lr=0.005))
    return state_input, action_input, output, model

  def get_critic_gradients(self, x_states, x_actions):
    critic_gradients_val = self.sess.run(self.critic_gradients, feed_dict={
        self.state_input: x_states,
        self.action_input: x_actions
    })
    return critic_gradients_val[0]

####################
class ACAgent:
    def __init__(self, sess, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = 100000
        self.min_replay_size = 1000
        self.batch_size = 32
        self.epsilon = 1
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.001
        self.discount_factor = 0.99

        K.set_session(sess)

        # Actor Model
        self.actor = Actor(sess, self.state_size, self.action_size)

        # Critic Model
        self.critic = Critic(sess, self.state_size, self.action_size)

        self.memory = deque(maxlen=self.memory_size)

        sess.run(tf.initialize_all_variables())

    def get_action(self, state):
        if self.epsilon > self.min_epsilon and \
           len(self.memory) >=  self.min_replay_size:
            self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_epsilon)
            
        if np.random.uniform(0,1) < self.epsilon: # explore
            action = [0] * action_size
            action[np.random.randint(0, action_size)] = 1
            action = np.array(action, dtype=np.float32)
        else: # exploit
            action = self.actor.model.predict(np.expand_dims(state, axis=0))[0]
        return action

    def add_experience(self, state, action, reward, \
                       next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):

        if len(self.memory) < self.min_replay_size:
            return

        minibatch =  random.sample(self.memory, self.batch_size)

        x_states = []
        x_actions = []
        y = []
        for sample in minibatch: 
            state, action, reward, next_state, done = sample
            next_action = self.actor.model.predict(\
                        np.expand_dims(next_state,axis=0))

            if done:
                reward = -reward
                # reward = -100
            else:
                next_reward = self.critic.model.predict(\
                            [np.expand_dims(next_state, axis=0),\
                             next_action])[0][0]
                reward = reward + self.discount_factor * next_reward

            x_states.append(state)
            x_actions.append(action)
            y.append(reward)
            #end of for-loop for samples

        x_states = np.array(x_states)
        x_actions = np.array(x_actions)
        x = [x_states, x_actions]
        y = np.array(y)
        y = np.expand_dims(y, axis=1)

        # train the critic model
        # Model error = \gamma * Q(s', a') - Q(s,a)
        self.critic.model.fit(x, y, self.batch_size, verbose = 0)


        x_actions_new = []
        for sample in minibatch:
            x_actions_new.append(self.actor.model.predict(np.expand_dims(sample[0], axis=0))[0])
        x_actions_new = np.array(x_actions_new)

        critic_gradients_val = self.critic.get_critic_gradients(x_states, x_actions)
        
        # train the actor model
        self.actor.train(critic_gradients_val, x_states)

#################################
if __name__ == '__main__':
    sess = tf.Session()
    K.set_session(sess)

    env = gym.make('CartPole-v0')
    env.seed(0)  # for reproducing results

    action_size = env.action_space.n
    state_size = env.observation_space.shape

    agent = ACAgent(sess, state_size, action_size)

    max_episodes = 3000

    #pdb.set_trace()

    file = open('./data1/cp_ac_100k-2.txt','w')

    max_reward = 0
    scores = []
    avgscores = []
    avg100scores = []
    last100scores = deque(maxlen=100)
    for e in range(max_episodes):
        done = False
        state = env.reset()
        t = 0
        while not done:

            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(np.argmax(action))

            # add experiences to replay memory
            agent.add_experience(state, action, reward, next_state, done)

            state = next_state
            t += 1

            agent.train()

            if done:
                scores.append(t)
                last100scores.append(t)
                avgscores.append(np.mean(scores))
                avg100scores.append(np.mean(last100scores))
                file.write('{}\t{}\t{}\t{}\n'.format( \
                            e, t, np.mean(scores), \
                            np.mean(last100scores)))
                break

            # end of while loop

        if t > (env.spec.max_episode_steps - 5) and t > max_reward:
            agent.actor.model.save_weights('actor'+str(t)+".h5")
            agent.critic.model.save_weights('critic'+str(t)+".h5")
            max_reward = max(max_reward, t)


        if e % 100 == 0:
            print('Episode:{}, score:{}, avg score:{:.2f}, avg100scores: {:.2f}, \
                    epsilon:{:.3f}'.format(e, t, np.mean(scores), \
                                           np.mean(last100scores),
                                           agent.epsilon))

        if np.mean(last100scores) > env.spec.max_episode_steps-5:
            print('The problem is solved in {} steps. Exiting ...'.format(e))
            break
        # for loop ends here   

    file.close()
    # plotting
    plt.plot(scores, label='Scores')
    plt.plot(avgscores, label='Avg Scores')
    plt.plot(avg100scores, label='Avg100Scores')
    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.grid()
    plt.legend(loc='upper left')
    plt.savefig('./img/cp_ac_seed0.png')
    plt.show()


