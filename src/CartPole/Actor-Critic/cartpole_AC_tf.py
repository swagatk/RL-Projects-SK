"""
Actor-Critic using TD-error as the advantage

"""
import numpy as np
import tensorflow as tf
import gym
from collections import deque
import matplotlib.pyplot as plt

np.random.seed(2)
tf.set_random_seed(2)

# Hyperparameters

MAX_EPISODES = 3000
OUTPUT_GRAPH = False
MAX_EP_STEPS = 1000 # maximum time step in one episode
GAMMA = 0.9
LR_A = 0.001
LR_C = 0.01

env = gym.make('CartPole-v0')
env.seed(1)  # required for reproducibility

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

class Actor:
    def __init__(self, sess, state_size, action_size, lr=0.001):
        self.sess = sess

        self.state = tf.placeholder(tf.float32, \
                    shape = [1, state_size], name='state')
        self.action = tf.placeholder(tf.int32, \
                             shape = None, name='action')
        self.td_error = tf.placeholder(tf.float32, \
                             shape = None, name='td_error')

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs = self.state,
                units = 20,
                activation = tf.nn.relu,
                kernel_initializer = \
                            tf.random_normal_initializer(0., .1), 
                bias_initializer = tf.constant_initializer(0.1),
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs = l1,
                units = action_size, 
                activation = tf.nn.softmax, 
                kernel_initializer = \
                            tf.random_normal_initializer(0., .1),
                bias_initializer = tf.constant_initializer(0.1),
                name = 'acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.action])
            # advantage (TD-Error) guided loss
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)

        # minimize(-exp_v) = maximize(exp_v)
        with tf.variable_scope('train'):
            self.train_op = \
                    tf.train.AdamOptimizer(lr).minimize(-self.exp_v)
                                
        
    def train(self, state, action, td_error):
        state = state[np.newaxis, :] # create a row vector
        feed_dict = {self.state: state, self.action: action,\
                                    self.td_error: td_error}
        _,exp_v = self.sess.run([self.train_op, self.exp_v], \
                                                feed_dict)
        return exp_v

    def choose_action(self, state):
        state = state[np.newaxis, :] # row vector
        #get action probabilities for all states
        probs = self.sess.run(self.acts_prob, feed_dict={self.state: state})
        return np.random.choice(np.arange(probs.shape[1]),\
                                p=probs.ravel()) # returns an int

#####################
class Critic:
    def __init__(self, sess, state_size, lr = 0.01):
        self.sess = sess
        self.state = tf.placeholder(tf.float32, shape=[1, state_size],\
                                                 name='state')
        self.v_next = tf.placeholder(tf.float32, shape=[1,1], name='v_next')
        self.reward = tf.placeholder(tf.float32, shape=None, \
                                                name='reward')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs = self.state,
                units = 20,
                activation = tf.nn.relu,
                kernel_initializer = \
                        tf.random_normal_initializer(0., .1), 
                bias_initializer = tf.constant_initializer(0.1),
                name = 'l1'
            )

            self.v = tf.layers.dense(
                inputs = l1, 
                units = 1, # single output
                activation = 'linear', # None
                kernel_initializer = \
                    tf.random_normal_initializer(0., .1),
                bias_initializer = tf.constant_initializer(0.1),
                name = 'V'
            )

            with tf.variable_scope('squared_TD_error'):
                # TD error = (r + gamma * V_next) - V
                self.td_error = self.reward + GAMMA * self.v_next - self.v
                self.loss = tf.square(self.td_error) 
                
            # define optimizer for critic training
            with tf.variable_scope('train'):
                self.train_op = \
                        tf.train.AdamOptimizer(lr).minimize(self.loss)

    
    def train(self, state, reward, state_next):
        state = state[np.newaxis, :]
        state_next = state_next[np.newaxis, :]
        v_next = self.sess.run(self.v, \
                               feed_dict={self.state: state_next})
        td_error, _ = self.sess.run([self.td_error,self.train_op],
                                    feed_dict={self.state: state,
                                               self.v_next: v_next,
                                               self.reward: reward
                                              })
        return td_error

############################
sess = tf.Session()

actor = Actor(sess, state_size, action_size, lr=LR_A)
critic = Critic(sess, state_size, lr=LR_C)

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

last100scores = deque(maxlen=100)
scores = []
avgscores = []
avg100scores = []
for e in range(MAX_EPISODES):
    state = env.reset()
    t = 0
    track_r = []
    done = False
    while not done:

        action = actor.choose_action(state)
        state_next, reward, done, info = env.step(action)

        if done: reward = -100

        track_r.append(reward)

        # gradient = grad [ r+ gamma * V(s_next) - V ]
        td_error = critic.train(state, reward, state_next)

        # gradient = grad [logPi(s,a) * td_error]
        actor.train(state, action, td_error)

        state = state_next
        t += 1

        if done: # or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)
            scores.append(t)
            last100scores.append(t)
            avgscores.append(np.mean(scores))
            avg100scores.append(np.mean(last100scores))

            print('episode:{}, reward:{}, avgscores:{:.2f}, \
                            avg100scores:{:.2f}'.format(e, t, \
                                np.mean(scores), np.mean(last100scores)))
            break

        # end of while loop
    if np.mean(last100scores) > env.spec.max_episode_steps-5:
        print('Problem is solved in {} episodes. Exiting ..'.format(e))
        break
    # end of for episode loop

# plot
plt.plot(scores, 'r-', label='Scores')
plt.plot(avgscores, 'b-', label='Avg Scores')
plt.plot(avg100scores, 'g-', label='Avg last 100 scores')
plt.xlabel('Episodes')
plt.ylabel('Scores')
plt.legend(loc='lower right')
plt.title('AC-TD-Error-as-Advantage (env.seed(1))')
plt.grid()
plt.savefig('./img/cp_ac_tf_seed1.png')
plt.show()



        

