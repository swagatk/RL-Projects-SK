"""
Deep Deterministic Policy Gradient for solving the Pendulum Problem 
Tensorflow 1.x Code
Output: Average reward over 100 episodes =
Training happens over 10000 episodes
"""
import tensorflow as tf
import numpy as np
import gym
import time
from collections import deque
from datetime import datetime 
import matplotlib.pyplot as plt

np.random.seed(1)
tf.set_random_seed(1)

#####################################
# Hyperparameters

MAX_EPISODES = 10000
MAX_EP_STEPS = 200

LR_A = 0.001
LR_C = 0.001
GAMMA = 0.9

REPLACEMENT =[
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0] # you can try different target replacement strategies
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
OUTPUT_GRAPH = True
ENV_NAME = 'Pendulum-v0'
############################################
## ACTOR
##############################
class Actor:
    def __init__(self, sess, action_size, action_bound, learning_rate,
                 replacement):
        self.sess = sess
        self.action_size = action_size
        self.action_bound = action_bound 
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input - state and output is action
            self.a = self._build_net(S, scope='eval_net',
                                          trainable=True)

            self.a_ = self._build_net(S_, scope='target_net',
                                               trainable=False)
        self.e_params = \
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                          scope='Actor/eval_net')
        self.t_params =  \
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                          scope='Actor/target_net')

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t,e in
                                 zip(self.t_params, self.e_params)]
        else:
            self.soft_replace = [tf.assign(t,
                                           (1-self.replacement['tau'])
                                          * t +
                                           self.replacement['tau'] *
                                           e) for t,e in
                                 zip(self.t_params, self.e_params)]


    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            net = tf.layers.dense(s, 30, activation = tf.nn.relu,
                                 kernel_initializer=init_w,
                                  bias_initializer=init_b,
                                 name='l1')
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.action_size,
                                          activation=tf.nn.tanh,
                                          kernel_initializer=init_w,
                                          bias_initializer = init_b,
                                          name='a',
                                          trainable=trainable
                                         )
                scaled_action = tf.multiply(actions,
                                             self.action_bound,
                                             name='scaled_a')
        return scaled_action


    def train(self, state):
        self.sess.run(self.train_op, feed_dict={S: state})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter %  \
                self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1


    def choose_action(self, state):
        state = state[np.newaxis, :]
        return self.sess.run(self.a, feed_dict={S: state})[0]


    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            # ys = policy
            # xs = policy parameters
            # a_grads = grad of policy to maximize Q
            # tf.gradients will calculate dys/dxs 
            # so this, dq/da * dq/dparams
            self.policy_grads = tf.gradients(ys=self.a,
                                             xs=self.e_params, 
                                            grad_ys = a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(-self.lr) # ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads,
                                                   self.e_params))


########################################################
### Critic
#########################################
class Critic:
    def __init__(self, sess, state_size, action_size, learning_rate,
                 gamma, replacement, a, a_):

        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement

        with tf.variable_scope('Critic'):
            # input (s,a) and output: q
            self.a = tf.stop_gradient(a)
            self.q = self._build_net(S, self.a, 'eval_net',
                                     trainable=True)

            # input (s', a'), output: q_  for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', 
                                      trainable = False)

            self.e_params = \
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                        scope='Critic/eval_net')
            self.t_params = \
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss =  \
                tf.reduce_mean(tf.squared_difference(self.target_q,
                                                    self.q))
        with tf.variable_scope('C_train'):
            self.train_op =  \
                tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, self.a)[0]

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in
                                        zip(self.t_params,
                                            self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t,
                                                (1-self.replacement['tau'])
                                                * t +
                                                self.replacement['tau']*
                                                e) for t,e in
                                        zip(self.t_params,
                                            self.e_params)]


    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n_l1 = 30
                w1_s = tf.get_variable('w1_s', [self.state_size,
                                                n_l1], 
                                      initializer=init_w,
                                      trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.action_size,
                                                n_l1],
                                       initializer=init_w,
                                       trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], 
                                     initializer=init_b, 
                                     trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + \
                                 tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w,
                                   bias_initializer=init_b,
                                    trainable=trainable)
        return q

    def train(self, state, action, reward, state_next):
        self.sess.run(self.train_op, feed_dict={S:state, self.a:
                                                action, R:reward, S_:
                                               state_next})
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
        else:
            if self.t_replace_counter % \
               self.replacement['rep_iter_c']== 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1

########################################
class Memory:
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transitions(self, s, a, r, s_):
        transition = np.hstack((s,a,r,s_))
        index = self.pointer % self.capacity 
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been\
        filled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices,:]


##############################
### Main
##########

env = gym.make(ENV_NAME)
env.seed(1)
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
action_bound = env.action_space.high

print('Max steps per episode: {}'.format(env.spec.max_episode_steps))


with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, state_size], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, shape=[None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, state_size], name='s_')

sess = tf.Session()

actor = Actor(sess, action_size, action_bound, LR_A, REPLACEMENT)
critic = Critic(sess, state_size, action_size, LR_C, GAMMA,
                REPLACEMENT, actor.a, actor.a_)

actor.add_grad_to_graph(critic.a_grads)

sess.run(tf.global_variables_initializer())

M = Memory(MEMORY_CAPACITY, dims = 2*state_size + action_size + 1)

##########
## Tensorboard

now = datetime.now()
logdir = 'logs/' + now.strftime("%Y%m%d-%H%M%S") + '/'
writer = tf.summary.FileWriter(logdir, sess.graph)

summary = tf.Summary()

var = 3

file = open('data_log.txt','w')
scores = []
avgscores = []
avg100scores = []
last100scores = deque(maxlen=100)
t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):

        if RENDER:
            env.render()

        a = actor.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)
        s_, r, done, info = env.step(a)

        M.store_transitions(s, a, r/10, s_)

        if M.pointer > MEMORY_CAPACITY:
            var *= .9995 # decay the action randomness
            b_M = M.sample(BATCH_SIZE)
            b_s = b_M[:, :state_size]
            b_a = b_M[:, state_size: state_size+action_size]
            b_r = b_M[:, -state_size-1: -state_size]
            b_s_ = b_M[:, -state_size:]

            critic.train(b_s, b_a, b_r, b_s_)
            actor.train(b_s)

        s = s_
        ep_reward += r

        if j == MAX_EP_STEPS - 1:
            scores.append(ep_reward)
            last100scores.append(ep_reward)
            print('Episode:{}, Reward: {}, Explore:{:.2f}'.format(i,
                                                int(ep_reward), var))
            summary.value.add(tag='Score', simple_value=ep_reward)
            summary.value.add(tag='AvgScore', simple_value  =np.mean(scores))
            summary.value.add(tag='Avg100Score', simple_value=np.mean(last100scores))
            writer.add_summary(summary, i)
            file.write('{}\t{}\t{}\t{}\n'.format(i, int(ep_reward),
                                               np.mean(scores),
                                                 np.mean(last100scores)))

            #if(ep_reward>-100):
            #      RENDER = True
            break
print('Running time:', time.time()-t1)

writer.flush()
file.close()

###############
## plot
plt.plot(scores, 'r-', label='scores')
plt.plot(avgscores, 'g-', label='avg scores')
plt.plot(avg100scores, 'b-', label='avg 100 scores')
plt.legend('lower right')
plt.xlabel('Episodes')
plt.ylabel('Scores')
plt.title('DDPG-Pendulum-V0')
plt.savefig('./ddpg_pendu_tf.png')
plt.show()



