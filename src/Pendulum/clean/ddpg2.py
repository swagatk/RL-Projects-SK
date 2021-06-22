'''
Actor-Critic Model for implementing DDPG Algorithm
Tensorflow 2.0
This code works. average score over last 40 episodes > -200 after training over 50-60 episodes.
'''
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import gym
import os

# required for reproducing the result
#np.random.seed(1)
#tf.random.set_seed(1)


##########################
# NOISE MODEL
#################################
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


############################################
# ACTOR
##############################
class Actor:
    def __init__(self, state_size, action_size,
                 replacement, learning_rate,
                 upper_bound):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.replacement = replacement
        self.upper_bound = upper_bound
        self.train_step_count = 0

        # create NN models
        self.model = self._build_net()
        self.target = self._build_net()
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.target.set_weights(self.model.get_weights())

    def _build_net(self):
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        state_input = layers.Input(shape=(self.state_size, ))
        l1 = layers.Dense(256, activation='relu')(state_input)
        l2 = layers.Dense(256, activation='relu')(l1)
        net_out = layers.Dense(self.action_size, activation='tanh',
                              kernel_initializer=last_init)(l2)

        net_out = net_out * self.upper_bound
        model = keras.Model(state_input, net_out)
        model.summary()
        return model

    def update_target(self):

        if self.replacement['name'] == 'hard':
            if self.train_step_count % \
                    self.replacement['rep_iter_a'] == 0:
                self.target.set_weights(self.model.get_weights())
        else:
            w = np.array(self.model.get_weights())
            w_dash = np.array(self.target.get_weights())
            new_wts = self.replacement['tau'] * w + \
                        (1 - self.replacement['tau']) * w_dash
            self.target.set_weights(new_wts)

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)

    def train(self, state_batch, critic):
        self.train_step_count += 1
        with tf.GradientTape() as tape:
            actor_weights = self.model.trainable_variables
            actions = self.model(state_batch)
            critic_value = critic.model([state_batch, actions])
            # -ve value is used to maximize value function
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_weights)
        self.optimizer.apply_gradients(zip(actor_grad, actor_weights))
        return actor_loss


#######################
# CRITIC
######################
class Critic:
    def __init__(self, state_size, action_size,
                        replacement,
                        learning_rate=1e-3,
                        gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.replacement = replacement
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        self.model = self._build_net()
        self.target = self._build_net()
        self.gamma = gamma
        self.train_step_count = 0
        self.target.set_weights(self.model.get_weights())

    def _build_net(self):
        state_input = layers.Input(shape=(self.state_size,))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(self.action_size,))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through separate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        net_out = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model(inputs=[state_input, action_input], outputs=net_out)
        model.summary()
        return model

    def update_target(self):
        if self.replacement['name'] == 'hard':
            if self.train_step_count % \
                    self.replacement['rep_iter_a'] == 0:
                self.target.set_weights(self.model.get_weights())
        else:
            w = np.array(self.model.get_weights())
            w_dash = np.array(self.target.get_weights())
            new_wts = self.replacement['tau'] * w + \
                      (1 - self.replacement['tau']) * w_dash
            self.target.set_weights(new_wts)

    def train(self, state_batch, action_batch, reward_batch,
                                    next_state_batch, actor):
        self.train_step_count += 1
        with tf.GradientTape() as tape:
            critic_weights = self.model.trainable_variables
            target_actions = actor.target(next_state_batch)
            target_critic = self.target([next_state_batch, target_actions])
            y = reward_batch + self.gamma * target_critic
            critic_value = self.model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, critic_weights)
        self.optimizer.apply_gradients(zip(critic_grad, critic_weights))
        return critic_loss

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)


###########################
# REPLAY BUFFER
#######################
class Buffer:
    def __init__(self, state_size, action_size,
                 buffer_capacity=100000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.state_size = state_size
        self.action_size = action_size

        # number of times record() is called
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, self.state_size))
        self.action_buffer = np.zeros((self.buffer_capacity, self.action_size))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.state_size))

    # store <s, a, r, s'> observation tuple
    def record(self, obs_tuple):
        # if buffer capacity is exceeded, old records are
        # replaced by new ones
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    def sample(self):
        # get the sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        return state_batch, action_batch, reward_batch, next_state_batch

#########################
# DDPG AGENT
#########################
class DDPGAgent:
    def __init__(self, state_size, action_size,
                 replacement, lr_a, lr_c,
                 batch_size,
                 memory_capacity,
                 gamma,
                 upper_bound, lower_bound):
        self.state_size = state_size
        self.action_size = action_size
        self.replacement = replacement
        self.actor_lr = lr_a
        self.critic_lr = lr_c
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.gamma = gamma
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self.actor = Actor(self.state_size, self.action_size, self.replacement,
                           self.actor_lr, self.upper_bound)
        self.critic = Critic(self.state_size, self.action_size, self.replacement,
                             self.critic_lr, self.gamma)
        self.buffer = Buffer(self.state_size, self.action_size,
                             self.memory_capacity, self.batch_size)

        std_dev = 0.2
        self.noise_object = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

        # Initially make weights for target and model equal
        self.actor.target.set_weights(self.actor.model.get_weights())
        self.critic.target.set_weights(self.critic.model.get_weights())

    def policy(self, state):
        sampled_action = tf.squeeze(self.actor.model(state))
        noise = self.noise_object()

        # Add noise to the action
        sampled_action = sampled_action.numpy() + noise

        # Make sure that the action is within bounds
        valid_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        return [np.squeeze(valid_action)]

    def experience_replay(self):

        # sample from stored memory
        state_batch, action_batch, reward_batch,\
                        next_state_batch = self.buffer.sample()

        a_loss = self.actor.train(state_batch, self.critic)
        c_loss = self.critic.train(state_batch, action_batch, reward_batch,
                          next_state_batch, self.actor)
        return a_loss, c_loss

    def update_targets(self):
        self.actor.update_target()
        self.critic.update_target()


if __name__=='__main__':

    path = './'
    filename = path + 'result_ddpg.txt'

    if os.path.exists(filename):
        print('Deleting existing file. A new one will be created.')
        os.remove(filename)
    else:
        print('The file does not exist. It will be created.')

    # Hyper parameters
    MAX_EPISODES = 200

    LR_A = 0.001
    LR_C = 0.002
    GAMMA = 0.99

    replacement = [
        dict(name='soft', tau=0.005),
        dict(name='hard', rep_iter_a=600, rep_iter_c=500)
    ][0]  # you can try different target replacement strategies

    MEMORY_CAPACITY = 20000
    BATCH_SIZE = 64

    problem = "Pendulum-v0"
    env = gym.make(problem)

    num_states = env.observation_space.shape[0]
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_space.shape[0]
    print("Size of Action Space ->  {}".format(num_actions))

    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))

    agent = DDPGAgent(num_states, num_actions,
                     replacement, LR_A, LR_C,
                     BATCH_SIZE,
                     MEMORY_CAPACITY,
                     GAMMA,
                     upper_bound, lower_bound)

    ep_reward_list = []
    avg_reward_list = []
    frames = []
    best_score = -1000
    for ep in range(MAX_EPISODES):
        prev_state = env.reset()
        episodic_reward = 0
        while True:
            if ep > MAX_EPISODES-3:
                frames.append(env.render(mode='rgb_array'))
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            action = agent.policy(tf_prev_state)
            state, reward, done, _ = env.step(action)
            # store experience
            agent.buffer.record((prev_state, action, reward, state))
            episodic_reward += reward

            # training
            a_loss, c_loss = agent.experience_replay()

            # Update target models
            agent.update_targets()

            prev_state = state

            if done:
                if episodic_reward > best_score:
                    best_score = episodic_reward
                    agent.actor.save_weights('./pendu_actor_weights.h5')
                    agent.critic.save_weights('./pendu_critic_weights.h5')
                break

        ep_reward_list.append(episodic_reward)
        avg_reward = np.mean(ep_reward_list[-50:])
        print("Episode * {} * Avg Reward = {} ".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

        with open(filename, 'a') as file:
            file.write('{}\t{}\t{}\t{}\t{}\n'.format(ep, episodic_reward,
                                               avg_reward, a_loss, c_loss))
        #if avg_reward > -200:
        #    print('Problem is solved in {} episodes'.format(ep))
        #    break
    env.close()

    # plot
    plt.plot(avg_reward_list)
    plt.xlabel('Episodes')
    plt.ylabel('Average of last 50 episodic rewards')
    plt.grid()
    plt.savefig('./pendu_ddpg_tf2.png')
    plt.show()














