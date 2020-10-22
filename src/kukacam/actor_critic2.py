'''
DDPG Algorithm for Pendulum-v0 environment
Source: https://keras.io/examples/rl/ddpg_pendulum/
Tensorflow 2.0 / Keras implementation

Gives a average reward of about -180 to -200 (over 40 episodes)

Be careful with @tf.function decorator. Does not work for all functions. It does not work when I decorate the function
experience_replay() with this decorator.
'''
import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

print('Tensorflow version: ', tf.__version__)


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    print('Creating GIF Animation File. Wait ...')
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)
    print('done!!')


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


class Buffer:
    def __init__(self, state_size, action_size, buffer_capacity=100000, batch_size=64):
        self.num_states = state_size
        self.num_actions = action_size
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.num_states))

    # Takes (s,a,r,s') observation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # We compute the loss and update parameters
    def sample(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        return state_batch, action_batch, reward_batch, next_state_batch


class actor_critic:
    def __init__(self, state_size, action_size,
                 critic_lr, actor_lr, gamma, tau,
                 upper_bound, lower_bound,
                 memory_capacity, batch_size):

        self.num_states = state_size
        self.num_actions = action_size
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        # training models
        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        # target models
        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()

        # Initially both models share same weights
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

        # create memory buffer
        self.buffer = Buffer(self.num_states, self.num_actions,
                             self.memory_capacity, self.batch_size)

    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.num_states,))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(self.num_actions, activation="tanh",
                               kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 for Pendulum.
        outputs = outputs * self.upper_bound
        model = tf.keras.Model(inputs, outputs)

        return model

    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=(self.num_states,))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(self.num_actions,))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through separate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def policy(self, curr_state, noise_object):
        sampled_actions = tf.squeeze(self.actor_model(curr_state))
        noise = noise_object()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        return [np.squeeze(legal_action)]

    @staticmethod
    def _update_target(target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    @tf.function
    def update_target_models(self):
        actor_critic._update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
        actor_critic._update_target(self.target_critic.variables, self.critic_model.variables, self.tau)

    def experience_replay(self):
        # sample a batch of experience from memory buffer
        state_batch, action_batch, reward_batch, next_state_batch = self.buffer.sample()

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True)
            critic_value = self.critic_model([state_batch, action_batch],
                                             training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables))

    def save_weights(self, actorfile, criticfile):
        self.actor_model.save_weights("pendulum_actor.h5")
        self.critic_model.save_weights("pendulum_critic.h5")


if __name__ == '__main__':

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

    # Noise object
    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

    critic_lr = 0.002
    actor_lr = 0.001

    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    tau = 0.005

    agent = actor_critic(num_states, num_actions,
                         critic_lr, actor_lr,
                         gamma, tau,
                         upper_bound, lower_bound,
                         memory_capacity=50000,
                         batch_size=64)

    total_episodes = 100

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    # Takes about 4 min to train
    frames = []
    for ep in range(total_episodes):

        prev_state = env.reset()
        episodic_reward = 0

        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            if ep > total_episodes-5:
                frames.append(env.render(mode='rgb_array'))

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = agent.policy(tf_prev_state, ou_noise)
            # Receive state and reward from environment.
            state, reward, done, info = env.step(action)

            agent.buffer.record((prev_state, action, reward, state))
            episodic_reward += reward

            agent.experience_replay()
            agent.update_target_models()

            # End this episode when `done` is True
            if done:
                break

            prev_state = state

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)
    env.close()

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.grid()
    plt.savefig('./pendu_ddpg_tf2.png')
    print('Close the Graph window to proceed')
    plt.show()

    #save_frames_as_gif(frames)

    # Save the weights
    #agent.save_weights('pendulum_actor.h5', 'pendulum_critic.h5')

