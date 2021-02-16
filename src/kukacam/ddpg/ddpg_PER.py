"""
DDPG + PER (PRIORITY EXPERIENCE REPLAY)
Status: Not working. The average reward over 100 episodes remains below 0.3 even after 12K episodes.
Buffer size is limited to 30K only. It gives error on increasing the buffer size
Todo: Focus on code reuse
"""
import tensorflow as tf
import numpy as np
from per_memory_buffer import Memory
import pickle
from FeatureNet import FeatureNetwork
from OUActionNoise import OUActionNoise
from actor import KukaActor
from critic import KukaCritic
########################################
# check tensorflow version
from packaging import version
print("Tensorflow Version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This program requires Tensorflow 2.0 or above"
#######################################
_DEBUG = False
#######################################
# avoid CUDNN_STATUS_INTERNAL_ERROR
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


####################################
# ACTOR-CRITIC AGENT
##################################
class DDPG_PER_Agent:
    def __init__(self, state_size, action_size,
                 replacement,
                 lr_a, lr_c,
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

        self.start_episode = 0
        self.ep_reward_list = []
        self.avg_reward_list = []

        self.feature = FeatureNetwork(self.state_size, self.actor_lr)

        self.actor = KukaActor(self.state_size, self.action_size, self.replacement,
                               self.actor_lr, self.upper_bound, self.feature)
        self.critic = KukaCritic(self.state_size, self.action_size, self.replacement,
                                 self.critic_lr, self.gamma, self.feature)
        self.buffer = Memory(self.memory_capacity, self.batch_size,
                             self.state_size, self.action_size)

        std_dev = 0.2
        self.noise_object = OUActionNoise(mean=np.zeros(1),
                                          std_deviation=float(std_dev) * np.ones(1))

        # Initially make weights for target and model equal
        self.actor.target.set_weights(self.actor.model.get_weights())
        self.critic.target.set_weights(self.critic.model.get_weights())

    def policy(self, state):
        # Check the size of state: (w, h, c)
        # convert the numpy array state into a tensor of size (1, w, h, c)
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

        sampled_action = tf.squeeze(self.actor.model(tf_state))
        noise = self.noise_object()  # scalar value

        # convert into the same shape as that of the action vector
        noise_vec = noise * np.ones(self.action_size)

        # Add noise to the action
        sampled_action = sampled_action.numpy() + noise_vec

        # Make sure that the action is within bounds
        valid_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        return np.squeeze(valid_action)

    def record(self, experience: tuple):
        priority = self.get_per_error(experience)
        self.buffer.record(experience, priority)

    def experience_replay(self):
        # sample from stored memory
        state_batch, action_batch, reward_batch, \
                next_state_batch, done_batch,\
                 sampled_idxs, is_weights = self.buffer.sample()

        # convert to tensors
        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)
        done_batch = tf.convert_to_tensor(done_batch, dtype=tf.float32)

        # train actor & critic networks
        actor_loss = self.actor.train(state_batch, self.critic)
        critic_loss = self.critic.train(state_batch, action_batch, reward_batch,
                            next_state_batch, done_batch, self.actor)

        # update the priorities in the replay buffer
        for i in range(len(sampled_idxs)):
            self.buffer.update(sampled_idxs[i], is_weights[i])

        if _DEBUG:
            with open('./root_value.txt', 'a+') as file:
                file.write('{}\t{:0.3f}\n'.format(self.buffer.available_samples,
                                                self.buffer.sum_tree.root_node.value))

        return actor_loss, critic_loss

    def get_per_error(self, experience: tuple):
        '''
        Computes the time-delay error which is used to determine the relative priority
        of the experience. This will be used for PER-based training
        delta[i] = r_t + gamma * QT(s', a) - Q(s, a)
        :param experience:
        :return: error
        '''

        state, action, reward, next_state, done = experience

        # convert the numpy array state into a tensor of size (1, w, h, c)
        tf_state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        tf_next_state = tf.expand_dims(tf.convert_to_tensor(next_state, dtype=tf.float32), 0)
        tf_action = tf.expand_dims(tf.convert_to_tensor(action, dtype=tf.float32), 0)
        tf_done = tf.convert_to_tensor(done, dtype=tf.float32)  # scalar

        # predict Q(s,a) for a given batch of states and action
        prim_qt = tf.squeeze(self.critic.model([tf_state, tf_action]))

        # target Qtarget(s',a)
        target_qtp1 = tf.squeeze(self.critic.target([tf_next_state, tf_action]))

        target_qt = reward + self.gamma * (1 - tf_done) * target_qtp1

        # time-delay error
        error = tf.squeeze(target_qt - prim_qt).numpy()
        error = DDPG_PER_Agent.huber_loss(error)

        if _DEBUG:
            with open('./td_error.txt', 'a+') as file:
                file.write('{:0.2f}\t{:0.2f}\t{:0.2f}\n'.format(error, target_qt.numpy(), target_qtp1.numpy()))
        return error

    @staticmethod
    def huber_loss(error):
        return 0.5 * error ** 2 if abs(error) < 1.0 else abs(error) - 0.5

    def update_targets(self):
        self.actor.update_target()
        self.critic.update_target()

    def save_model(self, path, actor_filename,
                   critic_filename, buffer_filename):
        actor_file = path + actor_filename
        critic_file = path + critic_filename
        buffer_file = path + buffer_filename

        self.actor.save_weights(actor_file)
        self.critic.save_weights(critic_file)
        self.buffer.save_data(buffer_file)

    def load_model(self, path, actor_filename,
                   critic_filename, buffer_filename):
        actor_file = path + actor_filename
        critic_file = path + critic_filename
        buffer_file = path + buffer_filename

        self.actor.model.load_weights(actor_file)
        self.actor.target.load_weights(actor_file)
        self.critic.model.load_weights(critic_file)
        self.critic.target.load_weights(critic_file)
        self.buffer.load_data(buffer_file)


