""" main for running ppo and ipg agents """

# Imports
import tensorflow as tf
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv
import pybullet_envs.bullet.racecarZEDGymEnv as e
from packaging import version
import gym
import mujoco_py

# Local imports
from ppo.ppo2 import PPOAgent
from IPG.ipg import IPGAgent
from IPG.ipg_her import IPGHERAgent

if __name__ == "__main__":

    ########################################
    # check tensorflow version
    print("Tensorflow Version: ", tf.__version__)
    assert version.parse(tf.__version__).release[0] >= 2, \
        "This program requires Tensorflow 2.0 or above"
    #######################################

    ######################################
    # avoid CUDNN_STATUS_INTERNAL_ERROR
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    ################################################
    # check GPU device
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    ##############################################

    # #### Hyper-parameters
    SEASONS = 500       # 500 (Fetch) # 50 (Kuka)
    success_value = 70      # Really not required
    lr_a = 0.0002  # 0.0002
    lr_c = 0.0002  # 0.0002
    epochs = 20
    training_batch = 3072   # 5024   # 3072 (Fetch)  # 1024 (kuka), 512
    buffer_capacity = 100000      # 50000 (Fetch)    # 20000 (kuka)
    batch_size = 256    # 528    # 256  # 128 (kuka)
    epsilon = 0.2  # 0.07
    gamma = 0.993  # 0.99
    lmbda = 0.7  # 0.9

    use_attention = False  # enable/disable for attention model
    use_mujoco = True

    if use_mujoco:
        env = gym.make('FetchReach-v1')
    else:
        env = KukaDiverseObjectEnv(renders=False,
                                   isDiscrete=False,
                                   maxSteps=20,
                                   removeHeightHack=False)

        # env = KukaCamGymEnv(renders=False, isDiscrete=False)
        # env = e.RacecarZEDGymEnv(renders=False,
        #                          isDiscrete=False)

    # PPO Agent
    agent = PPOAgent(env, SEASONS, success_value, lr_a, lr_c, epochs, training_batch, batch_size,
                      epsilon, gamma, lmbda, use_attention, use_mujoco)
    # IPG Agent
    # agent = IPGAgent(env, SEASONS, success_value, lr_a, lr_c, epochs, training_batch, batch_size,
    #             buffer_capacity, epsilon, gamma, lmbda, use_attention, use_mujoco)

    # IPG HER Agent
    # agent = IPGHERAgent(env, SEASONS, success_value, lr_a, lr_c, epochs, training_batch, batch_size,
    #                     buffer_capacity, epsilon, gamma, lmbda, use_attention, use_mujoco)

    agent.run()
