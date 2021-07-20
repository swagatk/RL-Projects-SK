""" main for running ppo and ipg agents """

# Imports
import tensorflow as tf
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from pybullet_envs.bullet.kukaCamGymEnv import KukaCamGymEnv
from pybullet_envs.bullet.racecarZEDGymEnv import RacecarZEDGymEnv
from pybullet_envs.bullet.racecarGymEnv import RacecarGymEnv
from packaging import version
import gym

# Local imports
from ppo.ppo2 import PPOAgent
from IPG.ipg import IPGAgent
from IPG.ipg_her import IPGHERAgent
from SAC.sac import SACAgent
from common.TimeLimitWrapper import TimeLimitWrapper
from common.CustomGymWrapper import ObsvnResizeTimeLimitWrapper

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
    SEASONS = 50   # 35
    success_value = None
    lr_a = 0.0002  # 0.0002
    lr_c = 0.0002  # 0.0002
    epochs = 20
    training_batch = 1024   # 5120(racecar)  # 1024 (kuka), 512
    training_episodes = 10000    # needed for SAC
    buffer_capacity = 50000     # 50k (racecar)  # 20K (kuka)
    batch_size = 128    # 512 (racecar) #   28 (kuka)
    epsilon = 0.2  # 0.07
    gamma = 0.993  # 0.99
    lmbda = 0.7  # 0.9
    tau = 0.995             # polyak averaging factor
    alpha = 0.2             # Entropy Coefficient
    use_attention = False  # enable/disable for attention model
    use_mujoco = False




    # Kuka DiverseObject Environment
    env = KukaDiverseObjectEnv(renders=False,
                               isDiscrete=False,
                               maxSteps=20,
                               removeHeightHack=False)

    # RaceCar Bullet Environment with image observation
    # env = ObsvnResizeTimeLimitWrapper(RacecarZEDGymEnv(renders=False,
    #                            isDiscrete=False), shape=20, max_steps=20)

    # RaceCar Bullet Environment with vector observation
    # env = RacecarGymEnv(renders=False, isDiscrete=False)

    # PPO Agent
    # agent = PPOAgent(env, SEASONS, success_value, lr_a, lr_c, epochs, training_batch, batch_size, epsilon, gamma,
    #                  lmbda, use_attention, use_mujoco,
    #                  filename='rc_ppo_zed.txt', val_freq=None)
    # IPG Agent
    # agent = IPGAgent(env, SEASONS, success_value, lr_a, lr_c, epochs, training_batch, batch_size, buffer_capacity,
    #                  epsilon, gamma, lmbda, use_attention, use_mujoco,
    #                  filename='rc_ipg_zed.txt', val_freq=None)
    # IPG HER Agent
    # agent = IPGHERAgent(env, SEASONS, success_value, lr_a, lr_c, epochs, training_batch, batch_size,
    #                     buffer_capacity, epsilon, gamma, lmbda, use_attention,
    #                     use_mujoco, filename='rc_ipg_her.txt', val_freq=None)

    # SAC Agent
    agent = SACAgent(env, success_value,
                     epochs, training_episodes, batch_size, buffer_capacity, lr_a, lr_c, alpha,
                     gamma, tau, use_attention, use_mujoco,
                     filename='kuka_sac.txt', tb_log=False, val_freq=None)

    agent.run()