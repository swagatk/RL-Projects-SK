""" 
Environment: PBMG
Algorithm: CURL-SAC

Updates:
14/03/2022: Work in progress

"""
# Imports
import tensorflow as tf
import pybullet_multigoal_gym as pmg
from packaging import version
import os
import datetime
import numpy as np
import gym

# Add the current folder to python's import path
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir)) # parent director

# Local imports
from algo.curl_sac import curlSacAgent
from common.CustomGymWrapper import FrameStackWrapper
from common.utils import set_seed_everywhere

########################################
# check tensorflow version
print("Tensorflow Version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This program requires Tensorflow 2.0 or above"
#######################################
# Random seed
#######################################################
# np.random.seed(42)
# tf.random.set_seed(42)
######################################
# avoid CUDNN_STATUS_INTERNAL_ERROR
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            print(gpu)
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
# #### Hyper-parameters for RACECAR Environment
##########################################
config_dict = dict(
    buffer_capacity = 10000,    # 50k (racecar)  # 20K (kuka)
    batch_size = 128,  # 512 (racecar) #   128 (kuka)
    # use_attention = {'type': 'luong',   # type: luong, bahdanau
    #                  'arch': 0,         # arch: 0, 1, 2, 3
    #                  'return_scores': False},  # visualize attention maps       
    use_attention = None, 
    algo = 'curl_sac',               # choices: ppo, sac, ipg, sac_her, ipg_her
    env_name = 'pbmg',          # environment name
    # use_her = { 'strategy': 'final',
    #             'extract_feature' : False}, 
    use_her = False,
    stack_size = 3,             # input stack size
    use_lstm = False,             # enable /disable LSTM
    image_obsvn=True
)
#######################
WB_LOG = True
############################
save_path = './log/'
chkpt_freq = None         # wrt seasons
load_path = None
##############################################3
save_path = save_path + config_dict['env_name'] + '/' + config_dict['algo'] + '/'
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#current_time = datetime.datetime.now().strftime("%Y%m%d")
save_path = save_path + current_time + '/'
#logfile = config_dict['env_name'] + '_' + config_dict['algo'] + '.txt'
logfile = None  # Don't write  into file
###########################################
# wandb related configuration
if WB_LOG:
    import wandb
    #WANDB_API_KEY=os.environ['MY_WANDB_API_KEY']
    print("WandB version", wandb.__version__)
    wandb.login()
    wandb.init(project=config_dict['env_name'], config=config_dict)
#######################################################33
if __name__ == "__main__":


    if config_dict['env_name'] == 'pbmg':

        camera_setup = [
            {
                'cameraEyePosition': [-1.0, 0.25, 0.6],
                'cameraTargetPosition': [-0.6, 0.05, 0.2],
                'cameraUpVector': [0, 0, 1],
                'render_width': 100,
                'render_height': 100
            },
            {
                'cameraEyePosition': [-1.0, -0.25, 0.6],
                'cameraTargetPosition': [-0.6, -0.05, 0.2],
                'cameraUpVector': [0, 0, 1],
                'render_width': 100,
                'render_height': 100
            }
        ]

        env = pmg.make_env(
            # task args ['reach', 'push', 'slide', 'pick_and_place', 
            #            'block_stack', 'block_rearrange', 'chest_pick_and_place', 'chest_push']
            task='reach',
            gripper='parallel_jaw',
            num_block=4,  # only meaningful for multi-block tasks
            render=False,    # only for saving image
            binary_reward=True,
            max_episode_steps=20,
            # image observation args
            image_observation=config_dict['image_obsvn'],
            depth_image=False,
            goal_image=True,
            visualize_target=True,
            camera_setup=camera_setup,
            observation_cam_id=0,
            goal_cam_id=1,
            # curriculum args
            use_curriculum=True,
            num_goals_to_generate=90)

        obs_shape = env.observation_space['observation'].__getattribute__('shape')
        dtype_value = env.observation_space['observation'].dtype
        # Apply stacking wrapper to the environment
        env = FrameStackWrapper(env, config_dict['stack_size'],
        org_shape=obs_shape, dtype_value=dtype_value)
        #print('Observation:', env.observation_space)


    if config_dict['image_obsvn']: 
        #state_size = env.observation_space['observation'].__getattribute__('shape')
        state_size = env.observation_space.shape
    else:
        state_size = env.observation_space['state'].__getattribute__('shape')

    action_size = env.action_space.__getattribute__('shape')
    upper_bound = env.action_space.__getattribute__('high')
    

    print('State size:', state_size)
    print('Action size:', action_size)
    print('Action upper bound:', upper_bound)


    # set seed
    set_seed_everywhere(seed=42, env=env)

    ans = input('Do you want to continue?(Y/N)')
    if ans.lower() == 'n':
        exit()


    ######################################################
    if config_dict['algo'] == 'curl_sac': # curl_sac
        agent = curlSacAgent(
            state_size=state_size, 
            action_size=action_size,
            feature_dim=50,
            action_upper_bound=upper_bound)


        # Train
        agent.run(env, WB_LOG=WB_LOG)


    # test
    # print('Mean Episodic Reward:', agent.validate(env, max_eps=50, render=True, load_path='./best_model/'))
    


