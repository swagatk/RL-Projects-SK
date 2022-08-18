""" 
Environment: Deep Mind Control Environment (DMC) 
Algorithm: CURL-SAC

Updates:
14/03/2022: Work in progress
18/06/2022: use common encoder for actor & critic
29/07/2022: Incorporate Reconstruction loss in CURL
29/07/2022: Train DMC-Cheetah environment.

"""
# Imports
import dmc2gym
import tensorflow as tf
import pybullet_multigoal_gym as pmg
from packaging import version
import os
import datetime
import numpy as np

# Add the current folder to python's import path
import sys

from src.CURL_SAC.curl_utils import FrameStackWrapper4DMC
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir)) # parent director


# Local imports
#from  b.curl_sac_dir.curl_sac_2 import CurlSacAgent
from src.CURL_SAC.curl_sac import CurlSacAgent
from common.utils import check_gpu_availability, set_seed_everywhere
from curl_utils import FrameStackWrapper4DMC, Config
from config import CFG 
########################################
# check tensorflow version
print("Tensorflow Version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This program requires Tensorflow 2.0 or above"
#######################################
# Random seed
#######################################################
gpu_status = check_gpu_availability()
if gpu_status:
    raise SystemError('GPU device not found')
##################################### 
# generate seed

# #### Hyper-parameters 
##########################################
config = Config.from_json(CFG)

config_dict = dict(
    buffer_capacity = 30000,    # 50k (racecar)  # 20K (kuka)
    batch_size = 128,  
    use_attention = None, 
    algo = 'curl_sac',               
    env_name = 'cheetah',          # environment name
    task_name = 'run',             # task name
    image_obsvn=True,
    stack_size=3,
    include_reconst_loss=True,
    seed=-1,
    pre_transform_img_size=64,
)
#######################
WB_LOG = False
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

    if config_dict['seed'] == -1:
        config_dict['seed'] = np.random.randint(0, 100000)

    env = dmc2gym.make(
        domain_name=config_dict['env_name'],
        task_name=config_dict['task_name'],
        seed=config_dict['seed'],
        visualize_reward=False,
        from_pixels=True,
        height=config_dict['pre_transform_img_size'],
        width=config_dict['pre_transform_img_size'],
        frame_skip=10
    )
     # Apply stacking wrapper to the environment
    env = FrameStackWrapper4DMC(env, config_dict['stack_size'],
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
        agent = CurlSacAgent(
            state_size=state_size, 
            action_size=action_size,
            action_upper_bound=upper_bound,
            curl_feature_dim=50,
            buffer_capacity=config_dict['buffer_capacity'],
            batch_size=config_dict['batch_size'],
            include_reconst_loss=config_dict['include_reconst_loss']
        )

        # Train
        agent.run(env, WB_LOG=WB_LOG)


    # test
    # print('Mean Episodic Reward:', agent.validate(env, max_eps=50, render=True, load_path='./best_model/'))
    


