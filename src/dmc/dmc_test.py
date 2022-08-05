"""
If you are accessing over SSH, use the following command to avoid GLFW initialization error:

$ xvfb-run -a -s "-screen 0 1400x900x24" python dmc_test.py
"""
import dmc2gym
import sys 
import os 
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir)) # parent director
from common.video import VideoRecorder

SAVE_VIDEO=False


env = dmc2gym.make(
    domain_name='cheetah',
    task_name='run',
    seed=42,
    visualize_reward=False,
    from_pixels=True,
    height=64,
    width=64,
    frame_skip=10
)
if SAVE_VIDEO:
    video_dir='videos'
    video = VideoRecorder(dir_name=video_dir,
                            height=64,
                            width=64,
                            camera_id=0,)
    video.init()


print('\nObservation shape:', env.observation_space.shape)
print('\nAction shape:', env.action_space.shape)

for ep in range(2):
    obs = env.reset()
    done = False
    ep_reward = 0
    step_count = 0
    while not done:
        if SAVE_VIDEO: 
            video.record(env)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        ep_reward += reward 
        step_count += 1
    print(f'\nepisode: {ep}, steps: {step_count}, reward: {ep_reward}')

if SAVE_VIDEO:
    video.save('eval.mp4')

env.close()
