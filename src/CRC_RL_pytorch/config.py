"""Model config in json format"""

CFG = {
    "env": {
        "domain_name":"cartpole",
        "task_name": "swingup",
        "pre_transform_image_size": 100,
        "eval_pre_transform_image_size": 100,
        "image_size": 84,
        "action_repeat": 8,  # 2,#4,
        "frame_stack": 3,
        "episode_length":1000,
        "transform1": "random_crop",  # choose from (random_crop, center_crop_image, random_overlay)
        "transform2": "random_overlay",
    },
    "replay_buffer": {
        "capacity": 50000
        },
    "train": {
        "agent": "curl_sac",
        "init_steps": 1000,
        "num_train_steps": 100000,
        "num_updates": 1,
        "batch_size": 128, #512
    },
    "eval": {
        "eval_freq": 1000, 
        "num_eval_episodes": 10, 
        "mode": "video_easy", # mode in {"video_easy", "color_hard", "color_easy", "video_hard", "train"}
    },
    "critic": {
        "lr": 1e-3, 
        "beta": 0.9, 
        "tau": 0.01, 
        "target_update_freq": 2,
        "dense_layers" : [128, 64,],
    },
    "actor": {
        "lr": 1e-3,
        "beta": 0.9,
        "log_std_min": -10,
        "log_std_max": 2,
        "update_freq": 2,
        "dense_layers": [128, 64, ],
    },
    "encoder": {
        "feature_dim": 50,
        "lr": 1e-3,
        "tau": 0.05,
        "conv_layers" : [32, 32, 32],
        "dense_layers": [128, 128, 64],
    },
    "decoder": {
        "type": "pixel",
        "lr": 1e-3,
        "latent_lambda": 1e-6,
        "weight_lambda": 1e-7,
    },
    "predictor": {
        "dense_layers": [128, 128, ],
    },
    "sac": {
        "discount": 0.99, 
        "init_temp": 0.1, 
        "alpha_lr": 1e-4, 
        "alpha_beta": 0.5
    },
    "params": {
        "seed": -1,
        "work_dir": ".",
        "save_tb": False,
        "save_buffer": False,
        "save_video": False,
        "save_model": False,
        "detach_encoder": False,
        "log_interval": 100,
        "num_cls": 4,
        "c1": 0.33,         # contrastive loss weight
        "c2": 0.33,         # reconstruction loss weight
        "c3" : 0.33,        # consistency loss weight
    },
}
