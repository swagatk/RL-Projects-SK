"""Model config in json format"""

CFG = {
    "env": {
        "domain_name": "cheetah",
        "task_name": "run",
        "pre_transform_img_size": 100,
        "post_transform_img_size": 84,
        "frame_skip": 8,
        "stack_size": 3,
        "transform": "random_crop",  # choose from (random_crop, random_shift, random_conv)
    },
    "replay_buffer": {"capacity": 10000},
    "train": {
        "algo": "curl_sac",
        "init_steps": 1000,
        "num_train_steps": 100000,
        "num_updates": 1,           # not used
        "batch_size" : 128,
        "target_update_freq" : 2,
        "ac_update_freq" : 1,
        "enc_update_freq" : 2,
    },
    "eval": {"freq": 1000, "num_episodes": 10},
    "critic": {
        "lr": 1e-3, 
        "dense_layers" : [1024, 1024],
    },
    "actor": {
        "lr": 1e-3,
        "beta": 0.9,
        "log_std_min": -10,     # check !!
        "log_std_max": 2,
        "update_freq": 2,
        "dense_layers" : [1024, 1024]
    },
    "encoder": {
        "feature_dim": 50,
        "lr": 1e-3,
        "dense_layers": [128, 128,],
        "conv_layers" : [32, 32, 32, 32,],
    },
    "decoder": {
        "lr": 1e-3,
        "update_freq": 1,
        "latent_lambda": 1e-6,
        "weight_lambda": 1e-7,
    },
    "predictor" : {
        "lr" : 1e-3,
        "dense_layers" : [256, 256],
    },
    "sac": {
            "gamma": 0.99,      #discount 
            "alpha": 0.2, 
            "alpha_lr": 1e-4, 
            "alpha_beta": 0.5,
            "tau": 0.95,       # soft update factor
    },
    "params": {
        "seed": -1,
        "log_interval": 10000,
        "frozen_encoder" : False,
        "include_reconst_loss" : False,
        "include_constcy_loss" : False,
        "alpha_c" : 1.0,
        "alpha_r" : 0.0,
        "alpha_cy" : 0.0
    },
}