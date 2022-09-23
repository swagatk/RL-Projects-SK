"Model Confguration in JSON format"

CFG = {
    "env" : {
        "domain_name" : "cheetah",
        "task_name" : "run",
        "transform" : "random_crop",
        "pre_transform_image_size" : 100,
        "eval_pre_transform_image_size" : 100,
        "aug_image_size" : 84,
        "stack_size" : 3,
        "frame_skip" : 8,
    },
    "train": {
        "learning_rate" : 1e-3,
       "num_training_steps" : 100000,
       "batch_size" : 128,
       "init_steps" : 1000,
       "ac_update_freq" : 2,
       "enc_update_freq" : 1, 
       "target_wt_update_freq" : 1, 
    },
    "replay_buffer" : {
        "capacity" : 10000,
    },
    "eval" : {
        "num_episodes" : 10,
        "eval_freq" : 1000,
    },
    "encoder" : {
        "latent_dim" : 50,
        "tau" : 0.05,       # exponential moving average 
        "conv_layers" : [32, 32],
        "strides" : (1,1),
        "padding" :  "same",
    },
    "decoder" : {
        "tau" : 0.005,      # polyak averaging constant   
        "conv_layers" : [32, 32],
        "strides" : (1,1),
        "padding" :  "same",
    },
    "predictor" : {
        "dense_layers" : [256, 128],
    },
    "actor" : {
        "dense_layers" : [128, 64],
    },
    "critic" : {
        "gamma" : 0.95,         # discount factor
        "dense_layers" : [32, 32],
        "target_update_freq" : 2,
    },
    "curl" : {
        "alpha" : 0.2,          # entropy coefficient
        "gamma" : 0.95,         # discount factor
        "polyak" : 0.995,       # averaging coeff
        "latent_dim" : 128,
        "cropped_img_size" : 84,
        "stack_size" : 3,
        "include_reconst_loss" : False,
        "include_constcy_loss" : False,
        "alpha_r" : 0.33,           # coeff for reconstruction loss
        "alpha_c" : 0.33,           # coeff for contrastive loss
        "alpha_cy" : 0.33,          # coefficient for consistency loss
    },
    "params" : {
        "seed" : 12345,
        "algo" : 'curl_sac',  # choose from ('sac' , 'curl_sac')
    
    },
}