"Model Confguration in JSON format"

CFG = {
    "env" : {

    },
    "replay_buffer" : {
        "capacity" : 30000,
    },
    "eval" : {
        "num_episodes" : 100,
        "eval_freq" : 100,
    },
    "encoder" : {
        "lr" : 1e-3,
        "tau" : 0.05,
        "conv_layers" : [32, 32],
        "strides" : (1,1),
        "padding" :  "same",
    },
    "decoder" : {
        "lr" : 1e-3,
        "tau" : 0.05,
        "conv_layers" : [32, 32],
        "strides" : (1,1),
        "padding" :  "same",
    },
    "feature_predictor" : {
        "lr" : 1e-3,
        "dense_layers" : [256, 128],
    },
    "actor" : {
        "lr" : 1e-3,
        "dense_layers" : [128, 64],
    },
    "q_function" : {
        "lr" : 1e-3,
        "dense_layers" : [32, 32],
    },
    "critic" : {
        "lr" : 1e-3,
        "gamma" : 0.95,
        "dense_layers" : [32, 32],
    },
    "curl" : {
        "lr" : 1e-3,
    },
    "agent" : {
        "lr" : 1e-3,
        "alpha" : 0.2,
        "gamma" : 0.95,
        "polyak" : 0.995,
        "latent_dim" : 128,
        "cropped_img_size" : 84,
        "stack_size" : 3,
        "init_steps" : 500,
        "eval_freq" : 1000,
        "eval_episodes" : 20,
        "ac_update_freq" : 2,
        "encoder_update_freq" : 1,
        "target_update_freq" : 2,
        "batch_size" : 32,
        "buffer_capacity" : 30000,
        "num_training_steps" : 100000,
        "include_reconstruction_loss" : True,
        "include_consistency_loss" : True,
    }
}