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
    }


}