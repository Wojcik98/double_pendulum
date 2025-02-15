from configclass import configclass

from rsl_rl_cfg_defs import RunnerCfg, PPOCfg


@configclass
class AIOlympicsPPOCfg(PPOCfg):
    benchmark = False
    action_min = -1.0
    action_max = 1.0
    return_steps = 1
    polyak = 0.995
    critic_input_normalization = False
    actor_input_normalization = False
    batch_count = 5
    batch_size = 10000
    gamma = 0.995
    actor_noise_std = 0.3
    learning_rate = 1e-3
    schedule = "fixed"
    actor_activations = ["elu", "elu", "linear"]
    actor_hidden_dims = [16, 16]
    actor_init_gain = 0.5
    critic_activations = ["elu", "elu", "elu", "linear"]
    critic_hidden_dims = [32, 32, 32]
    critic_init_gain = 0.5
    value_coeff = 1.0


@configclass
class AIOlympicsRunnerCfg(RunnerCfg):
    num_steps_per_env = 100
