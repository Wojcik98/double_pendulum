from configclass import configclass

from rsl_rl_cfg_defs import SACCfg, RunnerCfg


@configclass
class AIOlympicsSACCfg(SACCfg):
    benchmark = False
    gamma = 0.99
    action_min = -1.0
    action_max = 1.0
    actor_activations = ["elu", "elu", "linear"]
    actor_hidden_dims = [64, 64]
    actor_init_gain = 0.5
    actor_input_normalization = False
    actor_noise_std = 0.02
    actor_lr = 1e-5
    chimera = False
    critic_activations = ["elu", "elu", "linear"]
    critic_hidden_dims = [64, 64]
    critic_init_gain = 0.5
    critic_input_normalization = False
    critic_lr = 1e-2
    alpha_lr = 1e-3
    alpha = 0.1
    polyak = 0.995
    return_steps = 1
    batch_size = 10000
    batch_count = 1
    gradient_clip = 1.0
    storage_size = 10_000_000
    target_entropy = -2.0


@configclass
class AIOlympicsRunnerCfg(RunnerCfg):
    num_steps_per_env = 100
    # num_steps_per_env = 500
