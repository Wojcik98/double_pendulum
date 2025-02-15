from configclass import configclass

from rsl_rl_cfg_defs import DQNCfg


@configclass
class AIOlympicsDQNCfg(DQNCfg):
    batch_size = 64
    gamma = 0.999
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 1000
    tau = 1e-2
    learning_rate = 1e-3
    activation = "relu"
    hidden_dims = [32, 32, 32]
    mem_size = 10000
    policy_sampling = 31
    num_steps_per_env = 100
