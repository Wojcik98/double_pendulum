from dataclasses import MISSING
from typing import List
from configclass import configclass


@configclass
class RunnerCfg:
    """Configuration for the runner."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""


@configclass
class AgentCfg:
    """Configuration for the agent."""

    gamma: float = MISSING
    """The environment discount factor."""

    benchmark: bool = False
    """Whether to benchmark runtime."""

    action_max: float = MISSING
    """The maximum action value."""

    action_min: float = MISSING
    """The minimum action value."""


@configclass
class ActorCriticCfg(AgentCfg):
    """Configuration for the actor-critic networks."""

    actor_activations: List[str] = MISSING
    """The activation functions for the actor network."""

    actor_hidden_dims: List[int] = MISSING
    """The hidden dimensions of the actor network."""

    actor_init_gain: float = 0.5
    """Network initialization gain for the actor network."""

    actor_input_normalization: bool = False
    """Whether to normalize the input for the actor network."""

    actor_recurrent_layers: int = 0
    """The number of recurrent layers for the actor network."""

    actor_recurrent_module: str = "LSTM"
    """The recurrent module for the actor network."""

    actor_recurrent_tf_context_length: int = 64
    """The context length for the actor network."""

    actor_recurrent_tf_head_count: int = 8
    """The head count for the actor network."""

    actor_shared_dims: int = None
    """The shared dimensions for the actor network."""

    batch_count: int = 10
    """The batch count for the actor-critic networks."""

    batch_size: int = 1
    """The batch size for the actor-critic networks."""

    critic_activations: List[str] = MISSING
    """The activation functions for the critic network."""

    critic_hidden_dims: List[int] = MISSING
    """The hidden dimensions of the critic network."""

    critic_init_gain: float = 0.5
    """Network initialization gain for the critic network."""

    critic_input_normalization: bool = False
    """Whether to normalize the input for the critic network."""

    critic_recurrent_layers: int = 0
    """The number of recurrent layers for the critic network."""

    critic_recurrent_module: str = "LSTM"
    """The recurrent module for the critic network."""

    critic_recurrent_tf_context_length: int = 64
    """The context length for the critic network."""

    critic_recurrent_tf_head_count: int = 8
    """The head count for the critic network."""

    critic_shared_dims: int = None
    """The shared dimensions for the critic network."""

    polyak: float = 0.995
    """The polyak factor for the target networks."""

    recurrent: bool = False
    """Whether to use recurrent networks."""

    return_steps: int = 1
    """The number of steps to return for the actor-critic networks."""


@configclass
class SACCfg(ActorCriticCfg):
    """Configuration for the SAC agent."""

    actor_lr: float = 1e-4
    """The learning rate for the actor network."""

    actor_noise_std: float = 1.0
    """The standard deviation of the actor noise."""

    alpha: float = 0.2
    """The entropy regularization coefficient."""

    alpha_lr: float = 1e-3
    """The learning rate for the entropy regularization coefficient."""

    chimera: bool = True
    """Whether to use separate heads for computing action mean and std (True)
    or treat the std as a tunable parameter."""

    critic_lr: float = 1e-3
    """The learning rate for the critic network."""

    gradient_clip: float = 1.0
    """The gradient clipping value."""

    log_std_max: float = 4.0
    """The maximum log standard deviation value."""

    log_std_min: float = -20.0
    """The minimum log standard deviation value."""

    storage_initial_size: int = 0
    """The initial size of the storage."""

    storage_size: int = 100000
    """The size of the storage."""

    target_entropy: float = None
    """The target entropy value."""
