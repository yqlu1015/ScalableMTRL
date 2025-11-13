# SAC vs MTSAC Comparison

This document outlines the key differences between the SAC (Soft Actor-Critic) and MTSAC (Multi-Task Soft Actor-Critic) implementations.

## Overview

- **SAC**: Single-task implementation of Soft Actor-Critic
- **MTSAC**: Multi-task extension of SAC designed for learning multiple tasks simultaneously

## Key Differences

### 1. Temperature Parameter (Alpha)

#### SAC
- **Single scalar temperature**: Uses `Temperature` module with a single `log_alpha` parameter
- Shared across all tasks/environments
- Shape: `(1,)`

```python
class Temperature(nn.Module):
    initial_temperature: float = 1.0
    # Single log_alpha parameter
```

#### MTSAC
- **Per-task temperature**: Uses `MultiTaskTemperature` module with task-specific `log_alpha` parameters
- Each task has its own temperature parameter
- Shape: `(num_tasks,)`
- Temperature is computed via task IDs: `task_ids @ log_alpha`

```python
class MultiTaskTemperature(nn.Module):
    num_tasks: int
    initial_temperature: float = 1.0
    # Vector of log_alpha parameters, one per task
```

### 2. Replay Buffer

#### SAC
- Uses `ReplayBuffer`: Single buffer for all samples
- Samples are mixed across tasks (if multiple tasks exist)
- Shape: `(batch_size, obs_dim)`, `(batch_size, action_dim)`, etc.

#### MTSAC
- Uses `MultiTaskReplayBuffer`: Separate storage per task
- Maintains task structure in the buffer
- Shape: `(batch_size, num_tasks, obs_dim)`, etc.
- Samples are organized by task and then flattened

### 3. Task Identification

#### SAC
- No explicit task identification
- Treats all samples uniformly

#### MTSAC
- **Task IDs extracted from observations**: Assumes task IDs are in the last `num_tasks` dimensions of observations
- Extracts task IDs: `task_ids = data.observations[..., -self.num_tasks:]`
- Uses task IDs to:
  - Compute per-task temperature values
  - Compute task weights (if enabled)
  - Split data by task for multi-task optimizers

### 4. Task Weights

#### SAC
- No task weighting mechanism
- All samples contribute equally to loss

#### MTSAC
- **Optional task weights**: Can use `use_task_weights=True` to weight tasks
- Weights computed from temperature parameters: `extract_task_weights()`
- Applied to both critic and actor losses
- Formula: `task_weights = softmax(-log_alpha) * num_tasks`

### 5. Update Logic Structure

#### SAC
- **Monolithic update**: All losses computed together in `_update_inner()`
- Nested function structure:
  - `update_critic()` nested inside `actor_loss()`
  - `update_alpha()` nested inside `actor_loss()`
- Single gradient computation for actor

#### MTSAC
- **Separated updates**: Three distinct methods
  - `update_critic()`: Separate method for critic updates
  - `update_actor()`: Separate method for actor updates
  - `update_alpha()`: Separate method for temperature updates
- More modular design
- Allows for per-task loss computation

### 6. Multi-Task Optimizer Support

#### SAC
- No support for multi-task optimizers (e.g., PCGrad)
- Standard optimizer updates

#### MTSAC
- **Supports multi-task optimizers**: Can use PCGrad and similar methods
- `split_actor_losses` and `split_critic_losses` flags
- When enabled:
  - Data is split by task using `split_data_by_tasks()`
  - Losses computed per task using `jax.vmap()`
  - Optimizer receives per-task losses for gradient manipulation
- Supports `optimizer_extra_args` with `task_losses` and `key`

### 7. Q-Value Clipping

#### SAC
- No Q-value clipping
- Q-values can grow unbounded

#### MTSAC
- **Optional Q-value clipping**: `max_q_value` parameter
- Clips both predicted and target Q-values to `[-max_q_value, max_q_value]`
- Helps stabilize training in multi-task settings
- Comment notes: "Clipping Q values to approximate theoretical maximum for Metaworld"

### 8. Data Splitting/Unsplit Utilities

#### SAC
- No data splitting utilities needed

#### MTSAC
- `split_data_by_tasks()`: Splits batch data by task
- `unsplit_data_by_tasks()`: Reconstructs original batch order
- Used when `split_actor_losses` or `split_critic_losses` is enabled
- Maintains proper indexing for gradient computation

### 9. Configuration Options

#### SAC Config
```python
@dataclasses.dataclass(frozen=True)
class SACConfig(AlgorithmConfig):
    actor_config: ContinuousActionPolicyConfig
    critic_config: QValueFunctionConfig
    temperature_optimizer_config: OptimizerConfig
    initial_temperature: float = 1.0
    num_critics: int = 2
    tau: float = 0.005
```

#### MTSAC Config
```python
@dataclasses.dataclass(frozen=True)
class MTSACConfig(AlgorithmConfig):
    actor_config: ContinuousActionPolicyConfig
    critic_config: QValueFunctionConfig
    temperature_optimizer_config: OptimizerConfig
    initial_temperature: float = 1.0
    num_critics: int = 2
    tau: float = 0.005
    use_task_weights: bool = False  # NEW
    max_q_value: float | None = 5000  # NEW
```

### 10. Loss Computation

#### SAC
- Standard SAC loss computation
- All samples in batch contribute equally
- Single loss value per component (actor, critic, alpha)

#### MTSAC
- Supports task-weighted losses when `use_task_weights=True`
- Can compute per-task losses when `split_actor_losses` or `split_critic_losses=True`
- Losses can be:
  - Averaged across tasks (standard mode)
  - Weighted by task weights (weighted mode)
  - Computed per-task and passed to multi-task optimizer (split mode)

### 11. Alpha (Temperature) Update

#### SAC
- Updates single temperature parameter
- Loss: `-log_alpha * (log_probs + target_entropy)`

#### MTSAC
- Updates per-task temperature parameters
- Computes task-specific log_alpha: `task_ids @ log_alpha`
- Loss: `-log_alpha * (log_probs + target_entropy)` (but log_alpha is task-specific)

### 12. Intermediate Activations

#### SAC
- Has `_get_intermediates()` method for monitoring
- Returns actor and critic intermediate activations

#### MTSAC
- No `_get_intermediates()` method
- Focused on multi-task learning rather than detailed monitoring

## When to Use Which?

### Use SAC when:
- Learning a single task
- Simple, straightforward implementation needed
- No need for task-specific temperature control
- Standard replay buffer is sufficient
- **Note**: SAC can technically be used for multi-task learning (see `examples/multi_task/sac_mt10.py`), but it treats all tasks uniformly with a single temperature parameter

### Use MTSAC when:
- Learning multiple tasks simultaneously
- Need per-task temperature adaptation
- Want to use multi-task optimizers (PCGrad, etc.)
- Need task-weighted learning
- Working with Metaworld or similar multi-task benchmarks
- Require Q-value clipping for stability
- Want explicit multi-task handling and task-specific hyperparameters

## Implementation Notes

1. **MTSAC assumes task IDs in observations**: The implementation extracts task IDs from the last `num_tasks` dimensions of observations. Make sure your environment provides this.

2. **Buffer compatibility**: MTSAC requires `MultiTaskReplayBuffer`, which has a different structure than `ReplayBuffer`. The buffer maintains task structure.

3. **Optimizer requirements**: To use multi-task optimizers like PCGrad, set `split_actor_losses=True` or `split_critic_losses=True` in the network config optimizer.

4. **Task weights**: Task weights are computed from temperature parameters, providing adaptive weighting based on task difficulty (lower temperature = higher weight).

5. **Performance**: MTSAC may be slower due to:
   - Per-task computations
   - Data splitting/unsplitting overhead
   - More complex gradient computations

