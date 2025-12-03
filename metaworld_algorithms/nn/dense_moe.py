import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from metaworld_algorithms.config.nn import DenseMoEConfig

from .base import MLP


class DenseMoENetwork(nn.Module):
    config: DenseMoEConfig

    head_dim: int  # o, 1 for Q networks and 2 * action_dim for policy networks
    head_kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_normal()
    head_bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        batch_dim = x.shape[0]
        assert self.config.num_tasks is not None, "Number of tasks must be provided."
        task_idx = x[..., -self.config.num_tasks :]
        x = x[..., : -self.config.num_tasks]

        # Task ID embedding
        task_embedding = nn.Dense(
            self.config.num_experts,
            use_bias=False,
            kernel_init=self.config.kernel_init(),
        )(task_idx)

        # MOORE torso
        experts_out = nn.vmap(
            MLP,
            variable_axes={"params": 0, "intermediates": 1},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,  # pyright: ignore [reportArgumentType]
            out_axes=-2,
            axis_size=self.config.num_experts,
        )(
            self.config.width,
            self.config.depth - 1,
            self.config.width,
            self.config.activation,
            self.config.kernel_init(),
            self.config.bias_init(),
            self.config.use_bias,
            activate_last=False,
        )(x)
        features_out = jnp.einsum("bnk,bn->bk", experts_out, task_embedding)
        features_out = jax.nn.tanh(features_out)
        self.sow("intermediates", "torso_output", features_out)

        # MH
        x = nn.vmap(
            nn.Dense,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,  # pyright: ignore [reportArgumentType]
            out_axes=1,
            axis_size=self.config.num_tasks,
        )(
            self.head_dim,
            kernel_init=self.head_kernel_init,
            bias_init=self.head_bias_init,
            use_bias=self.config.use_bias,
        )(features_out)

        task_indices = task_idx.argmax(axis=-1)
        x = x[jnp.arange(batch_dim), task_indices]

        return x
