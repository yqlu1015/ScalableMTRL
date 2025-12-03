import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from metaworld_algorithms.config.nn import SparseMoEConfig

from .base import MLP


class SparseMoENetwork(nn.Module):
    config: SparseMoEConfig

    head_dim: int  # output dim; e.g., 1 for Q, 2 * action_dim for policy
    head_kernel_init: jax.nn.initializers.Initializer = jax.nn.initializers.he_normal()
    head_bias_init: jax.nn.initializers.Initializer = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        chex.assert_rank(x, 2)
        batch_dim = x.shape[0]
        assert self.config.num_tasks is not None, "Number of tasks must be provided."

        # Split out task one-hot from observation
        task_idx = x[..., -self.config.num_tasks :]
        x = x[..., : -self.config.num_tasks]

        # Determine expert counts
        num_routed = self.config.num_experts
        num_shared = self.config.num_shared_experts or 0
        k_active = self.config.k_active_experts
        assert 0 <= num_shared
        assert 1 <= k_active <= num_routed, "k_active_experts must be in [1, num_experts]"

        num_hidden_layers = max(self.config.depth - 1, 0)
        hidden_dim = self.config.width

        # Gating from task id -> routed expert logits
        gating_logits = nn.Dense(
            num_routed,
            use_bias=False,
            kernel_init=self.config.kernel_init(),
            name="gating",
        )(task_idx)  # (B, Nr)  
        gating_scores = jax.nn.softmax(gating_logits, axis=-1)  # (B, Nr)

        # Add learnable bias term for top-k selection
        gating_bias = self.param(
            "gating_bias",
            jax.nn.initializers.zeros,
            (num_routed,),
        )  # (Nr,)

        # Compute top-k indices per sample
        if self.config.load_balancing:
            topk_idx = jnp.argsort(-gating_scores-gating_bias, axis=-1)[:, :k_active]  # (B, K)
        else:
            topk_idx = jnp.argsort(-gating_scores, axis=-1)[:, :k_active]  # (B, K)
        # Gather top-k logits and normalize within top-k
        topk_logits = jnp.take_along_axis(gating_logits, topk_idx, axis=-1)  # (B, K)
        topk_weights = jax.nn.softmax(topk_logits, axis=-1)  # (B, K)

        # Monitor expert assignments: count how many samples are assigned to each expert
        # Create one-hot encoding for selected experts: (B, K, num_routed)
        expert_one_hot = jax.nn.one_hot(topk_idx, num_classes=num_routed)  # (B, K, num_routed)
        # Sum over batch and k dimensions to get total assignments per expert: (num_routed,)
        expert_assignments = expert_one_hot.sum(axis=(0, 1))  # (num_routed,)

        # Update load balancing bias
        if self.config.load_balancing:
            load_violation = jnp.sign(expert_assignments.mean(keepdims=True) - expert_assignments)
            gating_bias = gating_bias + self.config.u * load_violation

        # Store as intermediate for logging
        self.sow("intermediates", "expert_assignments", expert_assignments)

        # Compute routed experts using vmap with MLP (like MOORE, but without orthogonalization)
        # Note: vmap computes all experts, but we'll select only top-k per sample
        if num_hidden_layers == 0:
            # No hidden layers: just pass through input
            routed_all = jnp.tile(x[:, None, :], (1, num_routed, 1))  # (B, Nr, dim)
        else:
            routed_all = nn.vmap(
                MLP,
                variable_axes={"params": 0, "intermediates": 1},
                split_rngs={"params": True, "dropout": True},
                in_axes=None,  # pyright: ignore [reportArgumentType]
                out_axes=-2,
                axis_size=num_routed,
                # name="routed_experts",
            )(
                hidden_dim,
                num_hidden_layers,
                hidden_dim,
                self.config.activation,
                self.config.kernel_init(),
                self.config.bias_init(),
                self.config.use_bias,
                activate_last=False,
            )(x)  # (B, Nr, width)

        # Select only top-k experts per sample and weight them
        batch_indices = jnp.arange(batch_dim)[:, None]  # (B, 1)
        routed_topk = routed_all[batch_indices, topk_idx]  # (B, K, width)
        # Weight the top-k expert outputs: (B, K, width) * (B, K) -> (B, width)
        routed_weighted = jnp.einsum("bkw,bk->bw", routed_topk, topk_weights)  # (B, width)

        # Compute shared experts: always activated, then average (using vmap like MOORE)
        if num_shared > 0:
            if num_hidden_layers == 0:
                # No layers: just average the input (all shared experts return same output)
                shared_all = jnp.tile(x[:, None, :], (1, num_shared, 1))  # (B, S, dim)
            else:
                shared_all = nn.vmap(
                    MLP,
                    variable_axes={"params": 0, "intermediates": 1},
                    split_rngs={"params": True, "dropout": True},
                    in_axes=None,  # pyright: ignore [reportArgumentType]
                    out_axes=-2,
                    axis_size=num_shared,
                    # name="shared_experts",
                )(
                    hidden_dim,
                    num_hidden_layers,
                    hidden_dim,
                    self.config.activation,
                    self.config.kernel_init(),
                    self.config.bias_init(),
                    self.config.use_bias,
                    activate_last=False,
                )(x)  # (B, S, width)
            # Average over shared experts: (B, S, width) -> (B, width)
            shared_weighted = shared_all.mean(axis=1)  # (B, width)
        else:
            if num_hidden_layers == 0:
                shared_weighted = jnp.zeros_like(x)  # (B, dim)
            else:
                shared_weighted = jnp.zeros((batch_dim, hidden_dim), dtype=x.dtype)

        # Combine routed and shared experts
        features_out = routed_weighted + shared_weighted  # (B, width)
        features_out = jax.nn.tanh(features_out)
        self.sow("intermediates", "torso_output", features_out)

        # Multi-head output: one head per task, pick head by task_idx
        heads_out = nn.vmap(
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
        )(features_out)  # (B, T, head_dim)

        task_indices = task_idx.argmax(axis=-1)
        out = heads_out[jnp.arange(batch_dim), task_indices]
        return out