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
        in_dim = x.shape[-1]

        # Gating from task id -> routed expert logits
        gating_logits = nn.Dense(
            num_routed,
            use_bias=False,
            kernel_init=self.config.kernel_init(),
            name="gating",
        )(task_idx)  # (B, Nr)

        # Compute top-k indices per sample
        topk_idx = jnp.argsort(-gating_logits, axis=-1)[:, :k_active]  # (B, K)
        # Gather top-k logits and normalize within top-k
        topk_logits = jnp.take_along_axis(gating_logits, topk_idx, axis=-1)  # (B, K)
        topk_weights = jax.nn.softmax(topk_logits, axis=-1)  # (B, K)

        # Monitor expert assignments: count how many samples are assigned to each expert
        # Create one-hot encoding for selected experts: (B, K, num_routed)
        expert_one_hot = jax.nn.one_hot(topk_idx, num_classes=num_routed)  # (B, K, num_routed)
        # Sum over batch and k dimensions to get total assignments per expert: (num_routed,)
        expert_assignments = expert_one_hot.sum(axis=(0, 1))  # (num_routed,)
        # Store as intermediate for logging
        self.sow("intermediates", "expert_assignments", expert_assignments)

        # Manually create parameters for all routed experts (for initialization)
        # Then we'll compute only the top-k experts by gathering their parameters
        routed_kernels = []
        routed_biases = []
        prev_dim = in_dim
        for layer_id in range(num_hidden_layers):
            # Create parameters for all routed experts: (num_routed, in_dim, out_dim)
            kernel = self.param(
                f"routed_expert_layer_{layer_id}_kernel",
                self.config.kernel_init(),
                (num_routed, prev_dim, hidden_dim),
            )
            if self.config.use_bias:
                bias = self.param(
                    f"routed_expert_layer_{layer_id}_bias",
                    self.config.bias_init(),
                    (num_routed, hidden_dim),
                )
            else:
                bias = None
            routed_kernels.append(kernel)
            routed_biases.append(bias)
            prev_dim = hidden_dim

        # Manually create parameters for shared experts (always activated)
        shared_kernels = []
        shared_biases = []
        prev_dim = in_dim
        for layer_id in range(num_hidden_layers):
            if num_shared > 0:
                kernel = self.param(
                    f"shared_expert_layer_{layer_id}_kernel",
                    self.config.kernel_init(),
                    (num_shared, prev_dim, hidden_dim),
                )
                if self.config.use_bias:
                    bias = self.param(
                        f"shared_expert_layer_{layer_id}_bias",
                        self.config.bias_init(),
                        (num_shared, hidden_dim),
                    )
                else:
                    bias = None
            else:
                kernel = jnp.zeros((0, prev_dim, hidden_dim), dtype=x.dtype)
                bias = None
            shared_kernels.append(kernel)
            shared_biases.append(bias)
            prev_dim = hidden_dim

        # Compute routed experts: ONLY top-k experts per sample
        if num_hidden_layers == 0:
            # No hidden layers: just pass through input
            routed_weighted = x  # (B, dim)
        else:
            # Compute only top-k experts per sample using vectorized operations
            # topk_idx: (B, K) - expert indices for each sample
            
            # Gather kernels and biases for top-k experts: (B, K, in_dim, out_dim) and (B, K, out_dim)
            # We need to gather for each layer
            def compute_expert_layer(h, layer_kernels, layer_biases, expert_indices):
                """Compute one layer for top-k experts.
                
                Args:
                    h: (B, K, in_dim) - hidden states for top-k experts
                    layer_kernels: (num_routed, in_dim, out_dim) - all expert kernels
                    layer_biases: (num_routed, out_dim) - all expert biases  
                    expert_indices: (B, K) - expert indices for each sample
                
                Returns:
                    (B, K, out_dim) - output for top-k experts
                """
                # Gather kernels and biases for selected experts
                selected_kernels = layer_kernels[expert_indices]  # (B, K, in_dim, out_dim)
                
                # Compute: (B, K, in_dim) @ (B, K, in_dim, out_dim) -> (B, K, out_dim)
                # Use einsum for batched matmul
                out = jnp.einsum("bki,bkij->bkj", h, selected_kernels)
                if layer_biases is not None:
                    selected_biases = layer_biases[expert_indices]  # (B, K, out_dim)
                    out = out + selected_biases
                out = self.config.activation(out)
                return out
            
            # Initialize hidden states: broadcast x to (B, K, in_dim)
            h = jnp.tile(x[:, None, :], (1, k_active, 1))  # (B, K, in_dim)
            
            # Forward through all layers
            for layer_id in range(num_hidden_layers):
                h = compute_expert_layer(
                    h, routed_kernels[layer_id], routed_biases[layer_id], topk_idx
                )  # (B, K, width)
            
            # Weight the top-k expert outputs: (B, K, width) * (B, K) -> (B, width)
            routed_weighted = jnp.einsum("bkw,bk->bw", h, topk_weights)  # (B, width)

        # Compute shared experts: always activated, then average
        if num_shared > 0:
            if num_hidden_layers == 0:
                # No layers: just average the input (all shared experts return same output)
                shared_weighted = x  # (B, dim)
            else:
                # Compute all shared experts using vectorized operations
                # Broadcast x to (B, S, in_dim) for all shared experts
                h_shared = jnp.tile(x[:, None, :], (1, num_shared, 1))  # (B, S, in_dim)
                
                # Forward through all layers for all shared experts
                for layer_id in range(num_hidden_layers):
                    # shared_kernels[layer_id]: (num_shared, in_dim, out_dim)
                    # Compute: (B, S, in_dim) @ (S, in_dim, out_dim) -> (B, S, out_dim)
                    h_shared = jnp.einsum("bsi,sij->bsj", h_shared, shared_kernels[layer_id])
                    if shared_biases[layer_id] is not None:
                        # shared_biases[layer_id]: (num_shared, out_dim)
                        h_shared = h_shared + shared_biases[layer_id][None, :, :]  # (B, S, out_dim)
                    h_shared = self.config.activation(h_shared)
                
                # Average over shared experts: (B, S, width) -> (B, width)
                shared_weighted = h_shared.mean(axis=1)  # (B, width)
        else:
            if num_hidden_layers == 0:
                shared_weighted = jnp.zeros_like(x)  # (B, dim)
            else:
                shared_weighted = jnp.zeros((batch_dim, hidden_dim), dtype=x.dtype)

        # Combine routed and shared experts
        if num_hidden_layers == 0:
            features_out = routed_weighted + shared_weighted  # (B, dim)
        else:
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