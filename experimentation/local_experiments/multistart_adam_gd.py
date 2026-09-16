"""Batched multi-start variant of AdamGD: N independent trajectories per step,
evaluated in one obj.vmap_value_and_grad call instead of one obj.value_and_grad
call per trajectory.

Each trajectory gets its own optax state and its own gradient-clip norm
(vmapped init/update) -- this is N independent optimizers running under one
batched objective call, not one optimizer sharing a global clip norm across
all N.
"""

import jax
import optax
from jaxtyping import Array, Float

from dfbench import OptimizationAlgorithm, Objective


class MultiStartAdamGD(OptimizationAlgorithm):
    """Adam with N parallel random-restart trajectories via vmap."""

    algorithm_str = "multistart_adam_gd"

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        learning_rate: float = 0.1,
        n_starts: int = 8,
        **adam_kwargs,
    ) -> None:
        obj = objective
        self.prepare(obj, unbounded=True, random_seed=random_seed)

        params = (
            init_params
            if init_params is not None
            else obj.random_params_unbounded(n_samples=n_starts)
        )
        # n_samples=1 squeezes away the batch dim; vmap needs it explicit.
        params = params.reshape(n_starts, -1)

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(learning_rate, **adam_kwargs)
        )
        # vmap init/update over the batch axis so each trajectory has its own
        # state and its own (independent) clip norm, instead of one clip norm
        # computed over the flattened (n_starts, n_params) block.
        state = jax.vmap(optimizer.init)(params)
        vmapped_update = jax.vmap(optimizer.update)

        obj.warmup_vmap_value_and_grad()
        obj.start_logging()

        while not obj.budget_exceeded:
            losses, grads = obj.vmap_value_and_grad(params)
            updates, state = vmapped_update(grads, state, params)
            params = optax.apply_updates(params, updates)
