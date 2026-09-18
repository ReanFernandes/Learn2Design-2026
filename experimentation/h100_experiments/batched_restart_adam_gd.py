"""Batched Adam with per-trajectory restart-on-plateau, designed for H100.

Synthesizes three things this session found reason to trust:
  - Batched trajectories via vmap (cheap on H100: ~2.9x cost for a
    batch~15-19 value_and_grad call, per the documented H100 batch-scaling
    chart -- NOT the ~linear cost this same idea showed on CPU).
  - Restart-with-perturbation on plateau, alternating perturbed-best vs.
    fresh-random -- the same pattern BFGS's restart loop uses here, that
    NAAdamGD (noise injection) uses, that Urania (the source PRX paper's
    discovery engine) uses, and that this dataset's own "Reoptimized"
    category was generated with. Independently corroborated from four
    directions, not just a dataset correlation.
  - Tuned hyperparameters from experimentation/local_experiments/01_adamgd_hpo_voyager
    (lr, b1, b2) as a starting point -- NOT re-validated on H100/UIFO yet,
    treat as a reasonable prior, not a proven-optimal choice.

Each of the n_starts trajectories tracks its own best loss/params and its
own steps-since-improvement counter. A trajectory that plateaus for
`patience` steps gets reset independently of the others -- the batch as a
whole never stalls just because one member got stuck.

NOTE ON LOCAL TESTING: this has only been run locally on CPU for
correctness (does it run, does it produce sane output) -- NOT for
performance. Batch width, patience, and perturbation scale all need real
calibration on H100 before trusting any specific setting. See
experimentation/h100_experiments/PLAN.md.
"""

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

from dfbench import OptimizationAlgorithm, Objective

# From experimentation/local_experiments/01_adamgd_hpo_voyager.md (random-search
# best on ConstrainedVoyagerProblem, CPU, short budget). Starting prior only.
TUNED_LR = 0.25771
TUNED_B1 = 0.9282
TUNED_B2 = 0.95808


class BatchedRestartAdamGD(OptimizationAlgorithm):
    """n_starts parallel Adam trajectories, each independently restarted
    (perturbed-best or fresh-random, alternating) after `patience` steps
    without improvement."""

    algorithm_str = "batched_restart_adam_gd"

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        n_starts: int = 16,
        learning_rate: float = TUNED_LR,
        b1: float = TUNED_B1,
        b2: float = TUNED_B2,
        patience: int = 200,
        perturbation_scale: float = 0.3,
        **adam_kwargs,
    ) -> None:
        obj = objective
        self.prepare(obj, unbounded=True, random_seed=random_seed)

        params = (
            init_params
            if init_params is not None
            else obj.random_params_unbounded(n_samples=n_starts)
        )
        params = params.reshape(n_starts, -1)
        n_params = params.shape[1]

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate, b1=b1, b2=b2, **adam_kwargs),
        )
        vmapped_init = jax.vmap(optimizer.init)
        vmapped_update = jax.vmap(optimizer.update)
        state = vmapped_init(params)

        best_params = params
        best_loss = jnp.full((n_starts,), jnp.inf)
        steps_since_improvement = jnp.zeros((n_starts,), dtype=jnp.int32)
        restart_count = jnp.zeros((n_starts,), dtype=jnp.int32)

        obj.warmup_vmap_value_and_grad(batch_size=n_starts)
        obj.start_logging()

        while not obj.budget_exceeded:
            losses, grads = obj.vmap_value_and_grad(params)

            improved = losses < best_loss
            best_loss = jnp.where(improved, losses, best_loss)
            best_params = jnp.where(improved[:, None], params, best_params)
            steps_since_improvement = jnp.where(
                improved, 0, steps_since_improvement + 1
            )

            updates, state = vmapped_update(grads, state, params)
            params = optax.apply_updates(params, updates)

            needs_restart = steps_since_improvement >= patience
            if jnp.any(needs_restart):
                key = jax.random.PRNGKey(
                    int(jnp.sum(restart_count)) + int(obj.eval_count)
                )
                fresh = obj.random_params_unbounded(n_samples=n_starts).reshape(
                    n_starts, -1
                )
                noise = perturbation_scale * jax.random.normal(key, params.shape)
                perturbed = best_params + noise
                # Alternate perturbed-best vs. fresh-random per trajectory,
                # same exploit/explore alternation BFGS's restart loop uses.
                use_fresh = (restart_count % 2) == 1
                restart_target = jnp.where(use_fresh[:, None], fresh, perturbed)

                mask = needs_restart[:, None]
                params = jnp.where(mask, restart_target, params)
                fresh_state = vmapped_init(params)
                state = jax.tree.map(
                    lambda new, old: jnp.where(
                        needs_restart.reshape((-1,) + (1,) * (new.ndim - 1)),
                        new,
                        old,
                    ),
                    fresh_state,
                    state,
                )
                steps_since_improvement = jnp.where(needs_restart, 0, steps_since_improvement)
                restart_count = jnp.where(needs_restart, restart_count + 1, restart_count)
