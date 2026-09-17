"""How does obj.vmap_value_and_grad latency scale with batch size on GPU?

Despite the filename (kept for continuity with earlier runs/notes), this is
backend-agnostic -- runs on whatever device JAX finds.

Usage:
    python scratch_batch_scaling_cpu.py voyager   # ConstrainedVoyagerProblem (default)
    python scratch_batch_scaling_cpu.py uifo      # UIFOProblem(size=3) -- the real target problem
"""

import sys
import time

from dfbench import Objective
from dfbench.problems import ConstrainedVoyagerProblem, UIFOProblem

problem_name = sys.argv[1] if len(sys.argv) > 1 else "voyager"

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

for batch_size in batch_sizes:
    if problem_name == "uifo":
        problem = UIFOProblem(size=3, topology_seed=0)
    else:
        problem = ConstrainedVoyagerProblem()

    objective = Objective(problem, max_evals=batch_size * 100, max_time=300, verbose=0)
    params_batch = objective.random_params_unbounded(n_samples=batch_size).reshape(
        batch_size, -1
    )

    try:
        objective.warmup_vmap_value_and_grad(batch_size=batch_size)
        objective.start_logging()

        t0 = time.time()
        n_calls = 10
        for _ in range(n_calls):
            losses, grads = objective.vmap_value_and_grad(params_batch)
            losses.block_until_ready()
        elapsed = time.time() - t0
    except Exception as e:
        print(f"batch={batch_size:4d} | FAILED: {type(e).__name__}: {str(e)[:200]}")
        print("STOPPING_SWEEP")
        break

    per_call = elapsed / n_calls
    per_candidate = per_call / batch_size
    print(
        f"batch={batch_size:4d} | per_call={per_call * 1000:7.1f}ms | "
        f"per_candidate={per_candidate * 1000:7.2f}ms | "
        f"candidates/sec={batch_size / per_call:8.1f}"
    )
