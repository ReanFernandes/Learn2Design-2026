"""How does obj.vmap_value_and_grad latency scale with batch size on CPU (Apple M4)?"""

import time

from dfbench import Objective
from dfbench.problems import ConstrainedVoyagerProblem

for batch_size in [1, 2, 4, 8, 16, 32]:
    problem = ConstrainedVoyagerProblem()
    objective = Objective(problem, max_evals=batch_size * 100, max_time=300, verbose=0)
    params_batch = objective.random_params_unbounded(n_samples=batch_size).reshape(
        batch_size, -1
    )

    objective.warmup_vmap_value_and_grad()
    objective.start_logging()

    t0 = time.time()
    n_calls = 10
    for _ in range(n_calls):
        losses, grads = objective.vmap_value_and_grad(params_batch)
        losses.block_until_ready()
    elapsed = time.time() - t0

    per_call = elapsed / n_calls
    per_candidate = per_call / batch_size
    print(
        f"batch={batch_size:3d} | per_call={per_call * 1000:7.1f}ms | "
        f"per_candidate={per_candidate * 1000:7.2f}ms | "
        f"candidates/sec={batch_size / per_call:8.1f}"
    )
