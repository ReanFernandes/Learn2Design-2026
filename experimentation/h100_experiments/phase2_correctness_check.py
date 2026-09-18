"""Phase 2 -- correctness re-check of BatchedRestartAdamGD on real GPU.

Confirms the restart-on-plateau logic behaves the same as the local-CPU
check (does it run, no NaN/crash, restart actually fires) -- not a
performance test. See PLAN.md Phase 2.
"""

from dfbench import Objective
from dfbench.problems import ConstrainedVoyagerProblem

from experimentation.h100_experiments.batched_restart_adam_gd import BatchedRestartAdamGD

obj = Objective(ConstrainedVoyagerProblem(), max_time=30, save=["is_feasible"])
algo = BatchedRestartAdamGD()
algo.optimize(obj, random_seed=1, n_starts=16, patience=200)

print(f"eval_count={obj.eval_count}")
print(f"best_loss={obj.best_loss}")
print(f"best_is_feasible={obj.best_is_feasible}")
