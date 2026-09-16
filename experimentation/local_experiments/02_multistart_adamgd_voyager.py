"""AdamGD (single trajectory) vs MultiStartAdamGD (n_starts parallel
trajectories via vmap), same wall-clock budget, ConstrainedVoyagerProblem.

Uses the tuned hyperparameters found in 01_adamgd_hpo_voyager (lr, b1, b2) for
all configs, isolating the "one deep trajectory vs many shallow parallel
trajectories under the same wall-clock budget" question.

See 02_multistart_adamgd_voyager.md for the write-up.
"""

import time

from dfbench import Objective
from dfbench.problems import ConstrainedVoyagerProblem

from learn2design.example_algorithms import AdamGD
from multistart_adam_gd import MultiStartAdamGD

# hyperparameters from 01_adamgd_hpo_voyager.md (random-search best)
LR = 0.25771
B1 = 0.9282
B2 = 0.95808

BUDGET_SECONDS = 60
N_SEEDS = 5
N_STARTS_VALUES = [1, 8, 16]  # 1 = plain AdamGD baseline


def best_feasible_loss(objective: Objective) -> float | None:
    feasible = [
        float(loss)
        for loss, feas in zip(objective.loss_history, objective.is_feasible_history)
        if feas
    ]
    return min(feasible) if feasible else None


def run_once(n_starts: int, seed: int) -> tuple[float | None, int]:
    problem = ConstrainedVoyagerProblem()
    objective = Objective(
        problem, max_time=BUDGET_SECONDS, save=["is_feasible"], verbose=0
    )
    if n_starts == 1:
        algo = AdamGD()
        algo.optimize(objective, random_seed=seed, learning_rate=LR, b1=B1, b2=B2)
    else:
        algo = MultiStartAdamGD()
        algo.optimize(
            objective, random_seed=seed, learning_rate=LR, b1=B1, b2=B2, n_starts=n_starts
        )
    return best_feasible_loss(objective), objective.eval_count


def main() -> None:
    print(f"ConstrainedVoyagerProblem | budget={BUDGET_SECONDS}s | {N_SEEDS} seeds")
    print(f"hyperparams: lr={LR} b1={B1} b2={B2}\n")

    summary = []
    for n_starts in N_STARTS_VALUES:
        label = "AdamGD (single)" if n_starts == 1 else f"MultiStartAdamGD (n={n_starts})"
        print(f"{'=' * 70}\n{label}\n{'=' * 70}")

        results = []
        for s in range(N_SEEDS):
            seed = 2000 + s
            t0 = time.time()
            loss, n_evals = run_once(n_starts, seed)
            wall = time.time() - t0
            results.append((loss, n_evals))
            loss_str = f"{loss:.4f}" if loss is not None else "INFEASIBLE"
            print(f"  seed={seed} best_feasible={loss_str} n_evals={n_evals:6d} wall={wall:.1f}s")

        feasible_losses = [l for l, _ in results if l is not None]
        n_feasible_runs = len(feasible_losses)
        mean_loss = sum(feasible_losses) / n_feasible_runs if feasible_losses else None
        mean_evals = sum(n for _, n in results) / len(results)
        mean_str = f"{mean_loss:.4f}" if mean_loss is not None else "N/A"
        print(
            f"--- {label}: feasible_runs={n_feasible_runs}/{N_SEEDS} "
            f"mean_loss={mean_str} mean_evals={mean_evals:.0f} ---\n"
        )
        summary.append((label, n_feasible_runs, mean_loss, mean_evals))

    print(f"{'=' * 70}\nFINAL SUMMARY\n{'=' * 70}")
    for label, n_feasible, mean_loss, mean_evals in summary:
        mean_str = f"{mean_loss:.4f}" if mean_loss is not None else "N/A"
        print(
            f"{label:28s} | feasible={n_feasible}/{N_SEEDS} | "
            f"mean_loss={mean_str} | mean_evals={mean_evals:.0f}"
        )


if __name__ == "__main__":
    main()
