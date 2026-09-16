"""HPO over AdamGD's (learning_rate, b1, b2), scored on ConstrainedVoyagerProblem.

Single-fidelity: fixed short budget, fixed seed set, no clip-norm tuning
(hardcoded in AdamGD, left untouched). Compares plain random search against a
light BoTorch GP + qLogEI loop, same total trial budget for both.

See 01_adamgd_hpo_voyager.md for the write-up.
"""

import math
import time

import torch
from botorch.acquisition import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

from dfbench import Objective
from dfbench.problems import ConstrainedVoyagerProblem
from learn2design.example_algorithms import AdamGD

# --- search space: u in [0,1]^3, log-uniform mapped to real ranges ---
# LR lower bound calibrated empirically: lr=0.01/0.05 never reached feasibility
# within 45s on this problem (0/N feasible evals at any tested budget), so that
# region is a dead zone for this fidelity. Narrowed to stay in the informative
# range. See 01_adamgd_hpo_voyager.md.
LR_RANGE = (0.05, 0.3)
ONE_MINUS_B1_RANGE = (0.02, 0.5)
ONE_MINUS_B2_RANGE = (1e-4, 1e-1)

N_SEEDS = 3
INNER_BUDGET_SECONDS = 30
N_TRIALS = 12
N_INITIAL = 5  # Sobol init points for BO, within N_TRIALS
INFEASIBLE_PENALTY = 10.0


def _log_uniform(u: float, low: float, high: float) -> float:
    return math.exp(math.log(low) + u * (math.log(high) - math.log(low)))


def u_to_params(u: torch.Tensor) -> tuple[float, float, float]:
    lr = _log_uniform(u[0].item(), *LR_RANGE)
    b1 = 1.0 - _log_uniform(u[1].item(), *ONE_MINUS_B1_RANGE)
    b2 = 1.0 - _log_uniform(u[2].item(), *ONE_MINUS_B2_RANGE)
    return lr, b1, b2


def run_adamgd_once(lr: float, b1: float, b2: float, seed: int) -> float:
    problem = ConstrainedVoyagerProblem()
    objective = Objective(
        problem, max_time=INNER_BUDGET_SECONDS, save=["is_feasible"], verbose=0
    )
    algo = AdamGD()
    algo.optimize(objective, random_seed=seed, learning_rate=lr, b1=b1, b2=b2)

    feasible_losses = [
        float(loss)
        for loss, feasible in zip(objective.loss_history, objective.is_feasible_history)
        if feasible
    ]
    return min(feasible_losses) if feasible_losses else INFEASIBLE_PENALTY


def meta_objective(u: torch.Tensor) -> float:
    lr, b1, b2 = u_to_params(u)
    losses = [run_adamgd_once(lr, b1, b2, seed=1000 + s) for s in range(N_SEEDS)]
    score = sum(losses) / len(losses)
    print(
        f"  lr={lr:.5f} b1={b1:.4f} b2={b2:.5f} -> "
        f"seed_losses={[f'{l:.3f}' for l in losses]} mean={score:.4f}"
    )
    return score


def random_search(n_trials: int) -> list[tuple[torch.Tensor, float]]:
    print(f"\n{'=' * 70}\nRANDOM SEARCH ({n_trials} trials)\n{'=' * 70}")
    torch.manual_seed(0)
    results = []
    for i in range(n_trials):
        u = torch.rand(3)
        t0 = time.time()
        print(f"[{i + 1}/{n_trials}]")
        score = meta_objective(u)
        results.append((u, score))
        print(f"  wall={time.time() - t0:.1f}s")
    return results


def bo_search(n_trials: int, n_initial: int) -> list[tuple[torch.Tensor, float]]:
    print(f"\n{'=' * 70}\nBOTORCH GP + qLogEI ({n_trials} trials, {n_initial} initial)\n{'=' * 70}")
    bounds = torch.stack([torch.zeros(3), torch.ones(3)])

    init_u = draw_sobol_samples(bounds=bounds, n=n_initial, q=1, seed=0).squeeze(1)
    results: list[tuple[torch.Tensor, float]] = []
    for i in range(n_initial):
        print(f"[init {i + 1}/{n_initial}]")
        score = meta_objective(init_u[i])
        results.append((init_u[i], score))

    train_x = torch.stack([u for u, _ in results])
    # GP maximizes; we minimize loss, so feed negative loss.
    train_y = torch.tensor([[-s] for _, s in results])

    for i in range(n_trials - n_initial):
        gp = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        acqf = qLogExpectedImprovement(gp, best_f=train_y.max())
        candidate, _ = optimize_acqf(
            acqf, bounds=bounds, q=1, num_restarts=10, raw_samples=256
        )
        u_new = candidate.squeeze(0)

        print(f"[bo {i + 1}/{n_trials - n_initial}]")
        score = meta_objective(u_new)
        results.append((u_new, score))

        train_x = torch.cat([train_x, u_new.unsqueeze(0)])
        train_y = torch.cat([train_y, torch.tensor([[-score]])])

    return results


def summarize(name: str, results: list[tuple[torch.Tensor, float]]) -> None:
    best_u, best_score = min(results, key=lambda r: r[1])
    lr, b1, b2 = u_to_params(best_u)
    print(f"\n--- {name} BEST ---")
    print(f"lr={lr:.5f} b1={b1:.4f} b2={b2:.5f} -> mean_loss={best_score:.4f}")


if __name__ == "__main__":
    print(
        f"ConstrainedVoyagerProblem | {N_SEEDS} seeds x {INNER_BUDGET_SECONDS}s "
        f"| {N_TRIALS} trials each method"
    )

    rs_results = random_search(N_TRIALS)
    summarize("RANDOM SEARCH", rs_results)

    bo_results = bo_search(N_TRIALS, N_INITIAL)
    summarize("BO", bo_results)

    print(f"\n{'=' * 70}\nFINAL SUMMARY\n{'=' * 70}")
    summarize("RANDOM SEARCH", rs_results)
    summarize("BO", bo_results)
