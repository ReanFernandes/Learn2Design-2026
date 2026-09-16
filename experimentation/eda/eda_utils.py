"""Shared helpers for the dataset EDA scripts."""

import h5py
import numpy as np
import pandas as pd

DATASET_PATH = "/Users/reanfernandes/Learn2Design-2026/dataset/dataset.h5"


def load_params_for_topology(
    topology_string: str, path: str = DATASET_PATH
) -> tuple[np.ndarray, np.ndarray]:
    """All bounded_params rows (+ aligned loss) for every entry matching this
    exact topology string. params shape (n_entries, n_params) -- index i means
    the same physical parameter for every row, since a fixed topology fixes
    the parameterization. Returns (params, loss).
    """
    with h5py.File(path, "r") as f:
        entries = f["entries"][:]
        topo = entries["topology_string"].astype(str)
        matching = entries[topo == topology_string]
        if len(matching) == 0:
            raise ValueError(f"no entries found for topology {topology_string!r}")

        pool = f["bounded_params"]
        rows = [
            pool[int(e["param_offset"]) : int(e["param_offset"]) + int(e["param_length"])]
            for e in matching
        ]
    return np.array(rows), np.asarray(matching["loss"])


def load_metadata_df(path: str = DATASET_PATH) -> pd.DataFrame:
    """Flat scalar fields from the `entries` table -- no big pool slicing."""
    with h5py.File(path, "r") as f:
        entries = f["entries"][:]

    return pd.DataFrame(
        {
            "unique_hash": entries["unique_hash"].astype(str),
            "initialized_from": entries["initialized_from"].astype(str),
            "topology_string": entries["topology_string"].astype(str),
            "size": entries["size"],
            "loss": entries["loss"],
            "complexity": entries["complexity"],
            "run_id": entries["run_id"].astype(str),
        }
    )


# Interior codes A-D = beamsplitter (4 orientations), E-H = directional_beamsplitter.
# Boundary codes: L=laser, S=squeezer, D=detector, H=balanced_homodyne.
_INTERIOR_ROLE = {c: "beamsplitter" for c in "ABCD"} | {
    c: "directional_beamsplitter" for c in "EFGH"
}
_BOUNDARY_ROLE = {"L": "laser", "S": "squeezer", "D": "detector", "H": "homodyne"}


def parse_topology(topology_string: str) -> dict:
    """Component-role counts for one topology string, ignoring exact
    position/orientation -- just "how many of each component type"."""
    interior, boundary = topology_string.split("-")

    counts = {
        "n_beamsplitter": 0,
        "n_directional_beamsplitter": 0,
        "n_laser": 0,
        "n_squeezer": 0,
        "n_detector": 0,
        "n_homodyne": 0,
    }
    for ch in interior:
        counts[f"n_{_INTERIOR_ROLE[ch]}"] += 1
    for ch in boundary:
        counts[f"n_{_BOUNDARY_ROLE[ch]}"] += 1
    return counts


def regime_split_normalized_std(
    topology_string: str, threshold: float, path: str = DATASET_PATH
) -> dict:
    """Split a topology's replicate entries into good/common regimes by a
    loss threshold, and compute per-parameter normalized std within each,
    plus pooled. Needs dfbench/differometor -- imported lazily so the rest
    of eda_utils stays lightweight for scripts that don't need it.
    """
    from dfbench import Objective
    from dfbench.problems import UIFOProblem

    params, loss = load_params_for_topology(topology_string, path)
    size = 4 if "-" in topology_string and len(topology_string.split("-")[0]) == 16 else 3

    problem = UIFOProblem(size=size, topology=topology_string)
    obj = Objective(problem)
    bound_range = np.asarray(obj.bounds)[1] - np.asarray(obj.bounds)[0]

    good = params[loss < threshold]
    common = params[loss >= threshold]

    return {
        "topology_string": topology_string,
        "n_total": len(loss),
        "n_good": len(good),
        "n_common": len(common),
        "pooled_std": params.std(axis=0) / bound_range,
        "good_std": good.std(axis=0) / bound_range if len(good) > 1 else None,
        "common_std": common.std(axis=0) / bound_range if len(common) > 1 else None,
    }


# --- Semantic edit-distance ("genotype network" / Hamming graph, weighted) ---
# Interior: same char = 0, same type different orientation (A-D or E-H) = 1,
# different type (beamsplitter vs directional_beamsplitter) = 2.
# Boundary: same char = 0, any difference = 2 (no orientation sub-structure).
_INTERIOR_ALPHABET = "ABCDEFGH"
_BOUNDARY_ALPHABET = "LSDH"


def _cost_matrix(alphabet: str, has_orientation_groups: bool) -> np.ndarray:
    n = len(alphabet)
    m = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if i == j:
                m[i, j] = 0
            elif has_orientation_groups and (i // 4) == (j // 4):
                m[i, j] = 1
            else:
                m[i, j] = 2
    return m


_INTERIOR_COST = _cost_matrix(_INTERIOR_ALPHABET, has_orientation_groups=True)
_BOUNDARY_COST = _cost_matrix(_BOUNDARY_ALPHABET, has_orientation_groups=False)


def encode_topology_string(topology_string: str) -> tuple[np.ndarray, np.ndarray]:
    """Topology string -> (interior_codes, boundary_codes), integer arrays."""
    interior, boundary = topology_string.split("-")
    interior_codes = np.array([_INTERIOR_ALPHABET.index(c) for c in interior], dtype=np.int8)
    boundary_codes = np.array([_BOUNDARY_ALPHABET.index(c) for c in boundary], dtype=np.int8)
    return interior_codes, boundary_codes


def pairwise_semantic_distance(topology_strings: list[str]) -> np.ndarray:
    """Full (n, n) semantic edit-distance matrix across a list of topology
    strings (must all share the same size/grid). Vectorized over positions.
    """
    interior_all = np.stack([encode_topology_string(t)[0] for t in topology_strings])
    boundary_all = np.stack([encode_topology_string(t)[1] for t in topology_strings])
    n = len(topology_strings)

    dist = np.zeros((n, n), dtype=np.int16)
    for p in range(interior_all.shape[1]):
        col = interior_all[:, p]
        dist += _INTERIOR_COST[col[:, None], col[None, :]]
    for p in range(boundary_all.shape[1]):
        col = boundary_all[:, p]
        dist += _BOUNDARY_COST[col[:, None], col[None, :]]
    return dist


def add_topology_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach component-role count columns, derived from `topology_string`."""
    features = pd.DataFrame(
        [parse_topology(t) for t in df["topology_string"]], index=df.index
    )
    return pd.concat([df, features], axis=1)
