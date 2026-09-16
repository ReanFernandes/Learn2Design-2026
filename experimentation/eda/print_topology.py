"""Pretty-print a UIFO topology string as an ASCII grid, so you can read a
string like "ABBGDGBEA-SLLLSDSLSLSL" without doing the position arithmetic
by hand. Position ordering verified against dfbench's own
topology_from_string / _interior_positions / _boundary_positions.

Usage: edit TOPOLOGY/SIZE below, or import parse_and_print(topology, size).
"""

from dfbench.problems.uifo.uifo_problem import topology_from_string

_CENTER_ABBR = {"beamsplitter": "BS", "directional_beamsplitter": "DBS"}
_ORIENT_ARROW = {"left": "<", "right": ">", "top": "^", "bottom": "v"}
_BOUNDARY_ABBR = {
    "laser": "LASER",
    "squeezer": "SQZ",
    "detector": "DET",
    "balanced_homodyne": "HOMO",
}


def parse_and_print(topology: str, size: int) -> None:
    centers, boundaries = topology_from_string(topology, size)
    grid = size + 2

    cell_w = 9
    print(f"topology: {topology}  (size={size})\n")

    for r in range(grid):
        row_cells = []
        for c in range(grid):
            pos = f"{r}{c}"
            if r == 0 or r == grid - 1 or c == 0 or c == grid - 1:
                # boundary ring (corners are empty -- not part of the grid)
                if pos in boundaries:
                    label = _BOUNDARY_ABBR[boundaries[pos]]
                else:
                    label = ""
            else:
                comp, orient = centers[pos]
                label = f"{_CENTER_ABBR[comp]}{_ORIENT_ARROW[orient]}"
            row_cells.append(label.center(cell_w))
        print("|".join(row_cells))
    print()

    n_bs = sum(1 for c, _ in centers.values() if c == "beamsplitter")
    n_dbs = sum(1 for c, _ in centers.values() if c == "directional_beamsplitter")
    from collections import Counter

    b_counts = Counter(boundaries.values())
    print(f"interior: {n_bs} beamsplitter, {n_dbs} directional_beamsplitter")
    print(f"boundary: {dict(b_counts)}")


if __name__ == "__main__":
    parse_and_print("ABBGDGBEA-SLLLSDSLSLSL", size=3)
