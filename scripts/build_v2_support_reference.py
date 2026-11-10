"""Build the source-only v2 unseen-support reference.

This script reads frozen source/configuration only. It never reads checkpoints,
training logs, evaluation rows, or learned summaries. Its enumeration is kept
independent of ``emcomm.combos`` so the generated artifact can audit that
module's unseen sampling support rather than merely restating its output.

Usage: uv run python scripts/build_v2_support_reference.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLOR_NAMES = ("red", "yellow", "green", "blue", "purple")
SHAPE_NAMES = ("cube", "sphere", "cone", "cylinder", "ring")
AUDITED_SOURCES = (
    "configs/v2.toml",
    "src/emcomm/combos.py",
    "docs/PREREGISTRATION_V2.md",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combo_name(combo: tuple[int, int]) -> str:
    return f"{COLOR_NAMES[combo[0]]} {SHAPE_NAMES[combo[1]]}"


def build_reference(root: Path = ROOT) -> dict:
    with open(root / "configs/v2.toml", "rb") as fh:
        env = tomllib.load(fh)["env"]
    n_values, combo_seed = int(env["n_values"]), int(env["combo_seed"])
    if n_values != len(COLOR_NAMES) or n_values != len(SHAPE_NAMES):
        raise ValueError("the audited name tables require exactly five colors and five shapes")

    permutation = np.random.default_rng(combo_seed).permutation(n_values)
    held_out = tuple((color, int(permutation[color])) for color in range(n_values))
    held_set = set(held_out)
    states = []
    for target in held_out:
        other_colors = [color for color in range(n_values) if color != target[0]]
        other_shapes = [shape for shape in range(n_values) if shape != target[1]]
        for other_color in other_colors:
            for other_shape in other_shapes:
                colors = tuple(sorted((target[0], other_color)))
                shapes = tuple(sorted((target[1], other_shape)))
                cells = tuple((color, shape) for color in colors for shape in shapes)
                candidates = tuple(combo for combo in cells if combo in held_set)
                states.append(
                    {
                        "target": combo_name(target),
                        "square": [combo_name(combo) for combo in cells],
                        "held_out_candidates": [combo_name(combo) for combo in candidates],
                        "candidate_count": len(candidates),
                    }
                )

    candidate_counts = {
        str(count): sum(state["candidate_count"] == count for state in states)
        for count in sorted({state["candidate_count"] for state in states})
    }
    expected_identification = sum(1.0 / state["candidate_count"] for state in states) / len(states)
    squares_per_target = (n_values - 1) ** 2
    assert len(states) == n_values * squares_per_target
    assert candidate_counts == {"1": 60, "2": 20}
    assert expected_identification == 0.875

    return {
        "schema_version": 1,
        "kind": "source_only_mathematical_reference",
        "learned_data_used": False,
        "provenance": {
            "generator": "scripts/build_v2_support_reference.py",
            "audited_source_sha256": {path: sha256(root / path) for path in AUDITED_SOURCES},
            "method": (
                "Independent exhaustive enumeration of the frozen unseen target prior: a uniform held-out target "
                "and a uniform 2x2 square containing it."
            ),
        },
        "n_values": n_values,
        "combo_seed": combo_seed,
        "held_out_combinations": [combo_name(combo) for combo in held_out],
        "unseen_prior": {
            "targets": n_values,
            "squares_per_target": squares_per_target,
            "equiprobable_target_square_states": len(states),
            "held_out_candidate_count_distribution": candidate_counts,
        },
        "support_aware_no_message_identification_rate": expected_identification,
        "one_property_plus_matching_identification_rate": 1.0,
        "fixed_h4_reference": 0.5,
        "conclusion": (
            "The fixed 50% H4 reference is not an unseen one-property ceiling. Before navigation, support alone "
            "identifies the target in 60 states and leaves two candidates in 20 states, for 87.5% expected "
            "identification with uniform tie-breaking."
        ),
        "states": states,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/v2/support_reference.json")
    args = parser.parse_args()
    result = build_reference()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
