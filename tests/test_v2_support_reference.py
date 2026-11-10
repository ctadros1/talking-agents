import hashlib
import importlib.util
import json
from pathlib import Path

from emcomm.combos import held_out_combos
from emcomm.config import load_config

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/build_v2_support_reference.py"
ARTIFACT = ROOT / "results/v2/support_reference.json"

spec = importlib.util.spec_from_file_location("build_v2_support_reference", SCRIPT)
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_support_reference_is_reproducible_and_uses_source_only():
    generated = support.build_reference(ROOT)
    assert generated == json.loads(ARTIFACT.read_text())
    assert generated["kind"] == "source_only_mathematical_reference"
    assert generated["learned_data_used"] is False
    assert set(generated["provenance"]["audited_source_sha256"]) == {
        "configs/v2.toml",
        "src/emcomm/combos.py",
        "docs/PREREGISTRATION_V2.md",
    }
    source = SCRIPT.read_text()
    assert "from emcomm" not in source and "import emcomm" not in source
    assert "runs_v2" not in source and "summary.json" not in source and "per_run.csv" not in source


def test_support_reference_exhaustively_proves_the_unseen_prior_counts():
    proof = json.loads(ARTIFACT.read_text())
    states = proof["states"]
    assert len(states) == proof["unseen_prior"]["equiprobable_target_square_states"] == 80
    assert proof["unseen_prior"]["held_out_candidate_count_distribution"] == {"1": 60, "2": 20}
    assert {len(state["square"]) for state in states} == {4}
    assert {state["candidate_count"] for state in states} == {1, 2}
    assert all(state["target"] in state["held_out_candidates"] for state in states)
    expected = sum(1 / state["candidate_count"] for state in states) / len(states)
    assert expected == proof["support_aware_no_message_identification_rate"] == 0.875
    assert proof["fixed_h4_reference"] == 0.5
    assert proof["support_aware_no_message_identification_rate"] > proof["fixed_h4_reference"]


def test_independent_enumeration_matches_frozen_combo_source_and_hashes():
    proof = json.loads(ARTIFACT.read_text())
    cfg, _ = load_config(ROOT / "configs/v2.toml")
    canonical = [f"{support.COLOR_NAMES[c]} {support.SHAPE_NAMES[s]}" for c, s in held_out_combos(cfg)]
    assert proof["held_out_combinations"] == canonical
    for path, recorded in proof["provenance"]["audited_source_sha256"].items():
        assert recorded == _sha256(ROOT / path)
    assert proof["provenance"]["audited_source_sha256"] == {
        "configs/v2.toml": "3e67be8d66766cf100f12d4966ce7878a7ef68417112847c53deb2b03a07336b",
        "src/emcomm/combos.py": "a97dab94b5671b9920374f19f949826853aa69a2d0e26f23e36c569c85423274",
        "docs/PREREGISTRATION_V2.md": "a46f0cdc01a539270d133dc1ffac9ca5ace5a9b3cd11172e3fc171713438cc8f",
    }
