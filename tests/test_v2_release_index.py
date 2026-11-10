import csv
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest

from emcomm import benchmark_v2
from emcomm.channel import TRAIN_CHANNEL
from emcomm.config import CONDITIONS

PROJECT_ROOT = Path(__file__).parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "build_v2_release_index.py"
SPEC = importlib.util.spec_from_file_location("build_v2_release_index", SCRIPT)
assert SPEC and SPEC.loader
release_index = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_index)

N_EPISODES = 4
SEEDS = tuple(range(20))
FROZEN_PATHS = (
    "configs/v2.toml",
    "docs/PREREGISTRATION_V2.md",
    "src/emcomm/benchmark_v2.py",
    "src/emcomm/analysis_v2.py",
    "src/emcomm/stats.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episode(condition: str, seed: int, split: str, intervention: str, index: int) -> dict:
    color = (seed + index) % 5
    shape = (seed + 2 * index + (split == "unseen")) % 5
    success = index < N_EPISODES // 2
    picked = [color, shape] if success else [(color + 1) % 5, shape]
    steps = 8 + index
    runner_opt = 5
    return {
        "condition": condition,
        "seed": seed,
        "split": split,
        "intervention": intervention,
        "episode": index,
        "layout_id": 0 if split == "train" else 48,
        "target": index % 4,
        "success": success,
        "return": 0.8 if success else -0.2,
        "steps": steps,
        "runner_opt": runner_opt,
        "path_efficiency": runner_opt / steps if success else float("nan"),
        "runner_pick": index % 4 if success else (index + 1) % 4,
        "target_attrs": [color, shape],
        "picked_attrs": picked,
        "scout_msgs": [[color, shape] for _ in range(steps)],
    }


def _write_sources(root: Path) -> dict[str, bytes]:
    snapshot = {}
    for relative in FROZEN_PATHS:
        source = PROJECT_ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        snapshot[relative] = destination.read_bytes()
    return snapshot


def _fixture(root: Path) -> dict[str, bytes]:
    snapshot = _write_sources(root)
    table_rows = []

    for name, (base_condition, _) in benchmark_v2.SCRIPTED.items():
        condition = f"scripted_{name}"
        intervention = TRAIN_CHANNEL[base_condition]
        for split in benchmark_v2.SPLITS:
            records = [_episode(condition, -1, split, intervention, index) for index in range(N_EPISODES)]
            table_rows.append(
                {
                    "condition": condition,
                    "seed": -1,
                    "split": split,
                    "intervention": intervention,
                    **benchmark_v2.summarize(records, n_values=5),
                }
            )

    for condition in CONDITIONS:
        for seed in SEEDS:
            run_dir = root / "runs_v2" / condition / f"seed_{seed:04d}"
            run_dir.mkdir(parents=True)
            (run_dir / "final.pt").write_bytes(f"synthetic {condition} {seed}\n".encode())
            records_by_pair = {}
            all_records = []
            for split, intervention in benchmark_v2.eval_plan(condition):
                records = [_episode(condition, seed, split, intervention, index) for index in range(N_EPISODES)]
                records_by_pair[(split, intervention)] = records
                all_records.extend(records)
            with (run_dir / "eval.jsonl").open("w") as fh:
                fh.writelines(json.dumps(record) + "\n" for record in all_records)

            pooled = [
                *records_by_pair[("held_out", TRAIN_CHANNEL[condition])],
                *records_by_pair[("unseen", TRAIN_CHANNEL[condition])],
            ]
            run_level = benchmark_v2.run_level(pooled)
            for (split, intervention), records in records_by_pair.items():
                table_rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "split": split,
                        "intervention": intervention,
                        **benchmark_v2.summarize(records, n_values=5),
                        **run_level,
                    }
                )

    table = root / "results/v2/per_run.csv"
    table.parent.mkdir(parents=True)
    with table.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=benchmark_v2.RUN_FIELDS_V2)
        writer.writeheader()
        writer.writerows(table_rows)
    return snapshot


def _build(root: Path, snapshot: dict[str, bytes], run_auditor=None):
    calls = []

    def audit(*args):
        calls.append(args)

    index = release_index._build_v2_release_index(
        root=root,
        run_auditor=run_auditor or audit,
        resolve_ref=lambda _root, _ref: release_index.EXPECTED_PREREG_COMMIT,
        read_ref_file=lambda _root, _ref, relative: snapshot[relative],
        bind_validator_sources=False,
        n_eval_episodes=N_EPISODES,
    )
    return index, calls


def _evaluation(root: Path) -> Path:
    return root / "runs_v2/comm/seed_0000/eval.jsonl"


def _mutate_first_record(root: Path, mutate) -> None:
    path = _evaluation(root)
    lines = path.read_text().splitlines()
    record = json.loads(lines[0])
    mutate(record)
    lines[0] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n")


def _prepend_duplicate_member(root: Path, member: str) -> None:
    path = _evaluation(root)
    lines = path.read_text().splitlines()
    assert lines[0].startswith("{")
    lines[0] = "{" + member + "," + lines[0][1:]
    path.write_text("\n".join(lines) + "\n")


def test_builds_closed_index_with_lower_median_tie_and_exact_digests(tmp_path):
    snapshot = _fixture(tmp_path)

    index, calls = _build(tmp_path, snapshot)

    assert len(calls) == 80
    assert index == {
        "schema_version": "v2-release-input-index-1",
        "release_tag": "v2.0",
        "config": {"path": "configs/v2.toml", "sha256": _sha256(tmp_path / "configs/v2.toml")},
        "results_table": {
            "path": "results/v2/per_run.csv",
            "sha256": _sha256(tmp_path / "results/v2/per_run.csv"),
        },
        "representative": {
            "condition": "comm",
            "seed": 9,
            "checkpoint": {
                "path": "runs_v2/comm/seed_0009/final.pt",
                "sha256": _sha256(tmp_path / "runs_v2/comm/seed_0009/final.pt"),
            },
            "evaluation": {
                "path": "runs_v2/comm/seed_0009/eval.jsonl",
                "sha256": _sha256(tmp_path / "runs_v2/comm/seed_0009/eval.jsonl"),
            },
        },
    }
    output = tmp_path / "results/v2/release_index.json"
    assert json.loads(output.read_text()) == index
    assert type(index["representative"]["seed"]) is int

    original = output.read_bytes()
    second, _ = _build(tmp_path, snapshot)
    assert second == index
    assert output.read_bytes() == original


@pytest.mark.parametrize("fault", ["missing", "unexpected"])
def test_requires_exactly_the_80_canonical_final_checkpoint_paths(tmp_path, fault):
    snapshot = _fixture(tmp_path)
    if fault == "missing":
        (tmp_path / "runs_v2/oracle/seed_0019/final.pt").unlink()
    else:
        extra = tmp_path / "runs_v2/other/seed_0000/final.pt"
        extra.parent.mkdir(parents=True)
        extra.write_bytes(b"unexpected")

    with pytest.raises(release_index.ReleaseIndexError, match="checkpoint set"):
        _build(tmp_path, snapshot)


def test_rejects_config_that_is_stale_against_prereg_v2(tmp_path):
    snapshot = _fixture(tmp_path)
    config = tmp_path / "configs/v2.toml"
    config.write_bytes(config.read_bytes() + b"\n# stale\n")

    with pytest.raises(release_index.ReleaseIndexError, match="differs from prereg-v2"):
        _build(tmp_path, snapshot)


def test_rejects_stale_table_value_that_does_not_match_evaluation(tmp_path):
    snapshot = _fixture(tmp_path)
    table = tmp_path / "results/v2/per_run.csv"
    with table.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        if (
            row["condition"] == "comm"
            and row["seed"] == "0"
            and row["split"] == "held_out"
            and row["intervention"] == "intact"
        ):
            row["success"] = "0.75"
            break
    with table.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=benchmark_v2.RUN_FIELDS_V2)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(release_index.ReleaseIndexError, match="does not match eval.jsonl"):
        _build(tmp_path, snapshot)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "out_of_range", "boolean"])
def test_rejects_incomplete_or_nonstrict_episode_identity(tmp_path, fault):
    snapshot = _fixture(tmp_path)

    def mutate(record):
        if fault == "missing":
            record.pop("episode")
        elif fault == "duplicate":
            record["episode"] = 1
        elif fault == "out_of_range":
            record["episode"] = N_EPISODES
        else:
            record["episode"] = True

    _mutate_first_record(tmp_path, mutate)

    with pytest.raises(release_index.ReleaseIndexError, match="episode"):
        _build(tmp_path, snapshot)


@pytest.mark.parametrize(
    "member",
    [
        '"condition":"bogus"',
        '"return":NaN',
        '"extra":{"nested":1,"nested":2}',
    ],
)
def test_rejects_duplicate_json_members(tmp_path, member):
    snapshot = _fixture(tmp_path)
    _prepend_duplicate_member(tmp_path, member)

    with pytest.raises(release_index.ReleaseIndexError, match="duplicate JSON member"):
        _build(tmp_path, snapshot)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("condition", 1),
        ("seed", True),
        ("split", 1),
        ("intervention", 1),
        ("layout_id", True),
        ("target", True),
        ("target_attrs", [0, True]),
    ],
)
def test_rejects_wrong_v2_record_types(tmp_path, field, value):
    snapshot = _fixture(tmp_path)
    _mutate_first_record(tmp_path, lambda record: record.__setitem__(field, value))

    with pytest.raises(release_index.ReleaseIndexError, match=field):
        _build(tmp_path, snapshot)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("return", float("nan")),
        ("path_efficiency", float("inf")),
    ],
)
def test_rejects_nonfinite_evaluation_fields_outside_failed_path_sentinel(tmp_path, field, value):
    snapshot = _fixture(tmp_path)
    _mutate_first_record(tmp_path, lambda record: record.__setitem__(field, value))

    with pytest.raises(release_index.ReleaseIndexError, match="non-finite evaluation"):
        _build(tmp_path, snapshot)


def test_rejects_symlink_traversal_for_evaluation(tmp_path):
    snapshot = _fixture(tmp_path)
    evaluation = tmp_path / "runs_v2/comm/seed_0009/eval.jsonl"
    target = tmp_path / "outside-eval.jsonl"
    evaluation.replace(target)
    evaluation.symlink_to(target)

    with pytest.raises(release_index.ReleaseIndexError, match="symlink"):
        _build(tmp_path, snapshot)


def test_refuses_existing_different_index(tmp_path):
    snapshot = _fixture(tmp_path)
    output = tmp_path / "results/v2/release_index.json"
    output.write_text('{"different":true}\n')

    with pytest.raises(release_index.ReleaseIndexError, match="different content"):
        _build(tmp_path, snapshot)
    assert output.read_text() == '{"different":true}\n'


def test_atomic_publish_leaves_no_partial_output_on_link_failure(tmp_path, monkeypatch):
    snapshot = _fixture(tmp_path)
    output = tmp_path / "results/v2/release_index.json"

    def fail_link(_source, _destination):
        raise OSError("injected publish failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(release_index.ReleaseIndexError, match="atomic publish"):
        _build(tmp_path, snapshot)
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.*"))


def test_cli_help_is_available(capsys):
    with pytest.raises(SystemExit) as exc:
        release_index._build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    assert "release_index.json" in capsys.readouterr().out
