"""Validate the complete v2 release inputs and write results/v2/release_index.json.

The command is intentionally fixed to the pre-registered v2.0 experiment: four
learned conditions, seeds 0-19, configs/v2.toml, runs_v2, and
results/v2/per_run.csv. It does not train, evaluate, select a checkpoint, or
modify any input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

PACKAGE_CHECKPOINTS_PATH = Path(__file__).resolve().with_name("package_checkpoints.py")
PACKAGE_CHECKPOINTS_SPEC = importlib.util.spec_from_file_location(
    "_v2_release_package_checkpoints", PACKAGE_CHECKPOINTS_PATH
)
if PACKAGE_CHECKPOINTS_SPEC is None or PACKAGE_CHECKPOINTS_SPEC.loader is None:
    raise ImportError(f"cannot load checkpoint validator: {PACKAGE_CHECKPOINTS_PATH}")
package_checkpoints = importlib.util.module_from_spec(PACKAGE_CHECKPOINTS_SPEC)
PACKAGE_CHECKPOINTS_SPEC.loader.exec_module(package_checkpoints)
PackagingError = package_checkpoints.PackagingError
_bind_validator_sources = package_checkpoints._bind_validator_sources
_lexical_path = package_checkpoints._lexical_path
_regular_file = package_checkpoints._regular_file
_run_git = package_checkpoints._run_git
_sha256_file = package_checkpoints._sha256_file
_validate_run = package_checkpoints._validate_run
_verify_imported_validator_sources = package_checkpoints._verify_imported_validator_sources


SCHEMA_VERSION = "v2-release-input-index-1"
RELEASE_TAG = "v2.0"
PREREG_REF = "prereg-v2"
EXPECTED_PREREG_COMMIT = "5aaea7acac7493cc04955a83971e9c1239f9d7a5"
EXPECTED_CONFIG_SHA256 = "3e67be8d66766cf100f12d4966ce7878a7ef68417112847c53deb2b03a07336b"
CONFIG_PATH = "configs/v2.toml"
TABLE_PATH = "results/v2/per_run.csv"
OUTPUT_PATH = "results/v2/release_index.json"
RUNS_PATH = "runs_v2"
CONDITIONS = ("comm", "no_comm", "random_msg", "oracle")
SEEDS = tuple(range(20))
FROZEN_PATHS = (
    CONFIG_PATH,
    "docs/PREREGISTRATION_V2.md",
    "src/emcomm/benchmark_v2.py",
    "src/emcomm/analysis_v2.py",
    "src/emcomm/stats.py",
)
EVALUATION_RECORD_FIELDS = frozenset(
    {
        "condition",
        "seed",
        "split",
        "intervention",
        "episode",
        "layout_id",
        "target",
        "success",
        "return",
        "steps",
        "runner_opt",
        "path_efficiency",
        "runner_pick",
        "scout_msgs",
        "target_attrs",
        "picked_attrs",
    }
)

ResolveRef = Callable[[Path, str], str]
ReadRefFile = Callable[[Path, str, str], bytes]
RunAuditor = Callable[[Path, Path, str, int, Any, Any], Any]


class ReleaseIndexError(PackagingError):
    """A release input failed a strict v2 provenance or completeness check."""


class _DuplicateJsonMember(ValueError):
    pass


def _resolve_git_ref(root: Path, ref: str) -> str:
    return _run_git(root, "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}").decode().strip()


def _read_git_file(root: Path, ref: str, relative: str) -> bytes:
    return _run_git(root, "show", f"{ref}:{relative}")


def _checked_file(root: Path, path: Path) -> Path:
    try:
        return _regular_file(root, path)
    except PackagingError as exc:
        raise ReleaseIndexError(str(exc)) from exc


def _checked_path(root: Path, path: Path, *, must_exist: bool = True) -> Path:
    try:
        return _lexical_path(root, path, must_exist=must_exist)
    except PackagingError as exc:
        raise ReleaseIndexError(str(exc)) from exc


def _verify_frozen_sources(root: Path, resolve_ref: ResolveRef, read_ref_file: ReadRefFile) -> bytes:
    try:
        resolved = resolve_ref(root, PREREG_REF)
    except (OSError, PackagingError) as exc:
        raise ReleaseIndexError(f"cannot resolve {PREREG_REF}: {exc}") from exc
    if resolved != EXPECTED_PREREG_COMMIT:
        raise ReleaseIndexError(
            f"{PREREG_REF} resolves to {resolved!r}, expected immutable commit {EXPECTED_PREREG_COMMIT}"
        )

    config_bytes = b""
    for relative in FROZEN_PATHS:
        current_path = _checked_file(root, root / relative)
        current = current_path.read_bytes()
        try:
            frozen = read_ref_file(root, PREREG_REF, relative)
        except (KeyError, OSError, PackagingError) as exc:
            raise ReleaseIndexError(f"cannot read {relative} from {PREREG_REF}: {exc}") from exc
        if current != frozen:
            raise ReleaseIndexError(f"{relative} differs from {PREREG_REF}")
        if relative == CONFIG_PATH:
            config_bytes = current

    config_digest = hashlib.sha256(config_bytes).hexdigest()
    if config_digest != EXPECTED_CONFIG_SHA256:
        raise ReleaseIndexError(
            f"{CONFIG_PATH} has SHA-256 {config_digest}, expected frozen digest {EXPECTED_CONFIG_SHA256}"
        )
    return config_bytes


def _expected_final_paths(root: Path) -> set[Path]:
    return {
        root / RUNS_PATH / condition / f"seed_{seed:04d}" / "final.pt" for condition in CONDITIONS for seed in SEEDS
    }


def _verify_checkpoint_set(root: Path) -> None:
    expected = _expected_final_paths(root)
    runs = _checked_path(root, root / RUNS_PATH)
    if not runs.is_dir():
        raise ReleaseIndexError(f"expected run directory: {runs}")
    found = set(runs.glob("*/seed_*/final.pt"))
    if found != expected:
        missing = sorted(path.relative_to(root).as_posix() for path in expected - found)
        unexpected = sorted(path.relative_to(root).as_posix() for path in found - expected)
        raise ReleaseIndexError(
            f"final checkpoint set is not the exact canonical 80 paths; missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )


def _strict_int(value: Any, *, field: str) -> int:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ReleaseIndexError(f"{field} must be a canonical integer, got {value!r}")
    if value == "-0" or (value.startswith("0") and value != "0") or value.startswith("+"):
        raise ReleaseIndexError(f"{field} must be a canonical integer, got {value!r}")
    if value.startswith("-") and (len(value) == 1 or value[1:].startswith("0")):
        raise ReleaseIndexError(f"{field} must be a canonical integer, got {value!r}")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ReleaseIndexError(f"{field} must be a canonical integer, got {value!r}") from exc
    if str(parsed) != value:
        raise ReleaseIndexError(f"{field} must be a canonical integer, got {value!r}")
    return parsed


def _table_number(value: Any, *, field: str) -> float:
    if not isinstance(value, str):
        raise ReleaseIndexError(f"table field {field} is not textual CSV data")
    if value == "":
        return float("nan")
    try:
        number = float(value)
    except ValueError as exc:
        raise ReleaseIndexError(f"table field {field} is not numeric: {value!r}") from exc
    if math.isinf(number):
        raise ReleaseIndexError(f"table field {field} is infinite")
    return number


def _expected_table_keys(benchmark_v2: Any, train_channel: dict[str, str]) -> set[tuple[str, int, str, str]]:
    keys = {
        (condition, seed, split, intervention)
        for condition in CONDITIONS
        for seed in SEEDS
        for split, intervention in benchmark_v2.eval_plan(condition)
    }
    for name, (base_condition, _) in benchmark_v2.SCRIPTED.items():
        keys.update((f"scripted_{name}", -1, split, train_channel[base_condition]) for split in benchmark_v2.SPLITS)
    return keys


def _validate_metric_semantics(
    row: dict[str, Any], row_fields: tuple[str, ...], run_level_fields: tuple[str, ...]
) -> None:
    identity = f"{row['condition']}/seed {row['seed']}/{row['split']}/{row['intervention']}"
    always_finite = (
        "n",
        "success",
        "mean_return",
        "mi_first_msg",
        "runner_pick_rate",
        "mi_w1_color",
        "mi_w1_shape",
        "mi_w2_color",
        "mi_w2_shape",
    )
    for field in always_finite:
        if not math.isfinite(float(row[field])):
            raise ReleaseIndexError(f"non-finite required metric {field} in {identity}")
    if int(row["n"]) != row["n"] or row["n"] <= 0:
        raise ReleaseIndexError(f"invalid evaluation count in {identity}")
    for field in ("success", "runner_pick_rate", "color_acc", "shape_acc", *row_fields[2:5]):
        value = float(row[field])
        if math.isfinite(value) and not 0.0 <= value <= 1.0:
            raise ReleaseIndexError(f"out-of-range metric {field} in {identity}")
    if row["success"] > 0 and not all(math.isfinite(row[field]) for field in ("steps_success", "path_efficiency")):
        raise ReleaseIndexError(f"non-finite success metric in {identity}")
    if row["runner_pick_rate"] > 0 and not all(math.isfinite(row[field]) for field in ("color_acc", "shape_acc")):
        raise ReleaseIndexError(f"non-finite pick-accuracy metric in {identity}")

    twins = [row[field] for field in ("twin_color", "twin_shape", "twin_opposite")]
    if any(math.isfinite(value) for value in twins):
        if not all(math.isfinite(value) for value in twins) or not math.isclose(
            sum(twins), 1.0, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ReleaseIndexError(f"invalid twin-error decomposition in {identity}")
    for field in run_level_fields:
        if row["condition"].startswith("scripted_") and math.isfinite(row[field]):
            raise ReleaseIndexError(f"scripted row unexpectedly defines run-level metric {field} in {identity}")


def _read_table(
    table_path: Path,
    benchmark_v2: Any,
    train_channel: dict[str, str],
    n_eval_episodes: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, str, str], dict[str, Any]], bytes]:
    raw = table_path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleaseIndexError(f"{TABLE_PATH} is not UTF-8: {exc}") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames != benchmark_v2.RUN_FIELDS_V2:
        raise ReleaseIndexError(f"{TABLE_PATH} columns do not exactly match RUN_FIELDS_V2")

    rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    numeric_fields = benchmark_v2.RUN_FIELDS_V2[4:]
    for line_number, source in enumerate(reader, 2):
        if None in source:
            raise ReleaseIndexError(f"{TABLE_PATH}:{line_number} has extra columns")
        condition, split, intervention = (source[name] for name in ("condition", "split", "intervention"))
        if not condition or not split or not intervention:
            raise ReleaseIndexError(f"{TABLE_PATH}:{line_number} has an empty identity field")
        seed = _strict_int(source["seed"], field=f"{TABLE_PATH}:{line_number} seed")
        row: dict[str, Any] = {
            "condition": condition,
            "seed": seed,
            "split": split,
            "intervention": intervention,
        }
        row.update({field: _table_number(source[field], field=field) for field in numeric_fields})
        if _strict_int(source["n"], field=f"{TABLE_PATH}:{line_number} n") != n_eval_episodes:
            raise ReleaseIndexError(
                f"{TABLE_PATH}:{line_number} n must equal the fixed {n_eval_episodes} evaluation episodes"
            )
        row["n"] = n_eval_episodes
        key = (condition, seed, split, intervention)
        if key in by_key:
            raise ReleaseIndexError(f"duplicate {TABLE_PATH} row: {key}")
        _validate_metric_semantics(row, benchmark_v2.ROW_FIELDS, benchmark_v2.RUN_LEVEL_FIELDS)
        by_key[key] = row
        rows.append(row)

    expected = _expected_table_keys(benchmark_v2, train_channel)
    if set(by_key) != expected:
        missing = sorted(expected - set(by_key))
        unexpected = sorted(set(by_key) - expected)
        raise ReleaseIndexError(
            f"{TABLE_PATH} does not contain the exact prescribed rows; missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    return rows, by_key, raw


def _reject_duplicate_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonMember(f"duplicate JSON member {key!r}")
        value[key] = item
    return value


def _finite_json_numbers(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_json_numbers(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_json_numbers(item) for item in value.values())
    return False


def _record_int(record: dict, field: str, identity: str, *, minimum: int, maximum: int) -> int:
    value = record[field]
    if type(value) is not int or not minimum <= value <= maximum:
        raise ReleaseIndexError(
            f"{identity} {field} must be a strict JSON integer in [{minimum}, {maximum}], got {value!r}"
        )
    return value


def _attribute_pair(record: dict, field: str, identity: str, n_values: int) -> list[int]:
    value = record[field]
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) is not int or not 0 <= item < n_values for item in value)
    ):
        raise ReleaseIndexError(
            f"{identity} {field} must be two strict JSON integers in [0, {n_values - 1}], got {value!r}"
        )
    return value


def _validate_v2_evaluation_record(
    record: dict,
    *,
    condition: str,
    seed: int,
    expected_pairs: set[tuple[str, str]],
    env_cfg: Any,
    identity: str,
    n_eval_episodes: int,
) -> tuple[str, str]:
    fields = set(record)
    if fields != EVALUATION_RECORD_FIELDS:
        missing = sorted(EVALUATION_RECORD_FIELDS - fields)
        unexpected = sorted(fields - EVALUATION_RECORD_FIELDS)
        raise ReleaseIndexError(
            f"{identity} evaluation record schema mismatch; missing={missing}, unexpected={unexpected}"
        )

    values_without_path = dict(record)
    path_efficiency = values_without_path.pop("path_efficiency")
    failed_path_sentinel = record["success"] is False and type(path_efficiency) is float and math.isnan(path_efficiency)
    if not _finite_json_numbers(values_without_path) or (
        not failed_path_sentinel and not _finite_json_numbers(path_efficiency)
    ):
        raise ReleaseIndexError(f"non-finite evaluation value in {identity}")

    for field in ("condition", "split", "intervention"):
        if type(record[field]) is not str:
            raise ReleaseIndexError(f"{identity} {field} must be a JSON string, got {record[field]!r}")
    if type(record["seed"]) is not int:
        raise ReleaseIndexError(f"{identity} seed must be a strict JSON integer, got {record['seed']!r}")
    if record["condition"] != condition or record["seed"] != seed:
        raise ReleaseIndexError(f"{identity} has the wrong condition or seed")
    pair = (record["split"], record["intervention"])
    if pair not in expected_pairs:
        raise ReleaseIndexError(f"{identity} has an unexpected split/intervention {pair}")

    _record_int(record, "episode", identity, minimum=0, maximum=n_eval_episodes - 1)
    if record["split"] == "train":
        layout_minimum, layout_maximum = 0, env_cfg.n_train_layouts - 1
    else:
        layout_minimum = env_cfg.n_train_layouts
        layout_maximum = env_cfg.n_train_layouts + env_cfg.n_eval_layouts - 1
    _record_int(record, "layout_id", identity, minimum=layout_minimum, maximum=layout_maximum)
    target = _record_int(record, "target", identity, minimum=0, maximum=env_cfg.n_item_types - 1)
    if type(record["success"]) is not bool:
        raise ReleaseIndexError(f"{identity} success must be a JSON Boolean, got {record['success']!r}")
    if type(record["return"]) is not float:
        raise ReleaseIndexError(f"{identity} return must be a JSON float, got {record['return']!r}")
    steps = _record_int(record, "steps", identity, minimum=1, maximum=env_cfg.max_steps)
    runner_opt = _record_int(record, "runner_opt", identity, minimum=1, maximum=env_cfg.max_steps)
    runner_pick = _record_int(record, "runner_pick", identity, minimum=-1, maximum=env_cfg.n_item_types - 1)

    if type(path_efficiency) is not float:
        raise ReleaseIndexError(f"{identity} path_efficiency must be a JSON float, got {path_efficiency!r}")
    if record["success"]:
        expected_efficiency = runner_opt / steps
        if (
            runner_opt > steps
            or not 0.0 < path_efficiency <= 1.0
            or not math.isclose(path_efficiency, expected_efficiency, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ReleaseIndexError(f"{identity} path_efficiency is inconsistent with runner_opt and steps")
        if runner_pick != target:
            raise ReleaseIndexError(f"{identity} successful runner_pick does not equal target")
    elif not failed_path_sentinel:
        raise ReleaseIndexError(f"{identity} failed episode must use the path_efficiency NaN sentinel")

    target_attrs = _attribute_pair(record, "target_attrs", identity, env_cfg.n_values)
    picked_attrs = record["picked_attrs"]
    if runner_pick == -1:
        if picked_attrs is not None:
            raise ReleaseIndexError(f"{identity} picked_attrs must be null when runner_pick is -1")
    else:
        picked_attrs = _attribute_pair(record, "picked_attrs", identity, env_cfg.n_values)
    if record["success"] and picked_attrs != target_attrs:
        raise ReleaseIndexError(f"{identity} successful picked_attrs does not equal target_attrs")

    messages = record["scout_msgs"]
    if not isinstance(messages, list) or len(messages) != steps:
        raise ReleaseIndexError(f"{identity} scout_msgs must contain exactly one message per step")
    for message_index, message in enumerate(messages):
        if (
            not isinstance(message, list)
            or len(message) != env_cfg.msg_words
            or any(type(word) is not int or not 0 <= word < env_cfg.vocab_size for word in message)
        ):
            raise ReleaseIndexError(
                f"{identity} scout_msgs[{message_index}] must contain {env_cfg.msg_words} strict symbol integers"
            )
    return pair


def _read_evaluation(
    root: Path,
    path: Path,
    condition: str,
    seed: int,
    expected_pairs: list[tuple[str, str]],
    n_eval_episodes: int,
    env_cfg: Any,
) -> tuple[dict[tuple[str, str], list[dict]], str]:
    checked = _checked_file(root, path)
    digest_before = _sha256_file(checked)
    groups = {pair: [] for pair in expected_pairs}
    pair_set = set(groups)
    try:
        with checked.open() as fh:
            for line_number, line in enumerate(fh, 1):
                record = json.loads(line, object_pairs_hook=_reject_duplicate_members)
                if not isinstance(record, dict):
                    raise ReleaseIndexError(f"{checked}:{line_number} is not a JSON object")
                pair = _validate_v2_evaluation_record(
                    record,
                    condition=condition,
                    seed=seed,
                    expected_pairs=pair_set,
                    env_cfg=env_cfg,
                    identity=f"{checked}:{line_number}",
                    n_eval_episodes=n_eval_episodes,
                )
                groups[pair].append(record)
    except _DuplicateJsonMember as exc:
        raise ReleaseIndexError(f"invalid evaluation JSONL: {checked}: {exc}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseIndexError(f"invalid evaluation JSONL: {checked}: {exc}") from exc

    for pair, records in groups.items():
        if len(records) != n_eval_episodes:
            raise ReleaseIndexError(f"{checked} has {len(records)} records for {pair}, expected {n_eval_episodes}")
        episode_ids = [record["episode"] for record in records]
        if set(episode_ids) != set(range(n_eval_episodes)):
            raise ReleaseIndexError(
                f"{checked} episode IDs for {pair} must be exactly 0..{n_eval_episodes - 1}; got {episode_ids}"
            )
    digest_after = _sha256_file(checked)
    if digest_before != digest_after:
        raise ReleaseIndexError(f"evaluation changed while it was being validated: {checked}")
    return groups, digest_after


def _same_number(actual: float, expected: float) -> bool:
    if math.isnan(expected):
        return math.isnan(actual)
    return math.isfinite(actual) and math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)


def _verify_table_matches_evaluations(
    by_key: dict[tuple[str, int, str, str], dict[str, Any]],
    evaluations: dict[tuple[str, int], dict[tuple[str, str], list[dict]]],
    benchmark_v2: Any,
    train_channel: dict[str, str],
) -> None:
    metric_fields = benchmark_v2.RUN_FIELDS_V2[4:]
    for condition in CONDITIONS:
        for seed in SEEDS:
            groups = evaluations[(condition, seed)]
            pooled = [
                *groups[("held_out", train_channel[condition])],
                *groups[("unseen", train_channel[condition])],
            ]
            run_level = benchmark_v2.run_level(pooled)
            for (split, intervention), records in groups.items():
                expected = benchmark_v2.summarize(records, n_values=5)
                expected.update(run_level)
                actual = by_key[(condition, seed, split, intervention)]
                for field in metric_fields:
                    if not _same_number(float(actual[field]), float(expected[field])):
                        raise ReleaseIndexError(
                            f"{TABLE_PATH} does not match eval.jsonl for "
                            f"{condition}/seed {seed}/{split}/{intervention}: {field}"
                        )


def _publish_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ReleaseIndexError(f"output is not a regular file: {path}")
        if path.read_bytes() == payload:
            return
        raise ReleaseIndexError(f"refusing to overwrite existing index with different content: {path}")

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        temporary.chmod(0o644)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
                return
            raise ReleaseIndexError(f"refusing to overwrite existing index with different content: {path}") from None
        except OSError as exc:
            raise ReleaseIndexError(f"atomic publish failed for {path}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _build_v2_release_index(
    *,
    root: str | Path,
    run_auditor: RunAuditor,
    resolve_ref: ResolveRef,
    read_ref_file: ReadRefFile,
    bind_validator_sources: bool,
    n_eval_episodes: int,
) -> dict:
    root_argument = Path(root)
    if root_argument.is_symlink():
        raise ReleaseIndexError(f"repository root may not be a symlink: {root_argument}")
    try:
        root_path = root_argument.resolve(strict=True)
    except OSError as exc:
        raise ReleaseIndexError(f"repository root does not exist: {root_argument}") from exc
    if not root_path.is_dir():
        raise ReleaseIndexError(f"repository root is not a directory: {root_path}")
    if type(n_eval_episodes) is not int or n_eval_episodes <= 0:
        raise ReleaseIndexError("evaluation episode count must be a positive strict integer")

    config_bytes = _verify_frozen_sources(root_path, resolve_ref, read_ref_file)
    validator_sources = None
    if bind_validator_sources:
        try:
            head = _resolve_git_ref(root_path, "HEAD")
            indexer = _checked_file(root_path, root_path / "scripts/build_v2_release_index.py")
            if indexer.resolve(strict=True) != Path(__file__).resolve(strict=True):
                raise ReleaseIndexError("running release-index helper is not the canonical repository file")
            if _read_git_file(root_path, "HEAD", "scripts/build_v2_release_index.py") != indexer.read_bytes():
                raise ReleaseIndexError("release-index helper must be committed and match HEAD")
            validator_sources = _bind_validator_sources(root_path, head)
        except (OSError, PackagingError) as exc:
            raise ReleaseIndexError(str(exc)) from exc

    from emcomm import benchmark_v2
    from emcomm.channel import TRAIN_CHANNEL
    from emcomm.config import load_config
    from emcomm.plots import representative_seed

    config_path = _checked_file(root_path, root_path / CONFIG_PATH)
    try:
        env_cfg, ppo_cfg = load_config(config_path)
    except Exception as exc:
        raise ReleaseIndexError(f"invalid frozen config {CONFIG_PATH}: {exc}") from exc
    if config_path.read_bytes() != config_bytes:
        raise ReleaseIndexError(f"{CONFIG_PATH} changed while it was being validated")

    _verify_checkpoint_set(root_path)
    evaluations: dict[tuple[str, int], dict[tuple[str, str], list[dict]]] = {}
    evaluation_digests: dict[tuple[str, int], str] = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            run_dir = _checked_path(root_path, root_path / RUNS_PATH / condition / f"seed_{seed:04d}")
            if not run_dir.is_dir():
                raise ReleaseIndexError(f"expected run directory: {run_dir}")
            _checked_file(root_path, run_dir / "final.pt")
            try:
                run_auditor(root_path, run_dir, condition, seed, env_cfg, ppo_cfg)
            except PackagingError as exc:
                raise ReleaseIndexError(str(exc)) from exc
            groups, digest = _read_evaluation(
                root_path,
                run_dir / "eval.jsonl",
                condition,
                seed,
                benchmark_v2.eval_plan(condition),
                n_eval_episodes,
                env_cfg,
            )
            evaluations[(condition, seed)] = groups
            evaluation_digests[(condition, seed)] = digest

    if validator_sources is not None:
        try:
            _verify_imported_validator_sources(root_path, validator_sources)
        except PackagingError as exc:
            raise ReleaseIndexError(str(exc)) from exc

    table_path = _checked_file(root_path, root_path / TABLE_PATH)
    rows, by_key, table_bytes = _read_table(table_path, benchmark_v2, TRAIN_CHANNEL, n_eval_episodes)
    _verify_table_matches_evaluations(by_key, evaluations, benchmark_v2, TRAIN_CHANNEL)

    comm_median_rows = [
        row
        for row in rows
        if row["condition"] == "comm" and row["split"] == "held_out" and row["intervention"] == TRAIN_CHANNEL["comm"]
    ]
    if len(comm_median_rows) != 20 or {row["seed"] for row in comm_median_rows} != set(SEEDS):
        raise ReleaseIndexError("representative selection requires exactly the 20 comm seeds")
    representative = representative_seed(rows, condition="comm")
    if type(representative) is not int or representative not in SEEDS:
        raise ReleaseIndexError("representative_seed returned an invalid seed")

    checkpoint_relative = f"{RUNS_PATH}/comm/seed_{representative:04d}/final.pt"
    evaluation_relative = f"{RUNS_PATH}/comm/seed_{representative:04d}/eval.jsonl"
    checkpoint_path = _checked_file(root_path, root_path / checkpoint_relative)
    _checked_file(root_path, root_path / evaluation_relative)
    index = {
        "schema_version": SCHEMA_VERSION,
        "release_tag": RELEASE_TAG,
        "config": {"path": CONFIG_PATH, "sha256": hashlib.sha256(config_bytes).hexdigest()},
        "results_table": {"path": TABLE_PATH, "sha256": hashlib.sha256(table_bytes).hexdigest()},
        "representative": {
            "condition": "comm",
            "seed": representative,
            "checkpoint": {"path": checkpoint_relative, "sha256": _sha256_file(checkpoint_path)},
            "evaluation": {
                "path": evaluation_relative,
                "sha256": evaluation_digests[("comm", representative)],
            },
        },
    }
    payload = (json.dumps(index, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    output = _checked_path(root_path, root_path / OUTPUT_PATH, must_exist=False)
    _publish_atomic(output, payload)
    return index


def build_v2_release_index(root: str | Path = ".") -> dict:
    """Run the non-bypassable production audit and publish the canonical index."""
    return _build_v2_release_index(
        root=root,
        run_auditor=_validate_run,
        resolve_ref=_resolve_git_ref,
        read_ref_file=_read_git_file,
        bind_validator_sources=True,
        n_eval_episodes=200,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root (default: current directory)")
    parser.epilog = f"Writes {OUTPUT_PATH} only after every fixed v2.0 release-input gate passes."
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        index = build_v2_release_index(args.root)
    except ReleaseIndexError as exc:
        parser.error(str(exc))
    print(
        f"wrote {OUTPUT_PATH}: representative comm seed {index['representative']['seed']:04d}; "
        f"table {index['results_table']['sha256']}"
    )


if __name__ == "__main__":
    main()
