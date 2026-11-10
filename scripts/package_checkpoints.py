"""Build and verify complete, checksummed checkpoint release archives.

Package one archive per stable experiment family. Verification checks the archive
and every member before an optional safe restore; it never loads PyTorch files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

SCHEMA = "emcomm-checkpoint-archive/v1"
ALLOWLIST = ("final.pt", "run.json", "metrics.csv", "train.log", "eval.jsonl")
HASH_CHUNK = 1024 * 1024


class PackagingError(ValueError):
    """A release archive or source run failed a strict validation."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_stream(fh: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: fh.read(HASH_CHUNK), b""):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _run_git(root: Path, *args: str) -> bytes:
    try:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = (
            exc.stderr.decode(errors="replace").strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        )
        raise PackagingError(f"git {' '.join(args)} failed: {detail}") from exc


def _lexical_path(root: Path, value: str | Path, *, must_exist: bool = True) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise PackagingError(f"path escapes --root: {value}") from exc
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.exists() or cursor.is_symlink():
            if cursor.is_symlink():
                raise PackagingError(f"symlink is not allowed: {cursor}")
        elif must_exist:
            raise PackagingError(f"path does not exist: {cursor}")
    if must_exist and not candidate.exists():
        raise PackagingError(f"path does not exist: {candidate}")
    return candidate


def _regular_file(root: Path, path: Path) -> Path:
    checked = _lexical_path(root, path)
    if not checked.is_file():
        raise PackagingError(f"expected a regular file: {checked}")
    return checked


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PackagingError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PackagingError(f"expected a JSON object in {path}")
    return value


def _validate_metrics(path: Path, ppo_cfg: Any) -> None:
    from emcomm.train import METRIC_FIELDS

    if ppo_cfg.n_updates * ppo_cfg.steps_per_update != ppo_cfg.total_env_steps:
        raise PackagingError(f"config budget is not a whole number of updates: {path}")
    try:
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != METRIC_FIELDS:
                raise PackagingError(f"metrics columns do not match the training schema: {path}")
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise PackagingError(f"cannot read metrics: {path}: {exc}") from exc
    if len(rows) != ppo_cfg.n_updates:
        raise PackagingError(f"incomplete metrics: {path}: {len(rows)} rows, expected {ppo_cfg.n_updates}")
    for expected_update, row in enumerate(rows, 1):
        if None in row:
            raise PackagingError(f"metrics row has unexpected extra values at update {expected_update}: {path}")
        try:
            update = int(row["update"])
            env_steps = int(row["env_steps"])
            numeric = [float(row[field]) for field in METRIC_FIELDS]
        except (TypeError, ValueError) as exc:
            raise PackagingError(f"non-numeric metric at update {expected_update}: {path}") from exc
        if update != expected_update:
            raise PackagingError(f"metrics update sequence is incomplete at row {expected_update}: {path}")
        if env_steps != expected_update * ppo_cfg.steps_per_update:
            raise PackagingError(f"metrics budget mismatch at update {expected_update}: {path}")
        if not all(math.isfinite(value) for value in numeric):
            raise PackagingError(f"non-finite metric at update {expected_update}: {path}")


def _validate_eval_jsonl(path: Path) -> None:
    def finite_numbers(value: Any) -> bool:
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return True
        if isinstance(value, (int, float)):
            return math.isfinite(value)
        if isinstance(value, list):
            return all(finite_numbers(item) for item in value)
        if isinstance(value, dict):
            return all(finite_numbers(item) for item in value.values())
        return False

    try:
        with path.open() as fh:
            for line_number, line in enumerate(fh, 1):
                record = json.loads(line)
                if (
                    not isinstance(record, dict)
                    or not isinstance(record.get("success"), bool)
                    or "path_efficiency" not in record
                ):
                    raise PackagingError(f"invalid or non-finite evaluation record at {path}:{line_number}")
                values = dict(record)
                path_efficiency = values.pop("path_efficiency", None)
                failed_sentinel = (
                    record["success"] is False and isinstance(path_efficiency, float) and math.isnan(path_efficiency)
                )
                if not finite_numbers(values) or (not failed_sentinel and not finite_numbers(path_efficiency)):
                    raise PackagingError(f"invalid or non-finite evaluation record at {path}:{line_number}")
    except (OSError, json.JSONDecodeError) as exc:
        raise PackagingError(f"invalid evaluation JSONL: {path}: {exc}") from exc


def _validate_run(
    root: Path,
    run_dir: Path,
    condition: str,
    seed: int,
    env_cfg: Any,
    ppo_cfg: Any,
) -> list[Path]:
    import torch

    from emcomm.config import EnvConfig, PPOConfig
    from emcomm.train import load_checkpoint

    torch.set_num_threads(1)
    run_dir = _lexical_path(root, run_dir)
    if not run_dir.is_dir():
        raise PackagingError(f"expected run directory: {run_dir}")
    required = {name: _regular_file(root, run_dir / name) for name in ("final.pt", "run.json", "metrics.csv")}
    run_json = _load_json(required["run.json"])
    if set(run_json) != {"condition", "seed", "env_cfg", "ppo_cfg"}:
        raise PackagingError(f"run.json has an unexpected schema: {required['run.json']}")
    try:
        run_env = EnvConfig(**run_json["env_cfg"])
        run_ppo = PPOConfig(**run_json["ppo_cfg"])
    except (TypeError, ValueError) as exc:
        raise PackagingError(f"invalid typed config in {required['run.json']}: {exc}") from exc
    if run_json["condition"] != condition or run_json["seed"] != seed or run_env != env_cfg or run_ppo != ppo_cfg:
        raise PackagingError(f"run identity or frozen config mismatch: {required['run.json']}")
    _validate_metrics(required["metrics.csv"], ppo_cfg)
    try:
        nets, checkpoint_env, checkpoint_ppo, checkpoint = load_checkpoint(required["final.pt"])
    except Exception as exc:
        raise PackagingError(f"checkpoint failed emcomm validation: {required['final.pt']}: {exc}") from exc
    expected_steps = ppo_cfg.n_updates * ppo_cfg.steps_per_update
    if checkpoint_env != env_cfg or checkpoint_ppo != ppo_cfg:
        raise PackagingError(f"checkpoint frozen config mismatch: {required['final.pt']}")
    if (
        checkpoint.get("condition") != condition
        or checkpoint.get("seed") != seed
        or checkpoint.get("update") != ppo_cfg.n_updates
        or checkpoint.get("env_steps") != expected_steps
        or len(checkpoint.get("nets", ())) != 2
        or len(nets) != 2
    ):
        raise PackagingError(f"checkpoint identity or final budget mismatch: {required['final.pt']}")
    for net_index, net in enumerate(nets):
        for tensor_name, tensor in net.state_dict().items():
            if not tensor.isfinite().all().item():
                raise PackagingError(
                    f"non-finite checkpoint tensor in agent {net_index} {tensor_name}: {required['final.pt']}"
                )
    included = []
    for name in ALLOWLIST:
        path = run_dir / name
        if path.exists() or path.is_symlink():
            included.append(_regular_file(root, path))
    eval_path = run_dir / "eval.jsonl"
    if eval_path in included:
        _validate_eval_jsonl(eval_path)
    return included


def _tar_info(archive_path: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(archive_path)
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _write_archive(path: Path, members: list[tuple[str, Path]]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for archive_path, source in members:
                    with source.open("rb") as fh:
                        tar.addfile(_tar_info(archive_path, source.stat().st_size), fh)


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in ("emcomm", "matplotlib", "numpy", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _bind_validator_sources(root: Path, commit: str) -> list[dict[str, str]]:
    source_root = _lexical_path(root, root / "src" / "emcomm")
    if not source_root.is_dir():
        raise PackagingError(f"validator source directory is missing: {source_root}")
    tracked_output = _run_git(
        root,
        "ls-tree",
        "-r",
        "--name-only",
        commit,
        "--",
        "src/emcomm",
    ).decode()
    tracked_python = sorted(line for line in tracked_output.splitlines() if line.endswith(".py"))
    working_python = sorted(_relative(root, path) for path in source_root.rglob("*.py"))
    if tracked_python != working_python:
        raise PackagingError("working validator source set differs from packager commit")
    paths = ["scripts/package_checkpoints.py", "pyproject.toml", "uv.lock", *tracked_python]
    records = []
    for relative in paths:
        path = _regular_file(root, root / relative)
        committed = _run_git(root, "show", f"{commit}:{relative}")
        if committed != path.read_bytes():
            raise PackagingError(f"validator source differs from packager commit: {relative}")
        records.append({"path": relative, "sha256": _sha256_file(path)})
    return records


def _verify_imported_validator_sources(root: Path, records: list[dict[str, str]]) -> None:
    expected = {record["path"]: record["sha256"] for record in records}
    imported = []
    for module_name, module in sorted(sys.modules.items()):
        if module_name != "emcomm" and not module_name.startswith("emcomm."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None or not module_file.endswith(".py"):
            continue
        actual = Path(module_file).resolve(strict=True)
        try:
            relative = actual.relative_to(root).as_posix()
        except ValueError as exc:
            raise PackagingError(f"imported emcomm source is outside --root: {module_name}: {actual}") from exc
        if relative not in expected or _sha256_file(actual) != expected[relative]:
            raise PackagingError(f"imported emcomm source differs from packager commit: {module_name}: {actual}")
        imported.append(relative)
    if not imported or "src/emcomm/config.py" not in imported or "src/emcomm/train.py" not in imported:
        raise PackagingError("required imported emcomm validator sources were not observed")


def _sidecars(archive: Path) -> tuple[Path, Path]:
    return Path(f"{archive}.manifest.json"), Path(f"{archive}.sha256")


def _validate_archive_path(name: str) -> None:
    path = PurePosixPath(name)
    if "\\" in name or path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise PackagingError(f"unsafe archive member path: {name!r}")
    if str(path) != name:
        raise PackagingError(f"non-canonical archive member path: {name!r}")


def _verify_members(archive: Path, manifest: dict) -> None:
    records = manifest.get("members")
    if not isinstance(records, list):
        raise PackagingError("manifest members must be a list")
    expected = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size"}:
            raise PackagingError("manifest member record has an invalid schema")
        name = record["path"]
        _validate_archive_path(name)
        if name in expected:
            raise PackagingError(f"duplicate manifest member: {name}")
        expected[name] = record
    if list(expected) != sorted(expected):
        raise PackagingError("manifest members are not deterministically ordered")
    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            members = tar.getmembers()
            names = [member.name for member in members]
            if names != sorted(names) or names != list(expected):
                raise PackagingError("archive members do not exactly match the manifest")
            for member in members:
                _validate_archive_path(member.name)
                if not member.isfile() or member.issym() or member.islnk():
                    raise PackagingError(f"archive contains a non-regular member: {member.name}")
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise PackagingError(f"cannot read archive member: {member.name}")
                digest, size = _sha256_stream(extracted)
                record = expected[member.name]
                if digest != record["sha256"] or size != record["size"] or member.size != record["size"]:
                    raise PackagingError(f"member hash or size mismatch: {member.name}")
    except (EOFError, OSError, tarfile.TarError) as exc:
        raise PackagingError(f"invalid archive: {archive}: {exc}") from exc


def package_checkpoints(
    *,
    root: str | Path,
    runs: list[str | Path],
    configs: list[str | Path],
    seeds: list[int],
    conditions: list[str],
    out: str | Path,
    version: str,
    source_ref: str,
) -> dict:
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise PackagingError(f"--root is not a directory: {root}")
    if not version.strip() or not source_ref.strip():
        raise PackagingError("--version and --ref must be explicit non-empty values")
    if len(runs) != len(configs) or not runs:
        raise PackagingError("provide one --config for each --runs path")
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise PackagingError("seeds must be a non-empty, unique, non-negative set")
    if not conditions or len(conditions) != len(set(conditions)):
        raise PackagingError("conditions must be explicit, non-empty, and unique")

    out = _lexical_path(root, out, must_exist=False)
    if out.suffixes[-2:] != [".tar", ".gz"]:
        raise PackagingError("--out must end in .tar.gz")
    if any(character.isspace() for character in out.name):
        raise PackagingError("--out basename must not contain whitespace")
    manifest_path, checksum_path = _sidecars(out)
    for path in (out, manifest_path, checksum_path):
        _lexical_path(root, path, must_exist=False)
        if path.exists() or path.is_symlink():
            raise PackagingError(f"output already exists: {path}")
    ignored = subprocess.run(["git", "check-ignore", "-q", "--no-index", str(out)], cwd=root).returncode
    if ignored != 0:
        raise PackagingError(f"--out must be ignored release storage (for example dist/): {out}")

    commit = _run_git(root, "rev-parse", "--verify", "--end-of-options", f"{source_ref}^{{commit}}").decode().strip()
    packager_commit = _run_git(root, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    packager_path = _regular_file(root, root / "scripts" / "package_checkpoints.py")
    packager_rel = _relative(root, packager_path)
    if _run_git(root, "show", f"{packager_commit}:{packager_rel}") != packager_path.read_bytes():
        raise PackagingError("packager script must be committed and match HEAD before creating release artifacts")
    validator_sources = _bind_validator_sources(root, packager_commit)
    importlib.import_module("emcomm.config")
    importlib.import_module("emcomm.train")
    _verify_imported_validator_sources(root, validator_sources)
    archive_members: dict[str, Path] = {}
    source_records = []
    from emcomm.config import load_config

    for runs_value, config_value in zip(runs, configs, strict=True):
        runs_path = _lexical_path(root, runs_value)
        if not runs_path.is_dir():
            raise PackagingError(f"--runs is not a directory: {runs_path}")
        config_candidate = Path(config_value)
        config_path = _regular_file(
            root, config_candidate if config_candidate.is_absolute() else root / config_candidate
        )
        runs_rel, config_rel = _relative(root, runs_path), _relative(root, config_path)
        try:
            frozen_config = _run_git(root, "show", f"{commit}:{config_rel}")
        except PackagingError as exc:
            raise PackagingError(f"config is not available from source ref {source_ref!r}: {config_rel}") from exc
        if frozen_config != config_path.read_bytes():
            raise PackagingError(f"config differs from source ref {source_ref!r}: {config_rel}")
        try:
            env_cfg, ppo_cfg = load_config(config_path)
        except Exception as exc:
            raise PackagingError(f"invalid frozen config {config_path}: {exc}") from exc

        expected_finals = {
            runs_path / condition / f"seed_{seed:04d}" / "final.pt" for condition in conditions for seed in seeds
        }
        found_finals = set(runs_path.glob("*/seed_*/final.pt"))
        unexpected = sorted(str(path) for path in found_finals - expected_finals)
        if unexpected:
            raise PackagingError(f"unexpected direct final checkpoints under {runs_path}: {unexpected}")
        for condition in conditions:
            for seed in seeds:
                run_dir = runs_path / condition / f"seed_{seed:04d}"
                for path in _validate_run(root, run_dir, condition, seed, env_cfg, ppo_cfg):
                    archive_path = _relative(root, path)
                    if archive_path in archive_members:
                        raise PackagingError(f"duplicate archive path: {archive_path}")
                    archive_members[archive_path] = path
        if config_rel in archive_members:
            raise PackagingError(f"duplicate archive path: {config_rel}")
        archive_members[config_rel] = config_path
        source_records.append(
            {
                "runs": runs_rel,
                "config": config_rel,
                "config_sha256": _sha256_file(config_path),
            }
        )
    _verify_imported_validator_sources(root, validator_sources)

    ordered_members = sorted(archive_members.items())
    out.parent.mkdir(parents=True, exist_ok=True)
    temp_paths: list[Path] = []
    try:
        fd, temp_name = tempfile.mkstemp(prefix=f".{out.name}.", dir=out.parent)
        os.close(fd)
        temp_archive = Path(temp_name)
        temp_paths.append(temp_archive)
        _write_archive(temp_archive, ordered_members)
        archive_hash = _sha256_file(temp_archive)
        manifest = {
            "schema": SCHEMA,
            "archive": {"name": out.name, "sha256": archive_hash, "size": temp_archive.stat().st_size},
            "source": {"version": version, "ref": source_ref, "resolved_commit": commit},
            "packager": {
                "resolved_commit": packager_commit,
                "script": packager_rel,
                "script_sha256": _sha256_file(packager_path),
                "validator_sources": validator_sources,
            },
            "expected": {"conditions": conditions, "seeds": seeds},
            "sources": source_records,
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": _package_versions(),
            },
            "members": [
                {"path": archive_path, "sha256": _sha256_file(path), "size": path.stat().st_size}
                for archive_path, path in ordered_members
            ],
        }
        _verify_members(temp_archive, manifest)

        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        checksum_bytes = (f"{archive_hash}  {out.name}\n{manifest_hash}  {manifest_path.name}\n").encode()
        for destination, payload in (
            (manifest_path, manifest_bytes),
            (checksum_path, checksum_bytes),
        ):
            fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=out.parent)
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
            temp = Path(name)
            temp_paths.append(temp)
        if any(path.exists() or path.is_symlink() for path in (out, manifest_path, checksum_path)):
            raise PackagingError("an output appeared while packaging; refusing to overwrite")
        temp_archive.replace(out)
        temp_paths[1].replace(manifest_path)
        temp_paths[2].replace(checksum_path)
        temp_paths.clear()
        return manifest
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)


def _read_checksums(path: Path, expected_names: set[str]) -> dict[str, str]:
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise PackagingError(f"cannot read checksum sidecar: {path}: {exc}") from exc
    checksums = {}
    for line in lines:
        parts = line.split()
        if len(parts) != 2 or len(parts[0]) != 64 or parts[1] in checksums:
            raise PackagingError(f"invalid checksum sidecar: {path}")
        try:
            int(parts[0], 16)
        except ValueError as exc:
            raise PackagingError(f"invalid checksum sidecar: {path}") from exc
        checksums[parts[1]] = parts[0].lower()
    if set(checksums) != expected_names:
        raise PackagingError(f"checksum sidecar does not name exactly {sorted(expected_names)}: {path}")
    return checksums


def _restore(archive: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()):
            raise PackagingError(f"restore destination must be an empty directory: {destination}")
    else:
        destination.mkdir(parents=True)
    destination = destination.resolve(strict=True)
    with tarfile.open(archive, mode="r:gz") as tar:
        for member in tar.getmembers():
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            try:
                target.relative_to(destination)
            except ValueError as exc:
                raise PackagingError(f"restore path escapes destination: {member.name}") from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise PackagingError(f"cannot restore archive member: {member.name}")
            with target.open("xb") as output:
                shutil.copyfileobj(source, output, HASH_CHUNK)
            target.chmod(0o644)


def verify_archive(
    archive: str | Path,
    *,
    manifest_path: str | Path | None = None,
    checksum_path: str | Path | None = None,
    restore: str | Path | None = None,
) -> dict:
    archive = Path(archive)
    if not archive.is_file() or archive.is_symlink():
        raise PackagingError(f"archive is not a regular file: {archive}")
    default_manifest, default_checksum = _sidecars(archive)
    manifest_path = Path(manifest_path) if manifest_path is not None else default_manifest
    checksum_path = Path(checksum_path) if checksum_path is not None else default_checksum
    checksums = _read_checksums(checksum_path, {archive.name, manifest_path.name})
    if _sha256_file(manifest_path) != checksums[manifest_path.name]:
        raise PackagingError("manifest SHA-256 does not match its checksum sidecar")
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != SCHEMA:
        raise PackagingError(f"unsupported manifest schema: {manifest.get('schema')!r}")
    archive_record = manifest.get("archive")
    if not isinstance(archive_record, dict) or archive_record.get("name") != archive.name:
        raise PackagingError("manifest archive identity does not match the selected archive")
    expected_hash = checksums[archive.name]
    actual_hash = _sha256_file(archive)
    if expected_hash != actual_hash or archive_record.get("sha256") != actual_hash:
        raise PackagingError("archive SHA-256 does not match its manifest and sidecar")
    if archive_record.get("size") != archive.stat().st_size:
        raise PackagingError("archive size does not match its manifest")
    _verify_members(archive, manifest)
    if restore is not None:
        _restore(archive, Path(restore))
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    package = subparsers.add_parser("package", help="validate complete runs and create a release archive")
    package.add_argument("--root", required=True, help="repository root")
    package.add_argument("--runs", action="append", required=True, help="run root; repeat with matching --config")
    package.add_argument("--config", action="append", required=True, help="frozen config; repeat for each --runs")
    package.add_argument("--seeds", default="0-19")
    package.add_argument("--conditions", required=True, help="explicit comma-separated conditions")
    package.add_argument("--out", required=True, help="ignored .tar.gz path under --root")
    package.add_argument("--version", required=True, help="release version represented by this artifact")
    package.add_argument("--ref", required=True, dest="source_ref", help="Git ref providing source provenance")

    verify = subparsers.add_parser("verify", help="offline hash verification and optional safe restore")
    verify.add_argument("--archive", required=True)
    verify.add_argument("--manifest")
    verify.add_argument("--sha256")
    verify.add_argument("--restore", help="nonexistent or empty destination directory")
    return parser


def _parse_seed_spec(spec: str) -> list[int]:
    seeds = []
    try:
        for part in spec.split(","):
            lo, separator, hi = part.partition("-")
            seeds.extend(range(int(lo), int(hi if separator else lo) + 1))
    except ValueError as exc:
        raise PackagingError(f"invalid seed specification: {spec!r}") from exc
    return seeds


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "package":
            manifest = package_checkpoints(
                root=args.root,
                runs=args.runs,
                configs=args.config,
                seeds=_parse_seed_spec(args.seeds),
                conditions=args.conditions.split(","),
                out=args.out,
                version=args.version,
                source_ref=args.source_ref,
            )
            print(
                f"packaged {len(manifest['members'])} files: {args.out}\n"
                f"manifest: {args.out}.manifest.json\nchecksum: {args.out}.sha256"
            )
        else:
            manifest = verify_archive(
                args.archive,
                manifest_path=args.manifest,
                checksum_path=args.sha256,
                restore=args.restore,
            )
            suffix = f"; restored to {args.restore}" if args.restore else ""
            print(f"verified {len(manifest['members'])} files in {args.archive}{suffix}")
    except PackagingError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
