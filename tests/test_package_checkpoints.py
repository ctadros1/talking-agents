import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from emcomm.config import load_config
from emcomm.train import METRIC_FIELDS, make_nets, save_checkpoint

SCRIPT = Path(__file__).parents[1] / "scripts" / "package_checkpoints.py"
PROJECT_ROOT = SCRIPT.parent.parent
SPEC = importlib.util.spec_from_file_location("package_checkpoints", SCRIPT)
assert SPEC and SPEC.loader
package_checkpoints_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package_checkpoints_module)
PackagingError = package_checkpoints_module.PackagingError
verify_archive = package_checkpoints_module.verify_archive


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, text=True, capture_output=True).stdout.strip()


def _fixture(root: Path) -> tuple[Path, Path]:
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(root, "config", "user.name", "Fixture")
    key = root / ".fixture-signing-key"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    public_key = " ".join(key.with_suffix(".pub").read_text().split()[:2])
    allowed = root / ".fixture-allowed-signers"
    allowed.write_text(f"fixture@example.invalid {public_key}\n")
    _git(root, "config", "gpg.format", "ssh")
    _git(root, "config", "gpg.ssh.program", "ssh-keygen")
    _git(root, "config", "user.signingkey", str(key))
    _git(root, "config", "gpg.ssh.allowedSignersFile", str(allowed))
    _git(root, "config", "commit.gpgsign", "true")
    (root / ".gitignore").write_text("dist/\n.fixture-*\n")
    config = root / "configs" / "tiny.toml"
    config.parent.mkdir()
    config.write_text(
        """[env]
max_steps = 8

[ppo]
n_envs = 2
rollout_len = 4
total_env_steps = 16
hidden = 8
n_minibatches = 1
checkpoint_every = 2
"""
    )
    _git(root, "add", ".gitignore", "configs/tiny.toml")
    _git(root, "commit", "-qm", "fixture config")
    _git(root, "tag", "fixture-source")
    packager = root / "scripts" / "package_checkpoints.py"
    packager.parent.mkdir()
    shutil.copyfile(SCRIPT, packager)
    shutil.copytree(PROJECT_ROOT / "src" / "emcomm", root / "src" / "emcomm")
    shutil.copyfile(PROJECT_ROOT / "pyproject.toml", root / "pyproject.toml")
    shutil.copyfile(PROJECT_ROOT / "uv.lock", root / "uv.lock")
    _git(root, "add", "scripts/package_checkpoints.py", "src/emcomm", "pyproject.toml", "uv.lock")
    _git(root, "commit", "-qm", "checkpoint packager")

    env_cfg, ppo_cfg = load_config(config)
    run = root / "runs" / "comm" / "seed_0000"
    run.mkdir(parents=True)
    (run / "run.json").write_text(
        json.dumps(
            {
                "condition": "comm",
                "seed": 0,
                "env_cfg": asdict(env_cfg),
                "ppo_cfg": asdict(ppo_cfg),
            }
        )
    )
    with (run / "metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for update in range(1, ppo_cfg.n_updates + 1):
            row = {field: 1.0 for field in METRIC_FIELDS}
            row.update(
                update=update,
                env_steps=update * ppo_cfg.steps_per_update,
                episodes=2,
            )
            writer.writerow(row)
    save_checkpoint(
        run / "final.pt",
        make_nets(env_cfg, ppo_cfg),
        env_cfg,
        ppo_cfg,
        "comm",
        0,
        ppo_cfg.n_updates,
    )
    (run / "train.log").write_text("finished\n")
    (run / "update_00002.pt").write_bytes(b"excluded intermediate")
    (run / ".env").write_text("EXCLUDED=secret\n")
    return config, root / "runs"


def _package(
    root: Path,
    config: Path,
    runs: Path,
    name: str = "tiny-checkpoints.tar.gz",
    pythonpath: Path | None = None,
) -> Path:
    out = root / "dist" / name
    env = os.environ.copy()
    env["PYTHONPATH"] = str(pythonpath or root / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "package_checkpoints.py"),
            "package",
            "--root",
            str(root),
            "--runs",
            str(runs),
            "--config",
            str(config),
            "--seeds",
            "0",
            "--conditions",
            "comm",
            "--out",
            str(out),
            "--version",
            "test-1.0",
            "--ref",
            "fixture-source",
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise PackagingError(result.stderr)
    return out


def test_fixture_signing_is_local_and_uses_only_its_temporary_key(tmp_path):
    _fixture(tmp_path)
    assert _git(tmp_path, "config", "--local", "gpg.ssh.program") == "ssh-keygen"
    assert _git(tmp_path, "config", "--local", "commit.gpgsign") == "true"
    assert _git(tmp_path, "config", "--local", "user.signingkey") == str(tmp_path / ".fixture-signing-key")
    _git(tmp_path, "verify-commit", "fixture-source")
    _git(tmp_path, "verify-commit", "HEAD")
    assert ".fixture-signing-key" not in _git(tmp_path, "ls-files").splitlines()


def test_package_verify_and_restore_real_checkpoint(tmp_path):
    config, runs = _fixture(tmp_path)
    archive = _package(tmp_path, config, runs)

    manifest = verify_archive(archive)
    names = [member["path"] for member in manifest["members"]]
    assert names == sorted(names)
    assert names == [
        "configs/tiny.toml",
        "runs/comm/seed_0000/final.pt",
        "runs/comm/seed_0000/metrics.csv",
        "runs/comm/seed_0000/run.json",
        "runs/comm/seed_0000/train.log",
    ]
    assert manifest["source"]["resolved_commit"] == _git(tmp_path, "rev-parse", "fixture-source")
    assert manifest["packager"]["resolved_commit"] == _git(tmp_path, "rev-parse", "HEAD")
    assert manifest["source"]["resolved_commit"] != manifest["packager"]["resolved_commit"]
    expected_validator_sources = {
        "scripts/package_checkpoints.py",
        "pyproject.toml",
        "uv.lock",
        *(path.relative_to(tmp_path).as_posix() for path in (tmp_path / "src" / "emcomm").glob("*.py")),
    }
    assert {record["path"] for record in manifest["packager"]["validator_sources"]} == expected_validator_sources
    assert all(member["sha256"] for member in manifest["members"])

    restored = tmp_path / "restored"
    verify_archive(archive, restore=restored)
    assert (restored / "runs/comm/seed_0000/final.pt").read_bytes() == (runs / "comm/seed_0000/final.pt").read_bytes()
    assert not (restored / "runs/comm/seed_0000/update_00002.pt").exists()
    assert not (restored / "runs/comm/seed_0000/.env").exists()


def test_accepts_legacy_run_config_with_new_fields_defaulted(tmp_path):
    config, runs = _fixture(tmp_path)
    run_json = runs / "comm" / "seed_0000" / "run.json"
    data = json.loads(run_json.read_text())
    for field in ("channel", "vocal_noise", "task", "n_values", "msg_words", "combo_seed"):
        data["env_cfg"].pop(field)
    run_json.write_text(json.dumps(data))
    archive = _package(tmp_path, config, runs)
    verify_archive(archive)


def test_accepts_protocol_nan_only_for_failed_path_efficiency(tmp_path):
    config, runs = _fixture(tmp_path)
    eval_path = runs / "comm" / "seed_0000" / "eval.jsonl"
    records = [
        {"success": True, "return": 0.8, "steps": 12, "path_efficiency": 0.75},
        {"success": False, "return": -0.32, "steps": 32, "path_efficiency": float("nan")},
    ]
    eval_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    archive = _package(tmp_path, config, runs)
    manifest = verify_archive(archive)
    assert any(member["path"].endswith("eval.jsonl") for member in manifest["members"])


@pytest.mark.parametrize(
    "record",
    [
        {"success": False, "return": 0.0, "path_efficiency": float("inf")},
        {"success": True, "return": 0.8, "path_efficiency": float("nan")},
        {"success": float("nan"), "return": 0.0, "path_efficiency": float("nan")},
        {"success": False, "return": float("nan"), "path_efficiency": float("nan")},
    ],
)
def test_rejects_other_nonfinite_evaluation_values(tmp_path, record):
    config, runs = _fixture(tmp_path)
    eval_path = runs / "comm" / "seed_0000" / "eval.jsonl"
    eval_path.write_text(json.dumps(record) + "\n")
    with pytest.raises(PackagingError, match="non-finite evaluation"):
        _package(tmp_path, config, runs)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_rejects_nonfinite_checkpoint_tensor(tmp_path, nonfinite):
    config, runs = _fixture(tmp_path)
    env_cfg, ppo_cfg = load_config(config)
    nets = make_nets(env_cfg, ppo_cfg)
    with torch.no_grad():
        nets[0].move_head.weight[0, 0] = nonfinite
    save_checkpoint(
        runs / "comm" / "seed_0000" / "final.pt",
        nets,
        env_cfg,
        ppo_cfg,
        "comm",
        0,
        ppo_cfg.n_updates,
    )
    with pytest.raises(PackagingError, match="non-finite checkpoint tensor"):
        _package(tmp_path, config, runs)


def test_rejects_dirty_validator_source(tmp_path):
    config, runs = _fixture(tmp_path)
    train_source = tmp_path / "src" / "emcomm" / "train.py"
    train_source.write_text(train_source.read_text() + "\n# dirty\n")
    with pytest.raises(PackagingError, match="validator source differs from packager commit"):
        _package(tmp_path, config, runs)


def test_rejects_imported_validator_source_from_different_bytes(tmp_path):
    config, runs = _fixture(tmp_path)
    train_source = tmp_path / "src" / "emcomm" / "train.py"
    train_source.write_text(train_source.read_text() + "\n# committed but foreign\n")
    _git(tmp_path, "add", "src/emcomm/train.py")
    _git(tmp_path, "commit", "-qm", "foreign validator")
    with pytest.raises(PackagingError, match="imported emcomm source"):
        _package(tmp_path, config, runs, pythonpath=PROJECT_ROOT / "src")


@pytest.mark.parametrize("fault", ["missing", "seed", "config", "budget"])
def test_rejects_missing_or_mismatched_run(tmp_path, fault):
    config, runs = _fixture(tmp_path)
    run = runs / "comm" / "seed_0000"
    if fault == "missing":
        (run / "final.pt").unlink()
    elif fault == "seed":
        data = json.loads((run / "run.json").read_text())
        data["seed"] = 7
        (run / "run.json").write_text(json.dumps(data))
    elif fault == "config":
        data = json.loads((run / "run.json").read_text())
        data["env_cfg"]["grid_size"] = 99
        (run / "run.json").write_text(json.dumps(data))
    else:
        rows = list(csv.DictReader((run / "metrics.csv").open()))
        rows[-1]["env_steps"] = "8"
        with (run / "metrics.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=METRIC_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    with pytest.raises(PackagingError):
        _package(tmp_path, config, runs)


def test_verify_rejects_corrupted_archive_bytes(tmp_path):
    config, runs = _fixture(tmp_path)
    archive = _package(tmp_path, config, runs)
    with archive.open("ab") as fh:
        fh.write(b"corruption")
    with pytest.raises(PackagingError, match="archive SHA-256"):
        verify_archive(archive)


def test_rejects_symlink_member(tmp_path):
    config, runs = _fixture(tmp_path)
    final = runs / "comm" / "seed_0000" / "final.pt"
    target = tmp_path / "checkpoint.pt"
    final.replace(target)
    final.symlink_to(target)
    with pytest.raises(PackagingError, match="symlink"):
        _package(tmp_path, config, runs)


def test_rejects_output_collision(tmp_path):
    config, runs = _fixture(tmp_path)
    out = tmp_path / "dist" / "tiny-checkpoints.tar.gz"
    out.parent.mkdir()
    out.write_bytes(b"existing")
    with pytest.raises(PackagingError, match="already exists"):
        _package(tmp_path, config, runs)
