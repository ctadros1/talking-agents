import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "coursework_package", Path(__file__).parents[1] / "scripts/package_coursework.py"
)
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


@pytest.fixture
def source_tree(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "config", "core.autocrlf", "input"], cwd=root, check=True)
    for name, data in {
        ".gitattributes": "*.pdf binary\n",
        ".gitignore": ".venv/\nruns/\n",
        "README.md": "Course project\n",
        "results/ablation_unbiased/summary.json": json.dumps({"n_seeds": 20, "exploratory_post_hoc": True}),
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--no-gpg-sign",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    for phase, seed in (("runs", 19), ("runs_v2", 9)):
        for condition in ("comm", "no_comm"):
            folder = root / phase / condition / f"seed_{seed:04d}"
            folder.mkdir(parents=True)
            (folder / "final.pt").write_bytes(b"fixture")
            (folder / "run.json").write_text("{}\n")
    return root


def test_coursework_snapshot_preserves_repository_file_rules(tmp_path, source_tree):
    archive = tmp_path / "snapshot.zip"
    package.package(source_tree, archive)
    manifest = package.verify_archive(archive)
    assert {".gitattributes", ".gitignore"} <= manifest["files"].keys()


def test_normalized_clean_diff_cannot_hide_different_payload_bytes(tmp_path, source_tree):
    (source_tree / "README.md").write_bytes(b"Course project\r\n")
    assert not subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=source_tree)
    with pytest.raises(ValueError, match="bytes differ"):
        package.package(source_tree, tmp_path / "bad-source.zip")


def test_coursework_package_roundtrip_detects_changed_payload(tmp_path):
    archive = tmp_path / "course.zip"
    payloads = {"README.md": (b"readme\n", 0o644), "RUN_DEMO.command": (b"#!/bin/sh\n", 0o755)}
    package.write_archive(payloads, {"source_commit": "a" * 40}, archive)
    extracted = tmp_path / "extracted"
    manifest = package.verify_archive(archive, extracted)
    assert manifest["source_commit"] == "a" * 40
    assert (extracted / "CAP6938-Project/README.md").read_bytes() == b"readme\n"
    assert (extracted / "CAP6938-Project/RUN_DEMO.command").stat().st_mode & 0o111
    with zipfile.ZipFile(archive) as source:
        members = {name: source.read(name) for name in source.namelist()}
    members["CAP6938-Project/README.md"] = b"changed\n"
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(tampered, "w") as target:
        for name, data in members.items():
            target.writestr(name, data)
    with pytest.raises(ValueError, match="hash"):
        package.verify_archive(tampered)


def test_forbidden_payloads_and_path_traversal_cannot_enter_zip(tmp_path):
    for name in ("../outside", ".env", ".git/config", "runs/comm/final.pt", "cache/private.key"):
        with pytest.raises(ValueError, match="path"):
            package.write_archive({name: (b"x", 0o644)}, {}, tmp_path / "bad.zip")
