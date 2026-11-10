from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "package_submission.py"
SPEC = importlib.util.spec_from_file_location("package_submission", SCRIPT)
assert SPEC and SPEC.loader
packager = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = packager
SPEC.loader.exec_module(packager)
SubmissionError = packager.SubmissionError

EXPECTED_COMMON_PATHS = {
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "report/report.md",
    "report/report.pdf",
    "docs/guide/how-it-works.md",
    "docs/guide/how-it-works.pdf",
    "docs/PREREGISTRATION.md",
    "docs/PREREGISTRATION_V2.md",
    "docs/DEVIATIONS.md",
    "docs/STATUS.md",
    "docs/SUBMISSION.md",
    "docs/REPRODUCIBILITY.md",
    "viewer/vendor/three.LICENSE",
    "viewer/vendor/three.min.js",
}
EXPECTED_V2_PATHS = {
    "results/v2/summary.json",
    "results/v2/release_index.json",
    "viewer/v2.html",
    "viewer/v2.js",
    "viewer/v2.css",
    "viewer/data/replays_v2.js",
}
EXPECTED_REQUIRED_PATHS = {
    "v2.0": EXPECTED_COMMON_PATHS | EXPECTED_V2_PATHS,
    "v2.1": EXPECTED_COMMON_PATHS
    | EXPECTED_V2_PATHS
    | {
        "docs/PREREGISTRATION_V21.md",
        "results/v2.1/summary.json",
        "results/v2.1/preregistration_manifest.json",
    },
    "v3.0": EXPECTED_COMMON_PATHS
    | EXPECTED_V2_PATHS
    | {
        "docs/PREREGISTRATION_V21.md",
        "docs/PREREGISTRATION_V3.md",
        "results/v2.1/summary.json",
        "results/v2.1/preregistration_manifest.json",
        "results/v3/summary.json",
        "results/v3/primary_inventory.json",
        "results/v3/reproducibility.json",
        "results/v3/per_run.manifest.json",
        "viewer/v3.html",
        "viewer/v3.js",
        "viewer/v3.css",
        "viewer/data/replays_v3.js",
    },
}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, text=True, capture_output=True).stdout.strip()


def _write(root: Path, relative: str, data: str | bytes = "fixture\n") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data)


def _required_v2_files(root: Path) -> None:
    for relative in sorted(packager.REQUIRED_PATHS["v2.0"]):
        _write(root, relative, b"%PDF-1.4\nfixture\n" if relative.endswith(".pdf") else "fixture\n")
    _write(root, "src/emcomm/__init__.py", '__version__ = "fixture"\n')
    _write(root, "scripts/executable_fixture.py", "#!/usr/bin/env python3\n")
    (root / "scripts/executable_fixture.py").chmod(0o755)
    _write(root, "docs/releases/v1.0-recovery.md", "# Historical recovery\n")


@pytest.fixture
def signed_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Submission Fixture")
    _git(root, "config", "user.email", "fixture@example.invalid")

    key = tmp_path / "fixture_signer"
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "fixture", "-f", str(key)],
        check=True,
    )
    public = Path(f"{key}.pub").read_text().split()
    signer_key = " ".join(public[:2])
    monkeypatch.setattr(packager, "TRUSTED_SIGNER_KEY", signer_key)
    monkeypatch.setattr(packager, "TRUSTED_SIGNER_FINGERPRINT", packager._ssh_key_fingerprint(signer_key))
    monkeypatch.setattr(packager, "TRUSTED_SIGNER_PRINCIPAL", "fixture@example.invalid")

    _git(root, "config", "gpg.format", "ssh")
    _git(root, "config", "gpg.ssh.program", "ssh-keygen")
    _git(root, "config", "user.signingkey", str(key))
    _required_v2_files(root)
    _git(root, "add", ".")
    _git(root, "commit", "-qS", "-m", "signed release fixture")
    _git(root, "tag", "-s", "v2.0", "-m", "nonproduction fixture v2.0")
    return root, tmp_path / "submission.zip"


def _rewrite_sidecar(archive: Path) -> None:
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(f"{archive.name}.sha256").write_text(f"{digest}  {archive.name}\n")


def test_package_uses_only_tagged_bytes_and_verifies_offline(
    signed_release: tuple[Path, Path],
) -> None:
    root, archive = signed_release
    tagged_readme = (root / "README.md").read_bytes()
    (root / "README.md").write_text("dirty working tree bytes\n")

    manifest = packager.package_submission(root=root, tag="v2.0", out=archive)
    verified = packager.verify_submission(archive)

    assert verified == manifest
    assert manifest["source"] == {
        "commit": _git(root, "rev-parse", "v2.0^{commit}"),
        "tag": "v2.0",
    }
    assert manifest["status"] == "candidate_pending_extraction_pdf_demo_checks"
    with zipfile.ZipFile(archive) as package:
        assert package.read("README.md") == tagged_readme
        executable = package.getinfo("scripts/executable_fixture.py")
        assert executable.external_attr >> 16 & 0o777 == 0o755
        package_readme = package.read("SUBMISSION_PACKAGE_README.md").decode()
        assert "candidate source bundle" in package_readme
        assert "open viewer/v2.html" in package_readme
        assert "docs/REPRODUCIBILITY.md" in package_readme
        assert "docs/releases/v1.0-recovery.md" in package_readme
        assert "does not authenticate a publisher" in package_readme

    restored = archive.parent / "restored"
    packager.verify_submission(archive, extract=restored)
    assert (restored / "README.md").read_bytes() == tagged_readme
    assert (restored / "SUBMISSION_MANIFEST.json").is_file()
    assert (restored / "scripts/executable_fixture.py").stat().st_mode & 0o777 == 0o755


def test_repeated_package_is_byte_identical_but_different_output_is_never_overwritten(
    signed_release: tuple[Path, Path],
) -> None:
    root, archive = signed_release
    packager.package_submission(root=root, tag="v2.0", out=archive)
    before = archive.read_bytes()
    packager.package_submission(root=root, tag="v2.0", out=archive)
    assert archive.read_bytes() == before

    archive.write_bytes(b"different existing bytes")
    with pytest.raises(SubmissionError, match="refusing to overwrite"):
        packager.package_submission(root=root, tag="v2.0", out=archive)
    assert archive.read_bytes() == b"different existing bytes"


def test_rejects_unsigned_commit_even_when_tag_is_signed(
    signed_release: tuple[Path, Path],
) -> None:
    root, archive = signed_release
    (root / "README.md").write_text("unsigned release commit\n")
    _git(root, "add", "README.md")
    _git(root, "commit", "-q", "--no-gpg-sign", "-m", "unsigned release commit")
    _git(root, "tag", "-s", "v2.1", "-m", "signed tag over unsigned commit")

    with pytest.raises(SubmissionError, match="commit signature"):
        packager.package_submission(root=root, tag="v2.1", out=archive)


def test_rejects_release_signed_by_another_key(
    signed_release: tuple[Path, Path],
) -> None:
    root, archive = signed_release
    other_key = archive.parent / "other_signer"
    subprocess.run(
        [
            "ssh-keygen",
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-C",
            "other",
            "-f",
            str(other_key),
        ],
        check=True,
    )
    _git(root, "config", "user.signingkey", str(other_key))
    (root / "README.md").write_text("wrong signer release\n")
    _git(root, "add", "README.md")
    _git(root, "commit", "-qS", "-m", "wrong signer release")
    _git(root, "tag", "-s", "v2.1", "-m", "wrong signer release")

    with pytest.raises(SubmissionError, match="pinned public signer"):
        packager.package_submission(root=root, tag="v2.1", out=archive)


def test_rejects_tag_ref_whose_signed_internal_name_differs(
    signed_release: tuple[Path, Path],
) -> None:
    root, archive = signed_release
    _git(root, "update-ref", "refs/tags/v2.1", "refs/tags/v2.0")

    with pytest.raises(SubmissionError, match="tag object name"):
        packager.package_submission(root=root, tag="v2.1", out=archive)


def test_git_replace_cannot_substitute_tagged_source(
    signed_release: tuple[Path, Path],
) -> None:
    root, archive = signed_release
    tagged_commit = _git(root, "rev-parse", "v2.0^{commit}")
    tagged_readme = (root / "README.md").read_bytes()
    (root / "README.md").write_text("replacement commit bytes\n")
    _git(root, "add", "README.md")
    _git(root, "commit", "-qS", "-m", "signed replacement fixture")
    replacement_commit = _git(root, "rev-parse", "HEAD")
    _git(root, "replace", tagged_commit, replacement_commit)

    manifest = packager.package_submission(root=root, tag="v2.0", out=archive)
    assert manifest["source"]["commit"] == tagged_commit
    with zipfile.ZipFile(archive) as package:
        assert package.read("README.md") == tagged_readme


def test_git_environment_cannot_redirect_release_reads(
    signed_release: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, archive = signed_release
    monkeypatch.setenv("GIT_DIR", str(archive.parent / "missing-attacker-repository"))
    manifest = packager.package_submission(root=root, tag="v2.0", out=archive)
    assert manifest["source"]["tag"] == "v2.0"


def test_rejects_tagged_collision_with_generated_package_names(
    signed_release: tuple[Path, Path],
) -> None:
    root, archive = signed_release
    (root / "SUBMISSION_MANIFEST.json").write_text("tagged collision\n")
    _git(root, "add", "SUBMISSION_MANIFEST.json")
    _git(root, "commit", "-qS", "-m", "reserved path fixture")
    _git(root, "tag", "-s", "v2.1", "-m", "reserved path fixture")

    with pytest.raises(SubmissionError, match="reserved generated package paths"):
        packager.package_submission(root=root, tag="v2.1", out=archive)


def test_rejects_missing_contract_file_and_tracked_symlink(signed_release: tuple[Path, Path], tmp_path: Path) -> None:
    root, archive = signed_release
    (root / "viewer/data/replays_v2.js").unlink()
    _git(root, "add", "viewer/data/replays_v2.js")
    _git(root, "commit", "-qS", "-m", "missing replay")
    _git(root, "tag", "-s", "v2.1", "-m", "missing replay fixture")
    with pytest.raises(SubmissionError, match="missing required tagged files"):
        packager.package_submission(root=root, tag="v2.1", out=archive)

    other = tmp_path / "symlink-repo"
    shutil.copytree(root, other, symlinks=True, ignore=shutil.ignore_patterns(".git"))
    _git(other, "init", "-q")
    _git(other, "config", "user.name", "Submission Fixture")
    _git(other, "config", "user.email", "fixture@example.invalid")
    target = other / "target.txt"
    target.write_text("target\n")
    (other / "unsafe-link").symlink_to(target.name)
    _git(other, "add", ".")
    # Tree validation is intentionally tested before signing through the private bounded helper.
    commit = _git(other, "write-tree")
    with pytest.raises(SubmissionError, match="symlink"):
        packager._read_tagged_tree(other, commit)


@pytest.mark.parametrize("member_name", ["../escape", "/absolute", "extra.txt"])
def test_verifier_rejects_traversal_and_extra_members(signed_release: tuple[Path, Path], member_name: str) -> None:
    root, archive = signed_release
    packager.package_submission(root=root, tag="v2.0", out=archive)
    with zipfile.ZipFile(archive, "a") as package:
        package.writestr(member_name, b"unexpected")
    _rewrite_sidecar(archive)

    with pytest.raises(SubmissionError, match="unsafe|exactly match|deterministically sorted"):
        packager.verify_submission(archive)


def test_verifier_rejects_duplicate_member_and_hash_tampering(signed_release: tuple[Path, Path]) -> None:
    root, archive = signed_release
    packager.package_submission(root=root, tag="v2.0", out=archive)
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(archive, "a") as package:
            package.writestr("README.md", b"duplicate")
    _rewrite_sidecar(archive)
    with pytest.raises(SubmissionError, match="duplicate archive member"):
        packager.verify_submission(archive)

    clean = archive.parent / "clean.zip"
    packager.package_submission(root=root, tag="v2.0", out=clean)
    payloads: dict[str, bytes] = {}
    with zipfile.ZipFile(clean) as source:
        for info in source.infolist():
            payloads[info.filename] = source.read(info)
    payloads["README.md"] = b"tampered tagged bytes"
    with zipfile.ZipFile(clean, "w") as destination:
        for name, payload in payloads.items():
            destination.writestr(packager._zip_info(name), payload)
    _rewrite_sidecar(clean)
    with pytest.raises(SubmissionError, match="hash or size mismatch"):
        packager.verify_submission(clean)


def test_verifier_rejects_zip_symlink_metadata(signed_release: tuple[Path, Path]) -> None:
    root, archive = signed_release
    packager.package_submission(root=root, tag="v2.0", out=archive)
    rewritten = archive.parent / "symlink.zip"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(rewritten, "w") as destination:
        for source_info in source.infolist():
            info = packager._zip_info(source_info.filename, source_info.external_attr >> 16 & 0o777)
            if source_info.filename == "README.md":
                info.external_attr = (0o120000 | 0o777) << 16
            destination.writestr(info, source.read(source_info))
    _rewrite_sidecar(rewritten)

    with pytest.raises(SubmissionError, match="unsafe or non-normalized"):
        packager.verify_submission(rewritten)


def test_verifier_rejects_uninventoried_zip_metadata(signed_release: tuple[Path, Path]) -> None:
    root, archive = signed_release
    packager.package_submission(root=root, tag="v2.0", out=archive)
    with zipfile.ZipFile(archive, "a") as package:
        package.comment = b"uninventoried comment"
    _rewrite_sidecar(archive)
    with pytest.raises(SubmissionError, match="archive comment"):
        packager.verify_submission(archive)

    clean = archive.parent / "metadata.zip"
    packager.package_submission(root=root, tag="v2.0", out=clean)
    rewritten = archive.parent / "metadata-rewritten.zip"
    with zipfile.ZipFile(clean) as source, zipfile.ZipFile(rewritten, "w") as destination:
        for index, source_info in enumerate(source.infolist()):
            info = packager._zip_info(source_info.filename, source_info.external_attr >> 16 & 0o777)
            if index == 0:
                info.extra = b"\x01\x00\x00\x00"
            destination.writestr(info, source.read(source_info))
    _rewrite_sidecar(rewritten)
    with pytest.raises(SubmissionError, match="unsafe or non-normalized"):
        packager.verify_submission(rewritten)


def test_verifier_rejects_sidecar_and_manifest_schema_tampering(signed_release: tuple[Path, Path]) -> None:
    root, archive = signed_release
    packager.package_submission(root=root, tag="v2.0", out=archive)
    archive.with_name(f"{archive.name}.sha256").write_text(f"{'0' * 64}  {archive.name}\n")
    with pytest.raises(SubmissionError, match="does not match its sidecar"):
        packager.verify_submission(archive)

    with pytest.raises(SubmissionError, match="duplicate JSON key"):
        packager._strict_json(b'{"schema":"one","schema":"two"}\n')
    valid = {
        "files": [],
        "schema": packager.SCHEMA,
        "source": {"commit": "a" * 40, "tag": "v2.0"},
        "status": packager.CANDIDATE_STATUS,
    }
    with pytest.raises(SubmissionError, match="canonical JSON"):
        packager._validate_manifest(valid, json.dumps(valid, indent=2).encode())


@pytest.mark.parametrize(("field", "value"), [("tag", []), ("mode", [])])
def test_manifest_rejects_unhashable_value_types(field: str, value: list) -> None:
    manifest = {
        "files": [
            {
                "mode": value if field == "mode" else "0644",
                "path": packager.PACKAGE_README_NAME,
                "sha256": "a" * 64,
                "size": 1,
            }
        ],
        "schema": packager.SCHEMA,
        "source": {"commit": "a" * 40, "tag": value if field == "tag" else "v2.0"},
        "status": packager.CANDIDATE_STATUS,
    }
    raw = packager._canonical_json(manifest)
    with pytest.raises(SubmissionError, match="source record|hash or size"):
        packager._validate_manifest(manifest, raw)


def test_manifest_parser_rejects_excessive_nesting() -> None:
    hostile = b"[" * 2_000 + b"0" + b"]" * 2_000
    with pytest.raises(SubmissionError, match="invalid submission manifest"):
        packager._strict_json(hostile)


def test_packager_rechecks_generated_and_archive_size_bounds(
    signed_release: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, archive = signed_release
    commit = _git(root, "rev-parse", "v2.0^{commit}")
    source_total = sum(len(payload) for payload, _mode in packager._read_tagged_tree(root, commit).values())
    original_total = packager.MAX_TOTAL_SIZE
    original_manifest = packager.MAX_MANIFEST_SIZE
    original_archive = packager.MAX_ARCHIVE_SIZE

    monkeypatch.setattr(packager, "MAX_TOTAL_SIZE", source_total)
    with pytest.raises(SubmissionError, match="final package inventory"):
        packager.package_submission(root=root, tag="v2.0", out=archive)
    assert not archive.exists()

    monkeypatch.setattr(packager, "MAX_TOTAL_SIZE", original_total)
    monkeypatch.setattr(packager, "MAX_MANIFEST_SIZE", 1)
    with pytest.raises(SubmissionError, match="generated submission manifest"):
        packager.package_submission(root=root, tag="v2.0", out=archive)
    assert not archive.exists()

    monkeypatch.setattr(packager, "MAX_MANIFEST_SIZE", original_manifest)
    monkeypatch.setattr(packager, "MAX_ARCHIVE_SIZE", 1)
    with pytest.raises(SubmissionError, match="generated archive"):
        packager.package_submission(root=root, tag="v2.0", out=archive)
    assert not archive.exists()
    monkeypatch.setattr(packager, "MAX_ARCHIVE_SIZE", original_archive)


def test_safe_extraction_refuses_nonempty_destination(signed_release: tuple[Path, Path]) -> None:
    root, archive = signed_release
    packager.package_submission(root=root, tag="v2.0", out=archive)
    destination = archive.parent / "existing"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep\n")

    with pytest.raises(SubmissionError, match="new or empty directory"):
        packager.verify_submission(archive, extract=destination)
    assert sentinel.read_text() == "keep\n"


def test_extraction_uses_the_same_verified_archive_handle(
    signed_release: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, archive = signed_release
    packager.package_submission(root=root, tag="v2.0", out=archive)
    replacement = archive.parent / "replacement.zip"
    with zipfile.ZipFile(replacement, "w") as malicious:
        malicious.writestr("../escaped.txt", b"unverified replacement")

    original_validate = packager._validate_manifest

    def replace_after_manifest(manifest, raw):
        validated = original_validate(manifest, raw)
        os.replace(replacement, archive)
        return validated

    monkeypatch.setattr(packager, "_validate_manifest", replace_after_manifest)
    destination = archive.parent / "same-handle-extract"
    packager.verify_submission(archive, extract=destination)

    assert (destination / "README.md").is_file()
    assert not (archive.parent / "escaped.txt").exists()


def test_failed_exclusive_copy_removes_only_its_partial_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"complete payload")
    destination = tmp_path / "new-output"

    def fail_after_partial(_source, output, _length):
        output.write(b"partial")
        raise OSError("fixture copy failure")

    monkeypatch.setattr(packager.shutil, "copyfileobj", fail_after_partial)
    with pytest.raises(OSError, match="fixture copy failure"):
        packager._publish_exclusive(destination, source)
    assert not destination.exists()

    existing = tmp_path / "existing-output"
    existing.write_bytes(b"preserve")
    with pytest.raises(SubmissionError, match="refusing to overwrite"):
        packager._publish_exclusive(existing, source)
    assert existing.read_bytes() == b"preserve"


def test_cli_help_and_invalid_release_are_bounded(tmp_path: Path) -> None:
    help_result = subprocess.run([sys.executable, str(SCRIPT), "--help"], text=True, capture_output=True)
    assert help_result.returncode == 0
    assert "package" in help_result.stdout
    assert "verify" in help_result.stdout

    with pytest.raises(SubmissionError, match="supported stable release tag"):
        packager.package_submission(root=tmp_path, tag="v1.0", out=tmp_path / "x.zip")

    assert packager.REQUIRED_PATHS == EXPECTED_REQUIRED_PATHS
    for forbidden in (
        ".env",
        "runs_vocal/comm/seed_0000/final.pt",
        "release_artifacts/v3.0-checkpoints.tar.gz",
        "weights/final.pt.zip",
        "weights/model.safetensors",
        "weights/final.bin",
        "release_artifacts/checkpoints.tar.xz",
        ".aws/credentials",
        ".netrc",
        "client_secret.json",
        "id_ecdsa",
        "private.pem",
    ):
        with pytest.raises(SubmissionError):
            packager._reject_forbidden_source(forbidden)
    for unsafe in ("line\nfeed", "carriage\rreturn", "delete\x7fcharacter"):
        with pytest.raises(SubmissionError, match="unsafe archive path"):
            packager._validate_member_path(unsafe)


@pytest.mark.parametrize("basename", ["course submission.zip", "submissión.zip"])
def test_package_rejects_unsupported_checksum_basename(signed_release: tuple[Path, Path], basename: str) -> None:
    root, archive = signed_release
    output = archive.parent / basename
    with pytest.raises(SubmissionError, match="basename must be ASCII|contain no whitespace"):
        packager.package_submission(root=root, tag="v2.0", out=output)
    assert not output.exists()


@pytest.mark.parametrize("tag", ["v2.1", "v3.0"])
def test_package_readme_keeps_scientific_reproduction_in_authenticated_clone(tag: str) -> None:
    readme = packager._package_readme(tag, "a" * 40, set()).decode()
    assert "Git-free extracted bundle is for static inspection and the offline demo only" in readme
    assert "git checkout --detach " + "a" * 40 in readme
    assert "verify-tag refs/tags/" + tag in readme
    assert f'refs/tags/{tag}^{{commit}})" = "' + "a" * 40 in readme
    assert "verify-commit " + "a" * 40 in readme
    assert "docs/REPRODUCIBILITY.md" in readme
    assert "run_v21_interpreter.py" not in readme
    assert "scripts/evaluate_v3.py" not in readme


@pytest.mark.parametrize("tag", ["v2.1", "v3.0"])
def test_later_release_contracts_package_when_complete(signed_release: tuple[Path, Path], tag: str) -> None:
    root, archive = signed_release
    for relative in sorted(packager.REQUIRED_PATHS[tag]):
        if not (root / relative).exists():
            _write(
                root,
                relative,
                b"%PDF-1.4\nfixture\n" if relative.endswith(".pdf") else "fixture\n",
            )
    _git(root, "add", ".")
    _git(root, "commit", "-qS", "-m", f"complete nonproduction {tag} fixture")
    _git(root, "tag", "-s", tag, "-m", f"complete nonproduction {tag} fixture")

    manifest = packager.package_submission(root=root, tag=tag, out=archive)
    assert manifest["source"]["tag"] == tag
    assert packager.verify_submission(archive) == manifest
