#!/usr/bin/env python3
"""Build and verify one deterministic source-only course submission candidate."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO

SCHEMA = "emcomm-submission-source/v1"
CANDIDATE_STATUS = "candidate_pending_extraction_pdf_demo_checks"
MANIFEST_NAME = "SUBMISSION_MANIFEST.json"
PACKAGE_README_NAME = "SUBMISSION_PACKAGE_README.md"
REPOSITORY_URL = "https://github.com/ctadros1/cap6938-emergent-communication.git"
HASH_CHUNK = 1024 * 1024
MAX_FILES = 5_000
MAX_FILE_SIZE = 128 * 1024 * 1024
MAX_TOTAL_SIZE = 1024 * 1024 * 1024
MAX_MANIFEST_SIZE = 8 * 1024 * 1024
MAX_ARCHIVE_SIZE = MAX_TOTAL_SIZE + 4 * MAX_MANIFEST_SIZE + MAX_FILES * 1024

TRUSTED_SIGNER_PRINCIPAL = "ctadros101@gmail.com"
TRUSTED_SIGNER_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILfIoH67Yi7/qv9LsbGEQxL0E7q/F8jbIYjkdxchaJd7"
TRUSTED_SIGNER_FINGERPRINT = "SHA256:PJZf/txTjoXik5mogVE/7PMFZ9uGF+O8ELR2pEM9PxI"

COMMON_REQUIRED = frozenset(
    {
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
)
V2_VIEWER_REQUIRED = frozenset(
    {
        "results/v2/summary.json",
        "results/v2/release_index.json",
        "viewer/v2.html",
        "viewer/v2.js",
        "viewer/v2.css",
        "viewer/data/replays_v2.js",
    }
)
V3_VIEWER_REQUIRED = frozenset(
    {
        "viewer/v3.html",
        "viewer/v3.js",
        "viewer/v3.css",
        "viewer/data/replays_v3.js",
    }
)
REQUIRED_PATHS = {
    "v2.0": COMMON_REQUIRED | V2_VIEWER_REQUIRED,
    "v2.1": COMMON_REQUIRED
    | V2_VIEWER_REQUIRED
    | {
        "docs/PREREGISTRATION_V21.md",
        "results/v2.1/summary.json",
        "results/v2.1/preregistration_manifest.json",
    },
    "v3.0": COMMON_REQUIRED
    | V2_VIEWER_REQUIRED
    | V3_VIEWER_REQUIRED
    | {
        "docs/PREREGISTRATION_V21.md",
        "docs/PREREGISTRATION_V3.md",
        "results/v2.1/summary.json",
        "results/v2.1/preregistration_manifest.json",
        "results/v3/summary.json",
        "results/v3/primary_inventory.json",
        "results/v3/reproducibility.json",
        "results/v3/per_run.manifest.json",
    },
}

_FORBIDDEN_PARTS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".cache",
        ".aws",
        ".gnupg",
        ".ssh",
        "cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        "dist",
        "runs",
        "runs_v2",
        "runs_v3",
        "checkpoints",
    }
)
_SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx")
_CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors")
_ARCHIVE_SUFFIXES = (".7z", ".rar", ".tar", ".tar.gz", ".tar.xz", ".tar.zst", ".tgz", ".zip")
_SECRET_BASENAMES = frozenset(
    {
        ".netrc",
        ".pypirc",
        "client_secret.json",
        "credentials",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
    }
)


class SubmissionError(ValueError):
    """A submission source package failed a bounded release check."""


def _run_git(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update({"LANG": "C", "LC_ALL": "C"})
    try:
        return subprocess.run(
            ("git", "-C", str(root), "--no-replace-objects", *arguments),
            check=check,
            capture_output=True,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = (
            exc.stderr.decode("utf-8", errors="replace").strip()
            if isinstance(exc, subprocess.CalledProcessError)
            else str(exc)
        )
        raise SubmissionError(f"git {' '.join(arguments)} failed: {detail}") from exc


def _ssh_key_fingerprint(key: str) -> str:
    parts = key.split()
    if len(parts) != 2 or parts[0] != "ssh-ed25519":
        raise SubmissionError("pinned signer must be one Ed25519 public key")
    try:
        blob = base64.b64decode(parts[1], validate=True)
    except ValueError as exc:
        raise SubmissionError("pinned signer contains invalid base64") from exc
    return "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")


def _require_terminal_ssh_signature(raw: bytes, label: str) -> None:
    begin = b"-----BEGIN SSH SIGNATURE-----"
    end = b"-----END SSH SIGNATURE-----"
    foreign = (
        b"-----BEGIN PGP SIGNATURE-----",
        b"-----BEGIN PGP MESSAGE-----",
        b"-----BEGIN SIGNED MESSAGE-----",
    )
    terminal = raw[:-1] if raw.endswith(b"\n") else raw
    if (
        raw.count(begin) != 1
        or raw.count(end) != 1
        or any(marker in raw for marker in foreign)
        or b"\n" + begin + b"\n" not in raw
        or not terminal.endswith(b"\n" + end)
    ):
        raise SubmissionError(f"{label} must contain exactly one terminal SSH signature")


def _require_commit_ssh_signature(raw: bytes, label: str) -> None:
    begin = b"gpgsig -----BEGIN SSH SIGNATURE-----\n"
    end = b" -----END SSH SIGNATURE-----\n"
    foreign = (b"BEGIN PGP SIGNATURE", b"BEGIN PGP MESSAGE", b"BEGIN SIGNED MESSAGE")
    header = raw.split(b"\n\n", 1)[0] + b"\n"
    if (
        header.count(begin) != 1
        or header.count(end) != 1
        or any(marker in header for marker in foreign)
        or header.find(begin) > header.find(end)
    ):
        raise SubmissionError(f"{label} must contain exactly one SSH commit signature")


def _verify_release_source(root: Path, tag: str) -> str:
    if tag not in REQUIRED_PATHS:
        raise SubmissionError(f"tag must be a supported stable release tag: {', '.join(REQUIRED_PATHS)}")
    if _ssh_key_fingerprint(TRUSTED_SIGNER_KEY) != TRUSTED_SIGNER_FINGERPRINT:
        raise SubmissionError("pinned signer key does not match its pinned fingerprint")

    reference = f"refs/tags/{tag}"
    tag_oid_result = _run_git(root, "rev-parse", "--verify", f"{reference}^{{tag}}", check=False)
    if tag_oid_result.returncode:
        raise SubmissionError(f"required signed annotated stable tag is absent: {tag}")
    tag_oid = tag_oid_result.stdout.decode().strip()
    tag_type = _run_git(root, "cat-file", "-t", tag_oid, check=False)
    if tag_type.returncode or tag_type.stdout.strip() != b"tag":
        raise SubmissionError(f"required signed annotated stable tag is absent: {tag}")
    tag_object = _run_git(root, "cat-file", "tag", tag_oid).stdout
    _require_terminal_ssh_signature(tag_object, f"tag {tag}")
    header = tag_object.split(b"\n\n", 1)[0].splitlines()
    fields = dict(line.split(b" ", 1) for line in header if b" " in line)
    if fields.get(b"tag") != tag.encode("ascii"):
        raise SubmissionError(f"signed tag object name does not match requested stable tag: {tag}")
    if fields.get(b"type") != b"commit":
        raise SubmissionError(f"stable tag must point directly to a commit: {tag}")

    commit = fields.get(b"object", b"").decode("ascii", errors="ignore")
    if len(commit) not in {40, 64} or any(character not in "0123456789abcdef" for character in commit):
        raise SubmissionError(f"stable tag object does not bind one exact commit: {tag}")
    commit_type = _run_git(root, "cat-file", "-t", commit, check=False)
    if commit_type.returncode or commit_type.stdout.strip() != b"commit":
        raise SubmissionError(f"stable tag object does not directly bind a commit: {tag}")
    commit_object = _run_git(root, "cat-file", "commit", commit).stdout
    _require_commit_ssh_signature(commit_object, f"release commit {commit}")

    with tempfile.TemporaryDirectory(prefix="submission-allowed-signers-") as directory:
        allowed = Path(directory) / "allowed_signers"
        allowed.write_text(f"{TRUSTED_SIGNER_PRINCIPAL} {TRUSTED_SIGNER_KEY}\n", encoding="ascii")
        allowed.chmod(0o600)
        common = (
            "-c",
            "gpg.format=ssh",
            "-c",
            "gpg.ssh.program=ssh-keygen",
            "-c",
            f"gpg.ssh.allowedSignersFile={allowed}",
            "-c",
            "gpg.minTrustLevel=fully",
        )
        tag_check = _run_git(root, *common, "verify-tag", tag_oid, check=False)
        if tag_check.returncode:
            raise SubmissionError(f"tag signature does not match pinned public signer: {tag}")
        commit_check = _run_git(root, *common, "verify-commit", commit, check=False)
        if commit_check.returncode:
            raise SubmissionError(f"release commit signature does not match pinned public signer: {commit}")
    return commit


def _validate_member_path(name: str) -> None:
    if not name or "\\" in name or any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise SubmissionError(f"unsafe archive path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or str(path) != name or any(part in {"", ".", ".."} for part in path.parts):
        raise SubmissionError(f"unsafe archive path: {name!r}")


def _reject_forbidden_source(name: str) -> None:
    _validate_member_path(name)
    parts = PurePosixPath(name).parts
    lowered = tuple(part.casefold() for part in parts)
    basename = lowered[-1]
    if any(part in _FORBIDDEN_PARTS for part in lowered) or lowered[0].startswith("runs"):
        raise SubmissionError(f"forbidden run, cache, build, or checkpoint path is tracked: {name}")
    if basename in {".env", ".envrc"} or basename.startswith(".env."):
        raise SubmissionError(f"environment file is tracked: {name}")
    if basename in _SECRET_BASENAMES or basename.startswith("client_secret") or basename.endswith(_SECRET_SUFFIXES):
        raise SubmissionError(f"possible private key is tracked: {name}")
    checkpoint_archive = basename.endswith(_ARCHIVE_SUFFIXES) and (
        "checkpoint" in basename or any(f"{suffix}." in basename for suffix in _CHECKPOINT_SUFFIXES)
    )
    checkpoint_binary = basename.endswith(".bin") and any(
        marker in basename for marker in ("checkpoint", "final", "model", "weights")
    )
    if (
        basename.endswith(_CHECKPOINT_SUFFIXES)
        or "checkpoint" in lowered[:-1]
        or checkpoint_archive
        or checkpoint_binary
    ):
        raise SubmissionError(f"checkpoint payload must remain a release asset: {name}")


def _read_tagged_tree(root: Path, treeish: str) -> dict[str, tuple[bytes, int]]:
    raw = _run_git(root, "ls-tree", "-rz", "--full-tree", treeish).stdout
    records = [record for record in raw.split(b"\0") if record]
    if not records or len(records) > MAX_FILES:
        raise SubmissionError(f"tagged source tree must contain 1..{MAX_FILES} files")
    files: dict[str, tuple[bytes, int]] = {}
    total = 0
    for record in records:
        try:
            metadata, raw_name = record.split(b"\t", 1)
            mode, kind, object_id = metadata.split(b" ", 2)
            name = raw_name.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise SubmissionError("tagged tree contains an invalid or non-UTF-8 entry") from exc
        if mode == b"120000":
            raise SubmissionError(f"tracked symlink is not allowed in a submission: {name}")
        if kind != b"blob" or mode not in {b"100644", b"100755"}:
            raise SubmissionError(f"tagged tree contains a non-regular file: {name}")
        _reject_forbidden_source(name)
        if name in files:
            raise SubmissionError(f"duplicate tagged source path: {name}")
        object_name = object_id.decode("ascii")
        try:
            blob_size = int(_run_git(root, "cat-file", "-s", object_name).stdout)
        except ValueError as exc:
            raise SubmissionError(f"tagged blob has an invalid size: {name}") from exc
        if blob_size > MAX_FILE_SIZE:
            raise SubmissionError(f"tagged file exceeds bounded package size: {name}")
        payload = _run_git(root, "cat-file", "blob", object_name).stdout
        if len(payload) != blob_size:
            raise SubmissionError(f"tagged blob size changed while packaging: {name}")
        total += blob_size
        if total > MAX_TOTAL_SIZE:
            raise SubmissionError("tagged source exceeds bounded package size")
        files[name] = (payload, 0o755 if mode == b"100755" else 0o644)
    return files


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _sha256_stream(stream: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: stream.read(HASH_CHUNK), b""):
        digest.update(chunk)
        size += len(chunk)
        if size > MAX_FILE_SIZE:
            raise SubmissionError("archive member exceeds bounded verification size")
    return digest.hexdigest(), size


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def _package_readme(tag: str, commit: str, paths: set[str]) -> bytes:
    recovery_docs = ["docs/REPRODUCIBILITY.md", "docs/SUBMISSION.md"]
    recovery_docs.extend(sorted(path for path in paths if path.startswith("docs/releases/") and path.endswith(".md")))
    demo = "open viewer/v3.html" if tag == "v3.0" else "open viewer/v2.html"
    if tag == "v3.0":
        phase_note = (
            "The v3 release tree supplies the report and demo. Its production scientific tools require the "
            "signed `prereg-v3` source and signed predecessor tags; use the exact tagged recovery procedure."
        )
    elif tag == "v2.1":
        phase_note = (
            "The v2.1 offline replay and scoring guards require the authenticated Git tag chain and restored "
            "release artifacts; they are not runnable as scientific verification from this Git-free ZIP alone."
        )
    else:
        phase_note = (
            "The v2.0 scientific rerun and checkpoint restoration use the exact tagged commands and matching "
            "release assets documented in `docs/REPRODUCIBILITY.md`."
        )
    docs = "\n".join(f"- `{path}`" for path in dict.fromkeys(recovery_docs))
    candidate_notice = (
        "This is a candidate source bundle reconstructed only from the immutable tagged Git tree. "
        "It is pending clean extraction, full PDF page inspection, offline demo exercise, and independent "
        "checkpoint-asset recovery. The ZIP itself contains no trained checkpoint archive. Its SHA-256 "
        "sidecar detects changed bytes but does not authenticate a publisher; source authentication was "
        "performed during packaging by verifying the tag and release commit against the pinned public SSH signer."
    )
    recovery_notice = (
        f"Download matching checkpoint archives from the GitHub release for `{tag}` and follow the tagged "
        "recovery document. Do not substitute checkpoints from another version."
    )
    signer_file = ".git/submission-verification/allowed_signers"
    write_signer = f"printf '%s\\n' '{TRUSTED_SIGNER_PRINCIPAL} {TRUSTED_SIGNER_KEY}' > {signer_file}"
    verify_options = (
        "-c gpg.format=ssh -c gpg.ssh.program=ssh-keygen "
        f"-c gpg.ssh.allowedSignersFile={signer_file} -c gpg.minTrustLevel=fully"
    )
    verify_tag = f"git {verify_options} verify-tag refs/tags/{tag}"
    verify_commit = f"git {verify_options} verify-commit {commit}"
    return f"""# CAP6938 submission source candidate

- Source tag: `{tag}`
- Source commit: `{commit}`

{candidate_notice}

## Authenticate source and reproduce

The Git-free extracted bundle is for static inspection and the offline demo only. Perform scientific
reproduction in an authenticated clone that retains Git tags, then restore the matching release assets.

```bash
git clone --no-checkout {REPOSITORY_URL} cap6938-emergent-communication
cd cap6938-emergent-communication
mkdir -p .git/submission-verification
{write_signer}
{verify_tag}
test "$(git rev-parse --verify refs/tags/{tag}^{{commit}})" = "{commit}"
{verify_commit}
git checkout --detach {commit}
uv sync --frozen
```

{phase_note}

The exact version-specific checkpoint download, offline restore, rerun, evaluation, and export commands are in
`docs/REPRODUCIBILITY.md` at this tag. Follow them without substituting another tag or checkpoint set.

## Verify this ZIP and open the static offline demo

Run the verifier from the authenticated `{tag}` checkout. Choose a new or empty extraction directory.

```bash
python3 scripts/package_submission.py verify --archive ../{tag}-submission.zip --extract ../{tag}-submission
cd ../{tag}-submission
{demo}
```

## Recovery documents

{docs}

{recovery_notice}
""".encode()


def _zip_info(name: str, mode: int = 0o644) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | mode) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _write_zip(path: Path, members: dict[str, tuple[bytes, int]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(members):
            payload, mode = members[name]
            archive.writestr(_zip_info(name, mode), payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _publish_exclusive(destination: Path, payload_path: Path, expected: bytes | None = None) -> bool:
    payload_hash, payload_size = _sha256_path(payload_path)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise SubmissionError(f"output is not a regular file: {destination}")
        if expected is not None:
            identical = destination.read_bytes() == expected
        else:
            existing_hash, existing_size = _sha256_path(destination)
            identical = existing_hash == payload_hash and existing_size == payload_size
        if not identical:
            raise SubmissionError(f"refusing to overwrite different existing bytes: {destination}")
        return False
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise SubmissionError(f"output appeared while packaging: {destination}") from exc
    try:
        with os.fdopen(descriptor, "wb") as output, payload_path.open("rb") as source:
            shutil.copyfileobj(source, output, HASH_CHUNK)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return True


def package_submission(*, root: str | Path, tag: str, out: str | Path) -> dict:
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise SubmissionError(f"repository root is not a directory: {root}")
    if tag not in REQUIRED_PATHS:
        raise SubmissionError(f"tag must be a supported stable release tag: {', '.join(REQUIRED_PATHS)}")
    out = Path(out)
    if out.suffix.lower() != ".zip":
        raise SubmissionError("output must end in .zip")
    if not out.name.isascii() or any(character.isspace() for character in out.name):
        raise SubmissionError("output basename must be ASCII and contain no whitespace")
    out.parent.mkdir(parents=True, exist_ok=True)
    out = out.absolute()
    checksum_path = out.with_name(f"{out.name}.sha256")
    if (out.exists() or out.is_symlink()) != (checksum_path.exists() or checksum_path.is_symlink()):
        raise SubmissionError("archive and checksum sidecar must either both exist or both be absent")

    commit = _verify_release_source(root, tag)
    source_files = _read_tagged_tree(root, commit)
    reserved = sorted({MANIFEST_NAME, PACKAGE_README_NAME} & source_files.keys())
    if reserved:
        raise SubmissionError(f"tagged source uses reserved generated package paths: {reserved}")
    missing = sorted(REQUIRED_PATHS[tag] - source_files.keys())
    if missing:
        raise SubmissionError(f"missing required tagged files for {tag}: {missing}")
    empty = sorted(path for path in REQUIRED_PATHS[tag] if not source_files[path][0])
    if empty:
        raise SubmissionError(f"required tagged files are empty for {tag}: {empty}")
    invalid_pdfs = sorted(
        path for path in REQUIRED_PATHS[tag] if path.endswith(".pdf") and not source_files[path][0].startswith(b"%PDF-")
    )
    if invalid_pdfs:
        raise SubmissionError(f"required tagged PDFs lack a PDF header for {tag}: {invalid_pdfs}")

    files = dict(source_files)
    files[PACKAGE_README_NAME] = (_package_readme(tag, commit, set(source_files)), 0o644)
    inventory = [
        {
            "mode": format(mode, "04o"),
            "path": name,
            "sha256": _sha256_bytes(payload),
            "size": len(payload),
        }
        for name, (payload, mode) in sorted(files.items())
    ]
    if sum(record["size"] for record in inventory) > MAX_TOTAL_SIZE:
        raise SubmissionError("final package inventory exceeds bounded uncompressed size")
    manifest = {
        "files": inventory,
        "schema": SCHEMA,
        "source": {"commit": commit, "tag": tag},
        "status": CANDIDATE_STATUS,
    }
    manifest_bytes = _canonical_json(manifest)
    if len(manifest_bytes) > MAX_MANIFEST_SIZE:
        raise SubmissionError("generated submission manifest exceeds bounded size")
    files[MANIFEST_NAME] = (manifest_bytes, 0o644)

    created: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="submission-package-") as directory:
        temporary_zip = Path(directory) / out.name
        _write_zip(temporary_zip, files)
        if temporary_zip.stat().st_size > MAX_ARCHIVE_SIZE:
            raise SubmissionError("generated archive exceeds bounded size")
        archive_hash, _archive_size = _sha256_path(temporary_zip)
        checksum = f"{archive_hash}  {out.name}\n".encode("ascii")
        temporary_checksum = Path(directory) / checksum_path.name
        temporary_checksum.write_bytes(checksum)
        try:
            if _publish_exclusive(out, temporary_zip):
                created.append(out)
            if _publish_exclusive(checksum_path, temporary_checksum, checksum):
                created.append(checksum_path)
        except Exception:
            for path in created:
                path.unlink(missing_ok=True)
            raise
    return manifest


def _strict_json(raw: bytes) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_pairs)
        stack = [(value, 0)]
        while stack:
            item, depth = stack.pop()
            if depth > 64:
                raise ValueError("JSON nesting exceeds 64 levels")
            if type(item) is list:
                stack.extend((child, depth + 1) for child in item)
            elif type(item) is dict:
                stack.extend((child, depth + 1) for child in item.values())
        return value
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise SubmissionError(f"invalid submission manifest: {exc}") from exc


def _validate_manifest(manifest: object, raw: bytes) -> dict:
    if type(manifest) is not dict or set(manifest) != {"files", "schema", "source", "status"}:
        raise SubmissionError("manifest has an invalid closed schema")
    try:
        canonical = _canonical_json(manifest)
    except (RecursionError, TypeError, ValueError) as exc:
        raise SubmissionError(f"manifest cannot be represented as canonical JSON: {exc}") from exc
    if raw != canonical:
        raise SubmissionError("manifest is not canonical JSON")
    if manifest["schema"] != SCHEMA or manifest["status"] != CANDIDATE_STATUS:
        raise SubmissionError("manifest schema or candidate status is unsupported")
    source = manifest["source"]
    if (
        type(source) is not dict
        or set(source) != {"commit", "tag"}
        or type(source["tag"]) is not str
        or source["tag"] not in REQUIRED_PATHS
        or type(source["commit"]) is not str
        or len(source["commit"]) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in source["commit"])
    ):
        raise SubmissionError("manifest source record is invalid")
    records = manifest["files"]
    if type(records) is not list or not records or len(records) > MAX_FILES + 1:
        raise SubmissionError("manifest files must be a bounded non-empty list")
    previous = ""
    seen: set[str] = set()
    total = 0
    for record in records:
        if type(record) is not dict or set(record) != {"mode", "path", "sha256", "size"}:
            raise SubmissionError("manifest file record has an invalid closed schema")
        name, digest, size, mode = (
            record["path"],
            record["sha256"],
            record["size"],
            record["mode"],
        )
        if type(name) is not str:
            raise SubmissionError("manifest file path must be a string")
        _validate_member_path(name)
        if name != PACKAGE_README_NAME:
            _reject_forbidden_source(name)
        if name in seen:
            raise SubmissionError(f"duplicate manifest member: {name}")
        if name <= previous:
            raise SubmissionError("manifest file inventory is not strictly sorted")
        if (
            type(mode) is not str
            or mode not in {"0644", "0755"}
            or type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or type(size) is not int
            or isinstance(size, bool)
            or not 0 <= size <= MAX_FILE_SIZE
        ):
            raise SubmissionError(f"manifest hash or size is invalid: {name}")
        total += size
        if total > MAX_TOTAL_SIZE:
            raise SubmissionError("manifest inventory exceeds bounded package size")
        seen.add(name)
        previous = name
    if MANIFEST_NAME in seen or PACKAGE_README_NAME not in seen:
        raise SubmissionError("manifest inventory has invalid generated-member coverage")
    if not REQUIRED_PATHS[source["tag"]].issubset(seen):
        raise SubmissionError("manifest omits a required release file")
    return manifest


def _read_checksum(path: Path, archive_name: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise SubmissionError(f"checksum sidecar is not a regular file: {path}")
    if path.stat().st_size > 256:
        raise SubmissionError("checksum sidecar exceeds bounded size")
    raw = path.read_bytes()
    parts = raw.split()
    try:
        archive_name_bytes = archive_name.encode("ascii")
    except UnicodeEncodeError as exc:
        raise SubmissionError("archive basename must be ASCII for checksum verification") from exc
    if any(character.isspace() for character in archive_name):
        raise SubmissionError("archive basename cannot contain whitespace")
    if len(parts) != 2 or parts[1] != archive_name_bytes or raw != parts[0] + b"  " + parts[1] + b"\n":
        raise SubmissionError("checksum sidecar must contain exactly the selected archive")
    try:
        digest = parts[0].decode("ascii")
    except UnicodeDecodeError as exc:
        raise SubmissionError("checksum sidecar digest is not ASCII") from exc
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SubmissionError("checksum sidecar digest is invalid")
    return digest


def _extract_verified_archive(archive: zipfile.ZipFile, infos: list[zipfile.ZipInfo], destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir() or any(destination.iterdir()):
            raise SubmissionError(f"extraction destination must be a new or empty directory: {destination}")
    else:
        destination.mkdir(parents=True)
    destination = destination.resolve(strict=True)
    for info in infos:
        _validate_member_path(info.filename)
        target = destination.joinpath(*PurePosixPath(info.filename).parts)
        try:
            target.relative_to(destination)
        except ValueError as exc:
            raise SubmissionError(f"extraction path escapes destination: {info.filename}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info, "r") as source, target.open("xb") as output:
            shutil.copyfileobj(source, output, HASH_CHUNK)
        target.chmod(stat.S_IMODE(info.external_attr >> 16))


def _verify_zip(archive_stream: BinaryIO, archive_label: Path, extract: Path | None = None) -> dict:
    try:
        with zipfile.ZipFile(archive_stream, "r") as archive:
            if archive.comment:
                raise SubmissionError("archive comment is not allowed")
            infos = archive.infolist()
            if not 2 <= len(infos) <= MAX_FILES + 2:
                raise SubmissionError("archive has an invalid or unbounded member count")
            names = [info.filename for info in infos]
            for name in names:
                _validate_member_path(name)
            if len(names) != len(set(names)):
                duplicate = next(name for name in names if names.count(name) > 1)
                raise SubmissionError(f"duplicate archive member: {duplicate}")
            if names != sorted(names):
                raise SubmissionError("archive members are not deterministically sorted")
            if MANIFEST_NAME not in names:
                raise SubmissionError("archive is missing its manifest")
            manifest_info = infos[names.index(MANIFEST_NAME)]
            if manifest_info.file_size > MAX_MANIFEST_SIZE:
                raise SubmissionError("submission manifest exceeds bounded size")
            manifest_raw = archive.read(manifest_info)
            manifest = _validate_manifest(_strict_json(manifest_raw), manifest_raw)
            records = {record["path"]: record for record in manifest["files"]}
            if names != sorted({*records, MANIFEST_NAME}):
                raise SubmissionError("archive members do not exactly match the manifest")
            total = 0
            for info in infos:
                mode = info.external_attr >> 16
                expected_flags = 0 if info.filename.isascii() else 0x800
                if (
                    info.create_version != 20
                    or info.create_system != 3
                    or info.extract_version != 20
                    or info.reserved != 0
                    or info.flag_bits != expected_flags
                    or info.volume != 0
                    or info.internal_attr != 0
                    or info.extra
                    or info.comment
                    or stat.S_IFMT(mode) != stat.S_IFREG
                    or stat.S_IMODE(mode)
                    != (0o644 if info.filename == MANIFEST_NAME else int(records[info.filename]["mode"], 8))
                    or info.date_time != (1980, 1, 1, 0, 0, 0)
                    or info.compress_type != zipfile.ZIP_DEFLATED
                ):
                    raise SubmissionError(f"archive member metadata is unsafe or non-normalized: {info.filename}")
                if info.filename == MANIFEST_NAME:
                    continue
                with archive.open(info, "r") as stream:
                    digest, size = _sha256_stream(stream)
                total += size
                record = records[info.filename]
                if digest != record["sha256"] or size != record["size"] or info.file_size != size:
                    raise SubmissionError(f"member hash or size mismatch: {info.filename}")
            if total > MAX_TOTAL_SIZE:
                raise SubmissionError("archive contents exceed bounded verification size")
            if extract is not None:
                _extract_verified_archive(archive, infos, extract)
            return manifest
    except (OSError, EOFError, zipfile.BadZipFile, RuntimeError) as exc:
        raise SubmissionError(f"invalid submission ZIP: {archive_label}: {exc}") from exc


def verify_submission(
    archive: str | Path,
    *,
    checksum_path: str | Path | None = None,
    extract: str | Path | None = None,
) -> dict:
    archive = Path(archive)
    if archive.is_symlink() or not archive.is_file():
        raise SubmissionError(f"archive is not a regular file: {archive}")
    if archive.stat().st_size > MAX_ARCHIVE_SIZE:
        raise SubmissionError("archive exceeds bounded verification size")
    sidecar = Path(checksum_path) if checksum_path is not None else archive.with_name(f"{archive.name}.sha256")
    expected = _read_checksum(sidecar, archive.name)
    with archive.open("rb") as archive_stream:
        digest = hashlib.sha256()
        size = 0
        for chunk in iter(lambda: archive_stream.read(HASH_CHUNK), b""):
            digest.update(chunk)
            size += len(chunk)
            if size > MAX_ARCHIVE_SIZE:
                raise SubmissionError("archive exceeds bounded verification size")
        if digest.hexdigest() != expected:
            raise SubmissionError("archive SHA-256 does not match its sidecar")
        archive_stream.seek(0)
        return _verify_zip(
            archive_stream,
            archive,
            Path(extract) if extract is not None else None,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("package", help="build from one signed stable release tag")
    package.add_argument("--root", required=True, help="Git repository root")
    package.add_argument("--tag", required=True, choices=tuple(REQUIRED_PATHS))
    package.add_argument("--out", required=True, help="new or byte-identical .zip output")
    verify = commands.add_parser("verify", help="verify offline and optionally extract safely")
    verify.add_argument("--archive", required=True)
    verify.add_argument("--sha256", dest="checksum_path")
    verify.add_argument("--extract", help="new or empty destination")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "package":
            manifest = package_submission(root=args.root, tag=args.tag, out=args.out)
            print(
                f"built candidate from {manifest['source']['tag']} at {manifest['source']['commit']}: {args.out}\n"
                f"checksum: {args.out}.sha256\nstatus: {manifest['status']}"
            )
        else:
            manifest = verify_submission(args.archive, checksum_path=args.checksum_path, extract=args.extract)
            suffix = f"; extracted to {args.extract}" if args.extract else ""
            print(f"verified {len(manifest['files'])} inventoried files{suffix}")
    except SubmissionError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
