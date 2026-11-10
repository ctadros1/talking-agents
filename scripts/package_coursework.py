"""Package a verified coursework snapshot, separately from immutable product releases.

The existing package_submission.py retains its signed stable-release gates.
This helper never creates a release tag or promotes a scientific gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

PREFIX = "CAP6938-Project/"
MANIFEST = PREFIX + "COURSEWORK_MANIFEST.json"
FORBIDDEN = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "runs",
    "runs_v2",
    "runs_vocal",
    "dist",
    "cache",
}


def safe_path(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or any(part in FORBIDDEN or part.startswith(".env") for part in path.parts)
        or path.suffix in {".key", ".pem", ".pfx", ".p12"}
    ):
        raise ValueError(f"forbidden package path: {name}")
    if path.suffix == ".pt" and not (
        len(path.parts) == 5
        and path.parts[0] == "demo_checkpoints"
        and path.parts[1] in {"v1", "v2"}
        and path.parts[2] in {"comm", "no_comm"}
        and path.parts[3] in {"seed_0019", "seed_0009"}
        and path.name == "final.pt"
    ):
        raise ValueError(f"unexpected checkpoint path: {name}")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_archive(payloads: dict[str, tuple[bytes, int]], metadata: dict, out: Path) -> None:
    if out.exists():
        raise ValueError(f"archive already exists: {out}")
    for name in payloads:
        safe_path(name)
    manifest = {
        "schema": "cap6938-coursework/v1",
        **metadata,
        "files": {
            name: {"sha256": digest(data), "size": len(data), "mode": mode}
            for name, (data, mode) in sorted(payloads.items())
        },
    }
    complete = {PREFIX + name: (data, mode) for name, (data, mode) in payloads.items()}
    complete[MANIFEST] = ((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(), 0o644)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, (data, mode) in sorted(complete.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 9, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    out.with_suffix(out.suffix + ".sha256").write_text(f"{digest(out.read_bytes())}  {out.name}\n")


def verify_archive(archive: Path, extract: Path | None = None) -> dict:
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(names) != len(set(names)) or len(names) > 5000 or MANIFEST not in names:
            raise ValueError("duplicate, excessive or missing manifest entries")
        if sum(info.file_size for info in source.infolist()) > 256 * 1024 * 1024:
            raise ValueError("unexpectedly large coursework archive")
        manifest = json.loads(source.read(MANIFEST))
        expected = {PREFIX + name for name in manifest["files"]} | {MANIFEST}
        if set(names) != expected:
            raise ValueError("archive member list does not match manifest")
        for name, record in manifest["files"].items():
            safe_path(name)
            data = source.read(PREFIX + name)
            if len(data) != record["size"] or digest(data) != record["sha256"]:
                raise ValueError(f"payload hash mismatch: {name}")
        if extract is not None:
            if extract.exists() and any(extract.iterdir()):
                raise ValueError("extraction directory must be new or empty")
            extract.mkdir(parents=True, exist_ok=True)
            for name in names:
                path = extract / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source.read(name))
                if name != MANIFEST:
                    path.chmod(manifest["files"][name.removeprefix(PREFIX)]["mode"])
    return manifest


def package(root: Path, out: Path) -> None:
    root = root.resolve()
    ablation = json.loads((root / "results/ablation_unbiased/summary.json").read_text())
    if ablation.get("n_seeds") != 20 or ablation.get("exploratory_post_hoc") is not True:
        raise ValueError("the complete declared exploratory ablation is required")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    prefixes = ("src/", "tests/", "configs/", "scripts/", "viewer/", "results/", "report/", "docs/", "presentation/")
    top = {
        ".gitattributes",
        ".gitignore",
        "README.md",
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "START_HERE.html",
        "RUN_DEMO.command",
        "RUN_DEMO.bat",
    }
    files = [name for name in tracked if name and (name.startswith(prefixes) or name in top)]
    changed = subprocess.check_output(["git", "diff", "HEAD", "--name-only", "--", *files], cwd=root).decode()
    if changed:
        raise ValueError(f"commit the final coursework source before packaging:\n{changed}")
    payloads = {}
    for name in files:
        path = root / name
        if path.is_symlink():
            raise ValueError(f"symlink is not a submission payload: {name}")
        safe_path(name)
        data = path.read_bytes()
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=root)
        if data != committed:
            raise ValueError(f"working file bytes differ from the committed snapshot: {name}")
        payloads[name] = (data, 0o755 if path.stat().st_mode & 0o111 else 0o644)
    for phase, seed, runs in (("v1", 19, "runs"), ("v2", 9, "runs_v2")):
        for condition in ("comm", "no_comm"):
            folder = f"{condition}/seed_{seed:04d}"
            for filename in ("final.pt", "run.json"):
                payloads[f"demo_checkpoints/{phase}/{folder}/{filename}"] = (
                    (root / runs / folder / filename).read_bytes(),
                    0o644,
                )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    metadata = {
        "source_commit": commit,
        "purpose": "CAP6938 scientific-track coursework submission snapshot",
        "release_status": "v1.0 released; v2 experiment artifacts validated, not a stable v2.0 product release",
        "checkpoints": "Four representative final checkpoints for fresh inference; full training archives are separate",
        "video": "Student-narrated video is a separate required submission file",
    }
    write_archive(payloads, metadata, out)
    verify_archive(out)
    print(f"Verified coursework ZIP: {out} ({len(payloads)} payloads, source {commit})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("package")
    build.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    build.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--extract", type=Path)
    args = parser.parse_args()
    if args.command == "package":
        package(args.root, args.out)
    else:
        manifest = verify_archive(args.archive, args.extract)
        print(f"Verified {len(manifest['files'])} payloads, source {manifest['source_commit']}")


if __name__ == "__main__":
    main()
