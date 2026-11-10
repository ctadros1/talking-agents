"""Build an explicitly allowlisted static site; Python standard library only."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist/site"


def build():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for name in (
        "index.html",
        "styles.css",
        "app.mjs",
        "engine.mjs",
        "evidence.mjs",
        "scene3d.mjs",
        "icons.mjs",
        "icon-nodes.mjs",
        "icons.svg",
        "icons.css",
        "icons-source.json",
        "lucide.LICENSE",
        "replay-icons.js",
        "favicon.svg",
        "models",
        "data",
    ):
        source = ROOT / "web" / name
        if source.is_dir():
            shutil.copytree(source, OUT / name)
        else:
            shutil.copyfile(source, OUT / name)
    shutil.copytree(ROOT / "viewer", OUT / "replays")
    for page in (OUT / "replays").glob("*.html"):
        content = page.read_text()
        content = content.replace("</head>", '<link rel="stylesheet" href="../icons.css"></head>', 1)
        content = content.replace("</body>", '<script src="../replay-icons.js" defer></script></body>', 1)
        content = content.replace(
            "<header>",
            '<header><p><a class="back-link" href="../">'
            '<svg class="icon" aria-hidden="true" viewBox="0 0 24 24"><use href="../icons.svg#arrow-left"></use></svg>'
            "Talking Agents · live sandbox</a></p>",
            1,
        )
        page.write_text(content)
    data = {}
    for key, name in (
        ("v1", "summary.json"),
        ("v2", "v2/summary.json"),
        ("vocal", "vocal/summary.json"),
        ("ablation", "ablation_unbiased/summary.json"),
    ):
        data[key] = json.loads((ROOT / "results" / name).read_text())
    data["support"] = json.loads((ROOT / "results/v2/support_reference.json").read_text())
    (OUT / "data/results.json").write_text(json.dumps(data, separators=(",", ":")) + "\n")
    downloads = OUT / "downloads"
    downloads.mkdir()
    shutil.copyfile(ROOT / "report/report.pdf", downloads / "research-report.pdf")
    with zipfile.ZipFile(downloads / "results-and-protocol.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in ("results",):
            for path in sorted((ROOT / folder).rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(ROOT))
        for name in (
            "docs/PREREGISTRATION.md",
            "docs/PREREGISTRATION_V2.md",
            "docs/DEVIATIONS.md",
            "docs/SUPPLEMENTARY_ABLATION.md",
            "research-provenance.json",
            "README.md",
            "LICENSE",
        ):
            archive.write(ROOT / name, name)
    print(f"Built {OUT} ({sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()):,} bytes)")
    return OUT


if __name__ == "__main__":
    build()
