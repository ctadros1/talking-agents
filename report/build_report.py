"""Build report/report.pdf from report/report.md and results/summary.json.

Numbers are never typed into the report: `{{path.to.value|fmt}}` placeholders are
filled from results/summary.json, plus results/calibration.json under `calibration.` and
results/pilot_<rung>.json under `pilot.<rung>.`, results/vocal/summary.json under `vocal.`,
and results/v2/summary.json (+ calibration, pilots, support reference, meta) under `v2.`.
An unknown placeholder is an error.
Formats: pct, ci, pp, bits, num, num2, yesno, raw (default).

Usage: uv run python report/build_report.py
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import markdown

from emcomm.combos import COLOR_NAMES, SHAPE_NAMES, held_out_combos
from emcomm.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """
@page { size: letter; margin: 0.6in 0.55in; }
body { font: 9.4pt/1.38 "Helvetica Neue", Helvetica, Arial, sans-serif; color: #1f2933;
       columns: 2; column-gap: 0.25in; }
h1 { column-span: all; font-size: 16pt; margin: 0 0 2pt; color: #102a43; text-align: center; }
h1 + p { column-span: all; text-align: center; color: #52606d; margin: 0 0 10pt; }
h2 { font-size: 10.5pt; color: #102a43; margin: 9pt 0 3pt; break-after: avoid; }
p { margin: 0 0 5pt; text-align: justify; }
ul { margin: 0 0 5pt; padding-left: 13pt; } li { margin: 0 0 2pt; }
img { width: 100%; break-inside: avoid; }
table { width: 100%; border-collapse: collapse; font-size: 7.6pt; margin: 4pt 0 6pt; break-inside: avoid; }
th { background: #102a43; color: #fff; text-align: left; } th, td { padding: 2pt 3pt; border: 0.5pt solid #bcccdc; }
em.cap { display: block; font-size: 7.6pt; color: #52606d; text-align: center; margin-bottom: 6pt; }
code { font-size: 8pt; }
"""


def lookup(data: dict, path: str):
    value = data
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"report placeholder {path!r} not found in results")
        value = value[key]
    return value


def _not_estimable(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and not math.isfinite(value):
        return True
    if isinstance(value, dict) and "mean" in value:
        return any(_not_estimable(value[key]) for key in ("mean", "lo", "hi") if key in value)
    return False


def fmt(value, style: str) -> str:
    if _not_estimable(value):
        return "not estimable"
    if style == "pct":
        return f"{value * 100:.1f}%"
    if style == "mpct":  # the mean of a {mean, lo, hi} result, as a percentage without its CI
        return f"{value['mean'] * 100:.1f}%"
    if style == "ci":
        return f"{value['mean'] * 100:.1f}% (95% CI {value['lo'] * 100:.1f}–{value['hi'] * 100:.1f})"
    if style == "pp":
        return f"{value['mean'] * 100:+.1f} pp (95% CI {value['lo'] * 100:+.1f} to {value['hi'] * 100:+.1f})"
    if style == "bits":
        return f"{value['mean']:.2f} bits"
    if style == "num":
        return f"{value['mean']:.1f}"
    if style == "num2":
        return f"{value['mean']:.2f}"
    if style == "yesno":
        if not isinstance(value, bool):
            raise ValueError(f"yesno format requires a boolean, got {type(value).__name__}")
        return "yes" if value else "no"
    if style == "raw":
        return str(value)
    raise ValueError(f"unknown format {style!r}")


def _hypothesis_status(claims: dict, keys: tuple[str, ...]) -> str:
    if not claims or any(key not in claims for key in keys):
        return "not estimable"
    return "met" if all(claims[key] for key in keys) else "not met"


def _v2_report_fields(summary: dict) -> dict[str, str]:
    """Turn the fixed-sequence verdict into null-safe, non-overclaiming prose."""
    claims = summary.get("claims") or {}
    fields = {
        "h1_status": _hypothesis_status(claims, ("h1_gain_at_least_15pp", "h1_ci_excludes_zero")),
        "h2_status": _hypothesis_status(claims, ("h2_drop_at_least_10pp", "h2_ci_excludes_zero")),
        "h3_status": _hypothesis_status(claims, ("h3_gain_at_least_15pp", "h3_ci_excludes_zero")),
        "h4_status": _hypothesis_status(claims, ("h4_mean_above_reference", "h4_ci_above_reference")),
    }
    if not claims or any(key not in claims for key in ("claim_a", "claim_b", "claim_c")):
        fields.update(
            claim_a_status="not estimable",
            claim_b_status="not estimable",
            claim_c_status="not estimable",
            interpretation=(
                "The fixed-sequence verdict is not estimable because at least one required result is unavailable. "
                "No confirmatory claim is made from incomplete estimates. The lexicon, word interventions, "
                "property accuracies, topographic similarity, and positional disentanglement remain descriptive."
            ),
        )
        return fields
    if not claims["claim_a"]:
        fields.update(
            claim_a_status="not supported",
            claim_b_status="not reached",
            claim_c_status="not reached",
            interpretation=(
                "Claim A was not supported, so Claims B and C were not reached in the fixed sequence. "
                "The unseen estimates are shown for completeness and do not authorize a generalization claim. "
                "The lexicon and structural metrics remain descriptive."
            ),
        )
        return fields
    if not claims.get("claim_b", False):
        fields.update(
            claim_a_status="supported",
            claim_b_status="not supported",
            claim_c_status="not reached",
            interpretation=(
                "Claim A was supported, but Claim B was not, so Claim C was not reached. The experiment meets "
                "the seen-combination communication criterion but not the pre-registered generalization criterion. "
                "The unseen estimates and structural metrics remain descriptive."
            ),
        )
        return fields
    if not claims.get("claim_c", False):
        fields.update(
            claim_a_status="supported",
            claim_b_status="supported",
            claim_c_status="not supported",
            interpretation=(
                "Claims A and B were supported, but Claim C was not. The learned protocol meets the specified "
                "seen and unseen gain criteria, while unseen success does not meet H4's fixed 50% reference. "
                "This result does not identify which property transferred or establish compositionality."
            ),
        )
        return fields
    fields.update(
        claim_a_status="supported",
        claim_b_status="supported",
        claim_c_status="supported",
        interpretation=(
            "Claims A, B, and C were supported: learned communication meets the specified criteria on seen and "
            "unseen combinations, and unseen success exceeds H4's fixed 50% reference. H4 does not establish "
            "transfer of both properties or compositionality; the lexicon and structural metrics remain descriptive."
        ),
    )
    return fields


def render(template: str, data: dict) -> str:
    def substitute(match: re.Match) -> str:
        path, _, style = match.group(1).strip().partition("|")
        return fmt(lookup(data, path.strip()), style.strip() or "raw")

    return re.sub(r"\{\{([^}]+)\}\}", substitute, template)


def load_data(root: Path = ROOT) -> dict:
    """Everything a placeholder can refer to: summary, calibration, meta, pilot rungs, and the vocal arm."""
    results = root / "results"
    data = json.loads((results / "summary.json").read_text())
    data["calibration"] = json.loads((results / "calibration.json").read_text())
    _, ppo = load_config(root / "configs" / "base.toml")
    data["meta"] = {
        **asdict(ppo),
        "env_steps_millions": f"{ppo.n_updates * ppo.steps_per_update / 1e6:.1f}",
        "signal_note": (
            f" Every agent in every condition also received a cue-informativeness bias with weight {ppo.sig_coef}, "
            "a targeted form of the positive-signalling bias of Eccles et al. (2019): it rewards symbols that "
            "depend on the agent's private target cue, and never specifies which symbol should mean what."
            if ppo.sig_coef
            else ""
        ),
    }
    vocal = results / "vocal" / "summary.json"  # exploratory vocal arm (plan Task 26), if it was run
    if vocal.exists():
        data["vocal"] = json.loads(vocal.read_text())
        _, vocal_ppo = load_config(root / "configs/vocal.toml")
        data["vocal"]["meta"] = {"env_steps_millions": f"{vocal_ppo.n_updates * vocal_ppo.steps_per_update / 1e6:.1f}"}
    data["pilot"] = {
        p.stem.removeprefix("pilot_"): json.loads(p.read_text()) for p in sorted(results.glob("pilot_*.json"))
    }
    v2 = results / "v2"
    if (v2 / "summary.json").exists():  # v2: compositional communication (spec 2026-10-03)
        data["v2"] = json.loads((v2 / "summary.json").read_text())
        data["v2"]["calibration"] = json.loads((v2 / "calibration.json").read_text())
        data["v2"]["support_reference"] = json.loads((v2 / "support_reference.json").read_text())
        data["v2"]["pilot"] = {
            p.stem.removeprefix("pilot_"): json.loads(p.read_text()) for p in sorted(v2.glob("pilot_*.json"))
        }
        env2, ppo2 = load_config(root / "configs" / "v2.toml")
        data["v2"]["meta"] = {
            "env_steps_millions": f"{ppo2.n_updates * ppo2.steps_per_update / 1e6:.1f}",
            "sig_coef": str(ppo2.sig_coef),
            "held_out_combos": ", ".join(f"{COLOR_NAMES[c]} {SHAPE_NAMES[s]}" for c, s in held_out_combos(env2)),
        }
        data["v2"]["report"] = _v2_report_fields(data["v2"])
    ablation = results / "ablation_unbiased" / "summary.json"
    data["ablation"] = {"report": "The supplementary experiment is still running; no result is reported yet."}
    if ablation.exists():
        data["ablation"] = json.loads(ablation.read_text())
        data["ablation"]["report"] = render(
            "Across {{n_seeds}} full-budget seeds, unbiased IPPO reaches {{unbiased.success|ci}} held-out success, "
            "compared with {{biased.success|ci}} for cue-directed IPPO. The estimated biased minus unbiased "
            "difference is {{biased_minus_unbiased|pp}}. The unbiased baseline's intact minus permuted "
            "drop is {{unbiased.intact_minus_permuted|pp}}. The final-checkpoint protocol is retained; "
            "method differences bootstrap independent seed sets, while permutation drops pair each policy with itself. "
            "The difference "
            "estimate is exploratory because the comparison was added after the original outcomes were known; "
            "it does not retroactively expand the pre-registered hypotheses.",
            data["ablation"],
        )
    return data


def write_pdf(markdown_text: str, css: str, html: Path, pdf: Path) -> None:
    """Render Markdown (tables, fenced code, inline HTML/SVG allowed) to HTML, then print it to PDF with Chrome."""
    body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])
    html.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"
    )
    if not Path(CHROME).exists():
        sys.exit(f"Chrome not found; open {html} in a browser and print to PDF")
    subprocess.run(
        [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", html.as_uri()],
        check=True,
        capture_output=True,
    )
    print(f"wrote {pdf}")


def main() -> None:
    text = render((ROOT / "report" / "report.md").read_text(), load_data())
    # Keep a readable HTML companion; the submission PDF uses the course's CVPR template.
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    (ROOT / "report" / "report.html").write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    )
    from build_cvpr import build_cvpr

    build_cvpr(text)


if __name__ == "__main__":
    main()
