"""Render spoken result numbers from the same saved summaries as the report."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    def load(path):
        return json.loads((ROOT / path).read_text())

    v1 = load("results/summary.json")
    v2 = load("results/v2/summary.json")
    ablation = load("results/ablation_unbiased/summary.json")
    diagnostics = load("results/report_diagnostics/summary.json")["conditions"]
    difference = ablation["biased_minus_unbiased"]
    values = {
        "biased": ablation["biased"]["success"]["mean"],
        "unbiased": ablation["unbiased"]["success"]["mean"],
        "method_gain": difference["mean"],
        "method_lo": difference["lo"],
        "method_hi": difference["hi"],
        "channel_gain": v1["comparisons"]["comm_minus_no_comm"]["mean"],
        "no_comm": v1["conditions"]["no_comm"]["success"]["mean"],
        "permutation_drop": v1["comparisons"]["intact_minus_permuted"]["mean"],
        "seen": v2["conditions"]["comm"]["held_out"]["success"]["mean"],
        "unseen": v2["conditions"]["comm"]["unseen"]["success"]["mean"],
        "support": load("results/v2/support_reference.json")["support_aware_no_message_identification_rate"],
        "wrong_seen": diagnostics["comm"]["held_out"]["wrong_pick"]["mean"],
        "wrong_unseen": diagnostics["comm"]["unseen"]["wrong_pick"]["mean"],
    }
    template = (ROOT / "presentation/Read-Aloud-Script.template.md").read_text()
    text = re.sub(r"\{\{speech\.([a-z_]+)\}\}", lambda m: f"{values[m[1]] * 100:.1f}", template)
    if "{{" in text:
        raise ValueError("unresolved script placeholder")
    (ROOT / "presentation/Read-Aloud-Script.md").write_text(text)
    print("Rendered the complete narration script from saved summaries")


if __name__ == "__main__":
    main()
