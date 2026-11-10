"""Build the course presentation, narration notes and recording data from saved results.

Authoring-only dependency: reportlab (available in the Codex document runtime).
No experiment is run or selected here.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation"


def pct(value):
    return f"{value * 100:.1f}%"


def mean(value):
    return pct(value["mean"])


def pp(value):
    return f"{value['mean'] * 100:.1f} percentage points"


def ablation_slide(summary):
    if summary.get("n_seeds") != 20 or summary.get("exploratory_post_hoc") is not True:
        raise ValueError("a complete, labeled 20-seed exploratory ablation is required")
    effect = summary["biased_minus_unbiased"]
    interval = f"{effect['lo'] * 100:+.1f} to {effect['hi'] * 100:+.1f} pp"
    if effect["lo"] > 0:
        interpretation = "The interval favors the cue-directed method in this selected task and configuration."
    elif effect["hi"] < 0:
        interpretation = "The interval favors unbiased IPPO in this selected task and configuration."
    else:
        interpretation = "The interval includes zero and does not establish an advantage for either method."
    return {
        "title": "Does the added objective help?",
        "eyebrow": "Original-method control · Post-hoc exploratory",
        "kind": "comparison",
        "bullets": [
            f"Unbiased IPPO: {mean(summary['unbiased']['success'])} held-out success",
            f"Cue-directed IPPO: {mean(summary['biased']['success'])}",
            f"Difference: {effect['mean'] * 100:+.1f} pp · 95% CI {interval}",
        ],
        "notes": (
            "This direct comparison removes only the cue-information objective. The architecture, "
            "environment, optimizer and full training budget are equal, with twenty seeds per method. "
            f"Unbiased IPPO reaches {mean(summary['unbiased']['success'])} held-out success, compared with "
            f"{mean(summary['biased']['success'])} for cue-directed IPPO. The difference is "
            f"{pp(effect)}, with a ninety-five-percent seed-bootstrap interval from {interval}. "
            f"{interpretation} This is exploratory because it was added after the original outcomes "
            "were known. It does not compare against the complete Eccles method or establish a new "
            "state of the art."
        ),
    }


def slides():
    v1 = json.loads((ROOT / "results/summary.json").read_text())
    v2 = json.loads((ROOT / "results/v2/summary.json").read_text())
    vocal = json.loads((ROOT / "results/vocal/summary.json").read_text())
    support = json.loads((ROOT / "results/v2/support_reference.json").read_text())
    return [
        {
            "title": "Can agents invent useful messages?",
            "eyebrow": "CAP6938 · Scientific track",
            "bullets": [
                "Emergent Communication in a Grid World",
                "Christian Tadros · University of Central Florida",
                "Learned coordination, causal tests, and limits of transfer",
            ],
            "notes": (
                "My project asks whether two independently learning agents can invent a useful message "
                "protocol when I never assign meanings to their symbols. I built a small world where "
                "one agent has the information and the other can act on it. I will show what changed in "
                "the learning method, the controlled results, and an extension where communication "
                "works but the planned generalization claim fails."
            ),
            "kind": "title",
        },
        {
            "title": "The scout knows. The runner acts.",
            "eyebrow": "Information structure",
            "bullets": [
                "A wall blocks movement between rooms",
                "Only the scout sees the target",
                "The runner sees nearby items and must deliver the right one",
            ],
            "image": "../docs/guide/figures/map.png",
            "kind": "image",
            "notes": (
                "The scout is in the control room and knows which item is wanted. The runner is in the "
                "warehouse, sees only a local window, and does not know the target. Neither agent's "
                "visual input contains the other agent; the wall prevents physical contact. The local "
                "view is a square crop, rather than line-of-sight vision. Geometry is sampled before the target, and "
                "every item must be deliverable, so the map does not leak the answer. Messages arrive "
                "one step after they are sent. A wrong delivery ends the episode."
            ),
        },
        {
            "title": "One targeted change to independent PPO",
            "eyebrow": "Topic 7 · Agent RL and multi-agent systems",
            "bullets": [
                "Baseline: two separate recurrent PPO agents",
                "Modification: reward message information about the private cue",
                "No programmed codebook or listener supervision",
            ],
            "kind": "method",
            "notes": (
                "The starting method is independent recurrent PPO. Each agent has its own recurrent "
                "policy, value function and optimizer. The modification adds a cue-information "
                "objective for agents that actually see a cue. It encourages different message "
                "distributions for different targets without deciding which symbol should mean which "
                "target. This adapts the positive-signalling idea of Eccles and colleagues. I am not "
                "claiming to have invented that idea. The direct ablation removes only this objective, "
                "and is labeled exploratory because it was added after the original results were known."
            ),
        },
        {
            "title": "A benchmark of trained policies",
            "eyebrow": "Evaluation protocol",
            "bullets": [
                "20 independent training seeds per condition",
                "200 fixed evaluation episodes per seed and split",
                "Final checkpoints only · 95% seed-bootstrap intervals",
            ],
            "kind": "protocol",
            "notes": (
                "The main comparison has learned communication, no communication, random messages, and "
                "a full-information oracle. The observation and action spaces, network and budget are "
                "matched within each experiment. Every condition has twenty independent training seeds. "
                "Each policy is evaluated on two hundred fixed episodes per split. Confidence intervals "
                "bootstrap the training seeds, rather than treating all episodes as independent trained "
                "systems. Pilots use separate seeds and training layouts, and the benchmark evaluates "
                "only final checkpoints."
            ),
        },
        {
            "title": "Communication improves held-out success",
            "eyebrow": "v1 · Frozen benchmark",
            "bullets": [
                f"Learned communication: {mean(v1['conditions']['comm']['success'])}",
                f"No communication: {mean(v1['conditions']['no_comm']['success'])}",
                f"Gain: {pp(v1['comparisons']['comm_minus_no_comm'])}",
            ],
            "image": "../results/figures/fig_success.png",
            "kind": "image",
            "notes": (
                f"On held-out layouts, learned communication succeeds "
                f"{mean(v1['conditions']['comm']['success'])} of the time, compared with "
                f"{mean(v1['conditions']['no_comm']['success'])} without communication. The difference "
                f"is {pp(v1['comparisons']['comm_minus_no_comm'])}. Random messages give little benefit, "
                f"while the oracle indicates how well the runner can navigate when it already knows the "
                f"target. These are aggregate results across the twenty seeds. The scripted lines are "
                f"privileged references, not learned competitors."
            ),
        },
        {
            "title": "Does the listener use the meanings?",
            "eyebrow": "Same episode · Different received messages",
            "bullets": [
                "Left: learned messages intact",
                "Right: a fresh symbol permutation",
                "The replay illustrates the aggregate intervention test",
            ],
            "kind": "replay",
            "notes": (
                f"Here is the same held-out episode under intact and permuted messages. The scout and "
                f"runner use the same learned networks, but a fresh permutation changes the received "
                f"symbol identities. Watch the runner's target choice. Across all seeds, the "
                f"intact-minus-permuted drop is {pp(v1['comparisons']['intact_minus_permuted'])}. This "
                f"supports reliance on stable message meanings. A replay alone would not establish that "
                f"result; the controlled aggregate comparison does."
            ),
        },
        {
            "title": "Informative speech is not enough",
            "eyebrow": "Positive signalling versus positive listening",
            "bullets": [
                "Unheard scouts can still encode the target",
                "MI describes the speaker",
                "Interventions test whether the listener depends on the channel",
            ],
            "image": "../results/figures/fig_protocol.png",
            "kind": "image",
            "notes": (
                "The representative codebook contains arbitrary symbol assignments and some synonyms. "
                "But seeing a neat codebook does not prove communication. The signalling objective also "
                "makes scouts informative in conditions where nobody hears their outputs. The "
                "mutual-information statistic describes the speaker; it does not prove the runner "
                "listens. This follows the measurement concern raised by Lowe and colleagues and is why "
                "the intervention test matters."
            ),
        },
        {
            "title": "Two words do not guarantee generalization",
            "eyebrow": "v2 · Separately pre-registered",
            "bullets": [
                f"Seen combinations: {mean(v2['conditions']['comm']['held_out']['success'])}",
                f"Unseen combinations: {mean(v2['conditions']['comm']['unseen']['success'])}",
                "Communication supported · Generalization criterion not met",
            ],
            "image": "../results/v2/figures/fig_v2_success.png",
            "kind": "image",
            "notes": (
                f"The second experiment asks about targets with a color and a shape, communicated using "
                f"two words. The objective rewards each word's information about the joint cue; it does not "
                "enforce one word per property. Five combinations never appear in training, even as distractors. "
                f"Communication reaches {mean(v2['conditions']['comm']['held_out']['success'])} on seen "
                f"combinations and {mean(v2['conditions']['comm']['unseen']['success'])} on unseen "
                f"combinations. The frozen communication claim passes, but the generalization claim "
                f"fails its predefined effect-size requirement. The later claim is therefore not "
                f"reached. I retain this negative result rather than tuning after seeing it."
            ),
        },
        {
            "title": "Read the unseen result carefully",
            "eyebrow": "A limitation in the target prior",
            "bullets": [
                "The held-out combinations form a one-to-one matching",
                (
                    f"Support-aware identification: "
                    f"{pct(support['support_aware_no_message_identification_rate'])} before navigation"
                ),
                "The fixed 50% reference does not prove compositionality",
            ],
            "kind": "limitation",
            "notes": (
                "The unseen task has a limitation that matters for interpretation. Its target "
                "combinations form a fixed one-to-one matching. A strategy that knows this support can "
                "often identify the target without learning a compositional message. The source-only "
                "support calculation is shown here; it is not learned navigation performance. The fixed "
                "fifty-percent reference was preserved, but it cannot be called a one-property ceiling "
                "for this split. The lexicon and structural metrics are descriptive, not proof of "
                "compositional language."
            ),
        },
        {
            "title": "Exploratory: a continuous vocal channel",
            "eyebrow": "Separate arm · A longer training budget",
            "bullets": [
                f"Vocal communication: {mean(vocal['conditions']['comm']['success'])}",
                f"Intact minus shuffled voice: {pp(vocal['comparisons']['intact_minus_shuffled'])}",
                "Distinct notes form · Cross-arm rates are descriptive",
            ],
            "image": "../results/vocal/figures/fig_pitch.png",
            "kind": "image",
            "notes": (
                "An exploratory extension replaces discrete symbols with a noisy continuous pitch. The "
                "agents develop distinct pitch categories, and hearing a voice from another episode "
                "removes much of the benefit. This tests whether the listener uses the actual signal "
                "rather than just its presence. The vocal arm has a longer training budget, so its "
                "absolute success rate is not a clean one-variable comparison with the discrete arm. It "
                "remains outside the original confirmatory hypotheses."
            ),
        },
        {
            "title": "The implementation runs from saved weights",
            "eyebrow": "Fresh checkpoint inference",
            "bullets": [
                "Run RUN_DEMO.command, or the command below",
                "uv run python scripts/demo_inference.py --phase both",
                "New episode logs, metrics, receipt and offline replays",
            ],
            "kind": "execution",
            "notes": (
                "For the live demonstration I run fresh inference from the included final checkpoints. "
                "This command loads the learned policies and computes actions again; it does not merely "
                "play a saved animation. It writes a new output directory containing episode records, "
                "metrics, an execution receipt and regenerated replays. During preparation, all "
                "fourteen representative-seed summary rows matched the canonical results exactly. This "
                "is a representative execution check, not a rerun of the entire training sweep. I can "
                "run this command in the terminal during the live presentation."
            ),
        },
        {
            "title": "Useful coordination, with bounded claims",
            "eyebrow": "Conclusion and attribution",
            "bullets": [
                "Cue-directed signalling supports coordination in this task",
                "Causal message use differs from message informativeness",
                "The harder compositional generalization claim fails",
            ],
            "kind": "conclusion",
            "notes": (
                "The main result is that a task-specific signalling bias can support useful "
                "coordination between independently learning agents in this world, with interventions "
                "showing that the runner uses stable message identities. The extension shows the limit: "
                "communication success does not automatically give compositional generalization. The "
                "code, data and report preserve the controls, negative findings and exploratory labels. "
                "The work builds on PPO, independent multi-agent learning and prior "
                "emergent-communication methods. Claude and ChatGPT or Codex assisted with planning, "
                "implementation, checks and report refinement, as disclosed in the report. This "
                "recording uses my own narration."
            ),
        },
    ]


def make_pdf(items):
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen.canvas import Canvas

    canvas = Canvas(str(OUT / "Presentation.pdf"), pagesize=(960, 540))
    canvas.setTitle("Emergent Communication in a Grid World — Christian Tadros")
    canvas.setAuthor("Christian Tadros")

    def replay_pair():
        source = (ROOT / "viewer/data/replays.js").read_text()
        replay = json.loads(source.removeprefix("window.REPLAYS = ").rstrip().removesuffix(";"))
        key = next(
            key
            for key in replay["episodes"]
            if replay["variants"]["comm/intact"][key]["record"]["success"]
            and not replay["variants"]["comm/permuted"][key]["record"]["success"]
        )
        episode = replay["episodes"][key]
        for x, variant, label in ((110, "comm/intact", "INTACT"), (550, "comm/permuted", "PERMUTED")):
            run = replay["variants"][variant][key]
            frame = run["steps"][-1]
            canvas.setFillColor(HexColor("#ffcc55"))
            canvas.setFont("Helvetica-Bold", 15)
            canvas.drawString(x, 410, label)
            cell, board_y = 24, 100
            for row, walls in enumerate(episode["walls"]):
                for col, wall in enumerate(walls):
                    canvas.setFillColor(HexColor("#526071" if wall else "#27233c" if col < 4 else "#1b2837"))
                    canvas.rect(x + col * cell, board_y + (11 - row) * cell, cell - 1, cell - 1, fill=1, stroke=0)
            depot_row, depot_col = episode["depot"]
            canvas.setStrokeColor(HexColor("#50dfa0"))
            canvas.setLineWidth(2)
            canvas.rect(x + depot_col * cell + 3, board_y + (11 - depot_row) * cell + 3, 18, 18, fill=0, stroke=1)
            colors = ("#ec6475", "#ffcc55", "#53d2dc", "#c48cff")
            for item, (row, col) in enumerate(episode["items"]):
                if not frame["item_present"][item]:
                    continue
                canvas.setFillColor(HexColor(colors[item]))
                canvas.rect(x + col * cell + 4, board_y + (11 - row) * cell + 4, 16, 16, fill=1, stroke=0)
            for agent, position in enumerate(frame["pos"]):
                y, col = position
                canvas.setFillColor(HexColor("#ff963e" if agent else "#aa82ff"))
                canvas.circle(x + (col + 0.5) * cell, board_y + (11.5 - y) * cell, 9, fill=1, stroke=0)
            canvas.setFillColor(HexColor("#50dfa0" if run["record"]["success"] else "#ff7c87"))
            canvas.setFont("Helvetica-Bold", 13)
            canvas.drawString(x, 73, "CORRECT DELIVERY" if run["record"]["success"] else "FAILURE")
        canvas.setFillColor(HexColor("#bdcad8"))
        canvas.setFont("Helvetica", 11)
        canvas.drawString(
            110,
            47,
            f"Held-out episode {key}, representative seed {replay['seed']}. Terminal states; animate in record.html.",
        )

    for index, slide in enumerate(items):
        canvas.setFillColor(HexColor("#10151c"))
        canvas.rect(0, 0, 960, 540, fill=1, stroke=0)
        canvas.setFillColor(HexColor("#ffcc55"))
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawString(44, 498, slide["eyebrow"].upper())
        canvas.setFillColor(white)
        canvas.setFont("Helvetica-Bold", 26)
        canvas.drawString(44, 454, slide["title"])
        width = 430 if slide.get("image") else 870
        y = 390
        for bullet in [] if slide["kind"] == "replay" else slide["bullets"]:
            words, lines, line = bullet.split(), [], ""
            for word in words:
                trial = (line + " " + word).strip()
                if canvas.stringWidth(trial, "Helvetica", 17) > width:
                    lines.append(line)
                    line = word
                else:
                    line = trial
            lines.append(line)
            canvas.setFillColor(HexColor("#ffcc55"))
            canvas.circle(48, y + 5, 3, fill=1, stroke=0)
            canvas.setFillColor(white)
            canvas.setFont("Helvetica", 17)
            for line in lines:
                canvas.drawString(64, y, line)
                y -= 25
            y -= 22
        if slide.get("image"):
            path = (OUT / slide["image"]).resolve()
            image = ImageReader(str(path))
            w, h = image.getSize()
            scale = min(420 / w, 320 / h)
            canvas.setFillColor(white)
            canvas.rect(510, 95, 420, 320, fill=1, stroke=0)
            canvas.drawImage(
                image, 510 + (420 - w * scale) / 2, 95 + (320 - h * scale) / 2, width=w * scale, height=h * scale
            )
        if slide["kind"] == "replay":
            replay_pair()
        if slide["kind"] == "method":
            canvas.setFillColor(HexColor("#1c2b3c"))
            canvas.rect(64, 120, 830, 105, fill=1, stroke=0)
            canvas.setFillColor(HexColor("#ffcc55"))
            canvas.setFont("Helvetica-Bold", 22)
            canvas.drawString(86, 180, "PPO loss - lambda x cue information")
            canvas.setFont("Helvetica", 13)
            canvas.setFillColor(HexColor("#bdcad8"))
            canvas.drawString(86, 146, "Task-specific adaptation of positive signalling (Eccles et al., 2019)")
        canvas.setFont("Helvetica", 10)
        canvas.setFillColor(HexColor("#bdcad8"))
        canvas.drawString(44, 24, "Christian Tadros  |  CAP6938  |  Emergent Communication in a Grid World")
        canvas.drawRightString(916, 24, f"{index + 1} / {len(items)}")
        canvas.showPage()
    canvas.save()


def main():
    OUT.mkdir(exist_ok=True)
    items = slides()
    ablation = ROOT / "results/ablation_unbiased/summary.json"
    if ablation.exists():
        items.insert(3, ablation_slide(json.loads(ablation.read_text())))
    for slide in items:
        if slide.get("image"):
            # Data URLs keep the canvas origin-clean even when opened from disk.
            slide["imageData"] = "data:image/png;base64," + base64.b64encode(
                (OUT / slide["image"]).resolve().read_bytes()
            ).decode("ascii")
    (OUT / "slides.js").write_text("window.PRESENTATION = " + json.dumps(items, ensure_ascii=False) + ";\n")
    notes = [
        "# Narration script\n",
        (
            "Use this as a draft in your own voice. Edit it so every statement reflects your "
            "understanding. Aim for 7–9 minutes; no fixed video duration was stated in the "
            "assignment. Advance slides manually while recording.\n"
        ),
    ]
    for index, slide in enumerate(items, 1):
        notes.extend([f"\n## {index}. {slide['title']}\n", slide["notes"] + "\n"])
    (OUT / "Narration-Script.md").write_text("\n".join(notes))
    make_pdf(items)
    print(f"Built {len(items)} slides, narration script and presentation PDF")


if __name__ == "__main__":
    main()
