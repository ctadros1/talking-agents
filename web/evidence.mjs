export const pct = (x) => (100 * x).toFixed(1) + "%";
export const pp = (x) => (100 * x).toFixed(1) + " pp";
export const ci = (x, unit = pct) => `${unit(x.lo)}–${unit(x.hi)} · 95% CI`;
export function resultView(summary, phase) {
  const d = summary[phase],
    c = d.conditions;
  let cards,
    story,
    foot,
    bars = [];
  const card = (key, label, value, unit = "percent") => ({
    key,
    label,
    value,
    unit,
  });
  const bar = (condition, split, label, value) => ({
    key: `${condition}-${split}`,
    label,
    value,
  });
  if (phase === "v2") {
    const seen = c.comm.held_out.success,
      unseen = c.comm.unseen.success,
      gain = d.comparisons.unseen.comm_minus_no_comm;
    cards = [
      card("primary", "Familiar combinations · new maps", seen),
      card("secondary", "Unseen color–shape combinations", unseen),
      card("advantage", "Unseen gain over no messages", gain, "points"),
    ];
    story =
      "<strong>Communication transferred; the stronger generalization claim did not.</strong> Two-word messages helped on familiar color–shape combinations. On unseen combinations, the gain was positive but below the frozen 15 percentage point requirement. Claim A passed; Claim B failed; the hierarchy stopped before Claim C.";
    for (const [key, label] of [
      ["comm", "Learned messages"],
      ["no_comm", "No messages"],
      ["random_msg", "Random messages"],
      ["oracle", "Oracle"],
    ])
      bars.push(
        bar(key, "held_out", `${label} · seen`, c[key].held_out.success.mean),
      );
    for (const [key, label] of [
      ["comm", "Learned messages"],
      ["no_comm", "No messages"],
      ["oracle", "Oracle"],
    ])
      bars.push(
        bar(key, "unseen", `${label} · unseen`, c[key].unseen.success.mean),
      );
    foot = `80 learned runs · 20 seeds × 4 conditions · 20.0 million environment steps per run. “Seen” uses familiar combinations on held-out layouts. The unseen support also permits a target-identification shortcut (${pct(summary.support.support_aware_no_message_identification_rate)} theoretically, before navigation); it is not a measured learned-policy success rate. This limits a compositional-language interpretation.`;
  } else {
    cards = [
      card("primary", "Success with learned messages", c.comm.success),
      card("secondary", "Success without messages", c.no_comm.success),
      card(
        "advantage",
        "Communication advantage",
        d.comparisons.comm_minus_no_comm,
        "points",
      ),
    ];
    const conditions =
      phase === "v1"
        ? [
            ["comm", "Learned messages"],
            ["no_comm", "No messages"],
            ["random_msg", "Random messages"],
            ["oracle", "Oracle · target revealed"],
          ]
        : [
            ["comm", "Learned vocal messages"],
            ["no_comm", "No messages"],
          ];
    bars = conditions.map(([key, label]) =>
      bar(key, "held_out", label, c[key].success.mean),
    );
    if (phase === "v1") {
      story = `<strong>Stable, learned symbols improved coordination.</strong> The communicating pairs beat independently trained no-message and random-message controls. Relabeling their symbols reduced success by ${pp(d.comparisons.intact_minus_permuted.mean)} (${ci(d.comparisons.intact_minus_permuted, pp)}), supporting a causal role for stable symbol identities. The pre-registered criterion was met.`;
      foot =
        "80 learned runs · 20 seeds × 4 conditions · 10.0 million environment steps per run. Bars compare separately trained conditions. This study used a cue-directed signalling objective; useful communication is not evidence that unmodified PPO learns it equally well.";
    } else {
      story = `<strong>A continuous pitch channel also carried useful information.</strong> Both policies learn with a pitch-valued message instead of a discrete symbol. Shuffling the channel reduced success by ${pp(d.comparisons.intact_minus_shuffled.mean)}. This is an exploratory extension with a different training budget, so its score is not a controlled ranking against v1.`;
      foot =
        "40 learned runs · 20 seeds × 2 conditions · 20.0 million environment steps per run. Pitch is continuous; discretized bins are only for analysis. Use the vocal replay viewer to hear saved episodes.";
    }
  }
  return { cards, bars, story, foot };
}
const numberAnimations = new WeakMap();
function animateNumber(element, value, format, reduced) {
  const previous = numberAnimations.get(element);
  if (previous?.frame) cancelAnimationFrame(previous.frame);
  const state = { value: previous?.value ?? value, frame: null };
  numberAnimations.set(element, state);
  if (!previous || reduced) {
    state.value = value;
    element.textContent = format(value);
    return;
  }
  const from = state.value,
    start = performance.now();
  function tick(now) {
    const t = Math.min(1, (now - start) / 650),
      ease = 1 - (1 - t) ** 3;
    state.value = from + (value - from) * ease;
    element.textContent = format(state.value);
    if (t < 1) state.frame = requestAnimationFrame(tick);
  }
  state.frame = requestAnimationFrame(tick);
}
export class EvidenceView {
  constructor(root) {
    this.root = root;
    this.cards = new Map();
    this.bars = new Map();
    this.phase = null;
    root.innerHTML =
      '<div class="metric-grid"></div><p class="result-story"></p><div class="bar-table" aria-label="Success by condition"></div><p class="result-foot"></p>';
  }
  update(summary, phase) {
    if (this.phase === phase) return;
    const first = this.phase === null,
      view = resultView(summary, phase),
      reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.phase = phase;
    this.root.setAttribute("aria-labelledby", "tab-" + phase);
    for (const [index, data] of view.cards.entries()) {
      let element = this.cards.get(data.key);
      if (!element) {
        element = document.createElement("article");
        element.dataset.key = data.key;
        element.className = "metric" + (index === 0 ? " highlight" : "");
        element.innerHTML =
          '<div class="metric-label"></div><div class="metric-value" aria-hidden="true"></div><div class="metric-ci"></div>';
        this.cards.set(data.key, element);
        this.root.querySelector(".metric-grid").append(element);
      }
      const unit = data.unit === "points" ? pp : pct;
      element.querySelector(".metric-label").textContent = data.label;
      element.querySelector(".metric-ci").textContent = ci(data.value, unit);
      element.setAttribute(
        "aria-label",
        `${data.label}: ${unit(data.value.mean)}. ${ci(data.value, unit)}`,
      );
      animateNumber(
        element.querySelector(".metric-value"),
        data.value.mean,
        unit,
        reduced,
      );
    }
    const active = new Set(view.bars.map((b) => b.key));
    // Keep semantic rows mounted, including temporarily absent comparisons.
    for (const [key, element] of this.bars) {
      if (!active.has(key)) {
        element.classList.add("is-inactive");
        element.setAttribute("aria-hidden", "true");
      }
    }
    for (const data of view.bars) {
      let element = this.bars.get(data.key),
        created = false;
      if (!element) {
        created = true;
        element = document.createElement("div");
        element.className = "bar-slot is-inactive";
        element.dataset.key = data.key;
        element.innerHTML =
          '<div class="bar-clip"><div class="bar-row"><span class="bar-label"></span><div class="bar-track"><div class="bar-fill"></div></div><strong class="bar-number" aria-hidden="true"></strong></div></div>';
        this.bars.set(data.key, element);
        this.root.querySelector(".bar-table").append(element);
      }
      const fill = element.querySelector(".bar-fill"),
        width = Math.max(0, Math.min(100, data.value * 100));
      element.querySelector(".bar-label").textContent = data.label;
      element.setAttribute("aria-label", `${data.label}: ${pct(data.value)}`);
      element.removeAttribute("aria-hidden");
      if (created && !first && !reduced) {
        fill.style.width = "0%";
        element.getBoundingClientRect();
      }
      element.classList.remove("is-inactive");
      fill.style.width = width + "%";
      animateNumber(
        element.querySelector(".bar-number"),
        data.value,
        pct,
        reduced,
      );
    }
    const story = this.root.querySelector(".result-story");
    story.innerHTML = view.story;
    this.root.querySelector(".result-foot").textContent = view.foot;
    if (!first && !reduced)
      story.animate(
        [
          { opacity: 0.25, transform: "translateY(5px)" },
          { opacity: 1, transform: "translateY(0)" },
        ],
        { duration: 280, easing: "ease-out" },
      );
    document
      .querySelector(".result-tabs")
      ?.style.setProperty("--tab-index", ["v1", "v2", "vocal"].indexOf(phase));
    document.querySelectorAll("[data-phase]").forEach((b) => {
      const selected = b.dataset.phase === phase;
      b.setAttribute("aria-selected", String(selected));
      b.tabIndex = selected ? 0 : -1;
    });
  }
}
