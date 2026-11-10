import { Policy, Episode } from "./engine.mjs";
const $ = (id) => document.getElementById(id),
  colors = ["#c66b49", "#537fa4", "#ba9d3c", "#6c9270"],
  letters = ["A", "B", "C", "D"];
const symbol = (n) => (n === 8 ? "–" : String(n));
let episode,
  bank,
  playing = false,
  timer,
  requestVersion = 0,
  ready = false,
  summary;
const models = new Map();
async function json(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Could not load ${url} (${r.status})`);
  return r.json();
}
async function loadModel(condition) {
  if (!models.has(condition))
    models.set(
      condition,
      (async () => {
        const manifest = await json(`models/${condition}.json`),
          r = await fetch(`models/${condition}.bin`);
        if (!r.ok) throw new Error("Could not load trained model weights.");
        const bytes = await r.arrayBuffer();
        if (crypto.subtle) {
          const digest = Array.from(
            new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
            (x) => x.toString(16).padStart(2, "0"),
          ).join("");
          if (digest !== manifest.binary_sha256)
            throw new Error("Model integrity check failed. Reload this page.");
        }
        const weights = new Float32Array(bytes);
        return manifest.agents.map((t) => new Policy(t, weights));
      })(),
    );
  return models.get(condition);
}
const descriptions = {
  intact: "The scout's symbol reaches the runner one step later.",
  permuted:
    "A fixed random relabeling is applied for this episode. The policies are unchanged.",
  silenced:
    "The communicating policies still send symbols, but both agents hear silence.",
  no_comm:
    "A different pair of policies, trained with a silent channel from the start.",
};
function pause() {
  playing = false;
  clearTimeout(timer);
  $("play").textContent = episode?.world.done
    ? "▶ Run again"
    : "▶ Run episode";
}
async function reset() {
  pause();
  ready = false;
  const version = ++requestVersion;
  ["play", "step", "restart"].forEach((id) => ($(id).disabled = true));
  const mode = $("channel").value;
  $("channel-help").textContent = descriptions[mode];
  $("model-status").textContent = "Loading trained agents…";
  try {
    const policies = await loadModel(mode === "no_comm" ? "no_comm" : "comm");
    if (version !== requestVersion) return;
    const scenario = Math.max(
      1,
      Math.min(200, Math.floor(Number($("scenario").value) || 1)),
    );
    $("scenario").value = scenario;
    const seed = Math.max(
      0,
      Math.min(4294967295, Math.floor(Number($("seed").value) || 0)),
    );
    $("seed").value = seed;
    episode = new Episode(bank[scenario - 1], policies, { seed, mode });
    ready = true;
    ["play", "step", "restart"].forEach((id) => ($(id).disabled = false));
    $("model-status").textContent =
      `v1 · seed 19 · ${mode === "no_comm" ? "no-message model" : "communication model"}`;
    render();
  } catch (error) {
    $("model-status").textContent = "Model could not load";
    $("outcome").textContent = error.message;
    console.error(error);
  }
}
function tick() {
  if (!ready) return;
  episode.step();
  render();
  if (episode.world.done) {
    pause();
    $("step").disabled = true;
  } else if (playing) timer = setTimeout(tick, Number($("speed").value));
}
$("play").addEventListener("click", () => {
  if (!ready) return;
  if (playing) {
    pause();
    return;
  }
  if (episode.world.done) {
    episode.reset();
    $("step").disabled = false;
    render();
  }
  playing = true;
  $("play").textContent = "Ⅱ Pause";
  tick();
});
$("step").addEventListener("click", () => {
  pause();
  tick();
});
$("restart").addEventListener("click", reset);
for (const id of ["channel", "scenario", "seed"])
  $(id).addEventListener("change", reset);
$("new-scenario").addEventListener("click", () => {
  $("scenario").value = (Number($("scenario").value) % 200) + 1;
  reset();
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) pause();
});
const canvas = $("world"),
  ctx = canvas.getContext("2d");
function draw() {
  const w = episode.world,
    cell = canvas.width / 12;
  ctx.clearRect(0, 0, 720, 720);
  ctx.fillStyle = "#fafaf4";
  ctx.fillRect(0, 0, 720, 720);
  for (let r = 0; r < 12; r++)
    for (let c = 0; c < 12; c++) {
      ctx.fillStyle = w.walls[r][c] ? "#a9b4a3" : c < 4 ? "#eeecf4" : "#f1f3e9";
      ctx.fillRect(c * cell + 1, r * cell + 1, cell - 2, cell - 2);
    }
  const [rr, rc] = w.pos[1];
  ctx.fillStyle = "rgba(96,137,76,.09)";
  ctx.strokeStyle = "#8ba27c";
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 5]);
  const left = Math.max(0, rc - 3) * cell,
    top = Math.max(0, rr - 3) * cell,
    right = Math.min(12, rc + 4) * cell,
    bottom = Math.min(12, rr + 4) * cell;
  ctx.fillRect(left, top, right - left, bottom - top);
  ctx.strokeRect(left + 2, top + 2, right - left - 4, bottom - top - 4);
  ctx.setLineDash([]);
  const [dr, dc] = w.depot;
  ctx.fillStyle = "#204f3b";
  ctx.fillRect(dc * cell + 9, dr * cell + 9, cell - 18, cell - 18);
  ctx.fillStyle = "#fffef9";
  ctx.font = "bold 30px Arial";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("⌂", (dc + 0.5) * cell, (dr + 0.5) * cell);
  for (let j = 0; j < 4; j++)
    if (w.item_present[j]) {
      const [r, c] = w.item_pos[j];
      ctx.fillStyle = colors[j];
      ctx.beginPath();
      ctx.roundRect(c * cell + 12, r * cell + 12, cell - 24, cell - 24, 6);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "bold 22px Arial";
      ctx.fillText(letters[j], (c + 0.5) * cell, (r + 0.5) * cell + 1);
    }
  for (let a = 0; a < 2; a++) {
    const [r, c] = w.pos[a];
    ctx.fillStyle = a === 0 ? "#7861ad" : "#c37532";
    ctx.strokeStyle = "#fffef9";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc((c + 0.5) * cell, (r + 0.5) * cell, 21, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.font = "bold 23px Arial";
    ctx.fillText(a === 0 ? "S" : "R", (c + 0.5) * cell, (r + 0.5) * cell + 1);
    if (w.carrying[a] >= 0) {
      ctx.fillStyle = colors[w.carrying[a]];
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc((c + 0.82) * cell, (r + 0.2) * cell, 9, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }
}
function render() {
  const w = episode.world,
    last = episode.tape.at(-1);
  $("target").textContent = `Collect item ${letters[w.target]}`;
  $("target").style.color = colors[w.target];
  $("steps").textContent = `${w.t} / 32`;
  $("carrying").textContent =
    w.carrying[1] < 0 ? "Nothing" : `Item ${letters[w.carrying[1]]}`;
  $("sent").textContent = last ? symbol(last.sent[0]) : "—";
  $("heard").textContent =
    last && last.heard[1] !== 8 ? symbol(last.heard[1]) : "Silence";
  $("reward").textContent = episode.return.toFixed(2);
  const end = w.done;
  $("outcome").textContent = end
    ? episode.outcome.success
      ? "Correct delivery. Goal reached."
      : w.t >= 32
        ? "Time limit reached. Try another channel."
        : "Wrong item delivered. Episode ended."
    : w.t
      ? "Agents are choosing their next action."
      : "Ready when you are.";
  $("outcome").classList.toggle("fail", end && !episode.outcome.success);
  for (const [id, key, agent] of [
    ["tape-sent", "sent", 0],
    ["tape-heard", "heard", 1],
  ]) {
    $(id).replaceChildren(
      ...Array.from({ length: 32 }, (_, i) => {
        const el = document.createElement("span"),
          row = episode.tape[i];
        el.className = "tape-cell" + (!row ? " empty" : "");
        el.textContent = row ? symbol(row[key][agent]) : "·";
        el.title = `Step ${i + 1}${row ? ": " + (row[key][agent] === 8 ? "silence" : "symbol " + row[key][agent]) : ""}`;
        return el;
      }),
    );
  }
  draw();
}
const pct = (x) => (100 * x).toFixed(1) + "%",
  pp = (x) => (100 * x).toFixed(1) + " pp",
  ci = (x, unit = pct) => `${unit(x.lo)}–${unit(x.hi)} · 95% CI`;
const metric = (label, x, highlight = false, unit = pct) =>
  `<article class="metric${highlight ? " highlight" : ""}"><div class="metric-label">${label}</div><div class="metric-value">${unit(x.mean)}</div><div class="metric-ci">${ci(x, unit)}</div></article>`;
const bar = (label, x) =>
  `<div class="bar-row"><span>${label}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.max(0, Math.min(100, x * 100))}%"></div></div><strong>${pct(x)}</strong></div>`;
function showResults(phase) {
  if (!summary) return;
  const d = summary[phase],
    c = d.conditions;
  let cards, story, rows, foot;
  if (phase === "v2") {
    const seen = c.comm.held_out.success,
      unseen = c.comm.unseen.success,
      gain = d.comparisons.unseen.comm_minus_no_comm;
    cards =
      metric("Familiar combinations · new maps", seen, true) +
      metric("Unseen color–shape combinations", unseen) +
      metric("Unseen gain over no messages", gain, false, pp);
    story =
      "<strong>Communication transferred; the stronger generalization claim did not.</strong> Two-word messages helped on familiar color–shape combinations. On unseen combinations, the gain was positive but below the frozen 15 percentage point requirement. Claim A passed; Claim B failed; the hierarchy stopped before Claim C.";
    rows =
      bar("Learned messages · seen", seen.mean) +
      bar("No messages · seen", c.no_comm.held_out.success.mean) +
      bar("Learned messages · unseen", unseen.mean) +
      bar("No messages · unseen", c.no_comm.unseen.success.mean) +
      bar("Oracle · unseen", c.oracle.unseen.success.mean);
    foot = `80 learned runs · 20 seeds × 4 conditions · 20.0 million environment steps per run. “Seen” uses familiar combinations on held-out layouts. The unseen support also permits a target-identification shortcut (${pct(summary.support.support_aware_no_message_identification_rate)} theoretically, before navigation); it is not a measured learned-policy success rate. This limits a compositional-language interpretation.`;
  } else {
    const s = c.comm.success,
      b = c.no_comm.success,
      g = d.comparisons.comm_minus_no_comm;
    cards =
      metric("Success with learned messages", s, true) +
      metric("Success without messages", b) +
      metric("Communication advantage", g, false, pp);
    if (phase === "v1") {
      story = `<strong>Stable, learned symbols improved coordination.</strong> The communicating pairs beat independently trained no-message and random-message controls. Relabeling their symbols reduced success by ${pp(d.comparisons.intact_minus_permuted.mean)} (${ci(d.comparisons.intact_minus_permuted, pp)}), supporting a causal role for stable symbol identities. The pre-registered criterion was met.`;
      rows =
        bar("Learned messages", s.mean) +
        bar("No messages", b.mean) +
        bar("Random messages", c.random_msg.success.mean) +
        bar("Oracle · target revealed", c.oracle.success.mean);
      foot =
        "80 learned runs · 20 seeds × 4 conditions · 10.0 million environment steps per run. Bars compare separately trained conditions. This study used a cue-directed signalling objective; useful communication is not evidence that unmodified PPO learns it equally well.";
    } else {
      story = `<strong>A continuous pitch channel also carried useful information.</strong> Both policies learn with a pitch-valued message instead of a discrete symbol. Shuffling the channel reduced success by ${pp(d.comparisons.intact_minus_shuffled.mean)}. This is an exploratory extension with a different training budget, so its score is not a controlled ranking against v1.`;
      rows = bar("Learned vocal messages", s.mean) + bar("No messages", b.mean);
      foot =
        "40 learned runs · 20 seeds × 2 conditions · 20.0 million environment steps per run. Pitch is continuous; discretized bins are only for analysis. Use the vocal replay viewer to hear saved episodes.";
    }
  }
  $("result-content").innerHTML =
    `<div class="metric-grid">${cards}</div><p class="result-story">${story}</p><div class="bar-table">${rows}</div><p class="result-foot">${foot}</p>`;
  $("result-content").setAttribute("aria-labelledby", "tab-" + phase);
  document.querySelectorAll("[data-phase]").forEach((b) => {
    b.setAttribute("aria-selected", String(b.dataset.phase === phase));
    b.tabIndex = b.dataset.phase === phase ? 0 : -1;
  });
}
document.querySelectorAll("[data-phase]").forEach((button) => {
  button.addEventListener("click", () => showResults(button.dataset.phase));
  button.addEventListener("keydown", (event) => {
    if (!["ArrowRight", "ArrowLeft", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const tabs = [...document.querySelectorAll("[data-phase]")],
      index = tabs.indexOf(button),
      next =
        event.key === "Home"
          ? 0
          : event.key === "End"
            ? 2
            : (index + (event.key === "ArrowRight" ? 1 : 2)) % 3;
    showResults(tabs[next].dataset.phase);
    tabs[next].focus();
  });
});
(async () => {
  try {
    summary = await json("data/results.json");
    showResults("v1");
    const a = summary.ablation;
    $("ablation-body").innerHTML =
      `<p class="result-story">With objective: <strong>${pct(a.biased.success.mean)}</strong>. Original objective: <strong>${pct(a.unbiased.success.mean)}</strong>. Paired advantage: <strong>${pp(a.biased_minus_unbiased.mean)}</strong> (${ci(a.biased_minus_unbiased, pp)}).</p>`;
  } catch (error) {
    $("result-content").textContent =
      "Saved results could not load. Read the report or download the results below.";
    console.error(error);
  }
})();
(async () => {
  try {
    bank = await json("data/scenarios.json");
    await reset();
  } catch (error) {
    $("model-status").textContent = "Unable to load scenarios";
    $("outcome").textContent = error.message;
    console.error(error);
  }
})();
