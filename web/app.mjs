import { Policy, Episode } from "./engine.mjs";
import { buttonLabel, canvasIcon } from "./icons.mjs";
import { createLiveScene } from "./scene3d.mjs";
import { EvidenceView, pct, pp, ci } from "./evidence.mjs";
const $ = (id) => document.getElementById(id),
  colors = ["#c66b49", "#537fa4", "#ba9d3c", "#6c9270"],
  letters = ["A", "B", "C", "D"];
const v2Colors = ["#dc5b4b", "#bd8b13", "#4e8754", "#4f88be", "#9572bf"],
  colorNames = ["red", "yellow", "green", "blue", "purple"],
  shapeNames = ["cube", "sphere", "cone", "cylinder", "ring"];
const symbol = (n) => (n === 8 ? "–" : String(n)),
  message = (words) =>
    Array.isArray(words) ? words.map(symbol).join(" · ") : symbol(words);
const itemName = (world, k) =>
  world.spec.item_attrs
    ? `${colorNames[world.spec.item_attrs[k][0]]} ${shapeNames[world.spec.item_attrs[k][1]]}`
    : `item ${letters[k]}`;
let episode,
  playing = false,
  timer,
  requestVersion = 0,
  ready = false,
  summary,
  simulation = "v2",
  scene,
  scenePromise,
  sceneFailed = false,
  selectedEvidence = "v1";
const models = new Map(),
  banks = new Map();
async function json(url) {
  const response = await fetch(url);
  if (!response.ok)
    throw new Error(`Could not load ${url} (${response.status})`);
  return response.json();
}
async function loadModel(key) {
  if (!models.has(key))
    models.set(
      key,
      (async () => {
        const manifest = await json(`models/${key}.json`),
          response = await fetch(`models/${key}.bin`);
        if (!response.ok)
          throw new Error("Could not load trained model weights.");
        const bytes = await response.arrayBuffer();
        if (crypto.subtle) {
          const digest = Array.from(
            new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
            (x) => x.toString(16).padStart(2, "0"),
          ).join("");
          if (digest !== manifest.binary_sha256)
            throw new Error("Model integrity check failed. Reload this page.");
        }
        const weights = new Float32Array(bytes);
        return {
          manifest,
          policies: manifest.agents.map((t) => new Policy(t, weights)),
        };
      })(),
    );
  return models.get(key);
}
function loadBank(phase) {
  if (!banks.has(phase))
    banks.set(
      phase,
      json(phase === "v1" ? "data/scenarios.json" : "data/scenarios-v2.json"),
    );
  return banks.get(phase);
}
const descriptions = {
  intact:
    "Both words arrive one step later. Their symbol meanings were learned.",
  permuted:
    "A separate fixed random relabeling is applied to each word for this episode.",
  permuted_w1:
    "Only the first word is relabeled; the second word reaches its partner intact.",
  permuted_w2:
    "Only the second word is relabeled; the first word reaches its partner intact.",
  silenced:
    "The communicating policies still send messages, but both agents hear silence.",
  no_comm:
    "A different pair of policies, trained with a silent channel from the start.",
};
function pause() {
  playing = false;
  clearTimeout(timer);
  buttonLabel(
    $("play"),
    "play",
    episode?.world.done ? "Run again" : "Run episode",
  );
}
function setSimulation(phase) {
  simulation = phase;
  document.querySelectorAll("[data-simulation]").forEach((b) => {
    const selected = b.dataset.simulation === phase;
    b.setAttribute("aria-selected", String(selected));
    b.tabIndex = selected ? 0 : -1;
  });
  $("simulator").setAttribute("aria-labelledby", "sim-" + phase);
  $("split-field").hidden = phase === "v1";
  document.querySelectorAll("[data-v2-only]").forEach((option) => {
    option.hidden = phase === "v1";
    option.disabled = phase === "v1";
  });
  if (phase === "v1" && $("channel").value.startsWith("permuted_w"))
    $("channel").value = "intact";
  $("channel").querySelector('[value="permuted"]').textContent =
    phase === "v2" ? "Scramble both words" : "Scramble symbol meanings";
  reset();
}
async function reset() {
  pause();
  ready = false;
  const version = ++requestVersion,
    phase = simulation,
    mode = $("channel").value,
    split = $("split").value;
  ["play", "step", "restart"].forEach((id) => ($(id).disabled = true));
  $("model-status").textContent = "Loading trained agents…";
  $("channel-help").textContent =
    phase === "v1" && mode === "intact"
      ? "The scout's symbol reaches the runner one step later."
      : phase === "v1" && mode === "permuted"
        ? "A fixed random relabeling is applied for this episode."
        : descriptions[mode];
  try {
    const [{ manifest, policies }, bank] = await Promise.all([
      loadModel(
        (phase === "v2" ? "v2-" : "") +
          (mode === "no_comm" ? "no_comm" : "comm"),
      ),
      loadBank(phase),
    ]);
    if (version !== requestVersion) return;
    const scenarios = phase === "v1" ? bank : bank[split];
    const scenario = Math.max(
      1,
      Math.min(scenarios.length, Math.floor(Number($("scenario").value) || 1)),
    );
    $("scenario").value = scenario;
    $("scenario").max = scenarios.length;
    const seed = Math.max(
      0,
      Math.min(4294967295, Math.floor(Number($("seed").value) || 0)),
    );
    $("seed").value = seed;
    const next = new Episode(scenarios[scenario - 1], policies, {
      seed,
      mode,
      hidden: manifest.hidden,
    });
    if (phase === "v2" && !scene && !sceneFailed) {
      try {
        scenePromise ??= createLiveScene($("scene-live"));
        scene = await scenePromise;
      } catch (error) {
        sceneFailed = true;
        console.warn(
          "3D unavailable; the live simulation will use its 2D view.",
          error.message,
        );
      }
    }
    if (version !== requestVersion) return;
    episode = next;
    ready = true;
    const use3d = phase === "v2" && scene && !sceneFailed;
    $("scene-live").hidden = !use3d;
    canvas.hidden = !!use3d;
    $("view-help").hidden = phase === "v1";
    $("view-help").textContent = use3d
      ? "Drag to rotate · camera controls to zoom · keyboard arrows to orbit"
      : "3D is unavailable on this device. The live model is running in 2D.";
    if (scene) {
      scene.setActive(!!use3d);
      if (use3d) scene.load(episode.world);
    }
    ["play", "step", "restart"].forEach((id) => ($(id).disabled = false));
    $("model-status").textContent =
      `${phase} · seed ${manifest.seed} · ${mode === "no_comm" ? "no-message model" : "communication model"}`;
    $("sandbox-caption").textContent =
      `Live browser inference: ${phase} final checkpoints, training seed ${manifest.seed}. ${phase === "v2" ? "Two-word messages; 200 fixed familiar and 200 fixed never-seen target scenarios. The 3D scene renders the grid state." : "Choose from the 200 fixed held-out benchmark scenarios."} Actions are sampled live; JavaScript arithmetic and random sampling differ from PyTorch. These episodes illustrate behavior; published statistics come from the canonical Python evaluation across all training seeds.`;
    render();
  } catch (error) {
    if (version !== requestVersion) return;
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
    if (simulation === "v2" && scene) scene.load(episode.world);
    render();
  }
  playing = true;
  buttonLabel($("play"), "pause", "Pause");
  tick();
});
$("step").addEventListener("click", () => {
  pause();
  tick();
});
$("restart").addEventListener("click", reset);
for (const id of ["channel", "scenario", "seed", "split"])
  $(id).addEventListener("change", reset);
$("new-scenario").addEventListener("click", () => {
  $("scenario").value =
    (Number($("scenario").value) % Number($("scenario").max)) + 1;
  reset();
});
const simTabs = [...document.querySelectorAll("[data-simulation]")];
simTabs.forEach((button) => {
  button.addEventListener("click", () =>
    setSimulation(button.dataset.simulation),
  );
  button.addEventListener("keydown", (event) => {
    if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      const index =
        event.key === "Home"
          ? 0
          : event.key === "End"
            ? 1
            : 1 - simTabs.indexOf(button);
      setSimulation(simTabs[index].dataset.simulation);
      simTabs[index].focus();
    }
  });
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) pause();
  else scene?.schedule();
});
$("scene-live").addEventListener("renderer-lost", () => {
  sceneFailed = true;
  scene?.setActive(false);
  $("scene-live").hidden = true;
  canvas.hidden = false;
  $("view-help").textContent =
    "The 3D view was interrupted. Live inference continues in 2D.";
  draw();
});
const canvas = $("world"),
  ctx = canvas.getContext("2d");
function draw() {
  if (simulation === "v2" && scene && !sceneFailed) {
    scene.update(episode.world, episode.tape.at(-1));
    return;
  }
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
  canvasIcon(
    ctx,
    "house",
    dc * cell + 18,
    dr * cell + 18,
    cell - 36,
    "#fffef9",
  );
  for (let j = 0; j < 4; j++)
    if (w.item_present[j]) {
      const [r, c] = w.item_pos[j];
      ctx.fillStyle = w.spec.item_attrs
        ? v2Colors[w.spec.item_attrs[j][0]]
        : colors[j];
      ctx.beginPath();
      ctx.roundRect(c * cell + 12, r * cell + 12, cell - 24, cell - 24, 6);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "bold 22px Arial";
      ctx.fillText(
        w.spec.item_attrs
          ? ["□", "○", "△", "▱", "◎"][w.spec.item_attrs[j][1]]
          : letters[j],
        (c + 0.5) * cell,
        (r + 0.5) * cell + 1,
      );
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
      ctx.fillStyle = w.spec.item_attrs
        ? v2Colors[w.spec.item_attrs[w.carrying[a]][0]]
        : colors[w.carrying[a]];
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
  const world = episode.world,
    last = episode.tape.at(-1);
  $("target").textContent = `Collect ${itemName(world, world.target)}`;
  $("target").style.color = world.spec.item_attrs
    ? v2Colors[world.spec.item_attrs[world.target][0]]
    : colors[world.target];
  $("steps").textContent = `${world.t} / 32`;
  $("carrying").textContent =
    world.carrying[1] < 0 ? "Nothing" : itemName(world, world.carrying[1]);
  $("sent").textContent = last ? message(last.sent[0]) : "—";
  $("heard").textContent = last ? message(last.heard[1]) : "Silence";
  $("reward").textContent = episode.return.toFixed(2);
  const ended = world.done;
  $("outcome").textContent = ended
    ? episode.outcome.success
      ? "Correct delivery. Goal reached."
      : world.t >= 32
        ? "Time limit reached. Try another channel."
        : "Wrong item delivered. Episode ended."
    : world.t
      ? "Agents are choosing their next action."
      : "Ready when you are.";
  $("outcome").classList.toggle("fail", ended && !episode.outcome.success);
  for (const [id, key, agent] of [
    ["tape-sent", "sent", 0],
    ["tape-heard", "heard", 1],
  ]) {
    $(id).classList.toggle("two-words", episode.words === 2);
    $(id).replaceChildren(
      ...Array.from({ length: 32 }, (_, i) => {
        const el = document.createElement("span"),
          row = episode.tape[i];
        el.className = "tape-cell" + (!row ? " empty" : "");
        el.textContent = row ? message(row[key][agent]) : "·";
        el.title = `Step ${i + 1}${row ? ": " + message(row[key][agent]) : ""}`;
        return el;
      }),
    );
  }
  draw();
}
const evidence = new EvidenceView($("result-content"));
function showResults(phase) {
  selectedEvidence = phase;
  if (summary) evidence.update(summary, phase);
}
const evidenceTabs = [...document.querySelectorAll("[data-phase]")];
evidenceTabs.forEach((button) => {
  button.addEventListener("click", () => showResults(button.dataset.phase));
  button.addEventListener("keydown", (event) => {
    if (!["ArrowRight", "ArrowLeft", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const index = evidenceTabs.indexOf(button),
      next =
        event.key === "Home"
          ? 0
          : event.key === "End"
            ? 2
            : (index + (event.key === "ArrowRight" ? 1 : 2)) % 3;
    showResults(evidenceTabs[next].dataset.phase);
    evidenceTabs[next].focus();
  });
});
(async () => {
  try {
    summary = await json("data/results.json");
    showResults(selectedEvidence);
    const a = summary.ablation;
    $("ablation-body").innerHTML =
      `<p class="result-story">With objective: <strong>${pct(a.biased.success.mean)}</strong>. Original objective: <strong>${pct(a.unbiased.success.mean)}</strong>. Paired advantage: <strong>${pp(a.biased_minus_unbiased.mean)}</strong> (${ci(a.biased_minus_unbiased, pp)}).</p>`;
  } catch (error) {
    $("result-content").textContent =
      "Saved results could not load. Read the report or download the results below.";
    console.error(error);
  }
})();
setSimulation("v2");
