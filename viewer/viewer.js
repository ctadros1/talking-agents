"use strict";
(function () {
  const data = window.REPLAYS;
  if (!data) { document.getElementById("missing").hidden = false; return; }

  const ITEM_NAMES = "ABCD";
  const ITEM_COLORS = ["#dc2626", "#ca8a04", "#0891b2", "#c026d3"];
  const SYMBOL_COLORS = ["#1d4ed8", "#b45309", "#047857", "#be123c", "#6d28d9", "#0e7490", "#4d7c0f", "#a21caf"];
  const PENTATONIC_HZ = [261.63, 293.66, 329.63, 392.0, 440.0, 523.25, 587.33, 659.25]; // symbol k -> note k
  const cfg = data.config;
  const vocal = cfg.channel === "vocal";
  const VARIANT_LABELS = {
    "comm/intact": vocal ? "Vocal: intact" : "Comm: intact",
    "comm/permuted": "Comm: permuted",
    "comm/shuffled": "Vocal: shuffled",
    "no_comm/silenced": "No comm",
  };
  const canvas = document.getElementById("grid");
  const ctx = canvas.getContext("2d");
  const $ = (id) => document.getElementById(id);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const state = { episode: Object.keys(data.episodes)[0], variant: Object.keys(data.variants)[0], frame: 0, timer: null };
  const sound = { on: false, audio: null, lastKey: null };

  function frames() { return data.variants[state.variant][state.episode].steps; }
  function record() { return data.variants[state.variant][state.episode].record; }
  function silent(s) { return vocal ? s === null || Number.isNaN(s) : s >= cfg.vocab_size; }
  function symbolLabel(s) { return silent(s) ? "·" : vocal ? (s >= 0 ? "+" : "") + s.toFixed(2) : String(s); }
  function symbolColor(s) { return vocal ? `hsl(${Math.round(((s + 1) / 2) * 280)}, 70%, 42%)` : SYMBOL_COLORS[s % SYMBOL_COLORS.length]; }
  function frequency(s) { return vocal ? 220 * 2 ** (s + 1) : PENTATONIC_HZ[s % PENTATONIC_HZ.length]; } // vocal: 220-880 Hz

  function playTone(hz) {
    const ac = sound.audio, now = ac.currentTime;
    const osc = ac.createOscillator(), gain = ac.createGain();
    osc.type = vocal ? "sine" : "triangle";
    osc.frequency.value = hz;
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.25, now + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.18);
    osc.connect(gain).connect(ac.destination);
    osc.start(now);
    osc.stop(now + 0.2);
  }

  function playHeard() {
    const key = `${state.variant}|${state.episode}|${state.frame}`;
    if (!sound.on || !sound.audio || key === sound.lastKey) return;
    sound.lastKey = key;
    const f = frames()[state.frame];
    if (f.received && !silent(f.received[1])) playTone(frequency(f.received[1]));
  }

  function drawGrid() {
    const g = cfg.grid_size, cell = canvas.width / g, ep = data.episodes[state.episode];
    const f = frames()[state.frame];
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let r = 0; r < g; r++) for (let c = 0; c < g; c++) {
      ctx.fillStyle = ep.walls[r][c] ? css("--wall") : c < cfg.wall_col ? css("--control") : css("--warehouse");
      ctx.fillRect(c * cell, r * cell, cell, cell);
    }
    const [rr, rc] = f.pos[1], rad = cfg.view_radius;
    ctx.fillStyle = css("--view");
    ctx.fillRect((rc - rad) * cell, (rr - rad) * cell, (2 * rad + 1) * cell, (2 * rad + 1) * cell);
    ctx.strokeStyle = css("--line");
    for (let i = 0; i <= g; i++) {
      ctx.beginPath(); ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, canvas.height); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i * cell); ctx.lineTo(canvas.width, i * cell); ctx.stroke();
    }
    const [dr, dc] = ep.depot;
    ctx.strokeStyle = css("--depot"); ctx.lineWidth = 4;
    ctx.strokeRect(dc * cell + 4, dr * cell + 4, cell - 8, cell - 8); ctx.lineWidth = 1;
    ep.items.forEach(([r, c], k) => {
      if (!f.item_present[k]) return;
      ctx.fillStyle = ITEM_COLORS[k];
      ctx.fillRect(c * cell + 7, r * cell + 7, cell - 14, cell - 14);
      if (k === ep.target) { ctx.strokeStyle = css("--fg"); ctx.lineWidth = 2; ctx.strokeRect(c * cell + 4, r * cell + 4, cell - 8, cell - 8); ctx.lineWidth = 1; }
      label(ITEM_NAMES[k], c, r, cell, "#fff");
    });
    [["--scout", "S"], ["--runner", "R"]].forEach(([color, name], a) => {
      const [r, c] = f.pos[a];
      ctx.fillStyle = css(color);
      ctx.beginPath(); ctx.arc((c + 0.5) * cell, (r + 0.5) * cell, cell * 0.36, 0, 2 * Math.PI); ctx.fill();
      label(f.carrying[a] >= 0 ? name + ITEM_NAMES[f.carrying[a]] : name, c, r, cell, "#fff");
    });
  }

  function label(text, c, r, cell, color) {
    ctx.fillStyle = color; ctx.font = `700 ${Math.round(cell * 0.34)}px system-ui`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(text, (c + 0.5) * cell, (r + 0.5) * cell);
  }

  function drawTape() {
    const fs = frames();
    for (const [id, key, agent] of [["tape-said", "sent", 0], ["tape-heard", "received", 1]]) {
      const row = $(id); row.replaceChildren();
      fs.forEach((f, t) => {
        if (f[key] === null) return;
        const s = f[key][agent], el = document.createElement("div");
        el.className = "cell" + (silent(s) ? " silence" : "") + (t === state.frame ? " now" : "");
        if (!silent(s)) el.style.background = symbolColor(s);
        el.textContent = vocal && !silent(s) ? "♪" : symbolLabel(s);
        el.title = `step ${t}: ${symbolLabel(s)}`;
        el.onclick = () => { stop(); state.frame = t; render(); };
        row.appendChild(el);
      });
    }
  }

  function drawPanel() {
    const fs = frames(), f = fs[state.frame], ep = data.episodes[state.episode], rec = record();
    const last = state.frame === fs.length - 1;
    $("step").textContent = `${f.t} / ${cfg.max_steps}`;
    $("target").textContent = `${ITEM_NAMES[ep.target]}`;
    $("said").textContent = f.sent ? symbolLabel(f.sent[0]) : "—";
    $("heard").textContent = f.received ? symbolLabel(f.received[1]) : "·";
    $("carrying").textContent = f.carrying[1] >= 0 ? ITEM_NAMES[f.carrying[1]] : "nothing";
    const outcome = $("outcome");
    outcome.textContent = last ? (rec.success ? `delivered ${ITEM_NAMES[ep.target]} in ${rec.steps} steps (optimal ${ep.runner_opt})` : "failed") : "in progress";
    outcome.className = last ? (rec.success ? "ok" : "bad") : "";
    $("scrub").max = fs.length - 1; $("scrub").value = state.frame;
  }

  function render() { drawGrid(); drawTape(); drawPanel(); playHeard(); }
  function stop() { clearInterval(state.timer); state.timer = null; $("play").textContent = "Play"; }
  function play() {
    if (state.frame >= frames().length - 1) state.frame = 0;
    $("play").textContent = "Pause";
    state.timer = setInterval(() => {
      if (state.frame >= frames().length - 1) { stop(); return; }
      state.frame++; render();
    }, Number($("speed").value));
  }

  const epSelect = $("episode");
  Object.entries(data.episodes).forEach(([id, ep]) => {
    const ok = data.variants["comm/intact"][id].record.success;
    epSelect.add(new Option(`#${id}: target ${ITEM_NAMES[ep.target]} (${ok ? "comm succeeds" : "comm fails"})`, id));
  });
  epSelect.onchange = () => { stop(); state.episode = epSelect.value; state.frame = 0; render(); };
  const variantBox = $("variants");
  Object.keys(data.variants).forEach((v) => {
    const b = document.createElement("button");
    b.textContent = VARIANT_LABELS[v] || v; b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", String(v === state.variant));
    b.onclick = () => {
      stop(); state.variant = v; state.frame = 0;
      variantBox.querySelectorAll("button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
      render();
    };
    variantBox.appendChild(b);
  });
  $("sound").onclick = () => {
    sound.audio = sound.audio || new (window.AudioContext || window.webkitAudioContext)();
    sound.audio.resume();
    sound.on = !sound.on; sound.lastKey = null;
    $("sound").setAttribute("aria-pressed", String(sound.on));
    $("sound").textContent = sound.on ? "Sound on" : "Sound off";
    playHeard();
  };
  $("play").onclick = () => (state.timer ? stop() : play());
  $("prev").onclick = () => { stop(); state.frame = Math.max(0, state.frame - 1); render(); };
  $("next").onclick = () => { stop(); state.frame = Math.min(frames().length - 1, state.frame + 1); render(); };
  $("scrub").oninput = (e) => { stop(); state.frame = Number(e.target.value); render(); };
  document.addEventListener("keydown", (e) => {
    if (e.key === " ") { e.preventDefault(); $("play").click(); }
    if (e.key === "ArrowLeft") $("prev").click();
    if (e.key === "ArrowRight") $("next").click();
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
  render();
})();
