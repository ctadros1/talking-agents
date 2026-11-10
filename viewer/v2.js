"use strict";
(function () {
  const $ = (id) => document.getElementById(id);
  const data = window.REPLAYS_V2;
  if (!data || !window.THREE) { $("missing").hidden = false; return; }
  if (data.development === true) {
    $("development-banner").hidden = false;
    $("replay-source").textContent = "Scripted reference development replays.";
    $("dictionary-title").textContent = "The scripted scout's dictionary";
    $("dictionary-section").setAttribute("aria-label", "Scripted reference dictionary");
    $("dictionary-explainer").textContent = "How often the scripted scout's first message uses each symbol for each target color or shape in this development replay source. Outlined cells show the current target and words. These patterns are scripted references, not learned meanings.";
  }

  const cfg = data.config;
  const COLOR_HEX = ["#dc2626", "#eab308", "#16a34a", "#2563eb", "#9333ea"]; // red, yellow, green, blue, purple
  const SYMBOL_COLORS = ["#1d4ed8", "#b45309", "#047857", "#be123c", "#6d28d9", "#0e7490", "#4d7c0f", "#a21caf"];
  const PENTATONIC_HZ = [261.63, 293.66, 329.63, 392.0, 440.0, 523.25, 587.33, 659.25]; // symbol k -> note k
  const VARIANT_LABELS = {
    "comm/intact": "Talking",
    "comm/permuted": "Both words scrambled",
    "comm/permuted_w1": "Word 1 scrambled",
    "no_comm/silenced": "No talking",
  };
  const BODY = [0xe7e5e4, 0xfb923c]; // scout (pale stone), runner (orange): unlike any item color
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const itemName = ([c, s]) => `${cfg.colors[c]} ${cfg.shapes[s]}`;
  const silent = (s) => s >= cfg.vocab_size;
  const wrap = (x) => Math.atan2(Math.sin(x), Math.cos(x));

  const state = { episode: null, variant: Object.keys(data.variants)[0], frame: 0, timer: null };
  const sound = { on: false, audio: null, lastKey: null };
  const frames = () => data.variants[state.variant][state.episode].steps;
  const record = () => data.variants[state.variant][state.episode].record;
  const episode = () => data.episodes[state.episode];

  // ---- scene: grid (row, col) maps to world (x = col, z = row), centred on the board ----
  const sceneBox = $("scene");
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  sceneBox.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 4 / 3, 0.1, 200);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x556070, 2.2));
  const sun = new THREE.DirectionalLight(0xffffff, 2.4);
  sun.position.set(-6, 14, 9);
  scene.add(sun);
  const board = new THREE.Group();
  scene.add(board);
  const g = cfg.grid_size;
  const at = (r, c, y = 0) => new THREE.Vector3(c - (g - 1) / 2, y, r - (g - 1) / 2);

  const DEFAULT_ORBIT = { theta: -0.5, phi: 0.92, radius: 19 };
  const orbit = { ...DEFAULT_ORBIT };
  function placeCamera() {
    const { theta, phi, radius } = orbit;
    camera.position.set(radius * Math.sin(phi) * Math.sin(theta), radius * Math.cos(phi), radius * Math.sin(phi) * Math.cos(theta));
    camera.lookAt(0, 0, 0);
  }
  let drag = null;
  const canvas = renderer.domElement;
  canvas.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    orbit.theta -= (e.clientX - drag.x) * 0.008;
    orbit.phi = Math.min(1.45, Math.max(0.2, orbit.phi - (e.clientY - drag.y) * 0.008));
    drag = { x: e.clientX, y: e.clientY };
  });
  canvas.addEventListener("pointerup", () => { drag = null; });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    orbit.radius = Math.min(34, Math.max(9, orbit.radius * (1 + e.deltaY * 0.001)));
  }, { passive: false });
  $("reset-camera").onclick = () => Object.assign(orbit, DEFAULT_ORBIT);

  const standard = (color, roughness = 0.6) => new THREE.MeshStandardMaterial({ color, roughness });
  const SHAPES = [
    () => new THREE.BoxGeometry(0.5, 0.5, 0.5),
    () => new THREE.SphereGeometry(0.3, 28, 18),
    () => new THREE.ConeGeometry(0.3, 0.62, 28),
    () => new THREE.CylinderGeometry(0.25, 0.25, 0.55, 28),
    () => new THREE.TorusGeometry(0.22, 0.09, 14, 32), // stands upright: the ring faces the +z axis
  ];
  function itemModel([color, shape]) {
    const group = new THREE.Group();
    const mesh = new THREE.Mesh(SHAPES[shape](), standard(COLOR_HEX[color], 0.4));
    mesh.position.y = 0.32;
    group.add(mesh);
    return group;
  }
  function creature(bodyColor) {
    const group = new THREE.Group();
    const body = new THREE.Mesh(new THREE.SphereGeometry(0.34, 28, 20), standard(bodyColor, 0.5));
    body.scale.set(1, 1.15, 1);
    body.position.y = 0.42;
    group.add(body);
    const white = standard(0xffffff, 0.3), black = standard(0x111111, 0.3);
    for (const side of [-1, 1]) { // eyes look along +z, the creature's forward direction
      const eye = new THREE.Mesh(new THREE.SphereGeometry(0.085, 16, 12), white);
      eye.position.set(0.13 * side, 0.58, 0.27);
      const pupil = new THREE.Mesh(new THREE.SphereGeometry(0.045, 12, 10), black);
      pupil.position.set(0.13 * side, 0.58, 0.345);
      group.add(eye, pupil);
    }
    return group;
  }

  const actors = { items: [], agents: [], heading: [0, 0] };
  function buildBoard() {
    const ep = episode();
    board.clear();
    scene.background = new THREE.Color(css("--bg"));
    const tile = new THREE.BoxGeometry(0.96, 0.08, 0.96), block = new THREE.BoxGeometry(1, 0.9, 1);
    const control = standard(css("--control"), 0.9), warehouse = standard(css("--warehouse"), 0.9);
    const wall = standard(css("--wall"), 0.8);
    for (let r = 0; r < g; r++) for (let c = 0; c < g; c++) {
      const isWall = ep.walls[r][c] === 1;
      const mesh = new THREE.Mesh(isWall ? block : tile, isWall ? wall : c < cfg.wall_col ? control : warehouse);
      mesh.position.copy(at(r, c, isWall ? 0.45 : -0.04));
      board.add(mesh);
    }
    const depot = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.42, 0.05, 32), standard(css("--depot"), 0.5));
    depot.position.copy(at(ep.depot[0], ep.depot[1], 0.03));
    board.add(depot);
    actors.items = ep.attrs.map((attrs, k) => {
      const model = itemModel(attrs);
      model.position.copy(at(ep.items[k][0], ep.items[k][1]));
      board.add(model);
      return model;
    });
    actors.agents = BODY.map((color) => { const a = creature(color); board.add(a); return a; });
    actors.heading = [Math.PI / 2, Math.PI]; // scout faces the wall (east), runner faces north
    updateActors(true);
  }

  function itemGoal(k, f) { // on the floor, carried above an agent, or gone (delivered)
    for (let a = 0; a < 2; a++) if (f.carrying[a] === k) { const [r, c] = f.pos[a]; return at(r, c, 0.8); }
    if (f.item_present[k]) { const [r, c] = episode().items[k]; return at(r, c); }
    return null;
  }
  function updateActors(snap) {
    const f = frames()[state.frame];
    f.pos.forEach(([r, c], a) => {
      const agent = actors.agents[a], goal = at(r, c);
      const dx = goal.x - agent.position.x, dz = goal.z - agent.position.z;
      if (Math.abs(dx) + Math.abs(dz) > 0.05) actors.heading[a] = Math.atan2(dx, dz);
      agent.position.lerp(goal, snap ? 1 : 0.18);
      agent.position.y = snap ? 0 : 0.04 * Math.abs(Math.sin(performance.now() / 160 + a * 1.7));
      agent.rotation.y += wrap(actors.heading[a] - agent.rotation.y) * (snap ? 1 : 0.2);
    });
    actors.items.forEach((model, k) => {
      const goal = itemGoal(k, f);
      model.visible = goal !== null;
      if (goal) model.position.lerp(goal, snap ? 1 : 0.25);
    });
  }

  function wordChip(s) {
    const el = document.createElement("span");
    el.className = "word" + (silent(s) ? " silence" : "");
    el.textContent = silent(s) ? "·" : String(s);
    if (!silent(s)) el.style.background = SYMBOL_COLORS[s];
    return el;
  }
  function message(words) {
    const box = document.createDocumentFragment();
    words.forEach((s) => box.appendChild(wordChip(s)));
    return box;
  }
  function swatch(color) {
    const el = document.createElement("span");
    el.className = "swatch";
    el.style.background = COLOR_HEX[color];
    return el;
  }
  function placeBubble(el, a) {
    if (el.hidden || !actors.agents[a]) return;
    const p = actors.agents[a].position.clone();
    p.y += 1.2;
    p.project(camera);
    el.style.left = `${((p.x + 1) / 2) * sceneBox.clientWidth}px`;
    el.style.top = `${((1 - p.y) / 2) * sceneBox.clientHeight}px`;
  }

  function drawPanel() {
    const fs = frames(), f = fs[state.frame], ep = episode(), rec = record();
    const last = state.frame === fs.length - 1, target = ep.attrs[ep.target];
    $("step").textContent = `${f.t} / ${cfg.max_steps}`;
    $("target").replaceChildren(swatch(target[0]), document.createTextNode(itemName(target)));
    $("said").replaceChildren(f.sent ? message(f.sent[0]) : document.createTextNode("—"));
    $("heard").replaceChildren(f.received ? message(f.received[1]) : document.createTextNode("·"));
    $("carrying").textContent = f.carrying[1] >= 0 ? itemName(ep.attrs[f.carrying[1]]) : "nothing";
    const outcome = $("outcome");
    if (!last) { outcome.textContent = "in progress"; outcome.className = ""; }
    else if (rec.success) {
      outcome.textContent = `delivered the ${itemName(target)} in ${rec.steps} steps (optimal ${ep.runner_opt})`;
      outcome.className = "ok";
    } else {
      outcome.textContent = rec.picked_attrs && f.carrying[1] < 0
        ? `delivered the ${itemName(rec.picked_attrs)}: wrong item` : "ran out of time";
      outcome.className = "bad";
    }
    const unseen = ep.split === "unseen";
    $("badge").textContent = unseen ? "never-seen combination" : "seen combination";
    $("badge").className = "badge" + (unseen ? " unseen" : "");
    $("scrub").max = fs.length - 1;
    $("scrub").value = state.frame;
    const scout = $("bubble-scout"), runner = $("bubble-runner");
    scout.hidden = !f.sent;
    if (f.sent) scout.replaceChildren(message(f.sent[0]));
    const heard = f.received && f.received[1].some((s) => !silent(s));
    runner.hidden = !heard;
    if (heard) runner.replaceChildren(document.createTextNode("hears "), message(f.received[1]));
  }

  function drawTape() {
    for (const [id, key, agent] of [["tape-said", "sent", 0], ["tape-heard", "received", 1]]) {
      const row = $(id);
      row.replaceChildren();
      frames().forEach((f, t) => {
        if (f[key] === null) return;
        const cell = document.createElement("div");
        cell.className = "pair" + (t === state.frame ? " now" : "");
        cell.title = `step ${t}: ${f[key][agent].map((s) => (silent(s) ? "·" : s)).join(" ")}`;
        cell.appendChild(message(f[key][agent]));
        cell.onclick = () => { stop(); state.frame = t; render(); };
        row.appendChild(cell);
      });
    }
  }

  function drawDictionary() {
    const box = $("dictionary");
    box.replaceChildren();
    const tables = [
      ["word1_given_color", cfg.colors, "Word 1, by target color"], ["word1_given_shape", cfg.shapes, "Word 1, by target shape"],
      ["word2_given_color", cfg.colors, "Word 2, by target color"], ["word2_given_shape", cfg.shapes, "Word 2, by target shape"],
    ];
    for (const [key, names, title] of tables) {
      const table = document.createElement("table");
      table.dataset.property = key.endsWith("color") ? "0" : "1";
      table.dataset.word = key.startsWith("word1") ? "0" : "1";
      table.createCaption().textContent = title;
      const head = table.createTHead().insertRow();
      head.appendChild(document.createElement("th"));
      for (let s = 0; s < cfg.vocab_size; s++) { const th = document.createElement("th"); th.appendChild(wordChip(s)); head.appendChild(th); }
      const body = table.createTBody();
      data.lexicon[key].forEach((probs, v) => {
        const row = body.insertRow();
        row.dataset.value = String(v);
        const th = document.createElement("th");
        th.scope = "row";
        th.textContent = names[v];
        row.appendChild(th);
        probs.forEach((p, symbol) => {
          const td = row.insertCell();
          td.dataset.symbol = String(symbol);
          td.textContent = p >= 0.05 ? String(Math.round(p * 100)) : "";
          td.style.background = `rgba(37, 99, 235, ${p})`;
          if (p > 0.5) td.style.color = "#fff";
        });
      });
      box.appendChild(table);
    }
  }

  function highlightDictionary() {
    const f = frames()[state.frame], attrs = episode().attrs[episode().target];
    $("dictionary").querySelectorAll("table").forEach((table) => {
      const value = attrs[Number(table.dataset.property)];
      const symbol = f.sent ? f.sent[0][Number(table.dataset.word)] : null;
      table.querySelectorAll("tbody tr").forEach((row) => {
        const active = Number(row.dataset.value) === value;
        row.classList.toggle("active-property", active);
        row.querySelectorAll("td").forEach((cell) => {
          cell.classList.toggle("active-word", active && Number(cell.dataset.symbol) === symbol);
        });
      });
    });
  }

  function tone(hz, delay, type) {
    const ac = sound.audio, t0 = ac.currentTime + delay;
    const osc = ac.createOscillator(), gain = ac.createGain();
    osc.type = type;
    osc.frequency.value = hz;
    gain.gain.setValueAtTime(0.0001, t0);
    gain.gain.exponentialRampToValueAtTime(0.22, t0 + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.16);
    osc.connect(gain).connect(ac.destination);
    osc.start(t0);
    osc.stop(t0 + 0.18);
  }
  function playHeard() {
    const key = `${state.variant}|${state.episode}|${state.frame}`;
    if (!sound.on || !sound.audio || key === sound.lastKey) return;
    sound.lastKey = key;
    const f = frames()[state.frame];
    if (!f.received) return;
    const [w1, w2] = f.received[1];
    if (!silent(w1)) tone(PENTATONIC_HZ[w1], 0, "triangle");
    if (!silent(w2)) tone(PENTATONIC_HZ[w2] / 2, 0.13, "sine"); // word 2: an octave lower
  }

  function render() { drawPanel(); drawTape(); highlightDictionary(); playHeard(); }
  function stop() { clearInterval(state.timer); state.timer = null; $("play").textContent = "Play"; }
  function play() {
    if (state.frame >= frames().length - 1) { state.frame = 0; render(); }
    $("play").textContent = "Pause";
    state.timer = setInterval(() => {
      if (state.frame >= frames().length - 1) { stop(); return; }
      state.frame++;
      render();
    }, Number($("speed").value));
  }

  const epSelect = $("episode");
  for (const [split, label] of [["held_out", "Seen combinations"], ["unseen", "Never-seen combinations"]]) {
    const group = document.createElement("optgroup");
    group.label = label;
    Object.entries(data.episodes).filter(([, ep]) => ep.split === split).forEach(([id, ep]) => {
      const ok = data.variants["comm/intact"][id].record.success;
      group.appendChild(new Option(`${itemName(ep.attrs[ep.target])} (talking ${ok ? "succeeds" : "fails"})`, id));
    });
    if (group.children.length) epSelect.appendChild(group);
  }
  state.episode = epSelect.value;
  epSelect.onchange = () => { stop(); state.episode = epSelect.value; state.frame = 0; buildBoard(); render(); };
  const variantBox = $("variants");
  Object.keys(data.variants).forEach((v) => {
    const b = document.createElement("button");
    b.textContent = VARIANT_LABELS[v] || v;
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", String(v === state.variant));
    b.onclick = () => {
      stop();
      state.variant = v;
      state.frame = 0;
      variantBox.querySelectorAll("button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
      updateActors(true);
      render();
    };
    variantBox.appendChild(b);
  });
  $("sound").onclick = () => {
    sound.audio = sound.audio || new (window.AudioContext || window.webkitAudioContext)();
    sound.audio.resume();
    sound.on = !sound.on;
    sound.lastKey = null;
    $("sound").setAttribute("aria-pressed", String(sound.on));
    $("sound").textContent = sound.on ? "Sound on" : "Sound off";
    playHeard();
  };
  $("play").onclick = () => (state.timer ? stop() : play());
  $("speed").onchange = () => { if (state.timer) { stop(); play(); } };
  $("prev").onclick = () => { stop(); state.frame = Math.max(0, state.frame - 1); render(); };
  $("next").onclick = () => { stop(); state.frame = Math.min(frames().length - 1, state.frame + 1); render(); };
  $("scrub").oninput = (e) => { stop(); state.frame = Number(e.target.value); render(); };
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (e.key === " ") { e.preventDefault(); $("play").click(); }
    if (e.key === "ArrowLeft") $("prev").click();
    if (e.key === "ArrowRight") $("next").click();
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { buildBoard(); render(); });

  function resize() {
    const w = sceneBox.clientWidth, h = sceneBox.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(sceneBox);
  function animate() {
    placeCamera();
    updateActors(false);
    renderer.render(scene, camera);
    placeBubble($("bubble-scout"), 0);
    placeBubble($("bubble-runner"), 1);
    requestAnimationFrame(animate);
  }
  resize();
  buildBoard();
  drawDictionary();
  render();
  animate();
})();
