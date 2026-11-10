/* Local-only canvas presentation and microphone recording. No network requests. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const canvas = $("video"), ctx = canvas.getContext("2d"), slides = window.PRESENTATION;
  let index = 0, slideStarted = performance.now(), recorder = null, microphone = null;
  let started = 0, chunks = [], timer = null, capture = null, testRecording = false;
  const pictures = new Map();
  for (const slide of slides) if (slide.image) {
    const img = new Image(); img.src = slide.imageData || slide.image; pictures.set(slide.image, img);
  }
  const replay = window.REPLAYS;
  const episode = replay && Object.keys(replay.episodes).find(key =>
    replay.variants["comm/intact"][key].record.success && !replay.variants["comm/permuted"][key].record.success
  );

  function text(value, x, y, size = 34, color = "#edf3fa", weight = "400") {
    ctx.font = `${weight} ${size}px Arial, sans-serif`; ctx.fillStyle = color; ctx.fillText(value, x, y);
  }
  function wrap(value, x, y, width, size = 34, color = "#edf3fa", gap = 48) {
    const words = value.split(/\s+/); let line = "";
    ctx.font = `${size}px Arial, sans-serif`;
    for (const word of words) {
      const next = `${line} ${word}`.trim();
      if (ctx.measureText(next).width > width && line) { text(line, x, y, size, color); y += gap; line = word; }
      else line = next;
    }
    if (line) { text(line, x, y, size, color); y += gap; }
    return y;
  }
  function circle(x, y, radius, color) { ctx.beginPath(); ctx.arc(x, y, radius, 0, 2 * Math.PI); ctx.fillStyle = color; ctx.fill(); }
  function drawMap(variant, x, time, title) {
    if (!episode) { text("No paired replay is available.", x, 300); return; }
    const spec = replay.episodes[episode], run = replay.variants[variant][episode];
    const n = spec.walls.length, cell = 48, boardX = x + 100, boardY = 280;
    const frameIndex = Math.min(run.steps.length - 1, Math.floor(time / 350));
    const frame = run.steps[frameIndex];
    text(title, x + 40, 220, 34, "#ffcc55", "700");
    for (let row = 0; row < n; row++) for (let col = 0; col < n; col++) {
      ctx.fillStyle = spec.walls[row][col] ? "#526071" : col < 4 ? "#27233c" : "#1b2837";
      ctx.fillRect(boardX + col * cell, boardY + row * cell, cell - 1, cell - 1);
    }
    const point = pos => [boardX + (pos[1] + .5) * cell, boardY + (pos[0] + .5) * cell];
    const [dx, dy] = point(spec.depot); ctx.strokeStyle = "#50dfa0"; ctx.lineWidth = 4; ctx.strokeRect(dx - 17, dy - 17, 34, 34);
    const colors = ["#ec6475", "#ffcc55", "#53d2dc", "#c48cff"];
    spec.items.forEach((pos, item) => {
      if (frame.item_present && !frame.item_present[item]) return;
      const [px, py] = point(pos); ctx.fillStyle = colors[item]; ctx.fillRect(px - 16, py - 16, 32, 32);
      text(String.fromCharCode(65 + item), px - 9, py + 9, 25, "#10151c", "700");
    });
    frame.pos.forEach((pos, agent) => { const [px, py] = point(pos); circle(px, py, 18, agent ? "#ff963e" : "#aa82ff"); text(agent ? "R" : "S", px - 9, py + 9, 24, "#111", "700"); });
    const target = String.fromCharCode(65 + spec.target);
    text(`Target ${target} · Step ${frame.t} / ${replay.config.max_steps}`, x + 40, 920, 28, "#bdcad8");
    const said = frame.sent ? frame.sent[0] : "—";
    const heard = frame.received ? (frame.received[1] === replay.config.vocab_size ? "silence" : frame.received[1]) : "—";
    text(`Scout says: ${said}    Runner hears: ${heard}`, x + 40, 964, 28);
    if (frameIndex === run.steps.length - 1) text(run.record.success ? "CORRECT DELIVERY" : "FAILURE", x + 40, 1010, 30, run.record.success ? "#50dfa0" : "#ff7c87", "700");
  }
  function draw(now) {
    ctx.fillStyle = "#10151c"; ctx.fillRect(0, 0, 1920, 1080);
    const slide = slides[index];
    text(slide.eyebrow.toUpperCase(), 88, 78, 24, "#ffcc55", "700");
    text(slide.title, 88, 148, 52, "#fff", "700");
    if (slide.kind === "replay") {
      const time = (now - slideStarted) % 14500;
      drawMap("comm/intact", 30, time, "INTACT MESSAGES"); drawMap("comm/permuted", 990, time, "PERMUTED MESSAGES");
    } else {
      const image = pictures.get(slide.image); let y = 280;
      for (const bullet of slide.bullets) {
        circle(100, y - 10, 6, "#ffcc55"); y = wrap(bullet, 130, y, image ? 770 : 1650, 37) + 42;
      }
      if (image && image.complete && image.naturalWidth) {
        const x = 1000, boxY = 240, boxW = 840, boxH = 650;
        ctx.fillStyle = "#fff"; ctx.fillRect(x, boxY, boxW, boxH);
        const scale = Math.min(boxW / image.naturalWidth, boxH / image.naturalHeight);
        const w = image.naturalWidth * scale, h = image.naturalHeight * scale;
        ctx.drawImage(image, x + (boxW - w) / 2, boxY + (boxH - h) / 2, w, h);
      }
      if (slide.kind === "method") {
        ctx.fillStyle = "#1c2b3c"; ctx.fillRect(130, 620, 1620, 190);
        text("PPO loss  −  λ × cue information in the message", 180, 700, 44, "#ffcc55", "700");
        text("A task-specific adaptation of positive signalling (Eccles et al., 2019)", 180, 766, 30, "#bdcad8");
      }
      if (slide.kind === "title") {
        circle(420, 760, 75, "#aa82ff"); circle(1450, 760, 75, "#ff963e");
        text("SCOUT", 340, 890, 32, "#bdcad8", "700"); text("RUNNER", 1360, 890, 32, "#bdcad8", "700");
        text("learned messages →", 750, 775, 42, "#ffcc55", "700");
      }
      text("Christian Tadros  |  CAP6938  |  Emergent Communication in a Grid World", 88, 1030, 22, "#9baec3");
    }
    text(`${index + 1} / ${slides.length}`, 1760, 78, 25, "#9baec3");
    requestAnimationFrame(draw);
  }
  function change(delta) {
    index = Math.max(0, Math.min(slides.length - 1, index + delta)); slideStarted = performance.now();
    $("notes").textContent = slides[index].notes; $("counter").textContent = `${index + 1} / ${slides.length}`;
    $("previous").disabled = index === 0; $("next").disabled = index === slides.length - 1;
  }
  function cleanup() {
    if (microphone) microphone.getTracks().forEach(track => track.stop());
    if (capture) capture.getTracks().forEach(track => track.stop());
    microphone = null; capture = null; clearInterval(timer);
    $("start").disabled = false; $("stop").disabled = true;
  }
  async function start(silentTest = false) {
    if (recorder && recorder.state !== "inactive") return;
    testRecording = silentTest; chunks = [];
    try {
      if (!window.MediaRecorder || !canvas.captureStream) throw new Error("Please open this page in Google Chrome to record.");
      if (!silentTest) { index = 0; change(0); }
      if (!silentTest) microphone = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      capture = canvas.captureStream(30);
      const stream = new MediaStream([...capture.getVideoTracks(), ...(microphone ? microphone.getAudioTracks() : [])]);
      const types = ["video/mp4;codecs=avc1.42001E,mp4a.40.2", "video/mp4", "video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"];
      const mimeType = types.find(type => MediaRecorder.isTypeSupported(type));
      recorder = new MediaRecorder(stream, mimeType ? { mimeType, videoBitsPerSecond: 4500000 } : {});
      recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
      recorder.onerror = event => { $("status").textContent = `Recording stopped: ${event.error?.message || "please try again"}`; cleanup(); };
      recorder.onstop = () => {
        const type = recorder.mimeType, ext = type.includes("mp4") ? "mp4" : "webm";
        const blob = new Blob(chunks, { type }); const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = `${testRecording ? "CAP6938-SILENT-TEST" : "Christian-Tadros-CAP6938-Narration"}.${ext}`;
        link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 60000); cleanup();
        $("status").textContent = testRecording ? "Silent test saved. It is not a narrated submission." : "Recording saved. Play it back to check your voice, then finalize the video using the instructions.";
      };
      recorder.start(1000); started = Date.now();
      $("start").disabled = true; $("stop").disabled = false;
      $("status").textContent = silentTest ? "Recording a three-second silent export test." : "Recording your voice and the slides. Advance at your own pace; Stop and save when finished.";
      timer = setInterval(() => { const s = Math.floor((Date.now() - started) / 1000); $("clock").textContent = `${Math.floor(s / 60).toString().padStart(2, "0")}:${(s % 60).toString().padStart(2, "0")}`; }, 250);
      if (silentTest) setTimeout(() => { if (recorder.state === "recording") recorder.stop(); }, 3000);
    } catch (error) { cleanup(); $("status").textContent = `Could not start: ${error.message}. No recording was saved.`; }
  }
  $("previous").onclick = () => change(-1); $("next").onclick = () => change(1);
  $("start").onclick = () => start(false); $("stop").onclick = () => { if (recorder && recorder.state === "recording") recorder.stop(); };
  $("test").onclick = () => start(true);
  if (new URLSearchParams(location.search).has("test")) $("test").style.display = "inline-block";
  document.addEventListener("keydown", event => { if (event.key === "ArrowRight") change(1); if (event.key === "ArrowLeft") change(-1); });
  window.addEventListener("beforeunload", event => { if (recorder && recorder.state === "recording") { event.preventDefault(); event.returnValue = ""; } });
  change(0); requestAnimationFrame(draw);
})();
