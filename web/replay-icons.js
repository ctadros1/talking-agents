// Public replay UI enhancement. Archived scientific trajectories stay untouched.
(() => {
  const sprite = new URL("icons.svg", document.currentScript.src).href;
  const svg = (name) =>
    `<svg class="icon" aria-hidden="true" focusable="false" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><use href="${sprite}#${name}"></use></svg>`;
  function buttons() {
    for (const [id, name, label] of [
      ["prev", "step-back", ""],
      ["next", "step-forward", ""],
      ["reset-camera", "rotate-3d", "Reset view"],
    ]) {
      const el = document.getElementById(id);
      if (el && !el.querySelector("svg")) el.innerHTML = svg(name) + label;
    }
    for (const id of ["play", "sound"]) {
      const el = document.getElementById(id);
      if (!el || el.querySelector("svg")) continue;
      const label = el.textContent.trim();
      el.innerHTML =
        svg(
          id === "play"
            ? label === "Pause"
              ? "pause"
              : "play"
            : el.getAttribute("aria-pressed") === "true"
              ? "volume-2"
              : "volume-x",
        ) + `<span>${label}</span>`;
    }
    for (const cell of document.querySelectorAll(".cell"))
      if (cell.textContent === "♪") cell.innerHTML = svg("music-2");
  }
  document.querySelectorAll("a:not(.back-link)").forEach((a) => {
    if (!a.querySelector("svg"))
      a.insertAdjacentHTML("beforeend", svg("arrow-right"));
  });
  buttons();
  const observer = new MutationObserver((records) => {
    if (
      records.some(
        (r) =>
          r.target.nodeType === 1 && r.target.closest("#play,#sound,.cells"),
      )
    )
      buttons();
  });
  observer.observe(document.body, { childList: true, subtree: true });
})();
