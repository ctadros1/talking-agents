// Live Three.js view of the browser environment; no replay data is read here.
import { icon } from "./icons.mjs";
let threePromise;
function loadThree() {
  if (!threePromise)
    threePromise = new Promise((resolve, reject) => {
      if (window.THREE) {
        resolve(window.THREE);
        return;
      }
      const script = document.createElement("script");
      script.src = new URL(
        "./replays/vendor/three.min.js",
        import.meta.url,
      ).href;
      script.onload = () => resolve(window.THREE);
      script.onerror = () => {
        threePromise = null;
        reject(new Error("The 3D renderer could not load."));
      };
      document.head.append(script);
    });
  return threePromise;
}
export function positionCamera(camera, { theta, phi, radius }) {
  camera.position.set(
    radius * Math.sin(phi) * Math.sin(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.cos(theta),
  );
  camera.lookAt(0, 0, 0);
  // DOM speech bubbles project before the next render, so update the matrix now.
  camera.updateMatrixWorld();
}
export async function createLiveScene(container) {
  return new LiveScene(container, await loadThree());
}
class LiveScene {
  constructor(container, T) {
    this.T = T;
    this.container = container;
    this.active = false;
    this.visible = true;
    this.frame = null;
    this.world = null;
    this.disposed = false;
    this.renderer = new T.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
    this.renderer.domElement.setAttribute(
      "aria-label",
      "Live 3D grid world. Drag to rotate; use plus and minus to zoom.",
    );
    this.renderer.domElement.tabIndex = 0;
    this.renderer.domElement.setAttribute("role", "img");
    container.prepend(this.renderer.domElement);
    this.scene = new T.Scene();
    this.scene.background = new T.Color("#fafaf4");
    this.camera = new T.PerspectiveCamera(40, 1, 0.1, 100);
    this.board = new T.Group();
    this.scene.add(this.board);
    this.scene.add(new T.HemisphereLight(0xffffff, 0x76836d, 2.3));
    const sun = new T.DirectionalLight(0xffffff, 2.6);
    sun.position.set(-6, 14, 8);
    this.scene.add(sun);
    this.defaults = { theta: -0.55, phi: 0.8, radius: 21.5 };
    this.orbit = { ...this.defaults };
    container.insertAdjacentHTML(
      "beforeend",
      `<div class="scene-bubble scout-bubble" aria-hidden="true"></div><div class="scene-bubble runner-bubble" aria-hidden="true"></div><div class="camera-controls"><button type="button" class="camera-reset" aria-label="Reset 3D camera">${icon("rotate-3d")}</button><button type="button" class="camera-zoom-in" aria-label="Zoom in">${icon("zoom-in")}</button><button type="button" class="camera-zoom-out" aria-label="Zoom out">${icon("zoom-out")}</button></div>`,
    );
    container.querySelector(".camera-reset").onclick = () => {
      Object.assign(this.orbit, this.defaults);
      this.schedule();
    };
    container.querySelector(".camera-zoom-in").onclick = () => this.zoom(0.85);
    container.querySelector(".camera-zoom-out").onclick = () => this.zoom(1.15);
    const canvas = this.renderer.domElement;
    let drag = null;
    canvas.addEventListener("pointerdown", (e) => {
      drag = { x: e.clientX, y: e.clientY };
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!drag) return;
      this.orbit.theta -= (e.clientX - drag.x) * 0.008;
      this.orbit.phi = Math.min(
        1.4,
        Math.max(0.25, this.orbit.phi - (e.clientY - drag.y) * 0.008),
      );
      drag = { x: e.clientX, y: e.clientY };
      this.schedule();
    });
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"])
      canvas.addEventListener(name, () => {
        drag = null;
      });
    canvas.addEventListener(
      "wheel",
      (e) => {
        if (!canvas.matches(":focus")) return;
        e.preventDefault();
        this.zoom(Math.exp(e.deltaY * 0.001));
      },
      { passive: false },
    );
    canvas.addEventListener("keydown", (e) => {
      if (
        ![
          "ArrowLeft",
          "ArrowRight",
          "ArrowUp",
          "ArrowDown",
          "+",
          "=",
          "-",
          "0",
        ].includes(e.key)
      )
        return;
      e.preventDefault();
      if (e.key === "0") Object.assign(this.orbit, this.defaults);
      if (e.key === "ArrowLeft") this.orbit.theta -= 0.15;
      if (e.key === "ArrowRight") this.orbit.theta += 0.15;
      if (e.key === "ArrowUp")
        this.orbit.phi = Math.max(0.25, this.orbit.phi - 0.1);
      if (e.key === "ArrowDown")
        this.orbit.phi = Math.min(1.4, this.orbit.phi + 0.1);
      if (e.key === "+" || e.key === "=") this.zoom(0.85);
      if (e.key === "-") this.zoom(1.15);
      this.schedule();
    });
    canvas.addEventListener("webglcontextlost", (e) => {
      e.preventDefault();
      this.active = false;
      container.dispatchEvent(new Event("renderer-lost"));
    });
    this.resize = new ResizeObserver(() => {
      if (!container.clientWidth) return;
      this.renderer.setSize(
        container.clientWidth,
        container.clientHeight,
        false,
      );
      this.camera.aspect =
        container.clientWidth / Math.max(1, container.clientHeight);
      this.camera.updateProjectionMatrix();
      this.schedule();
    });
    this.resize.observe(container);
    this.intersection = new IntersectionObserver((entries) => {
      this.visible = entries[0].isIntersecting;
      if (this.visible) this.schedule();
    });
    this.intersection.observe(container);
  }
  zoom(factor) {
    this.orbit.radius = Math.min(32, Math.max(12, this.orbit.radius * factor));
    this.schedule();
  }
  at(r, c, y = 0) {
    return new this.T.Vector3(c - 5.5, y, r - 5.5);
  }
  material(color, roughness = 0.7) {
    return new this.T.MeshStandardMaterial({ color, roughness });
  }
  item(attrs) {
    const T = this.T,
      [color, shape] = attrs,
      geometries = [
        () => new T.BoxGeometry(0.5, 0.5, 0.5),
        () => new T.SphereGeometry(0.29, 20, 16),
        () => new T.ConeGeometry(0.3, 0.6, 24),
        () => new T.CylinderGeometry(0.25, 0.25, 0.55, 24),
        () => new T.TorusGeometry(0.22, 0.09, 12, 28),
      ];
    const model = new T.Group(),
      mesh = new T.Mesh(
        geometries[shape](),
        this.material(
          ["#dc5b4b", "#e6bb46", "#66a16d", "#4f88be", "#9572bf"][color],
          0.45,
        ),
      );
    mesh.position.y = 0.34;
    model.add(mesh);
    return model;
  }
  creature(color) {
    const T = this.T,
      g = new T.Group(),
      body = new T.Mesh(
        new T.SphereGeometry(0.33, 22, 16),
        this.material(color, 0.5),
      );
    body.position.y = 0.4;
    body.scale.y = 1.2;
    g.add(body);
    for (const side of [-1, 1]) {
      const eye = new T.Mesh(
        new T.SphereGeometry(0.085, 12, 10),
        this.material("#ffffff"),
      );
      eye.position.set(side * 0.13, 0.54, 0.28);
      const pupil = new T.Mesh(
        new T.SphereGeometry(0.04, 10, 8),
        this.material("#26342b"),
      );
      pupil.position.set(side * 0.13, 0.54, 0.35);
      g.add(eye, pupil);
    }
    return g;
  }
  clearBoard() {
    const geometries = new Set(),
      materials = new Set();
    this.board.traverse((o) => {
      if (o.geometry) geometries.add(o.geometry);
      if (o.material) materials.add(o.material);
    });
    this.board.clear();
    geometries.forEach((g) => g.dispose());
    materials.forEach((m) => m.dispose());
  }
  load(world) {
    this.clearBoard();
    this.world = world;
    const T = this.T;
    const tile = new T.BoxGeometry(0.965, 0.08, 0.965),
      block = new T.BoxGeometry(1, 0.78, 1),
      control = this.material("#ece7f3"),
      warehouse = this.material("#edf1e7"),
      wall = this.material("#a5b19f");
    for (let r = 0; r < 12; r++)
      for (let c = 0; c < 12; c++) {
        const blocked = world.walls[r][c],
          mesh = new T.Mesh(
            blocked ? block : tile,
            blocked ? wall : c < 4 ? control : warehouse,
          );
        mesh.position.copy(this.at(r, c, blocked ? 0.35 : -0.04));
        this.board.add(mesh);
      }
    const depot = new T.Mesh(
      new T.CylinderGeometry(0.43, 0.43, 0.06, 32),
      this.material("#477d5b"),
    );
    depot.position.copy(this.at(...world.depot, 0.04));
    this.board.add(depot);
    const ring = new T.Mesh(
      new T.TorusGeometry(0.3, 0.035, 8, 24),
      this.material("#f7f7eb"),
    );
    ring.rotation.x = Math.PI / 2;
    ring.position.copy(this.at(...world.depot, 0.08));
    this.board.add(ring);
    this.items = world.spec.item_attrs.map((attrs, k) => {
      const model = this.item(attrs);
      model.position.copy(this.at(...world.item_pos[k]));
      this.board.add(model);
      return model;
    });
    this.agents = ["#a58cc9", "#dc934d"].map((color) => {
      const a = this.creature(color);
      this.board.add(a);
      return a;
    });
    this.headings = [Math.PI / 2, Math.PI];
    const viewGeometry = new T.PlaneGeometry(1, 1),
      viewMaterial = new T.MeshBasicMaterial({
        color: "#6f9660",
        opacity: 0.11,
        transparent: true,
        depthWrite: false,
        side: T.DoubleSide,
      });
    this.view = new T.Mesh(viewGeometry, viewMaterial);
    this.view.rotation.x = -Math.PI / 2;
    this.view.position.y = 0.012;
    this.board.add(this.view);
    this.update(world, null, true);
  }
  update(world, last, snap = false) {
    this.world = world;
    this.last = last;
    if (snap) this.moveActors(1);
    this.schedule();
  }
  moveActors(alpha) {
    const w = this.world;
    if (!w) return;
    w.pos.forEach(([r, c], a) => {
      const actor = this.agents[a],
        goal = this.at(r, c),
        dx = goal.x - actor.position.x,
        dz = goal.z - actor.position.z;
      if (Math.abs(dx) + Math.abs(dz) > 0.05)
        this.headings[a] = Math.atan2(dx, dz);
      actor.position.lerp(goal, alpha);
      const difference = Math.atan2(
        Math.sin(this.headings[a] - actor.rotation.y),
        Math.cos(this.headings[a] - actor.rotation.y),
      );
      actor.rotation.y += difference * alpha;
    });
    this.items.forEach((item, k) => {
      const carrier = w.carrying.indexOf(k);
      item.visible = carrier >= 0 || w.item_present[k];
      if (item.visible)
        item.position.lerp(
          carrier >= 0
            ? this.at(...w.pos[carrier], 0.83)
            : this.at(...w.item_pos[k]),
          alpha,
        );
    });
    const [r, c] = w.pos[1],
      left = Math.max(0, c - 3),
      right = Math.min(12, c + 4),
      top = Math.max(0, r - 3),
      bottom = Math.min(12, r + 4);
    this.view.scale.set(right - left, bottom - top, 1);
    this.view.position.copy(
      this.at((top + bottom - 1) / 2, (left + right - 1) / 2, 0.012),
    );
  }
  placeBubble(selector, agent, text) {
    const el = this.container.querySelector(selector),
      point = this.agents[agent].position.clone();
    point.y += 1.3;
    point.project(this.camera);
    el.style.left = Math.max(13, Math.min(87, ((point.x + 1) / 2) * 100)) + "%";
    el.style.top = ((1 - point.y) / 2) * 100 + "%";
    el.textContent = text;
    el.hidden = !text || point.z > 1;
  }
  setActive(active) {
    this.active = active;
    if (active) this.schedule();
    else {
      cancelAnimationFrame(this.frame);
      this.frame = null;
    }
  }
  schedule() {
    if (this.active && this.visible && !this.frame && !this.disposed)
      this.frame = requestAnimationFrame(() => this.render());
  }
  render() {
    this.frame = null;
    if (!this.active || !this.world || document.hidden) return;
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.moveActors(reduced ? 1 : 0.25);
    positionCamera(this.camera, this.orbit);
    this.placeBubble(
      ".scout-bubble",
      0,
      this.last ? `S: ${this.last.sent[0].join(" · ")}` : "S · scout",
    );
    this.placeBubble(
      ".runner-bubble",
      1,
      this.last
        ? `R hears: ${this.last.heard[1].map((x) => (x === 8 ? "–" : x)).join(" · ")}`
        : "R · runner",
    );
    this.renderer.render(this.scene, this.camera);
    if (
      !reduced &&
      this.agents.some(
        (a, i) => a.position.distanceTo(this.at(...this.world.pos[i])) > 0.005,
      )
    )
      this.schedule();
  }
}
