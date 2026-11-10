// Browser-only inference adapter. The canonical experiment remains src/emcomm/.
export function seededRandom(seed) {
  let a = Number(seed) >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function permutation(random) {
  const p = Array.from({ length: 8 }, (_, i) => i);
  for (let i = 7; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [p[i], p[j]] = [p[j], p[i]];
  }
  return p;
}
export function sample(logits, random) {
  const max = Math.max(...logits),
    p = Array.from(logits, (x) => Math.exp(x - max));
  let draw = random() * p.reduce((a, b) => a + b, 0);
  for (let i = 0; i < p.length; i++) {
    draw -= p[i];
    if (draw < 0) return i;
  }
  return p.length - 1;
}
function linear(input, tensor, bias, weights) {
  const [rows, cols] = tensor.shape,
    out = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let v = weights[bias.offset + r],
      start = tensor.offset + r * cols;
    for (let c = 0; c < cols; c++) v += weights[start + c] * input[c];
    out[r] = v;
  }
  return out;
}
export class Policy {
  constructor(tensors, weights) {
    this.t = tensors;
    this.w = weights;
  }
  layer(name, x) {
    return linear(x, this.t[`${name}.weight`], this.t[`${name}.bias`], this.w);
  }
  forward(obs, hidden) {
    let x = this.layer("encoder.0", obs).map((v) => Math.max(0, v));
    x = this.layer("encoder.2", x).map((v) => Math.max(0, v));
    const i = linear(x, this.t["gru.weight_ih"], this.t["gru.bias_ih"], this.w);
    const g = linear(
      hidden,
      this.t["gru.weight_hh"],
      this.t["gru.bias_hh"],
      this.w,
    );
    const h = new Float32Array(hidden.length),
      n = h.length;
    for (let j = 0; j < n; j++) {
      const r = 1 / (1 + Math.exp(-i[j] - g[j])),
        z = 1 / (1 + Math.exp(-i[n + j] - g[n + j]));
      h[j] =
        (1 - z) * Math.tanh(i[2 * n + j] + r * g[2 * n + j]) + z * hidden[j];
    }
    return {
      move: this.layer("move_head", h),
      message: this.layer("msg_head", h),
      value: this.layer("value_head", h)[0],
      hidden: h,
    };
  }
}
export class World {
  constructor(spec) {
    this.spec = structuredClone(spec);
    this.walls = spec.walls;
    this.item_pos = spec.item_pos;
    this.depot = spec.depot;
    this.target = spec.target;
    this.pos = structuredClone(spec.agent_pos);
    this.carrying = [-1, -1];
    this.item_present = [true, true, true, true];
    this.t = 0;
    this.done = false;
  }
  observe(a, heard) {
    const out = new Float32Array(315),
      [r, c] = this.pos[a];
    for (let dr = -3; dr <= 3; dr++)
      for (let dc = -3; dc <= 3; dc++) {
        const rr = r + dr,
          cc = c + dc,
          offset = (dr + 3) * 7 + dc + 3;
        if (rr < 0 || rr >= 12 || cc < 0 || cc >= 12) {
          out[offset] = 1;
          continue;
        }
        out[offset] = Number(this.walls[rr][cc]);
        for (let j = 0; j < 4; j++)
          if (
            this.item_present[j] &&
            this.item_pos[j][0] === rr &&
            this.item_pos[j][1] === cc
          )
            out[(j + 1) * 49 + offset] = 1;
        if (this.depot[0] === rr && this.depot[1] === cc)
          out[5 * 49 + offset] = 1;
      }
    out[294] = r / 11;
    out[295] = c / 11;
    out[296 + this.carrying[a] + 1] = 1;
    if (a === 0) out[301 + this.target] = 1;
    out[305 + heard] = 1;
    out[314] = this.t / 32;
    return out;
  }
  step(moves) {
    if (this.done)
      throw new Error("Reset the episode before stepping after completion.");
    const delta = [
      [0, 0],
      [-1, 0],
      [1, 0],
      [0, -1],
      [0, 1],
      [0, 0],
    ];
    let reward = -0.01,
      success = false,
      done = false;
    for (let a = 0; a < 2; a++) {
      const [dr, dc] = delta[moves[a]],
        [r, c] = this.pos[a],
        rr = r + dr,
        cc = c + dc;
      if (rr >= 0 && rr < 12 && cc >= 0 && cc < 12 && !this.walls[rr][cc])
        this.pos[a] = [rr, cc];
      const same = (p) => p[0] === this.pos[a][0] && p[1] === this.pos[a][1];
      let picked = false;
      if (moves[a] === 5 && this.carrying[a] < 0) {
        const j = this.item_pos.findIndex(
          (p, j) => same(p) && this.item_present[j],
        );
        if (j >= 0) {
          this.carrying[a] = j;
          this.item_present[j] = false;
          reward += 0.1;
          picked = true;
        }
      }
      if (
        moves[a] === 5 &&
        !picked &&
        this.carrying[a] >= 0 &&
        same(this.depot) &&
        !done
      ) {
        const correct = this.carrying[a] === this.target;
        reward += correct ? 1 : 0;
        success = success || correct;
        done = true;
        this.carrying[a] = -1;
      }
    }
    this.t++;
    done = done || this.t >= 32;
    this.done = done;
    return { reward, done, success };
  }
  snapshot() {
    return {
      t: this.t,
      pos: structuredClone(this.pos),
      carrying: [...this.carrying],
      item_present: [...this.item_present],
    };
  }
}
export class Episode {
  constructor(
    spec,
    policies,
    { seed = 19, mode = "intact", hidden = 128 } = {},
  ) {
    this.spec = spec;
    this.policies = policies;
    this.seed = seed;
    this.mode = mode;
    this.width = hidden;
    this.reset();
  }
  reset() {
    this.random = seededRandom(this.seed);
    this.mapping = permutation(seededRandom(this.seed ^ 0x51eeda));
    this.world = new World(this.spec);
    this.hidden = [new Float32Array(this.width), new Float32Array(this.width)];
    this.heard = [8, 8];
    this.lastSent = [8, 8];
    this.tape = [];
    this.return = 0;
    this.outcome = null;
  }
  step() {
    if (this.world.done) return this.outcome;
    const oldHeard = [...this.heard],
      moves = [],
      sent = [];
    for (let a = 0; a < 2; a++) {
      const out = this.policies[a].forward(
        this.world.observe(a, this.heard[a]),
        this.hidden[a],
      );
      this.hidden[a] = out.hidden;
      moves.push(sample(out.move, this.random));
      sent.push(sample(out.message, this.random));
    }
    this.outcome = this.world.step(moves);
    this.return += this.outcome.reward;
    this.lastSent = sent;
    this.heard = sent.map((_, a) =>
      this.mode === "silenced" || this.mode === "no_comm"
        ? 8
        : this.mode === "permuted"
          ? this.mapping[sent[1 - a]]
          : sent[1 - a],
    );
    this.tape.push({
      t: this.world.t,
      sent: [...sent],
      heard: oldHeard,
      moves,
    });
    return this.outcome;
  }
}
