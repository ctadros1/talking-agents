import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  World,
  Policy,
  seededRandom,
  permutation,
  sample,
  Episode,
} from "../web/engine.mjs";
const read = async (p) =>
  JSON.parse(await readFile(new URL(p, import.meta.url)));
test("all exported policies agree with PyTorch on nonzero recurrent states", async () => {
  const fixture = await read("../web/fixtures/parity.json");
  for (const c of fixture.policies) {
    const manifest = await read(`../web/models/${c.condition}.json`);
    const bytes = await readFile(
      new URL(`../web/models/${c.condition}.bin`, import.meta.url),
    );
    const weights = new Float32Array(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    );
    const out = new Policy(manifest.agents[c.agent], weights).forward(
      c.obs,
      c.hidden,
    );
    for (const name of ["move", "message", "hidden", "value"]) {
      const expected = Array.isArray(c.expected[name])
        ? c.expected[name]
        : [c.expected[name]];
      const got =
        Array.isArray(out[name]) || ArrayBuffer.isView(out[name])
          ? out[name]
          : [out[name]];
      assert.equal(got.length, expected.length);
      const error = Math.max(...got.map((x, i) => Math.abs(x - expected[i])));
      assert.ok(
        error < 2e-5,
        `${c.condition}/${c.agent}/${name} max error ${error}`,
      );
    }
  }
});
test("environment observations, transitions and rewards match Python fixtures", async () => {
  const fixture = await read("../web/fixtures/parity.json");
  for (const c of fixture.worlds) {
    const world = new World(c.spec);
    if (c.state) Object.assign(world, structuredClone(c.state));
    for (let a = 0; a < 2; a++)
      assert.deepEqual(Array.from(world.observe(a, c.heard[a])), c.obs[a]);
    const got = world.step(c.moves);
    assert.deepEqual(world.snapshot(), c.expected.snapshot);
    assert.equal(got.done, c.expected.done);
    assert.equal(got.success, c.expected.success);
    assert.ok(Math.abs(got.reward - c.expected.reward) < 1e-6);
  }
});
test("seeded sampling is repeatable, varies across seeds, and remapping is bijective", () => {
  const draws = (seed) => {
    const r = seededRandom(seed);
    return Array.from({ length: 30 }, () => sample([0, 1, 2], r));
  };
  assert.deepEqual(draws(9), draws(9));
  assert.notDeepEqual(draws(9), draws(10));
  assert.deepEqual(
    [...permutation(seededRandom(20))].sort(),
    [0, 1, 2, 3, 4, 5, 6, 7],
  );
});
test("episode reset clears memory, messages are delayed, and silence is a separate intervention", async () => {
  const bank = await read("../web/data/scenarios.json");
  const fake = {
    forward(obs, hidden) {
      return {
        move: [0, -100, -100, -100, -100, -100],
        message: [-100, 0, -100, -100, -100, -100, -100, -100],
        hidden: Float32Array.from(hidden, (x) => x + 1),
      };
    },
  };
  const e = new Episode(bank[0], [fake, fake], { seed: 9, mode: "intact" });
  assert.deepEqual(e.heard, [8, 8]);
  e.step();
  assert.deepEqual(e.heard, [1, 1]);
  assert.equal(e.hidden[0][0], 1);
  e.reset();
  assert.deepEqual(e.heard, [8, 8]);
  assert.equal(e.hidden[0][0], 0);
  const silent = new Episode(bank[0], [fake, fake], {
    seed: 9,
    mode: "silenced",
  });
  silent.step();
  assert.deepEqual(silent.heard, [8, 8]);
});
