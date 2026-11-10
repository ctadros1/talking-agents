import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { resultView } from "../web/evidence.mjs";
const root = new URL("../", import.meta.url);
const load = async (p) => JSON.parse(await readFile(new URL(p, root)));
test("shared comparisons retain semantic keys across all evidence tabs", async () => {
  const summary = {
    v1: await load("results/summary.json"),
    v2: await load("results/v2/summary.json"),
    vocal: await load("results/vocal/summary.json"),
    support: await load("results/v2/support_reference.json"),
  };
  const views = ["v1", "v2", "vocal"].map((p) => resultView(summary, p));
  for (const key of ["comm-held_out", "no_comm-held_out"])
    for (const v of views) assert.ok(v.bars.some((b) => b.key === key));
  assert.equal(
    views[0].bars.find((b) => b.key === "comm-held_out").value,
    summary.v1.conditions.comm.success.mean,
  );
  assert.equal(
    views[1].bars.find((b) => b.key === "comm-held_out").value,
    summary.v2.conditions.comm.held_out.success.mean,
  );
  assert.ok(views[1].bars.some((b) => b.key === "comm-unseen"));
  assert.deepEqual(
    views[0].cards.map((c) => c.key),
    views[1].cards.map((c) => c.key),
  );
  assert.match(views[1].story, /Claim B failed/);
  assert.match(views[2].story, /exploratory/);
});
