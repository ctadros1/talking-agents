import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";
import { positionCamera } from "../web/scene3d.mjs";
test("camera zoom/orbit/reset update the projection matrix before positioning labels", async () => {
  const context = { console: { warn() {} } };
  vm.runInNewContext(
    await readFile(
      new URL("../viewer/vendor/three.min.js", import.meta.url),
      "utf8",
    ),
    context,
  );
  const T = context.THREE,
    camera = new T.PerspectiveCamera(40, 4 / 3, 0.1, 100),
    initial = { theta: -0.55, phi: 0.8, radius: 21.5 };
  const project = () => new T.Vector3(-4, 1, -2).project(camera).toArray();
  positionCamera(camera, initial);
  const start = project();
  positionCamera(camera, { ...initial, radius: initial.radius * 0.85 });
  assert.notDeepEqual(project(), start);
  positionCamera(camera, { ...initial, theta: initial.theta + 0.4 });
  assert.notDeepEqual(project(), start);
  positionCamera(camera, initial);
  assert.deepEqual(project(), start);
});
