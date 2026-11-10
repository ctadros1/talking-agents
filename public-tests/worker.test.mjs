import test from "node:test";
import assert from "node:assert/strict";
import worker from "../cloudflare/worker.mjs";
const env = {
  ASSETS: {
    fetch: async (r) =>
      new Response(new URL(r.url).pathname, {
        status: new URL(r.url).pathname === "/missing" ? 404 : 200,
      }),
  },
};
test("public path redirects then strips only its own prefix, preserving query", async () => {
  let r = await worker.fetch(
    new Request("https://ctad.dev/talking-agents?q=1"),
    env,
  );
  assert.equal(r.status, 308);
  assert.equal(
    r.headers.get("location"),
    "https://ctad.dev/talking-agents/?q=1",
  );
  r = await worker.fetch(
    new Request("https://ctad.dev/talking-agents/models/comm.bin"),
    env,
  );
  assert.equal(await r.text(), "/models/comm.bin");
  assert.equal(r.status, 200);
  const home = await worker.fetch(
    new Request("https://ctad.dev/talking-agents/"),
    env,
  );
  assert.equal(await home.text(), "/index.html");
  assert.equal(r.headers.get("X-Content-Type-Options"), "nosniff");
});
test("outside paths, unsupported methods and missing files never become the app", async () => {
  for (const path of ["/", "/tt-study", "/talking-agents-extra"])
    assert.equal(
      (await worker.fetch(new Request("https://ctad.dev" + path), env)).status,
      404,
    );
  assert.equal(
    (
      await worker.fetch(
        new Request("https://ctad.dev/talking-agents/", { method: "POST" }),
        env,
      )
    ).status,
    405,
  );
  assert.equal(
    (
      await worker.fetch(
        new Request("https://ctad.dev/talking-agents/missing"),
        env,
      )
    ).status,
    404,
  );
});
