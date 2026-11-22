import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { joinRoom } from "../lib/channel.js";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function until(predicate, timeoutMs = 3000, what = "condition") {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (predicate()) return;
    await sleep(5);
  }
  assert.fail(`timed out waiting for ${what}`);
}

/** A participant that records what joinRoom reports. */
function recorder() {
  const events = [];
  const states = [];
  return { events, states, onEvent: (event) => events.push(event), onState: (state) => states.push(state) };
}

const uniqueRoom = (prefix) => `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
const message = (text, from = "patient", extra = {}) => ({ type: "message", from, via: "sign", text, signs: [], ...extra });

// --- local mode: BroadcastChannel ----------------------------------------------

test("local: without a relay it uses BroadcastChannel and reports connected", async (t) => {
  const a = recorder();
  const channel = await joinRoom(uniqueRoom("local"), { ...a, base: null });
  t.after(() => channel.close());
  assert.equal(channel.transport, "local");
  assert.deepEqual(a.states, ["connected"]);
});

test("local: every participant, the sender included, gets each event exactly once", async (t) => {
  const room = uniqueRoom("local");
  const a = recorder();
  const b = recorder();
  const outsider = recorder();
  const ca = await joinRoom(room, { ...a, base: null });
  const cb = await joinRoom(room, { ...b, base: null });
  const co = await joinRoom(uniqueRoom("other"), { ...outsider, base: null });
  t.after(() => [ca, cb, co].forEach((c) => c.close()));

  // Without the relay there is no shared order for events sent at the same
  // moment, so the second is sent once the first has arrived everywhere.
  await ca.send(message("Je suis malade.", "patient", { signs: ["ENFERMO"] }));
  await until(() => a.events.length >= 1 && b.events.length >= 1, 2000, "the message on both sides");
  await cb.send({ type: "status", from: "doctor", state: "typing" });
  await until(() => a.events.length >= 2 && b.events.length >= 2, 2000, "the status on both sides");
  await sleep(50); // room for a duplicate to show up

  for (const side of [a, b]) {
    assert.equal(side.events.length, 2);
    const [sent, status] = side.events;
    assert.equal(sent.type, "message");
    assert.equal(sent.text, "Je suis malade.");
    assert.deepEqual(sent.signs, ["ENFERMO"]);
    assert.ok(typeof sent.id === "string" && sent.id.length > 0);
    assert.ok(Math.abs(sent.ts - Date.now()) < 5000);
    assert.equal(status.state, "typing");
  }
  assert.equal(a.events[0].id, b.events[0].id);
  assert.equal(outsider.events.length, 0);
});

test("local: the sender gets a copy of its event, with the given id kept", async (t) => {
  const a = recorder();
  const channel = await joinRoom(uniqueRoom("local"), { ...a, base: null });
  t.after(() => channel.close());
  const event = message("x", "doctor", { id: "given-id" });
  await channel.send(event);
  event.text = "changed after sending";
  assert.equal(a.events.length, 1);
  assert.equal(a.events[0].id, "given-id");
  assert.equal(a.events[0].text, "x");
});

test("local: a bad event is refused, the same way as by the relay", async (t) => {
  const a = recorder();
  const channel = await joinRoom(uniqueRoom("local"), { ...a, base: null });
  t.after(() => channel.close());
  await assert.rejects(channel.send({ type: "shout", from: "patient" }), /type/);
  await assert.rejects(channel.send({ type: "message", text: "x" }), /from/);
  await assert.rejects(channel.send({ type: "status", from: "doctor", state: "dancing" }), /state/);
  await assert.rejects(channel.send(message("x".repeat(17000))), /large/);
  await assert.rejects(channel.send(null), /object/);
  assert.equal(a.events.length, 0);
});

test("a bad room name is refused", async () => {
  await assert.rejects(joinRoom("bad room", { base: null }), /room/);
  await assert.rejects(joinRoom("", { base: null }), /room/);
  await assert.rejects(joinRoom("a".repeat(33), { base: null }), /room/);
});

test("local: after close nothing more arrives and send is refused", async (t) => {
  const room = uniqueRoom("local");
  const a = recorder();
  const b = recorder();
  const ca = await joinRoom(room, { ...a, base: null });
  const cb = await joinRoom(room, { ...b, base: null });
  t.after(() => ca.close());
  cb.close();
  cb.close(); // twice is fine
  await ca.send(message("after close"));
  await sleep(50);
  assert.equal(b.events.length, 0);
  await assert.rejects(cb.send(message("x")), /closed/);
});

// --- relay mode, against a scripted fake relay ------------------------------------

function json(status, body) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

/**
 * A fake relay: fetch answers ping and send; each EventSource created takes the
 * next behaviour from `plan` ("open", "fail" or "hang"; the last one repeats).
 */
function fakeRelay({ ping = () => json(200, { ok: true }), plan = ["open"] } = {}) {
  const sources = [];
  const posts = [];
  const relay = { sources, posts, plan, sendAnswer: () => json(200, { ok: true, id: "srv-1" }) };
  class FakeEventSource {
    constructor(url) {
      this.url = url;
      this.closed = false;
      sources.push(this);
      const step = relay.plan.length > 1 ? relay.plan.shift() : relay.plan[0];
      if (step === "open") setTimeout(() => this.open(), 0);
      if (step === "fail") setTimeout(() => this.fail(), 0);
    }
    open() {
      if (!this.closed) this.onopen?.({ type: "open" });
    }
    emit(event) {
      if (!this.closed) this.onmessage?.({ type: "message", data: JSON.stringify(event) });
    }
    fail() {
      if (!this.closed) this.onerror?.({ type: "error" });
    }
    close() {
      this.closed = true;
    }
  }
  relay.EventSource = FakeEventSource;
  relay.fetch = async (url, init = {}) => {
    const { pathname, searchParams } = new URL(url);
    if (pathname === "/relay/ping") return ping();
    if (pathname === "/relay/send") {
      posts.push({ room: searchParams.get("room"), method: init.method, event: JSON.parse(init.body) });
      return relay.sendAnswer();
    }
    return json(404, { ok: false, error: "unknown relay path" });
  };
  relay.options = (extra = {}) => ({
    base: "http://relay.test/relay/",
    fetch: relay.fetch,
    EventSource: FakeEventSource,
    retryMs: [5],
    ...extra,
  });
  return relay;
}

test("relay: uses the relay when ping answers and reports connected", async (t) => {
  const relay = fakeRelay();
  const a = recorder();
  const channel = await joinRoom("demo", { ...a, ...relay.options() });
  t.after(() => channel.close());
  assert.equal(channel.transport, "relay");
  assert.deepEqual(a.states, ["connected"]);
  assert.equal(relay.sources.length, 1);
  assert.equal(relay.sources[0].url, "http://relay.test/relay/events?room=demo");
});

test("relay: send posts the event and leaves the echo to the stream", async (t) => {
  const relay = fakeRelay();
  const a = recorder();
  const channel = await joinRoom("demo", { ...a, ...relay.options() });
  t.after(() => channel.close());
  await channel.send(message("Bonjour", "doctor"));
  assert.deepEqual(relay.posts, [{ room: "demo", method: "POST", event: message("Bonjour", "doctor") }]);
  assert.equal(a.events.length, 0, "the sender's copy comes back through the stream, not from send");
  relay.sources[0].emit({ ...message("Bonjour", "doctor"), id: "srv-1", ts: 1 });
  assert.equal(a.events.length, 1);
});

test("relay: send rejects with the relay's reason, or when the relay cannot be reached", async (t) => {
  const relay = fakeRelay();
  const channel = await joinRoom("demo", relay.options());
  t.after(() => channel.close());
  relay.sendAnswer = () => json(400, { ok: false, error: "type must be message, status or clear" });
  await assert.rejects(channel.send(message("x")), /type must be/);
  relay.sendAnswer = () => {
    throw new TypeError("Failed to fetch");
  };
  await assert.rejects(channel.send(message("x")), /relay/);
  await assert.rejects(channel.send({ type: "nope" }), /type/);
});

test("relay: after a reconnection the stored events are not delivered twice", async (t) => {
  const relay = fakeRelay();
  const a = recorder();
  const channel = await joinRoom("demo", { ...a, ...relay.options() });
  t.after(() => channel.close());
  const m1 = { ...message("one"), id: "m1", ts: 1 };
  const m2 = { ...message("two"), id: "m2", ts: 2 };
  const m3 = { ...message("three"), id: "m3", ts: 3 };
  relay.sources[0].emit(m1);
  relay.sources[0].emit(m2);
  relay.sources[0].fail();
  assert.equal(relay.sources[0].closed, true);
  await until(() => a.states.at(-1) === "connected" && relay.sources.length === 2, 2000, "reconnection");
  relay.sources[1].emit(m1); // the relay replays its history on every connection
  relay.sources[1].emit(m2);
  relay.sources[1].emit(m3);
  assert.deepEqual(a.events.map((e) => e.id), ["m1", "m2", "m3"]);
  assert.deepEqual(a.states, ["connected", "reconnecting", "connected"]);
});

test("relay: status events and events without id are passed on as they come", async (t) => {
  const relay = fakeRelay();
  const a = recorder();
  const channel = await joinRoom("demo", { ...a, ...relay.options() });
  t.after(() => channel.close());
  relay.sources[0].emit({ type: "status", from: "doctor", state: "typing", id: "s1" });
  relay.sources[0].emit({ type: "status", from: "doctor", state: "idle", id: "s2" });
  relay.sources[0].emit({ type: "clear" });
  relay.sources[0].onmessage({ data: "not json" });
  assert.deepEqual(a.events.map((e) => e.state ?? e.type), ["typing", "idle", "clear"]);
});

test("relay: a long outage is reported as offline, then connected again", async (t) => {
  const relay = fakeRelay({ plan: ["open", "fail"] });
  const a = recorder();
  const channel = await joinRoom("demo", { ...a, ...relay.options({ offlineAfterMs: 40 }) });
  t.after(() => channel.close());
  relay.sources[0].fail();
  await until(() => a.states.includes("offline"), 2000, "offline");
  await sleep(30);
  assert.ok(relay.sources.length > 3, "it keeps trying");
  relay.plan.splice(0, relay.plan.length, "open");
  await until(() => a.states.at(-1) === "connected", 2000, "back online");
  assert.deepEqual(a.states, ["connected", "reconnecting", "offline", "connected"]);
});

test("relay: a first connection that never opens still gives a relay channel", async (t) => {
  const relay = fakeRelay({ plan: ["hang"] });
  const a = recorder();
  const started = Date.now();
  const channel = await joinRoom("demo", { ...a, ...relay.options({ connectTimeoutMs: 30 }) });
  t.after(() => channel.close());
  assert.ok(Date.now() - started < 1000);
  assert.equal(channel.transport, "relay");
  assert.deepEqual(a.states, ["reconnecting"]);
  relay.sources[0].open();
  assert.deepEqual(a.states, ["reconnecting", "connected"]);
});

test("relay: after close there are no more events, states or reconnections", async () => {
  const relay = fakeRelay();
  const a = recorder();
  const channel = await joinRoom("demo", { ...a, ...relay.options() });
  const source = relay.sources[0];
  channel.close();
  assert.equal(source.closed, true);
  source.onmessage?.({ data: JSON.stringify({ ...message("late"), id: "late" }) });
  source.onerror?.({});
  await sleep(30);
  assert.equal(a.events.length, 0);
  assert.deepEqual(a.states, ["connected"]);
  assert.equal(relay.sources.length, 1);
  await assert.rejects(channel.send(message("x")), /closed/);
});

test("relay: falls back to local when ping fails, errs or hangs", async (t) => {
  const cases = {
    "404": () => json(404, { ok: false }),
    "not the relay": () => json(200, { hello: true }),
    error: () => {
      throw new TypeError("Failed to fetch");
    },
    hang: () => new Promise(() => {}),
  };
  for (const [name, ping] of Object.entries(cases)) {
    const relay = fakeRelay({ ping });
    const channel = await joinRoom(uniqueRoom("fallback"), relay.options({ pingTimeoutMs: 30 }));
    t.after(() => channel.close());
    assert.equal(channel.transport, "local", name);
    assert.equal(relay.sources.length, 0, name);
  }
});

test("relay: falls back to local when the browser has no EventSource", async (t) => {
  const relay = fakeRelay();
  const channel = await joinRoom(uniqueRoom("noes"), relay.options({ EventSource: null }));
  t.after(() => channel.close());
  assert.equal(channel.transport, "local");
});

// --- relay mode, against the real web/serve.py --------------------------------------

/**
 * Just enough of EventSource for Node 22 (which has none without a flag), on
 * top of fetch. It does not reconnect: joinRoom does that itself.
 */
class FetchEventSource {
  constructor(url) {
    this.url = url;
    this.readyState = 0;
    this.controller = new AbortController();
    this.run();
  }

  async run() {
    try {
      const response = await fetch(this.url, { headers: { Accept: "text/event-stream" }, signal: this.controller.signal });
      if (!response.ok || !String(response.headers.get("content-type")).startsWith("text/event-stream")) {
        throw new Error(`bad answer ${response.status}`);
      }
      this.readyState = 1;
      this.onopen?.({ type: "open" });
      const decoder = new TextDecoder();
      let buffer = "";
      for await (const chunk of response.body) {
        buffer += decoder.decode(chunk, { stream: true });
        let end;
        while ((end = buffer.indexOf("\n\n")) >= 0) {
          const block = buffer.slice(0, end);
          buffer = buffer.slice(end + 2);
          const data = block.split("\n").filter((line) => line.startsWith("data:")).map((line) => line.slice(5).replace(/^ /, ""));
          if (data.length && this.readyState === 1) this.onmessage?.({ type: "message", data: data.join("\n") });
        }
      }
      throw new Error("stream ended");
    } catch {
      if (this.readyState !== 2) {
        this.readyState = 2;
        this.onerror?.({ type: "error" });
      }
    }
  }

  close() {
    this.readyState = 2;
    this.controller.abort();
  }
}

const SERVE = fileURLToPath(new URL("../serve.py", import.meta.url));
const PYTHON = process.env.PYTHON || "python3";

/** Starts web/serve.py on `port` (0: any free port); resolves { port, stop() } or null. */
function startServer(port = 0) {
  return new Promise((resolve) => {
    const child = spawn(PYTHON, [SERVE, "--port", String(port)], { stdio: ["ignore", "pipe", "ignore"] });
    let output = "";
    const exited = new Promise((done) => child.once("exit", done));
    const stop = async () => {
      if (child.exitCode === null && child.signalCode === null) child.kill("SIGTERM");
      await exited;
    };
    child.once("error", () => resolve(null));
    child.stdout.on("data", (chunk) => {
      output += chunk;
      const match = output.match(/http:\/\/127\.0\.0\.1:(\d+)\//);
      if (match) {
        child.stdout.removeAllListeners("data");
        child.stdout.resume();
        resolve({ port: Number(match[1]), stop });
      }
    });
    child.once("exit", () => resolve(null));
  });
}

test("relay: two screens through the real serve.py, then a relay restart", { timeout: 20000 }, async (t) => {
  let server = await startServer();
  if (!server) {
    t.skip(`could not start ${SERVE} with ${PYTHON}`);
    return;
  }
  const port = server.port;
  t.after(() => server?.stop());
  const options = { base: `http://127.0.0.1:${port}/relay/`, EventSource: FetchEventSource, retryMs: [50], offlineAfterMs: 10000 };
  const room = uniqueRoom("real");
  const patient = recorder();
  const doctor = recorder();
  const cp = await joinRoom(room, { ...patient, ...options });
  const cd = await joinRoom(room, { ...doctor, ...options });
  t.after(() => [cp, cd].forEach((c) => c.close()));
  assert.equal(cp.transport, "relay");
  assert.equal(cd.transport, "relay");

  const started = performance.now();
  await cp.send(message("J'ai de la fièvre.", "patient", { signs: ["FIEBRE2"] }));
  await until(() => patient.events.length === 1 && doctor.events.length === 1, 3000, "the message on both screens");
  const delay = performance.now() - started;
  await cd.send({ type: "status", from: "doctor", state: "typing" });
  await until(() => patient.events.length === 2 && doctor.events.length === 2, 3000, "the status on both screens");
  await sleep(100);
  assert.equal(patient.events.length, 2);
  assert.equal(doctor.events.length, 2);
  assert.equal(patient.events[0].id, doctor.events[0].id);
  assert.equal(patient.events[0].text, "J'ai de la fièvre.");
  assert.ok(delay < 1000, `delay ${delay} ms`);

  // A screen that joins late gets the stored message once, and not the status.
  const late = recorder();
  const cl = await joinRoom(room, { ...late, ...options });
  t.after(() => cl.close());
  await until(() => late.events.length === 1, 3000, "the history");
  await sleep(100);
  assert.deepEqual(late.events.map((e) => e.text), ["J'ai de la fièvre."]);

  // The relay restarts on the same port: both screens reconnect on their own.
  await server.stop();
  await until(() => patient.states.at(-1) === "reconnecting" && doctor.states.at(-1) === "reconnecting", 3000, "reconnecting");
  server = await startServer(port);
  assert.ok(server, "the relay restarts");
  await until(() => patient.states.at(-1) === "connected" && doctor.states.at(-1) === "connected", 5000, "connected again");
  await cd.send(message("Toussez-vous ?", "doctor", { via: "text", signs: ["TOSER"] }));
  await until(() => patient.events.length === 3 && doctor.events.length === 3, 3000, "the reply on both screens");
  await sleep(100);
  assert.deepEqual(patient.events.map((e) => e.type), ["message", "status", "message"]);
  assert.deepEqual(doctor.events.map((e) => e.type), ["message", "status", "message"]);
  assert.deepEqual(patient.states, ["connected", "reconnecting", "connected"]);
});
