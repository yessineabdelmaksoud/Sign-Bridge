import assert from "node:assert/strict";
import test from "node:test";

import { joinUrl, newRoomName, roomFrom, shareBase } from "../lib/invite.js";

const place = (href) => {
  const url = new URL(href);
  return { origin: url.origin, hostname: url.hostname, pathname: url.pathname };
};

test("a new room is four digits, easy to read out", () => {
  assert.equal(newRoomName(() => 0), "1000");
  assert.equal(newRoomName(() => 0.99999), "9999");
  for (let i = 0; i < 50; i++) assert.match(newRoomName(), /^[1-9]\d{3}$/);
});

test("the room comes from ?room=, and only an allowed name counts", () => {
  assert.equal(roomFrom("?room=4821"), "4821");
  assert.equal(roomFrom("?room=salle_2&x=1"), "salle_2");
  assert.equal(roomFrom(""), null);
  assert.equal(roomFrom("?room="), null);
  assert.equal(roomFrom("?room=a%20b"), null);
  assert.equal(roomFrom(undefined), null);
});

test("joinUrl opens the other screen in the same room", () => {
  assert.equal(joinUrl("http://10.0.0.5:8520/", "patient.html", "4821"), "http://10.0.0.5:8520/patient.html?room=4821");
});

test("a page opened on the network shares its own address", async () => {
  const calls = [];
  const fetch = async (url) => { calls.push(url); throw new Error("not asked"); };
  assert.deepEqual(await shareBase(place("http://10.0.0.5:8520/doctor.html?room=1"), fetch),
    { base: "http://10.0.0.5:8520/", reachable: true });
  assert.deepEqual(await shareBase(place("https://clinic.example/app/doctor.html"), fetch),
    { base: "https://clinic.example/app/", reachable: true });
  assert.deepEqual(calls, []);
});

test("a page opened on localhost asks the server for its network address", async () => {
  const fetch = async (url) => {
    assert.equal(url, "relay/address");
    return { ok: true, json: async () => ({ ok: true, urls: ["http://192.168.1.20:8520"] }) };
  };
  assert.deepEqual(await shareBase(place("http://localhost:8520/doctor.html"), fetch),
    { base: "http://192.168.1.20:8520/", reachable: true });
});

test("without a network address, the code only works on this computer, and says so", async () => {
  const none = async () => ({ ok: true, json: async () => ({ ok: true, urls: [] }) });
  const down = async () => { throw new TypeError("no server"); };
  for (const fetch of [none, down]) {
    assert.deepEqual(await shareBase(place("http://127.0.0.1:8520/doctor.html"), fetch),
      { base: "http://127.0.0.1:8520/", reachable: false });
  }
});
