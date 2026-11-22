// The real-time link between the patient and doctor screens (web/CONVERSATION.md).
//
//   const channel = await joinRoom("demo", { onEvent, onState });
//   channel.transport;            // "relay" (web/serve.py) or "local" (BroadcastChannel)
//   await channel.send(event);    // rejects with an Error saying why when it was not sent
//   channel.close();
//
// Every event sent reaches onEvent of every participant, the sender included,
// exactly once. onState("connected" | "reconnecting" | "offline") reports the
// link, only when it changes.

export const ROOM_PATTERN = /^[A-Za-z0-9_-]{1,32}$/;
const MAX_BYTES = 16 * 1024;
const TYPES = ["message", "status", "clear"];
const SENDERS = ["patient", "doctor"];
const VIA = ["sign", "text", "voice"];
const STATES = ["online", "idle", "signing", "typing", "listening"];
const SEEN_LIMIT = 5000;

/** The relay beside this file when it was loaded over http(s): web/lib/ -> /relay/. */
function defaultBase() {
  try {
    const here = new URL(import.meta.url);
    if (here.protocol === "http:" || here.protocol === "https:") return new URL("../relay/", here).href;
  } catch {
    // not loaded from a URL
  }
  return null;
}

/** The same checks as the relay, so both transports refuse the same events. */
function checkEvent(event) {
  if (!event || typeof event !== "object" || Array.isArray(event)) throw new TypeError("event must be an object");
  if (!TYPES.includes(event.type)) throw new TypeError("type must be message, status or clear");
  if (event.type !== "clear" && !SENDERS.includes(event.from)) throw new TypeError("from must be patient or doctor");
  if (event.type === "status" && !STATES.includes(event.state)) {
    throw new TypeError("state must be online, idle, signing, typing or listening");
  }
  if (event.type === "message") {
    if ("via" in event && !VIA.includes(event.via)) throw new TypeError("via must be sign, text or voice");
    if ("text" in event && typeof event.text !== "string") throw new TypeError("text must be a string");
    const signs = event.signs ?? [];
    if (!Array.isArray(signs) || !signs.every((code) => typeof code === "string")) {
      throw new TypeError("signs must be a list of sign codes");
    }
  }
  const body = JSON.stringify(event);
  if (new TextEncoder().encode(body).length > MAX_BYTES) throw new TypeError(`event too large: at most ${MAX_BYTES} bytes`);
  return body;
}

function newId() {
  const bytes = new Uint8Array(8);
  if (globalThis.crypto?.getRandomValues) globalThis.crypto.getRandomValues(bytes);
  else for (let i = 0; i < bytes.length; i++) bytes[i] = Math.floor(Math.random() * 256);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

async function relayAnswers(base, fetchFn, timeoutMs) {
  const controller = typeof AbortController === "function" ? new AbortController() : null;
  let timer;
  const timeout = new Promise((resolve) => {
    timer = setTimeout(() => {
      controller?.abort();
      resolve(false);
    }, timeoutMs);
  });
  const ping = (async () => {
    try {
      const response = await fetchFn(new URL("ping", base).href, { cache: "no-store", signal: controller?.signal });
      if (!response.ok) return false;
      return (await response.json())?.ok === true;
    } catch {
      return false;
    }
  })();
  try {
    return await Promise.race([ping, timeout]);
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Joins a room. Uses the relay when `/relay/ping` answers, otherwise
 * BroadcastChannel("signbridge:" + room): two tabs of the same browser, no history.
 *
 * Besides onEvent and onState, the options are for tests and special setups:
 * base (relay URL ending in "/relay/", null forces local), fetch, EventSource,
 * BroadcastChannel, pingTimeoutMs, connectTimeoutMs, retryMs (delays between
 * reconnection attempts, the last one repeats), offlineAfterMs.
 */
export async function joinRoom(room, options = {}) {
  if (typeof room !== "string" || !ROOM_PATTERN.test(room)) {
    throw new TypeError("bad room name: use 1 to 32 letters, digits, - or _");
  }
  const {
    onEvent,
    onState,
    base = defaultBase(),
    fetch: fetchFn = globalThis.fetch?.bind(globalThis),
    EventSource: EventSourceClass = globalThis.EventSource,
    BroadcastChannel: BroadcastChannelClass = globalThis.BroadcastChannel,
    pingTimeoutMs = 2000,
    connectTimeoutMs = 4000,
    retryMs = [500, 1000, 2000, 4000],
    offlineAfterMs = 8000,
  } = options;
  const hooks = { onEvent: onEvent ?? (() => {}), onState: onState ?? (() => {}) };
  if (base && typeof fetchFn === "function" && typeof EventSourceClass === "function") {
    const relayBase = new URL(String(base).endsWith("/") ? String(base) : `${base}/`).href;
    if (await relayAnswers(relayBase, fetchFn, pingTimeoutMs)) {
      return relayChannel(room, relayBase, hooks, {
        fetchFn,
        EventSourceClass,
        connectTimeoutMs,
        retryMs: retryMs.length ? retryMs : [1000],
        offlineAfterMs,
      });
    }
  }
  return localChannel(room, hooks, BroadcastChannelClass);
}

// --- relay: EventSource in, POST out -------------------------------------------

async function relayChannel(room, base, { onEvent, onState }, settings) {
  const { fetchFn, EventSourceClass, connectTimeoutMs, retryMs, offlineAfterMs } = settings;
  const eventsUrl = new URL(`events?room=${encodeURIComponent(room)}`, base).href;
  const sendUrl = new URL(`send?room=${encodeURIComponent(room)}`, base).href;
  const seen = new Set(); // ids of stored events already delivered: the relay replays them on every connection
  let source = null;
  let closed = false;
  let state = null;
  let attempt = 0;
  let retryTimer = null;
  let offlineTimer = null;

  const setState = (next) => {
    if (closed || next === state) return;
    state = next;
    onState(next);
  };

  const startOfflineTimer = () => {
    if (offlineTimer === null) offlineTimer = setTimeout(() => setState("offline"), offlineAfterMs);
  };

  const deliver = (event) => {
    if (!event || typeof event !== "object") return;
    if (event.id != null && event.type !== "status") {
      if (seen.has(event.id)) return;
      seen.add(event.id);
      if (seen.size > SEEN_LIMIT) seen.delete(seen.values().next().value);
    }
    onEvent(event);
  };

  const scheduleReconnect = () => {
    if (closed || retryTimer !== null) return;
    if (state !== "offline") setState("reconnecting"); // stay "offline" until it works again
    startOfflineTimer();
    const delay = retryMs[Math.min(attempt, retryMs.length - 1)];
    attempt += 1;
    retryTimer = setTimeout(() => {
      retryTimer = null;
      connect();
    }, delay);
  };

  // Resolves true once the stream is open, false if this attempt failed.
  const connect = () =>
    new Promise((resolve) => {
      if (closed) return resolve(false);
      let current;
      try {
        current = new EventSourceClass(eventsUrl);
      } catch {
        resolve(false);
        scheduleReconnect();
        return;
      }
      source = current;
      current.onopen = () => {
        if (closed || current !== source) return;
        attempt = 0;
        clearTimeout(offlineTimer);
        offlineTimer = null;
        setState("connected");
        resolve(true);
      };
      current.onmessage = (message) => {
        if (closed || current !== source) return;
        let event;
        try {
          event = JSON.parse(message.data);
        } catch {
          return;
        }
        deliver(event);
      };
      current.onerror = () => {
        if (closed || current !== source) return;
        current.close(); // reconnect on our own schedule, whatever the browser would do
        source = null;
        resolve(false);
        scheduleReconnect();
      };
    });

  let timer;
  const opened = await Promise.race([
    connect(),
    new Promise((resolve) => {
      timer = setTimeout(() => resolve(null), connectTimeoutMs);
    }),
  ]);
  clearTimeout(timer);
  if (opened === null && !closed) {
    setState("reconnecting");
    startOfflineTimer();
  }

  return {
    transport: "relay",
    async send(event) {
      if (closed) throw new Error("channel closed");
      const body = checkEvent(event);
      let response;
      try {
        response = await fetchFn(sendUrl, { method: "POST", headers: { "Content-Type": "application/json" }, body });
      } catch (error) {
        throw new Error("relay unreachable: the event was not sent", { cause: error });
      }
      let answer = null;
      try {
        answer = await response.json();
      } catch {
        // not JSON: reported below
      }
      if (!response.ok || answer?.ok !== true) {
        throw new Error(answer?.error ? `relay refused the event: ${answer.error}` : `relay answered ${response.status}`);
      }
    },
    close() {
      if (closed) return;
      closed = true;
      clearTimeout(retryTimer);
      clearTimeout(offlineTimer);
      source?.close();
      source = null;
    },
  };
}

// --- local: BroadcastChannel ---------------------------------------------------------

function localChannel(room, { onEvent, onState }, BroadcastChannelClass) {
  let channel = null;
  try {
    channel = typeof BroadcastChannelClass === "function" ? new BroadcastChannelClass(`signbridge:${room}`) : null;
  } catch {
    channel = null;
  }
  let closed = false;
  if (channel) {
    channel.onmessage = (message) => {
      if (!closed) onEvent(message.data);
    };
  }
  onState(channel ? "connected" : "offline"); // without BroadcastChannel only this screen hears itself

  return {
    transport: "local",
    async send(event) {
      if (closed) throw new Error("channel closed");
      checkEvent(event);
      const stamped = {
        ...event,
        id: event.id == null || event.id === "" ? newId() : event.id,
        ts: typeof event.ts === "number" ? event.ts : Date.now(),
      };
      const wire = JSON.parse(JSON.stringify(stamped)); // what the relay would deliver
      channel?.postMessage(wire);
      onEvent(wire); // BroadcastChannel does not deliver to its own sender
    },
    close() {
      if (closed) return;
      closed = true;
      channel?.close();
    },
  };
}
