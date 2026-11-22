// Rooms and their QR codes (web/CONVERSATION.md). The doctor's screen makes a room
// and shows its code; the patient scans it with the phone's camera and arrives in the
// same room. The patient's screen can show a code for the doctor the same way.

export const ROOM_PATTERN = /^[A-Za-z0-9_-]{1,32}$/;
const LOOPBACK = new Set(["localhost", "127.0.0.1", "[::1]", "::1"]);

/** A new room name: four digits, easy to read out ("4821"). */
export function newRoomName(random = Math.random) {
  return String(1000 + Math.min(8999, Math.floor(random() * 9000)));
}

/** The room in a page address (?room=...), or null when there is none or it is not allowed. */
export function roomFrom(search) {
  const asked = new URLSearchParams(search ?? "").get("room");
  return asked && ROOM_PATTERN.test(asked) ? asked : null;
}

/**
 * Where another device can open this app: this page's own address unless it is
 * localhost, else the network address the server gives (GET relay/address).
 * Returns { base, reachable }; base ends with the app's folder ("http://10.0.0.5:8520/").
 * reachable is false when only this computer can open it.
 */
export async function shareBase(location, fetchImpl) {
  const folder = location.pathname.replace(/[^/]*$/, "") || "/";
  if (!LOOPBACK.has(location.hostname)) return { base: `${location.origin}${folder}`, reachable: true };
  try {
    const response = await (fetchImpl ?? ((...args) => globalThis.fetch(...args)))("relay/address", { cache: "no-store" });
    const urls = response.ok ? (await response.json())?.urls : null;
    if (Array.isArray(urls) && urls.length) {
      return { base: `${String(urls[0]).replace(/\/+$/, "")}${folder}`, reachable: true };
    }
  } catch { /* no server: this computer only */ }
  return { base: `${location.origin}${folder}`, reachable: false };
}

/** The address that opens `page` ("patient.html" or "doctor.html") in `room`. */
export function joinUrl(base, page, room) {
  return `${base}${page}?room=${encodeURIComponent(room)}`;
}
