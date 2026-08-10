/** A URL stimulus is loaded into an iframe, and an iframe src without a scheme
 * is a *relative path*: "ryankm.com/lab" resolves against this app's own
 * origin and the participant is shown this site's 404 instead of the page
 * being tested. Nothing about that failure looks like a typo, so the address
 * is resolved and checked here rather than at the point it breaks.
 *
 * This runs in two places for the same reason a preflight exists at all:
 * at the form, so a bad address is refused before it is saved, and again at
 * render time, because studies saved before the form checked anything are
 * still sitting in IndexedDB with raw text in the url field. Stored data
 * outlives every validation added after it was written.
 */
export function normaliseStimulusUrl(raw: string): { url: string } | { problem: string } {
  const candidate = /^[a-z][a-z0-9+.-]*:/i.test(raw) ? raw : `https://${raw}`;

  let parsed: URL;
  try {
    parsed = new URL(candidate);
  } catch {
    return { problem: "That does not look like a web address. Try something like example.com/pricing." };
  }

  if (parsed.protocol !== "https:" && parsed.protocol !== "http:") {
    return { problem: "Only http and https addresses can be loaded as a stimulus." };
  }
  // A bare word like "j" parses as a hostname, but a hostname without a dot
  // is not something a participant's browser will ever resolve, so it is
  // refused with the same message a malformed address gets.
  if (!parsed.hostname.includes(".") && parsed.hostname !== "localhost") {
    return { problem: "That does not look like a web address. Try something like example.com/pricing." };
  }
  if (parsed.hostname === window.location.hostname) {
    return { problem: "That is this tool's own address. Point it at the page you want tested." };
  }
  if (parsed.protocol === "http:" && window.location.protocol === "https:") {
    return { problem: "A plain http page cannot be embedded in a secure page. Use https, or test a screenshot instead." };
  }

  return { url: parsed.toString() };
}
