/** Minimal DOM helpers. The app is mostly canvas work, so a framework would
 * mostly get in the way of the parts that matter. */

type Child = Node | string | null | undefined | false;

/** An attribute value. `null`, `undefined` and `false` drop the attribute
 * entirely, so a conditional one can be written inline; `true` sets it bare. */
type AttrValue = string | number | boolean | null | undefined;

/** Attributes, plus event handlers under `on`-prefixed keys. */
export type Attrs = Record<string, AttrValue | EventListener>;

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Attrs = {},
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);

  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;

    if (typeof value === "function") {
      // Whether something is a listener is decided by the key, never inferred
      // from the value's runtime type. Inferring it means a plain-looking
      // `value: fn` quietly registers a "value" listener nobody ever fires
      // while the attribute the caller asked for is never set — a failure with
      // nothing at all to see. Refusing it is the only way it gets noticed.
      if (!key.startsWith("on")) {
        throw new TypeError(`el(): "${key}" was given a function; event keys must start with "on"`);
      }
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === "class") {
      node.className = String(value);
    } else if (value === true) {
      node.setAttribute(key, "");
    } else {
      node.setAttribute(key, String(value));
    }
  }

  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    node.append(typeof child === "string" ? document.createTextNode(child) : child);
  }

  return node;
}

export function clear(node: HTMLElement): void {
  while (node.firstChild) node.removeChild(node.firstChild);
}

/**
 * Makes every sibling of a full-screen overlay inert while it is up. The
 * screen underneath stays in the DOM, and without this Tab still reaches its
 * buttons and a screen reader still reads straight through the overlay.
 * Returns a restore function; anything already inert is left alone.
 */
export function inertSiblings(host: HTMLElement, overlay: HTMLElement): () => void {
  const covered = Array.from(host.children).filter(
    (node): node is HTMLElement =>
      node instanceof HTMLElement && node !== overlay && !node.hasAttribute("inert")
  );
  for (const node of covered) node.setAttribute("inert", "");
  return () => {
    for (const node of covered) node.removeAttribute("inert");
  };
}

/**
 * A two-step destructive button: the first press arms it, a second press
 * within three seconds fires. Everything in this app lives in one browser's
 * storage, so deletion has no server copy and no undo — nothing destructive
 * should fire on a single click, and a native confirm() would break the
 * page's voice to ask.
 *
 * The armed state wears `--signal-bad` (see `.btn.is-armed`), because it used
 * to arm in the same teal outline as every other secondary control: the one
 * irreversible action in the app asked for confirmation more quietly than the
 * low-signal quality badge two lines above it.
 *
 * Both labels are always in the button, stacked in one grid cell with the
 * inactive one hidden. That fixes the width at the wider of the two, so arming
 * cannot reflow the row it sits in — the recordings list used to push the
 * participant's stats onto a second line at the exact moment the operator was
 * deciding whether to go through with it. `visibility: hidden` also keeps the
 * hidden label out of the accessibility tree, so it is never read twice.
 */
export function confirmButton(
  label: string,
  armedLabel: string,
  onConfirm: () => void,
  className = "btn btn-ghost btn-small"
): HTMLButtonElement {
  const btn = el(
    "button",
    { class: `${className} confirm-btn`, type: "button" },
    el("span", { class: "confirm-rest" }, label),
    el("span", { class: "confirm-armed" }, armedLabel)
  );
  let disarm = 0;
  btn.addEventListener("click", () => {
    if (btn.classList.contains("is-armed")) {
      window.clearTimeout(disarm);
      btn.classList.remove("is-armed");
      onConfirm();
      return;
    }
    btn.classList.add("is-armed");
    disarm = window.setTimeout(() => btn.classList.remove("is-armed"), 3000);
  });
  return btn;
}

export function formatMs(ms: number): string {
  if (!Number.isFinite(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/**
 * Coarse on purpose: "3 days ago" is what a researcher scans for, and a
 * to-the-minute stamp on a list of ten studies is noise. Exact time is
 * available where it matters, as the title on the element this labels.
 *
 * Shared by the study list and the recordings list rather than duplicated:
 * "yesterday" on one screen and "1 day ago" on the other is the kind of drift
 * that makes two lists look like two products.
 */
export function relativeDay(timestamp: number, now = Date.now()): string {
  const days = Math.floor((now - timestamp) / 86_400_000);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 30) return `${days} days ago`;
  const months = Math.round(days / 30);
  if (months < 12) return `${months} month${months === 1 ? "" : "s"} ago`;
  const years = Math.round(days / 365);
  return `${years} year${years === 1 ? "" : "s"} ago`;
}

/** Waits for the next paint, so measurements happen after layout settles. */
export function nextFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
