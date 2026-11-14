import { el } from "./dom";

/**
 * The site chrome, shared by every screen.
 *
 * This tool is part of the Lab on ryankm.com and is hosted off-site, so the
 * bar is the only thing carrying the brand and the only route home. The
 * session and results screens replace the whole page, and when they dropped
 * the bar the deepest screens — the ones a visitor spends the most time on —
 * were the least identifiable: a bare ghost link floating on cream, with no
 * way back to the site at all.
 */

export const SITE_URL = "https://ryankm.com";
export const LAB_URL = `${SITE_URL}/lab`;

/** A bar, not a breadcrumb. A breadcrumb is a same-origin device: it says "you
 * are inside this section", and here the parent link leaves the origin, so it
 * would be an exit dressed as a way back up. The wordmark and one route home
 * are honest about being somewhere else, and they give a visitor who arrived
 * from a search result something to arrive at.
 *
 * The wordmark is the site's: the full name, set in tracked capitals by the
 * stylesheet (the text stays "Ryan McCarty" for a screen reader). The old
 * mark, the name with a vermillion full stop, went with the redesign. The way
 * back is one of the header's pills. */
export function appBar(): HTMLElement {
  return el(
    "div",
    { class: "site-bar" },
    el(
      "div",
      { class: "container bar-inner" },
      el("a", { class: "wordmark", href: SITE_URL, "aria-label": "Ryan McCarty, ryankm.com" }, "Ryan McCarty"),
      el(
        "a",
        { class: "bar-back", href: LAB_URL },
        "Back to the Lab",
        el("span", { "aria-hidden": "true" }, "↗")
      )
    )
  );
}
