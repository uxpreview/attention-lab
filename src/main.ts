import "./styles.css";

import { gradeTracking, TRACKING_BAD } from "./analysis/quality";
import { deleteStudy, listRecordings, listStudies, newId, saveStudy } from "./data/store";
import { normaliseStimulusUrl } from "./data/stimulusUrl";
import type { Study } from "./data/types";
import { FEATURE_BASIS_VERSION, FEATURE_DIM } from "./tracker/features";
import { GazeEngine, type TrackerStatus } from "./tracker/gaze";
import { deserialiseModel, isSerialisedModel, serialiseModel } from "./tracker/regression";
import { describeAccuracy, runCalibration } from "./ui/calibration";
import { appBar, LAB_URL } from "./ui/chrome";
import { clear, confirmButton, el, relativeDay } from "./ui/dom";
import { controlBandHeight, FIT_SCALE_FLOOR, fitStimulus, runRecording } from "./ui/record";
import { renderResults } from "./ui/results";

const app = mountPoint("app");

/** Fails loudly at startup if index.html and this module disagree about the
 * mount point, instead of casting the null away and crashing later. */
function mountPoint(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (!node) throw new Error(`index.html no longer has the #${id} mount point this module renders into`);
  return node;
}

const engine = new GazeEngine();

/** The active session's status-listener unsubscribe. Torn down in
 * showStudyList alongside engine.stop(), because the study list is the one
 * screen every abandoned session — back button, cancelled calibration —
 * funnels through. A listener left behind would fire on every frame of the
 * next session, writing into the dead session's detached DOM and keeping it
 * out of GC's reach. */
let releaseStatusListener: (() => void) | null = null;

/** Calibration is per-person, but a repeat participant in the same sitting can
 * reuse theirs. Session storage, not local: a stale calibration from yesterday
 * is worse than no calibration, because it fails silently. */
const CALIBRATION_KEY = "eyetrack.calibration";

interface StoredCalibration {
  /** Which feature basis the model was fit on. A stored model from an older
   * basis would be silently misapplied to features it never saw, so a
   * mismatch invalidates the calibration entirely. */
  basis: number;
  model: ReturnType<typeof serialiseModel>;
  validationError: number | null;
  savedAt: number;
  participant: string;
}

function loadStoredCalibration(): StoredCalibration | null {
  try {
    const raw = sessionStorage.getItem(CALIBRATION_KEY);
    if (!raw) return null;
    // Storage is an untrusted boundary: the cast below is only honest because
    // every field is checked first. The basis version catches a model fit on
    // an older feature set; the shape check catches everything else. A bad
    // payload means a fresh calibration is offered, never a silently wrong one.
    const stored: unknown = JSON.parse(raw);
    if (typeof stored !== "object" || stored === null) return null;
    const candidate = stored as Partial<StoredCalibration>;
    if (
      candidate.basis !== FEATURE_BASIS_VERSION ||
      !isSerialisedModel(candidate.model, FEATURE_DIM) ||
      (candidate.validationError !== null && typeof candidate.validationError !== "number") ||
      typeof candidate.savedAt !== "number" ||
      typeof candidate.participant !== "string"
    ) {
      return null;
    }
    return candidate as StoredCalibration;
  } catch {
    return null;
  }
}

function storeCalibration(value: StoredCalibration | null): void {
  try {
    if (value) sessionStorage.setItem(CALIBRATION_KEY, JSON.stringify(value));
    else sessionStorage.removeItem(CALIBRATION_KEY);
  } catch {
    /* private browsing; calibration simply will not persist */
  }
}

// --- The experiment page shell -------------------------------------------
// This tool is EXP-038 in the Lab on ryankm.com and it is hosted here rather
// than there, so the page has to do the work the site's /lab/<slug> route
// normally does: the trail back up, the record voice (number, kind, state,
// then the stack), the claim in display type, and the lede. Same order, same
// tokens, same measure. See docs/lab-strategy.md on the site, under
// "Experiments that live off-site".

/** A phone cannot run a session, and the honest thing is to say so rather than
 * let someone spend a participant finding out.
 *
 * Three reasons, none of them fixable by design. The camera is a hand's length
 * from the face and off to one side, so the iris is a few pixels across and
 * the head pose estimate is working from almost nothing. Calibration asks you
 * to look at a point and tap it, and on a handheld your thumb covers the point
 * and your arm moves the camera in the same motion. And a screen this size
 * cannot show a desktop stimulus at a size anyone can read, so the task itself
 * stops being the task.
 *
 * Reading the page and looking at results a session already produced are fine,
 * and stay available. */
function isHandheld(): boolean {
  return (
    typeof window !== "undefined" &&
    window.matchMedia("(pointer: coarse)").matches &&
    Math.min(window.innerWidth, window.innerHeight) < 820
  );
}

/** The smallest window a session can honestly run in. Calibration puts points
 * at 6% and 94% of each axis, which below this lands them within a couple of
 * dozen pixels of the window edge — close enough that a participant's gaze
 * leaves the screen rather than reaching the dot — and a 1280px stimulus has
 * nowhere to be read at. */
const MIN_SESSION_EDGE = 700;

/**
 * Why a session cannot run here, or null if it can.
 *
 * The handheld notice was pointer-based only: it wanted `(pointer: coarse)`
 * AND a small screen, so a 390px-wide *desktop* window sailed straight past it
 * with the full flow armed — as did a tablet with a trackpad. The honesty of
 * the notice is worth nothing if resizing a window defeats it, so the size
 * guard now stands on its own, independent of what kind of pointer is driving.
 */
function sessionBlockReason(): string | null {
  if (isHandheld()) return "Sessions need a laptop or a desktop";
  if (Math.min(window.innerWidth, window.innerHeight) < MIN_SESSION_EDGE) {
    return `This window is too small to calibrate in — make it at least ${MIN_SESSION_EDGE}px in both directions`;
  }
  return null;
}

/**
 * Everything on the study list whose state depends on whether a session can
 * run: the notice at the top and every Run session button.
 *
 * They are updated in place rather than by re-rendering, because the block
 * depends on the *window size* — the one screen state a user changes by hand,
 * often while the create form has half a study typed into it. Cleared at the
 * start of each render, so detached nodes are not kept alive by the listener.
 */
let blockWatchers: Array<(reason: string | null) => void> = [];

function watchSessionBlock(paint: (reason: string | null) => void): void {
  blockWatchers.push(paint);
  paint(sessionBlockReason());
}

let lastBlockReason = sessionBlockReason();
let blockResizeTimer = 0;
window.addEventListener("resize", () => {
  window.clearTimeout(blockResizeTimer);
  blockResizeTimer = window.setTimeout(() => {
    const reason = sessionBlockReason();
    if (reason === lastBlockReason) return;
    lastBlockReason = reason;
    for (const paint of blockWatchers) paint(reason);
  }, 200);
});

function experimentHead(): HTMLElement {
  // The bar is shared with the session and results screens, which used to drop
  // the chrome entirely — see ui/chrome.ts.
  const bar = appBar();

  const meta = el(
    "div",
    { class: "exp-meta" },
    el("span", { class: "label label-strong" }, "EXP-038"),
    el("span", { class: "label" }, "tool"),
    el(
      "span",
      { class: "label label-strong exp-state" },
      el("span", { class: "status-dot", "aria-hidden": "true" }),
      "Live"
    ),
    el("span", { class: "exp-rule", "aria-hidden": "true" }),
    el("span", { class: "label exp-stack" }, "MediaPipe · Ridge regression · Client-side only")
  );

  return el(
    "header",
    { class: "app-header" },
    bar,
    el(
      "div",
      { class: "container exp-head" },
      meta,
      el("h1", { class: "t-h1" }, "Attention Lab", el("span", { class: "dot" }, ".")),
      // Two sentences, not six. This is a tool people open to run a session,
      // and the lede used to push every control below a 900px fold — the page
      // read as an essay with a tool bolted underneath. The accuracy caveat
      // that used to close it is stated where it actually binds a decision:
      // the calibration readout, and the first bench note.
      el(
        "p",
        { class: "t-lede" },
        "Point a webcam at a wireframe and find out where people actually look. Upload a screen, set a task, and get back a heatmap, a scanpath, and attention numbers per region — accurate to a block rather than a word."
      )
    )
  );
}

// --- Study list ----------------------------------------------------------

/** The study being edited, if the list was re-rendered from an Edit button.
 * Consumed by the next render: the form opens pre-filled, then the flag
 * clears so navigation elsewhere returns the form to create mode. */
let editingStudy: Study | null = null;

async function showStudyList(): Promise<void> {
  // The study list is the one screen that never needs the camera, so releasing
  // it here covers every way of leaving a session — including abandoning
  // calibration, where nothing else would turn the camera light off.
  engine.stop();
  releaseStatusListener?.();
  releaseStatusListener = null;

  clear(app);
  blockWatchers = [];
  const studies = await listStudies();
  const editing = editingStudy;
  editingStudy = null;

  // One listRecordings pass per study, up front. Without it the only way to
  // find the study that has data is to open Results on each in turn.
  const stats = new Map<string, StudyStats>(
    await Promise.all(
      studies.map(async (study): Promise<[string, StudyStats]> => {
        const recordings = await listRecordings(study.id);
        return [
          study.id,
          {
            count: recordings.length,
            lastRun: recordings.reduce((latest, r) => Math.max(latest, r.createdAt), 0) || null,
            tracked: [...recordings]
              .sort((a, b) => a.createdAt - b.createdAt)
              .map((r) => r.quality.trackingRatio),
          },
        ];
      })
    )
  );

  const header = experimentHead();

  const list = el("section", { class: "study-list" });

  /** Nothing matched the filter — distinct from having no studies at all, and
   * the difference matters: one is an empty tool, the other is a typo. */
  const noMatches = el(
    "div",
    { class: "empty", hidden: true },
    el("h3", {}, "No studies match"),
    el("p", { class: "muted" }, "Clear the search, or drop the “with recordings” filter.")
  );

  /**
   * Client-side filtering over the cards already rendered.
   *
   * Every study is in the DOM and every count is already in `stats`, so this is
   * a pass over nodes rather than a re-render — which keeps the thumbnails from
   * reloading their blobs on every keystroke.
   */
  const applyFilter = (query: string, onlyWithRecordings: boolean): number => {
    const needle = query.trim().toLowerCase();
    let shown = 0;
    for (const card of Array.from(list.querySelectorAll<HTMLElement>(".study-card"))) {
      const haystack = `${card.dataset.name ?? ""} ${card.dataset.task ?? ""}`.toLowerCase();
      const matches =
        (needle === "" || haystack.includes(needle)) &&
        (!onlyWithRecordings || card.dataset.count !== "0");
      card.hidden = !matches;
      if (matches) shown++;
    }
    noMatches.hidden = shown > 0 || studies.length === 0;
    return shown;
  };

  if (studies.length === 0) {
    /* The zero state is the setup panel, so it is not also said in a box.
     *
     * This was a 1320×173 dashed rectangle holding "No studies yet" and one
     * muted sentence, sitting directly under a 563px panel whose display
     * heading, lede and drop zone already say the same thing and say it better.
     * It was the largest element on the lower half of the landing page and
     * carried the least — an empty state announcing emptiness underneath the
     * thing that fills it. The dashed treatment stays where it earns its space:
     * the filter miss below, where "nothing matched" is news.
     *
     * A handheld gets the sentence, because there it is not a duplicate: the
     * setup form is deliberately not rendered on a phone (a study lives in the
     * storage of the browser that made it), so without this the list would be
     * nothing at all. One muted line, not a bordered void. */
    if (isHandheld()) {
      list.append(
        el(
          "p",
          { class: "list-zero muted" },
          "Nothing has been recorded in this browser. Results of a session run on another machine stay on that machine."
        )
      );
    }
  } else {
    for (const study of studies) {
      list.append(
        studyCard(study, stats.get(study.id) ?? { count: 0, lastRun: null, tracked: [] })
      );
    }
    list.append(noMatches);
  }

  // The tool sits inside the page the way an experiment sits inside a
  // /lab/<slug> route: its own section, on the same measure as the head.
  const body = el(
    "section",
    { class: "exp-body" },
    // No setup form on a handheld. A study lives in the storage of the browser
    // it was made in, so one built on a phone could never be opened on the
    // machine that could run it: the form would only ever waste someone's
    // time. The notice above says so rather than leaving a hole.
    el(
      "div",
      { class: "container" },
      blockNoticeSlot(),
      isHandheld() ? null : setupSection(editing, studies.length, applyFilter),
      list
    )
  );

  app.append(header, body, footer());
}

/**
 * The create form, or the button that opens it.
 *
 * The panel is ~530px tall and used to render on every visit to the list, which
 * put the first study card at roughly y=940 on a 1440×900 screen: you scrolled
 * every single time to reach the work you came for, past a form you mostly did
 * not want. It earns that space in exactly two situations — the zero state,
 * where the display heading and lede are the only thing on the screen worth
 * reading, and an edit, which is a form by definition. Otherwise it collapses
 * to one button that expands it in place.
 */
function setupSection(
  editing: Study | null,
  studyCount: number,
  /** Narrows the rendered list; returns how many cards survived. */
  applyFilter: (query: string, onlyWithRecordings: boolean) => number
): HTMLElement {
  const slot = el("div", { class: "setup-slot" });

  const showForm = (existing: Study | null, collapsible: boolean): void => {
    clear(slot);
    slot.append(newStudyForm(existing, collapsible ? showBar : null));
  };

  /**
   * Search and a data filter, once there are enough studies to need them.
   *
   * The list renders every study, newest first, with no way to find one but
   * scrolling 107px rows. Three studies is fine; a researcher accumulates
   * dozens, and every project list in this category — Maze, Hotjar — leads with
   * search for that reason. Both filters are pure client-side passes over cards
   * that are already in the DOM and counts that were already computed for the
   * meta line, so nothing is fetched and no thumbnail is re-decoded.
   *
   * Below the threshold they are not drawn at all: a search field over three
   * rows is furniture that makes a small list look like a big one.
   */
  const FILTER_FROM = 4;

  function showBar(): void {
    clear(slot);
    const count = el("span", { class: "pill pill-count" }, String(studyCount));
    let onlyWithRecordings = false;

    const search = el("input", {
      class: "input input-search",
      type: "search",
      id: "study-search",
      placeholder: "Search name or task",
      "aria-label": "Search studies by name or task",
    });
    const withData = el(
      "button",
      {
        class: "btn btn-quiet btn-small",
        type: "button",
        "aria-pressed": "false",
      },
      "With recordings"
    );

    const refresh = (): void => {
      const shown = applyFilter(search.value, onlyWithRecordings);
      // The pill counts what is on the screen, and says so when that is not
      // everything — a list head reading "12" over four visible rows is the
      // filter lying about its own effect.
      count.textContent = shown === studyCount ? String(studyCount) : `${shown} of ${studyCount}`;
    };

    search.addEventListener("input", refresh);
    withData.addEventListener("click", () => {
      onlyWithRecordings = !onlyWithRecordings;
      withData.classList.toggle("is-active", onlyWithRecordings);
      withData.setAttribute("aria-pressed", onlyWithRecordings ? "true" : "false");
      refresh();
    });

    slot.append(
      el(
        "div",
        { class: "list-head" },
        el("h2", { class: "list-head-title" }, "Studies", count),
        studyCount >= FILTER_FROM
          ? el("div", { class: "list-tools" }, search, withData)
          : null,
        // Outlined, not filled. Every filled pill on this screen is a row's
        // "Run session" — one per study, in one column — and adding a fourth
        // fill in a different band for the occasional action would put the
        // list back where it started: a screen with no single anchor.
        el(
          "button",
          { class: "btn", type: "button", onclick: () => showForm(null, true) },
          "New study"
        )
      )
    );
  }

  if (editing || studyCount === 0) showForm(editing, false);
  else showBar();

  return slot;
}

/**
 * Why a session cannot be run here, said on the screen rather than in a
 * tooltip.
 *
 * This used to be handheld-only, while the block itself is not: a 660×820
 * desktop window with a mouse rendered three prominent teal "Run session" pills
 * that did nothing, with the explanation living entirely in a `title` — which
 * is to say, nowhere. Anything that blocks a session now says so out loud, and
 * updates itself when the window it is complaining about is resized.
 */
function blockNoticeSlot(): HTMLElement {
  const slot = el("div", { class: "notice-slot" });
  watchSessionBlock((reason) => {
    clear(slot);
    if (!reason) return;
    slot.append(isHandheld() ? handheldNotice() : windowTooSmallNotice(reason));
  });
  return slot;
}

function windowTooSmallNotice(reason: string): HTMLElement {
  return el(
    "aside",
    { class: "panel handheld" },
    el("p", { class: "label eyebrow" }, "Sessions are paused in this window"),
    el(
      "p",
      { class: "handheld-body" },
      `${reason}. Calibration puts its targets at 6% and 94% of each axis, and in a window this size those land close enough to the edge that a participant's gaze leaves the screen rather than reaching the dot — the model would fit to points nobody actually looked at.`
    ),
    el(
      "p",
      { class: "handheld-body" },
      "Everything else works. Studies can be created and edited, regions drawn, and existing results read; Run session comes back as soon as the window is big enough."
    )
  );
}

/** Said once, at the top, before anyone builds a study they cannot run. */
function handheldNotice(): HTMLElement {
  const copy = el(
    "button",
    { class: "btn btn-small", type: "button" },
    "Copy the link"
  );
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      copy.textContent = "Copied";
      setTimeout(() => (copy.textContent = "Copy the link"), 2000);
    } catch {
      // Clipboard access is refused in plenty of ordinary situations. Select
      // the address instead of pretending nothing happened.
      copy.textContent = "Copy from the address bar";
    }
  });

  return el(
    "aside",
    { class: "panel handheld" },
    el("p", { class: "label eyebrow" }, "Read here, run it on a desktop"),
    el(
      "p",
      { class: "handheld-body" },
      "Sessions need a laptop or a desktop. A phone camera sits a hand's length from your face and off to one side, so the iris is a few pixels across; calibration asks you to look at a point and tap it, which puts your thumb over the point and moves the camera at the same time; and a screen this size cannot show a desktop stimulus at a size anyone could actually read. The numbers would come out looking exactly like real ones."
    ),
    el(
      "p",
      { class: "handheld-body" },
      "Setting one up is not offered here either, and that is the reason rather than tidiness: studies live in the storage of the browser they were made in, so one built on this phone could never be opened on the machine that could run it."
    ),
    el(
      "p",
      { class: "handheld-body" },
      "Everything else works. Read the page, and open the results of any session already recorded in this browser."
    ),
    el("div", { class: "handheld-actions" }, copy)
  );
}

/** What a researcher needs to see without opening the study. */
interface StudyStats {
  count: number;
  /** Timestamp of the most recent recording, or null if there are none. */
  lastRun: number | null;
  /**
   * Tracking ratio per recording, oldest first — the study's signal quality at a
   * glance. It is the number that decides whether a study's results are evidence
   * or decoration, and the only way to see it used to be to open each study in
   * turn.
   */
  tracked: number[];
}

function studyCard(study: Study, stats: StudyStats): HTMLElement {
  const thumb = el("div", { class: "study-thumb" });
  if (study.stimulus.kind === "image") {
    const url = URL.createObjectURL(study.stimulus.blob);
    const img = el("img", { alt: "" });
    // The blob is only needed until the image decodes; revoking on load (or
    // error, so a broken blob does not pin its URL forever) releases it
    // without guessing at the card's lifetime. Same pattern as
    // readImageDimensions below.
    img.addEventListener("load", () => URL.revokeObjectURL(url), { once: true });
    img.addEventListener("error", () => URL.revokeObjectURL(url), { once: true });
    img.src = url;
    thumb.append(img);
  } else {
    thumb.append(el("span", { class: "study-thumb-url" }, "URL"));
  }

  return el(
    "article",
    {
      class: "study-card",
      // What the list head's search filters on, kept on the node so filtering is
      // a pass over the DOM rather than a re-render that would re-decode every
      // thumbnail on each keystroke.
      "data-name": study.name,
      "data-task": study.task,
      "data-count": String(stats.count),
    },
    thumb,
    el(
      "div",
      { class: "study-body" },
      el("h3", {}, study.name),
      el("p", { class: "muted" }, study.task || "No task set"),
      // The label voice belongs on the static fragments, never on the data. The
      // whole line used to be uppercased, which printed a case-sensitive URL as
      // "HTTPS://EXAMPLE.COM/" — factually wrong, not just loud — and turned
      // "15s" into the non-unit "15S".
      el(
        "p",
        { class: "study-meta" },
        el(
          "span",
          { class: stats.count > 0 ? "study-count" : "study-count is-empty" },
          stats.count > 0
            ? `${stats.count} recording${stats.count === 1 ? "" : "s"}`
            : "No recordings"
        ),
        stats.lastRun === null ? null : " · ",
        stats.lastRun === null ? null : el("span", { class: "meta-k" }, "last run "),
        stats.lastRun === null ? null : relativeDay(stats.lastRun),
        " · ",
        study.stimulus.kind === "image"
          ? `${study.stimulus.width}×${study.stimulus.height}`
          : el("span", { class: "meta-url" }, study.stimulus.url),
        study.stimulus.kind === "image" ? el("span", { class: "meta-k" }, " image") : null,
        " · ",
        study.duration > 0
          ? `${study.duration}s`
          : el("span", { class: "meta-k" }, "manual stop")
      )
    ),
    studySignal(stats),
    el(
      "div",
      { class: "study-actions" },
      // Two fixed slots, two fixed treatments, and the row's verb is the filled
      // one.
      //
      // The fill used to move: filled on Run session at zero recordings and on
      // Results once there were any, so down a list of three studies the one
      // high-contrast teal pill jumped between the first and the second column
      // and the eye could not establish either. The cure was to flatten
      // everything, which left a list of twelve identical outline pills in
      // which the app's primary verb carried no more weight than Edit — same
      // border, same ink, same 600 weight, differing only in width — and no
      // entry point anywhere in the list.
      //
      // What was actually wrong was the *moving*, not the fill. Run session is
      // the same column on every row, so filling it consistently costs nothing
      // and gives the list the one thing it lacked. The list head's "New study"
      // steps back to an outline in exchange: every filled pill on this screen
      // now means "start work on this study", which is the rule .btn-primary
      // states for itself. "This study has data" is still said where a
      // researcher scans for it — the recording count at the head of the meta
      // line, --strong at 600 with data and muted without (see .study-count).
      //
      // Where a session cannot run the button is disabled rather than hidden: a
      // control that vanishes leaves you wondering whether the tool is broken,
      // and the title says which of the two reasons applies.
      runSessionButton(study),
      // Not disabled at zero: there is nothing to read yet, but the results
      // screen is also where regions are drawn, and a study can usefully be
      // marked up before the first participant sits down — the screen keeps its
      // region tools at zero recordings, so the tooltip is a promise the app
      // actually honours.
      el(
        "button",
        {
          class: "btn",
          type: "button",
          title: stats.count > 0 ? null : "No recordings yet — regions can still be drawn",
          onclick: () => void openResults(study),
        },
        "Results"
      ),
      // Quiet, but bordered. These two were `.btn-ghost`, which with Results
      // also borderless at zero recordings left three of the four controls
      // reading as text links of identical weight — with the app's only
      // irreversible action sitting last in that undifferentiated run.
      el(
        "button",
        {
          class: "btn btn-quiet btn-small",
          type: "button",
          onclick: () => {
            // Recordings are irreplaceable participant time, so a typo in a
            // name or task must never cost a delete-and-recreate.
            editingStudy = study;
            void showStudyList();
          },
        },
        "Edit"
      ),
      studyDeleteButton(study)
    )
  );
}

/**
 * The study's signal quality, in the track that used to be nothing.
 *
 * The card's third column was a measured 324px of empty cream on every row — a
 * void the eye had to cross on its way to Run session, three or four times down
 * the page. What belongs in it is the fact a researcher would otherwise have to
 * open the study to learn: how well each session tracked. One bar per recording,
 * oldest to newest, filled to the tracking ratio and coloured by the same grade
 * the results screen paints — so a study with a run of amber bars is visible as
 * a problem from the list, and a run improving left to right is visible as a rig
 * that got better.
 */
function studySignal(stats: StudyStats): HTMLElement {
  if (stats.tracked.length === 0) {
    // The column keeps its heading rather than collapsing, so an empty study is
    // recognisably the same row shape as a full one — and the line says what
    // will appear there rather than repeating the "No recordings" already in the
    // meta line above.
    return el(
      "div",
      { class: "study-signal is-empty" },
      el("span", { class: "label" }, "Tracked"),
      el("span", { class: "signal-none" }, "After the first session")
    );
  }

  // The most recent dozen. Past that the bars are hairlines and the card is a
  // chart, which is not what a list row is for.
  const shown = stats.tracked.slice(-12);
  const hidden = stats.tracked.length - shown.length;
  const percent = (ratio: number): string => `${Math.round(ratio * 100)}%`;
  const mean = stats.tracked.reduce((a, b) => a + b, 0) / stats.tracked.length;

  return el(
    "div",
    { class: "study-signal" },
    el("span", { class: "label" }, "Tracked"),
    el(
      "div",
      {
        class: "signal-bars",
        // The bars are a picture, and a picture of numbers owes a screen reader
        // the numbers. One label for the group beats twelve tooltips no keyboard
        // can reach.
        role: "img",
        "aria-label": `Tracking ratio, oldest first: ${shown.map(percent).join(", ")}`,
      },
      ...shown.map((ratio) => {
        const bar = el("span", { class: "signal-bar", title: `${percent(ratio)} tracked` });
        const fill = el("span", { class: `signal-fill signal-${gradeTracking(ratio)}` });
        // A floor, so a 4%-tracked session is a visible mark rather than an
        // absence that would read as "no recording here".
        fill.style.setProperty("--v", Math.max(0.08, Math.min(1, ratio)).toFixed(3));
        bar.append(fill);
        return bar;
      }),
      hidden > 0 ? el("span", { class: "signal-more" }, `+${hidden}`) : null
    ),
    /* The grade as a number, beside the picture of it.
     *
     * The bars alone answered "how did these sessions compare to each other"
     * and nothing else: a 44% bar and a 91% bar differ only in fill height, with
     * no baseline and no scale, so "is this study healthy?" needed a hover on
     * every bar. Twelve 6px bars also left a measured 166px of the row's third
     * column empty, which was the void this column was introduced to fill. The
     * mean answers the question outright, the threshold rule drawn across each
     * track (see .signal-bar::before) gives the bars the scale they were
     * missing, and between them the track carries information instead of air.
     *
     * The mean, not the last run: the rightmost bar already *is* the last run,
     * so a number repeating it would add nothing. The title says which it is,
     * because "82%" under a heading reading "Tracked" is otherwise a number
     * without a scope. */
    el(
      "span",
      {
        class: `signal-value signal-${gradeTracking(mean)}`,
        title: `Mean ${percent(mean)} of gaze samples tracked across ${stats.tracked.length} recording${stats.tracked.length === 1 ? "" : "s"}. Under ${percent(TRACKING_BAD)} is excluded from an aggregate by default.`,
      },
      percent(mean)
    )
  );
}

function runSessionButton(study: Study): HTMLButtonElement {
  const btn = el(
    "button",
    {
      // Filled, in the same column on every row — see the note in studyCard.
      class: "btn btn-primary",
      type: "button",
      // Re-checked at click time rather than trusted from render time: the
      // window can be resized between the two.
      onclick: () => {
        if (sessionBlockReason()) return;
        void runSession(study);
      },
    },
    "Run session"
  );
  // Disabled rather than hidden — a control that vanishes leaves you wondering
  // whether the tool is broken — and the reason is now also stated on the
  // screen, in the notice at the top, rather than only in this title.
  watchSessionBlock((reason) => {
    btn.disabled = reason !== null;
    if (reason) btn.setAttribute("title", reason);
    else btn.removeAttribute("title");
  });
  return btn;
}

/** Results, with the route back and — when this machine can run one — the way
 * on to a session, so a study with no data yet is not a dead end. */
function openResults(study: Study): Promise<void> {
  // The study list's nodes are about to be replaced; nothing should keep its
  // buttons alive on the resize listener.
  blockWatchers = [];
  return renderResults(app, study, () => void showStudyList(), {
    onRunSession: sessionBlockReason() ? null : () => void runSession(study),
    runBlockedReason: sessionBlockReason(),
    // Whose calibration is cached for this sitting, if anyone's — a fact the
    // empty results rail uses to say what the next session will actually cost.
    // Read through the same loader the session flow uses, so the two can never
    // disagree about whether a cached model is still valid.
    reusableCalibration: loadStoredCalibration()?.participant ?? null,
  });
}

function studyDeleteButton(study: Study): HTMLButtonElement {
  const btn = confirmButton(
    "Delete",
    "Really delete?",
    async () => {
      await deleteStudy(study.id);
      void showStudyList();
    },
    // Bordered like Edit beside it, rather than a fourth borderless word in the
    // row. It still arms into --signal-bad on the first press.
    "btn btn-quiet btn-small"
  );
  btn.setAttribute("aria-label", `Delete study ${study.name} and all of its recordings`);
  return btn;
}

/** The study form: creates when `existing` is null, edits it otherwise.
 * Editing keeps the study's id, recordings and regions; an image stimulus
 * stays locked because every recording is normalised against it.
 *
 * `onCollapse`, when given, is what Cancel does — used by the collapsed create
 * affordance to fold the panel back into its button without a re-render. */
function newStudyForm(
  existing: Study | null = null,
  onCollapse: (() => void) | null = null
): HTMLElement {
  const lockedImage = existing !== null && existing.stimulus.kind === "image";

  const nameInput = el("input", {
    class: "input",
    placeholder: "Checkout wireframe v3",
    value: existing?.name ?? "",
  });
  const taskInput = el("input", {
    class: "input",
    placeholder: "Find where you would enter a discount code",
    value: existing?.task ?? "",
  });
  const urlInput = el("input", {
    class: "input",
    placeholder: "https://example.com",
    value: existing?.stimulus.kind === "url" ? existing.stimulus.url : "",
  });
  const durationInput = el("input", {
    class: "input input-with-suffix",
    type: "number",
    value: existing ? String(existing.duration) : "15",
    min: "0",
    "aria-describedby": "duration-hint",
  });
  // "Duration (seconds, 0 = manual)" wrapped to two lines at 12px uppercase and
  // shoved its input out of alignment with everything else: a parenthetical
  // spec crammed into a label slot. The unit belongs in the field, and the
  // special case belongs under it.
  const durationField = el(
    "div",
    { class: "input-suffix" },
    durationInput,
    el("span", { class: "input-suffix-unit", "aria-hidden": "true" }, "sec")
  );
  const fileInput = el("input", { class: "file-input", type: "file", accept: "image/*" });

  const dropTitle = el("span", { class: "drop-title" }, "Drop a wireframe or screenshot");
  const dropZone = el(
    "label",
    { class: "drop-zone" },
    fileInput,
    // A 1140×105 empty dashed rectangle reads as a gap in the layout rather
    // than a target. The glyph gives it a centre of gravity.
    el("span", { class: "drop-icon", "aria-hidden": "true" }, "↑"),
    dropTitle,
    el("span", { class: "muted" }, "PNG or JPG · or use a URL")
  );

  /**
   * What the participant will actually see, said before a participant is spent.
   *
   * A 1280×1600 full-page screenshot letterboxed into a 1440×804 recording
   * stage renders 643px wide — every label at half its designed size, on the
   * screen the finding is made on. The recording stage now switches such a
   * stimulus to full width and scrolls it (see ui/record.ts), but the operator
   * should still be told which of the two they are getting, and warned when
   * even full width leaves the text small. This is that line, computed from the
   * machine the study is being set up on.
   */
  const scaleNote = el("p", { class: "stimulus-scale", hidden: true, role: "status" });
  const describeStimulusScale = (width: number, height: number): void => {
    if (width <= 0 || height <= 0) {
      scaleNote.hidden = true;
      return;
    }
    // The same rule the recording stage applies, called rather than restated —
    // a setup form quoting a different number from the one a participant gets
    // would be worse than saying nothing.
    const fit = fitStimulus(
      { width, height },
      {
        width: window.innerWidth,
        height: Math.max(1, window.innerHeight - controlBandHeight(window.innerHeight)),
      }
    );
    const scale = fit.scale;
    const percent = Math.round(scale * 100);
    const scrolls = fit.mode === "width";
    scaleNote.hidden = false;
    scaleNote.classList.toggle("is-warn", scale < FIT_SCALE_FLOOR);
    scaleNote.textContent =
      scale < FIT_SCALE_FLOOR
        ? `${width}×${height}. On a screen this size a participant sees it at about ${percent}% — small labels may be unreadable, which puts the finding in doubt rather than only the look of it. Crop it, or export the wireframe larger.`
        : scrolls
          ? `${width}×${height}. Taller than the recording stage, so it is shown at full width (about ${percent}%) and scrolled.`
          : `${width}×${height}. Fits the recording stage at about ${percent}%.`;
  };

  let file: File | null = null;
  const setFile = (next: File | null) => {
    file = next;
    dropZone.classList.toggle("has-file", next !== null);
    dropTitle.textContent = next ? next.name : "Drop a wireframe or screenshot";
    // The uploaded image wins over the URL field. Saying so — and parking the
    // field — beats silently ignoring whatever was typed there.
    urlInput.disabled = next !== null;
    urlInput.placeholder = next ? "Using the uploaded image" : "https://example.com";
    if (!next) {
      scaleNote.hidden = true;
      return;
    }
    // Racy by nature — a second drop can land while the first is decoding — so
    // the result is discarded unless it is still the selected file.
    void readImageDimensions(next).then((dimensions) => {
      if (file !== next) return;
      describeStimulusScale(dimensions.width, dimensions.height);
    });
  };

  fileInput.addEventListener("change", () => setFile(fileInput.files?.[0] ?? null));
  dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("is-over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("is-over"));
  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-over");
    const dropped = (event as DragEvent).dataTransfer?.files?.[0];
    if (dropped && dropped.type.startsWith("image/")) setFile(dropped);
  });

  // role="alert" so it is heard rather than only seen, and an id so the field
  // at fault can point at it. It sat two fields away from the input it
  // described, in a three-column grid, with nothing marking which of four
  // inputs was wrong.
  const error = el("p", { class: "error", id: "setup-error", role: "alert" });

  /** Names the field at fault, marks it, moves the cursor to it, and says why.
   * Everything a validation failure owes the person who hit it. */
  const fail = (input: HTMLInputElement | null, message: string): void => {
    error.textContent = message;
    if (!input) return;
    input.classList.add("is-invalid");
    input.setAttribute("aria-invalid", "true");
    input.setAttribute("aria-describedby", "setup-error");
    input.focus();
  };

  const clearInvalid = (): void => {
    error.textContent = "";
    for (const input of [nameInput, urlInput, durationInput]) {
      input.classList.remove("is-invalid");
      input.removeAttribute("aria-invalid");
      if (input !== durationInput) input.removeAttribute("aria-describedby");
    }
  };

  // Typing is the operator answering the complaint; the mark comes off then
  // rather than surviving until the next submit.
  for (const input of [nameInput, urlInput]) {
    input.addEventListener("input", () => {
      if (input.classList.contains("is-invalid")) clearInvalid();
    });
  }

  const submit = async () => {
    clearInvalid();
    const name = nameInput.value.trim();
    const url = urlInput.value.trim();

    if (!name) {
      fail(nameInput, "Give the study a name.");
      return;
    }
    if (!existing && !file && !url) {
      fail(urlInput, "Add an image or a URL to look at.");
      return;
    }

    let stimulus: Study["stimulus"];
    if (lockedImage) {
      stimulus = existing.stimulus;
    } else if (file) {
      const dimensions = await readImageDimensions(file);
      stimulus = {
        kind: "image",
        blob: file,
        width: dimensions.width,
        height: dimensions.height,
        name: file.name,
      };
    } else {
      if (!url) {
        fail(urlInput, "Add a URL to look at.");
        return;
      }
      const resolved = normaliseStimulusUrl(url);
      if ("problem" in resolved) {
        fail(urlInput, resolved.problem);
        return;
      }
      stimulus = { kind: "url", url: resolved.url };
    }

    const study: Study = {
      id: existing?.id ?? newId("study"),
      name,
      task: taskInput.value.trim(),
      stimulus,
      aois: existing?.aois ?? [],
      createdAt: existing?.createdAt ?? Date.now(),
      duration: Math.max(0, Number(durationInput.value) || 0),
    };

    await saveStudy(study);
    void showStudyList();
  };

  // Two columns, because the panel has two measures in it and they disagree:
  // the copy wants a reading measure and the controls want the full width.
  // Stacked, the display heading wrapped at ~380px with 60% of the panel empty
  // beside it, and then the form ran full-bleed underneath. Side by side, the
  // reading column *is* the measure, and the form owns the rest.
  const copy = el(
    "div",
    { class: "setup-copy" },
    // The panel opens the way an experiment opens on the site: eyebrow, then
    // the claim in display type, then how to work it.
    el(
      "p",
      { class: "label eyebrow" },
      existing ? "Edit study" : "Upload · Task · Thirteen clicks"
    ),
    el(
      "h2",
      { class: "t-display" },
      existing ? `Adjust “${existing.name}”.` : "See where they actually looked."
    ),
    el(
      "p",
      { class: "panel-lede" },
      existing
        ? lockedImage
          ? "Name, task and duration are safe to change mid-study. The stimulus image stays locked: every existing recording is aligned to it."
          : "Name, task, duration and the page URL can all change. Recordings already made keep the gaze they captured."
        : "Add the screen you want tested and the task you want done. Running a session calibrates to whoever is sitting there, records them looking, and writes the result to this browser."
    ),
    // What the tool can actually resolve, said where someone is deciding to use
    // it rather than only in the bench notes at the foot of the page.
    //
    // This column used to close the gap between itself and the taller controls
    // beside it by stretching the lede's box: `flex: 1 0 auto` gave a 78px
    // paragraph a 252px box, so the panel's own most-read column opened with
    // 174px of cream between the lede and the guarantee under it — a paragraph
    // that looked like it had lost its second half. Stretching a box is not a
    // way to have something to say. This is: the resolution claim is the one
    // fact that decides whether a study is worth running at all, and someone
    // reading it here rather than 1,500px further down is someone who will not
    // over-read their own heatmap.
    //
    // One sentence, and deliberately not the bench note's. This pair works the
    // way the privacy pair already does — the panel states the claim, the note
    // at the foot of the page explains it — so the numbers that qualify it (50
    // to 120 pixels, why it is a component-sized fact) stay down there rather
    // than being restated 200px above themselves. The note also has to keep
    // carrying it: on a return visit this panel is collapsed to a button, and
    // the bench notes are then the only place the caveat is made at all.
    existing
      ? null
      : el(
          "p",
          { class: "panel-lede setup-accuracy" },
          "Gaze lands within 2 to 4 degrees of visual angle — enough to tell you which block someone read, never which word."
        ),
    // At the foot of the copy column, which is where the imbalance was: the copy
    // ran out at the end of the lede while the controls column carried on
    // through the drop zone, four fields and a submit. The guarantee also reads
    // better here than as the tail of a paragraph — a standing claim with a lock
    // beside it, on the screen where someone decides whether to upload their
    // client's unreleased wireframe.
    el(
      "p",
      { class: "privacy-note setup-privacy" },
      el("span", { class: "lock-glyph", "aria-hidden": "true" }),
      "Nothing you upload and no frame of video leaves this machine. There is no server: studies, recordings and the image itself live in this browser's own storage."
    )
  );

  const controls = el(
    "div",
    { class: "setup-controls" },
    existing ? null : dropZone,
    existing ? null : scaleNote,
    // Two columns rather than auto-fit's three: at three, Duration was left
    // alone on a row of its own with two thirds of it empty directly above the
    // submit button. Paired with the URL field, the grid closes.
    el(
      "div",
      { class: "field-grid" },
      field("Study name", nameInput),
      field("Task prompt", taskInput),
      lockedImage ? null : field(existing ? "Page URL" : "Or a page URL", urlInput),
      field(
        "Duration",
        durationField,
        el("span", { class: "field-hint", id: "duration-hint" }, "0 = until the moderator stops")
      )
    ),
    error,
    el(
      "div",
      { class: "form-actions" },
      el("button", { class: "btn btn-primary", type: "submit" }, existing ? "Save changes" : "Create study"),
      existing || onCollapse
        ? el(
            "button",
            {
              class: "btn btn-ghost",
              type: "button",
              onclick: onCollapse ?? (() => void showStudyList()),
            },
            "Cancel"
          )
        : null
    )
  );

  const form = el(
    "form",
    {
      class: "panel setup-panel",
      // A short form should submit on Enter; the button alone does not give
      // the fields that behavior.
      onsubmit: (event: Event) => {
        event.preventDefault();
        void submit();
      },
    },
    copy,
    controls
  );

  // Editing arrives from a button further down the page, and the create form
  // now arrives from a button too; either way the operator asked for this form,
  // so put the cursor in its first field rather than making them reach for it.
  // Not in the zero state, where the form is simply what the page is.
  if (existing || onCollapse) queueMicrotask(() => nameInput.focus());

  return form;
}

function field(label: string, input: HTMLElement, hint?: HTMLElement): HTMLElement {
  return el("label", { class: "field" }, el("span", {}, label), input, hint ?? null);
}


// Bench notes, in the site's ledger voice: a label rail on the left and the
// note beside it. Shorter than a /lab/<slug> page's set, because this one has
// to earn its space above a working tool rather than below a finished one.
const BENCH_NOTES: { k: string; v: string }[] = [
  {
    k: "What the error means",
    v: "Expect 2 to 4 degrees of visual angle, which is 50 to 120 pixels at a normal viewing distance. That is enough to tell you which block someone read and never enough to tell you which word. Every number this gives you should be read at the size of a component, not a line of copy.",
  },
  {
    k: "Calibrate every participant",
    v: "Calibration is per person and per seating position, and it is cached for one sitting only. A stale calibration does not announce itself: it just quietly returns gaze that is wrong by a consistent amount, which looks exactly like data.",
  },
  {
    k: "Give a real task",
    v: "A free-viewing heatmap mostly shows you where the biggest image is. A task-driven one shows you whether the interface works. The task is the part of the setup that decides whether the output is worth anything.",
  },
  {
    k: "It stays on this machine",
    v: "The camera feed is read frame by frame in the page and never recorded or sent anywhere. Studies and recordings live in this browser's own storage. There is no server, which is also why clearing site data deletes everything.",
  },
];

function footer(): HTMLElement {
  const notes = el("dl", { class: "notes" });
  for (const note of BENCH_NOTES) {
    notes.append(
      el("div", { class: "note-row" }, el("dt", { class: "label" }, note.k), el("dd", {}, note.v))
    );
  }

  return el(
    "footer",
    { class: "app-footer" },
    el(
      "div",
      { class: "container" },
      el("h2", { class: "t-display notes-title" }, "Bench notes"),
      notes,
      el(
        "p",
        { class: "footer-credit" },
        el("a", { class: "arrow-link", href: LAB_URL }, "EXP-038 in the Lab at ryankm.com", el("span", { "aria-hidden": "true" }, "→"))
      )
    )
  );
}

/** A live page can refuse to be embedded, and a cross-origin frame does not
 * report that in any way a script can read: it simply arrives empty. Finding
 * out at the point a participant is already calibrated and looking wastes the
 * one thing a session cannot get back, which is that person's first pass over
 * the screen. So the operator sees the framed page here, before anyone is
 * calibrated, and can decide it is broken with their own eyes. */
function stimulusCheck(study: Study): HTMLElement | null {
  if (study.stimulus.kind !== "url") return null;

  // The stored address is checked again here, not just at the form: a study
  // saved before the form validated anything still has raw text in this
  // field, and an unchecked iframe src turns that into this app's own 404
  // dressed up as the participant's page.
  const resolved = normaliseStimulusUrl(study.stimulus.url);
  if ("problem" in resolved) {
    return el(
      "div",
      { class: "stimulus-check" },
      el("p", { class: "label" }, "The page they will see"),
      el(
        "p",
        { class: "note error" },
        `This study's address (“${study.stimulus.url}”) cannot be loaded. ${resolved.problem} Delete this study and create it again, or run it against a screenshot.`
      )
    );
  }

  // A salvageable address (one that only lacked its scheme) is written back,
  // so the repair happens once instead of on every render.
  if (resolved.url !== study.stimulus.url) {
    study.stimulus.url = resolved.url;
    void saveStudy(study);
  }

  return el(
    "div",
    { class: "stimulus-check" },
    el("p", { class: "label" }, "The page they will see"),
    el(
      "div",
      { class: "stimulus-frame" },
      el("iframe", {
        src: resolved.url,
        title: "Stimulus preview",
        loading: "lazy",
        referrerpolicy: "no-referrer",
      })
    ),
    el(
      "p",
      { class: "note" },
      "If that box is blank or shows an error, the site refuses to be embedded and no session will work against it. Take a screenshot of the page and run the study against the image instead, which is the more rigorous choice anyway: every participant then sees byte-identical content."
    )
  );
}

// --- Session flow --------------------------------------------------------

async function runSession(study: Study): Promise<void> {
  clear(app);
  blockWatchers = [];

  // role="status" makes this a polite live region: the camera states below are
  // exactly what a non-sighted operator needs to hear to know it is working.
  const status = el("p", { class: "muted session-status", role: "status" }, "Starting camera…");

  /**
   * The one place the status line is written.
   *
   * Two things used to go wrong here. The severity was invisible — "No face
   * detected. Check your lighting and framing." rendered in exactly the same
   * grey at the same size as "Ready to calibrate.", so the one blocking state
   * in the flow looked like the ready state. And the tracker listener kept a
   * `lastMessage` closure to avoid rewriting the live region every frame, while
   * other code paths wrote `status.textContent` directly and never told it: a
   * cancelled calibration left "Calibration cancelled." on screen permanently,
   * because the face state had not transitioned since. Both are fixed by
   * giving the line a single owner, and by keeping the outcome of the last
   * attempt out of it entirely (see setOutcome).
   */
  let lastStatusMessage = "";
  const setStatus = (message: string, tone: "info" | "ok" | "warn" | "bad" = "info"): void => {
    lastStatusMessage = message;
    status.textContent = message;
    status.classList.toggle("is-ok", tone === "ok");
    status.classList.toggle("is-warn", tone === "warn");
    status.classList.toggle("error", tone === "bad");
  };

  /**
   * What happened last time, which is a different kind of fact from what the
   * camera is doing now.
   *
   * These used to be written over the live status line, which put the two in
   * direct conflict: either the outcome was wiped by the next tracked frame, or
   * — as shipped — the line stopped updating and reported a cancelled
   * calibration for as long as the operator stood there, at exactly the moment
   * they were deciding whether the rig was working. They are separate lines
   * now, and each says only what it knows.
   */
  const outcomeLine = el("p", { class: "note session-outcome", role: "status", hidden: true });
  const setOutcome = (message: string, tone: "warn" | "bad" = "warn"): void => {
    outcomeLine.textContent = message;
    outcomeLine.classList.toggle("note-warn", tone === "warn");
    outcomeLine.classList.toggle("note-bad", tone === "bad");
    outcomeLine.hidden = false;
  };
  const clearOutcome = (): void => {
    outcomeLine.hidden = true;
    outcomeLine.textContent = "";
  };

  // Something to aim at. The preview is the largest element on the screen and
  // offered no guidance at all: no head-position guide, no distance cue, and
  // the only feedback a grey sentence underneath it. The oval takes the
  // face-detected state, so a participant can fix their own framing.
  const guide = el(
    "div",
    { class: "camera-guide", "aria-hidden": "true" },
    el("span", { class: "camera-guide-oval" }),
    el("span", { class: "camera-guide-hint" }, "Line your face up here")
  );
  const preview = el("div", { class: "camera-preview" }, engine.tracker.video, guide);
  // Said on the screen where the camera light is actually on, which is when a
  // participant asks. It was on the setup panel and in the bench notes —
  // everywhere except here.
  const privacyNote = el(
    "p",
    { class: "privacy-note" },
    el("span", { class: "lock-glyph", "aria-hidden": "true" }),
    "Nothing leaves this machine. The camera is read frame by frame in this page; no video is recorded, uploaded, or sent anywhere."
  );
  const stored = loadStoredCalibration();

  const participantInput = el("input", {
    class: "input",
    placeholder: `P${Date.now().toString().slice(-4)}`,
    value: stored?.participant ?? "",
  });
  const gazeDotToggle = el("input", { type: "checkbox" });
  const actions = el("div", { class: "session-actions" });

  /**
   * The controls that only mean anything once a camera is running.
   *
   * Both dead ends on this screen used to leave these live and pointless. The
   * unloadable-URL branch removed them; the camera-failure branch did not, so a
   * red "Could not start the camera" sat sandwiched between a working
   * participant-label field and a working "show live gaze dot" checkbox, neither
   * of which could lead anywhere. One helper, called by both — hiding rather
   * than removing, because a retry that succeeds has to be able to put them back
   * with whatever was already typed into them.
   */
  const sessionControls = el(
    "div",
    { class: "session-controls" },
    field("Participant label", participantInput),
    el(
      "label",
      { class: "checkbox" },
      gazeDotToggle,
      el("span", {}, "Show live gaze dot (demo mode, distracting for real studies)")
    )
  );
  const showSessionControls = (visible: boolean): void => {
    sessionControls.hidden = !visible;
  };

  /**
   * The session screen, in two columns where there is room for two.
   *
   * This is the screen a participant sits in front of while framing their face,
   * and it used to be a ~620px card about 330px tall floating in a 1440×900
   * window with 450px of empty cream underneath it. The camera preview — the
   * element that decides whether calibration will work at all — got a fraction
   * of the available space, and the head-position oval inside it was
   * correspondingly small. The preview and its framing guide now take a column
   * of their own and everything the moderator reads or types takes the other.
   * Below 1100px it stacks back into the single card it was.
   */
  const mediaColumn = el("div", { class: "session-media" }, preview, privacyNote);
  const detailColumn = el(
    "div",
    { class: "session-detail" },
    el("h2", {}, study.name),
    el("p", { class: "muted" }, study.task || "No task set"),
    stimulusCheck(study),
    status,
    sessionControls,
    outcomeLine,
    actions
  );
  const panel = el("section", { class: "panel session-panel" }, mediaColumn, detailColumn);
  /** Collapses to one column when there is no picture to give one to. */
  const setMediaVisible = (visible: boolean): void => {
    panel.classList.toggle("has-no-media", !visible);
  };

  app.append(
    // The session screen replaces the page, so it carries the site bar with it
    // rather than leaving a participant on an unbranded card.
    appBar(),
    el(
      "div",
      // The session panel is its own measure — one card at small sizes, two
      // columns once the preview has room to be worth looking at — and it is
      // centred inside the site's 1840px shell rather than left-aligned in it.
      // Results keeps the wide shell.
      { class: "container screen screen-session" },
      el(
        "div",
        { class: "screen-head" },
        el("button", { class: "btn btn-ghost", type: "button", onclick: () => void showStudyList() }, "← Studies")
      ),
      panel
    )
  );

  // stimulusCheck above has already repaired a salvageable stored address; if
  // the address is still unloadable, running the session would calibrate a
  // participant and then show them a 404, so the camera never starts.
  if (study.stimulus.kind === "url" && "problem" in normaliseStimulusUrl(study.stimulus.url)) {
    setStatus("This study's address cannot be loaded, so a session cannot run against it.", "bad");
    // The rest of the session apparatus would only dress up a dead end. This
    // one is permanent — no retry can fix a stored address — so the camera
    // frame goes with it.
    mediaColumn.remove();
    setMediaVisible(false);
    showSessionControls(false);
    return;
  }

  /** Starts the camera and arms the screen, or explains why it could not and
   * offers the one thing worth doing next. The old version of this printed
   * "Could not start the camera: undefined" whenever the loader rejected with
   * a non-Error, and then returned before rendering any action at all — the
   * operator was left with a dead teal rectangle and no way forward. */
  const startCamera = async (): Promise<void> => {
    clear(actions);
    clearOutcome();
    setStatus("Starting camera…");
    preview.classList.remove("is-dead");
    setMediaVisible(true);
    showSessionControls(true);

    try {
      await engine.start((message) => setStatus(message));
    } catch (err) {
      setStatus(describeCameraFailure(err), "bad");
      // A camera that never started has no picture to show. Hiding the frame
      // beats leaving a dead solid-teal rectangle on the screen; it comes back
      // if a retry succeeds, and so do the controls.
      preview.classList.add("is-dead");
      setMediaVisible(false);
      showSessionControls(false);
      clear(actions);
      actions.append(
        // One way forward, not two that read as the same thing twice. "Reload
        // the page" was a second equal-weight button for an action "Try again"
        // already covers in every case where it would have helped.
        el(
          "button",
          { class: "btn btn-primary", type: "button", onclick: () => void startCamera() },
          "Try again"
        ),
        el("button", { class: "btn btn-ghost", type: "button", onclick: () => void showStudyList() }, "Back to studies")
      );
      return;
    }

    setStatus("Camera running. Sit about an arm's length away, square to the screen.");
    watchTracker();
    renderActions();
  };

  /** Puts the live camera readout back on the screen. Attached when the camera
   * starts and again after a recording hands control back, because the
   * recording stage takes the listener with it — without the re-attach, the
   * one live indicator of whether the rig is working stays dead for the rest
   * of the sitting, and the Calibrate gate freezes with it. */
  function watchTracker(): void {
    releaseStatusListener?.();
    releaseStatusListener = engine.onStatus(onTrackerStatus);
  }

  function applyTrackerState(s: TrackerStatus): void {
    settledUsable = s.usable;
    // The oval is the peripheral read — a participant fixing their own framing
    // is looking at the picture, not at the sentence under it.
    preview.classList.toggle("is-tracking", s.usable);
    preview.classList.toggle("is-lost", !s.faceVisible);
    updateCalibrateGate();

    // Only rewrite the live region when the sentence actually changes: assigning
    // on every frame would have a screen reader narrating the fps counter. The
    // rate is quantised rather than left out of the comparison, because a figure
    // read off the first tracked frame and then frozen would advertise a still-
    // settling estimate as the session's throughput for the rest of the sitting.
    const message = !s.faceVisible
      ? "No face detected. Check your lighting and framing."
      : !s.usable
        ? "Face detected, but turned too far or too close to the edge of frame."
        : `Tracking at ${Math.round(s.fps / 5) * 5} fps. Ready to calibrate.`;
    if (message === lastStatusMessage) return;
    setStatus(message, s.usable ? "ok" : "warn");
  }

  /**
   * Holds the readout steady through a blink.
   *
   * `usable` drops for every blink and for the settle frames after it, so a
   * screen wired straight to it flashes the guide amber, rewrites the status
   * sentence, and — now that the Calibrate button is gated on the same fact —
   * disables the primary action, several times a minute. Good news applies at
   * once; bad news has to persist for most of a second before it is shown, and
   * whatever is true when the wait ends is what gets shown.
   */
  const BLINK_HOLD_MS = 800;
  let lastRawStatus: TrackerStatus | null = null;
  let holdTimer = 0;

  function onTrackerStatus(s: TrackerStatus): void {
    lastRawStatus = s;
    if (s.usable) {
      window.clearTimeout(holdTimer);
      holdTimer = 0;
      applyTrackerState(s);
      return;
    }
    // Already reporting a problem: keep reporting the current one, so "no face"
    // and "face turned away" swap without waiting on each other.
    if (!settledUsable) {
      applyTrackerState(s);
      return;
    }
    if (holdTimer) return;
    holdTimer = window.setTimeout(() => {
      holdTimer = 0;
      if (lastRawStatus) applyTrackerState(lastRawStatus);
    }, BLINK_HOLD_MS);
  }

  const beginRecording = async (validationError: number | null) => {
    // The recording screen has its own face-lost readout; this one would only
    // talk over it.
    releaseStatusListener?.();
    releaseStatusListener = null;
    // Nothing is watching the tracker while the recording stage owns the
    // screen, so the gate must not carry a stale "yes" back out of it.
    window.clearTimeout(holdTimer);
    holdTimer = 0;
    settledUsable = false;
    updateCalibrateGate();
    const outcome = await runRecording(app, {
      study,
      engine,
      participant: participantInput.value.trim() || `P${Date.now().toString().slice(-4)}`,
      validationError,
      showGazeDot: gazeDotToggle.checked,
    });

    if (outcome.status === "saved") {
      engine.stop();
      await openResults(study);
      return;
    }

    if (outcome.status === "empty") {
      // The session ran to the end but no usable gaze arrived. Returning to
      // the study list silently here would spend a participant with no
      // explanation — say what happened, and keep the camera on so the
      // operator can recalibrate immediately.
      setOutcome(
        "The recording captured no usable gaze, so nothing was saved. Check lighting and framing, then recalibrate.",
        "bad"
      );
      watchTracker();
      renderActions();
      return;
    }

    engine.stop();
    void showStudyList();
  };

  const calibrateThenRecord = async () => {
    clear(actions);
    clearOutcome();

    let outcome: Awaited<ReturnType<typeof runCalibration>>;
    try {
      outcome = await runCalibration(engine, app);
    } catch (err) {
      // A participant who has just clicked eighteen dots has to be told what
      // happened and offered the retry. This used to be an unhandled rejection:
      // the real message ("Not enough calibration data (0 of 66 samples)") went
      // to the console, the overlay unmounted, and the session panel was left
      // with an empty actions container — no button, no error, thirty seconds
      // of someone's work gone with the app looking simply broken.
      setOutcome(
        err instanceof Error && err.message
          ? err.message
          : "Calibration failed. Check lighting and framing, then try again.",
        "bad"
      );
      renderActions();
      return;
    }

    if (outcome.cancelled) {
      setOutcome("Calibration cancelled. Nothing was saved.");
      renderActions();
      return;
    }

    const accuracy = describeAccuracy(outcome.validationError);
    const model = engine.getModel();
    if (model) {
      storeCalibration({
        basis: FEATURE_BASIS_VERSION,
        model: serialiseModel(model),
        validationError: outcome.validationError,
        savedAt: Date.now(),
        participant: participantInput.value.trim(),
      });
    }

    clear(actions);
    actions.append(
      el(
        "div",
        { class: `accuracy accuracy-${accuracy.grade}`, role: "status" },
        el("strong", {}, accuracy.label),
        el("p", { class: "muted" }, accuracy.detail)
      ),
      el(
        "button",
        {
          class: "btn btn-primary",
          type: "button",
          onclick: () => void beginRecording(outcome.validationError),
        },
        "Start recording"
      ),
      el("button", { class: "btn", type: "button", onclick: () => void calibrateThenRecord() }, "Recalibrate")
    );
  };

  /** Whether the tracker is returning gaze it would fit a model on, held
   * steady through blinks by onTrackerStatus. Calibration cannot succeed
   * without it, so the button says so rather than letting someone find out at
   * the end of eighteen clicks. */
  let settledUsable = false;
  let calibrateBtn: HTMLButtonElement | null = null;

  function updateCalibrateGate(): void {
    if (!calibrateBtn) return;
    calibrateBtn.disabled = !settledUsable;
    if (settledUsable) calibrateBtn.removeAttribute("title");
    else {
      calibrateBtn.setAttribute(
        "title",
        "Waiting for a clear view of the face — calibration cannot succeed until the camera is tracking"
      );
    }
  }

  function renderActions(): void {
    clear(actions);
    calibrateBtn = el(
      "button",
      { class: "btn btn-primary", type: "button", onclick: () => void calibrateThenRecord() },
      "Calibrate"
    );
    actions.append(calibrateBtn);
    updateCalibrateGate();

    if (stored) {
      const age = Math.round((Date.now() - stored.savedAt) / 60000);
      actions.append(
        el(
          "button",
          {
            class: "btn",
            type: "button",
            onclick: () => {
              engine.setModel(deserialiseModel(stored.model));
              void beginRecording(stored.validationError);
            },
          },
          `Reuse calibration (${age}m old)`
        )
      );
    }
  }

  void startCamera();
}

/**
 * A sentence an operator can act on, whatever the camera stack threw.
 *
 * MediaPipe's loader can reject with something that is not an Error at all,
 * and interpolating `.message` off that produced the literal string
 * "Could not start the camera: undefined" — the worst kind of error message,
 * because it tells you only that the code did not expect to be here.
 */
function describeCameraFailure(err: unknown): string {
  if (err instanceof DOMException) {
    if (err.name === "NotAllowedError") {
      return "Camera access was blocked. Allow the camera in your browser settings, then try again.";
    }
    if (err.name === "NotFoundError" || err.name === "OverconstrainedError") {
      return "No camera was found. Connect one — or check that another profile has not claimed it — then try again.";
    }
    if (err.name === "NotReadableError") {
      return "The camera is already in use by another app or tab. Close that one, then try again.";
    }
  }

  const detail =
    err instanceof Error && err.message
      ? err.message
      : typeof err === "string" && err
        ? err
        : "";
  return detail
    ? `Could not start the camera: ${detail}`
    : "Could not start the camera, and the browser gave no reason. This is usually a blocked permission, a camera another app is holding, or the face model failing to download.";
}

function readImageDimensions(file: File): Promise<{ width: number; height: number }> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve({ width: 0, height: 0 });
    };
    img.src = url;
  });
}

void showStudyList();
