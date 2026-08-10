import "./styles.css";

import { deleteStudy, listRecordings, listStudies, newId, saveStudy } from "./data/store";
import { normaliseStimulusUrl } from "./data/stimulusUrl";
import type { Study } from "./data/types";
import { FEATURE_BASIS_VERSION, FEATURE_DIM } from "./tracker/features";
import { GazeEngine, type TrackerStatus } from "./tracker/gaze";
import { deserialiseModel, isSerialisedModel, serialiseModel } from "./tracker/regression";
import { describeAccuracy, runCalibration } from "./ui/calibration";
import { appBar, LAB_URL } from "./ui/chrome";
import { clear, confirmButton, el } from "./ui/dom";
import { runRecording } from "./ui/record";
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
          },
        ];
      })
    )
  );

  const header = experimentHead();

  const list = el("section", { class: "study-list" });

  if (studies.length === 0) {
    list.append(
      el(
        "div",
        { class: "empty" },
        el("h3", {}, "No studies yet"),
        el(
          "p",
          { class: "muted" },
          isHandheld()
            ? "Nothing has been recorded in this browser. Results of a session run on another machine stay on that machine."
            : "Use the form above to add a wireframe or a URL — every session you run will collect here."
        )
      )
    );
  } else {
    for (const study of studies) {
      list.append(studyCard(study, stats.get(study.id) ?? { count: 0, lastRun: null }));
    }
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
      handheldNotice(),
      isHandheld() ? null : newStudyForm(editing),
      list
    )
  );

  app.append(header, body, footer());
}

/** Said once, at the top, before anyone builds a study they cannot run. */
function handheldNotice(): HTMLElement | null {
  if (!isHandheld()) return null;

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
}

/** Coarse on purpose: "3 days ago" is what a researcher scans for, and a
 * to-the-minute stamp on a list of ten studies is noise. */
function relativeDay(timestamp: number, now = Date.now()): string {
  const days = Math.floor((now - timestamp) / 86_400_000);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 30) return `${days} days ago`;
  const months = Math.round(days / 30);
  if (months < 12) return `${months} month${months === 1 ? "" : "s"} ago`;
  const years = Math.round(days / 365);
  return `${years} year${years === 1 ? "" : "s"} ago`;
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
    { class: "study-card" },
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
    el(
      "div",
      { class: "study-actions" },
      // Where a session cannot run the button is disabled rather than hidden: a
      // control that vanishes leaves you wondering whether the tool is broken,
      // and the title says which of the two reasons applies.
      runSessionButton(study),
      // Softened rather than disabled at zero: there is nothing to read yet,
      // but the results screen is also where regions are drawn, and a study
      // can usefully be marked up before the first participant sits down — the
      // screen keeps its region tools at zero recordings, so the tooltip is a
      // promise the app now actually honours.
      el(
        "button",
        {
          class: stats.count > 0 ? "btn" : "btn btn-ghost",
          type: "button",
          title: stats.count > 0 ? null : "No recordings yet — regions can still be drawn",
          onclick: () => void openResults(study),
        },
        "Results"
      ),
      el(
        "button",
        {
          class: "btn btn-ghost btn-small",
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

function runSessionButton(study: Study): HTMLButtonElement {
  const blocked = sessionBlockReason();
  if (blocked) {
    return el(
      "button",
      { class: "btn btn-primary", type: "button", disabled: true, title: blocked },
      "Run session"
    );
  }
  return el(
    "button",
    { class: "btn btn-primary", type: "button", onclick: () => void runSession(study) },
    "Run session"
  );
}

/** Results, with the route back and — when this machine can run one — the way
 * on to a session, so a study with no data yet is not a dead end. */
function openResults(study: Study): Promise<void> {
  return renderResults(app, study, () => void showStudyList(), {
    onRunSession: sessionBlockReason() ? null : () => void runSession(study),
    runBlockedReason: sessionBlockReason(),
  });
}

function studyDeleteButton(study: Study): HTMLButtonElement {
  const btn = confirmButton("Delete", "Really delete?", async () => {
    await deleteStudy(study.id);
    void showStudyList();
  });
  btn.setAttribute("aria-label", `Delete study ${study.name} and all of its recordings`);
  return btn;
}

/** The study form: creates when `existing` is null, edits it otherwise.
 * Editing keeps the study's id, recordings and regions; an image stimulus
 * stays locked because every recording is normalised against it. */
function newStudyForm(existing: Study | null = null): HTMLElement {
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

  let file: File | null = null;
  const setFile = (next: File | null) => {
    file = next;
    dropZone.classList.toggle("has-file", next !== null);
    dropTitle.textContent = next ? next.name : "Drop a wireframe or screenshot";
    // The uploaded image wins over the URL field. Saying so — and parking the
    // field — beats silently ignoring whatever was typed there.
    urlInput.disabled = next !== null;
    urlInput.placeholder = next ? "Using the uploaded image" : "https://example.com";
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
        : "Add the screen you want tested and the task you want done. Running a session calibrates to whoever is sitting there, records them looking, and writes the result to this browser. Nothing you upload and no frame of video ever leaves this machine."
    )
  );

  const controls = el(
    "div",
    { class: "setup-controls" },
    existing ? null : dropZone,
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
      existing
        ? el("button", { class: "btn btn-ghost", type: "button", onclick: () => void showStudyList() }, "Cancel")
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

  // Editing arrives from a button further down the page; bring the form and
  // its first field to the operator rather than making them scroll to it.
  if (existing) queueMicrotask(() => nameInput.focus());

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

  const panel = el(
    "section",
    { class: "panel session-panel" },
    el("h2", {}, study.name),
    el("p", { class: "muted" }, study.task || "No task set"),
    stimulusCheck(study),
    preview,
    privacyNote,
    status,
    field("Participant label", participantInput),
    el(
      "label",
      { class: "checkbox" },
      gazeDotToggle,
      el("span", {}, "Show live gaze dot (demo mode, distracting for real studies)")
    ),
    outcomeLine,
    actions
  );

  app.append(
    // The session screen replaces the page, so it carries the site bar with it
    // rather than leaving a participant on an unbranded card.
    appBar(),
    el(
      "div",
      // screen-narrow centres the whole column: the session panel is a single
      // 620px card, and left-aligning it inside the site's 1840px shell reads
      // as a mistake rather than a layout. Results keeps the wide shell.
      { class: "container screen screen-narrow" },
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
    // The rest of the session apparatus would only dress up a dead end.
    preview.remove();
    privacyNote.remove();
    panel.querySelectorAll(".field, .checkbox").forEach((node) => node.remove());
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

    try {
      await engine.start((message) => setStatus(message));
    } catch (err) {
      setStatus(describeCameraFailure(err), "bad");
      // A camera that never started has no picture to show. Hiding the frame
      // beats leaving a dead solid-teal rectangle on the screen; it comes back
      // if a retry succeeds.
      preview.classList.add("is-dead");
      clear(actions);
      actions.append(
        el(
          "button",
          { class: "btn btn-primary", type: "button", onclick: () => void startCamera() },
          "Try again"
        ),
        el(
          "button",
          { class: "btn", type: "button", onclick: () => window.location.reload() },
          "Reload the page"
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
