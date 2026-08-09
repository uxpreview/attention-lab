import "./styles.css";

import { deleteStudy, listStudies, newId, saveStudy } from "./data/store";
import { normaliseStimulusUrl } from "./data/stimulusUrl";
import type { Study } from "./data/types";
import { GazeEngine } from "./tracker/gaze";
import { deserialiseModel, serialiseModel } from "./tracker/regression";
import { describeAccuracy, runCalibration } from "./ui/calibration";
import { clear, el } from "./ui/dom";
import { runRecording } from "./ui/record";
import { renderResults } from "./ui/results";

const app = document.getElementById("app") as HTMLElement;
const engine = new GazeEngine();

/** Calibration is per-person, but a repeat participant in the same sitting can
 * reuse theirs. Session storage, not local: a stale calibration from yesterday
 * is worse than no calibration, because it fails silently. */
const CALIBRATION_KEY = "eyetrack.calibration";

interface StoredCalibration {
  model: ReturnType<typeof serialiseModel>;
  validationError: number | null;
  savedAt: number;
  participant: string;
}

function loadStoredCalibration(): StoredCalibration | null {
  try {
    const raw = sessionStorage.getItem(CALIBRATION_KEY);
    return raw ? (JSON.parse(raw) as StoredCalibration) : null;
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

const SITE_URL = "https://ryankm.com";
const LAB_URL = `${SITE_URL}/lab`;

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

function experimentHead(): HTMLElement {
  // A bar rather than a breadcrumb. A breadcrumb is a same-origin device: it
  // says "you are inside this section", and here the parent link leaves the
  // origin, so it would be an exit dressed as a way back up. The wordmark and
  // one route home are honest about being somewhere else, and they give a
  // visitor who arrived from a search result something to arrive at.
  const bar = el(
    "div",
    { class: "site-bar" },
    el(
      "div",
      { class: "container bar-inner" },
      el("a", { class: "wordmark", href: SITE_URL }, "Ryan McCarty", el("span", { class: "dot" }, ".")),
      el("a", { class: "bar-back", href: LAB_URL }, "Back to the Lab", el("span", { "aria-hidden": "true" }, "↗"))
    )
  );

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
      el(
        "p",
        { class: "t-lede" },
        "Point a webcam at a wireframe and find out where people actually look. Upload a screen, give someone a task, and this rebuilds their gaze from the iris after thirteen calibration clicks, then hands back a heatmap, a scanpath, and attention numbers per region. Accurate to a block rather than a word, which is the honest limit of doing this without a lab rig, and enough to settle most arguments about hierarchy."
      )
    )
  );
}

// --- Study list ----------------------------------------------------------

async function showStudyList(): Promise<void> {
  // The study list is the one screen that never needs the camera, so releasing
  // it here covers every way of leaving a session — including abandoning
  // calibration, where nothing else would turn the camera light off.
  engine.stop();

  clear(app);
  const studies = await listStudies();

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
            : "Add a wireframe or a URL to start collecting attention data."
        )
      )
    );
  } else {
    for (const study of studies) {
      list.append(studyCard(study));
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
      isHandheld() ? null : newStudyForm(),
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

function studyCard(study: Study): HTMLElement {
  const thumb = el("div", { class: "study-thumb" });
  if (study.stimulus.kind === "image") {
    const url = URL.createObjectURL(study.stimulus.blob);
    thumb.append(el("img", { src: url, alt: "" }));
    // Card lifetime is the screen's lifetime; release when it leaves the DOM.
    setTimeout(() => URL.revokeObjectURL(url), 60_000);
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
      el(
        "p",
        { class: "study-meta" },
        study.stimulus.kind === "image"
          ? `${study.stimulus.width}×${study.stimulus.height} image`
          : study.stimulus.url,
        " · ",
        study.duration > 0 ? `${study.duration}s` : "manual stop"
      )
    ),
    el(
      "div",
      { class: "study-actions" },
      // On a handheld the run button is disabled rather than hidden: a control
      // that vanishes leaves you wondering whether the tool is broken, and the
      // notice above the list has already given the reason.
      isHandheld()
        ? el(
            "button",
            {
              class: "btn btn-primary",
              type: "button",
              disabled: true,
              title: "Sessions need a laptop or a desktop",
            },
            "Run session"
          )
        : el(
            "button",
            { class: "btn btn-primary", type: "button", onclick: () => void runSession(study) },
            "Run session"
          ),
      el(
        "button",
        {
          class: "btn",
          type: "button",
          onclick: () => void renderResults(app, study, () => void showStudyList()),
        },
        "Results"
      ),
      el(
        "button",
        {
          class: "btn btn-ghost btn-small",
          type: "button",
          onclick: async () => {
            if (!confirm(`Delete "${study.name}" and all of its recordings?`)) return;
            await deleteStudy(study.id);
            void showStudyList();
          },
        },
        "Delete"
      )
    )
  );
}

function newStudyForm(): HTMLElement {
  const nameInput = el("input", { class: "input", placeholder: "Checkout wireframe v3" });
  const taskInput = el("input", {
    class: "input",
    placeholder: "Find where you would enter a discount code",
  });
  const urlInput = el("input", { class: "input", placeholder: "https://example.com" });
  const durationInput = el("input", { class: "input", type: "number", value: "15", min: "0" });
  const fileInput = el("input", { class: "file-input", type: "file", accept: "image/*" });

  const dropZone = el(
    "label",
    { class: "drop-zone" },
    fileInput,
    el("span", { class: "drop-title" }, "Drop a wireframe or screenshot"),
    el("span", { class: "muted" }, "PNG or JPG · or use a URL below")
  );

  let file: File | null = null;
  const setFile = (next: File | null) => {
    file = next;
    dropZone.classList.toggle("has-file", next !== null);
    const title = dropZone.querySelector(".drop-title") as HTMLElement;
    title.textContent = next ? next.name : "Drop a wireframe or screenshot";
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

  const error = el("p", { class: "error" });

  const submit = async () => {
    error.textContent = "";
    const name = nameInput.value.trim();
    const url = urlInput.value.trim();

    if (!name) {
      error.textContent = "Give the study a name.";
      return;
    }
    if (!file && !url) {
      error.textContent = "Add an image or a URL to look at.";
      return;
    }

    let stimulus: Study["stimulus"];
    if (file) {
      const dimensions = await readImageDimensions(file);
      stimulus = {
        kind: "image",
        blob: file,
        width: dimensions.width,
        height: dimensions.height,
        name: file.name,
      };
    } else {
      const resolved = normaliseStimulusUrl(url);
      if ("problem" in resolved) {
        error.textContent = resolved.problem;
        return;
      }
      stimulus = { kind: "url", url: resolved.url };
    }

    const study: Study = {
      id: newId("study"),
      name,
      task: taskInput.value.trim(),
      stimulus,
      aois: [],
      createdAt: Date.now(),
      duration: Math.max(0, Number(durationInput.value) || 0),
    };

    await saveStudy(study);
    void showStudyList();
  };

  return el(
    "section",
    { class: "panel" },
    // The panel opens the way an experiment opens on the site: eyebrow, then
    // the claim in display type, then how to work it.
    el("p", { class: "label eyebrow" }, "Upload · Task · Thirteen clicks"),
    el("h2", { class: "t-display" }, "See where they actually looked."),
    el(
      "p",
      { class: "panel-lede" },
      "Add the screen you want tested and the task you want done. Running a session calibrates to whoever is sitting there, records them looking, and writes the result to this browser. Nothing you upload and no frame of video ever leaves this machine."
    ),
    dropZone,
    el(
      "div",
      { class: "field-grid" },
      field("Study name", nameInput),
      field("Task prompt", taskInput),
      field("Or a page URL", urlInput),
      field("Duration (seconds, 0 = manual)", durationInput)
    ),
    error,
    el("button", { class: "btn btn-primary", type: "button", onclick: () => void submit() }, "Create study")
  );
}

function field(label: string, input: HTMLElement): HTMLElement {
  return el("label", { class: "field" }, el("span", {}, label), input);
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

  const status = el("p", { class: "muted" }, "Starting camera…");
  const preview = el("div", { class: "camera-preview" }, engine.tracker.video);
  const stored = loadStoredCalibration();

  const participantInput = el("input", {
    class: "input",
    placeholder: `P${Date.now().toString().slice(-4)}`,
    value: stored?.participant ?? "",
  });
  const gazeDotToggle = el("input", { type: "checkbox" });

  const panel = el(
    "section",
    { class: "panel session-panel" },
    el("h2", {}, study.name),
    el("p", { class: "muted" }, study.task || "No task set"),
    stimulusCheck(study),
    preview,
    status,
    field("Participant label", participantInput),
    el(
      "label",
      { class: "checkbox" },
      gazeDotToggle,
      el("span", {}, "Show live gaze dot (demo mode, distracting for real studies)")
    ),
    el("div", { class: "session-actions" })
  );

  const actions = panel.querySelector(".session-actions") as HTMLElement;
  app.append(
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
    status.textContent = "This study's address cannot be loaded, so a session cannot run against it.";
    status.classList.add("error");
    // The rest of the session apparatus would only dress up a dead end.
    panel.querySelector(".camera-preview")?.remove();
    panel.querySelectorAll(".field, .checkbox").forEach((node) => node.remove());
    return;
  }

  try {
    await engine.start((message) => {
      status.textContent = message;
    });
  } catch (err) {
    status.textContent =
      err instanceof DOMException && err.name === "NotAllowedError"
        ? "Camera access was blocked. Allow the camera in your browser settings and reload."
        : `Could not start the camera: ${(err as Error).message}`;
    status.classList.add("error");
    return;
  }

  status.textContent = "Camera running. Sit about an arm's length away, square to the screen.";

  const unsubscribe = engine.onStatus((s) => {
    if (!s.faceVisible) {
      status.textContent = "No face detected. Check your lighting and framing.";
    } else if (!s.usable) {
      status.textContent = "Face detected, but turned too far or too close to the edge of frame.";
    } else {
      status.textContent = `Tracking at ${Math.round(s.fps)} fps. Ready to calibrate.`;
    }
  });

  const beginRecording = async (validationError: number | null) => {
    unsubscribe();
    const recording = await runRecording(app, {
      study,
      engine,
      participant: participantInput.value.trim() || `P${Date.now().toString().slice(-4)}`,
      validationError,
      showGazeDot: gazeDotToggle.checked,
    });

    engine.stop();

    if (recording) {
      await renderResults(app, study, () => void showStudyList());
    } else {
      void showStudyList();
    }
  };

  const calibrateThenRecord = async () => {
    clear(actions);
    const outcome = await runCalibration(engine, app);

    if (outcome.cancelled) {
      status.textContent = "Calibration cancelled.";
      renderActions();
      return;
    }

    const accuracy = describeAccuracy(outcome.validationError);
    const model = engine.getModel();
    if (model) {
      storeCalibration({
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
        { class: `accuracy accuracy-${accuracy.grade}` },
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

  function renderActions(): void {
    clear(actions);
    actions.append(
      el(
        "button",
        { class: "btn btn-primary", type: "button", onclick: () => void calibrateThenRecord() },
        "Calibrate"
      )
    );

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

  renderActions();
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
