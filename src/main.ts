import "./styles.css";

import { deleteStudy, listStudies, newId, saveStudy } from "./data/store";
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

// --- Study list ----------------------------------------------------------

async function showStudyList(): Promise<void> {
  // The study list is the one screen that never needs the camera, so releasing
  // it here covers every way of leaving a session — including abandoning
  // calibration, where nothing else would turn the camera light off.
  engine.stop();

  clear(app);
  const studies = await listStudies();

  const header = el(
    "header",
    { class: "app-header" },
    el(
      "div",
      {},
      el("h1", {}, "Attention Lab"),
      el("p", { class: "muted" }, "Webcam eye tracking for wireframes and live pages")
    ),
    el(
      "div",
      { class: "header-side" },
      el("span", { class: "pill pill-quiet" }, "Everything stays in this browser"),
      // The way back to the section this belongs to. It is EXP-038 in the Lab
      // on ryankm.com, and a tool sitting on its own subdomain with no route
      // home reads like something abandoned there.
      el(
        "a",
        { class: "back-link", href: "https://ryankm.com/lab" },
        el("span", { "aria-hidden": "true" }, "←"),
        "EXP-038 in the Lab"
      )
    )
  );

  const list = el("section", { class: "study-list" });

  if (studies.length === 0) {
    list.append(
      el(
        "div",
        { class: "empty" },
        el("h3", {}, "No studies yet"),
        el("p", { class: "muted" }, "Add a wireframe or a URL to start collecting attention data.")
      )
    );
  } else {
    for (const study of studies) {
      list.append(studyCard(study));
    }
  }

  app.append(header, newStudyForm(), list, footer());
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
      el(
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
      stimulus = { kind: "url", url };
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
    el("h2", {}, "New study"),
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

function footer(): HTMLElement {
  return el(
    "footer",
    { class: "app-footer" },
    el(
      "p",
      { class: "muted" },
      "Webcam gaze estimation is approximate. Expect 2-4° of error, enough to tell which block someone read, not which word. No video ever leaves this device."
    ),
    el(
      "p",
      { class: "muted footer-credit" },
      "An experiment from the Lab at ",
      el("a", { class: "back-link back-link-plain", href: "https://ryankm.com/lab" }, "ryankm.com")
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
      "header",
      { class: "app-header" },
      el("button", { class: "btn btn-ghost", type: "button", onclick: () => void showStudyList() }, "← Studies")
    ),
    panel
  );

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
