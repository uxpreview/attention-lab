import { newId, saveRecording } from "../data/store";
import type { Recording, Study } from "../data/types";
import type { GazeEngine } from "../tracker/gaze";
import { el, inertSiblings, nextFrame } from "./dom";

/**
 * The recording screen: shows the stimulus full-bleed and captures gaze.
 *
 * Gaze is stored in normalised stimulus coordinates rather than screen pixels,
 * so recordings from different screen sizes aggregate into one heatmap. That
 * conversion has to happen at capture time, because the stimulus rect is only
 * known while it is on screen.
 */

export interface RecordOptions {
  study: Study;
  engine: GazeEngine;
  participant: string;
  validationError: number | null;
  /** Draws a live gaze dot. Useful for demos, distracting for real studies. */
  showGazeDot: boolean;
}

/**
 * How a recording session ended. "empty" means the session ran to completion
 * but captured no usable gaze — the caller owes the operator an explanation,
 * because a participant's first pass over the screen has just been spent.
 */
export type RecordOutcome =
  | { status: "saved"; recording: Recording }
  | { status: "cancelled" }
  | { status: "empty" };

export async function runRecording(
  host: HTMLElement,
  options: RecordOptions
): Promise<RecordOutcome> {
  const { study, engine, participant } = options;

  const stage = el("div", {
    class: "record-stage",
    role: "dialog",
    "aria-modal": "true",
    "aria-label": "Recording",
    tabindex: "-1",
  });
  const stimulusLayer = el("div", { class: "record-stimulus" });
  const chrome = el("div", { class: "record-chrome" });
  const progressBar = el("div", { class: "record-progress" });
  stage.append(stimulusLayer);
  if (study.duration > 0) {
    // The fill sits in a dark track: peach alone is ~1.6:1 over a typical
    // light stimulus, and the time-remaining signal has to survive that.
    stage.append(el("div", { class: "record-progress-track" }, progressBar));
  }
  stage.append(chrome);
  host.append(stage);

  const restoreBackground = inertSiblings(host, stage);
  stage.focus();

  let objectUrl: string | null = null;
  let stimulusEl: HTMLElement;

  if (study.stimulus.kind === "image") {
    objectUrl = URL.createObjectURL(study.stimulus.blob);
    const img = el("img", { class: "record-image", src: objectUrl, alt: "" });
    stimulusLayer.append(img);
    stimulusEl = img;
    await new Promise<void>((resolve) => {
      if (img.complete) resolve();
      else img.addEventListener("load", () => resolve(), { once: true });
    });
  } else {
    const frame = el("iframe", {
      class: "record-frame",
      src: study.stimulus.url,
      referrerpolicy: "no-referrer",
    });
    stimulusLayer.append(frame);
    stimulusEl = frame;
  }

  // Task prompt first: without a task, participants free-view, and free-viewing
  // heatmaps mostly show you where the biggest picture is.
  const proceed = await showTaskPrompt(chrome, study);
  if (!proceed) {
    restoreBackground();
    stage.remove();
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    return { status: "cancelled" };
  }

  await nextFrame();
  // Cached rather than measured per sample — one layout read per recording,
  // not thirty per second. But a resize, zoom, or scroll mid-recording moves
  // the stimulus under the cached rect and silently corrupts every sample
  // after it, so those events re-measure: one layout read per event.
  let rect = stimulusEl.getBoundingClientRect();
  const remeasure = () => {
    rect = stimulusEl.getBoundingClientRect();
  };
  window.addEventListener("resize", remeasure);
  // Capture phase, because scroll events do not bubble from inner containers.
  window.addEventListener("scroll", remeasure, true);

  const points: Recording["points"] = [];
  let firstTimestamp: number | null = null;
  let insideCount = 0;
  let totalCount = 0;

  const dot = options.showGazeDot ? el("div", { class: "gaze-dot" }) : null;
  if (dot) stage.append(dot);

  const offGaze = engine.onGaze((sample) => {
    totalCount++;
    if (firstTimestamp === null) firstTimestamp = sample.t;

    if (dot) {
      dot.style.transform = `translate(${sample.x}px, ${sample.y}px)`;
    }

    const nx = (sample.x - rect.left) / rect.width;
    const ny = (sample.y - rect.top) / rect.height;
    // A small margin outside the stimulus is kept and clamped: gaze estimates
    // near an edge routinely land just past it, and discarding those would bias
    // edge content downward in every heatmap.
    if (nx < -0.05 || nx > 1.05 || ny < -0.05 || ny > 1.05) return;

    insideCount++;
    points.push({
      x: Math.min(1, Math.max(0, nx)),
      y: Math.min(1, Math.max(0, ny)),
      t: sample.t - firstTimestamp,
    });
  });

  const durationMs = study.duration > 0 ? study.duration * 1000 : 0;
  const stopped = await waitForStop(progressBar, durationMs, chrome, engine);
  offGaze();
  window.removeEventListener("resize", remeasure);
  window.removeEventListener("scroll", remeasure, true);

  restoreBackground();
  stage.remove();
  if (objectUrl) URL.revokeObjectURL(objectUrl);

  if (!stopped) return { status: "cancelled" };
  if (points.length === 0) return { status: "empty" };

  const recording: Recording = {
    id: newId("rec"),
    studyId: study.id,
    participant,
    createdAt: Date.now(),
    points,
    quality: {
      validationError: options.validationError,
      trackingRatio: totalCount > 0 ? insideCount / totalCount : 0,
      meanFps: engine.tracker.fps,
      viewportWidth: window.innerWidth,
      viewportHeight: window.innerHeight,
      stimulusRect: { x: rect.left, y: rect.top, width: rect.width, height: rect.height },
    },
  };

  await saveRecording(recording);
  return { status: "saved", recording };
}

function showTaskPrompt(chrome: HTMLElement, study: Study): Promise<boolean> {
  return new Promise((resolve) => {
    const panel = el(
      "div",
      { class: "task-prompt" },
      el("p", { class: "task-label" }, "Your task"),
      el("h2", {}, study.task || "Look at this screen as you naturally would."),
      el(
        "p",
        { class: "task-hint" },
        study.duration > 0
          ? `Recording stops automatically after ${study.duration} seconds.`
          : "Finish with the space bar or the on-screen button when you are done."
      ),
      el("button", { class: "btn btn-primary", type: "button" }, "Start, or press space")
    );

    const start = () => {
      cleanup();
      resolve(true);
    };
    const cancel = () => {
      cleanup();
      resolve(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.code === "Space" || event.code === "Enter") {
        event.preventDefault();
        start();
      } else if (event.key === "Escape") {
        cancel();
      }
    };
    const cleanup = () => {
      window.removeEventListener("keydown", onKey);
      panel.remove();
    };

    panel.querySelector("button")?.addEventListener("click", start);
    window.addEventListener("keydown", onKey);
    chrome.append(panel);
  });
}

/**
 * Resolves true when the recording completed, false when it was discarded.
 *
 * The on-screen buttons are the guaranteed exit, not a courtesy: the moment a
 * participant clicks into a cross-origin iframe stimulus, keyboard focus
 * moves into a document this window cannot hear, and Space and Escape stop
 * arriving. The shortcuts stay for the common case; the buttons always work.
 */
function waitForStop(
  progressBar: HTMLElement,
  durationMs: number,
  chrome: HTMLElement,
  engine: GazeEngine
): Promise<boolean> {
  return new Promise((resolve) => {
    const start = performance.now();
    let rafId = 0;
    let disarmTimer = 0;
    let armed = false;

    const baseHint =
      durationMs > 0 ? "Recording… space or Finish ends early" : "Recording… space or Finish when done";
    const hint = el("div", { class: "record-hint" }, baseHint);
    const finishBtn = el("button", { class: "btn btn-small", type: "button" }, "Finish");
    const discardBtn = el("button", { class: "btn btn-ghost btn-small", type: "button" }, "Discard");
    const controls = el("div", { class: "record-controls" }, hint, finishBtn, discardBtn);
    chrome.append(controls);

    // The one in-recording quality signal: gaze quietly stops arriving when
    // the face is lost, and the operator should not discover that afterwards.
    let faceLost = false;
    const offStatus = engine.onStatus((status) => {
      const lost = !status.faceVisible;
      if (lost === faceLost) return;
      faceLost = lost;
      hint.classList.toggle("is-lost", lost);
      hint.textContent = lost ? "Face lost — check lighting and framing" : baseHint;
    });

    const tick = () => {
      const elapsed = performance.now() - start;
      progressBar.style.width = `${Math.min(100, (elapsed / durationMs) * 100)}%`;
      if (elapsed >= durationMs) {
        finish(true);
        return;
      }
      rafId = requestAnimationFrame(tick);
    };

    // Discarding is two-step: this recording is a participant's one first
    // pass over the screen, and a single slip of Escape must not erase it.
    const requestDiscard = () => {
      if (armed) {
        finish(false);
        return;
      }
      armed = true;
      discardBtn.textContent = "Really discard?";
      discardBtn.classList.add("is-active");
      disarmTimer = window.setTimeout(() => {
        armed = false;
        discardBtn.textContent = "Discard";
        discardBtn.classList.remove("is-active");
      }, 3000);
    };

    const onKey = (event: KeyboardEvent) => {
      if (event.code === "Space") {
        event.preventDefault();
        finish(true);
      } else if (event.key === "Escape") {
        requestDiscard();
      }
    };

    const finish = (completed: boolean) => {
      cancelAnimationFrame(rafId);
      window.clearTimeout(disarmTimer);
      offStatus();
      window.removeEventListener("keydown", onKey);
      controls.remove();
      resolve(completed);
    };

    finishBtn.addEventListener("click", () => finish(true));
    discardBtn.addEventListener("click", requestDiscard);
    window.addEventListener("keydown", onKey);
    // A manual-stop recording has no progress bar to animate, so the frame
    // loop only runs when there is a deadline to draw toward.
    if (durationMs > 0) rafId = requestAnimationFrame(tick);
  });
}
