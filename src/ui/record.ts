import { newId, saveRecording } from "../data/store";
import type { Recording, Study } from "../data/types";
import type { GazeEngine } from "../tracker/gaze";
import { el, nextFrame } from "./dom";

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

export async function runRecording(
  host: HTMLElement,
  options: RecordOptions
): Promise<Recording | null> {
  const { study, engine, participant } = options;

  const stage = el("div", { class: "record-stage" });
  const stimulusLayer = el("div", { class: "record-stimulus" });
  const chrome = el("div", { class: "record-chrome" });
  const progressBar = el("div", { class: "record-progress" });
  stage.append(stimulusLayer, progressBar, chrome);
  host.append(stage);

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
    // A page-length wireframe squeezed into the viewport is unreadable, so tall
    // images render full-width and scroll instead. Gaze is mapped through the
    // image's live rect per sample, so scrolling keeps document coordinates
    // correct.
    if (isTallStimulus(img.naturalWidth, img.naturalHeight)) {
      stimulusLayer.classList.add("is-scrollable");
      img.classList.add("record-image-full");
    }
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
    stage.remove();
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    return null;
  }

  await nextFrame();
  const rect = stimulusEl.getBoundingClientRect();

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

    // Measured per sample, not once: a scrollable stimulus moves under the
    // gaze, and the live rect is what converts screen position into document
    // position on the stimulus.
    const live = stimulusEl.getBoundingClientRect();
    const nx = (sample.x - live.left) / live.width;
    const ny = (sample.y - live.top) / live.height;
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
  const stopped = await waitForStop(progressBar, durationMs, chrome);
  offGaze();

  stage.remove();
  if (objectUrl) URL.revokeObjectURL(objectUrl);

  if (!stopped || points.length === 0) return null;

  const recording: Recording = {
    id: newId("rec"),
    studyId: study.id,
    participant,
    createdAt: Date.now(),
    startedAt: firstTimestamp ?? 0,
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
  return recording;
}

/**
 * A stimulus notably taller than the screen is worth scrolling; the 1.2 slack
 * keeps near-viewport images in the simpler letterboxed mode.
 */
export function isTallStimulus(width: number, height: number): boolean {
  if (width <= 0 || height <= 0) return false;
  return height / width > (window.innerHeight / window.innerWidth) * 1.2;
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
          : "Press space again when you are done.",
        study.stimulus.kind === "url"
          ? " The page is frozen during recording — scrolling and clicking are disabled so every participant sees the same thing."
          : ""
      ),
      el("button", { class: "btn btn-primary", type: "button" }, "Start — or press space")
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

/** Resolves true when the recording completed, false when it was abandoned. */
function waitForStop(
  progressBar: HTMLElement,
  durationMs: number,
  chrome: HTMLElement
): Promise<boolean> {
  return new Promise((resolve) => {
    const start = performance.now();
    let rafId = 0;

    const hint = el(
      "div",
      { class: "record-hint" },
      durationMs > 0 ? "Recording…" : "Recording… press space to finish"
    );
    chrome.append(hint);

    const tick = () => {
      if (durationMs > 0) {
        const elapsed = performance.now() - start;
        progressBar.style.width = `${Math.min(100, (elapsed / durationMs) * 100)}%`;
        if (elapsed >= durationMs) {
          finish(true);
          return;
        }
      }
      rafId = requestAnimationFrame(tick);
    };

    const onKey = (event: KeyboardEvent) => {
      if (event.code === "Space") {
        event.preventDefault();
        finish(true);
      } else if (event.key === "Escape") {
        finish(false);
      }
    };

    const finish = (completed: boolean) => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("keydown", onKey);
      hint.remove();
      resolve(completed);
    };

    window.addEventListener("keydown", onKey);
    rafId = requestAnimationFrame(tick);
  });
}
