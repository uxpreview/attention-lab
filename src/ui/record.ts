import { newId, saveRecording } from "../data/store";
import type { Recording, Study } from "../data/types";
import type { GazeEngine } from "../tracker/gaze";
import type { BiasEstimate, ResidualSample } from "../tracker/regression";
import { confirmButton, el, inertSiblings, nextFrame } from "./dom";

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
  /** The residuals behind `validationError`, and the offset taken out of them.
   * Stored with the recording so a doubtful heatmap can be diagnosed rather
   * than only re-run — see RecordingQuality. */
  residuals: ResidualSample[];
  bias: BiasEstimate | null;
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

/**
 * How far outside the stimulus a gaze sample is still kept and clamped in.
 *
 * Gaze estimates near an edge routinely land just past it, and throwing those
 * away would bias edge content downward in every heatmap — so a sample within
 * this fraction of the stimulus is pulled back onto the nearest edge instead.
 * It is exported because it is not only a filter constant: it sets how far the
 * moderator's own chrome has to stay clear of the stimulus (see
 * {@link controlBandHeight}).
 */
export const EDGE_TOLERANCE = 0.05;

/**
 * Height of the control strip at the bottom of the reserved band, including
 * its inset from the screen edge. Mirrored by `min-height` and `bottom` on
 * `.record-controls`, which is the pair that has to stay true.
 */
export const CONTROL_STRIP_PX = 54;

/**
 * Height of the band reserved at the foot of the stage for the moderator's
 * controls, for a stage this tall.
 *
 * This is a data-integrity calculation, not a layout preference. The controls
 * used to be absolutely positioned over the stimulus, and every gaze sample is
 * normalised against the stimulus rect: a participant glancing at the Finish
 * button, the discard button or the "RECORDING…" hint produced samples that
 * were recorded as attention to whatever the chrome was sitting on, which on a
 * full-bleed stimulus is the footer. The moderator's own furniture was
 * manufacturing attention data in the exact region it covered.
 *
 * Letterboxing the stimulus above the strip is necessary but not sufficient,
 * because of {@link EDGE_TOLERANCE}: 5% of an 800px stimulus is 40px of chrome
 * that would still be clamped back onto the footer. So the band is solved for
 * instead — it is exactly deep enough that the top of the control strip sits
 * below the tolerance line:
 *
 *     stageHeight - CONTROL_STRIP_PX >= (1 + EDGE_TOLERANCE) * (stageHeight - band)
 *
 * Rearranged and rounded up, which is what this returns.
 */
export function controlBandHeight(stageHeight: number): number {
  const height = Math.max(0, stageHeight);
  // The plus one is not slop: the tolerance test keeps a sample at *exactly*
  // 1.05, so solving the inequality as an equality leaves the top row of the
  // strip on the wrong side of it.
  return Math.ceil((EDGE_TOLERANCE * height + CONTROL_STRIP_PX) / (1 + EDGE_TOLERANCE)) + 1;
}

/**
 * True when a control strip in the reserved band could still be normalised
 * into the stimulus rect — the property the band exists to make impossible.
 * Exported for the test suite, which asserts it is false at every plausible
 * stage size.
 *
 * This covers the contained fit, where the stimulus is letterboxed above the
 * band and its rect therefore ends where the band begins. The scrolling fit
 * (see {@link fitStimulus}) breaks that assumption — a stimulus taller than its
 * viewport has a rect that runs off both ends of the screen, so the band is
 * inside it by construction — and is guarded in screen space instead, by
 * {@link isInsideViewport}.
 */
export function chromeCanContaminate(stageHeight: number): boolean {
  const stimulusHeight = stageHeight - controlBandHeight(stageHeight);
  if (stimulusHeight <= 0) return true;
  const topOfControls = stageHeight - CONTROL_STRIP_PX;
  return topOfControls / stimulusHeight <= 1 + EDGE_TOLERANCE;
}

/**
 * The smallest scale at which a stimulus is still worth showing contained.
 *
 * Below this a 1280px-wide wireframe is being rendered at well under its
 * designed size and its labels stop being readable — which does not merely look
 * bad, it invalidates the study: the task asks someone to find a control on
 * text they cannot read, and the heatmap that comes back is a picture of them
 * squinting. 0.85 is where a 13px label lands around 11px, the floor at which
 * UI copy is still scanned rather than deciphered.
 */
export const FIT_SCALE_FLOOR = 0.85;

/**
 * How a stimulus of this size should be shown in a viewport of that size.
 *
 * "contain" is the letterbox that has always been used: the whole stimulus at
 * once, which is right whenever it survives at a readable scale. "width" gives
 * the stimulus the full width of the viewport and lets it scroll vertically —
 * the way every full-page screenshot is read — and is chosen only when
 * containing it would fall below {@link FIT_SCALE_FLOOR} *and* fitting the
 * width does measurably better. Where the width is already the binding
 * constraint — anything wider than it is tall relative to the stage — scrolling
 * buys nothing, so it stays contained.
 *
 * Pure, and exported, because it is the one rule deciding what a participant
 * sees, and the setup form quotes it back to the operator before they spend a
 * participant on an unreadable screen.
 */
export function fitStimulus(
  stimulus: { width: number; height: number },
  viewport: { width: number; height: number }
): { mode: "contain" | "width"; scale: number } {
  if (stimulus.width <= 0 || stimulus.height <= 0) return { mode: "contain", scale: 1 };
  const contain = Math.min(viewport.width / stimulus.width, viewport.height / stimulus.height);
  const byWidth = viewport.width / stimulus.width;
  if (contain >= FIT_SCALE_FLOOR || byWidth <= contain) return { mode: "contain", scale: contain };
  // Never past 1:1. A 1280px stimulus in a 1440px window is shown at its native
  // size and centred, not stretched to fill — upscaling a screenshot softens
  // exactly the labels this mode exists to make readable.
  return { mode: "width", scale: Math.min(1, byWidth) };
}

/**
 * True when a gaze sample is over the part of the stimulus the participant can
 * actually see.
 *
 * In the contained fit this is implied — the stimulus is the visible rect — and
 * the normalised range check does the whole job. In the scrolling fit it is
 * not: the stimulus rect extends past the top and bottom of its viewport, and
 * the moderator's control strip sits below that viewport, inside the rect. A
 * glance at Finish would normalise to a perfectly plausible point three
 * quarters of the way down the page. So the sample is required to be in the
 * viewport box first, with the same edge tolerance applied on the axes where
 * the stimulus edge and the viewport edge coincide — never below the foot,
 * which is where the chrome is.
 */
export function isInsideViewport(
  sample: { x: number; y: number },
  viewport: { left: number; top: number; width: number; height: number }
): boolean {
  const padX = viewport.width * EDGE_TOLERANCE;
  const padY = viewport.height * EDGE_TOLERANCE;
  return (
    sample.x >= viewport.left - padX &&
    sample.x <= viewport.left + viewport.width + padX &&
    sample.y >= viewport.top - padY &&
    sample.y <= viewport.top + viewport.height
  );
}

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

  // The band the stimulus is letterboxed above, published to CSS so the one
  // number is derived in one place. See controlBandHeight: this is what keeps
  // the moderator's controls out of the measured rect.
  const applyControlBand = (): void => {
    stage.style.setProperty("--record-band", `${controlBandHeight(window.innerHeight)}px`);
  };
  applyControlBand();

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

  /**
   * Fit the stimulus so the participant can read it.
   *
   * `object-fit: contain` with no width-fit path meant a 1280×1600 full-page
   * screenshot rendered 643px wide in a 1440px stage — half its designed size,
   * 55% of the participant's screen empty teal, and a task prompt asking them
   * to find a control on labels they cannot read. That does not merely look
   * bad: it is the screen the finding is made on, so it invalidates the study.
   * Below the readable floor the stimulus takes the full width and scrolls,
   * which is how a full-page screenshot is read everywhere else.
   *
   * The choice is re-made on resize, because the window can change shape
   * mid-session and the fit is a function of the box.
   */
  let scrolling = false;
  const applyFit = (): void => {
    if (study.stimulus.kind !== "image") return;
    const box = stimulusLayer.getBoundingClientRect();
    const fit = fitStimulus(
      { width: study.stimulus.width, height: study.stimulus.height },
      { width: box.width, height: box.height }
    );
    scrolling = fit.mode === "width";
    stimulusLayer.classList.toggle("is-scroll", scrolling);
    stimulusEl.classList.toggle("is-fit-width", scrolling);
  };
  applyFit();

  await nextFrame();
  // Cached rather than measured per sample — one layout read per recording,
  // not thirty per second. But a resize, zoom, or scroll mid-recording moves
  // the stimulus under the cached rect and silently corrupts every sample
  // after it, so those events re-measure: one layout read per event.
  let rect = stimulusEl.getBoundingClientRect();
  /** The window onto the stimulus. Identical to `rect` in the contained fit;
   * in the scrolling fit it is the part of a taller stimulus that is on screen,
   * and it is what keeps the moderator's controls out of the measurement — see
   * isInsideViewport. */
  let viewport = stimulusLayer.getBoundingClientRect();
  /** The measured rect, published to CSS.
   *
   * The progress track used to span the whole stage: with a 1400×900 stimulus
   * letterboxed into a 1440×900 window it ran 95px past the picture on both
   * sides and its fill started hard against the stage's bottom-left corner,
   * attached to nothing. It underlines the stimulus, so it is as wide as the
   * stimulus — and off the same rect every gaze sample is normalised against,
   * rather than a second measurement that could disagree with it. */
  const publishRect = (): void => {
    stage.style.setProperty("--stim-left", `${rect.left}px`);
    stage.style.setProperty("--stim-width", `${rect.width}px`);
  };
  publishRect();
  const remeasure = () => {
    // The band is a function of the stage height, so a resize has to move it
    // before the rect is read back — otherwise a window made taller mid-session
    // leaves the strip inside the tolerance zone of the new, taller stimulus.
    applyControlBand();
    applyFit();
    rect = stimulusEl.getBoundingClientRect();
    viewport = stimulusLayer.getBoundingClientRect();
    publishRect();
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

    // In the scrolling fit the stimulus rect runs off both ends of its window,
    // so "inside the rect" no longer implies "on screen" — and the moderator's
    // control strip, which sits below that window, would normalise to a
    // plausible point partway down the page. The window is checked first.
    if (scrolling && !isInsideViewport(sample, viewport)) return;

    const nx = (sample.x - rect.left) / rect.width;
    const ny = (sample.y - rect.top) / rect.height;
    // A small margin outside the stimulus is kept and clamped: gaze estimates
    // near an edge routinely land just past it, and discarding those would bias
    // edge content downward in every heatmap. Nothing interactive is allowed
    // inside that margin — controlBandHeight is solved against this constant so
    // that a look at the moderator's controls falls outside it and is dropped
    // rather than being recorded as a look at the stimulus footer.
    const limit = 1 + EDGE_TOLERANCE;
    if (nx < -EDGE_TOLERANCE || nx > limit || ny < -EDGE_TOLERANCE || ny > limit) return;

    insideCount++;
    points.push({
      x: Math.min(1, Math.max(0, nx)),
      y: Math.min(1, Math.max(0, ny)),
      t: sample.t - firstTimestamp,
    });
  });

  const durationMs = study.duration > 0 ? study.duration * 1000 : 0;
  const stopped = await waitForStop(
    progressBar,
    durationMs,
    chrome,
    engine,
    participant,
    stimulusLayer
  );
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
      calibrationResiduals: options.residuals,
      calibrationBias: options.bias,
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
      el(
        "button",
        { class: "btn btn-primary", type: "button", onclick: () => start() },
        "Start, or press space"
      )
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
  engine: GazeEngine,
  participant: string,
  /** The letterbox the stimulus is drawn in, so the face-lost warning can put a
   * cue inside the participant's field of view rather than only in the
   * moderator's strip at the foot of the screen. */
  stimulusLayer: HTMLElement
): Promise<boolean> {
  return new Promise((resolve) => {
    const start = performance.now();
    let rafId = 0;
    let clockTimer = 0;

    const baseHint =
      durationMs > 0 ? "Recording… space or Finish ends early" : "Recording… space or Finish when done";
    const hint = el("div", { class: "record-hint" }, baseHint);
    // Who is being recorded, and how long is left. A moderator running
    // back-to-back sessions otherwise has no way to confirm they are capturing
    // P04 and not P03, and a progress sliver at the foot of the screen is not
    // a clock.
    const clock = el("span", { class: "record-clock" }, durationMs > 0 ? formatClock(durationMs) : "0:00");
    // Nothing on this screen read as "live". The only state cues were a 13px
    // grey sentence and a peach clock: a moderator glancing across a desk
    // could not tell a running capture from a paused one, and a participant
    // had no cue that the camera was on — which matters most in the tool whose
    // selling point is that the camera never leaves the machine. The dot is
    // the universal one, in --signal-bad, beside the label it applies to.
    const who = el(
      "div",
      { class: "record-who" },
      el("span", { class: "record-live", "aria-hidden": "true" }),
      el("span", { class: "record-participant" }, participant),
      clock
    );
    // The safe completion is the loudest control in the strip. Finish was an
    // outline button while an armed Discard filled itself in --signal-bad, so
    // at the exact moment a moderator is under time pressure and reaching for
    // a button, the destructive one was the most prominent element on the
    // screen. Primary here is peach on teal, per the band's token remap.
    const finishBtn = el("button", { class: "btn btn-primary btn-small", type: "button" }, "Finish");
    // Two-step, and the same two-step every other destructive control in the
    // app uses: this is a participant's one first pass over the screen and a
    // single slip of Escape must not erase it. It used to be hand-rolled here
    // with an `.is-active` class that the strip's own teal backing then
    // flattened into something indistinguishable from Finish beside it;
    // confirmButton arms in --signal-bad and holds its width while it does.
    const discardBtn = confirmButton(
      "Discard",
      "Really discard?",
      () => finish(false),
      "btn btn-ghost btn-small"
    );
    const controls = el("div", { class: "record-controls" }, who, hint, finishBtn, discardBtn);
    chrome.append(controls);

    const paintClock = () => {
      const elapsed = performance.now() - start;
      clock.textContent = formatClock(durationMs > 0 ? Math.max(0, durationMs - elapsed) : elapsed);
    };
    // A timed recording already has a frame loop; a manual one does not, and
    // does not need one — four ticks a second is plenty for a wall clock.
    if (durationMs === 0) clockTimer = window.setInterval(paintClock, 250);

    // The one in-recording quality signal: gaze quietly stops arriving when
    // the face is lost, and the operator should not discover that afterwards.
    //
    // It is signalled twice, because the two people in the room are looking at
    // different things. The chip in the strip is the moderator's readout. The
    // ring on the stimulus is the participant's: the strip is ~830px below
    // where they are actually looking, and calibration already solves the same
    // condition this way — the dot's ring changes colour inside their foveal
    // field *and* the status line appears. Nothing else is allowed inside the
    // measured rect, but while the face is lost the tracker emits no samples,
    // so there is no measurement for the ring to contaminate.
    let faceLost = false;
    const offStatus = engine.onStatus((status) => {
      const lost = !status.faceVisible;
      if (lost === faceLost) return;
      faceLost = lost;
      hint.classList.toggle("is-lost", lost);
      stimulusLayer.classList.toggle("is-lost", lost);
      hint.textContent = lost ? "Face lost — check lighting and framing" : baseHint;
    });

    const tick = () => {
      const elapsed = performance.now() - start;
      progressBar.style.width = `${Math.min(100, (elapsed / durationMs) * 100)}%`;
      paintClock();
      if (elapsed >= durationMs) {
        finish(true);
        return;
      }
      rafId = requestAnimationFrame(tick);
    };

    const onKey = (event: KeyboardEvent) => {
      if (event.code === "Space") {
        event.preventDefault();
        finish(true);
      } else if (event.key === "Escape") {
        // Escape arms and fires the same button the mouse does, so the
        // keyboard cannot reach a one-press discard the pointer cannot.
        discardBtn.click();
      }
    };

    const finish = (completed: boolean) => {
      cancelAnimationFrame(rafId);
      window.clearInterval(clockTimer);
      offStatus();
      window.removeEventListener("keydown", onKey);
      // The ring outlives the listener that raised it otherwise, and a
      // discarded-then-retried session would open already flagged.
      stimulusLayer.classList.remove("is-lost");
      controls.remove();
      resolve(completed);
    };

    finishBtn.addEventListener("click", () => finish(true));
    window.addEventListener("keydown", onKey);
    // A manual-stop recording has no progress bar to animate, so the frame
    // loop only runs when there is a deadline to draw toward.
    if (durationMs > 0) rafId = requestAnimationFrame(tick);
  });
}

/** m:ss, or plain seconds under a minute — the shortest reading that is still
 * unambiguous at a glance from across a desk. */
function formatClock(ms: number): string {
  const total = Math.max(0, Math.round(ms / 1000));
  if (total < 60) return `${total}s`;
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}
