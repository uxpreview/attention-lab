import { ERROR_BAD_DEG, ERROR_GOOD_DEG, pxToDegrees } from "../analysis/quality";
import type { CalibrationSample, GazeEngine } from "../tracker/gaze";
import { confirmButton, el, inertSiblings, sleep } from "./dom";

/**
 * Calibration and validation flow.
 *
 * Calibration quality dominates everything downstream — a sloppy calibration
 * produces a confident-looking heatmap that is simply wrong, which is worse
 * than no heatmap. So the flow is deliberately strict: the participant clicks
 * each dot (which proves they are looking at it, rather than trusting them to),
 * samples are only collected after a settle delay, and a separate validation
 * pass measures error on points the model never saw.
 */

/** 13 points: corners, edge midpoints, centre, plus an inner ring. The inner
 * ring matters because most content sits in the middle of the screen and a
 * 9-point grid leaves the model interpolating across a wide gap there. */
const CALIBRATION_POINTS: Array<[number, number]> = [
  [0.08, 0.08], [0.5, 0.06], [0.92, 0.08],
  [0.06, 0.5], [0.5, 0.5], [0.94, 0.5],
  [0.08, 0.92], [0.5, 0.94], [0.92, 0.92],
  [0.28, 0.28], [0.72, 0.28], [0.28, 0.72], [0.72, 0.72],
];

/** Validation points sit deliberately off the calibration grid. */
const VALIDATION_POINTS: Array<[number, number]> = [
  [0.2, 0.2], [0.8, 0.2], [0.5, 0.35], [0.2, 0.8], [0.8, 0.8],
];

/** Milliseconds to wait after the click before trusting the gaze has landed. */
const SETTLE_MS = 350;
/** Milliseconds of samples to collect per point. */
const DWELL_MS = 900;
/** How long the participant has to hold their gaze after clicking. */
const HOLD_MS = SETTLE_MS + DWELL_MS;

const SVG_NS = "http://www.w3.org/2000/svg";
/** Ring geometry, in the ring's own 48-unit viewBox. */
const RING_RADIUS = 21;
const RING_CIRCUMFERENCE = 2 * Math.PI * RING_RADIUS;

/**
 * A determinate ring around the active dot, sweeping over the settle + dwell
 * window.
 *
 * Before this, the only cue that sampling was in progress was a colour swap,
 * and the instructions said "keep looking until it stops pulsing" — the exact
 * inverse of the truth, because the dot stops pulsing when it is clicked, at
 * the *start* of the hold. A participant who let go on that cue gave every
 * point 0ms of usable dwell, and nothing about the resulting calibration would
 * have looked wrong.
 *
 * Driven by the Web Animations API rather than CSS: the page neutralises every
 * CSS animation under prefers-reduced-motion, and this one is a progress
 * readout rather than decoration — removing it would take the only cue with it.
 */
function calibrationRing(): SVGSVGElement {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", "calib-ring");
  svg.setAttribute("viewBox", "0 0 48 48");
  svg.setAttribute("aria-hidden", "true");

  const track = document.createElementNS(SVG_NS, "circle");
  track.setAttribute("class", "calib-ring-track");
  track.setAttribute("cx", "24");
  track.setAttribute("cy", "24");
  track.setAttribute("r", String(RING_RADIUS));

  const sweep = document.createElementNS(SVG_NS, "circle");
  sweep.setAttribute("class", "calib-ring-sweep");
  sweep.setAttribute("cx", "24");
  sweep.setAttribute("cy", "24");
  sweep.setAttribute("r", String(RING_RADIUS));
  sweep.setAttribute("stroke-dasharray", String(RING_CIRCUMFERENCE));
  sweep.setAttribute("stroke-dashoffset", String(RING_CIRCUMFERENCE));

  svg.append(track, sweep);
  return svg;
}

/** Starts the sweep and returns a function that stops and resets it. */
function sweepRing(dot: HTMLElement): () => void {
  const sweep = dot.querySelector<SVGCircleElement>(".calib-ring-sweep");
  if (!sweep || typeof sweep.animate !== "function") return () => {};
  const animation = sweep.animate(
    [{ strokeDashoffset: RING_CIRCUMFERENCE }, { strokeDashoffset: 0 }],
    { duration: HOLD_MS, easing: "linear", fill: "forwards" }
  );
  return () => animation.cancel();
}

export interface CalibrationOutcome {
  cancelled: boolean;
  /** Mean validation error in CSS pixels, or null if validation was skipped. */
  validationError: number | null;
}

export async function runCalibration(
  engine: GazeEngine,
  host: HTMLElement
): Promise<CalibrationOutcome> {
  const overlay = el("div", {
    class: "calib-overlay",
    role: "dialog",
    "aria-modal": "true",
    "aria-label": "Calibration",
    tabindex: "-1",
  });
  const heading = el("h2", {}, "Calibration");
  const guidance = el(
    "p",
    { class: "calib-guidance" },
    "Keep your head still and your face lit from the front. Look at each dot and click it, then keep looking until the ring around it completes. Esc cancels."
  );
  // What survives the dimming. The full block used to go to opacity 0 after the
  // first dot, leaving seventeen more clicks with no statement of the one rule
  // that decides whether the calibration is worth anything — and the rule is
  // not guessable: a participant who lets go on the click gives every point
  // zero usable dwell, and nothing downstream looks wrong. Dimmed, the block
  // keeps this line and moves out of the middle of the screen, where the
  // centre dot is.
  const shortGuidance = el(
    "p",
    { class: "calib-short" },
    "Click the dot, then keep looking at it until the ring fills. Esc cancels."
  );
  const instruction = el("div", { class: "calib-instruction" }, heading, guidance, shortGuidance);
  // The counter lives outside the instruction block: the instructions dim
  // after the first dot, and progress is the one thing that has to stay
  // visible for all eighteen clicks — it is what buys participant patience.
  const progress = el("p", { class: "calib-progress", role: "status" });
  /**
   * The one condition that invalidates the whole sequence, said while it is
   * still recoverable.
   *
   * The recording stage has flagged a lost face since it was built; calibration
   * did not, and calibration is where it costs the most. A participant who
   * drifts out of frame at dot 3 goes on to click ten more dots collecting
   * nothing, and finds out at the end via "Not enough calibration data" —
   * a minute of session time spent, and the sequence starts again. The dot's
   * ring turns warn-coloured and this line appears beside the counter.
   */
  const lostHint = el(
    "p",
    { class: "calib-lost", role: "status", hidden: true },
    "Face lost — move back into frame and check your lighting"
  );
  const dot = el(
    "button",
    { class: "calib-dot", type: "button", "aria-label": "Calibration point" },
    calibrationRing()
  );
  // Two-step, like every other destructive control in the app: this one throws
  // away up to thirty seconds of a participant's work with no undo, and it
  // shares a corner with the dots. It also gets out of the way entirely when a
  // dot lands near it — see keepCancelClear.
  const cancelBtn = confirmButton(
    "Cancel",
    "Stop calibration?",
    () => cancel(),
    "btn btn-ghost btn-small calib-cancel"
  );
  // One line at the foot of the screen: the counter, and beside it the reason
  // the counter has stopped meaning anything.
  overlay.append(
    instruction,
    el("div", { class: "calib-status" }, progress, lostHint),
    dot,
    cancelBtn
  );
  host.append(overlay);

  const restoreBackground = inertSiblings(host, overlay);
  overlay.focus();

  // Same subscription the recording stage uses, so "face lost" cannot mean two
  // different things in two phases of the same session.
  let faceLost = false;
  const offStatus = engine.onStatus((status) => {
    const lost = !status.faceVisible;
    if (lost === faceLost) return;
    faceLost = lost;
    dot.classList.toggle("is-lost", lost);
    lostHint.hidden = !lost;
  });

  const samples: CalibrationSample[] = [];
  let cancelled = false;

  // One cancellation path for the Escape key and the on-screen button: both
  // abort whatever wait is in flight, and the loops check the flag between
  // points. Escape fires immediately — a deliberate keystroke is already the
  // second step — while the button arms first (see cancelBtn above).
  const abort = new AbortController();
  const cancel = () => {
    cancelled = true;
    abort.abort();
  };
  const onKey = (event: KeyboardEvent) => {
    if (event.key === "Escape") cancel();
  };
  window.addEventListener("keydown", onKey);

  try {
    for (let i = 0; i < CALIBRATION_POINTS.length; i++) {
      if (cancelled) break;
      const [nx, ny] = CALIBRATION_POINTS[i];
      // Phase-labelled, because an unlabelled counter restarting at "1 / 5"
      // after thirteen dots reads as the whole thing starting over.
      progress.textContent = `Calibration ${i + 1} / ${CALIBRATION_POINTS.length}`;
      keepCancelClear(cancelBtn, nx, ny);
      let collected: CollectResult;
      do {
        collected = await collectAtPoint(engine, dot, nx, ny, samples, abort.signal);
      } while (collected === "retry");
      if (collected === "abandoned") {
        cancelled = true;
        break;
      }
      // Dim the instruction panel after the first point so it stops competing
      // for attention with the dot — down to one line, moved out of the centre
      // of the screen. It used to go to opacity 0, which took the only
      // statement of the rule that decides whether any of this is usable.
      if (i === 0) instruction.classList.add("is-dim");
    }

    if (cancelled) return { cancelled: true, validationError: null };

    progress.textContent = "Fitting model…";
    engine.calibrate(samples);

    instruction.classList.remove("is-dim");
    heading.textContent = "Accuracy check";
    guidance.textContent =
      "Five more dots — the last of it. Same again: click, then hold until the ring completes. These measure how accurate the calibration actually is.";

    const pointErrors: number[] = [];
    for (let i = 0; i < VALIDATION_POINTS.length; i++) {
      if (cancelled) break;
      const [nx, ny] = VALIDATION_POINTS[i];
      progress.textContent = `Accuracy check ${i + 1} / ${VALIDATION_POINTS.length}`;
      keepCancelClear(cancelBtn, nx, ny);
      let error: number | "retry" | null;
      do {
        error = await measureAtPoint(engine, dot, nx, ny, abort.signal);
      } while (error === "retry");
      if (error === null) {
        cancelled = true;
        break;
      }
      pointErrors.push(error);
      if (i === 0) instruction.classList.add("is-dim");
    }

    if (cancelled || pointErrors.length === 0) {
      return { cancelled, validationError: null };
    }

    const mean = pointErrors.reduce((a, b) => a + b, 0) / pointErrors.length;
    return { cancelled: false, validationError: mean };
  } finally {
    window.removeEventListener("keydown", onKey);
    offStatus();
    engine.stopCollecting();
    restoreBackground();
    overlay.remove();
  }
}

/** Clearance kept between a calibration dot and the Cancel button, in pixels. */
const CANCEL_CLEARANCE = 120;

/**
 * Gets Cancel out of the way of the dot.
 *
 * Cancel is in the bottom-left corner, which no calibration or validation point
 * occupies — but the corner points still come within a dot's halo of it at some
 * viewport sizes, and the two nearest, [0.08, 0.92] and [0.2, 0.8], are the ones
 * a participant aims at hardest. A 10px overshoot landing on Cancel used to
 * abort the whole calibration on a single click. Now the button both arms
 * before it fires and vanishes while a dot is anywhere near it; Esc stays
 * available throughout and the dimmed instruction line keeps saying so.
 */
function keepCancelClear(cancelBtn: HTMLElement, nx: number, ny: number): void {
  const x = nx * window.innerWidth;
  const y = ny * window.innerHeight;
  const rect = cancelBtn.getBoundingClientRect();
  const dx = Math.max(rect.left - x, 0, x - rect.right);
  const dy = Math.max(rect.top - y, 0, y - rect.bottom);
  cancelBtn.classList.toggle("is-away", Math.hypot(dx, dy) < CANCEL_CLEARANCE);
}

function placeDot(dot: HTMLElement, nx: number, ny: number): { x: number; y: number } {
  const x = nx * window.innerWidth;
  const y = ny * window.innerHeight;
  dot.style.left = `${x}px`;
  dot.style.top = `${y}px`;
  return { x, y };
}

type CollectResult = "ok" | "retry" | "abandoned";

/**
 * Collects dwell samples at one dot. Resolves "retry" if the window lost
 * focus mid-dwell — a notification or an alt-tab takes the eyes with it, and
 * samples collected anyway would fold silently into the model as a
 * consistent, invisible bias — so that point's samples are discarded and the
 * same dot is presented again. Resolves "abandoned" on cancel.
 */
async function collectAtPoint(
  engine: GazeEngine,
  dot: HTMLButtonElement,
  nx: number,
  ny: number,
  into: CalibrationSample[],
  signal: AbortSignal
): Promise<CollectResult> {
  const { x, y } = placeDot(dot, nx, ny);
  dot.classList.remove("is-active");

  const clicked = await waitForClick(dot, signal);
  if (!clicked) return "abandoned";

  dot.classList.add("is-active");
  const stopRing = sweepRing(dot);

  let blurred = false;
  const onBlur = () => {
    blurred = true;
  };
  window.addEventListener("blur", onBlur);
  const before = into.length;

  await sleep(SETTLE_MS);
  engine.startCollecting(x, y, into);
  await sleep(DWELL_MS);
  engine.stopCollecting();

  window.removeEventListener("blur", onBlur);
  stopRing();
  dot.classList.remove("is-active");

  if (signal.aborted) return "abandoned";
  if (blurred) {
    into.length = before;
    return "retry";
  }
  return "ok";
}

/** Returns the median gaze error at this point in pixels, "retry" if focus
 * was lost mid-dwell (same reasoning as collection), or null if abandoned. */
async function measureAtPoint(
  engine: GazeEngine,
  dot: HTMLButtonElement,
  nx: number,
  ny: number,
  signal: AbortSignal
): Promise<number | "retry" | null> {
  const { x, y } = placeDot(dot, nx, ny);
  dot.classList.remove("is-active");

  const clicked = await waitForClick(dot, signal);
  if (!clicked) return null;

  dot.classList.add("is-active");
  const stopRing = sweepRing(dot);

  let blurred = false;
  const onBlur = () => {
    blurred = true;
  };
  window.addEventListener("blur", onBlur);

  await sleep(SETTLE_MS);
  const errors: number[] = [];
  const off = engine.onGaze((sample) => {
    errors.push(Math.hypot(sample.x - x, sample.y - y));
  });
  await sleep(DWELL_MS);
  off();

  window.removeEventListener("blur", onBlur);
  stopRing();
  dot.classList.remove("is-active");

  if (signal.aborted) return null;
  if (blurred) return "retry";
  if (errors.length === 0) return Infinity;
  // Median rather than mean: a single blink-adjacent outlier should not decide
  // whether we tell the researcher their calibration is good.
  errors.sort((a, b) => a - b);
  return errors[Math.floor(errors.length / 2)];
}

function waitForClick(dot: HTMLButtonElement, signal: AbortSignal): Promise<boolean> {
  return new Promise((resolve) => {
    if (signal.aborted) {
      resolve(false);
      return;
    }
    const onClick = () => {
      cleanup();
      resolve(true);
    };
    const onAbort = () => {
      cleanup();
      resolve(false);
    };
    const cleanup = () => {
      dot.removeEventListener("click", onClick);
      signal.removeEventListener("abort", onAbort);
    };
    dot.addEventListener("click", onClick);
    signal.addEventListener("abort", onAbort);
  });
}

/**
 * Turns validation error into language a researcher can act on. The thresholds
 * are in degrees of visual angle at an assumed 60cm viewing distance, which is
 * the unit eye-tracking literature uses; commercial trackers claim 0.5°, and
 * webcam tracking lands around 1.5-3° on a good day. They live in
 * analysis/quality.ts, shared with the results screen's quality flags, so the
 * word "poor" here and a flagged row there can never mean different things.
 */
export function describeAccuracy(errorPx: number | null): {
  grade: "good" | "usable" | "poor" | "unknown";
  label: string;
  detail: string;
} {
  if (errorPx === null || !Number.isFinite(errorPx)) {
    return { grade: "unknown", label: "Not measured", detail: "Run the accuracy check to see error." };
  }

  const degrees = pxToDegrees(errorPx);
  if (degrees < ERROR_GOOD_DEG) {
    return {
      grade: "good",
      label: `Good: ~${degrees.toFixed(1)}° (${Math.round(errorPx)}px)`,
      detail: "Reliable enough for region-level conclusions on a wireframe.",
    };
  }
  if (degrees < ERROR_BAD_DEG) {
    return {
      grade: "usable",
      label: `Usable: ~${degrees.toFixed(1)}° (${Math.round(errorPx)}px)`,
      detail: "Good for big blocks. Do not read individual words or small links from this.",
    };
  }
  return {
    grade: "poor",
    label: `Poor: ~${degrees.toFixed(1)}° (${Math.round(errorPx)}px)`,
    detail: "Recalibrate: improve lighting, sit square to the screen, and stay still.",
  };
}
