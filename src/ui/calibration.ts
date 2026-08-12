import { ERROR_BAD_DEG, ERROR_GOOD_DEG, pxToDegrees } from "../analysis/quality";
import type { CalibrationSample, GazeEngine } from "../tracker/gaze";
import {
  measureBias,
  withBias,
  type BiasEstimate,
  type ResidualSample,
} from "../tracker/regression";
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
  /**
   * The residual at each validation dot: where the model looked, minus where
   * the participant demonstrably was.
   *
   * The mean above is a magnitude, and a magnitude cannot say what kind of
   * wrong a calibration is. A rigid offset, a field contracted toward the
   * screen centre, and pure per-sample scatter all average to the same number
   * and want three different responses. Keeping the vectors is what lets
   * {@link measureBias} separate the correctable part from the rest, and what
   * lets a recording be diagnosed after the fact instead of re-run.
   */
  residuals: ResidualSample[];
  /** The correctable component of {@link residuals}, already applied to the
   * engine's model. Null when validation was skipped. */
  bias: BiasEstimate | null;
}

/**
 * "full" fits a new model from 13 dots and then measures it with 5 more.
 * "recheck" skips the fitting and measures the model already installed —
 * see {@link recheckAccuracy}.
 */
type Mode = "full" | "recheck";

/** Fits a calibration and measures it. */
export function runCalibration(engine: GazeEngine, host: HTMLElement): Promise<CalibrationOutcome> {
  return runFlow(engine, host, "full");
}

/**
 * Re-runs only the accuracy check against a calibration that is already
 * installed, measuring what has drifted since it was fitted.
 *
 * This is the cheap half of calibration — five dots rather than eighteen — and
 * it exists because reusing a stored calibration used to carry its original
 * accuracy figure forward unexamined. A participant who has shifted in their
 * seat since fitting has a model with a constant offset, and both that offset
 * and the stale number reported alongside it are wrong in the same direction:
 * confidently.
 */
export function recheckAccuracy(
  engine: GazeEngine,
  host: HTMLElement
): Promise<CalibrationOutcome> {
  return runFlow(engine, host, "recheck");
}

async function runFlow(
  engine: GazeEngine,
  host: HTMLElement,
  mode: Mode
): Promise<CalibrationOutcome> {
  const recheck = mode === "recheck";
  const overlay = el("div", {
    class: "calib-overlay",
    role: "dialog",
    "aria-modal": "true",
    "aria-label": recheck ? "Accuracy check" : "Calibration",
    tabindex: "-1",
  });
  const heading = el("h2", {}, recheck ? "Accuracy check" : "Calibration");
  const guidance = el(
    "p",
    { class: "calib-guidance" },
    recheck
      ? "Five dots, to check the calibration still fits how you are sitting now. Look at each one and click it, then keep looking until the ring completes. Esc cancels."
      : "Keep your head still and your face lit from the front. Look at each dot and click it, then keep looking until the ring around it completes. Esc cancels."
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
  //
  // It used to be the quietest thing on the screen it was carrying: 14px in a
  // low-contrast grey-teal, centred at the foot of the window — 42px from where
  // the y=0.92 dot row lands, so through the bottom third of the sequence the
  // counter sat inside the halo of the dot the participant was aiming at. And a
  // bare fraction is not progress for an eighteen-step sequence that changes
  // its own denominator halfway through: "Calibration 1 / 13" says nothing
  // about the five dots that follow it. So: a determinate bar across both
  // phases, a louder counter, a line naming what comes next, and a corner
  // anchor that steps out of the dot's way (see keepStatusClear).
  const progress = el("p", { class: "calib-progress", role: "status" });
  const progressFill = el("div", { class: "calib-bar-fill" });
  const progressTrack = el("div", { class: "calib-bar", "aria-hidden": "true" }, progressFill);
  const phaseNote = el("p", { class: "calib-phase" });
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
  // One block in a corner: where you are, how far that is through the whole
  // sequence, what follows, and the reason the counter has stopped meaning
  // anything.
  const status = el(
    "div",
    { class: "calib-status" },
    progress,
    progressTrack,
    phaseNote,
    lostHint
  );
  overlay.append(instruction, status, dot, cancelBtn);
  host.append(overlay);

  /** Every dot the participant is asked to click, both phases together. The
   * bar spans this rather than restarting at the accuracy check, because what a
   * participant wants to know is how much of *this* is left. */
  const TOTAL_POINTS = recheck
    ? VALIDATION_POINTS.length
    : CALIBRATION_POINTS.length + VALIDATION_POINTS.length;
  const setProgress = (done: number, label: string, phase: string): void => {
    progress.textContent = label;
    phaseNote.textContent = phase;
    progressFill.style.width = `${Math.min(100, (done / TOTAL_POINTS) * 100)}%`;
  };

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
    for (let i = 0; !recheck && i < CALIBRATION_POINTS.length; i++) {
      if (cancelled) break;
      const [nx, ny] = CALIBRATION_POINTS[i];
      // Phase-labelled, because an unlabelled counter restarting at "1 / 5"
      // after thirteen dots reads as the whole thing starting over — and the
      // phase line says at dot 1 that a second phase is coming, rather than
      // springing it after thirteen clicks.
      setProgress(
        i + 1,
        `Calibration ${i + 1} / ${CALIBRATION_POINTS.length}`,
        `Then a ${VALIDATION_POINTS.length}-dot accuracy check`
      );
      keepCancelClear(cancelBtn, nx, ny);
      keepStatusClear(status, nx, ny);
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

    if (cancelled) return { cancelled: true, validationError: null, residuals: [], bias: null };

    if (!recheck) {
      setProgress(CALIBRATION_POINTS.length, "Fitting model…", "One moment");
      engine.calibrate(samples);

      instruction.classList.remove("is-dim");
      heading.textContent = "Accuracy check";
      guidance.textContent =
        "Five more dots — the last of it. Same again: click, then hold until the ring completes. These measure how accurate the calibration actually is.";
    }

    const done = recheck ? 0 : CALIBRATION_POINTS.length;
    const measurement = await runValidation(engine, dot, abort.signal, (i) => {
      setProgress(
        done + i + 1,
        `Accuracy check ${i + 1} / ${VALIDATION_POINTS.length}`,
        recheck ? "" : "The last of it"
      );
      keepCancelClear(cancelBtn, VALIDATION_POINTS[i][0], VALIDATION_POINTS[i][1]);
      keepStatusClear(status, VALIDATION_POINTS[i][0], VALIDATION_POINTS[i][1]);
      // On a recheck the instructions are the whole context the participant
      // has, so they survive one dot longer than in the full flow.
      if (i === (recheck ? 1 : 0)) instruction.classList.add("is-dim");
    });

    if (measurement === null) {
      return { cancelled: true, validationError: null, residuals: [], bias: null };
    }
    return { cancelled: false, ...measurement };
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

/**
 * Gets the progress block out of the way of the dot, without ever hiding it.
 *
 * Cancel can simply vanish when a dot comes near it; the counter cannot — it is
 * what buys eighteen clicks of patience. So it has two anchors instead: bottom
 * right by default, top right when a dot lands near the bottom right. At most
 * one of the two can be contested at a time, because the flip is only triggered
 * by a dot that is itself in the bottom corner.
 *
 * The clearance is measured against the *default* anchor's box rather than the
 * element's current one. Measuring where it happens to be would make the
 * decision depend on the previous decision: flipped to the top and then asked
 * about a bottom-right dot, it would read "far away", drop back down, and land
 * on the dot it was avoiding.
 */
function keepStatusClear(status: HTMLElement, nx: number, ny: number): void {
  const x = nx * window.innerWidth;
  const y = ny * window.innerHeight;
  const box = status.getBoundingClientRect();
  // Mirrors .calib-status's inset in styles.css.
  const inset = 18;
  const right = window.innerWidth - inset;
  const bottom = window.innerHeight - inset;
  const left = right - box.width;
  const top = bottom - box.height;
  const dx = Math.max(left - x, 0, x - right);
  const dy = Math.max(top - y, 0, y - bottom);
  status.classList.toggle("is-flipped", Math.hypot(dx, dy) < STATUS_CLEARANCE);
}

/** Clearance kept between a dot and the progress block, in pixels. Larger than
 * the Cancel button's, because this block is wider and taller than that pill
 * and a participant aiming at a corner dot should not have text under it. */
const STATUS_CLEARANCE = 150;

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

/** The measured part of a {@link CalibrationOutcome}. */
type ValidationResult = Pick<CalibrationOutcome, "validationError" | "residuals" | "bias">;

/**
 * Runs the validation dots, then measures and installs a bias correction.
 *
 * Correcting here rather than at the call site is deliberate: the residuals are
 * measured against the model currently installed on the engine, so the
 * correction is only valid for that model. Applying it anywhere else risks
 * pairing an offset with a calibration it was not measured against, which would
 * displace every gaze point by a constant and — this being the failure mode
 * this app keeps rediscovering — look exactly like data.
 *
 * Returns null if the participant abandoned the run.
 */
async function runValidation(
  engine: GazeEngine,
  dot: HTMLButtonElement,
  signal: AbortSignal,
  onPoint: (index: number) => void
): Promise<ValidationResult | null> {
  const residuals: ResidualSample[] = [];
  const measurements: PointMeasurement[] = [];

  for (let i = 0; i < VALIDATION_POINTS.length; i++) {
    const [nx, ny] = VALIDATION_POINTS[i];
    onPoint(i);

    let measured: PointMeasurement | "retry" | null;
    do {
      measured = await measureAtPoint(engine, dot, nx, ny, signal);
    } while (measured === "retry");
    if (measured === null) return null;

    measurements.push(measured);
    residuals.push({
      targetX: nx * window.innerWidth,
      targetY: ny * window.innerHeight,
      dx: measured.dx,
      dy: measured.dy,
    });
  }

  if (measurements.length === 0) return { validationError: null, residuals: [], bias: null };

  const bias = measureBias(residuals);
  const model = engine.getModel();

  // An offset too large to be posture is a calibration that has stopped
  // describing this participant. Report it, but do not subtract it: the
  // measured error stays honest and the UI can tell them to recalibrate.
  const correcting = Boolean(model) && bias.correctable;
  if (model && correcting) engine.setModel(withBias(model, bias));

  // Recompute per-sample error against the model that will actually record.
  // Reporting the uncorrected figure would overstate the error of a corrected
  // model, and this number both grades the recording and sizes the heatmap
  // kernel — so it has to describe the gaze that gets stored, not the gaze the
  // uncorrected model would have produced.
  const perPoint = measurements.map((m) =>
    m.offsets.length === 0
      ? m.error
      : medianOf(
          m.offsets.map(([dx, dy]) =>
            correcting ? Math.hypot(dx - bias.x, dy - bias.y) : Math.hypot(dx, dy)
          )
        )
  );

  return {
    validationError: perPoint.reduce((a, b) => a + b, 0) / perPoint.length,
    residuals,
    bias,
  };
}

/** What one validation dot measured. */
interface PointMeasurement {
  /** Median per-sample distance from the dot: the error a single recorded gaze
   * point carries, which is what the heatmap kernel has to cover. */
  error: number;
  /** Median signed residual, component-wise. Taking the median per axis before
   * combining is what separates a consistent lean from scatter — the magnitude
   * above cannot, because distance throws the sign away. */
  dx: number;
  dy: number;
  /**
   * Every sample's signed offset, kept so the error can be recomputed once the
   * offset correction is known.
   *
   * The alternative is to estimate the corrected error from the summary
   * numbers, and there is no honest way to do that: `dx`/`dy` are medians over
   * a dot's samples, so they have already averaged away most of the per-sample
   * scatter a recording will actually carry. Subtracting the bias from *those*
   * and calling the remainder the error would report a figure two or three
   * times better than the gaze being stored — and that figure sizes the
   * heatmap kernel.
   */
  offsets: Array<[number, number]>;
}

/** Returns the gaze error at this point, "retry" if focus was lost mid-dwell
 * (same reasoning as collection), or null if abandoned. */
async function measureAtPoint(
  engine: GazeEngine,
  dot: HTMLButtonElement,
  nx: number,
  ny: number,
  signal: AbortSignal
): Promise<PointMeasurement | "retry" | null> {
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
  const offsetsX: number[] = [];
  const offsetsY: number[] = [];
  const off = engine.onGaze((sample) => {
    offsetsX.push(sample.x - x);
    offsetsY.push(sample.y - y);
  });
  await sleep(DWELL_MS);
  off();

  window.removeEventListener("blur", onBlur);
  stopRing();
  dot.classList.remove("is-active");

  if (signal.aborted) return null;
  if (blurred) return "retry";
  if (offsetsX.length === 0) return { error: Infinity, dx: 0, dy: 0, offsets: [] };

  return {
    // The typical distance of one sample, not the distance of the typical
    // sample: a recording stores individual gaze points, so this is the figure
    // that describes them. Taking the magnitude first is deliberate.
    error: medianOf(offsetsX.map((dx, i) => Math.hypot(dx, offsetsY[i]))),
    dx: medianOf(offsetsX),
    dy: medianOf(offsetsY),
    offsets: offsetsX.map((dx, i) => [dx, offsetsY[i]] as [number, number]),
  };
}

/** Median rather than mean throughout this file: a single blink-adjacent
 * outlier should not decide whether we tell the researcher their calibration
 * is good, nor how far it leans. */
function medianOf(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
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

/**
 * An offset smaller than this is not worth mentioning, in CSS pixels.
 *
 * Below roughly a quarter of a degree the correction is inside the scatter it
 * was measured through, and saying "corrected a 4px lean" invites a researcher
 * to read precision into a number that is noise.
 */
const NOTABLE_BIAS_PX = 18;

/**
 * Says which way a calibration leaned, in the terms a participant can act on.
 *
 * This is the half of the accuracy check that the single averaged number could
 * never carry. "~2.9°" tells a researcher how wrong the gaze is; it cannot tell
 * them that all of it is one direction, which is both the most alarming kind of
 * wrong — the heatmap is the right shape in the wrong place — and the only kind
 * that can simply be subtracted. Returns null when there is nothing to say.
 */
export function describeBias(bias: BiasEstimate | null): string | null {
  if (!bias || !Number.isFinite(bias.x) || !Number.isFinite(bias.y)) return null;

  const magnitude = Math.hypot(bias.x, bias.y);
  if (!bias.correctable) {
    return (
      `Gaze is landing ${Math.round(magnitude)}px from where you look — too far off to correct. ` +
      `Recalibrate rather than trusting this.`
    );
  }
  if (magnitude < NOTABLE_BIAS_PX) return null;

  // Named from the participant's point of view: where the estimate was
  // landing, not the sign of the residual that produced it.
  const parts: string[] = [];
  if (Math.abs(bias.y) >= NOTABLE_BIAS_PX / 2) {
    parts.push(`${Math.round(Math.abs(bias.y))}px ${bias.y > 0 ? "below" : "above"}`);
  }
  if (Math.abs(bias.x) >= NOTABLE_BIAS_PX / 2) {
    parts.push(`${Math.round(Math.abs(bias.x))}px ${bias.x > 0 ? "right of" : "left of"}`);
  }

  return (
    `Gaze was landing ${parts.join(" and ")} where you looked. ` +
    `That constant offset has been measured and subtracted.`
  );
}
