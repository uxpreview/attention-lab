import type { CalibrationSample, GazeEngine } from "../tracker/gaze";
import { el, sleep } from "./dom";

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

export interface CalibrationOutcome {
  cancelled: boolean;
  /** Mean validation error in CSS pixels, or null if validation was skipped. */
  validationError: number | null;
  /** Per-point validation errors, for the quality readout. */
  pointErrors: number[];
}

export async function runCalibration(
  engine: GazeEngine,
  host: HTMLElement
): Promise<CalibrationOutcome> {
  const overlay = el("div", { class: "calib-overlay" });
  const instruction = el(
    "div",
    { class: "calib-instruction" },
    el("h2", {}, "Calibration"),
    el(
      "p",
      {},
      "Keep your head still and your face lit from the front. Look at each dot and click it — keep looking until it stops pulsing."
    ),
    el("p", { class: "calib-progress" }, `0 / ${CALIBRATION_POINTS.length}`)
  );
  const dot = el("button", { class: "calib-dot", type: "button" });
  overlay.append(instruction, dot);
  host.append(overlay);

  const progress = instruction.querySelector(".calib-progress") as HTMLElement;
  const samples: CalibrationSample[] = [];
  let cancelled = false;

  const onKey = (event: KeyboardEvent) => {
    if (event.key === "Escape") cancelled = true;
  };
  window.addEventListener("keydown", onKey);

  try {
    for (let i = 0; i < CALIBRATION_POINTS.length; i++) {
      if (cancelled) break;
      const [nx, ny] = CALIBRATION_POINTS[i];
      progress.textContent = `${i} / ${CALIBRATION_POINTS.length}`;
      const collected = await collectAtPoint(engine, dot, nx, ny, samples);
      if (!collected) {
        cancelled = true;
        break;
      }
      // Hide the instruction panel after the first point so it stops competing
      // for attention with the dot the participant is meant to be looking at.
      if (i === 0) instruction.classList.add("is-dim");
    }

    if (cancelled) return { cancelled: true, validationError: null, pointErrors: [] };

    progress.textContent = "Fitting model…";
    engine.calibrate(samples);

    instruction.classList.remove("is-dim");
    (instruction.querySelector("h2") as HTMLElement).textContent = "Accuracy check";
    (instruction.querySelector("p") as HTMLElement).textContent =
      "Five more dots. These measure how accurate the calibration actually is.";

    const pointErrors: number[] = [];
    for (let i = 0; i < VALIDATION_POINTS.length; i++) {
      if (cancelled) break;
      const [nx, ny] = VALIDATION_POINTS[i];
      progress.textContent = `${i} / ${VALIDATION_POINTS.length}`;
      const error = await measureAtPoint(engine, dot, nx, ny);
      if (error === null) {
        cancelled = true;
        break;
      }
      pointErrors.push(error);
      if (i === 0) instruction.classList.add("is-dim");
    }

    if (cancelled || pointErrors.length === 0) {
      return { cancelled, validationError: null, pointErrors };
    }

    const mean = pointErrors.reduce((a, b) => a + b, 0) / pointErrors.length;
    return { cancelled: false, validationError: mean, pointErrors };
  } finally {
    window.removeEventListener("keydown", onKey);
    engine.stopCollecting();
    overlay.remove();
  }
}

function placeDot(dot: HTMLElement, nx: number, ny: number): { x: number; y: number } {
  const x = nx * window.innerWidth;
  const y = ny * window.innerHeight;
  dot.style.left = `${x}px`;
  dot.style.top = `${y}px`;
  return { x, y };
}

/** Resolves false if the participant abandoned the point (window blur, escape). */
async function collectAtPoint(
  engine: GazeEngine,
  dot: HTMLButtonElement,
  nx: number,
  ny: number,
  into: CalibrationSample[]
): Promise<boolean> {
  const { x, y } = placeDot(dot, nx, ny);
  dot.classList.remove("is-active");

  const clicked = await waitForClick(dot);
  if (!clicked) return false;

  dot.classList.add("is-active");
  await sleep(SETTLE_MS);

  engine.startCollecting(x, y, into);
  await sleep(DWELL_MS);
  engine.stopCollecting();

  dot.classList.remove("is-active");
  return true;
}

/** Returns the mean gaze error at this point in pixels, or null if abandoned. */
async function measureAtPoint(
  engine: GazeEngine,
  dot: HTMLButtonElement,
  nx: number,
  ny: number
): Promise<number | null> {
  const { x, y } = placeDot(dot, nx, ny);
  dot.classList.remove("is-active");

  const clicked = await waitForClick(dot);
  if (!clicked) return null;

  dot.classList.add("is-active");
  await sleep(SETTLE_MS);

  const errors: number[] = [];
  const off = engine.onGaze((sample) => {
    errors.push(Math.hypot(sample.x - x, sample.y - y));
  });
  await sleep(DWELL_MS);
  off();
  dot.classList.remove("is-active");

  if (errors.length === 0) return Infinity;
  // Median rather than mean: a single blink-adjacent outlier should not decide
  // whether we tell the researcher their calibration is good.
  errors.sort((a, b) => a - b);
  return errors[Math.floor(errors.length / 2)];
}

function waitForClick(dot: HTMLButtonElement): Promise<boolean> {
  return new Promise((resolve) => {
    const onClick = () => {
      cleanup();
      resolve(true);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        cleanup();
        resolve(false);
      }
    };
    const cleanup = () => {
      dot.removeEventListener("click", onClick);
      window.removeEventListener("keydown", onKey);
    };
    dot.addEventListener("click", onClick);
    window.addEventListener("keydown", onKey);
  });
}

/**
 * Turns validation error into language a researcher can act on. The thresholds
 * are in degrees of visual angle at an assumed 60cm viewing distance, which is
 * the unit eye-tracking literature uses; commercial trackers claim 0.5°, and
 * webcam tracking lands around 1.5-3° on a good day.
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
  if (degrees < 2) {
    return {
      grade: "good",
      label: `Good — ~${degrees.toFixed(1)}° (${Math.round(errorPx)}px)`,
      detail: "Reliable enough for region-level conclusions on a wireframe.",
    };
  }
  if (degrees < 4) {
    return {
      grade: "usable",
      label: `Usable — ~${degrees.toFixed(1)}° (${Math.round(errorPx)}px)`,
      detail: "Good for big blocks. Do not read individual words or small links from this.",
    };
  }
  return {
    grade: "poor",
    label: `Poor — ~${degrees.toFixed(1)}° (${Math.round(errorPx)}px)`,
    detail: "Recalibrate: improve lighting, sit square to the screen, and stay still.",
  };
}

/**
 * Converts pixels to approximate degrees of visual angle. Assumes ~96 CSS ppi
 * and a 60cm viewing distance — both are rough, which is why the UI always
 * shows the raw pixel figure alongside.
 */
export function pxToDegrees(px: number, viewingDistanceCm = 60): number {
  const cm = (px / 96) * 2.54;
  return (Math.atan2(cm, viewingDistanceCm) * 180) / Math.PI;
}
