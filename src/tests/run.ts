/**
 * Headless checks for the parts that can be wrong silently.
 *
 * The webcam path needs a browser, but the maths underneath it does not, and a
 * gaze model that is subtly wrong still produces a confident-looking heatmap.
 * These run a simulated eye through the real fitting code and assert that the
 * pipeline recovers a mapping it has never seen.
 *
 * Run with: npm test
 */

import { detectFixations, summarise } from "../analysis/fixations";
import {
  contourBandColours,
  DEFAULT_KERNEL_RATIO,
  fieldCeiling,
  fieldPercentile,
  kernelRatio,
  MAX_KERNEL_RATIO,
  MAX_KERNEL_SIGMA_PX,
  MIN_KERNEL_SIGMA_PX,
  paintField,
  rampColour,
  renderHeatmap,
  SPOTLIGHT_MAX_DIM,
} from "../analysis/heatmap";
import { legendFor, participantColour, PARTICIPANT_COLOURS } from "../analysis/legend";
import { gradeError, gradeRecording, gradeTracking, isLowSignal } from "../analysis/quality";
import {
  CIRCLE_ALPHA,
  layoutOrdinals,
  ORDINAL_BUDGET,
  ORDINAL_HALO,
  planOrdinals,
  scanpathColour,
  selectOrdinals,
  type OrdinalLabel,
} from "../analysis/scanpath";
import { analyseAois, aggregateAois, type Aoi } from "../analysis/aoi";
import {
  chromeCanContaminate,
  controlBandHeight,
  CONTROL_STRIP_PX,
  EDGE_TOLERANCE,
  FIT_SCALE_FLOOR,
  fitStimulus,
  isInsideViewport,
} from "../ui/record";
import { formatOnset, SAMPLE_PERIOD_MS } from "../ui/dom";
import {
  AUTO_FIT_RATIO,
  scopeCaption,
  scopeNote,
  scopePill,
  scopeSentence,
  shouldFitWidth,
  stageCap,
  STAGE_MIN_HEIGHT,
  type ReportedScope,
} from "../ui/results";
import type { RecordingQuality } from "../data/types";
import { buildFeatureVector, FEATURE_DIM, type FaceState } from "../tracker/features";
import { MedianPoint, OneEuroPoint } from "../tracker/filter";
import {
  deserialiseModel,
  fitRidge,
  isSerialisedModel,
  MAX_CORRECTABLE_BIAS_PX,
  measureBias,
  predict,
  serialiseModel,
  withBias,
  type ResidualSample,
} from "../tracker/regression";
import { describeBias } from "../ui/calibration";

let failures = 0;
let checks = 0;

function check(name: string, condition: boolean, detail = ""): void {
  checks++;
  if (condition) {
    console.log(`  ok   ${name}${detail ? ` — ${detail}` : ""}`);
  } else {
    failures++;
    console.log(`  FAIL ${name}${detail ? ` — ${detail}` : ""}`);
  }
}

function section(name: string): void {
  console.log(`\n${name}`);
}

/** The four channels of a ramp stop, read back out of the CSS string the
 * legend is built from — so these assertions are made against exactly what the
 * legend promises, not against a private copy of the table. */
function rampParts(t: number): [number, number, number, number] {
  const parts = rampColour(t).match(/[\d.]+/g);
  if (!parts || parts.length < 4) throw new Error(`rampColour(${t}) is not rgba: ${rampColour(t)}`);
  return [Number(parts[0]), Number(parts[1]), Number(parts[2]), Number(parts[3])];
}
function rampRgb(t: number): [number, number, number] {
  const [r, g, b] = rampParts(t);
  return [r, g, b];
}

/**
 * Just enough of a canvas for `renderHeatmap` to draw onto in Node: it clears,
 * makes an ImageData and puts it back, and nothing else. The point is to measure
 * what the real renderer paints rather than to re-implement it — see the kernel
 * section, which reads the alpha channel back out of `pixels`.
 */
function stubCanvas(width: number, height: number): HTMLCanvasElement {
  const pixels = new Uint8ClampedArray(width * height * 4);
  const context = {
    clearRect: () => pixels.fill(0),
    createImageData: (w: number, h: number) => ({
      width: w,
      height: h,
      data: new Uint8ClampedArray(w * h * 4),
    }),
    putImageData: (image: { data: Uint8ClampedArray }) => pixels.set(image.data),
  };
  return {
    width,
    height,
    pixels,
    getContext: () => context,
  } as unknown as HTMLCanvasElement;
}
function rampAlpha(t: number): number {
  return rampParts(t)[3];
}

/** Deterministic PRNG so a failure is reproducible. */
function makeRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

const SCREEN_W = 1440;
const SCREEN_H = 900;

/**
 * A forward model of an eye looking at a screen point, deliberately non-linear
 * and head-pose coupled, so it exercises the interaction terms rather than
 * flattering a purely linear fit.
 */
function simulateFace(
  targetX: number,
  targetY: number,
  head: { yaw: number; pitch: number; x: number; y: number; scale: number },
  noise: number,
  rand: () => number
): FaceState {
  // Screen position in [-0.5, 0.5], relative to screen centre.
  const sx = targetX / SCREEN_W - 0.5;
  const sy = targetY / SCREEN_H - 0.5;

  // Eye rotation needed to hit that point, minus whatever the head already
  // contributes — this is the coupling the model has to learn.
  const eyeYaw = Math.atan2(sx * 0.6, 0.55) - head.yaw;
  const eyePitch = Math.atan2(sy * 0.38, 0.55) - head.pitch;

  // Iris offset is roughly sin of eye rotation, scaled by apparent eye size.
  const gain = 0.42 / head.scale;
  const jitter = () => (rand() - 0.5) * noise;
  const offsetX = Math.sin(eyeYaw) * gain;
  const offsetY = Math.sin(eyePitch) * gain;

  // The two eyes differ slightly through vergence. Landmark noise is applied
  // per eye, not shared: the mesh jitters each iris independently, which is
  // what keeps the vergence signal from being implausibly clean.
  const eye = (dx: number, dy: number) => ({
    iris: { x: 0.5 + dx, y: 0.5 + dy },
    offset: { x: dx + jitter(), y: dy + jitter() },
    openness: 0.32,
  });

  return {
    left: eye(offsetX + 0.012 * sx, offsetY),
    right: eye(offsetX - 0.012 * sx, offsetY),
    yaw: head.yaw,
    pitch: head.pitch,
    roll: 0,
    headX: head.x,
    headY: head.y,
    scale: head.scale,
    openness: 0.32,
  };
}

const CALIBRATION_GRID: Array<[number, number]> = [
  [0.08, 0.08], [0.5, 0.06], [0.92, 0.08],
  [0.06, 0.5], [0.5, 0.5], [0.94, 0.5],
  [0.08, 0.92], [0.5, 0.94], [0.92, 0.92],
  [0.28, 0.28], [0.72, 0.28], [0.28, 0.72], [0.72, 0.72],
];

interface FitResult {
  meanError: number;
  p95Error: number;
  model: ReturnType<typeof fitRidge>;
}

const BASE_HEAD = { yaw: 0.02, pitch: -0.03, x: 0.5, y: 0.48, scale: 0.62 };

/** Simulates a full calibration run: bursts of frames per dot, with small
 * natural head movement throughout. */
function buildCalibrationData(
  noise: number,
  rand: () => number
): { rows: number[][]; targetX: number[]; targetY: number[] } {
  const rows: number[][] = [];
  const targetX: number[] = [];
  const targetY: number[] = [];

  for (const [nx, ny] of CALIBRATION_GRID) {
    for (let s = 0; s < 22; s++) {
      const head = {
        yaw: BASE_HEAD.yaw + (rand() - 0.5) * 0.06,
        pitch: BASE_HEAD.pitch + (rand() - 0.5) * 0.05,
        x: BASE_HEAD.x + (rand() - 0.5) * 0.02,
        y: BASE_HEAD.y + (rand() - 0.5) * 0.02,
        scale: BASE_HEAD.scale + (rand() - 0.5) * 0.01,
      };
      const face = simulateFace(nx * SCREEN_W, ny * SCREEN_H, head, noise, rand);
      rows.push(buildFeatureVector(face));
      targetX.push(nx * SCREEN_W);
      targetY.push(ny * SCREEN_H);
    }
  }

  return { rows, targetX, targetY };
}

/**
 * Runs a full calibrate-then-test cycle. `drift` simulates the participant's
 * head moving between calibration and the recording, which is the single
 * biggest source of real-world degradation.
 */
function calibrateAndTest(noise: number, drift: number, seed: number): FitResult {
  const rand = makeRandom(seed);
  const { rows, targetX, targetY } = buildCalibrationData(noise, rand);

  const model = fitRidge(rows, targetX, targetY);

  // Test on points the model never saw, with the head shifted by `drift`.
  const errors: number[] = [];
  for (let i = 0; i < 300; i++) {
    const tx = (0.08 + rand() * 0.84) * SCREEN_W;
    const ty = (0.08 + rand() * 0.84) * SCREEN_H;
    const head = {
      yaw: BASE_HEAD.yaw + (rand() - 0.5) * drift,
      pitch: BASE_HEAD.pitch + (rand() - 0.5) * drift * 0.8,
      x: BASE_HEAD.x + (rand() - 0.5) * drift * 0.3,
      y: BASE_HEAD.y + (rand() - 0.5) * drift * 0.3,
      scale: BASE_HEAD.scale + (rand() - 0.5) * drift * 0.15,
    };
    const face = simulateFace(tx, ty, head, noise, rand);
    const [px, py] = predict(model, buildFeatureVector(face));
    errors.push(Math.hypot(px - tx, py - ty));
  }

  errors.sort((a, b) => a - b);
  return {
    meanError: errors.reduce((a, b) => a + b, 0) / errors.length,
    p95Error: errors[Math.floor(errors.length * 0.95)],
    model,
  };
}

// --- Feature basis -------------------------------------------------------

section("Feature extraction");
{
  const rand = makeRandom(7);
  const face = simulateFace(700, 400, { yaw: 0, pitch: 0, x: 0.5, y: 0.5, scale: 0.62 }, 0, rand);
  const vector = buildFeatureVector(face);
  check("feature vector length matches FEATURE_DIM", vector.length === FEATURE_DIM, `${vector.length}`);
  check("feature vector is all finite", vector.every(Number.isFinite));

  // Looking right should move the irises right relative to looking left.
  const left = buildFeatureVector(
    simulateFace(150, 450, { yaw: 0, pitch: 0, x: 0.5, y: 0.5, scale: 0.62 }, 0, rand)
  );
  const right = buildFeatureVector(
    simulateFace(1290, 450, { yaw: 0, pitch: 0, x: 0.5, y: 0.5, scale: 0.62 }, 0, rand)
  );
  // dx and dy sit at indices 0 and 1 of the basis.
  check("horizontal gaze moves the iris offset monotonically", right[0] > left[0]);

  const up = buildFeatureVector(
    simulateFace(720, 80, { yaw: 0, pitch: 0, x: 0.5, y: 0.5, scale: 0.62 }, 0, rand)
  );
  const down = buildFeatureVector(
    simulateFace(720, 820, { yaw: 0, pitch: 0, x: 0.5, y: 0.5, scale: 0.62 }, 0, rand)
  );
  check("vertical gaze moves the iris offset monotonically", down[1] > up[1]);
}

// --- Regression ----------------------------------------------------------

section("Ridge solver");
{
  // Ground truth for the solver itself: a noiseless linear system the fit must
  // reproduce almost exactly. The end-to-end thresholds below are loose enough
  // to hide a subtle sign or indexing error; this is not.
  const rand = makeRandom(31);
  const rows: number[][] = [];
  const tx: number[] = [];
  const ty: number[] = [];
  for (let i = 0; i < 80; i++) {
    const x1 = rand() * 4 - 2;
    const x2 = rand() * 4 - 2;
    rows.push([x1, x2]);
    tx.push(3 * x1 - 2 * x2 + 5);
    ty.push(-x1 + 4 * x2 - 7);
  }
  const model = fitRidge(rows, tx, ty);

  let worst = 0;
  for (let i = 0; i < 20; i++) {
    const x1 = rand() * 4 - 2;
    const x2 = rand() * 4 - 2;
    const [px, py] = predict(model, [x1, x2]);
    worst = Math.max(worst, Math.abs(px - (3 * x1 - 2 * x2 + 5)), Math.abs(py - (-x1 + 4 * x2 - 7)));
  }
  check("recovers an exact linear system", worst < 0.05, `worst error ${worst.toExponential(2)}`);
}

section("Degenerate calibration data");
{
  const rand = makeRandom(41);
  const randomRow = () => Array.from({ length: FEATURE_DIM }, () => rand());

  // Fewer samples than features: no amount of ridge makes that a gaze model.
  const n = FEATURE_DIM - 1;
  let threw = false;
  try {
    fitRidge(
      Array.from({ length: n }, randomRow),
      Array.from({ length: n }, () => rand() * 1000),
      Array.from({ length: n }, () => rand() * 800)
    );
  } catch {
    threw = true;
  }
  check("refuses an underdetermined system", threw);

  // Just past the floor: every CV fold's training split is <= FEATURE_DIM rows
  // and gets skipped, so lambda must come from the fallback — and it must be a
  // regularising value, not the weakest entry on the grid.
  const rows: number[][] = [];
  const targetX: number[] = [];
  const targetY: number[] = [];
  for (let i = 0; i < FEATURE_DIM + 4; i++) {
    rows.push(randomRow());
    targetX.push(i % 2 === 0 ? 200 : 1200);
    targetY.push(450);
  }
  const model = fitRidge(rows, targetX, targetY);
  check("falls back to a mid-grid lambda when no CV fold can run", model.lambda === 1, `lambda=${model.lambda}`);
  check("reports cvError as NaN rather than a fabricated number", Number.isNaN(model.cvError));
  check("still produces finite weights", Array.from(model.wx).every(Number.isFinite));
}

section("Gaze model fitting");
{
  const clean = calibrateAndTest(0, 0.02, 11);
  check(
    "recovers a noiseless mapping to sub-degree accuracy",
    clean.meanError < 25,
    `mean ${clean.meanError.toFixed(1)}px`
  );

  // Noise of 0.004 in iris-offset units is around the jitter a 720p webcam
  // gives after the mesh has been fit, so this is the realistic case.
  const realistic = calibrateAndTest(0.004, 0.05, 12);
  check(
    "stays usable with realistic sensor noise and head movement",
    realistic.meanError < 90,
    `mean ${realistic.meanError.toFixed(1)}px, p95 ${realistic.p95Error.toFixed(1)}px`
  );

  const drifting = calibrateAndTest(0.004, 0.16, 13);
  check(
    "degrades gracefully rather than diverging under head drift",
    drifting.meanError < 220,
    `mean ${drifting.meanError.toFixed(1)}px`
  );

  check("cross-validation picked a finite lambda", Number.isFinite(realistic.model.lambda), `lambda=${realistic.model.lambda}`);
  check("all weights are finite", Array.from(realistic.model.wx).every(Number.isFinite));
  check("cross-validation reports a finite error", Number.isFinite(realistic.model.cvError), `${realistic.model.cvError.toFixed(1)}px`);

  // Grouped CV holding out whole calibration dots should see noise for what it
  // is. Interleaved folds leak near-duplicate neighbouring frames into the
  // training set and would let a noisy fit get away with weak regularisation.
  check(
    "noisier data is regularised at least as hard as clean data",
    realistic.model.lambda >= clean.model.lambda,
    `clean lambda=${clean.model.lambda}, noisy lambda=${realistic.model.lambda}`
  );

  // Leakage guard: refit the identical samples with every row's target nudged
  // to be unique, which collapses the grouping to per-row folds — the
  // interleaved assignment this codebase used to have. Because each held-out
  // frame then has near-duplicate neighbours in the training set, the reported
  // error flatters substantially. Grouped CV must not.
  const { rows, targetX, targetY } = buildCalibrationData(0.004, makeRandom(12));
  const grouped = fitRidge(rows, targetX, targetY);
  const leaky = fitRidge(rows, targetX.map((v, i) => v + i * 1e-9), targetY);
  check(
    "holding out whole dots reports honestly larger CV error than leaky folds",
    grouped.cvError > leaky.cvError * 1.3,
    `grouped ${grouped.cvError.toFixed(1)}px vs leaked ${leaky.cvError.toFixed(1)}px`
  );

  // Round-tripping matters: a persisted calibration that predicts differently
  // would be worse than not persisting at all.
  const restored = deserialiseModel(serialiseModel(realistic.model));
  const rand = makeRandom(99);
  const face = simulateFace(600, 500, { yaw: 0.01, pitch: 0, x: 0.5, y: 0.5, scale: 0.62 }, 0, rand);
  const features = buildFeatureVector(face);
  const [ax, ay] = predict(realistic.model, features);
  const [bx, by] = predict(restored, features);
  check("serialise/deserialise round-trips exactly", ax === bx && ay === by);
}

section("Stored calibration guard");
{
  // Persisted models arrive from session storage, which the type system
  // cannot vouch for. The guard has to be exactly as strict as predict()
  // needs: a dimension mismatch reads past the feature array (NaN, gaze
  // silently never emits) or misaligns every column (confidently wrong gaze).
  const rand = makeRandom(61);
  const rows: number[][] = [];
  const targetX: number[] = [];
  const targetY: number[] = [];
  for (let i = 0; i < 40; i++) {
    const a = rand() * 2 - 1;
    const b = rand() * 2 - 1;
    const c = rand() * 2 - 1;
    rows.push([a, b, c]);
    targetX.push(2 * a - b + 3);
    targetY.push(a + 4 * c - 1);
  }
  const data = serialiseModel(fitRidge(rows, targetX, targetY));
  const roundTripped: unknown = JSON.parse(JSON.stringify(data));

  check("accepts its own serialised form", isSerialisedModel(data, 3));
  check("accepts a JSON round-trip", isSerialisedModel(roundTripped, 3));
  check("rejects a model fit on a different basis size", !isSerialisedModel(data, 4));
  check("rejects a truncated weight vector", !isSerialisedModel({ ...data, wx: data.wx.slice(1) }, 3));
  check("rejects non-numeric entries", !isSerialisedModel({ ...data, std: [1, "1", 1] }, 3));
  check(
    "rejects payloads that are not models at all",
    !isSerialisedModel(null, 3) && !isSerialisedModel("{}", 3) && !isSerialisedModel({}, 3)
  );

  // Bias is optional so calibrations stored before it existed still load, but a
  // non-finite offset would silently displace every gaze point — the same
  // confidently-wrong failure the rest of this guard exists to catch.
  const { biasX: _x, biasY: _y, ...legacy } = data;
  check("accepts a calibration stored before bias correction existed", isSerialisedModel(legacy, 3));
  check(
    "and loads it with no correction rather than undefined",
    deserialiseModel(legacy as typeof data).biasX === 0
  );
  check("rejects a non-finite offset", !isSerialisedModel({ ...data, biasY: NaN }, 3));
  check("rejects a non-numeric offset", !isSerialisedModel({ ...data, biasX: "12" }, 3));
}

section("Constant offset: telling it apart from scatter, and taking it out");
{
  const at = (dx: number, dy: number, i = 0): ResidualSample => ({
    targetX: 200 + i * 200,
    targetY: 150 + i * 120,
    dx,
    dy,
  });

  // A model that leans: every dot missed the same way. This is the fault a
  // single averaged error cannot name and the only one that can be subtracted.
  const leaning = measureBias([at(40, -70, 0), at(38, -66, 1), at(44, -72, 2), at(39, -69, 3), at(41, -71, 4)]);
  check(
    "reads a consistent lean as an offset",
    Math.abs(leaning.x - 40) < 3 && Math.abs(leaning.y + 70) < 3,
    `(${leaning.x.toFixed(0)}, ${leaning.y.toFixed(0)})`
  );
  check("and reports the leftover spread as small", leaning.scatter < 6, `${leaning.scatter.toFixed(1)}px`);
  check("an offset this size is correctable", leaning.correctable);

  // The same mean error, arranged as scatter instead. Nothing here should be
  // subtracted: there is no direction to subtract.
  const scattered = measureBias([at(80, 0, 0), at(-80, 0, 1), at(0, 80, 2), at(0, -80, 3), at(0, 0, 4)]);
  check(
    "reads symmetric scatter as no offset at all",
    Math.hypot(scattered.x, scattered.y) < 12,
    `(${scattered.x.toFixed(0)}, ${scattered.y.toFixed(0)})`
  );
  check("but does report the spread", scattered.scatter > 40, `${scattered.scatter.toFixed(0)}px`);

  // Five dots is few enough that one blink or glance away would drag a mean
  // correction with it, which is why the offset is a component-wise median.
  const oneBadDot = measureBias([at(40, -70, 0), at(38, -66, 1), at(44, -72, 2), at(39, -69, 3), at(600, 900, 4)]);
  check(
    "one bad dot does not drag the correction",
    Math.abs(oneBadDot.x - 40) < 5 && Math.abs(oneBadDot.y + 70) < 5,
    `(${oneBadDot.x.toFixed(0)}, ${oneBadDot.y.toFixed(0)})`
  );

  // Past a point an offset is not a settled posture, it is a calibration that
  // has stopped describing this participant. Subtracting it would hand back
  // gaze that looks plausible and is not.
  const huge = MAX_CORRECTABLE_BIAS_PX + 60;
  check("refuses an offset too large to be posture", !measureBias([at(huge, 0), at(huge, 0), at(huge, 0)]).correctable);
  check("and says so rather than silently declining", (describeBias(measureBias([at(huge, 0), at(huge, 0), at(huge, 0)])) ?? "").includes("Recalibrate"));

  check("no residuals means no correction", !measureBias([]).correctable);

  // The wording is the participant-facing half: it has to name the direction
  // the gaze *landed*, not the sign of the residual behind it.
  check(
    "names the direction the gaze landed, not the sign of the residual",
    (describeBias(leaning) ?? "").includes("70px above"),
    describeBias(leaning) ?? "(nothing said)"
  );
  check("says nothing about an offset inside the noise", describeBias(measureBias([at(3, -4), at(2, -5), at(4, -3)])) === null);
}

section("Correcting the offset moves the gaze back");
{
  const rand = makeRandom(77);
  const { rows, targetX, targetY } = buildCalibrationData(0.004, rand);
  const model = fitRidge(rows, targetX, targetY);

  const features = buildFeatureVector(
    simulateFace(600, 500, { yaw: 0.01, pitch: 0, x: 0.5, y: 0.5, scale: 0.62 }, 0, rand)
  );
  const [rawX, rawY] = predict(model, features);
  const [shiftedX, shiftedY] = predict(withBias(model, { x: 30, y: -20 }), features);
  check(
    "an offset subtracts exactly, in both axes",
    Math.abs(shiftedX - (rawX - 30)) < 1e-9 && Math.abs(shiftedY - (rawY + 20)) < 1e-9
  );
  check("a fresh fit carries no offset, because the intercept absorbed it", model.biasX === 0 && model.biasY === 0);
  check(
    "an offset survives being persisted and reloaded",
    deserialiseModel(serialiseModel(withBias(model, { x: 30, y: -20 }))).biasY === -20
  );

  /**
   * The case this was all built for: a participant who calibrates, then settles
   * into their chair before recording. Their model is the right shape in the
   * wrong place, and until now nothing in the app could notice — reusing a
   * stored calibration carried its original accuracy figure forward untouched.
   */
  const settle = 0.12;
  const seated = (r: () => number) => ({
    yaw: 0.02 + (r() - 0.5) * 0.04,
    pitch: -0.03 + settle + (r() - 0.5) * 0.03,
    x: 0.5 + (r() - 0.5) * 0.012,
    y: 0.48 + settle * 0.35 + (r() - 0.5) * 0.012,
    scale: 0.62 + (r() - 0.5) * 0.005,
  });

  // The five-dot accuracy check, as the app runs it.
  const VALIDATION: Array<[number, number]> = [[0.2, 0.2], [0.8, 0.2], [0.5, 0.35], [0.2, 0.8], [0.8, 0.8]];
  const residuals: ResidualSample[] = VALIDATION.map(([nx, ny]) => {
    const tx = nx * SCREEN_W;
    const ty = ny * SCREEN_H;
    const dxs: number[] = [];
    const dys: number[] = [];
    for (let s = 0; s < 27; s++) {
      const [px, py] = predict(model, buildFeatureVector(simulateFace(tx, ty, seated(rand), 0.004, rand)));
      dxs.push(px - tx);
      dys.push(py - ty);
    }
    dxs.sort((a, b) => a - b);
    dys.sort((a, b) => a - b);
    return { targetX: tx, targetY: ty, dx: dxs[13], dy: dys[13] };
  });

  const bias = measureBias(residuals);
  check("the settle shows up as an offset worth correcting", bias.correctable && Math.hypot(bias.x, bias.y) > 20, `${Math.hypot(bias.x, bias.y).toFixed(0)}px`);

  const corrected = withBias(model, bias);
  let before = 0;
  let after = 0;
  const trials = 900;
  for (let i = 0; i < trials; i++) {
    const tx = (0.08 + rand() * 0.84) * SCREEN_W;
    const ty = (0.08 + rand() * 0.84) * SCREEN_H;
    const face = buildFeatureVector(simulateFace(tx, ty, seated(rand), 0.004, rand));
    const [ax, ay] = predict(model, face);
    const [bx, by] = predict(corrected, face);
    before += Math.hypot(ax - tx, ay - ty);
    after += Math.hypot(bx - tx, by - ty);
  }
  before /= trials;
  after /= trials;
  check(
    "correcting it measurably improves gaze the model never saw",
    after < before * 0.9,
    `${before.toFixed(0)}px → ${after.toFixed(0)}px`
  );
}

// --- Filter --------------------------------------------------------------

section("One Euro filter");
{
  const filter = new OneEuroPoint({ minCutoff: 0.9, beta: 0.012 });
  const rand = makeRandom(21);

  // A stationary noisy signal should come out much quieter than it went in.
  let rawSpread = 0;
  let filteredSpread = 0;
  let lastRaw = 500;
  let lastFiltered = 500;
  for (let i = 0; i < 200; i++) {
    const raw = 500 + (rand() - 0.5) * 40;
    const [fx] = filter.filter(raw, 400, i * 33);
    if (i > 20) {
      rawSpread += Math.abs(raw - lastRaw);
      filteredSpread += Math.abs(fx - lastFiltered);
    }
    lastRaw = raw;
    lastFiltered = fx;
  }
  check(
    "suppresses jitter while the eye is still",
    filteredSpread < rawSpread * 0.35,
    `${(filteredSpread / rawSpread).toFixed(2)}x residual`
  );

  // A step change should be tracked within a few frames, not smoothed away.
  const stepFilter = new OneEuroPoint({ minCutoff: 0.9, beta: 0.012 });
  let t = 0;
  for (let i = 0; i < 30; i++) stepFilter.filter(200, 200, (t += 33));
  let final = 200;
  for (let i = 0; i < 12; i++) [final] = stepFilter.filter(900, 200, (t += 33));
  check("follows a saccade rather than smearing it", final > 780, `reached ${final.toFixed(0)} of 900`);
}

// --- Fixation detection --------------------------------------------------

section("Fixation detection");
{
  // Three fixations with saccades between them.
  const points: Array<{ x: number; y: number; t: number }> = [];
  const rand = makeRandom(5);
  let t = 0;
  const addFixation = (x: number, y: number, ms: number) => {
    for (let i = 0; i < ms / 33; i++) {
      points.push({ x: x + (rand() - 0.5) * 12, y: y + (rand() - 0.5) * 12, t: (t += 33) });
    }
  };
  const addSaccade = (from: [number, number], to: [number, number]) => {
    for (let i = 1; i <= 3; i++) {
      points.push({
        x: from[0] + ((to[0] - from[0]) * i) / 4,
        y: from[1] + ((to[1] - from[1]) * i) / 4,
        t: (t += 33),
      });
    }
  };

  addFixation(200, 200, 300);
  addSaccade([200, 200], [800, 300]);
  addFixation(800, 300, 400);
  addSaccade([800, 300], [500, 700]);
  addFixation(500, 700, 250);

  const fixations = detectFixations(points, { dispersion: 45, minDuration: 100 });
  check("finds exactly the three simulated fixations", fixations.length === 3, `found ${fixations.length}`);

  if (fixations.length === 3) {
    check("first fixation is at the right place", Math.hypot(fixations[0].x - 200, fixations[0].y - 200) < 20);
    check("second fixation is at the right place", Math.hypot(fixations[1].x - 800, fixations[1].y - 300) < 20);
    check("third fixation is at the right place", Math.hypot(fixations[2].x - 500, fixations[2].y - 700) < 20);
    check(
      "durations are approximately right",
      fixations[0].duration > 230 && fixations[1].duration > 330,
      `${Math.round(fixations[0].duration)}ms, ${Math.round(fixations[1].duration)}ms`
    );
    check("fixations are ordered in time", fixations[0].start < fixations[1].start);
  }

  const stats = summarise(fixations, 0);
  check("summary counts match", stats.fixationCount === fixations.length);
  check("scanpath length is non-zero", stats.scanpathLength > 500, `${Math.round(stats.scanpathLength)}px`);

  // Pure noise should not manufacture fixations.
  const noisy: Array<{ x: number; y: number; t: number }> = [];
  for (let i = 0; i < 200; i++) {
    noisy.push({ x: rand() * 1400, y: rand() * 900, t: i * 33 });
  }
  check(
    "does not invent fixations from random gaze",
    detectFixations(noisy, { dispersion: 45, minDuration: 100 }).length === 0
  );

  check("handles an empty recording", detectFixations([]).length === 0);

  // A tracking dropout inside a stationary dwell: maxGap decides whether the
  // dwell is split into two fixations or bridged into one.
  const interrupted: Array<{ x: number; y: number; t: number }> = [];
  let dropT = 0;
  for (let i = 0; i < 9; i++) {
    interrupted.push({ x: 400 + (rand() - 0.5) * 10, y: 300 + (rand() - 0.5) * 10, t: (dropT += 33) });
  }
  dropT += 200; // dropout: next sample arrives 233ms after the previous one
  for (let i = 0; i < 9; i++) {
    interrupted.push({ x: 400 + (rand() - 0.5) * 10, y: 300 + (rand() - 0.5) * 10, t: (dropT += 33) });
  }
  check(
    "a dropout longer than maxGap splits a dwell in two",
    detectFixations(interrupted, { dispersion: 45, minDuration: 100, maxGap: 150 }).length === 2
  );
  check(
    "a generous maxGap bridges the same dropout",
    detectFixations(interrupted, { dispersion: 45, minDuration: 100, maxGap: 300 }).length === 1
  );
}

// --- Despike filter ------------------------------------------------------

section("Median despike");
{
  // A single-frame blink artifact — the eyelid dragging the iris centroid down
  // for one frame — must be deleted, not smoothed into a downward saccade.
  const median = new MedianPoint();
  median.filter(500, 400);
  median.filter(502, 401);
  const [, spikeY] = median.filter(505, 700); // the artifact frame
  const [, afterY] = median.filter(503, 402);
  check("a one-frame spike never reaches the output", spikeY < 410, `${spikeY.toFixed(0)}px`);
  check("output returns to the true position after the spike", afterY < 410, `${afterY.toFixed(0)}px`);

  // A genuine saccade persists across frames and must pass through.
  const saccade = new MedianPoint();
  saccade.filter(200, 300);
  saccade.filter(201, 301);
  saccade.filter(800, 300);
  const [followX] = saccade.filter(802, 301);
  check("a sustained jump passes through with one frame of lag", followX > 780, `${followX.toFixed(0)}px`);
}

// --- Heatmap scaling -----------------------------------------------------

section("Heatmap percentile ceiling");
{
  const field = new Float32Array(100);
  for (let i = 0; i < 100; i++) field[i] = i + 1;
  const median = fieldPercentile(field, 0.5);
  const p98 = fieldPercentile(field, 0.98);
  check("finds the median of a uniform ramp", Math.abs(median - 50) < 1, `${median.toFixed(1)}`);
  check("finds a high percentile of a uniform ramp", Math.abs(p98 - 98) < 1, `${p98.toFixed(1)}`);
  check("zeroes are excluded from the distribution", fieldPercentile(Float32Array.of(0, 0, 0, 8), 0.5) > 7);
  check("an empty field yields a zero ceiling", fieldPercentile(new Float32Array(16), 0.98) === 0);
}

// --- The display ceiling -------------------------------------------------

section("Heatmap display ceiling");
{
  // Eight clusters in a row, one of them five times the rest: the seeded-study
  // case that rendered as a single red blob and seven pure-blue ones under a
  // legend reading "barely looked at". The old ceiling was a percentile of the
  // non-zero *pixels*, which the dominant blob sets by itself.
  const W = 400;
  const H = 40;
  const radius = 16;
  const sigmaSq = 2 * (radius / 2) * (radius / 2);
  const field = new Float32Array(W * H);
  const weights = [5, 1, 1, 1, 1, 1, 1, 1];
  weights.forEach((weight, i) => {
    const cx = 25 + i * 50;
    const cy = 20;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
        if (d2 > radius * radius) continue;
        field[y * W + x] += weight * Math.exp(-d2 / sigmaSq);
      }
    }
  });

  let max = 0;
  for (let i = 0; i < field.length; i++) max = Math.max(max, field[i]);
  const secondary = max / 5;

  const pixelP98 = fieldPercentile(field, 0.98);
  const ceiling = fieldCeiling(field, W, H, radius, 0.9);

  // The old clamp: it lands so close to the dominant peak that every other
  // cluster is pushed into the bottom third of the ramp — the cold end the
  // legend labels "barely looked at".
  check(
    "the old pixel percentile barely clamped the dominant blob",
    pixelP98 > max * 0.6,
    `${((pixelP98 / max) * 100).toFixed(0)}% of the peak`
  );
  check(
    "the old pixel percentile froze the other clusters out",
    secondary / pixelP98 < 0.3,
    `${((secondary / pixelP98) * 100).toFixed(0)}% of the old ceiling`
  );
  check(
    "the blob ceiling sits below the dominant peak",
    ceiling < max * 0.7,
    `${((ceiling / max) * 100).toFixed(0)}% of the peak`
  );
  check(
    "the dominant blob still saturates the ramp",
    max / ceiling >= 1,
    `${(max / ceiling).toFixed(2)}x the ceiling`
  );
  check(
    "the other clusters reach the warm half of the ramp",
    secondary / ceiling > 0.4,
    `${((secondary / ceiling) * 100).toFixed(0)}% of the ceiling`
  );

  // One cluster on its own has no other blob to be measured against, and its
  // own peak is the only honest ceiling there is.
  const lone = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const d2 = (x - 200) * (x - 200) + (y - 20) * (y - 20);
      if (d2 <= radius * radius) lone[y * W + x] = Math.exp(-d2 / sigmaSq);
    }
  }
  const loneCeiling = fieldCeiling(lone, W, H, radius, 0.9);
  check("a single cluster still scales to its own peak", Math.abs(loneCeiling - 1) < 0.01, `${loneCeiling.toFixed(3)}`);

  check("an empty field yields a zero ceiling", fieldCeiling(new Float32Array(64), 8, 8, 4) === 0);

  // What the legend is allowed to print.
  //
  // The strip under the stage now carries millisecond values, and the only
  // defensible source for them is the renderer itself — the ceiling is a
  // percentile of this selection's own blob peaks, so anything recomputed
  // beside it would be a second opinion. renderHeatmap hands it back in the
  // units the weights came in, which is what makes "≥1.2s" a fact about the
  // picture rather than a label stuck under it.
  const shim = {
    width: 200,
    height: 200,
    getContext: () => ({
      clearRect: () => {},
      createImageData: (w: number, h: number) => ({ data: new Uint8ClampedArray(w * h * 4) }),
      putImageData: () => {},
    }),
  } as unknown as HTMLCanvasElement;

  // One fixation, alone on the stimulus: the hot end of the ramp is worth
  // exactly that fixation, so the legend can print its duration. The kernel's
  // rim subtraction would otherwise report it 13.5% low.
  const dwell = 900;
  const returned = renderHeatmap(shim, [{ x: 0.5, y: 0.5, weight: dwell }], { radiusRatio: 0.055 });
  check(
    "the renderer hands back its ceiling in the weights' own units",
    Math.abs(returned - dwell) / dwell < 0.02,
    `${returned.toFixed(0)} for a ${dwell}ms fixation`
  );
  // Two fixations in one spot are two fixations' worth of looking, which is
  // what makes the axis a duration axis rather than an index.
  const doubled = renderHeatmap(
    shim,
    [
      { x: 0.5, y: 0.5, weight: dwell },
      { x: 0.5, y: 0.5, weight: dwell },
    ],
    { radiusRatio: 0.055 }
  );
  check(
    "stacked dwell sums on the axis",
    Math.abs(doubled - 2 * dwell) / (2 * dwell) < 0.02,
    `${doubled.toFixed(0)} for two ${dwell}ms fixations`
  );
  check(
    "nothing to draw means no scale to print",
    renderHeatmap(shim, []) === 0
  );
}

// --- Overlay painting ----------------------------------------------------

section("Heatmap overlay painting");
{
  // A splat's rim: hot centre, then a tail that fades to nothing. The tail is
  // where a spotlight can grow holes, so the field deliberately includes values
  // that round to zero intensity.
  const field = Float32Array.from([100, 40, 6, 0.2, 0.02, 0]);
  const scale = 255 / 100;
  const alphaOf = (dst: Uint8ClampedArray, i: number): number => dst[i * 4 + 3];

  const spotlight = new Uint8ClampedArray(field.length * 4);
  paintField(field, spotlight, "spotlight", scale, 0.72);

  let dimmed = true;
  let monotonic = true;
  for (let i = 1; i < field.length; i++) {
    // Index 0 is hot enough to be fully revealed, which is the point of it.
    if (alphaOf(spotlight, i) === 0) dimmed = false;
    if (alphaOf(spotlight, i) < alphaOf(spotlight, i - 1)) monotonic = false;
  }

  // The regression: pixels holding a splat too faint to round up to one unit of
  // intensity were skipped entirely, leaving them transparent — an undimmed
  // halo punched through the mask around every hot spot.
  check("spotlight dims a splat's faint rim rather than punching a hole", dimmed);
  check("spotlight dim rises as attention falls", monotonic);
  check("spotlight fully reveals its hottest point", alphaOf(spotlight, 0) === 0);
  check("spotlight dims unlooked regions without going opaque", alphaOf(spotlight, 5) < 255);
  // A revealed pixel returns the stimulus at 100% and never more: the mask can
  // only give back dimming it added, so no amount of attention can brighten the
  // picture past what was uploaded.
  check(
    "the mask never subtracts past zero",
    Array.from({ length: field.length }, (_, i) => alphaOf(spotlight, i)).every((a) => a >= 0)
  );
  // And the floor is a dim, not a blackout. At 232/255 the unlooked page was
  // gone, so the reveals read as glows in a void with nothing to place them
  // against — an opacity map has to leave the screen legible as context.
  check(
    "unlooked content stays legible as context",
    SPOTLIGHT_MAX_DIM <= 200,
    `${SPOTLIGHT_MAX_DIM}/255 dim`
  );

  const heat = new Uint8ClampedArray(field.length * 4);
  paintField(field, heat, "heat", scale, 0.72);
  check("heat leaves unlooked regions transparent", alphaOf(heat, 5) === 0);
  check("heat is strongest at the hottest point", alphaOf(heat, 0) > alphaOf(heat, 2));

  // The jet ramp this replaced painted its coldest stop a saturated blue at a
  // flat 0.72 opacity, so a region with the faintest trace of gaze carried as
  // much visual weight as the hot core: a reader's eye reported attention where
  // there was effectively none. The floor of the ramp has to fade out, and it
  // has to fade out *in the lookup table*, which is what the legend reads.
  check("the cold floor of the ramp is fully transparent", rampAlpha(0) === 0);
  check(
    "the ramp's alpha rises with attention",
    Array.from({ length: 32 }, (_, i) => rampAlpha(i / 31)).every(
      (a, i, all) => i === 0 || a > all[i - 1]
    )
  );
  check("the hot end of the ramp is opaque", rampAlpha(1) === 1);
  // A cold-blue floor is the specific failure; nothing in the ramp may be cool.
  const cool = Array.from({ length: 64 }, (_, i) => rampRgb(i / 63)).filter(
    ([r, g, b]) => b >= r || g > r
  );
  check("no stop in the ramp is a cool colour", cool.length === 0, `${cool.length} cool stops`);
  check(
    "the ramp passes through the brand's vermillion",
    Array.from({ length: 256 }, (_, i) => rampRgb(i / 255)).some(
      ([r, g, b]) => Math.abs(r - 231) <= 3 && Math.abs(g - 61) <= 6 && b <= 8
    )
  );
  // The overlay's own opacity still has the last word: the ramp scales it
  // rather than replacing it.
  const faint = new Uint8ClampedArray(field.length * 4);
  paintField(field, faint, "heat", scale, 0.3);
  check("overlay opacity still scales the ramp", alphaOf(faint, 0) < alphaOf(heat, 0));
}

// --- Recording stage geometry --------------------------------------------

section("Recording chrome stays out of the measured rect");
{
  // The controls used to be absolutely positioned over the stimulus, and every
  // gaze sample is normalised against the stimulus rect: a participant looking
  // at Finish or Discard was recorded as looking at whatever the chrome
  // covered. The band below the stimulus is solved against the same edge
  // tolerance the sample filter uses, so a look at the chrome falls outside the
  // kept range instead of being clamped onto the footer.
  const heights = [700, 720, 768, 800, 900, 1024, 1080, 1200, 1440, 1600, 2160];

  /** What runRecording does to a sample at screen y, for a stage of this
   * height: normalise against the letterboxed stimulus rect. */
  const normalisedY = (screenY: number, stageHeight: number): number =>
    screenY / (stageHeight - controlBandHeight(stageHeight));
  const kept = (ny: number): boolean => ny <= 1 + EDGE_TOLERANCE;

  const contaminating = heights.filter((h) => chromeCanContaminate(h));
  check(
    "no stage height lets the control strip reach the measured rect",
    contaminating.length === 0,
    contaminating.length ? `fails at ${contaminating.join(", ")}px` : `${heights.length} heights`
  );

  const leaks = heights.filter((h) => kept(normalisedY(h - CONTROL_STRIP_PX, h)));
  check(
    "gaze on the top edge of the controls is dropped, not clamped to the footer",
    leaks.length === 0,
    leaks.length ? `leaks at ${leaks.join(", ")}px` : "every height"
  );
  const bottomLeaks = heights.filter((h) => kept(normalisedY(h - 1, h)));
  check("gaze on the bottom edge of the screen is dropped too", bottomLeaks.length === 0);

  // The band still has to be a band, not a letterbox that eats the stimulus.
  const band = controlBandHeight(900);
  check("the band fits the control strip", band > CONTROL_STRIP_PX, `${band}px at 900px`);
  check(
    "the band costs the stimulus under a seventh of the stage",
    band / 900 < 0.14,
    `${((band / 900) * 100).toFixed(1)}% of a 900px stage`
  );
  check(
    "a taller stage reserves a proportionally larger band",
    controlBandHeight(1440) > controlBandHeight(900)
  );
  // The last row the participant can actually look at still reaches the
  // heatmap: the fix must not start throwing away real edge gaze.
  check(
    "gaze on the last row of the stimulus is still kept",
    kept(normalisedY(900 - controlBandHeight(900) - 1, 900))
  );
}

section("Showing a stimulus at a size the participant can read");
{
  // A 1280x1600 full-page screenshot letterboxed into a 1440x804 stage rendered
  // 643px wide: every label at half its designed size, on the screen the
  // finding is made on. That does not merely look bad — the task asks someone
  // to find a control on text they cannot read, so the study is invalid.
  const stage = { width: 1440, height: 804 };
  const fullPage = fitStimulus({ width: 1280, height: 1600 }, stage);
  check("a tall screenshot is shown at full width, not contained", fullPage.mode === "width");
  check(
    "and at full width it is near its designed size",
    fullPage.scale > FIT_SCALE_FLOOR,
    `${(fullPage.scale * 100).toFixed(0)}%`
  );

  // The common case must not change: a wireframe that already fits stays
  // letterboxed, because seeing the whole screen at once is the point.
  const wireframe = fitStimulus({ width: 1400, height: 760 }, stage);
  check("a stimulus that fits comfortably stays contained", wireframe.mode === "contain");
  const slightlyTall = fitStimulus({ width: 1280, height: 820 }, stage);
  check(
    "a marginally tall stimulus stays contained rather than scrolling",
    slightlyTall.mode === "contain",
    `${(slightlyTall.scale * 100).toFixed(0)}%`
  );
  // Scrolling buys nothing when the width is already the binding constraint: a
  // wide stimulus is the same size either way, and the scrollbar would be the
  // only difference.
  const wide = fitStimulus({ width: 4000, height: 1000 }, stage);
  check("a too-wide stimulus is contained, since scrolling would not help", wide.mode === "contain");
  check("a degenerate size does not divide by zero", fitStimulus({ width: 0, height: 0 }, stage).scale === 1);

  // The scrolling fit breaks the invariant controlBandHeight guarantees: a
  // stimulus taller than its window has a rect that runs past the window on
  // both ends, so the moderator's strip is inside the rect by construction and
  // "inside the rect" stops implying "on screen". The window check is what
  // replaces it.
  const band = controlBandHeight(900);
  const windowBox = { left: 0, top: 0, width: 1440, height: 900 - band };
  const stripY = 900 - CONTROL_STRIP_PX;
  check(
    "gaze on the control strip is dropped in the scrolling fit",
    !isInsideViewport({ x: 720, y: stripY }, windowBox),
    `strip at y=${stripY}, window ends at ${windowBox.height}`
  );
  check(
    "gaze anywhere below the window is dropped, not clamped",
    !isInsideViewport({ x: 720, y: windowBox.height + 2 }, windowBox)
  );
  check(
    "gaze on the last visible row of the stimulus is kept",
    isInsideViewport({ x: 720, y: windowBox.height - 1 }, windowBox)
  );
  // Edge tolerance survives where the window edge is a real stimulus edge.
  check(
    "gaze just past the left edge is still kept",
    isInsideViewport({ x: -20, y: 300 }, windowBox)
  );
  check(
    "gaze well off the left edge is dropped",
    !isInsideViewport({ x: -400, y: 300 }, windowBox)
  );
}

section("Sub-sample latencies do not print as zero");
{
  // "Time to first region: 0ms" rendered as a headline stat and as a table
  // cell. Capture begins with the participant already looking at the screen, so
  // anyone already on a region produces a value below one sample interval, and
  // a bare "0ms" reads as a broken counter rather than a measurement.
  const instant = formatOnset(0);
  check("an onset of zero does not print as 0ms", instant.label === `<${SAMPLE_PERIOD_MS}ms`, instant.label);
  check("and it says why", instant.note !== null && /already fixating/i.test(instant.note));
  check(
    "anything under one sample takes the same floor",
    formatOnset(SAMPLE_PERIOD_MS - 1).label === `<${SAMPLE_PERIOD_MS}ms`
  );
  // Above the floor it is an ordinary measurement again, in the units the rail
  // beside it uses.
  check("a real onset prints its value", formatOnset(420).label === "420ms");
  check("and carries no caveat", formatOnset(420).note === null);
  check("a slow onset reads in seconds", formatOnset(2400).label === "2.4s");
  check("an unmeasurable onset is a dash", formatOnset(Number.NaN).label === "—");
}

// --- AOIs ----------------------------------------------------------------

section("Area of interest analysis");
{
  const aois: Aoi[] = [
    { id: "hero", label: "Hero", x: 0, y: 0, width: 1, height: 0.3 },
    { id: "cta", label: "CTA", x: 0.6, y: 0.6, width: 0.3, height: 0.2 },
    { id: "unseen", label: "Footer", x: 0, y: 0.9, width: 1, height: 0.1 },
  ];

  const fixations = [
    { x: 0.5, y: 0.1, duration: 400, start: 100, samples: 12 },
    { x: 0.2, y: 0.2, duration: 200, start: 600, samples: 6 },
    { x: 0.7, y: 0.65, duration: 600, start: 900, samples: 18 },
  ];

  const results = analyseAois(aois, fixations, 0);
  const hero = results.find((r) => r.aoiId === "hero")!;
  const cta = results.find((r) => r.aoiId === "cta")!;
  const footer = results.find((r) => r.aoiId === "unseen")!;

  check("counts fixations inside an AOI", hero.fixationCount === 2, `${hero.fixationCount}`);
  check("sums dwell inside an AOI", hero.dwell === 600, `${hero.dwell}ms`);
  check("time to first fixation uses the first hit", hero.timeToFirstFixation === 100);
  check("CTA dwell is correct", cta.dwell === 600 && cta.fixationCount === 1);
  check("unlooked AOI reports null TTFF", footer.timeToFirstFixation === null && footer.dwell === 0);
  check(
    "dwell shares sum to at most 1",
    results.reduce((sum, r) => sum + r.dwellShare, 0) <= 1.0001
  );

  // Two participants: one found the CTA, one did not.
  const other = analyseAois(aois, [{ x: 0.5, y: 0.1, duration: 500, start: 50, samples: 15 }], 0);
  const aggregates = aggregateAois(aois, [results, other]);
  const ctaAgg = aggregates.find((a) => a.aoiId === "cta")!;
  const heroAgg = aggregates.find((a) => a.aoiId === "hero")!;
  check("hit rate reflects who found it", ctaAgg.hitRate === 0.5, `${ctaAgg.hitRate}`);
  check("hit rate is 1 when everyone looked", heroAgg.hitRate === 1);
  check("mean TTFF only averages participants who found it", ctaAgg.meanTimeToFirstFixation === 900);
  check("aggregate reports participant count", ctaAgg.participants === 2);
}

// --- Recording quality ---------------------------------------------------

section("Recording quality grading");
{
  const quality = (trackingRatio: number, validationError: number | null): RecordingQuality => ({
    validationError,
    trackingRatio,
    meanFps: 30,
    viewportWidth: SCREEN_W,
    viewportHeight: SCREEN_H,
    stimulusRect: { x: 0, y: 0, width: SCREEN_W, height: SCREEN_H },
  });

  check("a well-tracked session grades good", gradeTracking(0.91) === "good");
  check("a middling session grades warn", gradeTracking(0.7) === "warn");
  check("the screenshot's 44%-tracked session grades bad", gradeTracking(0.44) === "bad");

  // 4° at 60cm is ~159 CSS px, so 210px is over the line and 130px is not.
  check("a 210px calibration error grades bad", gradeError(210) === "bad");
  check("a 130px calibration error is not bad", gradeError(130) !== "bad");
  check("an unmeasured calibration is warned about, not condemned", gradeError(null) === "warn");

  check("the worse axis decides the row", gradeRecording(quality(0.95, 210)) === "bad");
  check("a clean recording is not flagged", isLowSignal(quality(0.91, 90)) === false);
  check("the flagged recording is excluded-worthy", isLowSignal(quality(0.44, 210)) === true);
  // The regression this exists to prevent: a 44%-tracked, 210px-error session
  // being averaged into the aggregate in the same grey as a clean one.
  check("low tracking alone is enough to flag", isLowSignal(quality(0.44, 60)) === true);
}

// --- Scanpath ordinals ---------------------------------------------------

section("Scanpath ordinal placement");
{
  const collides = (a: OrdinalLabel, b: OrdinalLabel): boolean =>
    Math.abs(a.x - b.x) < a.halfWidth + b.halfWidth &&
    Math.abs(a.y - b.y) < a.halfHeight + b.halfHeight;

  const countCollisions = (labels: OrdinalLabel[]): number => {
    let hits = 0;
    for (let i = 0; i < labels.length; i++) {
      for (let j = i + 1; j < labels.length; j++) {
        if (collides(labels[i], labels[j])) hits++;
      }
    }
    return hits;
  };

  // A sparse path: every number belongs at the centre of its own circle.
  const sparse = layoutOrdinals([
    { x: 100, y: 100, radius: 20 },
    { x: 300, y: 140, radius: 24 },
    { x: 200, y: 400, radius: 18 },
  ]);
  check("a sparse path keeps its labels centred", sparse.every((l) => !l.leader));
  check("a sparse path has no collisions", countCollisions(sparse) === 0);

  // The failure from the screenshots: a tight cluster, where a heatmap-dense
  // reader makes several fixations within a few pixels of each other and the
  // ordinals printed on top of one another.
  const clusterCircles = Array.from({ length: 6 }, (_, i) => ({
    x: 200 + i * 3,
    y: 200 + i * 2,
    radius: 22,
  }));
  const cluster = layoutOrdinals(clusterCircles);
  check("a dense cluster still emits every ordinal", cluster.length === 6);
  check(
    "a dense cluster's ordinals do not overlap",
    countCollisions(cluster) === 0,
    `${countCollisions(cluster)} collisions`
  );
  check("the first ordinal keeps its circle's centre", cluster[0].leader === false);
  check(
    "a dense cluster displaces most of its ordinals",
    cluster.filter((l) => l.leader).length >= 4,
    `${cluster.filter((l) => l.leader).length} of 6 displaced`
  );
  // A displaced label has to end up outside the circle it belongs to, or the
  // leader line would start and end in the same place.
  check(
    "every displaced ordinal clears its own circle",
    cluster.every(
      (l, i) =>
        !l.leader ||
        Math.hypot(l.x - clusterCircles[i].x, l.y - clusterCircles[i].y) >= clusterCircles[i].radius
    )
  );

  // Exact co-location is the worst case: eight callout slots for eight labels.
  const stacked = layoutOrdinals(Array.from({ length: 8 }, () => ({ x: 150, y: 150, radius: 16 })));
  check("perfectly co-located fixations all keep a number", stacked.length === 8);
  check(
    "perfectly co-located fixations are separated",
    countCollisions(stacked) === 0,
    `${countCollisions(stacked)} collisions across 8 labels`
  );

  // Past eight, one ring of callout positions is exhausted. This is the case
  // the placer used to give up on — "every position taken: park it above the
  // circle anyway" — which printed 10 over 7 as "107" on exactly the studies
  // this tool is for: one task, one CTA, every fixation in one place.
  const pileUp = layoutOrdinals(
    Array.from({ length: 14 }, (_, i) => ({ x: 300 + (i % 3), y: 300 - (i % 2), radius: 20 }))
  );
  check("a fourteen-deep pile-up keeps every ordinal", pileUp.length === 14);
  check(
    "a fourteen-deep pile-up has no unreadable pairs",
    countCollisions(pileUp) === 0,
    `${countCollisions(pileUp)} collisions across 14 labels`
  );
  // Displaced labels have to stay attached to something a leader line can
  // reach; a number parked half a screen away belongs to nothing.
  const furthest = Math.max(...pileUp.map((l) => Math.hypot(l.x - 300, l.y - 300)));
  check(
    "displaced ordinals stay near the cluster they belong to",
    furthest < 220,
    `${furthest.toFixed(0)}px from the cluster`
  );

  // Edge clamping. Without it a fixation near the top of the stimulus pushed
  // its number off the drawing surface — ordinal "2" arrived sliced in half by
  // the canvas edge — and a scanpath that loses a step loses the only thing it
  // says.
  const bounds = { width: 400, height: 300 };
  const edges = layoutOrdinals(
    [
      { x: 4, y: 3, radius: 22 },
      { x: 396, y: 297, radius: 22 },
      { x: 6, y: 5, radius: 22 },
      { x: 200, y: 2, radius: 22 },
    ],
    1,
    bounds
  );
  check(
    "every ordinal stays inside the canvas",
    edges.every(
      (l) =>
        l.x - l.halfWidth >= -0.001 &&
        l.x + l.halfWidth <= bounds.width + 0.001 &&
        l.y - l.halfHeight >= -0.001 &&
        l.y + l.halfHeight <= bounds.height + 0.001
    )
  );
  check(
    "clamped ordinals still do not collide",
    countCollisions(edges) === 0,
    `${countCollisions(edges)} collisions at the canvas edge`
  );
  // A number that had to leave its circle is drawn at the legibility floor, so
  // a displaced "36" recedes behind an ordinal sitting inside its own circle
  // rather than competing with it.
  const displaced = edges.filter((l) => l.leader);
  check(
    "displaced ordinals drop to the minimum size",
    displaced.length > 0 && displaced.every((l) => l.fontSize < edges[0].fontSize),
    `${displaced.length} displaced`
  );
  // Unbounded placement is unchanged: the tests above depend on it, and so
  // does any caller that has no canvas to clamp against.
  check(
    "without bounds nothing is clamped",
    layoutOrdinals([{ x: -50, y: -50, radius: 20 }])[0].x === -50
  );
  // A subset of a path still has to say where in the *whole* path each circle
  // sits, so the printed number comes from the circle rather than its position
  // in the array handed to the layout pass.
  const renumbered = layoutOrdinals([{ x: 100, y: 100, radius: 20, ordinal: 42 }]);
  check("a stated ordinal is what gets measured", renumbered[0].halfWidth > 0);
}

// --- Scanpath thinning and ramp -------------------------------------------

section("Scanpath ordinals and ramp");
{
  // Under the budget nothing is dropped: thinning is a response to density, not
  // a fixed cap on how much of a path can be numbered.
  const short = selectOrdinals([100, 200, 300]);
  check("a short path keeps every ordinal", short.length === 3 && short[2] === 2);

  // 40 fixations, with the longest ones deliberately not at the ends.
  const durations = Array.from({ length: 40 }, (_, i) => 100 + (i % 8) * 50);
  const kept = selectOrdinals(durations);
  check("a dense path is thinned to the budget", kept.length === ORDINAL_BUDGET, `${kept.length}`);
  check("the thinned ordinals stay in sequence order", kept.every((v, i) => i === 0 || v > kept[i - 1]));
  // Where a path starts and where it ends are the two facts a reader looks for
  // first, whatever those two fixations happened to be worth.
  check(
    "the first and last fixations always keep their number",
    kept[0] === 0 && kept[kept.length - 1] === durations.length - 1
  );
  // Duration is what makes a fixation worth citing, and the circles are already
  // sized by it — so the surviving numbers land on the marks a reader is
  // already looking at.
  const middle = kept.filter((i) => i !== 0 && i !== durations.length - 1);
  const dropped = durations.map((_, i) => i).filter((i) => !kept.includes(i));
  const shortestKept = Math.min(...middle.map((i) => durations[i]));
  const longestDropped = Math.max(...dropped.map((i) => durations[i]));
  check(
    "the surviving ordinals are the longest fixations",
    shortestKept >= longestDropped,
    `kept ≥ ${shortestKept}ms, dropped ≤ ${longestDropped}ms`
  );

  // The rainbow this replaced was flagged for exactly this: hue carries no
  // order, so first-to-last only reads if the ramp climbs in lightness. That
  // also makes it survive greyscale and colour vision deficiency.
  const luminance = (colour: string): number => {
    const [r, g, b] = colour.match(/[\d.]+/g)!.slice(0, 3).map(Number);
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  const ramp = Array.from({ length: 9 }, (_, i) => luminance(scanpathColour(i / 8)));
  check(
    "the order ramp climbs in lightness at every step",
    ramp.every((v, i) => i === 0 || v > ramp[i - 1]),
    ramp.map((v) => Math.round(v)).join(" → ")
  );
  // The rim drawn around each circle is the same colour, darker — the parameter
  // kept its old HSL-percentage meaning through the ramp change.
  check(
    "a lower lightness darkens the same colour",
    luminance(scanpathColour(0.7, 1, 40)) < luminance(scanpathColour(0.7, 1, 55))
  );
  check("the ramp carries the alpha it is asked for", scanpathColour(0.5, 0.85).endsWith(", 0.85)"));
}

// --- What a drawn scanpath actually prints --------------------------------

section("Scanpath numbering reads as an order");
{
  const bounds = { width: 1000, height: 700 };

  // A sparse path: every fixation numbered, with its own place in the sequence.
  const sparsePlan = planOrdinals(
    [
      { x: 120, y: 120, radius: 20 },
      { x: 500, y: 200, radius: 26 },
      { x: 300, y: 500, radius: 18 },
    ],
    [200, 400, 250],
    1,
    bounds
  );
  check(
    "a sparse path numbers every fixation by its true place",
    sparsePlan.circles.map((c) => c.ordinal).join(",") === "1,2,3"
  );

  // The failure from the screenshots: 61 fixations over six clusters. Numbering
  // all of them displaced almost every label, and the thinning then printed the
  // surviving *indices* — circles reading 1, 32, 14, 25, 26, 57 with grey
  // numerals orbiting them on leader stubs and large circles carrying nothing.
  const clusters = [
    [200, 160],
    [760, 200],
    [480, 380],
    [180, 560],
    [820, 560],
    [520, 640],
  ];
  const dense = Array.from({ length: 61 }, (_, i) => {
    const [cx, cy] = clusters[i % clusters.length];
    return {
      x: cx + ((i * 37) % 40) - 20,
      y: cy + ((i * 53) % 30) - 15,
      radius: 14 + (i % 7) * 4,
      // Carried through the plan so the test can ask which fixations survived;
      // the planner copies whatever a circle brings with it.
      seq: i,
    };
  });
  const denseDurations = dense.map((_, i) => 120 + (i % 9) * 60);
  const plan = planOrdinals(dense, denseDurations, 1, bounds);

  check(
    "a dense path is thinned rather than numbered in full",
    plan.circles.length < dense.length && plan.circles.length > 3,
    `${plan.circles.length} of ${dense.length} numbered`
  );
  // The whole point: what is printed is 1, 2, 3 … over the marks that keep a
  // number, so the picture reads as an order instead of as a bag of indices.
  check(
    "the numbers printed on a dense path count 1..n",
    plan.circles.every((c, i) => c.ordinal === i + 1),
    plan.circles.map((c) => c.ordinal).join(",")
  );
  // No numeral outside a circle with a leader stub running to a mark the eye
  // cannot find — the "orphan numerals" the picture was full of.
  check(
    "no ordinal on a dense path is orphaned outside its circle",
    plan.labels.every((label) => !label.leader),
    `${plan.labels.filter((l) => l.leader).length} displaced`
  );
  check(
    "every numbered circle has exactly one label",
    plan.labels.length === plan.circles.length
  );
  // Renumbering must not reorder: the marks are still read in the sequence the
  // participant looked in.
  const chosen = plan.circles.map((c) => c.seq);
  check(
    "the numbered marks stay in the order they were looked at",
    chosen.every((v, i) => i === 0 || v > chosen[i - 1])
  );
}

// --- Legends -------------------------------------------------------------

section("Overlay legends");
{
  const heat = legendFor("heat", ["P01", "P02"]);
  check("the heat legend is a gradient", heat.stops !== null && heat.stops.length > 2);
  check("the heat legend is not banded", heat.banded === false);
  // The legend must come from the same lookup table as the pixels, or it drifts
  // into promising colours the renderer does not paint.
  check("the heat legend ends match the ramp", heat.stops?.[0] === rampColour(0));
  check(
    "the heat legend's hot end matches the ramp",
    heat.stops?.[heat.stops.length - 1] === rampColour(1)
  );
  // The legend has to carry the ramp's transparency, or the strip under the
  // stage advertises a solid floor over regions the overlay leaves clear.
  check("the heat legend's cold end is transparent", heat.stops?.[0].endsWith(", 0.000)") === true);
  check("the heat legend names its units", /fixation duration/i.test(heat.note));
  check("the heat legend says the scale is relative", /relative to this selection/i.test(heat.note));

  // The numbered axis. A legend in a tool whose product is numbers should say
  // what the hot end is worth — but only when the caller knows, because the
  // ceiling is a percentile of one particular selection's blob peaks.
  check("a heat legend with no scale prints no numbers", heat.ticks === null);
  const scaled = legendFor("heat", ["P01"], { ceiling: 1240 });
  check("a scale puts ticks on the heat legend", scaled.ticks?.length === 3);
  check("the ticks span the whole strip", scaled.ticks?.[0].at === 0 && scaled.ticks?.[2].at === 1);
  check("the cold end is zero", scaled.ticks?.[0].label === "0ms");
  // Above the ceiling saturates, so the top of the axis is a bound and says so.
  check("the hot end names the ceiling", scaled.ticks?.[2].label === "≥1.2s", scaled.ticks?.[2].label);
  check("the midpoint is half the ceiling", scaled.ticks?.[1].label === "620ms", scaled.ticks?.[1].label);
  // Sub-second ceilings stay in the units the rail beside them uses.
  check(
    "a short ceiling reads in milliseconds",
    legendFor("heat", [], { ceiling: 480 }).ticks?.[2].label === "≥480ms"
  );
  // A ceiling of zero is "nothing was drawn", not "the hot end is 0ms".
  check("an empty selection prints no numbers", legendFor("heat", [], { ceiling: 0 }).ticks === null);
  // The caveat is the reason the numbers are safe to print at all, so it stays
  // under a legend that now has numbers on it.
  check(
    "the numbered legend keeps the cross-study caveat",
    /not comparable between studies/i.test(scaled.note)
  );

  const contour = legendFor("contour", []);
  check("the contour legend is banded", contour.banded === true);
  check(
    "the contour legend has one swatch per drawn band",
    contour.stops?.length === contourBandColours().length
  );
  // The bands are equal slices of the same scale, so they take the same ticks.
  check(
    "contour bands are numbered from the same scale",
    legendFor("contour", [], { ceiling: 1240 }).ticks?.[2].label === "≥1.2s"
  );
  // A mask encodes "revealed or not"; milliseconds under it would name a
  // quantity the picture does not carry.
  check(
    "the spotlight mask stays unnumbered",
    legendFor("spotlight", [], { ceiling: 1240 }).ticks === null
  );

  const scanpath = legendFor("scanpath", ["P01"]);
  check("the scanpath legend keys time", scanpath.minLabel === "First" && scanpath.maxLabel === "Last");

  // The note is the caveat a reader needs before citing the figure, and it is
  // 68 words — so it folds into a disclosure and a one-clause caption stays
  // inline. Every mode has to carry both, or a view loses its key entirely.
  const modes = ["heat", "contour", "spotlight", "scanpath", "raw"] as const;
  const captionless = modes.filter((m) => legendFor(m, ["P01"]).caption.trim().length === 0);
  check("every overlay names itself in one clause", captionless.length === 0, captionless.join(", "));
  const verbose = modes.filter((m) => legendFor(m, ["P01"]).caption.split(/\s+/).length > 14);
  check(
    "and the caption stays a caption rather than becoming the note again",
    verbose.length === 0,
    verbose.join(", ")
  );

  const raw = legendFor("raw", ["P01", "P02", "P03"]);
  check("the raw legend keys every participant", raw.swatches?.length === 3);
  check(
    "the raw legend's swatches match the dots the canvas draws",
    raw.swatches?.[1].colour === participantColour(1)
  );
  check("the raw legend labels the participants", raw.swatches?.[2].label === "P03");
  check("raw has no gradient", raw.stops === null);

  // The old hsla(i * 67) sweep put P01 on red and P02 on green — the one pair
  // a red-green colour-blind viewer cannot separate.
  const distinct = new Set(PARTICIPANT_COLOURS);
  check("participant colours are all distinct", distinct.size === PARTICIPANT_COLOURS.length);
  check("the palette wraps rather than running out", participantColour(0) === participantColour(PARTICIPANT_COLOURS.length));
  check("participant colours carry alpha when asked", participantColour(0, 0.35).endsWith(", 0.35)"));
}

// --- One screen, one scope ------------------------------------------------

section("The results screen states one scope, not three");
{
  /**
   * The measured failure, reproduced as data: a five-recording study with one
   * low-signal session auto-excluded, viewed in Scanpath. The header pill said
   * "4 of 5 recordings", the rail said "Summary — P01", the region table under
   * them printed the four-participant aggregate, all four scoped export rows
   * said "4 of 5 recordings", and the PNG they produced held P01's path under a
   * caption reading "All participants".
   */
  const solo: ReportedScope = {
    participants: ["P01"],
    total: 5,
    perView: true,
    flagged: 1,
    excludingFlagged: true,
  };
  const picked: ReportedScope = { ...solo, participants: ["P02"], perView: false };
  const aggregate: ReportedScope = {
    participants: ["P01", "P02", "P03", "P05"],
    total: 5,
    perView: false,
    flagged: 1,
    excludingFlagged: true,
  };
  const included: ReportedScope = {
    participants: ["P01", "P02", "P03", "P04", "P05"],
    total: 5,
    perView: false,
    flagged: 1,
    excludingFlagged: false,
  };
  const clean: ReportedScope = {
    participants: ["P01", "P02"],
    total: 2,
    perView: false,
    flagged: 0,
    excludingFlagged: false,
  };
  const lone: ReportedScope = {
    participants: ["P01"],
    total: 1,
    perView: false,
    flagged: 0,
    excludingFlagged: false,
  };

  /**
   * How many recordings a sentence claims to describe, read back out of its own
   * words. This is the assertion that matters: the four statements are written
   * in four registers and are *meant* to differ in wording, so comparing them
   * literally would only prove they are identical. Comparing what each one
   * claims proves they agree.
   */
  const claimed = (text: string): number => {
    const partial = text.match(/(\d+) of (\d+)/);
    if (partial) return Number(partial[1]);
    const all = text.match(/all (\d+) recording/i);
    if (all) return Number(all[1]);
    const bare = text.match(/^(\d+) recording/);
    if (bare) return Number(bare[1]);
    return NaN;
  };

  const worded = (scope: ReportedScope): string[] => [
    scopePill(scope).text,
    scopeSentence(scope),
    scopeNote(scope),
  ];

  const cases: Array<[string, ReportedScope]> = [
    ["a scanpath's one participant", solo],
    ["a picked participant", picked],
    ["the aggregate with a session excluded", aggregate],
    ["the aggregate with every session included", included],
    ["a study with nothing flagged", clean],
    ["a study of one recording", lone],
  ];

  const disagreeing = cases.filter(([, scope]) =>
    worded(scope).some((text) => claimed(text) !== scope.participants.length)
  );
  check(
    "the pill, the menu row and the file note all count the same recordings",
    disagreeing.length === 0,
    disagreeing.map(([name, s]) => `${name}: ${worded(s).map(claimed).join("/")} vs ${s.participants.length}`).join("; ")
  );

  const mislabelled = cases.filter(
    ([, scope]) =>
      (scope.participants.length === 1 && scope.total > 1) !==
      scopeCaption(scope).startsWith(scope.participants[0])
  );
  check(
    "the exported PNG's caption names a participant exactly when the picture is one",
    mislabelled.length === 0,
    mislabelled.map(([name]) => name).join(", ")
  );

  // The four statements the critic read off one screen, now.
  check(
    "the header pill names the participant a per-person view is showing",
    scopePill(solo).text === "P01 — 1 of 5 recordings",
    scopePill(solo).text
  );
  check("the export rows say the same", scopeSentence(solo) === "P01 only — 1 of 5 recordings", scopeSentence(solo));
  check("the PNG caption says the same", scopeCaption(solo) === "P01", scopeCaption(solo));
  check(
    "and the file note carries the reason the set is one person",
    scopeNote(solo).includes("P01") && scopeNote(solo).includes("scanpath"),
    scopeNote(solo)
  );
  // A participant picked from the dropdown is also one person, but not because
  // of the view — so the file does not blame the scanpath for it.
  check(
    "a picked participant is not attributed to the view",
    !scopeNote(picked).includes("scanpath") && scopeNote(picked).includes("P02"),
    scopeNote(picked)
  );

  // The other direction: an aggregate must not put one person's name on
  // anything, or the fix would have traded one wrong scope for another.
  const named = [...worded(aggregate), scopeCaption(aggregate)].filter((text) =>
    aggregate.participants.some((p) => text.includes(p))
  );
  check("an aggregate names no individual", named.length === 0, named.join(" | "));
  check(
    "the aggregate still says which sessions it dropped and why",
    scopeNote(aggregate).includes("low-signal excluded") && scopeNote(aggregate).includes("1 below"),
    scopeNote(aggregate)
  );
  check(
    "a study with nothing to exclude just states its size",
    scopePill(clean).text === "2 recordings" && scopeSentence(clean) === "all 2 recordings"
  );
  check("and a one-recording study is not described as a selection", scopeCaption(lone) === "All participants");
}

// --- The stage's ceiling --------------------------------------------------

section("The stage ends where the legend that decodes it still fits");
{
  /**
   * Measured at 1440×900 on a fresh open of a populated study: `--stage-cap`
   * resolved to 614px and `.legend-slot` measured top 849 / bottom 963 — 63px
   * past the fold, so the title strip was visible and the 0ms / 540ms / ≥1.1s
   * ticks that make the picture readable were not. The legend is 114px tall on
   * a 14px margin, which fixes the chrome above the stage at 221px.
   */
  const reference = { viewport: 900, above: 221, under: 114 + 14 };
  const cap = stageCap(reference);
  const legendBottom = reference.above + cap + reference.under;
  check("the stage's ceiling leaves the legend its height", cap === 551, `${cap}px`);
  check(
    "the legend's bottom edge lands on the fold rather than past it",
    legendBottom <= reference.viewport,
    `bottom ${legendBottom} of ${reference.viewport}`
  );

  // The term that was wrong, restated: the rail asked for 614 and its bound —
  // `min(fillsRow, viewport - above)` — let it through because the bound left
  // out the very chrome the first term subtracts.
  const previous = Math.max(cap, Math.min(614, reference.viewport - reference.above));
  check(
    "the 63px the old rail-slack bound overshot by are gone",
    previous - cap === 63 && reference.above + previous + reference.under - reference.viewport === 63,
    `was ${previous}px, now ${cap}px`
  );

  // And it is a property, not a lucky number: no combination of viewport,
  // chrome and reserve may put the legend past the fold unless the stage is
  // already at the floor its own stylesheet sets.
  const offenders: string[] = [];
  for (const viewport of [1080, 900, 800, 768, 700]) {
    for (const above of [140, 190, 221, 300]) {
      for (const under of [0, 90, 128, 176]) {
        const c = stageCap({ viewport, above, under });
        if (c === STAGE_MIN_HEIGHT) continue;
        if (above + c + under > viewport) offenders.push(`${viewport}/${above}/${under}`);
      }
    }
  }
  check("no viewport puts the stage and its legend past the fold", offenders.length === 0, offenders.join(" "));
  check(
    "a window too short for either still gets the stage's declared minimum",
    stageCap({ viewport: 650, above: 300, under: 176 }) === STAGE_MIN_HEIGHT
  );
}

// --- Fit-width ------------------------------------------------------------

section("The stage offers fit-width when the stimulus is starved of it");
{
  /**
   * Sampled ten times at 80ms from first paint at each width, on a fresh load:
   * `.stage-fit[aria-pressed]` was "false" in all forty samples while the
   * settled ratios were all under the threshold the toggle exists to enforce.
   */
  const settled = [
    { width: 1180, stage: 1098, figure: 687 },
    { width: 1100, stage: 1018, figure: 687 },
    { width: 1024, stage: 942, figure: 689 },
  ];
  const missed = settled.filter((s) => !shouldFitWidth(s.figure, s.stage));
  check(
    "every width the affordance was measured at is under its own threshold",
    missed.length === 0,
    settled.map((s) => `${s.width}: ${(s.figure / s.stage).toFixed(3)}`).join("  ")
  );

  /**
   * Why it never fired. It ran on the line after the first `draw()`, and
   * `draw()` is what fills the legend — so the cap it measured against had
   * reserved nothing for a 114px legend, the height-bound figure was that much
   * taller, and at the stimulus's 1.28 aspect ratio that is 146px wider. At
   * 1180 the ratio it actually tested was 0.759 against a threshold of 0.75: it
   * missed by seven thousandths of the number it was comparing against.
   */
  const preCap = settled[0].figure + 114 * 1.28;
  check(
    "the pre-legend measurement it used to take clears the threshold",
    !shouldFitWidth(preCap, settled[0].stage),
    `${(preCap / settled[0].stage).toFixed(3)} vs ${AUTO_FIT_RATIO}`
  );
  /**
   * And what fixing the stage's ceiling does to this decision, which is the one
   * place the two repairs touch. At 1440×900 the same 1280×1000 stimulus was
   * 748px wide in a 944px scroller — 0.792, over the threshold, correctly left
   * contained. With the cap honest at 551 the figure is 667px, and it now fires:
   * the 113px of empty ground on each side of the artboard goes, at the cost of
   * a stage that scrolls on Y with a visible clipped edge. One click undoes it.
   */
  const aspect = 748 / 584;
  const figureAt = (cap: number): number => (cap - 30) * aspect;
  check(
    "the old ceiling left 1440×900 contained",
    !shouldFitWidth(figureAt(614), 944),
    `${(figureAt(614) / 944).toFixed(3)}`
  );
  check(
    "and the honest one takes it below the threshold",
    shouldFitWidth(figureAt(551), 944),
    `${(figureAt(551) / 944).toFixed(3)}`
  );
  check("a figure that fills its stage is left contained", !shouldFitWidth(900, 944));
  check("and a stage with no width yet decides nothing", !shouldFitWidth(687, 0));
}

// --- The heat kernel ------------------------------------------------------

section("The heat kernel is the calibration error, not a constant");
{
  // σ is what the kernel means, and renderHeatmap takes σ as half the splat
  // radius — so the derived ratio has to come back out as exactly the error
  // that went in. That is the whole claim of the change.
  const stimulus = 1000;
  const sigmaFor = (errorPx: number | null): number =>
    (kernelRatio(errorPx, stimulus) * stimulus) / 2;
  check("a 60px calibration error blurs by 60px", Math.abs(sigmaFor(60) - 60) < 0.5, `${sigmaFor(60).toFixed(1)}px`);
  check(
    "two recordings with different error do not get the same blob",
    kernelRatio(48, stimulus) !== kernelRatio(184, stimulus),
    `${kernelRatio(48, stimulus).toFixed(3)} vs ${kernelRatio(184, stimulus).toFixed(3)}`
  );
  // The app's own setup panel says gaze lands within "2 to 4 degrees of visual
  // angle, which is 50 to 120 pixels". The constant drew σ = 28px on this
  // stimulus: half the floor of the uncertainty the tool states about itself,
  // three inches from a rail printing the measured value per selection.
  check(
    "the constant it replaces sat under the app's own stated uncertainty",
    (DEFAULT_KERNEL_RATIO * stimulus) / 2 < MIN_KERNEL_SIGMA_PX,
    `σ ${((DEFAULT_KERNEL_RATIO * stimulus) / 2).toFixed(0)}px against a stated floor of ${MIN_KERNEL_SIGMA_PX}px`
  );
  const outside = [0, 10, 48, 60, 100, 150, 184, 400, null].filter((error) => {
    const sigma = sigmaFor(error);
    return sigma < MIN_KERNEL_SIGMA_PX - 0.5 || sigma > MAX_KERNEL_SIGMA_PX + 0.5;
  });
  check(
    "every kernel it draws sits inside that band",
    outside.length === 0,
    outside.map((error) => `${error}px → σ ${sigmaFor(error).toFixed(0)}px`).join(", ")
  );
  check(
    "an unmeasured calibration takes the band's floor rather than a finer guess",
    sigmaFor(null) === MIN_KERNEL_SIGMA_PX
  );
  check(
    "with no stimulus rect to measure against, the old constant still stands in",
    kernelRatio(60, 0) === DEFAULT_KERNEL_RATIO
  );
  check(
    "and on a small stimulus the blob is capped at a fraction of the picture",
    kernelRatio(120, 300) === MAX_KERNEL_RATIO,
    `${kernelRatio(120, 300)}`
  );

  /**
   * What that is worth on the picture. The critic measured a 27-fixation study
   * rendering 5.5% of the overlay with any paint at all and 0.44% past half
   * alpha — pinpricks, where the evidence this tool claims to produce is
   * component-scale. Below is the same shape of study — 27 fixations over three
   * clusters, the way a task-driven session actually lands — pushed through the
   * real renderer onto a stub canvas.
   */
  const clusters = [
    { x: 0.3, y: 0.2 },
    { x: 0.68, y: 0.44 },
    { x: 0.4, y: 0.78 },
  ];
  const points = Array.from({ length: 27 }, (_, i) => {
    const centre = clusters[i % clusters.length];
    return {
      x: centre.x + (((i * 37) % 11) - 5) * 0.012,
      y: centre.y + (((i * 23) % 9) - 4) * 0.014,
      weight: 200 + (i % 5) * 180,
    };
  });
  const coverage = (ratio: number): { any: number; strong: number } => {
    const canvas = stubCanvas(320, 250);
    renderHeatmap(canvas, points, { style: "heat", radiusRatio: ratio });
    const pixels = (canvas as unknown as { pixels: Uint8ClampedArray }).pixels;
    let any = 0;
    let strong = 0;
    for (let i = 3; i < pixels.length; i += 4) {
      if (pixels[i] > 0) any++;
      if (pixels[i] > 127) strong++;
    }
    const total = pixels.length / 4;
    return { any: any / total, strong: strong / total };
  };
  const before = coverage(DEFAULT_KERNEL_RATIO);
  const after = coverage(kernelRatio(60, stimulus));
  check(
    "a measured kernel paints the evidence at component scale",
    after.any > before.any * 2,
    `any paint ${(before.any * 100).toFixed(1)}% → ${(after.any * 100).toFixed(1)}%`
  );
  check(
    "and it stays a finding rather than becoming a wash",
    after.any < 0.6,
    `${(after.any * 100).toFixed(1)}% of the overlay painted`
  );

  // A kernel is a parameter, not a fact, so the picture states the one it was
  // drawn at — the way Tobii Pro Lab states its own. In both units: degrees are
  // the comparable one, pixels are the one visible on the image.
  const keyed = legendFor("heat", ["P01"], { ceiling: 1240, blur: { degrees: 1.5, pixels: 60 } });
  check("the legend states the kernel the field was blurred at", keyed.caption.includes("Blur ≈1.5° (60px)"), keyed.caption);
  check(
    "and saying so keeps the caption a caption",
    keyed.caption.split(/\s+/).length <= 14,
    `${keyed.caption.split(/\s+/).length} words`
  );
  check(
    "a scale with no measured blur says nothing about one",
    !legendFor("heat", ["P01"], { ceiling: 1240 }).caption.includes("Blur")
  );
}

// --- Scanpath ordinals ----------------------------------------------------

section("A fixation's number survives the circle it is drawn on");
{
  /**
   * The circle as it actually lands: the ramp at 55% over whatever the stimulus
   * has there. Four grounds a wireframe really contains, because the renderer
   * draws on a transparent overlay and cannot sample the picture underneath —
   * which is the whole reason the numeral cannot pick its fill from the ramp.
   */
  const grounds: Array<[string, [number, number, number]]> = [
    ["white wireframe", [255, 255, 255]],
    ["cream card", [238, 232, 216]],
    ["dark teal bar", [13, 74, 82]],
    ["near-black bar", [24, 37, 41]],
  ];
  const channel = (v: number): number => {
    const c = v / 255;
    return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  };
  const luminance = ([r, g, b]: number[]): number =>
    0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
  const ratio = (a: number, b: number): number =>
    (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
  const rgbOf = (colour: string): number[] =>
    colour.match(/[\d.]+/g)!.slice(0, 3).map(Number);
  const over = (fg: number[], alpha: number, bg: number[]): number[] =>
    fg.map((v, i) => v * alpha + bg[i] * (1 - alpha));

  const disc = (t: number, ground: number[]): number =>
    luminance(over(rgbOf(scanpathColour(t)), CIRCLE_ALPHA, ground));
  const ink = luminance(rgbOf("rgba(255,255,255,1)"));
  const casing = luminance(rgbOf(ORDINAL_HALO));
  const wash = (t: number, ground: number[]): number =>
    luminance(over(rgbOf(ORDINAL_HALO), 0.75, over(rgbOf(scanpathColour(t)), CIRCLE_ALPHA, ground)));

  const samples = Array.from({ length: 21 }, (_, i) => i / 20);

  // The measured failure: white on the pale end of viridis over a white
  // wireframe. It is real, and it is why the numeral cannot be read from its
  // fill alone.
  check(
    "a white numeral has no contrast at all on the light end of the ramp",
    ratio(disc(1, [255, 255, 255]), ink) < 1.3,
    `${ratio(disc(1, [255, 255, 255]), ink).toFixed(2)}:1 over a white wireframe`
  );
  // And why picking the ink from the ramp instead does not fix it: the same
  // late circle over a dark nav bar is the other way round.
  check(
    "and the same circle over a dark bar is the opposite problem",
    ratio(disc(1, [13, 74, 82]), ink) > 2.5,
    `${ratio(disc(1, [13, 74, 82]), ink).toFixed(2)}:1 over a dark teal bar`
  );

  // So the numeral is read from its casing, and the pair has to separate on
  // every ground: either the fill or the casing carries it.
  let worstNow = Infinity;
  let worstBefore = Infinity;
  let where = "";
  for (const t of samples) {
    for (const [name, ground] of grounds) {
      const level = disc(t, ground);
      const now = Math.max(ratio(level, ink), ratio(level, casing));
      const before = Math.max(ratio(level, ink), ratio(level, wash(t, ground)));
      if (now < worstNow) {
        worstNow = now;
        where = `${name} at t=${t.toFixed(2)}`;
      }
      worstBefore = Math.min(worstBefore, before);
    }
  }
  check(
    "either the numeral or its casing clears 4:1 on every ground, everywhere on the ramp",
    worstNow >= 4,
    `worst ${worstNow.toFixed(2)}:1 — ${where}`
  );
  check(
    "an opaque casing beats the 75% wash it replaces",
    worstNow > worstBefore,
    `worst case ${worstBefore.toFixed(2)}:1 → ${worstNow.toFixed(2)}:1`
  );
  const regressions = samples.flatMap((t) =>
    grounds.filter(([, ground]) => {
      const level = disc(t, ground);
      return ratio(level, casing) < ratio(level, wash(t, ground)) - 1e-9;
    })
  );
  check("and it is not worse anywhere", regressions.length === 0, `${regressions.length} cases`);
  check(
    "the pale end of the ramp gains the most, which is where the failure was",
    ratio(disc(1, [255, 255, 255]), casing) > 15,
    `${ratio(disc(1, [255, 255, 255]), wash(1, [255, 255, 255])).toFixed(1)}:1 → ${ratio(disc(1, [255, 255, 255]), casing).toFixed(1)}:1`
  );
}

// --- Result --------------------------------------------------------------

console.log(`\n${checks - failures}/${checks} checks passed`);
if (failures > 0) {
  console.log(`${failures} FAILED`);
  process.exit(1);
}
