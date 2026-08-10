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
import { fieldPercentile } from "../analysis/heatmap";
import { analyseAois, aggregateAois, type Aoi } from "../analysis/aoi";
import { buildFeatureVector, FEATURE_DIM, type FaceState } from "../tracker/features";
import { MedianPoint, OneEuroPoint } from "../tracker/filter";
import {
  deserialiseModel,
  fitRidge,
  isSerialisedModel,
  predict,
  serialiseModel,
} from "../tracker/regression";

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

// --- Result --------------------------------------------------------------

console.log(`\n${checks - failures}/${checks} checks passed`);
if (failures > 0) {
  console.log(`${failures} FAILED`);
  process.exit(1);
}
