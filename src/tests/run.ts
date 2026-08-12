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
import { analyseAois, aggregateAois, type Aoi } from "../analysis/aoi";
import { buildFeatureVector, FEATURE_DIM, type FaceState } from "../tracker/features";
import { OneEuroPoint } from "../tracker/filter";
import { assessSetup, isReady, type SetupReading } from "../ui/setup";
import { deserialiseModel, fitRidge, predict, serialiseModel } from "../tracker/regression";

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
  const offsetX = Math.sin(eyeYaw) * gain + jitter();
  const offsetY = Math.sin(eyePitch) * gain + jitter();

  // The two eyes differ slightly through vergence.
  const eye = (dx: number, dy: number) => ({
    iris: { x: 0.5 + dx, y: 0.5 + dy },
    offset: { x: dx, y: dy },
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
    interocular: 0.12,
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

/**
 * Runs a full calibrate-then-test cycle. `drift` simulates the participant's
 * head moving between calibration and the recording, which is the single
 * biggest source of real-world degradation.
 */
function calibrateAndTest(noise: number, drift: number, seed: number): FitResult {
  const rand = makeRandom(seed);
  const baseHead = { yaw: 0.02, pitch: -0.03, x: 0.5, y: 0.48, scale: 0.62 };

  const rows: number[][] = [];
  const targetX: number[] = [];
  const targetY: number[] = [];

  for (const [nx, ny] of CALIBRATION_GRID) {
    for (let s = 0; s < 22; s++) {
      // Small natural head movement during calibration.
      const head = {
        yaw: baseHead.yaw + (rand() - 0.5) * 0.06,
        pitch: baseHead.pitch + (rand() - 0.5) * 0.05,
        x: baseHead.x + (rand() - 0.5) * 0.02,
        y: baseHead.y + (rand() - 0.5) * 0.02,
        scale: baseHead.scale + (rand() - 0.5) * 0.01,
      };
      const face = simulateFace(nx * SCREEN_W, ny * SCREEN_H, head, noise, rand);
      rows.push(buildFeatureVector(face));
      targetX.push(nx * SCREEN_W);
      targetY.push(ny * SCREEN_H);
    }
  }

  const model = fitRidge(rows, targetX, targetY);

  // Test on points the model never saw, with the head shifted by `drift`.
  const errors: number[] = [];
  for (let i = 0; i < 300; i++) {
    const tx = (0.08 + rand() * 0.84) * SCREEN_W;
    const ty = (0.08 + rand() * 0.84) * SCREEN_H;
    const head = {
      yaw: baseHead.yaw + (rand() - 0.5) * drift,
      pitch: baseHead.pitch + (rand() - 0.5) * drift * 0.8,
      x: baseHead.x + (rand() - 0.5) * drift * 0.3,
      y: baseHead.y + (rand() - 0.5) * drift * 0.3,
      scale: baseHead.scale + (rand() - 0.5) * drift * 0.15,
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
  check("horizontal gaze moves the iris offset monotonically", right[4] > left[4]);

  const up = buildFeatureVector(
    simulateFace(720, 80, { yaw: 0, pitch: 0, x: 0.5, y: 0.5, scale: 0.62 }, 0, rand)
  );
  const down = buildFeatureVector(
    simulateFace(720, 820, { yaw: 0, pitch: 0, x: 0.5, y: 0.5, scale: 0.62 }, 0, rand)
  );
  check("vertical gaze moves the iris offset monotonically", down[5] > up[5]);
}

// --- Regression ----------------------------------------------------------

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

// --- Setup checks --------------------------------------------------------

{
  section("Setup assessment");

  const good: SetupReading = {
    faceVisible: true,
    interocular: 0.12,
    headX: 0.5,
    headY: 0.5,
    fps: 28,
    faceLuma: 140,
    frameLuma: 120,
  };

  const state = (reading: SetupReading, id: string) =>
    assessSetup(reading).find((c) => c.id === id)!.state;

  check("well-set-up participant passes every check", isReady(assessSetup(good)));
  check("no face fails", state({ ...good, faceVisible: false }, "face") === "fail");
  check("too far warns", state({ ...good, interocular: 0.05 }, "distance") === "warn");
  check("too close warns", state({ ...good, interocular: 0.25 }, "distance") === "warn");
  check("off-centre warns", state({ ...good, headX: 0.15 }, "centering") === "warn");
  check("dark face fails lighting", state({ ...good, faceLuma: 30 }, "lighting") === "fail");
  check(
    "backlight warns",
    state({ ...good, faceLuma: 90, frameLuma: 180 }, "lighting") === "warn"
  );
  check("low fps warns", state({ ...good, fps: 9 }, "fps") === "warn");
  check(
    "missing luma sample is unknown, not a failure",
    state({ ...good, faceLuma: null, frameLuma: null }, "lighting") === "unknown"
  );
  check("not ready while any check warns", !isReady(assessSetup({ ...good, fps: 9 })));
}

// --- Result --------------------------------------------------------------

console.log(`\n${checks - failures}/${checks} checks passed`);
if (failures > 0) {
  console.log(`${failures} FAILED`);
  process.exit(1);
}
