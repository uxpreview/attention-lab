import { FaceTracker, type TrackerFrame } from "./faceTracker";
import {
  BLINK_CLOSE_OPENNESS,
  BLINK_OPEN_OPENNESS,
  buildFeatureVector,
  FEATURE_DIM,
  isUsableFace,
  readFaceState,
  type FaceState,
} from "./features";
import { MedianPoint, OneEuroPoint } from "./filter";
import { fitRidge, predict, type RidgeModel } from "./regression";

/**
 * The gaze engine: owns the tracker, the calibration model, and the smoothing
 * filter, and emits a stream of screen-space gaze estimates.
 */

export interface GazeSample {
  /** Screen x in CSS pixels, relative to the viewport. */
  x: number;
  /** Screen y in CSS pixels, relative to the viewport. */
  y: number;
  /** performance.now() timestamp of the source video frame. */
  t: number;
  /** Face state behind this sample, for diagnostics and quality gating. */
  face: FaceState;
}

export interface TrackerStatus {
  faceVisible: boolean;
  usable: boolean;
  calibrated: boolean;
  fps: number;
  /** Mean cross-validated calibration error in pixels, or null when uncalibrated. */
  calibrationError: number | null;
}

export type GazeListener = (sample: GazeSample) => void;
export type StatusListener = (status: TrackerStatus) => void;

export interface CalibrationSample {
  features: number[];
  targetX: number;
  targetY: number;
}

/**
 * Fitting fewer samples than ~3x the feature dimension invites an
 * ill-conditioned fit even with ridge behind it; 60 also guarantees several
 * distinct targets at the ~25 frames a calibration dot collects.
 */
const MIN_CALIBRATION_SAMPLES = Math.max(60, 3 * FEATURE_DIM);

/** Frames dropped after a blink ends, while the mesh re-settles on the iris. */
const POST_BLINK_SETTLE_FRAMES = 2;

export class GazeEngine {
  readonly tracker = new FaceTracker();

  private model: RidgeModel | null = null;
  private despike = new MedianPoint();
  private filter = new OneEuroPoint({ minCutoff: 0.9, beta: 0.012 });
  private gazeListeners = new Set<GazeListener>();
  private statusListeners = new Set<StatusListener>();
  private lastFace: FaceState | null = null;
  private faceVisible = false;
  private blinking = false;
  private blinkSettleFrames = 0;
  private collecting: CalibrationSample[] | null = null;
  private collectTarget: { x: number; y: number } | null = null;

  constructor() {
    this.tracker.onFrame(this.handleFrame);
  }

  get isCalibrated(): boolean {
    return this.model !== null;
  }

  get calibrationError(): number | null {
    // cvError is NaN when cross-validation could not run; report that as
    // "unknown" rather than letting NaN leak into the UI.
    return this.model && Number.isFinite(this.model.cvError) ? this.model.cvError : null;
  }

  get currentFace(): FaceState | null {
    return this.lastFace;
  }

  async start(onProgress?: (message: string) => void): Promise<void> {
    await this.tracker.start(onProgress);
  }

  stop(): void {
    this.tracker.stop();
    this.despike.reset();
    this.filter.reset();
    this.lastFace = null;
    this.faceVisible = false;
    this.blinking = false;
    this.blinkSettleFrames = 0;
  }

  onGaze(listener: GazeListener): () => void {
    this.gazeListeners.add(listener);
    return () => this.gazeListeners.delete(listener);
  }

  onStatus(listener: StatusListener): () => void {
    this.statusListeners.add(listener);
    return () => this.statusListeners.delete(listener);
  }

  /**
   * Begins accumulating calibration samples against a fixed screen target.
   * Samples keep arriving until {@link stopCollecting} is called, so the caller
   * controls dwell time per point.
   */
  startCollecting(targetX: number, targetY: number, into: CalibrationSample[]): void {
    this.collecting = into;
    this.collectTarget = { x: targetX, y: targetY };
  }

  stopCollecting(): void {
    this.collecting = null;
    this.collectTarget = null;
  }

  /** Fits a new gaze model from collected samples and installs it. */
  calibrate(samples: CalibrationSample[]): RidgeModel {
    if (samples.length < MIN_CALIBRATION_SAMPLES) {
      throw new Error(
        `Not enough calibration data (${samples.length} of ${MIN_CALIBRATION_SAMPLES} samples). Try again and hold your gaze on each dot.`
      );
    }
    const model = fitRidge(
      samples.map((s) => s.features),
      samples.map((s) => s.targetX),
      samples.map((s) => s.targetY)
    );
    this.setModel(model);
    return model;
  }

  setModel(model: RidgeModel | null): void {
    this.model = model;
    this.despike.reset();
    this.filter.reset();
    this.emitStatus();
  }

  getModel(): RidgeModel | null {
    return this.model;
  }

  private handleFrame = (frame: TrackerFrame): void => {
    const face = readFaceState(frame.result, frame.videoWidth, frame.videoHeight);
    this.faceVisible = face !== null;
    this.lastFace = face;

    if (!face) {
      this.emitStatus();
      return;
    }

    // Blink hysteresis. On the way out of a blink the lid still occludes part
    // of the iris and the mesh needs a moment to re-settle, so a blink only
    // ends once openness clears a higher threshold than the one that started
    // it, and the first frames after that are dropped too.
    const minOpenness = Math.min(face.left.openness, face.right.openness);
    if (minOpenness < BLINK_CLOSE_OPENNESS) {
      this.blinking = true;
    } else if (this.blinking && minOpenness >= BLINK_OPEN_OPENNESS) {
      this.blinking = false;
      this.blinkSettleFrames = POST_BLINK_SETTLE_FRAMES;
    }

    const settling = this.blinkSettleFrames > 0;
    if (settling) this.blinkSettleFrames--;

    const usable = isUsableFace(face) && !this.blinking && !settling;
    this.emitStatus(usable);
    if (!usable) return;

    const features = buildFeatureVector(face);

    if (this.collecting && this.collectTarget) {
      this.collecting.push({
        features,
        targetX: this.collectTarget.x,
        targetY: this.collectTarget.y,
      });
    }

    if (!this.model) return;

    const [rawX, rawY] = predict(this.model, features);
    if (!Number.isFinite(rawX) || !Number.isFinite(rawY)) return;

    // Median-of-three before One Euro: the One Euro filter is built to follow
    // fast jumps, so single-frame artifacts (a lid clipping the iris mid-blink)
    // must be removed before it, not by it.
    const [mx, my] = this.despike.filter(rawX, rawY);
    const [x, y] = this.filter.filter(mx, my, frame.timestamp);
    const sample: GazeSample = { x, y, t: frame.timestamp, face };
    for (const listener of this.gazeListeners) listener(sample);
  };

  private emitStatus(usable = false): void {
    if (this.statusListeners.size === 0) return;
    const status: TrackerStatus = {
      faceVisible: this.faceVisible,
      usable,
      calibrated: this.model !== null,
      fps: this.tracker.fps,
      calibrationError: this.calibrationError,
    };
    for (const listener of this.statusListeners) listener(status);
  }
}
