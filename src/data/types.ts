import type { Aoi } from "../analysis/aoi";
import type { BiasEstimate, ResidualSample } from "../tracker/regression";

/**
 * A study is a stimulus plus a task prompt. Recordings belong to a study, one
 * per participant, so heatmaps can be aggregated across everyone who saw the
 * same screen.
 */
export interface Study {
  id: string;
  name: string;
  /** The task the participant was asked to perform, shown before recording. */
  task: string;
  stimulus: Stimulus;
  aois: Aoi[];
  createdAt: number;
  /** Seconds; 0 means "until the moderator stops it". */
  duration: number;
}

export type Stimulus =
  | { kind: "image"; blob: Blob; width: number; height: number; name: string }
  | { kind: "url"; url: string };

/** One participant's session against one study. */
export interface Recording {
  id: string;
  studyId: string;
  participant: string;
  createdAt: number;
  /** Gaze in normalised stimulus coordinates, times in ms relative to the
   * recording's first sample — so t=0 is the moment gaze first arrived. */
  points: Array<{ x: number; y: number; t: number }>;
  /** Quality metadata captured at recording time. */
  quality: RecordingQuality;
}

export interface RecordingQuality {
  /** Mean validation error in pixels, measured after calibration. */
  validationError: number | null;
  /**
   * The per-dot validation residuals behind {@link validationError}, and the
   * constant offset extracted from them.
   *
   * Kept because the scalar above cannot say what kind of wrong a recording is,
   * and the kinds want different responses: an offset means the heat is the
   * right shape in the wrong place, scatter means the kernel is too tight, and
   * only the first can be subtracted. Without these a doubtful heatmap can only
   * be re-run, never diagnosed.
   *
   * Optional: recordings saved before this was captured have neither.
   */
  calibrationResiduals?: ResidualSample[];
  calibrationBias?: BiasEstimate | null;
  /** Fraction of expected samples actually captured. */
  trackingRatio: number;
  meanFps: number;
  /** Viewport size at record time, needed to convert px thresholds. */
  viewportWidth: number;
  viewportHeight: number;
  /** The stimulus display rect within the viewport, in CSS pixels. */
  stimulusRect: { x: number; y: number; width: number; height: number };
}
