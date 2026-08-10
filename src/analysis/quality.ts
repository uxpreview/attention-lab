import type { RecordingQuality } from "../data/types";

/**
 * Recording quality thresholds, in one place.
 *
 * The app's own bench notes warn that a bad calibration "looks exactly like
 * data": it does not announce itself, it just returns gaze that is wrong by a
 * consistent amount. So the two numbers that decide whether a recording is
 * worth citing — how much of the session was actually tracked, and how far off
 * the validation pass landed — are graded here rather than being formatted as
 * neutral grey text at three different call sites.
 *
 * "bad" is the threshold at which a recording is excluded from the aggregate
 * by default. It is deliberately generous: the point is to catch sessions that
 * are wrong, not to police ones that are merely imperfect.
 */

export type QualityGrade = "good" | "warn" | "bad";

/** Fraction of gaze samples that landed on the stimulus. */
export const TRACKING_GOOD = 0.8;
export const TRACKING_BAD = 0.6;

/** Validation error in degrees of visual angle. Commercial rigs claim 0.5°;
 * webcam tracking lands around 1.5-3° on a good day, and past 4° a heatmap is
 * no longer resolving anything the size of a component. */
export const ERROR_GOOD_DEG = 2;
export const ERROR_BAD_DEG = 4;

/**
 * Converts pixels to approximate degrees of visual angle. Assumes ~96 CSS ppi
 * and a 60cm viewing distance — both are rough, which is why the UI always
 * shows the raw pixel figure alongside.
 */
export function pxToDegrees(px: number, viewingDistanceCm = 60): number {
  const cm = (px / 96) * 2.54;
  return (Math.atan2(cm, viewingDistanceCm) * 180) / Math.PI;
}

export function gradeTracking(ratio: number): QualityGrade {
  if (ratio >= TRACKING_GOOD) return "good";
  if (ratio >= TRACKING_BAD) return "warn";
  return "bad";
}

/** An unmeasured calibration is "warn", never "bad": not knowing the error is
 * a reason to be careful, but it is not evidence the recording is wrong. */
export function gradeError(errorPx: number | null): QualityGrade {
  if (errorPx === null || !Number.isFinite(errorPx)) return "warn";
  const degrees = pxToDegrees(errorPx);
  if (degrees < ERROR_GOOD_DEG) return "good";
  if (degrees < ERROR_BAD_DEG) return "warn";
  return "bad";
}

const RANK: Record<QualityGrade, number> = { good: 0, warn: 1, bad: 2 };

/** The worse of the two grades: one broken axis is enough to spoil a session. */
export function gradeRecording(quality: RecordingQuality): QualityGrade {
  const tracking = gradeTracking(quality.trackingRatio);
  const error = gradeError(quality.validationError);
  return RANK[error] > RANK[tracking] ? error : tracking;
}

/** Below the line where a recording should be folded into an aggregate. */
export function isLowSignal(quality: RecordingQuality): boolean {
  return gradeRecording(quality) === "bad";
}

/** Why a recording was flagged, for the badge's tooltip. */
export function lowSignalReason(quality: RecordingQuality): string {
  const reasons: string[] = [];
  if (gradeTracking(quality.trackingRatio) === "bad") {
    reasons.push(`only ${Math.round(quality.trackingRatio * 100)}% of samples landed on the stimulus`);
  }
  if (gradeError(quality.validationError) === "bad" && quality.validationError !== null) {
    reasons.push(
      `calibration error ~${pxToDegrees(quality.validationError).toFixed(1)}° (${Math.round(quality.validationError)}px)`
    );
  }
  return reasons.join("; ");
}
