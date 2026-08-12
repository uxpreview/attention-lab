import type { FaceLandmarkerResult } from "@mediapipe/tasks-vision";

/**
 * Turns a face mesh into the feature vector the gaze regression consumes.
 *
 * The physics we are approximating: where you look is determined by where your
 * eyeballs point *plus* where your head is, which way it faces, and how far
 * back it sits. Iris offset alone works only if the head never moves, which no
 * participant manages. So the basis includes head pose, viewing distance and,
 * crucially, interaction terms — a given iris offset means a different screen
 * position depending on head yaw and on how close the participant is leaning,
 * and both couplings are multiplicative, not additive.
 */

// Landmark indices in MediaPipe's 478-point refined mesh. "Left" and "right"
// follow MediaPipe's convention (the subject's own left/right).
const LEFT_IRIS = [474, 475, 476, 477] as const;
const RIGHT_IRIS = [469, 470, 471, 472] as const;
const LEFT_EYE_OUTER = 263;
const LEFT_EYE_INNER = 362;
const LEFT_EYE_TOP = 386;
const LEFT_EYE_BOTTOM = 374;
const RIGHT_EYE_OUTER = 33;
const RIGHT_EYE_INNER = 133;
const RIGHT_EYE_TOP = 159;
const RIGHT_EYE_BOTTOM = 145;
const NOSE_TIP = 1;
const CHIN = 152;
const FOREHEAD = 10;

/** Number of entries in the vector returned by {@link buildFeatureVector}. */
export const FEATURE_DIM = 25;

/**
 * Bumped whenever the basis changes shape, order, or meaning. Persisted
 * calibrations carry this so an old model is invalidated rather than silently
 * applied to features it was never fit on.
 */
export const FEATURE_BASIS_VERSION = 3;

/** Per-eye openness below which the lid is occluding the iris (blink). */
export const BLINK_CLOSE_OPENNESS = 0.16;
/** Openness a closing eye must recover past before frames are trusted again.
 * Higher than the close threshold on purpose: hysteresis stops a half-open
 * lid from flickering the tracker on and off mid-blink. */
export const BLINK_OPEN_OPENNESS = 0.2;

export interface Point2 {
  x: number;
  y: number;
}

export interface EyeState {
  /** Iris centre in normalised video coordinates. */
  iris: Point2;
  /** Iris offset from the eye centre, normalised by eye width. */
  offset: Point2;
  /** Lid separation over eye width. Below {@link BLINK_CLOSE_OPENNESS} the
   * lid is far enough down to occlude the iris. */
  openness: number;
}

export interface FaceState {
  left: EyeState;
  right: EyeState;
  /** Head rotation in radians, derived from the transformation matrix. */
  yaw: number;
  pitch: number;
  roll: number;
  /** Nose tip in normalised video coordinates: where the head sits in frame. */
  headX: number;
  headY: number;
  /**
   * Outer-corner interocular distance as a fraction of the frame width.
   *
   * This is the viewing-distance signal: it grows as the face approaches the
   * camera. Unnormalised on purpose — every ratio of two lengths on a face is
   * distance-invariant, so dividing this by anything else on the face destroys
   * exactly the information it is here to carry.
   */
  interocular: number;
  /**
   * Interocular distance over face height: apparent eye size, which is
   * deliberately distance-*invariant*.
   *
   * It varies with pitch — a face tilted away foreshortens vertically and the
   * ratio climbs — and between participants, so it is a useful pose and
   * identity cue. It is not a distance cue, despite reading like one.
   */
  scale: number;
  /** Mean eye openness across both eyes. */
  openness: number;
}

interface Landmark {
  x: number;
  y: number;
  z: number;
}

function centroid(landmarks: Landmark[], indices: readonly number[]): Point2 {
  let x = 0;
  let y = 0;
  for (const i of indices) {
    x += landmarks[i].x;
    y += landmarks[i].y;
  }
  return { x: x / indices.length, y: y / indices.length };
}

function dist(a: Landmark | Point2, b: Landmark | Point2): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function eyeState(
  landmarks: Landmark[],
  irisIndices: readonly number[],
  outerIdx: number,
  innerIdx: number,
  topIdx: number,
  bottomIdx: number,
  aspect: number
): EyeState {
  const iris = centroid(landmarks, irisIndices);
  const outer = landmarks[outerIdx];
  const inner = landmarks[innerIdx];
  const top = landmarks[topIdx];
  const bottom = landmarks[bottomIdx];

  // Normalised landmark coordinates are relative to the frame, so x and y are
  // on different physical scales. Correct y by the aspect ratio before any
  // distance is taken, otherwise vertical gaze is systematically compressed.
  const cx = (outer.x + inner.x) / 2;
  const cy = (outer.y + inner.y) / 2;
  const width = Math.hypot(outer.x - inner.x, (outer.y - inner.y) * aspect);
  const safeWidth = width > 1e-6 ? width : 1e-6;

  return {
    iris,
    offset: {
      x: (iris.x - cx) / safeWidth,
      y: ((iris.y - cy) * aspect) / safeWidth,
    },
    openness: Math.hypot(top.x - bottom.x, (top.y - bottom.y) * aspect) / safeWidth,
  };
}

/**
 * Extracts head pose from MediaPipe's 4x4 model-to-camera matrix.
 * The matrix is column-major; the upper-left 3x3 block is the rotation.
 */
function headPose(matrix: number[] | undefined): { yaw: number; pitch: number; roll: number } {
  if (!matrix || matrix.length < 16) return { yaw: 0, pitch: 0, roll: 0 };

  const r00 = matrix[0];
  const r10 = matrix[1];
  const r20 = matrix[2];
  const r21 = matrix[6];
  const r22 = matrix[10];

  const sy = Math.hypot(r00, r10);
  if (sy > 1e-6) {
    return {
      pitch: Math.atan2(r21, r22),
      yaw: Math.atan2(-r20, sy),
      roll: Math.atan2(r10, r00),
    };
  }

  // Gimbal lock: cos(yaw) ~ 0, so pitch and roll are no longer separable.
  // Standard fallback for the Rz*Ry*Rx decomposition (Slabaugh, "Computing
  // Euler angles from a rotation matrix"): fix roll = 0 and read the combined
  // angle from the first row, whose sign depends on which pole yaw hit.
  // Unreachable in practice — isUsableFace rejects |yaw| > 0.7 rad long before
  // this — but kept correct for anyone checking the maths.
  const r01 = matrix[4];
  const r02 = matrix[8];
  return {
    pitch: r20 < 0 ? Math.atan2(r01, r02) : Math.atan2(-r01, -r02),
    yaw: Math.atan2(-r20, sy),
    roll: 0,
  };
}

/**
 * Reads a face mesh result into a {@link FaceState}, or null when no usable
 * face is present.
 */
export function readFaceState(
  result: FaceLandmarkerResult,
  videoWidth: number,
  videoHeight: number
): FaceState | null {
  const landmarks: Landmark[] | undefined = result.faceLandmarks?.[0];
  // Fewer than 478 points means the iris refinement is missing, and without
  // irises there is no gaze signal to extract.
  if (!landmarks || landmarks.length < 478) return null;

  const aspect = videoHeight > 0 ? videoHeight / videoWidth : 1;

  const left = eyeState(
    landmarks,
    LEFT_IRIS,
    LEFT_EYE_OUTER,
    LEFT_EYE_INNER,
    LEFT_EYE_TOP,
    LEFT_EYE_BOTTOM,
    aspect
  );
  const right = eyeState(
    landmarks,
    RIGHT_IRIS,
    RIGHT_EYE_OUTER,
    RIGHT_EYE_INNER,
    RIGHT_EYE_TOP,
    RIGHT_EYE_BOTTOM,
    aspect
  );

  const { yaw, pitch, roll } = headPose(result.facialTransformationMatrixes?.[0]?.data);

  const nose = landmarks[NOSE_TIP];
  const interocular = dist(landmarks[LEFT_EYE_OUTER], landmarks[RIGHT_EYE_OUTER]);
  const faceHeight = dist(landmarks[FOREHEAD], landmarks[CHIN]);

  return {
    left,
    right,
    yaw,
    pitch,
    roll,
    headX: nose.x,
    headY: nose.y,
    interocular,
    scale: faceHeight > 1e-6 ? interocular / faceHeight : interocular,
    openness: (left.openness + right.openness) / 2,
  };
}

/**
 * Builds the regression basis. Order matters only in that it must stay stable
 * between fitting and prediction — bump {@link FEATURE_BASIS_VERSION} on any
 * change so persisted calibrations are invalidated rather than silently
 * misread.
 */
export function buildFeatureVector(face: FaceState): number[] {
  // Mean-and-difference parameterisation of the two eyes: dx/dy carry gaze
  // direction, vx/vy carry vergence — the eyes' disagreement, a direct
  // viewing-distance cue. The raw per-eye offsets are deliberately absent:
  // they are exact linear combinations of these four, and feeding both would
  // leave ridge splitting weight arbitrarily between collinear columns.
  const dx = (face.left.offset.x + face.right.offset.x) / 2;
  const dy = (face.left.offset.y + face.right.offset.y) / 2;
  const vx = face.left.offset.x - face.right.offset.x;
  const vy = face.left.offset.y - face.right.offset.y;

  return [
    dx,
    dy,
    vx,
    vy,
    dx * dx,
    dy * dy,
    dx * dy,
    face.yaw,
    face.pitch,
    face.roll,
    face.headX,
    face.headY,
    face.scale,
    // Interaction terms: the same iris offset maps to a different screen point
    // depending on head pose and distance. Both same-axis and cross-axis pose
    // pairs are here because a webcam above or below the eye line couples
    // horizontal iris offset with vertical pose, and vice versa.
    dx * face.yaw,
    dy * face.pitch,
    dx * face.pitch,
    dy * face.yaw,
    dx * face.headX,
    dy * face.headY,
    dx * face.scale,
    dy * face.scale,
    face.yaw * face.pitch,
    // Viewing distance, and the two terms that let it modulate gaze gain.
    //
    // Sitting closer means the same screen point sits at a wider angle, so the
    // iris has to travel further to reach it — the gain from iris offset to
    // screen position is a function of distance, and no amount of fitting on
    // pose alone can stand in for it. Nothing else in this basis carries that:
    // `scale` is a ratio of two face lengths, which is distance-invariant by
    // construction, and so are the two interactions built on it.
    face.interocular,
    dx * face.interocular,
    dy * face.interocular,
  ];
}

/**
 * Rejects frames that should not feed the model: blinks, extreme head turns,
 * and faces so far off-centre that the mesh is unreliable.
 */
export function isUsableFace(face: FaceState): boolean {
  // Gate on the more closed eye, not the mean: a one-eyed wink halves the
  // gaze signal but barely moves the average.
  if (Math.min(face.left.openness, face.right.openness) < BLINK_CLOSE_OPENNESS) return false;
  if (Math.abs(face.yaw) > 0.7 || Math.abs(face.pitch) > 0.7) return false;
  if (face.headX < 0.05 || face.headX > 0.95) return false;
  if (face.headY < 0.05 || face.headY > 0.95) return false;
  return true;
}
