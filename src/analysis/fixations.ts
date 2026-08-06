/**
 * Fixation detection.
 *
 * Raw gaze is a mix of fixations (the eye parked on something, ~100-400ms) and
 * saccades (ballistic jumps, during which you are functionally blind). Attention
 * lives in the fixations, so both scanpaths and the more useful heatmap variant
 * are built from fixations rather than from raw samples — otherwise the
 * in-between saccade samples smear heat across regions nobody actually read.
 *
 * The algorithm is I-DT (Salvucci & Goldberg, 2000): grow a window while the
 * points inside it stay within a dispersion threshold; if the window lasts long
 * enough, emit its centroid as a fixation.
 */

export interface RawPoint {
  x: number;
  y: number;
  t: number;
}

export interface Fixation {
  x: number;
  y: number;
  /** Duration in milliseconds. */
  duration: number;
  /** Start timestamp, in the same clock as the input points. */
  start: number;
  /** Number of raw samples that composed this fixation. */
  samples: number;
}

export interface FixationOptions {
  /**
   * Maximum allowed dispersion, in the same units as the point coordinates.
   * Roughly one degree of visual angle: about 40px at a typical desk setup.
   */
  dispersion?: number;
  /** Minimum duration in milliseconds for a window to count as a fixation. */
  minDuration?: number;
  /**
   * Gap in milliseconds that forcibly ends a window. Guards against tracking
   * dropouts being bridged into one implausibly long fixation.
   */
  maxGap?: number;
}

function dispersionOf(points: RawPoint[], from: number, to: number): number {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = from; i <= to; i++) {
    const p = points[i];
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  // Salvucci & Goldberg's cheap dispersion measure: summed ranges, not spread.
  return maxX - minX + (maxY - minY);
}

export function detectFixations(points: RawPoint[], options: FixationOptions = {}): Fixation[] {
  const dispersion = options.dispersion ?? 45;
  const minDuration = options.minDuration ?? 100;
  const maxGap = options.maxGap ?? 150;

  const fixations: Fixation[] = [];
  if (points.length === 0) return fixations;

  let start = 0;

  while (start < points.length) {
    // Grow the window until it *reaches* the minimum duration. The test is on
    // the window as it currently stands, not on the candidate point: testing
    // the candidate stops one point short and no window ever qualifies.
    let end = start;
    while (
      end + 1 < points.length &&
      points[end].t - points[start].t < minDuration &&
      points[end + 1].t - points[end].t <= maxGap
    ) {
      end++;
    }

    if (points[end].t - points[start].t < minDuration) {
      // Ran out of points, or a dropout cut the window short.
      start = end + 1;
      continue;
    }

    if (dispersionOf(points, start, end) > dispersion) {
      // The eye was moving through this window: not a fixation, advance one.
      start++;
      continue;
    }

    // Window qualifies; extend it while it stays compact and continuous.
    while (
      end + 1 < points.length &&
      points[end + 1].t - points[end].t <= maxGap &&
      dispersionOf(points, start, end + 1) <= dispersion
    ) {
      end++;
    }

    let sumX = 0;
    let sumY = 0;
    for (let i = start; i <= end; i++) {
      sumX += points[i].x;
      sumY += points[i].y;
    }
    const count = end - start + 1;

    fixations.push({
      x: sumX / count,
      y: sumY / count,
      duration: points[end].t - points[start].t,
      start: points[start].t,
      samples: count,
    });

    start = end + 1;
  }

  return fixations;
}

export interface ScanStats {
  fixationCount: number;
  meanFixationDuration: number;
  totalDwell: number;
  /** Total distance travelled between consecutive fixations. */
  scanpathLength: number;
  /** Time from recording start to the first fixation, in milliseconds. */
  timeToFirstFixation: number;
}

export function summarise(fixations: Fixation[], recordingStart: number): ScanStats {
  if (fixations.length === 0) {
    return {
      fixationCount: 0,
      meanFixationDuration: 0,
      totalDwell: 0,
      scanpathLength: 0,
      timeToFirstFixation: 0,
    };
  }

  let dwell = 0;
  let length = 0;
  for (let i = 0; i < fixations.length; i++) {
    dwell += fixations[i].duration;
    if (i > 0) {
      length += Math.hypot(fixations[i].x - fixations[i - 1].x, fixations[i].y - fixations[i - 1].y);
    }
  }

  return {
    fixationCount: fixations.length,
    meanFixationDuration: dwell / fixations.length,
    totalDwell: dwell,
    scanpathLength: length,
    timeToFirstFixation: fixations[0].start - recordingStart,
  };
}
