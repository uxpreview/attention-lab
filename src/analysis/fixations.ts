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

    // Salvucci & Goldberg's cheap dispersion measure: summed x and y ranges,
    // not spread. The bounds are kept as running min/max — the window start is
    // pinned from here on and points are only appended, so each candidate
    // point updates them in O(1) instead of rescanning the whole window.
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (let i = start; i <= end; i++) {
      const p = points[i];
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }

    if (maxX - minX + (maxY - minY) > dispersion) {
      // The eye was moving through this window: not a fixation, advance one.
      start++;
      continue;
    }

    // Window qualifies; extend it while it stays compact and continuous.
    while (end + 1 < points.length && points[end + 1].t - points[end].t <= maxGap) {
      const p = points[end + 1];
      const nextMinX = Math.min(minX, p.x);
      const nextMaxX = Math.max(maxX, p.x);
      const nextMinY = Math.min(minY, p.y);
      const nextMaxY = Math.max(maxY, p.y);
      if (nextMaxX - nextMinX + (nextMaxY - nextMinY) > dispersion) break;
      minX = nextMinX;
      maxX = nextMaxX;
      minY = nextMinY;
      maxY = nextMaxY;
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
      // Inter-sample span, the usual I-DT convention: the last sample's own
      // dwell (~one frame) is not counted. Worth knowing when comparing AOI
      // dwell sums against wall-clock time.
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
  /**
   * Time from `recordingStart` to the first fixation, in milliseconds.
   *
   * Only meaningful for a single recording's own fixations against that
   * recording's own start. It is *not* a research "time to first fixation":
   * capture begins with the participant already looking at the screen, so the
   * first fixation lands within a frame or two of t=0 and this is near zero by
   * construction. Over a concatenated list from several recordings it is
   * meaningless outright — it describes whichever recording happened to be
   * first. The results screen reports time to the first fixation inside a
   * region instead, which is what the metric is normally taken to mean.
   */
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
