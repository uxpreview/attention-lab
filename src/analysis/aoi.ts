import type { Fixation } from "./fixations";

/**
 * Areas of interest.
 *
 * A heatmap shows you where attention went; an AOI turns that into a number you
 * can put in a deck — "72% of participants looked at the CTA, and it took them
 * 4.1s to find it". For wireframe testing this is usually the actual
 * deliverable, so AOI stats are computed per participant and then aggregated.
 */

export interface Aoi {
  id: string;
  label: string;
  /** Normalised stimulus rect, all values in [0, 1]. */
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface AoiResult {
  aoiId: string;
  label: string;
  /** Total time fixated inside the AOI, in milliseconds. */
  dwell: number;
  fixationCount: number;
  /**
   * Time from recording start to the first fixation inside the AOI, or null if
   * it was never looked at.
   */
  timeToFirstFixation: number | null;
  /** Share of this participant's total fixation time spent in the AOI. */
  dwellShare: number;
}

export interface AoiAggregate {
  aoiId: string;
  label: string;
  /** Share of participants who fixated the AOI at least once. */
  hitRate: number;
  meanDwell: number;
  meanFixationCount: number;
  /** Mean TTFF across participants who found it at all. */
  meanTimeToFirstFixation: number | null;
  participants: number;
}

function contains(aoi: Aoi, f: Fixation): boolean {
  return (
    f.x >= aoi.x && f.x <= aoi.x + aoi.width && f.y >= aoi.y && f.y <= aoi.y + aoi.height
  );
}

export function analyseAois(
  aois: Aoi[],
  fixations: Fixation[],
  recordingStart: number
): AoiResult[] {
  let totalDwell = 0;
  for (const f of fixations) totalDwell += f.duration;

  return aois.map((aoi) => {
    let dwell = 0;
    let count = 0;
    let ttff: number | null = null;

    for (const f of fixations) {
      if (!contains(aoi, f)) continue;
      dwell += f.duration;
      count++;
      if (ttff === null) ttff = f.start - recordingStart;
    }

    return {
      aoiId: aoi.id,
      label: aoi.label,
      dwell,
      fixationCount: count,
      timeToFirstFixation: ttff,
      dwellShare: totalDwell > 0 ? dwell / totalDwell : 0,
    };
  });
}

/** Rolls per-participant AOI results into study-level numbers. */
export function aggregateAois(aois: Aoi[], perParticipant: AoiResult[][]): AoiAggregate[] {
  const participants = perParticipant.length;

  return aois.map((aoi) => {
    let hits = 0;
    let dwellSum = 0;
    let countSum = 0;
    let ttffSum = 0;
    let ttffCount = 0;

    for (const results of perParticipant) {
      const r = results.find((x) => x.aoiId === aoi.id);
      if (!r) continue;
      dwellSum += r.dwell;
      countSum += r.fixationCount;
      if (r.fixationCount > 0) hits++;
      if (r.timeToFirstFixation !== null) {
        ttffSum += r.timeToFirstFixation;
        ttffCount++;
      }
    }

    return {
      aoiId: aoi.id,
      label: aoi.label,
      hitRate: participants > 0 ? hits / participants : 0,
      meanDwell: participants > 0 ? dwellSum / participants : 0,
      meanFixationCount: participants > 0 ? countSum / participants : 0,
      meanTimeToFirstFixation: ttffCount > 0 ? ttffSum / ttffCount : null,
      participants,
    };
  });
}
