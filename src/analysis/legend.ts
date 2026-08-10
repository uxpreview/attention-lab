import { contourBandColours, rampColour, SPOTLIGHT_MAX_DIM, type HeatmapStyle } from "./heatmap";
import { scanpathColour } from "./scanpath";

/**
 * What each overlay's colours mean, described once.
 *
 * Every commercial tool ships a scale strip, and for a good reason: a
 * coloured wash with no key is a picture, not a measurement. The spec below is
 * built from the same lookup tables the pixels come from — alpha included, so
 * the strip fades out at the cold end exactly where the overlay does — and a
 * change to RAMP or to the scanpath hue sweep moves the legend with it. It is
 * deliberately renderer-agnostic — the results screen draws it as DOM, the PNG
 * export draws the same spec into the file's caption band, and neither one
 * gets to invent its own version.
 */

export type OverlayMode = HeatmapStyle | "scanpath" | "raw";

export const OVERLAY_LABELS: Record<OverlayMode, string> = {
  heat: "Heatmap",
  spotlight: "Spotlight",
  contour: "Contours",
  scanpath: "Scanpath",
  raw: "Raw gaze",
};

/**
 * Participant colours for the raw-gaze view.
 *
 * The previous `hsla(i * 67, …)` sweep put participant 1 on red and
 * participant 2 on green, which is the one pair a red-green colour-blind
 * viewer cannot separate — and three overlapping dot clouds are unreadable if
 * you cannot tell whose is whose. This is the Okabe-Ito qualitative palette,
 * designed to stay distinguishable under all common forms of colour vision
 * deficiency.
 */
export const PARTICIPANT_COLOURS = [
  "#0072b2", // blue
  "#e69f00", // orange
  "#009e73", // bluish green
  "#cc79a7", // reddish purple
  "#56b4e9", // sky blue
  "#d55e00", // vermillion
  "#8a5fbf", // violet
  "#332288", // indigo
];

export function participantColour(index: number, alpha = 1): string {
  const hex = PARTICIPANT_COLOURS[index % PARTICIPANT_COLOURS.length];
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export interface LegendSwatch {
  colour: string;
  label: string;
}

export interface LegendSpec {
  /** What the colour axis is measuring. */
  title: string;
  /** Gradient stops, coldest/earliest first, or null when the key is discrete. */
  stops: string[] | null;
  /** Draw the stops as hard steps rather than a blend. */
  banded: boolean;
  minLabel: string;
  maxLabel: string;
  /** A colour-per-thing key, or null when the scale is continuous. */
  swatches: LegendSwatch[] | null;
  /** The caveat a reader needs before citing the picture. */
  note: string;
}

function heatStops(count = 9): string[] {
  const out: string[] = [];
  for (let i = 0; i < count; i++) out.push(rampColour(i / (count - 1)));
  return out;
}

function scanpathStops(count = 7): string[] {
  const out: string[] = [];
  for (let i = 0; i < count; i++) out.push(scanpathColour(i / (count - 1), 0.85));
  return out;
}

/**
 * The key for one overlay over one selection. `participants` is in the same
 * order the raw view draws them, because that is what makes its swatches
 * attributable.
 */
export function legendFor(mode: OverlayMode, participants: string[]): LegendSpec {
  switch (mode) {
    case "heat":
      return {
        title: "Fixation time per area",
        stops: heatStops(),
        banded: false,
        minLabel: "Clear: no attention",
        maxLabel: "Most looked at",
        swatches: null,
        // The percentile named here is the one renderHeatmap actually defaults
        // to, and it is a percentile of the per-blob peaks rather than of the
        // pixels — see fieldCeiling. It said "98th" for a clamp that no longer
        // exists.
        note: "Colour is total fixation duration (ms) summed over the selected participants, scaled so the 90th percentile of the attention peaks saturates the hot end. Areas nobody fixated are left clear rather than tinted, so the paint is the finding. The scale is relative to this selection — colours are not comparable between studies.",
      };
    case "contour":
      return {
        title: "Fixation time per area",
        stops: contourBandColours(),
        banded: true,
        minLabel: "Lowest drawn band",
        maxLabel: "Most looked at",
        swatches: null,
        note: "The same fixation-duration scale as the heatmap, quantised into equal bands so a region can be cited by band. Below the first band is left clear.",
      };
    case "spotlight":
      return {
        title: "Where they looked",
        stops: [
          `rgba(0, 0, 0, ${(SPOTLIGHT_MAX_DIM / 255).toFixed(2)})`,
          "rgba(0, 0, 0, 0.45)",
          "rgba(0, 0, 0, 0)",
        ],
        banded: false,
        minLabel: "Dimmed: unlooked",
        maxLabel: "Clear: looked at",
        swatches: null,
        note: "A mask rather than a heat overlay: the stimulus is dimmed everywhere except where fixations landed. Reveal is boosted, so a moderately-attended region clears fully.",
      };
    case "scanpath":
      return {
        title: "Order of fixations",
        stops: scanpathStops(),
        banded: false,
        minLabel: "First",
        maxLabel: "Last",
        swatches: null,
        note: "Circle area is dwell time and the numeral is the ordinal. One participant at a time — an averaged scanpath would be a path nobody took.",
      };
    case "raw":
      return {
        title: "Gaze samples by participant",
        stops: null,
        banded: false,
        minLabel: "",
        maxLabel: "",
        swatches: participants.map((label, i) => ({ colour: participantColour(i), label })),
        note: "Every gaze sample, roughly 30 per second, before fixation detection. Density reads as saturation; a lone dot is as likely to be tracker noise as a look.",
      };
  }
}
