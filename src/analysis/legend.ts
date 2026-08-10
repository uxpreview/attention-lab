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

/** A value printed under the strip, at the fraction of it that carries it. */
export interface LegendTick {
  /** 0 at the cold end, 1 at the hot end. */
  at: number;
  label: string;
}

/**
 * What the top of the colour axis is worth, for the selection on the stage.
 *
 * `renderHeatmap` returns this — a percentile of the per-blob peaks of the
 * accumulated field — and the field is summed fixation duration in
 * milliseconds, Gaussian-weighted by distance within a splat radius. So the
 * number is "roughly this many milliseconds of looking, gathered in one spot",
 * which is what a reader asking "how much is dark red?" wants and is close
 * enough to a dwell figure to be worth printing. It is a ceiling, not a maximum:
 * anything above it saturates, hence the ≥ on the label.
 */
export interface LegendScale {
  /** Field units at the hot end of the ramp; the cold end is always 0. */
  ceiling: number;
}

/** Rounds to something a person would read off an axis, in the same units the
 * rail's own numbers are in (see formatMs in ui/dom.ts). Coarse on purpose: the
 * ceiling is a percentile estimate over blob peaks, and printing "1,237ms"
 * would claim a precision the statistic does not have. */
function formatScaleValue(ms: number): string {
  if (!Number.isFinite(ms) || ms <= 0) return "0ms";
  if (ms >= 1000) {
    const seconds = ms / 1000;
    return `${seconds >= 10 ? Math.round(seconds) : seconds.toFixed(1)}s`;
  }
  const step = ms >= 100 ? 10 : 5;
  return `${Math.max(step, Math.round(ms / step) * step)}ms`;
}

/** Both ends and the midpoint of a duration axis. Three is what a 460px strip
 * holds without the labels touching, and the midpoint is the one interior value
 * a reader actually uses — "is this blob half of the hot end, or a tenth of
 * it?". The ceiling carries a ≥ because everything above it saturates. */
function durationTicks(ceiling: number): LegendTick[] {
  return [
    { at: 0, label: "0ms" },
    { at: 0.5, label: formatScaleValue(ceiling * 0.5) },
    { at: 1, label: `≥${formatScaleValue(ceiling)}` },
  ];
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
  /** Values printed along the strip, when the axis has units and the caller
   * knew what the scale was set to. Null for the qualitative overlays, and for
   * a heatmap drawn before the ceiling is known. */
  ticks: LegendTick[] | null;
  /** A colour-per-thing key, or null when the scale is continuous. */
  swatches: LegendSwatch[] | null;
  /**
   * One clause naming what the picture is, shown inline beside the ramp.
   *
   * The full {@link note} is what a reader needs *once*, and it was being
   * printed in full on every view: a measured 68 words and 174px of the primary
   * analysis screen, every mode, forever, pushing the region table below the
   * fold. The caption is what stays visible; the note moves into a disclosure
   * beside it (and is forced open on paper, where a figure keeps its caption).
   */
  caption: string;
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
export function legendFor(
  mode: OverlayMode,
  participants: string[],
  scale: LegendScale | null = null
): LegendSpec {
  switch (mode) {
    case "heat":
      return {
        title: "Fixation time per area",
        stops: heatStops(),
        banded: false,
        minLabel: "Clear: no attention",
        maxLabel: "Most looked at",
        ticks: scale && scale.ceiling > 0 ? durationTicks(scale.ceiling) : null,
        swatches: null,
        caption: "Total fixation time per area, scaled to this selection.",
        // The percentile named here is the one renderHeatmap actually defaults
        // to, and it is a percentile of the per-blob peaks rather than of the
        // pixels — see fieldCeiling. It said "98th" for a clamp that no longer
        // exists.
        note: "Colour is total fixation duration (ms) summed over the selected participants, scaled so the 90th percentile of the attention peaks saturates the hot end — the figure printed at the top of the axis, which anything hotter reaches too. Areas nobody fixated are left clear rather than tinted, so the paint is the finding. The scale is relative to this selection — colours are not comparable between studies.",
      };
    case "contour":
      return {
        title: "Fixation time per area",
        stops: contourBandColours(),
        banded: true,
        minLabel: "Lowest drawn band",
        maxLabel: "Most looked at",
        // The bands are equal slices of the same scale the heatmap uses, so the
        // same ticks name them.
        ticks: scale && scale.ceiling > 0 ? durationTicks(scale.ceiling) : null,
        swatches: null,
        caption: "The heatmap's scale, quantised into bands you can cite.",
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
        // A mask has no unit: reveal is a boosted function of the same field,
        // and printing milliseconds under it would name a quantity the picture
        // does not encode.
        ticks: null,
        swatches: null,
        caption: "The stimulus dimmed everywhere except where fixations landed.",
        note: "A mask rather than a heat overlay: the stimulus is dimmed everywhere except where fixations landed, and a fully revealed area is the stimulus at full strength — never brighter. The dim stops short of black, so the rest of the screen stays legible as context rather than disappearing. Reveal is boosted, so a moderately-attended region clears.",
      };
    case "scanpath":
      return {
        title: "Order of fixations",
        stops: scanpathStops(),
        banded: false,
        minLabel: "First",
        maxLabel: "Last",
        ticks: null,
        swatches: null,
        caption: "Fixations in order, dark first through light last; area is dwell.",
        // The ramp is viridis rather than a hue sweep, so the strip above reads
        // as an ordering on its own — dark to light — and survives greyscale and
        // colour vision deficiency.
        //
        // The thinning is stated, and so is what the numerals actually count. On
        // a dense path only the longest fixations are numbered and they are
        // numbered 1, 2, 3 in sequence rather than by their true index — which
        // is what stops the picture reading as "1, 32, 14, 25" — so a reader who
        // counts circles and numbers and finds they disagree is owed the reason
        // in the caption rather than left to guess at it.
        note: "Circle area is dwell time, and colour runs dark for the first fixation through light for the last. Where fixations pile up, only the longest are numbered — the numerals count those marks in order, 1, 2, 3, and the unnumbered circles are the shorter stops between them, drawn in the same path. Every fixation keeps its true position in the sequence in the exported Fixations CSV. One participant at a time — an averaged scanpath would be a path nobody took.",
      };
    case "raw":
      return {
        title: "Gaze samples by participant",
        stops: null,
        banded: false,
        minLabel: "",
        maxLabel: "",
        ticks: null,
        swatches: participants.map((label, i) => ({ colour: participantColour(i), label })),
        caption: "Every gaze sample, before fixation detection.",
        note: "Every gaze sample, roughly 30 per second, before fixation detection. Density reads as saturation; a lone dot is as likely to be tracker noise as a look.",
      };
  }
}
