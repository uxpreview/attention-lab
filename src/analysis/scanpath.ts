import type { Fixation } from "./fixations";

/**
 * Scanpath rendering: numbered fixation circles joined by saccade lines, sized
 * by dwell. Answers "in what order did they read this", which a heatmap cannot.
 */

/** The UI's display family, spelled the way canvas wants it. Canvas cannot
 * read a CSS custom property, so `--font-display` has to be mirrored here; if
 * the two ever disagree the numerals on the stage render in a different
 * typeface from every label around them. */
export const CANVAS_FONT_FAMILY = '"Figtree", ui-sans-serif, system-ui, sans-serif';

/**
 * The order ramp: first to last, dark to light.
 *
 * This was `hsl(210 - t*210)` — a blue → cyan → green → yellow → red sweep.
 * A rainbow ramp is the one scale a data-literate reviewer will flag on sight:
 * hue carries no intrinsic order, so a reader cannot tell whether green comes
 * before or after yellow without consulting the key, the perceived brightness
 * jumps around inside it, and it collapses under deuteranopia and in greyscale
 * — which sat oddly beside this app's deliberate Okabe-Ito participant colours
 * and its deliberate rejection of the jet ramp for heat.
 *
 * Viridis instead: monotonically increasing in lightness, so first → last
 * survives a greyscale print and every common form of colour vision deficiency,
 * and reads as an ordering even with the legend covered up. Anchors are the
 * standard nine stops of the matplotlib ramp, interpolated in sRGB — close
 * enough to the true perceptual path at this sample count that the difference
 * is invisible, and it keeps this file free of a colour-space dependency.
 */
const VIRIDIS: Array<[number, number, number]> = [
  [68, 1, 84],
  [72, 40, 120],
  [62, 73, 137],
  [49, 104, 142],
  [38, 130, 142],
  [31, 158, 137],
  [53, 183, 121],
  [110, 206, 88],
  [253, 231, 37],
];

/** Early (dark) to late (light), the same mapping the legend draws. `t` is the
 * fixation's position in the sequence, 0 to 1. `lightness` keeps the meaning it
 * had as an HSL percentage: 55 is the ramp as sampled, and lower values darken
 * it for the rim drawn around each circle's fill. */
export function scanpathColour(t: number, alpha = 1, lightness = 55): string {
  const clamped = Math.max(0, Math.min(1, t));
  const pos = clamped * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(pos));
  const f = pos - i;
  const a = VIRIDIS[i];
  const b = VIRIDIS[i + 1];
  const shade = lightness / 55;
  const channel = (lo: number, hi: number): number =>
    Math.max(0, Math.min(255, Math.round((lo + (hi - lo) * f) * shade)));
  return `rgba(${channel(a[0], b[0])}, ${channel(a[1], b[1])}, ${channel(a[2], b[2])}, ${alpha})`;
}

export interface ScanpathOptions {
  /** Radius in pixels for the shortest fixation. */
  minRadius?: number;
  /** Radius in pixels for the longest fixation. */
  maxRadius?: number;
  showNumbers?: boolean;
  /** Draw a colour gradient over time (early = cool, late = warm). */
  colourByTime?: boolean;
  /**
   * Device pixels per CSS pixel. Radii are already given in device pixels, and
   * the numerals have to be too: with the ordinal clamp fixed in CSS pixels, a
   * retina canvas rendered a 20px numeral that landed on screen at 10px, half
   * of it eaten by its own contrast halo.
   */
  scale?: number;
}

/** A fixation circle, as far as label placement is concerned. */
export interface OrdinalCircle {
  x: number;
  y: number;
  radius: number;
  /** The number to print. Defaults to the circle's index + 1; stated only when
   * a subset of the path is being numbered and the ordinals still have to
   * refer to positions in the whole sequence. */
  ordinal?: number;
}

export interface OrdinalLabel {
  /** Label centre. */
  x: number;
  y: number;
  halfWidth: number;
  halfHeight: number;
  fontSize: number;
  /** True when the label had to leave its circle and needs a leader line. */
  leader: boolean;
}

type Placed = Pick<OrdinalLabel, "x" | "y" | "halfWidth" | "halfHeight">;

/** The drawing surface an ordinal has to stay inside, in the same pixels as
 * the circles. Optional: the pure placement tests have no canvas. */
export interface OrdinalBounds {
  width: number;
  height: number;
}

/** Ordinal size bounds, in CSS pixels before {@link ScanpathOptions.scale}.
 * The floor is the payload's legibility floor: the order of the sequence is the
 * only thing a scanpath says, and a numeral that needs leaning in to read says
 * it to nobody. */
const MIN_ORDINAL_PX = 14;
const MAX_ORDINAL_PX = 30;

/** Canvas text metrics are unavailable in the headless figure shim, and
 * measureText is overkill for two or three digits: the numerals are tabular
 * enough that a per-character advance is within a pixel of the truth. */
function labelHalfWidth(text: string, fontSize: number): number {
  return (text.length * fontSize * 0.58) / 2;
}

function overlaps(a: Placed, b: Placed): boolean {
  return (
    Math.abs(a.x - b.x) < a.halfWidth + b.halfWidth + 2 &&
    Math.abs(a.y - b.y) < a.halfHeight + b.halfHeight + 2
  );
}

/** Eight positions around the circle, starting above-right and going clockwise
 * — the order an annotator would try by hand. */
const CALLOUT_ANGLES = [-Math.PI / 4, -Math.PI / 2, 0, -(3 * Math.PI) / 4, Math.PI / 4, Math.PI, Math.PI / 2, (3 * Math.PI) / 4];

/** How far out the search for clear space goes, in rings of eight positions.
 * Six rings hold ~48 labels around one point, well past the density at which a
 * scanpath is worth reading at all; past that the leader lines would be longer
 * than the saccades. */
const MAX_LABEL_RINGS = 6;

/**
 * Places the ordinal for each fixation, avoiding collisions.
 *
 * The order of a scanpath is the only thing it communicates, so the numbers
 * are the payload rather than decoration — and real fixation density stacks
 * them into unreadable collisions ("11" printed under "12", a pair reading as
 * "78"). Each label goes at its circle's centre if that is free, and otherwise
 * walks out to the circle's edge through eight callout positions until it
 * finds clear space. Earlier labels win ties: the start of a path is the part
 * a reader is following.
 *
 * Pure, and exported, because this is the part worth testing — the drawing
 * around it is not.
 */
export function layoutOrdinals(
  circles: OrdinalCircle[],
  scale = 1,
  bounds?: OrdinalBounds
): OrdinalLabel[] {
  const placed: Placed[] = [];
  const out: OrdinalLabel[] = [];

  /** Keeps a label wholly on the drawing surface. Without this a fixation near
   * an edge pushed its number off the canvas — ordinal "2" arrived sliced in
   * half by the top of the stimulus — and the sequence lost a step at exactly
   * the place a reader is trying to follow it. */
  const clamp = (spot: Placed): Placed => {
    if (!bounds) return spot;
    const lockX = Math.max(spot.halfWidth, bounds.width - spot.halfWidth);
    const lockY = Math.max(spot.halfHeight, bounds.height - spot.halfHeight);
    return {
      ...spot,
      x: Math.min(Math.max(spot.x, spot.halfWidth), lockX),
      y: Math.min(Math.max(spot.y, spot.halfHeight), lockY),
    };
  };

  for (let i = 0; i < circles.length; i++) {
    const c = circles[i];
    const text = String(c.ordinal ?? i + 1);
    const centreFont = Math.max(
      MIN_ORDINAL_PX * scale,
      Math.min(MAX_ORDINAL_PX * scale, c.radius * 0.85)
    );

    let fontSize = centreFont;
    let halfWidth = labelHalfWidth(text, fontSize);
    let halfHeight = fontSize * 0.55;

    let spot: Placed = clamp({ x: c.x, y: c.y, halfWidth, halfHeight });
    let leader = false;

    if (placed.some((other) => overlaps(spot, other))) {
      // A displaced number is drawn at the floor size, not at its circle's
      // size. A "36" sitting 60px away from its circle at the same weight as a
      // number inside one reads as a loose digit in a field of loose digits;
      // at the legibility floor it recedes behind the in-circle ordinals and
      // the path stays the figure.
      fontSize = MIN_ORDINAL_PX * scale;
      halfWidth = labelHalfWidth(text, fontSize);
      halfHeight = fontSize * 0.55;

      const base = c.radius + halfHeight + 4;
      let found: Placed | undefined;

      // Expanding rings rather than one ring and a shrug.
      //
      // This used to try the eight callout positions once and then "park it
      // above the circle anyway" when all eight were taken. On the studies this
      // tool is built for — a task, a CTA, gaze landing in one place — that
      // branch is not the rare case, it is the normal one: eight or more
      // fixations inside one radius, and from the ninth onward every label
      // landed on top of an earlier one. "10" over "7" reads as "107", and the
      // ordering, which is the only thing a scanpath says, stopped being
      // readable at exactly the density that makes a scanpath interesting.
      //
      // So each ring steps out by a label height and is rotated off the last by
      // a fraction of the angular step, which walks outward through the gaps
      // instead of stacking labels along the same eight spokes. The leader-line
      // machinery below draws whatever distance this ends up needing.
      const ringStep = halfHeight * 2 + 6;
      const twist = Math.PI / CALLOUT_ANGLES.length;
      for (let ring = 0; ring < MAX_LABEL_RINGS && !found; ring++) {
        const reach = base + ring * ringStep;
        found = CALLOUT_ANGLES.map((angle) => {
          const a = angle + ring * twist;
          // Clamped *before* the overlap test, or a candidate that passes the
          // test off-canvas is then pulled back on top of a label that was
          // already there.
          return clamp({
            x: c.x + Math.cos(a) * (reach + halfWidth * Math.abs(Math.cos(a))),
            y: c.y + Math.sin(a) * reach,
            halfWidth,
            halfHeight,
          });
        }).find((candidate) => !placed.some((other) => overlaps(candidate, other)));
      }

      // Nothing clear even at the outermost ring — only reachable with a truly
      // absurd pile-up. Park it at the top of that ring, where it is at least
      // far from the circles: a label that still collides is no worse than the
      // collision it started with, and dropping the number outright would lose
      // a step of the sequence.
      spot =
        found ??
        clamp({
          x: c.x,
          y: c.y - (base + (MAX_LABEL_RINGS - 1) * ringStep),
          halfWidth,
          halfHeight,
        });
      leader = true;
    }

    placed.push(spot);
    out.push({ ...spot, fontSize, leader });
  }

  return out;
}

/** How many displaced labels a picture can carry before numbering every
 * fixation stops communicating the order it exists to communicate. Below this,
 * leader lines are a handful of tidy callouts; above it they are a thicket. */
export const MAX_DISPLACED_ORDINALS = 20;

/** How many ordinals survive the thinning. Twenty numbers is already more than
 * a reader will trace in order; the point of the cut is that the ones that
 * remain are readable. */
export const ORDINAL_BUDGET = 16;

/**
 * Which fixations keep their number when the path is too dense to number all of
 * them.
 *
 * With 67 fixations over six clusters every circle's ordinal is displaced, and
 * the satellites pile into each other around the spots — "53, 28, 29, 31, 3, 30,
 * 54" orbiting the first two circles with their leader lines crossing. The
 * ordering, which is the only thing a scanpath says, then survives only for the
 * few numbers that happened to land inside a circle.
 *
 * So the longest fixations keep their ordinals — duration is what makes a
 * fixation worth citing, and the circles are already sized by it, so the
 * numbers land on the marks a reader is looking at anyway. The first and last
 * are always kept whatever their duration: where the path starts and where it
 * ends are the two facts a reader looks for first.
 *
 * Returns indices into the original sequence, in sequence order. Pure and
 * exported because this is the part worth testing.
 */
export function selectOrdinals(durations: number[], budget = ORDINAL_BUDGET): number[] {
  if (durations.length <= budget) return durations.map((_, i) => i);

  const keep = new Set<number>([0, durations.length - 1]);
  const byDuration = durations
    .map((duration, index) => ({ duration, index }))
    // Longest first; ties break toward the earlier fixation so the selection is
    // deterministic and leans toward the start of the path.
    .sort((a, b) => b.duration - a.duration || a.index - b.index);

  for (const { index } of byDuration) {
    if (keep.size >= budget) break;
    keep.add(index);
  }

  return [...keep].sort((a, b) => a - b);
}

export function renderScanpath(
  canvas: HTMLCanvasElement,
  fixations: Fixation[],
  options: ScanpathOptions = {}
): void {
  const minRadius = options.minRadius ?? 10;
  const maxRadius = options.maxRadius ?? 46;
  const showNumbers = options.showNumbers ?? true;
  const colourByTime = options.colourByTime ?? true;
  const scale = options.scale ?? 1;

  const width = canvas.width;
  const height = canvas.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, width, height);
  if (fixations.length === 0) return;

  let maxDuration = 0;
  for (const f of fixations) maxDuration = Math.max(maxDuration, f.duration);
  if (maxDuration <= 0) maxDuration = 1;

  // Geometry for every fixation first, because the saccades need to know the
  // radius at both of their ends before they can be drawn.
  const circles: Array<{ x: number; y: number; radius: number; t: number }> = fixations.map(
    (f, i) => ({
      x: f.x * width,
      y: f.y * height,
      // Area, not radius, scales with duration — otherwise long fixations read
      // as far more dominant than they are.
      radius: minRadius + (maxRadius - minRadius) * Math.sqrt(f.duration / maxDuration),
      t: colourByTime ? i / Math.max(1, fixations.length - 1) : 0.5,
    })
  );

  // Saccades, clipped at each circle's rim rather than run through its middle.
  // Drawing the polyline underneath the circles is not enough on its own: the
  // fills are translucent, so a line crossing a circle still showed through and
  // struck out the numeral inside it. Each segment now starts and ends on the
  // rim, which is also how a saccade actually reads — a jump *between* two
  // fixations, not a line through them.
  //
  // Drawn twice: a light halo underneath a dark line, so the path stays legible
  // over both pale wireframes and dark screenshots.
  const saccades = new Path2D();
  for (let i = 1; i < circles.length; i++) {
    const a = circles[i - 1];
    const b = circles[i];
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const length = Math.hypot(dx, dy);
    // Overlapping circles have no gap to draw a saccade in; the ordering is
    // carried by the ordinals there.
    if (length <= a.radius + b.radius + 1) continue;
    saccades.moveTo(a.x + (dx / length) * a.radius, a.y + (dy / length) * a.radius);
    saccades.lineTo(b.x - (dx / length) * b.radius, b.y - (dy / length) * b.radius);
  }
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.lineWidth = 5 * scale;
  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  ctx.stroke(saccades);
  ctx.lineWidth = 2 * scale;
  ctx.strokeStyle = "rgba(20,26,38,0.8)";
  ctx.stroke(saccades);

  for (const c of circles) {
    ctx.fillStyle = scanpathColour(c.t, 0.55);
    ctx.strokeStyle = scanpathColour(c.t, 0.95, 40);
    ctx.lineWidth = 2 * scale;

    ctx.beginPath();
    ctx.arc(c.x, c.y, c.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  if (!showNumbers) return;

  // Ordinals in a second pass, so a later circle cannot paint over an earlier
  // number.
  //
  // The first pass is also the density measurement: every label that had to
  // leave its circle reports one collision, and past MAX_DISPLACED_ORDINALS of
  // them the numbers have stopped being readable as an ordering. In that case
  // the longest fixations keep their numbers and the pass is re-run over only
  // those — which is not just fewer labels but better-placed ones, since most
  // of them can now sit inside their own circle where they belong.
  const bounds = { width, height };
  let numbered = circles.map((c, i) => ({ ...c, ordinal: i + 1 }));
  let labels = layoutOrdinals(numbered, scale, bounds);

  if (labels.filter((label) => label.leader).length > MAX_DISPLACED_ORDINALS) {
    const keep = selectOrdinals(fixations.map((f) => f.duration));
    numbered = keep.map((i) => ({ ...circles[i], ordinal: i + 1 }));
    labels = layoutOrdinals(numbered, scale, bounds);
  }

  for (let i = 0; i < numbered.length; i++) {
    const c = numbered[i];
    const text = String(c.ordinal);
    const spot = labels[i];

    if (spot.leader) {
      // A hairline from the circle's edge to the label, so a number sitting
      // outside its circle still reads as belonging to it.
      const dx = spot.x - c.x;
      const dy = spot.y - c.y;
      const length = Math.hypot(dx, dy) || 1;
      ctx.lineWidth = 3 * scale;
      ctx.strokeStyle = "rgba(255,255,255,0.85)";
      ctx.beginPath();
      ctx.moveTo(c.x + (dx / length) * c.radius, c.y + (dy / length) * c.radius);
      ctx.lineTo(spot.x, spot.y);
      ctx.stroke();
      ctx.lineWidth = 1.25 * scale;
      ctx.strokeStyle = scanpathColour(c.t, 0.95, 40);
      ctx.beginPath();
      ctx.moveTo(c.x + (dx / length) * c.radius, c.y + (dy / length) * c.radius);
      ctx.lineTo(spot.x, spot.y);
      ctx.stroke();
    }

    // The same family as the rest of the UI: canvas text in the system stack
    // renders in a visibly different typeface from every label around it.
    //
    // The halo is a fraction of the glyph rather than a fixed 3px. At the old
    // sizes a 3px stroke closed most of the counter of a numeral and the white
    // fill barely survived it, which is how "large white numeral, dark halo"
    // rendered on screen as a small dark-grey one.
    ctx.font = `700 ${spot.fontSize}px ${CANVAS_FONT_FAMILY}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineJoin = "round";
    ctx.lineWidth = Math.max(2 * scale, spot.fontSize * 0.22);
    ctx.strokeStyle = "rgba(12,18,24,0.75)";
    ctx.strokeText(text, spot.x, spot.y);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, spot.x, spot.y);
  }
}
