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

/** Early (cool) to late (warm), the same mapping the legend draws. `t` is the
 * fixation's position in the sequence, 0 to 1. */
export function scanpathColour(t: number, alpha = 1, lightness = 55): string {
  const clamped = Math.max(0, Math.min(1, t));
  const hue = 210 - clamped * 210;
  return `hsla(${hue}, 85%, ${lightness}%, ${alpha})`;
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
export function layoutOrdinals(circles: OrdinalCircle[], scale = 1): OrdinalLabel[] {
  const placed: Placed[] = [];
  const out: OrdinalLabel[] = [];

  for (let i = 0; i < circles.length; i++) {
    const c = circles[i];
    const fontSize = Math.max(
      MIN_ORDINAL_PX * scale,
      Math.min(MAX_ORDINAL_PX * scale, c.radius * 0.85)
    );
    const halfWidth = labelHalfWidth(String(i + 1), fontSize);
    const halfHeight = fontSize * 0.55;

    let spot: Placed = { x: c.x, y: c.y, halfWidth, halfHeight };
    let leader = false;

    if (placed.some((other) => overlaps(spot, other))) {
      const reach = c.radius + halfHeight + 4;
      const found = CALLOUT_ANGLES.map((angle) => ({
        x: c.x + Math.cos(angle) * (reach + halfWidth * Math.abs(Math.cos(angle))),
        y: c.y + Math.sin(angle) * reach,
        halfWidth,
        halfHeight,
      })).find((candidate) => !placed.some((other) => overlaps(candidate, other)));

      // Every position taken: park it above the circle anyway. A label that
      // still collides is no worse than the collision it started with, and
      // dropping the number outright would lose a step of the sequence.
      spot = found ?? { x: c.x, y: c.y - reach, halfWidth, halfHeight };
      leader = true;
    }

    placed.push(spot);
    out.push({ ...spot, fontSize, leader });
  }

  return out;
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
  const labels = layoutOrdinals(circles, scale);

  for (let i = 0; i < circles.length; i++) {
    const c = circles[i];
    const text = String(i + 1);
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
