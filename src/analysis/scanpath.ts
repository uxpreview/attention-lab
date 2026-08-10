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
export function layoutOrdinals(circles: OrdinalCircle[]): OrdinalLabel[] {
  const placed: Placed[] = [];
  const out: OrdinalLabel[] = [];

  for (let i = 0; i < circles.length; i++) {
    const c = circles[i];
    const fontSize = Math.max(11, Math.min(20, c.radius * 0.85));
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

  const width = canvas.width;
  const height = canvas.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, width, height);
  if (fixations.length === 0) return;

  let maxDuration = 0;
  for (const f of fixations) maxDuration = Math.max(maxDuration, f.duration);
  if (maxDuration <= 0) maxDuration = 1;

  const px = (f: Fixation) => ({ x: f.x * width, y: f.y * height });

  // Saccades first, so the fixation circles sit on top of the connecting lines.
  // Drawn twice: a light halo underneath a dark line, so the path stays legible
  // over both pale wireframes and dark screenshots.
  const path = new Path2D();
  for (let i = 0; i < fixations.length; i++) {
    const p = px(fixations[i]);
    if (i === 0) path.moveTo(p.x, p.y);
    else path.lineTo(p.x, p.y);
  }
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.lineWidth = 5;
  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  ctx.stroke(path);
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(20,26,38,0.8)";
  ctx.stroke(path);

  const circles: Array<{ x: number; y: number; radius: number; t: number }> = [];

  for (let i = 0; i < fixations.length; i++) {
    const f = fixations[i];
    const p = px(f);
    // Area, not radius, scales with duration — otherwise long fixations read as
    // far more dominant than they are.
    const scaled = Math.sqrt(f.duration / maxDuration);
    const radius = minRadius + (maxRadius - minRadius) * scaled;
    const t = colourByTime ? i / Math.max(1, fixations.length - 1) : 0.5;
    circles.push({ x: p.x, y: p.y, radius, t });

    ctx.fillStyle = scanpathColour(t, 0.55);
    ctx.strokeStyle = scanpathColour(t, 0.95, 40);
    ctx.lineWidth = 2;

    ctx.beginPath();
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  if (!showNumbers) return;

  // Ordinals in a second pass, so a later circle cannot paint over an earlier
  // number.
  const labels = layoutOrdinals(circles);

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
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(255,255,255,0.85)";
      ctx.beginPath();
      ctx.moveTo(c.x + (dx / length) * c.radius, c.y + (dy / length) * c.radius);
      ctx.lineTo(spot.x, spot.y);
      ctx.stroke();
      ctx.lineWidth = 1.25;
      ctx.strokeStyle = scanpathColour(c.t, 0.95, 40);
      ctx.beginPath();
      ctx.moveTo(c.x + (dx / length) * c.radius, c.y + (dy / length) * c.radius);
      ctx.lineTo(spot.x, spot.y);
      ctx.stroke();
    }

    // The same family as the rest of the UI: canvas text in the system stack
    // renders in a visibly different typeface from every label around it.
    ctx.font = `600 ${spot.fontSize}px ${CANVAS_FONT_FAMILY}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(0,0,0,0.55)";
    ctx.strokeText(text, spot.x, spot.y);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, spot.x, spot.y);
  }
}
