/**
 * Regenerates the README figures in docs/figures/.
 *
 * The point is that the pictures cannot drift from the product: a synthetic
 * scanpath over a wireframe is pushed through the same `detectFixations` →
 * `renderHeatmap` / `renderScanpath` path the results screen uses, so whatever
 * those renderers draw is what lands in the README. Nothing here re-implements
 * the look of a heatmap.
 *
 * What it does implement is the browser surface those renderers draw onto: a
 * small RGBA raster plus the exact slice of `CanvasRenderingContext2D` they
 * touch (splat compositing, discs, polylines, digits) and a PNG encoder over
 * `node:zlib`. That is the price of no new dependencies, and it is bounded —
 * if a renderer starts using an API the shim lacks, this script throws rather
 * than quietly drawing something else.
 *
 * Run with: npm run figures
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { deflateSync } from "node:zlib";

import { detectFixations, type Fixation, type RawPoint } from "../src/analysis/fixations";
import { renderHeatmap, type HeatmapStyle } from "../src/analysis/heatmap";
import { renderScanpath } from "../src/analysis/scanpath";

/* ── Raster ──────────────────────────────────────────────────────────────── */

interface Rgba {
  r: number;
  g: number;
  b: number;
  a: number;
}

/** An RGBA pixel buffer with source-over compositing and analytic coverage. */
class Raster {
  readonly data: Uint8ClampedArray;

  constructor(
    readonly width: number,
    readonly height: number,
    fill: Rgba = { r: 0, g: 0, b: 0, a: 0 }
  ) {
    this.data = new Uint8ClampedArray(width * height * 4);
    this.fillRect(0, 0, width, height, fill);
  }

  /** Source-over blend of one pixel, `coverage` scaling the source alpha. */
  blend(x: number, y: number, c: Rgba, coverage = 1): void {
    if (x < 0 || y < 0 || x >= this.width || y >= this.height) return;
    const alpha = c.a * coverage;
    if (alpha <= 0) return;
    const i = (y * this.width + x) * 4;
    const inv = 1 - alpha;
    this.data[i] = c.r * alpha + this.data[i] * inv;
    this.data[i + 1] = c.g * alpha + this.data[i + 1] * inv;
    this.data[i + 2] = c.b * alpha + this.data[i + 2] * inv;
    this.data[i + 3] = 255 * alpha + this.data[i + 3] * inv;
  }

  /** Axis-aligned rectangle with fractional edges, so nothing shimmers. */
  fillRect(x: number, y: number, w: number, h: number, c: Rgba): void {
    for (let py = Math.floor(y); py < Math.ceil(y + h); py++) {
      const rowCover = Math.min(y + h, py + 1) - Math.max(y, py);
      for (let px = Math.floor(x); px < Math.ceil(x + w); px++) {
        const colCover = Math.min(x + w, px + 1) - Math.max(x, px);
        this.blend(px, py, c, Math.max(0, rowCover) * Math.max(0, colCover));
      }
    }
  }

  /** Disc, or a ring of the given thickness when `thickness` is set. */
  fillDisc(cx: number, cy: number, radius: number, c: Rgba, thickness?: number): void {
    const outer = radius + (thickness ?? 0) / 2 + 1;
    for (let py = Math.floor(cy - outer); py <= Math.ceil(cy + outer); py++) {
      for (let px = Math.floor(cx - outer); px <= Math.ceil(cx + outer); px++) {
        const d = Math.hypot(px + 0.5 - cx, py + 0.5 - cy);
        // One pixel of feathering at the edge is enough at these radii.
        const cover =
          thickness === undefined
            ? clamp01(radius + 0.5 - d)
            : clamp01(thickness / 2 + 0.5 - Math.abs(d - radius));
        this.blend(px, py, c, cover);
      }
    }
  }

  /** Round-capped line segment. */
  strokeSegment(x0: number, y0: number, x1: number, y1: number, width: number, c: Rgba): void {
    const half = width / 2;
    const minX = Math.floor(Math.min(x0, x1) - half - 1);
    const maxX = Math.ceil(Math.max(x0, x1) + half + 1);
    const minY = Math.floor(Math.min(y0, y1) - half - 1);
    const maxY = Math.ceil(Math.max(y0, y1) + half + 1);
    const dx = x1 - x0;
    const dy = y1 - y0;
    const lengthSq = dx * dx + dy * dy;

    for (let py = minY; py <= maxY; py++) {
      for (let px = minX; px <= maxX; px++) {
        const vx = px + 0.5 - x0;
        const vy = py + 0.5 - y0;
        const t = lengthSq > 0 ? clamp01((vx * dx + vy * dy) / lengthSq) : 0;
        const d = Math.hypot(vx - dx * t, vy - dy * t);
        this.blend(px, py, c, clamp01(half + 0.5 - d));
      }
    }
  }

  /** Composites another raster of the same size on top of this one. */
  overlay(top: Uint8ClampedArray): void {
    for (let i = 0; i < this.data.length; i += 4) {
      const a = top[i + 3] / 255;
      if (a <= 0) continue;
      const inv = 1 - a;
      this.data[i] = top[i] * a + this.data[i] * inv;
      this.data[i + 1] = top[i + 1] * a + this.data[i + 1] * inv;
      this.data[i + 2] = top[i + 2] * a + this.data[i + 2] * inv;
      this.data[i + 3] = 255 * a + this.data[i + 3] * inv;
    }
  }

  /** Copies this raster into `target` at an offset. */
  blitInto(target: Raster, atX: number, atY: number): void {
    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        const i = (y * this.width + x) * 4;
        target.blend(atX + x, atY + y, {
          r: this.data[i],
          g: this.data[i + 1],
          b: this.data[i + 2],
          a: this.data[i + 3] / 255,
        });
      }
    }
  }
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/* ── Colour ──────────────────────────────────────────────────────────────── */

/** Parses the CSS colour forms the renderers actually emit. */
function parseColour(css: string): Rgba {
  const value = css.trim();

  const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(value);
  if (hex) {
    const h = hex[1].length === 3 ? hex[1].replace(/./g, (d) => d + d) : hex[1];
    return {
      r: parseInt(h.slice(0, 2), 16),
      g: parseInt(h.slice(2, 4), 16),
      b: parseInt(h.slice(4, 6), 16),
      a: 1,
    };
  }

  const rgb = /^rgba?\(([^)]+)\)$/i.exec(value);
  if (rgb) {
    const parts = rgb[1].split(",").map((p) => Number(p));
    return { r: parts[0], g: parts[1], b: parts[2], a: parts.length > 3 ? parts[3] : 1 };
  }

  const hsl = /^hsla?\(([^)]+)\)$/i.exec(value);
  if (hsl) {
    const parts = hsl[1].split(",").map((p) => Number(p.replace("%", "")));
    return hslToRgb(parts[0], parts[1] / 100, parts[2] / 100, parts.length > 3 ? parts[3] : 1);
  }

  throw new Error(`Unsupported colour in figure shim: ${css}`);
}

function hslToRgb(hue: number, s: number, l: number, a: number): Rgba {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const h = ((hue % 360) + 360) % 360;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  const [r, g, b] =
    h < 60
      ? [c, x, 0]
      : h < 120
        ? [x, c, 0]
        : h < 180
          ? [0, c, x]
          : h < 240
            ? [0, x, c]
            : h < 300
              ? [x, 0, c]
              : [c, 0, x];
  return { r: (r + m) * 255, g: (g + m) * 255, b: (b + m) * 255, a };
}

/* ── The canvas surface the renderers draw onto ──────────────────────────── */

type PathItem =
  | { kind: "poly"; points: Array<{ x: number; y: number }> }
  | { kind: "disc"; x: number; y: number; radius: number };

/** Stands in for the DOM `Path2D` the scanpath renderer builds its saccades in. */
class ShimPath {
  readonly items: PathItem[] = [];

  moveTo(x: number, y: number): void {
    this.items.push({ kind: "poly", points: [{ x, y }] });
  }

  lineTo(x: number, y: number): void {
    const last = this.items[this.items.length - 1];
    if (last?.kind === "poly") last.points.push({ x, y });
    else this.moveTo(x, y);
  }

  arc(x: number, y: number, radius: number): void {
    this.items.push({ kind: "disc", x, y, radius });
  }
}

/**
 * The subset of `CanvasRenderingContext2D` used by renderHeatmap and
 * renderScanpath. Everything else is deliberately absent: an unshimmed call
 * throws, rather than silently producing a figure that flatters the code.
 */
class ShimContext {
  fillStyle = "#000";
  strokeStyle = "#000";
  lineWidth = 1;
  lineJoin = "round";
  lineCap = "round";
  font = "10px sans-serif";
  textAlign = "start";
  textBaseline = "alphabetic";

  private current = new ShimPath();

  constructor(private readonly raster: Raster) {}

  clearRect(): void {
    this.raster.data.fill(0);
  }

  createImageData(
    width: number,
    height: number
  ): { width: number; height: number; data: Uint8ClampedArray } {
    return { width, height, data: new Uint8ClampedArray(width * height * 4) };
  }

  putImageData(image: { data: Uint8ClampedArray }): void {
    this.raster.data.set(image.data);
  }

  beginPath(): void {
    this.current = new ShimPath();
  }

  moveTo(x: number, y: number): void {
    this.current.moveTo(x, y);
  }

  lineTo(x: number, y: number): void {
    this.current.lineTo(x, y);
  }

  arc(x: number, y: number, radius: number): void {
    this.current.arc(x, y, radius);
  }

  fill(path?: ShimPath): void {
    const colour = parseColour(this.fillStyle);
    for (const item of (path ?? this.current).items) {
      if (item.kind === "disc") this.raster.fillDisc(item.x, item.y, item.radius, colour);
    }
  }

  stroke(path?: ShimPath): void {
    const colour = parseColour(this.strokeStyle);
    for (const item of (path ?? this.current).items) {
      if (item.kind === "disc") {
        this.raster.fillDisc(item.x, item.y, item.radius, colour, this.lineWidth);
      } else {
        for (let i = 1; i < item.points.length; i++) {
          const a = item.points[i - 1];
          const b = item.points[i];
          this.raster.strokeSegment(a.x, a.y, b.x, b.y, this.lineWidth, colour);
        }
      }
    }
  }

  fillText(text: string, x: number, y: number): void {
    this.drawDigits(text, x, y, parseColour(this.fillStyle), 0);
  }

  /** Approximates a text outline by dilating the glyph, which is all the
   * scanpath renderer wants it for: a dark halo under white numerals. */
  strokeText(text: string, x: number, y: number): void {
    this.drawDigits(text, x, y, parseColour(this.strokeStyle), this.lineWidth / 2);
  }

  private drawDigits(text: string, x: number, y: number, colour: Rgba, dilate: number): void {
    const size = Number(/(\d+(?:\.\d+)?)px/.exec(this.font)?.[1] ?? 10);
    const cell = size / 7;
    const glyphWidth = 5 * cell;
    const advance = glyphWidth + cell;
    const totalWidth = text.length * advance - cell;
    // The renderer centres its numerals; nothing else is needed.
    const left = this.textAlign === "center" ? x - totalWidth / 2 : x;
    const top = this.textBaseline === "middle" ? y - (7 * cell) / 2 : y - 7 * cell;

    for (let c = 0; c < text.length; c++) {
      const glyph = DIGITS[text[c]];
      if (!glyph) continue;
      for (let row = 0; row < 7; row++) {
        for (let col = 0; col < 5; col++) {
          if (!(glyph[row] & (1 << (4 - col)))) continue;
          this.raster.fillRect(
            left + c * advance + col * cell - dilate,
            top + row * cell - dilate,
            cell + dilate * 2,
            cell + dilate * 2,
            colour
          );
        }
      }
    }
  }
}

/** 5×7 numerals, one bit per cell — enough for fixation indices. */
const DIGITS: Record<string, number[]> = {
  "0": [0x0e, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0e],
  "1": [0x04, 0x0c, 0x04, 0x04, 0x04, 0x04, 0x0e],
  "2": [0x0e, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1f],
  "3": [0x1f, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0e],
  "4": [0x02, 0x06, 0x0a, 0x12, 0x1f, 0x02, 0x02],
  "5": [0x1f, 0x10, 0x1e, 0x01, 0x01, 0x11, 0x0e],
  "6": [0x06, 0x08, 0x10, 0x1e, 0x11, 0x11, 0x0e],
  "7": [0x1f, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
  "8": [0x0e, 0x11, 0x11, 0x0e, 0x11, 0x11, 0x0e],
  "9": [0x0e, 0x11, 0x11, 0x0f, 0x01, 0x02, 0x0c],
};

// The scanpath renderer builds its saccade line as `new Path2D()`, which only
// exists in a document. Node needs it put on the global before that call.
(globalThis as unknown as { Path2D: unknown }).Path2D = ShimPath;

/** A canvas the analysis renderers accept, backed by a raster we can encode. */
function makeCanvas(width: number, height: number): { canvas: HTMLCanvasElement; raster: Raster } {
  const raster = new Raster(width, height);
  const context = new ShimContext(raster);
  const canvas = { width, height, getContext: () => context };
  return { canvas: canvas as unknown as HTMLCanvasElement, raster };
}

/* ── The stimulus ────────────────────────────────────────────────────────── */

const PAPER: Rgba = { r: 255, g: 255, b: 255, a: 1 };
const RULE: Rgba = { r: 226, g: 219, b: 205, a: 1 };
const BLOCK: Rgba = { r: 236, g: 231, b: 222, a: 1 };
const TEXT: Rgba = { r: 205, g: 198, b: 186, a: 1 };
const INK: Rgba = { r: 24, g: 37, b: 40, a: 1 };
const TEAL: Rgba = { r: 0, g: 63, b: 72, a: 1 };

/**
 * A pricing page as a wireframe — the kind of stimulus this tool is pointed at.
 * Laid out in normalised coordinates so the same drawing works at any size.
 */
function paintStimulus(width: number, height: number): Raster {
  const page = new Raster(width, height, PAPER);
  const box = (x: number, y: number, w: number, h: number, c: Rgba): void =>
    page.fillRect(x * width, y * height, w * width, h * height, c);
  const line = (x: number, y: number, w: number): void => box(x, y, w, 0.014, TEXT);

  // Header: wordmark, four nav links, a rule under the lot.
  box(0.06, 0.045, 0.09, 0.026, INK);
  for (let i = 0; i < 4; i++) box(0.62 + i * 0.085, 0.05, 0.06, 0.016, TEXT);
  box(0, 0.105, 1, 0.002, RULE);

  // Hero: headline over two body lines, a primary and a secondary action.
  box(0.06, 0.17, 0.36, 0.05, INK);
  box(0.06, 0.245, 0.24, 0.05, INK);
  line(0.06, 0.35, 0.33);
  line(0.06, 0.39, 0.29);
  box(0.06, 0.45, 0.14, 0.055, TEAL);
  box(0.22, 0.45, 0.12, 0.055, BLOCK);

  // The image everyone looks at whether or not it says anything.
  box(0.54, 0.17, 0.4, 0.34, BLOCK);

  // Three plan cards; the middle one carries the answer to the task.
  for (let i = 0; i < 3; i++) {
    const x = 0.06 + i * 0.31;
    box(x, 0.6, 0.28, 0.32, BLOCK);
    box(x + 0.03, 0.64, 0.1, 0.022, INK);
    box(x + 0.03, 0.69, 0.14, 0.045, INK);
    line(x + 0.03, 0.765, 0.2);
    line(x + 0.03, 0.8, 0.17);
    box(x + 0.03, 0.845, 0.1, 0.04, i === 1 ? TEAL : RULE);
  }

  box(0, 0.96, 1, 0.002, RULE);
  return page;
}

/* ── A synthetic session over that stimulus ──────────────────────────────── */

/** Where the eye rested, and for how long: a participant hunting for a price. */
const SCANPATH: Array<{ x: number; y: number; ms: number }> = [
  { x: 0.19, y: 0.19, ms: 320 }, // headline
  { x: 0.73, y: 0.3, ms: 520 }, // the picture
  { x: 0.2, y: 0.26, ms: 240 }, // back to the headline
  { x: 0.17, y: 0.355, ms: 300 }, // body copy
  { x: 0.15, y: 0.395, ms: 220 },
  { x: 0.12, y: 0.475, ms: 460 }, // the primary CTA
  { x: 0.14, y: 0.66, ms: 280 }, // first plan
  { x: 0.44, y: 0.66, ms: 340 }, // second plan
  { x: 0.45, y: 0.71, ms: 700 }, // its price — the answer
  { x: 0.76, y: 0.67, ms: 260 }, // third plan, a glance
];

/** Deterministic PRNG, so a regenerated figure is byte-identical. */
function makeRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

/**
 * One participant's recording: the shared scanpath, displaced and re-timed for
 * this person, expanded into a 30fps gaze stream — jitter around each target, a
 * few in-flight samples across each saccade. The figures are then built from
 * something `detectFixations` has to actually work on.
 */
function simulateGaze(width: number, height: number, participant: number): RawPoint[] {
  const rand = makeRandom(20260806 + participant * 7919);
  const spread = participant === 0 ? 0 : 0.085;
  const points: RawPoint[] = [];
  const frame = 1000 / 30;
  let t = 0;

  const targets = SCANPATH.map((target) => ({
    x: target.x + (rand() - 0.5) * spread,
    y: target.y + (rand() - 0.5) * spread * 0.7,
    // Dwell varies more between people than position does.
    ms: target.ms * (participant === 0 ? 1 : 0.55 + rand() * 0.95),
  }));

  targets.forEach((target, index) => {
    if (index > 0) {
      const from = targets[index - 1];
      for (let s = 1; s <= 3; s++) {
        const f = s / 4;
        points.push({
          x: (from.x + (target.x - from.x) * f) * width,
          y: (from.y + (target.y - from.y) * f) * height,
          t,
        });
        t += frame;
      }
    }
    const jitter = (): number => (rand() - 0.5) * 0.009;
    for (let elapsed = 0; elapsed <= target.ms; elapsed += frame) {
      points.push({ x: (target.x + jitter()) * width, y: (target.y + jitter()) * height, t });
      t += frame;
    }
  });

  return points;
}

/** The results screen's own analysis settings, so the figures match the app. */
function analyse(width: number, height: number, participant: number): Fixation[] {
  return detectFixations(simulateGaze(width, height, participant), {
    dispersion: Math.max(30, Math.min(width, height) * 0.045),
    minDuration: 100,
  }).map((f) => ({ ...f, x: f.x / width, y: f.y / height }));
}

/* ── Figures ─────────────────────────────────────────────────────────────── */

/** Heatmaps aggregate; six participants is a realistic study, and the figure is
 * built the way the results screen builds one — every fixation from every
 * recording, weighted by dwell. */
const PARTICIPANTS = 6;

function heatFigure(width: number, height: number, style: HeatmapStyle): Raster {
  const page = paintStimulus(width, height);
  const { canvas, raster } = makeCanvas(width, height);
  const points = [];
  for (let p = 0; p < PARTICIPANTS; p++) {
    for (const f of analyse(width, height, p)) points.push({ x: f.x, y: f.y, weight: f.duration });
  }
  renderHeatmap(canvas, points, { style, radiusRatio: 0.055 });
  page.overlay(raster.data);
  return page;
}

/** Scanpaths do not aggregate — an averaged reading order means nothing — so
 * this one is a single participant, as in the app. */
function scanpathFigure(width: number, height: number): Raster {
  const page = paintStimulus(width, height);
  const { canvas, raster } = makeCanvas(width, height);
  const scale = width / 1024;
  renderScanpath(canvas, analyse(width, height, 0), {
    minRadius: 10 * scale,
    maxRadius: 40 * scale,
    // Same contract as the results screen: the ordinal size bounds are in CSS
    // pixels, so a canvas drawn at another scale has to say so.
    scale,
  });
  page.overlay(raster.data);
  return page;
}

/** Two figures on one row, on the site's cream, for a side-by-side in the README. */
function pair(left: Raster, right: Raster): Raster {
  const gutter = 24;
  const sheet = new Raster(
    left.width + right.width + gutter,
    Math.max(left.height, right.height),
    { r: 254, g: 246, b: 233, a: 1 }
  );
  left.blitInto(sheet, 0, 0);
  right.blitInto(sheet, left.width + gutter, 0);
  return sheet;
}

/* ── PNG encoding ────────────────────────────────────────────────────────── */

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type: string, body: Uint8Array): Buffer {
  const head = Buffer.alloc(8);
  head.writeUInt32BE(body.length, 0);
  head.write(type, 4, "ascii");
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([head.subarray(4), body])), 0);
  return Buffer.concat([head, body, crc]);
}

/** 8-bit RGBA PNG, Paeth-filtered — the filter that pays off on soft gradients. */
function encodePng(raster: Raster): Buffer {
  const stride = raster.width * 4;
  const raw = Buffer.alloc((stride + 1) * raster.height);

  for (let y = 0; y < raster.height; y++) {
    const rowStart = y * stride;
    const out = y * (stride + 1);
    raw[out] = 4;
    for (let i = 0; i < stride; i++) {
      const left = i >= 4 ? raster.data[rowStart + i - 4] : 0;
      const up = y > 0 ? raster.data[rowStart - stride + i] : 0;
      const upLeft = y > 0 && i >= 4 ? raster.data[rowStart - stride + i - 4] : 0;
      raw[out + 1 + i] = (raster.data[rowStart + i] - paeth(left, up, upLeft)) & 0xff;
    }
  }

  const header = Buffer.alloc(13);
  header.writeUInt32BE(raster.width, 0);
  header.writeUInt32BE(raster.height, 4);
  header[8] = 8; // bit depth
  header[9] = 6; // truecolour with alpha

  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", header),
    chunk("IDAT", deflateSync(raw, { level: 9 })),
    chunk("IEND", new Uint8Array()),
  ]);
}

function paeth(a: number, b: number, c: number): number {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

/* ── Output ──────────────────────────────────────────────────────────────── */

// Resolved against the working directory rather than import.meta.url: this file
// is bundled into node_modules/.cache before it runs, so its own path is not the
// repo. npm run puts the cwd at the package root.
const outDir = resolve(process.cwd(), "docs/figures");
mkdirSync(outDir, { recursive: true });

const HERO_W = 1024;
const HERO_H = 640;
const PANEL_W = 500;
const PANEL_H = 313;

const figures: Array<[string, Raster]> = [
  ["heatmap.png", heatFigure(HERO_W, HERO_H, "heat")],
  [
    "spotlight-scanpath.png",
    pair(heatFigure(PANEL_W, PANEL_H, "spotlight"), scanpathFigure(PANEL_W, PANEL_H)),
  ],
];

for (const [name, raster] of figures) {
  const png = encodePng(raster);
  writeFileSync(resolve(outDir, name), png);
  process.stdout.write(`Wrote docs/figures/${name} (${(png.length / 1024).toFixed(0)} KB)\n`);
}
