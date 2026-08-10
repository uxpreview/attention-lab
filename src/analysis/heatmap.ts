/**
 * Heatmap rendering.
 *
 * Two passes: accumulate weighted Gaussian splats into a float intensity
 * field, then map the summed field through a colour ramp. Doing the colouring
 * as a second pass is what keeps overlapping blobs from banding — colour has
 * to be applied to the *summed* field, not per splat.
 *
 * The accumulation is done in a Float32Array rather than by compositing
 * radial gradients into a canvas: canvas channels are 8-bit, so a couple of
 * dozen overlapping splats clip at 255 and flatten the hot end of a
 * multi-participant map exactly where the density ordering matters most,
 * while faint splats quantise into visible bands. Floats cost a few
 * milliseconds at these sizes and let weights stay in real units (ms of
 * dwell) instead of a canvas-friendly alpha mapping.
 */

export interface HeatPoint {
  /** Normalised stimulus coordinates in [0, 1]. */
  x: number;
  y: number;
  /** Relative weight, typically fixation duration in milliseconds. */
  weight: number;
}

export type HeatmapStyle = "heat" | "spotlight" | "contour";

export interface HeatmapOptions {
  /** Splat radius as a fraction of the smaller output dimension. */
  radiusRatio?: number;
  /** Opacity of the overlay at peak intensity. */
  opacity?: number;
  style?: HeatmapStyle;
  /**
   * Clamp the intensity scale to a percentile of observed values rather than to
   * the maximum, so one long stare does not flatten everything else to blue.
   */
  percentile?: number;
}

/** Blue → cyan → green → yellow → red, the convention for attention maps. */
const RAMP: Array<[number, number, number, number]> = [
  [0.0, 0, 0, 255],
  [0.25, 0, 200, 255],
  [0.45, 0, 220, 120],
  [0.65, 240, 240, 0],
  [0.85, 255, 130, 0],
  [1.0, 255, 0, 0],
];

function buildRamp(): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let lo = RAMP[0];
    let hi = RAMP[RAMP.length - 1];
    for (let s = 0; s < RAMP.length - 1; s++) {
      if (t >= RAMP[s][0] && t <= RAMP[s + 1][0]) {
        lo = RAMP[s];
        hi = RAMP[s + 1];
        break;
      }
    }
    const span = hi[0] - lo[0];
    const f = span > 0 ? (t - lo[0]) / span : 0;
    lut[i * 3] = lo[1] + (hi[1] - lo[1]) * f;
    lut[i * 3 + 1] = lo[2] + (hi[2] - lo[2]) * f;
    lut[i * 3 + 2] = lo[3] + (hi[3] - lo[3]) * f;
  }
  return lut;
}

const COLOUR_LUT = buildRamp();

/** Width of one contour band in intensity units, and how many that makes. */
const BAND_STEP = 32;
export const CONTOUR_BANDS = 256 / BAND_STEP;

/**
 * The ramp as a CSS colour, for legends.
 *
 * A legend that hard-codes its own gradient is a legend that drifts: change
 * RAMP above and the strip under the stage keeps promising the old colours. So
 * the only way to get a swatch is to ask the same lookup table the pixels came
 * from. `t` is 0 (coldest) to 1 (the display ceiling).
 */
export function rampColour(t: number): string {
  const i = Math.max(0, Math.min(255, Math.round(t * 255)));
  return `rgb(${COLOUR_LUT[i * 3]}, ${COLOUR_LUT[i * 3 + 1]}, ${COLOUR_LUT[i * 3 + 2]})`;
}

/** The visible contour bands, coldest first. Band 0 is drawn transparent. */
export function contourBandColours(): string[] {
  const out: string[] = [];
  for (let band = 1; band < CONTOUR_BANDS; band++) {
    out.push(rampColour((band * BAND_STEP) / 255));
  }
  return out;
}

/** How aggressively the spotlight view reveals moderately-attended regions. */
const SPOTLIGHT_BOOST = 1.8;
/** Dimming applied where nobody looked. Short of opaque, so context survives.
 * Exported so the spotlight legend can show the real dim rather than a guess. */
export const SPOTLIGHT_MAX_DIM = 232;

/**
 * Renders points onto a canvas, replacing its contents. The canvas is expected
 * to already be sized to the stimulus display area.
 */
export function renderHeatmap(
  canvas: HTMLCanvasElement,
  points: HeatPoint[],
  options: HeatmapOptions = {}
): void {
  const style = options.style ?? "heat";
  const radiusRatio = options.radiusRatio ?? 0.06;
  const opacity = options.opacity ?? 0.72;
  const percentile = options.percentile ?? 0.98;

  const width = canvas.width;
  const height = canvas.height;
  const ctx = canvas.getContext("2d");
  if (!ctx || width === 0 || height === 0) return;

  ctx.clearRect(0, 0, width, height);
  if (points.length === 0) return;

  const radius = Math.max(12, Math.min(width, height) * radiusRatio);

  // Pass 1: accumulate a Gaussian splat per point into the float field.
  const field = new Float32Array(width * height);
  const sigma = radius / 2;
  const twoSigmaSq = 2 * sigma * sigma;
  // The kernel is truncated at `radius`; subtracting its value at the rim
  // brings the splat smoothly to zero there instead of stepping off a ledge.
  const rim = Math.exp(-(radius * radius) / twoSigmaSq);

  // Weights are usually fixation durations in ms. If a caller passes all-zero
  // weights, fall back to counting points rather than rendering nothing.
  const weighted = points.some((p) => p.weight > 0);

  for (const p of points) {
    const w = weighted ? Math.max(p.weight, 0) : 1;
    if (w === 0) continue;
    const cx = p.x * width;
    const cy = p.y * height;
    const x0 = Math.max(0, Math.floor(cx - radius));
    const x1 = Math.min(width - 1, Math.ceil(cx + radius));
    const y0 = Math.max(0, Math.floor(cy - radius));
    const y1 = Math.min(height - 1, Math.ceil(cy + radius));

    for (let py = y0; py <= y1; py++) {
      const dy = py - cy;
      const rowBase = py * width;
      for (let px = x0; px <= x1; px++) {
        const dx = px - cx;
        const k = Math.exp(-(dx * dx + dy * dy) / twoSigmaSq) - rim;
        if (k > 0) field[rowBase + px] += w * k;
      }
    }
  }

  // Scale to a high percentile instead of the max so a single long dwell does
  // not compress the rest of the map into the cold end of the ramp.
  const ceiling = fieldPercentile(field, percentile);
  const scale = ceiling > 0 ? 255 / ceiling : 0;

  // Pass 2: colour the accumulated field.
  const out = ctx.createImageData(width, height);
  paintField(field, out.data, style, scale, opacity);

  // Note: putImageData ignores globalAlpha and composite state, so overlay
  // opacity has to be baked into the alpha channel by paintField rather than
  // set here.
  ctx.putImageData(out, 0, 0);
}

/**
 * Maps an accumulated intensity field onto RGBA overlay pixels, writing into
 * `dst` (which is assumed to start transparent). `scale` converts field units
 * to the 0-255 intensity the ramp is indexed by.
 *
 * Exported for the test suite. Spotlight in particular is easy to get subtly
 * wrong: it is a mask rather than an overlay, so it has to write *every* pixel.
 */
export function paintField(
  field: Float32Array,
  dst: Uint8ClampedArray,
  style: HeatmapStyle,
  scale: number,
  opacity: number
): void {
  if (style === "spotlight") {
    // Inverse map: dim everything, reveal what was looked at. The reveal is
    // boosted because a linear one leaves moderately-attended regions almost as
    // dark as ignored ones, which reads as "nobody looked at anything".
    //
    // Every pixel is written, including the faint rim of a splat: skipping the
    // ones that round to zero intensity leaves them fully transparent, which
    // punches an undimmed halo through the mask around every hot spot. The
    // colour channels stay at zero — the dimming is pure alpha.
    for (let j = 0; j < field.length; j++) {
      const reveal = Math.min(1, ((field[j] * scale) / 255) * SPOTLIGHT_BOOST);
      dst[j * 4 + 3] = Math.round(SPOTLIGHT_MAX_DIM * (1 - reveal));
    }
    return;
  }

  for (let j = 0; j < field.length; j++) {
    const raw = field[j];
    if (raw === 0) continue;
    const intensity = Math.min(255, Math.round(raw * scale));
    if (intensity === 0) continue;
    const i = j * 4;

    if (style === "contour") {
      // Banded ramp: reads as an isoline map, easier to cite exact regions from.
      const band = Math.floor(intensity / BAND_STEP) * BAND_STEP;
      dst[i] = COLOUR_LUT[band * 3];
      dst[i + 1] = COLOUR_LUT[band * 3 + 1];
      dst[i + 2] = COLOUR_LUT[band * 3 + 2];
      dst[i + 3] = band === 0 ? 0 : Math.round(opacity * 255);
    } else {
      dst[i] = COLOUR_LUT[intensity * 3];
      dst[i + 1] = COLOUR_LUT[intensity * 3 + 1];
      dst[i + 2] = COLOUR_LUT[intensity * 3 + 2];
      // Fade the cold tail out so the stimulus stays readable underneath.
      const fade = Math.min(1, intensity / 60);
      dst[i + 3] = Math.round(opacity * 255 * fade);
    }
  }
}

/**
 * Approximate percentile of the non-zero entries of an intensity field, via a
 * 1024-bin histogram against the maximum — plenty of resolution for a display
 * ceiling, and O(n) where an exact sort of a megapixel field would not be.
 * Exported for the test suite.
 */
export function fieldPercentile(field: Float32Array, percentile: number): number {
  const BINS = 1024;
  let max = 0;
  for (let i = 0; i < field.length; i++) {
    if (field[i] > max) max = field[i];
  }
  if (max <= 0) return 0;

  const histogram = new Uint32Array(BINS);
  let count = 0;
  for (let i = 0; i < field.length; i++) {
    const v = field[i];
    if (v > 0) {
      histogram[Math.min(BINS - 1, Math.floor((v / max) * BINS))]++;
      count++;
    }
  }

  const target = count * percentile;
  let running = 0;
  for (let b = 0; b < BINS; b++) {
    running += histogram[b];
    // The upper edge of the bin that crosses the target rank.
    if (running >= target) return ((b + 1) / BINS) * max;
  }
  return max;
}
