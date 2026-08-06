/**
 * Heatmap rendering.
 *
 * Two passes: accumulate weighted radial gradients into an intensity buffer
 * (using additive compositing so the GPU does the Gaussian summation for us),
 * then map intensity through a colour ramp. Doing the colouring as a second
 * pass is what keeps overlapping blobs from banding — colour has to be applied
 * to the *summed* field, not per splat.
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

/** How aggressively the spotlight view reveals moderately-attended regions. */
const SPOTLIGHT_BOOST = 1.8;
/** Dimming applied where nobody looked. Short of opaque, so context survives. */
const SPOTLIGHT_MAX_DIM = 232;

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
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx || width === 0 || height === 0) return;

  ctx.clearRect(0, 0, width, height);
  if (points.length === 0) return;

  const radius = Math.max(12, Math.min(width, height) * radiusRatio);

  // Pass 1: accumulate intensity in the alpha channel.
  const acc = document.createElement("canvas");
  acc.width = width;
  acc.height = height;
  const accCtx = acc.getContext("2d", { willReadFrequently: true });
  if (!accCtx) return;

  let maxWeight = 0;
  for (const p of points) maxWeight = Math.max(maxWeight, p.weight);
  if (maxWeight <= 0) maxWeight = 1;

  accCtx.globalCompositeOperation = "lighter";
  for (const p of points) {
    const cx = p.x * width;
    const cy = p.y * height;
    // Alpha per splat is deliberately low: intensity comes from accumulation,
    // and letting a single splat saturate would hide density differences.
    const alpha = Math.min(1, (p.weight / maxWeight) * 0.45 + 0.05);
    const gradient = accCtx.createRadialGradient(cx, cy, 0, cx, cy, radius);
    gradient.addColorStop(0, `rgba(0,0,0,${alpha})`);
    gradient.addColorStop(1, "rgba(0,0,0,0)");
    accCtx.fillStyle = gradient;
    accCtx.beginPath();
    accCtx.arc(cx, cy, radius, 0, Math.PI * 2);
    accCtx.fill();
  }

  const accData = accCtx.getImageData(0, 0, width, height);
  const src = accData.data;

  // Scale to a high percentile instead of the max so a single long dwell does
  // not compress the rest of the map into the cold end of the ramp.
  const ceiling = percentileOfAlpha(src, percentile);
  const scale = ceiling > 0 ? 255 / ceiling : 1;

  // Pass 2: colour the accumulated field.
  const out = ctx.createImageData(width, height);
  const dst = out.data;

  for (let i = 0; i < src.length; i += 4) {
    const raw = src[i + 3];
    if (raw === 0) continue;
    const intensity = Math.min(255, Math.round(raw * scale));
    if (intensity === 0) continue;

    if (style === "spotlight") {
      // Inverse map: dim everything, reveal what was looked at. The reveal is
      // boosted because a linear one leaves moderately-attended regions almost
      // as dark as ignored ones, which reads as "nobody looked at anything".
      const reveal = Math.min(1, (intensity / 255) * SPOTLIGHT_BOOST);
      dst[i] = 0;
      dst[i + 1] = 0;
      dst[i + 2] = 0;
      dst[i + 3] = Math.round(SPOTLIGHT_MAX_DIM * (1 - reveal));
    } else if (style === "contour") {
      // Banded ramp: reads as an isoline map, easier to cite exact regions from.
      const band = Math.floor(intensity / 32) * 32;
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

  if (style === "spotlight") {
    // Regions with no gaze at all get the full dim treatment — but not opaque
    // black, so the reader can still see what was ignored.
    for (let i = 0; i < dst.length; i += 4) {
      if (src[i + 3] === 0) {
        dst[i] = 0;
        dst[i + 1] = 0;
        dst[i + 2] = 0;
        dst[i + 3] = SPOTLIGHT_MAX_DIM;
      }
    }
  }

  // Note: putImageData ignores globalAlpha and composite state, so overlay
  // opacity has to be baked into the alpha channel above rather than set here.
  ctx.putImageData(out, 0, 0);
}

function percentileOfAlpha(data: Uint8ClampedArray, percentile: number): number {
  const histogram = new Uint32Array(256);
  let count = 0;
  for (let i = 3; i < data.length; i += 4) {
    const v = data[i];
    if (v > 0) {
      histogram[v]++;
      count++;
    }
  }
  if (count === 0) return 0;

  const target = count * percentile;
  let running = 0;
  for (let v = 1; v < 256; v++) {
    running += histogram[v];
    if (running >= target) return v;
  }
  return 255;
}
