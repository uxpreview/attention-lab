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
   * Where the top of the colour ramp sits, as a percentile of the *per-blob
   * peak* intensities — not of the pixel distribution. See {@link fieldCeiling}
   * for why that distinction is the difference between a working clamp and a
   * no-op.
   */
  percentile?: number;
}

/**
 * The splat radius to fall back to when there is no stimulus rect to measure
 * against. It is what every heatmap on this screen used to be drawn at,
 * whatever the recordings said.
 */
export const DEFAULT_KERNEL_RATIO = 0.055;

/**
 * The band the app states about its own instrument, in the participant's screen
 * pixels: the setup panel and the bench note both say webcam gaze lands within
 * "2 to 4 degrees of visual angle, which is 50 to 120 pixels".
 *
 * The kernel is held inside it. Below the floor the picture claims a precision
 * the technique does not have — which is exactly what the old constant did, at
 * σ = 27px on a 1000px stimulus, less than half the floor of the app's own
 * sentence. Above the ceiling the recording is not localising at all, and the
 * honest response to that is the low-signal exclusion rather than a blob the
 * size of the page.
 */
export const MIN_KERNEL_SIGMA_PX = 50;
export const MAX_KERNEL_SIGMA_PX = 120;

/** A backstop in the stimulus's own terms: on a small stimulus even a floor-of-
 * the-band kernel can be most of the picture, and a single blob covering a third
 * of the screen has stopped being a measurement of anything. */
export const MAX_KERNEL_RATIO = 0.3;

/**
 * The blur to draw attention at, derived from how far off the gaze actually was.
 *
 * The ratio was a hard-coded 0.055 — 55 stimulus px on a 1280×1000 wireframe,
 * σ = 27px — applied identically to a recording that validated at 48px of error
 * and one that validated at 184px, while the rail three inches from the picture
 * printed that very number per selection. The app's own setup panel says gaze
 * lands within "2 to 4 degrees of visual angle, which is 50 to 120 pixels", so
 * the fixed kernel was drawing at less than half the floor of the uncertainty
 * the tool itself states. The measured effect on a 27-fixation study: 5.5% of
 * the overlay carried any paint at all and 0.44% passed half alpha — pinpricks
 * where the evidence is component-scale.
 *
 * σ is set to the measured error, so the blob covers the region the gaze could
 * plausibly have been in, one standard deviation out. `renderHeatmap` takes σ as
 * half the splat radius, hence the factor of two. Tobii Pro Lab expresses the
 * same idea in degrees of visual angle and lets you set it; this app knows the
 * degrees and, until now, ignored them.
 *
 * `errorPx` and `stimulusMinDim` are both in the participant's own CSS pixels —
 * the error as validation measured it, the dimension as the stimulus was
 * displayed — so their ratio is a property of the picture and not of the screen
 * it is being reviewed on. An unmeasured calibration takes the floor of the
 * stated band rather than the old constant: not knowing how far off a recording
 * was is not evidence that it was far off, but it is no reason to draw it finer
 * than the technique resolves either.
 */
export function kernelRatio(errorPx: number | null, stimulusMinDim: number): number {
  if (!Number.isFinite(stimulusMinDim) || stimulusMinDim <= 0) return DEFAULT_KERNEL_RATIO;
  const measured =
    errorPx !== null && Number.isFinite(errorPx) && errorPx > 0 ? errorPx : MIN_KERNEL_SIGMA_PX;
  const sigma = Math.min(MAX_KERNEL_SIGMA_PX, Math.max(MIN_KERNEL_SIGMA_PX, measured));
  return Math.min(MAX_KERNEL_RATIO, (2 * sigma) / stimulusMinDim);
}

/**
 * The attention ramp: [position, r, g, b, alpha].
 *
 * This used to run blue → cyan → green → yellow → red and call itself "the
 * convention for attention maps". It is not; it is jet, the GIS/matplotlib
 * default, and on an attention map it does two harmful things. Visually it
 * fights a warm cream page — the contour view rendered as a blue-green weather
 * map. More seriously it is dishonest: jet's floor is a *saturated* colour, so
 * a region that got the faintest brush of gaze is painted a solid blue that
 * carries as much visual weight as the red core, and a reader's eye reports
 * attention where there was effectively none.
 *
 * So the ramp carries its own alpha and starts at zero. Nothing looked at is
 * nothing drawn — the transition from "no data" to "a little data" is a fade
 * rather than a step onto a coloured plateau — and the hues stay inside the
 * product's palette: pale amber through --accent-warm vermillion (#e73d00)
 * into a deep oxblood at the ceiling. Intensity therefore reads three ways at
 * once (more opaque, more saturated, darker), which also survives greyscale
 * printing and the common colour vision deficiencies, none of which jet does.
 */
const RAMP: Array<[number, number, number, number, number]> = [
  [0.0, 250, 214, 137, 0.0],
  [0.2, 247, 190, 88, 0.34],
  [0.45, 240, 139, 30, 0.62],
  [0.72, 231, 61, 0, 0.85],
  [1.0, 125, 20, 8, 1.0],
];

/** RGBA, 256 entries. Alpha is stored 0-255 like the rest. */
function buildRamp(): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256 * 4);
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
    lut[i * 4] = lo[1] + (hi[1] - lo[1]) * f;
    lut[i * 4 + 1] = lo[2] + (hi[2] - lo[2]) * f;
    lut[i * 4 + 2] = lo[3] + (hi[3] - lo[3]) * f;
    lut[i * 4 + 3] = (lo[4] + (hi[4] - lo[4]) * f) * 255;
  }
  return lut;
}

const COLOUR_LUT = buildRamp();

/** Width of one contour band in intensity units, and how many that makes. */
const BAND_STEP = 32;
export const CONTOUR_BANDS = 256 / BAND_STEP;

/**
 * Floor on a contour band's opacity, as a fraction of the overlay's.
 *
 * The heat view wants the ramp's own alpha curve unmodified. The contour view
 * cannot: its payload is the *band edges*, and the outermost drawn band sits at
 * around 0.2 of the ramp's alpha, which is a band you cannot see and therefore
 * cannot cite. Band 0 — the genuinely cold floor — is still drawn at zero, so
 * the honesty property survives; this only lifts bands that are already above
 * the noise line into being visible as bands.
 */
const CONTOUR_MIN_ALPHA = 0.55;

function contourAlpha(alpha: number): number {
  return Math.max(CONTOUR_MIN_ALPHA * 255, alpha);
}

/**
 * The ramp as a CSS colour, for legends.
 *
 * A legend that hard-codes its own gradient is a legend that drifts: change
 * RAMP above and the strip under the stage keeps promising the old colours. So
 * the only way to get a swatch is to ask the same lookup table the pixels came
 * from. `t` is 0 (coldest) to 1 (the display ceiling).
 *
 * Alpha comes along, because with this ramp the transparency *is* half the
 * message — a legend drawn as opaque swatches would promise a solid amber floor
 * the renderer never paints. The overlay's own `opacity` is not folded in: the
 * legend shows the shape of the scale, not the strength of one overlay, and the
 * strip is given a ground to sit on in CSS so the fade reads correctly.
 */
export function rampColour(t: number): string {
  const i = Math.max(0, Math.min(255, Math.round(t * 255)));
  return rgba(i * 4, COLOUR_LUT[i * 4 + 3]);
}

function rgba(offset: number, alpha: number): string {
  return `rgba(${COLOUR_LUT[offset]}, ${COLOUR_LUT[offset + 1]}, ${COLOUR_LUT[offset + 2]}, ${(alpha / 255).toFixed(3)})`;
}

/** The visible contour bands, coldest first. Band 0 is drawn transparent. */
export function contourBandColours(): string[] {
  const out: string[] = [];
  for (let band = 1; band < CONTOUR_BANDS; band++) {
    const i = band * BAND_STEP;
    out.push(rgba(i * 4, contourAlpha(COLOUR_LUT[i * 4 + 3])));
  }
  return out;
}

/**
 * How aggressively the spotlight view reveals moderately-attended regions.
 *
 * 1.8 cleared a region fully at 56% of the ramp's ceiling, so most of a
 * populated map sat at the flat top of the curve: the reveal stopped encoding
 * how much a region was looked at and became a binary. 1.5 keeps the boost the
 * setting exists for — a linear reveal leaves a genuinely-read region nearly as
 * dark as an ignored one — while leaving the top third of the scale with
 * somewhere to go.
 */
const SPOTLIGHT_BOOST = 1.5;
/**
 * Dimming applied where nobody looked, out of 255.
 *
 * It was 232 — 91% black — which is a room with the lights off rather than
 * down: the wireframe outside the reveals was gone, so the revealed blobs read
 * as white glows floating in a void with no context to place them against, and
 * the thumbnails and text bars inside a partial reveal washed out with it. An
 * opacity map is supposed to say "this is what they saw *of this screen*",
 * which needs the screen to still be faintly there. 184 is about 72%: unlooked
 * content stays legible as shape and layout, and the contrast between looked-at
 * and not is still unmistakable — which is how Tobii's opacity map behaves.
 *
 * Exported so the spotlight legend and the stage's own dark ground can show the
 * real dim rather than a guess.
 */
export const SPOTLIGHT_MAX_DIM = 184;

/**
 * Renders points onto a canvas, replacing its contents. The canvas is expected
 * to already be sized to the stimulus display area.
 *
 * Returns the value the hot end of the ramp was scaled to, in the same units
 * the weights came in — milliseconds of summed fixation duration, gathered
 * within about a splat radius and Gaussian-weighted by distance. The legend
 * needs it: a colour axis in a tool whose product is numbers should say what
 * the top of it is worth, and only the renderer knows, because the ceiling is a
 * percentile of this particular selection's blob peaks. Zero when there was
 * nothing to draw.
 */
export function renderHeatmap(
  canvas: HTMLCanvasElement,
  points: HeatPoint[],
  options: HeatmapOptions = {}
): number {
  const style = options.style ?? "heat";
  const radiusRatio = options.radiusRatio ?? 0.06;
  const opacity = options.opacity ?? 0.72;
  const percentile = options.percentile ?? 0.9;

  const width = canvas.width;
  const height = canvas.height;
  const ctx = canvas.getContext("2d");
  if (!ctx || width === 0 || height === 0) return 0;

  ctx.clearRect(0, 0, width, height);
  if (points.length === 0) return 0;

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

  // Scale to a high percentile of the *blob peaks* rather than of the pixels,
  // so a single long dwell does not compress the rest of the map into the cold
  // end of the ramp. fieldCeiling explains why the pixel percentile could not.
  const ceiling = fieldCeiling(field, width, height, radius, percentile);
  const scale = ceiling > 0 ? 255 / ceiling : 0;

  // Pass 2: colour the accumulated field.
  const out = ctx.createImageData(width, height);
  paintField(field, out.data, style, scale, opacity);

  // Note: putImageData ignores globalAlpha and composite state, so overlay
  // opacity has to be baked into the alpha channel by paintField rather than
  // set here.
  ctx.putImageData(out, 0, 0);

  // Back into the units the weights arrived in. The kernel is truncated at
  // `radius` and has its rim value subtracted so a splat reaches zero smoothly,
  // which costs every splat a fixed fraction of its weight — at sigma = radius/2
  // that is exp(-2), about 13.5%. Left in, a lone 900ms fixation would put "780ms"
  // on the legend, and the axis would read low by a constant nobody could see. So
  // the reported ceiling is divided back out: one fixation dead centre of the
  // hottest cluster is worth its own duration.
  return ceiling / (1 - rim);
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
      // Clamped at both ends on purpose. A fully-revealed pixel returns the
      // stimulus at exactly 100% and no further — the mask only ever subtracts
      // dimming it added, it never brightens — and the floor stops at
      // SPOTLIGHT_MAX_DIM rather than at black, so the unlooked page survives as
      // context.
      const reveal = Math.max(0, Math.min(1, ((field[j] * scale) / 255) * SPOTLIGHT_BOOST));
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
      const lut = band * 4;
      dst[i] = COLOUR_LUT[lut];
      dst[i + 1] = COLOUR_LUT[lut + 1];
      dst[i + 2] = COLOUR_LUT[lut + 2];
      dst[i + 3] = band === 0 ? 0 : Math.round(opacity * contourAlpha(COLOUR_LUT[lut + 3]));
    } else {
      const lut = intensity * 4;
      dst[i] = COLOUR_LUT[lut];
      dst[i + 1] = COLOUR_LUT[lut + 1];
      dst[i + 2] = COLOUR_LUT[lut + 2];
      // The cold tail fades out because the ramp itself does — see RAMP. This
      // used to be a separate `min(1, intensity / 60)` fudge layered over an
      // opaque ramp, which is the same idea done in a place the legend could
      // not see, so the legend promised a solid blue floor the pixels did not
      // have.
      dst[i + 3] = Math.round(opacity * COLOUR_LUT[lut + 3]);
    }
  }
}

/**
 * The top of the colour ramp, computed from one peak per blob.
 *
 * This used to be `fieldPercentile(field, 0.98)` — the 98th percentile of the
 * non-zero *pixels* — and the comment above it claimed that stopped one long
 * dwell flattening the map. It did the opposite. A dominant blob covers a small
 * fraction of the looked-at area, so its hottest pixels sit comfortably inside
 * the top 2% of the pixel distribution and set the ceiling themselves: the
 * clamp landed within a few percent of the global maximum and was, in the case
 * it was written for, a no-op. Eight clusters with one long dwell among them
 * rendered as one red blob and seven pure-blue ones, under a legend reading
 * "barely looked at" — a false finding, printed confidently.
 *
 * So reduce the field to one value per blob first: take a block maximum over
 * cells the size of the splat radius, keep the cells that beat all eight of
 * their neighbours, and read a high percentile off *those*. A dominant blob is
 * then one sample among many instead of the entire top of the distribution — it
 * saturates, and every other cluster keeps the range it earned. Interpolated
 * rather than nearest-rank, because with eight blobs the nearest rank at p90 is
 * still the maximum.
 *
 * O(n) in the field, and exported for the test suite.
 */
export function fieldCeiling(
  field: Float32Array,
  width: number,
  height: number,
  cell: number,
  percentile = 0.9
): number {
  const size = Math.max(2, Math.round(cell));
  const cols = Math.max(1, Math.ceil(width / size));
  const rows = Math.max(1, Math.ceil(height / size));

  const blockMax = new Float32Array(cols * rows);
  for (let y = 0; y < height; y++) {
    const rowBase = Math.floor(y / size) * cols;
    const fieldBase = y * width;
    for (let x = 0; x < width; x++) {
      const v = field[fieldBase + x];
      const i = rowBase + Math.floor(x / size);
      if (v > blockMax[i]) blockMax[i] = v;
    }
  }

  const peaks: number[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = blockMax[r * cols + c];
      if (v <= 0) continue;
      let isPeak = true;
      for (let dr = -1; dr <= 1 && isPeak; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          if (dr === 0 && dc === 0) continue;
          const rr = r + dr;
          const cc = c + dc;
          if (rr < 0 || cc < 0 || rr >= rows || cc >= cols) continue;
          if (blockMax[rr * cols + cc] > v) {
            isPeak = false;
            break;
          }
        }
      }
      if (isPeak) peaks.push(v);
    }
  }

  // A field too small to have neighbourhoods, or an empty one: fall back to the
  // pixel percentile rather than returning a ceiling of zero, which would paint
  // the whole overlay at full intensity.
  if (peaks.length === 0) return fieldPercentile(field, percentile);

  peaks.sort((a, b) => a - b);
  const rank = Math.max(0, Math.min(peaks.length - 1, percentile * (peaks.length - 1)));
  const lo = Math.floor(rank);
  const hi = Math.min(peaks.length - 1, lo + 1);
  return peaks[lo] + (peaks[hi] - peaks[lo]) * (rank - lo);
}

/**
 * Approximate percentile of the non-zero entries of an intensity field, via a
 * 1024-bin histogram against the maximum — plenty of resolution for a display
 * ceiling, and O(n) where an exact sort of a megapixel field would not be.
 * Kept as the degenerate-case fallback for {@link fieldCeiling}, and exported
 * for the test suite.
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
