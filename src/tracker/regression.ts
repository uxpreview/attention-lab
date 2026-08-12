/**
 * Ridge regression with feature standardisation and k-fold selection of the
 * regularisation strength.
 *
 * The gaze model is a linear map from a hand-built feature basis (iris offsets,
 * head pose, and their interactions) to screen coordinates. Ridge rather than
 * plain least squares because calibration gives us only a few hundred samples
 * for ~20 correlated features, so the unregularised solution is unstable —
 * small head movements during calibration blow up the weights.
 */

export interface RidgeModel {
  /** Feature means, used to centre inputs at predict time. */
  mean: Float64Array;
  /** Feature standard deviations, used to scale inputs at predict time. */
  std: Float64Array;
  /** Weights for the x target, including a leading intercept. */
  wx: Float64Array;
  /** Weights for the y target, including a leading intercept. */
  wy: Float64Array;
  /** The lambda chosen by cross-validation. */
  lambda: number;
  /** Mean cross-validated error in target units, or NaN when CV could not run. */
  cvError: number;
  /** Number of samples the model was fit on. */
  sampleCount: number;
  /**
   * A constant offset subtracted from every prediction, in target units.
   *
   * The fit itself cannot carry this: least squares puts the intercept exactly
   * where the calibration data says it belongs, so a freshly fitted model has
   * no constant error by construction. Bias appears *after* fitting — the
   * participant settles into the chair, or a calibration is reused in a later
   * sitting — and it is the one component of gaze error that can be measured
   * and removed rather than merely reported. {@link measureBias} estimates it
   * from the validation dots; zero until something measures it.
   */
  biasX: number;
  biasY: number;
}

const LAMBDA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 5, 20, 100];

/**
 * Solves `A w = B` for multiple right-hand sides via Gauss-Jordan elimination
 * with partial pivoting. `A` is n x n, `B` is n x m. Returns n x m.
 * Both inputs are consumed (modified in place).
 */
function solveInPlace(A: Float64Array, B: Float64Array, n: number, m: number): Float64Array {
  for (let col = 0; col < n; col++) {
    // Partial pivot: find the row with the largest magnitude in this column.
    let pivotRow = col;
    let pivotVal = Math.abs(A[col * n + col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(A[r * n + col]);
      if (v > pivotVal) {
        pivotVal = v;
        pivotRow = r;
      }
    }

    if (pivotVal < 1e-12) {
      // Singular even after ridge. Should not happen with lambda > 0, but bail
      // gracefully rather than emitting NaNs into the gaze stream.
      continue;
    }

    if (pivotRow !== col) {
      for (let c = 0; c < n; c++) {
        const tmp = A[col * n + c];
        A[col * n + c] = A[pivotRow * n + c];
        A[pivotRow * n + c] = tmp;
      }
      for (let c = 0; c < m; c++) {
        const tmp = B[col * m + c];
        B[col * m + c] = B[pivotRow * m + c];
        B[pivotRow * m + c] = tmp;
      }
    }

    const diag = A[col * n + col];
    for (let c = 0; c < n; c++) A[col * n + c] /= diag;
    for (let c = 0; c < m; c++) B[col * m + c] /= diag;

    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = A[r * n + col];
      if (factor === 0) continue;
      for (let c = 0; c < n; c++) A[r * n + c] -= factor * A[col * n + c];
      for (let c = 0; c < m; c++) B[r * m + c] -= factor * B[col * m + c];
    }
  }

  return B;
}

interface Standardisation {
  mean: Float64Array;
  std: Float64Array;
}

function standardise(rows: number[][], dim: number): Standardisation {
  const mean = new Float64Array(dim);
  const std = new Float64Array(dim);

  for (const row of rows) {
    for (let d = 0; d < dim; d++) mean[d] += row[d];
  }
  for (let d = 0; d < dim; d++) mean[d] /= rows.length;

  for (const row of rows) {
    for (let d = 0; d < dim; d++) {
      const diff = row[d] - mean[d];
      std[d] += diff * diff;
    }
  }
  for (let d = 0; d < dim; d++) {
    std[d] = Math.sqrt(std[d] / rows.length);
    // Guard constant features: a zero std would divide to Infinity.
    if (std[d] < 1e-8) std[d] = 1;
  }

  return { mean, std };
}

/**
 * Fits ridge weights on pre-standardised design rows. Returns weights of length
 * dim + 1, with the intercept first. The intercept is not penalised, which is
 * why it is handled as a separate mean-offset rather than an extra column.
 */
function fitWeights(
  rows: number[][],
  targetX: number[],
  targetY: number[],
  st: Standardisation,
  dim: number,
  lambda: number
): { wx: Float64Array; wy: Float64Array } {
  const n = rows.length;

  // Targets are centred so the intercept is simply the target mean.
  let meanX = 0;
  let meanY = 0;
  for (let i = 0; i < n; i++) {
    meanX += targetX[i];
    meanY += targetY[i];
  }
  meanX /= n;
  meanY /= n;

  const XtX = new Float64Array(dim * dim);
  const XtY = new Float64Array(dim * 2);
  const z = new Float64Array(dim);

  for (let i = 0; i < n; i++) {
    const row = rows[i];
    for (let d = 0; d < dim; d++) z[d] = (row[d] - st.mean[d]) / st.std[d];

    const dx = targetX[i] - meanX;
    const dy = targetY[i] - meanY;

    for (let a = 0; a < dim; a++) {
      const za = z[a];
      if (za === 0) continue;
      for (let b = a; b < dim; b++) XtX[a * dim + b] += za * z[b];
      XtY[a * 2] += za * dx;
      XtY[a * 2 + 1] += za * dy;
    }
  }

  // Mirror the upper triangle we filled into the lower triangle.
  for (let a = 0; a < dim; a++) {
    for (let b = a + 1; b < dim; b++) XtX[b * dim + a] = XtX[a * dim + b];
    XtX[a * dim + a] += lambda * n;
  }

  const solved = solveInPlace(XtX, XtY, dim, 2);

  const wx = new Float64Array(dim + 1);
  const wy = new Float64Array(dim + 1);
  wx[0] = meanX;
  wy[0] = meanY;
  for (let d = 0; d < dim; d++) {
    wx[d + 1] = solved[d * 2];
    wy[d + 1] = solved[d * 2 + 1];
  }

  return { wx, wy };
}

function predictStandardised(
  features: ArrayLike<number>,
  st: Standardisation,
  wx: Float64Array,
  wy: Float64Array,
  dim: number
): [number, number] {
  let x = wx[0];
  let y = wy[0];
  for (let d = 0; d < dim; d++) {
    const z = (features[d] - st.mean[d]) / st.std[d];
    x += z * wx[d + 1];
    y += z * wy[d + 1];
  }
  return [x, y];
}

/** Mid-grid fallback when no CV fold could run: the data-starved case is
 * exactly where weak regularisation does the most damage, so err strong. */
const FALLBACK_LAMBDA = 1;

/**
 * Fits a gaze model, choosing lambda by grouped k-fold cross-validation on
 * mean Euclidean error. Samples arrive as bursts of consecutive video frames
 * per calibration target, so adjacent rows are near-duplicates of each other.
 * Folds therefore hold out whole targets: interleaving samples would put a
 * near-copy of every test row into the training set, and that leaked CV error
 * both flatters the reported accuracy and picks a lambda too weak to survive
 * real head movement. Holding out whole dots instead measures the spatial
 * interpolation error that actually matters at prediction time. Interleaved
 * assignment survives only as a fallback for data with too few distinct
 * targets to group.
 */
export function fitRidge(
  rows: number[][],
  targetX: number[],
  targetY: number[],
  folds = 5
): RidgeModel {
  if (rows.length === 0) throw new Error("Cannot fit a gaze model with no samples");
  const dim = rows[0].length;
  if (rows.length <= dim) {
    throw new Error(
      `Cannot fit a gaze model on ${rows.length} samples of ${dim} features: the system is underdetermined`
    );
  }

  // Group rows by calibration target, numbering groups in order of appearance.
  const groupIds = new Map<string, number>();
  const groupOf = new Array<number>(rows.length);
  for (let i = 0; i < rows.length; i++) {
    const key = `${targetX[i]},${targetY[i]}`;
    let id = groupIds.get(key);
    if (id === undefined) {
      id = groupIds.size;
      groupIds.set(key, id);
    }
    groupOf[i] = id;
  }
  const grouped = groupIds.size >= folds;
  const effectiveFolds = Math.min(folds, grouped ? groupIds.size : rows.length);

  let bestLambda = NaN;
  let bestError = Infinity;

  if (effectiveFolds >= 2) {
    for (const lambda of LAMBDA_GRID) {
      let totalError = 0;
      let counted = 0;

      for (let fold = 0; fold < effectiveFolds; fold++) {
        const trainRows: number[][] = [];
        const trainX: number[] = [];
        const trainY: number[] = [];
        const testIdx: number[] = [];

        for (let i = 0; i < rows.length; i++) {
          const bucket = grouped ? groupOf[i] : i;
          if (bucket % effectiveFolds === fold) {
            testIdx.push(i);
          } else {
            trainRows.push(rows[i]);
            trainX.push(targetX[i]);
            trainY.push(targetY[i]);
          }
        }
        if (trainRows.length <= dim || testIdx.length === 0) continue;

        const st = standardise(trainRows, dim);
        const { wx, wy } = fitWeights(trainRows, trainX, trainY, st, dim, lambda);

        for (const i of testIdx) {
          const [px, py] = predictStandardised(rows[i], st, wx, wy, dim);
          totalError += Math.hypot(px - targetX[i], py - targetY[i]);
          counted++;
        }
      }

      if (counted > 0) {
        const meanError = totalError / counted;
        if (meanError < bestError) {
          bestError = meanError;
          bestLambda = lambda;
        }
      }
    }
  }

  if (!Number.isFinite(bestLambda)) {
    // Every fold was skipped (training splits no larger than the feature
    // dimension), so nothing was learned about lambda. Falling back to the
    // weakest grid entry here would hand the least-regularised fit to the
    // least-determined data; fall back to the middle of the grid instead.
    bestLambda = FALLBACK_LAMBDA;
  }

  const st = standardise(rows, dim);
  const { wx, wy } = fitWeights(rows, targetX, targetY, st, dim, bestLambda);

  return {
    mean: st.mean,
    std: st.std,
    wx,
    wy,
    lambda: bestLambda,
    cvError: Number.isFinite(bestError) ? bestError : NaN,
    sampleCount: rows.length,
    // A fresh fit has no constant error: the intercept absorbed it.
    biasX: 0,
    biasY: 0,
  };
}

export function predict(model: RidgeModel, features: ArrayLike<number>): [number, number] {
  const [x, y] = predictStandardised(
    features,
    { mean: model.mean, std: model.std },
    model.wx,
    model.wy,
    model.mean.length
  );
  return [x - model.biasX, y - model.biasY];
}

/**
 * The largest constant offset worth correcting, in target units (CSS pixels).
 *
 * Past this the offset is not a settled posture, it is a calibration that has
 * stopped describing the participant — a different person at the keyboard, a
 * moved laptop, a model fitted on someone else's face. Subtracting a shift that
 * large would paper over a broken calibration and hand back gaze that looks
 * plausible and is not, so the caller is told to recalibrate instead.
 */
export const MAX_CORRECTABLE_BIAS_PX = 320;

export interface ResidualSample {
  /** Where the participant was known to be looking. */
  targetX: number;
  targetY: number;
  /** Where the model said they were looking, minus where they were. */
  dx: number;
  dy: number;
}

export interface BiasEstimate {
  /** The constant offset to subtract, in target units. */
  x: number;
  y: number;
  /** Typical distance of a residual from the offset: the part that cannot be
   * corrected, only smoothed or re-measured. */
  scatter: number;
  /** False when the offset exceeds {@link MAX_CORRECTABLE_BIAS_PX}. */
  correctable: boolean;
}

/**
 * Splits validation residuals into the part that is a constant offset and the
 * part that is scatter.
 *
 * These are different faults with different remedies and the same mean error,
 * which is why a single averaged number cannot tell a researcher which one they
 * have. A constant offset moves every gaze point the same way — the heatmap is
 * the right shape in the wrong place, and subtracting the offset fixes it.
 * Scatter is per-sample noise; no offset removes it, and the honest responses
 * are a wider kernel or a recalibration.
 *
 * The offset is the component-wise median rather than the mean: five dots is
 * few enough that one bad dot — a blink, a glance away — would otherwise drag
 * the correction along with it.
 */
export function measureBias(residuals: ResidualSample[]): BiasEstimate {
  if (residuals.length === 0) {
    return { x: 0, y: 0, scatter: NaN, correctable: false };
  }

  const median = (values: number[]): number => {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
  };

  const x = median(residuals.map((r) => r.dx));
  const y = median(residuals.map((r) => r.dy));
  const scatter = median(residuals.map((r) => Math.hypot(r.dx - x, r.dy - y)));

  return {
    x,
    y,
    scatter,
    correctable: Math.hypot(x, y) <= MAX_CORRECTABLE_BIAS_PX,
  };
}

/** A copy of `model` that subtracts `bias` from every prediction. */
export function withBias(model: RidgeModel, bias: { x: number; y: number }): RidgeModel {
  return { ...model, biasX: bias.x, biasY: bias.y };
}

/** Serialisable form, for persisting a calibration between sessions. */
export interface SerialisedModel {
  mean: number[];
  std: number[];
  wx: number[];
  wy: number[];
  lambda: number;
  cvError: number;
  sampleCount: number;
  /** Optional: calibrations stored before bias correction existed omit these. */
  biasX?: number;
  biasY?: number;
}

export function serialiseModel(model: RidgeModel): SerialisedModel {
  return {
    mean: Array.from(model.mean),
    std: Array.from(model.std),
    wx: Array.from(model.wx),
    wy: Array.from(model.wy),
    lambda: model.lambda,
    cvError: model.cvError,
    sampleCount: model.sampleCount,
    biasX: model.biasX,
    biasY: model.biasY,
  };
}

/**
 * True when `data` has the shape and dimensions of a model serialised from a
 * `dim`-feature basis. Persisted models cross a storage boundary the type
 * system cannot see across, so the shape is checked rather than trusted: a
 * dimension mismatch at predict time either reads past the feature array
 * (NaN, and gaze silently never emits) or misaligns every column (confidently
 * wrong gaze), and neither failure announces itself.
 */
export function isSerialisedModel(data: unknown, dim: number): data is SerialisedModel {
  if (typeof data !== "object" || data === null) return false;
  const record = data as Record<string, unknown>;
  const finiteVector = (value: unknown, length: number): boolean =>
    Array.isArray(value) &&
    value.length === length &&
    value.every((v) => typeof v === "number" && Number.isFinite(v));

  // Bias is optional so calibrations stored before it existed still load, but
  // it is checked when present: a NaN offset would silently move every gaze
  // point, which is the same confidently-wrong failure the rest of this guard
  // exists to catch.
  const optionalOffset = (value: unknown): boolean =>
    value === undefined || (typeof value === "number" && Number.isFinite(value));

  if (!optionalOffset(record.biasX) || !optionalOffset(record.biasY)) return false;

  return (
    finiteVector(record.mean, dim) &&
    finiteVector(record.std, dim) &&
    // Weights carry a leading intercept on top of the feature columns.
    finiteVector(record.wx, dim + 1) &&
    finiteVector(record.wy, dim + 1) &&
    typeof record.lambda === "number" &&
    Number.isFinite(record.lambda) &&
    typeof record.sampleCount === "number"
    // cvError is deliberately unchecked: it is allowed to be NaN, which JSON
    // round-trips as null.
  );
}

export function deserialiseModel(data: SerialisedModel): RidgeModel {
  return {
    mean: Float64Array.from(data.mean),
    std: Float64Array.from(data.std),
    wx: Float64Array.from(data.wx),
    wy: Float64Array.from(data.wy),
    lambda: data.lambda,
    cvError: data.cvError,
    sampleCount: data.sampleCount,
    biasX: data.biasX ?? 0,
    biasY: data.biasY ?? 0,
  };
}
