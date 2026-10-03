import {
  DataValidationError,
  DeepboxError,
  DTypeError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../core/errors";
import { type Tensor, Tensor as TensorClass } from "../ndarray";
import { __random } from "../random/random";
import {
  assert2D,
  assertNumericTensor,
  createRandomStream,
  getShape2D,
  getStrides2D,
  shuffleIndicesInPlace,
} from "./_internal";

const EPS = Number.EPSILON;
const SQRT2 = Math.SQRT2;
const SQRT_PI = Math.sqrt(Math.PI);
const TWO_OVER_SQRT_PI = 2 / SQRT_PI;

// ---------------------------------------------------------------------------
// Input handling helpers
// ---------------------------------------------------------------------------

function getNumericData(X: Tensor, name: string): ArrayLike<number | bigint> {
  if (X.dtype === "string") {
    throw new DTypeError(`${name} must be numeric`);
  }
  if (Array.isArray(X.data)) {
    throw new DeepboxError("Internal error: invalid numeric tensor storage");
  }
  return X.data;
}

function parseBooleanOption(value: unknown, name: string, defaultValue: boolean): boolean {
  if (value === undefined) {
    return defaultValue;
  }
  if (typeof value !== "boolean") {
    throw new InvalidParameterError(`${name} must be a boolean`, name, value);
  }
  return value;
}

type DenseMatrix = { data: Float64Array; nSamples: number; nFeatures: number };

/**
 * Read a 2-D numeric tensor into a new dense row-major Float64Array (honouring
 * strides and offset). With `requireFinite` the same pass rejects NaN and
 * Infinity. The returned buffer is private to the caller and may be modified.
 */
function denseRowMajor(X: Tensor, name: string, requireFinite: boolean): DenseMatrix {
  const [nSamples, nFeatures] = getShape2D(X);
  const src = getNumericData(X, name);
  const [stride0, stride1] = getStrides2D(X);
  const out = new Float64Array(nSamples * nFeatures);
  const offset = X.offset;
  let pos = 0;
  for (let i = 0; i < nSamples; i++) {
    let idx = offset + i * stride0;
    for (let j = 0; j < nFeatures; j++) {
      const v = Number(src[idx]);
      idx += stride1;
      if (requireFinite && !Number.isFinite(v)) {
        throw new DataValidationError(`${name} contains NaN or Infinity at index ${pos}`);
      }
      out[pos++] = v;
    }
  }
  return { data: out, nSamples, nFeatures };
}

/** Validate and read the training matrix passed to `fit` / `partialFit`. */
function readFitMatrix(X: Tensor): DenseMatrix {
  if (X.size === 0) {
    if (X.ndim === 2 && (X.shape[0] ?? 0) > 0) {
      throw new InvalidParameterError("X must contain at least one feature", "X");
    }
    throw new InvalidParameterError("X must contain at least one sample", "X");
  }
  assert2D(X, "X");
  assertNumericTensor(X, "X");
  return denseRowMajor(X, "X", true);
}

/** Validate and read a matrix passed to `transform` / `inverseTransform`. */
function readApplyMatrix(
  X: Tensor,
  expectedFeatures: number,
  owner: string,
  requireFinite: boolean
): DenseMatrix {
  assert2D(X, "X");
  assertNumericTensor(X, "X");
  const nFeatures = X.shape[1] ?? 0;
  if (nFeatures !== expectedFeatures) {
    throw new ShapeError(
      `X has ${nFeatures} features, but ${owner} was fitted with ${expectedFeatures} features`,
      { expected: [X.shape[0] ?? 0, expectedFeatures], received: X.shape, context: owner }
    );
  }
  return denseRowMajor(X, "X", requireFinite);
}

function makeMatrix(
  data: Float64Array,
  nSamples: number,
  nFeatures: number,
  device: Tensor["device"]
): Tensor {
  return TensorClass.fromTypedArray({
    data,
    shape: [nSamples, nFeatures],
    dtype: "float64",
    device,
  });
}

/** Copy a fitted per-feature array into a fresh 1-D float64 tensor. */
function makeVector(values: Float64Array): Tensor {
  return TensorClass.fromTypedArray({
    data: values.slice(),
    shape: [values.length],
    dtype: "float64",
    device: "cpu",
  });
}

function assertSameFeatureCount(
  nFeatures: number,
  fittedFeatures: number,
  owner: string,
  method: string
): void {
  if (nFeatures !== fittedFeatures) {
    throw new ShapeError(
      `X has ${nFeatures} features, but ${owner}.${method} was previously called with ${fittedFeatures} features`,
      { expected: [fittedFeatures], received: [nFeatures], context: owner }
    );
  }
}

/** Replace exact zeros by one so that constant features are left unscaled. */
function nonZeroScale(values: Float64Array): Float64Array {
  const out = new Float64Array(values.length);
  for (let j = 0; j < values.length; j++) {
    const v = values[j] as number;
    out[j] = v === 0 ? 1 : v;
  }
  return out;
}

/**
 * Linear interpolation that is exact at both ends and monotone, with the same
 * branch split NumPy uses for percentiles. `lerp(a, a, t)` is exactly `a`.
 */
function lerp(a: number, b: number, t: number): number {
  const d = b - a;
  return t >= 0.5 ? b - d * (1 - t) : a + d * t;
}

/** Linear-interpolated quantile `q` in [0, 1] of an ascending sorted array. */
function quantileSorted(sorted: Float64Array, q: number): number {
  const n = sorted.length;
  if (n === 1) return sorted[0] as number;
  const position = q * (n - 1);
  const lower = Math.floor(position);
  if (lower >= n - 1) return sorted[n - 1] as number;
  return lerp(sorted[lower] as number, sorted[lower + 1] as number, position - lower);
}

// ---------------------------------------------------------------------------
// Normal distribution helpers (double precision)
// ---------------------------------------------------------------------------

/**
 * Complementary error function with relative error around 1e-14 or better.
 *
 * Uses the Maclaurin series of erf for |x| < 1 and a continued fraction for
 * larger |x|, so the tails keep full relative accuracy.
 */
function erfc(x: number): number {
  if (Number.isNaN(x)) return Number.NaN;
  const ax = Math.abs(x);
  let result: number;
  if (ax < 1) {
    const x2 = ax * ax;
    let term = ax;
    let sum = ax;
    for (let n = 1; n < 60; n++) {
      term *= -x2 / n;
      const add = term / (2 * n + 1);
      sum += add;
      if (Math.abs(add) < 1e-17 * Math.abs(sum)) break;
    }
    result = 1 - TWO_OVER_SQRT_PI * sum;
  } else if (ax > 27) {
    result = 0;
  } else {
    let t = ax;
    for (let k = 150; k >= 1; k--) t = ax + k / 2 / t;
    result = Math.exp(-ax * ax) / (SQRT_PI * t);
  }
  return x < 0 ? 2 - result : result;
}

function normalCdf(z: number): number {
  return 0.5 * erfc(-z / SQRT2);
}

/**
 * Inverse of the standard normal CDF. Starts from Acklam's rational
 * approximation (relative error about 1e-9) and polishes it with one Halley
 * step against the accurate CDF above. Returns -Infinity / Infinity at 0 / 1.
 */
function normalQuantile(p: number): number {
  if (Number.isNaN(p) || p < 0 || p > 1) return Number.NaN;
  if (p === 0) return Number.NEGATIVE_INFINITY;
  if (p === 1) return Number.POSITIVE_INFINITY;
  if (p > 0.5) return -normalQuantile(1 - p);

  const plow = 0.02425;
  let z: number;
  if (p < plow) {
    const q = Math.sqrt(-2 * Math.log(p));
    z =
      (((((-7.784894002430293e-3 * q + -3.223964580411365e-1) * q + -2.400758277161838) * q +
        -2.549732539343734) *
        q +
        4.374664141464968) *
        q +
        2.938163982698783) /
      ((((7.784695709041462e-3 * q + 3.224671290700398e-1) * q + 2.445134137142996) * q +
        3.754408661907416) *
        q +
        1);
  } else {
    const q = p - 0.5;
    const r = q * q;
    z =
      ((((((-3.969683028665376e1 * r + 2.209460984245205e2) * r + -2.759285104469687e2) * r +
        1.38357751867269e2) *
        r +
        -3.066479806614716e1) *
        r +
        2.506628277459239) *
        q) /
      (((((-5.447609879822406e1 * r + 1.615858368580409e2) * r + -1.556989798598866e2) * r +
        6.680131188771972e1) *
        r +
        -1.328068155288572e1) *
        r +
        1);
  }

  const e = normalCdf(z) - p;
  const u = e * Math.sqrt(2 * Math.PI) * Math.exp((z * z) / 2);
  if (Number.isFinite(u)) {
    z -= u / (1 + (z * u) / 2);
  }
  return z;
}

/** Output of the normal QuantileTransformer is clipped here (as scikit-learn does). */
const NORMAL_CLIP_PROBABILITY = 1e-7 - EPS;
const NORMAL_CLIP_MIN = normalQuantile(NORMAL_CLIP_PROBABILITY);
const NORMAL_CLIP_MAX = normalQuantile(1 - NORMAL_CLIP_PROBABILITY);

/** The normal `inverseTransform` snaps probabilities closer than this to 0 or 1 to the data range ends. */
const INVERSE_BOUNDS_THRESHOLD = 1e-7;

// ---------------------------------------------------------------------------
// StandardScaler
// ---------------------------------------------------------------------------

/**
 * Standardize features by removing the mean and scaling to unit variance.
 *
 * **Formula**: z = (x - μ) / σ, with σ the population standard deviation
 * (divisor n, as in scikit-learn).
 *
 * Features with zero variance are only centered; their scale is treated as 1.
 * Statistics are accumulated in a numerically stable way, so
 * {@link StandardScaler.partialFit} can be called on successive batches and
 * gives the same result as a single `fit` on all rows.
 *
 * **Fitted attributes** (read-only getters): `mean`, `variance`, `scale`,
 * `nSamplesSeen`, `nFeaturesIn`.
 *
 * @example
 * ```js
 * import { StandardScaler } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [3, 4], [5, 6]]);
 * const scaler = new StandardScaler();
 * const XScaled = scaler.fitTransform(X);
 * const XBack = scaler.inverseTransform(XScaled);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox Scalers}
 */
export class StandardScaler {
  private withMean: boolean;
  private withStd: boolean;
  private nSamplesSeen_ = 0;
  private nFeaturesIn_ = 0;
  private runMean_: Float64Array | undefined;
  private runM2_: Float64Array | undefined;

  /**
   * Creates a new StandardScaler.
   *
   * @param options - Configuration options
   * @param options.withMean - Center data before scaling (default: true)
   * @param options.withStd - Scale data to unit variance (default: true)
   * @param options.copy - Accepted for API parity; transforms are always out-of-place (default: true)
   * @throws {InvalidParameterError} If an option is not a boolean
   */
  constructor(options: { withMean?: boolean; withStd?: boolean; copy?: boolean } = {}) {
    this.withMean = parseBooleanOption(options.withMean, "withMean", true);
    this.withStd = parseBooleanOption(options.withStd, "withStd", true);
    parseBooleanOption(options.copy, "copy", true);
  }

  private get fitted(): boolean {
    return this.runMean_ !== undefined;
  }

  /** Per-feature mean seen during fitting, or `undefined` if unfitted or `withMean` is false. */
  get mean(): Tensor | undefined {
    return this.withMean && this.runMean_ ? makeVector(this.runMean_) : undefined;
  }

  /** Per-feature population variance, or `undefined` if unfitted or `withStd` is false. */
  get variance(): Tensor | undefined {
    return this.withStd && this.runM2_ ? makeVector(this.varianceArray()) : undefined;
  }

  /** Per-feature scale (standard deviation, 1 for constant features), or `undefined` if unfitted or `withStd` is false. */
  get scale(): Tensor | undefined {
    return this.withStd && this.runM2_ ? makeVector(this.scaleArray()) : undefined;
  }

  /** Number of samples seen so far (0 when unfitted). */
  get nSamplesSeen(): number {
    return this.nSamplesSeen_;
  }

  /** Number of features seen during fitting (0 when unfitted). */
  get nFeaturesIn(): number {
    return this.nFeaturesIn_;
  }

  private varianceArray(): Float64Array {
    const m2 = this.runM2_ as Float64Array;
    const out = new Float64Array(m2.length);
    const n = this.nSamplesSeen_;
    for (let j = 0; j < m2.length; j++) out[j] = (m2[j] as number) / n;
    return out;
  }

  private scaleArray(): Float64Array {
    const variance = this.varianceArray();
    const mean = this.runMean_ as Float64Array;
    const n = this.nSamplesSeen_;
    const out = new Float64Array(variance.length);
    for (let j = 0; j < variance.length; j++) {
      const v = variance[j] as number;
      const m = mean[j] as number;
      // A variance that is within rounding noise of zero means a constant feature.
      const noiseBound = n * EPS * v + (n * m * EPS) ** 2;
      out[j] = v <= noiseBound ? 1 : Math.sqrt(v);
    }
    return out;
  }

  private accumulate(batch: DenseMatrix, reset: boolean): void {
    const { data, nSamples, nFeatures } = batch;
    const batchMean = new Float64Array(nFeatures);
    const lows = new Float64Array(nFeatures).fill(Number.POSITIVE_INFINITY);
    const highs = new Float64Array(nFeatures).fill(Number.NEGATIVE_INFINITY);
    let pos = 0;
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        const v = data[pos++] as number;
        batchMean[j] = (batchMean[j] as number) + v;
        if (v < (lows[j] as number)) lows[j] = v;
        if (v > (highs[j] as number)) highs[j] = v;
      }
    }
    for (let j = 0; j < nFeatures; j++) {
      // A mean always lies between the smallest and largest value; the clamp
      // removes summation round-off, so a constant feature has an exact mean.
      const mean = (batchMean[j] as number) / nSamples;
      batchMean[j] = Math.min(Math.max(mean, lows[j] as number), highs[j] as number);
    }
    const batchM2 = new Float64Array(nFeatures);
    pos = 0;
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        const d = (data[pos++] as number) - (batchMean[j] as number);
        batchM2[j] = (batchM2[j] as number) + d * d;
      }
    }

    const prevN = reset || !this.runMean_ || !this.runM2_ ? 0 : this.nSamplesSeen_;
    let nextMean = batchMean;
    let nextM2 = batchM2;
    if (prevN > 0 && this.runMean_ && this.runM2_) {
      const total = prevN + nSamples;
      nextMean = new Float64Array(nFeatures);
      nextM2 = new Float64Array(nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        const delta = (batchMean[j] as number) - (this.runMean_[j] as number);
        nextMean[j] = (this.runMean_[j] as number) + (delta * nSamples) / total;
        nextM2[j] =
          (this.runM2_[j] as number) +
          (batchM2[j] as number) +
          (delta * delta * prevN * nSamples) / total;
      }
    }
    for (let j = 0; j < nFeatures; j++) {
      if (!Number.isFinite(nextM2[j] as number)) {
        throw new DataValidationError(
          `Feature ${j} has values too large for its variance to be represented`
        );
      }
    }
    this.runMean_ = nextMean;
    this.runM2_ = nextM2;
    this.nSamplesSeen_ = prevN + nSamples;
    this.nFeaturesIn_ = nFeatures;
  }

  /**
   * Compute the mean and standard deviation of each feature, discarding any
   * previous fit.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (present for API compatibility)
   * @returns this
   * @throws {InvalidParameterError} If X has no samples or no features
   * @throws {ShapeError} If X is not 2-D
   * @throws {DTypeError} If X is not real numeric
   * @throws {DataValidationError} If X contains NaN or Infinity, or a feature's variance would overflow
   */
  fit(X: Tensor, _y?: Tensor): this {
    this.accumulate(readFitMatrix(X), true);
    return this;
  }

  /**
   * Update the running mean and variance with another batch of samples.
   * Calling it on an unfitted scaler starts a new fit.
   *
   * @param X - Batch of shape (n_samples, n_features)
   * @param _y - Ignored (present for API compatibility)
   * @returns this
   * @throws {ShapeError} If the number of features differs from earlier batches
   * @throws {DataValidationError} If a feature's variance would overflow
   */
  partialFit(X: Tensor, _y?: Tensor): this {
    const batch = readFitMatrix(X);
    if (this.fitted) {
      assertSameFeatureCount(batch.nFeatures, this.nFeaturesIn_, "StandardScaler", "partialFit");
    }
    this.accumulate(batch, false);
    return this;
  }

  /**
   * Center and scale X with the fitted statistics.
   *
   * @param X - Data of shape (n_samples, n_features)
   * @returns Float64 tensor of the same shape
   * @throws {NotFittedError} If the scaler has not been fitted
   * @throws {ShapeError} If the number of features differs from the fit
   * @throws {DataValidationError} If X contains NaN or Infinity
   */
  transform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("StandardScaler must be fitted before transform");
    }
    const { data, nSamples, nFeatures } = readApplyMatrix(
      X,
      this.nFeaturesIn_,
      "StandardScaler",
      true
    );
    const off = this.withMean ? (this.runMean_ as Float64Array) : new Float64Array(nFeatures);
    const sc = this.withStd ? this.scaleArray() : new Float64Array(nFeatures).fill(1);

    let pos = 0;
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        data[pos] = ((data[pos] as number) - (off[j] as number)) / (sc[j] as number);
        pos++;
      }
    }
    return makeMatrix(data, nSamples, nFeatures, X.device);
  }

  /**
   * Fit to X, then transform it.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (present for API compatibility)
   * @returns Scaled data
   */
  fitTransform(X: Tensor, _y?: Tensor): Tensor {
    return this.fit(X).transform(X);
  }

  /**
   * Undo the scaling: `x * scale + mean`. Results that are within a few
   * rounding errors of an integer are returned as that integer, so integer
   * data survives a round trip exactly.
   *
   * @param X - Scaled data of shape (n_samples, n_features)
   * @returns Data in the original feature space
   * @throws {NotFittedError} If the scaler has not been fitted
   * @throws {ShapeError} If the number of features differs from the fit
   */
  inverseTransform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("StandardScaler must be fitted before inverseTransform");
    }
    const { data, nSamples, nFeatures } = readApplyMatrix(
      X,
      this.nFeaturesIn_,
      "StandardScaler",
      false
    );
    const off = this.withMean ? (this.runMean_ as Float64Array) : new Float64Array(nFeatures);
    const sc = this.withStd ? this.scaleArray() : new Float64Array(nFeatures).fill(1);

    let pos = 0;
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        const scaled = (data[pos] as number) * (sc[j] as number);
        const shift = off[j] as number;
        data[pos] = snapToInteger(scaled + shift, Math.abs(scaled) + Math.abs(shift));
        pos++;
      }
    }
    return makeMatrix(data, nSamples, nFeatures, X.device);
  }

  /** Constructor options of this scaler. */
  getParams(): Record<string, unknown> {
    return { withMean: this.withMean, withStd: this.withStd };
  }

  /**
   * Change options. Fitted statistics are kept; they cover both options.
   *
   * @throws {InvalidParameterError} If a recognised option is not a boolean
   */
  setParams(params: Record<string, unknown>): this {
    if (params["withMean"] !== undefined) {
      this.withMean = parseBooleanOption(params["withMean"], "withMean", this.withMean);
    }
    if (params["withStd"] !== undefined) {
      this.withStd = parseBooleanOption(params["withStd"], "withStd", this.withStd);
    }
    return this;
  }

  /** Create an unfitted scaler with the same options. */
  clone(): StandardScaler {
    return new StandardScaler({ withMean: this.withMean, withStd: this.withStd });
  }
}

/**
 * Round `value` to the nearest integer when it lies within a few units of
 * rounding error (relative to the magnitude of the operands that produced it).
 */
function snapToInteger(value: number, magnitude: number): number {
  if (!Number.isFinite(value)) return value;
  const rounded = Math.round(value);
  return Math.abs(value - rounded) <= 4 * EPS * magnitude ? rounded : value;
}

// ---------------------------------------------------------------------------
// MinMaxScaler
// ---------------------------------------------------------------------------

/**
 * Scale each feature to a given range, by default [0, 1].
 *
 * **Formula**: X_scaled = (X - X.min) / (X.max - X.min) * (max - min) + min
 *
 * A feature with a single distinct value is shifted so that this value maps to
 * `featureRange[0]`. Values outside the fitted range map outside the target
 * range unless `clip` is set.
 *
 * **Fitted attributes** (read-only getters): `dataMin`, `dataMax`, `dataRange`,
 * `scale`, `nSamplesSeen`, `nFeaturesIn`.
 *
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox Scalers}
 */
export class MinMaxScaler {
  private featureRange: [number, number];
  private clip: boolean;
  private nSamplesSeen_ = 0;
  private nFeaturesIn_ = 0;
  private dataMin_: Float64Array | undefined;
  private dataMax_: Float64Array | undefined;

  /**
   * Creates a new MinMaxScaler.
   *
   * @param options - Configuration options
   * @param options.featureRange - Desired feature range [min, max] (default: [0, 1])
   * @param options.clip - Clip transformed values to featureRange (default: false)
   * @param options.copy - Accepted for API parity; transforms are always out-of-place (default: true)
   * @throws {InvalidParameterError} If featureRange is not a finite ascending pair
   */
  constructor(
    options: {
      featureRange?: [number, number];
      clip?: boolean;
      copy?: boolean;
    } = {}
  ) {
    this.featureRange = validateFeatureRange(options.featureRange ?? [0, 1]);
    this.clip = parseBooleanOption(options.clip, "clip", false);
    parseBooleanOption(options.copy, "copy", true);
  }

  private get fitted(): boolean {
    return this.dataMin_ !== undefined;
  }

  /** Per-feature minimum seen during fitting, or `undefined` if unfitted. */
  get dataMin(): Tensor | undefined {
    return this.dataMin_ ? makeVector(this.dataMin_) : undefined;
  }

  /** Per-feature maximum seen during fitting, or `undefined` if unfitted. */
  get dataMax(): Tensor | undefined {
    return this.dataMax_ ? makeVector(this.dataMax_) : undefined;
  }

  /** Per-feature range (max - min) seen during fitting, or `undefined` if unfitted. */
  get dataRange(): Tensor | undefined {
    return this.dataMin_ && this.dataMax_ ? makeVector(this.rangeArray()) : undefined;
  }

  /** Per-feature multiplier applied by `transform`, or `undefined` if unfitted. */
  get scale(): Tensor | undefined {
    if (!this.dataMin_ || !this.dataMax_) return undefined;
    const [lo, hi] = this.featureRange;
    const range = nonZeroScale(this.rangeArray());
    const out = new Float64Array(range.length);
    for (let j = 0; j < range.length; j++) out[j] = (hi - lo) / (range[j] as number);
    return makeVector(out);
  }

  /** Number of samples seen so far (0 when unfitted). */
  get nSamplesSeen(): number {
    return this.nSamplesSeen_;
  }

  /** Number of features seen during fitting (0 when unfitted). */
  get nFeaturesIn(): number {
    return this.nFeaturesIn_;
  }

  private rangeArray(): Float64Array {
    const lo = this.dataMin_ as Float64Array;
    const hi = this.dataMax_ as Float64Array;
    const out = new Float64Array(lo.length);
    for (let j = 0; j < lo.length; j++) out[j] = (hi[j] as number) - (lo[j] as number);
    return out;
  }

  private accumulate(batch: DenseMatrix, reset: boolean): void {
    const { data, nSamples, nFeatures } = batch;
    let mins: Float64Array;
    let maxs: Float64Array;
    if (reset || !this.dataMin_ || !this.dataMax_) {
      mins = new Float64Array(nFeatures).fill(Number.POSITIVE_INFINITY);
      maxs = new Float64Array(nFeatures).fill(Number.NEGATIVE_INFINITY);
      this.nSamplesSeen_ = 0;
    } else {
      mins = this.dataMin_;
      maxs = this.dataMax_;
    }
    let pos = 0;
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        const v = data[pos++] as number;
        if (v < (mins[j] as number)) mins[j] = v;
        if (v > (maxs[j] as number)) maxs[j] = v;
      }
    }
    this.dataMin_ = mins;
    this.dataMax_ = maxs;
    this.nSamplesSeen_ += nSamples;
    this.nFeaturesIn_ = nFeatures;
  }

  /**
   * Compute the per-feature minimum and maximum, discarding any previous fit.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (present for API compatibility)
   * @returns this
   * @throws {InvalidParameterError} If X has no samples or no features
   * @throws {ShapeError} If X is not 2-D
   * @throws {DataValidationError} If X contains NaN or Infinity
   */
  fit(X: Tensor, _y?: Tensor): this {
    this.accumulate(readFitMatrix(X), true);
    return this;
  }

  /**
   * Update the running minimum and maximum with another batch of samples.
   *
   * @param X - Batch of shape (n_samples, n_features)
   * @param _y - Ignored (present for API compatibility)
   * @returns this
   * @throws {ShapeError} If the number of features differs from earlier batches
   */
  partialFit(X: Tensor, _y?: Tensor): this {
    const batch = readFitMatrix(X);
    if (this.fitted) {
      assertSameFeatureCount(batch.nFeatures, this.nFeaturesIn_, "MinMaxScaler", "partialFit");
    }
    this.accumulate(batch, false);
    return this;
  }

  /**
   * Map X into the feature range using the fitted minimum and maximum.
   *
   * @param X - Data of shape (n_samples, n_features)
   * @returns Float64 tensor of the same shape
   * @throws {NotFittedError} If the scaler has not been fitted
   * @throws {ShapeError} If the number of features differs from the fit
   * @throws {DataValidationError} If X contains NaN or Infinity
   */
  transform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("MinMaxScaler must be fitted before transform");
    }
    const { data, nSamples, nFeatures } = readApplyMatrix(
      X,
      this.nFeaturesIn_,
      "MinMaxScaler",
      true
    );
    const [minRange, maxRange] = this.featureRange;
    const minArr = this.dataMin_ as Float64Array;
    const maxArr = this.dataMax_ as Float64Array;
    const range = nonZeroScale(this.rangeArray());
    const clip = this.clip;

    // t is 0 at the fitted minimum and 1 at the fitted maximum; blending the
    // two range ends keeps both of them exact.
    let pos = 0;
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        const x = data[pos] as number;
        const t = Number.isFinite(range[j] as number)
          ? (x - (minArr[j] as number)) / (range[j] as number)
          : // max - min overflows: halve every term, which keeps the ratio and avoids Infinity.
            (x / 2 - (minArr[j] as number) / 2) /
            ((maxArr[j] as number) / 2 - (minArr[j] as number) / 2);
        let scaled = minRange * (1 - t) + maxRange * t;
        if (clip) scaled = Math.max(minRange, Math.min(maxRange, scaled));
        data[pos++] = scaled;
      }
    }
    return makeMatrix(data, nSamples, nFeatures, X.device);
  }

  /**
   * Fit to X, then transform it.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (present for API compatibility)
   * @returns Scaled data
   */
  fitTransform(X: Tensor, _y?: Tensor): Tensor {
    return this.fit(X).transform(X);
  }

  /**
   * Map data from the feature range back to the original feature space.
   *
   * @param X - Scaled data of shape (n_samples, n_features)
   * @returns Data in the original feature space
   * @throws {NotFittedError} If the scaler has not been fitted
   * @throws {ShapeError} If the number of features differs from the fit
   */
  inverseTransform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("MinMaxScaler must be fitted before inverseTransform");
    }
    const { data, nSamples, nFeatures } = readApplyMatrix(
      X,
      this.nFeaturesIn_,
      "MinMaxScaler",
      false
    );
    const [minRange, maxRange] = this.featureRange;
    const span = maxRange - minRange;
    const minArr = this.dataMin_ as Float64Array;
    const maxArr = this.dataMax_ as Float64Array;
    const range = nonZeroScale(this.rangeArray());

    let pos = 0;
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        const u = ((data[pos] as number) - minRange) / span;
        // max - min overflows: blend the two ends instead of multiplying by the range.
        data[pos] = Number.isFinite(range[j] as number)
          ? u * (range[j] as number) + (minArr[j] as number)
          : (minArr[j] as number) * (1 - u) + (maxArr[j] as number) * u;
        pos++;
      }
    }
    return makeMatrix(data, nSamples, nFeatures, X.device);
  }

  /** Constructor options of this scaler. */
  getParams(): Record<string, unknown> {
    return { featureRange: [this.featureRange[0], this.featureRange[1]], clip: this.clip };
  }

  /**
   * Change options. Fitted statistics are kept.
   *
   * @throws {InvalidParameterError} If a recognised option is invalid
   */
  setParams(params: Record<string, unknown>): this {
    if (params["featureRange"] !== undefined) {
      this.featureRange = validateFeatureRange(params["featureRange"]);
    }
    if (params["clip"] !== undefined) {
      this.clip = parseBooleanOption(params["clip"], "clip", this.clip);
    }
    return this;
  }

  /** Create an unfitted scaler with the same options. */
  clone(): MinMaxScaler {
    return new MinMaxScaler({ featureRange: [...this.featureRange], clip: this.clip });
  }
}

function validateFeatureRange(value: unknown): [number, number] {
  if (
    !Array.isArray(value) ||
    value.length !== 2 ||
    typeof value[0] !== "number" ||
    typeof value[1] !== "number" ||
    !Number.isFinite(value[0]) ||
    !Number.isFinite(value[1]) ||
    value[0] >= value[1]
  ) {
    throw new InvalidParameterError(
      "featureRange must be [min, max] with min < max",
      "featureRange",
      value
    );
  }
  return [value[0], value[1]];
}

// ---------------------------------------------------------------------------
// MaxAbsScaler
// ---------------------------------------------------------------------------

/**
 * Scale each feature by its maximum absolute value.
 *
 * The result lies in [-1, 1]. No centering is applied, so zeros stay zeros.
 * Features that are entirely zero are left unchanged.
 *
 * **Fitted attributes** (read-only getters): `maxAbs`, `scale`,
 * `nSamplesSeen`, `nFeaturesIn`.
 *
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox Scalers}
 */
export class MaxAbsScaler {
  private nSamplesSeen_ = 0;
  private nFeaturesIn_ = 0;
  private maxAbs_: Float64Array | undefined;

  /**
   * Creates a new MaxAbsScaler.
   *
   * @param options - Configuration options
   * @param options.copy - Accepted for API parity; transforms are always out-of-place (default: true)
   */
  constructor(options: { copy?: boolean } = {}) {
    parseBooleanOption(options.copy, "copy", true);
  }

  private get fitted(): boolean {
    return this.maxAbs_ !== undefined;
  }

  /** Per-feature maximum absolute value, or `undefined` if unfitted. */
  get maxAbs(): Tensor | undefined {
    return this.maxAbs_ ? makeVector(this.maxAbs_) : undefined;
  }

  /** Per-feature divisor applied by `transform` (1 for all-zero features), or `undefined` if unfitted. */
  get scale(): Tensor | undefined {
    return this.maxAbs_ ? makeVector(nonZeroScale(this.maxAbs_)) : undefined;
  }

  /** Number of samples seen so far (0 when unfitted). */
  get nSamplesSeen(): number {
    return this.nSamplesSeen_;
  }

  /** Number of features seen during fitting (0 when unfitted). */
  get nFeaturesIn(): number {
    return this.nFeaturesIn_;
  }

  private accumulate(batch: DenseMatrix, reset: boolean): void {
    const { data, nSamples, nFeatures } = batch;
    let maxAbs: Float64Array;
    if (reset || !this.maxAbs_) {
      maxAbs = new Float64Array(nFeatures);
      this.nSamplesSeen_ = 0;
    } else {
      maxAbs = this.maxAbs_;
    }
    let pos = 0;
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        const a = Math.abs(data[pos++] as number);
        if (a > (maxAbs[j] as number)) maxAbs[j] = a;
      }
    }
    this.maxAbs_ = maxAbs;
    this.nSamplesSeen_ += nSamples;
    this.nFeaturesIn_ = nFeatures;
  }

  /**
   * Compute the maximum absolute value of each feature, discarding any
   * previous fit.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (present for API compatibility)
   * @returns this
   * @throws {InvalidParameterError} If X has no samples or no features
   * @throws {ShapeError} If X is not 2-D
   * @throws {DataValidationError} If X contains NaN or Infinity
   */
  fit(X: Tensor, _y?: Tensor): this {
    this.accumulate(readFitMatrix(X), true);
    return this;
  }

  /**
   * Update the running maximum absolute values with another batch of samples.
   *
   * @param X - Batch of shape (n_samples, n_features)
   * @param _y - Ignored (present for API compatibility)
   * @returns this
   * @throws {ShapeError} If the number of features differs from earlier batches
   */
  partialFit(X: Tensor, _y?: Tensor): this {
    const batch = readFitMatrix(X);
    if (this.fitted) {
      assertSameFeatureCount(batch.nFeatures, this.nFeaturesIn_, "MaxAbsScaler", "partialFit");
    }
    this.accumulate(batch, false);
    return this;
  }

  /**
   * Divide each feature by its fitted maximum absolute value.
   *
   * @param X - Data of shape (n_samples, n_features)
   * @returns Float64 tensor of the same shape
   * @throws {NotFittedError} If the scaler has not been fitted
   * @throws {ShapeError} If the number of features differs from the fit
   * @throws {DataValidationError} If X contains NaN or Infinity
   */
  transform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("MaxAbsScaler must be fitted before transform");
    }
    const { data, nSamples, nFeatures } = readApplyMatrix(
      X,
      this.nFeaturesIn_,
      "MaxAbsScaler",
      true
    );
    const sc = nonZeroScale(this.maxAbs_ as Float64Array);
    let pos = 0;
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        data[pos] = (data[pos] as number) / (sc[j] as number);
        pos++;
      }
    }
    return makeMatrix(data, nSamples, nFeatures, X.device);
  }

  /**
   * Fit to X, then transform it.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (present for API compatibility)
   * @returns Scaled data
   */
  fitTransform(X: Tensor, _y?: Tensor): Tensor {
    return this.fit(X).transform(X);
  }

  /**
   * Multiply each feature by its fitted maximum absolute value.
   *
   * @param X - Scaled data of shape (n_samples, n_features)
   * @returns Data in the original feature space
   * @throws {NotFittedError} If the scaler has not been fitted
   * @throws {ShapeError} If the number of features differs from the fit
   */
  inverseTransform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("MaxAbsScaler must be fitted before inverseTransform");
    }
    const { data, nSamples, nFeatures } = readApplyMatrix(
      X,
      this.nFeaturesIn_,
      "MaxAbsScaler",
      false
    );
    const sc = nonZeroScale(this.maxAbs_ as Float64Array);
    let pos = 0;
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        data[pos] = (data[pos] as number) * (sc[j] as number);
        pos++;
      }
    }
    return makeMatrix(data, nSamples, nFeatures, X.device);
  }

  /** Constructor options of this scaler. */
  getParams(): Record<string, unknown> {
    return {};
  }

  /** This scaler has no options; present for API consistency. */
  setParams(_params: Record<string, unknown>): this {
    return this;
  }

  /** Create an unfitted scaler. */
  clone(): MaxAbsScaler {
    return new MaxAbsScaler();
  }
}

// ---------------------------------------------------------------------------
// RobustScaler
// ---------------------------------------------------------------------------

function validateQuantileRange(value: unknown, unitVariance: boolean): [number, number] {
  if (
    !Array.isArray(value) ||
    value.length !== 2 ||
    typeof value[0] !== "number" ||
    typeof value[1] !== "number" ||
    !Number.isFinite(value[0]) ||
    !Number.isFinite(value[1]) ||
    value[0] < 0 ||
    value[1] > 100 ||
    value[0] >= value[1]
  ) {
    throw new InvalidParameterError(
      "quantileRange must be a valid ascending percentile range",
      "quantileRange",
      value
    );
  }
  if (unitVariance && (value[0] <= 0 || value[1] >= 100)) {
    throw new InvalidParameterError(
      "quantileRange must lie strictly between 0 and 100 when unitVariance is true",
      "quantileRange",
      value
    );
  }
  return [value[0], value[1]];
}

/**
 * Scale features using statistics that are less sensitive to outliers.
 *
 * Each feature has its median removed and is divided by an inter-quantile
 * range (the IQR, from the 25th to the 75th percentile, by default).
 * Quantiles use linear interpolation, like `numpy.percentile`. Features whose
 * range is zero are only centered.
 *
 * **Fitted attributes** (read-only getters): `center`, `scale`, `nFeaturesIn`.
 *
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox Scalers}
 */
export class RobustScaler {
  private center_: Float64Array | undefined;
  private scale_: Float64Array | undefined;
  private nFeaturesIn_ = 0;
  private withCentering: boolean;
  private withScaling: boolean;
  private quantileRange: [number, number];
  private unitVariance: boolean;

  /**
   * Creates a new RobustScaler.
   *
   * @param options - Configuration options
   * @param options.withCentering - Center data using median (default: true)
   * @param options.withScaling - Scale data using IQR (default: true)
   * @param options.quantileRange - Quantile range for IQR as percentiles (default: [25, 75])
   * @param options.unitVariance - Divide the range by the same range of a standard normal, so that normally distributed features get unit variance (default: false). Requires a quantileRange strictly inside (0, 100).
   * @param options.copy - Accepted for API parity; transforms are always out-of-place (default: true)
   * @throws {InvalidParameterError} If an option is invalid
   */
  constructor(
    options: {
      withCentering?: boolean;
      withScaling?: boolean;
      quantileRange?: [number, number];
      unitVariance?: boolean;
      copy?: boolean;
    } = {}
  ) {
    this.withCentering = parseBooleanOption(options.withCentering, "withCentering", true);
    this.withScaling = parseBooleanOption(options.withScaling, "withScaling", true);
    this.unitVariance = parseBooleanOption(options.unitVariance, "unitVariance", false);
    parseBooleanOption(options.copy, "copy", true);
    this.quantileRange = validateQuantileRange(
      options.quantileRange ?? [25, 75],
      this.unitVariance
    );
  }

  private get fitted(): boolean {
    return this.center_ !== undefined;
  }

  /** Per-feature median, or `undefined` if unfitted or `withCentering` is false. */
  get center(): Tensor | undefined {
    return this.withCentering && this.center_ ? makeVector(this.center_) : undefined;
  }

  /** Per-feature scale (1 where the range is zero), or `undefined` if unfitted or `withScaling` is false. */
  get scale(): Tensor | undefined {
    return this.withScaling && this.scale_ ? makeVector(nonZeroScale(this.scale_)) : undefined;
  }

  /** Number of features seen during fitting (0 when unfitted). */
  get nFeaturesIn(): number {
    return this.nFeaturesIn_;
  }

  /**
   * Compute the median and quantile range of each feature.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (present for API compatibility)
   * @returns this
   * @throws {InvalidParameterError} If X has no samples or no features
   * @throws {ShapeError} If X is not 2-D
   * @throws {DataValidationError} If X contains NaN or Infinity
   */
  fit(X: Tensor, _y?: Tensor): this {
    const { data, nSamples, nFeatures } = readFitMatrix(X);
    const centers = new Float64Array(nFeatures);
    const scales = new Float64Array(nFeatures);

    const lowerFraction = this.quantileRange[0] / 100;
    const upperFraction = this.quantileRange[1] / 100;
    const normalizer = this.unitVariance
      ? normalQuantile(upperFraction) - normalQuantile(lowerFraction)
      : 1;

    const column = new Float64Array(nSamples);
    for (let j = 0; j < nFeatures; j++) {
      for (let i = 0; i < nSamples; i++) column[i] = data[i * nFeatures + j] as number;
      column.sort();
      centers[j] = quantileSorted(column, 0.5);
      const range = quantileSorted(column, upperFraction) - quantileSorted(column, lowerFraction);
      scales[j] = range / normalizer;
    }

    this.center_ = centers;
    this.scale_ = scales;
    this.nFeaturesIn_ = nFeatures;
    return this;
  }

  /**
   * Remove the median and divide by the quantile range.
   *
   * @param X - Data of shape (n_samples, n_features)
   * @returns Float64 tensor of the same shape
   * @throws {NotFittedError} If the scaler has not been fitted
   * @throws {ShapeError} If the number of features differs from the fit
   * @throws {DataValidationError} If X contains NaN or Infinity
   */
  transform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("RobustScaler must be fitted before transform");
    }
    const { data, nSamples, nFeatures } = readApplyMatrix(
      X,
      this.nFeaturesIn_,
      "RobustScaler",
      true
    );
    const off = this.withCentering ? (this.center_ as Float64Array) : new Float64Array(nFeatures);
    const sc = this.withScaling
      ? nonZeroScale(this.scale_ as Float64Array)
      : new Float64Array(nFeatures).fill(1);

    let pos = 0;
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        data[pos] = ((data[pos] as number) - (off[j] as number)) / (sc[j] as number);
        pos++;
      }
    }
    return makeMatrix(data, nSamples, nFeatures, X.device);
  }

  /**
   * Fit to X, then transform it.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (present for API compatibility)
   * @returns Scaled data
   */
  fitTransform(X: Tensor, _y?: Tensor): Tensor {
    return this.fit(X).transform(X);
  }

  /**
   * Undo the scaling: `x * scale + center`.
   *
   * @param X - Scaled data of shape (n_samples, n_features)
   * @returns Data in the original feature space
   * @throws {NotFittedError} If the scaler has not been fitted
   * @throws {ShapeError} If the number of features differs from the fit
   */
  inverseTransform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("RobustScaler must be fitted before inverseTransform");
    }
    const { data, nSamples, nFeatures } = readApplyMatrix(
      X,
      this.nFeaturesIn_,
      "RobustScaler",
      false
    );
    const off = this.withCentering ? (this.center_ as Float64Array) : new Float64Array(nFeatures);
    const sc = this.withScaling
      ? nonZeroScale(this.scale_ as Float64Array)
      : new Float64Array(nFeatures).fill(1);

    let pos = 0;
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        data[pos] = (data[pos] as number) * (sc[j] as number) + (off[j] as number);
        pos++;
      }
    }
    return makeMatrix(data, nSamples, nFeatures, X.device);
  }

  /** Constructor options of this scaler. */
  getParams(): Record<string, unknown> {
    return {
      withCentering: this.withCentering,
      withScaling: this.withScaling,
      quantileRange: [this.quantileRange[0], this.quantileRange[1]],
      unitVariance: this.unitVariance,
    };
  }

  /**
   * Change options. Changing `quantileRange` or `unitVariance` discards the
   * fitted state, because the stored scale depends on them.
   *
   * @throws {InvalidParameterError} If a recognised option is invalid
   */
  setParams(params: Record<string, unknown>): this {
    const withCentering = parseBooleanOption(
      params["withCentering"],
      "withCentering",
      this.withCentering
    );
    const withScaling = parseBooleanOption(params["withScaling"], "withScaling", this.withScaling);
    const unitVariance = parseBooleanOption(
      params["unitVariance"],
      "unitVariance",
      this.unitVariance
    );
    const quantileRange = validateQuantileRange(
      params["quantileRange"] ?? this.quantileRange,
      unitVariance
    );
    const changesFit =
      unitVariance !== this.unitVariance ||
      quantileRange[0] !== this.quantileRange[0] ||
      quantileRange[1] !== this.quantileRange[1];
    this.withCentering = withCentering;
    this.withScaling = withScaling;
    this.unitVariance = unitVariance;
    this.quantileRange = quantileRange;
    if (changesFit) {
      this.center_ = undefined;
      this.scale_ = undefined;
      this.nFeaturesIn_ = 0;
    }
    return this;
  }

  /** Create an unfitted scaler with the same options. */
  clone(): RobustScaler {
    return new RobustScaler({
      withCentering: this.withCentering,
      withScaling: this.withScaling,
      quantileRange: [this.quantileRange[0], this.quantileRange[1]],
      unitVariance: this.unitVariance,
    });
  }
}

// ---------------------------------------------------------------------------
// Normalizer
// ---------------------------------------------------------------------------

/**
 * Scale each sample (row) to unit norm.
 *
 * Rows whose norm is zero are left unchanged. The transformer is stateless:
 * `fit` only validates its input.
 *
 * @example
 * ```js
 * import { Normalizer } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[3, 4], [1, 0]]);
 * const Xn = new Normalizer({ norm: 'l2' }).transform(X); // [[0.6, 0.8], [1, 0]]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox Scalers}
 */
export class Normalizer {
  private norm: "l1" | "l2" | "max";

  /**
   * Creates a new Normalizer.
   *
   * @param options - Configuration options
   * @param options.norm - Norm to use: "l1", "l2" or "max" (default: "l2")
   * @param options.copy - Accepted for API parity; transforms are always out-of-place (default: true)
   * @throws {InvalidParameterError} If norm is not one of the supported values
   */
  constructor(options: { norm?: "l1" | "l2" | "max"; copy?: boolean } = {}) {
    this.norm = parseNorm(options.norm ?? "l2");
    parseBooleanOption(options.copy, "copy", true);
  }

  /**
   * Validate X. The normalizer has nothing to learn.
   *
   * @param X - Data of shape (n_samples, n_features)
   * @param _y - Ignored (present for API compatibility)
   * @returns this
   * @throws {ShapeError} If X is not 2-D
   * @throws {DTypeError} If X is not real numeric
   */
  fit(X: Tensor, _y?: Tensor): this {
    assert2D(X, "X");
    assertNumericTensor(X, "X");
    return this;
  }

  /**
   * Divide each row by its norm.
   *
   * @param X - Data of shape (n_samples, n_features)
   * @returns Float64 tensor of the same shape
   * @throws {ShapeError} If X is not 2-D
   * @throws {DataValidationError} If X contains NaN or Infinity
   */
  transform(X: Tensor): Tensor {
    assert2D(X, "X");
    assertNumericTensor(X, "X");
    const { data: out, nSamples, nFeatures } = denseRowMajor(X, "X", true);
    const kind = this.norm;

    for (let i = 0; i < nSamples; i++) {
      const rowBase = i * nFeatures;
      let norm = 0;
      if (kind === "l2") {
        for (let j = 0; j < nFeatures; j++) {
          const v = out[rowBase + j] as number;
          norm += v * v;
        }
        norm = Math.sqrt(norm);
        // Squares overflow or underflow for very large or very small rows.
        if (!Number.isFinite(norm) || norm < 1e-150) {
          normalizeRowScaled(out, rowBase, nFeatures, true);
          continue;
        }
      } else if (kind === "l1") {
        for (let j = 0; j < nFeatures; j++) norm += Math.abs(out[rowBase + j] as number);
        if (!Number.isFinite(norm)) {
          normalizeRowScaled(out, rowBase, nFeatures, false);
          continue;
        }
      } else {
        for (let j = 0; j < nFeatures; j++) {
          norm = Math.max(norm, Math.abs(out[rowBase + j] as number));
        }
      }
      if (norm !== 0) {
        for (let j = 0; j < nFeatures; j++) {
          out[rowBase + j] = (out[rowBase + j] as number) / norm;
        }
      }
    }

    return makeMatrix(out, nSamples, nFeatures, X.device);
  }

  /**
   * Transform X (there is nothing to fit).
   *
   * @param X - Data of shape (n_samples, n_features)
   * @param _y - Ignored (present for API compatibility)
   * @returns Normalized data
   */
  fitTransform(X: Tensor, _y?: Tensor): Tensor {
    return this.fit(X).transform(X);
  }

  /** Constructor options of this normalizer. */
  getParams(): Record<string, unknown> {
    return { norm: this.norm };
  }

  /**
   * Change options.
   *
   * @throws {InvalidParameterError} If `norm` is not one of the supported values
   */
  setParams(params: Record<string, unknown>): this {
    if (params["norm"] !== undefined) {
      this.norm = parseNorm(params["norm"]);
    }
    return this;
  }

  /** Create a normalizer with the same options. */
  clone(): Normalizer {
    return new Normalizer({ norm: this.norm });
  }
}

function parseNorm(value: unknown): "l1" | "l2" | "max" {
  if (value !== "l1" && value !== "l2" && value !== "max") {
    throw new InvalidParameterError("norm must be one of: l1, l2, max", "norm", value);
  }
  return value;
}

/**
 * Normalize one row in place to unit L2 (`squares`) or L1 norm, working relative to
 * its largest entry so that neither the norm nor its squares overflow or underflow.
 */
function normalizeRowScaled(
  data: Float64Array,
  start: number,
  length: number,
  squares: boolean
): void {
  let max = 0;
  for (let j = 0; j < length; j++) max = Math.max(max, Math.abs(data[start + j] as number));
  if (max === 0) return;
  let sum = 0;
  for (let j = 0; j < length; j++) {
    const r = (data[start + j] as number) / max;
    data[start + j] = r;
    sum += squares ? r * r : Math.abs(r);
  }
  const unit = squares ? Math.sqrt(sum) : sum;
  for (let j = 0; j < length; j++) data[start + j] = (data[start + j] as number) / unit;
}

// ---------------------------------------------------------------------------
// QuantileTransformer
// ---------------------------------------------------------------------------

/**
 * Transform features so that their distribution is uniform or normal.
 *
 * Each feature is mapped through its empirical quantile function, estimated
 * from `nQuantiles` evenly spaced quantiles of the training data and
 * interpolated linearly in between. Values outside the training range are
 * mapped to the ends of the output range. Tied training values map to the
 * average of the first and last quantile they cover. With
 * `outputDistribution: "normal"` the output is clipped to about ±5.2.
 *
 * Unlike scikit-learn, `transform` does not snap values lying within an absolute
 * distance of 1e-7 of the training minimum or maximum to the ends of the output range, because
 * that tolerance depends on the data scale. Results differ from scikit-learn only for such values,
 * by at most the local slope times 1e-7.
 *
 * **Fitted attributes** (read-only getters): `quantiles`, `references`,
 * `nFeaturesIn`.
 *
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox Scalers}
 */
export class QuantileTransformer {
  private nQuantiles: number;
  private outputDistribution: "uniform" | "normal";
  private subsample: number | undefined;
  private randomState: number | undefined;
  private nFeaturesIn_ = 0;
  private nQuantilesFitted_ = 0;
  /** Feature-major quantile table: feature j occupies [j * n, (j + 1) * n). */
  private quantiles_: Float64Array | undefined;
  private references_: Float64Array | undefined;

  /**
   * Creates a new QuantileTransformer.
   *
   * @param options - Configuration options
   * @param options.nQuantiles - Number of quantiles to use; reduced to the number of samples when larger (default: 1000)
   * @param options.outputDistribution - "uniform" or "normal" (default: "uniform")
   * @param options.subsample - Maximum number of samples used to estimate the quantiles (default: use all samples)
   * @param options.randomState - Seed for subsampling reproducibility
   * @param options.copy - Accepted for API parity; transforms are always out-of-place (default: true)
   * @throws {InvalidParameterError} If an option is invalid
   */
  constructor(
    options: {
      nQuantiles?: number;
      outputDistribution?: "uniform" | "normal";
      subsample?: number;
      randomState?: number;
      copy?: boolean;
    } = {}
  ) {
    this.nQuantiles = parseNQuantiles(options.nQuantiles ?? 1000);
    this.outputDistribution = parseOutputDistribution(options.outputDistribution ?? "uniform");
    this.subsample = parseSubsample(options.subsample);
    this.randomState = options.randomState;
    parseBooleanOption(options.copy, "copy", true);
  }

  /** Fitted quantiles as a tensor of shape (n_quantiles, n_features), or `undefined` if unfitted. */
  get quantiles(): Tensor | undefined {
    if (!this.quantiles_) return undefined;
    const n = this.nQuantilesFitted_;
    const p = this.nFeaturesIn_;
    const out = new Float64Array(n * p);
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < n; k++) out[k * p + j] = this.quantiles_[j * n + k] as number;
    }
    return makeMatrix(out, n, p, "cpu");
  }

  /** Probabilities that the fitted quantiles correspond to, or `undefined` if unfitted. */
  get references(): Tensor | undefined {
    return this.references_ ? makeVector(this.references_) : undefined;
  }

  /** Number of features seen during fitting (0 when unfitted). */
  get nFeaturesIn(): number {
    return this.nFeaturesIn_;
  }

  /**
   * Estimate the quantiles of each feature, discarding any previous fit.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (present for API compatibility)
   * @returns this
   * @throws {InvalidParameterError} If X has no samples or no features
   * @throws {ShapeError} If X is not 2-D
   * @throws {DataValidationError} If X contains NaN or Infinity
   */
  fit(X: Tensor, _y?: Tensor): this {
    const { data, nSamples, nFeatures } = readFitMatrix(X);

    const sampleCount =
      this.subsample !== undefined ? Math.min(this.subsample, nSamples) : nSamples;
    const n = Math.min(this.nQuantiles, sampleCount);
    const references = new Float64Array(n);
    if (n === 1) {
      references[0] = 0.5;
    } else {
      for (let i = 0; i < n; i++) references[i] = i / (n - 1);
    }

    let sampleIndices: number[] | undefined;
    if (sampleCount < nSamples) {
      const all = Array.from({ length: nSamples }, (_, i) => i);
      const random =
        this.randomState !== undefined ? createRandomStream(this.randomState) : __random;
      shuffleIndicesInPlace(all, random);
      sampleIndices = all.slice(0, sampleCount);
    }

    const quantiles = new Float64Array(nFeatures * n);
    const column = new Float64Array(sampleCount);
    for (let j = 0; j < nFeatures; j++) {
      for (let k = 0; k < sampleCount; k++) {
        const row = sampleIndices ? (sampleIndices[k] as number) : k;
        column[k] = data[row * nFeatures + j] as number;
      }
      column.sort();
      for (let k = 0; k < n; k++)
        quantiles[j * n + k] = quantileSorted(column, references[k] as number);
    }

    this.quantiles_ = quantiles;
    this.references_ = references;
    this.nQuantilesFitted_ = n;
    this.nFeaturesIn_ = nFeatures;
    return this;
  }

  /**
   * Map X to the output distribution.
   *
   * @param X - Data of shape (n_samples, n_features)
   * @returns Float64 tensor of the same shape
   * @throws {NotFittedError} If the transformer has not been fitted
   * @throws {ShapeError} If the number of features differs from the fit
   * @throws {DataValidationError} If X contains NaN or Infinity
   */
  transform(X: Tensor): Tensor {
    if (!this.quantiles_ || !this.references_) {
      throw new NotFittedError("QuantileTransformer must be fitted before transform");
    }
    const { data, nSamples, nFeatures } = readApplyMatrix(
      X,
      this.nFeaturesIn_,
      "QuantileTransformer",
      true
    );
    const n = this.nQuantilesFitted_;
    const normal = this.outputDistribution === "normal";

    for (let j = 0; j < nFeatures; j++) {
      const base = j * n;
      for (let i = 0; i < nSamples; i++) {
        const pos = i * nFeatures + j;
        const u = valueToProbability(
          data[pos] as number,
          this.quantiles_,
          base,
          n,
          this.references_
        );
        data[pos] = normal
          ? Math.max(NORMAL_CLIP_MIN, Math.min(NORMAL_CLIP_MAX, normalQuantile(u)))
          : u;
      }
    }
    return makeMatrix(data, nSamples, nFeatures, X.device);
  }

  /**
   * Fit to X, then transform it.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (present for API compatibility)
   * @returns Transformed data
   */
  fitTransform(X: Tensor, _y?: Tensor): Tensor {
    return this.fit(X).transform(X);
  }

  /**
   * Inverse transform data back to the original feature space.
   *
   * If `outputDistribution="normal"`, values are first mapped back to uniform
   * quantiles before being projected into the original data distribution.
   * Values outside the output range map to the training minimum or maximum.
   *
   * @param X - Transformed data (2D tensor)
   * @returns Data in the original feature space
   * @throws {NotFittedError} If transformer is not fitted
   * @throws {ShapeError} If the number of features differs from the fit
   * @throws {DataValidationError} If X contains NaN
   */
  inverseTransform(X: Tensor): Tensor {
    if (!this.quantiles_) {
      throw new NotFittedError("QuantileTransformer must be fitted before inverseTransform");
    }
    const { data, nSamples, nFeatures } = readApplyMatrix(
      X,
      this.nFeaturesIn_,
      "QuantileTransformer",
      false
    );
    const n = this.nQuantilesFitted_;
    const normal = this.outputDistribution === "normal";

    for (let j = 0; j < nFeatures; j++) {
      const base = j * n;
      for (let i = 0; i < nSamples; i++) {
        const pos = i * nFeatures + j;
        const raw = data[pos] as number;
        if (Number.isNaN(raw)) {
          throw new DataValidationError(`X contains NaN at index ${pos}`);
        }
        const p = Math.max(0, Math.min(1, normal ? normalCdf(raw) : raw));
        // For the normal output, probabilities within INVERSE_BOUNDS_THRESHOLD of 0 or 1 are
        // the clipped ends of the output range, so they map to the training minimum or maximum.
        if (normal && p < INVERSE_BOUNDS_THRESHOLD) {
          data[pos] = this.quantiles_[base] as number;
        } else if (normal && p > 1 - INVERSE_BOUNDS_THRESHOLD) {
          data[pos] = this.quantiles_[base + n - 1] as number;
        } else {
          data[pos] = probabilityToValue(p, this.quantiles_, base, n);
        }
      }
    }
    return makeMatrix(data, nSamples, nFeatures, X.device);
  }

  /** Constructor options of this transformer. */
  getParams(): Record<string, unknown> {
    const params: Record<string, unknown> = {
      nQuantiles: this.nQuantiles,
      outputDistribution: this.outputDistribution,
    };
    if (this.subsample !== undefined) params["subsample"] = this.subsample;
    if (this.randomState !== undefined) params["randomState"] = this.randomState;
    return params;
  }

  /**
   * Change options. Changing `nQuantiles`, `outputDistribution`, `subsample`
   * or `randomState` is only safe before fitting, so it discards the fitted
   * state.
   *
   * @throws {InvalidParameterError} If a recognised option is invalid
   */
  setParams(params: Record<string, unknown>): this {
    let changed = false;
    if (params["nQuantiles"] !== undefined) {
      const v = parseNQuantiles(params["nQuantiles"]);
      changed ||= v !== this.nQuantiles;
      this.nQuantiles = v;
    }
    if (params["outputDistribution"] !== undefined) {
      const v = parseOutputDistribution(params["outputDistribution"]);
      changed ||= v !== this.outputDistribution;
      this.outputDistribution = v;
    }
    if ("subsample" in params) {
      const v = parseSubsample(params["subsample"]);
      changed ||= v !== this.subsample;
      this.subsample = v;
    }
    if ("randomState" in params) {
      const v = params["randomState"];
      if (v !== undefined && typeof v !== "number") {
        throw new InvalidParameterError("randomState must be a number", "randomState", v);
      }
      changed ||= v !== this.randomState;
      this.randomState = v;
    }
    if (changed) {
      this.quantiles_ = undefined;
      this.references_ = undefined;
      this.nFeaturesIn_ = 0;
      this.nQuantilesFitted_ = 0;
    }
    return this;
  }

  /** Create an unfitted transformer with the same options. */
  clone(): QuantileTransformer {
    return new QuantileTransformer(
      this.getParams() as ConstructorParameters<typeof QuantileTransformer>[0]
    );
  }
}

function parseNQuantiles(value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 2) {
    throw new InvalidParameterError("nQuantiles must be an integer >= 2", "nQuantiles", value);
  }
  return value;
}

function parseOutputDistribution(value: unknown): "uniform" | "normal" {
  if (value !== "uniform" && value !== "normal") {
    throw new InvalidParameterError(
      "outputDistribution must be 'uniform' or 'normal'",
      "outputDistribution",
      value
    );
  }
  return value;
}

function parseSubsample(value: unknown): number | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "number" || !Number.isInteger(value) || value < 2) {
    throw new InvalidParameterError("subsample must be an integer >= 2", "subsample", value);
  }
  return value;
}

/**
 * Map a feature value to its empirical probability using the quantile table
 * `qs[base .. base + n)`. Equivalent to averaging a forward and a reversed
 * linear interpolation, so tied quantiles map to the mean of their references.
 */
function valueToProbability(
  value: number,
  qs: Float64Array,
  base: number,
  n: number,
  refs: Float64Array
): number {
  if (n === 1) return refs[0] as number;
  if (value <= (qs[base] as number)) return 0;
  if (value >= (qs[base + n - 1] as number)) return 1;

  // Invariant: qs[lo] < value <= qs[hi].
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >>> 1;
    if ((qs[base + mid] as number) < value) lo = mid;
    else hi = mid;
  }
  const qHi = qs[base + hi] as number;
  if (qHi === value) {
    // Find the last quantile equal to value (qs[n - 1] > value).
    let a = hi;
    let b = n - 1;
    while (b - a > 1) {
      const mid = (a + b) >>> 1;
      if ((qs[base + mid] as number) <= value) a = mid;
      else b = mid;
    }
    return 0.5 * ((refs[hi] as number) + (refs[a] as number));
  }
  const qLo = qs[base + lo] as number;
  const rLo = refs[lo] as number;
  return rLo + ((value - qLo) / (qHi - qLo)) * ((refs[hi] as number) - rLo);
}

/** Inverse of {@link valueToProbability} on the evenly spaced reference grid. */
function probabilityToValue(p: number, qs: Float64Array, base: number, n: number): number {
  if (n === 1) return qs[base] as number;
  const position = p * (n - 1);
  const lower = Math.min(Math.floor(position), n - 2);
  return lerp(qs[base + lower] as number, qs[base + lower + 1] as number, position - lower);
}

// ---------------------------------------------------------------------------
// PowerTransformer
// ---------------------------------------------------------------------------

function boxCox(x: number, lambda: number): number {
  if (lambda === 1) return x - 1;
  return Math.abs(lambda) < EPS ? Math.log(x) : Math.expm1(lambda * Math.log(x)) / lambda;
}

function yeoJohnson(x: number, lambda: number): number {
  if (lambda === 1) return x;
  if (x >= 0) {
    return Math.abs(lambda) < EPS ? Math.log1p(x) : Math.expm1(lambda * Math.log1p(x)) / lambda;
  }
  const mu = 2 - lambda;
  return Math.abs(mu) < EPS ? -Math.log1p(-x) : -Math.expm1(mu * Math.log1p(-x)) / mu;
}

function boxCoxInverse(y: number, lambda: number): number {
  if (lambda === 1) {
    if (y <= -1) {
      throw new InvalidParameterError("Box-Cox inverse encountered an invalid value", "X", y);
    }
    return y + 1;
  }
  if (Math.abs(lambda) < EPS) return Math.exp(y);
  const base = lambda * y + 1;
  if (base <= 0) {
    throw new InvalidParameterError("Box-Cox inverse encountered an invalid value", "X", y);
  }
  return Math.exp(Math.log1p(lambda * y) / lambda);
}

function yeoJohnsonInverse(y: number, lambda: number): number {
  if (lambda === 1) return y;
  if (y >= 0) {
    if (Math.abs(lambda) < EPS) return Math.expm1(y);
    if (lambda * y + 1 <= 0) {
      throw new InvalidParameterError("Yeo-Johnson inverse encountered an invalid value", "X", y);
    }
    return Math.expm1(Math.log1p(lambda * y) / lambda);
  }
  const mu = 2 - lambda;
  if (Math.abs(mu) < EPS) return -Math.expm1(-y);
  if (1 - mu * y <= 0) {
    throw new InvalidParameterError("Yeo-Johnson inverse encountered an invalid value", "X", y);
  }
  return -Math.expm1(Math.log1p(-mu * y) / mu);
}

/**
 * Apply a power transform to make data more Gaussian-like.
 *
 * Supports Box-Cox (strictly positive data) and Yeo-Johnson (any real data).
 * The exponent λ of each feature is chosen by maximum likelihood on the
 * training data, searching λ in [-5, 5] and widening the search up to [-160, 160]
 * when the optimum lies at the edge. With `standardize: true` the transformed
 * features are additionally centered and scaled to unit variance.
 *
 * **Difference from scikit-learn:** `standardize` defaults to `false` here, while
 * scikit-learn's `PowerTransformer` defaults to `True`. Pass `{ standardize: true }` to
 * match scikit-learn. The default is kept for compatibility with earlier Deepbox releases.
 *
 * **Fitted attributes** (read-only getters): `lambdas`, `mean`, `scale`,
 * `nFeaturesIn`.
 *
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox Scalers}
 */
export class PowerTransformer {
  private method: "box-cox" | "yeo-johnson";
  private standardize: boolean;
  private nFeaturesIn_ = 0;
  private lambdas_: Float64Array | undefined;
  private mean_: Float64Array | undefined;
  private scale_: Float64Array | undefined;

  /**
   * Creates a new PowerTransformer.
   *
   * @param options - Configuration options
   * @param options.method - "box-cox" or "yeo-johnson" (default: "yeo-johnson")
   * @param options.standardize - Whether to standardize transformed features (default: false;
   *   scikit-learn defaults to true, so pass `true` to match it)
   * @param options.copy - Accepted for API parity; transforms are always out-of-place (default: true)
   * @throws {InvalidParameterError} If an option is invalid
   */
  constructor(
    options: {
      method?: "box-cox" | "yeo-johnson";
      standardize?: boolean;
      copy?: boolean;
    } = {}
  ) {
    this.method = parsePowerMethod(options.method ?? "yeo-johnson");
    this.standardize = parseBooleanOption(options.standardize, "standardize", false);
    parseBooleanOption(options.copy, "copy", true);
  }

  /** Fitted exponent of each feature, or `undefined` if unfitted. */
  get lambdas(): Tensor | undefined {
    return this.lambdas_ ? makeVector(this.lambdas_) : undefined;
  }

  /** Mean of each transformed feature, or `undefined` if unfitted or `standardize` is false. */
  get mean(): Tensor | undefined {
    return this.standardize && this.mean_ ? makeVector(this.mean_) : undefined;
  }

  /** Standard deviation of each transformed feature (1 if constant), or `undefined` if unfitted or `standardize` is false. */
  get scale(): Tensor | undefined {
    return this.standardize && this.scale_ ? makeVector(this.scale_) : undefined;
  }

  /** Number of features seen during fitting (0 when unfitted). */
  get nFeaturesIn(): number {
    return this.nFeaturesIn_;
  }

  /**
   * Estimate the exponent of each feature by maximum likelihood.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (present for API compatibility)
   * @returns this
   * @throws {InvalidParameterError} If X has no samples, or for Box-Cox if X has values <= 0
   * @throws {ShapeError} If X is not 2-D
   * @throws {DataValidationError} If X contains NaN or Infinity
   */
  fit(X: Tensor, _y?: Tensor): this {
    const { data, nSamples, nFeatures } = readFitMatrix(X);
    const boxcox = this.method === "box-cox";

    const lambdas = new Float64Array(nFeatures);
    const means = new Float64Array(nFeatures);
    const scales = new Float64Array(nFeatures);
    const column = new Float64Array(nSamples);
    const transformed = new Float64Array(nSamples);

    for (let j = 0; j < nFeatures; j++) {
      for (let i = 0; i < nSamples; i++) {
        const value = data[i * nFeatures + j] as number;
        if (boxcox && value <= 0) {
          throw new InvalidParameterError(
            `Box-Cox requires strictly positive values in fit data (feature ${j})`,
            "X",
            value
          );
        }
        column[i] = value;
      }
      const lambda = optimizeLambda(column, boxcox, transformed);
      lambdas[j] = lambda;

      let maxAbs = 0;
      let low = Number.POSITIVE_INFINITY;
      let high = Number.NEGATIVE_INFINITY;
      for (let i = 0; i < nSamples; i++) {
        const t = boxcox
          ? boxCox(column[i] as number, lambda)
          : yeoJohnson(column[i] as number, lambda);
        transformed[i] = t;
        maxAbs = Math.max(maxAbs, Math.abs(t));
        if (t < low) low = t;
        if (t > high) high = t;
      }
      // Work relative to the largest magnitude so that huge outputs cannot overflow.
      const unit = maxAbs > 1e100 ? maxAbs : 1;
      let sum = 0;
      for (let i = 0; i < nSamples; i++) sum += (transformed[i] as number) / unit;
      // The clamp removes summation round-off, so a constant feature has an exact mean.
      const scaledMean = Math.min(Math.max(sum / nSamples, low / unit), high / unit);
      let sumSq = 0;
      for (let i = 0; i < nSamples; i++) {
        const d = (transformed[i] as number) / unit - scaledMean;
        sumSq += d * d;
      }
      const variance = sumSq / nSamples;
      // A variance within rounding noise of zero means a constant feature (as in StandardScaler).
      const noiseBound = nSamples * EPS * variance + (nSamples * scaledMean * EPS) ** 2;
      const std = variance <= noiseBound ? 0 : Math.sqrt(variance) * unit;
      means[j] = scaledMean * unit;
      scales[j] = std === 0 || !Number.isFinite(std) ? 1 : std;
    }

    this.lambdas_ = lambdas;
    this.mean_ = means;
    this.scale_ = scales;
    this.nFeaturesIn_ = nFeatures;
    return this;
  }

  /**
   * Apply the fitted power transform.
   *
   * @param X - Data of shape (n_samples, n_features)
   * @returns Float64 tensor of the same shape
   * @throws {NotFittedError} If the transformer has not been fitted
   * @throws {ShapeError} If the number of features differs from the fit
   * @throws {DataValidationError} If X contains NaN or Infinity
   * @throws {InvalidParameterError} For Box-Cox, if X has values <= 0
   */
  transform(X: Tensor): Tensor {
    if (!this.lambdas_ || !this.mean_ || !this.scale_) {
      throw new NotFittedError("PowerTransformer must be fitted before transform");
    }
    const { data, nSamples, nFeatures } = readApplyMatrix(
      X,
      this.nFeaturesIn_,
      "PowerTransformer",
      true
    );
    const boxcox = this.method === "box-cox";
    const standardize = this.standardize;

    let pos = 0;
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        const val = data[pos] as number;
        const lambda = this.lambdas_[j] as number;
        let t: number;
        if (boxcox) {
          if (val <= 0) {
            throw new InvalidParameterError("Box-Cox requires strictly positive values", "X", val);
          }
          t = boxCox(val, lambda);
        } else {
          t = yeoJohnson(val, lambda);
        }
        if (standardize) t = (t - (this.mean_[j] as number)) / (this.scale_[j] as number);
        data[pos++] = t;
      }
    }
    return makeMatrix(data, nSamples, nFeatures, X.device);
  }

  /**
   * Inverse transform data back to the original feature space.
   * If `standardize=true`, de-standardizes before applying the inverse power transform.
   *
   * @param X - Transformed data (2D tensor)
   * @returns Data in the original feature space
   * @throws {NotFittedError} If transformer is not fitted
   * @throws {ShapeError} If the number of features differs from the fit
   * @throws {InvalidParameterError} If a value is outside the range of the transform
   */
  inverseTransform(X: Tensor): Tensor {
    if (!this.lambdas_ || !this.mean_ || !this.scale_) {
      throw new NotFittedError("PowerTransformer must be fitted before inverseTransform");
    }
    const { data, nSamples, nFeatures } = readApplyMatrix(
      X,
      this.nFeaturesIn_,
      "PowerTransformer",
      false
    );
    const boxcox = this.method === "box-cox";
    const standardize = this.standardize;

    let pos = 0;
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        let val = data[pos] as number;
        if (standardize) val = val * (this.scale_[j] as number) + (this.mean_[j] as number);
        const lambda = this.lambdas_[j] as number;
        data[pos++] = boxcox ? boxCoxInverse(val, lambda) : yeoJohnsonInverse(val, lambda);
      }
    }
    return makeMatrix(data, nSamples, nFeatures, X.device);
  }

  /**
   * Fit to X, then transform it.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (present for API compatibility)
   * @returns Transformed data
   */
  fitTransform(X: Tensor, _y?: Tensor): Tensor {
    return this.fit(X).transform(X);
  }

  /** Constructor options of this transformer. */
  getParams(): Record<string, unknown> {
    return { method: this.method, standardize: this.standardize };
  }

  /**
   * Change options. Changing `method` discards the fitted state; changing
   * `standardize` keeps it.
   *
   * @throws {InvalidParameterError} If a recognised option is invalid
   */
  setParams(params: Record<string, unknown>): this {
    if (params["method"] !== undefined) {
      const method = parsePowerMethod(params["method"]);
      if (method !== this.method) {
        this.method = method;
        this.lambdas_ = undefined;
        this.mean_ = undefined;
        this.scale_ = undefined;
        this.nFeaturesIn_ = 0;
      }
    }
    if (params["standardize"] !== undefined) {
      this.standardize = parseBooleanOption(params["standardize"], "standardize", this.standardize);
    }
    return this;
  }

  /** Create an unfitted transformer with the same options. */
  clone(): PowerTransformer {
    return new PowerTransformer({ method: this.method, standardize: this.standardize });
  }
}

function parsePowerMethod(value: unknown): "box-cox" | "yeo-johnson" {
  if (value !== "box-cox" && value !== "yeo-johnson") {
    throw new InvalidParameterError("method must be 'box-cox' or 'yeo-johnson'", "method", value);
  }
  return value;
}

/** Smallest positive normal double; transformed variances below it are rejected. */
const TINY_VARIANCE = 2.2250738585072014e-308;

/** Transformed values above this (divided by the sample count) are treated as overflow. */
const OVERFLOW_GUARD = 1e300;

/**
 * Profile log-likelihood of a power transform with exponent `lambda`
 * (up to a constant). `jacobianSum` is the sum of log |dy/dx| terms that do
 * not depend on lambda, see {@link optimizeLambda}.
 */
function logLikelihood(
  values: Float64Array,
  lambda: number,
  boxcox: boolean,
  jacobianSum: number,
  scratch: Float64Array
): number {
  const n = values.length;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const v = values[i] as number;
    const t = boxcox ? boxCox(v, lambda) : yeoJohnson(v, lambda);
    // Reject exponents whose output is so large that its sum would overflow.
    if (!Number.isFinite(t) || Math.abs(t) > OVERFLOW_GUARD / n) return Number.NEGATIVE_INFINITY;
    scratch[i] = t;
    sum += t;
  }
  const mean = sum / n;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const d = (scratch[i] as number) - mean;
    sumSq += d * d;
  }
  const variance = sumSq / n;
  if (!Number.isFinite(variance)) {
    // For large lambda the transformed values are finite but their squares
    // overflow (typical for tightly clustered data far from zero).
    if (!boxcox || lambda <= 0) return Number.NEGATIVE_INFINITY;
    const logVariance = boxCoxLogVariance(values, lambda);
    if (!Number.isFinite(logVariance)) return Number.NEGATIVE_INFINITY;
    return -0.5 * n * logVariance + (lambda - 1) * jacobianSum;
  }
  if (variance < TINY_VARIANCE) return Number.NEGATIVE_INFINITY;
  return -0.5 * n * Math.log(variance) + (lambda - 1) * jacobianSum;
}

/**
 * log Var((x^lambda - 1) / lambda) for lambda > 0 without forming x^lambda,
 * by factoring out the largest power. Returns -Infinity for constant data.
 */
function boxCoxLogVariance(values: Float64Array, lambda: number): number {
  const n = values.length;
  let maxLog = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < n; i++) maxLog = Math.max(maxLog, lambda * Math.log(values[i] as number));
  let sum = 0;
  for (let i = 0; i < n; i++) sum += Math.exp(lambda * Math.log(values[i] as number) - maxLog);
  const mean = sum / n;
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    const d = Math.exp(lambda * Math.log(values[i] as number) - maxLog) - mean;
    sumSq += d * d;
  }
  const scaledVariance = sumSq / n;
  if (scaledVariance <= 0) return Number.NEGATIVE_INFINITY;
  return 2 * maxLog + Math.log(scaledVariance) - 2 * Math.log(lambda);
}

/** Golden-section maximisation of the log-likelihood over [lo, hi]. */
function goldenSearch(
  lo: number,
  hi: number,
  f: (lambda: number) => number
): { lambda: number; score: number } {
  const phi = (Math.sqrt(5) - 1) / 2;
  let left = lo;
  let right = hi;
  let c = right - phi * (right - left);
  let d = left + phi * (right - left);
  let fc = f(c);
  let fd = f(d);
  for (let iter = 0; iter < 100 && right - left > 1e-9; iter++) {
    // When both probes overflow, step back toward the finite region around 0.
    const bothOverflow = fc === Number.NEGATIVE_INFINITY && fd === Number.NEGATIVE_INFINITY;
    if (bothOverflow ? c + d > 0 : fc > fd) {
      right = d;
      d = c;
      fd = fc;
      c = right - phi * (right - left);
      fc = f(c);
    } else {
      left = c;
      c = d;
      fc = fd;
      d = left + phi * (right - left);
      fd = f(d);
    }
  }
  let best = { lambda: 1, score: Number.NEGATIVE_INFINITY };
  for (const lambda of [left, right, (left + right) / 2]) {
    const score = f(lambda);
    if (score > best.score) best = { lambda, score };
  }
  return best;
}

/**
 * Maximum-likelihood exponent for one feature. The log-likelihood is concave
 * in lambda for both transforms, so a bracketed golden-section search finds
 * the global optimum. Returns 1 (identity-like) for constant features.
 */
function optimizeLambda(values: Float64Array, boxcox: boolean, scratch: Float64Array): number {
  const n = values.length;
  if (n < 2) return 1;

  let minValue = Number.POSITIVE_INFINITY;
  let maxValue = Number.NEGATIVE_INFINITY;
  let jacobianSum = 0;
  for (let i = 0; i < n; i++) {
    const v = values[i] as number;
    if (v < minValue) minValue = v;
    if (v > maxValue) maxValue = v;
    // d/dx of the transform is x^(lambda-1) (Box-Cox) or (1+|x|)^(sign(x)(lambda-1)).
    jacobianSum += boxcox ? Math.log(v) : Math.sign(v) * Math.log1p(Math.abs(v));
  }
  if (minValue === maxValue) return 1;

  const f = (lambda: number): number => logLikelihood(values, lambda, boxcox, jacobianSum, scratch);
  let best = { lambda: 1, score: Number.NEGATIVE_INFINITY };
  for (const limit of [5, 10, 20, 40, 80, 160]) {
    best = goldenSearch(-limit, limit, f);
    if (Math.abs(best.lambda) < 0.99 * limit || !Number.isFinite(best.score)) break;
  }
  return Number.isFinite(best.score) ? best.lambda : 1;
}
