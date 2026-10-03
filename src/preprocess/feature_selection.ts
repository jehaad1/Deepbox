/**
 * Feature selection transformers.
 *
 * @module preprocess/feature_selection
 * @see {@link https://deepbox.dev/docs/preprocess-features | Deepbox Feature Selection}
 */

import {
  DataValidationError,
  DTypeError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../core/errors";
import { getEstimatorTags } from "../ml/base";
import { type DType, type Tensor, Tensor as TensorClass, type TypedArray } from "../ndarray";
import { assert2D, assertNumericTensor, getShape2D, getStrides2D } from "./_internal";

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/** Throw unless `params` only contains names from `known`. */
function assertKnownParams(
  params: Record<string, unknown>,
  known: readonly string[],
  who: string
): void {
  for (const key of Object.keys(params)) {
    if (!known.includes(key)) {
      throw new InvalidParameterError(
        `Invalid parameter '${key}' for ${who}; valid parameters are ${known.join(", ")}`,
        key,
        params[key]
      );
    }
  }
}

/** Drop `undefined` entries so they do not override existing values. */
function definedEntries(params: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) out[key] = value;
  }
  return out;
}

/**
 * Throw if any element of a 2D tensor is infinite, or NaN when `allowNaN` is false.
 * Reads through the strides, so views are handled without a copy.
 */
function assertFiniteMatrix(X: Tensor, who: string, allowNaN: boolean): void {
  const [nSamples, nFeatures] = getShape2D(X);
  const [rs, cs] = getStrides2D(X);
  const src = X.data as ArrayLike<number | bigint>;
  const base = X.offset;
  for (let i = 0; i < nSamples; i++) {
    const rowBase = base + i * rs;
    for (let j = 0; j < nFeatures; j++) {
      const v = Number(src[rowBase + j * cs]);
      if (Number.isFinite(v)) continue;
      if (Number.isNaN(v) && allowNaN) continue;
      throw new DataValidationError(
        allowNaN
          ? `${who}: X contains infinity (row ${i}, column ${j})`
          : `${who}: X contains NaN or infinity (row ${i}, column ${j})`
      );
    }
  }
}

/**
 * Read a 1D target (or an `[n, 1]` column) into a Float64Array.
 * Honors strides and rejects non-numeric dtypes and wrong lengths.
 */
function readTarget(y: Tensor, nSamples: number, who: string, finiteOnly: boolean): Float64Array {
  if (y.dtype === "string") {
    throw new DTypeError(`${who} requires a numeric target y`);
  }
  const isColumn = y.ndim === 2 && y.shape[1] === 1;
  if (y.ndim !== 1 && !isColumn) {
    throw new ShapeError(`y must be a 1D tensor, got shape [${y.shape.join(", ")}]`);
  }
  const length = y.shape[0] ?? 0;
  if (length !== nSamples) {
    throw new ShapeError(`X and y have inconsistent numbers of samples: ${nSamples} and ${length}`);
  }
  const stride = y.strides[0] ?? 1;
  const src = y.data as ArrayLike<number | bigint>;
  const out = new Float64Array(nSamples);
  for (let i = 0; i < nSamples; i++) {
    const v = Number(src[y.offset + i * stride]);
    if (finiteOnly && !Number.isFinite(v)) {
      throw new DataValidationError(`${who}: y contains NaN or infinity (index ${i})`);
    }
    out[i] = v;
  }
  return out;
}

/** Allocate an empty array of the same element type as `data`. */
function allocLike(data: unknown, length: number): { [index: number]: number | bigint } {
  const Ctor = (data as { constructor: new (length: number) => unknown }).constructor;
  return new Ctor(length) as { [index: number]: number | bigint };
}

/**
 * Gather columns of a 2D tensor into a new `[nSamples, cols.length]` tensor.
 * The dtype of the input is preserved, including string tensors.
 */
function gatherColumns(X: Tensor, cols: readonly number[]): Tensor {
  const [nSamples] = getShape2D(X);
  const [rs, cs] = getStrides2D(X);
  const nOut = cols.length;
  const base = X.offset;
  if (X.dtype === "string") {
    const src = X.data as string[];
    const out = new Array<string>(nSamples * nOut);
    let pos = 0;
    for (let i = 0; i < nSamples; i++) {
      const rowBase = base + i * rs;
      for (let c = 0; c < nOut; c++) out[pos++] = src[rowBase + (cols[c] as number) * cs] as string;
    }
    return TensorClass.fromStringArray({ data: out, shape: [nSamples, nOut], device: X.device });
  }
  const src = X.data as ArrayLike<number | bigint>;
  const out = allocLike(X.data, nSamples * nOut);
  let pos = 0;
  for (let i = 0; i < nSamples; i++) {
    const rowBase = base + i * rs;
    for (let c = 0; c < nOut; c++) {
      out[pos++] = src[rowBase + (cols[c] as number) * cs] as number | bigint;
    }
  }
  return TensorClass.fromTypedArray({
    data: out as unknown as TypedArray,
    shape: [nSamples, nOut],
    dtype: X.dtype as Exclude<DType, "string">,
    device: X.device,
  });
}

/**
 * Gather rows of a 1D or 2D tensor. The dtype is preserved. Used to build
 * cross-validation splits.
 */
function gatherRows(T: Tensor, rows: readonly number[]): Tensor {
  const nCols = T.ndim === 2 ? (T.shape[1] ?? 0) : 1;
  const rs = T.strides[0] ?? 1;
  const cs = T.ndim === 2 ? (T.strides[1] ?? 1) : 0;
  const shape: [number] | [number, number] = T.ndim === 2 ? [rows.length, nCols] : [rows.length];
  const base = T.offset;
  if (T.dtype === "string") {
    const src = T.data as string[];
    const out = new Array<string>(rows.length * nCols);
    let pos = 0;
    for (const r of rows) {
      for (let j = 0; j < nCols; j++) out[pos++] = src[base + r * rs + j * cs] as string;
    }
    return TensorClass.fromStringArray({ data: out, shape, device: T.device });
  }
  const src = T.data as ArrayLike<number | bigint>;
  const out = allocLike(T.data, rows.length * nCols);
  let pos = 0;
  for (const r of rows) {
    for (let j = 0; j < nCols; j++) out[pos++] = src[base + r * rs + j * cs] as number | bigint;
  }
  return TensorClass.fromTypedArray({
    data: out as unknown as TypedArray,
    shape,
    dtype: T.dtype as Exclude<DType, "string">,
    device: T.device,
  });
}

/**
 * Inverse of {@link gatherColumns}: place the columns of `Xt` at positions
 * `cols` of a `[nSamples, nFeaturesIn]` tensor and fill the rest with zeros.
 */
function scatterColumns(Xt: Tensor, cols: readonly number[], nFeaturesIn: number): Tensor {
  const [nSamples, nSel] = getShape2D(Xt);
  if (nSel !== cols.length) {
    throw new InvalidParameterError(`Expected ${cols.length} features, got ${nSel}`, "X", nSel);
  }
  const [rs, cs] = getStrides2D(Xt);
  const base = Xt.offset;
  if (Xt.dtype === "string") {
    const src = Xt.data as string[];
    const out = new Array<string>(nSamples * nFeaturesIn).fill("");
    for (let i = 0; i < nSamples; i++) {
      for (let c = 0; c < nSel; c++) {
        out[i * nFeaturesIn + (cols[c] as number)] = src[base + i * rs + c * cs] as string;
      }
    }
    return TensorClass.fromStringArray({
      data: out,
      shape: [nSamples, nFeaturesIn],
      device: Xt.device,
    });
  }
  const src = Xt.data as ArrayLike<number | bigint>;
  const out = allocLike(Xt.data, nSamples * nFeaturesIn);
  for (let i = 0; i < nSamples; i++) {
    for (let c = 0; c < nSel; c++) {
      out[i * nFeaturesIn + (cols[c] as number)] = src[base + i * rs + c * cs] as number | bigint;
    }
  }
  return TensorClass.fromTypedArray({
    data: out as unknown as TypedArray,
    shape: [nSamples, nFeaturesIn],
    dtype: Xt.dtype as Exclude<DType, "string">,
    device: Xt.device,
  });
}

function maskToIndices(mask: readonly boolean[]): number[] {
  const cols: number[] = [];
  for (let j = 0; j < mask.length; j++) {
    if (mask[j]) cols.push(j);
  }
  return cols;
}

/** Validate a 2D input against the feature count seen during fit. */
function assertFeatureCount(X: Tensor, nFeaturesIn: number): void {
  assert2D(X, "X");
  const [, nFeatures] = getShape2D(X);
  if (nFeatures !== nFeaturesIn) {
    throw new InvalidParameterError(
      `Expected ${nFeaturesIn} features, got ${nFeatures}`,
      "X",
      nFeatures
    );
  }
}

/** Support mask as booleans, or as the indices of the selected features. */
function supportOf(mask: readonly boolean[], indices: boolean): boolean[] | number[] {
  return indices ? maskToIndices(mask) : [...mask];
}

/** Indices that sort `values` ascending (stable, NaN last). */
function argsortAscending(values: readonly number[]): number[] {
  const idx = values.map((_, i) => i);
  idx.sort((a, b) => {
    const va = values[a] as number;
    const vb = values[b] as number;
    const na = Number.isNaN(va);
    const nb = Number.isNaN(vb);
    if (na || nb) return na === nb ? 0 : na ? 1 : -1;
    return va - vb;
  });
  return idx;
}

/** Mask of the `k` highest scores. NaN ranks lowest; ties keep the lowest index. */
function selectTopK(scores: readonly number[], k: number): boolean[] {
  const idx = scores.map((_, i) => i);
  idx.sort((a, b) => {
    const va = scores[a] as number;
    const vb = scores[b] as number;
    const na = Number.isNaN(va);
    const nb = Number.isNaN(vb);
    if (na || nb) return na === nb ? 0 : na ? 1 : -1;
    return vb - va;
  });
  const mask = new Array<boolean>(scores.length).fill(false);
  for (let i = 0; i < k; i++) mask[idx[i] as number] = true;
  return mask;
}

/** Fresh unfitted copy of an estimator, or the estimator itself when it cannot be cloned. */
function cloneEstimator<T extends object>(estimator: T): T {
  const withClone = estimator as { clone?: () => T; getParams?: () => Record<string, unknown> };
  if (typeof withClone.clone === "function") {
    return withClone.clone();
  }
  if (typeof withClone.getParams === "function") {
    const Ctor = estimator.constructor as new (params: Record<string, unknown>) => T;
    try {
      return new Ctor(withClone.getParams());
    } catch {
      return estimator;
    }
  }
  return estimator;
}

// ---------------------------------------------------------------------------
// VarianceThreshold
// ---------------------------------------------------------------------------

/**
 * Feature selector that removes all low-variance features.
 *
 * Features with a variance lower than or equal to the threshold are removed.
 * By default only zero-variance (constant) features are removed. Variances are
 * the population variances (divisor n), computed with NaN values ignored.
 * Constant columns always get a variance of exactly 0, so the default threshold
 * is not defeated by rounding error in the column mean.
 *
 * `transform` keeps the dtype of the input.
 *
 * @example
 * ```ts
 * import { VarianceThreshold } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1]]);
 * const selector = new VarianceThreshold({ threshold: 0.0 });
 * selector.fit(X);
 * const Xt = selector.transform(X); // removes constant columns
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-features | Deepbox Feature Selection}
 */
export class VarianceThreshold {
  private threshold: number;
  private variances_: number[] | undefined;
  private mask_: boolean[] | undefined;
  private nFeaturesIn_: number | undefined;

  /**
   * @param options.threshold - Features with variance <= threshold are removed (default 0)
   */
  constructor(options: { threshold?: number } = {}) {
    this.threshold = options.threshold ?? 0.0;
    if (!Number.isFinite(this.threshold) || this.threshold < 0) {
      throw new InvalidParameterError(
        "threshold must be a non-negative number",
        "threshold",
        this.threshold
      );
    }
  }

  /**
   * Learn the per-feature variances.
   *
   * @param X - Data of shape (n_samples, n_features). NaN values are ignored.
   * @throws {DTypeError} If X is a string tensor
   * @throws {DataValidationError} If X contains infinity
   */
  fit(X: Tensor, _y?: Tensor): this {
    if (X.dtype === "string") {
      throw new DTypeError("VarianceThreshold requires numeric input");
    }
    assertNumericTensor(X, "X");
    assert2D(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const [rowStride, colStride] = getStrides2D(X);

    if (nSamples === 0) {
      throw new InvalidParameterError("Cannot fit on empty array", "X", nSamples);
    }
    assertFiniteMatrix(X, "VarianceThreshold", true);

    const src = X.data as ArrayLike<number | bigint>;
    const base = X.offset;
    const variances: number[] = [];

    for (let j = 0; j < nFeatures; j++) {
      let count = 0;
      let sum = 0;
      let min = Number.POSITIVE_INFINITY;
      let max = Number.NEGATIVE_INFINITY;
      for (let i = 0; i < nSamples; i++) {
        const v = Number(src[base + i * rowStride + j * colStride]);
        if (Number.isNaN(v)) continue;
        count++;
        sum += v;
        if (v < min) min = v;
        if (v > max) max = v;
      }
      if (count === 0) {
        variances.push(Number.NaN);
        continue;
      }
      if (min === max) {
        variances.push(0);
        continue;
      }
      // Corrected two-pass algorithm: avoids the cancellation of E[x^2] - E[x]^2.
      const mean = sum / count;
      let s1 = 0;
      let s2 = 0;
      for (let i = 0; i < nSamples; i++) {
        const v = Number(src[base + i * rowStride + j * colStride]);
        if (Number.isNaN(v)) continue;
        const d = v - mean;
        s1 += d;
        s2 += d * d;
      }
      variances.push(Math.max(0, (s2 - (s1 * s1) / count) / count));
    }

    this.variances_ = variances;
    this.mask_ = variances.map((v) => v > this.threshold);
    this.nFeaturesIn_ = nFeatures;

    return this;
  }

  /**
   * Keep only the features whose variance exceeds the threshold.
   *
   * @param X - Data of shape (n_samples, n_features)
   * @returns Tensor of shape (n_samples, n_selected) with the same dtype as X
   */
  transform(X: Tensor): Tensor {
    if (!this.mask_ || !this.variances_ || this.nFeaturesIn_ === undefined) {
      throw new NotFittedError("VarianceThreshold must be fitted before transform");
    }
    if (X.dtype === "string") {
      throw new DTypeError("VarianceThreshold requires numeric input");
    }
    assertFeatureCount(X, this.nFeaturesIn_);
    return gatherColumns(X, maskToIndices(this.mask_));
  }

  fitTransform(X: Tensor, y?: Tensor): Tensor {
    this.fit(X, y);
    return this.transform(X);
  }

  /**
   * Map reduced data back to the original number of features. Removed
   * features are filled with zeros.
   */
  inverseTransform(X: Tensor): Tensor {
    if (!this.mask_ || this.nFeaturesIn_ === undefined) {
      throw new NotFittedError("VarianceThreshold must be fitted before inverseTransform");
    }
    assert2D(X, "X");
    return scatterColumns(X, maskToIndices(this.mask_), this.nFeaturesIn_);
  }

  /**
   * Get the selection mask (true = kept), or the indices of the kept features
   * when `indices` is true.
   */
  getSupport(): boolean[];
  getSupport(indices: false): boolean[];
  getSupport(indices: true): number[];
  getSupport(indices?: boolean): boolean[] | number[];
  getSupport(indices = false): boolean[] | number[] {
    if (!this.mask_) {
      throw new NotFittedError("VarianceThreshold must be fitted before getSupport");
    }
    return supportOf(this.mask_, indices);
  }

  /** Per-feature variances (NaN ignored; NaN for an all-NaN column). */
  get variances(): number[] {
    if (!this.variances_) {
      throw new NotFittedError("VarianceThreshold must be fitted before accessing variances");
    }
    return [...this.variances_];
  }

  /** Number of features seen during fit. */
  get nFeaturesIn(): number {
    if (this.nFeaturesIn_ === undefined) {
      throw new NotFittedError("VarianceThreshold must be fitted before accessing nFeaturesIn");
    }
    return this.nFeaturesIn_;
  }

  getParams(): Record<string, unknown> {
    return { threshold: this.threshold };
  }

  /**
   * Update parameters. A new threshold is applied to the fitted variances
   * immediately, without refitting.
   */
  setParams(params: Record<string, unknown>): this {
    assertKnownParams(params, ["threshold"], "VarianceThreshold");
    const next = new VarianceThreshold({
      ...this.getParams(),
      ...definedEntries(params),
    } as { threshold?: number });
    this.threshold = next.threshold;
    if (this.variances_) {
      this.mask_ = this.variances_.map((v) => v > this.threshold);
    }
    return this;
  }

  /** Create an unfitted copy with the same parameters. */
  clone(): VarianceThreshold {
    return new VarianceThreshold({ threshold: this.threshold });
  }
}

// ---------------------------------------------------------------------------
// Scoring functions
// ---------------------------------------------------------------------------

/** Scoring function type for SelectKBest. Returns one score per feature. */
export type ScoreFunc = (X: Tensor, y: Tensor) => number[];

function prepareScoringInput(
  X: Tensor,
  y: Tensor,
  who: string
): { nSamples: number; nFeatures: number; yv: Float64Array } {
  if (X.dtype === "string") {
    throw new DTypeError(`${who} requires numeric features`);
  }
  assertNumericTensor(X, "X");
  assert2D(X, "X");
  const [nSamples, nFeatures] = getShape2D(X);
  if (nSamples === 0) {
    throw new InvalidParameterError(`${who} requires at least one sample`, "X", nSamples);
  }
  const yv = readTarget(y, nSamples, who, true);
  assertFiniteMatrix(X, who, false);
  return { nSamples, nFeatures, yv };
}

/**
 * ANOVA F-value between each feature and the target classes.
 *
 * Returns one F statistic per feature, `MSB / MSW`, where the class labels in
 * `y` define the groups. Same statistic as `sklearn.feature_selection.f_classif`
 * (without the p-values). Constant features and targets with a single class
 * score 0; an exactly zero within-class variance with separated class means
 * scores `Infinity`.
 *
 * @param X - Features of shape (n_samples, n_features); must be finite
 * @param y - Class labels of shape (n_samples,)
 * @returns F statistics, one per feature
 * @throws {ShapeError} If X is not 2D or y does not match X
 * @throws {DataValidationError} If X or y contain NaN or infinity
 *
 * @deprecated Prefer {@link fClassif}.
 */
export function f_classif(X: Tensor, y: Tensor): number[] {
  const { nSamples, nFeatures, yv } = prepareScoringInput(X, y, "f_classif");
  const [rowStride, colStride] = getStrides2D(X);
  const src = X.data as ArrayLike<number | bigint>;
  const base = X.offset;

  const classMap = new Map<number, number[]>();
  for (let i = 0; i < nSamples; i++) {
    const label = yv[i] as number;
    let arr = classMap.get(label);
    if (!arr) {
      arr = [];
      classMap.set(label, arr);
    }
    arr.push(i);
  }
  const classes = [...classMap.values()];
  const nClasses = classes.length;
  const dfB = nClasses - 1;
  const dfW = nSamples - nClasses;

  const scores: number[] = [];
  for (let j = 0; j < nFeatures; j++) {
    let grandSum = 0;
    for (let i = 0; i < nSamples; i++)
      grandSum += Number(src[base + i * rowStride + j * colStride]);
    const grandMean = grandSum / nSamples;
    let ssBetween = 0;
    let ssWithin = 0;
    for (const members of classes) {
      let gs = 0;
      for (const idx of members) gs += Number(src[base + idx * rowStride + j * colStride]);
      const gm = gs / members.length;
      ssBetween += members.length * (gm - grandMean) ** 2;
      for (const idx of members) {
        const d = Number(src[base + idx * rowStride + j * colStride]) - gm;
        ssWithin += d * d;
      }
    }
    if (dfB <= 0) {
      scores.push(0);
    } else if (dfW <= 0 || ssWithin === 0) {
      scores.push(ssBetween > 0 ? Number.POSITIVE_INFINITY : 0);
    } else {
      scores.push(ssBetween / dfB / (ssWithin / dfW));
    }
  }
  return scores;
}

/**
 * Univariate F statistic of the linear regression of `y` on each feature.
 *
 * For each feature the Pearson correlation `r` with `y` is turned into
 * `F = r^2 / (1 - r^2) * (n - 2)`, as in `sklearn.feature_selection.f_regression`
 * with `center=True` (p-values are not returned). Constant features, a constant
 * target and fewer than three samples score 0; a perfect correlation scores
 * `Infinity`.
 *
 * @param X - Features of shape (n_samples, n_features); must be finite
 * @param y - Continuous target of shape (n_samples,)
 * @returns F statistics, one per feature
 * @throws {ShapeError} If X is not 2D or y does not match X
 * @throws {DataValidationError} If X or y contain NaN or infinity
 *
 * @deprecated Prefer {@link fRegression}.
 */
export function f_regression(X: Tensor, y: Tensor): number[] {
  const { nSamples, nFeatures, yv } = prepareScoringInput(X, y, "f_regression");
  const [rowStride, colStride] = getStrides2D(X);
  const src = X.data as ArrayLike<number | bigint>;
  const base = X.offset;

  let ySum = 0;
  for (let i = 0; i < nSamples; i++) ySum += yv[i] as number;
  const yMean = ySum / nSamples;
  const yc = new Float64Array(nSamples);
  let ySS = 0;
  for (let i = 0; i < nSamples; i++) {
    const v = (yv[i] as number) - yMean;
    yc[i] = v;
    ySS += v * v;
  }
  const dfDen = nSamples - 2;

  const scores: number[] = [];
  for (let j = 0; j < nFeatures; j++) {
    let xSum = 0;
    for (let i = 0; i < nSamples; i++) xSum += Number(src[base + i * rowStride + j * colStride]);
    const xMean = xSum / nSamples;
    let xSS = 0;
    let xyCov = 0;
    for (let i = 0; i < nSamples; i++) {
      const xv = Number(src[base + i * rowStride + j * colStride]) - xMean;
      xSS += xv * xv;
      xyCov += xv * (yc[i] as number);
    }
    if (xSS === 0 || ySS === 0) {
      scores.push(0);
      continue;
    }
    const r2 = Math.min(1, (xyCov * xyCov) / (xSS * ySS));
    if (r2 >= 1) {
      scores.push(Number.POSITIVE_INFINITY);
    } else {
      scores.push(dfDen <= 0 ? 0 : (r2 / (1 - r2)) * dfDen);
    }
  }
  return scores;
}

// ---------------------------------------------------------------------------
// SelectKBest
// ---------------------------------------------------------------------------

/**
 * Select features according to the k highest scores.
 *
 * Features with NaN scores rank last. Among equal scores the feature with the
 * lowest index is kept first. `transform` keeps the dtype of the input.
 *
 * @example
 * ```ts
 * import { SelectKBest, fClassif } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]);
 * const y = tensor([0,0,1,1]);
 * const skb = new SelectKBest({ scoreFunc: fClassif, k: 2 });
 * skb.fit(X, y);
 * const Xt = skb.transform(X);
 * ```
 */
export class SelectKBest {
  private k: number | "all";
  private scoreFunc: ScoreFunc;
  private scores_: number[] | undefined;
  private mask_: boolean[] | undefined;
  private nFeaturesIn_: number | undefined;

  /**
   * @param options.scoreFunc - Function returning one score per feature (default: fClassif)
   * @param options.k - Number of features to keep, or "all" (default 10)
   */
  constructor(options: { scoreFunc?: ScoreFunc; k?: number | "all" } = {}) {
    this.k = options.k ?? 10;
    this.scoreFunc = options.scoreFunc ?? f_classif;
    if (this.k !== "all" && (!Number.isInteger(this.k) || this.k < 1)) {
      throw new InvalidParameterError('k must be a positive integer or "all"', "k", this.k);
    }
    if (typeof this.scoreFunc !== "function") {
      throw new InvalidParameterError("scoreFunc must be a function", "scoreFunc", this.scoreFunc);
    }
  }

  private resolveK(nFeatures: number): number {
    const k = this.k === "all" ? nFeatures : this.k;
    if (k > nFeatures) {
      throw new InvalidParameterError(
        `k=${k} exceeds number of features (${nFeatures})`,
        "k",
        this.k
      );
    }
    return k;
  }

  /**
   * Score every feature and select the k best.
   *
   * @throws {InvalidParameterError} If k exceeds the number of features
   */
  fit(X: Tensor, y: Tensor): this {
    if (X.dtype === "string") throw new DTypeError("SelectKBest requires numeric input");
    assert2D(X, "X");
    const [, nFeatures] = getShape2D(X);
    const k = this.resolveK(nFeatures);
    const scores = this.scoreFunc(X, y);
    if (!Array.isArray(scores) || scores.length !== nFeatures) {
      throw new InvalidParameterError(
        `scoreFunc must return an array with one score per feature (${nFeatures}); got ${
          Array.isArray(scores) ? scores.length : typeof scores
        }`,
        "scoreFunc"
      );
    }
    this.scores_ = [...scores];
    this.mask_ = selectTopK(scores, k);
    this.nFeaturesIn_ = nFeatures;
    return this;
  }

  /**
   * Keep only the selected features.
   *
   * @returns Tensor of shape (n_samples, k) with the same dtype as X
   */
  transform(X: Tensor): Tensor {
    if (!this.mask_ || this.nFeaturesIn_ === undefined) {
      throw new NotFittedError("SelectKBest must be fitted before transform");
    }
    if (X.dtype === "string") throw new DTypeError("SelectKBest requires numeric input");
    assertFeatureCount(X, this.nFeaturesIn_);
    return gatherColumns(X, maskToIndices(this.mask_));
  }

  fitTransform(X: Tensor, y: Tensor): Tensor {
    this.fit(X, y);
    return this.transform(X);
  }

  /** Map reduced data back to the original features; dropped features become zeros. */
  inverseTransform(X: Tensor): Tensor {
    if (!this.mask_ || this.nFeaturesIn_ === undefined) {
      throw new NotFittedError("SelectKBest must be fitted before inverseTransform");
    }
    assert2D(X, "X");
    return scatterColumns(X, maskToIndices(this.mask_), this.nFeaturesIn_);
  }

  /** Selection mask (true = kept), or the kept indices when `indices` is true. */
  getSupport(): boolean[];
  getSupport(indices: false): boolean[];
  getSupport(indices: true): number[];
  getSupport(indices?: boolean): boolean[] | number[];
  getSupport(indices = false): boolean[] | number[] {
    if (!this.mask_) throw new NotFittedError("SelectKBest must be fitted before getSupport");
    return supportOf(this.mask_, indices);
  }

  /** Feature scores computed during fit (NaN scores are kept as returned). */
  get scores(): number[] {
    if (!this.scores_)
      throw new NotFittedError("SelectKBest must be fitted before accessing scores");
    return [...this.scores_];
  }

  /** Number of features seen during fit. */
  get nFeaturesIn(): number {
    if (this.nFeaturesIn_ === undefined) {
      throw new NotFittedError("SelectKBest must be fitted before accessing nFeaturesIn");
    }
    return this.nFeaturesIn_;
  }

  getParams(): Record<string, unknown> {
    return { k: this.k, scoreFunc: this.scoreFunc };
  }

  /**
   * Update parameters. A new `k` is applied to the fitted scores immediately;
   * a new `scoreFunc` discards the fitted state.
   */
  setParams(params: Record<string, unknown>): this {
    assertKnownParams(params, ["k", "scoreFunc"], "SelectKBest");
    const next = new SelectKBest({
      ...this.getParams(),
      ...definedEntries(params),
    } as { scoreFunc?: ScoreFunc; k?: number | "all" });
    const funcChanged = next.scoreFunc !== this.scoreFunc;
    if (!funcChanged && this.scores_ && this.nFeaturesIn_ !== undefined) {
      // Validate k against the fitted feature count before changing any state.
      const k = next.k === "all" ? this.nFeaturesIn_ : next.k;
      if (k > this.nFeaturesIn_) {
        throw new InvalidParameterError(
          `k=${k} exceeds number of features (${this.nFeaturesIn_})`,
          "k",
          next.k
        );
      }
      this.mask_ = selectTopK(this.scores_, k);
    }
    this.k = next.k;
    this.scoreFunc = next.scoreFunc;
    if (funcChanged) {
      this.scores_ = undefined;
      this.mask_ = undefined;
      this.nFeaturesIn_ = undefined;
    }
    return this;
  }

  /** Create an unfitted copy with the same parameters. */
  clone(): SelectKBest {
    return new SelectKBest({ k: this.k, scoreFunc: this.scoreFunc });
  }
}

// ---------------------------------------------------------------------------
// Importance extraction (SelectFromModel / RFE / RFECV)
// ---------------------------------------------------------------------------

/**
 * Estimator interface expected by SelectFromModel, RFE and RFECV.
 *
 * After `fit(X, y)` the estimator must expose one of:
 * - `coef` / `coef_`: a Tensor, typed array or array of shape `[n_features]`
 *   or `[n_targets, n_features]` (linear and SVM models)
 * - `featureImportances` / `featureImportances_`: values of shape `[n_features]`
 *   (tree-based models)
 */
export interface ImportanceEstimator {
  fit(X: Tensor, y: Tensor): unknown;
}

/** Estimator with a `predict` method, required by {@link RFECV} for scoring. */
export interface ScoringEstimator extends ImportanceEstimator {
  predict(X: Tensor): Tensor;
}

type NumericBlock = { values: Float64Array; shape: number[] };

function readNumericBlock(value: unknown): NumericBlock | undefined {
  if (value instanceof TensorClass) {
    if (value.dtype === "string") return undefined;
    const shape = [...value.shape];
    const size = shape.reduce((a, b) => a * b, 1);
    const values = new Float64Array(size);
    const src = value.data as ArrayLike<number | bigint>;
    const strides = value.strides;
    const nd = shape.length;
    const counter = new Array<number>(nd).fill(0);
    for (let p = 0; p < size; p++) {
      let idx = value.offset;
      for (let a = 0; a < nd; a++) idx += (counter[a] as number) * (strides[a] as number);
      values[p] = Number(src[idx]);
      for (let a = nd - 1; a >= 0; a--) {
        counter[a] = (counter[a] as number) + 1;
        if ((counter[a] as number) < (shape[a] as number)) break;
        counter[a] = 0;
      }
    }
    return { values, shape };
  }
  if (Array.isArray(value)) {
    if (value.length > 0 && Array.isArray(value[0])) {
      const rows = value as unknown[][];
      const nCols = rows[0]?.length ?? 0;
      const values = new Float64Array(rows.length * nCols);
      for (let r = 0; r < rows.length; r++) {
        const row = rows[r] as unknown[];
        if (row.length !== nCols) return undefined;
        for (let c = 0; c < nCols; c++) values[r * nCols + c] = Number(row[c]);
      }
      return { values, shape: [rows.length, nCols] };
    }
    return {
      values: Float64Array.from(value as ArrayLike<unknown>, Number),
      shape: [value.length],
    };
  }
  if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
    const arr = value as unknown as ArrayLike<number | bigint>;
    return { values: Float64Array.from(arr, Number), shape: [arr.length] };
  }
  return undefined;
}

/**
 * Read per-feature importances from a fitted estimator.
 *
 * Linear models use `coef`: `|coef|` for 1D, and for `[n_targets, n_features]`
 * the sum over targets of `|coef|` (`"l1"`, used by SelectFromModel) or of
 * `coef^2` (`"l2"`, used by RFE). Tree models use `featureImportances` as is.
 */
function extractImportances(
  estimator: ImportanceEstimator,
  nFeatures: number,
  norm: "l1" | "l2"
): number[] {
  const rec = estimator as unknown as Record<string, unknown>;
  for (const name of ["coef_", "coef"]) {
    const raw = rec[name];
    if (raw === undefined || raw === null) continue;
    const block = readNumericBlock(raw);
    if (!block) {
      throw new InvalidParameterError(`Estimator ${name} must be numeric`, "estimator");
    }
    const { values, shape } = block;
    const out = new Array<number>(nFeatures).fill(0);
    const term = (v: number): number => (norm === "l1" ? Math.abs(v) : v * v);
    if (shape.length === 1) {
      const len = shape[0] as number;
      if (len === nFeatures) {
        for (let j = 0; j < nFeatures; j++) out[j] = term(values[j] as number);
      } else if (nFeatures > 0 && len > nFeatures && len % nFeatures === 0) {
        // Row-major `[n_targets, n_features]` flattened into one array (SGDClassifier).
        for (let t = 0; t < len / nFeatures; t++) {
          for (let j = 0; j < nFeatures; j++)
            out[j] = (out[j] as number) + term(values[t * nFeatures + j] as number);
        }
      } else {
        throw new InvalidParameterError(
          `Estimator ${name} has ${len} values but X has ${nFeatures} features`,
          "estimator"
        );
      }
    } else if (shape.length === 2) {
      const [r, c] = shape as [number, number];
      if (c === nFeatures) {
        for (let t = 0; t < r; t++) {
          for (let j = 0; j < nFeatures; j++)
            out[j] = (out[j] as number) + term(values[t * c + j] as number);
        }
      } else if (r === nFeatures) {
        for (let j = 0; j < nFeatures; j++) {
          for (let t = 0; t < c; t++)
            out[j] = (out[j] as number) + term(values[j * c + t] as number);
        }
      } else {
        throw new InvalidParameterError(
          `Estimator ${name} has shape [${r}, ${c}], which does not match ${nFeatures} features`,
          "estimator"
        );
      }
    } else {
      throw new InvalidParameterError(`Estimator ${name} must be 1D or 2D`, "estimator");
    }
    assertFiniteImportances(out);
    return out;
  }
  for (const name of ["featureImportances_", "featureImportances"]) {
    const raw = rec[name];
    if (raw === undefined || raw === null) continue;
    const block = readNumericBlock(raw);
    if (block === undefined || block.shape.length !== 1) {
      throw new InvalidParameterError(`Estimator ${name} must be a 1D numeric array`, "estimator");
    }
    if (block.values.length !== nFeatures) {
      throw new InvalidParameterError(
        `Estimator ${name} has ${block.values.length} values but X has ${nFeatures} features`,
        "estimator"
      );
    }
    const out = Array.from(block.values);
    assertFiniteImportances(out);
    return out;
  }
  throw new InvalidParameterError(
    "Estimator must expose coef (or coef_) or featureImportances (or featureImportances_) after fitting",
    "estimator"
  );
}

function assertFiniteImportances(values: readonly number[]): void {
  for (const v of values) {
    if (Number.isNaN(v)) {
      throw new DataValidationError("Estimator feature importances contain NaN");
    }
  }
}

function resolveStep(step: number, nFeatures: number): number {
  return step > 0 && step < 1 ? Math.max(1, Math.floor(step * nFeatures)) : step;
}

function validateStep(step: number): void {
  const isInteger = Number.isInteger(step) && step >= 1;
  const isFraction = Number.isFinite(step) && step > 0 && step < 1;
  if (!isInteger && !isFraction) {
    throw new InvalidParameterError(
      "step must be a positive integer or a fraction in (0, 1)",
      "step",
      step
    );
  }
}

/**
 * Recursive feature elimination down to `nTarget` features.
 *
 * Returns the ranking (1 = kept; features dropped in the first round get the
 * largest rank) and the number of elimination rounds.
 */
function runElimination(
  estimator: ImportanceEstimator,
  X: Tensor,
  y: Tensor,
  nTarget: number,
  step: number
): number[] {
  const [, nFeatures] = getShape2D(X);
  let active = Array.from({ length: nFeatures }, (_, i) => i);
  const rounds: number[][] = [];
  let currentX = X;

  while (active.length > nTarget) {
    estimator.fit(currentX, y);
    const importances = extractImportances(estimator, active.length, "l2");
    const nRemove = Math.min(step, active.length - nTarget);
    const order = argsortAscending(importances);
    const removeLocal = new Set<number>(order.slice(0, nRemove));
    rounds.push([...removeLocal].map((local) => active[local] as number));
    active = active.filter((_, local) => !removeLocal.has(local));
    currentX = gatherColumns(X, active);
  }

  const ranking = new Array<number>(nFeatures).fill(1);
  const total = rounds.length;
  for (let r = 0; r < total; r++) {
    for (const idx of rounds[r] as number[]) ranking[idx] = total - r + 1;
  }
  return ranking;
}

// ---------------------------------------------------------------------------
// SelectFromModel
// ---------------------------------------------------------------------------

/**
 * Meta-transformer for selecting features based on importance weights
 * from a fitted estimator.
 *
 * The estimator is cloned and fitted on the data; the user's instance is not
 * modified when it can be cloned (it has `clone()` or a `getParams()` that its
 * constructor accepts). The fitted copy is available as `fittedEstimator`.
 *
 * Importances come from `coef`/`coef_` (absolute values; for multi-target
 * coefficients the sum of absolute values over targets, like scikit-learn) or
 * from `featureImportances`/`featureImportances_`. A feature is kept when its
 * importance is greater than or equal to the threshold, and at most
 * `maxFeatures` features with the largest importances are kept.
 *
 * @example
 * ```ts
 * import { SelectFromModel } from 'deepbox/preprocess';
 * import { RandomForestClassifier } from 'deepbox/ml';
 *
 * const selector = new SelectFromModel({
 *   estimator: new RandomForestClassifier(),
 *   threshold: 'mean',
 * });
 * selector.fit(X, y);
 * const Xt = selector.transform(X);
 * ```
 */
export class SelectFromModel {
  private estimator: ImportanceEstimator;
  private threshold: "mean" | "median" | number;
  private maxFeatures: number | undefined;
  private fitted_: ImportanceEstimator | undefined;
  private importances_: number[] | undefined;
  private mask_: boolean[] | undefined;
  private nFeaturesIn_: number | undefined;

  /**
   * @param options.estimator - Estimator exposing coefficients or feature importances after fit
   * @param options.threshold - "mean" (default), "median" or a non-negative number
   * @param options.maxFeatures - Upper bound on the number of selected features
   */
  constructor(options: {
    estimator: ImportanceEstimator;
    threshold?: "mean" | "median" | number;
    maxFeatures?: number;
  }) {
    if (!options || typeof options.estimator?.fit !== "function") {
      throw new InvalidParameterError(
        "estimator must be an object with a fit(X, y) method",
        "estimator"
      );
    }
    this.estimator = options.estimator;
    this.threshold = options.threshold ?? "mean";
    this.maxFeatures = options.maxFeatures;
    if (
      this.threshold !== "mean" &&
      this.threshold !== "median" &&
      (typeof this.threshold !== "number" || !Number.isFinite(this.threshold) || this.threshold < 0)
    ) {
      throw new InvalidParameterError(
        'threshold must be "mean", "median" or a non-negative number',
        "threshold",
        this.threshold
      );
    }
    if (
      this.maxFeatures !== undefined &&
      (!Number.isInteger(this.maxFeatures) || this.maxFeatures < 1)
    ) {
      throw new InvalidParameterError(
        "maxFeatures must be a positive integer",
        "maxFeatures",
        this.maxFeatures
      );
    }
  }

  private computeMask(importances: readonly number[]): boolean[] {
    const n = importances.length;
    let thresh: number;
    if (this.threshold === "mean") {
      // The mean lies between the extremes; clamping removes the rounding error that would
      // otherwise push it above the importances when all of them are equal (3 x 0.1).
      let sum = 0;
      let max = Number.NEGATIVE_INFINITY;
      for (const imp of importances) {
        sum += imp;
        if (imp > max) max = imp;
      }
      thresh = n === 0 ? 0 : Math.min(sum / n, max);
    } else if (this.threshold === "median") {
      const sorted = [...importances].sort((a, b) => a - b);
      const mid = Math.floor(n / 2);
      thresh =
        n === 0
          ? 0
          : n % 2 === 0
            ? ((sorted[mid - 1] as number) + (sorted[mid] as number)) / 2
            : (sorted[mid] as number);
    } else {
      thresh = this.threshold;
    }

    const mask = importances.map((imp) => imp >= thresh);
    if (this.maxFeatures !== undefined) {
      const candidates = maskToIndices(mask);
      if (candidates.length > this.maxFeatures) {
        const order = candidates.sort((a, b) => {
          const d = (importances[b] as number) - (importances[a] as number);
          return d !== 0 ? d : a - b;
        });
        const keep = new Set(order.slice(0, this.maxFeatures));
        for (let j = 0; j < n; j++) mask[j] = keep.has(j);
      }
    }
    return mask;
  }

  /**
   * Fit the estimator and select features by importance.
   *
   * @throws {InvalidParameterError} If the estimator exposes no usable importances
   */
  fit(X: Tensor, y: Tensor): this {
    assert2D(X, "X");
    const [, nFeatures] = getShape2D(X);
    const est = cloneEstimator(this.estimator);
    est.fit(X, y);
    const importances = extractImportances(est, nFeatures, "l1");
    this.fitted_ = est;
    this.importances_ = importances;
    this.mask_ = this.computeMask(importances);
    this.nFeaturesIn_ = nFeatures;
    return this;
  }

  /**
   * Keep only the selected features.
   *
   * @returns Tensor of shape (n_samples, n_selected) with the same dtype as X
   */
  transform(X: Tensor): Tensor {
    if (!this.mask_ || this.nFeaturesIn_ === undefined) {
      throw new NotFittedError("SelectFromModel must be fitted before transform");
    }
    assertFeatureCount(X, this.nFeaturesIn_);
    return gatherColumns(X, maskToIndices(this.mask_));
  }

  fitTransform(X: Tensor, y: Tensor): Tensor {
    this.fit(X, y);
    return this.transform(X);
  }

  /** Map reduced data back to the original features; dropped features become zeros. */
  inverseTransform(X: Tensor): Tensor {
    if (!this.mask_ || this.nFeaturesIn_ === undefined) {
      throw new NotFittedError("SelectFromModel must be fitted before inverseTransform");
    }
    assert2D(X, "X");
    return scatterColumns(X, maskToIndices(this.mask_), this.nFeaturesIn_);
  }

  /** Selection mask (true = kept), or the kept indices when `indices` is true. */
  getSupport(): boolean[];
  getSupport(indices: false): boolean[];
  getSupport(indices: true): number[];
  getSupport(indices?: boolean): boolean[] | number[];
  getSupport(indices = false): boolean[] | number[] {
    if (!this.mask_) {
      throw new NotFittedError("SelectFromModel must be fitted before getSupport");
    }
    return supportOf(this.mask_, indices);
  }

  /** Importances read from the estimator during fit. */
  get importances(): number[] {
    if (!this.importances_) {
      throw new NotFittedError("SelectFromModel must be fitted before accessing importances");
    }
    return [...this.importances_];
  }

  /** The estimator instance that was fitted (a clone of the one passed in, when clonable). */
  get fittedEstimator(): ImportanceEstimator {
    if (!this.fitted_) {
      throw new NotFittedError("SelectFromModel must be fitted before accessing fittedEstimator");
    }
    return this.fitted_;
  }

  /** Number of features seen during fit. */
  get nFeaturesIn(): number {
    if (this.nFeaturesIn_ === undefined) {
      throw new NotFittedError("SelectFromModel must be fitted before accessing nFeaturesIn");
    }
    return this.nFeaturesIn_;
  }

  getParams(): Record<string, unknown> {
    return {
      estimator: this.estimator,
      threshold: this.threshold,
      maxFeatures: this.maxFeatures,
    };
  }

  /**
   * Update parameters. New `threshold` / `maxFeatures` values are applied to
   * the fitted importances immediately; a new `estimator` discards the fitted state.
   */
  setParams(params: Record<string, unknown>): this {
    assertKnownParams(params, ["estimator", "threshold", "maxFeatures"], "SelectFromModel");
    const next = new SelectFromModel({
      ...definedEntries(this.getParams()),
      ...definedEntries(params),
    } as ConstructorParameters<typeof SelectFromModel>[0]);
    const estimatorChanged = next.estimator !== this.estimator;
    this.estimator = next.estimator;
    this.threshold = next.threshold;
    this.maxFeatures = next.maxFeatures;
    if (estimatorChanged) {
      this.fitted_ = undefined;
      this.importances_ = undefined;
      this.mask_ = undefined;
      this.nFeaturesIn_ = undefined;
    } else if (this.importances_) {
      this.mask_ = this.computeMask(this.importances_);
    }
    return this;
  }

  /** Create an unfitted copy; the inner estimator is cloned as well. */
  clone(): SelectFromModel {
    return new SelectFromModel({
      ...definedEntries({ threshold: this.threshold, maxFeatures: this.maxFeatures }),
      estimator: cloneEstimator(this.estimator),
    });
  }
}

// ---------------------------------------------------------------------------
// RFE
// ---------------------------------------------------------------------------

function resolveNFeaturesToSelect(value: number | undefined, nFeatures: number): number {
  let n: number;
  if (value === undefined) {
    n = Math.max(1, Math.floor(nFeatures / 2));
  } else if (value > 0 && value < 1) {
    n = Math.max(1, Math.floor(nFeatures * value));
  } else {
    n = value;
  }
  if (n > nFeatures) {
    throw new InvalidParameterError(
      `nFeaturesToSelect=${n} exceeds number of features (${nFeatures})`,
      "nFeaturesToSelect",
      value
    );
  }
  return n;
}

/**
 * Recursive Feature Elimination (RFE).
 *
 * Recursively removes the least important features, re-fitting the
 * estimator each time, until the desired number of features is reached.
 * Importances are read as described for {@link SelectFromModel}; for
 * multi-target coefficients the sum of squares over targets is ranked, as in
 * scikit-learn. The estimator is cloned when possible, so the instance passed
 * in is left untouched.
 *
 * `ranking` is 1 for selected features. Features removed in the first round get
 * the largest rank, features removed in the last round get 2.
 *
 * @example
 * ```ts
 * import { RFE } from 'deepbox/preprocess';
 * import { LogisticRegression } from 'deepbox/ml';
 *
 * const rfe = new RFE({
 *   estimator: new LogisticRegression(),
 *   nFeaturesToSelect: 3,
 * });
 * rfe.fit(X, y);
 * const Xt = rfe.transform(X);
 * ```
 */
export class RFE {
  private estimator: ImportanceEstimator;
  private nFeaturesToSelect: number | undefined;
  private step: number;
  private ranking_: number[] | undefined;
  private mask_: boolean[] | undefined;
  private nFeaturesIn_: number | undefined;

  /**
   * @param options.estimator - Estimator exposing coefficients or feature importances after fit
   * @param options.nFeaturesToSelect - Number of features to keep: a positive integer, or a
   *   fraction in (0, 1) of the feature count. Default: half of the features (at least 1).
   * @param options.step - Features removed per round: a positive integer, or a fraction in
   *   (0, 1) of the feature count (default 1)
   */
  constructor(options: {
    estimator: ImportanceEstimator;
    nFeaturesToSelect?: number;
    step?: number;
  }) {
    if (!options || typeof options.estimator?.fit !== "function") {
      throw new InvalidParameterError(
        "estimator must be an object with a fit(X, y) method",
        "estimator"
      );
    }
    this.estimator = options.estimator;
    this.nFeaturesToSelect = options.nFeaturesToSelect;
    this.step = options.step ?? 1;
    const n = this.nFeaturesToSelect;
    if (
      n !== undefined &&
      !((Number.isInteger(n) && n >= 1) || (Number.isFinite(n) && n > 0 && n < 1))
    ) {
      throw new InvalidParameterError(
        "nFeaturesToSelect must be a positive integer or a fraction in (0, 1)",
        "nFeaturesToSelect",
        n
      );
    }
    validateStep(this.step);
  }

  /**
   * Eliminate features recursively until `nFeaturesToSelect` remain.
   *
   * @throws {InvalidParameterError} If nFeaturesToSelect exceeds the number of features
   */
  fit(X: Tensor, y: Tensor): this {
    assert2D(X, "X");
    const [, nFeatures] = getShape2D(X);
    const target = resolveNFeaturesToSelect(this.nFeaturesToSelect, nFeatures);
    const est = cloneEstimator(this.estimator);
    const ranking = runElimination(est, X, y, target, resolveStep(this.step, nFeatures));
    this.ranking_ = ranking;
    this.mask_ = ranking.map((r) => r === 1);
    this.nFeaturesIn_ = nFeatures;
    return this;
  }

  /**
   * Keep only the selected features.
   *
   * @returns Tensor of shape (n_samples, n_selected) with the same dtype as X
   */
  transform(X: Tensor): Tensor {
    if (!this.mask_ || this.nFeaturesIn_ === undefined) {
      throw new NotFittedError("RFE must be fitted before transform");
    }
    assertFeatureCount(X, this.nFeaturesIn_);
    return gatherColumns(X, maskToIndices(this.mask_));
  }

  fitTransform(X: Tensor, y: Tensor): Tensor {
    this.fit(X, y);
    return this.transform(X);
  }

  /** Map reduced data back to the original features; dropped features become zeros. */
  inverseTransform(X: Tensor): Tensor {
    if (!this.mask_ || this.nFeaturesIn_ === undefined) {
      throw new NotFittedError("RFE must be fitted before inverseTransform");
    }
    assert2D(X, "X");
    return scatterColumns(X, maskToIndices(this.mask_), this.nFeaturesIn_);
  }

  /** Selection mask (true = kept), or the kept indices when `indices` is true. */
  getSupport(): boolean[];
  getSupport(indices: false): boolean[];
  getSupport(indices: true): number[];
  getSupport(indices?: boolean): boolean[] | number[];
  getSupport(indices = false): boolean[] | number[] {
    if (!this.mask_) {
      throw new NotFittedError("RFE must be fitted before getSupport");
    }
    return supportOf(this.mask_, indices);
  }

  /** Feature ranking (1 = selected, larger = eliminated earlier). */
  get ranking(): number[] {
    if (!this.ranking_) {
      throw new NotFittedError("RFE must be fitted before accessing ranking");
    }
    return [...this.ranking_];
  }

  /** Number of selected features. */
  get nFeatures(): number {
    if (!this.mask_) {
      throw new NotFittedError("RFE must be fitted before accessing nFeatures");
    }
    return this.mask_.filter(Boolean).length;
  }

  getParams(): Record<string, unknown> {
    return {
      estimator: this.estimator,
      nFeaturesToSelect: this.nFeaturesToSelect,
      step: this.step,
    };
  }

  /** Update parameters. The fitted state is discarded. */
  setParams(params: Record<string, unknown>): this {
    assertKnownParams(params, ["estimator", "nFeaturesToSelect", "step"], "RFE");
    const next = new RFE({
      ...definedEntries(this.getParams()),
      ...definedEntries(params),
    } as ConstructorParameters<typeof RFE>[0]);
    this.estimator = next.estimator;
    this.nFeaturesToSelect = next.nFeaturesToSelect;
    this.step = next.step;
    this.ranking_ = undefined;
    this.mask_ = undefined;
    this.nFeaturesIn_ = undefined;
    return this;
  }

  /** Create an unfitted copy; the inner estimator is cloned as well. */
  clone(): RFE {
    return new RFE({
      ...definedEntries({ nFeaturesToSelect: this.nFeaturesToSelect }),
      step: this.step,
      estimator: cloneEstimator(this.estimator),
    });
  }
}

// ---------------------------------------------------------------------------
// RFECV
// ---------------------------------------------------------------------------

/**
 * Scoring used by {@link RFECV}: a built-in name (higher is better) or a
 * function `(yTrue, yPred) => score` where higher is better.
 */
export type RFECVScoring =
  | "accuracy"
  | "r2"
  | "neg_mean_squared_error"
  | "neg_mean_absolute_error"
  | ((yTrue: Tensor, yPred: Tensor) => number);

const RFECV_SCORING_NAMES = ["accuracy", "r2", "neg_mean_squared_error", "neg_mean_absolute_error"];

function builtinScore(name: string, yTrue: Float64Array, yPred: Float64Array): number {
  const n = yTrue.length;
  if (n === 0) return Number.NaN;
  if (name === "accuracy") {
    let correct = 0;
    for (let i = 0; i < n; i++) if (yTrue[i] === yPred[i]) correct++;
    return correct / n;
  }
  if (name === "neg_mean_squared_error" || name === "neg_mean_absolute_error") {
    let acc = 0;
    for (let i = 0; i < n; i++) {
      const d = (yTrue[i] as number) - (yPred[i] as number);
      acc += name === "neg_mean_squared_error" ? d * d : Math.abs(d);
    }
    return -acc / n;
  }
  // r2
  let mean = 0;
  for (let i = 0; i < n; i++) mean += yTrue[i] as number;
  mean /= n;
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    const d = (yTrue[i] as number) - (yPred[i] as number);
    const t = (yTrue[i] as number) - mean;
    ssRes += d * d;
    ssTot += t * t;
  }
  if (ssTot === 0) return ssRes === 0 ? 1 : 0;
  return 1 - ssRes / ssTot;
}

function readPredictions(pred: Tensor, n: number): Float64Array {
  if (!(pred instanceof TensorClass) || pred.dtype === "string") {
    throw new InvalidParameterError("estimator.predict must return a numeric Tensor", "estimator");
  }
  if (pred.size !== n) {
    throw new ShapeError(`estimator.predict returned ${pred.size} values for ${n} samples`);
  }
  const block = readNumericBlock(pred) as NumericBlock;
  return block.values;
}

/**
 * Build validation folds: contiguous KFold blocks (the first `n % cv` folds get
 * one extra sample), or stratified folds following scikit-learn's StratifiedKFold
 * without shuffling.
 */
function makeFolds(nSamples: number, cv: number, labels: Float64Array | undefined): number[][] {
  if (cv > nSamples) {
    throw new InvalidParameterError(
      `cv=${cv} cannot be greater than the number of samples (${nSamples})`,
      "cv",
      cv
    );
  }
  const folds: number[][] = Array.from({ length: cv }, () => []);
  if (!labels) {
    const base = Math.floor(nSamples / cv);
    const extra = nSamples % cv;
    let start = 0;
    for (let f = 0; f < cv; f++) {
      const size = base + (f < extra ? 1 : 0);
      for (let i = start; i < start + size; i++) (folds[f] as number[]).push(i);
      start += size;
    }
    return folds;
  }
  // Classes are encoded in order of first appearance.
  const codeOf = new Map<number, number>();
  const members: number[][] = [];
  for (let i = 0; i < nSamples; i++) {
    const label = labels[i] as number;
    let code = codeOf.get(label);
    if (code === undefined) {
      code = members.length;
      codeOf.set(label, code);
      members.push([]);
    }
    (members[code] as number[]).push(i);
  }
  const maxCount = members.reduce((best, m) => Math.max(best, m.length), 0);
  if (cv > maxCount) {
    throw new InvalidParameterError(
      `cv=${cv} cannot be greater than the number of members in each class (largest class has ${maxCount})`,
      "cv",
      cv
    );
  }
  // Sorted class codes; fold f takes every cv-th position starting at f.
  let start = 0;
  for (const cls of members) {
    const perFold = new Array<number>(cv).fill(0);
    for (let p = start; p < start + cls.length; p++)
      perFold[p % cv] = (perFold[p % cv] as number) + 1;
    let at = 0;
    for (let f = 0; f < cv; f++) {
      for (let c = 0; c < (perFold[f] as number); c++) {
        (folds[f] as number[]).push(cls[at++] as number);
      }
    }
    start += cls.length;
  }
  for (const fold of folds) fold.sort((a, b) => a - b);
  return folds;
}

/**
 * Recursive Feature Elimination with Cross-Validation (RFECV).
 *
 * Runs RFE inside each cross-validation fold. On every fold the estimator is
 * fitted on the training part with the currently active features, scored on the
 * validation part, and then used to drop the least important features. The
 * feature count with the best mean score is chosen (the smallest count on ties)
 * and RFE is run once more on all data to pick the final features.
 *
 * Scoring: `scoring` if given. Otherwise the estimator's own `score(X, y)` is
 * used when it exists (accuracy for classifiers, R^2 for regressors), and
 * accuracy computed from `predict` when it does not. Folds are stratified for
 * classifiers with accuracy scoring (see `stratified`); otherwise they are
 * contiguous blocks, so shuffle ordered data beforehand.
 *
 * @example
 * ```ts
 * import { RFECV } from 'deepbox/preprocess';
 * import { LogisticRegression } from 'deepbox/ml';
 *
 * const rfecv = new RFECV({
 *   estimator: new LogisticRegression(),
 *   cv: 5,
 *   step: 1,
 * });
 * rfecv.fit(X, y);
 * console.log(rfecv.nFeatures); // optimal number of features
 * const Xt = rfecv.transform(X);
 * ```
 */
export class RFECV {
  private estimator: ScoringEstimator;
  private cv: number;
  private step: number;
  private minFeaturesToSelect: number;
  private scoring: RFECVScoring | undefined;
  private stratified: boolean | undefined;

  private ranking_: number[] | undefined;
  private mask_: boolean[] | undefined;
  private nFeaturesIn_: number | undefined;
  private nFeatures_: number | undefined;
  private cvScores_: Map<number, number> | undefined;
  private cvStd_: Map<number, number> | undefined;

  /**
   * @param options.estimator - Estimator with `fit`, `predict` and coefficients or importances
   * @param options.cv - Number of folds (default 5, at least 2)
   * @param options.step - Features removed per round: positive integer or fraction in (0, 1)
   * @param options.minFeaturesToSelect - Fewest features to consider (default 1)
   * @param options.scoring - Scoring name or function (see class description)
   * @param options.stratified - Force stratified (true) or contiguous (false) folds. Default:
   *   stratified when the estimator is a classifier and scoring is accuracy.
   */
  constructor(options: {
    estimator: ScoringEstimator;
    cv?: number;
    step?: number;
    minFeaturesToSelect?: number;
    scoring?: RFECVScoring;
    stratified?: boolean;
  }) {
    if (
      !options ||
      typeof options.estimator?.fit !== "function" ||
      typeof options.estimator?.predict !== "function"
    ) {
      throw new InvalidParameterError(
        "estimator must be an object with fit(X, y) and predict(X) methods",
        "estimator"
      );
    }
    this.estimator = options.estimator;
    this.cv = options.cv ?? 5;
    this.step = options.step ?? 1;
    this.minFeaturesToSelect = options.minFeaturesToSelect ?? 1;
    this.scoring = options.scoring;
    this.stratified = options.stratified;

    if (!Number.isInteger(this.cv) || this.cv < 2) {
      throw new InvalidParameterError("cv must be an integer >= 2", "cv", this.cv);
    }
    validateStep(this.step);
    if (!Number.isInteger(this.minFeaturesToSelect) || this.minFeaturesToSelect < 1) {
      throw new InvalidParameterError(
        "minFeaturesToSelect must be a positive integer",
        "minFeaturesToSelect",
        this.minFeaturesToSelect
      );
    }
    if (
      this.scoring !== undefined &&
      typeof this.scoring !== "function" &&
      !RFECV_SCORING_NAMES.includes(this.scoring)
    ) {
      throw new InvalidParameterError(
        `scoring must be a function or one of ${RFECV_SCORING_NAMES.join(", ")}`,
        "scoring",
        this.scoring
      );
    }
  }

  private scoreFold(
    est: ScoringEstimator,
    valX: Tensor,
    valY: Tensor,
    valYValues: Float64Array
  ): number {
    const scoring = this.scoring;
    if (typeof scoring === "function") {
      const pred = readPredictions(est.predict(valX), valYValues.length);
      return scoring(
        valY,
        TensorClass.fromTypedArray({
          data: pred,
          shape: [pred.length],
          dtype: "float64",
          device: "cpu",
        })
      );
    }
    if (scoring === undefined) {
      const withScore = est as unknown as { score?: (X: Tensor, y: Tensor) => number };
      if (typeof withScore.score === "function") {
        return withScore.score(valX, valY);
      }
      return builtinScore(
        "accuracy",
        valYValues,
        readPredictions(est.predict(valX), valYValues.length)
      );
    }
    return builtinScore(scoring, valYValues, readPredictions(est.predict(valX), valYValues.length));
  }

  /**
   * Choose the number of features by cross-validation, then select them.
   *
   * @throws {InvalidParameterError} If cv exceeds the number of samples or
   *   minFeaturesToSelect exceeds the number of features
   */
  fit(X: Tensor, y: Tensor): this {
    assert2D(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    if (this.minFeaturesToSelect > nFeatures) {
      throw new InvalidParameterError(
        `minFeaturesToSelect=${this.minFeaturesToSelect} exceeds number of features (${nFeatures})`,
        "minFeaturesToSelect",
        this.minFeaturesToSelect
      );
    }
    const yValues = readTarget(y, nSamples, "RFECV", false);

    const isClassifier =
      getEstimatorTags(this.estimator as unknown as Parameters<typeof getEstimatorTags>[0])
        .estimatorType === "classifier";
    const useStratified =
      this.stratified ??
      (isClassifier && (this.scoring === undefined || this.scoring === "accuracy"));
    const folds = makeFolds(nSamples, this.cv, useStratified ? yValues : undefined);
    const step = resolveStep(this.step, nFeatures);
    const est = cloneEstimator(this.estimator);

    // Feature counts visited by every fold (same for all folds).
    const scoresByCount = new Map<number, number[]>();
    for (const valRows of folds) {
      const inVal = new Set(valRows);
      const trainRows: number[] = [];
      for (let i = 0; i < nSamples; i++) if (!inVal.has(i)) trainRows.push(i);
      const trainX = gatherRows(X, trainRows);
      const trainY = gatherRows(y, trainRows);
      const valX = gatherRows(X, valRows);
      const valY = gatherRows(y, valRows);
      const valYValues = Float64Array.from(valRows, (i) => yValues[i] as number);

      let active = Array.from({ length: nFeatures }, (_, i) => i);
      for (;;) {
        const nActive = active.length;
        const subTrain = nActive === nFeatures ? trainX : gatherColumns(trainX, active);
        const subVal = nActive === nFeatures ? valX : gatherColumns(valX, active);
        est.fit(subTrain, trainY);
        const score = this.scoreFold(est, subVal, valY, valYValues);
        const list = scoresByCount.get(nActive);
        if (list) list.push(score);
        else scoresByCount.set(nActive, [score]);

        if (nActive <= this.minFeaturesToSelect) break;
        const importances = extractImportances(est, nActive, "l2");
        const nRemove = Math.min(step, nActive - this.minFeaturesToSelect);
        const drop = new Set(argsortAscending(importances).slice(0, nRemove));
        active = active.filter((_, local) => !drop.has(local));
      }
    }

    const means = new Map<number, number>();
    const stds = new Map<number, number>();
    let bestN = nFeatures;
    let bestScore = Number.NEGATIVE_INFINITY;
    for (const [n, list] of scoresByCount) {
      const mean = list.reduce((a, b) => a + b, 0) / list.length;
      const variance = list.reduce((a, b) => a + (b - mean) ** 2, 0) / list.length;
      means.set(n, mean);
      stds.set(n, Math.sqrt(variance));
      // Counts are visited from most to fewest features: on ties keep the fewest.
      if (mean >= bestScore) {
        bestScore = mean;
        bestN = n;
      }
    }

    const ranking = runElimination(est, X, y, bestN, step);

    this.cvScores_ = means;
    this.cvStd_ = stds;
    this.nFeatures_ = bestN;
    this.ranking_ = ranking;
    this.mask_ = ranking.map((r) => r === 1);
    this.nFeaturesIn_ = nFeatures;
    return this;
  }

  /**
   * Keep only the selected features.
   *
   * @returns Tensor of shape (n_samples, n_selected) with the same dtype as X
   */
  transform(X: Tensor): Tensor {
    if (!this.mask_ || this.nFeaturesIn_ === undefined) {
      throw new NotFittedError("RFECV must be fitted before transform");
    }
    assertFeatureCount(X, this.nFeaturesIn_);
    return gatherColumns(X, maskToIndices(this.mask_));
  }

  fitTransform(X: Tensor, y: Tensor): Tensor {
    this.fit(X, y);
    return this.transform(X);
  }

  /** Map reduced data back to the original features; dropped features become zeros. */
  inverseTransform(X: Tensor): Tensor {
    if (!this.mask_ || this.nFeaturesIn_ === undefined) {
      throw new NotFittedError("RFECV must be fitted before inverseTransform");
    }
    assert2D(X, "X");
    return scatterColumns(X, maskToIndices(this.mask_), this.nFeaturesIn_);
  }

  /** Selection mask (true = kept), or the kept indices when `indices` is true. */
  getSupport(): boolean[];
  getSupport(indices: false): boolean[];
  getSupport(indices: true): number[];
  getSupport(indices?: boolean): boolean[] | number[];
  getSupport(indices = false): boolean[] | number[] {
    if (!this.mask_) {
      throw new NotFittedError("RFECV must be fitted before getSupport");
    }
    return supportOf(this.mask_, indices);
  }

  /** Feature ranking (1 = selected, larger = eliminated earlier). */
  get ranking(): number[] {
    if (!this.ranking_) {
      throw new NotFittedError("RFECV must be fitted before accessing ranking");
    }
    return [...this.ranking_];
  }

  /** Optimal number of features selected by cross-validation. */
  get nFeatures(): number {
    if (this.nFeatures_ === undefined) {
      throw new NotFittedError("RFECV must be fitted before accessing nFeatures");
    }
    return this.nFeatures_;
  }

  /** Mean cross-validation score for each feature count tested (most features first). */
  get gridScores(): Map<number, number> {
    if (!this.cvScores_) {
      throw new NotFittedError("RFECV must be fitted before accessing gridScores");
    }
    return new Map(this.cvScores_);
  }

  /** Standard deviation of the cross-validation scores for each feature count tested. */
  get gridScoresStd(): Map<number, number> {
    if (!this.cvStd_) {
      throw new NotFittedError("RFECV must be fitted before accessing gridScoresStd");
    }
    return new Map(this.cvStd_);
  }

  getParams(): Record<string, unknown> {
    return {
      estimator: this.estimator,
      cv: this.cv,
      step: this.step,
      minFeaturesToSelect: this.minFeaturesToSelect,
      scoring: this.scoring,
      stratified: this.stratified,
    };
  }

  /** Update parameters. The fitted state is discarded. */
  setParams(params: Record<string, unknown>): this {
    assertKnownParams(
      params,
      ["estimator", "cv", "step", "minFeaturesToSelect", "scoring", "stratified"],
      "RFECV"
    );
    const next = new RFECV({
      ...definedEntries(this.getParams()),
      ...definedEntries(params),
    } as ConstructorParameters<typeof RFECV>[0]);
    this.estimator = next.estimator;
    this.cv = next.cv;
    this.step = next.step;
    this.minFeaturesToSelect = next.minFeaturesToSelect;
    this.scoring = next.scoring;
    this.stratified = next.stratified;
    this.ranking_ = undefined;
    this.mask_ = undefined;
    this.nFeaturesIn_ = undefined;
    this.nFeatures_ = undefined;
    this.cvScores_ = undefined;
    this.cvStd_ = undefined;
    return this;
  }

  /** Create an unfitted copy; the inner estimator is cloned as well. */
  clone(): RFECV {
    return new RFECV({
      ...definedEntries({ scoring: this.scoring, stratified: this.stratified }),
      cv: this.cv,
      step: this.step,
      minFeaturesToSelect: this.minFeaturesToSelect,
      estimator: cloneEstimator(this.estimator),
    });
  }
}

/**
 * ANOVA F-value between each feature and the target classes.
 *
 * Returns one F statistic per feature, `MSB / MSW`, where the class labels in
 * `y` define the groups. Same statistic as `sklearn.feature_selection.f_classif`
 * (without the p-values). Constant features and targets with a single class
 * score 0; an exactly zero within-class variance with separated class means
 * scores `Infinity`.
 *
 * @param X - Features of shape (n_samples, n_features); must be finite
 * @param y - Class labels of shape (n_samples,)
 * @returns F statistics, one per feature
 * @throws {ShapeError} If X is not 2D or y does not match X
 * @throws {DataValidationError} If X or y contain NaN or infinity
 */
export const fClassif = f_classif;
/**
 * Univariate F statistic of the linear regression of `y` on each feature.
 *
 * For each feature the Pearson correlation `r` with `y` is turned into
 * `F = r^2 / (1 - r^2) * (n - 2)`, as in `sklearn.feature_selection.f_regression`
 * with `center=True` (p-values are not returned). Constant features, a constant
 * target and fewer than three samples score 0; a perfect correlation scores
 * `Infinity`.
 *
 * @param X - Features of shape (n_samples, n_features); must be finite
 * @param y - Continuous target of shape (n_samples,)
 * @returns F statistics, one per feature
 * @throws {ShapeError} If X is not 2D or y does not match X
 * @throws {DataValidationError} If X or y contain NaN or infinity
 */
export const fRegression = f_regression;
