/**
 * Kernel support vector machines: {@link SVC} and {@link SVR}, plus the SMO solver and
 * kernel helpers that {@link NuSVC}, {@link NuSVR} and {@link OneClassSVM} share.
 *
 * The dual problems are solved with the working-set algorithm of LIBSVM (second-order
 * working-set selection, Fan, Chen and Lin 2005), so the fitted models agree with
 * scikit-learn's `SVC` and `SVR` up to the solver tolerance.
 *
 * @module ml/svm/KernelSVM
 * @see {@link https://deepbox.dev/docs/ml-svm | Deepbox SVM}
 */

import {
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
  warn,
} from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { toFloat64View, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier, Regressor } from "../base";

/** Kernel functions supported by the kernel SVM estimators. */
export type KernelType = "rbf" | "linear" | "poly" | "sigmoid";

/** Kernel coefficient: a positive number, `"scale"` (`1 / (n_features * X.var())`) or `"auto"` (`1 / n_features`). */
export type GammaOption = number | "scale" | "auto";

/** Per-class penalty multipliers: `"balanced"` or a map from class label to weight. */
export type ClassWeightOption = "balanced" | Record<number, number>;

/** Resolved kernel parameters used when evaluating the kernel. */
export type SvmKernelParams = {
  readonly kernel: KernelType;
  readonly gamma: number;
  readonly coef0: number;
  readonly degree: number;
};

// ---------------------------------------------------------------------------
// Parameter validation (shared by every estimator in this directory)
// ---------------------------------------------------------------------------

const KERNEL_NAMES: readonly KernelType[] = ["rbf", "linear", "poly", "sigmoid"];

/** @internal */
export function parseC(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new InvalidParameterError("C must be positive and finite", "C", value);
  }
  return value;
}

/** @internal */
export function parseMaxIter(value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new InvalidParameterError("maxIter must be a positive integer", "maxIter", value);
  }
  return value;
}

/** @internal */
export function parseTol(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new InvalidParameterError("tol must be >= 0 and finite", "tol", value);
  }
  return value;
}

/** @internal */
export function parseEpsilon(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new InvalidParameterError("epsilon must be >= 0 and finite", "epsilon", value);
  }
  return value;
}

/** @internal */
export function parseNu(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0 || value > 1) {
    throw new InvalidParameterError("nu must be in (0, 1]", "nu", value);
  }
  return value;
}

/** @internal */
export function parseKernel(value: unknown): KernelType {
  if (typeof value !== "string" || !KERNEL_NAMES.includes(value as KernelType)) {
    throw new InvalidParameterError(
      'kernel must be "rbf", "linear", "poly", or "sigmoid"',
      "kernel",
      value
    );
  }
  return value as KernelType;
}

/** @internal */
export function parseGamma(value: unknown): GammaOption {
  if (value === "scale" || value === "auto") return value;
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new InvalidParameterError(
      'gamma must be "scale", "auto", or a positive finite number',
      "gamma",
      value
    );
  }
  return value;
}

/** @internal */
export function parseCoef0(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new InvalidParameterError("coef0 must be a finite number", "coef0", value);
  }
  return value;
}

/** @internal */
export function parseDegree(value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new InvalidParameterError("degree must be an integer >= 1", "degree", value);
  }
  return value;
}

/** @internal */
export function parseCacheSize(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new InvalidParameterError(
      "cacheSize must be a positive number of megabytes",
      "cacheSize",
      value
    );
  }
  return value;
}

/** @internal */
export function parseClassWeight(value: unknown): ClassWeightOption {
  if (value === "balanced") return value;
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    const out: Record<number, number> = {};
    for (const [key, weight] of Object.entries(value)) {
      const label = Number(key);
      if (key.trim() === "" || Number.isNaN(label)) {
        throw new InvalidParameterError(
          `classWeight keys must be numeric class labels; got "${key}"`,
          "classWeight",
          value
        );
      }
      if (typeof weight !== "number" || !Number.isFinite(weight) || weight < 0) {
        throw new InvalidParameterError(
          `classWeight for class ${key} must be a finite number >= 0`,
          "classWeight",
          value
        );
      }
      out[label] = weight;
    }
    return out;
  }
  throw new InvalidParameterError(
    'classWeight must be "balanced" or an object mapping class labels to weights',
    "classWeight",
    value
  );
}

/** @internal */
export function copyClassWeight(value: ClassWeightOption): ClassWeightOption {
  return value === "balanced" ? value : { ...value };
}

/** Read `options[key]`, falling back to `fallback` when it is `undefined`, validating otherwise. */
export function optionOr<T>(
  options: Record<string, unknown>,
  key: string,
  fallback: T,
  parse: (value: unknown) => T
): T {
  const value = options[key];
  return value === undefined ? fallback : parse(value);
}

/**
 * Merge `params` into the current configuration for `setParams`.
 *
 * Unknown keys and `undefined` values (except for keys in `optionalKeys`, where `undefined`
 * clears the setting) are rejected. The merged record is validated by the caller before
 * anything is stored, so a failing call leaves the estimator unchanged.
 *
 * @internal
 */
export function mergeParams(
  current: Record<string, unknown>,
  params: Record<string, unknown>,
  allowed: readonly string[],
  optionalKeys: readonly string[] = []
): Record<string, unknown> {
  const next: Record<string, unknown> = { ...current };
  for (const [key, value] of Object.entries(params)) {
    if (!allowed.includes(key)) {
      throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
    }
    if (value === undefined) {
      if (!optionalKeys.includes(key)) {
        throw new InvalidParameterError(`${key} must not be undefined`, key, value);
      }
      delete next[key];
    } else {
      next[key] = value;
    }
  }
  return next;
}

// ---------------------------------------------------------------------------
// Input conversion and label helpers
// ---------------------------------------------------------------------------

/**
 * Encode class labels as `int32` when every label is a 32-bit integer, otherwise `float64`
 * so fractional labels are never truncated.
 *
 * @internal
 */
export function labelsToTensor(values: ArrayLike<number>): Tensor {
  let integral = true;
  for (let i = 0; i < values.length; i++) {
    const v = values[i] as number;
    if (!Number.isInteger(v) || v > 2147483647 || v < -2147483648) {
      integral = false;
      break;
    }
  }
  return integral
    ? tensor(Int32Array.from(values), { dtype: "int32" })
    : tensor(Float64Array.from(values), { dtype: "float64" });
}

/**
 * Sorted distinct labels of `y` and the class index of every sample.
 *
 * @internal
 */
export function encodeLabels(yv: Float64Array): { labels: Float64Array; index: Int32Array } {
  const sorted = Float64Array.from(yv, (v) => v + 0).sort();
  const distinct: number[] = [];
  for (let i = 0; i < sorted.length; i++) {
    const v = sorted[i] as number;
    if (i === 0 || v !== distinct[distinct.length - 1]) distinct.push(v);
  }
  const lookup = new Map<number, number>();
  distinct.forEach((v, k) => {
    lookup.set(v, k);
  });
  const index = new Int32Array(yv.length);
  for (let i = 0; i < yv.length; i++) index[i] = lookup.get((yv[i] as number) + 0) as number;
  return { labels: Float64Array.from(distinct), index };
}

/**
 * Per-class penalty multipliers for `classWeight`.
 *
 * `"balanced"` uses `n_samples / (n_classes * count)`. A map must only name labels that occur
 * in `y`; classes it does not mention keep weight 1.
 *
 * @internal
 */
export function resolveClassWeights(
  classWeight: ClassWeightOption | undefined,
  labels: Float64Array,
  classIndex: Int32Array
): Float64Array {
  const k = labels.length;
  const weights = new Float64Array(k).fill(1);
  if (classWeight === undefined) return weights;
  if (classWeight === "balanced") {
    const counts = new Float64Array(k);
    for (let i = 0; i < classIndex.length; i++) counts[classIndex[i] as number]!++;
    for (let c = 0; c < k; c++) weights[c] = classIndex.length / (k * (counts[c] as number));
    return weights;
  }
  for (const [key, weight] of Object.entries(classWeight)) {
    const label = Number(key);
    const c = labels.indexOf(label);
    if (c < 0) {
      throw new InvalidParameterError(
        `classWeight names class ${key}, which does not occur in y`,
        "classWeight",
        classWeight
      );
    }
    weights[c] = weight;
  }
  return weights;
}

/**
 * Validate and read a sample-weight vector.
 *
 * @internal
 */
export function readSampleWeight(
  sampleWeight: Tensor | undefined,
  nSamples: number
): Float64Array | undefined {
  if (sampleWeight === undefined) return undefined;
  if (sampleWeight.ndim !== 1) {
    throw new ShapeError(`sampleWeight must be 1-dimensional; got ndim=${sampleWeight.ndim}`);
  }
  if (sampleWeight.size !== nSamples) {
    throw new ShapeError(
      `sampleWeight must have one entry per sample; got ${sampleWeight.size} for ${nSamples} samples`
    );
  }
  const sw = toFloat64View(sampleWeight);
  for (let i = 0; i < sw.length; i++) {
    const v = sw[i] as number;
    if (!Number.isFinite(v) || v < 0) {
      throw new DataValidationError("sampleWeight must contain finite values >= 0");
    }
  }
  return sw;
}

/** Population variance of all entries of `X`, computed in two passes. */
function totalVariance(X: Float64Array): number {
  const total = X.length;
  if (total === 0) return 0;
  let mean = 0;
  for (let i = 0; i < total; i++) mean += X[i] as number;
  mean /= total;
  let variance = 0;
  for (let i = 0; i < total; i++) {
    const dv = (X[i] as number) - mean;
    variance += dv * dv;
  }
  return variance / total;
}

/**
 * Resolve the `gamma` option on training data.
 *
 * `"scale"` is `1 / (n_features * X.var())` (1 when the variance is 0), `"auto"` is
 * `1 / n_features`.
 *
 * @internal
 */
export function resolveGamma(gamma: GammaOption, X: Float64Array, nFeatures: number): number {
  if (typeof gamma === "number") return gamma;
  if (gamma === "auto") return 1 / nFeatures;
  const variance = totalVariance(X);
  return variance > 0 ? 1 / (nFeatures * variance) : 1;
}

/** Copy a flat float64 buffer into a new `(rows, cols)` tensor. */
function matrixTensor(data: Float64Array, rows: number, cols: number): Tensor {
  return tensor(Float64Array.from(data), { dtype: "float64" }).reshape([rows, cols]);
}

/** Numerically stable logistic function. */
export function stableSigmoid(z: number): number {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

/**
 * Mean accuracy of `predictions` against `y` after validating `y`.
 *
 * @internal
 */
export function accuracyOf(predictions: Tensor, y: Tensor): number {
  if (y.ndim !== 1) {
    throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
  }
  const yv = toFloat64View(y);
  if (yv.length === 0) {
    throw new DataValidationError("y must contain at least one sample");
  }
  for (let i = 0; i < yv.length; i++) {
    if (!Number.isFinite(yv[i])) {
      throw new DataValidationError("y contains non-finite values (NaN or Inf)");
    }
  }
  if (predictions.size !== yv.length) {
    throw new ShapeError(
      `X and y must have the same number of samples; got X=${predictions.size}, y=${yv.length}`
    );
  }
  const pv = toFloat64View(predictions);
  let correct = 0;
  for (let i = 0; i < yv.length; i++) if (pv[i] === yv[i]) correct++;
  return correct / yv.length;
}

/**
 * Coefficient of determination of `predictions` against `y` after validating `y`.
 *
 * A constant `y` scores 1 for a perfect fit and 0 otherwise.
 *
 * @internal
 */
export function r2Of(predictions: Tensor, y: Tensor): number {
  if (y.ndim !== 1) {
    throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
  }
  const yv = toFloat64View(y);
  if (yv.length === 0) {
    throw new DataValidationError("y must contain at least one sample");
  }
  for (let i = 0; i < yv.length; i++) {
    if (!Number.isFinite(yv[i])) {
      throw new DataValidationError("y contains non-finite values (NaN or Inf)");
    }
  }
  if (predictions.size !== yv.length) {
    throw new ShapeError(
      `X and y must have the same number of samples; got X=${predictions.size}, y=${yv.length}`
    );
  }
  const pv = toFloat64View(predictions);
  let mean = 0;
  for (let i = 0; i < yv.length; i++) mean += yv[i] as number;
  mean /= yv.length;
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < yv.length; i++) {
    const yi = yv[i] as number;
    const r = yi - (pv[i] as number);
    const t = yi - mean;
    ssRes += r * r;
    ssTot += t * t;
  }
  return ssTot === 0 ? (ssRes === 0 ? 1 : 0) : 1 - ssRes / ssTot;
}

// ---------------------------------------------------------------------------
// Kernel evaluation
// ---------------------------------------------------------------------------

/**
 * Evaluate the kernel between one vector and `m` vectors stored row-major in `B`.
 *
 * Writes `out[j] = K(a[aOff .. aOff + d), B[j * d .. (j + 1) * d))` for `j < m`.
 *
 * @internal
 */
export function kernelAgainst(
  a: Float64Array,
  aOff: number,
  B: Float64Array,
  m: number,
  d: number,
  kp: SvmKernelParams,
  out: Float64Array
): void {
  switch (kp.kernel) {
    case "rbf": {
      const g = -kp.gamma;
      for (let j = 0; j < m; j++) {
        const bo = j * d;
        let s = 0;
        for (let k = 0; k < d; k++) {
          const t = (a[aOff + k] as number) - (B[bo + k] as number);
          s += t * t;
        }
        out[j] = Math.exp(g * s);
      }
      return;
    }
    case "linear": {
      for (let j = 0; j < m; j++) {
        const bo = j * d;
        let s = 0;
        for (let k = 0; k < d; k++) s += (a[aOff + k] as number) * (B[bo + k] as number);
        out[j] = s;
      }
      return;
    }
    case "poly": {
      for (let j = 0; j < m; j++) {
        const bo = j * d;
        let s = 0;
        for (let k = 0; k < d; k++) s += (a[aOff + k] as number) * (B[bo + k] as number);
        out[j] = (kp.gamma * s + kp.coef0) ** kp.degree;
      }
      return;
    }
    case "sigmoid": {
      for (let j = 0; j < m; j++) {
        const bo = j * d;
        let s = 0;
        for (let k = 0; k < d; k++) s += (a[aOff + k] as number) * (B[bo + k] as number);
        out[j] = Math.tanh(kp.gamma * s + kp.coef0);
      }
      return;
    }
  }
}

const SINGLE = new Float64Array(1);

/** Kernel value of row `i` of `X` with itself. */
function kernelDiagonal(X: Float64Array, i: number, d: number, kp: SvmKernelParams): number {
  kernelAgainst(X, i * d, X.subarray(i * d, (i + 1) * d), 1, d, kp, SINGLE);
  return SINGLE[0] as number;
}

function assertFiniteRow(row: Float64Array, len: number, kp: SvmKernelParams): void {
  for (let j = 0; j < len; j++) {
    if (!Number.isFinite(row[j])) {
      throw new DataValidationError(
        `the ${kp.kernel} kernel produced a non-finite value; scale the features or reduce gamma, coef0 and degree`
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Q matrices
// ---------------------------------------------------------------------------

/**
 * Quadratic-term matrix of a dual problem, exposed row by row.
 *
 * A returned row stays valid until two further `getQ` calls; the solver never needs more
 * than two rows at once.
 *
 * @internal
 */
export interface QMatrix {
  /** Number of dual variables. */
  readonly size: number;
  /** Diagonal of Q. */
  readonly QD: Float64Array;
  getQ(i: number): Float64Array;
}

/** Least-recently-used cache of fixed-length rows. */
class RowCache {
  private readonly rows = new Map<number, Float64Array>();

  constructor(
    private readonly capacity: number,
    private readonly rowLength: number
  ) {}

  get(key: number, fill: (row: Float64Array) => void): Float64Array {
    const hit = this.rows.get(key);
    if (hit !== undefined) {
      this.rows.delete(key);
      this.rows.set(key, hit);
      return hit;
    }
    let row: Float64Array;
    if (this.rows.size >= this.capacity) {
      const oldest = this.rows.keys().next().value as number;
      row = this.rows.get(oldest) as Float64Array;
      this.rows.delete(oldest);
    } else {
      row = new Float64Array(this.rowLength);
    }
    fill(row);
    this.rows.set(key, row);
    return row;
  }
}

function cacheCapacity(cacheSizeMb: number, rowLength: number): number {
  const rows = Math.floor((cacheSizeMb * 1048576) / (8 * Math.max(1, rowLength)));
  return Math.max(2, Math.min(rowLength, rows));
}

/**
 * `Q[i][j] = y[i] * y[j] * K(x_i, x_j)` for classification and one-class problems.
 *
 * @internal
 */
export class ClassificationQ implements QMatrix {
  readonly size: number;
  readonly QD: Float64Array;
  private readonly cache: RowCache;

  constructor(
    private readonly X: Float64Array,
    n: number,
    private readonly d: number,
    private readonly y: Int8Array,
    private readonly kp: SvmKernelParams,
    cacheSizeMb: number
  ) {
    this.size = n;
    this.QD = new Float64Array(n);
    for (let i = 0; i < n; i++) this.QD[i] = kernelDiagonal(X, i, d, kp);
    this.cache = new RowCache(cacheCapacity(cacheSizeMb, n), n);
  }

  getQ(i: number): Float64Array {
    return this.cache.get(i, (row) => {
      const n = this.size;
      kernelAgainst(this.X, i * this.d, this.X, n, this.d, this.kp, row);
      assertFiniteRow(row, n, this.kp);
      const yi = this.y[i] as number;
      for (let j = 0; j < n; j++) row[j] = (row[j] as number) * yi * (this.y[j] as number);
    });
  }
}

/**
 * Q matrix of the epsilon and nu regression duals: `2 * n` variables, the first `n` with
 * sign +1 and the last `n` with sign -1, both tied to the same training samples.
 *
 * @internal
 */
export class RegressionQ implements QMatrix {
  readonly size: number;
  readonly QD: Float64Array;
  private readonly cache: RowCache;
  private readonly buffers: [Float64Array, Float64Array];
  private nextBuffer = 0;

  constructor(
    private readonly X: Float64Array,
    private readonly n: number,
    private readonly d: number,
    private readonly kp: SvmKernelParams,
    cacheSizeMb: number
  ) {
    this.size = 2 * n;
    this.QD = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) {
      const v = kernelDiagonal(X, i, d, kp);
      this.QD[i] = v;
      this.QD[i + n] = v;
    }
    this.cache = new RowCache(cacheCapacity(cacheSizeMb, n), n);
    this.buffers = [new Float64Array(2 * n), new Float64Array(2 * n)];
  }

  getQ(i: number): Float64Array {
    const n = this.n;
    const real = i < n ? i : i - n;
    const base = this.cache.get(real, (row) => {
      kernelAgainst(this.X, real * this.d, this.X, n, this.d, this.kp, row);
      assertFiniteRow(row, n, this.kp);
    });
    const buf = this.buffers[this.nextBuffer] as Float64Array;
    this.nextBuffer ^= 1;
    if (i < n) {
      for (let j = 0; j < n; j++) {
        const v = base[j] as number;
        buf[j] = v;
        buf[j + n] = -v;
      }
    } else {
      for (let j = 0; j < n; j++) {
        const v = base[j] as number;
        buf[j] = -v;
        buf[j + n] = v;
      }
    }
    return buf;
  }
}

// ---------------------------------------------------------------------------
// SMO solver (LIBSVM working-set algorithm)
// ---------------------------------------------------------------------------

/** Outcome of {@link solveSmo}. */
export type SmoResult = {
  /** Solution of the dual problem (modified in place from the starting point). */
  readonly alpha: Float64Array;
  /** Offset of the decision function. In nu mode this is `(r1 - r2) / 2`. */
  readonly rho: number;
  /** Scale `(r1 + r2) / 2` of the nu formulations (0 in standard mode). */
  readonly r: number;
  /** Number of working-set updates performed. */
  readonly iterations: number;
  /** Whether the stopping criterion was met within the iteration budget. */
  readonly converged: boolean;
};

const TAU = 1e-12;
const LOWER = 0;
const UPPER = 1;
const FREE = 2;
/** Variable with an upper bound of 0 (zero sample weight): never selected, never moves. */
const FIXED = 3;

/**
 * Offset from the tightest bounds `lb <= rho <= ub` of a problem without free variables.
 * Either bound can be missing (infinite) when every variable sits on one side, in which
 * case the finite one is the nearest valid offset.
 */
function boundedOffset(ub: number, lb: number): number {
  if (Number.isFinite(ub) && Number.isFinite(lb)) return (ub + lb) / 2;
  if (Number.isFinite(ub)) return ub;
  if (Number.isFinite(lb)) return lb;
  return 0;
}

/**
 * Solve `min 0.5 a'Qa + p'a` subject to `y'a = const` and `0 <= a_i <= C_i`.
 *
 * A variable with `C_i = 0` (a sample with zero weight) is left out of the problem.
 *
 * With `nuMode` the working set is chosen within each class separately, which keeps both
 * `sum(a_i : y_i = +1)` and `sum(a_i : y_i = -1)` constant (the nu-SVM formulations).
 *
 * @param Q - Quadratic term
 * @param p - Linear term
 * @param y - Variable signs, +1 or -1
 * @param alpha - Feasible starting point, updated in place
 * @param C - Upper bound of every variable
 * @param eps - Stopping tolerance on the maximal KKT violation
 * @param maxSteps - Upper bound on the number of working-set updates
 * @param nuMode - Use the nu working-set rule and rho computation
 *
 * @internal
 */
export function solveSmo(
  Q: QMatrix,
  p: Float64Array,
  y: Int8Array,
  alpha: Float64Array,
  C: Float64Array,
  eps: number,
  maxSteps: number,
  nuMode: boolean
): SmoResult {
  const l = Q.size;
  const QD = Q.QD;
  const G = Float64Array.from(p);
  const status = new Uint8Array(l);
  const updateStatus = (i: number): void => {
    const a = alpha[i] as number;
    const ci = C[i] as number;
    status[i] = !(ci > 0) ? FIXED : a >= ci ? UPPER : a <= 0 ? LOWER : FREE;
  };
  for (let i = 0; i < l; i++) updateStatus(i);
  for (let i = 0; i < l; i++) {
    if (status[i] === LOWER || status[i] === FIXED) continue;
    const Qi = Q.getQ(i);
    const ai = alpha[i] as number;
    for (let j = 0; j < l; j++) G[j] = (G[j] as number) + ai * (Qi[j] as number);
  }

  const empty = new Float64Array(0);
  let iter = 0;
  let converged = false;

  while (iter < maxSteps) {
    // ---- working set selection ----
    let iSel = -1;
    let jSel = -1;
    if (!nuMode) {
      let gMax = Number.NEGATIVE_INFINITY;
      let gMax2 = Number.NEGATIVE_INFINITY;
      let objMin = Number.POSITIVE_INFINITY;
      for (let t = 0; t < l; t++) {
        if (status[t] === FIXED) continue;
        if (y[t] === 1) {
          if (status[t] !== UPPER && -(G[t] as number) >= gMax) {
            gMax = -(G[t] as number);
            iSel = t;
          }
        } else if (status[t] !== LOWER && (G[t] as number) >= gMax) {
          gMax = G[t] as number;
          iSel = t;
        }
      }
      const Qi = iSel >= 0 ? Q.getQ(iSel) : empty;
      for (let j = 0; j < l; j++) {
        if (status[j] === FIXED) continue;
        const Gj = G[j] as number;
        if (y[j] === 1) {
          if (status[j] !== LOWER) {
            const gradDiff = gMax + Gj;
            if (Gj >= gMax2) gMax2 = Gj;
            if (gradDiff > 0) {
              const quad =
                (QD[iSel] as number) +
                (QD[j] as number) -
                2 * (y[iSel] as number) * (Qi[j] as number);
              const objDiff =
                quad > 0 ? -(gradDiff * gradDiff) / quad : -(gradDiff * gradDiff) / TAU;
              if (objDiff <= objMin) {
                jSel = j;
                objMin = objDiff;
              }
            }
          }
        } else if (status[j] !== UPPER) {
          const gradDiff = gMax - Gj;
          if (-Gj >= gMax2) gMax2 = -Gj;
          if (gradDiff > 0) {
            const quad =
              (QD[iSel] as number) +
              (QD[j] as number) +
              2 * (y[iSel] as number) * (Qi[j] as number);
            const objDiff = quad > 0 ? -(gradDiff * gradDiff) / quad : -(gradDiff * gradDiff) / TAU;
            if (objDiff <= objMin) {
              jSel = j;
              objMin = objDiff;
            }
          }
        }
      }
      if (gMax + gMax2 < eps || jSel === -1) {
        converged = true;
        break;
      }
    } else {
      let gMaxP = Number.NEGATIVE_INFINITY;
      let gMaxP2 = Number.NEGATIVE_INFINITY;
      let ipSel = -1;
      let gMaxN = Number.NEGATIVE_INFINITY;
      let gMaxN2 = Number.NEGATIVE_INFINITY;
      let inSel = -1;
      let objMin = Number.POSITIVE_INFINITY;
      for (let t = 0; t < l; t++) {
        if (status[t] === FIXED) continue;
        if (y[t] === 1) {
          if (status[t] !== UPPER && -(G[t] as number) >= gMaxP) {
            gMaxP = -(G[t] as number);
            ipSel = t;
          }
        } else if (status[t] !== LOWER && (G[t] as number) >= gMaxN) {
          gMaxN = G[t] as number;
          inSel = t;
        }
      }
      const Qip = ipSel >= 0 ? Q.getQ(ipSel) : empty;
      const Qin = inSel >= 0 ? Q.getQ(inSel) : empty;
      for (let j = 0; j < l; j++) {
        if (status[j] === FIXED) continue;
        const Gj = G[j] as number;
        if (y[j] === 1) {
          if (status[j] !== LOWER) {
            const gradDiff = gMaxP + Gj;
            if (Gj >= gMaxP2) gMaxP2 = Gj;
            if (gradDiff > 0) {
              const quad = (QD[ipSel] as number) + (QD[j] as number) - 2 * (Qip[j] as number);
              const objDiff =
                quad > 0 ? -(gradDiff * gradDiff) / quad : -(gradDiff * gradDiff) / TAU;
              if (objDiff <= objMin) {
                jSel = j;
                objMin = objDiff;
              }
            }
          }
        } else if (status[j] !== UPPER) {
          const gradDiff = gMaxN - Gj;
          if (-Gj >= gMaxN2) gMaxN2 = -Gj;
          if (gradDiff > 0) {
            const quad = (QD[inSel] as number) + (QD[j] as number) - 2 * (Qin[j] as number);
            const objDiff = quad > 0 ? -(gradDiff * gradDiff) / quad : -(gradDiff * gradDiff) / TAU;
            if (objDiff <= objMin) {
              jSel = j;
              objMin = objDiff;
            }
          }
        }
      }
      if (Math.max(gMaxP + gMaxP2, gMaxN + gMaxN2) < eps || jSel === -1) {
        converged = true;
        break;
      }
      iSel = y[jSel] === 1 ? ipSel : inSel;
    }

    // ---- two-variable update ----
    iter++;
    const i = iSel;
    const j = jSel;
    const Qi = Q.getQ(i);
    const Qj = Q.getQ(j);
    const Ci = C[i] as number;
    const Cj = C[j] as number;
    const oldI = alpha[i] as number;
    const oldJ = alpha[j] as number;
    let ai = oldI;
    let aj = oldJ;

    if (y[i] !== y[j]) {
      let quad = (QD[i] as number) + (QD[j] as number) + 2 * (Qi[j] as number);
      if (quad <= 0) quad = TAU;
      const delta = (-(G[i] as number) - (G[j] as number)) / quad;
      const diff = ai - aj;
      ai += delta;
      aj += delta;
      if (diff > 0) {
        if (aj < 0) {
          aj = 0;
          ai = diff;
        }
      } else if (ai < 0) {
        ai = 0;
        aj = -diff;
      }
      if (diff > Ci - Cj) {
        if (ai > Ci) {
          ai = Ci;
          aj = Ci - diff;
        }
      } else if (aj > Cj) {
        aj = Cj;
        ai = Cj + diff;
      }
    } else {
      let quad = (QD[i] as number) + (QD[j] as number) - 2 * (Qi[j] as number);
      if (quad <= 0) quad = TAU;
      const delta = ((G[i] as number) - (G[j] as number)) / quad;
      const sum = ai + aj;
      ai -= delta;
      aj += delta;
      if (sum > Ci) {
        if (ai > Ci) {
          ai = Ci;
          aj = sum - Ci;
        }
      } else if (aj < 0) {
        aj = 0;
        ai = sum;
      }
      if (sum > Cj) {
        if (aj > Cj) {
          aj = Cj;
          ai = sum - Cj;
        }
      } else if (ai < 0) {
        ai = 0;
        aj = sum;
      }
    }

    alpha[i] = ai;
    alpha[j] = aj;
    const dai = ai - oldI;
    const daj = aj - oldJ;
    for (let k = 0; k < l; k++) {
      G[k] = (G[k] as number) + (Qi[k] as number) * dai + (Qj[k] as number) * daj;
    }
    updateStatus(i);
    updateStatus(j);
  }

  // ---- offset ----
  if (!nuMode) {
    let ub = Number.POSITIVE_INFINITY;
    let lb = Number.NEGATIVE_INFINITY;
    let sumFree = 0;
    let nFree = 0;
    for (let i = 0; i < l; i++) {
      if (status[i] === FIXED) continue;
      const yG = (y[i] as number) * (G[i] as number);
      if (status[i] === UPPER) {
        if (y[i] === -1) ub = Math.min(ub, yG);
        else lb = Math.max(lb, yG);
      } else if (status[i] === LOWER) {
        if (y[i] === 1) ub = Math.min(ub, yG);
        else lb = Math.max(lb, yG);
      } else {
        nFree++;
        sumFree += yG;
      }
    }
    const rho = nFree > 0 ? sumFree / nFree : boundedOffset(ub, lb);
    return { alpha, rho, r: 0, iterations: iter, converged };
  }

  let ub1 = Number.POSITIVE_INFINITY;
  let ub2 = Number.POSITIVE_INFINITY;
  let lb1 = Number.NEGATIVE_INFINITY;
  let lb2 = Number.NEGATIVE_INFINITY;
  let sumFree1 = 0;
  let sumFree2 = 0;
  let nFree1 = 0;
  let nFree2 = 0;
  for (let i = 0; i < l; i++) {
    if (status[i] === FIXED) continue;
    const g = G[i] as number;
    if (y[i] === 1) {
      if (status[i] === UPPER) lb1 = Math.max(lb1, g);
      else if (status[i] === LOWER) ub1 = Math.min(ub1, g);
      else {
        nFree1++;
        sumFree1 += g;
      }
    } else if (status[i] === UPPER) lb2 = Math.max(lb2, g);
    else if (status[i] === LOWER) ub2 = Math.min(ub2, g);
    else {
      nFree2++;
      sumFree2 += g;
    }
  }
  const r1 = nFree1 > 0 ? sumFree1 / nFree1 : boundedOffset(ub1, lb1);
  const r2 = nFree2 > 0 ? sumFree2 / nFree2 : boundedOffset(ub2, lb2);
  return { alpha, rho: (r1 - r2) / 2, r: (r1 + r2) / 2, iterations: iter, converged };
}

/**
 * Emit a `ConvergenceWarning` for a solver run that used up its iteration budget.
 *
 * @internal
 */
export function warnNotConverged(estimator: string, maxIter: number): void {
  warn(
    `Solver did not converge within maxIter=${maxIter} passes over the data; ` +
      "increase maxIter, loosen tol, or scale the features",
    "ConvergenceWarning",
    estimator
  );
}

// ---------------------------------------------------------------------------
// Fitted kernel expansions
// ---------------------------------------------------------------------------

/**
 * Decision function `sum_i coef_i * K(sv_i, x) - rho` of a single-output kernel model
 * (regression and one-class SVMs).
 *
 * @internal
 */
export class KernelExpansion {
  readonly nSupport: number;

  private constructor(
    readonly supportX: Float64Array,
    readonly supportIndices: Int32Array,
    readonly coef: Float64Array,
    readonly rho: number,
    readonly nFeatures: number,
    readonly kp: SvmKernelParams
  ) {
    this.nSupport = coef.length;
  }

  /** Keep the training samples whose coefficient is not zero. */
  static fromCoefficients(
    X: Float64Array,
    n: number,
    d: number,
    coef: Float64Array,
    rho: number,
    kp: SvmKernelParams
  ): KernelExpansion {
    let nSv = 0;
    for (let i = 0; i < n; i++) if (coef[i] !== 0) nSv++;
    const supportX = new Float64Array(nSv * d);
    const idx = new Int32Array(nSv);
    const c = new Float64Array(nSv);
    let s = 0;
    for (let i = 0; i < n; i++) {
      if (coef[i] === 0) continue;
      supportX.set(X.subarray(i * d, (i + 1) * d), s * d);
      idx[s] = i;
      c[s] = coef[i] as number;
      s++;
    }
    return new KernelExpansion(supportX, idx, c, rho, d, kp);
  }

  /** `sum_i coef_i * K(sv_i, x)` for every row of `X` (the offset is not subtracted). */
  raw(X: Float64Array, n: number): Float64Array {
    const d = this.nFeatures;
    const out = new Float64Array(n);
    const kv = new Float64Array(this.nSupport);
    for (let i = 0; i < n; i++) {
      kernelAgainst(X, i * d, this.supportX, this.nSupport, d, this.kp, kv);
      let s = 0;
      for (let t = 0; t < this.nSupport; t++) s += (this.coef[t] as number) * (kv[t] as number);
      if (!Number.isFinite(s)) {
        throw new DataValidationError(
          `the ${this.kp.kernel} kernel produced a non-finite decision value for row ${i} of X; scale the features`
        );
      }
      out[i] = s;
    }
    return out;
  }
}

// ---------------------------------------------------------------------------
// One-vs-one classification model (shared by SVC and NuSVC)
// ---------------------------------------------------------------------------

/** Dual solution of one binary sub-problem. */
export type PairSolution = {
  /** `alpha_i * y_i` for every sample of the pair (`y = +1` for the lower class index). */
  readonly coef: Float64Array;
  readonly rho: number;
  readonly iterations: number;
  readonly converged: boolean;
};

type PairModel = {
  readonly positions: Int32Array;
  readonly coef: Float64Array;
  readonly rho: number;
};

/**
 * Fitted one-vs-one multiclass SVM in LIBSVM layout.
 *
 * Every class pair `(a, b)` with `a < b` has its own binary classifier whose decision
 * value is positive for class `a`. Support vectors are stored once and shared by all pairs.
 *
 * @internal
 */
export class OvoModel {
  private constructor(
    readonly labels: Float64Array,
    readonly kp: SvmKernelParams,
    readonly nFeatures: number,
    readonly supportX: Float64Array,
    readonly supportIndices: Int32Array,
    readonly nSupportPerClass: Int32Array,
    private readonly pairs: readonly PairModel[],
    private readonly dualCoefLibsvm: Float64Array,
    readonly nIter: number
  ) {}

  /**
   * Train every class pair with `solvePair` and assemble the model.
   *
   * @returns The model and whether all sub-problems converged
   */
  static fit(
    X: Float64Array,
    n: number,
    d: number,
    labels: Float64Array,
    classIndex: Int32Array,
    kp: SvmKernelParams,
    solvePair: (
      Xsub: Float64Array,
      m: number,
      y: Int8Array,
      sampleIdx: Int32Array,
      a: number,
      b: number
    ) => PairSolution
  ): { model: OvoModel; converged: boolean } {
    const K = labels.length;
    const members: number[][] = Array.from({ length: K }, () => []);
    for (let i = 0; i < n; i++) members[classIndex[i] as number]!.push(i);

    type Raw = { idx: Int32Array; sol: PairSolution };
    const raw: Raw[] = [];
    let converged = true;
    let nIter = 0;
    const isSupport = new Uint8Array(n);
    for (let a = 0; a < K; a++) {
      for (let b = a + 1; b < K; b++) {
        const ma = members[a] as number[];
        const mb = members[b] as number[];
        const m = ma.length + mb.length;
        const idx = new Int32Array(m);
        const y = new Int8Array(m);
        for (let t = 0; t < ma.length; t++) {
          idx[t] = ma[t] as number;
          y[t] = 1;
        }
        for (let t = 0; t < mb.length; t++) {
          idx[ma.length + t] = mb[t] as number;
          y[ma.length + t] = -1;
        }
        let identity = m === n;
        for (let t = 0; identity && t < m; t++) identity = idx[t] === t;
        let Xsub = X;
        if (!identity) {
          Xsub = new Float64Array(m * d);
          for (let t = 0; t < m; t++) {
            const g = idx[t] as number;
            Xsub.set(X.subarray(g * d, (g + 1) * d), t * d);
          }
        }
        const sol = solvePair(Xsub, m, y, idx, a, b);
        converged = converged && sol.converged;
        nIter = Math.max(nIter, sol.iterations);
        for (let t = 0; t < m; t++) if (sol.coef[t] !== 0) isSupport[idx[t] as number] = 1;
        raw.push({ idx, sol });
      }
    }

    // Support vectors grouped by class, in training order inside a class.
    const position = new Int32Array(n).fill(-1);
    const order: number[] = [];
    const nSupportPerClass = new Int32Array(K);
    for (let c = 0; c < K; c++) {
      for (const g of members[c] as number[]) {
        if (isSupport[g] === 1) {
          position[g] = order.length;
          order.push(g);
          nSupportPerClass[c]!++;
        }
      }
    }
    const nSv = order.length;
    const supportX = new Float64Array(nSv * d);
    const supportIndices = Int32Array.from(order);
    order.forEach((g, s) => {
      supportX.set(X.subarray(g * d, (g + 1) * d), s * d);
    });

    const pairs: PairModel[] = [];
    const dual = new Float64Array(Math.max(0, K - 1) * nSv);
    let p = 0;
    for (let a = 0; a < K; a++) {
      for (let b = a + 1; b < K; b++) {
        const { idx, sol } = raw[p++] as Raw;
        let nz = 0;
        for (let t = 0; t < idx.length; t++) if (sol.coef[t] !== 0) nz++;
        const positions = new Int32Array(nz);
        const coef = new Float64Array(nz);
        let q = 0;
        for (let t = 0; t < idx.length; t++) {
          const c = sol.coef[t] as number;
          if (c === 0) continue;
          const g = idx[t] as number;
          positions[q] = position[g] as number;
          coef[q] = c;
          q++;
          const row = (classIndex[g] as number) === a ? b - 1 : a;
          dual[row * nSv + (position[g] as number)] = c;
        }
        pairs.push({ positions, coef, rho: sol.rho });
      }
    }

    return {
      model: new OvoModel(
        labels,
        kp,
        d,
        supportX,
        supportIndices,
        nSupportPerClass,
        pairs,
        dual,
        nIter
      ),
      converged,
    };
  }

  get nClasses(): number {
    return this.labels.length;
  }

  get nSupport(): number {
    return this.supportIndices.length;
  }

  /** Pairwise decision values, one row per sample and one column per class pair (LIBSVM sign). */
  pairDecisions(X: Float64Array, n: number): Float64Array {
    const d = this.nFeatures;
    const nSv = this.nSupport;
    const nPairs = this.pairs.length;
    const out = new Float64Array(n * nPairs);
    const kv = new Float64Array(nSv);
    for (let i = 0; i < n; i++) {
      kernelAgainst(X, i * d, this.supportX, nSv, d, this.kp, kv);
      for (let q = 0; q < nPairs; q++) {
        const pair = this.pairs[q] as PairModel;
        let s = -pair.rho;
        for (let t = 0; t < pair.positions.length; t++) {
          s += (pair.coef[t] as number) * (kv[pair.positions[t] as number] as number);
        }
        if (!Number.isFinite(s)) {
          throw new DataValidationError(
            `the ${this.kp.kernel} kernel produced a non-finite decision value for row ${i} of X; scale the features`
          );
        }
        out[i * nPairs + q] = s;
      }
    }
    return out;
  }

  /** Class index chosen by one-vs-one voting (ties go to the smaller class index). */
  predictIndices(dec: Float64Array, n: number): Int32Array {
    const K = this.nClasses;
    const nPairs = this.pairs.length;
    const out = new Int32Array(n);
    const votes = new Int32Array(K);
    for (let i = 0; i < n; i++) {
      votes.fill(0);
      let q = 0;
      for (let a = 0; a < K; a++) {
        for (let b = a + 1; b < K; b++) {
          if ((dec[i * nPairs + q] as number) > 0) votes[a]!++;
          else votes[b]!++;
          q++;
        }
      }
      let best = 0;
      for (let c = 1; c < K; c++) if ((votes[c] as number) > (votes[best] as number)) best = c;
      out[i] = best;
    }
    return out;
  }

  /**
   * One-vs-rest scores built from the pairwise values: votes plus a confidence term in
   * (-1/3, 1/3) that only breaks ties, the same construction as scikit-learn's
   * `decision_function_shape="ovr"`.
   */
  ovrScores(dec: Float64Array, n: number): Float64Array {
    const K = this.nClasses;
    const nPairs = this.pairs.length;
    const out = new Float64Array(n * K);
    const conf = new Float64Array(K);
    for (let i = 0; i < n; i++) {
      conf.fill(0);
      let q = 0;
      for (let a = 0; a < K; a++) {
        for (let b = a + 1; b < K; b++) {
          const v = dec[i * nPairs + q] as number;
          conf[a]! += v;
          conf[b]! -= v;
          out[i * K + (v > 0 ? a : b)]! += 1;
          q++;
        }
      }
      for (let c = 0; c < K; c++) {
        const s = conf[c] as number;
        out[i * K + c]! += s / (3 * (Math.abs(s) + 1));
      }
    }
    return out;
  }

  /** `dual_coef_` in scikit-learn layout: `(1, n_SV)` for two classes, else `(n_classes - 1, n_SV)`. */
  dualCoefTensor(): Tensor {
    const nSv = this.nSupport;
    if (this.nClasses === 2) {
      const flipped = Float64Array.from(this.dualCoefLibsvm, (v) => -v);
      return matrixTensor(flipped, 1, nSv);
    }
    return matrixTensor(this.dualCoefLibsvm, this.nClasses - 1, nSv);
  }

  /** `intercept_` in scikit-learn layout: one entry per class pair. */
  interceptTensor(): Tensor {
    const binary = this.nClasses === 2;
    return tensor(
      Float64Array.from(this.pairs, (pr) => (binary ? pr.rho : -pr.rho)),
      { dtype: "float64" }
    );
  }

  supportVectorsTensor(): Tensor {
    return matrixTensor(this.supportX, this.nSupport, this.nFeatures);
  }
}

/**
 * Prediction methods shared by the one-vs-one classifiers.
 *
 * @internal
 */
export function ovoPredict(model: OvoModel, X: Tensor, name: string): Tensor {
  validatePredictInputs(X, model.nFeatures, name);
  const n = X.shape[0] ?? 0;
  const dec = model.pairDecisions(toFloat64View(X), n);
  const idx = model.predictIndices(dec, n);
  return labelsToTensor(Float64Array.from(idx, (c) => model.labels[c] as number));
}

/** @internal */
export function ovoDecisionFunction(model: OvoModel, X: Tensor, name: string): Tensor {
  validatePredictInputs(X, model.nFeatures, name);
  const n = X.shape[0] ?? 0;
  const dec = model.pairDecisions(toFloat64View(X), n);
  if (model.nClasses === 2) {
    return tensor(
      Float64Array.from(dec, (v) => -v),
      { dtype: "float64" }
    );
  }
  return matrixTensor(model.ovrScores(dec, n), n, model.nClasses);
}

/** @internal */
export function ovoPredictProba(model: OvoModel, X: Tensor, name: string): Tensor {
  validatePredictInputs(X, model.nFeatures, name);
  const n = X.shape[0] ?? 0;
  const dec = model.pairDecisions(toFloat64View(X), n);
  const K = model.nClasses;
  const out = new Float64Array(n * K);
  if (K === 2) {
    for (let i = 0; i < n; i++) {
      const p1 = stableSigmoid(-(dec[i] as number));
      out[i * 2] = 1 - p1;
      out[i * 2 + 1] = p1;
    }
    return matrixTensor(out, n, 2);
  }
  const scores = model.ovrScores(dec, n);
  for (let i = 0; i < n; i++) {
    let max = Number.NEGATIVE_INFINITY;
    for (let c = 0; c < K; c++) max = Math.max(max, scores[i * K + c] as number);
    let sum = 0;
    for (let c = 0; c < K; c++) {
      const e = Math.exp((scores[i * K + c] as number) - max);
      out[i * K + c] = e;
      sum += e;
    }
    for (let c = 0; c < K; c++) out[i * K + c] = (out[i * K + c] as number) / sum;
  }
  return matrixTensor(out, n, K);
}

// ---------------------------------------------------------------------------
// SVC
// ---------------------------------------------------------------------------

/** Constructor options of {@link SVC}. */
export type SVCOptions = {
  /** Penalty of margin violations, must be positive (default: 1.0). */
  readonly C?: number;
  /** Kernel function (default: "rbf"). */
  readonly kernel?: KernelType;
  /** Kernel coefficient of `rbf`, `poly` and `sigmoid` (default: "scale"). */
  readonly gamma?: GammaOption;
  /** Independent term of the `poly` and `sigmoid` kernels (default: 0). */
  readonly coef0?: number;
  /** Degree of the `poly` kernel, an integer >= 1 (default: 3). */
  readonly degree?: number;
  /**
   * Budget of SMO updates, expressed as passes over the data: the solver stops after at
   * most `maxIter * n` working-set updates per binary problem (default: 1000).
   */
  readonly maxIter?: number;
  /** Stopping tolerance on the maximal KKT violation (default: 1e-3). */
  readonly tol?: number;
  /** Per-class multiplier of `C`: "balanced" or a map from class label to weight. */
  readonly classWeight?: ClassWeightOption;
  /** Size of the kernel row cache in megabytes (default: 200). */
  readonly cacheSize?: number;
};

type SvcConfig = {
  C: number;
  kernel: KernelType;
  gamma: GammaOption;
  coef0: number;
  degree: number;
  maxIter: number;
  tol: number;
  classWeight: ClassWeightOption | undefined;
  cacheSize: number;
};

const SVC_KEYS = [
  "C",
  "kernel",
  "gamma",
  "coef0",
  "degree",
  "maxIter",
  "tol",
  "classWeight",
  "cacheSize",
] as const;

function normalizeSvcConfig(o: Record<string, unknown>): SvcConfig {
  return {
    C: optionOr(o, "C", 1.0, parseC),
    kernel: optionOr(o, "kernel", "rbf", parseKernel),
    gamma: optionOr<GammaOption>(o, "gamma", "scale", parseGamma),
    coef0: optionOr(o, "coef0", 0, parseCoef0),
    degree: optionOr(o, "degree", 3, parseDegree),
    maxIter: optionOr(o, "maxIter", 1000, parseMaxIter),
    tol: optionOr(o, "tol", 1e-3, parseTol),
    classWeight: o["classWeight"] === undefined ? undefined : parseClassWeight(o["classWeight"]),
    cacheSize: optionOr(o, "cacheSize", 200, parseCacheSize),
  };
}

/**
 * Support Vector Classification with the kernel trick.
 *
 * Solves the C-SVC dual with the LIBSVM working-set SMO algorithm and handles more than two
 * classes with one-vs-one voting, like scikit-learn's `SVC`. Supports RBF, polynomial,
 * sigmoid and linear kernels.
 *
 * `predictProba` returns a monotone squashing of the decision values, not calibrated
 * probabilities. Wrap the classifier in `CalibratedClassifierCV` when calibrated
 * probabilities matter.
 *
 * @example
 * ```ts
 * import { SVC } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0], [1, 1], [1, 0], [0, 1]]);
 * const y = tensor([0, 0, 1, 1]);
 *
 * const svc = new SVC({ kernel: 'rbf', C: 1.0 });
 * svc.fit(X, y);
 * const predictions = svc.predict(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-svm | Deepbox SVM}
 */
export class SVC implements Classifier {
  private cfg: SvcConfig;
  private model_: OvoModel | undefined;

  /**
   * @param options - Hyperparameters, see {@link SVCOptions}
   * @throws {InvalidParameterError} If an option is out of range
   */
  constructor(options: SVCOptions = {}) {
    this.cfg = normalizeSvcConfig(options as Record<string, unknown>);
  }

  private get fitted(): OvoModel {
    if (this.model_ === undefined) {
      throw new NotFittedError("SVC must be fitted before prediction");
    }
    return this.model_;
  }

  /**
   * Fit the classifier.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Class labels of shape (n_samples,), at least two distinct values
   * @param sampleWeight - Optional per-sample multipliers of `C`, shape (n_samples,)
   * @returns this
   * @throws {ShapeError} If X is not 2D, y is not 1D or the sample counts differ
   * @throws {DataValidationError} If X or y contain NaN/Inf or the kernel overflows
   * @throws {InvalidParameterError} If y has fewer than 2 classes
   */
  // biome-ignore lint/suspicious/noConfusingVoidType: `void` keeps this compatible with Estimator.fit(X, y, params)
  fit(X: Tensor, y: Tensor, sampleWeightArg?: Tensor | void): this {
    const sampleWeight = sampleWeightArg as Tensor | undefined;
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const Xf = toFloat64View(X);
    const { labels, index } = encodeLabels(toFloat64View(y));
    if (labels.length < 2) {
      throw new InvalidParameterError("SVC requires at least 2 classes", "y", labels.length);
    }
    const sw = readSampleWeight(sampleWeight, n);
    const cw = resolveClassWeights(this.cfg.classWeight, labels, index);
    const Cfull = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      Cfull[i] = this.cfg.C * (cw[index[i] as number] as number) * (sw ? (sw[i] as number) : 1);
    }
    const kp: SvmKernelParams = {
      kernel: this.cfg.kernel,
      gamma: resolveGamma(this.cfg.gamma, Xf, d),
      coef0: this.cfg.coef0,
      degree: this.cfg.degree,
    };

    const { model, converged } = OvoModel.fit(
      Xf,
      n,
      d,
      labels,
      index,
      kp,
      (Xsub, m, ySub, idx): PairSolution => {
        const Q = new ClassificationQ(Xsub, m, d, ySub, kp, this.cfg.cacheSize);
        const p = new Float64Array(m).fill(-1);
        const alpha = new Float64Array(m);
        const C = new Float64Array(m);
        for (let t = 0; t < m; t++) C[t] = Cfull[idx[t] as number] as number;
        const res = solveSmo(Q, p, ySub, alpha, C, this.cfg.tol, this.cfg.maxIter * m, false);
        const coef = new Float64Array(m);
        for (let t = 0; t < m; t++) coef[t] = (alpha[t] as number) * (ySub[t] as number);
        return { coef, rho: res.rho, iterations: res.iterations, converged: res.converged };
      }
    );
    if (!converged) warnNotConverged("SVC", this.cfg.maxIter);

    this.model_ = model;
    return this;
  }

  /**
   * Predict class labels by one-vs-one voting.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Labels of shape (n_samples,): int32 for integer classes, float64 otherwise
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    return ovoPredict(this.fitted, X, "SVC");
  }

  /**
   * Signed decision values.
   *
   * Two classes give shape (n_samples,) and a positive value means `classes[1]`. More classes
   * give shape (n_samples, n_classes) of one-vs-rest scores: one-vs-one votes plus a
   * confidence term in (-1/3, 1/3).
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @throws {NotFittedError} If the model has not been fitted
   */
  decisionFunction(X: Tensor): Tensor {
    return ovoDecisionFunction(this.fitted, X, "SVC");
  }

  /**
   * Class scores squashed into rows that sum to one: the logistic function of the decision
   * value for two classes, a softmax of the one-vs-rest scores otherwise. These are not
   * calibrated probabilities.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Shape (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   */
  predictProba(X: Tensor): Tensor {
    return ovoPredictProba(this.fitted, X, "SVC");
  }

  /**
   * Mean accuracy on the given data.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True labels of shape (n_samples,)
   * @returns Accuracy in [0, 1]
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional or the sample counts differ
   * @throws {DataValidationError} If y is empty or contains NaN/Inf
   */
  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    return accuracyOf(this.predict(X), y);
  }

  /** Sorted class labels seen during `fit`, or `undefined` before fitting. */
  get classes(): Tensor | undefined {
    return this.model_ === undefined ? undefined : labelsToTensor(this.model_.labels);
  }

  /** Support vectors, shape (n_SV, n_features), grouped by class. */
  get supportVectors(): Tensor {
    return this.fitted.supportVectorsTensor();
  }

  /** Indices of the support vectors in the training data, shape (n_SV,), dtype int32. */
  get supportIndices(): Tensor {
    return tensor(Int32Array.from(this.fitted.supportIndices), { dtype: "int32" });
  }

  /** Number of support vectors of each class, shape (n_classes,), dtype int32. */
  get nSupport(): Tensor {
    return tensor(Int32Array.from(this.fitted.nSupportPerClass), { dtype: "int32" });
  }

  /** Dual coefficients `alpha_i * y_i`, shape (n_classes - 1, n_SV), scikit-learn layout. */
  get dualCoef(): Tensor {
    return this.fitted.dualCoefTensor();
  }

  /** Decision-function offsets, one per class pair, scikit-learn sign convention. */
  get intercept(): Tensor {
    return this.fitted.interceptTensor();
  }

  /** Largest number of SMO updates used by any binary sub-problem. */
  get nIter(): number {
    return this.fitted.nIter;
  }

  /**
   * Get hyperparameters, including `classWeight` only when it is set.
   *
   * @returns Object that can be passed back to the constructor
   */
  getParams(): Record<string, unknown> {
    const { classWeight, ...rest } = this.cfg;
    return classWeight === undefined
      ? { ...rest }
      : { ...rest, classWeight: copyClassWeight(classWeight) };
  }

  /**
   * Set hyperparameters. The call is atomic: if any value is invalid nothing changes.
   *
   * @param params - Parameters to set
   * @throws {InvalidParameterError} If a parameter is unknown or its value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    this.cfg = normalizeSvcConfig(mergeParams(this.getParams(), params, SVC_KEYS, ["classWeight"]));
    return this;
  }

  /** Create an unfitted copy with the same hyperparameters. */
  clone(): SVC {
    return new SVC(this.getParams() as SVCOptions);
  }
}

// ---------------------------------------------------------------------------
// SVR
// ---------------------------------------------------------------------------

/** Constructor options of {@link SVR}. */
export type SVROptions = {
  /** Penalty of errors outside the epsilon tube, must be positive (default: 1.0). */
  readonly C?: number;
  /** Kernel function (default: "rbf"). */
  readonly kernel?: KernelType;
  /** Kernel coefficient of `rbf`, `poly` and `sigmoid` (default: "scale"). */
  readonly gamma?: GammaOption;
  /** Independent term of the `poly` and `sigmoid` kernels (default: 0). */
  readonly coef0?: number;
  /** Degree of the `poly` kernel, an integer >= 1 (default: 3). */
  readonly degree?: number;
  /** Half-width of the tube in which errors are not penalized (default: 0.1). */
  readonly epsilon?: number;
  /** Budget of SMO updates in passes over the 2 * n dual variables (default: 1000). */
  readonly maxIter?: number;
  /** Stopping tolerance on the maximal KKT violation (default: 1e-3). */
  readonly tol?: number;
  /** Size of the kernel row cache in megabytes (default: 200). */
  readonly cacheSize?: number;
};

type SvrConfig = {
  C: number;
  kernel: KernelType;
  gamma: GammaOption;
  coef0: number;
  degree: number;
  epsilon: number;
  maxIter: number;
  tol: number;
  cacheSize: number;
};

const SVR_KEYS = [
  "C",
  "kernel",
  "gamma",
  "coef0",
  "degree",
  "epsilon",
  "maxIter",
  "tol",
  "cacheSize",
] as const;

function normalizeSvrConfig(o: Record<string, unknown>): SvrConfig {
  return {
    C: optionOr(o, "C", 1.0, parseC),
    kernel: optionOr(o, "kernel", "rbf", parseKernel),
    gamma: optionOr<GammaOption>(o, "gamma", "scale", parseGamma),
    coef0: optionOr(o, "coef0", 0, parseCoef0),
    degree: optionOr(o, "degree", 3, parseDegree),
    epsilon: optionOr(o, "epsilon", 0.1, parseEpsilon),
    maxIter: optionOr(o, "maxIter", 1000, parseMaxIter),
    tol: optionOr(o, "tol", 1e-3, parseTol),
    cacheSize: optionOr(o, "cacheSize", 200, parseCacheSize),
  };
}

/**
 * Collapse the `2 * n` dual variables of a regression SVM into one coefficient per sample.
 *
 * @internal
 */
export function regressionCoefficients(alpha: Float64Array, n: number): Float64Array {
  const coef = new Float64Array(n);
  for (let i = 0; i < n; i++) coef[i] = (alpha[i] as number) - (alpha[i + n] as number);
  return coef;
}

/**
 * Tensor views of a fitted regression expansion (support vectors, indices, dual
 * coefficients and intercept), copied so callers cannot modify the model.
 *
 * @internal
 */
export function expansionTensors(model: KernelExpansion): {
  supportVectors: Tensor;
  supportIndices: Tensor;
  dualCoef: Tensor;
  intercept: Tensor;
} {
  return {
    supportVectors: matrixTensor(model.supportX, model.nSupport, model.nFeatures),
    supportIndices: tensor(Int32Array.from(model.supportIndices), { dtype: "int32" }),
    dualCoef: matrixTensor(model.coef, 1, model.nSupport),
    intercept: tensor(Float64Array.of(-model.rho), { dtype: "float64" }),
  };
}

/**
 * Predict with a fitted regression expansion: `sum_i coef_i K(sv_i, x) - rho`.
 *
 * @internal
 */
export function expansionPredict(model: KernelExpansion, X: Tensor, name: string): Tensor {
  validatePredictInputs(X, model.nFeatures, name);
  const n = X.shape[0] ?? 0;
  const f = model.raw(toFloat64View(X), n);
  for (let i = 0; i < n; i++) f[i] = (f[i] as number) - model.rho;
  return tensor(f, { dtype: "float64" });
}

/**
 * Epsilon-Support Vector Regression with the kernel trick.
 *
 * Solves the epsilon-SVR dual with the LIBSVM working-set SMO algorithm, so the result
 * agrees with scikit-learn's `SVR` up to the solver tolerance. Supports RBF, polynomial,
 * sigmoid and linear kernels.
 *
 * @example
 * ```ts
 * import { SVR } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1], [2], [3], [4], [5]]);
 * const y = tensor([1.2, 2.1, 2.9, 4.0, 5.1]);
 *
 * const svr = new SVR({ kernel: 'rbf', C: 10 });
 * svr.fit(X, y);
 * const predictions = svr.predict(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-svm | Deepbox SVM}
 */
export class SVR implements Regressor {
  private cfg: SvrConfig;
  private model_: KernelExpansion | undefined;

  /**
   * @param options - Hyperparameters, see {@link SVROptions}
   * @throws {InvalidParameterError} If an option is out of range
   */
  constructor(options: SVROptions = {}) {
    this.cfg = normalizeSvrConfig(options as Record<string, unknown>);
  }

  private get fitted(): KernelExpansion {
    if (this.model_ === undefined) {
      throw new NotFittedError("SVR must be fitted before prediction");
    }
    return this.model_;
  }

  /**
   * Fit the regressor.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Targets of shape (n_samples,)
   * @param sampleWeight - Optional per-sample multipliers of `C`, shape (n_samples,)
   * @returns this
   * @throws {ShapeError} If X is not 2D, y is not 1D or the sample counts differ
   * @throws {DataValidationError} If X or y contain NaN/Inf or the kernel overflows
   */
  // biome-ignore lint/suspicious/noConfusingVoidType: `void` keeps this compatible with Estimator.fit(X, y, params)
  fit(X: Tensor, y: Tensor, sampleWeightArg?: Tensor | void): this {
    const sampleWeight = sampleWeightArg as Tensor | undefined;
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const Xf = toFloat64View(X);
    const yv = toFloat64View(y);
    const sw = readSampleWeight(sampleWeight, n);
    const kp: SvmKernelParams = {
      kernel: this.cfg.kernel,
      gamma: resolveGamma(this.cfg.gamma, Xf, d),
      coef0: this.cfg.coef0,
      degree: this.cfg.degree,
    };

    const Q = new RegressionQ(Xf, n, d, kp, this.cfg.cacheSize);
    const p = new Float64Array(2 * n);
    const sign = new Int8Array(2 * n);
    const C = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) {
      const yi = yv[i] as number;
      p[i] = this.cfg.epsilon - yi;
      p[i + n] = this.cfg.epsilon + yi;
      sign[i] = 1;
      sign[i + n] = -1;
      const ci = this.cfg.C * (sw ? (sw[i] as number) : 1);
      C[i] = ci;
      C[i + n] = ci;
    }
    const res = solveSmo(
      Q,
      p,
      sign,
      new Float64Array(2 * n),
      C,
      this.cfg.tol,
      this.cfg.maxIter * 2 * n,
      false
    );
    if (!res.converged) warnNotConverged("SVR", this.cfg.maxIter);

    this.model_ = KernelExpansion.fromCoefficients(
      Xf,
      n,
      d,
      regressionCoefficients(res.alpha, n),
      res.rho,
      kp
    );
    return this;
  }

  /**
   * Predict target values.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predictions of shape (n_samples,), dtype float64
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    return expansionPredict(this.fitted, X, "SVR");
  }

  /**
   * Coefficient of determination R^2 on the given data.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True targets of shape (n_samples,)
   * @returns R^2 (1 is perfect, can be negative); a constant y scores 1 or 0
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional or the sample counts differ
   * @throws {DataValidationError} If y is empty or contains NaN/Inf
   */
  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    return r2Of(this.predict(X), y);
  }

  /** Support vectors, shape (n_SV, n_features). */
  get supportVectors(): Tensor {
    return expansionTensors(this.fitted).supportVectors;
  }

  /** Indices of the support vectors in the training data, shape (n_SV,), dtype int32. */
  get supportIndices(): Tensor {
    return expansionTensors(this.fitted).supportIndices;
  }

  /** Dual coefficients `alpha_i - alpha_i*`, shape (1, n_SV). */
  get dualCoef(): Tensor {
    return expansionTensors(this.fitted).dualCoef;
  }

  /** Intercept of the decision function, shape (1,). */
  get intercept(): Tensor {
    return expansionTensors(this.fitted).intercept;
  }

  /**
   * Get hyperparameters.
   *
   * @returns Object that can be passed back to the constructor
   */
  getParams(): Record<string, unknown> {
    return { ...this.cfg };
  }

  /**
   * Set hyperparameters. The call is atomic: if any value is invalid nothing changes.
   *
   * @param params - Parameters to set
   * @throws {InvalidParameterError} If a parameter is unknown or its value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    this.cfg = normalizeSvrConfig(mergeParams(this.getParams(), params, SVR_KEYS));
    return this;
  }

  /** Create an unfitted copy with the same hyperparameters. */
  clone(): SVR {
    return new SVR(this.getParams() as SVROptions);
  }
}
