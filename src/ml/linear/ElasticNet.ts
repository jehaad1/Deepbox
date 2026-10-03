/**
 * Elastic Net regression (combined L1 and L2 penalty) fitted by coordinate descent.
 *
 * This file also hosts the coordinate descent solver that {@link Lasso} reuses,
 * since Lasso is Elastic Net with `l1Ratio = 1`.
 *
 * @module ml/linear/ElasticNet
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError, warn } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __random, __randomBelow } from "../../random/random";
import { toFloat64View, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Regressor } from "../base";
import { createTreeRng } from "../tree/DecisionTree";
import { r2ScoreOf } from "./LinearRegression";

/** Constructor options of {@link ElasticNet}. */
export type ElasticNetOptions = {
  /** Overall penalty strength, must be >= 0 (default: 1.0). */
  readonly alpha?: number;
  /** Mix between L1 and L2, in [0, 1]; 1 is pure Lasso, 0 is pure Ridge (default: 0.5). */
  readonly l1Ratio?: number;
  /** Fit an intercept term (default: true). */
  readonly fitIntercept?: boolean;
  /** Scale each (centered) column to unit L2 norm before fitting (default: false). */
  readonly normalize?: boolean;
  /** Maximum number of full passes over the coordinates (default: 1000). */
  readonly maxIter?: number;
  /** Convergence tolerance (default: 1e-4). */
  readonly tol?: number;
  /** Start from the coefficients of the previous `fit` call (default: false). */
  readonly warmStart?: boolean;
  /** Constrain the coefficients to be non-negative (default: false). */
  readonly positive?: boolean;
  /** Order in which coordinates are visited: `"cyclic"` or `"random"` (default: "cyclic"). */
  readonly selection?: "cyclic" | "random";
  /** Seed for `selection: "random"`; the global generator is used when omitted. */
  readonly randomState?: number;
};

/** Fully resolved coordinate descent settings. @internal */
export type CoordinateDescentConfig = {
  readonly alpha: number;
  readonly l1Ratio: number;
  readonly fitIntercept: boolean;
  readonly normalize: boolean;
  readonly maxIter: number;
  readonly tol: number;
  readonly positive: boolean;
  readonly selection: "cyclic" | "random";
  readonly randomState: number | undefined;
  /** Coefficients (original feature scale) to start from, or undefined for zeros. */
  readonly warmCoef: Float64Array | undefined;
  /** Model name used in warning messages. */
  readonly modelName: string;
};

/** Result of {@link fitCoordinateDescent}. @internal */
export type CoordinateDescentResult = {
  readonly coef: Float64Array;
  readonly intercept: number;
  readonly nIter: number;
  readonly dualGap: number;
  readonly converged: boolean;
};

/**
 * Check the numeric hyper-parameters shared by Lasso and Elastic Net.
 *
 * @throws {InvalidParameterError} If a value is outside its valid range
 * @internal
 */
export function validateCoordinateDescentParams(
  alpha: number,
  l1Ratio: number,
  maxIter: number,
  tol: number,
  selection: unknown
): void {
  if (!(alpha >= 0) || !Number.isFinite(alpha)) {
    throw new InvalidParameterError(
      `alpha must be >= 0 and finite; received ${alpha}`,
      "alpha",
      alpha
    );
  }
  if (!(l1Ratio >= 0 && l1Ratio <= 1)) {
    throw new InvalidParameterError(
      `l1Ratio must be in [0, 1]; received ${l1Ratio}`,
      "l1Ratio",
      l1Ratio
    );
  }
  if (!Number.isInteger(maxIter) || maxIter < 1) {
    throw new InvalidParameterError(
      `maxIter must be a positive integer; received ${maxIter}`,
      "maxIter",
      maxIter
    );
  }
  if (!(tol >= 0) || !Number.isFinite(tol)) {
    throw new InvalidParameterError(
      `tol must be a non-negative finite number; received ${tol}`,
      "tol",
      tol
    );
  }
  if (selection !== "cyclic" && selection !== "random") {
    throw new InvalidParameterError(
      `selection must be "cyclic" or "random"; received ${String(selection)}`,
      "selection",
      selection
    );
  }
}

/**
 * Validate and normalize one hyper-parameter update for Lasso / Elastic Net.
 *
 * @param target - Option bag that receives the validated value
 * @param key - Parameter name
 * @param value - Candidate value
 * @param allowL1Ratio - Whether `l1Ratio` is a known parameter (Elastic Net only)
 * @throws {InvalidParameterError} If the name is unknown or the value is invalid
 * @internal
 */
export function applyCoordinateDescentParam(
  target: {
    alpha?: number;
    l1Ratio?: number;
    fitIntercept?: boolean;
    normalize?: boolean;
    maxIter?: number;
    tol?: number;
    warmStart?: boolean;
    positive?: boolean;
    selection?: "cyclic" | "random";
    randomState?: number;
  },
  key: string,
  value: unknown,
  allowL1Ratio: boolean
): void {
  const requireBoolean = (): boolean => {
    if (typeof value !== "boolean") {
      throw new InvalidParameterError(`${key} must be a boolean`, key, value);
    }
    return value;
  };
  switch (key) {
    case "alpha":
      if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
        throw new InvalidParameterError("alpha must be a finite number >= 0", "alpha", value);
      }
      target.alpha = value;
      return;
    case "l1Ratio":
      if (!allowL1Ratio) break;
      if (typeof value !== "number" || !(value >= 0 && value <= 1)) {
        throw new InvalidParameterError("l1Ratio must be in [0, 1]", "l1Ratio", value);
      }
      target.l1Ratio = value;
      return;
    case "maxIter":
      if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
        throw new InvalidParameterError("maxIter must be a positive integer", "maxIter", value);
      }
      target.maxIter = value;
      return;
    case "tol":
      if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
        throw new InvalidParameterError("tol must be a non-negative finite number", "tol", value);
      }
      target.tol = value;
      return;
    case "fitIntercept":
      target.fitIntercept = requireBoolean();
      return;
    case "normalize":
      target.normalize = requireBoolean();
      return;
    case "warmStart":
      target.warmStart = requireBoolean();
      return;
    case "positive":
      target.positive = requireBoolean();
      return;
    case "selection":
      if (value !== "cyclic" && value !== "random") {
        throw new InvalidParameterError(
          `Invalid selection: ${String(value)}; expected "cyclic" or "random"`,
          "selection",
          value
        );
      }
      target.selection = value;
      return;
    case "randomState":
      if (value === undefined) {
        delete target.randomState;
        return;
      }
      if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new InvalidParameterError(
          `randomState must be a finite number; received ${String(value)}`,
          "randomState",
          value
        );
      }
      target.randomState = value;
      return;
    default:
      break;
  }
  throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
}

/**
 * Minimize `1/(2n) ||y - Xw||^2 + alpha * l1Ratio * ||w||_1 + alpha * (1 - l1Ratio) / 2 * ||w||^2`
 * by cyclic or randomized coordinate descent.
 *
 * Follows the scikit-learn solver: the residual is updated incrementally, a pass is
 * considered finished when the largest coefficient update is small relative to the
 * largest coefficient, and the final decision uses the duality gap scaled by `||y||^2`.
 *
 * @internal
 */
export function fitCoordinateDescent(
  X: Tensor,
  y: Tensor,
  config: CoordinateDescentConfig
): CoordinateDescentResult {
  const m = X.shape[0] ?? 0;
  const n = X.shape[1] ?? 0;
  const xRaw = toFloat64View(X);
  const yRaw = toFloat64View(y);
  const { fitIntercept, normalize, positive, maxIter, tol } = config;

  // Means (two-pass for accuracy on data with a large offset).
  const xMean = new Float64Array(n);
  let yMean = 0;
  if (fitIntercept) {
    for (let i = 0; i < m; i++) {
      const base = i * n;
      for (let j = 0; j < n; j++) xMean[j] = (xMean[j] as number) + (xRaw[base + j] as number);
      yMean += yRaw[i] as number;
    }
    for (let j = 0; j < n; j++) xMean[j] = (xMean[j] as number) / m;
    yMean /= m;
  }

  // Column-major copy of the centered (and optionally scaled) design matrix so that every
  // coordinate update walks a contiguous slice.
  const xc = new Float64Array(n * m);
  const colScale = new Float64Array(n).fill(1);
  for (let j = 0; j < n; j++) {
    const mu = xMean[j] as number;
    const off = j * m;
    let sumSq = 0;
    for (let i = 0; i < m; i++) {
      const v = (xRaw[i * n + j] as number) - mu;
      xc[off + i] = v;
      sumSq += v * v;
    }
    if (normalize) {
      const s = Math.sqrt(sumSq);
      if (s > 0) {
        colScale[j] = s;
        for (let i = 0; i < m; i++) xc[off + i] = (xc[off + i] as number) / s;
      } else {
        xc.fill(0, off, off + m);
      }
    }
  }
  const yc = new Float64Array(m);
  for (let i = 0; i < m; i++) yc[i] = (yRaw[i] as number) - yMean;

  const normCols = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    const off = j * m;
    let s = 0;
    for (let i = 0; i < m; i++) s += (xc[off + i] as number) ** 2;
    normCols[j] = s;
  }

  const l1Reg = config.alpha * config.l1Ratio * m;
  const l2Reg = config.alpha * (1 - config.l1Ratio) * m;

  // Initial point and residual R = y - Xw.
  const w = new Float64Array(n);
  const warm = config.warmCoef;
  if (warm !== undefined && warm.length === n) {
    for (let j = 0; j < n; j++)
      w[j] = (warm[j] as number) * (normalize ? (colScale[j] as number) : 1);
  }
  const resid = Float64Array.from(yc);
  for (let j = 0; j < n; j++) {
    const wj = w[j] as number;
    if (wj === 0) continue;
    const off = j * m;
    for (let i = 0; i < m; i++) {
      resid[i] = (resid[i] as number) - wj * (xc[off + i] as number);
    }
  }

  let yy = 0;
  for (let i = 0; i < m; i++) yy += (yc[i] as number) ** 2;
  const gapTol = tol * yy;

  const rng: () => number =
    config.selection === "random" ? createTreeRng(config.randomState) : __random;
  const order = new Int32Array(n);
  for (let j = 0; j < n; j++) order[j] = j;

  let nIter = 0;
  let dualGap = Number.POSITIVE_INFINITY;
  let converged = false;

  for (let iter = 0; iter < maxIter; iter++) {
    nIter = iter + 1;
    if (config.selection === "random") {
      for (let k = n - 1; k > 0; k--) {
        const r = __randomBelow(rng, k + 1);
        const tmp = order[k] as number;
        order[k] = order[r] as number;
        order[r] = tmp;
      }
    }

    let wMax = 0;
    let dwMax = 0;
    for (let idx = 0; idx < n; idx++) {
      const j = order[idx] as number;
      const off = j * m;
      const wOld = w[j] as number;
      const nc = normCols[j] as number;
      let wNew = 0;
      if (nc !== 0) {
        // rho = x_j . (R + w_j x_j) without touching R first.
        let dotXR = 0;
        for (let i = 0; i < m; i++) dotXR += (xc[off + i] as number) * (resid[i] as number);
        const tmp = dotXR + wOld * nc;
        if (!(positive && tmp < 0)) {
          if (tmp > l1Reg) wNew = (tmp - l1Reg) / (nc + l2Reg);
          else if (tmp < -l1Reg) wNew = (tmp + l1Reg) / (nc + l2Reg);
        }
      }
      const delta = wNew - wOld;
      if (delta !== 0) {
        for (let i = 0; i < m; i++) {
          resid[i] = (resid[i] as number) - delta * (xc[off + i] as number);
        }
      }
      w[j] = wNew;
      const absDelta = Math.abs(delta);
      if (absDelta > dwMax) dwMax = absDelta;
      const absW = Math.abs(wNew);
      if (absW > wMax) wMax = absW;
    }

    if (wMax === 0 || dwMax / wMax <= tol || iter === maxIter - 1) {
      dualGap = elasticNetDualGap(xc, yc, w, resid, m, n, l1Reg, l2Reg, positive);
      if (dualGap <= gapTol) {
        converged = true;
        break;
      }
    }
  }

  if (!converged) {
    warn(
      `Coordinate descent did not converge within ${maxIter} iterations (duality gap ${dualGap.toExponential(3)}, tolerance ${gapTol.toExponential(3)}); increase maxIter or tol`,
      "ConvergenceWarning",
      config.modelName
    );
  }

  const coef = new Float64Array(n);
  let xMeanDotW = 0;
  for (let j = 0; j < n; j++) {
    coef[j] = (w[j] as number) / (colScale[j] as number);
    xMeanDotW += (xMean[j] as number) * (coef[j] as number);
  }
  const intercept = fitIntercept ? yMean - xMeanDotW : 0;
  return { coef, intercept, nIter, dualGap, converged };
}

/** Duality gap of the Elastic Net problem (scikit-learn's `enet_coordinate_descent`). */
function elasticNetDualGap(
  xc: Float64Array,
  yc: Float64Array,
  w: Float64Array,
  resid: Float64Array,
  m: number,
  n: number,
  l1Reg: number,
  l2Reg: number,
  positive: boolean
): number {
  let dualNormXtA = 0;
  for (let j = 0; j < n; j++) {
    const off = j * m;
    let s = 0;
    for (let i = 0; i < m; i++) s += (xc[off + i] as number) * (resid[i] as number);
    s -= l2Reg * (w[j] as number);
    const v = positive ? s : Math.abs(s);
    if (v > dualNormXtA) dualNormXtA = v;
  }
  let rNorm2 = 0;
  let ry = 0;
  for (let i = 0; i < m; i++) {
    const r = resid[i] as number;
    rNorm2 += r * r;
    ry += r * (yc[i] as number);
  }
  let wNorm2 = 0;
  let l1Norm = 0;
  for (let j = 0; j < n; j++) {
    const wj = w[j] as number;
    wNorm2 += wj * wj;
    l1Norm += Math.abs(wj);
  }
  let scale = 1;
  let gap = rNorm2;
  if (dualNormXtA > l1Reg) {
    scale = l1Reg / dualNormXtA;
    gap = 0.5 * (rNorm2 + rNorm2 * scale * scale);
  }
  return gap + l1Reg * l1Norm - scale * ry + 0.5 * l2Reg * (1 + scale * scale) * wNorm2;
}

/**
 * Elastic Net Regression (L1 + L2 Regularized Linear Regression).
 *
 * Elastic Net combines L1 (Lasso) and L2 (Ridge) penalties, controlled
 * by the `l1Ratio` parameter. It is more stable than Lasso when features are
 * correlated, while still driving some coefficients exactly to zero.
 *
 * **Objective**: minimize (1/(2*n)) ||y - Xw||² + α * l1Ratio * ||w||₁ + α * (1-l1Ratio)/2 * ||w||²
 *
 * - `l1Ratio = 1` → pure Lasso (L1 only)
 * - `l1Ratio = 0` → pure Ridge (L2 only, with the penalty scaled by 1/n relative to {@link Ridge})
 * - `0 < l1Ratio < 1` → mix of L1 and L2
 *
 * The solver and its stopping rule (coordinate updates followed by a duality gap check
 * scaled by `||y||²`) follow scikit-learn, so results agree with `sklearn.linear_model.ElasticNet`.
 * With `normalize: true` every centered column is divided by its L2 norm before fitting and the
 * coefficients are mapped back to the original scale afterwards.
 *
 * @example
 * ```ts
 * import { ElasticNet } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [2, 1], [3, 5], [4, 3]]);
 * const y = tensor([1, 2, 3, 5]);
 * const model = new ElasticNet({ alpha: 0.1, l1Ratio: 0.5 });
 * model.fit(X, y);
 * const predictions = model.predict(X);
 * ```
 *
 * @category Linear Models
 * @implements {Regressor}
 */
export class ElasticNet implements Regressor {
  private options: {
    alpha?: number;
    l1Ratio?: number;
    fitIntercept?: boolean;
    normalize?: boolean;
    maxIter?: number;
    tol?: number;
    warmStart?: boolean;
    positive?: boolean;
    selection?: "cyclic" | "random";
    randomState?: number;
  };

  private coef_?: Tensor;
  private intercept_ = 0;
  private nFeaturesIn_?: number;
  private nIter_: number | undefined;
  private dualGap_: number | undefined;
  private fitted = false;

  /**
   * Create a new Elastic Net model.
   *
   * @param options - Configuration options
   * @param options.alpha - Regularization strength (default: 1.0). Must be >= 0.
   * @param options.l1Ratio - Mix ratio between L1 and L2 (default: 0.5). 0 = Ridge, 1 = Lasso.
   * @param options.fitIntercept - Whether to calculate the intercept (default: true)
   * @param options.normalize - Whether to scale centered columns to unit L2 norm (default: false)
   * @param options.maxIter - Maximum passes of coordinate descent (default: 1000)
   * @param options.tol - Convergence tolerance (default: 1e-4)
   * @param options.warmStart - Reuse the previous solution as initialization (default: false)
   * @param options.positive - Force coefficients to be non-negative (default: false)
   * @param options.selection - Coordinate order: 'cyclic' or 'random' (default: 'cyclic')
   * @param options.randomState - Seed for the random coordinate order
   * @throws {InvalidParameterError} If `randomState` is not a finite number
   */
  constructor(options: ElasticNetOptions = {}) {
    this.options = { ...options };
    if (this.options.randomState !== undefined && !Number.isFinite(this.options.randomState)) {
      throw new InvalidParameterError(
        `randomState must be a finite number; received ${String(this.options.randomState)}`,
        "randomState",
        this.options.randomState
      );
    }
  }

  /**
   * Fit Elastic Net model using coordinate descent.
   *
   * Solves: minimize (1/(2*n)) ||y - Xw||² + α * l1Ratio * ||w||₁ + α * (1-l1Ratio)/2 * ||w||²
   *
   * Emits a `ConvergenceWarning` when `maxIter` passes were not enough to reach `tol`.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target values of shape (n_samples,)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D, y is not 1D, or their sample counts differ
   * @throws {DataValidationError} If X or y are empty or contain NaN/Inf
   * @throws {InvalidParameterError} If a hyper-parameter is out of range
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);

    const alpha = this.options.alpha ?? 1.0;
    const l1Ratio = this.options.l1Ratio ?? 0.5;
    const maxIter = this.options.maxIter ?? 1000;
    const tol = this.options.tol ?? 1e-4;
    const selection = this.options.selection ?? "cyclic";
    validateCoordinateDescentParams(alpha, l1Ratio, maxIter, tol, selection);

    const n = X.shape[1] ?? 0;
    let warmCoef: Float64Array | undefined;
    if (this.options.warmStart && this.coef_ && this.coef_.size === n) {
      warmCoef = Float64Array.from(toFloat64View(this.coef_));
    }

    const result = fitCoordinateDescent(X, y, {
      alpha,
      l1Ratio,
      fitIntercept: this.options.fitIntercept ?? true,
      normalize: this.options.normalize ?? false,
      maxIter,
      tol,
      positive: this.options.positive ?? false,
      selection,
      randomState: this.options.randomState,
      warmCoef,
      modelName: "ElasticNet",
    });

    this.nFeaturesIn_ = n;
    this.coef_ = tensor(result.coef, { dtype: "float64" });
    this.intercept_ = result.intercept;
    this.nIter_ = result.nIter;
    this.dualGap_ = result.dualGap;
    this.fitted = true;
    return this;
  }

  /**
   * Predict using the Elastic Net model.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predictions of shape (n_samples,) as a float64 tensor
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X is not 2D or has the wrong number of features
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.coef_) {
      throw new NotFittedError("ElasticNet must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "ElasticNet");

    const m = X.shape[0] ?? 0;
    const n = X.shape[1] ?? 0;
    const xv = toFloat64View(X);
    const w = toFloat64View(this.coef_);
    const pred = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      let s = this.intercept_;
      const base = i * n;
      for (let j = 0; j < n; j++) s += (xv[base + j] as number) * (w[j] as number);
      pred[i] = s;
    }
    return tensor(pred, { dtype: "float64" });
  }

  /**
   * Return the R² score on the given test data.
   *
   * A constant `y` scores 1 when predicted exactly and 0 otherwise.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True target values of shape (n_samples,)
   * @returns R² score (1 is perfect, can be negative)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-D or its length differs from the number of samples
   * @throws {DataValidationError} If y contains NaN/Inf
   */
  score(X: Tensor, y: Tensor): number {
    if (!this.fitted) {
      throw new NotFittedError("ElasticNet must be fitted before scoring");
    }
    return r2ScoreOf(y, () => this.predict(X));
  }

  /**
   * Coefficients of shape (n_features,) in the original feature scale.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get coef(): Tensor {
    if (!this.fitted || !this.coef_) {
      throw new NotFittedError("ElasticNet must be fitted to access coefficients");
    }
    return this.coef_;
  }

  /**
   * Intercept term (0 when `fitIntercept` is false).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get intercept(): number {
    if (!this.fitted) {
      throw new NotFittedError("ElasticNet must be fitted to access intercept");
    }
    return this.intercept_;
  }

  /**
   * Number of coordinate descent passes run by the last `fit`.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nIter(): number | undefined {
    if (!this.fitted) {
      throw new NotFittedError("ElasticNet must be fitted to access nIter");
    }
    return this.nIter_;
  }

  /**
   * Duality gap at the end of the last `fit` (an upper bound on the suboptimality of the
   * objective, in units of `n` times the objective).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get dualGap(): number | undefined {
    if (!this.fitted) {
      throw new NotFittedError("ElasticNet must be fitted to access dualGap");
    }
    return this.dualGap_;
  }

  /** Number of features seen during `fit`. */
  get nFeaturesIn(): number {
    if (!this.fitted) {
      throw new NotFittedError("ElasticNet must be fitted to access nFeaturesIn");
    }
    return this.nFeaturesIn_ ?? 0;
  }

  /**
   * Hyper-parameters of this estimator, with defaults filled in.
   *
   * The result can be passed back to the constructor to create an equivalent unfitted model.
   */
  getParams(): Record<string, unknown> {
    const params: Record<string, unknown> = {
      alpha: this.options.alpha ?? 1.0,
      l1Ratio: this.options.l1Ratio ?? 0.5,
      fitIntercept: this.options.fitIntercept ?? true,
      normalize: this.options.normalize ?? false,
      maxIter: this.options.maxIter ?? 1000,
      tol: this.options.tol ?? 1e-4,
      warmStart: this.options.warmStart ?? false,
      positive: this.options.positive ?? false,
      selection: this.options.selection ?? "cyclic",
    };
    if (this.options.randomState !== undefined) params["randomState"] = this.options.randomState;
    return params;
  }

  /**
   * Set hyper-parameters. All values are validated before any of them is applied.
   *
   * @param params - Parameters to change
   * @returns this
   * @throws {InvalidParameterError} If a name is unknown or a value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    const next = { ...this.options };
    for (const [key, value] of Object.entries(params)) {
      applyCoordinateDescentParam(next, key, value, true);
    }
    this.options = next;
    return this;
  }

  /** Create an unfitted copy with the same hyper-parameters. */
  clone(): ElasticNet {
    return new ElasticNet(this.getParams() as ElasticNetOptions);
  }
}
