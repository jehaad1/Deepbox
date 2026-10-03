/**
 * Ordinary least squares linear regression.
 *
 * @module ml/linear/LinearRegression
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { lstsq } from "../../linalg";
import { type Tensor, tensor } from "../../ndarray";
import {
  assertContiguous,
  toFloat64View,
  validateFitInputs,
  validatePredictInputs,
} from "../_validation";
import type { Regressor } from "../base";

/**
 * Coefficient of determination between a target tensor and the predictions of a model.
 *
 * `y` is validated first (1-D, contiguous, numeric, finite), then `predict` is called and
 * its length is checked against `y`. When `y` is constant the score is 1 for an exact fit
 * and 0 otherwise, matching `sklearn.metrics.r2_score`.
 *
 * @param y - True targets of shape (n_samples,)
 * @param predict - Produces the predictions to compare against
 * @returns R² score
 * @throws {ShapeError} If y is not 1-D or its length differs from the predictions
 * @throws {DataValidationError} If y is empty or contains NaN/Inf
 * @internal
 */
export function r2ScoreOf(y: Tensor, predict: () => Tensor): number {
  if (y.ndim !== 1) {
    throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
  }
  assertContiguous(y, "y");
  const yv = toFloat64View(y);
  const n = yv.length;
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(yv[i])) {
      throw new DataValidationError("y contains non-finite values (NaN or Inf)");
    }
  }
  if (n === 0) {
    throw new DataValidationError("y must contain at least one sample");
  }
  const pred = predict();
  if (pred.size !== n) {
    throw new ShapeError(
      `X and y must have the same number of samples; got X=${pred.size}, y=${n}`
    );
  }
  const pv = toFloat64View(pred);
  let yMean = 0;
  for (let i = 0; i < n; i++) yMean += yv[i] as number;
  yMean /= n;
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    const yi = yv[i] as number;
    const r = yi - (pv[i] as number);
    const t = yi - yMean;
    ssRes += r * r;
    ssTot += t * t;
  }
  if (ssTot === 0) return ssRes === 0 ? 1.0 : 0.0;
  return 1 - ssRes / ssTot;
}

/**
 * Ordinary Least Squares Linear Regression.
 *
 * Fits a linear model with coefficients w = (w1, ..., wp) to minimize
 * the residual sum of squares between the observed targets and the
 * targets predicted by the linear approximation. The system is solved with an
 * SVD-based least squares solver, so rank-deficient and under-determined problems
 * return the minimum-norm solution (as `numpy.linalg.lstsq` does).
 *
 * @example
 * ```ts
 * import { LinearRegression } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * // Create training data
 * const X = tensor([[1, 1], [1, 2], [2, 2], [2, 3]]);
 * const y = tensor([1, 2, 2, 3]);
 *
 * // Fit model
 * const model = new LinearRegression({ fitIntercept: true });
 * model.fit(X, y);
 *
 * // Make predictions
 * const X_test = tensor([[3, 5]]);
 * const predictions = model.predict(X_test);
 *
 * // Get R^2 score
 * const score = model.score(X, y);
 * ```
 */
export class LinearRegression implements Regressor {
  /** Model coefficients (weights) of shape (n_features,) */
  private coef_?: Tensor;

  /** Independent term (bias/intercept); undefined when fitIntercept is false */
  private intercept_: Tensor | undefined;

  /** Number of features seen during fit */
  private nFeaturesIn_?: number;

  /** Effective rank of the (centered) design matrix */
  private rank_ = 0;

  /** Singular values of the (centered) design matrix */
  private singular_?: Float64Array;

  /** Whether the model has been fitted */
  private fitted = false;

  private options: {
    fitIntercept?: boolean;
    normalize?: boolean;
    copyX?: boolean;
  };

  /**
   * Create a new Linear Regression model.
   *
   * @param options - Configuration options
   * @param options.fitIntercept - Whether to calculate the intercept (default: true)
   * @param options.normalize - Whether to scale every column to unit L2 norm before the fit.
   *   Columns are centered first when `fitIntercept` is true. Coefficients are reported in the
   *   original feature scale. (default: false)
   * @param options.copyX - When false, `fit` may overwrite `X` with its centered and scaled values
   *   if `X` is a float32 or float64 tensor (default: true)
   */
  constructor(
    options: {
      readonly fitIntercept?: boolean;
      readonly normalize?: boolean;
      readonly copyX?: boolean;
    } = {}
  ) {
    this.options = { ...options };
  }

  /**
   * Fit linear model using Ordinary Least Squares.
   *
   * Uses an SVD-based least squares solver for numerical stability.
   * When fitIntercept is true, centers the data before fitting.
   *
   * **Algorithm Complexity**: O(n * p^2) where n = samples, p = features
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target values of shape (n_samples,). Multi-output regression is not supported.
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D or y is not 1D
   * @throws {ShapeError} If X and y have different number of samples
   * @throws {DataValidationError} If X or y contain NaN/Inf values
   * @throws {DataValidationError} If X or y are empty
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);

    const m = X.shape[0] ?? 0;
    const n = X.shape[1] ?? 0;
    const fitIntercept = this.options.fitIntercept ?? true;
    const normalize = this.options.normalize ?? false;
    const copyX = this.options.copyX ?? true;
    const allowInPlace = !copyX && (X.dtype === "float32" || X.dtype === "float64");

    const xRaw = toFloat64View(X);
    const yRaw = toFloat64View(y);
    // A float64 view shares memory with X, so it can be overwritten directly.
    const xWork = allowInPlace && X.dtype === "float64" ? xRaw : Float64Array.from(xRaw);

    const xMean = new Float64Array(n);
    let yMean = 0;
    const yWork = Float64Array.from(yRaw);
    if (fitIntercept) {
      for (let i = 0; i < m; i++) {
        const base = i * n;
        for (let j = 0; j < n; j++) xMean[j] = (xMean[j] as number) + (xWork[base + j] as number);
        yMean += yWork[i] as number;
      }
      for (let j = 0; j < n; j++) xMean[j] = (xMean[j] as number) / m;
      yMean /= m;
      for (let i = 0; i < m; i++) {
        const base = i * n;
        for (let j = 0; j < n; j++) {
          xWork[base + j] = (xWork[base + j] as number) - (xMean[j] as number);
        }
        yWork[i] = (yWork[i] as number) - yMean;
      }
    }

    const scale = new Float64Array(n).fill(1);
    if (normalize) {
      const sumSq = new Float64Array(n);
      for (let i = 0; i < m; i++) {
        const base = i * n;
        for (let j = 0; j < n; j++)
          sumSq[j] = (sumSq[j] as number) + (xWork[base + j] as number) ** 2;
      }
      for (let j = 0; j < n; j++) scale[j] = Math.sqrt(sumSq[j] as number);
      for (let i = 0; i < m; i++) {
        const base = i * n;
        for (let j = 0; j < n; j++) {
          const s = scale[j] as number;
          xWork[base + j] = s === 0 ? 0 : (xWork[base + j] as number) / s;
        }
      }
    }

    const design = tensor(xWork, { dtype: "float64" }).reshape([m, n]);
    const result = lstsq(design, tensor(yWork, { dtype: "float64" }));

    if (allowInPlace && X.dtype === "float32") {
      // Honor copyX = false for float32 tensors, whose float64 view above is a copy.
      for (let k = 0; k < xWork.length; k++) X.data[X.offset + k] = xWork[k] as number;
    }

    const w = Float64Array.from(toFloat64View(result.x));
    let xMeanDotW = 0;
    for (let j = 0; j < n; j++) {
      const s = scale[j] as number;
      w[j] = s === 0 ? 0 : (w[j] as number) / s;
      xMeanDotW += (xMean[j] as number) * (w[j] as number);
    }

    this.nFeaturesIn_ = n;
    this.coef_ = tensor(w, { dtype: "float64" });
    this.intercept_ = fitIntercept ? tensor(yMean - xMeanDotW, { dtype: "float64" }) : undefined;
    this.rank_ = result.rank;
    this.singular_ = Float64Array.from(toFloat64View(result.s));
    this.fitted = true;
    return this;
  }

  /**
   * Predict using the linear model.
   *
   * Computes y_pred = X * coef_ + intercept_
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted values of shape (n_samples,) as a float64 tensor
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.coef_) {
      throw new NotFittedError("LinearRegression must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "LinearRegression");

    const m = X.shape[0] ?? 0;
    const n = X.shape[1] ?? 0;
    const xv = toFloat64View(X);
    const w = toFloat64View(this.coef_);
    const b =
      this.intercept_ === undefined ? 0 : Number(this.intercept_.data[this.intercept_.offset]);
    const out = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      let s = b;
      const base = i * n;
      for (let j = 0; j < n; j++) s += (xv[base + j] as number) * (w[j] as number);
      out[i] = s;
    }
    return tensor(out, { dtype: "float64" });
  }

  /**
   * Return the coefficient of determination R^2 of the prediction.
   *
   * R^2 = 1 - (SS_res / SS_tot)
   *
   * Where:
   * - SS_res = Σ(y_true - y_pred)^2 (residual sum of squares)
   * - SS_tot = Σ(y_true - y_mean)^2 (total sum of squares)
   *
   * Best possible score is 1.0, and it can be negative. A constant `y` scores
   * 1 when predicted exactly and 0 otherwise.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True target values of shape (n_samples,)
   * @returns R² score (best possible is 1.0, can be negative)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional or sample counts mismatch
   * @throws {DataValidationError} If y contains NaN/Inf values
   */
  score(X: Tensor, y: Tensor): number {
    if (!this.fitted) {
      throw new NotFittedError("LinearRegression must be fitted before scoring");
    }
    return r2ScoreOf(y, () => this.predict(X));
  }

  /**
   * Get the model coefficients (weights).
   *
   * @returns Coefficient tensor of shape (n_features,)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get coef(): Tensor {
    if (!this.fitted || !this.coef_) {
      throw new NotFittedError("LinearRegression must be fitted to access coefficients");
    }
    return this.coef_;
  }

  /**
   * Get the intercept (bias term).
   *
   * @returns 0-d tensor holding the intercept, or `undefined` when `fitIntercept` is false
   * @throws {NotFittedError} If the model has not been fitted
   */
  get intercept(): Tensor | undefined {
    if (!this.fitted) {
      throw new NotFittedError("LinearRegression must be fitted to access intercept");
    }
    return this.intercept_;
  }

  /**
   * The intercept as a plain number, the same type the other linear models return from
   * `intercept`. It is 0 when `fitIntercept` is false.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get interceptValue(): number {
    const intercept = this.intercept;
    if (intercept === undefined) return 0;
    return Number(intercept.data[intercept.offset] ?? 0);
  }

  /**
   * Effective rank of the design matrix used by the last fit.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get rank(): number {
    if (!this.fitted) {
      throw new NotFittedError("LinearRegression must be fitted to access rank");
    }
    return this.rank_;
  }

  /**
   * Singular values of the design matrix used by the last fit, in descending order.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get singular(): Float64Array {
    if (!this.fitted || !this.singular_) {
      throw new NotFittedError("LinearRegression must be fitted to access singular values");
    }
    return this.singular_;
  }

  /**
   * Number of features seen during `fit`.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nFeaturesIn(): number {
    if (!this.fitted) {
      throw new NotFittedError("LinearRegression must be fitted to access nFeaturesIn");
    }
    return this.nFeaturesIn_ ?? 0;
  }

  /**
   * Get parameters for this estimator.
   *
   * @returns Object containing all parameters
   */
  getParams(): Record<string, unknown> {
    return {
      fitIntercept: this.options.fitIntercept ?? true,
      normalize: this.options.normalize ?? false,
      copyX: this.options.copyX ?? true,
    };
  }

  /**
   * Set the parameters of this estimator. All values are validated before any is applied.
   *
   * @param params - Parameters to set
   * @returns this - The estimator
   * @throws {InvalidParameterError} If a name is unknown or a value is not a boolean
   */
  setParams(params: Record<string, unknown>): this {
    const next = { ...this.options };
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "fitIntercept":
        case "normalize":
        case "copyX":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError(
              `${key} must be a boolean; received ${String(value)}`,
              key,
              value
            );
          }
          next[key] = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    this.options = next;
    return this;
  }

  /** Create an unfitted copy with the same parameters. */
  clone(): LinearRegression {
    return new LinearRegression(
      this.getParams() as {
        fitIntercept?: boolean;
        normalize?: boolean;
        copyX?: boolean;
      }
    );
  }
}
