/**
 * Lasso regression (L1 penalty) fitted by coordinate descent.
 *
 * @module ml/linear/Lasso
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { toFloat64View, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Regressor } from "../base";
import {
  applyCoordinateDescentParam,
  fitCoordinateDescent,
  validateCoordinateDescentParams,
} from "./ElasticNet";
import { r2ScoreOf } from "./LinearRegression";

/** Constructor options of {@link Lasso}. */
export type LassoOptions = {
  /** L1 penalty strength, must be >= 0 (default: 1.0). */
  readonly alpha?: number;
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

/**
 * Lasso Regression (L1 Regularized Linear Regression).
 *
 * Lasso performs both regularization and feature selection by adding
 * an L1 penalty that can drive coefficients exactly to zero.
 *
 * **Objective**: minimize (1/(2*n)) ||y - Xw||² + α ||w||₁
 *
 * The solver and its stopping rule (coordinate updates followed by a duality gap check
 * scaled by `||y||²`) follow scikit-learn, so results agree with `sklearn.linear_model.Lasso`.
 *
 * @example
 * ```ts
 * import { Lasso } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 1], [1, 0], [2, 1], [3, 5]]);
 * const y = tensor([1, 3, 5, 7]);
 * const model = new Lasso({ alpha: 0.1, maxIter: 1000 });
 * model.fit(X, y);
 *
 * // Many coefficients will be exactly 0
 * console.log(model.coef);
 *
 * const predictions = model.predict(X);
 * ```
 *
 * @category Linear Models
 * @implements {Regressor}
 */
export class Lasso implements Regressor {
  /** Configuration options for the Lasso regression model */
  private options: {
    alpha?: number;
    fitIntercept?: boolean;
    normalize?: boolean;
    maxIter?: number;
    tol?: number;
    warmStart?: boolean;
    positive?: boolean;
    selection?: "cyclic" | "random";
    randomState?: number;
  };

  /** Model coefficients (weights) after fitting - shape (n_features,) */
  private coef_?: Tensor;

  /** Intercept (bias) term after fitting */
  private intercept_ = 0;

  /** Number of features seen during fit - used for validation */
  private nFeaturesIn_?: number;

  /** Number of iterations run by coordinate descent */
  private nIter_: number | undefined;

  /** Duality gap at the end of the last fit */
  private dualGap_: number | undefined;

  /** Whether the model has been fitted to data */
  private fitted = false;

  /**
   * Create a new Lasso Regression model.
   *
   * @param options - Configuration options
   * @param options.alpha - Regularization strength (default: 1.0). Must be >= 0. Controls sparsity of solution.
   * @param options.fitIntercept - Whether to calculate the intercept (default: true)
   * @param options.normalize - Whether to scale centered columns to unit L2 norm (default: false)
   * @param options.maxIter - Maximum passes of coordinate descent (default: 1000)
   * @param options.tol - Convergence tolerance (default: 1e-4). Smaller = more precise but slower.
   * @param options.warmStart - Whether to reuse previous solution as initialization (default: false)
   * @param options.positive - Whether to force coefficients to be non-negative (default: false)
   * @param options.selection - Coordinate order: 'cyclic' (default) or 'random'
   * @param options.randomState - Seed for the random coordinate order
   * @throws {InvalidParameterError} If `randomState` is not a finite number
   */
  constructor(options: LassoOptions = {}) {
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
   * Fit Lasso regression model using Coordinate Descent.
   *
   * Solves the L1-regularized least squares problem:
   * minimize (1/(2*n)) ||y - Xw||² + α||w||₁
   *
   * **Algorithm**: Coordinate Descent with Soft Thresholding
   * 1. Center (and optionally scale) the data, initialize coefficients (warm start if enabled)
   * 2. For each pass, visit every feature (cyclic or random order):
   *    - Compute the correlation of the feature with the partial residual
   *    - Apply the soft thresholding operator
   *    - Update the residual incrementally
   * 3. Stop when the duality gap is below `tol * ||y||²`
   *
   * **Time Complexity**: O(k * n * p) where k = iterations, n = samples, p = features
   * **Space Complexity**: O(n * p) for the column-major working copy of X
   *
   * Emits a `ConvergenceWarning` when `maxIter` passes were not enough to reach `tol`.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target values of shape (n_samples,)
   * @returns this - The fitted estimator for method chaining
   * @throws {ShapeError} If X is not 2D or y is not 1D
   * @throws {ShapeError} If X and y have different number of samples
   * @throws {DataValidationError} If X or y contain NaN/Inf values
   * @throws {DataValidationError} If X or y are empty
   * @throws {InvalidParameterError} If alpha < 0 or another hyper-parameter is out of range
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);

    const alpha = this.options.alpha ?? 1.0;
    const maxIter = this.options.maxIter ?? 1000;
    const tol = this.options.tol ?? 1e-4;
    const selection = this.options.selection ?? "cyclic";
    validateCoordinateDescentParams(alpha, 1, maxIter, tol, selection);

    const n = X.shape[1] ?? 0;
    let warmCoef: Float64Array | undefined;
    if (this.options.warmStart && this.coef_ && this.coef_.size === n) {
      warmCoef = Float64Array.from(toFloat64View(this.coef_));
    }

    const result = fitCoordinateDescent(X, y, {
      alpha,
      l1Ratio: 1,
      fitIntercept: this.options.fitIntercept ?? true,
      normalize: this.options.normalize ?? false,
      maxIter,
      tol,
      positive: this.options.positive ?? false,
      selection,
      randomState: this.options.randomState,
      warmCoef,
      modelName: "Lasso",
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
   * Get the model coefficients (weights).
   *
   * Many coefficients will be exactly zero due to L1 regularization (sparsity).
   *
   * @returns Coefficient tensor of shape (n_features,)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get coef(): Tensor {
    if (!this.fitted || !this.coef_) {
      throw new NotFittedError("Lasso must be fitted to access coefficients");
    }
    return this.coef_;
  }

  /**
   * Get the intercept (bias term).
   *
   * @returns Intercept value (0 when `fitIntercept` is false)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get intercept(): number {
    if (!this.fitted) {
      throw new NotFittedError("Lasso must be fitted to access intercept");
    }
    return this.intercept_;
  }

  /**
   * Get the number of passes run by coordinate descent in the last fit.
   *
   * @returns Number of passes until convergence (`maxIter` if it did not converge)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nIter(): number | undefined {
    if (!this.fitted) {
      throw new NotFittedError("Lasso must be fitted to access nIter");
    }
    return this.nIter_;
  }

  /**
   * Duality gap at the end of the last fit.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get dualGap(): number | undefined {
    if (!this.fitted) {
      throw new NotFittedError("Lasso must be fitted to access dualGap");
    }
    return this.dualGap_;
  }

  /**
   * Number of features seen during `fit`.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nFeaturesIn(): number {
    if (!this.fitted) {
      throw new NotFittedError("Lasso must be fitted to access nFeaturesIn");
    }
    return this.nFeaturesIn_ ?? 0;
  }

  /**
   * Predict using the Lasso regression model.
   *
   * Computes predictions as: ŷ = X @ coef + intercept
   *
   * **Time Complexity**: O(nm) where n = samples, m = features
   * **Space Complexity**: O(n)
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted values of shape (n_samples,) as a float64 tensor
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.coef_) {
      throw new NotFittedError("Lasso must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "Lasso");

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
   * Return the R² score on the given test data and target values.
   *
   * A constant `y` scores 1 when predicted exactly and 0 otherwise.
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
      throw new NotFittedError("Lasso must be fitted before scoring");
    }
    return r2ScoreOf(y, () => this.predict(X));
  }

  /**
   * Get hyperparameters for this estimator, with defaults filled in.
   *
   * The result can be passed back to the constructor to create an equivalent unfitted model.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    const params: Record<string, unknown> = {
      alpha: this.options.alpha ?? 1.0,
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
   * Set the parameters of this estimator. All values are validated before any is applied.
   *
   * @param params - Parameters to set (alpha, maxIter, tol, fitIntercept, normalize, warmStart,
   *   positive, selection, randomState)
   * @returns this
   * @throws {InvalidParameterError} If a name is unknown or a value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    const next = { ...this.options };
    for (const [key, value] of Object.entries(params)) {
      applyCoordinateDescentParam(next, key, value, false);
    }
    this.options = next;
    return this;
  }

  /** Create an unfitted copy with the same hyper-parameters. */
  clone(): Lasso {
    return new Lasso(this.getParams() as LassoOptions);
  }
}
