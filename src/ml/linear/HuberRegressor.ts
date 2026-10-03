/**
 * Huber regression: a linear model fitted with Huber loss.
 *
 * Uses Huber loss instead of squared loss, which is quadratic for small
 * residuals and linear for large residuals (outliers). The `epsilon`
 * parameter controls the threshold between the two regimes.
 *
 * @module ml/linear/HuberRegressor
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError, warn } from "../../core";
import { lstsq } from "../../linalg";
import { type Tensor, tensor } from "../../ndarray";
import { toFloat64View, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Regressor } from "../base";
import { r2ScoreOf } from "./LinearRegression";

/** Constructor options of {@link HuberRegressor}. */
export type HuberRegressorOptions = {
  /** Threshold between the quadratic and linear regimes, in units of the scale; must be > 1 (default: 1.35). */
  readonly epsilon?: number;
  /** Strength of the squared L2 penalty on the coefficients, must be >= 0 (default: 0.0001). */
  readonly alpha?: number;
  /** Maximum number of reweighting iterations (default: 100). */
  readonly maxIter?: number;
  /** Convergence tolerance relative to the standard deviation of y (default: 1e-5). */
  readonly tol?: number;
  /** Fit an intercept term (default: true). */
  readonly fitIntercept?: boolean;
  /**
   * Ignored. Earlier versions fitted by plain gradient descent with this step size; the solver
   * no longer needs one. The option is still accepted so existing code keeps working.
   *
   * @deprecated Has no effect.
   */
  readonly learningRate?: number;
};

type ResolvedOptions = {
  epsilon: number;
  alpha: number;
  maxIter: number;
  tol: number;
  fitIntercept: boolean;
  learningRate: number;
};

const OPTION_KEYS: readonly string[] = [
  "epsilon",
  "alpha",
  "maxIter",
  "tol",
  "fitIntercept",
  "learningRate",
];

/** Smallest scale the estimator is allowed to take (same bound as scikit-learn). */
const MIN_SCALE = 10 * Number.EPSILON;

function resolveOptions(options: HuberRegressorOptions): ResolvedOptions {
  const resolved: ResolvedOptions = {
    epsilon: options.epsilon ?? 1.35,
    alpha: options.alpha ?? 0.0001,
    maxIter: options.maxIter ?? 100,
    tol: options.tol ?? 1e-5,
    fitIntercept: options.fitIntercept ?? true,
    learningRate: options.learningRate ?? 0.01,
  };
  if (!(resolved.epsilon > 1.0) || !Number.isFinite(resolved.epsilon)) {
    throw new InvalidParameterError(
      "epsilon must be a finite number > 1.0",
      "epsilon",
      resolved.epsilon
    );
  }
  if (!(resolved.alpha >= 0) || !Number.isFinite(resolved.alpha)) {
    throw new InvalidParameterError("alpha must be >= 0", "alpha", resolved.alpha);
  }
  if (!Number.isInteger(resolved.maxIter) || resolved.maxIter < 1) {
    throw new InvalidParameterError(
      "maxIter must be a positive integer",
      "maxIter",
      resolved.maxIter
    );
  }
  if (!(resolved.tol >= 0) || !Number.isFinite(resolved.tol)) {
    throw new InvalidParameterError(
      "tol must be a non-negative finite number",
      "tol",
      resolved.tol
    );
  }
  if (!Number.isFinite(resolved.learningRate)) {
    throw new InvalidParameterError(
      "learningRate must be a finite number",
      "learningRate",
      resolved.learningRate
    );
  }
  return resolved;
}

/**
 * Value of the scale `sigma` that minimizes the concomitant Huber objective for fixed
 * absolute residuals.
 *
 * With `f(sigma) = n sigma + sum_{|r| <= eps sigma} r^2 / sigma + sum_{|r| > eps sigma} (2 eps |r| - eps^2 sigma)`
 * the derivative `n - sum_in r^2 / sigma^2 - eps^2 n_out` is non-decreasing, negative as
 * sigma -> 0 and non-negative at `sqrt(sum r^2 / n)`, so bisection finds the root.
 */
function solveScale(absResid: Float64Array, epsilon: number): number {
  const n = absResid.length;
  let ss = 0;
  for (let i = 0; i < n; i++) ss += (absResid[i] as number) ** 2;
  let hi = Math.sqrt(ss / n);
  if (!(hi > MIN_SCALE)) return MIN_SCALE;
  let lo = 0;
  const eps2 = epsilon * epsilon;
  for (let it = 0; it < 200 && hi - lo > 1e-15 * hi; it++) {
    const mid = 0.5 * (lo + hi);
    const bound = epsilon * mid;
    let inSS = 0;
    let nOut = 0;
    for (let i = 0; i < n; i++) {
      const r = absResid[i] as number;
      if (r <= bound) inSS += r * r;
      else nOut++;
    }
    const g = n - inSS / (mid * mid) - eps2 * nOut;
    if (g < 0) lo = mid;
    else hi = mid;
  }
  return Math.max(MIN_SCALE, 0.5 * (lo + hi));
}

/**
 * Linear regression with Huber loss, which limits the influence of outliers.
 *
 * Minimizes, over the coefficients `w`, the intercept `c` and a scale `sigma`,
 * `sum_i [ sigma + H(r_i / sigma) * sigma ] + alpha * ||w||^2` with `r_i = y_i - x_i . w - c` and
 * the Huber function `H(z) = z^2` for `|z| <= epsilon` and `2 epsilon |z| - epsilon^2` otherwise.
 * Because the scale is estimated jointly, `epsilon` is relative to the noise level and the fit does
 * not depend on the units of `y`. This is the objective of `sklearn.linear_model.HuberRegressor`.
 *
 * The problem is convex. It is solved by iteratively reweighted ridge regression (a
 * majorize-minimize scheme that never increases the objective) alternated with an exact update of
 * the scale.
 *
 * @example
 * ```ts
 * import { HuberRegressor } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1], [2], [3], [4], [100]]);
 * const y = tensor([2, 4, 6, 8, 200]);
 * const reg = new HuberRegressor({ epsilon: 1.35 });
 * reg.fit(X, y);
 * console.log(reg.coef); // close to 2
 * ```
 */
export class HuberRegressor implements Regressor {
  private opts: ResolvedOptions;

  private coef_?: Float64Array;
  private intercept_ = 0;
  private scale_ = 0;
  private nFeaturesIn_ = 0;
  private nIter_ = 0;
  private outliers_?: boolean[];
  private fitted = false;

  /**
   * Create a new Huber regressor.
   *
   * @param options - Configuration options, see {@link HuberRegressorOptions}
   * @throws {InvalidParameterError} If an option is outside its valid range
   */
  constructor(options: HuberRegressorOptions = {}) {
    this.opts = resolveOptions(options);
  }

  /**
   * Fit the model.
   *
   * Emits a `ConvergenceWarning` when `maxIter` iterations were not enough to reach `tol`.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target values of shape (n_samples,)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D, y is not 1D, or their sample counts differ
   * @throws {DataValidationError} If X or y are empty or contain NaN/Inf
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const p = X.shape[1] ?? 0;
    const { epsilon, alpha, maxIter, tol, fitIntercept } = this.opts;
    const q = fitIntercept ? p + 1 : p;

    const xv = toFloat64View(X);
    const yRaw = toFloat64View(y);

    // With an intercept, centering X and y changes neither the coefficients nor the residuals
    // (the intercept is not penalized) but keeps the least squares systems well conditioned when
    // the data has a large offset.
    const xMean = new Float64Array(p);
    let yShift = 0;
    if (fitIntercept) {
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < p; j++) xMean[j] = (xMean[j] as number) + (xv[i * p + j] as number);
        yShift += yRaw[i] as number;
      }
      for (let j = 0; j < p; j++) xMean[j] = (xMean[j] as number) / n;
      yShift /= n;
    }
    const yv = new Float64Array(n);
    for (let i = 0; i < n; i++) yv[i] = (yRaw[i] as number) - yShift;

    // Design matrix with the intercept as the last, unpenalized column.
    const A = new Float64Array(n * q);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < p; j++) A[i * q + j] = (xv[i * p + j] as number) - (xMean[j] as number);
      if (fitIntercept) A[i * q + p] = 1;
    }

    let yMean = 0;
    for (let i = 0; i < n; i++) yMean += yv[i] as number;
    yMean /= n;
    let yVar = 0;
    for (let i = 0; i < n; i++) yVar += ((yv[i] as number) - yMean) ** 2;
    const yStd = Math.sqrt(yVar / n);
    const stepTol = tol * (yStd > 0 ? yStd : 1);

    const residuals = (theta: Float64Array): Float64Array => {
      const r = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let pred = 0;
        for (let j = 0; j < q; j++) pred += (A[i * q + j] as number) * (theta[j] as number);
        r[i] = (yv[i] as number) - pred;
      }
      return r;
    };
    const absOf = (r: Float64Array): Float64Array => {
      const a = new Float64Array(n);
      for (let i = 0; i < n; i++) a[i] = Math.abs(r[i] as number);
      return a;
    };

    // Start from ordinary least squares.
    let theta = Float64Array.from(
      toFloat64View(
        lstsq(
          tensor(A, { dtype: "float64" }).reshape([n, q]),
          tensor(yv.slice(), { dtype: "float64" })
        ).x
      )
    );
    let resid = residuals(theta);
    let scale = solveScale(absOf(resid), epsilon);

    // Rows: n weighted samples plus one penalty row per penalized coefficient.
    const nPen = alpha > 0 ? p : 0;
    const rows = n + nPen;
    let nIter = 0;
    let converged = false;
    for (let iter = 0; iter < maxIter; iter++) {
      nIter = iter + 1;
      const bound = epsilon * scale;
      const B = new Float64Array(rows * q);
      const rhs = new Float64Array(rows);
      for (let i = 0; i < n; i++) {
        const r = Math.abs(resid[i] as number);
        // Weight 1 inside the quadratic zone, eps * sigma / |r| outside (majorizer of |r|).
        const sw = r <= bound ? 1 : Math.sqrt(bound / r);
        for (let j = 0; j < q; j++) B[i * q + j] = sw * (A[i * q + j] as number);
        rhs[i] = sw * (yv[i] as number);
      }
      const penSqrt = Math.sqrt(alpha * scale);
      for (let j = 0; j < nPen; j++) B[(n + j) * q + j] = penSqrt;

      const next = Float64Array.from(
        toFloat64View(
          lstsq(
            tensor(B, { dtype: "float64" }).reshape([rows, q]),
            tensor(rhs, { dtype: "float64" })
          ).x
        )
      );
      const nextResid = residuals(next);
      const nextScale = solveScale(absOf(nextResid), epsilon);

      let step = Math.abs(nextScale - scale);
      for (let i = 0; i < n; i++) {
        const d = Math.abs((nextResid[i] as number) - (resid[i] as number));
        if (d > step) step = d;
      }
      theta = next;
      resid = nextResid;
      scale = nextScale;
      if (step <= stepTol) {
        converged = true;
        break;
      }
    }
    if (!converged) {
      warn(
        `HuberRegressor did not converge within ${maxIter} iterations; increase maxIter or tol`,
        "ConvergenceWarning",
        "HuberRegressor"
      );
    }

    const bound = epsilon * scale;
    const outliers: boolean[] = new Array<boolean>(n);
    for (let i = 0; i < n; i++) outliers[i] = Math.abs(resid[i] as number) > bound;

    this.nFeaturesIn_ = p;
    const coef = theta.slice(0, p);
    let intercept = 0;
    if (fitIntercept) {
      intercept = yShift + (theta[p] as number);
      for (let j = 0; j < p; j++) intercept -= (xMean[j] as number) * (coef[j] as number);
    }
    this.coef_ = coef;
    this.intercept_ = intercept;
    this.scale_ = scale;
    this.nIter_ = nIter;
    this.outliers_ = outliers;
    this.fitted = true;
    return this;
  }

  /**
   * Predict target values.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predictions of shape (n_samples,) as a float64 tensor
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X is not 2D or has the wrong number of features
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.coef_) {
      throw new NotFittedError("HuberRegressor must be fitted before predict");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "HuberRegressor");

    const n = X.shape[0] ?? 0;
    const p = X.shape[1] ?? 0;
    const xv = toFloat64View(X);
    const w = this.coef_;
    const result = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let pred = this.intercept_;
      const base = i * p;
      for (let j = 0; j < p; j++) pred += (w[j] as number) * (xv[base + j] as number);
      result[i] = pred;
    }
    return tensor(result, { dtype: "float64" });
  }

  /**
   * Coefficient of determination R² on the given data.
   *
   * A constant `y` scores 1 when predicted exactly and 0 otherwise.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True target values of shape (n_samples,)
   * @returns R² score (1 is perfect, can be negative)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-D or its length differs from the number of samples
   */
  score(X: Tensor, y: Tensor): number {
    if (!this.fitted) {
      throw new NotFittedError("HuberRegressor must be fitted before scoring");
    }
    return r2ScoreOf(y, () => this.predict(X));
  }

  /** Fitted coefficients, shape (n_features,). */
  get coef(): Float64Array {
    if (!this.fitted) throw new NotFittedError("HuberRegressor must be fitted to access coef");
    return this.coef_ as Float64Array;
  }

  /**
   * Fitted coefficients as a float64 tensor of shape (n_features,), the same type as
   * `LinearRegression.coef`.
   */
  get coefTensor(): Tensor {
    return tensor(Float64Array.from(this.coef), { dtype: "float64" });
  }

  /** Intercept term (0 when `fitIntercept` is false). */
  get intercept(): number {
    if (!this.fitted) throw new NotFittedError("HuberRegressor must be fitted to access intercept");
    return this.intercept_;
  }

  /** Number of reweighting iterations run by the last `fit`. */
  get nIter(): number {
    if (!this.fitted) throw new NotFittedError("HuberRegressor must be fitted to access nIter");
    return this.nIter_;
  }

  /**
   * Estimated scale of the residuals. A training sample is flagged as an outlier when its
   * absolute residual exceeds `epsilon * scale`.
   */
  get scale(): number {
    if (!this.fitted) throw new NotFittedError("HuberRegressor must be fitted to access scale");
    return this.scale_;
  }

  /** For every training sample, whether its absolute residual exceeds `epsilon * scale`. */
  get outliers(): boolean[] {
    if (!this.fitted) throw new NotFittedError("HuberRegressor must be fitted to access outliers");
    return this.outliers_ as boolean[];
  }

  /** Number of features seen during `fit`. */
  get nFeaturesIn(): number {
    if (!this.fitted) {
      throw new NotFittedError("HuberRegressor must be fitted to access nFeaturesIn");
    }
    return this.nFeaturesIn_;
  }

  /** Hyper-parameters of this estimator, with defaults filled in. */
  getParams(): Record<string, unknown> {
    return {
      epsilon: this.opts.epsilon,
      alpha: this.opts.alpha,
      maxIter: this.opts.maxIter,
      tol: this.opts.tol,
      fitIntercept: this.opts.fitIntercept,
      learningRate: this.opts.learningRate,
    };
  }

  /**
   * Set hyper-parameters. All values are validated before any is applied.
   *
   * @param params - Parameters to change
   * @returns this
   * @throws {InvalidParameterError} If a name is unknown or a value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    const next: Record<string, unknown> = { ...this.opts };
    for (const [key, value] of Object.entries(params)) {
      if (!OPTION_KEYS.includes(key)) {
        throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
      if (key === "fitIntercept") {
        if (typeof value !== "boolean") {
          throw new InvalidParameterError("fitIntercept must be a boolean", key, value);
        }
      } else if (typeof value !== "number") {
        throw new InvalidParameterError(`${key} must be a number`, key, value);
      }
      next[key] = value;
    }
    this.opts = resolveOptions(next as HuberRegressorOptions);
    return this;
  }

  /** Create an unfitted copy with the same hyper-parameters. */
  clone(): HuberRegressor {
    return new HuberRegressor(this.getParams() as HuberRegressorOptions);
  }
}
