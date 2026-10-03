/**
 * Bayesian Ridge Regression.
 *
 * Fits a linear model using Bayesian inference with automatic
 * regularization parameter tuning via evidence maximization (type-II ML).
 * The model assumes Gaussian priors on the weights and noise.
 *
 * @module ml/linear/BayesianRidge
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../../core";
import { svd } from "../../linalg";
import { type Tensor, tensor } from "../../ndarray";
import { toFloat64View, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Regressor } from "../base";
import { r2ScoreOf } from "./LinearRegression";

/** Constructor options of {@link BayesianRidge}. */
export type BayesianRidgeOptions = {
  /** Maximum number of evidence-maximization iterations (default: 300). */
  readonly maxIter?: number;
  /**
   * Stop when the summed absolute change of the coefficients between two iterations is
   * below this value (default: 1e-3).
   */
  readonly tol?: number;
  /** Fit an intercept term (default: true). */
  readonly fitIntercept?: boolean;
  /** Initial noise precision. Defaults to `1 / (var(y) + eps)`. */
  readonly alphaInit?: number;
  /** Initial weight precision (default: 1). */
  readonly lambdaInit?: number;
  /** Shape parameter of the Gamma prior over the noise precision alpha (default: 1e-6). */
  readonly alpha1?: number;
  /** Rate parameter of the Gamma prior over the noise precision alpha (default: 1e-6). */
  readonly alpha2?: number;
  /** Shape parameter of the Gamma prior over the weight precision lambda (default: 1e-6). */
  readonly lambda1?: number;
  /** Rate parameter of the Gamma prior over the weight precision lambda (default: 1e-6). */
  readonly lambda2?: number;
  /** Record the log marginal likelihood at every iteration (default: false). */
  readonly computeScore?: boolean;
};

type ResolvedOptions = {
  maxIter: number;
  tol: number;
  fitIntercept: boolean;
  alphaInit: number | undefined;
  lambdaInit: number;
  alpha1: number;
  alpha2: number;
  lambda1: number;
  lambda2: number;
  computeScore: boolean;
};

const OPTION_KEYS: readonly string[] = [
  "maxIter",
  "tol",
  "fitIntercept",
  "alphaInit",
  "lambdaInit",
  "alpha1",
  "alpha2",
  "lambda1",
  "lambda2",
  "computeScore",
];

function resolveOptions(options: BayesianRidgeOptions): ResolvedOptions {
  const resolved: ResolvedOptions = {
    maxIter: options.maxIter ?? 300,
    tol: options.tol ?? 1e-3,
    fitIntercept: options.fitIntercept ?? true,
    alphaInit: options.alphaInit,
    lambdaInit: options.lambdaInit ?? 1,
    alpha1: options.alpha1 ?? 1e-6,
    alpha2: options.alpha2 ?? 1e-6,
    lambda1: options.lambda1 ?? 1e-6,
    lambda2: options.lambda2 ?? 1e-6,
    computeScore: options.computeScore ?? false,
  };
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
  if (
    resolved.alphaInit !== undefined &&
    !(resolved.alphaInit > 0 && Number.isFinite(resolved.alphaInit))
  ) {
    throw new InvalidParameterError("alphaInit must be > 0", "alphaInit", resolved.alphaInit);
  }
  if (!(resolved.lambdaInit > 0 && Number.isFinite(resolved.lambdaInit))) {
    throw new InvalidParameterError("lambdaInit must be > 0", "lambdaInit", resolved.lambdaInit);
  }
  for (const key of ["alpha1", "alpha2", "lambda1", "lambda2"] as const) {
    const v = resolved[key];
    if (!(v >= 0) || !Number.isFinite(v)) {
      throw new InvalidParameterError(`${key} must be a non-negative finite number`, key, v);
    }
  }
  return resolved;
}

/**
 * Bayesian Ridge Regression with automatic regularization.
 *
 * Iteratively updates the precision parameters alpha (noise) and
 * lambda (weights) with MacKay's evidence maximization, using Gamma hyper-priors
 * (`alpha1`, `alpha2`, `lambda1`, `lambda2`). The algorithm, defaults and stopping rule follow
 * `sklearn.linear_model.BayesianRidge`. Uncertainty is available through the posterior
 * covariance of the weights ({@link BayesianRidge.sigma}) and
 * {@link BayesianRidge.predictWithStd}.
 *
 * Each iteration works on the singular value decomposition of the centered design matrix,
 * so the cost per iteration does not depend on the number of samples.
 *
 * @example
 * ```ts
 * import { BayesianRidge } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1], [2], [3], [4], [5]]);
 * const y = tensor([2.1, 3.9, 6.1, 7.9, 10.2]);
 * const reg = new BayesianRidge();
 * reg.fit(X, y);
 * const pred = reg.predict(X);
 * ```
 */
export class BayesianRidge implements Regressor {
  private opts: ResolvedOptions;

  private coef_?: Float64Array;
  private intercept_ = 0;
  private alpha_ = 0; // noise precision
  private lambda_ = 0; // weight precision
  private nFeaturesIn_ = 0;
  private nIter_ = 0;
  private xMean_?: Float64Array;
  private sigma_?: Float64Array;
  private scores_: Float64Array | undefined;
  private fitted = false;

  /**
   * Create a new Bayesian Ridge model.
   *
   * @param options - Configuration options, see {@link BayesianRidgeOptions}
   * @throws {InvalidParameterError} If an option is outside its valid range
   */
  constructor(options: BayesianRidgeOptions = {}) {
    this.opts = resolveOptions(options);
  }

  /**
   * Fit the model by evidence maximization.
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
    const { fitIntercept, maxIter, tol, alpha1, alpha2, lambda1, lambda2 } = this.opts;

    const xRaw = toFloat64View(X);
    const yRaw = toFloat64View(y);
    const xc = Float64Array.from(xRaw);
    const yc = Float64Array.from(yRaw);
    const xMean = new Float64Array(p);
    let yMean = 0;
    if (fitIntercept) {
      for (let i = 0; i < n; i++) {
        const base = i * p;
        for (let j = 0; j < p; j++) xMean[j] = (xMean[j] as number) + (xc[base + j] as number);
        yMean += yc[i] as number;
      }
      for (let j = 0; j < p; j++) xMean[j] = (xMean[j] as number) / n;
      yMean /= n;
      for (let i = 0; i < n; i++) {
        const base = i * p;
        for (let j = 0; j < p; j++) xc[base + j] = (xc[base + j] as number) - (xMean[j] as number);
        yc[i] = (yc[i] as number) - yMean;
      }
    }

    // Variance of y about its mean, used for the default initial noise precision.
    let yVar = 0;
    {
      let m = 0;
      for (let i = 0; i < n; i++) m += yRaw[i] as number;
      m /= n;
      for (let i = 0; i < n; i++) yVar += ((yRaw[i] as number) - m) ** 2;
      yVar /= n;
    }

    // Thin SVD of the centered design matrix: X = U diag(s) Vt.
    const [uT, sT, vtT] = svd(tensor(xc, { dtype: "float64" }).reshape([n, p]), false);
    const U = toFloat64View(uT);
    const s = toFloat64View(sT);
    const Vt = toFloat64View(vtT);
    const k = s.length;
    const eig = new Float64Array(k);
    for (let i = 0; i < k; i++) eig[i] = (s[i] as number) ** 2;

    // U^T y and the part of y outside the column space of X.
    const uty = new Float64Array(k);
    for (let i = 0; i < n; i++) {
      const yi = yc[i] as number;
      const base = i * k;
      for (let c = 0; c < k; c++) uty[c] = (uty[c] as number) + (U[base + c] as number) * yi;
    }
    let perpSS = 0;
    for (let i = 0; i < n; i++) {
      let proj = 0;
      const base = i * k;
      for (let c = 0; c < k; c++) proj += (U[base + c] as number) * (uty[c] as number);
      perpSS += ((yc[i] as number) - proj) ** 2;
    }

    // Posterior mean for given precisions, together with the residual sum of squares.
    const updateCoef = (alpha: number, lambda: number): { coef: Float64Array; sse: number } => {
      const ratio = lambda / alpha;
      const coef = new Float64Array(p);
      let sse = perpSS;
      for (let c = 0; c < k; c++) {
        const d = (eig[c] as number) + ratio;
        const g = ((s[c] as number) * (uty[c] as number)) / d;
        const base = c * p;
        for (let j = 0; j < p; j++) coef[j] = (coef[j] as number) + g * (Vt[base + j] as number);
        const shrink = (ratio / d) * (uty[c] as number);
        sse += shrink * shrink;
      }
      return { coef, sse };
    };

    const logMarginalLikelihood = (
      alpha: number,
      lambda: number,
      coef: Float64Array,
      sse: number
    ): number => {
      let logdetSigma = (p - k) * Math.log(lambda);
      for (let c = 0; c < k; c++) logdetSigma += Math.log(lambda + alpha * (eig[c] as number));
      logdetSigma = -logdetSigma;
      let coefSS = 0;
      for (let j = 0; j < p; j++) coefSS += (coef[j] as number) ** 2;
      let score = lambda1 * Math.log(lambda) - lambda2 * lambda;
      score += alpha1 * Math.log(alpha) - alpha2 * alpha;
      score +=
        0.5 *
        (p * Math.log(lambda) +
          n * Math.log(alpha) -
          alpha * sse -
          lambda * coefSS +
          logdetSigma -
          n * Math.log(2 * Math.PI));
      return score;
    };

    let alpha = this.opts.alphaInit ?? 1 / (yVar + Number.EPSILON);
    let lambda = this.opts.lambdaInit;
    const scores: number[] = [];
    let coefOld: Float64Array | undefined;
    let iter = 0;
    for (; iter < maxIter; iter++) {
      const { coef, sse } = updateCoef(alpha, lambda);
      if (this.opts.computeScore) scores.push(logMarginalLikelihood(alpha, lambda, coef, sse));

      // MacKay (1992) updates of the precisions.
      let gamma = 0;
      for (let c = 0; c < k; c++) {
        const ae = alpha * (eig[c] as number);
        gamma += ae / (lambda + ae);
      }
      let coefSS = 0;
      for (let j = 0; j < p; j++) coefSS += (coef[j] as number) ** 2;
      lambda = (gamma + 2 * lambda1) / (coefSS + 2 * lambda2);
      alpha = (n - gamma + 2 * alpha1) / (sse + 2 * alpha2);

      if (iter !== 0 && coefOld !== undefined) {
        let change = 0;
        for (let j = 0; j < p; j++)
          change += Math.abs((coefOld[j] as number) - (coef[j] as number));
        if (change < tol) break;
      }
      coefOld = coef;
    }
    const nIter = Math.min(iter + 1, maxIter);

    const final = updateCoef(alpha, lambda);
    if (this.opts.computeScore) {
      scores.push(logMarginalLikelihood(alpha, lambda, final.coef, final.sse));
    }

    // Posterior covariance (alpha X^T X + lambda I)^-1; directions outside the row space of X
    // keep the prior variance 1 / lambda.
    const sigma = new Float64Array(p * p);
    for (let j = 0; j < p; j++) sigma[j * p + j] = 1 / lambda;
    for (let c = 0; c < k; c++) {
      const gainOverPrior = 1 / (alpha * (eig[c] as number) + lambda) - 1 / lambda;
      const base = c * p;
      for (let a = 0; a < p; a++) {
        const va = (Vt[base + a] as number) * gainOverPrior;
        for (let b = 0; b < p; b++) {
          sigma[a * p + b] = (sigma[a * p + b] as number) + va * (Vt[base + b] as number);
        }
      }
    }

    let xMeanDotW = 0;
    for (let j = 0; j < p; j++) xMeanDotW += (xMean[j] as number) * (final.coef[j] as number);

    this.nFeaturesIn_ = p;
    this.coef_ = final.coef;
    this.intercept_ = fitIntercept ? yMean - xMeanDotW : 0;
    this.alpha_ = alpha;
    this.lambda_ = lambda;
    this.nIter_ = nIter;
    this.xMean_ = xMean;
    this.sigma_ = sigma;
    this.scores_ = this.opts.computeScore ? Float64Array.from(scores) : undefined;
    this.fitted = true;
    return this;
  }

  /**
   * Predict the posterior mean of the target.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predictions of shape (n_samples,) as a float64 tensor
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X is not 2D or has the wrong number of features
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.coef_) {
      throw new NotFittedError("BayesianRidge must be fitted before predict");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "BayesianRidge");

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
   * Predict the posterior mean together with the standard deviation of the predictive
   * distribution.
   *
   * The variance of a sample `x` is `(x - xMean)^T Sigma (x - xMean) + 1 / alpha`, where `xMean`
   * is the training mean (zero when `fitIntercept` is false). Uncertainty in the intercept is not
   * included. scikit-learn evaluates the quadratic form at the uncentered `x`, so its standard
   * deviations differ when the features are not centered.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Object with `mean` and `std`, both of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X is not 2D or has the wrong number of features
   */
  predictWithStd(X: Tensor): { mean: Tensor; std: Tensor } {
    const mean = this.predict(X);
    const sigma = this.sigma_ as Float64Array;
    const xMean = this.xMean_ as Float64Array;
    const n = X.shape[0] ?? 0;
    const p = this.nFeaturesIn_;
    const xv = toFloat64View(X);
    const std = new Float64Array(n);
    const d = new Float64Array(p);
    for (let i = 0; i < n; i++) {
      const base = i * p;
      for (let j = 0; j < p; j++) d[j] = (xv[base + j] as number) - (xMean[j] as number);
      let q = 0;
      for (let a = 0; a < p; a++) {
        let row = 0;
        for (let b = 0; b < p; b++) row += (sigma[a * p + b] as number) * (d[b] as number);
        q += (d[a] as number) * row;
      }
      std[i] = Math.sqrt(Math.max(0, q) + 1 / this.alpha_);
    }
    return { mean, std: tensor(std, { dtype: "float64" }) };
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
      throw new NotFittedError("BayesianRidge must be fitted before scoring");
    }
    return r2ScoreOf(y, () => this.predict(X));
  }

  /** Posterior mean of the weights, shape (n_features,). */
  get coef(): Float64Array {
    if (!this.fitted) throw new NotFittedError("BayesianRidge must be fitted to access coef");
    return this.coef_ as Float64Array;
  }

  /**
   * Posterior mean of the weights as a float64 tensor of shape (n_features,), the same type as
   * `LinearRegression.coef`.
   */
  get coefTensor(): Tensor {
    return tensor(Float64Array.from(this.coef), { dtype: "float64" });
  }

  /** Intercept term (0 when `fitIntercept` is false). */
  get intercept(): number {
    if (!this.fitted) throw new NotFittedError("BayesianRidge must be fitted to access intercept");
    return this.intercept_;
  }

  /** Estimated precision of the noise. */
  get alpha(): number {
    if (!this.fitted) throw new NotFittedError("BayesianRidge must be fitted to access alpha");
    return this.alpha_;
  }

  /** Estimated precision of the weights. */
  get lambda(): number {
    if (!this.fitted) throw new NotFittedError("BayesianRidge must be fitted to access lambda");
    return this.lambda_;
  }

  /** Number of evidence-maximization iterations run by the last `fit`. */
  get nIter(): number {
    if (!this.fitted) throw new NotFittedError("BayesianRidge must be fitted to access nIter");
    return this.nIter_;
  }

  /** Posterior covariance matrix of the weights, shape (n_features, n_features). */
  get sigma(): Tensor {
    if (!this.fitted || !this.sigma_) {
      throw new NotFittedError("BayesianRidge must be fitted to access sigma");
    }
    return tensor(Float64Array.from(this.sigma_), { dtype: "float64" }).reshape([
      this.nFeaturesIn_,
      this.nFeaturesIn_,
    ]);
  }

  /**
   * Log marginal likelihood after every iteration plus the final value. Only recorded when the
   * model was created with `computeScore: true`; empty otherwise.
   */
  get scores(): Float64Array {
    if (!this.fitted) throw new NotFittedError("BayesianRidge must be fitted to access scores");
    return this.scores_ ?? new Float64Array(0);
  }

  /** Number of features seen during `fit`. */
  get nFeaturesIn(): number {
    if (!this.fitted) {
      throw new NotFittedError("BayesianRidge must be fitted to access nFeaturesIn");
    }
    return this.nFeaturesIn_;
  }

  /**
   * Hyper-parameters of this estimator, with defaults filled in. `alphaInit` is omitted while it
   * is derived from the data.
   */
  getParams(): Record<string, unknown> {
    const params: Record<string, unknown> = {
      maxIter: this.opts.maxIter,
      tol: this.opts.tol,
      fitIntercept: this.opts.fitIntercept,
      lambdaInit: this.opts.lambdaInit,
      alpha1: this.opts.alpha1,
      alpha2: this.opts.alpha2,
      lambda1: this.opts.lambda1,
      lambda2: this.opts.lambda2,
      computeScore: this.opts.computeScore,
    };
    if (this.opts.alphaInit !== undefined) params["alphaInit"] = this.opts.alphaInit;
    return params;
  }

  /**
   * Set hyper-parameters. All values are validated before any is applied; pass `alphaInit:
   * undefined` to go back to the data-derived initial noise precision.
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
      if (key === "fitIntercept" || key === "computeScore") {
        if (typeof value !== "boolean") {
          throw new InvalidParameterError(`${key} must be a boolean`, key, value);
        }
      } else if (value !== undefined && typeof value !== "number") {
        throw new InvalidParameterError(`${key} must be a number`, key, value);
      }
      next[key] = value;
    }
    this.opts = resolveOptions(next as BayesianRidgeOptions);
    return this;
  }

  /** Create an unfitted copy with the same hyper-parameters. */
  clone(): BayesianRidge {
    return new BayesianRidge(this.getParams() as BayesianRidgeOptions);
  }
}
