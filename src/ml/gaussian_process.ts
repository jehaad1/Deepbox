/**
 * Gaussian Process models.
 *
 * Implements Gaussian Process Regression (GPR) and Gaussian Process Classification (GPC)
 * with an RBF kernel. GPR provides mean predictions and uncertainty estimates.
 * GPC uses the Laplace approximation (Newton mode finding) for binary classification and
 * one-vs-rest for more than two classes. The kernel hyperparameters are fixed; they are
 * not optimized during `fit`.
 *
 * @module ml/gaussian_process
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../core";
import { type Tensor, tensor } from "../ndarray";
import { r2Score } from "./_internal";
import {
  assertContiguous,
  toFloat64View,
  validateFitInputs,
  validatePredictInputs,
} from "./_validation";
import type { Classifier, Regressor } from "./base";

// ---------------------------------------------------------------------------
// Shared numerical helpers
// ---------------------------------------------------------------------------

/**
 * Lower Cholesky factor `L` (row-major, `A = L Lᵀ`) of a symmetric positive definite matrix.
 * Returns `null` when a pivot is not larger than the rounding error of the factorization
 * (`n * eps * A[i, i]`), i.e. the matrix is not numerically positive definite.
 */
function choleskyLower(A: Float64Array, n: number): Float64Array | null {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    const rowI = i * n;
    for (let j = 0; j <= i; j++) {
      const rowJ = j * n;
      let sum = 0;
      for (let k = 0; k < j; k++) {
        sum += (L[rowI + k] ?? 0) * (L[rowJ + k] ?? 0);
      }
      if (i === j) {
        const aii = A[rowI + i] ?? 0;
        const diag = aii - sum;
        // A pivot below the rounding noise of the factorization means the matrix is
        // singular to working precision, and dividing by it would amplify noise.
        if (!(diag > n * Number.EPSILON * Math.abs(aii)) || !Number.isFinite(diag)) return null;
        L[rowI + i] = Math.sqrt(diag);
      } else {
        L[rowI + j] = ((A[rowI + j] ?? 0) - sum) / (L[rowJ + j] ?? 1);
      }
    }
  }
  return L;
}

/** Solve `L x = b` for lower triangular `L`, writing the result into `out`. */
function forwardSolveInto(L: Float64Array, b: Float64Array, n: number, out: Float64Array): void {
  for (let i = 0; i < n; i++) {
    const row = i * n;
    let sum = 0;
    for (let j = 0; j < i; j++) {
      sum += (L[row + j] ?? 0) * (out[j] ?? 0);
    }
    out[i] = ((b[i] ?? 0) - sum) / (L[row + i] ?? 1);
  }
}

/** Solve `Lᵀ x = b` for lower triangular `L`. */
function backwardSolve(L: Float64Array, b: Float64Array, n: number): Float64Array {
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < n; j++) {
      sum += (L[j * n + i] ?? 0) * (x[j] ?? 0);
    }
    x[i] = ((b[i] ?? 0) - sum) / (L[i * n + i] ?? 1);
  }
  return x;
}

/**
 * Fill `out[j]` with the RBF kernel `variance * exp(-|x - train_j|² / twoL2)` between the
 * row of `x` starting at `xOffset` and every row of `train`.
 */
function rbfRow(
  x: Float64Array,
  xOffset: number,
  train: Float64Array,
  nTrain: number,
  nF: number,
  variance: number,
  twoL2: number,
  out: Float64Array
): void {
  for (let j = 0; j < nTrain; j++) {
    const jo = j * nF;
    let sq = 0;
    for (let f = 0; f < nF; f++) {
      const d = (x[xOffset + f] ?? 0) - (train[jo + f] ?? 0);
      sq += d * d;
    }
    out[j] = variance * Math.exp(-sq / twoL2);
  }
}

/**
 * Training kernel matrix `K + diagAdd * I` as a dense row-major array.
 */
function rbfMatrix(
  xTrain: Float64Array,
  n: number,
  nF: number,
  variance: number,
  twoL2: number,
  diagAdd: number
): Float64Array {
  const K = new Float64Array(n * n);
  const row = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    rbfRow(xTrain, i * nF, xTrain, n, nF, variance, twoL2, row);
    for (let j = i; j < n; j++) {
      const kij = row[j] ?? 0;
      K[i * n + j] = kij;
      K[j * n + i] = kij;
    }
    K[i * n + i] = (K[i * n + i] ?? 0) + diagAdd;
  }
  return K;
}

/**
 * Error function with a relative error near machine precision.
 *
 * Uses erf(x) = 2/√π · exp(-x²) · Σ 2ⁿ x^(2n+1) / (1·3·…·(2n+1)), which has only positive
 * terms and therefore no cancellation. The value is exactly ±1 in double precision for
 * |x| >= 6.
 */
function erf(x: number): number {
  const ax = Math.abs(x);
  if (ax >= 6) return x < 0 ? -1 : 1;
  if (ax === 0) return x;
  const x2 = ax * ax;
  let term = ax;
  let sum = ax;
  for (let n = 0; n < 500; n++) {
    term *= (2 * x2) / (2 * n + 3);
    sum += term;
    if (term < sum * 1e-17) break;
  }
  const value = (2 / Math.sqrt(Math.PI)) * Math.exp(-x2) * sum;
  return x < 0 ? -value : value;
}

/**
 * Sigmoid, clamped so `exp` cannot overflow.
 */
function gpcSigmoid(x: number): number {
  const clamped = Math.max(-500, Math.min(500, x));
  return 1 / (1 + Math.exp(-clamped));
}

/** `log(1 + exp(-z))` without overflow. */
function softplusNeg(z: number): number {
  return Math.max(-z, 0) + Math.log1p(Math.exp(-Math.abs(z)));
}

// Williams & Barber (1998) approximation of the logistic sigmoid by five error functions,
// used to integrate the sigmoid against the Gaussian posterior of the latent function.
// These are the same constants scikit-learn uses.
const LAMBDAS = [0.41, 0.4, 0.37, 0.44, 0.39] as const;
const COEFS = [-1854.8214151, 3516.89893646, 221.29346712, 128.12323805, -2010.49422654] as const;
const COEFS_HALF_SUM = 0.5 * COEFS.reduce((a, b) => a + b, 0);

/**
 * Expected value of `sigmoid(f)` for `f ~ N(mean, variance)`.
 *
 * Closed form of the five-erf approximation: with `α = 1/(2·variance)` the integral of each
 * term reduces to `0.5 · erf(λ·mean / sqrt(1 + 2·variance·λ²))`, so `variance = 0` needs no
 * special case.
 */
function expectedSigmoid(mean: number, variance: number): number {
  let p = COEFS_HALF_SUM;
  for (let k = 0; k < COEFS.length; k++) {
    const lambda = LAMBDAS[k] ?? 0;
    p +=
      0.5 * (COEFS[k] ?? 0) * erf((lambda * mean) / Math.sqrt(1 + 2 * variance * lambda * lambda));
  }
  return Math.min(1, Math.max(0, p));
}

function validateKernelParam(name: "lengthScale" | "kernelVariance", value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new InvalidParameterError(`${name} must be a finite number > 0`, name, value);
  }
  if (name === "lengthScale") {
    const twoL2 = 2 * value * value;
    if (!(twoL2 > 0) || !Number.isFinite(twoL2)) {
      throw new InvalidParameterError(
        "lengthScale is too small or too large: 2 * lengthScale^2 must be a finite number > 0",
        name,
        value
      );
    }
  }
  return value;
}

function validateAlpha(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new InvalidParameterError("alpha must be a finite number >= 0", "alpha", value);
  }
  return value;
}

/**
 * Check a target vector passed to `score`.
 */
function checkScoreTarget(y: Tensor): void {
  if (y.ndim !== 1) {
    throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
  }
  assertContiguous(y, "y");
  if (y.size === 0) {
    throw new DataValidationError("y must contain at least one sample");
  }
}

// ---------------------------------------------------------------------------
// GaussianProcessRegressor
// ---------------------------------------------------------------------------

/**
 * Options for {@link GaussianProcessRegressor}.
 */
export type GaussianProcessRegressorOptions = {
  /** Value added to the diagonal of the kernel matrix, i.e. the noise variance (default: 1e-10). */
  readonly alpha?: number;
  /** RBF kernel length scale (default: 1.0). */
  readonly lengthScale?: number;
  /** RBF kernel variance, the prior variance of the function (default: 1.0). */
  readonly kernelVariance?: number;
  /**
   * Subtract the training mean and divide by the training standard deviation before fitting,
   * and undo this in the predictions (default: false). The prior mean of the process is zero,
   * so enable this when the targets are far from zero.
   */
  readonly normalizeY?: boolean;
};

type GprModel = {
  readonly xTrain: Float64Array;
  readonly nTrain: number;
  readonly nFeatures: number;
  /** Lower Cholesky factor of K + alpha * I. */
  readonly L: Float64Array;
  /** (K + alpha * I)⁻¹ y, in normalized-target units. */
  readonly weights: Float64Array;
  readonly lengthScale: number;
  readonly kernelVariance: number;
  readonly yMean: number;
  readonly yStd: number;
  readonly logMarginalLikelihood: number;
};

/**
 * Gaussian Process Regressor with RBF kernel.
 *
 * Matches scikit-learn's `GaussianProcessRegressor` with kernel
 * `ConstantKernel(kernelVariance) * RBF(lengthScale)` and `optimizer=None`: the
 * hyperparameters are used as given. The prior mean is zero (see `normalizeY`).
 *
 * @example
 * ```ts
 * import { GaussianProcessRegressor } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1], [2], [3], [4], [5]]);
 * const y = tensor([1, 4, 9, 16, 25]);
 * const gpr = new GaussianProcessRegressor({ alpha: 1e-6 });
 * gpr.fit(X, y);
 * const pred = gpr.predict(X);
 * const { mean, std } = gpr.predictWithStd(X);
 * ```
 */
export class GaussianProcessRegressor implements Regressor {
  private alpha: number;
  private lengthScale: number;
  private kernelVariance: number;
  private normalizeY: boolean;

  private model_: GprModel | undefined;

  /**
   * @param options.alpha - Noise added to the kernel diagonal (default: 1e-10)
   * @param options.lengthScale - RBF kernel length scale (default: 1.0)
   * @param options.kernelVariance - RBF kernel variance (default: 1.0)
   * @param options.normalizeY - Standardize the targets before fitting (default: false)
   * @throws {InvalidParameterError} If `alpha` is negative or a kernel parameter is not a positive finite number
   */
  constructor(options: GaussianProcessRegressorOptions = {}) {
    this.alpha = validateAlpha(options.alpha ?? 1e-10);
    this.lengthScale = validateKernelParam("lengthScale", options.lengthScale ?? 1.0);
    this.kernelVariance = validateKernelParam("kernelVariance", options.kernelVariance ?? 1.0);
    this.normalizeY = options.normalizeY ?? false;
    if (typeof this.normalizeY !== "boolean") {
      throw new InvalidParameterError(
        "normalizeY must be a boolean",
        "normalizeY",
        this.normalizeY
      );
    }
  }

  /**
   * Fit the model. Previous fitted state is discarded, and kept untouched if fitting throws.
   *
   * @throws {DataValidationError} If `K + alpha * I` is not numerically positive definite;
   * increase `alpha` or remove duplicate samples
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const nF = X.shape[1] ?? 0;

    // Copy: the model must not alias the caller's memory.
    const xTrain = Float64Array.from(toFloat64View(X));
    const yRaw = toFloat64View(y);

    let yMean = 0;
    let yStd = 1;
    const yFit = new Float64Array(n);
    if (this.normalizeY) {
      for (let i = 0; i < n; i++) yMean += yRaw[i] ?? 0;
      yMean /= n;
      let variance = 0;
      for (let i = 0; i < n; i++) {
        const d = (yRaw[i] ?? 0) - yMean;
        variance += d * d;
      }
      yStd = Math.sqrt(variance / n);
      // A constant target keeps scale 1, as scikit-learn does.
      if (!(yStd > 0)) yStd = 1;
      for (let i = 0; i < n; i++) yFit[i] = ((yRaw[i] ?? 0) - yMean) / yStd;
    } else {
      yFit.set(yRaw);
    }

    const twoL2 = 2 * this.lengthScale * this.lengthScale;
    const K = rbfMatrix(xTrain, n, nF, this.kernelVariance, twoL2, this.alpha);
    const L = choleskyLower(K, n);
    if (L === null) {
      throw new DataValidationError(
        "The kernel matrix is not positive definite. Increase alpha, increase lengthScale or remove duplicate samples."
      );
    }

    // (K + alpha I) w = y  =>  L z = y, Lᵀ w = z
    const z = new Float64Array(n);
    forwardSolveInto(L, yFit, n, z);
    const weights = backwardSolve(L, z, n);

    let quad = 0;
    let logDet = 0;
    for (let i = 0; i < n; i++) {
      quad += (yFit[i] ?? 0) * (weights[i] ?? 0);
      logDet += Math.log(L[i * n + i] ?? 1);
    }
    const logMarginalLikelihood = -0.5 * quad - logDet - 0.5 * n * Math.log(2 * Math.PI);

    this.model_ = {
      xTrain,
      nTrain: n,
      nFeatures: nF,
      L,
      weights,
      lengthScale: this.lengthScale,
      kernelVariance: this.kernelVariance,
      yMean,
      yStd,
      logMarginalLikelihood,
    };
    return this;
  }

  /**
   * Posterior mean at the rows of `X`.
   *
   * @returns Tensor of shape (n_samples,)
   */
  predict(X: Tensor): Tensor {
    const model = this.requireFitted("predict");
    validatePredictInputs(X, model.nFeatures, "GaussianProcessRegressor");
    return tensor(this.evaluate(X, model, false).mean, { dtype: "float64" });
  }

  /**
   * Posterior mean and standard deviation at the rows of `X`.
   *
   * The standard deviation is that of the noise-free latent function: it does not include
   * `alpha`, and numerically negative variances are clipped to 0.
   *
   * @returns `mean` and `std`, each of shape (n_samples,)
   */
  predictWithStd(X: Tensor): { mean: Tensor; std: Tensor } {
    const model = this.requireFitted("predictWithStd");
    validatePredictInputs(X, model.nFeatures, "GaussianProcessRegressor");
    const { mean, std } = this.evaluate(X, model, true);
    return {
      mean: tensor(mean, { dtype: "float64" }),
      std: tensor(std, { dtype: "float64" }),
    };
  }

  /**
   * Log marginal likelihood of the training targets under the fitted kernel.
   * With `normalizeY` it refers to the normalized targets, as in scikit-learn.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  logMarginalLikelihood(): number {
    return this.requireFitted("logMarginalLikelihood").logMarginalLikelihood;
  }

  /**
   * Coefficient of determination R^2 of `predict(X)` against `y`.
   *
   * A constant target gives 1 for a perfect fit and 0 otherwise, like scikit-learn.
   *
   * @throws {ShapeError} If `y` is not 1-D or its length differs from the number of samples in `X`
   * @throws {DataValidationError} If `y` is empty
   */
  score(X: Tensor, y: Tensor): number {
    checkScoreTarget(y);
    const pred = this.predict(X);
    const n = y.size;
    if (pred.size !== n) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${pred.size}, y=${n}`
      );
    }
    return r2Score(toFloat64View(y), toFloat64View(pred));
  }

  getParams(): Record<string, unknown> {
    return {
      alpha: this.alpha,
      lengthScale: this.lengthScale,
      kernelVariance: this.kernelVariance,
      normalizeY: this.normalizeY,
    };
  }

  /**
   * Change hyperparameters. They take effect on the next `fit`; a model that is already
   * fitted keeps predicting with the values it was fitted with.
   *
   * @throws {InvalidParameterError} On an unknown key or an invalid value
   */
  setParams(params: Record<string, unknown>): this {
    let { alpha, lengthScale, kernelVariance, normalizeY } = this;
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "alpha":
          alpha = validateAlpha(value);
          break;
        case "lengthScale":
          lengthScale = validateKernelParam("lengthScale", value);
          break;
        case "kernelVariance":
          kernelVariance = validateKernelParam("kernelVariance", value);
          break;
        case "normalizeY":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("normalizeY must be a boolean", "normalizeY", value);
          }
          normalizeY = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    this.alpha = alpha;
    this.lengthScale = lengthScale;
    this.kernelVariance = kernelVariance;
    this.normalizeY = normalizeY;
    return this;
  }

  /** Create an unfitted copy with the same hyperparameters. */
  clone(): GaussianProcessRegressor {
    return new GaussianProcessRegressor({
      alpha: this.alpha,
      lengthScale: this.lengthScale,
      kernelVariance: this.kernelVariance,
      normalizeY: this.normalizeY,
    });
  }

  private requireFitted(method: string): GprModel {
    if (this.model_ === undefined) {
      throw new NotFittedError(`GaussianProcessRegressor must be fitted before ${method}`);
    }
    return this.model_;
  }

  private evaluate(
    X: Tensor,
    model: GprModel,
    withStd: boolean
  ): { mean: Float64Array; std: Float64Array } {
    const x = toFloat64View(X);
    const nTest = X.shape[0] ?? 0;
    const { nTrain, nFeatures, L, weights, yMean, yStd } = model;
    const twoL2 = 2 * model.lengthScale * model.lengthScale;

    const mean = new Float64Array(nTest);
    const std = new Float64Array(withStd ? nTest : 0);
    const kStar = new Float64Array(nTrain);
    const v = new Float64Array(withStd ? nTrain : 0);

    for (let i = 0; i < nTest; i++) {
      rbfRow(x, i * nFeatures, model.xTrain, nTrain, nFeatures, model.kernelVariance, twoL2, kStar);

      let m = 0;
      for (let j = 0; j < nTrain; j++) m += (kStar[j] ?? 0) * (weights[j] ?? 0);
      mean[i] = m * yStd + yMean;

      if (withStd) {
        // var = k(x*, x*) - |L⁻¹ k*|²
        forwardSolveInto(L, kStar, nTrain, v);
        let vNorm = 0;
        for (let j = 0; j < nTrain; j++) vNorm += (v[j] ?? 0) * (v[j] ?? 0);
        std[i] = Math.sqrt(Math.max(0, model.kernelVariance - vNorm)) * yStd;
      }
    }
    return { mean, std };
  }
}

// ---------------------------------------------------------------------------
// GaussianProcessClassifier
// ---------------------------------------------------------------------------

/**
 * Options for {@link GaussianProcessClassifier}.
 */
export type GaussianProcessClassifierOptions = {
  /** Value added to the diagonal of the training kernel matrix (default: 1e-5). */
  readonly alpha?: number;
  /** RBF kernel length scale (default: 1.0). */
  readonly lengthScale?: number;
  /** RBF kernel variance (default: 1.0). */
  readonly kernelVariance?: number;
  /** Maximum Newton iterations of the Laplace approximation, per binary problem (default: 100). */
  readonly maxLaplaceIter?: number;
};

/** Laplace posterior of one binary problem. */
type LaplacePosterior = {
  /** `y - σ(f̂)`: the predictive latent mean at x* is `k*ᵀ a`. */
  readonly a: Float64Array;
  /** `sqrt(W)` with `W = σ(f̂)(1 - σ(f̂))`. */
  readonly wSr: Float64Array;
  /** Cholesky factor of `I + W^½ K W^½`. */
  readonly L: Float64Array;
};

type GpcModel = {
  readonly xTrain: Float64Array;
  readonly nTrain: number;
  readonly nFeatures: number;
  readonly classes: readonly number[];
  /** One posterior for binary problems, one per class (one-vs-rest) otherwise. */
  readonly posteriors: readonly LaplacePosterior[];
  readonly lengthScale: number;
  readonly kernelVariance: number;
  readonly integerLabels: boolean;
};

/**
 * Gaussian Process Classifier with RBF kernel.
 *
 * Uses the Laplace approximation with Newton mode finding (Rasmussen & Williams,
 * Algorithms 3.1 and 3.2), as scikit-learn does. For two classes it fits a single latent
 * GP with a logistic link; for more classes it fits one binary problem per class
 * (one-vs-rest) and normalizes the per-class probabilities. The predictive probability
 * integrates the sigmoid over the posterior variance of the latent function, so it is
 * less extreme than `sigmoid(mean)` far from the training data. Hyperparameters are fixed.
 *
 * @example
 * ```ts
 * import { GaussianProcessClassifier } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 1], [2, 2], [3, 3], [8, 8], [9, 9], [10, 10]]);
 * const y = tensor([0, 0, 0, 1, 1, 1]);
 *
 * const gpc = new GaussianProcessClassifier();
 * gpc.fit(X, y);
 * const predictions = gpc.predict(X);
 * const probabilities = gpc.predictProba(X);
 * ```
 */
export class GaussianProcessClassifier implements Classifier {
  private alpha: number;
  private lengthScale: number;
  private kernelVariance: number;
  private maxLaplaceIter: number;

  private model_: GpcModel | undefined;

  /**
   * @param options.alpha - Noise added to kernel diagonal for numerical stability (default: 1e-5)
   * @param options.lengthScale - RBF kernel length scale (default: 1.0)
   * @param options.kernelVariance - RBF kernel variance (default: 1.0)
   * @param options.maxLaplaceIter - Maximum Newton iterations of the Laplace approximation (default: 100)
   * @throws {InvalidParameterError} If a parameter is out of range
   */
  constructor(options: GaussianProcessClassifierOptions = {}) {
    this.alpha = validateAlpha(options.alpha ?? 1e-5);
    this.lengthScale = validateKernelParam("lengthScale", options.lengthScale ?? 1.0);
    this.kernelVariance = validateKernelParam("kernelVariance", options.kernelVariance ?? 1.0);
    this.maxLaplaceIter = validateMaxIter(options.maxLaplaceIter ?? 100);
  }

  /**
   * Fit the model. Previous fitted state is discarded, and kept untouched if fitting throws.
   *
   * @throws {DataValidationError} If `y` has fewer than two distinct classes
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const nF = X.shape[1] ?? 0;

    const xTrain = Float64Array.from(toFloat64View(X));
    const yData = toFloat64View(y);
    const classes = [...new Set(yData)].sort((a, b) => a - b);
    const nClasses = classes.length;
    if (nClasses < 2) {
      throw new DataValidationError(
        `GaussianProcessClassifier requires at least 2 distinct classes in y; got ${nClasses}`
      );
    }

    const twoL2 = 2 * this.lengthScale * this.lengthScale;
    const K = rbfMatrix(xTrain, n, nF, this.kernelVariance, twoL2, this.alpha);

    // Binary problems model P(y = classes[1]); otherwise one problem per class.
    const targets = nClasses === 2 ? [classes[1] ?? 0] : classes;
    const posteriors: LaplacePosterior[] = [];
    for (const cls of targets) {
      const yBin = new Float64Array(n);
      for (let i = 0; i < n; i++) yBin[i] = (yData[i] ?? 0) === cls ? 1 : 0;
      posteriors.push(this.laplaceBinary(K, yBin, n));
    }

    this.model_ = {
      xTrain,
      nTrain: n,
      nFeatures: nF,
      classes,
      posteriors,
      lengthScale: this.lengthScale,
      kernelVariance: this.kernelVariance,
      integerLabels: classes.every((v) => Number.isInteger(v) && Math.abs(v) <= 2147483647),
    };
    return this;
  }

  predict(X: Tensor): Tensor {
    const model = this.requireFitted();
    validatePredictInputs(X, model.nFeatures, "GaussianProcessClassifier");
    const { proba, latent } = this.evaluate(X, model);
    const nSamples = X.shape[0] ?? 0;
    const nClasses = model.classes.length;
    const out = new Float64Array(nSamples);

    for (let i = 0; i < nSamples; i++) {
      let best = 0;
      if (nClasses === 2) {
        // Same rule as scikit-learn: the sign of the latent mean decides.
        best = (latent[i] ?? 0) > 0 ? 1 : 0;
      } else {
        let maxP = Number.NEGATIVE_INFINITY;
        for (let c = 0; c < nClasses; c++) {
          const p = proba[i * nClasses + c] ?? 0;
          if (p > maxP) {
            maxP = p;
            best = c;
          }
        }
      }
      out[i] = model.classes[best] ?? 0;
    }
    return model.integerLabels ? tensor(Int32Array.from(out)) : tensor(out);
  }

  /**
   * Class probabilities. Columns follow {@link GaussianProcessClassifier.classes}.
   *
   * @returns Tensor of shape (n_samples, n_classes) whose rows sum to 1
   */
  predictProba(X: Tensor): Tensor {
    const model = this.requireFitted();
    validatePredictInputs(X, model.nFeatures, "GaussianProcessClassifier");
    const { proba } = this.evaluate(X, model);
    return tensor(proba, { dtype: "float64" }).reshape([X.shape[0] ?? 0, model.classes.length]);
  }

  /**
   * Mean accuracy of `predict(X)` against `y`.
   *
   * @throws {ShapeError} If `y` is not 1-D or its length differs from the number of samples in `X`
   * @throws {DataValidationError} If `y` is empty
   */
  score(X: Tensor, y: Tensor): number {
    checkScoreTarget(y);
    const yPred = this.predict(X);
    if (yPred.size !== y.size) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${yPred.size}, y=${y.size}`
      );
    }
    const yv = toFloat64View(y);
    const pv = toFloat64View(yPred);
    let correct = 0;
    for (let i = 0; i < y.size; i++) {
      if ((yv[i] ?? 0) === (pv[i] ?? 0)) correct++;
    }
    return correct / y.size;
  }

  /** Sorted class labels seen during `fit`, or `undefined` before fitting. */
  get classes(): Tensor | undefined {
    const model = this.model_;
    if (model === undefined) return undefined;
    return model.integerLabels
      ? tensor(Int32Array.from(model.classes))
      : tensor(Float64Array.from(model.classes));
  }

  getParams(): Record<string, unknown> {
    return {
      alpha: this.alpha,
      lengthScale: this.lengthScale,
      kernelVariance: this.kernelVariance,
      maxLaplaceIter: this.maxLaplaceIter,
    };
  }

  /**
   * Change hyperparameters. They take effect on the next `fit`; a model that is already
   * fitted keeps predicting with the values it was fitted with.
   *
   * @throws {InvalidParameterError} On an unknown key or an invalid value
   */
  setParams(params: Record<string, unknown>): this {
    let { alpha, lengthScale, kernelVariance, maxLaplaceIter } = this;
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "alpha":
          alpha = validateAlpha(value);
          break;
        case "lengthScale":
          lengthScale = validateKernelParam("lengthScale", value);
          break;
        case "kernelVariance":
          kernelVariance = validateKernelParam("kernelVariance", value);
          break;
        case "maxLaplaceIter":
          maxLaplaceIter = validateMaxIter(value);
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    this.alpha = alpha;
    this.lengthScale = lengthScale;
    this.kernelVariance = kernelVariance;
    this.maxLaplaceIter = maxLaplaceIter;
    return this;
  }

  /** Create an unfitted copy with the same hyperparameters. */
  clone(): GaussianProcessClassifier {
    return new GaussianProcessClassifier({
      alpha: this.alpha,
      lengthScale: this.lengthScale,
      kernelVariance: this.kernelVariance,
      maxLaplaceIter: this.maxLaplaceIter,
    });
  }

  // ---- Private helpers ----

  private requireFitted(): GpcModel {
    if (this.model_ === undefined) {
      throw new NotFittedError("GaussianProcessClassifier must be fitted before prediction");
    }
    return this.model_;
  }

  /**
   * Predictive class probabilities (row-major, n_samples x n_classes) and, for binary
   * problems, the latent mean of the positive class.
   */
  private evaluate(X: Tensor, model: GpcModel): { proba: Float64Array; latent: Float64Array } {
    const x = toFloat64View(X);
    const nTest = X.shape[0] ?? 0;
    const { nTrain, nFeatures, posteriors } = model;
    const nClasses = model.classes.length;
    const twoL2 = 2 * model.lengthScale * model.lengthScale;

    const proba = new Float64Array(nTest * nClasses);
    const latent = new Float64Array(nClasses === 2 ? nTest : 0);
    const kStar = new Float64Array(nTrain);
    const wK = new Float64Array(nTrain);
    const v = new Float64Array(nTrain);
    const perClass = new Float64Array(posteriors.length);

    for (let i = 0; i < nTest; i++) {
      // The kernel row is shared by every class.
      rbfRow(x, i * nFeatures, model.xTrain, nTrain, nFeatures, model.kernelVariance, twoL2, kStar);

      for (let c = 0; c < posteriors.length; c++) {
        const post = posteriors[c]!;
        let mean = 0;
        for (let j = 0; j < nTrain; j++) {
          mean += (kStar[j] ?? 0) * (post.a[j] ?? 0);
          wK[j] = (post.wSr[j] ?? 0) * (kStar[j] ?? 0);
        }
        forwardSolveInto(post.L, wK, nTrain, v);
        let vNorm = 0;
        for (let j = 0; j < nTrain; j++) vNorm += (v[j] ?? 0) * (v[j] ?? 0);
        const variance = Math.max(0, model.kernelVariance - vNorm);

        perClass[c] = expectedSigmoid(mean, variance);
        if (nClasses === 2) latent[i] = mean;
      }

      if (nClasses === 2) {
        const p1 = perClass[0] ?? 0;
        proba[i * 2] = 1 - p1;
        proba[i * 2 + 1] = p1;
      } else {
        let sum = 0;
        for (let c = 0; c < nClasses; c++) sum += perClass[c] ?? 0;
        for (let c = 0; c < nClasses; c++) {
          proba[i * nClasses + c] = sum > 0 ? (perClass[c] ?? 0) / sum : 1 / nClasses;
        }
      }
    }
    return { proba, latent };
  }

  /**
   * Laplace approximation of the posterior of the latent function for one binary problem
   * (Rasmussen & Williams, Algorithm 3.1), by Newton iteration from `f = 0`. Iteration stops
   * when the approximate log marginal likelihood improves by less than 1e-10, or after
   * `maxLaplaceIter` iterations.
   *
   * `B = I + W^½ K W^½` is positive definite for any positive semi-definite `K`, so the
   * Cholesky factorization does not depend on `alpha`.
   */
  private laplaceBinary(K: Float64Array, y: Float64Array, n: number): LaplacePosterior {
    let f: Float64Array = new Float64Array(n);
    let prevLml = Number.NEGATIVE_INFINITY;

    let pi = new Float64Array(n);
    let wSr = new Float64Array(n);
    let L: Float64Array = new Float64Array(0);
    const W = new Float64Array(n);
    const B = new Float64Array(n * n);
    const b = new Float64Array(n);
    const rhs = new Float64Array(n);
    const z = new Float64Array(n);

    for (let iter = 0; iter < this.maxLaplaceIter; iter++) {
      pi = new Float64Array(n);
      wSr = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const p = gpcSigmoid(f[i] ?? 0);
        pi[i] = p;
        W[i] = p * (1 - p);
        wSr[i] = Math.sqrt(W[i] ?? 0);
      }

      // B = I + W^½ K W^½
      for (let i = 0; i < n; i++) {
        const wi = wSr[i] ?? 0;
        for (let j = 0; j < n; j++) {
          B[i * n + j] = wi * (K[i * n + j] ?? 0) * (wSr[j] ?? 0) + (i === j ? 1 : 0);
        }
      }
      const factor = choleskyLower(B, n);
      if (factor === null) {
        throw new DataValidationError(
          "The Laplace approximation failed: the kernel matrix is not positive semi-definite. Check lengthScale and kernelVariance."
        );
      }
      L = factor;

      // b = W f + (y - pi)
      for (let i = 0; i < n; i++) {
        b[i] = (W[i] ?? 0) * (f[i] ?? 0) + ((y[i] ?? 0) - (pi[i] ?? 0));
      }
      // a = b - W^½ B⁻¹ W^½ K b
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += (K[i * n + j] ?? 0) * (b[j] ?? 0);
        rhs[i] = (wSr[i] ?? 0) * s;
      }
      forwardSolveInto(L, rhs, n, z);
      const u = backwardSolve(L, z, n);
      const a = new Float64Array(n);
      for (let i = 0; i < n; i++) a[i] = (b[i] ?? 0) - (wSr[i] ?? 0) * (u[i] ?? 0);

      // f = K a
      const fNew = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += (K[i * n + j] ?? 0) * (a[j] ?? 0);
        fNew[i] = s;
      }
      f = fNew;

      // Approximate log marginal likelihood, used as the convergence criterion.
      let lml = 0;
      for (let i = 0; i < n; i++) {
        lml -= 0.5 * (a[i] ?? 0) * (f[i] ?? 0);
        lml -= softplusNeg((2 * (y[i] ?? 0) - 1) * (f[i] ?? 0));
        lml -= Math.log(L[i * n + i] ?? 1);
      }
      if (lml - prevLml < 1e-10) break;
      prevLml = lml;
    }

    const aOut = new Float64Array(n);
    for (let i = 0; i < n; i++) aOut[i] = (y[i] ?? 0) - (pi[i] ?? 0);
    return { a: aOut, wSr, L };
  }
}

function validateMaxIter(value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new InvalidParameterError(
      "maxLaplaceIter must be an integer >= 1",
      "maxLaplaceIter",
      value
    );
  }
  return value;
}
