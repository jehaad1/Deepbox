/**
 * Gaussian Process models.
 *
 * Implements Gaussian Process Regression (GPR) and Gaussian Process Classification (GPC)
 * with an RBF kernel. GPR provides mean predictions and uncertainty estimates.
 * GPC uses Laplace approximation for binary and multi-class classification.
 *
 * @module ml/gaussian_process
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError, ShapeError } from "../core";
import { type Tensor, tensor } from "../ndarray";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "./_validation";
import type { Classifier, Regressor } from "./base";

/**
 * Gaussian Process Regressor with RBF kernel.
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
  private readonly alpha: number;
  private readonly lengthScale: number;
  private readonly kernelVariance: number;

  private xTrain_?: Float64Array;
  private yTrain_?: Float64Array;
  private nTrainSamples_ = 0;
  private nFeaturesIn_ = 0;
  private kInvY_?: Float64Array; // K_inv @ y_train
  private lMatrix_?: Float64Array; // Cholesky lower triangular of K + alpha*I
  private fitted = false;

  constructor(
    options: {
      readonly alpha?: number;
      readonly lengthScale?: number;
      readonly kernelVariance?: number;
    } = {}
  ) {
    this.alpha = options.alpha ?? 1e-10;
    this.lengthScale = options.lengthScale ?? 1.0;
    this.kernelVariance = options.kernelVariance ?? 1.0;

    if (this.alpha < 0) {
      throw new InvalidParameterError("alpha must be >= 0", "alpha", this.alpha);
    }
    if (this.lengthScale <= 0) {
      throw new InvalidParameterError("lengthScale must be > 0", "lengthScale", this.lengthScale);
    }
    if (this.kernelVariance <= 0) {
      throw new InvalidParameterError(
        "kernelVariance must be > 0",
        "kernelVariance",
        this.kernelVariance
      );
    }
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const nF = X.shape[1] ?? 0;
    this.nTrainSamples_ = n;
    this.nFeaturesIn_ = nF;

    // Store training data
    this.xTrain_ = new Float64Array(n * nF);
    this.yTrain_ = new Float64Array(n);
    for (let i = 0; i < n * nF; i++) {
      this.xTrain_[i] = Number(X.data[X.offset + i]);
    }
    for (let i = 0; i < n; i++) {
      this.yTrain_[i] = Number(y.data[y.offset + i]);
    }

    // Compute K(X_train, X_train) + alpha * I
    const K = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        const kij = this.rbfKernel(this.xTrain_, i, this.xTrain_, j, nF);
        K[i * n + j] = kij;
        K[j * n + i] = kij;
      }
      K[i * n + i] = (K[i * n + i] ?? 0) + this.alpha;
    }

    // Cholesky decomposition: K = L @ L^T
    this.lMatrix_ = this.cholesky(K, n);

    // Solve L @ L^T @ alpha = y  =>  L @ z = y  =>  L^T @ alpha = z
    const z = this.forwardSolve(this.lMatrix_, this.yTrain_, n);
    this.kInvY_ = this.backwardSolve(this.lMatrix_, z, n);

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("GaussianProcessRegressor must be fitted before predict");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "GaussianProcessRegressor");
    const nTest = X.shape[0] ?? 0;
    const nF = this.nFeaturesIn_;
    const nTrain = this.nTrainSamples_;

    const result = new Float64Array(nTest);
    for (let i = 0; i < nTest; i++) {
      // k_star = K(x_test_i, X_train)
      let mean = 0;
      for (let j = 0; j < nTrain; j++) {
        const kij = this.rbfKernelMixed(X, i, this.xTrain_!, j, nF);
        mean += kij * (this.kInvY_![j] ?? 0);
      }
      result[i] = mean;
    }

    return tensor(Array.from(result));
  }

  /**
   * Predict with uncertainty estimates.
   */
  predictWithStd(X: Tensor): { mean: Tensor; std: Tensor } {
    if (!this.fitted) {
      throw new NotFittedError("GaussianProcessRegressor must be fitted before predictWithStd");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "GaussianProcessRegressor");
    const nTest = X.shape[0] ?? 0;
    const nF = this.nFeaturesIn_;
    const nTrain = this.nTrainSamples_;

    const means = new Float64Array(nTest);
    const stds = new Float64Array(nTest);

    for (let i = 0; i < nTest; i++) {
      const kStar = new Float64Array(nTrain);
      for (let j = 0; j < nTrain; j++) {
        kStar[j] = this.rbfKernelMixed(X, i, this.xTrain_!, j, nF);
      }

      // mean = k_star^T @ K_inv @ y
      let mean = 0;
      for (let j = 0; j < nTrain; j++) {
        mean += (kStar[j] ?? 0) * (this.kInvY_![j] ?? 0);
      }
      means[i] = mean;

      // var = k(x*, x*) - k_star^T @ K_inv @ k_star
      const kSelf = this.kernelVariance;
      // Solve L @ v = k_star
      const v = this.forwardSolve(this.lMatrix_!, kStar, nTrain);
      let vNorm = 0;
      for (let j = 0; j < nTrain; j++) {
        vNorm += (v[j] ?? 0) * (v[j] ?? 0);
      }
      const variance = Math.max(0, kSelf - vNorm);
      stds[i] = Math.sqrt(variance);
    }

    return {
      mean: tensor(Array.from(means)),
      std: tensor(Array.from(stds)),
    };
  }

  score(X: Tensor, y: Tensor): number {
    const pred = this.predict(X);
    const n = y.size;
    let ssRes = 0;
    let ssTot = 0;
    let yMean = 0;
    for (let i = 0; i < n; i++) yMean += Number(y.data[y.offset + i]);
    yMean /= n;
    for (let i = 0; i < n; i++) {
      const yi = Number(y.data[y.offset + i]);
      const pi = Number(pred.data[pred.offset + i]);
      ssRes += (yi - pi) * (yi - pi);
      ssTot += (yi - yMean) * (yi - yMean);
    }
    return ssTot > 0 ? 1 - ssRes / ssTot : 0;
  }

  getParams(): Record<string, unknown> {
    return {
      alpha: this.alpha,
      lengthScale: this.lengthScale,
      kernelVariance: this.kernelVariance,
    };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }

  private rbfKernel(
    data: Float64Array,
    i: number,
    data2: Float64Array,
    j: number,
    nF: number
  ): number {
    let sq = 0;
    for (let f = 0; f < nF; f++) {
      const diff = (data[i * nF + f] ?? 0) - (data2[j * nF + f] ?? 0);
      sq += diff * diff;
    }
    return this.kernelVariance * Math.exp(-sq / (2 * this.lengthScale * this.lengthScale));
  }

  private rbfKernelMixed(
    X: Tensor,
    i: number,
    xTrain: Float64Array,
    j: number,
    nF: number
  ): number {
    let sq = 0;
    for (let f = 0; f < nF; f++) {
      const diff = Number(X.data[X.offset + i * nF + f]) - (xTrain[j * nF + f] ?? 0);
      sq += diff * diff;
    }
    return this.kernelVariance * Math.exp(-sq / (2 * this.lengthScale * this.lengthScale));
  }

  private cholesky(A: Float64Array, n: number): Float64Array {
    const L = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = 0;
        for (let k = 0; k < j; k++) {
          sum += (L[i * n + k] ?? 0) * (L[j * n + k] ?? 0);
        }
        if (i === j) {
          const diag = (A[i * n + i] ?? 0) - sum;
          L[i * n + i] = Math.sqrt(Math.max(diag, 1e-20));
        } else {
          const lJJ = L[j * n + j] ?? 1;
          L[i * n + j] = ((A[i * n + j] ?? 0) - sum) / lJJ;
        }
      }
    }
    return L;
  }

  private forwardSolve(L: Float64Array, b: Float64Array, n: number): Float64Array {
    // Solve L @ x = b
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < i; j++) {
        sum += (L[i * n + j] ?? 0) * (x[j] ?? 0);
      }
      const lII = L[i * n + i] ?? 1;
      x[i] = ((b[i] ?? 0) - sum) / lII;
    }
    return x;
  }

  private backwardSolve(L: Float64Array, b: Float64Array, n: number): Float64Array {
    // Solve L^T @ x = b
    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let sum = 0;
      for (let j = i + 1; j < n; j++) {
        sum += (L[j * n + i] ?? 0) * (x[j] ?? 0);
      }
      const lII = L[i * n + i] ?? 1;
      x[i] = ((b[i] ?? 0) - sum) / lII;
    }
    return x;
  }
}

/**
 * Gaussian Process Classifier with RBF kernel.
 *
 * Uses Laplace approximation for approximate inference. For binary classification,
 * fits a single GP with sigmoid link function. For multi-class, uses one-vs-rest.
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
  private readonly alpha: number;
  private readonly lengthScale: number;
  private readonly kernelVariance: number;
  private readonly maxLaplaceIter: number;

  private xTrain_?: Float64Array;
  private nTrainSamples_ = 0;
  private nFeaturesIn_ = 0;
  private classes_?: number[];
  private fHat_?: Float64Array;
  private ovrFHats_?: Float64Array[];
  private fitted = false;

  /**
   * @param options.alpha - Noise added to kernel diagonal for numerical stability (default: 1e-5)
   * @param options.lengthScale - RBF kernel length scale (default: 1.0)
   * @param options.kernelVariance - RBF kernel variance (default: 1.0)
   * @param options.maxLaplaceIter - Maximum Laplace approximation iterations (default: 20)
   */
  constructor(
    options: {
      readonly alpha?: number;
      readonly lengthScale?: number;
      readonly kernelVariance?: number;
      readonly maxLaplaceIter?: number;
    } = {}
  ) {
    this.alpha = options.alpha ?? 1e-5;
    this.lengthScale = options.lengthScale ?? 1.0;
    this.kernelVariance = options.kernelVariance ?? 1.0;
    this.maxLaplaceIter = options.maxLaplaceIter ?? 20;

    if (this.lengthScale <= 0) {
      throw new InvalidParameterError("lengthScale must be > 0", "lengthScale", this.lengthScale);
    }
    if (this.kernelVariance <= 0) {
      throw new InvalidParameterError(
        "kernelVariance must be > 0",
        "kernelVariance",
        this.kernelVariance
      );
    }
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const nF = X.shape[1] ?? 0;
    this.nTrainSamples_ = n;
    this.nFeaturesIn_ = nF;

    this.xTrain_ = new Float64Array(n * nF);
    for (let i = 0; i < n * nF; i++) {
      this.xTrain_[i] = Number(X.data[X.offset + i] ?? 0);
    }

    const yData: number[] = [];
    for (let i = 0; i < n; i++) {
      yData.push(Number(y.data[y.offset + i] ?? 0));
    }

    this.classes_ = [...new Set(yData)].sort((a, b) => a - b);
    const nClasses = this.classes_.length;

    const K = this.computeKernelMatrix(n, nF);

    if (nClasses === 2) {
      const yBin = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        yBin[i] = (yData[i] ?? 0) === (this.classes_[1] ?? 0) ? 1 : 0;
      }
      this.fHat_ = this.laplaceBinary(K, yBin, n);
      delete this.ovrFHats_;
    } else {
      this.ovrFHats_ = [];
      for (const cls of this.classes_) {
        const yBin = new Float64Array(n);
        for (let i = 0; i < n; i++) {
          yBin[i] = (yData[i] ?? 0) === cls ? 1 : 0;
        }
        this.ovrFHats_.push(this.laplaceBinary(K, yBin, n));
      }
      delete this.fHat_;
    }

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.classes_) {
      throw new NotFittedError("GaussianProcessClassifier must be fitted before prediction");
    }
    const proba = this.predictProba(X);
    const nSamples = proba.shape[0] ?? 0;
    const nClasses = this.classes_.length;
    const predictions: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      let maxP = -1;
      let maxC = 0;
      for (let c = 0; c < nClasses; c++) {
        const p = Number(proba.data[proba.offset + i * nClasses + c] ?? 0);
        if (p > maxP) {
          maxP = p;
          maxC = c;
        }
      }
      predictions.push(this.classes_[maxC] ?? 0);
    }

    return tensor(predictions, { dtype: "int32" });
  }

  predictProba(X: Tensor): Tensor {
    if (!this.fitted || !this.classes_ || !this.xTrain_) {
      throw new NotFittedError("GaussianProcessClassifier must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "GaussianProcessClassifier");

    const nTest = X.shape[0] ?? 0;
    const nF = this.nFeaturesIn_;
    const nTrain = this.nTrainSamples_;
    const nClasses = this.classes_.length;

    if (nClasses === 2) {
      const probabilities: number[][] = [];
      for (let i = 0; i < nTest; i++) {
        // GP posterior latent mean = k*ᵀ a, where a = y - σ(f̂) is stored in fHat_.
        let latent = 0;
        for (let j = 0; j < nTrain; j++) {
          const kij = this.rbfKernelMixed(X, i, this.xTrain_, j, nF);
          latent += kij * (this.fHat_![j] ?? 0);
        }
        const p1 = gpcSigmoid(latent);
        probabilities.push([1 - p1, p1]);
      }
      return tensor(probabilities);
    }

    const probabilities: number[][] = [];
    for (let i = 0; i < nTest; i++) {
      const scores: number[] = [];
      for (let c = 0; c < nClasses; c++) {
        let latent = 0;
        for (let j = 0; j < nTrain; j++) {
          const kij = this.rbfKernelMixed(X, i, this.xTrain_, j, nF);
          latent += kij * (this.ovrFHats_![c]![j] ?? 0);
        }
        scores.push(gpcSigmoid(latent));
      }
      const sum = scores.reduce((a, b) => a + b, 0) || 1;
      probabilities.push(scores.map((s) => s / sum));
    }

    return tensor(probabilities);
  }

  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    assertContiguous(y, "y");
    const yPred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.size; i++) {
      if (Number(y.data[y.offset + i] ?? 0) === Number(yPred.data[yPred.offset + i] ?? 0)) {
        correct++;
      }
    }
    return correct / y.size;
  }

  get classes(): Tensor | undefined {
    if (!this.fitted || !this.classes_) return undefined;
    return tensor(this.classes_, { dtype: "int32" });
  }

  getParams(): Record<string, unknown> {
    return {
      alpha: this.alpha,
      lengthScale: this.lengthScale,
      kernelVariance: this.kernelVariance,
      maxLaplaceIter: this.maxLaplaceIter,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const key of Object.keys(params)) {
      if (
        key !== "alpha" &&
        key !== "lengthScale" &&
        key !== "kernelVariance" &&
        key !== "maxLaplaceIter"
      ) {
        throw new InvalidParameterError(`Unknown parameter: ${key}`, key, params[key]);
      }
    }
    return this;
  }

  // ---- Private helpers ----

  private computeKernelMatrix(n: number, nF: number): Float64Array {
    const K = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        const kij = this.rbfKernel(this.xTrain_!, i, this.xTrain_!, j, nF);
        K[i * n + j] = kij;
        K[j * n + i] = kij;
      }
      K[i * n + i] = (K[i * n + i] ?? 0) + this.alpha;
    }
    return K;
  }

  /**
   * Laplace-approximation mode finding for binary GP classification
   * (Rasmussen & Williams, Algorithm 3.1). Returns the posterior weight vector
   * `a = y - σ(f̂)` at the mode, which is exactly what the predictive mean
   * `k*ᵀ a` needs. A damped fixed-point (γ<1) is used because the undamped
   * iteration `f ← K(y - σ(f))` oscillates/diverges on typical kernel matrices.
   */
  private laplaceBinary(K: Float64Array, y: Float64Array, n: number): Float64Array {
    const f = new Float64Array(n);
    const gamma = 0.5; // damping factor for stability

    for (let iter = 0; iter < this.maxLaplaceIter; iter++) {
      const grad = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        grad[i] = (y[i] ?? 0) - gpcSigmoid(f[i] ?? 0);
      }

      let maxChange = 0;
      const fTarget = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let j = 0; j < n; j++) sum += (K[i * n + j] ?? 0) * (grad[j] ?? 0);
        fTarget[i] = sum;
      }
      for (let i = 0; i < n; i++) {
        const next = (1 - gamma) * (f[i] ?? 0) + gamma * (fTarget[i] ?? 0);
        const change = Math.abs(next - (f[i] ?? 0));
        if (change > maxChange) maxChange = change;
        f[i] = next;
      }

      if (maxChange < 1e-8) break;
    }

    // Posterior weights a = y - σ(f̂); the predictive latent mean at a test
    // point x* is k*ᵀ a.
    const a = new Float64Array(n);
    for (let i = 0; i < n; i++) a[i] = (y[i] ?? 0) - gpcSigmoid(f[i] ?? 0);
    return a;
  }

  private rbfKernel(
    data: Float64Array,
    i: number,
    data2: Float64Array,
    j: number,
    nF: number
  ): number {
    let sq = 0;
    for (let f = 0; f < nF; f++) {
      const diff = (data[i * nF + f] ?? 0) - (data2[j * nF + f] ?? 0);
      sq += diff * diff;
    }
    return this.kernelVariance * Math.exp(-sq / (2 * this.lengthScale * this.lengthScale));
  }

  private rbfKernelMixed(
    X: Tensor,
    i: number,
    xTrain: Float64Array,
    j: number,
    nF: number
  ): number {
    let sq = 0;
    for (let f = 0; f < nF; f++) {
      const diff = Number(X.data[X.offset + i * nF + f] ?? 0) - (xTrain[j * nF + f] ?? 0);
      sq += diff * diff;
    }
    return this.kernelVariance * Math.exp(-sq / (2 * this.lengthScale * this.lengthScale));
  }
}

function gpcSigmoid(x: number): number {
  const clamped = Math.max(-500, Math.min(500, x));
  return 1 / (1 + Math.exp(-clamped));
}
