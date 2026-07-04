/**
 * Huber Regressor — Linear regression robust to outliers.
 *
 * Uses Huber loss instead of squared loss, which is quadratic for small
 * residuals and linear for large residuals (outliers). The `epsilon`
 * parameter controls the threshold between the two regimes.
 *
 * @module ml/linear/HuberRegressor
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Regressor } from "../base";

/**
 * Linear regression with Huber loss for robustness to outliers.
 *
 * The Huber loss is defined as:
 * - `0.5 * (y - X @ w)^2` if `|y - X @ w| <= epsilon`
 * - `epsilon * |y - X @ w| - 0.5 * epsilon^2` otherwise
 *
 * @example
 * ```ts
 * import { HuberRegressor } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1], [2], [3], [4], [100]]);
 * const y = tensor([2, 4, 6, 8, 200]); // last point is an outlier
 * const reg = new HuberRegressor({ epsilon: 1.35 });
 * reg.fit(X, y);
 * console.log(reg.coef); // close to 2
 * ```
 */
export class HuberRegressor implements Regressor {
  private readonly epsilon: number;
  private readonly alpha: number;
  private readonly maxIter: number;
  private readonly tol: number;
  private readonly fitInterceptOpt: boolean;
  private readonly learningRate: number;

  private coef_?: Float64Array;
  private intercept_ = 0;
  private nFeaturesIn_ = 0;
  private nIter_ = 0;
  private outliers_?: boolean[];
  private fitted = false;

  constructor(
    options: {
      readonly epsilon?: number;
      readonly alpha?: number;
      readonly maxIter?: number;
      readonly tol?: number;
      readonly fitIntercept?: boolean;
      readonly learningRate?: number;
    } = {}
  ) {
    this.epsilon = options.epsilon ?? 1.35;
    this.alpha = options.alpha ?? 0.0001;
    this.maxIter = options.maxIter ?? 100;
    this.tol = options.tol ?? 1e-5;
    this.fitInterceptOpt = options.fitIntercept ?? true;
    this.learningRate = options.learningRate ?? 0.01;

    if (this.epsilon <= 1.0) {
      throw new InvalidParameterError("epsilon must be > 1.0", "epsilon", this.epsilon);
    }
    if (this.alpha < 0) {
      throw new InvalidParameterError("alpha must be >= 0", "alpha", this.alpha);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new InvalidParameterError(
        "maxIter must be a positive integer",
        "maxIter",
        this.maxIter
      );
    }
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nFeatures;

    // Extract data
    const yData = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      yData[i] = Number(y.data[y.offset + i]);
    }

    const w = new Float64Array(nFeatures);
    let b = 0;
    const eps = this.epsilon;

    for (let iter = 0; iter < this.maxIter; iter++) {
      const gradW = new Float64Array(nFeatures);
      let gradB = 0;

      for (let i = 0; i < nSamples; i++) {
        const rowBase = X.offset + i * nFeatures;
        let pred = this.fitInterceptOpt ? b : 0;
        for (let j = 0; j < nFeatures; j++) {
          pred += (w[j] ?? 0) * Number(X.data[rowBase + j] ?? 0);
        }

        const residual = pred - (yData[i] ?? 0);
        const absR = Math.abs(residual);
        let dloss: number;

        if (absR <= eps) {
          // Quadratic regime
          dloss = residual;
        } else {
          // Linear regime
          dloss = eps * Math.sign(residual);
        }

        for (let j = 0; j < nFeatures; j++) {
          gradW[j] = (gradW[j] ?? 0) + dloss * Number(X.data[rowBase + j] ?? 0);
        }
        if (this.fitInterceptOpt) gradB += dloss;
      }

      // Average gradients + L2 regularization
      const invN = 1 / nSamples;
      let maxUpdate = 0;

      for (let j = 0; j < nFeatures; j++) {
        const g = (gradW[j] ?? 0) * invN + this.alpha * (w[j] ?? 0);
        const update = this.learningRate * g;
        w[j] = (w[j] ?? 0) - update;
        maxUpdate = Math.max(maxUpdate, Math.abs(update));
      }
      if (this.fitInterceptOpt) {
        const updateB = this.learningRate * gradB * invN;
        b -= updateB;
        maxUpdate = Math.max(maxUpdate, Math.abs(updateB));
      }

      this.nIter_ = iter + 1;
      if (maxUpdate < this.tol) break;
    }

    this.coef_ = w;
    this.intercept_ = b;

    // Identify outliers: points where |residual| > epsilon
    this.outliers_ = [];
    for (let i = 0; i < nSamples; i++) {
      const rowBase = X.offset + i * nFeatures;
      let pred = this.fitInterceptOpt ? b : 0;
      for (let j = 0; j < nFeatures; j++) {
        pred += (w[j] ?? 0) * Number(X.data[rowBase + j] ?? 0);
      }
      this.outliers_.push(Math.abs(pred - (yData[i] ?? 0)) > eps);
    }

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) throw new NotFittedError("HuberRegressor must be fitted before predict");
    validatePredictInputs(X, this.nFeaturesIn_, "HuberRegressor");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const w = this.coef_!;
    const result = new Float64Array(nSamples);

    for (let i = 0; i < nSamples; i++) {
      let pred = this.fitInterceptOpt ? this.intercept_ : 0;
      const rowBase = X.offset + i * nFeatures;
      for (let j = 0; j < nFeatures; j++) {
        pred += (w[j] ?? 0) * Number(X.data[rowBase + j] ?? 0);
      }
      result[i] = pred;
    }

    return tensor(Array.from(result));
  }

  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    assertContiguous(y, "y");
    const pred = this.predict(X);
    const nSamples = y.size;

    let yMean = 0;
    for (let i = 0; i < nSamples; i++) yMean += Number(y.data[y.offset + i]);
    yMean /= nSamples;

    let ssRes = 0;
    let ssTot = 0;
    for (let i = 0; i < nSamples; i++) {
      const yi = Number(y.data[y.offset + i]);
      const pi = Number(pred.data[pred.offset + i]);
      ssRes += (yi - pi) ** 2;
      ssTot += (yi - yMean) ** 2;
    }
    return ssTot === 0 ? 0 : 1 - ssRes / ssTot;
  }

  get coef(): Float64Array {
    if (!this.fitted) throw new NotFittedError("HuberRegressor must be fitted to access coef");
    return this.coef_!;
  }

  get intercept(): number {
    if (!this.fitted) throw new NotFittedError("HuberRegressor must be fitted to access intercept");
    return this.intercept_;
  }

  get nIter(): number {
    if (!this.fitted) throw new NotFittedError("HuberRegressor must be fitted to access nIter");
    return this.nIter_;
  }

  get outliers(): boolean[] {
    if (!this.fitted) throw new NotFittedError("HuberRegressor must be fitted to access outliers");
    return this.outliers_!;
  }

  getParams(): Record<string, unknown> {
    return {
      epsilon: this.epsilon,
      alpha: this.alpha,
      maxIter: this.maxIter,
      tol: this.tol,
      fitIntercept: this.fitInterceptOpt,
      learningRate: this.learningRate,
    };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}
