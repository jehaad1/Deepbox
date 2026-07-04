/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __random } from "../../random/random";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Regressor } from "../base";

/**
 * Elastic Net Regression (L1 + L2 Regularized Linear Regression).
 *
 * Elastic Net combines L1 (Lasso) and L2 (Ridge) penalties, controlled
 * by the `l1Ratio` parameter. This makes it more robust than Lasso when
 * there are correlated features, while still performing feature selection.
 *
 * **Objective**: minimize (1/(2*n)) ||y - Xw||² + α * l1Ratio * ||w||₁ + α * (1-l1Ratio)/2 * ||w||²
 *
 * - `l1Ratio = 1` → pure Lasso (L1 only)
 * - `l1Ratio = 0` → pure Ridge (L2 only)
 * - `0 < l1Ratio < 1` → mix of L1 and L2
 *
 * @example
 * ```ts
 * import { ElasticNet } from 'deepbox/ml';
 *
 * const model = new ElasticNet({ alpha: 0.1, l1Ratio: 0.5 });
 * model.fit(X_train, y_train);
 * const predictions = model.predict(X_test);
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
  private fitted = false;

  /**
   * Create a new Elastic Net model.
   *
   * @param options - Configuration options
   * @param options.alpha - Regularization strength (default: 1.0). Must be >= 0.
   * @param options.l1Ratio - Mix ratio between L1 and L2 (default: 0.5). 0 = Ridge, 1 = Lasso.
   * @param options.fitIntercept - Whether to calculate the intercept (default: true)
   * @param options.normalize - Whether to normalize features (default: false)
   * @param options.maxIter - Maximum iterations for coordinate descent (default: 1000)
   * @param options.tol - Tolerance for convergence (default: 1e-4)
   * @param options.warmStart - Reuse previous solution as init (default: false)
   * @param options.positive - Force coefficients to be positive (default: false)
   * @param options.selection - Coordinate selection: 'cyclic' or 'random' (default: 'cyclic')
   */
  constructor(
    options: {
      readonly alpha?: number;
      readonly l1Ratio?: number;
      readonly fitIntercept?: boolean;
      readonly normalize?: boolean;
      readonly maxIter?: number;
      readonly tol?: number;
      readonly warmStart?: boolean;
      readonly positive?: boolean;
      readonly selection?: "cyclic" | "random";
      readonly randomState?: number;
    } = {}
  ) {
    this.options = { ...options };
    if (this.options.randomState !== undefined && !Number.isFinite(this.options.randomState)) {
      throw new InvalidParameterError(
        `randomState must be a finite number; received ${String(this.options.randomState)}`,
        "randomState",
        this.options.randomState
      );
    }
  }

  private createRNG(): () => number {
    if (this.options.randomState !== undefined) {
      let seed = this.options.randomState;
      return () => {
        seed = (seed * 9301 + 49297) % 233280;
        return seed / 233280;
      };
    }
    return __random;
  }

  /**
   * Fit Elastic Net model using Coordinate Descent.
   *
   * Solves: minimize (1/(2*n)) ||y - Xw||² + α * l1Ratio * ||w||₁ + α * (1-l1Ratio)/2 * ||w||²
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target values of shape (n_samples,)
   * @returns this - The fitted estimator
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    this.nIter_ = undefined;

    const alpha = this.options.alpha ?? 1.0;
    if (!(alpha >= 0)) {
      throw new InvalidParameterError(`alpha must be >= 0; received ${alpha}`, "alpha", alpha);
    }

    const l1Ratio = this.options.l1Ratio ?? 0.5;
    if (l1Ratio < 0 || l1Ratio > 1) {
      throw new InvalidParameterError(
        `l1Ratio must be in [0, 1]; received ${l1Ratio}`,
        "l1Ratio",
        l1Ratio
      );
    }

    const maxIter = this.options.maxIter ?? 1000;
    const tol = this.options.tol ?? 1e-4;
    const fitIntercept = this.options.fitIntercept ?? true;
    const normalize = this.options.normalize ?? false;
    const positive = this.options.positive ?? false;
    const selection = this.options.selection ?? "cyclic";
    const rng = this.createRNG();

    // Decompose alpha into L1 and L2 components
    const l1Penalty = alpha * l1Ratio;
    const l2Penalty = alpha * (1 - l1Ratio);

    const m = X.shape[0] ?? 0;
    const n = X.shape[1] ?? 0;
    this.nFeaturesIn_ = n;

    // Compute means for centering
    let yMean = 0;
    const xMean = new Array<number>(n).fill(0);

    if (fitIntercept) {
      for (let i = 0; i < m; i++) {
        yMean += Number(y.data[y.offset + i] ?? 0);
      }
      for (let i = 0; i < m; i++) {
        const rowBase = X.offset + i * n;
        for (let j = 0; j < n; j++) {
          xMean[j] = (xMean[j] ?? 0) + Number(X.data[rowBase + j] ?? 0);
        }
      }
      const invM = m === 0 ? 0 : 1 / m;
      yMean *= invM;
      for (let j = 0; j < n; j++) {
        xMean[j] = (xMean[j] ?? 0) * invM;
      }
    }

    let xScale: number[] | undefined;
    if (normalize) {
      xScale = new Array<number>(n).fill(0);
      for (let i = 0; i < m; i++) {
        const rowBase = X.offset + i * n;
        for (let j = 0; j < n; j++) {
          const centered = Number(X.data[rowBase + j] ?? 0) - (fitIntercept ? (xMean[j] ?? 0) : 0);
          xScale[j] = (xScale[j] ?? 0) + centered * centered;
        }
      }
      for (let j = 0; j < n; j++) {
        xScale[j] = Math.sqrt(xScale[j] ?? 0);
      }
    }

    const getX = (sampleIndex: number, featureIndex: number): number => {
      const raw = Number(X.data[X.offset + sampleIndex * n + featureIndex] ?? 0);
      const centered = raw - (fitIntercept ? (xMean[featureIndex] ?? 0) : 0);
      if (normalize && xScale) {
        const s = xScale[featureIndex] ?? 0;
        return s === 0 ? 0 : centered / s;
      }
      return centered;
    };

    // Precompute column squared norms
    const colNorm2 = new Array<number>(n).fill(0);
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let i = 0; i < m; i++) {
        const xij = getX(i, j);
        s += xij * xij;
      }
      colNorm2[j] = m === 0 ? 0 : s / m;
    }

    // Initialize coefficients
    const w = new Array<number>(n).fill(0);
    if (this.options.warmStart && this.coef_ && this.coef_.ndim === 1 && this.coef_.size === n) {
      for (let j = 0; j < n; j++) {
        w[j] = Number(this.coef_.data[this.coef_.offset + j] ?? 0);
      }
    }

    // Maintain current predictions
    const yHat = new Array<number>(m).fill(0);
    for (let i = 0; i < m; i++) {
      let pred = 0;
      for (let j = 0; j < n; j++) {
        pred += getX(i, j) * (w[j] ?? 0);
      }
      yHat[i] = pred;
    }

    const invM = m === 0 ? 0 : 1 / m;

    // Coordinate descent
    for (let iter = 0; iter < maxIter; iter++) {
      let maxChange = 0;

      let indices: number[] | null = null;
      if (selection === "random") {
        indices = Array.from({ length: n }, (_, j) => j);
        for (let k = n - 1; k > 0; k--) {
          const r = Math.floor(rng() * (k + 1));
          const tmp = indices[k];
          indices[k] = indices[r] ?? 0;
          indices[r] = tmp ?? 0;
        }
      }

      const iterOrder = indices ?? Array.from({ length: n }, (_, j) => j);
      for (const j of iterOrder) {
        const denom = (colNorm2[j] ?? 0) + l2Penalty;

        if (denom === 0) {
          const prevW = w[j] ?? 0;
          if (prevW !== 0) {
            const delta = -prevW;
            for (let i = 0; i < m; i++) {
              yHat[i] = (yHat[i] ?? 0) + delta * getX(i, j);
            }
            maxChange = Math.max(maxChange, Math.abs(delta));
          }
          w[j] = 0;
          continue;
        }

        // Compute correlation
        let rho = 0;
        for (let i = 0; i < m; i++) {
          const xij = getX(i, j);
          const yi = Number(y.data[y.offset + i] ?? 0) - (fitIntercept ? yMean : 0);
          const r = yi - (yHat[i] ?? 0) + (w[j] ?? 0) * xij;
          rho += xij * r;
        }
        rho *= invM;

        // Soft threshold for L1, divide by (colNorm2 + l2Penalty) for L2
        let newW = this.softThreshold(rho, l1Penalty) / denom;

        if (positive && newW < 0) {
          newW = 0;
        }

        const delta = newW - (w[j] ?? 0);

        if (delta !== 0) {
          for (let i = 0; i < m; i++) {
            yHat[i] = (yHat[i] ?? 0) + delta * getX(i, j);
          }
        }

        w[j] = newW;
        maxChange = Math.max(maxChange, Math.abs(delta));
      }

      if (maxChange < tol) {
        this.nIter_ = iter + 1;
        break;
      }
    }

    if (this.nIter_ === undefined) {
      this.nIter_ = maxIter;
    }

    // Rescale if normalized
    if (normalize && xScale) {
      for (let j = 0; j < n; j++) {
        const s = xScale[j] ?? 1;
        w[j] = s === 0 ? 0 : (w[j] ?? 0) / s;
      }
    }

    this.coef_ = tensor(w);

    if (fitIntercept) {
      let xMeanDotW = 0;
      for (let j = 0; j < n; j++) {
        xMeanDotW += (xMean[j] ?? 0) * (w[j] ?? 0);
      }
      this.intercept_ = yMean - xMeanDotW;
    } else {
      this.intercept_ = 0;
    }

    this.fitted = true;
    return this;
  }

  private softThreshold(x: number, lambda: number): number {
    if (!Number.isFinite(x) || !Number.isFinite(lambda)) {
      throw new DataValidationError("Non-finite value encountered during soft-thresholding");
    }
    if (x > lambda) return x - lambda;
    if (x < -lambda) return x + lambda;
    return 0;
  }

  /**
   * Predict using the Elastic Net model.
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.coef_) {
      throw new NotFittedError("ElasticNet must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "ElasticNet");

    const m = X.shape[0] ?? 0;
    const n = X.shape[1] ?? 0;
    const pred = Array(m).fill(0);

    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        pred[i] +=
          Number(X.data[X.offset + i * n + j] ?? 0) *
          Number(this.coef_.data[this.coef_.offset + j] ?? 0);
      }
      pred[i] += this.intercept_;
    }

    return tensor(pred);
  }

  /**
   * Return the R² score on the given test data.
   */
  score(X: Tensor, y: Tensor): number {
    if (!this.fitted) {
      throw new NotFittedError("ElasticNet must be fitted before scoring");
    }
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    assertContiguous(y, "y");
    for (let i = 0; i < y.size; i++) {
      const val = y.data[y.offset + i] ?? 0;
      if (!Number.isFinite(val)) {
        throw new DataValidationError("y contains non-finite values (NaN or Inf)");
      }
    }
    const pred = this.predict(X);
    if (pred.size !== y.size) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${pred.size}, y=${y.size}`
      );
    }
    let ssRes = 0,
      ssTot = 0;
    let yMean = 0;
    for (let i = 0; i < y.size; i++) {
      yMean += Number(y.data[y.offset + i] ?? 0);
    }
    yMean /= y.size;
    for (let i = 0; i < y.size; i++) {
      const yVal = Number(y.data[y.offset + i] ?? 0);
      const predVal = Number(pred.data[pred.offset + i] ?? 0);
      ssRes += (yVal - predVal) ** 2;
      ssTot += (yVal - yMean) ** 2;
    }
    if (ssTot === 0) {
      return ssRes === 0 ? 1.0 : 0.0;
    }
    return 1 - ssRes / ssTot;
  }

  get coef(): Tensor {
    if (!this.fitted || !this.coef_) {
      throw new NotFittedError("ElasticNet must be fitted to access coefficients");
    }
    return this.coef_;
  }

  get intercept(): number {
    if (!this.fitted) {
      throw new NotFittedError("ElasticNet must be fitted to access intercept");
    }
    return this.intercept_;
  }

  get nIter(): number | undefined {
    if (!this.fitted) {
      throw new NotFittedError("ElasticNet must be fitted to access nIter");
    }
    return this.nIter_;
  }

  getParams(): Record<string, unknown> {
    return { ...this.options };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "alpha":
          if (typeof value !== "number" || !Number.isFinite(value)) {
            throw new InvalidParameterError("alpha must be a finite number", "alpha", value);
          }
          this.options.alpha = value;
          break;
        case "l1Ratio":
          if (typeof value !== "number" || value < 0 || value > 1) {
            throw new InvalidParameterError("l1Ratio must be in [0, 1]", "l1Ratio", value);
          }
          this.options.l1Ratio = value;
          break;
        case "maxIter":
          if (typeof value !== "number" || !Number.isFinite(value)) {
            throw new InvalidParameterError("maxIter must be a finite number", "maxIter", value);
          }
          this.options.maxIter = value;
          break;
        case "tol":
          if (typeof value !== "number" || !Number.isFinite(value)) {
            throw new InvalidParameterError("tol must be a finite number", "tol", value);
          }
          this.options.tol = value;
          break;
        case "fitIntercept":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError(
              "fitIntercept must be a boolean",
              "fitIntercept",
              value
            );
          }
          this.options.fitIntercept = value;
          break;
        case "normalize":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("normalize must be a boolean", "normalize", value);
          }
          this.options.normalize = value;
          break;
        case "warmStart":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("warmStart must be a boolean", "warmStart", value);
          }
          this.options.warmStart = value;
          break;
        case "positive":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("positive must be a boolean", "positive", value);
          }
          this.options.positive = value;
          break;
        case "selection":
          if (value !== "cyclic" && value !== "random") {
            throw new InvalidParameterError(
              `Invalid selection: ${String(value)}`,
              "selection",
              value
            );
          }
          this.options.selection = value;
          break;
        case "randomState":
          if (typeof value !== "number" || !Number.isFinite(value)) {
            throw new InvalidParameterError(
              `randomState must be a finite number; received ${String(value)}`,
              "randomState",
              value
            );
          }
          this.options.randomState = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
