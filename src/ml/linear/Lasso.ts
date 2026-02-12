import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Regressor } from "../base";

/**
 * Lasso Regression (L1 Regularized Linear Regression).
 *
 * Lasso performs both regularization and feature selection by adding
 * an L1 penalty that can drive coefficients exactly to zero.
 *
 * @example
 * ```ts
 * import { Lasso } from 'deepbox/ml';
 *
 * const model = new Lasso({ alpha: 0.1, maxIter: 1000 });
 * model.fit(X_train, y_train);
 *
 * // Many coefficients will be exactly 0
 * console.log(model.coef);
 *
 * const predictions = model.predict(X_test);
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

  /** Whether the model has been fitted to data */
  private fitted = false;

  /**
   * Create a new Lasso Regression model.
   *
   * @param options - Configuration options
   * @param options.alpha - Regularization strength (default: 1.0). Must be >= 0. Controls sparsity of solution.
   * @param options.fitIntercept - Whether to calculate the intercept (default: true)
   * @param options.normalize - Whether to normalize features before regression (default: false)
   * @param options.maxIter - Maximum iterations for coordinate descent (default: 1000)
   * @param options.tol - Tolerance for convergence (default: 1e-4). Smaller = more precise but slower.
   * @param options.warmStart - Whether to reuse previous solution as initialization (default: false)
   * @param options.positive - Whether to force coefficients to be positive (default: false)
   * @param options.selection - Coordinate selection: 'cyclic' (default) or 'random'
   */
  constructor(
    options: {
      readonly alpha?: number;
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
    return Math.random;
  }

  /**
   * Fit Lasso regression model using Coordinate Descent.
   *
   * Solves the L1-regularized least squares problem:
   * minimize (1/(2*n)) ||y - Xw||² + α||w||₁
   *
   * **Algorithm**: Coordinate Descent with Soft Thresholding
   * 1. Initialize coefficients (warm start if enabled)
   * 2. For each iteration:
   *    - For each feature (cyclic or random order):
   *      - Compute residual correlation
   *      - Apply soft thresholding operator
   *      - Update predictions incrementally
   * 3. Check convergence based on coefficient changes
   *
   * **Time Complexity**: O(k * n * p) where k = iterations, n = samples, p = features
   * **Space Complexity**: O(n + p)
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target values of shape (n_samples,)
   * @returns this - The fitted estimator for method chaining
   * @throws {ShapeError} If X is not 2D or y is not 1D
   * @throws {ShapeError} If X and y have different number of samples
   * @throws {DataValidationError} If X or y contain NaN/Inf values
   * @throws {DataValidationError} If X or y are empty
   * @throws {InvalidParameterError} If alpha < 0
   */
  fit(X: Tensor, y: Tensor): this {
    // Validate inputs (dimensions, empty data, NaN/Inf)
    validateFitInputs(X, y);
    this.nIter_ = undefined;

    // Extract and validate regularization parameter
    const alpha = this.options.alpha ?? 1.0;
    if (!(alpha >= 0)) {
      throw new InvalidParameterError(`alpha must be >= 0; received ${alpha}`, "alpha", alpha);
    }

    // Extract optimization parameters
    const maxIter = this.options.maxIter ?? 1000;
    const tol = this.options.tol ?? 1e-4;
    const fitIntercept = this.options.fitIntercept ?? true;
    const normalize = this.options.normalize ?? false;
    const positive = this.options.positive ?? false;
    const selection = this.options.selection ?? "cyclic";
    const rng = this.createRNG();

    // Extract dimensions: m = number of samples, n = number of features
    const m = X.shape[0] ?? 0;
    const n = X.shape[1] ?? 0;

    // Store number of features for prediction validation
    this.nFeaturesIn_ = n;

    // Compute means for centering (if fitIntercept is true)
    // Centering is crucial for proper intercept calculation and numerical stability
    let yMean = 0;
    const xMean = new Array<number>(n).fill(0);

    if (fitIntercept) {
      // Compute sum of y values
      for (let i = 0; i < m; i++) {
        yMean += Number(y.data[y.offset + i] ?? 0);
      }

      // Compute sum of each feature column
      for (let i = 0; i < m; i++) {
        const rowBase = X.offset + i * n;
        for (let j = 0; j < n; j++) {
          xMean[j] = (xMean[j] ?? 0) + Number(X.data[rowBase + j] ?? 0);
        }
      }

      // Convert sums to means
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

    // Precompute column squared norms of centered X: (1/m) * Σᵢ xᵢⱼ²
    // This is used in the coordinate descent update formula
    // Caching these norms significantly improves performance (O(n) vs O(nm) per iteration)
    const colNorm2 = new Array<number>(n).fill(0);
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let i = 0; i < m; i++) {
        const xij = getX(i, j);
        s += xij * xij;
      }
      // Normalize by number of samples
      colNorm2[j] = m === 0 ? 0 : s / m;
    }

    // Initialize coefficients
    // If warm_start is enabled and we have previous coefficients, reuse them
    // Otherwise initialize to zeros
    const w = new Array<number>(n).fill(0);
    if (this.options.warmStart && this.coef_ && this.coef_.ndim === 1 && this.coef_.size === n) {
      // Copy previous coefficients for warm start
      for (let j = 0; j < n; j++) {
        w[j] = Number(this.coef_.data[this.coef_.offset + j] ?? 0);
      }
    }

    // Maintain current predictions in centered space: ŷ = X_centered @ w
    // We update this incrementally during coordinate descent for efficiency
    // This avoids recomputing all predictions from scratch each iteration
    const yHat = new Array<number>(m).fill(0);
    for (let i = 0; i < m; i++) {
      let pred = 0;
      for (let j = 0; j < n; j++) {
        pred += getX(i, j) * (w[j] ?? 0);
      }
      yHat[i] = pred;
    }

    // Precompute 1/m for efficiency
    const invM = m === 0 ? 0 : 1 / m;

    // Coordinate descent main loop
    // Each iteration updates all coefficients once
    for (let iter = 0; iter < maxIter; iter++) {
      // Track maximum coefficient change for convergence check
      let maxChange = 0;

      // Determine order of coordinate updates
      // Cyclic: iterate through features in order (more cache-friendly)
      // Random: shuffle features each iteration (can help convergence)
      let indices: number[] | null = null;
      if (selection === "random") {
        indices = Array.from({ length: n }, (_, j) => j);
        // Fisher-Yates shuffle for random selection
        for (let k = n - 1; k > 0; k--) {
          const r = Math.floor(rng() * (k + 1));
          const tmp = indices[k];
          indices[k] = indices[r] ?? 0;
          indices[r] = tmp ?? 0;
        }
      }

      // Update each coefficient using coordinate descent
      const iterOrder = indices ?? Array.from({ length: n }, (_, j) => j);
      for (const j of iterOrder) {
        const denom = colNorm2[j] ?? 0;

        // Skip features with zero variance (constant columns)
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

        // Compute correlation between feature j and current residual
        // rho = (1/m) * Σᵢ xᵢⱼ * (yᵢ - ŷᵢ + wⱼ * xᵢⱼ)
        // The term (wⱼ * xᵢⱼ) adds back the contribution of feature j to the residual
        let rho = 0;
        for (let i = 0; i < m; i++) {
          const xij = getX(i, j);
          const yi = Number(y.data[y.offset + i] ?? 0) - (fitIntercept ? yMean : 0);
          // Current residual plus contribution from feature j
          const r = yi - (yHat[i] ?? 0) + (w[j] ?? 0) * xij;
          rho += xij * r;
        }
        rho *= invM;

        // Apply soft thresholding operator (proximal operator for L1 norm)
        // This is the key step that induces sparsity in Lasso
        // newW = soft_threshold(rho, α) / ||xⱼ||²
        let newW = this.softThreshold(rho, alpha) / denom;

        // Enforce non-negativity constraint if requested
        if (positive && newW < 0) {
          newW = 0;
        }

        // Compute change in coefficient
        const delta = newW - (w[j] ?? 0);

        // Update predictions incrementally if coefficient changed
        // This is much more efficient than recomputing all predictions
        // ŷ_new = ŷ_old + Δwⱼ * xⱼ
        if (delta !== 0) {
          for (let i = 0; i < m; i++) {
            yHat[i] = (yHat[i] ?? 0) + delta * getX(i, j);
          }
        }

        // Update coefficient
        w[j] = newW;

        // Track maximum change for convergence check
        maxChange = Math.max(maxChange, Math.abs(delta));
      }

      // Check convergence: if no coefficient changed by more than tol, we're done
      if (maxChange < tol) {
        this.nIter_ = iter + 1;
        break;
      }
    }

    // Store final iteration count if we didn't converge early
    if (this.nIter_ === undefined) {
      this.nIter_ = maxIter;
    }

    // Rescale coefficients back to original feature space if normalized
    if (normalize && xScale) {
      for (let j = 0; j < n; j++) {
        const s = xScale[j] ?? 1;
        w[j] = s === 0 ? 0 : (w[j] ?? 0) / s;
      }
    }

    // Store final coefficients
    this.coef_ = tensor(w);

    // Compute intercept if needed
    // intercept = mean(y) - mean(X) @ coef
    // This accounts for the centering we did during optimization
    if (fitIntercept) {
      let xMeanDotW = 0;
      for (let j = 0; j < n; j++) {
        xMeanDotW += (xMean[j] ?? 0) * (w[j] ?? 0);
      }
      this.intercept_ = yMean - xMeanDotW;
    } else {
      this.intercept_ = 0;
    }

    // Mark model as fitted
    this.fitted = true;
    return this;
  }

  /**
   * Soft thresholding operator (proximal operator for L1 norm).
   *
   * This is the key operation in Lasso that induces sparsity.
   *
   * Formula:
   * - If x > λ: return x - λ
   * - If x < -λ: return x + λ
   * - Otherwise: return 0
   *
   * Geometrically, this "shrinks" x towards zero by λ,
   * and sets it exactly to zero if |x| ≤ λ.
   *
   * **Time Complexity**: O(1)
   *
   * @param x - Input value
   * @param lambda - Threshold parameter (regularization strength)
   * @returns Soft-thresholded value
   */
  private softThreshold(x: number, lambda: number): number {
    // Guard against non-finite inputs
    if (!Number.isFinite(x) || !Number.isFinite(lambda)) {
      throw new DataValidationError("Non-finite value encountered during soft-thresholding");
    }
    if (x > lambda) return x - lambda;
    if (x < -lambda) return x + lambda;
    return 0;
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
   * @returns Intercept value
   * @throws {NotFittedError} If the model has not been fitted
   */
  get intercept(): number {
    if (!this.fitted) {
      throw new NotFittedError("Lasso must be fitted to access intercept");
    }
    return this.intercept_;
  }

  /**
   * Get the number of iterations run by coordinate descent.
   *
   * @returns Number of iterations until convergence
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nIter(): number | undefined {
    if (!this.fitted) {
      throw new NotFittedError("Lasso must be fitted to access nIter");
    }
    return this.nIter_;
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
   * @returns Predicted values of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    // Check if model has been fitted
    if (!this.fitted || !this.coef_) {
      throw new NotFittedError("Lasso must be fitted before prediction");
    }

    // Validate input
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "Lasso");

    const m = X.shape[0] ?? 0; // Number of samples to predict
    const n = X.shape[1] ?? 0; // Number of features
    const pred = Array(m).fill(0);

    // Compute predictions: ŷ[i] = Σⱼ X[i,j] * coef[j] + intercept
    for (let i = 0; i < m; i++) {
      // Compute weighted sum of features
      for (let j = 0; j < n; j++) {
        pred[i] +=
          Number(X.data[X.offset + i * n + j] ?? 0) *
          Number(this.coef_.data[this.coef_.offset + j] ?? 0);
      }
      // Add intercept
      pred[i] += this.intercept_;
    }

    return tensor(pred);
  }

  /**
   * Return the R² score on the given test data and target values.
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

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return { ...this.options };
  }

  /**
   * Set the parameters of this estimator.
   *
   * @param params - Parameters to set (alpha, maxIter, tol, fitIntercept, normalize)
   * @returns this
   * @throws {InvalidParameterError} If any parameter value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "alpha":
          if (typeof value !== "number" || !Number.isFinite(value)) {
            throw new InvalidParameterError("alpha must be a finite number", "alpha", value);
          }
          this.options.alpha = value;
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
