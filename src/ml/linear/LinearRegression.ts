import {
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  type ScalarDType,
  type Shape,
  ShapeError,
} from "../../core";
import { lstsq } from "../../linalg";
import { dot, mean, sub, type Tensor, tensor } from "../../ndarray";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Regressor } from "../base";

/**
 * Ordinary Least Squares Linear Regression.
 *
 * Fits a linear model with coefficients w = (w1, ..., wp) to minimize
 * the residual sum of squares between the observed targets and the
 * targets predicted by the linear approximation.
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
  /** Model coefficients (weights) of shape (n_features,) or (n_features, n_targets) */
  private coef_?: Tensor;

  /** Independent term (bias/intercept) in the linear model */
  private intercept_?: Tensor;

  /** Number of features seen during fit */
  private nFeaturesIn_?: number;

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
   * @param options.normalize - Whether to normalize features before regression (default: false)
   * @param options.copyX - Whether to copy X or overwrite it (default: true)
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
   * Uses SVD-based least squares solver for numerical stability.
   * When fitIntercept is true, centers the data before fitting.
   *
   * **Algorithm Complexity**: O(n * p^2) where n = samples, p = features
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target values of shape (n_samples,). Multi-output regression is not currently supported.
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D or y is not 1D
   * @throws {ShapeError} If X and y have different number of samples
   * @throws {DataValidationError} If X or y contain NaN/Inf values
   * @throws {DataValidationError} If X or y are empty
   */
  fit(X: Tensor, y: Tensor): this {
    // Validate inputs (dimensions, empty data, NaN/Inf)
    validateFitInputs(X, y);

    // Store number of features for validation during predict
    const nFeatures = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nFeatures;

    // Handle intercept by centering the data
    // This is more numerically stable than adding a column of ones
    const fitIntercept = this.options.fitIntercept ?? true;
    const copyX = this.options.copyX ?? true;
    const allowInPlace = copyX === false && (X.dtype === "float32" || X.dtype === "float64");

    if (fitIntercept) {
      // Compute mean of X along axis 0 (column means)
      const X_mean = mean(X, 0);
      // Compute mean of y
      const y_mean = mean(y);

      // Center X and y by subtracting means
      let X_processed = allowInPlace
        ? this.centerDataInPlace(X, X_mean)
        : this.centerData(X, X_mean);
      const y_centered = this.centerData(y, y_mean);

      // Handle normalization if requested
      let X_scale: Tensor | undefined;
      if (this.options.normalize) {
        // Compute L2 norm of centered X
        X_scale = this.computeL2Norm(X_processed);
        // Divide by norm
        X_processed = allowInPlace
          ? this.scaleDataInPlace(X_processed, X_scale)
          : this.scaleData(X_processed, X_scale);
      }

      // Solve least squares on processed data: X_processed * w = y_centered
      const result = lstsq(X_processed, y_centered);
      let w = result.x;

      // If normalized, rescale coefficients: w_original = w_normalized / scale
      if (X_scale) {
        w = this.rescaleCoefs(w, X_scale);
      }
      this.coef_ = w;

      // Compute intercept: b = y_mean - X_mean @ coef
      const X_mean_dot_coef = dot(X_mean, this.coef_);
      this.intercept_ = sub(y_mean, X_mean_dot_coef);
    } else {
      // No intercept - solve directly using least squares
      const result = lstsq(X, y);
      this.coef_ = result.x;
      // intercept_ remains undefined when fitIntercept is false
    }

    // Mark model as fitted
    this.fitted = true;
    return this;
  }

  /**
   * Center data by subtracting the mean.
   *
   * @param data - Input tensor to center
   * @param dataMean - Mean tensor to subtract
   * @returns Centered tensor
   */
  private centerData(data: Tensor, dataMean: Tensor): Tensor {
    // For 1D data, manually subtract mean to avoid dtype mismatch
    if (data.ndim === 1) {
      const n = data.size;
      const meanVal = Number(dataMean.data[dataMean.offset] ?? 0);
      const centered: number[] = [];
      for (let i = 0; i < n; i++) {
        centered.push(Number(data.data[data.offset + i] ?? 0) - meanVal);
      }
      return tensor(centered);
    }

    // For 2D data, broadcast subtract row mean from each row
    const nSamples = data.shape[0] ?? 0;
    const nFeatures = data.shape[1] ?? 0;

    const result: number[][] = [];

    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        const val = Number(data.data[data.offset + i * nFeatures + j] ?? 0);
        const meanVal = Number(dataMean.data[dataMean.offset + j] ?? 0);
        row.push(val - meanVal);
      }
      result.push(row);
    }

    return tensor(result);
  }

  private centerDataInPlace(data: Tensor, dataMean: Tensor): Tensor {
    if (data.ndim === 1) {
      const n = data.size;
      const meanVal = Number(dataMean.data[dataMean.offset] ?? 0);
      for (let i = 0; i < n; i++) {
        const idx = data.offset + i;
        data.data[idx] = Number(data.data[idx] ?? 0) - meanVal;
      }
      return data;
    }

    const nSamples = data.shape[0] ?? 0;
    const nFeatures = data.shape[1] ?? 0;
    for (let i = 0; i < nSamples; i++) {
      const rowBase = data.offset + i * nFeatures;
      for (let j = 0; j < nFeatures; j++) {
        const idx = rowBase + j;
        const meanVal = Number(dataMean.data[dataMean.offset + j] ?? 0);
        data.data[idx] = Number(data.data[idx] ?? 0) - meanVal;
      }
    }
    return data;
  }

  private computeL2Norm(X: Tensor): Tensor {
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const norms: number[] = new Array(nFeatures).fill(0);

    for (let j = 0; j < nFeatures; j++) {
      let sumSq = 0;
      for (let i = 0; i < nSamples; i++) {
        const val = Number(X.data[X.offset + i * nFeatures + j] ?? 0);
        sumSq += val * val;
      }
      norms[j] = Math.sqrt(sumSq);
    }
    return tensor(norms);
  }

  private scaleData(X: Tensor, scale: Tensor): Tensor {
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const result: number[][] = [];

    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        const val = Number(X.data[X.offset + i * nFeatures + j] ?? 0);
        const s = Number(scale.data[scale.offset + j] ?? 1);
        row.push(s === 0 ? 0 : val / s); // Handle zero norm
      }
      result.push(row);
    }
    return tensor(result);
  }

  private scaleDataInPlace(X: Tensor, scale: Tensor): Tensor {
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    for (let i = 0; i < nSamples; i++) {
      const rowBase = X.offset + i * nFeatures;
      for (let j = 0; j < nFeatures; j++) {
        const idx = rowBase + j;
        const s = Number(scale.data[scale.offset + j] ?? 1);
        X.data[idx] = s === 0 ? 0 : Number(X.data[idx] ?? 0) / s;
      }
    }
    return X;
  }

  private rescaleCoefs(coef: Tensor, scale: Tensor): Tensor {
    // coef is (n_features,) or (n_features, n_targets)
    // We need to divide each row j by scale[j]
    const nFeatures = coef.shape[0] ?? 0;
    const nTargets = coef.ndim > 1 ? (coef.shape[1] ?? 1) : 1;

    if (coef.ndim === 1) {
      const res: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        const c = Number(coef.data[coef.offset + j] ?? 0);
        const s = Number(scale.data[scale.offset + j] ?? 1);
        res.push(s === 0 ? 0 : c / s);
      }
      return tensor(res);
    }

    const result: number[][] = [];
    for (let j = 0; j < nFeatures; j++) {
      const row: number[] = [];
      const s = Number(scale.data[scale.offset + j] ?? 1);
      for (let k = 0; k < nTargets; k++) {
        const c = Number(coef.data[coef.offset + j * nTargets + k] ?? 0);
        row.push(s === 0 ? 0 : c / s);
      }
      result.push(row);
    }
    return tensor(result);
  }

  /**
   * Predict using the linear model.
   *
   * Computes y_pred = X * coef_ + intercept_
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted values of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor<Shape, ScalarDType> {
    if (!this.fitted || !this.coef_) {
      throw new NotFittedError("LinearRegression must be fitted before prediction");
    }

    // Validate input
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "LinearRegression");

    // Compute predictions: y_pred = X * w + b
    const y_pred_raw = dot(X, this.coef_);

    // Add intercept if model was fitted with fitIntercept=true
    if (this.intercept_ !== undefined) {
      // Manually add intercept to avoid dtype mismatch
      const interceptVal = Number(this.intercept_.data[this.intercept_.offset] ?? 0);
      const result: number[] = [];
      for (let i = 0; i < y_pred_raw.size; i++) {
        result.push(Number(y_pred_raw.data[y_pred_raw.offset + i] ?? 0) + interceptVal);
      }
      return tensor(result);
    }

    return y_pred_raw as Tensor<Shape, ScalarDType>;
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
   * Best possible score is 1.0, and it can be negative (worse than random).
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

    // Get predictions
    const y_pred = this.predict(X);
    if (y_pred.size !== y.size) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${y_pred.size}, y=${y.size}`
      );
    }

    // Compute residual sum of squares: SS_res = Σ(y_true - y_pred)^2
    // Manually compute to avoid dtype mismatch
    let ss_res = 0;
    for (let i = 0; i < y.size; i++) {
      const diff = Number(y.data[y.offset + i] ?? 0) - Number(y_pred.data[y_pred.offset + i] ?? 0);
      ss_res += diff * diff;
    }

    // Compute total sum of squares: SS_tot = Σ(y_true - y_mean)^2
    const y_mean_tensor = mean(y);
    const y_mean_val = Number(y_mean_tensor.data[y_mean_tensor.offset] ?? 0);
    let ss_tot = 0;
    for (let i = 0; i < y.size; i++) {
      const diff = Number(y.data[y.offset + i] ?? 0) - y_mean_val;
      ss_tot += diff * diff;
    }

    // R^2 = 1 - (SS_res / SS_tot)
    // Handle edge case where ss_tot is 0 (constant y)
    if (ss_tot === 0) {
      return ss_res === 0 ? 1.0 : 0.0;
    }

    return 1 - ss_res / ss_tot;
  }

  /**
   * Get the model coefficients (weights).
   *
   * @returns Coefficient tensor of shape (n_features,) or (n_features, n_targets)
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
   * @returns Intercept value or tensor
   * @throws {NotFittedError} If the model has not been fitted
   */
  get intercept(): Tensor | undefined {
    if (!this.fitted) {
      throw new NotFittedError("LinearRegression must be fitted to access intercept");
    }
    return this.intercept_;
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
   * Set the parameters of this estimator.
   *
   * @param params - Parameters to set
   * @returns this - The estimator
   */
  setParams(_params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(_params)) {
      switch (key) {
        case "fitIntercept":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError(
              `fitIntercept must be a boolean; received ${String(value)}`,
              "fitIntercept",
              value
            );
          }
          this.options.fitIntercept = value;
          break;
        case "normalize":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError(
              `normalize must be a boolean; received ${String(value)}`,
              "normalize",
              value
            );
          }
          this.options.normalize = value;
          break;
        case "copyX":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError(
              `copyX must be a boolean; received ${String(value)}`,
              "copyX",
              value
            );
          }
          this.options.copyX = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
