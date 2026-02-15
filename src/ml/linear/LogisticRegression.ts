import {
  DataValidationError,
  DeepboxError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../../core";
import { reshape, type Tensor, tensor } from "../../ndarray";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier } from "../base";

/**
 * Logistic Regression (Binary and Multiclass Classification).
 *
 * Logistic regression uses the logistic (sigmoid) function to model
 * the probability of class membership.
 *
 * @example
 * ```ts
 * import { LogisticRegression } from 'deepbox/ml';
 *
 * // Binary classification
 * const model = new LogisticRegression({ C: 1.0, maxIter: 100 });
 * model.fit(X_train, y_train);
 *
 * const predictions = model.predict(X_test);
 * const probabilities = model.predictProba(X_test);
 *
 * // Multiclass classification
 * const multiModel = new LogisticRegression({
 *   multiClass: 'multinomial',
 *   solver: 'lbfgs'
 * });
 * multiModel.fit(X_train_multi, y_train_multi);
 * ```
 *
 * @category Linear Models
 * @implements {Classifier}
 */
export class LogisticRegression implements Classifier {
  private options: {
    penalty?: "l2" | "none";
    tol?: number;
    C?: number;
    fitIntercept?: boolean;
    maxIter?: number;
    learningRate?: number;
    multiClass?: "ovr" | "auto";
  };

  private coef_?: Tensor; // Shape (n_features,) for binary, (n_classes, n_features) for multiclass
  private intercept_?: number | number[]; // Scalar for binary, Array for multiclass
  private nFeaturesIn_?: number;
  private classes_?: Tensor;
  private fitted = false;
  private multiclass_ = false;

  /**
   * Create a new Logistic Regression classifier.
   *
   * @param options - Configuration options
   * @param options.penalty - Regularization type: 'l2' or 'none' (default: 'l2')
   * @param options.C - Inverse regularization strength (default: 1.0). Must be > 0.
   * @param options.tol - Tolerance for stopping criterion (default: 1e-4)
   * @param options.maxIter - Maximum number of iterations (default: 100)
   * @param options.fitIntercept - Whether to fit intercept (default: true)
   * @param options.learningRate - Learning rate for gradient descent (default: 0.1)
   * @param options.multiClass - Multiclass strategy: 'ovr' (One-vs-Rest) or 'auto' (default: 'auto')
   */
  constructor(
    options: {
      readonly penalty?: "l2" | "none";
      readonly tol?: number;
      readonly C?: number;
      readonly fitIntercept?: boolean;
      readonly maxIter?: number;
      readonly learningRate?: number;
      readonly multiClass?: "ovr" | "auto";
    } = {}
  ) {
    this.options = { ...options };

    const penalty = this.options.penalty ?? "l2";
    if (penalty !== "l2" && penalty !== "none") {
      throw new InvalidParameterError(
        `Only penalty='l2' or 'none' is supported; received ${String(penalty)}`,
        "penalty",
        penalty
      );
    }
    this.options.penalty = penalty;

    const multiClass = this.options.multiClass ?? "auto";
    if (multiClass !== "ovr" && multiClass !== "auto") {
      throw new InvalidParameterError(
        `multiClass must be 'ovr' or 'auto'; received ${String(multiClass)}`,
        "multiClass",
        multiClass
      );
    }
    this.options.multiClass = multiClass;

    if (this.options.C !== undefined && this.options.C <= 0) {
      throw new InvalidParameterError(
        `C must be > 0; received ${this.options.C}`,
        "C",
        this.options.C
      );
    }
    if (
      this.options.maxIter !== undefined &&
      (!Number.isFinite(this.options.maxIter) || this.options.maxIter <= 0)
    ) {
      throw new InvalidParameterError(
        `maxIter must be a positive finite number; received ${this.options.maxIter}`,
        "maxIter",
        this.options.maxIter
      );
    }
    if (
      this.options.tol !== undefined &&
      (!Number.isFinite(this.options.tol) || this.options.tol < 0)
    ) {
      throw new InvalidParameterError(
        `tol must be a finite number >= 0; received ${this.options.tol}`,
        "tol",
        this.options.tol
      );
    }
    if (
      this.options.learningRate !== undefined &&
      (!Number.isFinite(this.options.learningRate) || this.options.learningRate <= 0)
    ) {
      throw new InvalidParameterError(
        `learningRate must be a positive finite number; received ${this.options.learningRate}`,
        "learningRate",
        this.options.learningRate
      );
    }
  }

  /**
   * Numerically stable sigmoid function.
   *
   * Uses different formulations for positive and negative inputs
   * to avoid overflow:
   * - For z >= 0: σ(z) = 1 / (1 + exp(-z))
   * - For z < 0: σ(z) = exp(z) / (1 + exp(z))
   *
   * @param z - Input value
   * @returns Sigmoid output in [0, 1]
   */
  private sigmoid(z: number): number {
    // Guard against non-finite inputs
    if (!Number.isFinite(z)) {
      return z > 0 ? 1 : 0;
    }
    if (z >= 0) {
      const ez = Math.exp(-z);
      return 1 / (1 + ez);
    }
    const ez = Math.exp(z);
    return ez / (1 + ez);
  }

  private ensureFitted(): void {
    if (!this.fitted || !this.coef_) {
      throw new NotFittedError("LogisticRegression must be fitted before using this method");
    }
  }

  private _fitBinary(
    X: Tensor,
    y: Tensor,
    m: number,
    n: number,
    lambda: number
  ): { w: number[]; b: number } {
    const maxIter = this.options.maxIter ?? 100;
    const tol = this.options.tol ?? 1e-4;
    const lr = this.options.learningRate ?? 0.1;
    const fitIntercept = this.options.fitIntercept ?? true;

    // Initialize weights and bias
    const w = new Array<number>(n).fill(0);
    let b = 0;

    // Gradient descent training loop
    for (let iter = 0; iter < maxIter; iter++) {
      const gradW = new Array<number>(n).fill(0);
      let gradB = 0;

      // Compute gradients over all samples
      for (let i = 0; i < m; i++) {
        let z = fitIntercept ? b : 0;
        const rowBase = X.offset + i * n;
        for (let j = 0; j < n; j++) {
          z += Number(X.data[rowBase + j] ?? 0) * (w[j] ?? 0);
        }

        const yi = Number(y.data[y.offset + i] ?? 0);
        const pi = this.sigmoid(z);
        const error = pi - yi;

        gradB += error;
        for (let j = 0; j < n; j++) {
          gradW[j] = (gradW[j] ?? 0) + error * Number(X.data[rowBase + j] ?? 0);
        }
      }

      // Update weights with gradient descent + L2 regularization
      const invM = m === 0 ? 0 : 1 / m;
      let maxUpdate = 0;
      for (let j = 0; j < n; j++) {
        const g = (gradW[j] ?? 0) * invM + lambda * (w[j] ?? 0);
        const update = lr * g;
        w[j] = (w[j] ?? 0) - update;
        maxUpdate = Math.max(maxUpdate, Math.abs(update));
      }
      if (fitIntercept) {
        const gB = gradB * invM;
        const updateB = lr * gB;
        b -= updateB;
        maxUpdate = Math.max(maxUpdate, Math.abs(updateB));
      }

      // Check convergence
      if (maxUpdate < tol) {
        break;
      }
    }
    return { w, b: fitIntercept ? b : 0 };
  }

  /**
   * Fit logistic regression model.
   *
   * Uses gradient descent with L2 regularization.
   * Supports binary and multiclass (One-vs-Rest) classification.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target labels of shape (n_samples,)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D or y is not 1D
   * @throws {ShapeError} If X and y have different number of samples
   * @throws {DataValidationError} If X or y contain NaN/Inf values
   * @throws {DataValidationError} If X or y are empty
   * @throws {InvalidParameterError} If C <= 0 or penalty is invalid
   */
  fit(X: Tensor, y: Tensor): this {
    // Validate inputs (dimensions, empty data, NaN/Inf)
    validateFitInputs(X, y);

    const m = X.shape[0] ?? 0;
    const n = X.shape[1] ?? 0;
    this.nFeaturesIn_ = n;

    // Identify unique classes
    const yData = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      yData[i] = Number(y.data[y.offset + i]);
    }
    const uniqueClasses = [...new Set(yData)].sort((a, b) => a - b);
    this.classes_ = tensor(uniqueClasses);

    // Extract and validate hyperparameters
    const penalty = this.options.penalty ?? "l2";
    const C = this.options.C ?? 1.0;
    if (!(C > 0)) {
      throw new InvalidParameterError(`C must be > 0; received ${C}`, "C", C);
    }
    const lambda = penalty === "l2" ? 1 / C : 0;
    const multiClass = this.options.multiClass ?? "auto";

    if (uniqueClasses.length <= 2) {
      // Binary classification
      this.multiclass_ = false;

      // Map labels to 0/1 if they are not already
      let yBinary = y;
      if (uniqueClasses.length === 2 && (uniqueClasses[0] !== 0 || uniqueClasses[1] !== 1)) {
        // Map uniqueClasses[0] -> 0, uniqueClasses[1] -> 1
        const mappedData = new Float64Array(m);
        for (let i = 0; i < m; i++) {
          mappedData[i] = yData[i] === uniqueClasses[1] ? 1 : 0;
        }
        yBinary = tensor(mappedData);
      } else if (uniqueClasses.length === 1) {
        // If only 1 class, we still need 0/1 for the math, though it's degenerate
        const mappedData = new Float64Array(m);
        const target = uniqueClasses[0] === 1 ? 1 : 0; // If only class is 1, treat as all 1s
        mappedData.fill(target);
        yBinary = tensor(mappedData);
      } else {
        // Check if they are 0 and 1
        for (const val of uniqueClasses) {
          if (val !== 0 && val !== 1) {
            // Should be caught above, but as fallback
            throw new DataValidationError("Binary classification expects labels 0 and 1");
          }
        }
      }

      const { w, b } = this._fitBinary(X, yBinary, m, n, lambda);
      this.coef_ = tensor(w);
      this.intercept_ = b;
    } else {
      // Multiclass (One-vs-Rest)
      if (multiClass !== "ovr" && multiClass !== "auto") {
        throw new InvalidParameterError(
          `multiClass must be 'ovr' or 'auto'; received ${String(multiClass)}`,
          "multiClass",
          multiClass
        );
      }
      this.multiclass_ = true;
      const nClasses = uniqueClasses.length;
      const allCoefs: number[] = []; // Flattened (nClasses * nFeatures)
      const allIntercepts: number[] = [];

      for (let k = 0; k < nClasses; k++) {
        const targetClass = uniqueClasses[k];
        // Create binary target for class k
        const yBinaryData = new Float64Array(m);
        for (let i = 0; i < m; i++) {
          yBinaryData[i] = yData[i] === targetClass ? 1 : 0;
        }
        const yBinary = tensor(yBinaryData);

        const { w, b } = this._fitBinary(X, yBinary, m, n, lambda);
        allCoefs.push(...w);
        allIntercepts.push(b);
      }

      const allCoefsTensor = tensor(allCoefs);
      this.coef_ = reshape(allCoefsTensor, [nClasses, n]);
      this.intercept_ = allIntercepts;
    }

    this.fitted = true;
    return this;
  }

  get classes(): Tensor | undefined {
    return this.classes_;
  }

  /**
   * Get the model coefficients (weights).
   *
   * @returns Coefficient tensor of shape (n_features,)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get coef(): Tensor {
    this.ensureFitted();
    const coef = this.coef_;
    if (!coef) {
      throw new DeepboxError("Internal error: coef_ is missing after ensureFitted() ");
    }
    return coef;
  }

  /**
   * Get the intercept (bias term).
   *
   * @returns Intercept value (scalar for binary, array for multiclass)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get intercept(): number | number[] {
    this.ensureFitted();
    if (this.intercept_ === undefined) {
      return 0;
    }
    return this.intercept_;
  }

  predict(X: Tensor): Tensor {
    const proba = this.predictProba(X);
    const m = X.shape[0] ?? 0;
    const pred = new Array<number>(m).fill(0);

    if (this.multiclass_) {
      const classes = this.classes_;
      if (!classes) {
        throw new NotFittedError("Model not fitted (classes_ missing)");
      }
      const nClasses = classes.size;
      for (let i = 0; i < m; i++) {
        let maxProb = -1;
        let maxClassIdx = 0;
        for (let k = 0; k < nClasses; k++) {
          const p = Number(proba.data[proba.offset + i * nClasses + k]);
          if (p > maxProb) {
            maxProb = p;
            maxClassIdx = k;
          }
        }
        // Map index back to class label
        pred[i] = Number(classes.data[classes.offset + maxClassIdx]);
      }
    } else {
      const classes = this.classes_;
      if (!classes) {
        throw new NotFittedError("Model not fitted (classes_ missing)");
      }

      // Handle binary classification mapping
      // If we have 2 classes, map 0->classes[0], 1->classes[1]
      // If we have 1 class, we always predict that class (degenerate case)
      if (classes.size === 1) {
        const cls = Number(classes.data[classes.offset]);
        pred.fill(cls);
      } else {
        const cls0 = Number(classes.data[classes.offset]);
        const cls1 = Number(classes.data[classes.offset + 1]);

        for (let i = 0; i < m; i++) {
          const p1 = Number(proba.data[proba.offset + i * 2 + 1]);
          pred[i] = p1 >= 0.5 ? cls1 : cls0;
        }
      }
    }
    return tensor(pred);
  }

  /**
   * Predict class probabilities for samples.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Probabilities of shape (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predictProba(X: Tensor): Tensor {
    this.ensureFitted();

    const coef = this.coef_;
    if (!coef) {
      throw new DeepboxError("Internal error: coef_ is missing after ensureFitted()");
    }

    // Validate input
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "LogisticRegression");

    const m = X.shape[0] ?? 0;
    const n = X.shape[1] ?? 0;

    if (this.multiclass_) {
      const nClasses = this.classes_?.size ?? 0;
      const proba = new Float64Array(m * nClasses);
      const interceptValue = this.intercept_;
      if (!Array.isArray(interceptValue)) {
        throw new DeepboxError("Internal error: intercept_ must be an array for multiclass");
      }

      for (let i = 0; i < m; i++) {
        const rowBase = X.offset + i * n;
        let sumExp = 0;
        const scores = new Array<number>(nClasses).fill(0);

        // Compute scores for each class
        for (let k = 0; k < nClasses; k++) {
          let z = interceptValue[k] ?? 0;
          const coefRowBase = coef.offset + k * n;
          for (let j = 0; j < n; j++) {
            z += Number(X.data[rowBase + j] ?? 0) * Number(coef.data[coefRowBase + j] ?? 0);
          }
          // For OvR, we apply sigmoid to get probability of class k vs rest
          // Then we normalize these probabilities to sum to 1
          scores[k] = this.sigmoid(z);
          sumExp += scores[k] ?? 0;
        }

        // Normalize
        for (let k = 0; k < nClasses; k++) {
          // If sum is 0 (unlikely with sigmoid), avoid NaN
          proba[i * nClasses + k] = sumExp > 0 ? (scores[k] ?? 0) / sumExp : 1.0 / nClasses;
        }
      }

      return tensor(proba, { dtype: "float64" }).reshape([m, nClasses]);
    } else {
      // Binary case
      const proba = new Array<number>(m * 2).fill(0);
      const interceptValue = this.intercept_;
      if (Array.isArray(interceptValue) || typeof interceptValue !== "number") {
        throw new DeepboxError("Internal error: intercept_ must be a number for binary case");
      }

      for (let i = 0; i < m; i++) {
        let z = interceptValue;
        const rowBase = X.offset + i * n;
        for (let j = 0; j < n; j++) {
          z += Number(X.data[rowBase + j] ?? 0) * Number(coef.data[coef.offset + j] ?? 0);
        }

        const p1 = this.sigmoid(z);
        proba[i * 2 + 0] = 1 - p1;
        proba[i * 2 + 1] = p1;
      }

      return tensor(proba).reshape([m, 2]);
    }
  }

  /**
   * Return the mean accuracy on the given test data and labels.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True labels of shape (n_samples,)
   * @returns Accuracy score in range [0, 1]
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional or sample counts mismatch
   * @throws {DataValidationError} If y contains NaN/Inf values
   */
  score(X: Tensor, y: Tensor): number {
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
    let correct = 0;

    for (let i = 0; i < y.size; i++) {
      if (Number(pred.data[pred.offset + i] ?? 0) === Number(y.data[y.offset + i] ?? 0)) {
        correct++;
      }
    }
    return correct / y.size;
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
   * @param params - Parameters to set (maxIter, tol, C, learningRate, penalty, fitIntercept)
   * @returns this
   * @throws {InvalidParameterError} If any parameter value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "maxIter":
          if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
            throw new InvalidParameterError(
              "maxIter must be a positive finite number",
              "maxIter",
              value
            );
          }
          this.options.maxIter = value;
          break;
        case "tol":
          if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
            throw new InvalidParameterError("tol must be a finite number >= 0", "tol", value);
          }
          this.options.tol = value;
          break;
        case "C":
          if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
            throw new InvalidParameterError("C must be a positive finite number", "C", value);
          }
          this.options.C = value;
          break;
        case "learningRate":
          if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
            throw new InvalidParameterError(
              "learningRate must be a positive finite number",
              "learningRate",
              value
            );
          }
          this.options.learningRate = value;
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
        case "penalty":
          if (value !== "none" && value !== "l2") {
            throw new InvalidParameterError(
              `Only penalty='l2' or 'none' is supported; received ${String(value)}`,
              "penalty",
              value
            );
          }
          this.options.penalty = value;
          break;
        case "multiClass":
          if (value !== "ovr" && value !== "auto") {
            throw new InvalidParameterError(
              `multiClass must be 'ovr' or 'auto'; received ${String(value)}`,
              "multiClass",
              value
            );
          }
          this.options.multiClass = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
