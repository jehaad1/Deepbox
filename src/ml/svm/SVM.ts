import {
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  NotImplementedError,
  ShapeError,
} from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier, Regressor } from "../base";

/**
 * Support Vector Machine (SVM) Classifier.
 *
 * Implements a linear SVM using sub-gradient descent on the hinge loss
 * with L2 regularization (soft margin). Suitable for binary classification tasks.
 *
 * **Algorithm**: Sub-gradient descent on hinge loss (linear kernel)
 *
 * **Mathematical Formulation**:
 * - Decision function: f(x) = sign(w · x + b)
 * - Optimization: minimize (1/2)||w||² + C * Σmax(0, 1 - y_i(w · x_i + b))
 *
 * @example
 * ```ts
 * import { LinearSVC } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [2, 3], [3, 1], [4, 2]]);
 * const y = tensor([0, 0, 1, 1]);
 *
 * const svm = new LinearSVC({ C: 1.0 });
 * svm.fit(X, y);
 * const predictions = svm.predict(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-svm | Deepbox SVM}
 */
export class LinearSVC implements Classifier {
  /** Regularization parameter (inverse of regularization strength) */
  private readonly C: number;

  /** Maximum number of iterations for optimization */
  private readonly maxIter: number;

  /** Tolerance for stopping criterion */
  private readonly tol: number;

  /** Per-class weight vectors (OvR for multiclass, single for binary) */
  private weightsPerClass: number[][] = [];

  /** Per-class bias terms */
  private biasPerClass: number[] = [];

  /** Number of features seen during fit */
  private nFeatures = 0;

  /** Unique class labels */
  private classLabels: number[] = [];

  /** Whether the model has been fitted */
  private fitted = false;

  /**
   * Create a new SVM Classifier.
   *
   * @param options - Configuration options
   * @param options.C - Regularization parameter (default: 1.0). Larger C = stronger penalty on errors = harder margin.
   * @param options.maxIter - Maximum iterations (default: 1000)
   * @param options.tol - Convergence tolerance (default: 1e-4)
   */
  constructor(
    options: {
      readonly C?: number;
      readonly maxIter?: number;
      readonly tol?: number;
    } = {}
  ) {
    this.C = options.C ?? 1.0;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-4;

    // Validate parameters
    if (!Number.isFinite(this.C) || this.C <= 0) {
      throw new InvalidParameterError("C must be positive", "C", this.C);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter <= 0) {
      throw new InvalidParameterError(
        "maxIter must be a positive integer",
        "maxIter",
        this.maxIter
      );
    }
    if (!Number.isFinite(this.tol) || this.tol < 0) {
      throw new InvalidParameterError("tol must be >= 0", "tol", this.tol);
    }
  }

  /**
   * Fit a single binary SVM using sub-gradient descent on hinge loss.
   * Maps labels to {-1, +1} and returns learned weights + bias.
   */
  private fitBinary(
    XData: number[][],
    yMapped: number[],
    nSamples: number,
    nFeatures: number
  ): { weights: number[]; bias: number } {
    const weights = new Array<number>(nFeatures).fill(0);
    let bias = 0;
    const learningRate = 0.01;

    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxViolation = 0;

      for (let i = 0; i < nSamples; i++) {
        const xi = XData[i];
        const yi = yMapped[i];
        if (xi === undefined || yi === undefined) continue;

        let decision = bias;
        for (let j = 0; j < nFeatures; j++) {
          decision += (weights[j] ?? 0) * (xi[j] ?? 0);
        }

        const margin = yi * decision;
        if (margin < 1) {
          maxViolation = Math.max(maxViolation, 1 - margin);
        }

        const effectiveLR = Math.min(learningRate, 1.0 / (this.C * 10));

        if (margin < 1) {
          for (let j = 0; j < nFeatures; j++) {
            weights[j] =
              (weights[j] ?? 0) * (1 - effectiveLR) + effectiveLR * this.C * yi * (xi[j] ?? 0);
          }
          bias += effectiveLR * this.C * yi;
        } else {
          for (let j = 0; j < nFeatures; j++) {
            weights[j] = (weights[j] ?? 0) * (1 - effectiveLR);
          }
        }
      }

      if (maxViolation < this.tol) break;
    }

    return { weights, bias };
  }

  /**
   * Compute decision value for a single binary classifier.
   */
  private decisionBinary(x: number[], classIdx: number): number {
    const w = this.weightsPerClass[classIdx];
    let d = this.biasPerClass[classIdx] ?? 0;
    if (w) {
      for (let j = 0; j < w.length; j++) {
        d += (w[j] ?? 0) * (x[j] ?? 0);
      }
    }
    return d;
  }

  /**
   * Fit the SVM classifier using sub-gradient descent.
   *
   * Supports both binary and multiclass classification (via OvR).
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target labels of shape (n_samples,). Must contain at least 2 classes.
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D or y is not 1D
   * @throws {ShapeError} If X and y have different number of samples
   * @throws {DataValidationError} If X or y contain NaN/Inf values
   * @throws {InvalidParameterError} If y does not contain at least 2 classes
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    this.nFeatures = nFeatures;

    const XData: number[][] = [];
    const yData: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        row.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      XData.push(row);
      yData.push(Number(y.data[y.offset + i]));
    }

    this.classLabels = [...new Set(yData)].sort((a, b) => a - b);
    if (this.classLabels.length < 2) {
      throw new InvalidParameterError(
        "LinearSVC requires at least 2 classes",
        "y",
        this.classLabels.length
      );
    }

    this.weightsPerClass = [];
    this.biasPerClass = [];

    if (this.classLabels.length === 2) {
      // Binary: single SVM, map to {-1, +1}
      const yMapped = yData.map((label) => (label === this.classLabels[0] ? -1 : 1));
      const { weights, bias } = this.fitBinary(XData, yMapped, nSamples, nFeatures);
      this.weightsPerClass.push(weights);
      this.biasPerClass.push(bias);
    } else {
      // Multiclass: One-vs-Rest — one binary SVM per class
      for (const classLabel of this.classLabels) {
        const yMapped = yData.map((label) => (label === classLabel ? 1 : -1));
        const { weights, bias } = this.fitBinary(XData, yMapped, nSamples, nFeatures);
        this.weightsPerClass.push(weights);
        this.biasPerClass.push(bias);
      }
    }

    this.fitted = true;
    return this;
  }

  /**
   * Predict class labels for samples in X.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("SVC must be fitted before prediction");
    }

    validatePredictInputs(X, this.nFeatures ?? 0, "LinearSVC");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const predictions: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      const xi: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        xi.push(Number(X.data[X.offset + i * nFeatures + j]));
      }

      if (this.classLabels.length === 2) {
        // Binary
        const d = this.decisionBinary(xi, 0);
        predictions.push(d >= 0 ? (this.classLabels[1] ?? 0) : (this.classLabels[0] ?? 0));
      } else {
        // Multiclass OvR: pick class with highest decision value
        let bestClass = 0;
        let bestScore = -Infinity;
        for (let c = 0; c < this.classLabels.length; c++) {
          const score = this.decisionBinary(xi, c);
          if (score > bestScore) {
            bestScore = score;
            bestClass = c;
          }
        }
        predictions.push(this.classLabels[bestClass] ?? 0);
      }
    }

    return tensor(predictions, { dtype: "int32" });
  }

  /**
   * Predict class probabilities using Platt scaling approximation.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Probability estimates of shape (n_samples, 2)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predictProba(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("LinearSVC must be fitted before prediction");
    }

    validatePredictInputs(X, this.nFeatures ?? 0, "LinearSVC");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const nClasses = this.classLabels.length;
    const proba: number[][] = [];

    for (let i = 0; i < nSamples; i++) {
      const xi: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        xi.push(Number(X.data[X.offset + i * nFeatures + j]));
      }

      if (nClasses === 2) {
        const d = this.decisionBinary(xi, 0);
        const p1 = 1 / (1 + Math.exp(-d));
        proba.push([1 - p1, p1]);
      } else {
        // Softmax over per-class sigmoid scores
        const sigScores: number[] = [];
        for (let c = 0; c < nClasses; c++) {
          sigScores.push(1 / (1 + Math.exp(-this.decisionBinary(xi, c))));
        }
        const total = sigScores.reduce((s, v) => s + v, 0) || 1;
        proba.push(sigScores.map((v) => v / total));
      }
    }

    return tensor(proba);
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
    const predictions = this.predict(X);
    if (predictions.size !== y.size) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${predictions.size}, y=${y.size}`
      );
    }
    let correct = 0;
    for (let i = 0; i < y.size; i++) {
      if (Number(predictions.data[predictions.offset + i]) === Number(y.data[y.offset + i])) {
        correct++;
      }
    }
    return correct / y.size;
  }

  /**
   * Get the weight vector.
   *
   * @returns Weight vector as tensor of shape (1, n_features)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get coef(): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("LinearSVC must be fitted to access coefficients");
    }
    return tensor(this.weightsPerClass);
  }

  /**
   * Get the bias terms.
   *
   * @returns Bias values as tensor
   * @throws {NotFittedError} If the model has not been fitted
   */
  get intercept(): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("LinearSVC must be fitted to access intercept");
    }
    return tensor(this.biasPerClass);
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      C: this.C,
      maxIter: this.maxIter,
      tol: this.tol,
    };
  }

  /**
   * Set the parameters of this estimator.
   *
   * @param _params - Parameters to set
   * @throws {NotImplementedError} Always — parameters cannot be changed after construction
   */
  setParams(_params: Record<string, unknown>): this {
    throw new NotImplementedError("LinearSVC does not support setParams after construction");
  }
}

/**
 * Support Vector Machine (SVM) Regressor.
 *
 * Implements epsilon-SVR (Support Vector Regression) using sub-gradient descent.
 *
 * @example
 * ```ts
 * import { LinearSVR } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1], [2], [3], [4]]);
 * const y = tensor([1.5, 2.5, 3.5, 4.5]);
 *
 * const svr = new LinearSVR({ C: 1.0, epsilon: 0.1 });
 * svr.fit(X, y);
 * const predictions = svr.predict(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-svm | Deepbox SVM}
 */
export class LinearSVR implements Regressor {
  /** Regularization parameter */
  private readonly C: number;

  /** Epsilon in the epsilon-SVR model */
  private readonly epsilon: number;

  /** Maximum number of iterations */
  private readonly maxIter: number;

  /** Tolerance for stopping criterion */
  private readonly tol: number;

  /** Weight vector */
  private weights: number[] = [];

  /** Bias term */
  private bias = 0;

  /** Number of features */
  private nFeatures = 0;

  /** Whether the model has been fitted */
  private fitted = false;

  constructor(
    options: {
      readonly C?: number;
      readonly epsilon?: number;
      readonly maxIter?: number;
      readonly tol?: number;
    } = {}
  ) {
    this.C = options.C ?? 1.0;
    this.epsilon = options.epsilon ?? 0.1;
    this.maxIter = options.maxIter ?? 1000;
    this.tol = options.tol ?? 1e-4;

    if (!Number.isFinite(this.C) || this.C <= 0) {
      throw new InvalidParameterError("C must be positive", "C", this.C);
    }
    if (!Number.isFinite(this.epsilon) || this.epsilon < 0) {
      throw new InvalidParameterError("epsilon must be >= 0", "epsilon", this.epsilon);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter <= 0) {
      throw new InvalidParameterError("maxIter must be positive", "maxIter", this.maxIter);
    }
    if (!Number.isFinite(this.tol) || this.tol < 0) {
      throw new InvalidParameterError("tol must be >= 0", "tol", this.tol);
    }
  }

  /**
   * Fit the SVR model using sub-gradient descent on epsilon-insensitive loss.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target values of shape (n_samples,)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D or y is not 1D
   * @throws {ShapeError} If X and y have different number of samples
   * @throws {DataValidationError} If X or y contain NaN/Inf values
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    this.nFeatures = nFeatures;

    // Extract data
    const XData: number[][] = [];
    const yData: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        row.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      XData.push(row);
      yData.push(Number(y.data[y.offset + i]));
    }

    // Initialize weights
    this.weights = new Array(nFeatures).fill(0);
    this.bias = 0;

    const learningRate = 0.01;

    for (let iter = 0; iter < this.maxIter; iter++) {
      let totalLoss = 0;

      for (let i = 0; i < nSamples; i++) {
        const xi = XData[i];
        const yi = yData[i];

        if (xi === undefined || yi === undefined) continue;

        // Compute prediction
        let pred = this.bias;
        for (let j = 0; j < nFeatures; j++) {
          pred += (this.weights[j] ?? 0) * (xi[j] ?? 0);
        }

        const error = pred - yi;
        const absError = Math.abs(error);

        // Epsilon-insensitive loss
        if (absError > this.epsilon) {
          totalLoss += absError - this.epsilon;

          // Sub-gradient
          const sign = error > 0 ? 1 : -1;

          for (let j = 0; j < nFeatures; j++) {
            this.weights[j] =
              (this.weights[j] ?? 0) -
              learningRate * (this.C * sign * (xi[j] ?? 0) + (this.weights[j] ?? 0));
          }
          this.bias -= learningRate * this.C * sign;
        } else {
          // Only regularization
          for (let j = 0; j < nFeatures; j++) {
            this.weights[j] = (this.weights[j] ?? 0) - learningRate * (this.weights[j] ?? 0);
          }
        }
      }

      if (totalLoss / nSamples < this.tol) {
        break;
      }
    }

    this.fitted = true;
    return this;
  }

  /**
   * Predict target values for samples in X.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted values of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("SVR must be fitted before prediction");
    }

    validatePredictInputs(X, this.nFeatures ?? 0, "SVR");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    const predictions: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      let pred = this.bias;
      for (let j = 0; j < nFeatures; j++) {
        pred += (this.weights[j] ?? 0) * Number(X.data[X.offset + i * nFeatures + j]);
      }
      predictions.push(pred);
    }

    return tensor(predictions);
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
    const predictions = this.predict(X);
    if (predictions.size !== y.size) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${predictions.size}, y=${y.size}`
      );
    }

    let ssRes = 0;
    let ssTot = 0;
    let yMean = 0;

    for (let i = 0; i < y.size; i++) {
      yMean += Number(y.data[y.offset + i]);
    }
    yMean /= y.size;

    for (let i = 0; i < y.size; i++) {
      const yTrue = Number(y.data[y.offset + i]);
      const yPred = Number(predictions.data[predictions.offset + i]);
      ssRes += (yTrue - yPred) ** 2;
      ssTot += (yTrue - yMean) ** 2;
    }

    return ssTot === 0 ? (ssRes === 0 ? 1.0 : 0.0) : 1 - ssRes / ssTot;
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      C: this.C,
      epsilon: this.epsilon,
      maxIter: this.maxIter,
      tol: this.tol,
    };
  }

  /**
   * Set the parameters of this estimator.
   *
   * @param _params - Parameters to set
   * @throws {NotImplementedError} Always — parameters cannot be changed after construction
   */
  setParams(_params: Record<string, unknown>): this {
    throw new NotImplementedError("LinearSVR does not support setParams after construction");
  }
}
