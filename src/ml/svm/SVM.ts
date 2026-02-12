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
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html | scikit-learn LinearSVC}
 */
export class LinearSVC implements Classifier {
  /** Regularization parameter (inverse of regularization strength) */
  private readonly C: number;

  /** Maximum number of iterations for optimization */
  private readonly maxIter: number;

  /** Tolerance for stopping criterion */
  private readonly tol: number;

  /** Weight vector of shape (n_features,) */
  private weights: number[] = [];

  /** Bias term */
  private bias = 0;

  /** Number of features seen during fit */
  private nFeatures = 0;

  /** Unique class labels [0, 1] mapped from original labels */
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
   * Fit the SVM classifier using sub-gradient descent.
   *
   * Uses a simplified hinge loss optimization with L2 regularization.
   * Objective: minimize (1/2)||w||² + C * Σmax(0, 1 - y_i(w · x_i + b))
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target labels of shape (n_samples,). Must contain exactly 2 classes.
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D or y is not 1D
   * @throws {ShapeError} If X and y have different number of samples
   * @throws {DataValidationError} If X or y contain NaN/Inf values
   * @throws {InvalidParameterError} If y does not contain exactly 2 classes
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

    // Get unique classes and map to {-1, 1} for SVM
    this.classLabels = [...new Set(yData)].sort((a, b) => a - b);
    if (this.classLabels.length !== 2) {
      throw new InvalidParameterError(
        "LinearSVC requires exactly 2 classes for binary classification",
        "y",
        this.classLabels.length
      );
    }

    // Map labels to {-1, 1}
    const yMapped = yData.map((label) => (label === this.classLabels[0] ? -1 : 1));

    // Initialize weights and bias
    this.weights = new Array(nFeatures).fill(0);
    this.bias = 0;

    // Sub-gradient descent optimization
    // We use a constant learning rate schedule for simplicity, scaled by 1/(lambda*n)
    // Here lambda = 1/C. So eta = C/n.
    // However, to ensure convergence, usually eta decays as 1/t.
    // For fixed iterations, small constant rate is often sufficient.
    // We choose eta such that eta * lambda * n = 1 (approx) implies eta = 1/(lambda*n) = C/n?
    // Let's use eta = 1 / (nSamples * lambda) = C / nSamples.
    // But if C is large, eta is large, which might be unstable.
    // Let's use a safe learning rate.
    const learningRate = 0.01; // Fixed small learning rate for stability

    for (let iter = 0; iter < this.maxIter; iter++) {
      let maxViolation = 0;

      for (let i = 0; i < nSamples; i++) {
        const xi = XData[i];
        const yi = yMapped[i];

        if (xi === undefined || yi === undefined) continue;

        // Compute decision function: w · x + b
        let decision = this.bias;
        for (let j = 0; j < nFeatures; j++) {
          decision += (this.weights[j] ?? 0) * (xi[j] ?? 0);
        }

        // Hinge loss margin: y * (w · x + b)
        const margin = yi * decision;

        // Track constraint violation for convergence check
        if (margin < 1) {
          maxViolation = Math.max(maxViolation, 1 - margin);
        }

        // Sub-gradient update
        // Objective: 0.5*|w|^2 + C * sum(max(0, 1 - y(wx+b)))
        // Grad w: w - C*y*x (if margin < 1)
        // Update: w <- w - eta * (w - C*y*x) = w(1-eta) + eta*C*y*x

        // Regularization part (always applied)
        // Let's trust the user input C implies hard margin.
        // We will use: w = w - learningRate * (w - C * y * x)
        // To prevent explosion, learningRate must be < 1/C.
        // So we adapt LR.
        const effectiveLR = Math.min(learningRate, 1.0 / (this.C * 10));

        if (margin < 1) {
          // Misclassified or within margin
          for (let j = 0; j < nFeatures; j++) {
            this.weights[j] =
              (this.weights[j] ?? 0) * (1 - effectiveLR) + effectiveLR * this.C * yi * (xi[j] ?? 0);
          }
          this.bias += effectiveLR * this.C * yi;
        } else {
          // Correctly classified outside margin: only apply regularization
          for (let j = 0; j < nFeatures; j++) {
            this.weights[j] = (this.weights[j] ?? 0) * (1 - effectiveLR);
          }
        }
      }

      // Check convergence
      if (maxViolation < this.tol) {
        break;
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
      // Compute decision function
      let decision = this.bias;
      for (let j = 0; j < nFeatures; j++) {
        decision += (this.weights[j] ?? 0) * Number(X.data[X.offset + i * nFeatures + j]);
      }

      // Map back to original labels
      const predictedClass = decision >= 0 ? this.classLabels[1] : this.classLabels[0];
      predictions.push(predictedClass ?? 0);
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

    const proba: number[][] = [];

    for (let i = 0; i < nSamples; i++) {
      // Compute decision function
      let decision = this.bias;
      for (let j = 0; j < nFeatures; j++) {
        decision += (this.weights[j] ?? 0) * Number(X.data[X.offset + i * nFeatures + j]);
      }

      // Use sigmoid for probability approximation (Platt scaling)
      const p1 = 1 / (1 + Math.exp(-decision));
      proba.push([1 - p1, p1]);
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
    return tensor([this.weights]);
  }

  /**
   * Get the bias term.
   *
   * @returns Bias value
   * @throws {NotFittedError} If the model has not been fitted
   */
  get intercept(): number {
    if (!this.fitted) {
      throw new NotFittedError("LinearSVC must be fitted to access intercept");
    }
    return this.bias;
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
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html | scikit-learn LinearSVR}
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
