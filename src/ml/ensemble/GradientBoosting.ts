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
import { DecisionTreeRegressor } from "../tree/DecisionTree";

/**
 * Gradient Boosting Regressor.
 *
 * Builds an additive model in a forward stage-wise fashion using
 * regression trees as weak learners. Optimizes squared error loss.
 *
 * **Algorithm**: Gradient Boosting with regression trees
 * - Stage-wise additive modeling
 * - Uses gradient of squared loss (residuals)
 *
 * @example
 * ```ts
 * import { GradientBoostingRegressor } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1], [2], [3], [4], [5]]);
 * const y = tensor([1.2, 2.1, 2.9, 4.0, 5.1]);
 *
 * const gbr = new GradientBoostingRegressor({ nEstimators: 100 });
 * gbr.fit(X, y);
 * const predictions = gbr.predict(X);
 * ```
 *
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html | scikit-learn GradientBoostingRegressor}
 */
export class GradientBoostingRegressor implements Regressor {
  /** Number of boosting stages (trees) */
  private nEstimators: number;

  /** Learning rate shrinks the contribution of each tree */
  private learningRate: number;

  /** Maximum depth of individual regression trees */
  private maxDepth: number;

  /** Minimum samples required to split */
  private minSamplesSplit: number;

  /** Array of weak learners (regression trees) */
  private estimators: DecisionTreeRegressor[] = [];

  /** Initial prediction (mean of targets) */
  private initPrediction = 0;

  /** Number of features */
  private nFeatures = 0;

  /** Whether the model has been fitted */
  private fitted = false;

  constructor(
    options: {
      readonly nEstimators?: number;
      readonly learningRate?: number;
      readonly maxDepth?: number;
      readonly minSamplesSplit?: number;
    } = {}
  ) {
    this.nEstimators = options.nEstimators ?? 100;
    this.learningRate = options.learningRate ?? 0.1;
    this.maxDepth = options.maxDepth ?? 3;
    this.minSamplesSplit = options.minSamplesSplit ?? 2;

    if (!Number.isInteger(this.nEstimators) || this.nEstimators <= 0) {
      throw new InvalidParameterError(
        "nEstimators must be a positive integer",
        "nEstimators",
        this.nEstimators
      );
    }
    if (!Number.isFinite(this.learningRate) || this.learningRate <= 0) {
      throw new InvalidParameterError(
        "learningRate must be positive",
        "learningRate",
        this.learningRate
      );
    }
    if (!Number.isInteger(this.maxDepth) || this.maxDepth < 1) {
      throw new InvalidParameterError(
        "maxDepth must be an integer >= 1",
        "maxDepth",
        this.maxDepth
      );
    }
    if (!Number.isInteger(this.minSamplesSplit) || this.minSamplesSplit < 2) {
      throw new InvalidParameterError(
        "minSamplesSplit must be an integer >= 2",
        "minSamplesSplit",
        this.minSamplesSplit
      );
    }
  }

  /**
   * Fit the gradient boosting regressor on training data.
   *
   * Builds an additive model by sequentially fitting regression trees
   * to the negative gradient (residuals) of the loss function.
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

    const yData: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      yData.push(Number(y.data[y.offset + i]));
    }

    // Initialize with mean prediction (F0)
    this.initPrediction = yData.reduce((sum, val) => sum + val, 0) / nSamples;

    // Current predictions
    const predictions = new Array<number>(nSamples).fill(this.initPrediction);

    // Build ensemble
    this.estimators = [];

    for (let m = 0; m < this.nEstimators; m++) {
      // Compute residuals (negative gradient of squared loss)
      const residuals: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        residuals.push((yData[i] ?? 0) - (predictions[i] ?? 0));
      }

      // Fit a regression tree to residuals
      const tree = new DecisionTreeRegressor({
        maxDepth: this.maxDepth,
        minSamplesSplit: this.minSamplesSplit,
        minSamplesLeaf: 1,
      });
      tree.fit(X, tensor(residuals));
      this.estimators.push(tree);

      // Update predictions
      const treePred = tree.predict(X);
      for (let i = 0; i < nSamples; i++) {
        predictions[i] =
          (predictions[i] ?? 0) + this.learningRate * Number(treePred.data[treePred.offset + i]);
      }
    }

    this.fitted = true;
    return this;
  }

  /**
   * Predict target values for samples in X.
   *
   * Aggregates the initial prediction and the scaled contributions of all trees.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted values of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("GradientBoostingRegressor must be fitted before prediction");
    }

    validatePredictInputs(X, this.nFeatures ?? 0, "GradientBoostingRegressor");

    const nSamples = X.shape[0] ?? 0;
    const predictions = new Array<number>(nSamples).fill(this.initPrediction);

    for (const tree of this.estimators) {
      const treePred = tree.predict(X);
      for (let i = 0; i < nSamples; i++) {
        predictions[i] =
          (predictions[i] ?? 0) + this.learningRate * Number(treePred.data[treePred.offset + i]);
      }
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
      nEstimators: this.nEstimators,
      learningRate: this.learningRate,
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
    };
  }

  /**
   * Set the parameters of this estimator.
   *
   * @param _params - Parameters to set
   * @throws {NotImplementedError} Always — parameters cannot be changed after construction
   */
  setParams(_params: Record<string, unknown>): this {
    throw new NotImplementedError(
      "GradientBoostingRegressor does not support setParams after construction"
    );
  }
}

/**
 * Gradient Boosting Classifier.
 *
 * Uses gradient boosting with shallow regression trees for binary classification.
 * Optimizes log loss (cross-entropy) using sigmoid function.
 *
 * @example
 * ```ts
 * import { GradientBoostingClassifier } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [2, 3], [3, 1], [4, 2]]);
 * const y = tensor([0, 0, 1, 1]);
 *
 * const gbc = new GradientBoostingClassifier({ nEstimators: 100 });
 * gbc.fit(X, y);
 * const predictions = gbc.predict(X);
 * ```
 *
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html | scikit-learn GradientBoostingClassifier}
 */
export class GradientBoostingClassifier implements Classifier {
  /** Number of boosting stages */
  private nEstimators: number;

  /** Learning rate */
  private learningRate: number;

  /** Maximum depth */
  private maxDepth: number;

  /** Minimum samples to split */
  private minSamplesSplit: number;

  /** Array of weak learners */
  private estimators: DecisionTreeRegressor[] = [];

  /** Initial log-odds prediction */
  private initPrediction = 0;

  /** Number of features */
  private nFeatures = 0;

  /** Unique class labels */
  private classLabels: number[] = [];

  /** Whether fitted */
  private fitted = false;

  constructor(
    options: {
      readonly nEstimators?: number;
      readonly learningRate?: number;
      readonly maxDepth?: number;
      readonly minSamplesSplit?: number;
    } = {}
  ) {
    this.nEstimators = options.nEstimators ?? 100;
    this.learningRate = options.learningRate ?? 0.1;
    this.maxDepth = options.maxDepth ?? 3;
    this.minSamplesSplit = options.minSamplesSplit ?? 2;

    if (!Number.isInteger(this.nEstimators) || this.nEstimators <= 0) {
      throw new InvalidParameterError(
        "nEstimators must be a positive integer",
        "nEstimators",
        this.nEstimators
      );
    }
    if (!Number.isFinite(this.learningRate) || this.learningRate <= 0) {
      throw new InvalidParameterError(
        "learningRate must be positive",
        "learningRate",
        this.learningRate
      );
    }
    if (!Number.isInteger(this.maxDepth) || this.maxDepth < 1) {
      throw new InvalidParameterError(
        "maxDepth must be an integer >= 1",
        "maxDepth",
        this.maxDepth
      );
    }
    if (!Number.isInteger(this.minSamplesSplit) || this.minSamplesSplit < 2) {
      throw new InvalidParameterError(
        "minSamplesSplit must be an integer >= 2",
        "minSamplesSplit",
        this.minSamplesSplit
      );
    }
  }

  /**
   * Fit the gradient boosting classifier on training data.
   *
   * Builds an additive model by sequentially fitting regression trees
   * to the pseudo-residuals (gradient of log loss).
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target class labels of shape (n_samples,). Must contain exactly 2 classes.
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

    const yData: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      yData.push(Number(y.data[y.offset + i]));
    }

    // Get unique classes
    this.classLabels = [...new Set(yData)].sort((a, b) => a - b);
    if (this.classLabels.length !== 2) {
      throw new InvalidParameterError(
        "GradientBoostingClassifier requires exactly 2 classes",
        "y",
        this.classLabels.length
      );
    }

    // Map to {0, 1}
    const yBinary = yData.map((label) => (label === this.classLabels[0] ? 0 : 1));

    // Initialize with log-odds
    const posCount = yBinary.filter((v) => v === 1).length;
    const negCount = nSamples - posCount;
    this.initPrediction = Math.log((posCount + 1) / (negCount + 1)); // Add smoothing

    // Current raw scores (log-odds)
    const rawScores = new Array<number>(nSamples).fill(this.initPrediction);

    // Build ensemble
    this.estimators = [];

    for (let m = 0; m < this.nEstimators; m++) {
      // Compute pseudo-residuals (gradient of log loss)
      const residuals: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        const prob = 1 / (1 + Math.exp(-(rawScores[i] ?? 0))); // Sigmoid
        const y_i = yBinary[i] ?? 0;
        residuals.push(y_i - prob);
      }

      // Fit regression tree to residuals
      const tree = new DecisionTreeRegressor({
        maxDepth: this.maxDepth,
        minSamplesSplit: this.minSamplesSplit,
        minSamplesLeaf: 1,
      });
      tree.fit(X, tensor(residuals));
      this.estimators.push(tree);

      // Update raw scores
      const treePred = tree.predict(X);
      for (let i = 0; i < nSamples; i++) {
        rawScores[i] =
          (rawScores[i] ?? 0) + this.learningRate * Number(treePred.data[treePred.offset + i]);
      }
    }

    this.fitted = true;
    return this;
  }

  /**
   * Predict class labels for samples in X.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted class labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("GradientBoostingClassifier must be fitted before prediction");
    }

    validatePredictInputs(X, this.nFeatures ?? 0, "GradientBoostingClassifier");

    const nSamples = X.shape[0] ?? 0;

    const rawScores = new Array<number>(nSamples).fill(this.initPrediction);
    for (const tree of this.estimators) {
      const treePred = tree.predict(X);
      for (let i = 0; i < nSamples; i++) {
        rawScores[i] =
          (rawScores[i] ?? 0) + this.learningRate * Number(treePred.data[treePred.offset + i]);
      }
    }

    const predictions: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      const prob = 1 / (1 + Math.exp(-(rawScores[i] ?? 0)));
      const predictedClass = prob >= 0.5 ? this.classLabels[1] : this.classLabels[0];
      predictions.push(predictedClass ?? 0);
    }

    return tensor(predictions, { dtype: "int32" });
  }

  /**
   * Predict class probabilities for samples in X.
   *
   * Returns a matrix of shape (n_samples, 2) where columns are
   * [P(class_0), P(class_1)].
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Class probability matrix of shape (n_samples, 2)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predictProba(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("GradientBoostingClassifier must be fitted before prediction");
    }

    validatePredictInputs(X, this.nFeatures ?? 0, "GradientBoostingClassifier");

    const nSamples = X.shape[0] ?? 0;
    const rawScores = new Array<number>(nSamples).fill(this.initPrediction);
    for (const tree of this.estimators) {
      const treePred = tree.predict(X);
      for (let i = 0; i < nSamples; i++) {
        rawScores[i] =
          (rawScores[i] ?? 0) + this.learningRate * Number(treePred.data[treePred.offset + i]);
      }
    }

    const proba: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
      const prob = 1 / (1 + Math.exp(-(rawScores[i] ?? 0)));
      proba.push([1 - prob, prob]);
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
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      nEstimators: this.nEstimators,
      learningRate: this.learningRate,
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
    };
  }

  /**
   * Set the parameters of this estimator.
   *
   * @param _params - Parameters to set
   * @throws {NotImplementedError} Always — parameters cannot be changed after construction
   */
  setParams(_params: Record<string, unknown>): this {
    throw new NotImplementedError(
      "GradientBoostingClassifier does not support setParams after construction"
    );
  }
}
