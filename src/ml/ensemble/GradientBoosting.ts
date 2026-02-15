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
 * @see {@link https://deepbox.dev/docs/ml-ensemble | Deepbox Ensemble Methods}
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
 * Uses gradient boosting with shallow regression trees for classification.
 * Supports both binary and multiclass classification.
 * - Binary: optimizes log loss using sigmoid function.
 * - Multiclass: uses One-vs-Rest (OvR) strategy, training one binary model per class.
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
 * @see {@link https://deepbox.dev/docs/ml-ensemble | Deepbox Ensemble Methods}
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

  /** Per-class arrays of weak learners (OvR for multiclass, single for binary) */
  private estimatorsPerClass: DecisionTreeRegressor[][] = [];

  /** Per-class initial log-odds predictions */
  private initPredictions: number[] = [];

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
   * Fit a single binary boosting ensemble.
   * Trains nEstimators regression trees to optimize log loss for a binary target.
   */
  private fitBinary(
    X: Tensor,
    yBinary: number[],
    nSamples: number
  ): { estimators: DecisionTreeRegressor[]; initPred: number } {
    const posCount = yBinary.filter((v) => v === 1).length;
    const negCount = nSamples - posCount;
    const initPred = Math.log((posCount + 1) / (negCount + 1));

    const rawScores = new Array<number>(nSamples).fill(initPred);
    const estimators: DecisionTreeRegressor[] = [];

    for (let m = 0; m < this.nEstimators; m++) {
      const residuals: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        const prob = 1 / (1 + Math.exp(-(rawScores[i] ?? 0)));
        residuals.push((yBinary[i] ?? 0) - prob);
      }

      const tree = new DecisionTreeRegressor({
        maxDepth: this.maxDepth,
        minSamplesSplit: this.minSamplesSplit,
        minSamplesLeaf: 1,
      });
      tree.fit(X, tensor(residuals));
      estimators.push(tree);

      const treePred = tree.predict(X);
      for (let i = 0; i < nSamples; i++) {
        rawScores[i] =
          (rawScores[i] ?? 0) + this.learningRate * Number(treePred.data[treePred.offset + i]);
      }
    }

    return { estimators, initPred };
  }

  /**
   * Compute raw scores for a single binary ensemble.
   */
  private predictRawBinary(X: Tensor, classIdx: number): number[] {
    const nSamples = X.shape[0] ?? 0;
    const rawScores = new Array<number>(nSamples).fill(this.initPredictions[classIdx] ?? 0);
    const estimators = this.estimatorsPerClass[classIdx] ?? [];
    for (const tree of estimators) {
      const treePred = tree.predict(X);
      for (let i = 0; i < nSamples; i++) {
        rawScores[i] =
          (rawScores[i] ?? 0) + this.learningRate * Number(treePred.data[treePred.offset + i]);
      }
    }
    return rawScores;
  }

  /**
   * Fit the gradient boosting classifier on training data.
   *
   * Builds an additive model by sequentially fitting regression trees
   * to the pseudo-residuals (gradient of log loss).
   * Supports binary (2 classes) and multiclass (>2 classes via OvR).
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target class labels of shape (n_samples,). Must contain at least 2 classes.
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

    const yData: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      yData.push(Number(y.data[y.offset + i]));
    }

    this.classLabels = [...new Set(yData)].sort((a, b) => a - b);
    if (this.classLabels.length < 2) {
      throw new InvalidParameterError(
        "GradientBoostingClassifier requires at least 2 classes",
        "y",
        this.classLabels.length
      );
    }

    this.estimatorsPerClass = [];
    this.initPredictions = [];

    if (this.classLabels.length === 2) {
      // Binary: single sigmoid model
      const yBinary = yData.map((label) => (label === this.classLabels[0] ? 0 : 1));
      const { estimators, initPred } = this.fitBinary(X, yBinary, nSamples);
      this.estimatorsPerClass.push(estimators);
      this.initPredictions.push(initPred);
    } else {
      // Multiclass: One-vs-Rest — one binary model per class
      for (const classLabel of this.classLabels) {
        const yBinary = yData.map((label) => (label === classLabel ? 1 : 0));
        const { estimators, initPred } = this.fitBinary(X, yBinary, nSamples);
        this.estimatorsPerClass.push(estimators);
        this.initPredictions.push(initPred);
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
    const predictions: number[] = [];

    if (this.classLabels.length === 2) {
      // Binary
      const rawScores = this.predictRawBinary(X, 0);
      for (let i = 0; i < nSamples; i++) {
        const prob = 1 / (1 + Math.exp(-(rawScores[i] ?? 0)));
        predictions.push(prob >= 0.5 ? (this.classLabels[1] ?? 0) : (this.classLabels[0] ?? 0));
      }
    } else {
      // Multiclass OvR: pick class with highest raw score
      const allScores: number[][] = [];
      for (let c = 0; c < this.classLabels.length; c++) {
        allScores.push(this.predictRawBinary(X, c));
      }
      for (let i = 0; i < nSamples; i++) {
        let bestClass = 0;
        let bestScore = -Infinity;
        for (let c = 0; c < this.classLabels.length; c++) {
          const score = allScores[c]?.[i] ?? 0;
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
   * Predict class probabilities for samples in X.
   *
   * Returns a matrix of shape (n_samples, n_classes).
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Class probability matrix of shape (n_samples, n_classes)
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
    const nClasses = this.classLabels.length;
    const proba: number[][] = [];

    if (nClasses === 2) {
      // Binary
      const rawScores = this.predictRawBinary(X, 0);
      for (let i = 0; i < nSamples; i++) {
        const prob = 1 / (1 + Math.exp(-(rawScores[i] ?? 0)));
        proba.push([1 - prob, prob]);
      }
    } else {
      // Multiclass OvR: softmax over per-class sigmoid scores
      const allScores: number[][] = [];
      for (let c = 0; c < nClasses; c++) {
        allScores.push(this.predictRawBinary(X, c));
      }
      for (let i = 0; i < nSamples; i++) {
        const sigScores: number[] = [];
        for (let c = 0; c < nClasses; c++) {
          sigScores.push(1 / (1 + Math.exp(-(allScores[c]?.[i] ?? 0))));
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
