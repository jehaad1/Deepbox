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
 * K-Nearest Neighbors base class.
 */
abstract class KNeighborsBase {
  protected readonly nNeighbors: number;
  protected readonly weights: "uniform" | "distance";
  protected readonly metric: "euclidean" | "manhattan";

  protected XTrain_?: Tensor;
  protected yTrain_?: Tensor;
  protected nFeaturesIn_?: number;
  protected fitted = false;

  constructor(
    options: {
      readonly nNeighbors?: number;
      readonly weights?: "uniform" | "distance";
      readonly metric?: "euclidean" | "manhattan";
    } = {}
  ) {
    this.nNeighbors = options.nNeighbors ?? 5;
    this.weights = options.weights ?? "uniform";
    this.metric = options.metric ?? "euclidean";

    if (!Number.isInteger(this.nNeighbors) || this.nNeighbors < 1) {
      throw new InvalidParameterError(
        "nNeighbors must be an integer >= 1",
        "nNeighbors",
        this.nNeighbors
      );
    }
    if (this.weights !== "uniform" && this.weights !== "distance") {
      throw new InvalidParameterError(
        `weights must be "uniform" or "distance"; received ${String(this.weights)}`,
        "weights",
        this.weights
      );
    }
    if (this.metric !== "euclidean" && this.metric !== "manhattan") {
      throw new InvalidParameterError(
        `metric must be "euclidean" or "manhattan"; received ${String(this.metric)}`,
        "metric",
        this.metric
      );
    }
  }

  protected calculateDistance(x1: number[], x2: number[]): number {
    let dist = 0;
    if (this.metric === "euclidean") {
      for (let i = 0; i < x1.length; i++) {
        const diff = (x1[i] ?? 0) - (x2[i] ?? 0);
        dist += diff * diff;
      }
      return Math.sqrt(dist);
    } else {
      // manhattan
      for (let i = 0; i < x1.length; i++) {
        dist += Math.abs((x1[i] ?? 0) - (x2[i] ?? 0));
      }
      return dist;
    }
  }

  protected findKNearest(sample: number[]): Array<{ index: number; distance: number }> {
    if (!this.XTrain_) {
      throw new NotFittedError("Model must be fitted before finding neighbors");
    }

    const nSamples = this.XTrain_.shape[0] ?? 0;
    const nFeatures = this.XTrain_.shape[1] ?? 0;

    const distances: Array<{ index: number; distance: number }> = [];

    for (let i = 0; i < nSamples; i++) {
      const trainSample: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        trainSample.push(Number(this.XTrain_.data[this.XTrain_.offset + i * nFeatures + j]));
      }
      const dist = this.calculateDistance(sample, trainSample);
      distances.push({ index: i, distance: dist });
    }

    distances.sort((a, b) => a.distance - b.distance);
    return distances.slice(0, this.nNeighbors);
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      nNeighbors: this.nNeighbors,
      weights: this.weights,
      metric: this.metric,
    };
  }

  /**
   * Set the parameters of this estimator.
   *
   * @param _params - Parameters to set
   * @throws {NotImplementedError} Always — parameters cannot be changed after construction
   */
  setParams(_params: Record<string, unknown>): this {
    throw new NotImplementedError("KNeighbors does not support setParams after construction");
  }
}

/**
 * K-Nearest Neighbors Classifier.
 *
 * Classification based on k nearest neighbors. Predicts class by majority vote
 * of k nearest training samples.
 *
 * **Algorithm**: Instance-based learning
 * 1. Store all training data
 * 2. For each test sample, find k nearest training samples
 * 3. Predict class by majority vote (or weighted vote)
 *
 * **Time Complexity**:
 * - Training: O(1) (just stores data)
 * - Prediction: O(n * d) per sample where n=training samples, d=features
 *
 * @example
 * ```ts
 * import { KNeighborsClassifier } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0], [1, 1], [2, 2], [3, 3]]);
 * const y = tensor([0, 0, 1, 1]);
 *
 * const knn = new KNeighborsClassifier({ nNeighbors: 3 });
 * knn.fit(X, y);
 *
 * const predictions = knn.predict(tensor([[1.5, 1.5]]));
 * ```
 *
 * @see {@link https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm | Wikipedia: KNN}
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html | scikit-learn KNeighborsClassifier}
 */
export class KNeighborsClassifier extends KNeighborsBase implements Classifier {
  /**
   * Fit the k-nearest neighbors classifier from the training set.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target class labels of shape (n_samples,)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D or y is not 1D
   * @throws {ShapeError} If X and y have different number of samples
   * @throws {DataValidationError} If X or y contain NaN/Inf values
   * @throws {InvalidParameterError} If nNeighbors > n_samples
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    if (this.nNeighbors > nSamples) {
      throw new InvalidParameterError(
        `nNeighbors must be <= n_samples; received ${this.nNeighbors} > ${nSamples}`,
        "nNeighbors",
        this.nNeighbors
      );
    }

    this.XTrain_ = X;
    this.yTrain_ = y;
    this.nFeaturesIn_ = nFeatures;
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
    if (!this.fitted || !this.XTrain_ || !this.yTrain_) {
      throw new NotFittedError("KNeighborsClassifier must be fitted before prediction");
    }

    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "KNeighborsClassifier");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const predictions: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      const sample: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        sample.push(Number(X.data[X.offset + i * nFeatures + j]));
      }

      const neighbors = this.findKNearest(sample);

      // Count votes for each class
      const votes = new Map<number, number>();

      for (const neighbor of neighbors) {
        const label = Number(this.yTrain_.data[this.yTrain_.offset + neighbor.index]);
        const weight = this.weights === "uniform" ? 1 : 1 / (neighbor.distance + 1e-10);
        votes.set(label, (votes.get(label) ?? 0) + weight);
      }

      // Find class with most votes
      let maxVotes = -1;
      let predictedClass = 0;
      for (const [label, voteCount] of votes.entries()) {
        if (voteCount > maxVotes) {
          maxVotes = voteCount;
          predictedClass = label;
        }
      }

      predictions.push(predictedClass);
    }

    return tensor(predictions, { dtype: "int32" });
  }

  /**
   * Predict class probabilities for samples in X.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Class probability matrix of shape (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predictProba(X: Tensor): Tensor {
    if (!this.fitted || !this.XTrain_ || !this.yTrain_) {
      throw new NotFittedError("KNeighborsClassifier must be fitted before prediction");
    }

    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "KNeighborsClassifier");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    // Get unique classes
    const classSet = new Set<number>();
    for (let i = 0; i < this.yTrain_.size; i++) {
      classSet.add(Number(this.yTrain_.data[this.yTrain_.offset + i]));
    }
    const classes = Array.from(classSet).sort((a, b) => a - b);

    const probabilities: number[][] = [];

    for (let i = 0; i < nSamples; i++) {
      const sample: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        sample.push(Number(X.data[X.offset + i * nFeatures + j]));
      }

      const neighbors = this.findKNearest(sample);

      // Count votes for each class
      const votes = new Map<number, number>();
      let totalWeight = 0;

      for (const neighbor of neighbors) {
        const label = Number(this.yTrain_.data[this.yTrain_.offset + neighbor.index]);
        const weight = this.weights === "uniform" ? 1 : 1 / (neighbor.distance + 1e-10);
        votes.set(label, (votes.get(label) ?? 0) + weight);
        totalWeight += weight;
      }

      // Convert to probabilities
      const probs: number[] = [];
      for (const cls of classes) {
        probs.push((votes.get(cls) ?? 0) / totalWeight);
      }
      probabilities.push(probs);
    }

    return tensor(probabilities);
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
    const yPred = this.predict(X);
    if (yPred.size !== y.size) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${yPred.size}, y=${y.size}`
      );
    }
    let correct = 0;

    for (let i = 0; i < y.size; i++) {
      if (Number(y.data[y.offset + i]) === Number(yPred.data[yPred.offset + i])) {
        correct++;
      }
    }

    return correct / y.size;
  }
}

/**
 * K-Nearest Neighbors Regressor.
 *
 * Regression based on k nearest neighbors. Predicts value as mean (or weighted mean)
 * of k nearest training samples.
 *
 * @example
 * ```ts
 * import { KNeighborsRegressor } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0], [1], [2], [3]]);
 * const y = tensor([0, 1, 4, 9]);
 *
 * const knn = new KNeighborsRegressor({ nNeighbors: 2 });
 * knn.fit(X, y);
 *
 * const predictions = knn.predict(tensor([[1.5]]));
 * ```
 *
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html | scikit-learn KNeighborsRegressor}
 */
export class KNeighborsRegressor extends KNeighborsBase implements Regressor {
  /**
   * Fit the k-nearest neighbors regressor from the training set.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target values of shape (n_samples,)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D or y is not 1D
   * @throws {ShapeError} If X and y have different number of samples
   * @throws {DataValidationError} If X or y contain NaN/Inf values
   * @throws {InvalidParameterError} If nNeighbors > n_samples
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    if (this.nNeighbors > nSamples) {
      throw new InvalidParameterError(
        `nNeighbors must be <= n_samples; received ${this.nNeighbors} > ${nSamples}`,
        "nNeighbors",
        this.nNeighbors
      );
    }

    this.XTrain_ = X;
    this.yTrain_ = y;
    this.nFeaturesIn_ = nFeatures;
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
    if (!this.fitted || !this.XTrain_ || !this.yTrain_) {
      throw new NotFittedError("KNeighborsRegressor must be fitted before prediction");
    }

    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "KNeighborsRegressor");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const predictions: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      const sample: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        sample.push(Number(X.data[X.offset + i * nFeatures + j]));
      }

      const neighbors = this.findKNearest(sample);

      // Calculate weighted mean
      let sumValues = 0;
      let sumWeights = 0;

      for (const neighbor of neighbors) {
        const value = Number(this.yTrain_.data[this.yTrain_.offset + neighbor.index]);
        const weight = this.weights === "uniform" ? 1 : 1 / (neighbor.distance + 1e-10);
        sumValues += value * weight;
        sumWeights += weight;
      }

      predictions.push(sumValues / sumWeights);
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
    const yPred = this.predict(X);
    if (yPred.size !== y.size) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${yPred.size}, y=${y.size}`
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
      const yPredVal = Number(yPred.data[yPred.offset + i]);
      ssRes += (yTrue - yPredVal) ** 2;
      ssTot += (yTrue - yMean) ** 2;
    }

    if (ssTot === 0) {
      return ssRes === 0 ? 1.0 : 0.0;
    }

    return 1 - ssRes / ssTot;
  }
}
