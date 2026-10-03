/**
 * K-nearest-neighbors estimators: classifier, regressor and unsupervised neighbor search.
 *
 * @module ml/neighbors
 * @see {@link https://deepbox.dev/docs/ml-neighbors | Deepbox Nearest Neighbors}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import {
  assertContiguous,
  toFloat64View,
  validateFitInputs,
  validatePredictInputs,
  validateUnsupervisedFitInputs,
} from "../_validation";
import type { Classifier, EstimatorTags, Regressor } from "../base";

type NeighborWeights = "uniform" | "distance";
type NeighborMetric = "euclidean" | "manhattan";

/**
 * Inverse-distance weights for a list of neighbor distances, as in scikit-learn.
 *
 * Each weight is `1 / distance`. If any neighbor sits at distance zero (or so close
 * that the inverse overflows), those neighbors get weight 1 and all others weight 0,
 * so an exact match decides the prediction without producing infinities.
 *
 * @param distances - Distances from one query point to its neighbors
 * @returns One weight per distance
 *
 * @internal
 */
export function inverseDistanceWeights(distances: ArrayLike<number>): number[] {
  const n = distances.length;
  const weights = new Array<number>(n);
  let anyExact = false;
  for (let i = 0; i < n; i++) {
    const w = 1 / (distances[i] as number);
    weights[i] = w;
    if (!Number.isFinite(w)) anyExact = true;
  }
  if (anyExact) {
    for (let i = 0; i < n; i++) weights[i] = Number.isFinite(weights[i] as number) ? 0 : 1;
  }
  return weights;
}

/**
 * Check a score target: 1-D, contiguous, finite and non-empty.
 *
 * @internal
 */
export function assertScoreTarget(y: Tensor): void {
  if (y.ndim !== 1) {
    throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
  }
  assertContiguous(y, "y");
  if (y.size === 0) {
    throw new DataValidationError("y must contain at least one sample");
  }
  for (let i = 0; i < y.size; i++) {
    const val = y.data[y.offset + i] ?? 0;
    if (typeof val === "number" && !Number.isFinite(val)) {
      throw new DataValidationError("y contains non-finite values (NaN or Inf)");
    }
  }
}

/**
 * Throw a ShapeError unless the prediction and target lengths match.
 *
 * @internal
 */
export function assertSameSampleCount(nPredicted: number, nTrue: number): void {
  if (nPredicted !== nTrue) {
    throw new ShapeError(
      `X and y must have the same number of samples; got X=${nPredicted}, y=${nTrue}`
    );
  }
}

/**
 * Build a 1-D label tensor: int32 when every class is an int32 integer, float64 otherwise.
 *
 * @internal
 */
export function labelTensor(values: ArrayLike<number>, classes: readonly number[]): Tensor {
  const integral = classes.every((c) => Number.isInteger(c) && c >= -2147483648 && c <= 2147483647);
  return integral ? tensor(Int32Array.from(values)) : tensor(Float64Array.from(values));
}

/**
 * K-Nearest Neighbors base class.
 */
abstract class KNeighborsBase {
  protected nNeighbors: number;
  protected weights: NeighborWeights;
  protected metric: NeighborMetric;

  /** Row-major snapshot of the training samples (taken at fit time). */
  protected xTrain_?: Float64Array;
  protected nTrain_ = 0;
  protected nFeaturesIn_ = 0;
  protected fitted = false;

  constructor(
    options: {
      readonly nNeighbors?: number;
      readonly weights?: NeighborWeights;
      readonly metric?: NeighborMetric;
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

  /** Copy the (validated) training matrix so later edits of `X` cannot change the model. */
  protected storeTrainingData(X: Tensor): void {
    this.xTrain_ = Float64Array.from(toFloat64View(X));
    this.nTrain_ = X.shape[0] ?? 0;
    this.nFeaturesIn_ = X.shape[1] ?? 0;
  }

  /**
   * Find the `k` nearest training samples of one query row.
   *
   * Results are ordered by distance; equal distances keep training order.
   *
   * @param query - Flat array holding the query row
   * @param offset - Index of the row's first feature in `query`
   * @param k - Number of neighbors to return (at most the number of candidates)
   * @param exclude - Training index to skip (used to leave out a sample itself), or -1
   */
  protected findNearest(
    query: Float64Array,
    offset: number,
    k: number,
    exclude = -1
  ): Array<{ index: number; distance: number }> {
    const train = this.xTrain_;
    if (!train) {
      throw new NotFittedError("Model must be fitted before finding neighbors");
    }
    const nF = this.nFeaturesIn_;
    const euclidean = this.metric === "euclidean";
    const bestKey: number[] = [];
    const bestIdx: number[] = [];

    for (let i = 0; i < this.nTrain_; i++) {
      if (i === exclude) continue;
      const full = bestKey.length === k;
      const limit = full ? (bestKey[k - 1] as number) : Number.POSITIVE_INFINITY;
      const base = i * nF;
      let acc = 0;
      for (let f = 0; f < nF; f++) {
        const diff = (query[offset + f] as number) - (train[base + f] as number);
        acc += euclidean ? diff * diff : Math.abs(diff);
        if (full && acc >= limit) break;
      }
      // Training indices ascend, so a candidate that only ties the k-th key loses.
      if (full && acc >= limit) continue;

      let pos = bestKey.length;
      while (pos > 0 && (bestKey[pos - 1] as number) > acc) pos--;
      bestKey.splice(pos, 0, acc);
      bestIdx.splice(pos, 0, i);
      if (bestKey.length > k) {
        bestKey.pop();
        bestIdx.pop();
      }
    }

    return bestKey.map((key, j) => ({
      index: bestIdx[j] as number,
      distance: euclidean ? Math.sqrt(key) : key,
    }));
  }

  /** Find every training sample within `radius` of one query row, nearest first. */
  protected findWithinRadius(
    query: Float64Array,
    offset: number,
    radius: number,
    exclude = -1
  ): Array<{ index: number; distance: number }> {
    const train = this.xTrain_;
    if (!train) {
      throw new NotFittedError("Model must be fitted before finding neighbors");
    }
    const nF = this.nFeaturesIn_;
    const euclidean = this.metric === "euclidean";
    const limit = euclidean ? radius * radius : radius;
    const hits: Array<{ index: number; key: number }> = [];

    for (let i = 0; i < this.nTrain_; i++) {
      if (i === exclude) continue;
      const base = i * nF;
      let acc = 0;
      for (let f = 0; f < nF; f++) {
        const diff = (query[offset + f] as number) - (train[base + f] as number);
        acc += euclidean ? diff * diff : Math.abs(diff);
        if (acc > limit) break;
      }
      if (acc <= limit) hits.push({ index: i, key: acc });
    }

    hits.sort((a, b) => a.key - b.key || a.index - b.index);
    return hits.map((h) => ({
      index: h.index,
      distance: euclidean ? Math.sqrt(h.key) : h.key,
    }));
  }

  /** Throw if `nNeighbors` (possibly changed by `setParams`) exceeds the training set size. */
  protected assertEnoughTrainingSamples(): void {
    if (this.nNeighbors > this.nTrain_) {
      throw new InvalidParameterError(
        `nNeighbors must be <= n_samples; received ${this.nNeighbors} > ${this.nTrain_}`,
        "nNeighbors",
        this.nNeighbors
      );
    }
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
   * @param params - Parameters to set
   * @throws {InvalidParameterError} If a parameter value is invalid or unknown
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nNeighbors":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError(
              "nNeighbors must be an integer >= 1",
              "nNeighbors",
              value
            );
          }
          this.nNeighbors = value;
          break;
        case "weights":
          if (value !== "uniform" && value !== "distance") {
            throw new InvalidParameterError(
              `weights must be "uniform" or "distance"; got ${String(value)}`,
              "weights",
              value
            );
          }
          this.weights = value;
          break;
        case "metric":
          if (value !== "euclidean" && value !== "manhattan") {
            throw new InvalidParameterError(
              `metric must be "euclidean" or "manhattan"; got ${String(value)}`,
              "metric",
              value
            );
          }
          this.metric = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}

/**
 * K-Nearest Neighbors Classifier.
 *
 * Classification based on k nearest neighbors. Predicts class by majority vote
 * of k nearest training samples. When classes tie, the smallest label wins, as in
 * scikit-learn.
 *
 * **Algorithm**: Instance-based learning
 * 1. Store all training data
 * 2. For each test sample, find k nearest training samples
 * 3. Predict class by majority vote (or inverse-distance weighted vote)
 *
 * **Time Complexity**:
 * - Training: O(n * d) (copies the data)
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
 * @see {@link https://deepbox.dev/docs/ml-neighbors | Deepbox Nearest Neighbors}
 */
export class KNeighborsClassifier extends KNeighborsBase implements Classifier {
  private classes_: number[] = [];
  /** Class index (position in `classes_`) of every training sample. */
  private yIndex_?: Int32Array;

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
    if (this.nNeighbors > nSamples) {
      throw new InvalidParameterError(
        `nNeighbors must be <= n_samples; received ${this.nNeighbors} > ${nSamples}`,
        "nNeighbors",
        this.nNeighbors
      );
    }

    const labels = toFloat64View(y);
    const classes = [...new Set(labels)].sort((a, b) => a - b);
    const classIndex = new Map<number, number>();
    classes.forEach((c, i) => {
      classIndex.set(c, i);
    });
    const yIndex = new Int32Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      yIndex[i] = classIndex.get(labels[i] as number) as number;
    }

    this.storeTrainingData(X);
    this.classes_ = classes;
    this.yIndex_ = yIndex;
    this.fitted = true;

    return this;
  }

  /**
   * Predict class labels for samples in X.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted class labels of shape (n_samples,); int32 when all training
   *   labels are integers, float64 otherwise
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    const proba = this.votes(X, "KNeighborsClassifier");
    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classes_.length;
    const predictions = new Float64Array(nSamples);

    for (let i = 0; i < nSamples; i++) {
      // Strict comparison over ascending classes: the smallest label wins ties.
      let best = 0;
      for (let c = 1; c < nClasses; c++) {
        if ((proba[i * nClasses + c] as number) > (proba[i * nClasses + best] as number)) best = c;
      }
      predictions[i] = this.classes_[best] as number;
    }

    return labelTensor(predictions, this.classes_);
  }

  /**
   * Predict class probabilities for samples in X.
   *
   * With `weights: "uniform"` the probability of a class is the fraction of the k
   * neighbors that carry it; with `weights: "distance"` neighbors are weighted by
   * inverse distance. Columns follow the sorted unique training labels.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Class probability matrix of shape (n_samples, n_classes), float64
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predictProba(X: Tensor): Tensor {
    const proba = this.votes(X, "KNeighborsClassifier");
    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classes_.length;
    for (let i = 0; i < nSamples; i++) {
      let total = 0;
      for (let c = 0; c < nClasses; c++) total += proba[i * nClasses + c] as number;
      for (let c = 0; c < nClasses; c++) {
        proba[i * nClasses + c] = (proba[i * nClasses + c] as number) / total;
      }
    }
    return tensor(proba).reshape([nSamples, nClasses]);
  }

  /**
   * Return the mean accuracy on the given test data and labels.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True labels of shape (n_samples,)
   * @returns Accuracy score in range [0, 1]
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional or sample counts mismatch
   * @throws {DataValidationError} If y is empty or contains NaN/Inf values
   */
  score(X: Tensor, y: Tensor): number {
    assertScoreTarget(y);
    const yPred = this.predict(X);
    assertSameSampleCount(yPred.size, y.size);

    let correct = 0;
    for (let i = 0; i < y.size; i++) {
      if (Number(y.data[y.offset + i]) === Number(yPred.data[yPred.offset + i])) {
        correct++;
      }
    }

    return correct / y.size;
  }

  /**
   * Sorted unique class labels seen during fit; columns of `predictProba` follow this order.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get classes(): Tensor {
    if (!this.fitted)
      throw new NotFittedError("KNeighborsClassifier must be fitted to access classes");
    return labelTensor(this.classes_, this.classes_);
  }

  /**
   * Create an unfitted copy with the same hyperparameters.
   *
   * @returns A new KNeighborsClassifier
   */
  clone(): KNeighborsClassifier {
    return new KNeighborsClassifier({
      nNeighbors: this.nNeighbors,
      weights: this.weights,
      metric: this.metric,
    });
  }

  /**
   * Weighted vote totals of the k nearest neighbors, flat `(n_samples, n_classes)`.
   */
  private votes(X: Tensor, name: string): Float64Array {
    if (!this.fitted || !this.xTrain_ || !this.yIndex_) {
      throw new NotFittedError(`${name} must be fitted before prediction`);
    }
    validatePredictInputs(X, this.nFeaturesIn_, name);
    this.assertEnoughTrainingSamples();

    const nSamples = X.shape[0] ?? 0;
    const nF = this.nFeaturesIn_;
    const nClasses = this.classes_.length;
    const rows = toFloat64View(X);
    const totals = new Float64Array(nSamples * nClasses);

    for (let i = 0; i < nSamples; i++) {
      const neighbors = this.findNearest(rows, i * nF, this.nNeighbors);
      const w =
        this.weights === "uniform"
          ? null
          : inverseDistanceWeights(neighbors.map((n) => n.distance));
      for (let j = 0; j < neighbors.length; j++) {
        const cls = this.yIndex_[(neighbors[j] as { index: number }).index] as number;
        totals[i * nClasses + cls] =
          (totals[i * nClasses + cls] as number) + (w ? (w[j] as number) : 1);
      }
    }
    return totals;
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
 * @see {@link https://deepbox.dev/docs/ml-neighbors | Deepbox Nearest Neighbors}
 */
export class KNeighborsRegressor extends KNeighborsBase implements Regressor {
  private yTrain_?: Float64Array;

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
    if (this.nNeighbors > nSamples) {
      throw new InvalidParameterError(
        `nNeighbors must be <= n_samples; received ${this.nNeighbors} > ${nSamples}`,
        "nNeighbors",
        this.nNeighbors
      );
    }

    this.storeTrainingData(X);
    this.yTrain_ = Float64Array.from(toFloat64View(y));
    this.fitted = true;

    return this;
  }

  /**
   * Predict target values for samples in X.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted values of shape (n_samples,), float64
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.xTrain_ || !this.yTrain_) {
      throw new NotFittedError("KNeighborsRegressor must be fitted before prediction");
    }

    validatePredictInputs(X, this.nFeaturesIn_, "KNeighborsRegressor");
    this.assertEnoughTrainingSamples();

    const nSamples = X.shape[0] ?? 0;
    const nF = this.nFeaturesIn_;
    const rows = toFloat64View(X);
    const predictions = new Float64Array(nSamples);

    for (let i = 0; i < nSamples; i++) {
      const neighbors = this.findNearest(rows, i * nF, this.nNeighbors);
      const w =
        this.weights === "uniform"
          ? null
          : inverseDistanceWeights(neighbors.map((n) => n.distance));

      let sumValues = 0;
      let sumWeights = 0;
      for (let j = 0; j < neighbors.length; j++) {
        const weight = w ? (w[j] as number) : 1;
        sumValues += (this.yTrain_[(neighbors[j] as { index: number }).index] as number) * weight;
        sumWeights += weight;
      }

      predictions[i] = sumValues / sumWeights;
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
   * @throws {DataValidationError} If y is empty or contains NaN/Inf values
   */
  score(X: Tensor, y: Tensor): number {
    assertScoreTarget(y);
    const yPred = this.predict(X);
    assertSameSampleCount(yPred.size, y.size);

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

  /**
   * Create an unfitted copy with the same hyperparameters.
   *
   * @returns A new KNeighborsRegressor
   */
  clone(): KNeighborsRegressor {
    return new KNeighborsRegressor({
      nNeighbors: this.nNeighbors,
      weights: this.weights,
      metric: this.metric,
    });
  }
}

/**
 * Unsupervised nearest-neighbor search.
 *
 * Stores the training samples and answers k-nearest and fixed-radius queries
 * with the chosen metric. Distances and indices are exact (brute force).
 *
 * @example
 * ```ts
 * import { NearestNeighbors } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0], [1, 0], [0, 2], [5, 5]]);
 * const nn = new NearestNeighbors({ nNeighbors: 2 }).fit(X);
 * const { distances, indices } = nn.kneighbors(tensor([[0.2, 0]]));
 * // indices -> [[0, 1]]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-neighbors | Deepbox Nearest Neighbors}
 */
export class NearestNeighbors extends KNeighborsBase {
  private radius: number;

  /**
   * @param options.nNeighbors - Default number of neighbors for `kneighbors` (default: 5)
   * @param options.radius - Default radius for `radiusNeighbors` (default: 1)
   * @param options.metric - "euclidean" (default) or "manhattan"
   */
  constructor(
    options: {
      readonly nNeighbors?: number;
      readonly radius?: number;
      readonly metric?: NeighborMetric;
    } = {}
  ) {
    const baseOptions: {
      readonly nNeighbors?: number;
      readonly metric?: NeighborMetric;
    } = {
      ...(options.nNeighbors !== undefined ? { nNeighbors: options.nNeighbors } : {}),
      ...(options.metric !== undefined ? { metric: options.metric } : {}),
    };
    super(baseOptions);
    this.radius = options.radius ?? 1;

    if (!Number.isFinite(this.radius) || this.radius <= 0) {
      throw new InvalidParameterError("radius must be a finite number > 0", "radius", this.radius);
    }
  }

  /**
   * Store the samples that later queries are answered against.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (present for API compatibility)
   * @returns this
   * @throws {ShapeError} If X is not 2D
   * @throws {DataValidationError} If X is empty or contains NaN/Inf values
   */
  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    this.storeTrainingData(X);
    this.fitted = true;
    return this;
  }

  /**
   * Find the k nearest training samples of each query row.
   *
   * When `X` is omitted the training samples themselves are queried and every
   * sample is excluded from its own neighbor list.
   *
   * @param X - Query samples of shape (n_queries, n_features); defaults to the training data
   * @param nNeighbors - Number of neighbors; defaults to the `nNeighbors` option
   * @returns `distances` (float64) and `indices` (int32), both of shape (n_queries, nNeighbors),
   *   nearest first
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {InvalidParameterError} If nNeighbors is invalid or larger than the number of
   *   available training samples
   * @throws {ShapeError} If X has the wrong dimensions or feature count
   */
  kneighbors(X?: Tensor, nNeighbors?: number): { distances: Tensor; indices: Tensor } {
    const train = this.xTrain_;
    if (!this.fitted || !train) {
      throw new NotFittedError("NearestNeighbors must be fitted before querying neighbors");
    }

    const k = nNeighbors ?? this.nNeighbors;
    if (!Number.isInteger(k) || k < 1) {
      throw new InvalidParameterError("nNeighbors must be an integer >= 1", "nNeighbors", k);
    }

    const queryingTrainingData = X === undefined;
    let query: Float64Array;
    let nQueries: number;
    if (queryingTrainingData) {
      query = train;
      nQueries = this.nTrain_;
    } else {
      validatePredictInputs(X, this.nFeaturesIn_, "NearestNeighbors");
      query = toFloat64View(X);
      nQueries = X.shape[0] ?? 0;
    }

    const maxNeighbors = queryingTrainingData ? this.nTrain_ - 1 : this.nTrain_;
    if (k > maxNeighbors) {
      throw new InvalidParameterError(
        `nNeighbors must be <= ${maxNeighbors} for this query; received ${k}`,
        "nNeighbors",
        k
      );
    }

    const nF = this.nFeaturesIn_;
    const distances = new Float64Array(nQueries * k);
    const indices = new Int32Array(nQueries * k);
    for (let i = 0; i < nQueries; i++) {
      const neighbors = this.findNearest(query, i * nF, k, queryingTrainingData ? i : -1);
      for (let j = 0; j < k; j++) {
        const neighbor = neighbors[j] as { index: number; distance: number };
        distances[i * k + j] = neighbor.distance;
        indices[i * k + j] = neighbor.index;
      }
    }

    return {
      distances: tensor(distances).reshape([nQueries, k]),
      indices: tensor(indices).reshape([nQueries, k]),
    };
  }

  /**
   * Find all training samples within a radius of each query row.
   *
   * When `X` is omitted the training samples themselves are queried and every
   * sample is excluded from its own neighbor list.
   *
   * @param X - Query samples of shape (n_queries, n_features); defaults to the training data
   * @param radius - Search radius; defaults to the `radius` option
   * @returns For every query, the neighbor distances and training indices, nearest first
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {InvalidParameterError} If radius is not a finite number > 0
   * @throws {ShapeError} If X has the wrong dimensions or feature count
   */
  radiusNeighbors(X?: Tensor, radius?: number): { distances: number[][]; indices: number[][] } {
    const train = this.xTrain_;
    if (!this.fitted || !train) {
      throw new NotFittedError("NearestNeighbors must be fitted before querying neighbors");
    }

    const r = radius ?? this.radius;
    if (typeof r !== "number" || !Number.isFinite(r) || r <= 0) {
      throw new InvalidParameterError("radius must be a finite number > 0", "radius", r);
    }

    const queryingTrainingData = X === undefined;
    let query: Float64Array;
    let nQueries: number;
    if (queryingTrainingData) {
      query = train;
      nQueries = this.nTrain_;
    } else {
      validatePredictInputs(X, this.nFeaturesIn_, "NearestNeighbors");
      query = toFloat64View(X);
      nQueries = X.shape[0] ?? 0;
    }

    const nF = this.nFeaturesIn_;
    const distanceRows: number[][] = [];
    const indexRows: number[][] = [];
    for (let i = 0; i < nQueries; i++) {
      const neighbors = this.findWithinRadius(query, i * nF, r, queryingTrainingData ? i : -1);
      distanceRows.push(neighbors.map((neighbor) => neighbor.distance));
      indexRows.push(neighbors.map((neighbor) => neighbor.index));
    }

    return { distances: distanceRows, indices: indexRows };
  }

  override getParams(): Record<string, unknown> {
    return {
      nNeighbors: this.nNeighbors,
      radius: this.radius,
      metric: this.metric,
    };
  }

  override setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nNeighbors":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError(
              "nNeighbors must be an integer >= 1",
              "nNeighbors",
              value
            );
          }
          this.nNeighbors = value;
          break;
        case "radius":
          if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
            throw new InvalidParameterError("radius must be a finite number > 0", "radius", value);
          }
          this.radius = value;
          break;
        case "metric":
          if (value !== "euclidean" && value !== "manhattan") {
            throw new InvalidParameterError(
              `metric must be "euclidean" or "manhattan"; got ${String(value)}`,
              "metric",
              value
            );
          }
          this.metric = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  /**
   * Estimator tags. The model has no `predict`, `transform` or `scoreSamples`, so tag
   * inference would report a classifier that needs `y`; it is unsupervised.
   *
   * @internal
   */
  _getTags(): Partial<EstimatorTags> {
    return { estimatorType: "transformer", requiresY: false };
  }

  /**
   * Create an unfitted copy with the same hyperparameters.
   *
   * @returns A new NearestNeighbors
   */
  clone(): NearestNeighbors {
    return new NearestNeighbors({
      nNeighbors: this.nNeighbors,
      radius: this.radius,
      metric: this.metric,
    });
  }
}
