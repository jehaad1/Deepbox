/**
 * Radius-based Neighbors Classifier and Regressor.
 *
 * Classifies/predicts based on all training samples within a given radius.
 *
 * @module ml/neighbors/RadiusNeighbors
 * @see {@link https://deepbox.dev/docs/ml-neighbors | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, warn } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { toFloat64View, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier, Regressor } from "../base";
import {
  assertSameSampleCount,
  assertScoreTarget,
  inverseDistanceWeights,
  labelTensor,
} from "./index";

type RadiusWeights = "uniform" | "distance";
type OutlierLabel = number | "most_frequent" | null;

function checkRadius(value: unknown): number {
  if (typeof value !== "number" || Number.isNaN(value) || value <= 0) {
    throw new InvalidParameterError("radius must be > 0", "radius", value);
  }
  return value;
}

function checkWeights(value: unknown): RadiusWeights {
  if (value !== "uniform" && value !== "distance") {
    throw new InvalidParameterError(
      `weights must be "uniform" or "distance"; received ${String(value)}`,
      "weights",
      value
    );
  }
  return value;
}

function checkOutlierLabel(value: unknown): OutlierLabel {
  if (value === null || value === "most_frequent") return value;
  if (typeof value === "number" && Number.isFinite(value)) return value;
  throw new InvalidParameterError(
    `outlierLabel must be a finite number, "most_frequent" or null; received ${String(value)}`,
    "outlierLabel",
    value
  );
}

/** Training samples within `radius` of one query row, with Euclidean distances. */
function neighborsWithin(
  train: Float64Array,
  nTrain: number,
  nF: number,
  query: Float64Array,
  offset: number,
  radius: number
): { index: number[]; distance: number[] } {
  const r2 = radius * radius;
  const index: number[] = [];
  const distance: number[] = [];
  for (let j = 0; j < nTrain; j++) {
    const base = j * nF;
    let d = 0;
    for (let f = 0; f < nF; f++) {
      const diff = (query[offset + f] as number) - (train[base + f] as number);
      d += diff * diff;
      if (d > r2) break;
    }
    if (d <= r2) {
      index.push(j);
      distance.push(Math.sqrt(d));
    }
  }
  return { index, distance };
}

function describeSamples(rows: readonly number[]): string {
  const shown = rows.slice(0, 5).join(", ");
  return rows.length > 5 ? `${shown}, ... (${rows.length} in total)` : shown;
}

/**
 * Classifier that votes among all training samples within a fixed radius.
 *
 * Unlike k-nearest neighbors, the number of voters varies per sample. A sample with no
 * training point inside the radius is an outlier: by default this is an error, like in
 * scikit-learn; set `outlierLabel` to assign a label to such samples instead. When classes
 * tie, the smallest label wins.
 *
 * @example
 * ```ts
 * import { RadiusNeighborsClassifier } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0], [0.1, 0], [5, 5], [5.1, 5]]);
 * const y = tensor([0, 0, 1, 1]);
 * const clf = new RadiusNeighborsClassifier({ radius: 1, outlierLabel: -1 }).fit(X, y);
 * clf.predict(tensor([[0.2, 0], [20, 20]])); // [0, -1]
 * ```
 */
export class RadiusNeighborsClassifier implements Classifier {
  private radius: number;
  private weights: RadiusWeights;
  private outlierLabel: OutlierLabel;

  private xTrain_?: Float64Array;
  private yIndex_?: Int32Array;
  private classes_: number[] = [];
  private nTrainSamples_ = 0;
  private nFeaturesIn_ = 0;
  private fitted = false;

  /**
   * @param options.radius - Neighborhood radius, > 0 (default: 1)
   * @param options.weights - "uniform" (default) or "distance" (inverse-distance votes)
   * @param options.outlierLabel - Label for samples without neighbors: a number,
   *   "most_frequent" (the most common training label) or `null` (default, throw an error).
   *   A number that is not a training class gives all-zero rows in `predictProba`.
   */
  constructor(
    options: {
      readonly radius?: number;
      readonly weights?: RadiusWeights;
      readonly outlierLabel?: OutlierLabel;
    } = {}
  ) {
    this.radius = checkRadius(options.radius ?? 1.0);
    this.weights = checkWeights(options.weights ?? "uniform");
    this.outlierLabel = checkOutlierLabel(options.outlierLabel ?? null);
  }

  /**
   * Store the training data.
   *
   * @param X - Training samples of shape (n_samples, n_features)
   * @param y - Class labels of shape (n_samples,)
   * @returns this
   * @throws {ShapeError} If X is not 2D, y is not 1D or their sample counts differ
   * @throws {DataValidationError} If X or y contain NaN/Inf
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const labels = toFloat64View(y);
    const classes = [...new Set(labels)].sort((a, b) => a - b);
    const classIndex = new Map<number, number>();
    classes.forEach((c, i) => {
      classIndex.set(c, i);
    });
    const yIndex = new Int32Array(n);
    for (let i = 0; i < n; i++) yIndex[i] = classIndex.get(labels[i] as number) as number;

    this.xTrain_ = Float64Array.from(toFloat64View(X));
    this.yIndex_ = yIndex;
    this.classes_ = classes;
    this.nTrainSamples_ = n;
    this.nFeaturesIn_ = X.shape[1] ?? 0;
    this.fitted = true;
    return this;
  }

  /**
   * Predict the class of each sample by (weighted) vote of the training samples in its radius.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted labels of shape (n_samples,); int32 when all labels are integers,
   *   float64 otherwise
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong dimensions or feature count
   * @throws {DataValidationError} If a sample has no neighbor in the radius and `outlierLabel`
   *   is `null`
   */
  predict(X: Tensor): Tensor {
    const votes = this.votes(X, "predict");
    const nTest = X.shape[0] ?? 0;
    const nClasses = this.classes_.length;
    const labels = new Float64Array(nTest);
    let fallback: number | undefined;

    for (let i = 0; i < nTest; i++) {
      const row = votes.subarray(i * nClasses, (i + 1) * nClasses);
      if (row.every((v) => v === 0)) {
        fallback ??= this.outlierValue();
        labels[i] = fallback;
        continue;
      }
      // Strict comparison over ascending classes: the smallest label wins ties.
      let best = 0;
      for (let c = 1; c < nClasses; c++) {
        if ((row[c] as number) > (row[best] as number)) best = c;
      }
      labels[i] = this.classes_[best] as number;
    }
    return labelTensor(
      labels,
      fallback === undefined ? this.classes_ : [...this.classes_, fallback]
    );
  }

  /**
   * Estimate class probabilities as the (weighted) fraction of in-radius neighbors per class.
   *
   * Rows of outliers follow `outlierLabel`: a one-hot row for that class, or all zeros when the
   * label is not a training class.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Probabilities of shape (n_samples, n_classes), float64; columns follow `classes`
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong dimensions or feature count
   * @throws {DataValidationError} If a sample has no neighbor in the radius and `outlierLabel`
   *   is `null`
   */
  predictProba(X: Tensor): Tensor {
    const votes = this.votes(X, "predictProba");
    const nTest = X.shape[0] ?? 0;
    const nClasses = this.classes_.length;
    let fallbackClass: number | undefined;

    for (let i = 0; i < nTest; i++) {
      let total = 0;
      for (let c = 0; c < nClasses; c++) total += votes[i * nClasses + c] as number;
      if (total > 0) {
        for (let c = 0; c < nClasses; c++) {
          votes[i * nClasses + c] = (votes[i * nClasses + c] as number) / total;
        }
      } else {
        fallbackClass ??= this.classes_.indexOf(this.outlierValue());
        if (fallbackClass >= 0) votes[i * nClasses + fallbackClass] = 1;
      }
    }
    return tensor(votes).reshape([nTest, nClasses]);
  }

  /**
   * Return the mean accuracy on the given test data and labels.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True labels of shape (n_samples,)
   * @returns Accuracy in [0, 1]
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-D or its length differs from the number of samples
   */
  score(X: Tensor, y: Tensor): number {
    assertScoreTarget(y);
    const pred = this.predict(X);
    assertSameSampleCount(pred.size, y.size);
    let correct = 0;
    for (let i = 0; i < y.size; i++) {
      if (Number(pred.data[pred.offset + i]) === Number(y.data[y.offset + i])) correct++;
    }
    return correct / y.size;
  }

  /**
   * Sorted unique class labels seen during fit.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get classes(): Tensor {
    if (!this.fitted) throw new NotFittedError("RadiusNeighborsClassifier must be fitted");
    return labelTensor(this.classes_, this.classes_);
  }

  getParams(): Record<string, unknown> {
    return { radius: this.radius, weights: this.weights, outlierLabel: this.outlierLabel };
  }

  /**
   * Set hyperparameters.
   *
   * @throws {InvalidParameterError} If a value is invalid or a key is unknown
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "radius":
          this.radius = checkRadius(value);
          break;
        case "weights":
          this.weights = checkWeights(value);
          break;
        case "outlierLabel":
          this.outlierLabel = checkOutlierLabel(value);
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  /**
   * Create an unfitted copy with the same hyperparameters.
   *
   * @returns A new RadiusNeighborsClassifier
   */
  clone(): RadiusNeighborsClassifier {
    return new RadiusNeighborsClassifier({
      radius: this.radius,
      weights: this.weights,
      outlierLabel: this.outlierLabel,
    });
  }

  /** Label assigned to samples without neighbors. */
  private outlierValue(): number {
    if (this.outlierLabel === null) {
      // Callers collect the offending rows before throwing; see votes().
      throw new DataValidationError("no neighbors within the radius");
    }
    if (this.outlierLabel === "most_frequent") {
      const counts = new Float64Array(this.classes_.length);
      for (const c of this.yIndex_ as Int32Array) counts[c] = (counts[c] as number) + 1;
      let best = 0;
      for (let c = 1; c < counts.length; c++) {
        if ((counts[c] as number) > (counts[best] as number)) best = c;
      }
      return this.classes_[best] as number;
    }
    return this.outlierLabel;
  }

  /** Weighted vote totals per class, flat `(n_samples, n_classes)`. */
  private votes(X: Tensor, method: string): Float64Array {
    if (!this.fitted || !this.xTrain_ || !this.yIndex_) {
      throw new NotFittedError(`RadiusNeighborsClassifier must be fitted before ${method}`);
    }
    validatePredictInputs(X, this.nFeaturesIn_, "RadiusNeighborsClassifier");
    const nTest = X.shape[0] ?? 0;
    const nF = this.nFeaturesIn_;
    const nClasses = this.classes_.length;
    const rows = toFloat64View(X);
    const totals = new Float64Array(nTest * nClasses);
    const outliers: number[] = [];

    for (let i = 0; i < nTest; i++) {
      const hits = neighborsWithin(
        this.xTrain_,
        this.nTrainSamples_,
        nF,
        rows,
        i * nF,
        this.radius
      );
      if (hits.index.length === 0) {
        outliers.push(i);
        continue;
      }
      const w = this.weights === "uniform" ? null : inverseDistanceWeights(hits.distance);
      for (let j = 0; j < hits.index.length; j++) {
        const cls = this.yIndex_[hits.index[j] as number] as number;
        totals[i * nClasses + cls] =
          (totals[i * nClasses + cls] as number) + (w ? (w[j] as number) : 1);
      }
    }

    if (outliers.length > 0 && this.outlierLabel === null) {
      throw new DataValidationError(
        `No neighbors found within radius ${this.radius} for test samples [${describeSamples(outliers)}]; ` +
          "use a larger radius, set outlierLabel to give such samples a label, or remove them from the data"
      );
    }
    return totals;
  }
}

/**
 * Regressor that averages the targets of all training samples within a fixed radius.
 *
 * A sample with no training point inside the radius gets the prediction NaN and a warning,
 * like in scikit-learn.
 *
 * @example
 * ```ts
 * import { RadiusNeighborsRegressor } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1], [2], [3], [4]]);
 * const y = tensor([2, 4, 6, 8]);
 * const reg = new RadiusNeighborsRegressor({ radius: 1.1 }).fit(X, y);
 * reg.predict(tensor([[2.5]])); // [5]
 * ```
 */
export class RadiusNeighborsRegressor implements Regressor {
  private radius: number;
  private weights: RadiusWeights;

  private xTrain_?: Float64Array;
  private yTrain_?: Float64Array;
  private nTrainSamples_ = 0;
  private nFeaturesIn_ = 0;
  private fitted = false;

  /**
   * @param options.radius - Neighborhood radius, > 0 (default: 1)
   * @param options.weights - "uniform" (default) or "distance" (inverse-distance weighted mean)
   */
  constructor(options: { readonly radius?: number; readonly weights?: RadiusWeights } = {}) {
    this.radius = checkRadius(options.radius ?? 1.0);
    this.weights = checkWeights(options.weights ?? "uniform");
  }

  /**
   * Store the training data.
   *
   * @param X - Training samples of shape (n_samples, n_features)
   * @param y - Targets of shape (n_samples,)
   * @returns this
   * @throws {ShapeError} If X is not 2D, y is not 1D or their sample counts differ
   * @throws {DataValidationError} If X or y contain NaN/Inf
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    this.xTrain_ = Float64Array.from(toFloat64View(X));
    this.yTrain_ = Float64Array.from(toFloat64View(y));
    this.nTrainSamples_ = X.shape[0] ?? 0;
    this.nFeaturesIn_ = X.shape[1] ?? 0;
    this.fitted = true;
    return this;
  }

  /**
   * Predict the (weighted) mean target of the training samples within the radius.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predictions of shape (n_samples,), float64; NaN for samples without neighbors
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong dimensions or feature count
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.xTrain_ || !this.yTrain_) {
      throw new NotFittedError("RadiusNeighborsRegressor must be fitted before predict");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "RadiusNeighborsRegressor");
    const nTest = X.shape[0] ?? 0;
    const nF = this.nFeaturesIn_;
    const rows = toFloat64View(X);
    const result = new Float64Array(nTest);
    let nOutliers = 0;

    for (let i = 0; i < nTest; i++) {
      const hits = neighborsWithin(
        this.xTrain_,
        this.nTrainSamples_,
        nF,
        rows,
        i * nF,
        this.radius
      );
      if (hits.index.length === 0) {
        result[i] = Number.NaN;
        nOutliers++;
        continue;
      }
      const w = this.weights === "uniform" ? null : inverseDistanceWeights(hits.distance);
      let sum = 0;
      let total = 0;
      for (let j = 0; j < hits.index.length; j++) {
        const weight = w ? (w[j] as number) : 1;
        sum += (this.yTrain_[hits.index[j] as number] as number) * weight;
        total += weight;
      }
      result[i] = sum / total;
    }

    if (nOutliers > 0) {
      warn(
        `${nOutliers} sample(s) have no neighbors within radius ${this.radius}; predicting NaN for them`,
        "UserWarning",
        "RadiusNeighborsRegressor.predict"
      );
    }
    return tensor(result);
  }

  /**
   * Return the R² score on the given test data and targets.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True targets of shape (n_samples,)
   * @returns R² (1 is perfect, can be negative); NaN if some sample has no neighbors
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-D or its length differs from the number of samples
   */
  score(X: Tensor, y: Tensor): number {
    assertScoreTarget(y);
    const pred = toFloat64View(this.predict(X));
    assertSameSampleCount(pred.length, y.size);
    const truth = toFloat64View(y);
    const n = truth.length;
    let ssRes = 0;
    let ssTot = 0;
    let yMean = 0;
    for (let i = 0; i < n; i++) yMean += truth[i] as number;
    yMean /= n;
    for (let i = 0; i < n; i++) {
      ssRes += ((truth[i] as number) - (pred[i] as number)) ** 2;
      ssTot += ((truth[i] as number) - yMean) ** 2;
    }
    if (ssTot === 0) return ssRes === 0 ? 1 : 0;
    return 1 - ssRes / ssTot;
  }

  getParams(): Record<string, unknown> {
    return { radius: this.radius, weights: this.weights };
  }

  /**
   * Set hyperparameters.
   *
   * @throws {InvalidParameterError} If a value is invalid or a key is unknown
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "radius":
          this.radius = checkRadius(value);
          break;
        case "weights":
          this.weights = checkWeights(value);
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  /**
   * Create an unfitted copy with the same hyperparameters.
   *
   * @returns A new RadiusNeighborsRegressor
   */
  clone(): RadiusNeighborsRegressor {
    return new RadiusNeighborsRegressor({ radius: this.radius, weights: this.weights });
  }
}
