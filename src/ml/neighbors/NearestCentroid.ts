/**
 * Nearest Centroid classifier.
 *
 * Classifies samples based on the nearest class centroid (mean or median of the
 * training samples of each class).
 *
 * @module ml/neighbors/NearestCentroid
 * @see {@link https://deepbox.dev/docs/ml-neighbors | Deepbox documentation}
 */

import {
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
  warn,
} from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { toFloat64View, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier } from "../base";

type CentroidMetric = "euclidean" | "manhattan";
type CentroidPriors = "uniform" | "empirical" | readonly number[];

/** Median of a non-empty list (mean of the two middle values for an even count). */
function median(values: number[]): number {
  const sorted = Float64Array.from(values).sort();
  const n = sorted.length;
  const mid = n >> 1;
  return n % 2 === 1
    ? (sorted[mid] as number)
    : ((sorted[mid - 1] as number) + (sorted[mid] as number)) / 2;
}

/**
 * Nearest Centroid classifier.
 *
 * Each class is represented by its centroid and a sample is assigned to the class
 * with the nearest centroid. With `metric: "euclidean"` the centroid is the
 * per-feature mean; with `metric: "manhattan"` it is the per-feature median.
 *
 * Optionally the centroids can be shrunk towards the overall centroid
 * (`shrinkThreshold`, the "nearest shrunken centroid" of Tibshirani et al., 2002),
 * which removes features that do not separate the classes.
 *
 * For `predictProba` and `decisionFunction` (Euclidean metric only) the features are
 * scaled by the pooled within-class standard deviation and the scores are
 * `-||x - centroid||^2 + 2 log(prior)`, like scikit-learn.
 *
 * @example
 * ```ts
 * import { NearestCentroid } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]);
 * const y = tensor([1, 1, 1, 2, 2, 2]);
 * const clf = new NearestCentroid().fit(X, y);
 * clf.predict(tensor([[-0.8, -1]])); // [1]
 * ```
 *
 * References:
 * - Tibshirani, R., Hastie, T., Narasimhan, B., Chu, G. (2002). Diagnosis of multiple cancer
 *   types by shrunken centroids of gene expression. PNAS 99(10).
 */
export class NearestCentroid implements Classifier {
  private metric: CentroidMetric;
  private shrinkThreshold: number | undefined;
  private priors: CentroidPriors;

  private centroids_?: Float64Array;
  private classes_: number[] = [];
  private classPrior_: number[] = [];
  private withinClassStd_?: Float64Array;
  private nFeaturesIn_ = 0;
  private fitted = false;

  /**
   * @param options.metric - "euclidean" (mean centroids, default) or "manhattan" (median centroids)
   * @param options.shrinkThreshold - Threshold (> 0) for shrinking centroids towards the
   *   overall centroid; omitted by default (no shrinking)
   * @param options.priors - "uniform" (default), "empirical" (class frequencies in the training
   *   data) or one non-negative prior per class. Priors only affect `predict`,
   *   `predictProba` and `decisionFunction` when they are not uniform.
   */
  constructor(
    options: {
      readonly metric?: CentroidMetric;
      readonly shrinkThreshold?: number;
      readonly priors?: CentroidPriors;
    } = {}
  ) {
    this.metric = NearestCentroid.checkMetric(options.metric ?? "euclidean");
    this.shrinkThreshold = NearestCentroid.checkShrink(options.shrinkThreshold);
    this.priors = NearestCentroid.checkPriors(options.priors ?? "uniform");
  }

  private static checkMetric(value: unknown): CentroidMetric {
    if (value !== "euclidean" && value !== "manhattan") {
      throw new InvalidParameterError(
        `metric must be "euclidean" or "manhattan"; received ${String(value)}`,
        "metric",
        value
      );
    }
    return value;
  }

  private static checkShrink(value: unknown): number | undefined {
    if (value === undefined) return undefined;
    if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
      throw new InvalidParameterError(
        `shrinkThreshold must be a finite number > 0; received ${String(value)}`,
        "shrinkThreshold",
        value
      );
    }
    return value;
  }

  private static checkPriors(value: unknown): CentroidPriors {
    if (value === "uniform" || value === "empirical") return value;
    if (
      Array.isArray(value) &&
      value.length > 0 &&
      value.every((p) => typeof p === "number" && Number.isFinite(p) && p >= 0)
    ) {
      return [...(value as number[])];
    }
    throw new InvalidParameterError(
      'priors must be "uniform", "empirical" or an array of non-negative numbers',
      "priors",
      value
    );
  }

  /**
   * Compute the class centroids from the training data.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Class labels of shape (n_samples,)
   * @returns this
   * @throws {ShapeError} If X is not 2D, y is not 1D or their sample counts differ
   * @throws {DataValidationError} If X or y contain NaN/Inf
   * @throws {InvalidParameterError} If `priors` is an array whose length differs from the number
   *   of classes or whose sum is zero
   * @throws {DataValidationError} If `shrinkThreshold` is set and there are not more samples
   *   than classes
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const nF = X.shape[1] ?? 0;
    const x = toFloat64View(X);
    const labels = toFloat64View(y);

    const classes = [...new Set(labels)].sort((a, b) => a - b);
    const nClasses = classes.length;
    const classIndex = new Map<number, number>();
    classes.forEach((c, i) => {
      classIndex.set(c, i);
    });
    const rowClass = new Int32Array(n);
    const counts = new Float64Array(nClasses);
    for (let i = 0; i < n; i++) {
      const c = classIndex.get(labels[i] as number) as number;
      rowClass[i] = c;
      counts[c] = (counts[c] as number) + 1;
    }

    const priors = this.resolvePriors(counts, n);

    // Class centroids: mean, or per-feature median for the Manhattan metric.
    const centroids = new Float64Array(nClasses * nF);
    if (this.metric === "euclidean") {
      for (let i = 0; i < n; i++) {
        const c = rowClass[i] as number;
        for (let f = 0; f < nF; f++) {
          centroids[c * nF + f] = (centroids[c * nF + f] as number) + (x[i * nF + f] as number);
        }
      }
      for (let c = 0; c < nClasses; c++) {
        for (let f = 0; f < nF; f++) {
          centroids[c * nF + f] = (centroids[c * nF + f] as number) / (counts[c] as number);
        }
      }
    } else {
      for (let c = 0; c < nClasses; c++) {
        for (let f = 0; f < nF; f++) {
          const column: number[] = [];
          for (let i = 0; i < n; i++) {
            if (rowClass[i] === c) column.push(x[i * nF + f] as number);
          }
          centroids[c * nF + f] = median(column);
        }
      }
    }

    // Pooled within-class standard deviation, measured from the unshrunk centroids.
    // It needs n > nClasses degrees of freedom; otherwise it is left at zero, which
    // disables the feature scaling in decisionFunction.
    const withinStd = new Float64Array(nF);
    if (n > nClasses) {
      for (let i = 0; i < n; i++) {
        const c = rowClass[i] as number;
        for (let f = 0; f < nF; f++) {
          const diff = (x[i * nF + f] as number) - (centroids[c * nF + f] as number);
          withinStd[f] = (withinStd[f] as number) + diff * diff;
        }
      }
      for (let f = 0; f < nF; f++) {
        withinStd[f] = Math.sqrt((withinStd[f] as number) / (n - nClasses));
      }
    }

    if (this.shrinkThreshold !== undefined && nClasses > 1) {
      if (n <= nClasses) {
        throw new DataValidationError(
          "shrinkThreshold needs more samples than classes to estimate the within-class spread"
        );
      }
      this.shrinkCentroids(centroids, x, n, nF, nClasses, counts, withinStd, this.shrinkThreshold);
    }

    this.nFeaturesIn_ = nF;
    this.classes_ = classes;
    this.classPrior_ = priors;
    this.centroids_ = centroids;
    this.withinClassStd_ = withinStd;
    this.fitted = true;
    return this;
  }

  /**
   * Predict the class of each sample.
   *
   * With uniform priors this is the class whose centroid is nearest in the chosen metric
   * (ties go to the smaller label). Otherwise the class with the largest
   * {@link decisionFunction} score is returned.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted class labels of shape (n_samples,), float64
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong dimensions or feature count
   */
  predict(X: Tensor): Tensor {
    this.assertFitted("predict");
    validatePredictInputs(X, this.nFeaturesIn_, "NearestCentroid");
    const nTest = X.shape[0] ?? 0;
    const nClasses = this.classes_.length;
    const labels = new Float64Array(nTest);

    if (this.hasUniformPriors()) {
      const x = toFloat64View(X);
      const nF = this.nFeaturesIn_;
      const centroids = this.centroids_ as Float64Array;
      const euclidean = this.metric === "euclidean";
      for (let i = 0; i < nTest; i++) {
        let bestC = 0;
        let bestD = Number.POSITIVE_INFINITY;
        for (let c = 0; c < nClasses; c++) {
          let d = 0;
          for (let f = 0; f < nF; f++) {
            const diff = (x[i * nF + f] as number) - (centroids[c * nF + f] as number);
            d += euclidean ? diff * diff : Math.abs(diff);
          }
          if (d < bestD) {
            bestD = d;
            bestC = c;
          }
        }
        labels[i] = this.classes_[bestC] as number;
      }
    } else {
      const scores = this.scores(X);
      for (let i = 0; i < nTest; i++) {
        let bestC = 0;
        for (let c = 1; c < nClasses; c++) {
          if ((scores[i * nClasses + c] as number) > (scores[i * nClasses + bestC] as number)) {
            bestC = c;
          }
        }
        labels[i] = this.classes_[bestC] as number;
      }
    }
    return tensor(labels);
  }

  /**
   * Estimate class probabilities as the softmax of the discriminant scores
   * (`-||x - centroid||^2 + 2 log(prior)` after scaling the features by the pooled
   * within-class standard deviation), as scikit-learn does.
   *
   * Only available with `metric: "euclidean"`. Note that with uniform priors
   * `predict` uses the unscaled distance, so its result can differ from the most
   * probable class here when the features have very different within-class spreads.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Probabilities of shape (n_samples, n_classes), float64; columns follow `classes`
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {InvalidParameterError} If the metric is not "euclidean"
   * @throws {ShapeError} If X has the wrong dimensions or feature count
   */
  predictProba(X: Tensor): Tensor {
    this.assertFitted("predictProba");
    this.assertEuclidean("predictProba");
    validatePredictInputs(X, this.nFeaturesIn_, "NearestCentroid");
    const nTest = X.shape[0] ?? 0;
    const nClasses = this.classes_.length;
    const scores = this.scores(X);

    for (let i = 0; i < nTest; i++) {
      let max = Number.NEGATIVE_INFINITY;
      for (let c = 0; c < nClasses; c++) max = Math.max(max, scores[i * nClasses + c] as number);
      let total = 0;
      for (let c = 0; c < nClasses; c++) {
        const e = Math.exp((scores[i * nClasses + c] as number) - max);
        scores[i * nClasses + c] = e;
        total += e;
      }
      for (let c = 0; c < nClasses; c++) {
        scores[i * nClasses + c] = (scores[i * nClasses + c] as number) / total;
      }
    }
    return tensor(scores).reshape([nTest, nClasses]);
  }

  /**
   * Discriminant scores of each sample.
   *
   * For two classes the result is the log-likelihood ratio of the second class, shape
   * `(n_samples,)`; otherwise it holds one score per class, shape `(n_samples, n_classes)`.
   * Only available with `metric: "euclidean"`.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Scores as float64
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {InvalidParameterError} If the metric is not "euclidean"
   * @throws {ShapeError} If X has the wrong dimensions or feature count
   */
  decisionFunction(X: Tensor): Tensor {
    this.assertFitted("decisionFunction");
    this.assertEuclidean("decisionFunction");
    validatePredictInputs(X, this.nFeaturesIn_, "NearestCentroid");
    const nTest = X.shape[0] ?? 0;
    const nClasses = this.classes_.length;
    const scores = this.scores(X);
    if (nClasses === 2) {
      const diff = new Float64Array(nTest);
      for (let i = 0; i < nTest; i++) {
        diff[i] = (scores[i * 2 + 1] as number) - (scores[i * 2] as number);
      }
      return tensor(diff);
    }
    return tensor(scores).reshape([nTest, nClasses]);
  }

  /**
   * Return the mean accuracy on the given test data and labels.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True labels of shape (n_samples,)
   * @returns Accuracy in [0, 1]
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-D or its length differs from the number of samples
   * @throws {DataValidationError} If y is empty or not contiguous
   */
  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    const truth = toFloat64View(y);
    if (truth.length === 0) {
      throw new DataValidationError("y must contain at least one sample");
    }
    const pred = toFloat64View(this.predict(X));
    if (pred.length !== truth.length) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${pred.length}, y=${truth.length}`
      );
    }
    let correct = 0;
    for (let i = 0; i < truth.length; i++) {
      if (pred[i] === truth[i]) correct++;
    }
    return correct / truth.length;
  }

  /**
   * Sorted unique class labels seen during fit.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get classes(): Tensor {
    if (!this.fitted) throw new NotFittedError("NearestCentroid must be fitted to access classes");
    return tensor(Float64Array.from(this.classes_));
  }

  /**
   * Class centroids of shape (n_classes, n_features), after shrinking if `shrinkThreshold` is set.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get centroids(): Tensor {
    if (!this.fitted || !this.centroids_) {
      throw new NotFittedError("NearestCentroid must be fitted to access centroids");
    }
    return tensor(Float64Array.from(this.centroids_)).reshape([
      this.classes_.length,
      this.nFeaturesIn_,
    ]);
  }

  /**
   * Class prior probabilities used by `predict`, `predictProba` and `decisionFunction`.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get classPrior(): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("NearestCentroid must be fitted to access class priors");
    }
    return tensor(Float64Array.from(this.classPrior_));
  }

  getParams(): Record<string, unknown> {
    return {
      metric: this.metric,
      shrinkThreshold: this.shrinkThreshold,
      priors: typeof this.priors === "string" ? this.priors : [...this.priors],
    };
  }

  /**
   * Set hyperparameters. Refit the model afterwards for the change to take effect.
   *
   * @throws {InvalidParameterError} If a value is invalid or a key is unknown
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "metric":
          this.metric = NearestCentroid.checkMetric(value);
          break;
        case "shrinkThreshold":
          this.shrinkThreshold = NearestCentroid.checkShrink(value);
          break;
        case "priors":
          this.priors = NearestCentroid.checkPriors(value);
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
   * @returns A new NearestCentroid
   */
  clone(): NearestCentroid {
    return new NearestCentroid({
      metric: this.metric,
      ...(this.shrinkThreshold !== undefined ? { shrinkThreshold: this.shrinkThreshold } : {}),
      priors: this.priors,
    });
  }

  private assertFitted(method: string): void {
    if (!this.fitted || !this.centroids_) {
      throw new NotFittedError(`NearestCentroid must be fitted before ${method}`);
    }
  }

  private assertEuclidean(method: string): void {
    if (this.metric !== "euclidean") {
      throw new InvalidParameterError(
        `${method} is only available with metric "euclidean"; got "${this.metric}"`,
        "metric",
        this.metric
      );
    }
  }

  private hasUniformPriors(): boolean {
    // Same tolerance as numpy.isclose: atol = 1e-8, rtol = 1e-5.
    const uniform = 1 / this.classPrior_.length;
    return this.classPrior_.every((p) => Math.abs(p - uniform) <= 1e-8 + 1e-5 * uniform);
  }

  /** Resolve the `priors` option into one normalized prior per class. */
  private resolvePriors(counts: Float64Array, n: number): number[] {
    const nClasses = counts.length;
    if (this.priors === "uniform") {
      return new Array<number>(nClasses).fill(1 / nClasses);
    }
    if (this.priors === "empirical") {
      return Array.from(counts, (c) => c / n);
    }
    if (this.priors.length !== nClasses) {
      throw new InvalidParameterError(
        `priors must have one value per class; got ${this.priors.length} for ${nClasses} classes`,
        "priors",
        this.priors
      );
    }
    const total = this.priors.reduce((a, b) => a + b, 0);
    if (!(total > 0)) {
      throw new InvalidParameterError("priors must not sum to zero", "priors", this.priors);
    }
    if (Math.abs(total - 1) > 1e-5 + 1e-8) {
      warn("The priors do not sum to 1. Normalizing such that it sums to one.", "UserWarning");
    }
    return this.priors.map((p) => p / total);
  }

  /**
   * Shrink the centroids in place (Tibshirani et al., 2002): soft-threshold each
   * standardized deviation of a class centroid from the overall centroid.
   */
  private shrinkCentroids(
    centroids: Float64Array,
    x: Float64Array,
    n: number,
    nF: number,
    nClasses: number,
    counts: Float64Array,
    withinStd: Float64Array,
    threshold: number
  ): void {
    const overall = new Float64Array(nF);
    for (let i = 0; i < n; i++) {
      for (let f = 0; f < nF; f++) {
        overall[f] = (overall[f] as number) + (x[i * nF + f] as number);
      }
    }
    for (let f = 0; f < nF; f++) overall[f] = (overall[f] as number) / n;

    const medianStd = median(Array.from(withinStd));
    for (let c = 0; c < nClasses; c++) {
      const m = Math.sqrt(1 / (counts[c] as number) - 1 / n);
      for (let f = 0; f < nF; f++) {
        const scale = m * ((withinStd[f] as number) + medianStd);
        // A zero scale means the feature carries no spread information: keep the centroid.
        if (!(scale > 0)) continue;
        const deviation = ((centroids[c * nF + f] as number) - (overall[f] as number)) / scale;
        const shrunk = Math.sign(deviation) * Math.max(Math.abs(deviation) - threshold, 0);
        centroids[c * nF + f] = (overall[f] as number) + scale * shrunk;
      }
    }
  }

  /** Discriminant scores, flat `(n_samples, n_classes)`. */
  private scores(X: Tensor): Float64Array {
    const nTest = X.shape[0] ?? 0;
    const nF = this.nFeaturesIn_;
    const nClasses = this.classes_.length;
    const x = toFloat64View(X);
    const centroids = this.centroids_ as Float64Array;
    const std = this.withinClassStd_ as Float64Array;
    const euclidean = this.metric === "euclidean";
    const out = new Float64Array(nTest * nClasses);

    for (let i = 0; i < nTest; i++) {
      for (let c = 0; c < nClasses; c++) {
        let d = 0;
        for (let f = 0; f < nF; f++) {
          const s = std[f] as number;
          let diff = (x[i * nF + f] as number) - (centroids[c * nF + f] as number);
          if (s !== 0) diff /= s;
          d += euclidean ? diff * diff : Math.abs(diff);
        }
        // scikit-learn squares the distance of either metric before scoring.
        const squared = euclidean ? d : d * d;
        out[i * nClasses + c] = -squared + 2 * Math.log(this.classPrior_[c] as number);
      }
    }
    return out;
  }
}
