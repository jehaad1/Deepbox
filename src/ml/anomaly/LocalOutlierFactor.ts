/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import {
  percentileSorted,
  toFloat64View,
  validatePredictInputs,
  validateUnsupervisedFitInputs,
} from "../_validation";
import type { OutlierDetector } from "../base";

/** Same epsilon scikit-learn adds to the mean reachability distance. */
const LRD_EPS = 1e-10;

function isValidContamination(value: unknown): boolean {
  return value === "auto" || (typeof value === "number" && value > 0 && value <= 0.5);
}

function contaminationError(value: unknown): InvalidParameterError {
  return new InvalidParameterError(
    `contamination must be "auto" or a number in (0, 0.5]; received ${String(value)}`,
    "contamination",
    value
  );
}

function nNeighborsError(value: unknown): InvalidParameterError {
  return new InvalidParameterError(
    `nNeighbors must be an integer >= 1; received ${String(value)}`,
    "nNeighbors",
    value
  );
}

/**
 * Select the `k` smallest entries of `dist` (ignoring index `skip`) and return
 * their indices ordered by increasing distance, ties broken by lower index.
 */
function kSmallest(dist: Float64Array, k: number, skip: number): Int32Array {
  const n = dist.length;
  const candidates = n - (skip >= 0 && skip < n ? 1 : 0);
  const take = Math.min(k, candidates);
  if (take > 32) {
    const order: number[] = [];
    for (let i = 0; i < n; i++) if (i !== skip) order.push(i);
    order.sort((a, b) => (dist[a] as number) - (dist[b] as number));
    return Int32Array.from(order.slice(0, take));
  }
  // Insertion into a small sorted buffer: O(n * take) with no allocation per row.
  const idx = new Int32Array(take);
  let size = 0;
  for (let i = 0; i < n; i++) {
    if (i === skip) continue;
    const d = dist[i] as number;
    if (size === take && d >= (dist[idx[take - 1] as number] as number)) continue;
    let pos = size < take ? size : take - 1;
    while (pos > 0 && d < (dist[idx[pos - 1] as number] as number)) {
      idx[pos] = idx[pos - 1] as number;
      pos--;
    }
    idx[pos] = i;
    if (size < take) size++;
  }
  return idx;
}

/**
 * Local Outlier Factor (LOF) for anomaly detection.
 *
 * Measures the local density deviation of a data point with respect to its
 * neighbors. Points that have substantially lower density than their neighbors
 * are considered outliers.
 *
 * **Algorithm**:
 * 1. For each point, find its k nearest neighbors.
 * 2. Compute reachability distances and local reachability density (LRD).
 * 3. LOF score = average ratio of neighbor LRDs to this point's LRD.
 * 4. LOF ≈ 1 means similar density as neighbors; LOF >> 1 means outlier.
 *
 * `fitPredict` labels the training samples with their fitted LOF.
 * `predict`, `scoreSamples` and `decisionFunction` evaluate arbitrary samples
 * against the training set (novelty mode). A query point that coincides exactly
 * with training points is scored as if those points were left out, so calling
 * `predict` on training data without duplicate rows reproduces the fitted labels.
 * This differs from scikit-learn, which counts a coinciding training point as a
 * neighbor at distance 0.
 *
 * With `contamination: "auto"` a sample is an outlier when its LOF is above 1.5;
 * otherwise the threshold is the `1 - contamination` quantile of the training LOF values.
 *
 * @example
 * ```ts
 * import { LocalOutlierFactor } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0], [0.1, -0.1], [0.2, 0.1], [100, 100]]);
 * const lof = new LocalOutlierFactor({ nNeighbors: 2, contamination: 0.25 });
 * lof.fit(X);
 * const labels = lof.predict(X); // -1 for outliers, 1 for inliers
 * ```
 */
export class LocalOutlierFactor implements OutlierDetector {
  private nNeighbors: number;
  private contamination: number | "auto";

  private trainData_?: Float64Array;
  private lofScores_?: Float64Array;
  /** LOF values above this are outliers. */
  private threshold_ = 1.5;
  // Per-training-point statistics needed to compute a proper LOF for NEW
  // (novelty) points: k-distance and local reachability density.
  private trainKDist_?: Float64Array;
  private trainLrd_?: Float64Array;
  /** Neighbor count actually used at fit time (nNeighbors clamped to nSamples - 1). */
  private nNeighborsFit_ = 0;
  private nFeaturesIn_ = 0;
  private nTrain_ = 0;
  private fitted = false;

  /**
   * @param options.nNeighbors - Number of neighbors for kNN (default: 20). Clamped to
   *   `nSamples - 1` at fit time when larger.
   * @param options.contamination - Expected proportion of outliers in (0, 0.5], or `"auto"` (default)
   * @throws {InvalidParameterError} If an option is out of range
   */
  constructor(
    options: {
      readonly nNeighbors?: number;
      readonly contamination?: number | "auto";
    } = {}
  ) {
    const nNeighbors = options.nNeighbors ?? 20;
    const contamination = options.contamination ?? "auto";
    if (!Number.isInteger(nNeighbors) || nNeighbors < 1) throw nNeighborsError(nNeighbors);
    if (!isValidContamination(contamination)) throw contaminationError(contamination);
    this.nNeighbors = nNeighbors;
    this.contamination = contamination;
  }

  /** Euclidean distances from `query` (row `qOff` of `q`) to every training row. */
  private distancesToTrain(q: Float64Array, qOff: number, out: Float64Array): void {
    const train = this.trainData_ as Float64Array;
    const d = this.nFeaturesIn_;
    for (let t = 0; t < this.nTrain_; t++) {
      let sum = 0;
      const tOff = t * d;
      for (let j = 0; j < d; j++) {
        const diff = (q[qOff + j] as number) - (train[tOff + j] as number);
        sum += diff * diff;
      }
      out[t] = Math.sqrt(sum);
    }
  }

  /**
   * Fit the model on the training data, computing k-distances, local
   * reachability densities and the LOF of every training sample.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (exists for API compatibility)
   * @returns this
   * @throws {ShapeError} If X is not 2D
   * @throws {DataValidationError} If X is empty, non-contiguous or contains NaN/Inf
   * @throws {InvalidParameterError} If X has fewer than 2 samples
   */
  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    const k = Math.min(this.nNeighbors, nSamples - 1);
    if (k < 1) {
      throw new InvalidParameterError(
        `Not enough samples for the specified nNeighbors: at least 2 samples are needed to find a neighbor; got ${nSamples}`,
        "nNeighbors",
        this.nNeighbors
      );
    }

    // Copy so later in-place edits of X cannot change the fitted model.
    const data = Float64Array.from(toFloat64View(X));
    this.trainData_ = data;
    this.nTrain_ = nSamples;
    this.nFeaturesIn_ = nFeatures;

    const neighborIdx: Int32Array[] = [];
    const neighborDist: Float64Array[] = [];
    const kDist = new Float64Array(nSamples);
    const dist = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      this.distancesToTrain(data, i * nFeatures, dist);
      const idx = kSmallest(dist, k, i);
      const nd = new Float64Array(idx.length);
      for (let n = 0; n < idx.length; n++) nd[n] = dist[idx[n] as number] as number;
      neighborIdx.push(idx);
      neighborDist.push(nd);
      // k-distance: distance to the k-th nearest neighbor.
      kDist[i] = nd[nd.length - 1] as number;
    }

    // Local reachability density: 1 / mean reach-dist, with
    // reach-dist(p, o) = max(k-distance(o), dist(p, o)).
    const lrd = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      const idx = neighborIdx[i] as Int32Array;
      const nd = neighborDist[i] as Float64Array;
      let reachSum = 0;
      for (let n = 0; n < idx.length; n++) {
        reachSum += Math.max(kDist[idx[n] as number] as number, nd[n] as number);
      }
      lrd[i] = 1 / (reachSum / idx.length + LRD_EPS);
    }

    // LOF(p) = mean over neighbors o of lrd(o) / lrd(p).
    const lof = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      const idx = neighborIdx[i] as Int32Array;
      let sum = 0;
      for (let n = 0; n < idx.length; n++) sum += lrd[idx[n] as number] as number;
      lof[i] = sum / idx.length / (lrd[i] as number);
    }

    this.nNeighborsFit_ = k;
    this.trainKDist_ = kDist;
    this.trainLrd_ = lrd;
    this.lofScores_ = lof;

    if (this.contamination === "auto") {
      this.threshold_ = 1.5;
    } else {
      // Linear-interpolated percentile of -LOF at `contamination`, as in
      // numpy.percentile; samples with a larger LOF are outliers.
      const neg = Float64Array.from(lof, (v) => -v).sort();
      this.threshold_ = -percentileSorted(neg, 100 * this.contamination);
    }

    this.fitted = true;
    return this;
  }

  /**
   * LOF of each query row computed against the training set.
   */
  private queryLof(X: Tensor, method: string): Float64Array {
    if (!this.fitted) {
      throw new NotFittedError(`LocalOutlierFactor must be fitted before ${method}`);
    }
    validatePredictInputs(X, this.nFeaturesIn_, "LocalOutlierFactor");

    const nQuery = X.shape[0] ?? 0;
    const q = toFloat64View(X);
    const nTrain = this.nTrain_;
    const d = this.nFeaturesIn_;
    const trainKDist = this.trainKDist_ as Float64Array;
    const trainLrd = this.trainLrd_ as Float64Array;
    const k = this.nNeighborsFit_;
    const out = new Float64Array(nQuery);
    const dist = new Float64Array(nTrain);
    const masked = new Float64Array(nTrain);

    for (let i = 0; i < nQuery; i++) {
      this.distancesToTrain(q, i * d, dist);

      // Training points at distance exactly 0 are the query itself (or exact
      // duplicates of it). Leave them out when enough other points remain, so
      // that scoring a training sample reproduces its fitted LOF.
      let zeros = 0;
      for (let t = 0; t < nTrain; t++) if (dist[t] === 0) zeros++;
      let source = dist;
      if (zeros > 0 && nTrain - zeros >= k) {
        for (let t = 0; t < nTrain; t++) {
          const v = dist[t] as number;
          masked[t] = v === 0 ? Infinity : v;
        }
        source = masked;
      }
      const idx = kSmallest(source, k, -1);

      // lrd(q) = 1 / mean reachDist(q, o), reachDist = max(k-dist(o), d(q, o))
      let reachSum = 0;
      for (let n = 0; n < idx.length; n++) {
        const t = idx[n] as number;
        reachSum += Math.max(trainKDist[t] as number, dist[t] as number);
      }
      const lrdQ = 1 / (reachSum / idx.length + LRD_EPS);
      // LOF(q) = mean lrd(o) / lrd(q)
      let lrdSum = 0;
      for (let n = 0; n < idx.length; n++) lrdSum += trainLrd[idx[n] as number] as number;
      out[i] = lrdSum / idx.length / lrdQ;
    }
    return out;
  }

  /**
   * Predict whether each sample is an outlier relative to the training set.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns int32 labels of shape (n_samples,): -1 for outliers, 1 for inliers
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has a different number of features than the training data
   */
  predict(X: Tensor): Tensor {
    const lof = this.queryLof(X, "prediction");
    const labels = new Int32Array(lof.length);
    for (let i = 0; i < lof.length; i++) {
      labels[i] = (lof[i] as number) > this.threshold_ ? -1 : 1;
    }
    return tensor(labels, { dtype: "int32" });
  }

  /**
   * Fit the model and label the training samples with their fitted LOF.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored
   * @returns int32 labels of shape (n_samples,): -1 for outliers, 1 for inliers
   */
  fitPredict(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    const lof = this.lofScores_ as Float64Array;
    const labels = new Int32Array(lof.length);
    for (let i = 0; i < lof.length; i++) {
      labels[i] = (lof[i] as number) > this.threshold_ ? -1 : 1;
    }
    return tensor(labels, { dtype: "int32" });
  }

  /**
   * Opposite of the LOF of each sample (lower is more abnormal), computed
   * against the training set.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Scores of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has a different number of features than the training data
   */
  scoreSamples(X: Tensor): Tensor {
    const lof = this.queryLof(X, "scoring");
    for (let i = 0; i < lof.length; i++) lof[i] = -(lof[i] as number);
    return tensor(lof, { dtype: "float64" });
  }

  /**
   * Shifted opposite of the LOF: `scoreSamples(X) - offset`. Negative values are outliers.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Decision values of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   */
  decisionFunction(X: Tensor): Tensor {
    const lof = this.queryLof(X, "decisionFunction");
    for (let i = 0; i < lof.length; i++) lof[i] = this.threshold_ - (lof[i] as number);
    return tensor(lof, { dtype: "float64" });
  }

  /**
   * Opposite of the LOF of the training samples, shape (n_samples,). The more
   * negative, the more abnormal the sample.
   *
   * Mirrors scikit-learn's `negative_outlier_factor_`.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get negativeOutlierFactor(): Tensor {
    if (!this.fitted || !this.lofScores_) {
      throw new NotFittedError("LocalOutlierFactor must be fitted to access LOF scores");
    }
    return tensor(
      Float64Array.from(this.lofScores_, (s) => -s),
      { dtype: "float64" }
    );
  }

  /**
   * Opposite of the LOF of the training samples.
   *
   * Alias of {@link LocalOutlierFactor.negativeOutlierFactor}. Values are negated LOF
   * scores, so lower (more negative) means more anomalous.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get negativeLofScores(): Tensor {
    return this.negativeOutlierFactor;
  }

  /**
   * Decision threshold in `scoreSamples` units: samples scoring below it are outliers.
   * Equals -1.5 when `contamination` is `"auto"`.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get offset(): number {
    if (!this.fitted) {
      throw new NotFittedError("LocalOutlierFactor must be fitted before accessing offset");
    }
    return -this.threshold_;
  }

  getParams(): Record<string, unknown> {
    return { nNeighbors: this.nNeighbors, contamination: this.contamination };
  }

  /**
   * Set parameters. All values are validated before any of them is applied.
   * Changing parameters does not affect an already fitted model until `fit` is called again.
   *
   * @throws {InvalidParameterError} If a name is unknown or a value is out of range
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nNeighbors":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw nNeighborsError(value);
          }
          break;
        case "contamination":
          if (!isValidContamination(value)) throw contaminationError(value);
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    for (const [key, value] of Object.entries(params)) {
      if (key === "nNeighbors") this.nNeighbors = value as number;
      else if (key === "contamination") this.contamination = value as number | "auto";
    }
    return this;
  }

  /**
   * Create an unfitted copy with the same parameters.
   */
  clone(): LocalOutlierFactor {
    return new LocalOutlierFactor({
      nNeighbors: this.nNeighbors,
      contamination: this.contamination,
    });
  }
}
