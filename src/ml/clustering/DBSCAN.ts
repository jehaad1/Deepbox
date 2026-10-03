import { InvalidParameterError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import {
  toFloat64View,
  validatePredictInputs,
  validateUnsupervisedFitInputs,
} from "../_validation";
import type { Clusterer } from "../base";

/**
 * DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
 *
 * Clusters points based on density. Points in high-density regions are
 * grouped together, while points in low-density regions are marked as noise.
 *
 * **Algorithm**:
 * 1. For each point, find all neighbors within eps distance (the point itself counts)
 * 2. If a point has at least minSamples neighbors, it's a core point
 * 3. Core points and their neighbors form clusters
 * 4. Points not reachable from any core point are noise (label = -1)
 *
 * A border point that lies within `eps` of core points from several clusters
 * joins the cluster that reaches it first, so labels of border points depend on
 * the order of the rows in `X`.
 *
 * **Advantages**:
 * - No need to specify number of clusters
 * - Can find arbitrarily shaped clusters
 * - Points in sparse regions are labeled as noise instead of distorting a cluster
 *
 * @example
 * ```ts
 * import { DBSCAN } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]]);
 * const dbscan = new DBSCAN({ eps: 3, minSamples: 2 });
 * const labels = dbscan.fitPredict(X);
 * // labels: [0, 0, 0, 1, 1, -1]  (-1 = noise)
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-clustering | Deepbox Clustering}
 */
export class DBSCAN implements Clusterer {
  private eps: number;
  private minSamples: number;
  private metric: "euclidean" | "manhattan";

  private labels_?: Tensor;
  private coreIndices_?: Int32Array;
  // Core samples (rows) and their cluster labels, kept for predict().
  private coreData_?: Float64Array;
  private coreLabels_?: Int32Array;
  private nFeaturesIn_ = 0;
  // Radius and metric of the last fit, so setParams() after fit does not change predict().
  private fitEps_ = 0;
  private fitMetric_: "euclidean" | "manhattan" = "euclidean";
  private fitted = false;

  /**
   * Create a DBSCAN model.
   *
   * @param options - Configuration options
   * @param options.eps - Neighborhood radius, finite and > 0 (default: 0.5)
   * @param options.minSamples - Neighbors (including the point itself) needed for a core point, integer >= 1 (default: 5)
   * @param options.metric - Distance metric: "euclidean" or "manhattan" (default: "euclidean")
   * @throws {InvalidParameterError} If any option is out of range
   */
  constructor(
    options: {
      readonly eps?: number;
      readonly minSamples?: number;
      readonly metric?: "euclidean" | "manhattan";
    } = {}
  ) {
    this.eps = options.eps ?? 0.5;
    this.minSamples = options.minSamples ?? 5;
    this.metric = options.metric ?? "euclidean";

    if (!Number.isFinite(this.eps) || this.eps <= 0) {
      throw new InvalidParameterError("eps must be a finite number > 0", "eps", this.eps);
    }
    if (!Number.isInteger(this.minSamples) || this.minSamples < 1) {
      throw new InvalidParameterError(
        "minSamples must be an integer >= 1",
        "minSamples",
        this.minSamples
      );
    }
    if (this.metric !== "euclidean" && this.metric !== "manhattan") {
      throw new InvalidParameterError(
        `metric must be "euclidean" or "manhattan"`,
        "metric",
        this.metric
      );
    }
  }

  /**
   * Whether the rows at `aBase` of `a` and `bBase` of `b` are at most `eps` apart.
   * Euclidean distances are compared squared, so no square root is taken.
   */
  private static within(
    a: Float64Array,
    aBase: number,
    b: Float64Array,
    bBase: number,
    d: number,
    eps: number,
    metric: "euclidean" | "manhattan"
  ): boolean {
    if (metric === "manhattan") {
      let sum = 0;
      for (let f = 0; f < d; f++) {
        sum += Math.abs((a[aBase + f] as number) - (b[bBase + f] as number));
        if (sum > eps) return false;
      }
      return true;
    }
    const epsSq = eps * eps;
    let sumSq = 0;
    for (let f = 0; f < d; f++) {
      const diff = (a[aBase + f] as number) - (b[bBase + f] as number);
      sumSq += diff * diff;
      if (sumSq > epsSq) return false;
    }
    return true;
  }

  /** Distance used to rank candidates: squared Euclidean or Manhattan (monotone in the true distance). */
  private static rankDistance(
    a: Float64Array,
    aBase: number,
    b: Float64Array,
    bBase: number,
    d: number,
    metric: "euclidean" | "manhattan"
  ): number {
    let s = 0;
    if (metric === "manhattan") {
      for (let f = 0; f < d; f++)
        s += Math.abs((a[aBase + f] as number) - (b[bBase + f] as number));
    } else {
      for (let f = 0; f < d; f++) {
        const diff = (a[aBase + f] as number) - (b[bBase + f] as number);
        s += diff * diff;
      }
    }
    return s;
  }

  /** Indices of all samples within `eps` of sample `idx` (including `idx`). */
  private static getNeighbors(
    data: Float64Array,
    n: number,
    d: number,
    idx: number,
    eps: number,
    metric: "euclidean" | "manhattan"
  ): number[] {
    const neighbors: number[] = [];
    const base = idx * d;
    for (let i = 0; i < n; i++) {
      if (DBSCAN.within(data, base, data, i * d, d, eps, metric)) neighbors.push(i);
    }
    return neighbors;
  }

  /**
   * Perform DBSCAN clustering on data X.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (exists for API compatibility)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D
   * @throws {DataValidationError} If X is empty or contains NaN/Inf values
   */
  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);

    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const data = toFloat64View(X);
    const eps = this.eps;
    const metric = this.metric;

    const UNVISITED = -2;
    const NOISE = -1;
    const labels = new Int32Array(n).fill(UNVISITED);
    const isCore = new Uint8Array(n);
    const queued = new Uint8Array(n);

    let clusterId = 0;
    for (let i = 0; i < n; i++) {
      if (labels[i] !== UNVISITED) continue;

      const neighbors = DBSCAN.getNeighbors(data, n, d, i, eps, metric);
      if (neighbors.length < this.minSamples) {
        // Noise for now; a later cluster may still claim it as a border point.
        labels[i] = NOISE;
        continue;
      }

      isCore[i] = 1;
      labels[i] = clusterId;

      // Breadth-first expansion; `queued` keeps each point in the queue at most once.
      const queue: number[] = neighbors;
      for (const q of queue) queued[q] = 1;
      for (let head = 0; head < queue.length; head++) {
        const q = queue[head] as number;
        if (labels[q] === NOISE) {
          // Border point previously seen as noise; it is not a core point.
          labels[q] = clusterId;
          continue;
        }
        if (labels[q] !== UNVISITED) continue;

        labels[q] = clusterId;
        const qNeighbors = DBSCAN.getNeighbors(data, n, d, q, eps, metric);
        if (qNeighbors.length >= this.minSamples) {
          isCore[q] = 1;
          for (const nb of qNeighbors) {
            if (queued[nb] === 0 && (labels[nb] === UNVISITED || labels[nb] === NOISE)) {
              queued[nb] = 1;
              queue.push(nb);
            }
          }
        }
      }
      for (const q of queue) queued[q] = 0;

      clusterId++;
    }

    let nCore = 0;
    for (let i = 0; i < n; i++) nCore += isCore[i] as number;
    const coreIndices = new Int32Array(nCore);
    const coreData = new Float64Array(nCore * d);
    const coreLabels = new Int32Array(nCore);
    let c = 0;
    for (let i = 0; i < n; i++) {
      if (isCore[i] === 0) continue;
      coreIndices[c] = i;
      coreLabels[c] = labels[i] as number;
      for (let f = 0; f < d; f++) coreData[c * d + f] = data[i * d + f] as number;
      c++;
    }

    this.labels_ = tensor(labels);
    this.coreIndices_ = coreIndices;
    this.coreData_ = coreData;
    this.coreLabels_ = coreLabels;
    this.nFeaturesIn_ = d;
    this.fitEps_ = eps;
    this.fitMetric_ = metric;
    this.fitted = true;

    return this;
  }

  /**
   * Predict cluster labels for new samples using nearest-neighbor assignment.
   *
   * Each new point is assigned the label of its nearest core sample from the
   * training data. Points with no core sample within `eps` distance are
   * labeled as noise (-1). The `eps` and `metric` of the last `fit` are used, even
   * if `setParams` changed them afterwards.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Cluster labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X is not 2D or has a different number of features than the training data
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.coreData_ || !this.coreLabels_) {
      throw new NotFittedError("DBSCAN must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "DBSCAN");

    const n = X.shape[0] ?? 0;
    const d = this.nFeaturesIn_;
    const data = toFloat64View(X);
    const core = this.coreData_;
    const coreLabels = this.coreLabels_;
    const nCore = coreLabels.length;
    const result = new Int32Array(n).fill(-1);

    for (let i = 0; i < n; i++) {
      let bestDist = Infinity;
      let bestLabel = -1;
      for (let c = 0; c < nCore; c++) {
        // Only core samples within eps can claim the point.
        if (!DBSCAN.within(data, i * d, core, c * d, d, this.fitEps_, this.fitMetric_)) continue;
        const dist = DBSCAN.rankDistance(data, i * d, core, c * d, d, this.fitMetric_);
        if (dist < bestDist) {
          bestDist = dist;
          bestLabel = coreLabels[c] as number;
        }
      }
      result[i] = bestLabel;
    }

    return tensor(result);
  }

  /**
   * Fit DBSCAN and return cluster labels.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (exists for API compatibility)
   * @returns Cluster labels of shape (n_samples,). Noise points are labeled -1.
   * @throws {ShapeError} If X is not 2D
   * @throws {DataValidationError} If X is empty or contains NaN/Inf values
   * @throws {NotFittedError} If fit did not produce labels (internal error)
   */
  fitPredict(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    if (!this.labels_) {
      throw new NotFittedError("DBSCAN fit did not produce labels");
    }
    return this.labels_;
  }

  /**
   * Get cluster labels assigned during fitting.
   *
   * @returns Tensor of cluster labels. Noise points are labeled -1.
   * @throws {NotFittedError} If the model has not been fitted
   */
  get labels(): Tensor {
    if (!this.fitted || !this.labels_) {
      throw new NotFittedError("DBSCAN must be fitted to access labels");
    }
    return this.labels_;
  }

  /**
   * Number of clusters found (excluding noise).
   *
   * @returns Number of distinct clusters (labels >= 0)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nClusters(): number {
    if (!this.fitted || !this.labels_) {
      throw new NotFittedError("DBSCAN must be fitted to access nClusters");
    }
    // Labels are assigned consecutively from 0, so the cluster count is max + 1.
    let max = -1;
    for (let i = 0; i < this.labels_.size; i++) {
      const label = Number(this.labels_.data[this.labels_.offset + i]);
      if (label > max) max = label;
    }
    return max + 1;
  }

  /**
   * Get indices of core samples discovered during fitting.
   *
   * Core samples are points with at least `minSamples` neighbors within `eps`
   * (the point itself counts as a neighbor).
   *
   * @returns Array of core sample indices in increasing order
   * @throws {NotFittedError} If the model has not been fitted
   */
  get coreIndices(): number[] {
    if (!this.fitted || !this.coreIndices_) {
      throw new NotFittedError("DBSCAN must be fitted to access core indices");
    }
    return Array.from(this.coreIndices_);
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      eps: this.eps,
      minSamples: this.minSamples,
      metric: this.metric,
    };
  }

  /**
   * Set the parameters of this estimator.
   *
   * @param params - Parameters to set (eps, minSamples, metric)
   * @returns this
   * @throws {InvalidParameterError} If any parameter value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "eps":
          if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
            throw new InvalidParameterError("eps must be a finite number > 0", "eps", value);
          }
          this.eps = value;
          break;
        case "minSamples":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError(
              "minSamples must be an integer >= 1",
              "minSamples",
              value
            );
          }
          this.minSamples = value;
          break;
        case "metric":
          if (value !== "euclidean" && value !== "manhattan") {
            throw new InvalidParameterError(
              `metric must be "euclidean" or "manhattan"`,
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
