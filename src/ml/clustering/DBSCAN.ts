import { InvalidParameterError, NotFittedError, NotImplementedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { validateUnsupervisedFitInputs } from "../_validation";
import type { Clusterer } from "../base";

/**
 * DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
 *
 * Clusters points based on density. Points in high-density regions are
 * grouped together, while points in low-density regions are marked as noise.
 *
 * **Algorithm**:
 * 1. For each point, find all neighbors within eps distance
 * 2. If a point has at least minSamples neighbors, it's a core point
 * 3. Core points and their neighbors form clusters
 * 4. Points not reachable from any core point are noise (label = -1)
 *
 * **Advantages**:
 * - No need to specify number of clusters
 * - Can find arbitrarily shaped clusters
 * - Robust to outliers
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
  private coreIndices_?: number[];
  private fitted = false;

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
   * Check if two points are neighbors (distance <= eps).
   */
  private isNeighbor(a: number[], b: number[]): boolean {
    if (this.metric === "manhattan") {
      let sum = 0;
      for (let i = 0; i < a.length; i++) {
        sum += Math.abs((a[i] ?? 0) - (b[i] ?? 0));
        if (sum > this.eps) return false;
      }
      return sum <= this.eps;
    }

    // Euclidean distance
    let sumSq = 0;
    const epsSq = this.eps * this.eps;
    for (let i = 0; i < a.length; i++) {
      const diff = (a[i] ?? 0) - (b[i] ?? 0);
      sumSq += diff * diff;
      if (sumSq > epsSq) return false;
    }
    return sumSq <= epsSq;
  }

  /**
   * Find all neighbors within eps distance.
   */
  private getNeighbors(data: number[][], pointIdx: number): number[] {
    const neighbors: number[] = [];
    const point = data[pointIdx];
    if (!point) return neighbors;

    for (let i = 0; i < data.length; i++) {
      const other = data[i];
      if (other && this.isNeighbor(point, other)) {
        neighbors.push(i);
      }
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
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    // Extract data
    const data: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        row.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      data.push(row);
    }

    // Initialize labels to undefined (-2 means unvisited)
    const labels: number[] = new Array(nSamples).fill(-2);
    const coreIndices: number[] = [];

    let clusterId = 0;

    for (let i = 0; i < nSamples; i++) {
      // Skip if already processed
      if (labels[i] !== -2) continue;

      const neighbors = this.getNeighbors(data, i);

      if (neighbors.length < this.minSamples) {
        // Mark as noise (for now, might be claimed by another cluster later)
        labels[i] = -1;
        continue;
      }

      // Start a new cluster
      coreIndices.push(i);
      labels[i] = clusterId;

      // Process neighbors
      const seedSet = new Set(neighbors);

      for (const q of seedSet) {
        // If noise, claim it for this cluster
        if (labels[q] === -1) {
          labels[q] = clusterId;
        }

        // If unvisited
        if (labels[q] === -2) {
          labels[q] = clusterId;

          const qNeighbors = this.getNeighbors(data, q);

          if (qNeighbors.length >= this.minSamples) {
            coreIndices.push(q);
            // Add new neighbors to seed set
            for (const n of qNeighbors) {
              if (labels[n] === -2 || labels[n] === -1) {
                seedSet.add(n);
              }
            }
          }
        }
      }

      clusterId++;
    }

    this.labels_ = tensor(labels, { dtype: "int32" });
    this.coreIndices_ = coreIndices;
    this.fitted = true;

    return this;
  }

  /**
   * Predict cluster labels for samples in X.
   *
   * @param _X - Samples (unused)
   * @throws {NotImplementedError} Always — DBSCAN is transductive and does not support prediction on new data
   */
  predict(_X: Tensor): Tensor {
    throw new NotImplementedError(
      "DBSCAN is a transductive clustering algorithm and does not support prediction on new data. Use fitPredict() instead."
    );
  }

  /**
   * Fit DBSCAN and return cluster labels.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (exists for API compatibility)
   * @returns Cluster labels of shape (n_samples,). Noise points are labeled -1.
   * @throws {ShapeError} If X is not 2D
   * @throws {DataValidationError} If X contains NaN/Inf values
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
    const unique = new Set<number>();
    for (let i = 0; i < this.labels_.size; i++) {
      const label = Number(this.labels_.data[this.labels_.offset + i]);
      if (label >= 0) unique.add(label);
    }
    return unique.size;
  }

  /**
   * Get indices of core samples discovered during fitting.
   *
   * Core samples are points with at least `minSamples` neighbors within `eps`.
   *
   * @returns Array of core sample indices
   * @throws {NotFittedError} If the model has not been fitted
   */
  get coreIndices(): number[] {
    if (!this.fitted || !this.coreIndices_) {
      throw new NotFittedError("DBSCAN must be fitted to access core indices");
    }
    return [...this.coreIndices_];
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
          if (typeof value !== "number" || value <= 0) {
            throw new InvalidParameterError("eps must be > 0", "eps", value);
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
