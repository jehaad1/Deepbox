import { InvalidParameterError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { validatePredictInputs, validateUnsupervisedFitInputs } from "../_validation";
import type { Clusterer } from "../base";

/**
 * K-Means clustering algorithm.
 *
 * Partitions n samples into k clusters by minimizing the within-cluster
 * sum of squared distances to cluster centroids.
 *
 * **Algorithm**: Lloyd's algorithm (iterative refinement)
 * 1. Initialize k centroids (random or k-means++)
 * 2. Assign each point to nearest centroid
 * 3. Update centroids as mean of assigned points
 * 4. Repeat until convergence or max iterations
 *
 * **Time Complexity**: O(n * k * i * d) where:
 * - n = number of samples
 * - k = number of clusters
 * - i = number of iterations
 * - d = number of features
 *
 * @example
 * ```ts
 * import { KMeans } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]);
 * const kmeans = new KMeans({ nClusters: 2, randomState: 42 });
 * kmeans.fit(X);
 *
 * const labels = kmeans.predict(X);
 * console.log('Cluster labels:', labels);
 * console.log('Centroids:', kmeans.clusterCenters);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-clustering | Deepbox Clustering}
 * @see {@link https://deepbox.dev/docs/ml-clustering | Deepbox Clustering}
 */
export class KMeans implements Clusterer {
  private nClusters: number;
  private maxIter: number;
  private tol: number;
  private init: "random" | "kmeans++";
  private randomState: number | undefined;

  private clusterCenters_?: Tensor;
  private labels_?: Tensor;
  private inertia_?: number;
  private nIter_?: number;
  private nFeaturesIn_?: number;
  private fitted = false;

  /**
   * Create a new K-Means clustering model.
   *
   * @param options - Configuration options
   * @param options.nClusters - Number of clusters (default: 8)
   * @param options.maxIter - Maximum number of iterations (default: 300)
   * @param options.tol - Tolerance for convergence (default: 1e-4)
   * @param options.init - Initialization method: 'random' or 'kmeans++' (default: 'kmeans++')
   * @param options.randomState - Random seed for reproducibility
   */
  constructor(
    options: {
      readonly nClusters?: number;
      readonly maxIter?: number;
      readonly tol?: number;
      readonly init?: "random" | "kmeans++";
      readonly randomState?: number;
    } = {}
  ) {
    this.nClusters = options.nClusters ?? 8;
    this.maxIter = options.maxIter ?? 300;
    this.tol = options.tol ?? 1e-4;
    this.init = options.init ?? "kmeans++";
    if (options.randomState !== undefined) {
      this.randomState = options.randomState;
    }

    if (!Number.isInteger(this.nClusters) || this.nClusters < 1) {
      throw new InvalidParameterError(
        "nClusters must be an integer >= 1",
        "nClusters",
        this.nClusters
      );
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new InvalidParameterError("maxIter must be an integer >= 1", "maxIter", this.maxIter);
    }
    if (!Number.isFinite(this.tol) || this.tol < 0) {
      throw new InvalidParameterError("tol must be a finite number >= 0", "tol", this.tol);
    }
    if (this.init !== "random" && this.init !== "kmeans++") {
      throw new InvalidParameterError(
        `init must be "random" or "kmeans++"; received ${String(this.init)}`,
        "init",
        this.init
      );
    }
    if (options.randomState !== undefined && !Number.isFinite(options.randomState)) {
      throw new InvalidParameterError(
        `randomState must be a finite number; received ${String(options.randomState)}`,
        "randomState",
        options.randomState
      );
    }
  }

  /**
   * Fit K-Means clustering on training data.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Ignored (exists for compatibility)
   * @returns this - The fitted estimator
   */
  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nFeatures;

    if (nSamples < this.nClusters) {
      throw new InvalidParameterError(
        `n_samples=${nSamples} should be >= n_clusters=${this.nClusters}`,
        "nClusters",
        this.nClusters
      );
    }

    // Initialize centroids
    let centroids = this.initializeCentroids(X);

    let prevInertia = Number.POSITIVE_INFINITY;

    for (let iter = 0; iter < this.maxIter; iter++) {
      // Assign points to nearest centroid
      const labels = this.assignClusters(X, centroids);

      // Update centroids
      const newCentroids: number[][] = [];
      for (let k = 0; k < this.nClusters; k++) {
        const clusterPoints: number[][] = [];
        for (let i = 0; i < nSamples; i++) {
          if (Number(labels.data[labels.offset + i]) === k) {
            const point: number[] = [];
            for (let j = 0; j < nFeatures; j++) {
              point.push(Number(X.data[X.offset + i * nFeatures + j]));
            }
            clusterPoints.push(point);
          }
        }

        if (clusterPoints.length > 0) {
          const centroid: number[] = [];
          for (let j = 0; j < nFeatures; j++) {
            let sum = 0;
            for (const point of clusterPoints) {
              sum += point[j] ?? 0;
            }
            centroid.push(sum / clusterPoints.length);
          }
          newCentroids.push(centroid);
        } else {
          // Keep old centroid if no points assigned
          const oldCentroid: number[] = [];
          for (let j = 0; j < nFeatures; j++) {
            oldCentroid.push(Number(centroids.data[centroids.offset + k * nFeatures + j]));
          }
          newCentroids.push(oldCentroid);
        }
      }

      centroids = tensor(newCentroids);

      // Calculate inertia (sum of squared distances to centroids)
      const inertia = this.calculateInertia(X, centroids, labels);

      // Check convergence
      if (Math.abs(prevInertia - inertia) < this.tol) {
        this.nIter_ = iter + 1;
        break;
      }

      prevInertia = inertia;
      this.nIter_ = iter + 1;
    }

    this.clusterCenters_ = centroids;
    this.labels_ = this.assignClusters(X, centroids);
    this.inertia_ = this.calculateInertia(X, centroids, this.labels_);
    this.fitted = true;

    return this;
  }

  /**
   * Predict cluster labels for samples.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Cluster labels of shape (n_samples,)
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.clusterCenters_) {
      throw new NotFittedError("KMeans must be fitted before prediction");
    }

    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "KMeans");

    return this.assignClusters(X, this.clusterCenters_);
  }

  /**
   * Fit and predict in one step.
   *
   * @param X - Training data
   * @param y - Ignored (exists for compatibility)
   * @returns Cluster labels
   */
  fitPredict(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    if (!this.labels_) {
      throw new NotFittedError("KMeans fit did not produce labels");
    }
    return this.labels_;
  }

  /**
   * Initialize centroids using specified method.
   */
  private initializeCentroids(X: Tensor): Tensor {
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    if (this.init === "random") {
      // Random initialization
      const indices = new Set<number>();
      const rng = this.createRNG();

      while (indices.size < this.nClusters) {
        indices.add(Math.floor(rng() * nSamples));
      }

      const centroids: number[][] = [];
      for (const idx of indices) {
        const centroid: number[] = [];
        for (let j = 0; j < nFeatures; j++) {
          centroid.push(Number(X.data[X.offset + idx * nFeatures + j]));
        }
        centroids.push(centroid);
      }

      return tensor(centroids);
    } else {
      // K-means++ initialization
      const rng = this.createRNG();
      const centroids: number[][] = [];
      const minDistSq = new Float64Array(nSamples).fill(Infinity);

      // Choose first centroid randomly
      const firstIdx = Math.floor(rng() * nSamples);
      const firstCentroid: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        firstCentroid.push(Number(X.data[X.offset + firstIdx * nFeatures + j]));
      }
      centroids.push(firstCentroid);

      // Update distances for first centroid
      for (let i = 0; i < nSamples; i++) {
        let dist = 0;
        for (let j = 0; j < nFeatures; j++) {
          const diff = Number(X.data[X.offset + i * nFeatures + j]) - (firstCentroid[j] ?? 0);
          dist += diff * diff;
        }
        minDistSq[i] = dist;
      }

      // Choose remaining centroids
      for (let k = 1; k < this.nClusters; k++) {
        // Choose next centroid with probability proportional to distance squared
        const totalDist = minDistSq.reduce((a, b) => a + b, 0);
        let r = rng() * totalDist;
        let nextIdx = 0;

        for (let i = 0; i < nSamples; i++) {
          r -= minDistSq[i] ?? 0;
          if (r <= 0) {
            nextIdx = i;
            break;
          }
        }

        const nextCentroid: number[] = [];
        for (let j = 0; j < nFeatures; j++) {
          nextCentroid.push(Number(X.data[X.offset + nextIdx * nFeatures + j]));
        }
        centroids.push(nextCentroid);

        // Update minimum squared distances
        for (let i = 0; i < nSamples; i++) {
          let dist = 0;
          for (let j = 0; j < nFeatures; j++) {
            const diff = Number(X.data[X.offset + i * nFeatures + j]) - (nextCentroid[j] ?? 0);
            dist += diff * diff;
          }
          if (dist < (minDistSq[i] ?? Infinity)) {
            minDistSq[i] = dist;
          }
        }
      }

      return tensor(centroids);
    }
  }

  /**
   * Assign each sample to nearest centroid.
   */
  private assignClusters(X: Tensor, centroids: Tensor): Tensor {
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const labels: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      let minDist = Number.POSITIVE_INFINITY;
      let minLabel = 0;

      for (let k = 0; k < this.nClusters; k++) {
        let dist = 0;
        for (let j = 0; j < nFeatures; j++) {
          const diff =
            Number(X.data[X.offset + i * nFeatures + j]) -
            Number(centroids.data[centroids.offset + k * nFeatures + j]);
          dist += diff * diff;
        }

        if (dist < minDist) {
          minDist = dist;
          minLabel = k;
        }
      }

      labels.push(minLabel);
    }

    return tensor(labels, { dtype: "int32" });
  }

  /**
   * Calculate inertia (sum of squared distances to centroids).
   */
  private calculateInertia(X: Tensor, centroids: Tensor, labels: Tensor): number {
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    let inertia = 0;

    for (let i = 0; i < nSamples; i++) {
      const label = Number(labels.data[labels.offset + i]);
      for (let j = 0; j < nFeatures; j++) {
        const diff =
          Number(X.data[X.offset + i * nFeatures + j]) -
          Number(centroids.data[centroids.offset + label * nFeatures + j]);
        inertia += diff * diff;
      }
    }

    return inertia;
  }

  /**
   * Create a simple RNG for reproducibility.
   */
  private createRNG(): () => number {
    if (this.randomState !== undefined) {
      let seed = this.randomState;
      return () => {
        seed = (seed * 9301 + 49297) % 233280;
        return seed / 233280;
      };
    }
    return Math.random;
  }

  /**
   * Get cluster centers.
   */
  get clusterCenters(): Tensor {
    if (!this.fitted || !this.clusterCenters_) {
      throw new NotFittedError("KMeans must be fitted to access cluster centers");
    }
    return this.clusterCenters_;
  }

  /**
   * Get training labels.
   */
  get labels(): Tensor {
    if (!this.fitted || !this.labels_) {
      throw new NotFittedError("KMeans must be fitted to access labels");
    }
    return this.labels_;
  }

  /**
   * Get inertia (sum of squared distances to centroids).
   */
  get inertia(): number {
    if (!this.fitted || this.inertia_ === undefined) {
      throw new NotFittedError("KMeans must be fitted to access inertia");
    }
    return this.inertia_;
  }

  /**
   * Get number of iterations run.
   */
  get nIter(): number {
    if (!this.fitted || this.nIter_ === undefined) {
      throw new NotFittedError("KMeans must be fitted to access n_iter");
    }
    return this.nIter_;
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      nClusters: this.nClusters,
      maxIter: this.maxIter,
      tol: this.tol,
      init: this.init,
      randomState: this.randomState,
    };
  }

  /**
   * Set the parameters of this estimator.
   *
   * @param params - Parameters to set (nClusters, maxIter, tol, init, randomState)
   * @returns this
   * @throws {InvalidParameterError} If any parameter value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nClusters":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError(
              "nClusters must be an integer >= 1",
              "nClusters",
              value
            );
          }
          this.nClusters = value;
          break;
        case "maxIter":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("maxIter must be an integer >= 1", "maxIter", value);
          }
          this.maxIter = value;
          break;
        case "tol":
          if (typeof value !== "number" || value < 0) {
            throw new InvalidParameterError("tol must be >= 0", "tol", value);
          }
          this.tol = value;
          break;
        case "init":
          if (value !== "random" && value !== "kmeans++") {
            throw new InvalidParameterError(`init must be "random" or "kmeans++"`, "init", value);
          }
          this.init = value;
          break;
        case "randomState":
          if (value !== undefined && (typeof value !== "number" || !Number.isFinite(value))) {
            throw new InvalidParameterError(
              "randomState must be a finite number",
              "randomState",
              value
            );
          }
          this.randomState = value === undefined ? undefined : value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
