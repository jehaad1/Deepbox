/**
 * Spectral Clustering.
 *
 * Performs clustering by embedding the data in the eigenspace of the
 * graph Laplacian, then applying KMeans in that space.
 *
 * **Algorithm** (as in scikit-learn):
 * 1. Build a symmetric affinity matrix W (RBF kernel, k-nearest-neighbor graph, or a
 *    precomputed matrix)
 * 2. Form the normalized Laplacian L = I - D^{-1/2} W D^{-1/2}
 * 3. Take the eigenvectors of the `nComponents` smallest eigenvalues of L and divide each
 *    row by sqrt(degree), which recovers the eigenvectors of the random-walk Laplacian
 * 4. Cluster the rows of that embedding with KMeans
 *
 * @module ml/clustering/SpectralClustering
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import {
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
  warn,
} from "../../core";
import { eigh } from "../../linalg";
import { type Tensor, tensor } from "../../ndarray";
import { kthSmallest } from "../_internal";
import {
  toFloat64View,
  validatePredictInputs,
  validateUnsupervisedFitInputs,
} from "../_validation";
import type { Clusterer } from "../base";
import { KMeans } from "./KMeans";

type Affinity = "rbf" | "nearest_neighbors" | "precomputed";

function checkNClusters(value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new InvalidParameterError("nClusters must be an integer >= 1", "nClusters", value);
  }
  return value;
}

function checkAffinity(value: unknown): Affinity {
  if (value !== "rbf" && value !== "nearest_neighbors" && value !== "precomputed") {
    throw new InvalidParameterError(
      `affinity must be "rbf", "nearest_neighbors" or "precomputed"`,
      "affinity",
      value
    );
  }
  return value;
}

function checkGamma(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new InvalidParameterError("gamma must be a finite number > 0", "gamma", value);
  }
  return value;
}

function checkNNeighbors(value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new InvalidParameterError("nNeighbors must be an integer >= 1", "nNeighbors", value);
  }
  return value;
}

function checkRandomState(value: unknown): number | undefined {
  if (value !== undefined && (typeof value !== "number" || !Number.isFinite(value))) {
    throw new InvalidParameterError("randomState must be a finite number", "randomState", value);
  }
  return value;
}

function checkNInit(value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new InvalidParameterError("nInit must be an integer >= 1", "nInit", value);
  }
  return value;
}

function checkNComponents(value: unknown): number | undefined {
  if (value !== undefined && (typeof value !== "number" || !Number.isInteger(value) || value < 1)) {
    throw new InvalidParameterError(
      "nComponents must be an integer >= 1 or undefined",
      "nComponents",
      value
    );
  }
  return value;
}

/**
 * Spectral Clustering using graph Laplacian eigenvectors.
 *
 * Particularly effective for non-convex clusters and clusters with
 * complex shapes that KMeans cannot handle well.
 *
 * Affinities:
 * - `"rbf"`: `exp(-gamma * ||x_i - x_j||^2)` between all pairs (default).
 * - `"nearest_neighbors"`: each sample is linked to its `nNeighbors` nearest samples, counting
 *   itself as the first (so it has `nNeighbors - 1` neighbors, as in scikit-learn); the graph is
 *   symmetrized as `(C + C^T) / 2`.
 * - `"precomputed"`: `X` is itself a square, non-negative affinity matrix; it is symmetrized
 *   as `(A + A^T) / 2` and the diagonal is ignored.
 *
 * The method builds and decomposes a dense n x n matrix, so memory is O(n^2) and time O(n^3).
 *
 * @example
 * ```ts
 * import { SpectralClustering } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0], [0.1, 0], [10, 10], [10.1, 10]]);
 * const sc = new SpectralClustering({ nClusters: 2 });
 * sc.fit(X);
 * console.log(sc.labels);
 * ```
 */
export class SpectralClustering implements Clusterer {
  private nClusters: number;
  private affinity: Affinity;
  private gamma: number;
  private nNeighbors: number;
  private randomState: number | undefined;
  private nInit: number;
  private nComponents: number | undefined;

  private labels_?: Tensor;
  private fitData_?: Float64Array;
  private nFeaturesIn_ = 0;
  private fitted = false;

  /**
   * Create a new Spectral Clustering model.
   *
   * @param options - Configuration options
   * @param options.nClusters - Number of clusters (default: 8)
   * @param options.affinity - 'rbf', 'nearest_neighbors' or 'precomputed' (default: 'rbf')
   * @param options.gamma - Kernel coefficient of the 'rbf' affinity (default: 1.0)
   * @param options.nNeighbors - Neighborhood size of the 'nearest_neighbors' affinity, the sample itself included (default: 10)
   * @param options.randomState - Random seed for the KMeans step
   * @param options.nInit - Number of KMeans initializations (default: 10)
   * @param options.nComponents - Number of eigenvectors used for the embedding (default: `nClusters`)
   * @throws {InvalidParameterError} If any option is out of range
   */
  constructor(
    options: {
      readonly nClusters?: number;
      readonly affinity?: Affinity;
      readonly gamma?: number;
      readonly nNeighbors?: number;
      readonly randomState?: number;
      readonly nInit?: number;
      readonly nComponents?: number;
    } = {}
  ) {
    this.nClusters = checkNClusters(options.nClusters ?? 8);
    this.affinity = checkAffinity(options.affinity ?? "rbf");
    this.gamma = checkGamma(options.gamma ?? 1.0);
    this.nNeighbors = checkNNeighbors(options.nNeighbors ?? 10);
    this.randomState = checkRandomState(options.randomState);
    this.nInit = checkNInit(options.nInit ?? 10);
    this.nComponents = checkNComponents(options.nComponents);
  }

  /**
   * Fit Spectral Clustering.
   *
   * Calling `fit` again replaces the previous model.
   *
   * @param X - Training data of shape (n_samples, n_features), or the (n_samples, n_samples)
   *   affinity matrix when `affinity` is 'precomputed'
   * @param _y - Ignored (exists for compatibility)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D, or not square with 'precomputed'
   * @throws {DataValidationError} If X is empty, contains NaN/Inf, or is a precomputed matrix with negative entries
   * @throws {InvalidParameterError} If there are fewer samples than clusters, or `nComponents` exceeds the number of samples
   */
  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;

    if (this.affinity === "precomputed" && n !== d) {
      throw new ShapeError(`A precomputed affinity matrix must be square; got shape [${n}, ${d}]`);
    }
    if (n < this.nClusters) {
      throw new InvalidParameterError(
        `n_samples=${n} should be >= n_clusters=${this.nClusters}`,
        "nClusters",
        this.nClusters
      );
    }
    const k = this.nComponents ?? this.nClusters;
    if (k > n) {
      throw new InvalidParameterError(
        `nComponents=${k} should be <= n_samples=${n}`,
        "nComponents",
        k
      );
    }

    const data = toFloat64View(X);

    // Step 1: symmetric affinity matrix with an ignored (zero) diagonal.
    const W = this.buildAffinityMatrix(data, n, d);

    if (!SpectralClustering.isConnected(W, n)) {
      warn(
        "Graph is not fully connected, spectral embedding may not work as expected.",
        "UserWarning",
        "SpectralClustering"
      );
    }

    // Step 2: L = I - D^{-1/2} W D^{-1/2}. Isolated samples (degree 0) use a scale of 1, which
    // leaves their row of L equal to the identity row.
    const scale = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let degree = 0;
      for (let j = 0; j < n; j++) degree += W[i * n + j] as number;
      scale[i] = degree > 0 ? Math.sqrt(degree) : 1;
    }
    // Reuse W's buffer; fill each pair from one expression so L is exactly symmetric.
    const L = W;
    for (let i = 0; i < n; i++) {
      L[i * n + i] = 1;
      for (let j = i + 1; j < n; j++) {
        const v = -(W[i * n + j] as number) / ((scale[i] as number) * (scale[j] as number));
        L[i * n + j] = v;
        L[j * n + i] = v;
      }
    }

    // Step 3: eigenvectors of the k smallest eigenvalues (eigh sorts ascending), scaled back
    // by sqrt(degree).
    const [, eigenvectors] = eigh(tensor(L).reshape([n, n]));
    const vectors = toFloat64View(eigenvectors);
    const embedding = new Float64Array(n * k);
    for (let i = 0; i < n; i++) {
      const s = scale[i] as number;
      for (let j = 0; j < k; j++) embedding[i * k + j] = (vectors[i * n + j] as number) / s;
    }

    // Step 4: KMeans on the embedding.
    const kmeans = new KMeans({
      nClusters: this.nClusters,
      nInit: this.nInit,
      ...(this.randomState !== undefined ? { randomState: this.randomState } : {}),
    });
    kmeans.fit(tensor(embedding).reshape([n, k]));

    this.labels_ = kmeans.labels;
    // Copy: the data view may share memory with the caller's tensor. A precomputed matrix is not needed later.
    this.fitData_ = this.affinity === "precomputed" ? new Float64Array(0) : data.slice();
    this.nFeaturesIn_ = d;
    this.fitted = true;
    return this;
  }

  /** Whether every sample can reach every other through positive affinities. */
  private static isConnected(W: Float64Array, n: number): boolean {
    const seen = new Uint8Array(n);
    const stack = [0];
    seen[0] = 1;
    let reached = 1;
    while (stack.length > 0) {
      const i = stack.pop() as number;
      for (let j = 0; j < n; j++) {
        if (!seen[j] && (W[i * n + j] as number) > 0) {
          seen[j] = 1;
          reached++;
          stack.push(j);
        }
      }
    }
    return reached === n;
  }

  /** Builds the dense n x n affinity matrix (row-major, zero diagonal, symmetric). */
  private buildAffinityMatrix(data: Float64Array, n: number, d: number): Float64Array {
    const W = new Float64Array(n * n);

    if (this.affinity === "precomputed") {
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const a = data[i * n + j] as number;
          const b = data[j * n + i] as number;
          if (a < 0 || b < 0) {
            throw new DataValidationError("A precomputed affinity matrix must be non-negative");
          }
          const w = (a + b) / 2;
          W[i * n + j] = w;
          W[j * n + i] = w;
        }
        if ((data[i * n + i] as number) < 0) {
          throw new DataValidationError("A precomputed affinity matrix must be non-negative");
        }
      }
      return W;
    }

    if (this.affinity === "rbf") {
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          let distSq = 0;
          for (let f = 0; f < d; f++) {
            const diff = (data[i * d + f] as number) - (data[j * d + f] as number);
            distSq += diff * diff;
          }
          const w = Math.exp(-this.gamma * distSq);
          W[i * n + j] = w;
          W[j * n + i] = w;
        }
      }
      return W;
    }

    // nearest_neighbors: each sample counts as its own first neighbor, leaving
    // nNeighbors - 1 links to other samples. C[i][j] = 1 for a link i -> j and
    // W = (C + C^T) / 2, so a mutual link has weight 1 and a one-way link 0.5.
    const links = Math.min(this.nNeighbors - 1, n - 1);
    if (links > 0) {
      const row = new Float64Array(n);
      const scratch = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          let distSq = 0;
          for (let f = 0; f < d; f++) {
            const diff = (data[i * d + f] as number) - (data[j * d + f] as number);
            distSq += diff * diff;
          }
          row[j] = distSq;
        }
        row[i] = Infinity;
        scratch.set(row);
        const threshold = kthSmallest(scratch, links - 1);
        // Everything strictly closer than the threshold, then ties at the threshold by lowest index.
        let chosen = 0;
        for (let j = 0; j < n; j++) {
          if ((row[j] as number) < threshold) {
            W[i * n + j] = (W[i * n + j] as number) + 0.5;
            W[j * n + i] = (W[j * n + i] as number) + 0.5;
            chosen++;
          }
        }
        for (let j = 0; j < n && chosen < links; j++) {
          if (row[j] === threshold) {
            W[i * n + j] = (W[i * n + j] as number) + 0.5;
            W[j * n + i] = (W[j * n + i] as number) + 0.5;
            chosen++;
          }
        }
      }
    }
    return W;
  }

  /**
   * Predict cluster labels for new samples using nearest-neighbor assignment.
   *
   * Each new point is assigned the label of its nearest training sample (Euclidean
   * distance). With a precomputed affinity, `X` holds the affinities of the new
   * samples to every training sample, shape (n_samples, n_train), and each sample
   * gets the label of the training sample it is most similar to. Spectral clustering
   * itself has no inductive prediction rule; this is a Deepbox convenience.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Cluster labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X is not 2D or has a different number of features than the training data
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.labels_ || !this.fitData_) {
      throw new NotFittedError("SpectralClustering must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "SpectralClustering");

    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const data = toFloat64View(X);
    const trainLabels = toFloat64View(this.labels_);
    const nTrain = trainLabels.length;
    const result = new Int32Array(n);

    if (this.affinity === "precomputed") {
      for (let i = 0; i < n; i++) {
        let best = 0;
        let bestAffinity = -Infinity;
        for (let t = 0; t < nTrain; t++) {
          const a = data[i * d + t] as number;
          if (a > bestAffinity) {
            bestAffinity = a;
            best = t;
          }
        }
        result[i] = trainLabels[best] as number;
      }
      return tensor(result, { dtype: "int32" });
    }

    const train = this.fitData_;
    for (let i = 0; i < n; i++) {
      let bestDist = Infinity;
      let bestLabel = 0;
      for (let t = 0; t < nTrain; t++) {
        let dist = 0;
        for (let f = 0; f < d; f++) {
          const diff = (data[i * d + f] as number) - (train[t * d + f] as number);
          dist += diff * diff;
          if (dist >= bestDist) break;
        }
        if (dist < bestDist) {
          bestDist = dist;
          bestLabel = trainLabels[t] as number;
        }
      }
      result[i] = bestLabel;
    }
    return tensor(result, { dtype: "int32" });
  }

  /**
   * Fit the model and return the training labels.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (exists for compatibility)
   * @returns Cluster labels of shape (n_samples,)
   */
  fitPredict(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    return this.labels;
  }

  /** Training labels of shape (n_samples,). */
  get labels(): Tensor {
    if (!this.fitted || !this.labels_) {
      throw new NotFittedError("SpectralClustering must be fitted to access labels");
    }
    return this.labels_;
  }

  /**
   * Mean of the training samples of every non-empty cluster in label order, shape (n_clusters, n_features).
   *
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {InvalidParameterError} With a precomputed affinity, where samples have no features
   */
  get clusterCenters(): Tensor {
    if (!this.fitted || !this.fitData_ || !this.labels_) {
      throw new NotFittedError("SpectralClustering must be fitted to access cluster centers");
    }
    if (this.affinity === "precomputed") {
      throw new InvalidParameterError(
        "Cluster centers are not defined for a precomputed affinity matrix",
        "affinity",
        this.affinity
      );
    }
    const d = this.nFeaturesIn_;
    const labels = toFloat64View(this.labels_);
    let nClusters = 0;
    for (let i = 0; i < labels.length; i++) {
      const label = labels[i] as number;
      if (label >= nClusters) nClusters = label + 1;
    }
    const sums = new Float64Array(nClusters * d);
    const counts = new Float64Array(nClusters);
    for (let i = 0; i < labels.length; i++) {
      const label = labels[i] as number;
      counts[label] = (counts[label] as number) + 1;
      for (let f = 0; f < d; f++) {
        sums[label * d + f] =
          (sums[label * d + f] as number) + (this.fitData_[i * d + f] as number);
      }
    }
    // An empty cluster (possible only with duplicate samples) has no mean and is skipped.
    const rows: number[] = [];
    for (let c = 0; c < nClusters; c++) {
      const count = counts[c] as number;
      if (count === 0) continue;
      for (let f = 0; f < d; f++) rows.push((sums[c * d + f] as number) / count);
    }
    return tensor(new Float64Array(rows)).reshape([rows.length / d, d]);
  }

  getParams(): Record<string, unknown> {
    return {
      nClusters: this.nClusters,
      affinity: this.affinity,
      gamma: this.gamma,
      nNeighbors: this.nNeighbors,
      randomState: this.randomState,
      nInit: this.nInit,
      nComponents: this.nComponents,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nClusters":
          this.nClusters = checkNClusters(value);
          break;
        case "affinity":
          this.affinity = checkAffinity(value);
          break;
        case "gamma":
          this.gamma = checkGamma(value);
          break;
        case "nNeighbors":
          this.nNeighbors = checkNNeighbors(value);
          break;
        case "randomState":
          this.randomState = checkRandomState(value);
          break;
        case "nInit":
          this.nInit = checkNInit(value);
          break;
        case "nComponents":
          this.nComponents = checkNComponents(value);
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
