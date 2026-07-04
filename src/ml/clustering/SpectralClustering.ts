/**
 * Spectral Clustering.
 *
 * Performs clustering by embedding the data in the eigenspace of the
 * graph Laplacian, then applying KMeans in that space.
 *
 * **Algorithm**:
 * 1. Build affinity matrix (RBF kernel or nearest neighbors)
 * 2. Compute normalized graph Laplacian
 * 3. Find the k smallest eigenvectors of the Laplacian
 * 4. Cluster rows of the eigenvector matrix using KMeans
 *
 * @module ml/clustering/SpectralClustering
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../../core";
import { eigh } from "../../linalg";
import { type Tensor, tensor } from "../../ndarray";
import { validateUnsupervisedFitInputs } from "../_validation";
import type { Clusterer } from "../base";
import { KMeans } from "./KMeans";

type Affinity = "rbf" | "nearest_neighbors";

/**
 * Spectral Clustering using graph Laplacian eigenvectors.
 *
 * Particularly effective for non-convex clusters and clusters with
 * complex shapes that KMeans cannot handle well.
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

  private labels_?: Tensor;
  private fitData_?: number[][];
  private fitted = false;

  constructor(
    options: {
      readonly nClusters?: number;
      readonly affinity?: Affinity;
      readonly gamma?: number;
      readonly nNeighbors?: number;
      readonly randomState?: number;
      readonly nInit?: number;
    } = {}
  ) {
    this.nClusters = options.nClusters ?? 8;
    this.affinity = options.affinity ?? "rbf";
    this.gamma = options.gamma ?? 1.0;
    this.nNeighbors = options.nNeighbors ?? 10;
    if (options.randomState !== undefined) this.randomState = options.randomState;
    this.nInit = options.nInit ?? 10;

    if (!Number.isInteger(this.nClusters) || this.nClusters < 1) {
      throw new InvalidParameterError(
        "nClusters must be an integer >= 1",
        "nClusters",
        this.nClusters
      );
    }
    if (this.gamma <= 0) {
      throw new InvalidParameterError("gamma must be > 0", "gamma", this.gamma);
    }
  }

  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    if (nSamples < this.nClusters) {
      throw new InvalidParameterError(
        `n_samples=${nSamples} should be >= n_clusters=${this.nClusters}`,
        "nClusters",
        this.nClusters
      );
    }

    // Extract data
    const data: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        row.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      data.push(row);
    }

    // Step 1: Build affinity matrix
    const W = this.buildAffinityMatrix(data, nSamples, nFeatures);

    // Step 2: Compute normalized Laplacian (symmetric normalization)
    // D = diag(W * 1), D^{-1/2}, L_sym = I - D^{-1/2} W D^{-1/2}
    const D_inv_sqrt = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      let sum = 0;
      for (let j = 0; j < nSamples; j++) {
        sum += W[i * nSamples + j] ?? 0;
      }
      D_inv_sqrt[i] = sum > 0 ? 1 / Math.sqrt(sum) : 0;
    }

    // L_sym = I - D^{-1/2} W D^{-1/2}
    const L = new Float64Array(nSamples * nSamples);
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nSamples; j++) {
        const val = (D_inv_sqrt[i] ?? 0) * (W[i * nSamples + j] ?? 0) * (D_inv_sqrt[j] ?? 0);
        L[i * nSamples + j] = i === j ? 1 - val : -val;
      }
    }

    // Step 3: Compute eigenvectors corresponding to smallest eigenvalues
    const Ltensor = tensor(Array.from(L)).reshape([nSamples, nSamples]);
    const [, eigenvectors] = eigh(Ltensor);

    // eigenvalues are sorted ascending, take first nClusters eigenvectors
    const k = this.nClusters;
    const embedding: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let j = 0; j < k; j++) {
        row.push(Number(eigenvectors.data[eigenvectors.offset + i * nSamples + j]));
      }
      embedding.push(row);
    }

    // Normalize rows of embedding to unit length
    for (let i = 0; i < nSamples; i++) {
      let norm = 0;
      for (let j = 0; j < k; j++) {
        norm += (embedding[i]![j] ?? 0) ** 2;
      }
      norm = Math.sqrt(norm);
      if (norm > 0) {
        for (let j = 0; j < k; j++) {
          embedding[i]![j] = (embedding[i]![j] ?? 0) / norm;
        }
      }
    }

    // Step 4: Cluster the embedding using KMeans
    const embeddingTensor = tensor(embedding.flat()).reshape([nSamples, k]);
    const kmeansOpts: Record<string, unknown> = {
      nClusters: this.nClusters,
      nInit: this.nInit,
    };
    if (this.randomState !== undefined) kmeansOpts["randomState"] = this.randomState;
    const kmeans = new KMeans(
      kmeansOpts as Parameters<typeof KMeans.prototype.fit>[0] extends Tensor
        ? never
        : ConstructorParameters<typeof KMeans>[0]
    );
    kmeans.fit(embeddingTensor);

    this.labels_ = kmeans.labels;
    this.fitData_ = data;
    this.fitted = true;
    return this;
  }

  private buildAffinityMatrix(data: number[][], nSamples: number, nFeatures: number): Float64Array {
    const W = new Float64Array(nSamples * nSamples);

    if (this.affinity === "rbf") {
      for (let i = 0; i < nSamples; i++) {
        for (let j = i + 1; j < nSamples; j++) {
          let distSq = 0;
          for (let f = 0; f < nFeatures; f++) {
            const diff = (data[i]![f] ?? 0) - (data[j]![f] ?? 0);
            distSq += diff * diff;
          }
          const w = Math.exp(-this.gamma * distSq);
          W[i * nSamples + j] = w;
          W[j * nSamples + i] = w;
        }
      }
    } else {
      // nearest_neighbors: build KNN graph, then symmetrize
      // For each point, find k nearest neighbors
      const k = Math.min(this.nNeighbors, nSamples - 1);
      for (let i = 0; i < nSamples; i++) {
        // Compute distances to all other points
        const dists: Array<{ idx: number; dist: number }> = [];
        for (let j = 0; j < nSamples; j++) {
          if (i === j) continue;
          let d = 0;
          for (let f = 0; f < nFeatures; f++) {
            const diff = (data[i]![f] ?? 0) - (data[j]![f] ?? 0);
            d += diff * diff;
          }
          dists.push({ idx: j, dist: d });
        }
        dists.sort((a, b) => a.dist - b.dist);
        // Connect to k nearest
        for (let n = 0; n < k; n++) {
          const j = dists[n]!.idx;
          W[i * nSamples + j] = 1;
          W[j * nSamples + i] = 1; // symmetrize
        }
      }
    }

    return W;
  }

  /**
   * Predict cluster labels for new samples using nearest-neighbor assignment.
   *
   * Each new point is assigned the label of its nearest training sample.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Cluster labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.labels_ || !this.fitData_) {
      throw new NotFittedError("SpectralClustering must be fitted before prediction");
    }

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const trainData = this.fitData_;
    const trainLabels = this.labels_;
    const nTrain = trainData.length;
    const result: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        row.push(Number(X.data[X.offset + i * nFeatures + j]));
      }

      let bestDist = Infinity;
      let bestLabel = 0;
      for (let t = 0; t < nTrain; t++) {
        const trainRow = trainData[t];
        if (!trainRow) continue;
        let dist = 0;
        for (let f = 0; f < nFeatures; f++) {
          const diff = (row[f] ?? 0) - (trainRow[f] ?? 0);
          dist += diff * diff;
        }
        if (dist < bestDist) {
          bestDist = dist;
          bestLabel = Number(trainLabels.data[trainLabels.offset + t]);
        }
      }
      result.push(bestLabel);
    }

    return tensor(result, { dtype: "int32" });
  }

  fitPredict(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    return this.labels_!;
  }

  get labels(): Tensor {
    if (!this.fitted || !this.labels_) {
      throw new NotFittedError("SpectralClustering must be fitted to access labels");
    }
    return this.labels_;
  }

  get clusterCenters(): Tensor {
    if (!this.fitted || !this.fitData_ || !this.labels_) {
      throw new NotFittedError("SpectralClustering must be fitted to access cluster centers");
    }
    const nFeatures = this.fitData_[0]?.length ?? 0;
    const clusterMap = new Map<number, number[][]>();
    for (let i = 0; i < this.fitData_.length; i++) {
      const label = Number(this.labels_.data[this.labels_.offset + i]);
      const row = this.fitData_[i];
      if (row) {
        const existing = clusterMap.get(label);
        if (existing) {
          existing.push(row);
        } else {
          clusterMap.set(label, [row]);
        }
      }
    }
    const sortedLabels = [...clusterMap.keys()].sort((a, b) => a - b);
    const centers: number[][] = [];
    for (const label of sortedLabels) {
      const members = clusterMap.get(label);
      if (!members || members.length === 0) continue;
      const centroid = new Array<number>(nFeatures).fill(0);
      for (const m of members) {
        for (let f = 0; f < nFeatures; f++) {
          centroid[f] = (centroid[f] ?? 0) + (m[f] ?? 0);
        }
      }
      for (let f = 0; f < nFeatures; f++) {
        centroid[f] = (centroid[f] ?? 0) / members.length;
      }
      centers.push(centroid);
    }
    return tensor(centers);
  }

  getParams(): Record<string, unknown> {
    return {
      nClusters: this.nClusters,
      affinity: this.affinity,
      gamma: this.gamma,
      nNeighbors: this.nNeighbors,
      randomState: this.randomState,
      nInit: this.nInit,
    };
  }

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
        case "affinity":
          if (value !== "rbf" && value !== "nearest_neighbors") {
            throw new InvalidParameterError(
              `affinity must be "rbf" or "nearest_neighbors"`,
              "affinity",
              value
            );
          }
          this.affinity = value;
          break;
        case "gamma":
          if (typeof value !== "number" || value <= 0) {
            throw new InvalidParameterError("gamma must be > 0", "gamma", value);
          }
          this.gamma = value;
          break;
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
        case "randomState":
          if (value !== undefined && (typeof value !== "number" || !Number.isFinite(value))) {
            throw new InvalidParameterError(
              "randomState must be a finite number",
              "randomState",
              value
            );
          }
          this.randomState = value;
          break;
        case "nInit":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("nInit must be an integer >= 1", "nInit", value);
          }
          this.nInit = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
