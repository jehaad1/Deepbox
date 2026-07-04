/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __random } from "../../random/random";
import { validatePredictInputs, validateUnsupervisedFitInputs } from "../_validation";
import type { Clusterer } from "../base";

/**
 * Mini-Batch K-Means clustering.
 *
 * A faster variant of KMeans that uses small random batches of data
 * to update centroids, trading a small amount of quality for significant
 * speed improvements on large datasets.
 *
 * @example
 * ```ts
 * import { MiniBatchKMeans } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]);
 * const km = new MiniBatchKMeans({ nClusters: 2, batchSize: 3 });
 * km.fit(X);
 * ```
 */
export class MiniBatchKMeans implements Clusterer {
  private nClusters: number;
  private maxIter: number;
  private batchSize: number;
  private tol: number;
  private nInit: number;
  private init: "random" | "kmeans++";
  private randomState: number | undefined;

  private clusterCenters_?: Tensor;
  private labels_?: Tensor;
  private inertia_?: number;
  private nFeaturesIn_?: number;
  private fitted = false;

  constructor(
    options: {
      readonly nClusters?: number;
      readonly maxIter?: number;
      readonly batchSize?: number;
      readonly tol?: number;
      readonly nInit?: number;
      readonly init?: "random" | "kmeans++";
      readonly randomState?: number;
    } = {}
  ) {
    this.nClusters = options.nClusters ?? 8;
    this.maxIter = options.maxIter ?? 100;
    this.batchSize = options.batchSize ?? 100;
    this.tol = options.tol ?? 0;
    this.nInit = options.nInit ?? 3;
    this.init = options.init ?? "kmeans++";
    this.randomState = options.randomState;

    if (!Number.isInteger(this.nClusters) || this.nClusters < 1) {
      throw new InvalidParameterError(
        "nClusters must be an integer >= 1",
        "nClusters",
        this.nClusters
      );
    }
    if (!Number.isInteger(this.batchSize) || this.batchSize < 1) {
      throw new InvalidParameterError(
        "batchSize must be an integer >= 1",
        "batchSize",
        this.batchSize
      );
    }
  }

  private createRNG(seed?: number): () => number {
    if (seed !== undefined) {
      let s = seed;
      return () => {
        s = (s * 9301 + 49297) % 233280;
        return s / 233280;
      };
    }
    return __random;
  }

  private initCentroids(
    data: number[][],
    nSamples: number,
    nFeatures: number,
    rng: () => number
  ): number[][] {
    if (this.init === "random") {
      const indices = new Set<number>();
      while (indices.size < this.nClusters) {
        indices.add(Math.floor(rng() * nSamples));
      }
      return [...indices].map((i) => [...data[i]!]);
    }

    // kmeans++
    const centroids: number[][] = [];
    const firstIdx = Math.floor(rng() * nSamples);
    centroids.push([...data[firstIdx]!]);

    const minDistSq = new Float64Array(nSamples).fill(Infinity);

    for (let c = 0; c < centroids.length; c++) {
      for (let i = 0; i < nSamples; i++) {
        let d = 0;
        for (let j = 0; j < nFeatures; j++) {
          const diff = (data[i]![j] ?? 0) - (centroids[c]![j] ?? 0);
          d += diff * diff;
        }
        if (d < (minDistSq[i] ?? Infinity)) minDistSq[i] = d;
      }
    }

    while (centroids.length < this.nClusters) {
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
      const next = [...data[nextIdx]!];
      centroids.push(next);

      for (let i = 0; i < nSamples; i++) {
        let d = 0;
        for (let j = 0; j < nFeatures; j++) {
          const diff = (data[i]![j] ?? 0) - (next[j] ?? 0);
          d += diff * diff;
        }
        if (d < (minDistSq[i] ?? Infinity)) minDistSq[i] = d;
      }
    }

    return centroids;
  }

  private assignPoint(point: number[], centroids: number[][], nFeatures: number): number {
    let bestK = 0;
    let bestDist = Infinity;
    for (let k = 0; k < centroids.length; k++) {
      let d = 0;
      for (let j = 0; j < nFeatures; j++) {
        const diff = (point[j] ?? 0) - (centroids[k]![j] ?? 0);
        d += diff * diff;
      }
      if (d < bestDist) {
        bestDist = d;
        bestK = k;
      }
    }
    return bestK;
  }

  private computeInertia(
    data: number[][],
    centroids: number[][],
    nSamples: number,
    nFeatures: number
  ): number {
    let inertia = 0;
    for (let i = 0; i < nSamples; i++) {
      const k = this.assignPoint(data[i]!, centroids, nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        const diff = (data[i]![j] ?? 0) - (centroids[k]![j] ?? 0);
        inertia += diff * diff;
      }
    }
    return inertia;
  }

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

    const data: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        row.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      data.push(row);
    }

    const effectiveBatch = Math.min(this.batchSize, nSamples);
    let bestCentroids: number[][] | undefined;
    let bestInertia = Infinity;

    const baseSeed = this.randomState;

    for (let run = 0; run < this.nInit; run++) {
      const rng = this.createRNG(baseSeed !== undefined ? baseSeed + run * 7919 : undefined);
      const centroids = this.initCentroids(data, nSamples, nFeatures, rng);
      const counts = new Array<number>(this.nClusters).fill(0);

      for (let iter = 0; iter < this.maxIter; iter++) {
        // Sample mini-batch
        const batchIdx: number[] = [];
        for (let b = 0; b < effectiveBatch; b++) {
          batchIdx.push(Math.floor(rng() * nSamples));
        }

        // Assign and update
        let maxShift = 0;
        for (const idx of batchIdx) {
          const k = this.assignPoint(data[idx]!, centroids, nFeatures);
          counts[k] = (counts[k] ?? 0) + 1;
          const lr = 1 / (counts[k] ?? 1);
          for (let j = 0; j < nFeatures; j++) {
            const old = centroids[k]![j] ?? 0;
            const newVal = old + lr * ((data[idx]![j] ?? 0) - old);
            centroids[k]![j] = newVal;
            const shift = Math.abs(newVal - old);
            if (shift > maxShift) maxShift = shift;
          }
        }

        if (this.tol > 0 && maxShift < this.tol) break;
      }

      const inertia = this.computeInertia(data, centroids, nSamples, nFeatures);
      if (inertia < bestInertia) {
        bestInertia = inertia;
        bestCentroids = centroids;
      }
    }

    // Assign final labels
    const labels: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      labels.push(this.assignPoint(data[i]!, bestCentroids!, nFeatures));
    }

    this.clusterCenters_ = tensor(bestCentroids!);
    this.labels_ = tensor(labels, { dtype: "int32" });
    this.inertia_ = bestInertia;
    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.clusterCenters_) {
      throw new NotFittedError("MiniBatchKMeans must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "MiniBatchKMeans");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const labels: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      let bestK = 0;
      let bestDist = Infinity;
      for (let k = 0; k < this.nClusters; k++) {
        let d = 0;
        for (let j = 0; j < nFeatures; j++) {
          const diff =
            Number(X.data[X.offset + i * nFeatures + j]) -
            Number(this.clusterCenters_.data[this.clusterCenters_.offset + k * nFeatures + j]);
          d += diff * diff;
        }
        if (d < bestDist) {
          bestDist = d;
          bestK = k;
        }
      }
      labels.push(bestK);
    }

    return tensor(labels, { dtype: "int32" });
  }

  fitPredict(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    return this.labels_!;
  }

  get clusterCenters(): Tensor {
    if (!this.fitted || !this.clusterCenters_) {
      throw new NotFittedError("MiniBatchKMeans must be fitted to access cluster centers");
    }
    return this.clusterCenters_;
  }

  get labels(): Tensor {
    if (!this.fitted || !this.labels_) {
      throw new NotFittedError("MiniBatchKMeans must be fitted to access labels");
    }
    return this.labels_;
  }

  get inertia(): number {
    if (!this.fitted || this.inertia_ === undefined) {
      throw new NotFittedError("MiniBatchKMeans must be fitted to access inertia");
    }
    return this.inertia_;
  }

  getParams(): Record<string, unknown> {
    return {
      nClusters: this.nClusters,
      maxIter: this.maxIter,
      batchSize: this.batchSize,
      tol: this.tol,
      nInit: this.nInit,
      init: this.init,
      randomState: this.randomState,
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
        case "maxIter":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("maxIter must be an integer >= 1", "maxIter", value);
          }
          this.maxIter = value;
          break;
        case "batchSize":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError(
              "batchSize must be an integer >= 1",
              "batchSize",
              value
            );
          }
          this.batchSize = value;
          break;
        case "tol":
          if (typeof value !== "number" || value < 0) {
            throw new InvalidParameterError("tol must be >= 0", "tol", value);
          }
          this.tol = value;
          break;
        case "nInit":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("nInit must be an integer >= 1", "nInit", value);
          }
          this.nInit = value;
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
          this.randomState = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
