/**
 * OPTICS — Ordering Points To Identify the Clustering Structure.
 *
 * A density-based clustering algorithm related to DBSCAN that creates
 * a reachability ordering of the data points. Unlike DBSCAN, OPTICS
 * can detect clusters of varying density.
 *
 * @module ml/clustering/OPTICS
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { validateUnsupervisedFitInputs } from "../_validation";
import type { Clusterer } from "../base";

/**
 * OPTICS clustering algorithm.
 *
 * Produces a reachability plot and extracts clusters using a threshold (xi)
 * or a simple epsilon cut.
 *
 * @example
 * ```ts
 * import { OPTICS } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0], [0.1, 0], [10, 10], [10.1, 10]]);
 * const optics = new OPTICS({ minSamples: 2 });
 * optics.fit(X);
 * console.log(optics.labels);
 * console.log(optics.reachability);
 * ```
 */
export class OPTICS implements Clusterer {
  private minSamples: number;
  private maxEps: number;
  private clusterMethod: "xi" | "dbscan";
  private xi: number;
  private eps: number | undefined;

  private labels_?: Tensor;
  private reachability_?: Float64Array;
  private ordering_?: Int32Array;
  private coreDistances_?: Float64Array;
  private fitData_?: number[][];
  private fitted = false;

  constructor(
    options: {
      readonly minSamples?: number;
      readonly maxEps?: number;
      readonly clusterMethod?: "xi" | "dbscan";
      readonly xi?: number;
      readonly eps?: number;
    } = {}
  ) {
    this.minSamples = options.minSamples ?? 5;
    this.maxEps = options.maxEps ?? Infinity;
    this.clusterMethod = options.clusterMethod ?? "dbscan";
    this.xi = options.xi ?? 0.05;
    this.eps = options.eps;

    if (!Number.isInteger(this.minSamples) || this.minSamples < 1) {
      throw new InvalidParameterError(
        "minSamples must be a positive integer",
        "minSamples",
        this.minSamples
      );
    }
    if (this.maxEps <= 0) {
      throw new InvalidParameterError("maxEps must be > 0", "maxEps", this.maxEps);
    }
  }

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

    // Precompute pairwise distances
    const dist = new Float64Array(nSamples * nSamples);
    for (let i = 0; i < nSamples; i++) {
      for (let j = i + 1; j < nSamples; j++) {
        let d = 0;
        for (let f = 0; f < nFeatures; f++) {
          const diff = (data[i]![f] ?? 0) - (data[j]![f] ?? 0);
          d += diff * diff;
        }
        d = Math.sqrt(d);
        dist[i * nSamples + j] = d;
        dist[j * nSamples + i] = d;
      }
    }

    // Compute core distances
    this.coreDistances_ = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      // Collect distances to all other points, sort, take minSamples-th
      const dists: number[] = [];
      for (let j = 0; j < nSamples; j++) {
        if (i === j) continue;
        const d = dist[i * nSamples + j] ?? 0;
        if (d <= this.maxEps) {
          dists.push(d);
        }
      }
      dists.sort((a, b) => a - b);
      if (dists.length >= this.minSamples - 1) {
        this.coreDistances_[i] = dists[this.minSamples - 2] ?? Infinity;
      } else {
        this.coreDistances_[i] = Infinity;
      }
    }

    // OPTICS ordering
    const processed = new Uint8Array(nSamples);
    this.reachability_ = new Float64Array(nSamples).fill(Infinity);
    this.ordering_ = new Int32Array(nSamples);
    let orderIdx = 0;

    for (let i = 0; i < nSamples; i++) {
      if (processed[i]) continue;

      // Use a priority queue (simple sorted array for correctness)
      processed[i] = 1;
      this.ordering_[orderIdx++] = i;

      if ((this.coreDistances_[i] ?? Infinity) >= Infinity) continue;

      // Seeds: priority queue of (reachability, index)
      const seeds = new Map<number, number>();
      this.updateSeeds(i, seeds, processed, dist, nSamples);

      while (seeds.size > 0) {
        // Find seed with smallest reachability
        let bestIdx = -1;
        let bestReach = Infinity;
        for (const [idx, reach] of seeds) {
          if (reach < bestReach) {
            bestReach = reach;
            bestIdx = idx;
          }
        }
        if (bestIdx === -1) break;

        seeds.delete(bestIdx);
        processed[bestIdx] = 1;
        this.reachability_[bestIdx] = bestReach;
        this.ordering_[orderIdx++] = bestIdx;

        if ((this.coreDistances_[bestIdx] ?? Infinity) < Infinity) {
          this.updateSeeds(bestIdx, seeds, processed, dist, nSamples);
        }
      }
    }

    // Store training data for nearest-neighbor predict
    this.fitData_ = data;

    // Extract clusters
    if (this.clusterMethod === "dbscan") {
      this.extractDBSCANClusters(nSamples);
    } else {
      this.extractXiClusters(nSamples);
    }

    this.fitted = true;
    return this;
  }

  private updateSeeds(
    pointIdx: number,
    seeds: Map<number, number>,
    processed: Uint8Array,
    dist: Float64Array,
    nSamples: number
  ): void {
    const coreDist = this.coreDistances_![pointIdx] ?? Infinity;

    for (let j = 0; j < nSamples; j++) {
      if (processed[j]) continue;
      const d = dist[pointIdx * nSamples + j] ?? Infinity;
      if (d > this.maxEps) continue;

      const reachDist = Math.max(coreDist, d);

      const existing = seeds.get(j);
      if (existing === undefined) {
        seeds.set(j, reachDist);
      } else if (reachDist < existing) {
        seeds.set(j, reachDist);
      }
    }
  }

  private extractDBSCANClusters(nSamples: number): void {
    // Use eps (or maxEps as fallback) to cut the reachability plot
    const cutEps =
      this.eps ?? (this.maxEps === Infinity ? this.estimateEps(nSamples) : this.maxEps);
    const labels = new Int32Array(nSamples).fill(-1);
    let currentCluster = -1;

    for (let i = 0; i < nSamples; i++) {
      const idx = this.ordering_![i] ?? 0;
      const reach = this.reachability_![idx] ?? Infinity;

      if (reach > cutEps) {
        // Check if this point is a core point that starts a new cluster
        if ((this.coreDistances_![idx] ?? Infinity) <= cutEps) {
          currentCluster++;
          labels[idx] = currentCluster;
        } else {
          labels[idx] = -1; // noise
        }
      } else {
        labels[idx] = currentCluster;
      }
    }

    this.labels_ = tensor(Array.from(labels), { dtype: "int32" });
  }

  private extractXiClusters(nSamples: number): void {
    // Simple xi-based extraction: look for steep downward/upward regions
    const labels = new Int32Array(nSamples).fill(-1);
    let currentCluster = -1;
    let inCluster = false;

    for (let i = 1; i < nSamples; i++) {
      const idx = this.ordering_![i] ?? 0;
      const prevIdx = this.ordering_![i - 1] ?? 0;
      const reach = this.reachability_![idx] ?? Infinity;
      const prevReach = this.reachability_![prevIdx] ?? Infinity;

      if (reach < Infinity && prevReach < Infinity) {
        const ratio = prevReach > 0 ? 1 - reach / prevReach : 0;
        if (ratio >= this.xi && !inCluster) {
          // Steep downward: start new cluster
          currentCluster++;
          inCluster = true;
        } else if (ratio <= -this.xi && inCluster) {
          // Steep upward: end cluster
          inCluster = false;
        }
      }

      if (inCluster && reach < Infinity) {
        labels[idx] = currentCluster;
      }
    }

    this.labels_ = tensor(Array.from(labels), { dtype: "int32" });
  }

  private estimateEps(nSamples: number): number {
    // Estimate eps from the reachability plot: use the median non-infinite value
    const reachVals: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      const r = this.reachability_![i] ?? Infinity;
      if (r < Infinity) reachVals.push(r);
    }
    if (reachVals.length === 0) return 1;
    reachVals.sort((a, b) => a - b);
    return reachVals[Math.floor(reachVals.length * 0.75)] ?? 1;
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
      throw new NotFittedError("OPTICS must be fitted before prediction");
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
      let bestLabel = -1;
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
      throw new NotFittedError("OPTICS must be fitted to access labels");
    }
    return this.labels_;
  }

  get reachability(): Float64Array {
    if (!this.fitted || !this.reachability_) {
      throw new NotFittedError("OPTICS must be fitted to access reachability");
    }
    return this.reachability_;
  }

  get ordering(): Int32Array {
    if (!this.fitted || !this.ordering_) {
      throw new NotFittedError("OPTICS must be fitted to access ordering");
    }
    return this.ordering_;
  }

  get coreDistances(): Float64Array {
    if (!this.fitted || !this.coreDistances_) {
      throw new NotFittedError("OPTICS must be fitted to access core distances");
    }
    return this.coreDistances_;
  }

  get clusterCenters(): Tensor {
    if (!this.fitted || !this.fitData_ || !this.labels_) {
      throw new NotFittedError("OPTICS must be fitted to access cluster centers");
    }
    const nFeatures = this.fitData_[0]?.length ?? 0;
    const clusterMap = new Map<number, number[][]>();
    for (let i = 0; i < this.fitData_.length; i++) {
      const label = Number(this.labels_.data[this.labels_.offset + i]);
      if (label < 0) continue; // skip noise
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
      minSamples: this.minSamples,
      maxEps: this.maxEps,
      clusterMethod: this.clusterMethod,
      xi: this.xi,
      eps: this.eps,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
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
        case "maxEps":
          if (typeof value !== "number" || value <= 0) {
            throw new InvalidParameterError("maxEps must be > 0", "maxEps", value);
          }
          this.maxEps = value;
          break;
        case "clusterMethod":
          if (value !== "xi" && value !== "dbscan") {
            throw new InvalidParameterError(
              `clusterMethod must be "xi" or "dbscan"`,
              "clusterMethod",
              value
            );
          }
          this.clusterMethod = value;
          break;
        case "xi":
          if (typeof value !== "number" || value <= 0 || value >= 1) {
            throw new InvalidParameterError("xi must be in (0, 1)", "xi", value);
          }
          this.xi = value;
          break;
        case "eps":
          if (value !== undefined && (typeof value !== "number" || value <= 0)) {
            throw new InvalidParameterError(
              "eps must be a positive number or undefined",
              "eps",
              value
            );
          }
          this.eps = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
