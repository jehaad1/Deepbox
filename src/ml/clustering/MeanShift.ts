/**
 * Mean Shift Clustering.
 *
 * A centroid-based algorithm that iteratively shifts each point towards
 * the mode (densest area) of a kernel density estimate. Does not require
 * specifying the number of clusters in advance.
 *
 * **Algorithm**:
 * 1. For each data point, compute a weighted mean of points within bandwidth
 * 2. Shift the point to that weighted mean
 * 3. Repeat until convergence
 * 4. Merge converged points that are within bandwidth of each other
 *
 * @module ml/clustering/MeanShift
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { validatePredictInputs, validateUnsupervisedFitInputs } from "../_validation";
import type { Clusterer } from "../base";

/**
 * Mean Shift clustering using a flat/Gaussian kernel.
 *
 * Automatically determines the number of clusters based on the data
 * density landscape. The `bandwidth` parameter controls the kernel size.
 *
 * @example
 * ```ts
 * import { MeanShift } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0], [0.5, 0], [10, 10], [10.5, 10]]);
 * const ms = new MeanShift({ bandwidth: 2 });
 * ms.fit(X);
 * console.log(ms.labels);
 * console.log(ms.clusterCenters);
 * ```
 */
export class MeanShift implements Clusterer {
  private bandwidth: number | "auto";
  private maxIter: number;
  private tol: number;
  private binSeeding: boolean;

  private clusterCenters_?: Tensor;
  private labels_?: Tensor;
  private nFeaturesIn_ = 0;
  private fitted = false;

  constructor(
    options: {
      readonly bandwidth?: number | "auto";
      readonly maxIter?: number;
      readonly tol?: number;
      readonly binSeeding?: boolean;
    } = {}
  ) {
    this.bandwidth = options.bandwidth ?? "auto";
    this.maxIter = options.maxIter ?? 300;
    this.tol = options.tol ?? 1e-3;
    this.binSeeding = options.binSeeding ?? false;

    if (typeof this.bandwidth === "number" && this.bandwidth <= 0) {
      throw new InvalidParameterError("bandwidth must be > 0", "bandwidth", this.bandwidth);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new InvalidParameterError(
        "maxIter must be a positive integer",
        "maxIter",
        this.maxIter
      );
    }
  }

  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nFeatures;

    // Extract data
    const data: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        row.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      data.push(row);
    }

    // Estimate bandwidth if auto
    const bw = this.estimateBandwidth(data, nSamples, nFeatures);

    // Initialize seeds (all data points or bin seeds)
    let seeds: number[][];
    if (this.binSeeding) {
      seeds = this.getBinSeeds(data, nSamples, nFeatures, bw);
    } else {
      seeds = data.map((row) => [...row]);
    }

    // Mean shift each seed
    const convergedSeeds: number[][] = [];
    const bwSq = bw * bw;

    for (const seed of seeds) {
      const current = [...seed];

      for (let iter = 0; iter < this.maxIter; iter++) {
        // Compute weighted mean within bandwidth
        const newMean = new Array<number>(nFeatures).fill(0);
        let totalWeight = 0;

        for (let i = 0; i < nSamples; i++) {
          let distSq = 0;
          for (let f = 0; f < nFeatures; f++) {
            const diff = (data[i]![f] ?? 0) - (current[f] ?? 0);
            distSq += diff * diff;
          }
          if (distSq <= bwSq) {
            // Flat kernel: weight = 1 within bandwidth
            totalWeight++;
            for (let f = 0; f < nFeatures; f++) {
              newMean[f] = (newMean[f] ?? 0) + (data[i]![f] ?? 0);
            }
          }
        }

        if (totalWeight === 0) break;

        for (let f = 0; f < nFeatures; f++) {
          newMean[f] = (newMean[f] ?? 0) / totalWeight;
        }

        // Check convergence
        let shift = 0;
        for (let f = 0; f < nFeatures; f++) {
          const diff = (newMean[f] ?? 0) - (current[f] ?? 0);
          shift += diff * diff;
        }

        for (let f = 0; f < nFeatures; f++) {
          current[f] = newMean[f] ?? 0;
        }

        if (Math.sqrt(shift) < this.tol) break;
      }

      convergedSeeds.push(current);
    }

    // Merge close centroids
    const mergedCenters: number[][] = [];
    const used = new Uint8Array(convergedSeeds.length);

    for (let i = 0; i < convergedSeeds.length; i++) {
      if (used[i]) continue;
      const group: number[][] = [convergedSeeds[i]!];
      used[i] = 1;

      for (let j = i + 1; j < convergedSeeds.length; j++) {
        if (used[j]) continue;
        let distSq = 0;
        for (let f = 0; f < nFeatures; f++) {
          const diff = (convergedSeeds[i]![f] ?? 0) - (convergedSeeds[j]![f] ?? 0);
          distSq += diff * diff;
        }
        if (distSq <= bwSq) {
          group.push(convergedSeeds[j]!);
          used[j] = 1;
        }
      }

      // Average the group
      const center = new Array<number>(nFeatures).fill(0);
      for (const pt of group) {
        for (let f = 0; f < nFeatures; f++) {
          center[f] = (center[f] ?? 0) + (pt[f] ?? 0);
        }
      }
      for (let f = 0; f < nFeatures; f++) {
        center[f] = (center[f] ?? 0) / group.length;
      }
      mergedCenters.push(center);
    }

    // Assign labels: each sample to nearest merged center
    const labels: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      let bestK = 0;
      let bestDist = Infinity;
      for (let k = 0; k < mergedCenters.length; k++) {
        let d = 0;
        for (let f = 0; f < nFeatures; f++) {
          const diff = (data[i]![f] ?? 0) - (mergedCenters[k]![f] ?? 0);
          d += diff * diff;
        }
        if (d < bestDist) {
          bestDist = d;
          bestK = k;
        }
      }
      labels.push(bestK);
    }

    this.clusterCenters_ = tensor(mergedCenters);
    this.labels_ = tensor(labels, { dtype: "int32" });
    this.fitted = true;
    return this;
  }

  private estimateBandwidth(data: number[][], nSamples: number, nFeatures: number): number {
    if (typeof this.bandwidth === "number") return this.bandwidth;

    // Estimate bandwidth using median of pairwise distances / quantile approach
    // Use a simpler approach: mean distance to k-th nearest neighbor
    const k = Math.max(1, Math.floor(nSamples * 0.3));
    let totalDist = 0;

    for (let i = 0; i < nSamples; i++) {
      const dists: number[] = [];
      for (let j = 0; j < nSamples; j++) {
        if (i === j) continue;
        let d = 0;
        for (let f = 0; f < nFeatures; f++) {
          const diff = (data[i]![f] ?? 0) - (data[j]![f] ?? 0);
          d += diff * diff;
        }
        dists.push(Math.sqrt(d));
      }
      dists.sort((a, b) => a - b);
      totalDist += dists[Math.min(k - 1, dists.length - 1)] ?? 0;
    }

    return Math.max(totalDist / nSamples, 1e-10);
  }

  private getBinSeeds(
    data: number[][],
    nSamples: number,
    nFeatures: number,
    bw: number
  ): number[][] {
    // Discretize data into bins of size bandwidth, use bin centers as seeds
    const bins = new Map<string, { sum: number[]; count: number }>();
    for (let i = 0; i < nSamples; i++) {
      const key: number[] = [];
      for (let f = 0; f < nFeatures; f++) {
        key.push(Math.round((data[i]![f] ?? 0) / bw));
      }
      const keyStr = key.join(",");
      const existing = bins.get(keyStr);
      if (existing) {
        for (let f = 0; f < nFeatures; f++) {
          existing.sum[f] = (existing.sum[f] ?? 0) + (data[i]![f] ?? 0);
        }
        existing.count++;
      } else {
        bins.set(keyStr, {
          sum: data[i]!.slice(),
          count: 1,
        });
      }
    }

    const seeds: number[][] = [];
    for (const { sum, count } of bins.values()) {
      seeds.push(sum.map((s) => s / count));
    }
    return seeds;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.clusterCenters_) {
      throw new NotFittedError("MeanShift must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "MeanShift");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const centers = this.clusterCenters_;
    const nClusters = centers.shape[0] ?? 0;
    const labels: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      let bestK = 0;
      let bestDist = Infinity;
      for (let k = 0; k < nClusters; k++) {
        let d = 0;
        for (let f = 0; f < nFeatures; f++) {
          const diff =
            Number(X.data[X.offset + i * nFeatures + f]) -
            Number(centers.data[centers.offset + k * nFeatures + f]);
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

  get labels(): Tensor {
    if (!this.fitted || !this.labels_) {
      throw new NotFittedError("MeanShift must be fitted to access labels");
    }
    return this.labels_;
  }

  get clusterCenters(): Tensor {
    if (!this.fitted || !this.clusterCenters_) {
      throw new NotFittedError("MeanShift must be fitted to access cluster centers");
    }
    return this.clusterCenters_;
  }

  getParams(): Record<string, unknown> {
    return {
      bandwidth: this.bandwidth,
      maxIter: this.maxIter,
      tol: this.tol,
      binSeeding: this.binSeeding,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "bandwidth":
          if (value !== "auto" && (typeof value !== "number" || value <= 0)) {
            throw new InvalidParameterError(
              'bandwidth must be "auto" or a positive number',
              "bandwidth",
              value
            );
          }
          this.bandwidth = value;
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
        case "binSeeding":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("binSeeding must be a boolean", "binSeeding", value);
          }
          this.binSeeding = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
