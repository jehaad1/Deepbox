/**
 * Mean Shift Clustering.
 *
 * A centroid-based algorithm that iteratively shifts each seed towards
 * the mode (densest area) of a kernel density estimate. Does not require
 * specifying the number of clusters in advance.
 *
 * **Algorithm**:
 * 1. For each seed, compute the mean of the samples within `bandwidth` (flat kernel)
 * 2. Move the seed to that mean
 * 3. Repeat until the shift is at most `tol * bandwidth` or `maxIter` is reached
 * 4. Sort the converged modes by the number of samples they cover and drop every
 *    mode that lies within `bandwidth` of a stronger one
 * 5. Label each sample with its nearest remaining mode
 *
 * @module ml/clustering/MeanShift
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError, warn } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { kthSmallest } from "../_internal";
import {
  toFloat64View,
  validatePredictInputs,
  validateUnsupervisedFitInputs,
} from "../_validation";
import type { Clusterer } from "../base";

/** Fraction of the samples used as neighborhood size when the bandwidth is estimated. */
const BANDWIDTH_QUANTILE = 0.3;

/** Rounds half to even, like `numpy.round`. */
function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** A converged mode together with the statistics used to rank it. */
type Mode = {
  readonly center: Float64Array;
  /** Number of samples within the bandwidth of the mode during the last iteration. */
  readonly intensity: number;
};

function checkBandwidth(value: unknown): number | "auto" {
  if (value === "auto") return "auto";
  if (typeof value !== "number" || !(value > 0)) {
    throw new InvalidParameterError(
      'bandwidth must be "auto" or a positive number',
      "bandwidth",
      value
    );
  }
  return value;
}

function checkMaxIter(value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new InvalidParameterError("maxIter must be an integer >= 1", "maxIter", value);
  }
  return value;
}

function checkTol(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new InvalidParameterError("tol must be a finite number >= 0", "tol", value);
  }
  return value;
}

function checkBoolean(name: string, value: unknown): boolean {
  if (typeof value !== "boolean") {
    throw new InvalidParameterError(`${name} must be a boolean`, name, value);
  }
  return value;
}

function checkMinBinFreq(value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new InvalidParameterError("minBinFreq must be an integer >= 1", "minBinFreq", value);
  }
  return value;
}

/**
 * Mean Shift clustering with a flat kernel.
 *
 * Automatically determines the number of clusters from the density landscape of the
 * data. The `bandwidth` parameter is the radius of the kernel. With `bandwidth: "auto"`
 * it is estimated from the data the same way as `sklearn.cluster.estimate_bandwidth`
 * with `quantile=0.3`.
 *
 * Clusters are ordered by the number of samples they attract (strongest mode first),
 * so label `0` is always the densest mode.
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
  private minBinFreq: number;
  private clusterAll: boolean;

  private clusterCenters_?: Tensor;
  private labels_?: Tensor;
  private nIter_ = 0;
  private nFeaturesIn_ = 0;
  private fitted = false;

  /**
   * Create a new Mean Shift model.
   *
   * @param options - Configuration options
   * @param options.bandwidth - Kernel radius, or `"auto"` to estimate it from the data (default: "auto")
   * @param options.maxIter - Maximum number of shift iterations per seed (default: 300)
   * @param options.tol - A seed has converged when it moves at most `tol * bandwidth` (default: 1e-3)
   * @param options.binSeeding - Start from the centers of occupied bandwidth-sized bins instead of every sample (default: false)
   * @param options.minBinFreq - With `binSeeding`, only bins holding at least this many samples are used as seeds (default: 1)
   * @param options.clusterAll - If false, samples farther than `bandwidth` from every mode get label -1 (default: true)
   * @throws {InvalidParameterError} If any option is out of range
   */
  constructor(
    options: {
      readonly bandwidth?: number | "auto";
      readonly maxIter?: number;
      readonly tol?: number;
      readonly binSeeding?: boolean;
      readonly minBinFreq?: number;
      readonly clusterAll?: boolean;
    } = {}
  ) {
    this.bandwidth = checkBandwidth(options.bandwidth ?? "auto");
    this.maxIter = checkMaxIter(options.maxIter ?? 300);
    this.tol = checkTol(options.tol ?? 1e-3);
    this.binSeeding = checkBoolean("binSeeding", options.binSeeding ?? false);
    this.minBinFreq = checkMinBinFreq(options.minBinFreq ?? 1);
    this.clusterAll = checkBoolean("clusterAll", options.clusterAll ?? true);
  }

  /**
   * Fit Mean Shift clustering.
   *
   * Calling `fit` again replaces the previous model.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (exists for compatibility)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D
   * @throws {DataValidationError} If X is empty or contains NaN/Inf
   * @throws {InvalidParameterError} If the bandwidth estimate is 0 (all samples identical or fewer
   *   than 4 samples), or if no sample lies within the bandwidth of any seed
   */
  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const data = toFloat64View(X);

    const bw = this.bandwidth === "auto" ? MeanShift.estimateBandwidth(data, n, d) : this.bandwidth;
    const bwSq = bw * bw;

    const seeds = this.binSeeding ? this.getBinSeeds(data, n, d, bw) : data;
    const nSeeds = seeds.length / d;

    const stopThreshold = this.tol * bw;
    const modes: Mode[] = [];
    let maxIterations = 0;
    const newMean = new Float64Array(d);

    for (let s = 0; s < nSeeds; s++) {
      const current = seeds.slice(s * d, (s + 1) * d);
      let count = 0;
      let iterations = 0;

      for (;;) {
        newMean.fill(0);
        count = 0;
        for (let i = 0; i < n; i++) {
          const base = i * d;
          let distSq = 0;
          for (let f = 0; f < d; f++) {
            const diff = (data[base + f] as number) - (current[f] as number);
            distSq += diff * diff;
            if (distSq > bwSq) break;
          }
          if (distSq <= bwSq) {
            count++;
            for (let f = 0; f < d; f++)
              newMean[f] = (newMean[f] as number) + (data[base + f] as number);
          }
        }
        if (count === 0) break;

        let shiftSq = 0;
        for (let f = 0; f < d; f++) {
          const mean = (newMean[f] as number) / count;
          const diff = mean - (current[f] as number);
          shiftSq += diff * diff;
          current[f] = mean;
        }
        if (Math.sqrt(shiftSq) <= stopThreshold || iterations === this.maxIter) break;
        iterations++;
      }

      if (count > 0) modes.push({ center: current, intensity: count });
      if (iterations > maxIterations) maxIterations = iterations;
    }

    if (modes.length === 0) {
      throw new InvalidParameterError(
        `No sample was within bandwidth=${bw} of any seed; use a different seeding strategy or increase the bandwidth`,
        "bandwidth",
        bw
      );
    }

    // Strongest mode first; ties fall back to the coordinates (descending).
    modes.sort((a, b) => {
      if (a.intensity !== b.intensity) return b.intensity - a.intensity;
      for (let f = 0; f < d; f++) {
        const diff = (b.center[f] as number) - (a.center[f] as number);
        if (diff !== 0) return diff;
      }
      return 0;
    });

    // Drop every mode that lies within `bandwidth` of a stronger, retained mode.
    const keep = new Uint8Array(modes.length).fill(1);
    for (let i = 0; i < modes.length; i++) {
      if (!keep[i]) continue;
      const ci = (modes[i] as Mode).center;
      for (let j = i + 1; j < modes.length; j++) {
        if (!keep[j]) continue;
        const cj = (modes[j] as Mode).center;
        let distSq = 0;
        for (let f = 0; f < d; f++) {
          const diff = (ci[f] as number) - (cj[f] as number);
          distSq += diff * diff;
        }
        if (distSq <= bwSq) keep[j] = 0;
      }
    }
    const kept = modes.filter((_, i) => keep[i] === 1);
    const nCenters = kept.length;
    const centers = new Float64Array(nCenters * d);
    kept.forEach((m, k) => {
      centers.set(m.center, k * d);
    });

    const labels = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const [best, bestDistSq] = MeanShift.nearest(data, i * d, centers, nCenters, d);
      labels[i] = this.clusterAll || bestDistSq <= bwSq ? best : -1;
    }

    this.nFeaturesIn_ = d;
    this.clusterCenters_ = tensor(centers).reshape([nCenters, d]);
    this.labels_ = tensor(labels, { dtype: "int32" });
    this.nIter_ = maxIterations;
    this.fitted = true;
    return this;
  }

  /**
   * Index and squared distance of the center nearest to the row at `base` (lowest index wins ties).
   */
  private static nearest(
    x: Float64Array,
    base: number,
    centers: Float64Array,
    k: number,
    d: number
  ): [number, number] {
    let bestK = 0;
    let bestDist = Infinity;
    for (let c = 0; c < k; c++) {
      let dist = 0;
      const cBase = c * d;
      for (let f = 0; f < d; f++) {
        const diff = (x[base + f] as number) - (centers[cBase + f] as number);
        dist += diff * diff;
        if (dist >= bestDist) break;
      }
      if (dist < bestDist) {
        bestDist = dist;
        bestK = c;
      }
    }
    return [bestK, bestDist];
  }

  /**
   * Bandwidth estimate: the mean, over all samples, of the distance to the
   * `floor(0.3 * n)`-th nearest sample (the sample itself counts as the first).
   * Matches `sklearn.cluster.estimate_bandwidth(X, quantile=0.3)`.
   */
  private static estimateBandwidth(data: Float64Array, n: number, d: number): number {
    const k = Math.max(1, Math.floor(n * BANDWIDTH_QUANTILE));
    const row = new Float64Array(n);
    let total = 0;
    for (let i = 0; i < n; i++) {
      const iBase = i * d;
      for (let j = 0; j < n; j++) {
        const jBase = j * d;
        let dist = 0;
        for (let f = 0; f < d; f++) {
          const diff = (data[iBase + f] as number) - (data[jBase + f] as number);
          dist += diff * diff;
        }
        row[j] = dist;
      }
      total += Math.sqrt(kthSmallest(row, k - 1));
    }
    const bw = total / n;
    if (!(bw > 0)) {
      throw new InvalidParameterError(
        'The estimated bandwidth is 0: the samples are (nearly) identical, or there are fewer than 4 samples (the estimate uses the floor(0.3 * n)-th nearest sample, counting the sample itself); pass an explicit positive "bandwidth"',
        "bandwidth",
        bw
      );
    }
    return bw;
  }

  /**
   * Seeds for binned seeding: the centers of the bandwidth-sized grid cells that hold at
   * least `minBinFreq` samples. Falls back to the samples themselves when binning does
   * not reduce the number of seeds.
   */
  private getBinSeeds(data: Float64Array, n: number, d: number, bw: number): Float64Array {
    // An infinite bandwidth puts every sample in one bin whose center would be 0 * Infinity.
    if (!Number.isFinite(bw)) return data;
    const bins = new Map<string, { readonly key: number[]; count: number }>();
    for (let i = 0; i < n; i++) {
      const key: number[] = [];
      for (let f = 0; f < d; f++) key.push(roundHalfEven((data[i * d + f] as number) / bw) + 0);
      const keyStr = key.join(",");
      const existing = bins.get(keyStr);
      if (existing) existing.count++;
      else bins.set(keyStr, { key, count: 1 });
    }

    const selected = [...bins.values()].filter((b) => b.count >= this.minBinFreq);
    if (selected.length === n) {
      warn(
        `Binning data failed with bin size ${bw}; using the samples as seeds.`,
        "UserWarning",
        "MeanShift"
      );
      return data;
    }
    const seeds = new Float64Array(selected.length * d);
    selected.forEach((b, s) => {
      for (let f = 0; f < d; f++) seeds[s * d + f] = (b.key[f] as number) * bw;
    });
    return seeds;
  }

  /**
   * Assign each sample to the nearest cluster center found during `fit`.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Cluster labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X is not 2D or has a different number of features than the training data
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.clusterCenters_) {
      throw new NotFittedError("MeanShift must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "MeanShift");

    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const data = toFloat64View(X);
    const centers = toFloat64View(this.clusterCenters_);
    const nClusters = this.clusterCenters_.shape[0] ?? 0;
    const labels = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      labels[i] = MeanShift.nearest(data, i * d, centers, nClusters, d)[0];
    }
    return tensor(labels, { dtype: "int32" });
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

  /** Training labels of shape (n_samples,); -1 marks samples left unclustered when `clusterAll` is false. */
  get labels(): Tensor {
    if (!this.fitted || !this.labels_) {
      throw new NotFittedError("MeanShift must be fitted to access labels");
    }
    return this.labels_;
  }

  /** Cluster centers of shape (n_clusters, n_features), strongest mode first. */
  get clusterCenters(): Tensor {
    if (!this.fitted || !this.clusterCenters_) {
      throw new NotFittedError("MeanShift must be fitted to access cluster centers");
    }
    return this.clusterCenters_;
  }

  /** Largest number of shift iterations any seed needed. */
  get nIter(): number {
    if (!this.fitted) {
      throw new NotFittedError("MeanShift must be fitted to access the iteration count");
    }
    return this.nIter_;
  }

  getParams(): Record<string, unknown> {
    return {
      bandwidth: this.bandwidth,
      maxIter: this.maxIter,
      tol: this.tol,
      binSeeding: this.binSeeding,
      minBinFreq: this.minBinFreq,
      clusterAll: this.clusterAll,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "bandwidth":
          this.bandwidth = checkBandwidth(value);
          break;
        case "maxIter":
          this.maxIter = checkMaxIter(value);
          break;
        case "tol":
          this.tol = checkTol(value);
          break;
        case "binSeeding":
          this.binSeeding = checkBoolean("binSeeding", value);
          break;
        case "minBinFreq":
          this.minBinFreq = checkMinBinFreq(value);
          break;
        case "clusterAll":
          this.clusterAll = checkBoolean("clusterAll", value);
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
