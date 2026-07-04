/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { validateUnsupervisedFitInputs } from "../_validation";
import type { OutlierDetector } from "../base";

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

  private trainData_?: number[][];
  private lofScores_?: number[];
  private threshold_ = 1.5;
  // Per-training-point statistics needed to compute a proper LOF for NEW
  // (novelty) points: k-distance and local reachability density.
  private trainKDist_?: number[];
  private trainLrd_?: number[];
  private fitted = false;

  /**
   * @param options.nNeighbors - Number of neighbors for kNN (default: 20)
   * @param options.contamination - Expected proportion of outliers (default: 'auto')
   */
  constructor(
    options: {
      readonly nNeighbors?: number;
      readonly contamination?: number | "auto";
    } = {}
  ) {
    this.nNeighbors = options.nNeighbors ?? 20;
    this.contamination = options.contamination ?? "auto";

    if (!Number.isInteger(this.nNeighbors) || this.nNeighbors < 1) {
      throw new InvalidParameterError(
        "nNeighbors must be an integer >= 1",
        "nNeighbors",
        this.nNeighbors
      );
    }
  }

  private euclideanDist(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const d = (a[i] ?? 0) - (b[i] ?? 0);
      sum += d * d;
    }
    return Math.sqrt(sum);
  }

  /**
   * Find k nearest neighbor indices and distances for a point.
   */
  private kNearestNeighbors(
    pointIdx: number,
    data: number[][],
    k: number
  ): { indices: number[]; distances: number[] } {
    const dists: Array<{ idx: number; dist: number }> = [];
    for (let i = 0; i < data.length; i++) {
      if (i === pointIdx) continue;
      dists.push({
        idx: i,
        dist: this.euclideanDist(data[pointIdx]!, data[i]!),
      });
    }
    dists.sort((a, b) => a.dist - b.dist);
    const topK = dists.slice(0, k);
    return {
      indices: topK.map((d) => d.idx),
      distances: topK.map((d) => d.dist),
    };
  }

  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    const effectiveK = Math.min(this.nNeighbors, nSamples - 1);
    if (effectiveK < 1) {
      throw new InvalidParameterError(
        "Not enough samples for the specified nNeighbors",
        "nNeighbors",
        this.nNeighbors
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
    this.trainData_ = data;

    // Compute kNN for all points
    const knnResults: Array<{ indices: number[]; distances: number[] }> = [];
    for (let i = 0; i < nSamples; i++) {
      knnResults.push(this.kNearestNeighbors(i, data, effectiveK));
    }

    // k-distance for each point = distance to k-th neighbor
    const kDist: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      const dists = knnResults[i]!.distances;
      kDist.push(dists[dists.length - 1] ?? 0);
    }

    // Reachability distance: max(k-distance(o), dist(p, o))
    // Local reachability density
    const lrd: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      const neighbors = knnResults[i]!;
      let reachSum = 0;
      for (let n = 0; n < neighbors.indices.length; n++) {
        const nIdx = neighbors.indices[n] ?? 0;
        const dist = neighbors.distances[n] ?? 0;
        reachSum += Math.max(kDist[nIdx] ?? 0, dist);
      }
      lrd.push(reachSum > 0 ? neighbors.indices.length / reachSum : 1);
    }

    this.trainKDist_ = kDist;
    this.trainLrd_ = lrd;

    // LOF score for each point
    this.lofScores_ = [];
    for (let i = 0; i < nSamples; i++) {
      const neighbors = knnResults[i]!;
      let lofSum = 0;
      for (const nIdx of neighbors.indices) {
        lofSum += (lrd[nIdx] ?? 1) / Math.max(lrd[i] ?? 1, 1e-10);
      }
      this.lofScores_.push(lofSum / neighbors.indices.length);
    }

    // Determine threshold
    if (this.contamination !== "auto") {
      const sorted = [...this.lofScores_].sort((a, b) => b - a);
      const cutoff = Math.floor(this.contamination * nSamples);
      this.threshold_ = sorted[Math.max(0, cutoff - 1)] ?? 1.5;
    } else {
      // Default threshold
      this.threshold_ = 1.5;
    }

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.lofScores_ || !this.trainData_) {
      throw new NotFittedError("LocalOutlierFactor must be fitted before prediction");
    }

    // Compute the true LOF of each query point against the training set
    // (novelty detection). Comparing the LOF ratio to the same threshold used
    // on training data is dimensionally correct — the previous version
    // compared a raw average distance to the LOF-ratio threshold and used a
    // row-count check that returned memorized labels for any same-sized input.
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const nTrain = this.trainData_.length;
    const trainKDist = this.trainKDist_ ?? [];
    const trainLrd = this.trainLrd_ ?? [];
    const effectiveK = Math.min(this.nNeighbors, nTrain);
    const labels: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      const xi: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        xi.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      const dists: Array<{ idx: number; dist: number }> = [];
      for (let t = 0; t < nTrain; t++) {
        dists.push({ idx: t, dist: this.euclideanDist(xi, this.trainData_[t]!) });
      }
      dists.sort((a, b) => a.dist - b.dist);
      // Exclude an exact self-match (distance 0) so querying a training point
      // reproduces its fitted LOF rather than being skewed by a zero-distance
      // "neighbor". New points have no zero-distance match, so this is a no-op.
      const nonSelf = dists.filter((d) => d.dist > 1e-12);
      const pool = nonSelf.length >= effectiveK ? nonSelf : dists;
      const topK = pool.slice(0, effectiveK);

      // lrd(q) = 1 / mean_i reachDist(q, o_i), reachDist = max(k-dist(o_i), d(q,o_i))
      let reachSum = 0;
      for (const { idx, dist } of topK) {
        reachSum += Math.max(trainKDist[idx] ?? 0, dist);
      }
      const lrdQ = reachSum > 0 ? topK.length / reachSum : 1;
      // LOF(q) = mean_i lrd(o_i) / lrd(q)
      let lofSum = 0;
      for (const { idx } of topK) lofSum += (trainLrd[idx] ?? 1) / Math.max(lrdQ, 1e-10);
      const lof = topK.length > 0 ? lofSum / topK.length : 1;
      labels.push(lof >= this.threshold_ ? -1 : 1);
    }

    return tensor(labels, { dtype: "int32" });
  }

  fitPredict(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    // Training-set labels use the precomputed LOF scores (sklearn's
    // fit_predict / negative_outlier_factor_ convention).
    if (!this.lofScores_) throw new NotFittedError("LocalOutlierFactor was not fitted");
    const labels = this.lofScores_.map((s) => (s >= this.threshold_ ? -1 : 1));
    return tensor(labels, { dtype: "int32" });
  }

  scoreSamples(X: Tensor): Tensor {
    if (!this.fitted || !this.lofScores_) {
      throw new NotFittedError("LocalOutlierFactor must be fitted before scoring");
    }

    const nSamples = X.shape[0] ?? 0;
    const scores: number[] = [];

    if (nSamples === (this.trainData_?.length ?? 0)) {
      // Return negative LOF scores (more negative = more anomalous)
      for (let i = 0; i < nSamples; i++) {
        scores.push(-(this.lofScores_[i] ?? 0));
      }
    } else {
      // For new data, return negative distance heuristic
      const nFeatures = X.shape[1] ?? 0;
      for (let i = 0; i < nSamples; i++) {
        const xi: number[] = [];
        for (let j = 0; j < nFeatures; j++) {
          xi.push(Number(X.data[X.offset + i * nFeatures + j]));
        }
        let minDist = Infinity;
        for (const train of this.trainData_!) {
          const d = this.euclideanDist(xi, train);
          if (d < minDist) minDist = d;
        }
        scores.push(-minDist);
      }
    }

    return tensor(scores);
  }

  /**
   * Get the computed LOF scores for training data.
   * Higher values indicate more anomalous points.
   */
  get negativeLofScores(): Tensor {
    if (!this.fitted || !this.lofScores_) {
      throw new NotFittedError("LocalOutlierFactor must be fitted to access LOF scores");
    }
    return tensor(this.lofScores_.map((s) => -s));
  }

  getParams(): Record<string, unknown> {
    return { nNeighbors: this.nNeighbors, contamination: this.contamination };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
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
        case "contamination":
          if (value !== "auto" && (typeof value !== "number" || value <= 0 || value > 0.5)) {
            throw new InvalidParameterError(
              'contamination must be "auto" or in (0, 0.5]',
              "contamination",
              value
            );
          }
          this.contamination = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
