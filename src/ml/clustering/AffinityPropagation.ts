/**
 * Affinity Propagation clustering.
 *
 * Clusters data by sending messages between pairs of samples until
 * convergence. Automatically determines the number of clusters from
 * the data using a preference parameter.
 *
 * @module ml/clustering/AffinityPropagation
 * @see {@link https://deepbox.dev/docs/ml-clustering | Deepbox documentation}
 */

import {
  DataValidationError,
  InvalidParameterError,
  MemoryError,
  NotFittedError,
  ShapeError,
  warn,
} from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __SeededRandom, __seedToUint64 } from "../../random/random";
import {
  toFloat64View,
  validatePredictInputs,
  validateUnsupervisedFitInputs,
} from "../_validation";
import type { Clusterer } from "../base";

type Affinity = "euclidean" | "precomputed";

/** Machine epsilon and smallest normal double: scale of the tie-breaking noise added to S. */
const EPS = 2.220446049250313e-16;
const TINY = 2.2250738585072014e-308;

/** Allocate an n-by-n matrix of doubles, reporting a failed allocation as a MemoryError. */
function allocSquare(n: number): Float64Array {
  try {
    return new Float64Array(n * n);
  } catch (error) {
    throw new MemoryError(
      `AffinityPropagation needs several ${n} x ${n} matrices of doubles, which could not be allocated`,
      { requestedBytes: n * n * 8, cause: error }
    );
  }
}

function isValidDamping(value: unknown): value is number {
  return typeof value === "number" && value >= 0.5 && value < 1;
}

function isPositiveInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 1;
}

/**
 * Affinity Propagation clustering algorithm.
 *
 * Follows the message-passing scheme of Frey and Dueck (2007) as implemented in
 * scikit-learn: similarities are negative squared Euclidean distances, the
 * diagonal of the similarity matrix holds the `preference` (default: the median
 * of all similarities, diagonal included), and tiny random noise breaks exact
 * ties. Samples with a positive `a(k,k) + r(k,k)` become exemplars.
 *
 * The algorithm needs three n-by-n matrices, so memory grows quadratically with
 * the number of samples. If no exemplar emerges, all labels are -1 and a
 * `ConvergenceWarning` is issued.
 *
 * @example
 * ```ts
 * import { AffinityPropagation } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0], [0.1, 0], [10, 10], [10.1, 10]]);
 * const ap = new AffinityPropagation({ damping: 0.5 });
 * ap.fit(X);
 * console.log(ap.labels);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-clustering | Deepbox Clustering}
 */
export class AffinityPropagation implements Clusterer {
  private damping: number;
  private maxIter: number;
  private convergenceIter: number;
  private preference: number | undefined;
  private affinity: Affinity;
  private randomState: number;

  private labels_?: Tensor;
  private clusterCentersIndices_?: Int32Array;
  private clusterCenters_?: Float64Array;
  private nFeaturesIn_ = 0;
  private nIter_ = 0;
  private converged_ = false;
  private fitted = false;

  /**
   * Create an Affinity Propagation model.
   *
   * @param options - Configuration options
   * @param options.damping - Damping factor in [0.5, 1) applied to the messages (default: 0.5)
   * @param options.maxIter - Maximum number of iterations, integer >= 1 (default: 200)
   * @param options.convergenceIter - Number of iterations with an unchanged exemplar set that ends the run, integer >= 1 (default: 15)
   * @param options.preference - Preference of every sample to become an exemplar. Larger values give more clusters. Default: median of the similarities.
   * @param options.affinity - "euclidean" to cluster feature vectors, or "precomputed" when `X` is a square similarity matrix (default: "euclidean")
   * @param options.randomState - Seed for the tie-breaking noise (default: 0)
   * @throws {InvalidParameterError} If any option is out of range
   */
  constructor(
    options: {
      readonly damping?: number;
      readonly maxIter?: number;
      readonly convergenceIter?: number;
      readonly preference?: number;
      readonly affinity?: Affinity;
      readonly randomState?: number;
    } = {}
  ) {
    this.damping = options.damping ?? 0.5;
    this.maxIter = options.maxIter ?? 200;
    this.convergenceIter = options.convergenceIter ?? 15;
    this.affinity = options.affinity ?? "euclidean";
    this.randomState = options.randomState ?? 0;
    if (options.preference !== undefined) this.preference = options.preference;

    if (!isValidDamping(this.damping)) {
      throw new InvalidParameterError("damping must be in [0.5, 1)", "damping", this.damping);
    }
    if (!isPositiveInteger(this.maxIter)) {
      throw new InvalidParameterError("maxIter must be an integer >= 1", "maxIter", this.maxIter);
    }
    if (!isPositiveInteger(this.convergenceIter)) {
      throw new InvalidParameterError(
        "convergenceIter must be an integer >= 1",
        "convergenceIter",
        this.convergenceIter
      );
    }
    if (this.preference !== undefined && !Number.isFinite(this.preference)) {
      throw new InvalidParameterError(
        "preference must be a finite number",
        "preference",
        this.preference
      );
    }
    if (this.affinity !== "euclidean" && this.affinity !== "precomputed") {
      throw new InvalidParameterError(
        `affinity must be "euclidean" or "precomputed"`,
        "affinity",
        this.affinity
      );
    }
    if (!Number.isFinite(this.randomState)) {
      throw new InvalidParameterError(
        "randomState must be a finite number",
        "randomState",
        this.randomState
      );
    }
  }

  /**
   * Run affinity propagation.
   *
   * @param X - Samples of shape (n_samples, n_features), or with `affinity: "precomputed"` a similarity matrix of shape (n_samples, n_samples)
   * @param _y - Ignored (exists for API compatibility)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D, or is not square with `affinity: "precomputed"`
   * @throws {DataValidationError} If X is empty or contains NaN/Inf
   * @throws {MemoryError} If the n-by-n message matrices cannot be allocated
   */
  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const precomputed = this.affinity === "precomputed";
    if (precomputed && n !== d) {
      throw new ShapeError(
        `affinity="precomputed" needs a square similarity matrix; got shape [${n}, ${d}]`
      );
    }
    const data = toFloat64View(X);

    // Similarity matrix S: negative squared Euclidean distances, or the given matrix.
    const S = allocSquare(n);
    if (precomputed) {
      S.set(data);
    } else {
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          let dist = 0;
          for (let f = 0; f < d; f++) {
            const diff = (data[i * d + f] as number) - (data[j * d + f] as number);
            dist += diff * diff;
          }
          S[i * n + j] = -dist;
          S[j * n + i] = -dist;
        }
      }
    }

    const preference = this.preference ?? AffinityPropagation.median(S);

    const result = this.propagate(S, n, preference);

    this.labels_ = tensor(result.labels);
    this.clusterCentersIndices_ = result.exemplars;
    this.nIter_ = result.nIter;
    this.converged_ = result.converged;
    this.nFeaturesIn_ = d;
    if (precomputed) {
      delete this.clusterCenters_;
    } else {
      const k = result.exemplars.length;
      const centers = new Float64Array(k * d);
      for (let c = 0; c < k; c++) {
        const idx = result.exemplars[c] as number;
        for (let f = 0; f < d; f++) centers[c * d + f] = data[idx * d + f] as number;
      }
      this.clusterCenters_ = centers;
    }
    this.fitted = true;
    return this;
  }

  /** Median of all entries of `values` (mean of the two middle values for an even count). */
  private static median(values: Float64Array): number {
    const sorted = values.slice().sort();
    const mid = sorted.length >> 1;
    return sorted.length % 2 === 1
      ? (sorted[mid] as number)
      : ((sorted[mid - 1] as number) + (sorted[mid] as number)) / 2;
  }

  /** Message passing on the similarity matrix; returns exemplars, labels and iteration info. */
  private propagate(
    S: Float64Array,
    n: number,
    preference: number
  ): { labels: Int32Array; exemplars: Int32Array; nIter: number; converged: boolean } {
    // Degenerate input: all off-diagonal similarities equal (and one common preference).
    if (n === 1 || AffinityPropagation.allOffDiagonalEqual(S, n)) {
      warn(
        "All samples have mutually equal similarities. Returning arbitrary cluster center(s).",
        "UserWarning",
        "AffinityPropagation"
      );
      const reference = n === 1 ? (S[0] as number) : (S[n - 1] as number);
      if (preference > reference) {
        const labels = new Int32Array(n);
        const exemplars = new Int32Array(n);
        for (let i = 0; i < n; i++) {
          labels[i] = i;
          exemplars[i] = i;
        }
        return { labels, exemplars, nIter: 0, converged: true };
      }
      return {
        labels: new Int32Array(n),
        exemplars: new Int32Array([0]),
        nIter: 0,
        converged: true,
      };
    }

    for (let i = 0; i < n; i++) S[i * n + i] = preference;

    // Remove degeneracies: noise far below the data scale breaks exact ties.
    const noise = new __SeededRandom(__seedToUint64(this.randomState));
    for (let i = 0; i < n * n; i++) {
      const v = S[i] as number;
      S[i] = v + (EPS * v + TINY * 100) * noise.nextNormal();
    }

    const damping = this.damping;
    const keep = 1 - damping;
    const R = allocSquare(n);
    const A = allocSquare(n);
    const posSum = new Float64Array(n);
    const rDiag = new Float64Array(n);
    const conv = this.convergenceIter;
    const history = new Uint8Array(n * conv);
    const exemplar = new Uint8Array(n);

    let nIter = 0;
    let converged = false;
    for (let it = 0; it < this.maxIter; it++) {
      nIter = it + 1;

      // Responsibilities: r(i,k) = s(i,k) - max_{k' != k} (a(i,k') + s(i,k')).
      for (let i = 0; i < n; i++) {
        const base = i * n;
        let first = -Infinity;
        let second = -Infinity;
        let arg = 0;
        for (let k = 0; k < n; k++) {
          const v = (A[base + k] as number) + (S[base + k] as number);
          if (v > first) {
            second = first;
            first = v;
            arg = k;
          } else if (v > second) {
            second = v;
          }
        }
        for (let k = 0; k < n; k++) {
          const fresh = (S[base + k] as number) - (k === arg ? second : first);
          R[base + k] = damping * (R[base + k] as number) + keep * fresh;
        }
      }

      // Availabilities from the positive part of the responsibilities of each column.
      posSum.fill(0);
      for (let i = 0; i < n; i++) {
        const base = i * n;
        for (let k = 0; k < n; k++) {
          const r = R[base + k] as number;
          if (r > 0) posSum[k] = (posSum[k] as number) + r;
        }
      }
      for (let k = 0; k < n; k++) rDiag[k] = R[k * n + k] as number;
      for (let i = 0; i < n; i++) {
        const base = i * n;
        for (let k = 0; k < n; k++) {
          const rkk = rDiag[k] as number;
          const rik = R[base + k] as number;
          const ownPositive = rkk > 0 ? rkk : 0;
          let fresh: number;
          if (i === k) {
            // a(k,k) = sum_{i' != k} max(0, r(i',k))
            fresh = (posSum[k] as number) - ownPositive;
          } else {
            // a(i,k) = min(0, r(k,k) + sum_{i' not in {i,k}} max(0, r(i',k)))
            const sum = rkk + (posSum[k] as number) - ownPositive - (rik > 0 ? rik : 0);
            fresh = sum < 0 ? sum : 0;
          }
          A[base + k] = damping * (A[base + k] as number) + keep * fresh;
        }
      }

      // Convergence: the exemplar set must stay unchanged for `convergenceIter` iterations.
      let nExemplars = 0;
      const slot = it % conv;
      for (let i = 0; i < n; i++) {
        const e = (A[i * n + i] as number) + (R[i * n + i] as number) > 0 ? 1 : 0;
        exemplar[i] = e;
        history[i * conv + slot] = e;
        nExemplars += e;
      }
      if (it >= conv) {
        let stable = 0;
        for (let i = 0; i < n; i++) {
          let seen = 0;
          for (let j = 0; j < conv; j++) seen += history[i * conv + j] as number;
          if (seen === conv || seen === 0) stable++;
        }
        if (stable === n && nExemplars > 0) {
          converged = true;
          break;
        }
      }
    }

    const exemplarIdx: number[] = [];
    for (let i = 0; i < n; i++) if (exemplar[i] === 1) exemplarIdx.push(i);
    const K = exemplarIdx.length;

    if (K === 0) {
      warn(
        "Affinity propagation did not converge and this model will not have any cluster centers.",
        "ConvergenceWarning",
        "AffinityPropagation"
      );
      return {
        labels: new Int32Array(n).fill(-1),
        exemplars: new Int32Array(0),
        nIter,
        converged: false,
      };
    }
    if (!converged) {
      warn(
        "Affinity propagation did not converge, this model may return degenerate cluster centers and labels.",
        "ConvergenceWarning",
        "AffinityPropagation"
      );
    }

    // Assign every sample to its most similar exemplar; exemplars stay in their own cluster.
    const assignTo = (exemplars: number[]): Int32Array => {
      const c = new Int32Array(n);
      for (let i = 0; i < n; i++) {
        let best = 0;
        let bestVal = -Infinity;
        for (let j = 0; j < exemplars.length; j++) {
          const v = S[i * n + (exemplars[j] as number)] as number;
          if (v > bestVal) {
            bestVal = v;
            best = j;
          }
        }
        c[i] = best;
      }
      for (let j = 0; j < exemplars.length; j++) c[exemplars[j] as number] = j;
      return c;
    };

    let c = assignTo(exemplarIdx);

    // Refine: the exemplar of a cluster is the member most similar to all members.
    for (let k = 0; k < K; k++) {
      const members: number[] = [];
      for (let i = 0; i < n; i++) if (c[i] === k) members.push(i);
      let bestMember = members[0] as number;
      let bestSum = -Infinity;
      for (const m of members) {
        let sum = 0;
        for (const i of members) sum += S[i * n + m] as number;
        if (sum > bestSum) {
          bestSum = sum;
          bestMember = m;
        }
      }
      exemplarIdx[k] = bestMember;
    }
    c = assignTo(exemplarIdx);

    // Reduce to sorted, gapless labels.
    const used = [...new Set(Array.from(c, (j) => exemplarIdx[j] as number))].sort((a, b) => a - b);
    const rank = new Map<number, number>();
    for (let pos = 0; pos < used.length; pos++) rank.set(used[pos] as number, pos);
    const labels = new Int32Array(n);
    for (let i = 0; i < n; i++)
      labels[i] = rank.get(exemplarIdx[c[i] as number] as number) as number;

    return { labels, exemplars: new Int32Array(used), nIter, converged };
  }

  /** Whether every off-diagonal entry of the n-by-n matrix `S` equals the first one. */
  private static allOffDiagonalEqual(S: Float64Array, n: number): boolean {
    const first = S[1] as number;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i !== j && (S[i * n + j] as number) !== first) return false;
      }
    }
    return true;
  }

  /**
   * Assign each sample to the nearest cluster center (exemplar).
   *
   * If the fit produced no exemplars, every sample gets label -1 and a
   * `ConvergenceWarning` is issued.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Cluster labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X is not 2D or has a different number of features than the training data
   * @throws {InvalidParameterError} If the model was fitted with `affinity: "precomputed"`
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.clusterCentersIndices_) {
      throw new NotFittedError("AffinityPropagation must be fitted before predict");
    }
    if (!this.clusterCenters_) {
      // Exemplar rows only exist when the model was fitted on feature vectors.
      throw new InvalidParameterError(
        'predict is not supported when affinity is "precomputed"',
        "affinity",
        "precomputed"
      );
    }
    validatePredictInputs(X, this.nFeaturesIn_, "AffinityPropagation");
    const nSamples = X.shape[0] ?? 0;
    const d = this.nFeaturesIn_;
    const nClusters = this.clusterCentersIndices_.length;
    const centers = this.clusterCenters_;
    const data = toFloat64View(X);

    if (nClusters === 0) {
      warn(
        "This model does not have any cluster centers because affinity propagation did not converge. Labeling every sample as -1.",
        "ConvergenceWarning",
        "AffinityPropagation"
      );
      return tensor(new Int32Array(nSamples).fill(-1));
    }

    const labels = new Int32Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      let bestC = 0;
      let bestD = Infinity;
      for (let c = 0; c < nClusters; c++) {
        let dist = 0;
        for (let j = 0; j < d; j++) {
          const diff = (data[i * d + j] as number) - (centers[c * d + j] as number);
          dist += diff * diff;
        }
        if (dist < bestD) {
          bestD = dist;
          bestC = c;
        }
      }
      labels[i] = bestC;
    }

    return tensor(labels);
  }

  /**
   * Fit the model and return the cluster labels of the training samples.
   *
   * @param X - Samples (or similarity matrix) as accepted by {@link AffinityPropagation.fit}
   * @param _y - Ignored (exists for API compatibility)
   * @returns Cluster labels of shape (n_samples,)
   */
  fitPredict(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    return this.labels_ as Tensor;
  }

  /**
   * Cluster labels of the training samples (-1 for all samples if no exemplar was found).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get labels(): Tensor {
    if (!this.fitted || !this.labels_) {
      throw new NotFittedError("AffinityPropagation must be fitted to access labels");
    }
    return this.labels_;
  }

  /**
   * Exemplar samples, one per cluster.
   *
   * @returns Tensor of shape (n_clusters, n_features)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {DataValidationError} If the model was fitted with `affinity: "precomputed"`
   */
  get clusterCenters(): Tensor {
    if (!this.fitted || !this.clusterCentersIndices_) {
      throw new NotFittedError("AffinityPropagation must be fitted to access cluster centers");
    }
    if (!this.clusterCenters_) {
      throw new DataValidationError(
        'cluster centers are not available when affinity is "precomputed"; use clusterCentersIndices'
      );
    }
    return tensor(this.clusterCenters_.slice()).reshape([
      this.clusterCentersIndices_.length,
      this.nFeaturesIn_,
    ]);
  }

  /**
   * Row indices of the exemplars in the training data (sorted, one per cluster).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get clusterCentersIndices(): Int32Array {
    if (!this.fitted || !this.clusterCentersIndices_) {
      throw new NotFittedError(
        "AffinityPropagation must be fitted to access cluster center indices"
      );
    }
    return this.clusterCentersIndices_;
  }

  /**
   * Number of iterations that were run.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nIter(): number {
    if (!this.fitted)
      throw new NotFittedError("AffinityPropagation must be fitted to access nIter");
    return this.nIter_;
  }

  /**
   * Whether the exemplar set stabilized before `maxIter` was reached.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get converged(): boolean {
    if (!this.fitted) {
      throw new NotFittedError("AffinityPropagation must be fitted to access converged");
    }
    return this.converged_;
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      damping: this.damping,
      maxIter: this.maxIter,
      convergenceIter: this.convergenceIter,
      preference: this.preference,
      affinity: this.affinity,
      randomState: this.randomState,
    };
  }

  /**
   * Set the parameters of this estimator.
   *
   * @param params - Parameters to set (damping, maxIter, convergenceIter, preference, affinity, randomState)
   * @returns this
   * @throws {InvalidParameterError} If any parameter value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "damping":
          if (!isValidDamping(value)) {
            throw new InvalidParameterError("damping must be in [0.5, 1)", "damping", value);
          }
          this.damping = value;
          break;
        case "maxIter":
          if (!isPositiveInteger(value)) {
            throw new InvalidParameterError("maxIter must be an integer >= 1", "maxIter", value);
          }
          this.maxIter = value;
          break;
        case "convergenceIter":
          if (!isPositiveInteger(value)) {
            throw new InvalidParameterError(
              "convergenceIter must be an integer >= 1",
              "convergenceIter",
              value
            );
          }
          this.convergenceIter = value;
          break;
        case "preference":
          if (value !== undefined && (typeof value !== "number" || !Number.isFinite(value))) {
            throw new InvalidParameterError(
              "preference must be a finite number or undefined",
              "preference",
              value
            );
          }
          this.preference = value;
          break;
        case "affinity":
          if (value !== "euclidean" && value !== "precomputed") {
            throw new InvalidParameterError(
              `affinity must be "euclidean" or "precomputed"`,
              "affinity",
              value
            );
          }
          this.affinity = value;
          break;
        case "randomState":
          if (typeof value !== "number" || !Number.isFinite(value)) {
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
