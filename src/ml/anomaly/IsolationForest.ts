/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __random, __randomBelow, __SeededRandom, __seedToUint64 } from "../../random/random";
import {
  percentileSorted,
  toFloat64View,
  validatePredictInputs,
  validateUnsupervisedFitInputs,
} from "../_validation";
import type { OutlierDetector } from "../base";

/** Euler-Mascheroni constant. */
const EULER_GAMMA = 0.5772156649015329;

/**
 * An isolation tree stored as flat arrays. Node 0 is the root. A node with
 * `feature[node] < 0` is a leaf and `leafValue[node]` holds the average path
 * length correction `c(size)` for the samples that ended up in it.
 */
type FlatTree = {
  readonly feature: Int32Array;
  readonly threshold: Float64Array;
  readonly left: Int32Array;
  readonly right: Int32Array;
  readonly leafValue: Float64Array;
};

/**
 * Average path length of an unsuccessful search in a binary search tree with
 * `n` nodes, as used to normalise isolation depths (Liu, Ting and Zhou, 2008).
 */
function averagePathLength(n: number): number {
  if (n <= 1) return 0;
  if (n === 2) return 1;
  return 2 * (Math.log(n - 1) + EULER_GAMMA) - (2 * (n - 1)) / n;
}

function isValidMaxSamples(value: unknown): boolean {
  if (value === "auto") return true;
  if (typeof value !== "number") return false;
  if (Number.isInteger(value)) return value >= 1;
  return value > 0 && value < 1;
}

function isValidContamination(value: unknown): boolean {
  return value === "auto" || (typeof value === "number" && value > 0 && value <= 0.5);
}

function isValidMaxFeatures(value: unknown): boolean {
  if (typeof value !== "number") return false;
  if (Number.isInteger(value)) return value >= 1;
  return value > 0 && value < 1;
}

function isValidRandomState(value: unknown): boolean {
  return value === undefined || (typeof value === "number" && Number.isFinite(value));
}

function maxSamplesError(value: unknown): InvalidParameterError {
  return new InvalidParameterError(
    `maxSamples must be "auto", an integer >= 1 (number of samples) or a fraction in (0, 1); received ${String(value)}`,
    "maxSamples",
    value
  );
}

function contaminationError(value: unknown): InvalidParameterError {
  return new InvalidParameterError(
    `contamination must be "auto" or a number in (0, 0.5]; received ${String(value)}`,
    "contamination",
    value
  );
}

function maxFeaturesError(value: unknown): InvalidParameterError {
  return new InvalidParameterError(
    `maxFeatures must be a fraction in (0, 1] or an integer >= 1 (number of features); received ${String(value)}`,
    "maxFeatures",
    value
  );
}

function nEstimatorsError(value: unknown): InvalidParameterError {
  return new InvalidParameterError(
    `nEstimators must be an integer >= 1; received ${String(value)}`,
    "nEstimators",
    value
  );
}

function randomStateError(value: unknown): InvalidParameterError {
  return new InvalidParameterError(
    `randomState must be a finite number; received ${String(value)}`,
    "randomState",
    value
  );
}

/**
 * Isolation Forest for anomaly/outlier detection.
 *
 * Isolates observations by randomly selecting a feature and then randomly
 * selecting a split value between the max and min of the selected feature.
 * Anomalies require fewer splits to be isolated, resulting in shorter path lengths.
 *
 * Follows scikit-learn's `IsolationForest`: every tree is grown on a subsample
 * drawn without replacement, trees are limited to depth `ceil(log2(maxSamples))`,
 * constant features are skipped when choosing a split, and
 * `scoreSamples` returns the negated anomaly score `-2^(-E[h(x)] / c(maxSamples))`
 * (the lower, the more abnormal). With `contamination: "auto"` a sample is an
 * outlier when its anomaly score is above 0.5; otherwise the threshold is the
 * `contamination` percentile of the training scores.
 *
 * @example
 * ```ts
 * import { IsolationForest } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0], [0.1, -0.1], [0.2, 0.1], [100, 100]]);
 * const ifo = new IsolationForest({ nEstimators: 100, contamination: 0.25, randomState: 0 });
 * ifo.fit(X);
 * const labels = ifo.predict(X); // -1 for outliers, 1 for inliers
 * ```
 */
export class IsolationForest implements OutlierDetector {
  private nEstimators: number;
  private maxSamples: number | "auto";
  private contamination: number | "auto";
  private maxFeatures: number;
  private randomState: number | undefined;

  private trees: FlatTree[] = [];
  private maxSamples_ = 0;
  private offset_ = -0.5;
  private nFeaturesIn_ = 0;
  private fitted = false;

  /**
   * @param options.nEstimators - Number of isolation trees (default: 100)
   * @param options.maxSamples - Samples drawn per tree: `"auto"` (default, `min(256, nSamples)`),
   *   an integer count, or a fraction in (0, 1) of the training samples
   * @param options.contamination - Expected proportion of outliers in (0, 0.5], or `"auto"` (default)
   * @param options.maxFeatures - Features drawn per tree: a fraction in (0, 1] (default 1.0 = all)
   *   or an integer count (clamped to the number of features)
   * @param options.randomState - Seed for reproducible forests; uses the global RNG when omitted
   * @throws {InvalidParameterError} If any option is out of range
   */
  constructor(
    options: {
      readonly nEstimators?: number;
      readonly maxSamples?: number | "auto";
      readonly contamination?: number | "auto";
      readonly maxFeatures?: number;
      readonly randomState?: number;
    } = {}
  ) {
    const nEstimators = options.nEstimators ?? 100;
    const maxSamples = options.maxSamples ?? "auto";
    const contamination = options.contamination ?? "auto";
    const maxFeatures = options.maxFeatures ?? 1.0;

    if (!Number.isInteger(nEstimators) || nEstimators < 1) throw nEstimatorsError(nEstimators);
    if (!isValidMaxSamples(maxSamples)) throw maxSamplesError(maxSamples);
    if (!isValidContamination(contamination)) throw contaminationError(contamination);
    if (!isValidMaxFeatures(maxFeatures)) throw maxFeaturesError(maxFeatures);
    if (!isValidRandomState(options.randomState)) throw randomStateError(options.randomState);

    this.nEstimators = nEstimators;
    this.maxSamples = maxSamples;
    this.contamination = contamination;
    this.maxFeatures = maxFeatures;
    this.randomState = options.randomState;
  }

  private createRNG(): () => number {
    const seed = this.randomState;
    if (seed !== undefined) {
      const gen = new __SeededRandom(__seedToUint64(seed));
      return () => gen.next();
    }
    return __random;
  }

  private resolveMaxSamples(nSamples: number): number {
    const ms = this.maxSamples;
    if (ms === "auto") return Math.min(256, nSamples);
    if (Number.isInteger(ms)) return Math.min(ms, nSamples);
    return Math.max(1, Math.floor(ms * nSamples));
  }

  private resolveMaxFeatures(nFeatures: number): number {
    const mf = this.maxFeatures;
    if (mf === 1) return nFeatures;
    if (Number.isInteger(mf)) return Math.min(mf, nFeatures);
    return Math.max(1, Math.floor(mf * nFeatures));
  }

  /**
   * Grow one isolation tree on the sample ids `sample[0..sample.length)`.
   * `sample` is partitioned in place; `cand` holds the candidate features of
   * this tree and is reordered while features are drawn.
   */
  private buildTree(
    data: Float64Array,
    nFeatures: number,
    sample: Int32Array,
    cand: Int32Array,
    maxDepth: number,
    rng: () => number
  ): FlatTree {
    const feature: number[] = [];
    const threshold: number[] = [];
    const left: number[] = [];
    const right: number[] = [];
    const leafValue: number[] = [];

    const grow = (lo: number, hi: number, depth: number): number => {
      const node = feature.length;
      feature.push(-1);
      threshold.push(0);
      left.push(-1);
      right.push(-1);
      leafValue.push(averagePathLength(hi - lo));
      if (hi - lo <= 1 || depth >= maxDepth) return node;

      // Draw candidate features without replacement until one is not constant
      // on this node's samples (constant features cannot split anything).
      let feat = -1;
      let minVal = 0;
      let maxVal = 0;
      for (let k = 0; k < cand.length; k++) {
        const j = k + __randomBelow(rng, cand.length - k);
        const f = cand[j] as number;
        cand[j] = cand[k] as number;
        cand[k] = f;
        let mn = Infinity;
        let mx = -Infinity;
        for (let s = lo; s < hi; s++) {
          const v = data[(sample[s] as number) * nFeatures + f] as number;
          if (v < mn) mn = v;
          if (v > mx) mx = v;
        }
        if (mn < mx) {
          feat = f;
          minVal = mn;
          maxVal = mx;
          break;
        }
      }
      if (feat < 0) return node;

      // Uniform split value in [min, max). The convex combination avoids
      // overflow of (max - min); a value that rounds up to max falls back to
      // min so that both sides stay non-empty.
      const r = rng();
      let split = minVal * (1 - r) + maxVal * r;
      if (!(split >= minVal && split < maxVal)) split = minVal;

      let i = lo;
      let j = hi - 1;
      while (i <= j) {
        if ((data[(sample[i] as number) * nFeatures + feat] as number) <= split) {
          i++;
        } else {
          const tmp = sample[i] as number;
          sample[i] = sample[j] as number;
          sample[j] = tmp;
          j--;
        }
      }

      feature[node] = feat;
      threshold[node] = split;
      left[node] = grow(lo, i, depth + 1);
      right[node] = grow(i, hi, depth + 1);
      return node;
    };

    grow(0, sample.length, 0);
    return {
      feature: Int32Array.from(feature),
      threshold: Float64Array.from(threshold),
      left: Int32Array.from(left),
      right: Int32Array.from(right),
      leafValue: Float64Array.from(leafValue),
    };
  }

  private static pathLength(tree: FlatTree, data: Float64Array, rowOffset: number): number {
    let node = 0;
    let depth = 0;
    for (;;) {
      const f = tree.feature[node] as number;
      if (f < 0) return depth + (tree.leafValue[node] as number);
      node =
        (data[rowOffset + f] as number) <= (tree.threshold[node] as number)
          ? (tree.left[node] as number)
          : (tree.right[node] as number);
      depth++;
    }
  }

  /** Anomaly score in (0, 1] of one row; higher means more abnormal. */
  private static anomalyScore(
    trees: readonly FlatTree[],
    maxSamples: number,
    data: Float64Array,
    rowOffset: number
  ): number {
    const c = averagePathLength(maxSamples);
    // With a single sample per tree every tree is one leaf and carries no
    // information; report the neutral score instead of dividing by zero.
    if (c === 0) return 0.5;
    let total = 0;
    for (const tree of trees) total += IsolationForest.pathLength(tree, data, rowOffset);
    return 2 ** (-total / trees.length / c);
  }

  private scoreRows(data: Float64Array, nSamples: number, nFeatures: number): Float64Array {
    const out = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      out[i] = -IsolationForest.anomalyScore(this.trees, this.maxSamples_, data, i * nFeatures);
    }
    return out;
  }

  /**
   * Fit the forest. Calling `fit` again discards the previous forest.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (exists for API compatibility)
   * @returns this
   * @throws {ShapeError} If X is not 2D
   * @throws {DataValidationError} If X is empty, non-contiguous or contains NaN/Inf
   */
  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const data = toFloat64View(X);

    const maxSamples = this.resolveMaxSamples(nSamples);
    const nFeaturesTree = this.resolveMaxFeatures(nFeatures);
    const maxDepth = Math.ceil(Math.log2(Math.max(maxSamples, 2)));
    const rng = this.createRNG();

    const perm = new Int32Array(nSamples);
    for (let i = 0; i < nSamples; i++) perm[i] = i;
    const featPerm = new Int32Array(nFeatures);
    for (let i = 0; i < nFeatures; i++) featPerm[i] = i;

    const trees: FlatTree[] = [];
    for (let t = 0; t < this.nEstimators; t++) {
      // Partial Fisher-Yates: the first `maxSamples` entries are a uniform
      // subsample without replacement.
      for (let i = 0; i < maxSamples; i++) {
        const j = i + __randomBelow(rng, nSamples - i);
        const tmp = perm[i] as number;
        perm[i] = perm[j] as number;
        perm[j] = tmp;
      }
      for (let i = 0; i < nFeaturesTree; i++) {
        const j = i + __randomBelow(rng, nFeatures - i);
        const tmp = featPerm[i] as number;
        featPerm[i] = featPerm[j] as number;
        featPerm[j] = tmp;
      }
      const sample = perm.slice(0, maxSamples);
      const cand = featPerm.slice(0, nFeaturesTree);
      trees.push(this.buildTree(data, nFeatures, sample, cand, maxDepth, rng));
    }

    this.trees = trees;
    this.maxSamples_ = maxSamples;
    this.nFeaturesIn_ = nFeatures;

    if (this.contamination === "auto") {
      this.offset_ = -0.5;
    } else {
      // Linear-interpolated percentile of the training scores (numpy default).
      const scores = this.scoreRows(data, nSamples, nFeatures).sort();
      this.offset_ = percentileSorted(scores, 100 * this.contamination);
    }

    this.fitted = true;
    return this;
  }

  /**
   * Opposite of the anomaly score, shifted so that negative values are outliers.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns `scoreSamples(X) - offset` of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   */
  decisionFunction(X: Tensor): Tensor {
    const scores = this.scoreSamplesRaw(X, "decisionFunction");
    for (let i = 0; i < scores.length; i++) scores[i] = (scores[i] as number) - this.offset_;
    return tensor(scores, { dtype: "float64" });
  }

  private scoreSamplesRaw(X: Tensor, method: string): Float64Array {
    if (!this.fitted) {
      throw new NotFittedError(`IsolationForest must be fitted before ${method}`);
    }
    validatePredictInputs(X, this.nFeaturesIn_, "IsolationForest");
    return this.scoreRows(toFloat64View(X), X.shape[0] ?? 0, this.nFeaturesIn_);
  }

  /**
   * Predict whether each sample is an outlier.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns int32 labels of shape (n_samples,): -1 for outliers, 1 for inliers
   * @throws {NotFittedError} If the model has not been fitted
   */
  predict(X: Tensor): Tensor {
    const scores = this.scoreSamplesRaw(X, "prediction");
    const labels = new Int32Array(scores.length);
    for (let i = 0; i < scores.length; i++) {
      labels[i] = (scores[i] as number) < this.offset_ ? -1 : 1;
    }
    return tensor(labels, { dtype: "int32" });
  }

  fitPredict(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    return this.predict(X);
  }

  /**
   * Anomaly score of each sample (lower is more abnormal).
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Scores in [-1, 0) of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   */
  scoreSamples(X: Tensor): Tensor {
    return tensor(this.scoreSamplesRaw(X, "scoring"), { dtype: "float64" });
  }

  /**
   * Decision threshold in `scoreSamples` units: samples scoring below it are outliers.
   * Equals -0.5 when `contamination` is `"auto"`.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get offset(): number {
    if (!this.fitted) {
      throw new NotFittedError("IsolationForest must be fitted before accessing offset");
    }
    return this.offset_;
  }

  getParams(): Record<string, unknown> {
    return {
      nEstimators: this.nEstimators,
      maxSamples: this.maxSamples,
      contamination: this.contamination,
      maxFeatures: this.maxFeatures,
      randomState: this.randomState,
    };
  }

  /**
   * Set parameters. All values are validated before any of them is applied.
   *
   * @throws {InvalidParameterError} If a name is unknown or a value is out of range
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nEstimators":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw nEstimatorsError(value);
          }
          break;
        case "maxSamples":
          if (!isValidMaxSamples(value)) throw maxSamplesError(value);
          break;
        case "contamination":
          if (!isValidContamination(value)) throw contaminationError(value);
          break;
        case "maxFeatures":
          if (!isValidMaxFeatures(value)) throw maxFeaturesError(value);
          break;
        case "randomState":
          if (!isValidRandomState(value)) throw randomStateError(value);
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nEstimators":
          this.nEstimators = value as number;
          break;
        case "maxSamples":
          this.maxSamples = value as number | "auto";
          break;
        case "contamination":
          this.contamination = value as number | "auto";
          break;
        case "maxFeatures":
          this.maxFeatures = value as number;
          break;
        case "randomState":
          this.randomState = value as number | undefined;
          break;
      }
    }
    return this;
  }

  /**
   * Create an unfitted copy with the same parameters.
   */
  clone(): IsolationForest {
    return new IsolationForest({
      nEstimators: this.nEstimators,
      maxSamples: this.maxSamples,
      contamination: this.contamination,
      maxFeatures: this.maxFeatures,
      ...(this.randomState !== undefined ? { randomState: this.randomState } : {}),
    });
  }
}
