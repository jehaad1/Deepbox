/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __random } from "../../random/random";
import { validatePredictInputs, validateUnsupervisedFitInputs } from "../_validation";
import type { OutlierDetector } from "../base";

/**
 * Internal node of an isolation tree.
 */
type ITreeNode = {
  readonly isLeaf: boolean;
  readonly size: number;
  readonly featureIndex?: number;
  readonly splitValue?: number;
  readonly left?: ITreeNode;
  readonly right?: ITreeNode;
};

/**
 * Average path length of an unsuccessful search in a BST (harmonic number correction).
 */
function averagePathLength(n: number): number {
  if (n <= 1) return 0;
  if (n === 2) return 1;
  return 2 * (Math.log(n - 1) + 0.5772156649) - (2 * (n - 1)) / n;
}

/**
 * Isolation Forest for anomaly/outlier detection.
 *
 * Isolates observations by randomly selecting a feature and then randomly
 * selecting a split value between the max and min of the selected feature.
 * Anomalies require fewer splits to be isolated, resulting in shorter path lengths.
 *
 * @example
 * ```ts
 * import { IsolationForest } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0], [0.1, -0.1], [0.2, 0.1], [100, 100]]);
 * const ifo = new IsolationForest({ nEstimators: 100, contamination: 0.25 });
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

  private trees: ITreeNode[] = [];
  private maxSamples_: number = 256;
  private threshold_ = 0;
  private nFeaturesIn_ = 0;
  private fitted = false;

  /**
   * @param options.nEstimators - Number of isolation trees (default: 100)
   * @param options.maxSamples - Number of samples to draw per tree (default: 'auto' = min(256, nSamples))
   * @param options.contamination - Expected proportion of outliers (default: 'auto')
   * @param options.maxFeatures - Number of features to draw per tree (default: 1.0 = all)
   * @param options.randomState - Random seed
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
    this.nEstimators = options.nEstimators ?? 100;
    this.maxSamples = options.maxSamples ?? "auto";
    this.contamination = options.contamination ?? "auto";
    this.maxFeatures = options.maxFeatures ?? 1.0;
    this.randomState = options.randomState;

    if (!Number.isInteger(this.nEstimators) || this.nEstimators < 1) {
      throw new InvalidParameterError(
        "nEstimators must be an integer >= 1",
        "nEstimators",
        this.nEstimators
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

  private buildTree(
    data: number[][],
    indices: number[],
    nFeatures: number,
    maxDepth: number,
    depth: number,
    rng: () => number
  ): ITreeNode {
    if (indices.length <= 1 || depth >= maxDepth) {
      return { isLeaf: true, size: indices.length };
    }

    // Pick random feature
    const feat = Math.floor(rng() * nFeatures);
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (const idx of indices) {
      const v = data[idx]![feat] ?? 0;
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }

    if (minVal === maxVal) {
      return { isLeaf: true, size: indices.length };
    }

    const splitVal = minVal + rng() * (maxVal - minVal);
    const leftIdx: number[] = [];
    const rightIdx: number[] = [];
    for (const idx of indices) {
      if ((data[idx]![feat] ?? 0) < splitVal) {
        leftIdx.push(idx);
      } else {
        rightIdx.push(idx);
      }
    }

    return {
      isLeaf: false,
      size: indices.length,
      featureIndex: feat,
      splitValue: splitVal,
      left: this.buildTree(data, leftIdx, nFeatures, maxDepth, depth + 1, rng),
      right: this.buildTree(data, rightIdx, nFeatures, maxDepth, depth + 1, rng),
    };
  }

  private pathLength(x: number[], node: ITreeNode, depth: number): number {
    if (node.isLeaf) {
      return depth + averagePathLength(node.size);
    }
    const feat = node.featureIndex ?? 0;
    const val = x[feat] ?? 0;
    if (val < (node.splitValue ?? 0)) {
      return this.pathLength(x, node.left!, depth + 1);
    }
    return this.pathLength(x, node.right!, depth + 1);
  }

  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nFeatures;

    // Resolve maxSamples
    if (this.maxSamples === "auto") {
      this.maxSamples_ = Math.min(256, nSamples);
    } else {
      this.maxSamples_ = Math.min(this.maxSamples, nSamples);
    }

    const maxDepth = Math.ceil(Math.log2(Math.max(this.maxSamples_, 2)));

    const data: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        row.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      data.push(row);
    }

    this.trees = [];
    const baseSeed = this.randomState;

    for (let t = 0; t < this.nEstimators; t++) {
      const rng = this.createRNG(baseSeed !== undefined ? baseSeed + t * 7919 : undefined);

      // Subsample
      const indices: number[] = [];
      for (let s = 0; s < this.maxSamples_; s++) {
        indices.push(Math.floor(rng() * nSamples));
      }

      this.trees.push(this.buildTree(data, indices, nFeatures, maxDepth, 0, rng));
    }

    // Compute anomaly scores on training data to find threshold
    if (this.contamination !== "auto") {
      const scores: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        scores.push(this.anomalyScore(data[i]!));
      }
      scores.sort((a, b) => b - a); // descending
      const cutoff = Math.floor(this.contamination * nSamples);
      this.threshold_ = scores[Math.max(0, cutoff - 1)] ?? 0.5;
    } else {
      this.threshold_ = 0.5;
    }

    this.fitted = true;
    return this;
  }

  private anomalyScore(x: number[]): number {
    let avgPath = 0;
    for (const tree of this.trees) {
      avgPath += this.pathLength(x, tree, 0);
    }
    avgPath /= this.trees.length;
    return 2 ** (-avgPath / averagePathLength(this.maxSamples_));
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("IsolationForest must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "IsolationForest");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const labels: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      const xi: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        xi.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      const score = this.anomalyScore(xi);
      labels.push(score >= this.threshold_ ? -1 : 1);
    }

    return tensor(labels, { dtype: "int32" });
  }

  fitPredict(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    return this.predict(X);
  }

  scoreSamples(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("IsolationForest must be fitted before scoring");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "IsolationForest");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const scores: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      const xi: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        xi.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      // Return negative anomaly score (more negative = more anomalous)
      scores.push(-this.anomalyScore(xi));
    }

    return tensor(scores);
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

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nEstimators":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError(
              "nEstimators must be an integer >= 1",
              "nEstimators",
              value
            );
          }
          this.nEstimators = value;
          break;
        case "maxSamples":
          if (value !== "auto" && (typeof value !== "number" || value <= 0)) {
            throw new InvalidParameterError(
              'maxSamples must be "auto" or a positive number',
              "maxSamples",
              value
            );
          }
          this.maxSamples = value;
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
        case "maxFeatures":
          if (typeof value !== "number" || value <= 0 || value > 1) {
            throw new InvalidParameterError("maxFeatures must be in (0, 1]", "maxFeatures", value);
          }
          this.maxFeatures = value;
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
