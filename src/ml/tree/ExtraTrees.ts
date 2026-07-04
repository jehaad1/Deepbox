/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import {
  DataValidationError,
  DeepboxError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __random } from "../../random/random";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier, Regressor } from "../base";

type TreeNode = {
  readonly isLeaf: boolean;
  readonly prediction?: number | undefined;
  readonly classProbabilities?: number[] | undefined;
  readonly featureIndex?: number;
  readonly threshold?: number;
  readonly left?: TreeNode;
  readonly right?: TreeNode;
  readonly nSamples?: number;
  readonly impurity?: number;
};

type ExtraTreesOptions = {
  readonly nEstimators?: number;
  readonly maxDepth?: number;
  readonly minSamplesSplit?: number;
  readonly minSamplesLeaf?: number;
  readonly maxFeatures?: "sqrt" | "log2" | number;
  readonly bootstrap?: boolean;
  readonly randomState?: number;
};

function createRNG(seed?: number): () => number {
  if (seed === undefined) return __random;
  let s = seed;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}

function validateOptions(opts: ExtraTreesOptions): void {
  const nEst = opts.nEstimators ?? 100;
  if (!Number.isInteger(nEst) || nEst < 1) {
    throw new InvalidParameterError(
      `nEstimators must be an integer >= 1; received ${nEst}`,
      "nEstimators",
      nEst
    );
  }
  const md = opts.maxDepth ?? 10;
  if (!Number.isInteger(md) || md < 1) {
    throw new InvalidParameterError(
      `maxDepth must be an integer >= 1; received ${md}`,
      "maxDepth",
      md
    );
  }
  const mss = opts.minSamplesSplit ?? 2;
  if (!Number.isInteger(mss) || mss < 2) {
    throw new InvalidParameterError(
      `minSamplesSplit must be an integer >= 2; received ${mss}`,
      "minSamplesSplit",
      mss
    );
  }
  const msl = opts.minSamplesLeaf ?? 1;
  if (!Number.isInteger(msl) || msl < 1) {
    throw new InvalidParameterError(
      `minSamplesLeaf must be an integer >= 1; received ${msl}`,
      "minSamplesLeaf",
      msl
    );
  }
  const mf = opts.maxFeatures;
  if (mf !== undefined && typeof mf === "number") {
    if (!Number.isInteger(mf) || mf < 1) {
      throw new InvalidParameterError(
        `maxFeatures must be an integer >= 1, "sqrt", or "log2"; received ${mf}`,
        "maxFeatures",
        mf
      );
    }
  } else if (mf !== undefined && mf !== "sqrt" && mf !== "log2") {
    throw new InvalidParameterError(
      `maxFeatures must be "sqrt", "log2", or a positive integer; received ${String(mf)}`,
      "maxFeatures",
      mf
    );
  }
  if (opts.randomState !== undefined && !Number.isFinite(opts.randomState)) {
    throw new InvalidParameterError(
      `randomState must be a finite number; received ${String(opts.randomState)}`,
      "randomState",
      opts.randomState
    );
  }
}

function resolveMaxFeatures(
  maxFeatures: "sqrt" | "log2" | number | undefined,
  nFeatures: number,
  defaultVal: "sqrt" | "log2" | number
): number {
  const mf = maxFeatures ?? defaultVal;
  let n: number;
  if (typeof mf === "number") {
    n = Math.min(mf, nFeatures);
  } else if (mf === "sqrt") {
    n = Math.floor(Math.sqrt(nFeatures));
  } else {
    n = Math.floor(Math.log2(nFeatures));
  }
  return Math.max(1, n);
}

function extractData(
  X: Tensor,
  y: Tensor
): { XData: number[][]; yData: number[]; nSamples: number; nFeatures: number } {
  const nSamples = X.shape[0] ?? 0;
  const nFeatures = X.shape[1] ?? 0;
  const XData: number[][] = [];
  const yData: number[] = [];
  for (let i = 0; i < nSamples; i++) {
    const row: number[] = [];
    for (let j = 0; j < nFeatures; j++) {
      row.push(Number(X.data[X.offset + i * nFeatures + j]));
    }
    XData.push(row);
    yData.push(Number(y.data[y.offset + i]));
  }
  return { XData, yData, nSamples, nFeatures };
}

// ── Classification tree with random thresholds ──────────────────────

function giniFromCounts(counts: Map<number, number>, n: number): number {
  let impurity = 1.0;
  for (const count of counts.values()) {
    const p = count / n;
    impurity -= p * p;
  }
  return impurity;
}

function buildClassificationTree(
  XData: number[][],
  yData: number[],
  indices: number[],
  depth: number,
  maxDepth: number,
  minSamplesSplit: number,
  minSamplesLeaf: number,
  nSelectFeatures: number,
  classLabels: number[],
  rng: () => number
): TreeNode {
  const n = indices.length;

  const getClassProbs = (): number[] => {
    const labelIndex = new Map<number, number>();
    for (let i = 0; i < classLabels.length; i++) {
      const v = classLabels[i];
      if (v !== undefined) labelIndex.set(v, i);
    }
    const counts = new Array<number>(classLabels.length).fill(0);
    for (const idx of indices) {
      const label = yData[idx] ?? 0;
      const li = labelIndex.get(label);
      if (li !== undefined) counts[li] = (counts[li] ?? 0) + 1;
    }
    return counts.map((c) => c / n);
  };

  const getMajority = (): number => {
    const counts = new Map<number, number>();
    for (const i of indices) {
      const label = yData[i] ?? 0;
      counts.set(label, (counts.get(label) ?? 0) + 1);
    }
    let maxCount = 0;
    let maxLabel = 0;
    for (const [label, count] of counts) {
      if (count > maxCount) {
        maxCount = count;
        maxLabel = label;
      }
    }
    return maxLabel;
  };

  // Stopping conditions
  if (depth >= maxDepth || n < minSamplesSplit || n < minSamplesLeaf) {
    return {
      isLeaf: true,
      prediction: getMajority(),
      classProbabilities: getClassProbs(),
      nSamples: n,
    };
  }

  const classes = new Set(indices.map((i) => yData[i]));
  if (classes.size === 1) {
    return {
      isLeaf: true,
      prediction: yData[indices[0] ?? 0] ?? 0,
      classProbabilities: getClassProbs(),
      nSamples: n,
    };
  }

  // Select random subset of features
  const nFeatures = XData[0]?.length ?? 0;
  const featurePool = Array.from({ length: nFeatures }, (_, i) => i);
  for (let i = 0; i < Math.min(nSelectFeatures, nFeatures); i++) {
    const j = i + Math.floor(rng() * (nFeatures - i));
    const tmp = featurePool[i]!;
    featurePool[i] = featurePool[j]!;
    featurePool[j] = tmp;
  }
  const candidateFeatures = featurePool.slice(0, nSelectFeatures);

  // ExtraTrees key difference: pick a RANDOM threshold for each feature
  let bestGini = Infinity;
  let bestFeature = 0;
  let bestThreshold = 0;
  let bestLeft: number[] = [];
  let bestRight: number[] = [];

  for (const f of candidateFeatures) {
    // Find min and max of this feature across indices
    let fMin = Infinity;
    let fMax = -Infinity;
    for (const idx of indices) {
      const val = XData[idx]?.[f] ?? 0;
      if (val < fMin) fMin = val;
      if (val > fMax) fMax = val;
    }
    if (fMin >= fMax) continue; // No variation

    // Random threshold between min and max
    const threshold = fMin + rng() * (fMax - fMin);

    const leftIdx: number[] = [];
    const rightIdx: number[] = [];
    for (const idx of indices) {
      if ((XData[idx]?.[f] ?? 0) <= threshold) {
        leftIdx.push(idx);
      } else {
        rightIdx.push(idx);
      }
    }

    if (leftIdx.length < minSamplesLeaf || rightIdx.length < minSamplesLeaf) {
      continue;
    }

    // Calculate weighted Gini
    const leftCounts = new Map<number, number>();
    for (const idx of leftIdx) {
      const label = yData[idx] ?? 0;
      leftCounts.set(label, (leftCounts.get(label) ?? 0) + 1);
    }
    const rightCounts = new Map<number, number>();
    for (const idx of rightIdx) {
      const label = yData[idx] ?? 0;
      rightCounts.set(label, (rightCounts.get(label) ?? 0) + 1);
    }

    const weightedGini =
      (leftIdx.length * giniFromCounts(leftCounts, leftIdx.length) +
        rightIdx.length * giniFromCounts(rightCounts, rightIdx.length)) /
      n;

    if (weightedGini < bestGini) {
      bestGini = weightedGini;
      bestFeature = f;
      bestThreshold = threshold;
      bestLeft = leftIdx;
      bestRight = rightIdx;
    }
  }

  if (bestLeft.length === 0 || bestRight.length === 0) {
    return {
      isLeaf: true,
      prediction: getMajority(),
      classProbabilities: getClassProbs(),
      nSamples: n,
    };
  }

  const left = buildClassificationTree(
    XData,
    yData,
    bestLeft,
    depth + 1,
    maxDepth,
    minSamplesSplit,
    minSamplesLeaf,
    nSelectFeatures,
    classLabels,
    rng
  );
  const right = buildClassificationTree(
    XData,
    yData,
    bestRight,
    depth + 1,
    maxDepth,
    minSamplesSplit,
    minSamplesLeaf,
    nSelectFeatures,
    classLabels,
    rng
  );

  return {
    isLeaf: false,
    featureIndex: bestFeature,
    threshold: bestThreshold,
    left,
    right,
    nSamples: n,
  };
}

// ── Regression tree with random thresholds ──────────────────────────

function mseForValues(vals: number[]): number {
  if (vals.length === 0) return 0;
  let sum = 0;
  for (const v of vals) sum += v;
  const mean = sum / vals.length;
  let mse = 0;
  for (const v of vals) mse += (v - mean) ** 2;
  return mse / vals.length;
}

function buildRegressionTree(
  XData: number[][],
  yData: number[],
  indices: number[],
  depth: number,
  maxDepth: number,
  minSamplesSplit: number,
  minSamplesLeaf: number,
  nSelectFeatures: number,
  rng: () => number
): TreeNode {
  const n = indices.length;

  const getMean = (): number => {
    let sum = 0;
    for (const idx of indices) sum += yData[idx] ?? 0;
    return sum / n;
  };

  if (depth >= maxDepth || n < minSamplesSplit || n < minSamplesLeaf) {
    return { isLeaf: true, prediction: getMean(), nSamples: n };
  }

  // Check if all y-values are the same
  const firstY = yData[indices[0] ?? 0] ?? 0;
  let allSame = true;
  for (const idx of indices) {
    if ((yData[idx] ?? 0) !== firstY) {
      allSame = false;
      break;
    }
  }
  if (allSame) {
    return { isLeaf: true, prediction: firstY, nSamples: n };
  }

  const nFeatures = XData[0]?.length ?? 0;
  const featurePool = Array.from({ length: nFeatures }, (_, i) => i);
  for (let i = 0; i < Math.min(nSelectFeatures, nFeatures); i++) {
    const j = i + Math.floor(rng() * (nFeatures - i));
    const tmp = featurePool[i]!;
    featurePool[i] = featurePool[j]!;
    featurePool[j] = tmp;
  }
  const candidateFeatures = featurePool.slice(0, nSelectFeatures);

  let bestMSE = Infinity;
  let bestFeature = 0;
  let bestThreshold = 0;
  let bestLeft: number[] = [];
  let bestRight: number[] = [];

  for (const f of candidateFeatures) {
    let fMin = Infinity;
    let fMax = -Infinity;
    for (const idx of indices) {
      const val = XData[idx]?.[f] ?? 0;
      if (val < fMin) fMin = val;
      if (val > fMax) fMax = val;
    }
    if (fMin >= fMax) continue;

    const threshold = fMin + rng() * (fMax - fMin);
    const leftIdx: number[] = [];
    const rightIdx: number[] = [];
    for (const idx of indices) {
      if ((XData[idx]?.[f] ?? 0) <= threshold) {
        leftIdx.push(idx);
      } else {
        rightIdx.push(idx);
      }
    }

    if (leftIdx.length < minSamplesLeaf || rightIdx.length < minSamplesLeaf) {
      continue;
    }

    const leftVals = leftIdx.map((i) => yData[i] ?? 0);
    const rightVals = rightIdx.map((i) => yData[i] ?? 0);
    const weightedMSE =
      (leftIdx.length * mseForValues(leftVals) + rightIdx.length * mseForValues(rightVals)) / n;

    if (weightedMSE < bestMSE) {
      bestMSE = weightedMSE;
      bestFeature = f;
      bestThreshold = threshold;
      bestLeft = leftIdx;
      bestRight = rightIdx;
    }
  }

  if (bestLeft.length === 0 || bestRight.length === 0) {
    return { isLeaf: true, prediction: getMean(), nSamples: n };
  }

  const left = buildRegressionTree(
    XData,
    yData,
    bestLeft,
    depth + 1,
    maxDepth,
    minSamplesSplit,
    minSamplesLeaf,
    nSelectFeatures,
    rng
  );
  const right = buildRegressionTree(
    XData,
    yData,
    bestRight,
    depth + 1,
    maxDepth,
    minSamplesSplit,
    minSamplesLeaf,
    nSelectFeatures,
    rng
  );

  return {
    isLeaf: false,
    featureIndex: bestFeature,
    threshold: bestThreshold,
    left,
    right,
    nSamples: n,
  };
}

// ── Prediction helper ───────────────────────────────────────────────

function predictSample(sample: number[], node: TreeNode): number {
  let current = node;
  while (!current.isLeaf) {
    const val = sample[current.featureIndex ?? 0] ?? 0;
    if (val <= (current.threshold ?? 0)) {
      if (!current.left) throw new DeepboxError("Invalid tree node: missing left child");
      current = current.left;
    } else {
      if (!current.right) throw new DeepboxError("Invalid tree node: missing right child");
      current = current.right;
    }
  }
  return current.prediction ?? 0;
}

function predictProbaFromTree(sample: number[], node: TreeNode): number[] {
  let current = node;
  while (!current.isLeaf) {
    const val = sample[current.featureIndex ?? 0] ?? 0;
    if (val <= (current.threshold ?? 0)) {
      if (!current.left) throw new DeepboxError("Invalid tree node: missing left child");
      current = current.left;
    } else {
      if (!current.right) throw new DeepboxError("Invalid tree node: missing right child");
      current = current.right;
    }
  }
  return current.classProbabilities ?? [];
}

// ── Importance helper ───────────────────────────────────────────────

function computeImportance(node: TreeNode, nFeatures: number): number[] {
  const imp = new Array<number>(nFeatures).fill(0);
  const walk = (n: TreeNode): void => {
    if (n.isLeaf || !n.left || !n.right) return;
    const f = n.featureIndex ?? 0;
    const nS = n.nSamples ?? 0;
    const leftS = n.left.nSamples ?? 0;
    const rightS = n.right.nSamples ?? 0;
    const nImp = n.impurity ?? 0;
    const leftImp = n.left.impurity ?? 0;
    const rightImp = n.right.impurity ?? 0;
    const decrease = nS * nImp - leftS * leftImp - rightS * rightImp;
    imp[f] = (imp[f] ?? 0) + Math.max(0, decrease);
    walk(n.left);
    walk(n.right);
  };
  walk(node);
  let total = 0;
  for (const v of imp) total += v;
  if (total > 0) for (let i = 0; i < nFeatures; i++) imp[i] = (imp[i] ?? 0) / total;
  return imp;
}

// ═════════════════════════════════════════════════════════════════════
// ExtraTreesClassifier
// ═════════════════════════════════════════════════════════════════════

/**
 * Extremely Randomized Trees Classifier.
 *
 * Like Random Forest, but uses random split thresholds instead of
 * searching for the best threshold, and by default does not bootstrap.
 * This leads to lower variance at the cost of slightly higher bias.
 *
 * @example
 * ```ts
 * import { ExtraTreesClassifier } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const clf = new ExtraTreesClassifier({ nEstimators: 100, randomState: 42 });
 * clf.fit(X_train, y_train);
 * const predictions = clf.predict(X_test);
 * ```
 */
export class ExtraTreesClassifier implements Classifier {
  private nEstimators: number;
  private maxDepth: number;
  private minSamplesSplit: number;
  private minSamplesLeaf: number;
  private maxFeatures: "sqrt" | "log2" | number;
  private bootstrap: boolean;
  private randomState: number | undefined;

  private trees: TreeNode[] = [];
  private classLabels?: number[];
  private nFeatures?: number;
  private fitted = false;

  constructor(options: ExtraTreesOptions = {}) {
    validateOptions(options);
    this.nEstimators = options.nEstimators ?? 100;
    this.maxDepth = options.maxDepth ?? 10;
    this.minSamplesSplit = options.minSamplesSplit ?? 2;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 1;
    this.maxFeatures = options.maxFeatures ?? "sqrt";
    this.bootstrap = options.bootstrap ?? false; // ExtraTrees default: no bootstrap
    if (options.randomState !== undefined) this.randomState = options.randomState;
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const { XData, yData, nSamples, nFeatures } = extractData(X, y);
    this.nFeatures = nFeatures;
    this.classLabels = [...new Set(yData)].sort((a, b) => a - b);

    const nSelect = resolveMaxFeatures(this.maxFeatures, nFeatures, "sqrt");
    const rng = createRNG(this.randomState);
    this.trees = [];

    for (let t = 0; t < this.nEstimators; t++) {
      let indices: number[];
      if (this.bootstrap) {
        indices = [];
        for (let i = 0; i < nSamples; i++) indices.push(Math.floor(rng() * nSamples));
      } else {
        indices = Array.from({ length: nSamples }, (_, i) => i);
      }
      const tree = buildClassificationTree(
        XData,
        yData,
        indices,
        0,
        this.maxDepth,
        this.minSamplesSplit,
        this.minSamplesLeaf,
        nSelect,
        this.classLabels,
        rng
      );
      this.trees.push(tree);
    }

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted)
      throw new NotFittedError("ExtraTreesClassifier must be fitted before prediction");
    validatePredictInputs(X, this.nFeatures ?? 0, "ExtraTreesClassifier");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const predictions: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      const sample: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        sample.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      // Majority voting
      const votes = new Map<number, number>();
      for (const tree of this.trees) {
        const pred = predictSample(sample, tree);
        votes.set(pred, (votes.get(pred) ?? 0) + 1);
      }
      let maxVotes = 0;
      let prediction = 0;
      for (const [label, count] of votes) {
        if (count > maxVotes) {
          maxVotes = count;
          prediction = label;
        }
      }
      predictions.push(prediction);
    }

    return tensor(predictions, { dtype: "int32" });
  }

  predictProba(X: Tensor): Tensor {
    if (!this.fitted)
      throw new NotFittedError("ExtraTreesClassifier must be fitted before prediction");
    validatePredictInputs(X, this.nFeatures ?? 0, "ExtraTreesClassifier");
    assertContiguous(X, "X");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const classLabels = this.classLabels ?? [];
    const nClasses = classLabels.length;
    if (nClasses === 0)
      throw new NotFittedError("ExtraTreesClassifier must be fitted before prediction");

    const proba: number[][] = Array.from({ length: nSamples }, () =>
      new Array<number>(nClasses).fill(0)
    );

    for (let i = 0; i < nSamples; i++) {
      const sample: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        sample.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      for (const tree of this.trees) {
        const treeProbs = predictProbaFromTree(sample, tree);
        for (let c = 0; c < Math.min(treeProbs.length, nClasses); c++) {
          const row = proba[i];
          if (row) row[c] = (row[c] ?? 0) + (treeProbs[c] ?? 0);
        }
      }
    }

    const invTrees = this.trees.length === 0 ? 0 : 1 / this.trees.length;
    for (const row of proba) {
      for (let j = 0; j < nClasses; j++) {
        row[j] = (row[j] ?? 0) * invTrees;
      }
    }

    return tensor(proba);
  }

  score(X: Tensor, y: Tensor): number {
    if (!this.fitted)
      throw new NotFittedError("ExtraTreesClassifier must be fitted before scoring");
    if (y.ndim !== 1) throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    assertContiguous(y, "y");
    for (let i = 0; i < y.size; i++) {
      const val = y.data[y.offset + i] ?? 0;
      if (!Number.isFinite(val))
        throw new DataValidationError("y contains non-finite values (NaN or Inf)");
    }
    const predictions = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.size; i++) {
      if (Number(predictions.data[predictions.offset + i]) === Number(y.data[y.offset + i]))
        correct++;
    }
    return correct / y.size;
  }

  get classes(): Tensor | undefined {
    if (!this.fitted || !this.classLabels) return undefined;
    return tensor(this.classLabels, { dtype: "int32" });
  }

  get featureImportances(): Tensor {
    if (!this.fitted || this.trees.length === 0 || this.nFeatures === undefined) {
      throw new NotFittedError("ExtraTreesClassifier must be fitted to access featureImportances");
    }
    const nF = this.nFeatures;
    const avg = new Array<number>(nF).fill(0);
    for (const tree of this.trees) {
      const treeImp = computeImportance(tree, nF);
      for (let j = 0; j < nF; j++) avg[j] = (avg[j] ?? 0) + (treeImp[j] ?? 0);
    }
    const nTrees = this.trees.length;
    let total = 0;
    for (let j = 0; j < nF; j++) {
      avg[j] = (avg[j] ?? 0) / nTrees;
      total += avg[j] ?? 0;
    }
    if (total > 0) for (let j = 0; j < nF; j++) avg[j] = (avg[j] ?? 0) / total;
    return tensor(avg);
  }

  getParams(): Record<string, unknown> {
    return {
      nEstimators: this.nEstimators,
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
      minSamplesLeaf: this.minSamplesLeaf,
      maxFeatures: this.maxFeatures,
      bootstrap: this.bootstrap,
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
        case "maxDepth":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("maxDepth must be an integer >= 1", "maxDepth", value);
          }
          this.maxDepth = value;
          break;
        case "minSamplesSplit":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 2) {
            throw new InvalidParameterError(
              "minSamplesSplit must be an integer >= 2",
              "minSamplesSplit",
              value
            );
          }
          this.minSamplesSplit = value;
          break;
        case "minSamplesLeaf":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError(
              "minSamplesLeaf must be an integer >= 1",
              "minSamplesLeaf",
              value
            );
          }
          this.minSamplesLeaf = value;
          break;
        case "maxFeatures":
          if (value !== "sqrt" && value !== "log2" && (typeof value !== "number" || value < 1)) {
            throw new InvalidParameterError(
              'maxFeatures must be "sqrt", "log2", or a number >= 1',
              "maxFeatures",
              value
            );
          }
          this.maxFeatures = value;
          break;
        case "bootstrap":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("bootstrap must be a boolean", "bootstrap", value);
          }
          this.bootstrap = value;
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

// ═════════════════════════════════════════════════════════════════════
// ExtraTreesRegressor
// ═════════════════════════════════════════════════════════════════════

/**
 * Extremely Randomized Trees Regressor.
 *
 * Like Random Forest Regressor, but uses random split thresholds and
 * by default does not bootstrap. Predictions are averaged across trees.
 *
 * @example
 * ```ts
 * import { ExtraTreesRegressor } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const reg = new ExtraTreesRegressor({ nEstimators: 100, randomState: 42 });
 * reg.fit(X_train, y_train);
 * const predictions = reg.predict(X_test);
 * ```
 */
export class ExtraTreesRegressor implements Regressor {
  private nEstimators: number;
  private maxDepth: number;
  private minSamplesSplit: number;
  private minSamplesLeaf: number;
  private maxFeatures: "sqrt" | "log2" | number;
  private bootstrap: boolean;
  private randomState: number | undefined;

  private trees: TreeNode[] = [];
  private nFeatures?: number;
  private fitted = false;

  constructor(options: ExtraTreesOptions = {}) {
    validateOptions(options);
    this.nEstimators = options.nEstimators ?? 100;
    this.maxDepth = options.maxDepth ?? 10;
    this.minSamplesSplit = options.minSamplesSplit ?? 2;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 1;
    this.maxFeatures = options.maxFeatures ?? "sqrt";
    this.bootstrap = options.bootstrap ?? false;
    if (options.randomState !== undefined) this.randomState = options.randomState;
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const { XData, yData, nSamples, nFeatures } = extractData(X, y);
    this.nFeatures = nFeatures;

    const nSelect = resolveMaxFeatures(this.maxFeatures, nFeatures, "sqrt");
    const rng = createRNG(this.randomState);
    this.trees = [];

    for (let t = 0; t < this.nEstimators; t++) {
      let indices: number[];
      if (this.bootstrap) {
        indices = [];
        for (let i = 0; i < nSamples; i++) indices.push(Math.floor(rng() * nSamples));
      } else {
        indices = Array.from({ length: nSamples }, (_, i) => i);
      }
      const tree = buildRegressionTree(
        XData,
        yData,
        indices,
        0,
        this.maxDepth,
        this.minSamplesSplit,
        this.minSamplesLeaf,
        nSelect,
        rng
      );
      this.trees.push(tree);
    }

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted)
      throw new NotFittedError("ExtraTreesRegressor must be fitted before prediction");
    validatePredictInputs(X, this.nFeatures ?? 0, "ExtraTreesRegressor");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const predictions: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      const sample: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        sample.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      let sum = 0;
      for (const tree of this.trees) sum += predictSample(sample, tree);
      predictions.push(sum / this.trees.length);
    }

    return tensor(predictions);
  }

  score(X: Tensor, y: Tensor): number {
    if (!this.fitted) throw new NotFittedError("ExtraTreesRegressor must be fitted before scoring");
    if (y.ndim !== 1) throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    assertContiguous(y, "y");
    for (let i = 0; i < y.size; i++) {
      const val = y.data[y.offset + i] ?? 0;
      if (!Number.isFinite(val))
        throw new DataValidationError("y contains non-finite values (NaN or Inf)");
    }
    const predictions = this.predict(X);

    let ssRes = 0;
    let ssTot = 0;
    let yMean = 0;
    for (let i = 0; i < y.size; i++) yMean += Number(y.data[y.offset + i]);
    yMean /= y.size;
    for (let i = 0; i < y.size; i++) {
      const yTrue = Number(y.data[y.offset + i]);
      const yPred = Number(predictions.data[predictions.offset + i]);
      ssRes += (yTrue - yPred) ** 2;
      ssTot += (yTrue - yMean) ** 2;
    }
    return ssTot === 0 ? (ssRes === 0 ? 1.0 : 0.0) : 1 - ssRes / ssTot;
  }

  get featureImportances(): Tensor {
    if (!this.fitted || this.trees.length === 0 || this.nFeatures === undefined) {
      throw new NotFittedError("ExtraTreesRegressor must be fitted to access featureImportances");
    }
    const nF = this.nFeatures;
    const avg = new Array<number>(nF).fill(0);
    for (const tree of this.trees) {
      const treeImp = computeImportance(tree, nF);
      for (let j = 0; j < nF; j++) avg[j] = (avg[j] ?? 0) + (treeImp[j] ?? 0);
    }
    const nTrees = this.trees.length;
    let total = 0;
    for (let j = 0; j < nF; j++) {
      avg[j] = (avg[j] ?? 0) / nTrees;
      total += avg[j] ?? 0;
    }
    if (total > 0) for (let j = 0; j < nF; j++) avg[j] = (avg[j] ?? 0) / total;
    return tensor(avg);
  }

  getParams(): Record<string, unknown> {
    return {
      nEstimators: this.nEstimators,
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
      minSamplesLeaf: this.minSamplesLeaf,
      maxFeatures: this.maxFeatures,
      bootstrap: this.bootstrap,
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
        case "maxDepth":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("maxDepth must be an integer >= 1", "maxDepth", value);
          }
          this.maxDepth = value;
          break;
        case "minSamplesSplit":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 2) {
            throw new InvalidParameterError(
              "minSamplesSplit must be an integer >= 2",
              "minSamplesSplit",
              value
            );
          }
          this.minSamplesSplit = value;
          break;
        case "minSamplesLeaf":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError(
              "minSamplesLeaf must be an integer >= 1",
              "minSamplesLeaf",
              value
            );
          }
          this.minSamplesLeaf = value;
          break;
        case "maxFeatures":
          if (value !== "sqrt" && value !== "log2" && (typeof value !== "number" || value < 1)) {
            throw new InvalidParameterError(
              'maxFeatures must be "sqrt", "log2", or a number >= 1',
              "maxFeatures",
              value
            );
          }
          this.maxFeatures = value;
          break;
        case "bootstrap":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("bootstrap must be a boolean", "bootstrap", value);
          }
          this.bootstrap = value;
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
