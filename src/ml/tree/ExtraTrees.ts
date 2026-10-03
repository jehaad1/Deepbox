/**
 * @see {@link https://deepbox.dev/docs/ml-tree | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __randomBelow } from "../../random/random";
import { toFloat64View, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier, Regressor } from "../base";
import {
  checkCcpAlpha,
  checkClassWeight,
  checkMaxLeafNodes,
  checkMinImpurityDecrease,
  classWeightPerClass,
  combineWeights,
  type ExpandResult,
  type ForestClassWeight,
  positiveWeightIndices,
  readSampleWeight,
  splitIsWorthwhile,
  type TreeGrowthOptions,
} from "./_growth";
import {
  type ClassificationCriterion,
  checkTreeCriterion,
  checkTreeInteger,
  checkTreeMaxDepth,
  checkTreeMaxFeatures,
  checkTreeRandomState,
  classImpurity,
  classLeaf,
  createTreeRng,
  descend,
  growTree,
  nodeTargetStats,
  normalizedImportances,
  partitionIndices,
  readScoreTargets,
  resolveTreeMaxFeatures,
  type SplitResult,
  TreeFeatureSampler,
  type TreeMaxFeatures,
  type TreeNode,
  toColumnMajor,
  treeLabelDType,
  treeLabelTensor,
} from "./DecisionTree";

/** Options shared by {@link ExtraTreesClassifier} and {@link ExtraTreesRegressor}. */
export type ExtraTreesOptions = {
  /** Number of trees. Default 100. */
  readonly nEstimators?: number;
  /** Maximum depth of every tree; `Infinity` grows them until the leaves are pure. Default 10. */
  readonly maxDepth?: number;
  /** Minimum number of samples a node needs to be split. Default 2. */
  readonly minSamplesSplit?: number;
  /** Minimum number of samples every leaf must keep. Default 1. */
  readonly minSamplesLeaf?: number;
  /**
   * Features drawn at each split: an exact count, `"sqrt"` or `"log2"`. Default `"sqrt"` for the
   * classifier and all features for the regressor.
   */
  readonly maxFeatures?: TreeMaxFeatures;
  /** Train every tree on a bootstrap sample. Default `false`. */
  readonly bootstrap?: boolean;
  /** Seed for the trees. Without it the global Deepbox generator is used. */
  readonly randomState?: number;
} & TreeGrowthOptions;

/** Options of {@link ExtraTreesClassifier}. */
export type ExtraTreesClassifierOptions = ExtraTreesOptions & {
  /** Split quality measure used to pick the best of the random candidate splits. Default `"gini"`. */
  readonly criterion?: ClassificationCriterion;
  /**
   * Class weights: `"balanced"`, `"balanced_subsample"` (the balanced weights recomputed on the
   * bootstrap sample of every tree) or a map from class label to weight. They multiply the
   * `sampleWeight` given to `fit`. Default: all classes weigh 1.
   */
  readonly classWeight?: ForestClassWeight;
};

type EnsembleParams = {
  nEstimators: number;
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
  maxFeatures: TreeMaxFeatures | undefined;
  bootstrap: boolean;
  randomState: number | undefined;
  minImpurityDecrease: number;
  maxLeafNodes: number | undefined;
  ccpAlpha: number;
};

function checkBootstrap(value: unknown): boolean {
  if (typeof value !== "boolean") {
    throw new InvalidParameterError(
      `bootstrap must be a boolean; received ${String(value)}`,
      "bootstrap",
      value
    );
  }
  return value;
}

/** Validate the options shared by both estimators and fill in the defaults. */
function resolveOptions(
  options: ExtraTreesOptions,
  defaultMaxFeatures: TreeMaxFeatures | undefined
): EnsembleParams {
  return {
    nEstimators: checkTreeInteger("nEstimators", options.nEstimators ?? 100, 1),
    maxDepth: checkTreeMaxDepth(options.maxDepth ?? 10),
    minSamplesSplit: checkTreeInteger("minSamplesSplit", options.minSamplesSplit ?? 2, 2),
    minSamplesLeaf: checkTreeInteger("minSamplesLeaf", options.minSamplesLeaf ?? 1, 1),
    maxFeatures: checkTreeMaxFeatures(options.maxFeatures ?? defaultMaxFeatures),
    bootstrap: checkBootstrap(options.bootstrap ?? false),
    randomState: checkTreeRandomState(options.randomState),
    minImpurityDecrease: checkMinImpurityDecrease(options.minImpurityDecrease ?? 0),
    maxLeafNodes: checkMaxLeafNodes(options.maxLeafNodes),
    ccpAlpha: checkCcpAlpha(options.ccpAlpha ?? 0),
  };
}

/** Apply one shared `setParams` entry to `next`; false when the key is not a shared one. */
function applySharedParam(next: EnsembleParams, key: string, value: unknown): boolean {
  switch (key) {
    case "nEstimators":
      next.nEstimators = checkTreeInteger("nEstimators", value, 1);
      return true;
    case "maxDepth":
      next.maxDepth = checkTreeMaxDepth(value);
      return true;
    case "minSamplesSplit":
      next.minSamplesSplit = checkTreeInteger("minSamplesSplit", value, 2);
      return true;
    case "minSamplesLeaf":
      next.minSamplesLeaf = checkTreeInteger("minSamplesLeaf", value, 1);
      return true;
    case "maxFeatures":
      next.maxFeatures = checkTreeMaxFeatures(value);
      return true;
    case "bootstrap":
      next.bootstrap = checkBootstrap(value);
      return true;
    case "randomState":
      next.randomState = checkTreeRandomState(value);
      return true;
    case "minImpurityDecrease":
      next.minImpurityDecrease = checkMinImpurityDecrease(value);
      return true;
    case "maxLeafNodes":
      next.maxLeafNodes = checkMaxLeafNodes(value);
      return true;
    case "ccpAlpha":
      next.ccpAlpha = checkCcpAlpha(value);
      return true;
    default:
      return false;
  }
}

/**
 * The entries of `indices` whose weight is positive (all of them without weights). When a
 * bootstrap sample holds only zero-weight rows, all positive-weight rows are used instead.
 */
function withoutZeroWeights(indices: Int32Array, weights: Float64Array | undefined): Int32Array {
  if (weights === undefined) return indices;
  let count = 0;
  for (let i = 0; i < indices.length; i++) {
    if ((weights[indices[i] as number] as number) > 0) count++;
  }
  if (count === indices.length) return indices;
  if (count === 0) return positiveWeightIndices(weights.length, weights);
  const out = new Int32Array(count);
  let k = 0;
  for (let i = 0; i < indices.length; i++) {
    const idx = indices[i] as number;
    if ((weights[idx] as number) > 0) out[k++] = idx;
  }
  return out;
}

/** Sum of the weights of the rows in `indices` (their count without weights). */
function totalWeightOf(indices: Int32Array, weights: Float64Array | undefined): number {
  if (weights === undefined) return indices.length;
  let total = 0;
  for (let i = 0; i < indices.length; i++) total += weights[indices[i] as number] as number;
  return total;
}

/** Indices of the training rows one tree is grown on. */
function drawSample(nSamples: number, bootstrap: boolean, rng: () => number): Int32Array {
  const indices = new Int32Array(nSamples);
  if (bootstrap) {
    for (let i = 0; i < nSamples; i++) indices[i] = __randomBelow(rng, nSamples);
  } else {
    for (let i = 0; i < nSamples; i++) indices[i] = i;
  }
  return indices;
}

/**
 * Uniform draw from `[lo, hi)` (`lo < hi`), never returning `hi` itself so both sides of the
 * split are non-empty.
 */
function randomThreshold(lo: number, hi: number, rng: () => number): number {
  const r = rng();
  const span = hi - lo;
  const t = Number.isFinite(span) ? lo + r * span : lo * (1 - r) + hi * r;
  return t >= lo && t < hi ? t : lo;
}

// ---------------------------------------------------------------------------
// Tree builders (random thresholds)
// ---------------------------------------------------------------------------

type ClassificationBuild = {
  readonly xc: Float64Array;
  readonly nSamples: number;
  readonly yCode: Int32Array;
  readonly nClasses: number;
  readonly labels: readonly number[];
  readonly isGini: boolean;
  readonly maxDepth: number;
  readonly minSamplesSplit: number;
  readonly minSamplesLeaf: number;
  readonly minImpurityDecrease: number;
  /** Per-row weights, `undefined` when every sample weighs 1. */
  readonly weights: Float64Array | undefined;
  /** Sum of the weights of the rows the current tree is grown on. */
  readonly totalWeight: number;
  readonly sampler: TreeFeatureSampler;
  readonly rng: () => number;
  readonly leftCounts: Float64Array;
};

function expandClassificationNode(
  b: ClassificationBuild,
  indices: Int32Array,
  depth: number
): ExpandResult {
  const n = indices.length;
  const counts = new Float64Array(b.nClasses);
  const { weights } = b;
  let weight = 0;
  for (let i = 0; i < n; i++) {
    const idx = indices[i] as number;
    const w = weights === undefined ? 1 : (weights[idx] as number);
    counts[b.yCode[idx] as number]! += w;
    weight += w;
  }

  let present = 0;
  let totalSq = 0;
  let nodeEntropy = 0;
  for (let c = 0; c < b.nClasses; c++) {
    const count = counts[c] as number;
    if (count > 0) {
      present++;
      totalSq += count * count;
      nodeEntropy -= count * Math.log2(count / weight);
    }
  }
  const impurity = present === 1 ? 0 : classImpurity(counts, weight, b.isGini);
  const makeLeaf = () => classLeaf(counts, weight, n, b.labels, impurity);

  if (depth >= b.maxDepth || n < b.minSamplesSplit || n < 2 * b.minSamplesLeaf || present === 1) {
    return { node: makeLeaf() };
  }

  const split = findRandomClassificationSplit(b, indices, counts, weight, totalSq, nodeEntropy);
  if (split === undefined) return { node: makeLeaf() };
  if (!splitIsWorthwhile(split.decrease, b.totalWeight, b.minImpurityDecrease)) {
    return { node: makeLeaf() };
  }

  const children = partitionIndices(b.xc, split.feature * b.nSamples, indices, split.threshold);
  if (children[0].length === 0 || children[1].length === 0) {
    return { node: makeLeaf() };
  }
  return {
    node: {
      isLeaf: false,
      featureIndex: split.feature,
      threshold: split.threshold,
      nSamples: n,
      weightedImpurityDecrease: split.decrease,
      impurity,
      weightedNSamples: weight,
    },
    children,
    leaf: makeLeaf,
  };
}

function buildClassificationTree(
  b: ClassificationBuild,
  indices: Int32Array,
  growth: { readonly maxLeafNodes: number | undefined; readonly ccpAlpha: number }
): TreeNode {
  return growTree(indices, (idx, depth) => expandClassificationNode(b, idx, depth), growth);
}

/**
 * Draw one random threshold for each candidate feature and keep the best of them. Candidates
 * keep being drawn until `limit` non-constant features were tried, so a node is only turned
 * into a leaf when every feature is constant or no draw satisfies `minSamplesLeaf`.
 */
function findRandomClassificationSplit(
  b: ClassificationBuild,
  indices: Int32Array,
  counts: Float64Array,
  weight: number,
  totalSq: number,
  nodeEntropy: number
): SplitResult | undefined {
  const n = indices.length;
  const { xc, nSamples, yCode, nClasses, leftCounts, sampler, rng, weights } = b;

  let bestScore = Number.NEGATIVE_INFINITY;
  let bestFeature = -1;
  let bestThreshold = 0;

  let visited = 0;
  let usable = 0;
  while (visited < sampler.nFeatures && usable < sampler.limit) {
    const f = sampler.draw(visited);
    visited++;

    const base = f * nSamples;
    let lo = Number.POSITIVE_INFINITY;
    let hi = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < n; i++) {
      const v = xc[base + (indices[i] as number)] as number;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (lo === hi) continue;
    usable++;

    const threshold = randomThreshold(lo, hi, rng);
    leftCounts.fill(0);
    let nLeft = 0;
    let leftWeight = 0;
    for (let i = 0; i < n; i++) {
      const idx = indices[i] as number;
      if ((xc[base + idx] as number) <= threshold) {
        const w = weights === undefined ? 1 : (weights[idx] as number);
        leftCounts[yCode[idx] as number]! += w;
        leftWeight += w;
        nLeft++;
      }
    }
    const nRight = n - nLeft;
    if (nLeft < b.minSamplesLeaf || nRight < b.minSamplesLeaf) continue;
    const rightWeight = weight - leftWeight;
    if (!(leftWeight > 0) || !(rightWeight > 0)) continue;

    let score: number;
    if (b.isGini) {
      let leftSq = 0;
      let rightSq = 0;
      for (let c = 0; c < nClasses; c++) {
        const l = leftCounts[c] as number;
        const r = (counts[c] as number) - l;
        leftSq += l * l;
        rightSq += r * r;
      }
      score = leftSq / leftWeight + rightSq / rightWeight;
    } else {
      let childEntropy = 0;
      for (let c = 0; c < nClasses; c++) {
        const l = leftCounts[c] as number;
        if (l > 0) childEntropy -= l * Math.log2(l / leftWeight);
        const r = (counts[c] as number) - l;
        if (r > 0) childEntropy -= r * Math.log2(r / rightWeight);
      }
      score = -childEntropy;
    }

    if (score > bestScore) {
      bestScore = score;
      bestFeature = f;
      bestThreshold = threshold;
    }
  }

  if (bestFeature < 0) return undefined;
  const decrease = b.isGini ? bestScore - totalSq / weight : nodeEntropy + bestScore;
  return { feature: bestFeature, threshold: bestThreshold, decrease: Math.max(0, decrease) };
}

type RegressionBuild = {
  readonly xc: Float64Array;
  readonly nSamples: number;
  readonly y: Float64Array;
  readonly maxDepth: number;
  readonly minSamplesSplit: number;
  readonly minSamplesLeaf: number;
  readonly minImpurityDecrease: number;
  /** Per-row weights, `undefined` when every sample weighs 1. */
  readonly weights: Float64Array | undefined;
  /** Sum of the weights of the rows the current tree is grown on. */
  readonly totalWeight: number;
  readonly sampler: TreeFeatureSampler;
  readonly rng: () => number;
  readonly yLocal: Float64Array;
};

function expandRegressionNode(
  b: RegressionBuild,
  indices: Int32Array,
  depth: number
): ExpandResult {
  const n = indices.length;
  const { weights } = b;
  const { mean, lo, hi, weight, impurity } = nodeTargetStats(b.y, indices, weights);
  const leaf: TreeNode = {
    isLeaf: true,
    prediction: mean,
    nSamples: n,
    weightedNSamples: weight,
    impurity,
  };

  if (depth >= b.maxDepth || n < b.minSamplesSplit || n < 2 * b.minSamplesLeaf || lo === hi) {
    return { node: leaf };
  }

  // Centered targets avoid the cancellation of sumL^2/WL + sumR^2/WR for large offsets.
  const { yLocal } = b;
  let totalSum = 0;
  for (let i = 0; i < n; i++) {
    const idx = indices[i] as number;
    const v = (b.y[idx] as number) - mean;
    yLocal[i] = v;
    totalSum += (weights === undefined ? 1 : (weights[idx] as number)) * v;
  }

  const split = findRandomRegressionSplit(b, indices, weight, totalSum);
  if (split === undefined) return { node: leaf };
  if (!splitIsWorthwhile(split.decrease, b.totalWeight, b.minImpurityDecrease)) {
    return { node: leaf };
  }

  const children = partitionIndices(b.xc, split.feature * b.nSamples, indices, split.threshold);
  if (children[0].length === 0 || children[1].length === 0) return { node: leaf };
  return {
    node: {
      isLeaf: false,
      featureIndex: split.feature,
      threshold: split.threshold,
      nSamples: n,
      weightedImpurityDecrease: split.decrease,
      impurity,
      weightedNSamples: weight,
    },
    children,
    leaf: () => leaf,
  };
}

function buildRegressionTree(
  b: RegressionBuild,
  indices: Int32Array,
  growth: { readonly maxLeafNodes: number | undefined; readonly ccpAlpha: number }
): TreeNode {
  return growTree(indices, (idx, depth) => expandRegressionNode(b, idx, depth), growth);
}

function findRandomRegressionSplit(
  b: RegressionBuild,
  indices: Int32Array,
  weight: number,
  totalSum: number
): SplitResult | undefined {
  const n = indices.length;
  const { xc, nSamples, yLocal, sampler, rng, weights } = b;

  let bestScore = Number.NEGATIVE_INFINITY;
  let bestFeature = -1;
  let bestThreshold = 0;

  let visited = 0;
  let usable = 0;
  while (visited < sampler.nFeatures && usable < sampler.limit) {
    const f = sampler.draw(visited);
    visited++;

    const base = f * nSamples;
    let lo = Number.POSITIVE_INFINITY;
    let hi = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < n; i++) {
      const v = xc[base + (indices[i] as number)] as number;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (lo === hi) continue;
    usable++;

    const threshold = randomThreshold(lo, hi, rng);
    let leftSum = 0;
    let leftWeight = 0;
    let nLeft = 0;
    for (let i = 0; i < n; i++) {
      const idx = indices[i] as number;
      if ((xc[base + idx] as number) <= threshold) {
        const w = weights === undefined ? 1 : (weights[idx] as number);
        leftSum += w * (yLocal[i] as number);
        leftWeight += w;
        nLeft++;
      }
    }
    const nRight = n - nLeft;
    if (nLeft < b.minSamplesLeaf || nRight < b.minSamplesLeaf) continue;
    const rightWeight = weight - leftWeight;
    if (!(leftWeight > 0) || !(rightWeight > 0)) continue;

    const rightSum = totalSum - leftSum;
    const score = (leftSum * leftSum) / leftWeight + (rightSum * rightSum) / rightWeight;
    if (score > bestScore) {
      bestScore = score;
      bestFeature = f;
      bestThreshold = threshold;
    }
  }

  if (bestFeature < 0) return undefined;
  return {
    feature: bestFeature,
    threshold: bestThreshold,
    decrease: Math.max(0, bestScore - (totalSum * totalSum) / weight),
  };
}

/** Mean of the per-tree normalized importances, renormalized to sum to 1. */
function forestImportances(trees: readonly TreeNode[], nFeatures: number): Tensor {
  const avg = new Float64Array(nFeatures);
  for (const tree of trees) {
    const imp = toFloat64View(normalizedImportances(tree, nFeatures));
    for (let j = 0; j < nFeatures; j++) avg[j] = (avg[j] as number) + (imp[j] as number);
  }
  let total = 0;
  for (let j = 0; j < nFeatures; j++) total += avg[j] as number;
  if (total > 0) for (let j = 0; j < nFeatures; j++) avg[j] = (avg[j] as number) / total;
  return tensor(avg, { dtype: "float64" });
}

function checkScoreSizes(predicted: number, actual: number): void {
  if (predicted !== actual) {
    throw new ShapeError(
      `X and y must have the same number of samples; got X=${predicted}, y=${actual}`
    );
  }
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
 * At every node a random threshold, drawn uniformly between the node's minimum and maximum, is
 * tried for each of `maxFeatures` randomly chosen features, and the candidate with the lowest
 * weighted impurity is used. `predictProba` averages the leaf class frequencies of all trees and
 * `predict` returns the class with the highest average (ties go to the smallest label).
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
  private params: EnsembleParams;
  private criterion: ClassificationCriterion;
  private classWeight: ForestClassWeight | undefined;

  private trees: TreeNode[] = [];
  private classLabels?: number[];
  private labelDType: "int32" | "float64" = "int32";
  private nFeatures?: number;
  private fitted = false;

  /**
   * @param options - Hyperparameters, see {@link ExtraTreesClassifierOptions}
   * @throws {InvalidParameterError} If an option is out of range
   */
  constructor(options: ExtraTreesClassifierOptions = {}) {
    this.params = resolveOptions(options, "sqrt");
    this.criterion = checkTreeCriterion(options.criterion ?? "gini");
    this.classWeight = checkClassWeight(options.classWeight, true);
  }

  /**
   * Fit the ensemble on training data.
   *
   * Every sample counts with the weight `sampleWeight[i] * classWeight[y[i]]` in the impurity
   * of a node and in the leaf class probabilities; `minSamplesSplit` and `minSamplesLeaf` still
   * count samples. Samples with weight 0 are ignored.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Class labels of shape (n_samples,)
   * @param sampleWeight - Optional non-negative weights of shape (n_samples,)
   * @returns this
   * @throws {ShapeError} If X is not 2D, y is not 1D or their sample counts differ
   * @throws {ShapeError} If `sampleWeight` is not 1D with one entry per sample
   * @throws {DataValidationError} If X or y contain NaN/Inf values, or the weights are negative,
   *   not finite or all zero
   * @throws {InvalidParameterError} If `classWeight` names a label that is not in y
   */
  // biome-ignore lint/suspicious/noConfusingVoidType: `void` keeps this compatible with Estimator.fit(X, y, params)
  fit(X: Tensor, y: Tensor, sampleWeightArg?: Tensor | void): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const sampleWeight = readSampleWeight(sampleWeightArg as Tensor | undefined, nSamples);

    const xc = toColumnMajor(toFloat64View(X), nSamples, nFeatures);
    const yv = toFloat64View(y);
    const labels = [...new Set(yv)].sort((a, b) => a - b);
    const codeOf = new Map<number, number>();
    for (let i = 0; i < labels.length; i++) codeOf.set(labels[i] as number, i);
    const yCode = new Int32Array(nSamples);
    for (let i = 0; i < nSamples; i++) yCode[i] = codeOf.get(yv[i] as number) as number;

    const p = this.params;
    const nClasses = labels.length;
    const cw = this.classWeight;
    let baseWeights = sampleWeight;
    if (cw !== undefined && cw !== "balanced_subsample") {
      const classCounts = new Float64Array(nClasses);
      for (let i = 0; i < nSamples; i++) classCounts[yCode[i] as number]! += 1;
      baseWeights = combineWeights(
        sampleWeight,
        classWeightPerClass(cw, labels, classCounts, nSamples),
        yCode
      );
    }
    const rng = createTreeRng(p.randomState);
    const build: ClassificationBuild = {
      xc,
      nSamples,
      yCode,
      nClasses,
      labels,
      isGini: this.criterion === "gini",
      maxDepth: p.maxDepth,
      minSamplesSplit: p.minSamplesSplit,
      minSamplesLeaf: p.minSamplesLeaf,
      minImpurityDecrease: p.minImpurityDecrease,
      weights: baseWeights,
      totalWeight: nSamples,
      sampler: new TreeFeatureSampler(
        nFeatures,
        resolveTreeMaxFeatures(p.maxFeatures, nFeatures),
        rng
      ),
      rng,
      leftCounts: new Float64Array(nClasses),
    };
    const growth = { maxLeafNodes: p.maxLeafNodes, ccpAlpha: p.ccpAlpha };

    const trees: TreeNode[] = [];
    for (let t = 0; t < p.nEstimators; t++) {
      const drawn = drawSample(nSamples, p.bootstrap, rng);
      let weights = baseWeights;
      if (cw === "balanced_subsample") {
        // The balanced weights are computed on this tree's bootstrap sample.
        const classCounts = new Float64Array(nClasses);
        for (let i = 0; i < drawn.length; i++)
          classCounts[yCode[drawn[i] as number] as number]! += 1;
        weights = combineWeights(
          baseWeights,
          classWeightPerClass("balanced", labels, classCounts, drawn.length),
          yCode
        );
      }
      const indices = withoutZeroWeights(drawn, weights);
      if (indices.length === 0) {
        throw new DataValidationError("every sample has a weight of zero");
      }
      trees.push(
        buildClassificationTree(
          { ...build, weights, totalWeight: totalWeightOf(indices, weights) },
          indices,
          growth
        )
      );
    }

    this.trees = trees;
    this.nFeatures = nFeatures;
    this.classLabels = labels;
    this.labelDType = treeLabelDType(labels);
    this.fitted = true;
    return this;
  }

  /** Mean leaf class frequencies over all trees, row-major `(nSamples, nClasses)`. */
  private averageProba(X: Tensor): Float64Array {
    if (!this.fitted || !this.classLabels) {
      throw new NotFittedError("ExtraTreesClassifier must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures ?? 0, "ExtraTreesClassifier");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const nClasses = this.classLabels.length;
    const x = toFloat64View(X);
    const out = new Float64Array(nSamples * nClasses);
    for (let i = 0; i < nSamples; i++) {
      const row = i * nClasses;
      for (const tree of this.trees) {
        const probabilities = descend(tree, x, i * nFeatures).classProbabilities;
        if (!probabilities) continue;
        for (let c = 0; c < nClasses; c++) {
          out[row + c] = (out[row + c] as number) + (probabilities[c] as number);
        }
      }
    }
    const inv = 1 / this.trees.length;
    for (let i = 0; i < out.length; i++) out[i] = (out[i] as number) * inv;
    return out;
  }

  /**
   * Predict class labels: the class with the highest mean probability over the trees.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Labels of shape (n_samples,): `int32` when all classes are integers, `float64`
   *   otherwise
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    const proba = this.averageProba(X);
    const labels = this.classLabels ?? [];
    const nClasses = labels.length;
    const nSamples = nClasses === 0 ? 0 : proba.length / nClasses;
    const out = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      let best = 0;
      for (let c = 1; c < nClasses; c++) {
        if ((proba[i * nClasses + c] as number) > (proba[i * nClasses + best] as number)) best = c;
      }
      out[i] = labels[best] as number;
    }
    return treeLabelTensor(out, this.labelDType);
  }

  /**
   * Predict class probabilities: the leaf class frequencies averaged over the trees.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns `float64` matrix of shape (n_samples, n_classes); columns follow
   *   {@link ExtraTreesClassifier.classes}
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predictProba(X: Tensor): Tensor {
    const proba = this.averageProba(X);
    const nClasses = this.classLabels?.length ?? 0;
    return tensor(proba, { dtype: "float64" }).reshape([proba.length / nClasses, nClasses]);
  }

  /**
   * Mean accuracy on the given test data and labels.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True labels of shape (n_samples,)
   * @returns Accuracy in [0, 1]
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional or sample counts mismatch
   * @throws {DataValidationError} If y is empty or contains NaN/Inf values
   */
  score(X: Tensor, y: Tensor): number {
    if (!this.fitted) {
      throw new NotFittedError("ExtraTreesClassifier must be fitted before scoring");
    }
    const truth = readScoreTargets(y);
    const predictions = toFloat64View(this.predict(X));
    checkScoreSizes(predictions.length, truth.length);
    let correct = 0;
    for (let i = 0; i < truth.length; i++) {
      if (predictions[i] === truth[i]) correct++;
    }
    return correct / truth.length;
  }

  /**
   * Sorted class labels seen during fit (`int32` for integer labels, `float64` otherwise), or
   * undefined if not fitted.
   */
  get classes(): Tensor | undefined {
    if (!this.fitted || !this.classLabels) return undefined;
    return treeLabelTensor(this.classLabels, this.labelDType);
  }

  /** Number of features seen during fit. */
  get nFeatures_(): number | undefined {
    return this.nFeatures;
  }

  /**
   * Mean decrease in impurity per feature, averaged over the trees (sums to 1, or all zeros if no
   * tree split).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get featureImportances(): Tensor {
    if (!this.fitted || this.trees.length === 0 || this.nFeatures === undefined) {
      throw new NotFittedError("ExtraTreesClassifier must be fitted to access featureImportances");
    }
    return forestImportances(this.trees, this.nFeatures);
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    const cw = this.classWeight;
    return {
      ...this.params,
      criterion: this.criterion,
      classWeight: cw === undefined || typeof cw === "string" ? cw : { ...cw },
    };
  }

  /**
   * Set parameters. Either all given parameters are applied or, if one is invalid, none.
   *
   * @param params - Parameters to set
   * @returns this
   * @throws {InvalidParameterError} If a parameter value is invalid or unknown
   */
  setParams(params: Record<string, unknown>): this {
    const next = { ...this.params };
    let criterion = this.criterion;
    let classWeight = this.classWeight;
    for (const [key, value] of Object.entries(params)) {
      if (key === "criterion") criterion = checkTreeCriterion(value);
      else if (key === "classWeight") classWeight = checkClassWeight(value, true);
      else if (!applySharedParam(next, key, value)) {
        throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    this.params = next;
    this.criterion = criterion;
    this.classWeight = classWeight;
    return this;
  }

  /** Create an unfitted copy with the same hyperparameters. */
  clone(): ExtraTreesClassifier {
    return new ExtraTreesClassifier().setParams(this.getParams());
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
 * Every split considers all features unless `maxFeatures` says otherwise.
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
  private params: EnsembleParams;

  private trees: TreeNode[] = [];
  private nFeatures?: number;
  private fitted = false;

  /**
   * @param options - Hyperparameters, see {@link ExtraTreesOptions}
   * @throws {InvalidParameterError} If an option is out of range
   */
  constructor(options: ExtraTreesOptions = {}) {
    this.params = resolveOptions(options, undefined);
  }

  /**
   * Fit the ensemble on training data.
   *
   * With `sampleWeight` the node means, the impurity and the split scores use weighted sums,
   * while `minSamplesSplit` and `minSamplesLeaf` still count samples. Samples with weight 0 are
   * ignored.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target values of shape (n_samples,)
   * @param sampleWeight - Optional non-negative weights of shape (n_samples,)
   * @returns this
   * @throws {ShapeError} If X is not 2D, y is not 1D or their sample counts differ
   * @throws {ShapeError} If `sampleWeight` is not 1D with one entry per sample
   * @throws {DataValidationError} If X or y contain NaN/Inf values, or the weights are negative,
   *   not finite or all zero
   */
  // biome-ignore lint/suspicious/noConfusingVoidType: `void` keeps this compatible with Estimator.fit(X, y, params)
  fit(X: Tensor, y: Tensor, sampleWeightArg?: Tensor | void): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const weights = readSampleWeight(sampleWeightArg as Tensor | undefined, nSamples);

    const p = this.params;
    const rng = createTreeRng(p.randomState);
    const build: RegressionBuild = {
      xc: toColumnMajor(toFloat64View(X), nSamples, nFeatures),
      nSamples,
      y: toFloat64View(y),
      maxDepth: p.maxDepth,
      minSamplesSplit: p.minSamplesSplit,
      minSamplesLeaf: p.minSamplesLeaf,
      minImpurityDecrease: p.minImpurityDecrease,
      weights,
      totalWeight: nSamples,
      sampler: new TreeFeatureSampler(
        nFeatures,
        resolveTreeMaxFeatures(p.maxFeatures, nFeatures),
        rng
      ),
      rng,
      yLocal: new Float64Array(nSamples),
    };
    const growth = { maxLeafNodes: p.maxLeafNodes, ccpAlpha: p.ccpAlpha };

    const trees: TreeNode[] = [];
    for (let t = 0; t < p.nEstimators; t++) {
      const indices = withoutZeroWeights(drawSample(nSamples, p.bootstrap, rng), weights);
      trees.push(
        buildRegressionTree(
          { ...build, totalWeight: totalWeightOf(indices, weights) },
          indices,
          growth
        )
      );
    }

    this.trees = trees;
    this.nFeatures = nFeatures;
    this.fitted = true;
    return this;
  }

  /**
   * Predict target values: the mean of the trees' predictions.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns `float64` predictions of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("ExtraTreesRegressor must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures ?? 0, "ExtraTreesRegressor");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const x = toFloat64View(X);
    const out = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      let sum = 0;
      for (const tree of this.trees) sum += descend(tree, x, i * nFeatures).prediction ?? 0;
      out[i] = sum / this.trees.length;
    }
    return tensor(out, { dtype: "float64" });
  }

  /**
   * R² score on the given test data. When all targets are equal the score is 1 for a perfect
   * prediction and 0 otherwise.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True target values of shape (n_samples,)
   * @returns R² (best possible 1.0, can be negative)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional or sample counts mismatch
   * @throws {DataValidationError} If y is empty or contains NaN/Inf values
   */
  score(X: Tensor, y: Tensor): number {
    if (!this.fitted) {
      throw new NotFittedError("ExtraTreesRegressor must be fitted before scoring");
    }
    const truth = readScoreTargets(y);
    const predictions = toFloat64View(this.predict(X));
    checkScoreSizes(predictions.length, truth.length);

    let yMean = 0;
    for (let i = 0; i < truth.length; i++) yMean += truth[i] as number;
    yMean /= truth.length;
    let ssRes = 0;
    let ssTot = 0;
    for (let i = 0; i < truth.length; i++) {
      const yTrue = truth[i] as number;
      ssRes += (yTrue - (predictions[i] as number)) ** 2;
      ssTot += (yTrue - yMean) ** 2;
    }
    return ssTot === 0 ? (ssRes === 0 ? 1.0 : 0.0) : 1 - ssRes / ssTot;
  }

  /** Number of features seen during fit. */
  get nFeatures_(): number | undefined {
    return this.nFeatures;
  }

  /**
   * Mean decrease in impurity per feature, averaged over the trees (sums to 1, or all zeros if no
   * tree split).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get featureImportances(): Tensor {
    if (!this.fitted || this.trees.length === 0 || this.nFeatures === undefined) {
      throw new NotFittedError("ExtraTreesRegressor must be fitted to access featureImportances");
    }
    return forestImportances(this.trees, this.nFeatures);
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return { ...this.params };
  }

  /**
   * Set parameters. Either all given parameters are applied or, if one is invalid, none.
   *
   * @param params - Parameters to set
   * @returns this
   * @throws {InvalidParameterError} If a parameter value is invalid or unknown
   */
  setParams(params: Record<string, unknown>): this {
    const next = { ...this.params };
    for (const [key, value] of Object.entries(params)) {
      if (!applySharedParam(next, key, value)) {
        throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    this.params = next;
    return this;
  }

  /** Create an unfitted copy with the same hyperparameters. */
  clone(): ExtraTreesRegressor {
    return new ExtraTreesRegressor().setParams(this.getParams());
  }
}
