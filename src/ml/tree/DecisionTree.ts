import {
  DataValidationError,
  DeepboxError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __random, __randomBelow, __SeededRandom, __seedToUint64 } from "../../random/random";
import {
  assertContiguous,
  toFloat64View,
  validateFitInputs,
  validatePredictInputs,
} from "../_validation";
import type { Classifier, Regressor } from "../base";
import {
  checkCcpAlpha,
  checkClassWeight,
  checkMaxLeafNodes,
  checkMinImpurityDecrease,
  classWeightPerClass,
  combineWeights,
  type ExpandResult,
  growTreeBestFirst,
  positiveWeightIndices,
  pruneTree,
  readSampleWeight,
  splitIsWorthwhile,
  type TreeClassWeight,
  type TreeGrowthOptions,
} from "./_growth";

/** @internal */
export type TreeNode = {
  readonly isLeaf: boolean;
  readonly prediction?: number | undefined;
  readonly classProbabilities?: number[] | undefined;
  readonly featureIndex?: number;
  readonly threshold?: number;
  readonly left?: TreeNode;
  readonly right?: TreeNode;
  /** Number of training samples that reached this node. */
  readonly nSamples?: number;
  // Weighted impurity decrease at this split (n*imp - nL*impL - nR*impR),
  // recorded so feature_importances_ is the standard MDI (mean decrease in
  // impurity) rather than a raw split count.
  readonly weightedImpurityDecrease?: number;
  /** Impurity of the node (Gini, base-2 entropy, or the variance of the targets). */
  readonly impurity?: number;
  /** Sum of the sample weights that reached the node (the sample count without weights). */
  readonly weightedNSamples?: number;
  /**
   * Only kept while a tree is grown with `ccpAlpha > 0`: the leaf this split node becomes when
   * it is pruned.
   */
  readonly collapsed?: TreeNode;
};

/**
 * Impurity criterion used by the classification trees.
 *
 * `"log_loss"` is an alias of `"entropy"`; both use the base-2 Shannon entropy, as scikit-learn
 * does (this matters for `minImpurityDecrease` and `ccpAlpha`).
 */
export type ClassificationCriterion = "gini" | "entropy" | "log_loss";

/**
 * Number of features examined at every split: an exact count, `"sqrt"`
 * (`floor(sqrt(nFeatures))`) or `"log2"` (`floor(log2(nFeatures))`), never below 1.
 */
export type TreeMaxFeatures = number | "sqrt" | "log2";

/** Options of {@link DecisionTreeClassifier}. */
export type DecisionTreeClassifierOptions = {
  /** Maximum depth of the tree; `Infinity` grows it until the leaves are pure. Default 10. */
  readonly maxDepth?: number;
  /** Minimum number of samples a node needs to be split. Default 2. */
  readonly minSamplesSplit?: number;
  /** Minimum number of samples every leaf must keep. Default 1. */
  readonly minSamplesLeaf?: number;
  /** Features examined per split. Default: all features. */
  readonly maxFeatures?: TreeMaxFeatures;
  /** Seed for the random feature subsets. Without it the global Deepbox generator is used. */
  readonly randomState?: number;
  /** Split quality measure. Default `"gini"`. */
  readonly criterion?: ClassificationCriterion;
  /**
   * Class weights, `"balanced"` or a map from class label to weight. They multiply the
   * `sampleWeight` given to `fit`. Default: all classes weigh 1.
   */
  readonly classWeight?: TreeClassWeight;
} & TreeGrowthOptions;

/** Options of {@link DecisionTreeRegressor}. */
export type DecisionTreeRegressorOptions = {
  /** Maximum depth of the tree; `Infinity` grows it until the leaves are pure. Default 10. */
  readonly maxDepth?: number;
  /** Minimum number of samples a node needs to be split. Default 2. */
  readonly minSamplesSplit?: number;
  /** Minimum number of samples every leaf must keep. Default 1. */
  readonly minSamplesLeaf?: number;
  /** Features examined per split. Default: all features. */
  readonly maxFeatures?: TreeMaxFeatures;
  /** Seed for the random feature subsets. Without it the global Deepbox generator is used. */
  readonly randomState?: number;
} & TreeGrowthOptions;

// ---------------------------------------------------------------------------
// Shared helpers (also used by ExtraTrees.ts; not part of the public API)
// ---------------------------------------------------------------------------

/**
 * Random source for one `fit` call: a private seeded stream when `randomState` is set
 * (SplitMix64-expanded, so consecutive seeds give unrelated streams), the global Deepbox
 * generator otherwise. Shared by the trees, the forests, the boosting and bagging ensembles and
 * the stochastic linear models.
 *
 * @internal
 */
export function createTreeRng(randomState: number | undefined): () => number {
  if (randomState === undefined) return __random;
  const generator = new __SeededRandom(__seedToUint64(randomState));
  return () => generator.next();
}

/** @internal */
export function checkTreeMaxDepth(value: unknown): number {
  if (
    typeof value !== "number" ||
    !((Number.isInteger(value) && value >= 1) || value === Number.POSITIVE_INFINITY)
  ) {
    throw new InvalidParameterError(
      `maxDepth must be an integer >= 1 or Infinity; received ${String(value)}`,
      "maxDepth",
      value
    );
  }
  return value;
}

/** @internal */
export function checkTreeInteger(name: string, value: unknown, min: number): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < min) {
    throw new InvalidParameterError(
      `${name} must be an integer >= ${min}; received ${String(value)}`,
      name,
      value
    );
  }
  return value;
}

/** @internal */
export function checkTreeMaxFeatures(value: unknown): TreeMaxFeatures | undefined {
  if (value === undefined || value === "sqrt" || value === "log2") return value;
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new InvalidParameterError(
      `maxFeatures must be an integer >= 1, "sqrt" or "log2"; received ${String(value)}`,
      "maxFeatures",
      value
    );
  }
  return value;
}

/** @internal */
export function checkTreeRandomState(value: unknown): number | undefined {
  if (value !== undefined && (typeof value !== "number" || !Number.isFinite(value))) {
    throw new InvalidParameterError(
      `randomState must be a finite number; received ${String(value)}`,
      "randomState",
      value
    );
  }
  return value;
}

/** @internal */
export function checkTreeCriterion(value: unknown): ClassificationCriterion {
  if (value !== "gini" && value !== "entropy" && value !== "log_loss") {
    throw new InvalidParameterError(
      `criterion must be "gini", "entropy", or "log_loss"; received ${String(value)}`,
      "criterion",
      value
    );
  }
  return value;
}

/**
 * Number of candidate features per split for a `maxFeatures` setting.
 * `undefined` means every feature.
 *
 * @internal
 */
export function resolveTreeMaxFeatures(
  maxFeatures: TreeMaxFeatures | undefined,
  nFeatures: number
): number {
  if (maxFeatures === undefined) return nFeatures;
  if (typeof maxFeatures === "number") return Math.max(1, Math.min(maxFeatures, nFeatures));
  if (maxFeatures === "sqrt") return Math.max(1, Math.floor(Math.sqrt(nFeatures)));
  return Math.max(1, Math.floor(Math.log2(nFeatures)));
}

/**
 * dtype of a label tensor: `int32` when every label is an integer that fits, `float64`
 * otherwise (so fractional labels are not silently truncated).
 *
 * @internal
 */
export function treeLabelDType(labels: ArrayLike<number>): "int32" | "float64" {
  for (let i = 0; i < labels.length; i++) {
    const v = labels[i] as number;
    if (!Number.isInteger(v) || v < -2147483648 || v > 2147483647) return "float64";
  }
  return "int32";
}

/** @internal */
export function treeLabelTensor(values: ArrayLike<number>, dtype: "int32" | "float64"): Tensor {
  return dtype === "int32"
    ? tensor(Int32Array.from(values), { dtype: "int32" })
    : tensor(Float64Array.from(values), { dtype: "float64" });
}

/**
 * Draws the features examined at one node: the first `limit` draws come from a lazy
 * Fisher-Yates shuffle, and callers keep drawing while no usable (non-constant) feature has
 * been seen, as scikit-learn does. Without a limit the features are visited in index order.
 *
 * @internal
 */
export class TreeFeatureSampler {
  private readonly perm: Int32Array | undefined;

  constructor(
    readonly nFeatures: number,
    readonly limit: number,
    private readonly rng: () => number
  ) {
    if (limit < nFeatures) {
      this.perm = new Int32Array(nFeatures);
      for (let i = 0; i < nFeatures; i++) this.perm[i] = i;
    }
  }

  /** Feature at draw position `visited` (0-based, strictly increasing within one node). */
  draw(visited: number): number {
    const perm = this.perm;
    if (perm === undefined) return visited;
    const j = visited + __randomBelow(this.rng, this.nFeatures - visited);
    const f = perm[j] as number;
    perm[j] = perm[visited] as number;
    perm[visited] = f;
    return f;
  }
}

/**
 * Row-major `(n, d)` data to a feature-major (column-contiguous) `Float64Array`.
 *
 * @internal
 */
export function toColumnMajor(rowMajor: Float64Array, n: number, d: number): Float64Array {
  const out = new Float64Array(n * d);
  for (let i = 0; i < n; i++) {
    const base = i * d;
    for (let j = 0; j < d; j++) out[j * n + i] = rowMajor[base + j] as number;
  }
  return out;
}

/** Midpoint of two adjacent distinct values that always lies in `[lo, hi)`. */
function splitThreshold(lo: number, hi: number): number {
  const mid = lo / 2 + hi / 2;
  return mid >= lo && mid < hi ? mid : lo;
}

/**
 * Leaf reached by `x[base..base + d)`; throws if the tree is malformed.
 *
 * @internal
 */
export function descend(root: TreeNode, x: Float64Array, base: number): TreeNode {
  let node = root;
  while (!node.isLeaf) {
    const next =
      (x[base + (node.featureIndex ?? 0)] as number) <= (node.threshold ?? 0)
        ? node.left
        : node.right;
    if (!next) throw new DeepboxError("Corrupted tree: internal node is missing a child");
    node = next;
  }
  return node;
}

function treeDepth(root: TreeNode): number {
  let deepest = 0;
  const stack: Array<[TreeNode, number]> = [[root, 0]];
  for (let top = stack.pop(); top !== undefined; top = stack.pop()) {
    const [node, depth] = top;
    if (depth > deepest) deepest = depth;
    if (node.left) stack.push([node.left, depth + 1]);
    if (node.right) stack.push([node.right, depth + 1]);
  }
  return deepest;
}

function treeLeafCount(root: TreeNode): number {
  let leaves = 0;
  const stack: TreeNode[] = [root];
  for (let node = stack.pop(); node !== undefined; node = stack.pop()) {
    if (node.isLeaf) {
      leaves++;
      continue;
    }
    if (node.left) stack.push(node.left);
    if (node.right) stack.push(node.right);
  }
  return leaves;
}

/** A tree node while it is being built (children are attached afterwards). @internal */
export type MutableTreeNode = { -readonly [K in keyof TreeNode]: TreeNode[K] };

/**
 * Builds a tree without recursion, so very deep trees (`maxDepth: Infinity` on data that splits
 * off a few samples at a time) cannot overflow the call stack. `expand` turns the samples of one
 * node into either a leaf or a split node plus the samples of its two children; nodes are
 * expanded in pre-order (a node, its whole left subtree, then its right subtree), the same order
 * as a recursive build, so seeded random draws are unaffected.
 *
 * With `maxLeafNodes` the tree grows best-first instead (see {@link growTreeBestFirst}); with
 * `ccpAlpha > 0` the finished tree is pruned (see {@link pruneTree}).
 *
 * @internal
 */
export function growTree(
  rootIndices: Int32Array,
  expand: (indices: Int32Array, depth: number) => ExpandResult,
  growth: {
    readonly maxLeafNodes?: number | undefined;
    readonly ccpAlpha?: number | undefined;
  } = {}
): TreeNode {
  const ccpAlpha = growth.ccpAlpha ?? 0;
  const prune = ccpAlpha > 0;
  // Keeps the leaf a split node turns into when it is pruned.
  const expandKeeping = (indices: Int32Array, depth: number): ExpandResult => {
    const result = expand(indices, depth);
    if (prune && result.leaf !== undefined) result.node.collapsed = result.leaf();
    return result;
  };
  const expandFn = prune ? expandKeeping : expand;

  let root: TreeNode;
  if (growth.maxLeafNodes !== undefined) {
    root = growTreeBestFirst(rootIndices, expandFn, growth.maxLeafNodes);
  } else {
    let first: TreeNode | undefined;
    type Job = { indices: Int32Array; depth: number; attach: (node: TreeNode) => void };
    const stack: Job[] = [
      {
        indices: rootIndices,
        depth: 0,
        attach: (node) => {
          first = node;
        },
      },
    ];
    for (let job = stack.pop(); job !== undefined; job = stack.pop()) {
      const { node, children } = expandFn(job.indices, job.depth);
      job.attach(node);
      if (children) {
        stack.push({
          indices: children[1],
          depth: job.depth + 1,
          attach: (child) => {
            node.right = child;
          },
        });
        stack.push({
          indices: children[0],
          depth: job.depth + 1,
          attach: (child) => {
            node.left = child;
          },
        });
      }
    }
    root = first as unknown as TreeNode;
  }
  return prune ? pruneTree(root, ccpAlpha) : root;
}

/** @internal */
export function normalizedImportances(tree: TreeNode, nFeatures: number): Tensor {
  const importances = new Float64Array(nFeatures);
  accumulateImportances(tree, importances);
  let total = 0;
  for (let i = 0; i < nFeatures; i++) total += importances[i] as number;
  if (total > 0) {
    for (let i = 0; i < nFeatures; i++) importances[i] = (importances[i] as number) / total;
  }
  return tensor(importances, { dtype: "float64" });
}

function accumulateImportances(root: TreeNode, importances: Float64Array): void {
  const stack: TreeNode[] = [root];
  for (let node = stack.pop(); node !== undefined; node = stack.pop()) {
    if (node.isLeaf) continue;
    const fi = node.featureIndex ?? 0;
    if (fi < importances.length) {
      // Mean decrease in impurity: the weighted impurity drop of every split on the
      // feature (scikit-learn convention), not a raw split count.
      importances[fi] = (importances[fi] as number) + (node.weightedImpurityDecrease ?? 0);
    }
    if (node.left) stack.push(node.left);
    if (node.right) stack.push(node.right);
  }
}

/**
 * Split `indices` into the samples with `x[feature] <= threshold` and the rest.
 *
 * @internal
 */
export function partitionIndices(
  xc: Float64Array,
  base: number,
  indices: Int32Array,
  threshold: number
): [Int32Array, Int32Array] {
  const n = indices.length;
  let nLeft = 0;
  for (let i = 0; i < n; i++) {
    if ((xc[base + (indices[i] as number)] as number) <= threshold) nLeft++;
  }
  const left = new Int32Array(nLeft);
  const right = new Int32Array(n - nLeft);
  let l = 0;
  let r = 0;
  for (let i = 0; i < n; i++) {
    const idx = indices[i] as number;
    if ((xc[base + idx] as number) <= threshold) left[l++] = idx;
    else right[r++] = idx;
  }
  return [left, right];
}

/**
 * Common validation of the y tensor shared by the `score` methods.
 *
 * @internal
 */
export function readScoreTargets(y: Tensor): Float64Array {
  if (y.ndim !== 1) {
    throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
  }
  assertContiguous(y, "y");
  const values = toFloat64View(y);
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) {
      throw new DataValidationError("y contains non-finite values (NaN or Inf)");
    }
  }
  if (values.length === 0) {
    throw new DataValidationError("y must contain at least one sample");
  }
  return values;
}

const treeSortF32 = new Float32Array(1);
const treeSortU32 = new Uint32Array(treeSortF32.buffer);
let treeSortPacked = new Float64Array(0);

const treeSortF64 = new Float64Array(1);
const treeSortF64Words = new Uint32Array(treeSortF64.buffer);
const TREE_SORT_HIGH_WORD = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1 ? 1 : 0;
const treeSortCounts = new Uint32Array(256);
let treeSortKeyHigh = new Uint32Array(0);
let treeSortKeyLow = new Uint32Array(0);
let treeSortScratch = new Int32Array(0);

/** Column length from which the radix sort beats the comparator sort. */
const TREE_RADIX_MIN = 96;

/**
 * Stable ascending sort of `positions[0..n)` by `col[pos]` for float64 columns that are not
 * float32-exact, with an LSD radix sort on the order-preserving 64-bit encoding of the values
 * (bytes whose value is the same for every element are skipped). `positions` must hold the
 * identity permutation `0..n-1` on entry. `-0` sorts before `+0`; the two compare equal in the
 * split search, so the ordering of the tree is not affected.
 */
function radixSortPositions(col: Float64Array, positions: Int32Array, n: number): void {
  if (treeSortKeyHigh.length < n) {
    const size = Math.max(n, 1024);
    treeSortKeyHigh = new Uint32Array(size);
    treeSortKeyLow = new Uint32Array(size);
    treeSortScratch = new Int32Array(size);
  }
  const keyHigh = treeSortKeyHigh;
  const keyLow = treeSortKeyLow;
  const low = 1 - TREE_SORT_HIGH_WORD;
  for (let i = 0; i < n; i++) {
    treeSortF64[0] = col[i] as number;
    let hi = treeSortF64Words[TREE_SORT_HIGH_WORD] as number;
    let lo = treeSortF64Words[low] as number;
    if (hi & 0x80000000) {
      hi = ~hi >>> 0;
      lo = ~lo >>> 0;
    } else {
      hi = (hi | 0x80000000) >>> 0;
    }
    keyHigh[i] = hi;
    keyLow[i] = lo;
  }

  const counts = treeSortCounts;
  let src: Int32Array = positions;
  let dst: Int32Array = treeSortScratch;
  for (let pass = 0; pass < 8; pass++) {
    const word = pass < 4 ? keyLow : keyHigh;
    const shift = (pass & 3) * 8;
    counts.fill(0);
    for (let i = 0; i < n; i++) {
      const bucket = ((word[src[i] as number] as number) >>> shift) & 255;
      counts[bucket] = (counts[bucket] as number) + 1;
    }
    let skip = false;
    let offset = 0;
    for (let b = 0; b < 256; b++) {
      const c = counts[b] as number;
      if (c === n) {
        skip = true;
        break;
      }
      counts[b] = offset;
      offset += c;
    }
    if (skip) continue;
    for (let i = 0; i < n; i++) {
      const id = src[i] as number;
      const bucket = ((word[id] as number) >>> shift) & 255;
      const at = counts[bucket] as number;
      dst[at] = id;
      counts[bucket] = at + 1;
    }
    const swap = src;
    src = dst;
    dst = swap;
  }
  if (src !== positions) positions.set(src.subarray(0, n));
}

/**
 * Sort `positions[0..n)` ascending by `col[pos]`, stably.
 *
 * When every column value is float32-exact (the common case: features come
 * from float32 tensors), values are bit-encoded into order-preserving
 * integers and packed with the position into a Float64Array so V8's
 * comparator-free typed sort applies (~3x faster than a comparator sort).
 * Otherwise long columns use a radix sort on the float64 bits and short ones a stable
 * comparator sort. All paths produce the same ordering of distinct values and keep the original
 * order of equal values. `positions` must hold the identity permutation `0..n-1` on entry.
 *
 * @internal
 */
export function sortPositionsByColumn(col: Float64Array, positions: Int32Array, n: number): void {
  // Already sorted (sorted time series, or the child of a split on this very feature): the
  // identity permutation the callers start from is the stable result.
  let sorted = true;
  for (let i = 1; i < n; i++) {
    if ((col[i] as number) < (col[i - 1] as number)) {
      sorted = false;
      break;
    }
  }
  if (sorted) return;
  let packable = n <= 2097152;
  if (packable) {
    for (let i = 0; i < n; i++) {
      const v = col[i]!;
      if (Math.fround(v) !== v) {
        packable = false;
        break;
      }
    }
  }
  if (packable) {
    if (treeSortPacked.length < n) treeSortPacked = new Float64Array(Math.max(n, 1024));
    const packed = treeSortPacked.subarray(0, n);
    for (let i = 0; i < n; i++) {
      treeSortF32[0] = col[i]!;
      const bits = treeSortU32[0]! >>> 0;
      const enc = bits & 0x80000000 ? ~bits >>> 0 : (bits | 0x80000000) >>> 0;
      packed[i] = enc * 2097152 + i;
    }
    packed.sort();
    for (let i = 0; i < n; i++) {
      const key = packed[i]!;
      positions[i] = key - Math.floor(key / 2097152) * 2097152;
    }
    return;
  }
  if (n >= TREE_RADIX_MIN) {
    radixSortPositions(col, positions, n);
    return;
  }
  positions.subarray(0, n).sort((a, b) => col[a]! - col[b]!);
}

/**
 * Leaf for a node with per-class weighted sample `counts` (`weight` is their sum, `n` the number
 * of samples). The predicted label is the class with the largest weight; ties go to the smallest
 * label (`labels` is sorted ascending).
 *
 * @internal
 */
export function classLeaf(
  counts: Float64Array,
  weight: number,
  n: number,
  labels: readonly number[],
  impurity: number
): TreeNode {
  const probabilities = new Array<number>(counts.length);
  let best = 0;
  for (let c = 0; c < counts.length; c++) {
    const count = counts[c] as number;
    probabilities[c] = count / weight;
    if (count > (counts[best] as number)) best = c;
  }
  return {
    isLeaf: true,
    prediction: labels[best] as number,
    classProbabilities: probabilities,
    nSamples: n,
    weightedNSamples: weight,
    impurity,
  };
}

/**
 * Impurity of a node from its weighted class `counts` (`weight` is their sum): Gini, or the
 * base-2 entropy.
 *
 * @internal
 */
export function classImpurity(counts: Float64Array, weight: number, isGini: boolean): number {
  let sum = 0;
  for (let c = 0; c < counts.length; c++) {
    const count = counts[c] as number;
    if (count <= 0) continue;
    if (isGini) sum += count * count;
    else sum -= count * Math.log2(count / weight);
  }
  return isGini ? 1 - sum / (weight * weight) : sum / weight;
}

/**
 * Weighted mean (compensated with a second pass), minimum, maximum, total weight and variance
 * (the regression impurity) of `y` over `indices`. `weights` are indexed like `y`; without them
 * every sample weighs 1.
 *
 * @internal
 */
export function nodeTargetStats(
  y: Float64Array,
  indices: Int32Array,
  weights?: Float64Array
): { mean: number; lo: number; hi: number; weight: number; impurity: number } {
  const n = indices.length;
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  if (weights === undefined) {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const v = y[indices[i] as number] as number;
      sum += v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const mean = sum / n;
    let correction = 0;
    let squares = 0;
    for (let i = 0; i < n; i++) {
      const d = (y[indices[i] as number] as number) - mean;
      correction += d;
      squares += d * d;
    }
    const variance = Math.max(0, squares - (correction * correction) / n) / n;
    return { mean: mean + correction / n, lo, hi, weight: n, impurity: variance };
  }
  let sum = 0;
  let weight = 0;
  for (let i = 0; i < n; i++) {
    const idx = indices[i] as number;
    const v = y[idx] as number;
    const w = weights[idx] as number;
    sum += w * v;
    weight += w;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const mean = sum / weight;
  let correction = 0;
  let squares = 0;
  for (let i = 0; i < n; i++) {
    const idx = indices[i] as number;
    const w = weights[idx] as number;
    const d = (y[idx] as number) - mean;
    correction += w * d;
    squares += w * d * d;
  }
  const variance = Math.max(0, squares - (correction * correction) / weight) / weight;
  return { mean: mean + correction / weight, lo, hi, weight, impurity: variance };
}

/** @internal */
export type SplitResult = {
  readonly feature: number;
  readonly threshold: number;
  readonly decrease: number;
};

/** Copy of a `classWeight` setting, so `getParams` does not expose the internal map. */
function copyClassWeight(value: TreeClassWeight | undefined): TreeClassWeight | undefined {
  return value === undefined || typeof value === "string" ? value : { ...value };
}

// ---------------------------------------------------------------------------
// Classifier
// ---------------------------------------------------------------------------

type ClassifierContext = {
  readonly xc: Float64Array;
  readonly nSamples: number;
  readonly yCode: Int32Array;
  readonly nClasses: number;
  readonly labels: readonly number[];
  /** Per-sample weights (`sampleWeight * classWeight`), `undefined` when every sample weighs 1. */
  readonly weights: Float64Array | undefined;
  /** Sum of the weights of all samples the tree is grown on. */
  readonly totalWeight: number;
  readonly sampler: TreeFeatureSampler;
  // Scratch buffers, sized once per fit and reused by every node.
  readonly col: Float64Array;
  readonly positions: Int32Array;
  readonly localLabels: Int32Array;
  readonly remap: Int32Array;
  readonly totalLocal: Float64Array;
  readonly leftCounts: Float64Array;
  readonly rightCounts: Float64Array;
};

/**
 * Decision Tree Classifier.
 *
 * A non-parametric supervised learning method that learns simple decision rules
 * inferred from the data features.
 *
 * **Algorithm**: CART (Classification and Regression Trees)
 * - Splits on Gini impurity (default) or Shannon entropy
 * - Recursively splits data based on feature thresholds (`x <= threshold` goes left)
 * - Supports `maxDepth`, `minSamplesSplit`, `minSamplesLeaf`, `minImpurityDecrease`,
 *   `maxLeafNodes` (best-first growth) and `ccpAlpha` (cost-complexity pruning) for regularization
 * - `fit` accepts per-sample weights and the `classWeight` option weights whole classes
 * - Ties between equally good classes resolve to the smallest class label
 *
 * Class labels may be any finite numbers. `predict` and `classes` are `int32` when all labels
 * are integers and `float64` otherwise.
 *
 * @example
 * ```ts
 * import { DecisionTreeClassifier } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [3, 4], [5, 6], [7, 8]]);
 * const y = tensor([0, 0, 1, 1]);
 *
 * const clf = new DecisionTreeClassifier({ maxDepth: 3 });
 * clf.fit(X, y);
 * const predictions = clf.predict(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-tree | Deepbox Decision Trees}
 */
export class DecisionTreeClassifier implements Classifier {
  private maxDepth: number;
  private minSamplesSplit: number;
  private minSamplesLeaf: number;
  private maxFeatures: TreeMaxFeatures | undefined;
  private randomState: number | undefined;
  private criterion: ClassificationCriterion;
  private minImpurityDecrease: number;
  private maxLeafNodes: number | undefined;
  private ccpAlpha: number;
  private classWeight: TreeClassWeight | undefined;

  private tree?: TreeNode;
  private nFeatures?: number;
  private classLabels?: number[];
  private labelDType: "int32" | "float64" = "int32";
  private fitted = false;

  /**
   * @param options - Hyperparameters, see {@link DecisionTreeClassifierOptions}
   * @throws {InvalidParameterError} If an option is out of range
   */
  constructor(options: DecisionTreeClassifierOptions = {}) {
    this.maxDepth = checkTreeMaxDepth(options.maxDepth ?? 10);
    this.minSamplesSplit = checkTreeInteger("minSamplesSplit", options.minSamplesSplit ?? 2, 2);
    this.minSamplesLeaf = checkTreeInteger("minSamplesLeaf", options.minSamplesLeaf ?? 1, 1);
    this.criterion = checkTreeCriterion(options.criterion ?? "gini");
    this.maxFeatures = checkTreeMaxFeatures(options.maxFeatures);
    this.randomState = checkTreeRandomState(options.randomState);
    this.minImpurityDecrease = checkMinImpurityDecrease(options.minImpurityDecrease ?? 0);
    this.maxLeafNodes = checkMaxLeafNodes(options.maxLeafNodes);
    this.ccpAlpha = checkCcpAlpha(options.ccpAlpha ?? 0);
    this.classWeight = checkClassWeight(options.classWeight, false) as TreeClassWeight | undefined;
  }

  /**
   * Build a decision tree classifier from the training set (X, y).
   *
   * Every sample counts with its weight: the impurity of a node, the split search, the class
   * probabilities of the leaves and `minImpurityDecrease` all use weighted class counts, while
   * `minSamplesSplit` and `minSamplesLeaf` still count samples. Samples with weight 0 are
   * ignored. The weight of a sample is `sampleWeight[i] * classWeight[y[i]]`.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target class labels of shape (n_samples,)
   * @param sampleWeight - Optional non-negative weights of shape (n_samples,)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D or y is not 1D
   * @throws {ShapeError} If X and y have different number of samples
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

    const nClasses = labels.length;
    let weights = sampleWeight;
    if (this.classWeight !== undefined) {
      const classCounts = new Float64Array(nClasses);
      for (let i = 0; i < nSamples; i++) classCounts[yCode[i] as number]! += 1;
      weights = combineWeights(
        sampleWeight,
        classWeightPerClass(this.classWeight, labels, classCounts, nSamples),
        yCode
      );
    }
    const indices = positiveWeightIndices(nSamples, weights);
    if (indices.length === 0) {
      throw new DataValidationError("every sample has a weight of zero");
    }
    let totalWeight = 0;
    for (let i = 0; i < indices.length; i++) {
      totalWeight += weights === undefined ? 1 : (weights[indices[i] as number] as number);
    }
    const ctx: ClassifierContext = {
      xc,
      nSamples,
      yCode,
      nClasses,
      labels,
      weights,
      totalWeight,
      sampler: new TreeFeatureSampler(
        nFeatures,
        resolveTreeMaxFeatures(this.maxFeatures, nFeatures),
        createTreeRng(this.randomState)
      ),
      col: new Float64Array(nSamples),
      positions: new Int32Array(nSamples),
      localLabels: new Int32Array(nSamples),
      remap: new Int32Array(nClasses),
      totalLocal: new Float64Array(nClasses),
      leftCounts: new Float64Array(nClasses),
      rightCounts: new Float64Array(nClasses),
    };

    this.tree = growTree(indices, (idx, depth) => this.expandNode(ctx, idx, depth), {
      maxLeafNodes: this.maxLeafNodes,
      ccpAlpha: this.ccpAlpha,
    });
    this.nFeatures = nFeatures;
    this.classLabels = labels;
    this.labelDType = treeLabelDType(labels);
    this.fitted = true;
    return this;
  }

  private expandNode(ctx: ClassifierContext, indices: Int32Array, depth: number): ExpandResult {
    const n = indices.length;
    const counts = new Float64Array(ctx.nClasses);
    const { weights } = ctx;
    let weight = n;
    if (weights === undefined) {
      for (let i = 0; i < n; i++) counts[ctx.yCode[indices[i] as number] as number]! += 1;
    } else {
      weight = 0;
      for (let i = 0; i < n; i++) {
        const idx = indices[i] as number;
        const w = weights[idx] as number;
        counts[ctx.yCode[idx] as number]! += w;
        weight += w;
      }
    }

    let present = 0;
    for (let c = 0; c < ctx.nClasses; c++) if ((counts[c] as number) > 0) present++;

    const isGini = this.criterion === "gini";
    const impurity = present === 1 ? 0 : classImpurity(counts, weight, isGini);
    const makeLeaf = (): TreeNode => classLeaf(counts, weight, n, ctx.labels, impurity);

    if (
      depth >= this.maxDepth ||
      n < this.minSamplesSplit ||
      n < 2 * this.minSamplesLeaf ||
      present === 1
    ) {
      return { node: makeLeaf() };
    }

    const split = this.findBestSplit(ctx, indices, counts, weight, present);
    if (split === undefined) return { node: makeLeaf() };
    if (!splitIsWorthwhile(split.decrease, ctx.totalWeight, this.minImpurityDecrease)) {
      return { node: makeLeaf() };
    }

    const children = partitionIndices(
      ctx.xc,
      split.feature * ctx.nSamples,
      indices,
      split.threshold
    );
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

  /**
   * Exhaustive search for the split with the lowest weighted child impurity.
   * `counts` are the weighted class counts of the node and `weight` their sum.
   * Returns `undefined` when no feature offers a valid split.
   */
  private findBestSplit(
    ctx: ClassifierContext,
    indices: Int32Array,
    counts: Float64Array,
    weight: number,
    present: number
  ): SplitResult | undefined {
    const n = indices.length;
    const { xc, nSamples, col, positions, localLabels, remap, totalLocal, weights } = ctx;
    const { leftCounts, rightCounts, sampler } = ctx;
    const minLeaf = this.minSamplesLeaf;
    const isGini = this.criterion === "gini";

    // Compact the classes present at this node into 0..present-1 so the scan loop
    // touches small typed arrays.
    let k = 0;
    for (let c = 0; c < ctx.nClasses; c++) {
      const count = counts[c] as number;
      if (count > 0) {
        remap[c] = k;
        totalLocal[k] = count;
        k++;
      }
    }
    for (let i = 0; i < n; i++) {
      localLabels[i] = remap[ctx.yCode[indices[i] as number] as number] as number;
    }

    let totalSq = 0;
    let nodeEntropy = 0;
    for (let c = 0; c < present; c++) {
      const count = totalLocal[c] as number;
      totalSq += count * count;
      nodeEntropy -= count * Math.log2(count / weight);
    }

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
        col[i] = v;
        positions[i] = i;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      if (lo === hi) continue; // constant here: cannot split, and does not use up the budget
      usable++;

      sortPositionsByColumn(col, positions, n);

      for (let c = 0; c < k; c++) {
        leftCounts[c] = 0;
        rightCounts[c] = totalLocal[c] as number;
      }
      // Without sample weights the sums of squared class counts are exact integers, so the Gini
      // score below is computed without accumulated rounding error.
      let leftSq = 0;
      let rightSq = totalSq;
      let leftWeight = 0;

      for (let i = 0; i < n - 1; i++) {
        const pos = positions[i] as number;
        const label = localLabels[pos] as number;
        const lc = leftCounts[label] as number;
        const rc = rightCounts[label] as number;
        let w = 1;
        if (weights === undefined) {
          leftSq += 2 * lc + 1;
          rightSq -= 2 * rc - 1;
        } else {
          w = weights[indices[pos] as number] as number;
          leftSq += w * (2 * lc + w);
          rightSq -= w * (2 * rc - w);
        }
        leftCounts[label] = lc + w;
        rightCounts[label] = rc - w;
        leftWeight += w;

        const val = col[pos] as number;
        const nextVal = col[positions[i + 1] as number] as number;
        if (val === nextVal) continue; // cannot split between equal values

        const leftSize = i + 1;
        const rightSize = n - leftSize;
        if (leftSize < minLeaf || rightSize < minLeaf) continue;
        const rightWeight = weight - leftWeight;
        if (!(rightWeight > 0)) continue;

        let score: number;
        if (isGini) {
          // Maximizing sum_side(sumsq / weight) minimizes the weighted Gini impurity.
          score = leftSq / leftWeight + rightSq / rightWeight;
        } else {
          let childEntropy = 0;
          for (let c = 0; c < k; c++) {
            const l = leftCounts[c] as number;
            if (l > 0) childEntropy -= l * Math.log2(l / leftWeight);
            const r = rightCounts[c] as number;
            if (r > 0) childEntropy -= r * Math.log2(r / rightWeight);
          }
          score = -childEntropy;
        }

        if (score > bestScore) {
          bestScore = score;
          bestFeature = f;
          bestThreshold = splitThreshold(val, nextVal);
        }
      }
    }

    if (bestFeature < 0) return undefined;
    // weight * impurity(node) - sum_side(weight_side * impurity(side)), both criteria.
    const decrease = isGini ? bestScore - totalSq / weight : nodeEntropy + bestScore;
    return {
      feature: bestFeature,
      threshold: bestThreshold,
      decrease: Math.max(0, decrease),
    };
  }

  /**
   * Predict class labels for samples in X.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted class labels of shape (n_samples,): `int32` when all classes are
   *   integers, `float64` otherwise
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    const tree = this.requireTree();
    validatePredictInputs(X, this.nFeatures ?? 0, "DecisionTreeClassifier");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const x = toFloat64View(X);
    const out = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      out[i] = descend(tree, x, i * nFeatures).prediction ?? 0;
    }
    return treeLabelTensor(out, this.labelDType);
  }

  /**
   * Predict class probabilities for samples in X.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns `float64` class probability matrix of shape (n_samples, n_classes); columns
   *   follow {@link DecisionTreeClassifier.classes}
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predictProba(X: Tensor): Tensor {
    const tree = this.requireTree();
    validatePredictInputs(X, this.nFeatures ?? 0, "DecisionTreeClassifier");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const nClasses = this.classLabels?.length ?? 0;
    const x = toFloat64View(X);
    const out = new Float64Array(nSamples * nClasses);
    for (let i = 0; i < nSamples; i++) {
      const probabilities = descend(tree, x, i * nFeatures).classProbabilities;
      if (probabilities) out.set(probabilities, i * nClasses);
    }
    return tensor(out, { dtype: "float64" }).reshape([nSamples, nClasses]);
  }

  /**
   * Return the mean accuracy on the given test data and labels.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True labels of shape (n_samples,)
   * @returns Accuracy score in range [0, 1]
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional or sample counts mismatch
   * @throws {DataValidationError} If y is empty or contains NaN/Inf values
   */
  score(X: Tensor, y: Tensor): number {
    this.requireTree();
    const truth = readScoreTargets(y);
    const predictions = toFloat64View(this.predict(X));
    if (predictions.length !== truth.length) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${predictions.length}, y=${truth.length}`
      );
    }
    let correct = 0;
    for (let i = 0; i < truth.length; i++) {
      if (predictions[i] === truth[i]) correct++;
    }
    return correct / truth.length;
  }

  /**
   * Get the unique class labels discovered during fitting.
   *
   * @returns Sorted class labels (`int32` for integer labels, `float64` otherwise), or
   *   undefined if not fitted
   */
  get classes(): Tensor | undefined {
    if (!this.fitted || !this.classLabels) {
      return undefined;
    }
    return treeLabelTensor(this.classLabels, this.labelDType);
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
      minSamplesLeaf: this.minSamplesLeaf,
      maxFeatures: this.maxFeatures,
      randomState: this.randomState,
      criterion: this.criterion,
      minImpurityDecrease: this.minImpurityDecrease,
      maxLeafNodes: this.maxLeafNodes,
      ccpAlpha: this.ccpAlpha,
      classWeight: copyClassWeight(this.classWeight),
    };
  }

  /**
   * Set the parameters of this estimator. Either all given parameters are applied or, if one
   * is invalid, none.
   *
   * @param params - Parameters to set
   * @returns this
   * @throws {InvalidParameterError} If a parameter value is invalid or unknown
   */
  setParams(params: Record<string, unknown>): this {
    const next = {
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
      minSamplesLeaf: this.minSamplesLeaf,
      maxFeatures: this.maxFeatures,
      randomState: this.randomState,
      criterion: this.criterion,
      minImpurityDecrease: this.minImpurityDecrease,
      maxLeafNodes: this.maxLeafNodes,
      ccpAlpha: this.ccpAlpha,
      classWeight: this.classWeight,
    };
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "maxDepth":
          next.maxDepth = checkTreeMaxDepth(value);
          break;
        case "minSamplesSplit":
          next.minSamplesSplit = checkTreeInteger("minSamplesSplit", value, 2);
          break;
        case "minSamplesLeaf":
          next.minSamplesLeaf = checkTreeInteger("minSamplesLeaf", value, 1);
          break;
        case "maxFeatures":
          next.maxFeatures = checkTreeMaxFeatures(value);
          break;
        case "criterion":
          next.criterion = checkTreeCriterion(value);
          break;
        case "randomState":
          next.randomState = checkTreeRandomState(value);
          break;
        case "minImpurityDecrease":
          next.minImpurityDecrease = checkMinImpurityDecrease(value);
          break;
        case "maxLeafNodes":
          next.maxLeafNodes = checkMaxLeafNodes(value);
          break;
        case "ccpAlpha":
          next.ccpAlpha = checkCcpAlpha(value);
          break;
        case "classWeight":
          next.classWeight = checkClassWeight(value, false) as TreeClassWeight | undefined;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    Object.assign(this, next);
    return this;
  }

  /** Access the internal tree structure (for export_text). */
  get tree_(): TreeNode | undefined {
    return this.tree;
  }

  /** Number of features seen during fit. */
  get nFeatures_(): number | undefined {
    return this.nFeatures;
  }

  /**
   * Depth of the fitted tree (a single leaf has depth 0).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  getDepth(): number {
    return treeDepth(this.requireTree());
  }

  /**
   * Number of leaves of the fitted tree.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  getNLeaves(): number {
    return treeLeafCount(this.requireTree());
  }

  /**
   * Normalized mean decrease in impurity per feature (sums to 1, or all zeros when the tree
   * never split).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get featureImportances(): Tensor {
    if (!this.fitted || !this.tree || this.nFeatures === undefined) {
      throw new NotFittedError(
        "DecisionTreeClassifier must be fitted to access feature_importances_"
      );
    }
    return normalizedImportances(this.tree, this.nFeatures);
  }

  /**
   * Create an unfitted copy with the same hyperparameters.
   */
  clone(): DecisionTreeClassifier {
    return new DecisionTreeClassifier(this.getParams() as DecisionTreeClassifierOptions);
  }

  private requireTree(): TreeNode {
    if (!this.fitted || !this.tree) {
      throw new NotFittedError("DecisionTreeClassifier must be fitted before prediction");
    }
    return this.tree;
  }
}

// ---------------------------------------------------------------------------
// Regressor
// ---------------------------------------------------------------------------

type RegressorContext = {
  readonly xc: Float64Array;
  readonly nSamples: number;
  readonly y: Float64Array;
  /** Per-sample weights, `undefined` when every sample weighs 1. */
  readonly weights: Float64Array | undefined;
  /** Sum of the weights of all samples the tree is grown on. */
  readonly totalWeight: number;
  readonly sampler: TreeFeatureSampler;
  readonly col: Float64Array;
  readonly positions: Int32Array;
  readonly yLocal: Float64Array;
};

/**
 * Decision Tree Regressor.
 *
 * Uses MSE reduction to find optimal splits for regression tasks. Leaves predict the mean
 * target of their training samples; `predict` returns `float64`.
 *
 * @example
 * ```ts
 * import { DecisionTreeRegressor } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1], [2], [3], [4]]);
 * const y = tensor([1.5, 1.7, 3.1, 3.3]);
 * const reg = new DecisionTreeRegressor({ maxDepth: 2 }).fit(X, y);
 * const predictions = reg.predict(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-tree | Deepbox Decision Trees}
 */
export class DecisionTreeRegressor implements Regressor {
  private maxDepth: number;
  private minSamplesSplit: number;
  private minSamplesLeaf: number;
  private maxFeatures: TreeMaxFeatures | undefined;
  private randomState: number | undefined;
  private minImpurityDecrease: number;
  private maxLeafNodes: number | undefined;
  private ccpAlpha: number;

  private tree?: TreeNode;
  private nFeatures?: number;
  private fitted = false;

  /**
   * @param options - Hyperparameters, see {@link DecisionTreeRegressorOptions}
   * @throws {InvalidParameterError} If an option is out of range
   */
  constructor(options: DecisionTreeRegressorOptions = {}) {
    this.maxDepth = checkTreeMaxDepth(options.maxDepth ?? 10);
    this.minSamplesSplit = checkTreeInteger("minSamplesSplit", options.minSamplesSplit ?? 2, 2);
    this.minSamplesLeaf = checkTreeInteger("minSamplesLeaf", options.minSamplesLeaf ?? 1, 1);
    this.maxFeatures = checkTreeMaxFeatures(options.maxFeatures);
    this.randomState = checkTreeRandomState(options.randomState);
    this.minImpurityDecrease = checkMinImpurityDecrease(options.minImpurityDecrease ?? 0);
    this.maxLeafNodes = checkMaxLeafNodes(options.maxLeafNodes);
    this.ccpAlpha = checkCcpAlpha(options.ccpAlpha ?? 0);
  }

  /**
   * Build a decision tree regressor from the training set (X, y).
   *
   * With `sampleWeight` the node means, the impurity and the split search use weighted sums,
   * while `minSamplesSplit` and `minSamplesLeaf` still count samples. Samples with weight 0 are
   * ignored.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target values of shape (n_samples,)
   * @param sampleWeight - Optional non-negative weights of shape (n_samples,)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D or y is not 1D
   * @throws {ShapeError} If X and y have different number of samples
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
    const indices = positiveWeightIndices(nSamples, weights);
    let totalWeight = 0;
    for (let i = 0; i < indices.length; i++) {
      totalWeight += weights === undefined ? 1 : (weights[indices[i] as number] as number);
    }

    const ctx: RegressorContext = {
      xc: toColumnMajor(toFloat64View(X), nSamples, nFeatures),
      nSamples,
      y: toFloat64View(y),
      weights,
      totalWeight,
      sampler: new TreeFeatureSampler(
        nFeatures,
        resolveTreeMaxFeatures(this.maxFeatures, nFeatures),
        createTreeRng(this.randomState)
      ),
      col: new Float64Array(nSamples),
      positions: new Int32Array(nSamples),
      yLocal: new Float64Array(nSamples),
    };

    this.tree = growTree(indices, (idx, depth) => this.expandNode(ctx, idx, depth), {
      maxLeafNodes: this.maxLeafNodes,
      ccpAlpha: this.ccpAlpha,
    });
    this.nFeatures = nFeatures;
    this.fitted = true;
    return this;
  }

  private expandNode(ctx: RegressorContext, indices: Int32Array, depth: number): ExpandResult {
    const n = indices.length;
    const { y, yLocal, weights } = ctx;

    const { mean, lo, hi, weight, impurity } = nodeTargetStats(y, indices, weights);
    const leaf: TreeNode = {
      isLeaf: true,
      prediction: mean,
      nSamples: n,
      weightedNSamples: weight,
      impurity,
    };

    if (
      depth >= this.maxDepth ||
      n < this.minSamplesSplit ||
      n < 2 * this.minSamplesLeaf ||
      lo === hi
    ) {
      return { node: leaf };
    }

    // Targets centered on the node mean keep the split scores free of the cancellation
    // that sumL^2/nL + sumR^2/nR suffers when |mean| dwarfs the spread.
    for (let i = 0; i < n; i++) yLocal[i] = (y[indices[i] as number] as number) - mean;

    const split = this.findBestSplit(ctx, indices, weight);
    if (split === undefined) return { node: leaf };
    if (!splitIsWorthwhile(split.decrease, ctx.totalWeight, this.minImpurityDecrease)) {
      return { node: leaf };
    }

    const children = partitionIndices(
      ctx.xc,
      split.feature * ctx.nSamples,
      indices,
      split.threshold
    );
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

  /**
   * Split search over the node's centered targets in `ctx.yLocal[0..n)`; `weight` is the sum of
   * the sample weights of the node.
   */
  private findBestSplit(
    ctx: RegressorContext,
    indices: Int32Array,
    weight: number
  ): SplitResult | undefined {
    const n = indices.length;
    const { xc, nSamples, col, positions, yLocal, sampler, weights } = ctx;
    const minLeaf = this.minSamplesLeaf;

    let totalSum = 0;
    for (let i = 0; i < n; i++) {
      const w = weights === undefined ? 1 : (weights[indices[i] as number] as number);
      totalSum += w * (yLocal[i] as number);
    }

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
        col[i] = v;
        positions[i] = i;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      if (lo === hi) continue;
      usable++;

      sortPositionsByColumn(col, positions, n);

      let leftSum = 0;
      let leftWeight = 0;
      for (let i = 0; i < n - 1; i++) {
        const pos = positions[i] as number;
        const w = weights === undefined ? 1 : (weights[indices[pos] as number] as number);
        leftSum += w * (yLocal[pos] as number);
        leftWeight += w;

        const val = col[pos] as number;
        const nextVal = col[positions[i + 1] as number] as number;
        if (val === nextVal) continue; // cannot split between equal values

        const leftCnt = i + 1;
        const rightCnt = n - leftCnt;
        if (leftCnt < minLeaf || rightCnt < minLeaf) continue;
        const rightWeight = weight - leftWeight;
        if (!(rightWeight > 0)) continue;

        // Maximizing sumL^2/WL + sumR^2/WR minimizes the weighted child MSE.
        const rightSum = totalSum - leftSum;
        const score = (leftSum * leftSum) / leftWeight + (rightSum * rightSum) / rightWeight;
        if (score > bestScore) {
          bestScore = score;
          bestFeature = f;
          bestThreshold = splitThreshold(val, nextVal);
        }
      }
    }

    if (bestFeature < 0) return undefined;
    // W * mse(node) - WL * mse(left) - WR * mse(right) on the centered targets.
    return {
      feature: bestFeature,
      threshold: bestThreshold,
      decrease: Math.max(0, bestScore - (totalSum * totalSum) / weight),
    };
  }

  /**
   * Predict target values for samples in X.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns `float64` predictions of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    const tree = this.requireTree();
    validatePredictInputs(X, this.nFeatures ?? 0, "DecisionTreeRegressor");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const x = toFloat64View(X);
    const out = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      out[i] = descend(tree, x, i * nFeatures).prediction ?? 0;
    }
    return tensor(out, { dtype: "float64" });
  }

  /**
   * Rewrite each leaf's prediction via `fn(oldValue)`. Used by gradient
   * boosting to apply the TreeBoost Newton leaf-value update so that both
   * training and inference use the corrected terminal values. Leaves are visited
   * left to right (depth first).
   */
  remapLeaves(fn: (leafValue: number) => number): void {
    if (!this.tree) return;
    const copyOf = (node: TreeNode): MutableTreeNode =>
      node.isLeaf ? { ...node, prediction: fn(node.prediction ?? 0) } : { ...node };
    let root: MutableTreeNode | undefined;
    type Job = { source: TreeNode; attach: (copy: MutableTreeNode) => void };
    const stack: Job[] = [
      {
        source: this.tree,
        attach: (copy) => {
          root = copy;
        },
      },
    ];
    for (let job = stack.pop(); job !== undefined; job = stack.pop()) {
      const copy = copyOf(job.source);
      job.attach(copy);
      const { left, right } = job.source;
      if (right) {
        stack.push({
          source: right,
          attach: (child) => {
            copy.right = child;
          },
        });
      }
      if (left) {
        stack.push({
          source: left,
          attach: (child) => {
            copy.left = child;
          },
        });
      }
    }
    this.tree = root as unknown as TreeNode;
  }

  /**
   * Return the R² score on the given test data and target values.
   *
   * R² = 1 - SS_res / SS_tot. When all targets are equal the score is 1 for a perfect
   * prediction and 0 otherwise.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True target values of shape (n_samples,)
   * @returns R² score (best possible is 1.0, can be negative)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional or sample counts mismatch
   * @throws {DataValidationError} If y is empty or contains NaN/Inf values
   */
  score(X: Tensor, y: Tensor): number {
    this.requireTree();
    const truth = readScoreTargets(y);
    const predictions = toFloat64View(this.predict(X));
    if (predictions.length !== truth.length) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${predictions.length}, y=${truth.length}`
      );
    }

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

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
      minSamplesLeaf: this.minSamplesLeaf,
      maxFeatures: this.maxFeatures,
      randomState: this.randomState,
      minImpurityDecrease: this.minImpurityDecrease,
      maxLeafNodes: this.maxLeafNodes,
      ccpAlpha: this.ccpAlpha,
    };
  }

  /**
   * Set the parameters of this estimator. Either all given parameters are applied or, if one
   * is invalid, none.
   *
   * @param params - Parameters to set
   * @returns this
   * @throws {InvalidParameterError} If a parameter value is invalid or unknown
   */
  setParams(params: Record<string, unknown>): this {
    const next = {
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
      minSamplesLeaf: this.minSamplesLeaf,
      maxFeatures: this.maxFeatures,
      randomState: this.randomState,
      minImpurityDecrease: this.minImpurityDecrease,
      maxLeafNodes: this.maxLeafNodes,
      ccpAlpha: this.ccpAlpha,
    };
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "maxDepth":
          next.maxDepth = checkTreeMaxDepth(value);
          break;
        case "minSamplesSplit":
          next.minSamplesSplit = checkTreeInteger("minSamplesSplit", value, 2);
          break;
        case "minSamplesLeaf":
          next.minSamplesLeaf = checkTreeInteger("minSamplesLeaf", value, 1);
          break;
        case "maxFeatures":
          next.maxFeatures = checkTreeMaxFeatures(value);
          break;
        case "randomState":
          next.randomState = checkTreeRandomState(value);
          break;
        case "minImpurityDecrease":
          next.minImpurityDecrease = checkMinImpurityDecrease(value);
          break;
        case "maxLeafNodes":
          next.maxLeafNodes = checkMaxLeafNodes(value);
          break;
        case "ccpAlpha":
          next.ccpAlpha = checkCcpAlpha(value);
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    Object.assign(this, next);
    return this;
  }

  /** Access the internal tree structure (for export_text). */
  get tree_(): TreeNode | undefined {
    return this.tree;
  }

  /** Number of features seen during fit. */
  get nFeatures_(): number | undefined {
    return this.nFeatures;
  }

  /**
   * Depth of the fitted tree (a single leaf has depth 0).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  getDepth(): number {
    return treeDepth(this.requireTree());
  }

  /**
   * Number of leaves of the fitted tree.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  getNLeaves(): number {
    return treeLeafCount(this.requireTree());
  }

  /**
   * Normalized mean decrease in impurity per feature (sums to 1, or all zeros when the tree
   * never split).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get featureImportances(): Tensor {
    if (!this.fitted || !this.tree || this.nFeatures === undefined) {
      throw new NotFittedError(
        "DecisionTreeRegressor must be fitted to access feature_importances_"
      );
    }
    return normalizedImportances(this.tree, this.nFeatures);
  }

  /**
   * Create an unfitted copy with the same hyperparameters.
   */
  clone(): DecisionTreeRegressor {
    return new DecisionTreeRegressor(this.getParams() as DecisionTreeRegressorOptions);
  }

  private requireTree(): TreeNode {
    if (!this.fitted || !this.tree) {
      throw new NotFittedError("DecisionTreeRegressor must be fitted before prediction");
    }
    return this.tree;
  }
}

/**
 * Build a text representation of a decision tree.
 *
 * Works with both DecisionTreeClassifier and DecisionTreeRegressor.
 *
 * @param tree - A fitted DecisionTreeClassifier or DecisionTreeRegressor
 * @param options - Optional feature names and the number of decimals (0 to 100, default 2)
 * @returns Multi-line string representation of the tree
 * @throws {NotFittedError} If the tree has not been fitted
 * @throws {InvalidParameterError} If `decimals` is not an integer in [0, 100]
 *
 * @example
 * ```ts
 * const clf = new DecisionTreeClassifier({ maxDepth: 3 });
 * clf.fit(X, y);
 * console.log(exportText(clf));
 * // |--- feature_1 <= 3.50
 * // |   |--- class: 0
 * // |--- feature_1 > 3.50
 * // |   |--- class: 1
 * ```
 *
 * @deprecated Prefer {@link exportText}.
 */
export function export_text(
  tree: DecisionTreeClassifier | DecisionTreeRegressor,
  options: { featureNames?: readonly string[]; decimals?: number } = {}
): string {
  const root = tree.tree_;
  if (!root) {
    throw new NotFittedError("Tree must be fitted before export");
  }

  const decimals = options.decimals ?? 2;
  if (!Number.isInteger(decimals) || decimals < 0 || decimals > 100) {
    throw new InvalidParameterError(
      `decimals must be an integer in [0, 100]; received ${String(decimals)}`,
      "decimals",
      decimals
    );
  }
  const featureNames = options.featureNames;
  const lines: string[] = [];

  function featureName(idx: number): string {
    if (featureNames && idx < featureNames.length) {
      return featureNames[idx] ?? `feature_${idx}`;
    }
    return `feature_${idx}`;
  }

  // Explicit stack instead of recursion: an entry is either a finished line or a node still to
  // print, so very deep trees do not overflow the call stack.
  type Entry = string | { readonly node: TreeNode; readonly prefix: string };
  const connector = "|--- ";
  const stack: Entry[] = [{ node: root, prefix: "" }];
  for (let entry = stack.pop(); entry !== undefined; entry = stack.pop()) {
    if (typeof entry === "string") {
      lines.push(entry);
      continue;
    }
    const { node, prefix } = entry;
    if (node.isLeaf) {
      if (node.classProbabilities !== undefined) {
        // Class labels are printed exactly as they are stored.
        lines.push(`${prefix}${connector}class: ${String(node.prediction ?? "?")}`);
        continue;
      }
      const val =
        node.prediction !== undefined
          ? Number.isInteger(node.prediction)
            ? String(node.prediction)
            : node.prediction.toFixed(decimals)
          : "?";
      lines.push(`${prefix}${connector}value: ${val}`);
      continue;
    }

    const fname = featureName(node.featureIndex ?? 0);
    const thresh = (node.threshold ?? 0).toFixed(decimals);
    lines.push(`${prefix}${connector}${fname} <= ${thresh}`);
    if (node.right) stack.push({ node: node.right, prefix: `${prefix}|   ` });
    stack.push(`${prefix}${connector}${fname} > ${thresh}`);
    if (node.left) stack.push({ node: node.left, prefix: `${prefix}|   ` });
  }
  return lines.join("\n");
}

/**
 * Build a text representation of a decision tree.
 *
 * Works with both DecisionTreeClassifier and DecisionTreeRegressor.
 *
 * @param tree - A fitted DecisionTreeClassifier or DecisionTreeRegressor
 * @param options - Optional feature names and the number of decimals (0 to 100, default 2)
 * @returns Multi-line string representation of the tree
 * @throws {NotFittedError} If the tree has not been fitted
 * @throws {InvalidParameterError} If `decimals` is not an integer in [0, 100]
 *
 * @example
 * ```ts
 * const clf = new DecisionTreeClassifier({ maxDepth: 3 });
 * clf.fit(X, y);
 * console.log(exportText(clf));
 * // |--- feature_1 <= 3.50
 * // |   |--- class: 0
 * // |--- feature_1 > 3.50
 * // |   |--- class: 1
 * ```
 */
export const exportText = export_text;
