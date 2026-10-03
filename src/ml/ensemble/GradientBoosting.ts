/**
 * @see {@link https://deepbox.dev/docs/ml-ensemble | Deepbox Ensemble Methods}
 */

import {
  DataValidationError,
  DeepboxError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __randomBelow, __SeededRandom } from "../../random/random";
import {
  assertContiguous,
  percentileSorted,
  toFloat64View,
  validateFitInputs,
  validatePredictInputs,
} from "../_validation";
import type { Classifier, Regressor } from "../base";
import {
  checkCcpAlpha,
  checkMaxLeafNodes,
  checkMinImpurityDecrease,
  readSampleWeight,
  type TreeGrowthOptions,
} from "../tree/_growth";
import { createTreeRng, DecisionTreeRegressor } from "../tree/DecisionTree";
import { applyTree, type FlatTree, flattenTree, predictRegressionTree } from "./_tree";

export { predictRegressionTree };

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** `out[i] += scale * tree(x_i)` for the `n` rows of `x`. */
function addTree(
  f: FlatTree,
  x: Float64Array,
  n: number,
  d: number,
  out: Float64Array,
  scale: number,
  scratch: Int32Array
): void {
  applyTree(f, x, n, d, scratch);
  const values = f.leafValue;
  for (let i = 0; i < n; i++) out[i] = out[i]! + scale * values[scratch[i]!]!;
}

/**
 * Store new leaf values on both the fitted tree (so it keeps predicting them) and the
 * flattened copy. `values` must be in the leaf order of {@link flattenTree}, which is
 * the order `remapLeaves` visits leaves in.
 */
function setLeafValues(tree: DecisionTreeRegressor, flat: FlatTree, values: Float64Array): void {
  let k = 0;
  tree.remapLeaves(() => values[k++] as number);
  flat.leafValue.set(values);
}

/** Group row positions by leaf with a counting sort. Returns `order` and per-leaf `start` offsets. */
function groupByLeaf(
  leafOfRow: Int32Array,
  rows: Int32Array | null,
  nRows: number,
  nLeaves: number
): { order: Int32Array; start: Int32Array } {
  const count = rows === null ? nRows : rows.length;
  const start = new Int32Array(nLeaves + 1);
  for (let k = 0; k < count; k++) {
    const leaf = leafOfRow[rows === null ? k : rows[k]!]!;
    start[leaf + 1] = start[leaf + 1]! + 1;
  }
  for (let l = 0; l < nLeaves; l++) start[l + 1] = start[l + 1]! + start[l]!;
  const cursor = start.slice(0, nLeaves);
  const order = new Int32Array(count);
  for (let k = 0; k < count; k++) {
    const row = rows === null ? k : rows[k]!;
    const leaf = leafOfRow[row]!;
    order[cursor[leaf]!] = row;
    cursor[leaf] = cursor[leaf]! + 1;
  }
  return { order, start };
}

/** Quantile `q` in [0, 1] of `values` with linear interpolation (like `numpy.quantile`). */
function quantileOf(values: ArrayLike<number>, q: number): number {
  const sorted = Float64Array.from(values).sort();
  return percentileSorted(sorted, q * 100);
}

/**
 * Smallest sample value whose cumulative share reaches `q` (the "inverted CDF" quantile, with
 * `numpy.quantile(..., method="inverted_cdf")` semantics). This is the exact minimizer of the
 * pinball loss on the sample and matches how scikit-learn picks leaf values and the Huber
 * transition point; for an even count the median is the lower of the two middle values.
 */
function invertedCdfQuantile(values: ArrayLike<number>, q: number): number {
  const sorted = Float64Array.from(values).sort();
  const rank = ((q * 100) / 100) * sorted.length;
  const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(rank) - 1));
  return sorted[index]!;
}

/**
 * Weighted quantile `q` in [0, 1]: the smallest value whose cumulative weight reaches
 * `q * totalWeight` (scikit-learn's weighted percentile). Without `weights` it is
 * {@link invertedCdfQuantile}.
 */
function weightedQuantile(
  values: ArrayLike<number>,
  weights: ArrayLike<number> | undefined,
  q: number
): number {
  if (weights === undefined) return invertedCdfQuantile(values, q);
  const n = values.length;
  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => values[a]! - values[b]!);
  let total = 0;
  for (let i = 0; i < n; i++) total += weights[i]!;
  const target = q * total;
  let cumulative = 0;
  for (let k = 0; k < n; k++) {
    cumulative += weights[order[k]!]!;
    if (cumulative >= target) return values[order[k]!]!;
  }
  return values[order[n - 1]!]!;
}

/** Sum of `weights`, or `count` when there are no weights. */
function totalOf(weights: Float64Array | undefined, count: number): number {
  if (weights === undefined) return count;
  let total = 0;
  for (let i = 0; i < weights.length; i++) total += weights[i]!;
  return total;
}

/** Draw `k` distinct row indices out of `n` (partial Fisher-Yates). */
function sampleWithoutReplacement(rng: () => number, n: number, k: number): Int32Array {
  const all = new Int32Array(n);
  for (let i = 0; i < n; i++) all[i] = i;
  for (let i = 0; i < k; i++) {
    const j = i + __randomBelow(rng, n - i);
    const tmp = all[i]!;
    all[i] = all[j]!;
    all[j] = tmp;
  }
  return all.slice(0, k);
}

function floatMatrix(data: Float64Array, rows: number, cols: number): Tensor {
  return tensor(data, { dtype: "float64" }).reshape([rows, cols]);
}

function gatherRows(x: Float64Array, d: number, rows: Int32Array): Float64Array {
  const out = new Float64Array(rows.length * d);
  for (let r = 0; r < rows.length; r++) {
    const src = rows[r]! * d;
    out.set(x.subarray(src, src + d), r * d);
  }
  return out;
}

function gatherValues(v: Float64Array, rows: Int32Array): Float64Array {
  const out = new Float64Array(rows.length);
  for (let r = 0; r < rows.length; r++) out[r] = v[rows[r]!]!;
  return out;
}

/** Numerically stable logistic function. */
function sigmoid(z: number): number {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

/**
 * Mean of the per-tree normalized importances over all trees, normalized to sum to one.
 * Trees that never split contribute zeros.
 */
function averageTreeImportances(trees: DecisionTreeRegressor[], nFeatures: number): Tensor {
  const avg = new Float64Array(nFeatures);
  for (const tree of trees) {
    const imp = toFloat64View(tree.featureImportances);
    for (let j = 0; j < nFeatures; j++) avg[j] = avg[j]! + imp[j]!;
  }
  let total = 0;
  for (let j = 0; j < nFeatures; j++) {
    avg[j] = avg[j]! / trees.length;
    total += avg[j]!;
  }
  if (total > 0) {
    for (let j = 0; j < nFeatures; j++) avg[j] = avg[j]! / total;
  }
  return tensor(avg);
}

/** Throw unless every label is an integer that fits in int32 (the dtype of `predict`). */
function assertIntegerLabels(y: Float64Array, who: string): void {
  for (let i = 0; i < y.length; i++) {
    const v = y[i]!;
    if (!Number.isInteger(v) || v < -2147483648 || v > 2147483647) {
      throw new DataValidationError(
        `${who} requires integer class labels in the int32 range; y[${i}] = ${v}`
      );
    }
  }
}

/** Shared `score` input checks; returns the targets as a flat array. */
function checkScoreTarget(y: Tensor, nPredicted: number): Float64Array {
  if (y.ndim !== 1) {
    throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
  }
  assertContiguous(y, "y");
  const yv = toFloat64View(y);
  for (let i = 0; i < yv.length; i++) {
    if (!Number.isFinite(yv[i])) {
      throw new DataValidationError("y contains non-finite values (NaN or Inf)");
    }
  }
  if (yv.length !== nPredicted) {
    throw new ShapeError(
      `X and y must have the same number of samples; got X=${nPredicted}, y=${yv.length}`
    );
  }
  if (yv.length === 0) {
    throw new DataValidationError("y must contain at least one sample");
  }
  return yv;
}

function checkNEstimators(value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new InvalidParameterError("nEstimators must be an integer >= 1", "nEstimators", value);
  }
  return value;
}

function checkLearningRate(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new InvalidParameterError(
      "learningRate must be a finite number > 0",
      "learningRate",
      value
    );
  }
  return value;
}

function checkMaxDepth(value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new InvalidParameterError("maxDepth must be an integer >= 1", "maxDepth", value);
  }
  return value;
}

function checkMinSamplesSplit(value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 2) {
    throw new InvalidParameterError(
      "minSamplesSplit must be an integer >= 2",
      "minSamplesSplit",
      value
    );
  }
  return value;
}

function checkMinSamplesLeaf(value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new InvalidParameterError(
      "minSamplesLeaf must be an integer >= 1",
      "minSamplesLeaf",
      value
    );
  }
  return value;
}

function checkWarmStart(value: unknown): boolean {
  if (typeof value !== "boolean") {
    throw new InvalidParameterError("warmStart must be a boolean", "warmStart", value);
  }
  return value;
}

function checkSubsample(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0 || value > 1) {
    throw new InvalidParameterError("subsample must be in (0, 1]", "subsample", value);
  }
  return value;
}

function checkMaxFeatures(value: unknown): "sqrt" | "log2" | number | undefined {
  if (
    value !== undefined &&
    value !== "sqrt" &&
    value !== "log2" &&
    (typeof value !== "number" || !Number.isInteger(value) || value < 1)
  ) {
    throw new InvalidParameterError(
      'maxFeatures must be "sqrt", "log2", an integer >= 1, or undefined',
      "maxFeatures",
      value
    );
  }
  return value;
}

function checkRandomState(value: unknown): number | undefined {
  if (value !== undefined && (typeof value !== "number" || !Number.isFinite(value))) {
    throw new InvalidParameterError("randomState must be a finite number", "randomState", value);
  }
  return value;
}

function checkValidationFraction(value: unknown): number {
  if (typeof value !== "number" || !(value > 0 && value < 1)) {
    throw new InvalidParameterError(
      "validationFraction must be in (0, 1)",
      "validationFraction",
      value
    );
  }
  return value;
}

function checkNIterNoChange(value: unknown): number | undefined {
  if (value !== undefined && (typeof value !== "number" || !Number.isInteger(value) || value < 1)) {
    throw new InvalidParameterError(
      "nIterNoChange must be an integer >= 1 or undefined",
      "nIterNoChange",
      value
    );
  }
  return value;
}

function checkAlpha(value: unknown): number {
  if (typeof value !== "number" || !(value > 0 && value < 1)) {
    throw new InvalidParameterError("alpha must be in (0, 1)", "alpha", value);
  }
  return value;
}

/** Number of features a tree may look at per split, or `undefined` for all of them. */
function resolveMaxFeatures(
  maxFeatures: "sqrt" | "log2" | number | undefined,
  nFeatures: number
): number | undefined {
  if (maxFeatures === undefined) return undefined;
  if (typeof maxFeatures === "number") return Math.max(1, Math.min(maxFeatures, nFeatures));
  if (maxFeatures === "sqrt") return Math.max(1, Math.floor(Math.sqrt(nFeatures)));
  return Math.max(1, Math.floor(Math.log2(nFeatures)));
}

type TreeOptions = {
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
  minImpurityDecrease: number;
  ccpAlpha: number;
  maxLeafNodes?: number;
  maxFeatures?: number;
  randomState?: number;
};

type TreeBase = {
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
  minImpurityDecrease: number;
  maxLeafNodes: number | undefined;
  ccpAlpha: number;
};

/**
 * Regression tree for one boosting stage. scikit-learn grows the stages with the `friedman_mse`
 * criterion, whose improvement is the weighted decrease of the squared error, not divided by the
 * total weight of the training rows, so `minImpurityDecrease` is scaled by `fitWeight` (the total
 * weight of the rows this tree is fitted on) to give the tree the same threshold.
 */
function newTree(
  opts: TreeBase,
  treeMaxFeatures: number | undefined,
  rng: () => number,
  fitWeight: number
): DecisionTreeRegressor {
  const treeOpts: TreeOptions = {
    maxDepth: opts.maxDepth,
    minSamplesSplit: opts.minSamplesSplit,
    minSamplesLeaf: opts.minSamplesLeaf,
    minImpurityDecrease: opts.minImpurityDecrease / fitWeight,
    ccpAlpha: opts.ccpAlpha,
  };
  if (opts.maxLeafNodes !== undefined) treeOpts.maxLeafNodes = opts.maxLeafNodes;
  if (treeMaxFeatures !== undefined) {
    treeOpts.maxFeatures = treeMaxFeatures;
    // The tree seeds its own feature sampling, so hand it a seed from our stream.
    treeOpts.randomState = Math.floor(rng() * 4294967296);
  }
  return new DecisionTreeRegressor(treeOpts);
}

function definedOptions<T>(params: Record<string, unknown>): T {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) out[key] = value;
  }
  return out as T;
}

// ---------------------------------------------------------------------------
// Regressor
// ---------------------------------------------------------------------------

type RegressionLoss = "ls" | "lad" | "huber" | "quantile";

/**
 * Loss names accepted by {@link GradientBoostingRegressor}. `"squared_error"` and
 * `"absolute_error"` are the scikit-learn spellings of `"ls"` and `"lad"`.
 */
export type GradientBoostingLoss = RegressionLoss | "squared_error" | "absolute_error";

function canonicalLoss(value: unknown): RegressionLoss {
  switch (value) {
    case "ls":
    case "squared_error":
      return "ls";
    case "lad":
    case "absolute_error":
      return "lad";
    case "huber":
    case "quantile":
      return value;
    default:
      throw new InvalidParameterError(
        `loss must be "ls", "lad", "huber", "quantile", "squared_error" or "absolute_error"`,
        "loss",
        value
      );
  }
}

function checkLoss(value: unknown): GradientBoostingLoss {
  canonicalLoss(value);
  return value as GradientBoostingLoss;
}

/** Tree growth options of the boosting stages. */
type BoostingTreeOptions = {
  /**
   * A split is made only if it lowers the weighted squared error of the residuals by at least
   * this value. As with scikit-learn's `friedman_mse` trees the decrease is not divided by the
   * total weight, so it grows with the number of samples. Default 0.
   */
  readonly minImpurityDecrease?: TreeGrowthOptions["minImpurityDecrease"];
  /** Grow every tree best-first up to this many leaves (an integer >= 2). Default: no limit. */
  readonly maxLeafNodes?: TreeGrowthOptions["maxLeafNodes"];
  /** Cost-complexity pruning strength of every tree (see {@link TreeGrowthOptions}). Default 0. */
  readonly ccpAlpha?: TreeGrowthOptions["ccpAlpha"];
};

/** Options of {@link GradientBoostingRegressor}. */
export type GradientBoostingRegressorOptions = {
  readonly nEstimators?: number;
  readonly learningRate?: number;
  readonly maxDepth?: number;
  readonly minSamplesSplit?: number;
  readonly minSamplesLeaf?: number;
  readonly warmStart?: boolean;
  readonly subsample?: number;
  readonly maxFeatures?: "sqrt" | "log2" | number;
  readonly validationFraction?: number;
  readonly nIterNoChange?: number;
  readonly loss?: GradientBoostingLoss;
  readonly alpha?: number;
  readonly randomState?: number;
} & BoostingTreeOptions;

/**
 * Gradient Boosting Regressor.
 *
 * Builds an additive model in a forward stage-wise fashion using regression trees as
 * weak learners. Each tree is fitted to the negative gradient of the loss; for the
 * `"lad"`, `"huber"` and `"quantile"` losses the leaf values are then replaced by the
 * line-search optimum for that loss (leaf median, Huber-corrected median or leaf
 * quantile of the residuals), as in Friedman's TreeBoost and scikit-learn.
 *
 * **Losses**: `"ls"` squared error (initial value: the mean of y), `"lad"` absolute error
 * (initial value: the median), `"huber"` (initial value: the median; the transition point is
 * the `alpha` quantile of the absolute residuals) and `"quantile"` (initial value: the
 * `alpha` quantile of y).
 *
 * With `subsample < 1` each stage trains on a random subset drawn without replacement.
 * With `nIterNoChange` set, `validationFraction` of the rows are held out at random and
 * boosting stops after `nIterNoChange` stages without a validation-loss improvement.
 *
 * @example
 * ```ts
 * import { GradientBoostingRegressor } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1], [2], [3], [4], [5]]);
 * const y = tensor([1.2, 2.1, 2.9, 4.0, 5.1]);
 *
 * const gbr = new GradientBoostingRegressor({ nEstimators: 100, randomState: 0 });
 * gbr.fit(X, y);
 * const predictions = gbr.predict(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-ensemble | Deepbox Ensemble Methods}
 */
export class GradientBoostingRegressor implements Regressor {
  private nEstimators: number;
  private learningRate: number;
  private maxDepth: number;
  private minSamplesSplit: number;
  private minSamplesLeaf: number;
  private warmStart: boolean;
  private subsample: number;
  private maxFeatures: "sqrt" | "log2" | number | undefined;
  private validationFraction: number;
  private nIterNoChange: number | undefined;
  private loss: GradientBoostingLoss;
  private alpha: number;
  private randomState: number | undefined;
  private minImpurityDecrease: number;
  private maxLeafNodes: number | undefined;
  private ccpAlpha: number;

  private estimators: DecisionTreeRegressor[] = [];
  private flatTrees: FlatTree[] = [];
  private initPrediction = 0;
  private nFeatures = 0;
  private fitted = false;
  private rng: (() => number) | undefined;
  private splitSeed: number | undefined;

  /**
   * @param options.nEstimators - Maximum number of boosting stages (default: 100)
   * @param options.learningRate - Shrinks each tree's contribution, must be > 0 (default: 0.1)
   * @param options.maxDepth - Maximum depth of each tree (default: 3)
   * @param options.minSamplesSplit - Minimum rows needed to split a node, >= 2 (default: 2)
   * @param options.minSamplesLeaf - Minimum rows in a leaf, >= 1 (default: 1)
   * @param options.warmStart - Keep the trees of the previous fit and add more (default: false)
   * @param options.subsample - Fraction of rows used per stage, in (0, 1] (default: 1.0)
   * @param options.maxFeatures - Features tried per split: `"sqrt"`, `"log2"` or an integer count
   * @param options.validationFraction - Fraction of rows held out for early stopping (default: 0.1)
   * @param options.nIterNoChange - Stop after this many stages without validation improvement;
   *   early stopping is off when undefined
   * @param options.loss - `"ls"`, `"lad"`, `"huber"` or `"quantile"` (default: "ls")
   * @param options.alpha - Quantile for the huber and quantile losses, in (0, 1) (default: 0.9)
   * @param options.randomState - Seed for subsampling, feature sampling and the validation split.
   *   Without it the global Deepbox generator is used, so `setSeed` also makes the fit reproducible.
   * @param options.minImpurityDecrease - Minimum decrease of the weighted squared error of the
   *   residuals for a split, not divided by the total weight (scikit-learn's `friedman_mse`
   *   improvement) (default: 0)
   * @param options.maxLeafNodes - Grow every tree best-first up to this many leaves (default: no limit)
   * @param options.ccpAlpha - Cost-complexity pruning strength of every tree (default: 0)
   * @throws {InvalidParameterError} If an option is out of range
   */
  constructor(options: GradientBoostingRegressorOptions = {}) {
    this.nEstimators = checkNEstimators(options.nEstimators ?? 100);
    this.learningRate = checkLearningRate(options.learningRate ?? 0.1);
    this.maxDepth = checkMaxDepth(options.maxDepth ?? 3);
    this.minSamplesSplit = checkMinSamplesSplit(options.minSamplesSplit ?? 2);
    this.minSamplesLeaf = checkMinSamplesLeaf(options.minSamplesLeaf ?? 1);
    this.warmStart = checkWarmStart(options.warmStart ?? false);
    this.subsample = checkSubsample(options.subsample ?? 1.0);
    this.maxFeatures = checkMaxFeatures(options.maxFeatures);
    this.validationFraction = checkValidationFraction(options.validationFraction ?? 0.1);
    this.nIterNoChange = checkNIterNoChange(options.nIterNoChange);
    this.loss = checkLoss(options.loss ?? "ls");
    this.alpha = checkAlpha(options.alpha ?? 0.9);
    this.randomState = checkRandomState(options.randomState);
    this.minImpurityDecrease = checkMinImpurityDecrease(options.minImpurityDecrease ?? 0);
    this.maxLeafNodes = checkMaxLeafNodes(options.maxLeafNodes);
    this.ccpAlpha = checkCcpAlpha(options.ccpAlpha ?? 0);
  }

  /**
   * Initial constant prediction that minimizes the loss on `y`: the (weighted) mean, median or
   * `alpha` quantile. With weights the quantiles are weighted percentiles that pick a sample
   * value, as in scikit-learn.
   */
  private computeInit(y: Float64Array, weights: Float64Array | undefined): number {
    const n = y.length;
    switch (canonicalLoss(this.loss)) {
      case "ls": {
        let sum = 0;
        let total = 0;
        for (let i = 0; i < n; i++) {
          const w = weights === undefined ? 1 : weights[i]!;
          sum += w * y[i]!;
          total += w;
        }
        return sum / total;
      }
      case "lad":
      case "huber":
        return weights === undefined ? quantileOf(y, 0.5) : weightedQuantile(y, weights, 0.5);
      case "quantile":
        return weights === undefined
          ? quantileOf(y, this.alpha)
          : weightedQuantile(y, weights, this.alpha);
    }
  }

  /** Huber transition point: the (weighted) `alpha` quantile of the absolute residuals. */
  private huberDelta(
    y: Float64Array,
    raw: Float64Array,
    weights: Float64Array | undefined
  ): number {
    const abs = new Float64Array(y.length);
    for (let i = 0; i < y.length; i++) abs[i] = Math.abs(y[i]! - raw[i]!);
    return weightedQuantile(abs, weights, this.alpha);
  }

  /** Negative gradient of the loss at the current predictions. */
  private negGradient(y: Float64Array, raw: Float64Array, delta: number): Float64Array {
    const n = y.length;
    const out = new Float64Array(n);
    switch (canonicalLoss(this.loss)) {
      case "ls":
        for (let i = 0; i < n; i++) out[i] = y[i]! - raw[i]!;
        break;
      case "lad":
        for (let i = 0; i < n; i++) out[i] = Math.sign(y[i]! - raw[i]!);
        break;
      case "huber":
        for (let i = 0; i < n; i++) {
          const diff = y[i]! - raw[i]!;
          out[i] = Math.abs(diff) <= delta ? diff : delta * Math.sign(diff);
        }
        break;
      case "quantile":
        for (let i = 0; i < n; i++) out[i] = y[i]! - raw[i]! >= 0 ? this.alpha : this.alpha - 1;
        break;
    }
    return out;
  }

  /** Weighted mean loss, used on the held-out rows for early stopping. */
  private meanLoss(
    y: Float64Array,
    raw: Float64Array,
    delta: number,
    weights: Float64Array | undefined
  ): number {
    const n = y.length;
    let loss = 0;
    let total = 0;
    for (let i = 0; i < n; i++) {
      const w = weights === undefined ? 1 : weights[i]!;
      total += w;
      switch (canonicalLoss(this.loss)) {
        case "ls":
          loss += w * (y[i]! - raw[i]!) ** 2;
          break;
        case "lad":
          loss += w * Math.abs(y[i]! - raw[i]!);
          break;
        case "huber": {
          const a = Math.abs(y[i]! - raw[i]!);
          loss += w * (a <= delta ? 0.5 * a * a : delta * (a - 0.5 * delta));
          break;
        }
        case "quantile": {
          const diff = y[i]! - raw[i]!;
          loss += w * (diff >= 0 ? this.alpha * diff : (this.alpha - 1) * diff);
          break;
        }
      }
    }
    return loss / total;
  }

  /**
   * Replace the leaf values of a freshly fitted tree by the optimum of the loss over the
   * rows the tree was trained on (`bagRows`, or all rows when null). Squared error needs
   * no update because the (weighted) leaf mean of the residuals already is the optimum.
   */
  private updateLeaves(
    tree: DecisionTreeRegressor,
    flat: FlatTree,
    leafOfRow: Int32Array,
    bagRows: Int32Array | null,
    y: Float64Array,
    raw: Float64Array,
    delta: number,
    weights: Float64Array | undefined
  ): void {
    const loss = canonicalLoss(this.loss);
    if (loss === "ls") return;
    const nLeaves = flat.leafValue.length;
    const { order, start } = groupByLeaf(leafOfRow, bagRows, y.length, nLeaves);
    const values = Float64Array.from(flat.leafValue);
    for (let l = 0; l < nLeaves; l++) {
      const from = start[l]!;
      const to = start[l + 1]!;
      if (to === from) continue;
      const diff = new Float64Array(to - from);
      for (let k = from; k < to; k++) diff[k - from] = y[order[k]!]! - raw[order[k]!]!;
      let leafWeights: Float64Array | undefined;
      if (weights !== undefined) {
        leafWeights = new Float64Array(to - from);
        for (let k = from; k < to; k++) leafWeights[k - from] = weights[order[k]!]!;
      }
      if (loss === "lad") {
        values[l] = weightedQuantile(diff, leafWeights, 0.5);
      } else if (loss === "quantile") {
        values[l] = weightedQuantile(diff, leafWeights, this.alpha);
      } else {
        const median = weightedQuantile(diff, leafWeights, 0.5);
        let term = 0;
        let total = 0;
        for (let k = 0; k < diff.length; k++) {
          const w = leafWeights === undefined ? 1 : leafWeights[k]!;
          const dev = diff[k]! - median;
          term += w * Math.sign(dev) * Math.min(delta, Math.abs(dev));
          total += w;
        }
        values[l] = median + term / total;
      }
    }
    setLeafValues(tree, flat, values);
  }

  /**
   * Fit the gradient boosting regressor.
   *
   * Without `warmStart` any previous ensemble is discarded. With `warmStart`, the fitted
   * trees are kept and stages are added until `nEstimators` is reached (nothing happens if
   * it already is); X must then have the same number of features as before.
   *
   * With `sampleWeight` the initial value, the tree splits, the leaf values, the Huber
   * transition point and the early-stopping loss all use the weights.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target values of shape (n_samples,)
   * @param sampleWeight - Optional non-negative weights of shape (n_samples,)
   * @returns this
   * @throws {ShapeError} If X is not 2D, y is not 1D, the sample counts differ, or a warm
   *   start sees a different number of features
   * @throws {ShapeError} If `sampleWeight` is not 1D with one entry per sample
   * @throws {DataValidationError} If X or y contain NaN/Inf, or the weights are negative, not
   *   finite or all zero
   */
  // biome-ignore lint/suspicious/noConfusingVoidType: `void` keeps this compatible with Estimator.fit(X, y, params)
  fit(X: Tensor, y: Tensor, sampleWeightArg?: Tensor | void): this {
    const warm = this.warmStart && this.fitted && this.estimators.length > 0;
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const sampleWeight = readSampleWeight(sampleWeightArg as Tensor | undefined, n);
    if (warm && d !== this.nFeatures) {
      throw new ShapeError(
        `X has ${d} features but the previous fit used ${this.nFeatures}; a warm start needs the same features`
      );
    }
    if (warm && this.estimators.length >= this.nEstimators) return this;

    const xv = toFloat64View(X);
    const yv = toFloat64View(y);
    if (!warm || this.rng === undefined) this.rng = createTreeRng(this.randomState);
    const rng = this.rng;

    // Hold out rows for early stopping.
    let trainX = xv;
    let trainY = yv;
    let trainW = sampleWeight;
    let trainN = n;
    let valX: Float64Array | undefined;
    let valY: Float64Array | undefined;
    let valW: Float64Array | undefined;
    let splitSeed = warm ? this.splitSeed : undefined;
    if (this.nIterNoChange !== undefined && n >= 2) {
      splitSeed ??= Math.floor(rng() * 4294967296);
      const splitRng = new __SeededRandom(BigInt(splitSeed));
      const valN = Math.min(n - 1, Math.max(1, Math.ceil(this.validationFraction * n - 1e-9)));
      const perm = sampleWithoutReplacement(() => splitRng.next(), n, n);
      const valRows = perm.slice(0, valN);
      const trainRows = perm.slice(valN);
      valX = gatherRows(xv, d, valRows);
      valY = gatherValues(yv, valRows);
      trainX = gatherRows(xv, d, trainRows);
      trainY = gatherValues(yv, trainRows);
      if (sampleWeight !== undefined) {
        valW = gatherValues(sampleWeight, valRows);
        trainW = gatherValues(sampleWeight, trainRows);
      }
      trainN = trainRows.length;
    }

    // Work on local copies and commit at the end, so a failed fit leaves the old model intact.
    const estimators = warm ? [...this.estimators] : [];
    const flatTrees = warm ? [...this.flatTrees] : [];
    if (trainW !== undefined && !(trainW.reduce((acc, w) => acc + w, 0) > 0)) {
      throw new DataValidationError(
        "sampleWeight must give the training rows (those left after the validation split) a positive total weight"
      );
    }
    const init = warm ? this.initPrediction : this.computeInit(trainY, trainW);

    const lr = this.learningRate;
    const raw = new Float64Array(trainN).fill(init);
    const scratch = new Int32Array(Math.max(trainN, valY?.length ?? 0));
    for (const flat of flatTrees) addTree(flat, trainX, trainN, d, raw, lr, scratch);
    const valN = valY?.length ?? 0;
    const valRaw = new Float64Array(valN).fill(init);
    if (valX !== undefined) {
      for (const flat of flatTrees) addTree(flat, valX, valN, d, valRaw, lr, scratch);
    }

    const drawSize = this.subsample < 1 ? Math.max(1, Math.floor(this.subsample * trainN)) : trainN;
    const bagged = drawSize < trainN;
    const treeMaxFeatures = resolveMaxFeatures(this.maxFeatures, d);
    const treeBase: TreeBase = {
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
      minSamplesLeaf: this.minSamplesLeaf,
      minImpurityDecrease: this.minImpurityDecrease,
      maxLeafNodes: this.maxLeafNodes,
      ccpAlpha: this.ccpAlpha,
    };
    const leafOfRow = new Int32Array(trainN);
    const isHuber = canonicalLoss(this.loss) === "huber";

    let bestValLoss = Number.POSITIVE_INFINITY;
    let noImprovement = 0;

    for (let m = estimators.length; m < this.nEstimators; m++) {
      const bagRows = bagged ? sampleWithoutReplacement(rng, trainN, drawSize) : null;
      // The Huber transition point is estimated on the rows this stage trains on.
      const bagW =
        trainW === undefined
          ? undefined
          : bagRows === null
            ? trainW
            : gatherValues(trainW, bagRows);
      const delta = isHuber
        ? bagRows === null
          ? this.huberDelta(trainY, raw, trainW)
          : this.huberDelta(gatherValues(trainY, bagRows), gatherValues(raw, bagRows), bagW)
        : 0;
      const gradient = this.negGradient(trainY, raw, delta);

      const fitX = bagRows === null ? trainX : gatherRows(trainX, d, bagRows);
      const fitG = bagRows === null ? gradient : gatherValues(gradient, bagRows);
      const nFit = bagRows === null ? trainN : bagRows.length;

      const tree = newTree(treeBase, treeMaxFeatures, rng, totalOf(bagW, nFit));
      tree.fit(
        floatMatrix(fitX, nFit, d),
        tensor(fitG, { dtype: "float64" }),
        bagW === undefined ? undefined : tensor(bagW, { dtype: "float64" })
      );
      const root = tree.tree_;
      if (!root) throw new DeepboxError("Internal error: fitted tree has no root");
      const flat = flattenTree(root);

      applyTree(flat, trainX, trainN, d, leafOfRow);
      this.updateLeaves(tree, flat, leafOfRow, bagRows, trainY, raw, delta, trainW);
      for (let i = 0; i < trainN; i++) raw[i] = raw[i]! + lr * flat.leafValue[leafOfRow[i]!]!;

      estimators.push(tree);
      flatTrees.push(flat);

      if (valX !== undefined && valY !== undefined && this.nIterNoChange !== undefined) {
        addTree(flat, valX, valN, d, valRaw, lr, scratch);
        const valLoss = this.meanLoss(valY, valRaw, delta, valW);
        // A relative tolerance keeps the test meaningful for targets of any scale.
        if (!Number.isFinite(bestValLoss) || valLoss < bestValLoss - 1e-7 * Math.abs(bestValLoss)) {
          bestValLoss = valLoss;
          noImprovement = 0;
        } else {
          noImprovement++;
          if (noImprovement >= this.nIterNoChange) break;
        }
      }
    }

    this.estimators = estimators;
    this.flatTrees = flatTrees;
    this.initPrediction = init;
    this.nFeatures = d;
    this.splitSeed = splitSeed;
    this.fitted = true;
    return this;
  }

  /** Raw model output for every row of X, in double precision. */
  private rawPredict(X: Tensor): Float64Array {
    const n = X.shape[0] ?? 0;
    const xv = toFloat64View(X);
    const out = new Float64Array(n).fill(this.initPrediction);
    const scratch = new Int32Array(n);
    for (const flat of this.flatTrees) {
      addTree(flat, xv, n, this.nFeatures, out, this.learningRate, scratch);
    }
    return out;
  }

  /**
   * Predict target values: the initial value plus the scaled contribution of every tree.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted values of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong rank or feature count
   * @throws {DataValidationError} If X contains NaN/Inf
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("GradientBoostingRegressor must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "GradientBoostingRegressor");
    return tensor(this.rawPredict(X));
  }

  /**
   * Coefficient of determination R^2 on the given data.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True target values of shape (n_samples,)
   * @returns R^2 (1.0 is perfect, it can be negative)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1D or its length differs from the number of rows in X
   * @throws {DataValidationError} If y is empty or contains NaN/Inf
   */
  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    const predictions = this.predict(X);
    const yv = checkScoreTarget(y, predictions.size);
    const pv = toFloat64View(predictions);
    let mean = 0;
    for (let i = 0; i < yv.length; i++) mean += yv[i]!;
    mean /= yv.length;
    let ssRes = 0;
    let ssTot = 0;
    for (let i = 0; i < yv.length; i++) {
      ssRes += (yv[i]! - pv[i]!) ** 2;
      ssTot += (yv[i]! - mean) ** 2;
    }
    return ssTot === 0 ? (ssRes === 0 ? 1.0 : 0.0) : 1 - ssRes / ssTot;
  }

  /**
   * Feature importances averaged across all boosting stages.
   *
   * @returns Tensor of shape (n_features,) that sums to 1 (all zeros if no tree split)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get featureImportances(): Tensor {
    if (!this.fitted || this.estimators.length === 0 || this.nFeatures === 0) {
      throw new NotFittedError(
        "GradientBoostingRegressor must be fitted to access feature_importances_"
      );
    }
    return averageTreeImportances(this.estimators, this.nFeatures);
  }

  /** Number of stages actually fitted (below `nEstimators` after early stopping). */
  get nEstimatorsFitted(): number {
    return this.estimators.length;
  }

  getParams(): Record<string, unknown> {
    return {
      nEstimators: this.nEstimators,
      learningRate: this.learningRate,
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
      minSamplesLeaf: this.minSamplesLeaf,
      warmStart: this.warmStart,
      subsample: this.subsample,
      maxFeatures: this.maxFeatures,
      validationFraction: this.validationFraction,
      nIterNoChange: this.nIterNoChange,
      loss: this.loss,
      alpha: this.alpha,
      randomState: this.randomState,
      minImpurityDecrease: this.minImpurityDecrease,
      maxLeafNodes: this.maxLeafNodes,
      ccpAlpha: this.ccpAlpha,
    };
  }

  /**
   * Set hyperparameters. Changing `randomState` restarts the random stream at the next
   * `fit`, so a later warm start no longer continues the earlier stream.
   *
   * @param params - Parameters to set
   * @throws {InvalidParameterError} If a parameter value is invalid or unknown
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nEstimators":
          this.nEstimators = checkNEstimators(value);
          break;
        case "learningRate":
          this.learningRate = checkLearningRate(value);
          break;
        case "maxDepth":
          this.maxDepth = checkMaxDepth(value);
          break;
        case "minSamplesSplit":
          this.minSamplesSplit = checkMinSamplesSplit(value);
          break;
        case "minSamplesLeaf":
          this.minSamplesLeaf = checkMinSamplesLeaf(value);
          break;
        case "warmStart":
          this.warmStart = checkWarmStart(value);
          break;
        case "subsample":
          this.subsample = checkSubsample(value);
          break;
        case "maxFeatures":
          this.maxFeatures = checkMaxFeatures(value);
          break;
        case "validationFraction":
          this.validationFraction = checkValidationFraction(value);
          break;
        case "nIterNoChange":
          this.nIterNoChange = checkNIterNoChange(value);
          break;
        case "loss":
          this.loss = checkLoss(value);
          break;
        case "alpha":
          this.alpha = checkAlpha(value);
          break;
        case "minImpurityDecrease":
          this.minImpurityDecrease = checkMinImpurityDecrease(value);
          break;
        case "maxLeafNodes":
          this.maxLeafNodes = checkMaxLeafNodes(value);
          break;
        case "ccpAlpha":
          this.ccpAlpha = checkCcpAlpha(value);
          break;
        case "randomState":
          this.randomState = checkRandomState(value);
          this.rng = undefined;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  /** Create an unfitted copy with the same hyperparameters. */
  clone(): GradientBoostingRegressor {
    return new GradientBoostingRegressor(
      definedOptions<GradientBoostingRegressorOptions>(this.getParams())
    );
  }
}

// ---------------------------------------------------------------------------
// Classifier
// ---------------------------------------------------------------------------

/** Options of {@link GradientBoostingClassifier}. */
export type GradientBoostingClassifierOptions = {
  readonly nEstimators?: number;
  readonly learningRate?: number;
  readonly maxDepth?: number;
  readonly minSamplesSplit?: number;
  readonly minSamplesLeaf?: number;
  readonly warmStart?: boolean;
  readonly subsample?: number;
  readonly maxFeatures?: "sqrt" | "log2" | number;
  readonly randomState?: number;
} & BoostingTreeOptions;

/**
 * Gradient Boosting Classifier.
 *
 * Uses gradient boosting with shallow regression trees for classification and the
 * binomial deviance (log loss). The leaf values are Newton steps (Friedman's TreeBoost),
 * computed on the rows each tree was trained on.
 *
 * - Two classes: one boosted model on the log-odds, started from the log of the class
 *   ratio.
 * - More classes: One-vs-Rest. One binary model is boosted per class, and probabilities are
 *   the per-class sigmoids normalized to sum to one. This differs from scikit-learn, which
 *   boosts all classes jointly with a multinomial deviance and a softmax.
 *
 * Class labels must be integers. With `subsample < 1` each stage trains on a random subset
 * drawn without replacement.
 *
 * @example
 * ```ts
 * import { GradientBoostingClassifier } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [2, 3], [3, 1], [4, 2]]);
 * const y = tensor([0, 0, 1, 1]);
 *
 * const gbc = new GradientBoostingClassifier({ nEstimators: 100, randomState: 0 });
 * gbc.fit(X, y);
 * const predictions = gbc.predict(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-ensemble | Deepbox Ensemble Methods}
 */
export class GradientBoostingClassifier implements Classifier {
  private nEstimators: number;
  private learningRate: number;
  private maxDepth: number;
  private minSamplesSplit: number;
  private minSamplesLeaf: number;
  private warmStart: boolean;
  private subsample: number;
  private maxFeatures: "sqrt" | "log2" | number | undefined;
  private randomState: number | undefined;
  private minImpurityDecrease: number;
  private maxLeafNodes: number | undefined;
  private ccpAlpha: number;

  /** Per-class arrays of weak learners (OvR for multiclass, a single array for binary) */
  private estimatorsPerClass: DecisionTreeRegressor[][] = [];
  private flatPerClass: FlatTree[][] = [];
  private initPredictions: number[] = [];
  private nFeatures = 0;
  private classLabels: number[] = [];
  private fitted = false;
  private rng: (() => number) | undefined;

  /**
   * @param options.nEstimators - Boosting stages per binary model (default: 100)
   * @param options.learningRate - Shrinks each tree's contribution, must be > 0 (default: 0.1)
   * @param options.maxDepth - Maximum depth of each tree (default: 3)
   * @param options.minSamplesSplit - Minimum rows needed to split a node, >= 2 (default: 2)
   * @param options.minSamplesLeaf - Minimum rows in a leaf, >= 1 (default: 1)
   * @param options.warmStart - Keep the trees of the previous fit and add more (default: false)
   * @param options.subsample - Fraction of rows used per stage, in (0, 1] (default: 1.0)
   * @param options.maxFeatures - Features tried per split: `"sqrt"`, `"log2"` or an integer count
   * @param options.randomState - Seed for subsampling and feature sampling. Without it the global
   *   Deepbox generator is used, so `setSeed` also makes the fit reproducible.
   * @param options.minImpurityDecrease - Minimum decrease of the weighted squared error of the
   *   residuals for a split, not divided by the total weight (scikit-learn's `friedman_mse`
   *   improvement) (default: 0)
   * @param options.maxLeafNodes - Grow every tree best-first up to this many leaves (default: no limit)
   * @param options.ccpAlpha - Cost-complexity pruning strength of every tree (default: 0)
   * @throws {InvalidParameterError} If an option is out of range
   */
  constructor(options: GradientBoostingClassifierOptions = {}) {
    this.nEstimators = checkNEstimators(options.nEstimators ?? 100);
    this.learningRate = checkLearningRate(options.learningRate ?? 0.1);
    this.maxDepth = checkMaxDepth(options.maxDepth ?? 3);
    this.minSamplesSplit = checkMinSamplesSplit(options.minSamplesSplit ?? 2);
    this.minSamplesLeaf = checkMinSamplesLeaf(options.minSamplesLeaf ?? 1);
    this.warmStart = checkWarmStart(options.warmStart ?? false);
    this.subsample = checkSubsample(options.subsample ?? 1.0);
    this.maxFeatures = checkMaxFeatures(options.maxFeatures);
    this.randomState = checkRandomState(options.randomState);
    this.minImpurityDecrease = checkMinImpurityDecrease(options.minImpurityDecrease ?? 0);
    this.maxLeafNodes = checkMaxLeafNodes(options.maxLeafNodes);
    this.ccpAlpha = checkCcpAlpha(options.ccpAlpha ?? 0);
  }

  /**
   * Fit (or extend) one binary boosted model on the 0/1 targets `yBinary`.
   */
  private fitBinary(
    xv: Float64Array,
    yBinary: Float64Array,
    weights: Float64Array | undefined,
    n: number,
    d: number,
    rng: () => number,
    existing: { estimators: DecisionTreeRegressor[]; flats: FlatTree[]; init: number } | undefined
  ): { estimators: DecisionTreeRegressor[]; flats: FlatTree[]; init: number } {
    const lr = this.learningRate;
    const scratch = new Int32Array(n);
    let init: number;
    let estimators: DecisionTreeRegressor[];
    let flats: FlatTree[];
    if (existing) {
      init = existing.init;
      estimators = [...existing.estimators];
      flats = [...existing.flats];
    } else {
      let pos = 0;
      let total = 0;
      for (let i = 0; i < n; i++) {
        const w = weights === undefined ? 1 : weights[i]!;
        pos += w * yBinary[i]!;
        total += w;
      }
      if (!(pos > 0) || !(total - pos > 0)) {
        throw new DataValidationError(
          "sampleWeight gives one of the classes a total weight of zero"
        );
      }
      // Log-odds of the (weighted) class prior.
      init = Math.log(pos / (total - pos));
      estimators = [];
      flats = [];
    }

    const raw = new Float64Array(n).fill(init);
    for (const flat of flats) addTree(flat, xv, n, d, raw, lr, scratch);

    const drawSize = this.subsample < 1 ? Math.max(1, Math.floor(this.subsample * n)) : n;
    const bagged = drawSize < n;
    const treeMaxFeatures = resolveMaxFeatures(this.maxFeatures, d);
    const treeBase: TreeBase = {
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
      minSamplesLeaf: this.minSamplesLeaf,
      minImpurityDecrease: this.minImpurityDecrease,
      maxLeafNodes: this.maxLeafNodes,
      ccpAlpha: this.ccpAlpha,
    };
    const residual = new Float64Array(n);
    const hessian = new Float64Array(n);
    const leafOfRow = new Int32Array(n);

    for (let m = estimators.length; m < this.nEstimators; m++) {
      for (let i = 0; i < n; i++) {
        const p = sigmoid(raw[i]!);
        residual[i] = yBinary[i]! - p;
        // p * (1 - p) with 1 - p taken as sigmoid(-raw), which does not round to 0 near p = 1.
        hessian[i] = p * sigmoid(-raw[i]!);
      }

      const bagRows = bagged ? sampleWithoutReplacement(rng, n, drawSize) : null;
      const fitX = bagRows === null ? xv : gatherRows(xv, d, bagRows);
      const fitR = bagRows === null ? residual : gatherValues(residual, bagRows);
      const nFit = bagRows === null ? n : bagRows.length;
      const fitW =
        weights === undefined
          ? undefined
          : bagRows === null
            ? weights
            : gatherValues(weights, bagRows);

      const tree = newTree(treeBase, treeMaxFeatures, rng, totalOf(fitW, nFit));
      tree.fit(
        floatMatrix(fitX, nFit, d),
        tensor(fitR, { dtype: "float64" }),
        fitW === undefined ? undefined : tensor(fitW, { dtype: "float64" })
      );
      const root = tree.tree_;
      if (!root) throw new DeepboxError("Internal error: fitted tree has no root");
      const flat = flattenTree(root);

      // TreeBoost Newton step (Friedman 2001): each leaf value becomes
      // sum(w * residual) / sum(w * p * (1 - p)) over the rows the tree was trained on.
      applyTree(flat, xv, n, d, leafOfRow);
      const nLeaves = flat.leafValue.length;
      const { order, start } = groupByLeaf(leafOfRow, bagRows, n, nLeaves);
      const values = Float64Array.from(flat.leafValue);
      for (let l = 0; l < nLeaves; l++) {
        const from = start[l]!;
        const to = start[l + 1]!;
        if (to === from) continue;
        let num = 0;
        let den = 0;
        for (let k = from; k < to; k++) {
          const row = order[k]!;
          const w = weights === undefined ? 1 : weights[row]!;
          num += w * residual[row]!;
          den += w * hessian[row]!;
        }
        values[l] = Math.abs(den) < 1e-150 ? 0 : num / den;
      }
      setLeafValues(tree, flat, values);

      for (let i = 0; i < n; i++) raw[i] = raw[i]! + lr * flat.leafValue[leafOfRow[i]!]!;
      estimators.push(tree);
      flats.push(flat);
    }

    return { estimators, flats, init };
  }

  /**
   * Fit the gradient boosting classifier.
   *
   * Without `warmStart` any previous ensemble is discarded. With `warmStart`, the fitted
   * trees are kept and stages are added until `nEstimators` is reached; X must then have the
   * same number of features and y the same set of classes as in the previous fit.
   *
   * With `sampleWeight` the initial log-odds, the tree splits and the Newton leaf values use
   * the weights.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Integer class labels of shape (n_samples,), with at least two classes
   * @param sampleWeight - Optional non-negative weights of shape (n_samples,)
   * @returns this
   * @throws {ShapeError} If X is not 2D, y is not 1D, the sample counts differ, or a warm
   *   start sees a different number of features
   * @throws {ShapeError} If `sampleWeight` is not 1D with one entry per sample
   * @throws {DataValidationError} If X or y contain NaN/Inf, y has non-integer labels, the
   *   weights are negative, not finite or give a class no weight, or a warm start sees a
   *   different set of classes
   * @throws {InvalidParameterError} If y contains a single class
   */
  // biome-ignore lint/suspicious/noConfusingVoidType: `void` keeps this compatible with Estimator.fit(X, y, params)
  fit(X: Tensor, y: Tensor, sampleWeightArg?: Tensor | void): this {
    const warmRequested = this.warmStart && this.fitted && this.estimatorsPerClass.length > 0;
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const sampleWeight = readSampleWeight(sampleWeightArg as Tensor | undefined, n);
    const xv = toFloat64View(X);
    const yv = toFloat64View(y);
    assertIntegerLabels(yv, "GradientBoostingClassifier");

    const classLabels = [...new Set(yv)].sort((a, b) => a - b);
    if (classLabels.length < 2) {
      throw new InvalidParameterError(
        "GradientBoostingClassifier requires at least 2 classes",
        "y",
        classLabels.length
      );
    }
    if (warmRequested) {
      if (d !== this.nFeatures) {
        throw new ShapeError(
          `X has ${d} features but the previous fit used ${this.nFeatures}; a warm start needs the same features`
        );
      }
      if (
        classLabels.length !== this.classLabels.length ||
        classLabels.some((label, i) => label !== this.classLabels[i])
      ) {
        throw new DataValidationError(
          "A warm start needs the same set of classes as the previous fit"
        );
      }
    }

    if (!warmRequested || this.rng === undefined) this.rng = createTreeRng(this.randomState);
    const rng = this.rng;
    const oldEstimators = warmRequested ? this.estimatorsPerClass : [];
    const oldFlats = warmRequested ? this.flatPerClass : [];
    const oldInit = warmRequested ? this.initPredictions : [];

    const estimatorsPerClass: DecisionTreeRegressor[][] = [];
    const flatPerClass: FlatTree[][] = [];
    const initPredictions: number[] = [];
    const nModels = classLabels.length === 2 ? 1 : classLabels.length;
    for (let c = 0; c < nModels; c++) {
      // Binary: class 1 is the larger label. Multiclass: class c against the rest.
      const positive = classLabels.length === 2 ? classLabels[1]! : classLabels[c]!;
      const yBinary = new Float64Array(n);
      for (let i = 0; i < n; i++) yBinary[i] = yv[i] === positive ? 1 : 0;
      const prior =
        warmRequested && oldEstimators[c] !== undefined
          ? { estimators: oldEstimators[c]!, flats: oldFlats[c]!, init: oldInit[c]! }
          : undefined;
      const model = this.fitBinary(xv, yBinary, sampleWeight, n, d, rng, prior);
      estimatorsPerClass.push(model.estimators);
      flatPerClass.push(model.flats);
      initPredictions.push(model.init);
    }

    this.estimatorsPerClass = estimatorsPerClass;
    this.flatPerClass = flatPerClass;
    this.initPredictions = initPredictions;
    this.classLabels = classLabels;
    this.nFeatures = d;
    this.fitted = true;
    return this;
  }

  private checkFitted(): void {
    if (!this.fitted) {
      throw new NotFittedError("GradientBoostingClassifier must be fitted before prediction");
    }
  }

  /** Raw scores of every binary model, flattened row-major to (n, nModels). */
  private rawScores(X: Tensor): { scores: Float64Array; nModels: number } {
    const n = X.shape[0] ?? 0;
    const xv = toFloat64View(X);
    const nModels = this.flatPerClass.length;
    const scores = new Float64Array(n * nModels);
    const column = new Float64Array(n);
    const scratch = new Int32Array(n);
    for (let c = 0; c < nModels; c++) {
      column.fill(this.initPredictions[c] as number);
      for (const flat of this.flatPerClass[c] as FlatTree[]) {
        addTree(flat, xv, n, this.nFeatures, column, this.learningRate, scratch);
      }
      for (let i = 0; i < n; i++) scores[i * nModels + c] = column[i]!;
    }
    return { scores, nModels };
  }

  /**
   * Raw (pre-sigmoid) scores of the boosted models.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Shape (n_samples,) with the log-odds of `classes[1]` for two classes, otherwise
   *   (n_samples, n_classes) with one One-vs-Rest score per class
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong rank or feature count
   */
  decisionFunction(X: Tensor): Tensor {
    this.checkFitted();
    validatePredictInputs(X, this.nFeatures, "GradientBoostingClassifier");
    const { scores, nModels } = this.rawScores(X);
    return nModels === 1 ? tensor(scores) : floatMatrix(scores, X.shape[0] ?? 0, nModels);
  }

  /**
   * Predict class labels. Two classes: `classes[1]` when the log-odds are positive.
   * More classes: the class with the largest One-vs-Rest score (ties go to the smallest label).
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Int32 labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong rank or feature count
   * @throws {DataValidationError} If X contains NaN/Inf
   */
  predict(X: Tensor): Tensor {
    this.checkFitted();
    validatePredictInputs(X, this.nFeatures, "GradientBoostingClassifier");
    const n = X.shape[0] ?? 0;
    const { scores, nModels } = this.rawScores(X);
    const out = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      if (nModels === 1) {
        out[i] = this.classLabels[scores[i]! > 0 ? 1 : 0]!;
      } else {
        let best = 0;
        let bestScore = -Infinity;
        for (let c = 0; c < nModels; c++) {
          const s = scores[i * nModels + c]!;
          if (s > bestScore) {
            bestScore = s;
            best = c;
          }
        }
        out[i] = this.classLabels[best]!;
      }
    }
    return tensor(out, { dtype: "int32" });
  }

  /**
   * Predict class probabilities. Two classes: the sigmoid of the log-odds. More classes: the
   * per-class sigmoids divided by their sum.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Probabilities of shape (n_samples, n_classes), columns ordered like `classes`
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong rank or feature count
   * @throws {DataValidationError} If X contains NaN/Inf
   */
  predictProba(X: Tensor): Tensor {
    this.checkFitted();
    validatePredictInputs(X, this.nFeatures, "GradientBoostingClassifier");
    const n = X.shape[0] ?? 0;
    const nClasses = this.classLabels.length;
    const { scores, nModels } = this.rawScores(X);
    const out = new Float64Array(n * nClasses);
    for (let i = 0; i < n; i++) {
      if (nModels === 1) {
        const z = scores[i]!;
        out[2 * i] = sigmoid(-z);
        out[2 * i + 1] = sigmoid(z);
      } else {
        let total = 0;
        for (let c = 0; c < nClasses; c++) {
          const p = sigmoid(scores[i * nClasses + c]!);
          out[i * nClasses + c] = p;
          total += p;
        }
        // Every sigmoid underflowing at once means no model claims the row: split evenly.
        for (let c = 0; c < nClasses; c++) {
          out[i * nClasses + c] = total > 0 ? out[i * nClasses + c]! / total : 1 / nClasses;
        }
      }
    }
    return floatMatrix(out, n, nClasses);
  }

  /**
   * Mean accuracy on the given data.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True labels of shape (n_samples,)
   * @returns Accuracy in [0, 1]
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1D or its length differs from the number of rows in X
   * @throws {DataValidationError} If y is empty or contains NaN/Inf
   */
  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    const predictions = this.predict(X);
    const yv = checkScoreTarget(y, predictions.size);
    const pv = toFloat64View(predictions);
    let correct = 0;
    for (let i = 0; i < yv.length; i++) {
      if (pv[i] === yv[i]) correct++;
    }
    return correct / yv.length;
  }

  /** Sorted class labels seen during fit, or `undefined` before fitting. */
  get classes(): Tensor | undefined {
    if (!this.fitted || this.classLabels.length === 0) return undefined;
    return tensor(this.classLabels, { dtype: "int32" });
  }

  /** Number of boosting stages per binary model after the last fit. */
  get nEstimatorsFitted(): number {
    return this.estimatorsPerClass[0]?.length ?? 0;
  }

  /**
   * Feature importances averaged across all boosting stages and binary models.
   *
   * @returns Tensor of shape (n_features,) that sums to 1 (all zeros if no tree split)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get featureImportances(): Tensor {
    if (!this.fitted || this.estimatorsPerClass.length === 0 || this.nFeatures === 0) {
      throw new NotFittedError(
        "GradientBoostingClassifier must be fitted to access feature_importances_"
      );
    }
    return averageTreeImportances(this.estimatorsPerClass.flat(), this.nFeatures);
  }

  getParams(): Record<string, unknown> {
    return {
      nEstimators: this.nEstimators,
      learningRate: this.learningRate,
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
      minSamplesLeaf: this.minSamplesLeaf,
      warmStart: this.warmStart,
      subsample: this.subsample,
      maxFeatures: this.maxFeatures,
      randomState: this.randomState,
      minImpurityDecrease: this.minImpurityDecrease,
      maxLeafNodes: this.maxLeafNodes,
      ccpAlpha: this.ccpAlpha,
    };
  }

  /**
   * Set hyperparameters. Changing `randomState` restarts the random stream at the next
   * `fit`, so a later warm start no longer continues the earlier stream.
   *
   * @param params - Parameters to set
   * @throws {InvalidParameterError} If a parameter value is invalid or unknown
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nEstimators":
          this.nEstimators = checkNEstimators(value);
          break;
        case "learningRate":
          this.learningRate = checkLearningRate(value);
          break;
        case "maxDepth":
          this.maxDepth = checkMaxDepth(value);
          break;
        case "minSamplesSplit":
          this.minSamplesSplit = checkMinSamplesSplit(value);
          break;
        case "minSamplesLeaf":
          this.minSamplesLeaf = checkMinSamplesLeaf(value);
          break;
        case "warmStart":
          this.warmStart = checkWarmStart(value);
          break;
        case "subsample":
          this.subsample = checkSubsample(value);
          break;
        case "maxFeatures":
          this.maxFeatures = checkMaxFeatures(value);
          break;
        case "minImpurityDecrease":
          this.minImpurityDecrease = checkMinImpurityDecrease(value);
          break;
        case "maxLeafNodes":
          this.maxLeafNodes = checkMaxLeafNodes(value);
          break;
        case "ccpAlpha":
          this.ccpAlpha = checkCcpAlpha(value);
          break;
        case "randomState":
          this.randomState = checkRandomState(value);
          this.rng = undefined;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  /** Create an unfitted copy with the same hyperparameters. */
  clone(): GradientBoostingClassifier {
    return new GradientBoostingClassifier(
      definedOptions<GradientBoostingClassifierOptions>(this.getParams())
    );
  }
}
