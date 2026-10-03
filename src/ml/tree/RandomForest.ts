import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __randomBelow, __SeededRandom } from "../../random/random";
import { toFloat64View, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier, Regressor } from "../base";
import {
  checkCcpAlpha,
  checkClassWeight,
  checkMaxLeafNodes,
  checkMinImpurityDecrease,
  classWeightPerClass,
  combineWeights,
  type ForestClassWeight,
  readSampleWeight,
  type TreeGrowthOptions,
} from "./_growth";
import {
  type ClassificationCriterion,
  createTreeRng,
  DecisionTreeClassifier,
  DecisionTreeRegressor,
} from "./DecisionTree";

// ---------------------------------------------------------------------------
// Options and parameter handling shared by both forests
// ---------------------------------------------------------------------------

/**
 * Number of features examined at every split of a forest tree.
 *
 * - an integer >= 1: that many features (capped at the number of features)
 * - a number in (0, 1): that fraction of the features, at least one
 * - `"sqrt"`: `floor(sqrt(nFeatures))`
 * - `"log2"`: `floor(log2(nFeatures))`
 * - `null`: every feature (scikit-learn's `None`), whatever the class default is
 *
 * Results are never below one feature.
 */
export type ForestMaxFeatures = number | "sqrt" | "log2" | null;

/** Options shared by {@link RandomForestClassifier} and {@link RandomForestRegressor}. */
export type RandomForestOptions = {
  /** Number of trees. Default 100. */
  readonly nEstimators?: number;
  /** Maximum depth of every tree; `Infinity` grows trees until the leaves are pure. Default 10. */
  readonly maxDepth?: number;
  /** Minimum number of samples a node needs to be split. Default 2. */
  readonly minSamplesSplit?: number;
  /** Minimum number of samples every leaf must keep. Default 1. */
  readonly minSamplesLeaf?: number;
  /** Features examined per split, see {@link ForestMaxFeatures}. */
  readonly maxFeatures?: ForestMaxFeatures;
  /** Draw each tree's training set with replacement. Default true. */
  readonly bootstrap?: boolean;
  /**
   * Keep the trees of a previous `fit` and only add the missing ones when `nEstimators`
   * was increased. Default false.
   */
  readonly warmStart?: boolean;
  /**
   * Seed. The bootstrap samples and the per-split feature subsets of every tree derive from
   * it, so the same seed always gives the same forest. Without it the global Deepbox random
   * generator is used.
   */
  readonly randomState?: number;
  /**
   * Size of each bootstrap sample (requires `bootstrap`): an integer >= 1 is a sample count
   * (capped at the number of samples), a number in (0, 1) a fraction of the samples.
   * Default: as many samples as the training set has.
   */
  readonly maxSamples?: number;
  /** Compute the out-of-bag score during `fit` (requires `bootstrap`). Default false. */
  readonly oobScore?: boolean;
} & TreeGrowthOptions;

/** Options of {@link RandomForestClassifier}. */
export type RandomForestClassifierOptions = RandomForestOptions & {
  /** Split quality measure of the trees. Default `"gini"`. */
  readonly criterion?: ClassificationCriterion;
  /**
   * Class weights: `"balanced"`, `"balanced_subsample"` (the balanced weights recomputed on the
   * bootstrap sample of every tree) or a map from class label to weight. They multiply the
   * `sampleWeight` given to `fit`. Default: all classes weigh 1.
   */
  readonly classWeight?: ForestClassWeight;
};

/** Options of {@link RandomForestRegressor}. */
export type RandomForestRegressorOptions = RandomForestOptions;

type ForestParams = {
  nEstimators: number;
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
  maxFeatures: ForestMaxFeatures;
  bootstrap: boolean;
  warmStart: boolean;
  randomState: number | undefined;
  maxSamples: number | undefined;
  oobScore: boolean;
  minImpurityDecrease: number;
  maxLeafNodes: number | undefined;
  ccpAlpha: number;
};

function defaultParams(maxFeatures: ForestMaxFeatures): ForestParams {
  return {
    nEstimators: 100,
    maxDepth: 10,
    minSamplesSplit: 2,
    minSamplesLeaf: 1,
    maxFeatures,
    bootstrap: true,
    warmStart: false,
    randomState: undefined,
    maxSamples: undefined,
    oobScore: false,
    minImpurityDecrease: 0,
    maxLeafNodes: undefined,
    ccpAlpha: 0,
  };
}

function checkInteger(name: string, value: unknown, min: number): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < min) {
    throw new InvalidParameterError(
      `${name} must be an integer >= ${min}; received ${String(value)}`,
      name,
      value
    );
  }
  return value;
}

function checkBoolean(name: string, value: unknown): boolean {
  if (typeof value !== "boolean") {
    throw new InvalidParameterError(
      `${name} must be a boolean; received ${String(value)}`,
      name,
      value
    );
  }
  return value;
}

function checkMaxFeatures(value: unknown): ForestMaxFeatures {
  if (value === "sqrt" || value === "log2" || value === null) return value;
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    !((Number.isInteger(value) && value >= 1) || (value > 0 && value < 1))
  ) {
    throw new InvalidParameterError(
      `maxFeatures must be "sqrt", "log2", null, an integer >= 1, or a fraction in (0, 1); received ${String(value)}`,
      "maxFeatures",
      value
    );
  }
  return value;
}

function checkMaxSamples(value: unknown): number | undefined {
  if (value === undefined) return undefined;
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    !((Number.isInteger(value) && value >= 1) || (value > 0 && value < 1))
  ) {
    throw new InvalidParameterError(
      `maxSamples must be an integer >= 1 or a fraction in (0, 1); received ${String(value)}`,
      "maxSamples",
      value
    );
  }
  return value;
}

function checkRandomState(value: unknown): number | undefined {
  if (value !== undefined && (typeof value !== "number" || !Number.isFinite(value))) {
    throw new InvalidParameterError(
      `randomState must be a finite number; received ${String(value)}`,
      "randomState",
      value
    );
  }
  return value;
}

/**
 * Validate one named parameter and store it in `target`.
 * Returns false when `key` is not a forest parameter.
 */
function assignParam(target: ForestParams, key: string, value: unknown): boolean {
  switch (key) {
    case "nEstimators":
      target.nEstimators = checkInteger("nEstimators", value, 1);
      return true;
    case "maxDepth":
      target.maxDepth =
        value === Number.POSITIVE_INFINITY ? value : checkInteger("maxDepth", value, 1);
      return true;
    case "minSamplesSplit":
      target.minSamplesSplit = checkInteger("minSamplesSplit", value, 2);
      return true;
    case "minSamplesLeaf":
      target.minSamplesLeaf = checkInteger("minSamplesLeaf", value, 1);
      return true;
    case "maxFeatures":
      target.maxFeatures = checkMaxFeatures(value);
      return true;
    case "bootstrap":
      target.bootstrap = checkBoolean("bootstrap", value);
      return true;
    case "warmStart":
      target.warmStart = checkBoolean("warmStart", value);
      return true;
    case "randomState":
      target.randomState = checkRandomState(value);
      return true;
    case "maxSamples":
      target.maxSamples = checkMaxSamples(value);
      return true;
    case "oobScore":
      target.oobScore = checkBoolean("oobScore", value);
      return true;
    case "minImpurityDecrease":
      target.minImpurityDecrease = checkMinImpurityDecrease(value);
      return true;
    case "maxLeafNodes":
      target.maxLeafNodes = checkMaxLeafNodes(value);
      return true;
    case "ccpAlpha":
      target.ccpAlpha = checkCcpAlpha(value);
      return true;
    default:
      return false;
  }
}

function paramsFromOptions(
  options: RandomForestOptions,
  maxFeatures: ForestMaxFeatures
): ForestParams {
  const params = defaultParams(maxFeatures);
  for (const [key, value] of Object.entries(options)) {
    if (value === undefined) continue;
    // Keys that are not forest parameters (such as `criterion`) are handled by the caller.
    assignParam(params, key, value);
  }
  assertConsistent(params);
  return params;
}

/** Parameter combinations that are only checkable once all parameters are known. */
function assertConsistent(params: ForestParams): void {
  if (params.oobScore && !params.bootstrap) {
    throw new InvalidParameterError("oobScore requires bootstrap=true", "oobScore", true);
  }
  if (params.maxSamples !== undefined && !params.bootstrap) {
    throw new InvalidParameterError(
      "maxSamples requires bootstrap=true",
      "maxSamples",
      params.maxSamples
    );
  }
}

function publicParams(params: ForestParams): Record<string, unknown> {
  return {
    nEstimators: params.nEstimators,
    maxDepth: params.maxDepth,
    minSamplesSplit: params.minSamplesSplit,
    minSamplesLeaf: params.minSamplesLeaf,
    maxFeatures: params.maxFeatures,
    bootstrap: params.bootstrap,
    warmStart: params.warmStart,
    randomState: params.randomState,
    maxSamples: params.maxSamples,
    oobScore: params.oobScore,
    minImpurityDecrease: params.minImpurityDecrease,
    maxLeafNodes: params.maxLeafNodes,
    ccpAlpha: params.ccpAlpha,
  };
}

/** Options object for a constructor: parameters that are `undefined` are left out. */
function definedOptions(params: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) out[key] = value;
  }
  return out;
}

/**
 * Candidate features per split. With `oneMeansAll` an integer 1 (indistinguishable from the
 * float 1.0 in JavaScript) selects every feature, which is the regressor's default.
 */
function resolveMaxFeatures(
  maxFeatures: ForestMaxFeatures,
  nFeatures: number,
  oneMeansAll: boolean
): number {
  if (maxFeatures === null) return nFeatures;
  if (maxFeatures === "sqrt") return Math.max(1, Math.floor(Math.sqrt(nFeatures)));
  if (maxFeatures === "log2") return Math.max(1, Math.floor(Math.log2(nFeatures)));
  if (oneMeansAll && maxFeatures === 1) return nFeatures;
  if (Number.isInteger(maxFeatures)) return Math.min(maxFeatures, nFeatures);
  return Math.max(1, Math.floor(maxFeatures * nFeatures));
}

/** Round half to even, like Python's `round`, so sample counts match scikit-learn. */
function roundHalfEven(value: number): number {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

function resolveDrawSize(maxSamples: number | undefined, nSamples: number): number {
  if (maxSamples === undefined) return nSamples;
  if (maxSamples < 1) return Math.max(1, roundHalfEven(maxSamples * nSamples));
  return Math.min(maxSamples, nSamples);
}

/**
 * Two seeds per tree (bootstrap stream, tree stream) for the trees `start..nEstimators - 1`.
 * With a `randomState` the seeds of tree `t` depend only on the seed and `t`, so a warm-started
 * forest equals the one grown in a single `fit`.
 */
function drawTreeSeeds(
  randomState: number | undefined,
  nEstimators: number,
  start: number
): number[] {
  const next = createTreeRng(randomState);
  if (randomState !== undefined) {
    for (let i = 0; i < 2 * start; i++) next();
  }
  const seeds: number[] = [];
  for (let t = start; t < nEstimators; t++) {
    seeds.push(Math.floor(next() * 4294967296), Math.floor(next() * 4294967296));
  }
  return seeds;
}

/**
 * Indices of one bootstrap sample drawn with replacement, plus the in-bag mask used for the
 * out-of-bag estimate.
 */
function drawBootstrap(
  seed: number,
  drawSize: number,
  nSamples: number
): { indices: Int32Array; inBag: Uint8Array } {
  const generator = new __SeededRandom(BigInt(seed));
  const next = (): number => generator.next();
  const indices = new Int32Array(drawSize);
  const inBag = new Uint8Array(nSamples);
  for (let i = 0; i < drawSize; i++) {
    const idx = Math.min(nSamples - 1, __randomBelow(next, nSamples));
    indices[i] = idx;
    inBag[idx] = 1;
  }
  return { indices, inBag };
}

/** Copy the rows `indices` of the row-major `(n, d)` array `x`. */
function gatherRows(x: Float64Array, d: number, indices: Int32Array): Float64Array {
  const out = new Float64Array(indices.length * d);
  for (let k = 0; k < indices.length; k++) {
    const row = (indices[k] as number) * d;
    out.set(x.subarray(row, row + d), k * d);
  }
  return out;
}

/** Weights of the rows `indices`, as a tensor; `undefined` when there are no weights. */
function gatherWeights(
  weights: Float64Array | undefined,
  indices: Int32Array
): Float64Array | undefined {
  if (weights === undefined) return undefined;
  const out = new Float64Array(indices.length);
  for (let k = 0; k < indices.length; k++) out[k] = weights[indices[k] as number] as number;
  return out;
}

/** Whether any weight is positive (always true without weights). */
function hasPositiveWeight(weights: Float64Array | undefined): boolean {
  if (weights === undefined) return true;
  for (let i = 0; i < weights.length; i++) if ((weights[i] as number) > 0) return true;
  return false;
}

/** Out-of-bag estimates need the bootstrap sample of every tree of a warm-started forest. */
function assertBagsAvailable(trees: number, bags: number): void {
  if (bags !== trees) {
    throw new InvalidParameterError(
      "oobScore needs every tree to have a bootstrap sample, but the existing trees were grown with bootstrap=false",
      "oobScore",
      true
    );
  }
}

/** Row-major `(n, d)` float64 tensor over `data`. */
function matrixTensor(data: Float64Array, n: number, d: number): Tensor {
  return tensor(data, { dtype: "float64" }).reshape([n, d]);
}

/** Rows of `x` that are not in `inBag`, with their original positions. */
function outOfBagRows(
  x: Float64Array,
  d: number,
  inBag: Uint8Array
): { rows: Float64Array; positions: Int32Array } {
  let count = 0;
  for (let i = 0; i < inBag.length; i++) if (inBag[i] === 0) count++;
  const rows = new Float64Array(count * d);
  const positions = new Int32Array(count);
  let k = 0;
  for (let i = 0; i < inBag.length; i++) {
    if (inBag[i] !== 0) continue;
    rows.set(x.subarray(i * d, (i + 1) * d), k * d);
    positions[k++] = i;
  }
  return { rows, positions };
}

/** `int32` when every label is an integer in range, `float64` otherwise. */
function labelDType(labels: readonly number[]): "int32" | "float64" {
  for (const v of labels) {
    if (!Number.isInteger(v) || v < -2147483648 || v > 2147483647) return "float64";
  }
  return "int32";
}

function labelTensor(values: ArrayLike<number>, dtype: "int32" | "float64"): Tensor {
  return dtype === "int32"
    ? tensor(Int32Array.from(values), { dtype: "int32" })
    : tensor(Float64Array.from(values), { dtype: "float64" });
}

/** Read and check the target of a `score` call. */
function readScoreTargets(y: Tensor): Float64Array {
  if (y.ndim !== 1) {
    throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
  }
  const values = toFloat64View(y);
  if (values.length === 0) {
    throw new DataValidationError("y must contain at least one sample");
  }
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) {
      throw new DataValidationError("y contains non-finite values (NaN or Inf)");
    }
  }
  return values;
}

/** Coefficient of determination; 1 for a perfect fit of constant targets, else 0 for them. */
function r2(yTrue: ArrayLike<number>, yPred: ArrayLike<number>, indices?: Int32Array): number {
  const n = indices ? indices.length : yTrue.length;
  const at = (i: number): number => (indices ? (indices[i] as number) : i);
  let mean = 0;
  for (let i = 0; i < n; i++) mean += yTrue[at(i)] as number;
  mean /= n;
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    const t = yTrue[at(i)] as number;
    const d = t - (yPred[at(i)] as number);
    ssRes += d * d;
    ssTot += (t - mean) * (t - mean);
  }
  if (ssTot === 0) return ssRes === 0 ? 1 : 0;
  return 1 - ssRes / ssTot;
}

/**
 * Mean of the (individually normalized) importances of all trees, renormalized to sum to 1
 * (all zeros when no tree ever split).
 */
function averageImportances(
  trees: readonly { readonly featureImportances: Tensor }[],
  nFeatures: number
): Tensor {
  const avg = new Float64Array(nFeatures);
  for (const tree of trees) {
    const imp = toFloat64View(tree.featureImportances);
    for (let j = 0; j < nFeatures; j++) avg[j] = (avg[j] as number) + (imp[j] as number);
  }
  let total = 0;
  for (let j = 0; j < nFeatures; j++) total += avg[j] as number;
  if (total > 0) {
    for (let j = 0; j < nFeatures; j++) avg[j] = (avg[j] as number) / total;
  }
  return tensor(avg, { dtype: "float64" });
}

// ---------------------------------------------------------------------------
// Classifier
// ---------------------------------------------------------------------------

/**
 * Random Forest Classifier.
 *
 * An ensemble of decision trees trained on bootstrap samples with a random subset of the
 * features examined at every split. Class probabilities are the average of the trees'
 * probabilities and `predict` returns the class with the highest average probability
 * (the smallest label on ties), as scikit-learn does.
 *
 * **Algorithm**:
 * 1. Draw `nEstimators` bootstrap samples from the training data
 * 2. Grow a decision tree on each sample, examining `maxFeatures` random features per split
 * 3. Average the trees' class probabilities
 *
 * Differences from scikit-learn defaults: `maxDepth` defaults to 10 (not unlimited); pass
 * `Infinity` to grow full trees.
 *
 * @example
 * ```ts
 * import { RandomForestClassifier } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X_train = tensor([[0, 0], [0, 1], [1, 0], [1, 1], [5, 5], [5, 6], [6, 5], [6, 6]]);
 * const y_train = tensor([0, 0, 0, 0, 1, 1, 1, 1]);
 * const clf = new RandomForestClassifier({ nEstimators: 50, randomState: 0 });
 * clf.fit(X_train, y_train);
 * const predictions = clf.predict(tensor([[0.5, 0.5], [5.5, 5.5]]));
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-tree | Deepbox Decision Trees}
 */
export class RandomForestClassifier implements Classifier {
  private params: ForestParams;
  private criterion: ClassificationCriterion;
  private classWeight: ForestClassWeight | undefined;

  private trees: DecisionTreeClassifier[] = [];
  private inBag: Uint8Array[] = [];
  private classLabels?: number[];
  private labelDType: "int32" | "float64" = "int32";
  private nFeatures?: number;
  private fitNSamples = 0;
  private fitted = false;
  private oobScore_?: number;
  private oobDecision_?: Float64Array;

  /**
   * @param options - Hyperparameters, see {@link RandomForestClassifierOptions}
   * @throws {InvalidParameterError} If an option is out of range or inconsistent
   */
  constructor(options: RandomForestClassifierOptions = {}) {
    this.params = paramsFromOptions(options, "sqrt");
    const criterion = options.criterion ?? "gini";
    if (criterion !== "gini" && criterion !== "entropy" && criterion !== "log_loss") {
      throw new InvalidParameterError(
        `criterion must be "gini", "entropy", or "log_loss"; received ${String(criterion)}`,
        "criterion",
        criterion
      );
    }
    this.criterion = criterion;
    this.classWeight = checkClassWeight(options.classWeight, true);
  }

  /**
   * Fit the random forest classifier on training data.
   *
   * Builds an ensemble of decision trees, each trained on a bootstrap sample with random
   * feature subsets. Calling `fit` again discards the previous forest unless `warmStart` is
   * set and `nEstimators` was increased, in which case only the missing trees are grown.
   * Labels may be any finite numbers (integer or fractional).
   *
   * Each tree is grown with the weights `sampleWeight[i] * classWeight[y[i]]` (for
   * `"balanced_subsample"`, with the balanced weights of its own bootstrap sample), see
   * {@link DecisionTreeClassifier.fit}. The out-of-bag score stays unweighted.
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
   * @throws {InvalidParameterError} If the parameters are inconsistent, if `classWeight` names a
   *   label that is not in y, or if a warm start would need fewer trees than the forest already
   *   has
   */
  // biome-ignore lint/suspicious/noConfusingVoidType: `void` keeps this compatible with Estimator.fit(X, y, params)
  fit(X: Tensor, y: Tensor, sampleWeightArg?: Tensor | void): this {
    validateFitInputs(X, y);
    const p = this.params;
    assertConsistent(p);

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const x = toFloat64View(X);
    const yv = toFloat64View(y);
    const sampleWeight = readSampleWeight(sampleWeightArg as Tensor | undefined, nSamples);

    const labels = [...new Set(yv)].sort((a, b) => a - b);
    const nClasses = labels.length;
    const codeOf = new Map<number, number>();
    for (let i = 0; i < nClasses; i++) codeOf.set(labels[i] as number, i);
    const yCode = new Int32Array(nSamples);
    for (let i = 0; i < nSamples; i++) yCode[i] = codeOf.get(yv[i] as number) as number;

    let trees: DecisionTreeClassifier[] = [];
    let inBag: Uint8Array[] = [];
    let start = 0;
    if (p.warmStart && this.fitted && this.trees.length > 0) {
      this.assertWarmStartCompatible(nSamples, nFeatures, labels);
      start = this.trees.length;
      if (start > p.nEstimators) {
        throw new InvalidParameterError(
          `nEstimators=${p.nEstimators} must be >= the ${start} trees already fitted when warmStart=true`,
          "nEstimators",
          p.nEstimators
        );
      }
      if (start === p.nEstimators) return this;
      trees = this.trees.slice();
      inBag = this.inBag.slice();
    }

    const perSplit = resolveMaxFeatures(p.maxFeatures, nFeatures, false);
    const drawSize = resolveDrawSize(p.maxSamples, nSamples);
    const seeds = drawTreeSeeds(p.randomState, p.nEstimators, start);
    const fullX = p.bootstrap ? undefined : matrixTensor(x, nSamples, nFeatures);
    const fullY = p.bootstrap ? undefined : tensor(yCode, { dtype: "int32" });

    // Weights of the whole training set. "balanced_subsample" is computed per bootstrap sample
    // below (without bootstrap it is the same as "balanced").
    const cw = this.classWeight;
    const perTreeBalance = cw === "balanced_subsample" && p.bootstrap;
    let baseWeights = sampleWeight;
    if (cw !== undefined && !perTreeBalance) {
      const classCounts = new Float64Array(nClasses);
      for (let i = 0; i < nSamples; i++) classCounts[yCode[i] as number]! += 1;
      baseWeights = combineWeights(
        sampleWeight,
        classWeightPerClass(cw, labels, classCounts, nSamples),
        yCode
      );
    }
    if (!hasPositiveWeight(baseWeights)) {
      throw new DataValidationError("every sample has a weight of zero");
    }
    const fullWeights =
      baseWeights === undefined || p.bootstrap
        ? undefined
        : tensor(baseWeights, { dtype: "float64" });

    for (let t = start; t < p.nEstimators; t++) {
      const bootSeed = seeds[2 * (t - start)] as number;
      const treeSeed = seeds[2 * (t - start) + 1] as number;
      const tree = new DecisionTreeClassifier({
        maxDepth: p.maxDepth,
        minSamplesSplit: p.minSamplesSplit,
        minSamplesLeaf: p.minSamplesLeaf,
        maxFeatures: perSplit,
        randomState: treeSeed,
        criterion: this.criterion,
        minImpurityDecrease: p.minImpurityDecrease,
        ...(p.maxLeafNodes === undefined ? {} : { maxLeafNodes: p.maxLeafNodes }),
        ccpAlpha: p.ccpAlpha,
      });
      if (p.bootstrap) {
        const bag = drawBootstrap(bootSeed, drawSize, nSamples);
        const ys = new Int32Array(bag.indices.length);
        for (let k = 0; k < ys.length; k++) ys[k] = yCode[bag.indices[k] as number] as number;
        let bagWeights = gatherWeights(baseWeights, bag.indices);
        if (perTreeBalance) {
          const classCounts = new Float64Array(nClasses);
          for (let k = 0; k < ys.length; k++) classCounts[ys[k] as number]! += 1;
          const perClass = classWeightPerClass("balanced", labels, classCounts, ys.length);
          bagWeights = combineWeights(bagWeights, perClass, ys);
        }
        const xs = matrixTensor(
          gatherRows(x, nFeatures, bag.indices),
          bag.indices.length,
          nFeatures
        );
        if (bagWeights !== undefined && !hasPositiveWeight(bagWeights)) {
          // A bootstrap sample of zero-weight rows only: grow this tree on the whole training set.
          tree.fit(
            matrixTensor(x, nSamples, nFeatures),
            tensor(yCode, { dtype: "int32" }),
            tensor(baseWeights as Float64Array, { dtype: "float64" })
          );
        } else {
          tree.fit(
            xs,
            tensor(ys, { dtype: "int32" }),
            bagWeights === undefined ? undefined : tensor(bagWeights, { dtype: "float64" })
          );
        }
        inBag.push(bag.inBag);
      } else {
        tree.fit(fullX as Tensor, fullY as Tensor, fullWeights);
      }
      trees.push(tree);
    }

    let oob: { score: number; decision: Float64Array } | undefined;
    if (p.oobScore) {
      oob = this.computeOob(trees, inBag, x, yCode, nSamples, nFeatures, nClasses);
    }

    this.trees = trees;
    this.inBag = inBag;
    this.classLabels = labels;
    this.labelDType = labelDType(labels);
    this.nFeatures = nFeatures;
    this.fitNSamples = nSamples;
    this.fitted = true;
    delete this.oobScore_;
    delete this.oobDecision_;
    if (oob) {
      this.oobScore_ = oob.score;
      this.oobDecision_ = oob.decision;
    }
    return this;
  }

  private assertWarmStartCompatible(
    nSamples: number,
    nFeatures: number,
    labels: readonly number[]
  ): void {
    if (nFeatures !== this.nFeatures) {
      throw new ShapeError(
        `X has ${nFeatures} features but the warm-started forest was fitted with ${this.nFeatures} features`
      );
    }
    const old = this.classLabels ?? [];
    if (labels.length !== old.length || labels.some((v, i) => v !== old[i])) {
      throw new DataValidationError(
        "y has different classes than the warm-started forest; set warmStart=false to refit from scratch"
      );
    }
    if (this.params.oobScore) assertBagsAvailable(this.trees.length, this.inBag.length);
    if (this.params.oobScore && nSamples !== this.fitNSamples) {
      throw new ShapeError(
        `X has ${nSamples} samples but the warm-started forest was fitted with ${this.fitNSamples}; the out-of-bag score needs the same training set`
      );
    }
  }

  /**
   * Out-of-bag estimate: every sample is scored by the average probabilities of the trees whose
   * bootstrap sample did not contain it. Samples that every tree saw are left out of the score
   * and have NaN rows in the decision matrix.
   */
  private computeOob(
    trees: readonly DecisionTreeClassifier[],
    inBag: readonly Uint8Array[],
    x: Float64Array,
    yCode: Int32Array,
    nSamples: number,
    nFeatures: number,
    nClasses: number
  ): { score: number; decision: Float64Array } {
    const decision = new Float64Array(nSamples * nClasses);
    const counts = new Int32Array(nSamples);
    for (let t = 0; t < trees.length; t++) {
      const tree = trees[t] as DecisionTreeClassifier;
      const { rows, positions } = outOfBagRows(x, nFeatures, inBag[t] as Uint8Array);
      const m = positions.length;
      if (m === 0) continue;
      const proba = toFloat64View(tree.predictProba(matrixTensor(rows, m, nFeatures)));
      const treeClasses = toFloat64View(tree.classes as Tensor);
      const k = treeClasses.length;
      for (let r = 0; r < m; r++) {
        const base = (positions[r] as number) * nClasses;
        for (let j = 0; j < k; j++) {
          const col = base + (treeClasses[j] as number);
          decision[col] = (decision[col] as number) + (proba[r * k + j] as number);
        }
        counts[positions[r] as number]! += 1;
      }
    }
    let correct = 0;
    let covered = 0;
    for (let i = 0; i < nSamples; i++) {
      const c = counts[i] as number;
      const base = i * nClasses;
      if (c === 0) {
        for (let j = 0; j < nClasses; j++) decision[base + j] = Number.NaN;
        continue;
      }
      let best = 0;
      for (let j = 0; j < nClasses; j++) {
        decision[base + j] = (decision[base + j] as number) / c;
        if ((decision[base + j] as number) > (decision[base + best] as number)) best = j;
      }
      covered++;
      if (best === yCode[i]) correct++;
    }
    return { score: covered === 0 ? Number.NaN : correct / covered, decision };
  }

  private requireFitted(action: string): {
    trees: DecisionTreeClassifier[];
    labels: number[];
    nFeatures: number;
  } {
    if (!this.fitted || !this.classLabels || this.nFeatures === undefined) {
      throw new NotFittedError(`RandomForestClassifier must be fitted before ${action}`);
    }
    return { trees: this.trees, labels: this.classLabels, nFeatures: this.nFeatures };
  }

  /** Average class probabilities over the trees: `(nSamples, nClasses)` row-major. */
  private averageProba(X: Tensor, action: string): { proba: Float64Array; nSamples: number } {
    const { trees, labels, nFeatures } = this.requireFitted(action);
    validatePredictInputs(X, nFeatures, "RandomForestClassifier");
    const nSamples = X.shape[0] ?? 0;
    const nClasses = labels.length;
    const proba = new Float64Array(nSamples * nClasses);
    if (nSamples === 0) return { proba, nSamples };

    for (const tree of trees) {
      const treeProba = toFloat64View(tree.predictProba(X));
      const treeClasses = toFloat64View(tree.classes as Tensor);
      const k = treeClasses.length;
      for (let j = 0; j < k; j++) {
        // The trees were fitted on class codes, so a tree's class value is the column index.
        const col = treeClasses[j] as number;
        for (let i = 0; i < nSamples; i++) {
          const idx = i * nClasses + col;
          proba[idx] = (proba[idx] as number) + (treeProba[i * k + j] as number);
        }
      }
    }
    for (let i = 0; i < proba.length; i++) proba[i] = (proba[i] as number) / trees.length;
    return { proba, nSamples };
  }

  /**
   * Predict class labels for samples in X.
   *
   * Returns the class with the highest probability averaged over the trees; ties go to the
   * smallest label.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted class labels of shape (n_samples,): `int32` when all classes are
   *   integers, `float64` otherwise
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    const { proba, nSamples } = this.averageProba(X, "prediction");
    const labels = this.classLabels as number[];
    const nClasses = labels.length;
    const out = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      let best = 0;
      for (let j = 1; j < nClasses; j++) {
        if ((proba[i * nClasses + j] as number) > (proba[i * nClasses + best] as number)) best = j;
      }
      out[i] = labels[best] as number;
    }
    return labelTensor(out, this.labelDType);
  }

  /**
   * Predict class probabilities for samples in X.
   *
   * Averages the predicted class probabilities from all trees in the ensemble.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns `float64` class probability matrix of shape (n_samples, n_classes); columns follow
   *   {@link RandomForestClassifier.classes}
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predictProba(X: Tensor): Tensor {
    const { proba, nSamples } = this.averageProba(X, "prediction");
    const nClasses = (this.classLabels as number[]).length;
    return matrixTensor(proba, nSamples, nClasses);
  }

  /**
   * Predict log class probabilities for samples in X (`-Infinity` for impossible classes).
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns `float64` matrix of shape (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predictLogProba(X: Tensor): Tensor {
    const { proba, nSamples } = this.averageProba(X, "prediction");
    const nClasses = (this.classLabels as number[]).length;
    for (let i = 0; i < proba.length; i++) proba[i] = Math.log(proba[i] as number);
    return matrixTensor(proba, nSamples, nClasses);
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
    this.requireFitted("scoring");
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
   * Out-of-bag accuracy: each training sample is classified by the trees whose bootstrap
   * sample did not contain it (probabilities averaged, as in scikit-learn). Samples that
   * every tree saw are ignored; the score is NaN if there are none left.
   * Only available when `oobScore=true` (which requires `bootstrap=true`) was set before `fit`.
   *
   * @returns OOB accuracy score in range [0, 1]
   * @throws {NotFittedError} If the model has not been fitted or oobScore was not enabled
   */
  get oobScore(): number {
    if (!this.fitted || this.oobScore_ === undefined) {
      throw new NotFittedError(
        "RandomForestClassifier must be fitted with oobScore=true to access oobScore"
      );
    }
    return this.oobScore_;
  }

  /**
   * Out-of-bag class probabilities of the training samples, shape (n_samples, n_classes).
   * Rows of samples that no tree left out are NaN. Only available with `oobScore=true`.
   *
   * @throws {NotFittedError} If the model has not been fitted or oobScore was not enabled
   */
  get oobDecisionFunction(): Tensor {
    if (!this.fitted || this.oobDecision_ === undefined || !this.classLabels) {
      throw new NotFittedError(
        "RandomForestClassifier must be fitted with oobScore=true to access oobDecisionFunction"
      );
    }
    return matrixTensor(this.oobDecision_.slice(), this.fitNSamples, this.classLabels.length);
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
    return labelTensor(this.classLabels, this.labelDType);
  }

  /**
   * Get feature importances averaged across all trees.
   *
   * @returns `float64` tensor of shape (n_features,) that sums to 1 (all zeros if no tree split)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get featureImportances(): Tensor {
    if (!this.fitted || this.trees.length === 0 || this.nFeatures === undefined) {
      throw new NotFittedError(
        "RandomForestClassifier must be fitted to access feature_importances_"
      );
    }
    return averageImportances(this.trees, this.nFeatures);
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    const cw = this.classWeight;
    return {
      ...publicParams(this.params),
      criterion: this.criterion,
      classWeight: cw === undefined || typeof cw === "string" ? cw : { ...cw },
    };
  }

  /**
   * Set the parameters of this estimator. Either all given parameters are applied or, if one
   * is invalid, none. Changes only affect the next `fit`.
   *
   * @param params - Parameters to set
   * @returns this
   * @throws {InvalidParameterError} If a parameter value is invalid or unknown
   */
  setParams(params: Record<string, unknown>): this {
    const next: ForestParams = { ...this.params };
    let criterion = this.criterion;
    let classWeight = this.classWeight;
    for (const [key, value] of Object.entries(params)) {
      if (key === "classWeight") {
        classWeight = checkClassWeight(value, true);
      } else if (key === "criterion") {
        if (value !== "gini" && value !== "entropy" && value !== "log_loss") {
          throw new InvalidParameterError(
            `criterion must be "gini", "entropy", or "log_loss"; received ${String(value)}`,
            "criterion",
            value
          );
        }
        criterion = value;
      } else if (!assignParam(next, key, value)) {
        throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    this.params = next;
    this.criterion = criterion;
    this.classWeight = classWeight;
    return this;
  }

  /**
   * Create an unfitted copy with the same hyperparameters.
   */
  clone(): RandomForestClassifier {
    return new RandomForestClassifier(
      definedOptions(this.getParams()) as RandomForestClassifierOptions
    );
  }
}

// ---------------------------------------------------------------------------
// Regressor
// ---------------------------------------------------------------------------

/**
 * Random Forest Regressor.
 *
 * An ensemble of decision tree regressors trained on bootstrap samples with a random subset
 * of the features examined at every split. Predictions are the mean of the trees' predictions.
 *
 * **Algorithm**:
 * 1. Draw `nEstimators` bootstrap samples from the training data
 * 2. Grow a decision tree on each sample, examining `maxFeatures` random features per split
 * 3. Average the trees' predictions
 *
 * By default every feature is examined at every split (`maxFeatures` 1, which JavaScript cannot
 * tell apart from the float 1.0 that scikit-learn uses for "all features"; `null` states it
 * explicitly). To examine fewer, pass `"sqrt"`, `"log2"`, an integer >= 2, or a fraction in
 * (0, 1). `maxDepth` defaults to 10 (not unlimited); pass `Infinity` to grow full trees.
 *
 * @example
 * ```ts
 * import { RandomForestRegressor } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X_train = tensor([[1], [2], [3], [4], [5], [6]]);
 * const y_train = tensor([1.1, 1.9, 3.2, 3.9, 5.1, 6.0]);
 * const reg = new RandomForestRegressor({ nEstimators: 50, randomState: 0 });
 * reg.fit(X_train, y_train);
 * const predictions = reg.predict(tensor([[2.5], [4.5]]));
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-tree | Deepbox Decision Trees}
 */
export class RandomForestRegressor implements Regressor {
  private params: ForestParams;

  private trees: DecisionTreeRegressor[] = [];
  private inBag: Uint8Array[] = [];
  private nFeatures?: number;
  private fitNSamples = 0;
  private fitted = false;
  private oobScore_?: number;
  private oobPrediction_?: Float64Array;

  /**
   * @param options - Hyperparameters, see {@link RandomForestRegressorOptions}
   * @throws {InvalidParameterError} If an option is out of range or inconsistent
   */
  constructor(options: RandomForestRegressorOptions = {}) {
    this.params = paramsFromOptions(options, 1.0);
  }

  /**
   * Fit the random forest regressor on training data.
   *
   * Builds an ensemble of decision trees, each trained on a bootstrap sample with random
   * feature subsets. Calling `fit` again discards the previous forest unless `warmStart` is
   * set and `nEstimators` was increased, in which case only the missing trees are grown.
   *
   * With `sampleWeight` every tree is grown with weighted sums, see
   * {@link DecisionTreeRegressor.fit}. The out-of-bag score stays unweighted.
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
   * @throws {InvalidParameterError} If the parameters are inconsistent, or a warm start would
   *   need fewer trees than the forest already has
   */
  // biome-ignore lint/suspicious/noConfusingVoidType: `void` keeps this compatible with Estimator.fit(X, y, params)
  fit(X: Tensor, y: Tensor, sampleWeightArg?: Tensor | void): this {
    validateFitInputs(X, y);
    const p = this.params;
    assertConsistent(p);

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const x = toFloat64View(X);
    const yv = toFloat64View(y);
    const weights = readSampleWeight(sampleWeightArg as Tensor | undefined, nSamples);

    let trees: DecisionTreeRegressor[] = [];
    let inBag: Uint8Array[] = [];
    let start = 0;
    if (p.warmStart && this.fitted && this.trees.length > 0) {
      if (nFeatures !== this.nFeatures) {
        throw new ShapeError(
          `X has ${nFeatures} features but the warm-started forest was fitted with ${this.nFeatures} features`
        );
      }
      if (p.oobScore) assertBagsAvailable(this.trees.length, this.inBag.length);
      if (p.oobScore && nSamples !== this.fitNSamples) {
        throw new ShapeError(
          `X has ${nSamples} samples but the warm-started forest was fitted with ${this.fitNSamples}; the out-of-bag score needs the same training set`
        );
      }
      start = this.trees.length;
      if (start > p.nEstimators) {
        throw new InvalidParameterError(
          `nEstimators=${p.nEstimators} must be >= the ${start} trees already fitted when warmStart=true`,
          "nEstimators",
          p.nEstimators
        );
      }
      if (start === p.nEstimators) return this;
      trees = this.trees.slice();
      inBag = this.inBag.slice();
    }

    const perSplit = resolveMaxFeatures(p.maxFeatures, nFeatures, true);
    const drawSize = resolveDrawSize(p.maxSamples, nSamples);
    const seeds = drawTreeSeeds(p.randomState, p.nEstimators, start);
    const fullX = p.bootstrap ? undefined : matrixTensor(x, nSamples, nFeatures);
    const fullY = p.bootstrap ? undefined : tensor(yv, { dtype: "float64" });
    const fullWeights =
      weights === undefined || p.bootstrap ? undefined : tensor(weights, { dtype: "float64" });

    for (let t = start; t < p.nEstimators; t++) {
      const bootSeed = seeds[2 * (t - start)] as number;
      const treeSeed = seeds[2 * (t - start) + 1] as number;
      const tree = new DecisionTreeRegressor({
        maxDepth: p.maxDepth,
        minSamplesSplit: p.minSamplesSplit,
        minSamplesLeaf: p.minSamplesLeaf,
        maxFeatures: perSplit,
        randomState: treeSeed,
        minImpurityDecrease: p.minImpurityDecrease,
        ...(p.maxLeafNodes === undefined ? {} : { maxLeafNodes: p.maxLeafNodes }),
        ccpAlpha: p.ccpAlpha,
      });
      if (p.bootstrap) {
        const bag = drawBootstrap(bootSeed, drawSize, nSamples);
        const ys = new Float64Array(bag.indices.length);
        for (let k = 0; k < ys.length; k++) ys[k] = yv[bag.indices[k] as number] as number;
        const bagWeights = gatherWeights(weights, bag.indices);
        const xs = matrixTensor(
          gatherRows(x, nFeatures, bag.indices),
          bag.indices.length,
          nFeatures
        );
        if (bagWeights !== undefined && !hasPositiveWeight(bagWeights)) {
          // A bootstrap sample of zero-weight rows only: grow this tree on the whole training set.
          tree.fit(
            matrixTensor(x, nSamples, nFeatures),
            tensor(yv, { dtype: "float64" }),
            tensor(weights as Float64Array, { dtype: "float64" })
          );
        } else {
          tree.fit(
            xs,
            tensor(ys, { dtype: "float64" }),
            bagWeights === undefined ? undefined : tensor(bagWeights, { dtype: "float64" })
          );
        }
        inBag.push(bag.inBag);
      } else {
        tree.fit(fullX as Tensor, fullY as Tensor, fullWeights);
      }
      trees.push(tree);
    }

    let oob: { score: number; prediction: Float64Array } | undefined;
    if (p.oobScore) {
      oob = this.computeOob(trees, inBag, x, yv, nSamples, nFeatures);
    }

    this.trees = trees;
    this.inBag = inBag;
    this.nFeatures = nFeatures;
    this.fitNSamples = nSamples;
    this.fitted = true;
    delete this.oobScore_;
    delete this.oobPrediction_;
    if (oob) {
      this.oobScore_ = oob.score;
      this.oobPrediction_ = oob.prediction;
    }
    return this;
  }

  /**
   * Out-of-bag estimate: every sample is predicted by the mean of the trees whose bootstrap
   * sample did not contain it. Samples that every tree saw get NaN and are left out of the score.
   */
  private computeOob(
    trees: readonly DecisionTreeRegressor[],
    inBag: readonly Uint8Array[],
    x: Float64Array,
    yv: Float64Array,
    nSamples: number,
    nFeatures: number
  ): { score: number; prediction: Float64Array } {
    const sum = new Float64Array(nSamples);
    const counts = new Int32Array(nSamples);
    for (let t = 0; t < trees.length; t++) {
      const tree = trees[t] as DecisionTreeRegressor;
      const { rows, positions } = outOfBagRows(x, nFeatures, inBag[t] as Uint8Array);
      const m = positions.length;
      if (m === 0) continue;
      const preds = toFloat64View(tree.predict(matrixTensor(rows, m, nFeatures)));
      for (let r = 0; r < m; r++) {
        const i = positions[r] as number;
        sum[i] = (sum[i] as number) + (preds[r] as number);
        counts[i]! += 1;
      }
    }
    const prediction = new Float64Array(nSamples);
    const covered: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      const c = counts[i] as number;
      if (c === 0) {
        prediction[i] = Number.NaN;
      } else {
        prediction[i] = (sum[i] as number) / c;
        covered.push(i);
      }
    }
    if (covered.length === 0) return { score: Number.NaN, prediction };
    return { score: r2(yv, prediction, Int32Array.from(covered)), prediction };
  }

  private requireFitted(action: string): { trees: DecisionTreeRegressor[]; nFeatures: number } {
    if (!this.fitted || this.nFeatures === undefined) {
      throw new NotFittedError(`RandomForestRegressor must be fitted before ${action}`);
    }
    return { trees: this.trees, nFeatures: this.nFeatures };
  }

  /**
   * Predict target values for samples in X.
   *
   * Averages predictions from all trees in the ensemble.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns `float64` predicted values of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    const { trees, nFeatures } = this.requireFitted("prediction");
    validatePredictInputs(X, nFeatures, "RandomForestRegressor");

    const nSamples = X.shape[0] ?? 0;
    const sum = new Float64Array(nSamples);
    if (nSamples > 0) {
      for (const tree of trees) {
        const preds = toFloat64View(tree.predict(X));
        for (let i = 0; i < nSamples; i++) sum[i] = (sum[i] as number) + (preds[i] as number);
      }
    }
    const nTrees = trees.length;
    for (let i = 0; i < nSamples; i++) sum[i] = (sum[i] as number) / nTrees;
    return tensor(sum, { dtype: "float64" });
  }

  /**
   * Return the R² score on the given test data and target values.
   *
   * R² = 1 - SS_res / SS_tot, where SS_res = Σ(y - ŷ)² and SS_tot = Σ(y - ȳ)². For constant
   * targets the score is 1 for a perfect prediction and 0 otherwise.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True target values of shape (n_samples,)
   * @returns R² score (best possible is 1.0, can be negative)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional or sample counts mismatch
   * @throws {DataValidationError} If y is empty or contains NaN/Inf values
   */
  score(X: Tensor, y: Tensor): number {
    this.requireFitted("scoring");
    const truth = readScoreTargets(y);
    const predictions = toFloat64View(this.predict(X));
    if (predictions.length !== truth.length) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${predictions.length}, y=${truth.length}`
      );
    }
    return r2(truth, predictions);
  }

  /**
   * Out-of-bag R²: each training sample is predicted by the mean of the trees whose bootstrap
   * sample did not contain it. Samples that every tree saw are ignored; the score is NaN if
   * there are none left. Only available when `oobScore=true` (which requires `bootstrap=true`)
   * was set before `fit`.
   *
   * @returns OOB R² score (best possible is 1.0, can be negative)
   * @throws {NotFittedError} If the model has not been fitted or oobScore was not enabled
   */
  get oobScore(): number {
    if (!this.fitted || this.oobScore_ === undefined) {
      throw new NotFittedError(
        "RandomForestRegressor must be fitted with oobScore=true to access oobScore"
      );
    }
    return this.oobScore_;
  }

  /**
   * Out-of-bag predictions of the training samples, shape (n_samples,); NaN for samples that
   * no tree left out. Only available with `oobScore=true`.
   *
   * @throws {NotFittedError} If the model has not been fitted or oobScore was not enabled
   */
  get oobPrediction(): Tensor {
    if (!this.fitted || this.oobPrediction_ === undefined) {
      throw new NotFittedError(
        "RandomForestRegressor must be fitted with oobScore=true to access oobPrediction"
      );
    }
    return tensor(this.oobPrediction_.slice(), { dtype: "float64" });
  }

  /**
   * Get feature importances averaged across all trees.
   *
   * @returns `float64` tensor of shape (n_features,) that sums to 1 (all zeros if no tree split)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get featureImportances(): Tensor {
    if (!this.fitted || this.trees.length === 0 || this.nFeatures === undefined) {
      throw new NotFittedError(
        "RandomForestRegressor must be fitted to access feature_importances_"
      );
    }
    return averageImportances(this.trees, this.nFeatures);
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return publicParams(this.params);
  }

  /**
   * Set the parameters of this estimator. Either all given parameters are applied or, if one
   * is invalid, none. Changes only affect the next `fit`.
   *
   * @param params - Parameters to set
   * @returns this
   * @throws {InvalidParameterError} If a parameter value is invalid or unknown
   */
  setParams(params: Record<string, unknown>): this {
    const next: ForestParams = { ...this.params };
    for (const [key, value] of Object.entries(params)) {
      if (!assignParam(next, key, value)) {
        throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    this.params = next;
    return this;
  }

  /**
   * Create an unfitted copy with the same hyperparameters.
   */
  clone(): RandomForestRegressor {
    return new RandomForestRegressor(
      definedOptions(this.getParams()) as RandomForestRegressorOptions
    );
  }
}
