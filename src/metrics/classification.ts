import { DTypeError, InvalidParameterError, ShapeError } from "../core/errors";
import { type Tensor, Tensor as TensorClass, tensor } from "../ndarray";
import { radixArgsortF64 } from "../ndarray/ops/radix";
import {
  assertSameSizeVectors,
  assertVectorLike,
  compensatedSum,
  type DenseLabels,
  denseLabels,
  nonZeroWeightSum,
  readSampleWeight,
  type SampleWeightInput,
  type WeightedMetricOptions,
} from "./_internal";

type Label = number | string | bigint;
type Average = "binary" | "micro" | "macro" | "weighted";

const AVERAGE_MODES: readonly string[] = ["binary", "micro", "macro", "weighted"];

/** Below this many samples a comparator sort beats the radix argsort's fixed setup cost. */
const SMALL_SORT_THRESHOLD = 256;

/**
 * Options object accepted by {@link precision}, {@link recall}, {@link f1Score},
 * {@link fbetaScore} and {@link jaccardScore} in place of the positional `average`.
 *
 * @example
 * ```ts
 * import { precision } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([0, 1, 2, 2, 1]);
 * const yPred = tensor([0, 2, 2, 2, 1]);
 * precision(yTrue, yPred, { average: "macro", sampleWeight: [1, 1, 2, 1, 1] });
 * ```
 */
export type AveragedMetricOptions = {
  /**
   * Averaging strategy: `"binary"` (positive class 1), `"micro"`, `"macro"`,
   * `"weighted"` (by support) or `null` for one value per class in label order.
   * When omitted, `"binary"` is used for at most two distinct labels and `"weighted"`
   * otherwise; with `labels` given and no `average`, `"weighted"` is used.
   */
  readonly average?: Average | null;
  /**
   * Classes to score, in the order they are reported for `average: null`. Samples whose
   * true or predicted label is not listed still count as errors of the listed classes,
   * but unlisted classes are not scored. Ignored with `average: "binary"`. Defaults to
   * the sorted union of the labels in yTrue and yPred. For int64 tensors plain integers
   * are converted to bigint.
   */
  readonly labels?: ReadonlyArray<number | string | bigint>;
  /**
   * Value returned when a score is undefined because its denominator is zero (a class
   * that is never predicted for precision, has no true samples for recall, and so on).
   * `0` or `1`, `NaN` (undefined scores are then left out of `"macro"` and `"weighted"`
   * averages) or `"warn"`, which acts as `0` because Deepbox does not emit warnings.
   * Default: `0`, and `1` for {@link jaccardScore} (two empty sets are identical).
   */
  readonly zeroDivision?: number | "warn";
  /** One weight per sample; every count and support is a weighted sum. */
  readonly sampleWeight?: SampleWeightInput;
};

type AveragedArg = Average | null | AveragedMetricOptions | undefined;

type LabelKind = "string" | "int64" | "numeric";

function labelKind(t: Tensor): LabelKind {
  return t.dtype === "string" ? "string" : t.dtype === "int64" ? "int64" : "numeric";
}

/**
 * Label kind both tensors are compared as. int64 and numeric (or bool) labels may be
 * mixed: the int64 values are then compared as numbers ("numeric"). Strings only match
 * strings.
 */
function assertComparableLabelTypes(yTrue: Tensor, yPred: Tensor): LabelKind {
  const trueKind = labelKind(yTrue);
  const predKind = labelKind(yPred);
  if (trueKind === predKind) return trueKind;
  if (trueKind !== "string" && predKind !== "string") return "numeric";
  throw new DTypeError("yTrue and yPred must use compatible label types");
}

function compareLabels(a: Label, b: Label): number {
  if (typeof a === "number" && typeof b === "number") return a - b;
  if (typeof a === "string" && typeof b === "string") return a < b ? -1 : a > b ? 1 : 0;
  if (typeof a === "bigint" && typeof b === "bigint") return a < b ? -1 : a > b ? 1 : 0;
  return 0;
}

function throwNonBinary(value: Label, name: string, index: number): never {
  throw new InvalidParameterError(
    `${name} must contain only binary values (0 or 1); found ${String(value)} at index ${index}`,
    name,
    value
  );
}

/**
 * Read a label tensor as 0/1 flags. Accepts numeric, bool and int64 tensors;
 * anything other than 0 or 1 (and any NaN/Infinity) is rejected.
 */
function binaryLabels(t: Tensor, name: string): Uint8Array {
  if (t.dtype === "string") {
    throw new DTypeError(`${name} must contain numeric binary labels (0 or 1), not strings`);
  }
  const labels = denseLabels(t, name);
  const out = new Uint8Array(labels.length);
  for (let i = 0; i < out.length; i++) {
    const v = labels[i] as number | bigint;
    if (v === 1 || v === 1n) out[i] = 1;
    else if (v !== 0 && v !== 0n) throwNonBinary(v, name, i);
  }
  return out;
}

/** Read scores/probabilities as finite float64 values (numeric, bool and int64 tensors). */
function denseScores(t: Tensor, name: string): Float64Array {
  if (t.dtype === "string") {
    throw new DTypeError(`${name} must be a numeric tensor; string tensors are not supported`);
  }
  const values = denseLabels(t, name);
  if (values instanceof Float64Array) return values;
  const out = new Float64Array(values.length);
  for (let i = 0; i < out.length; i++) out[i] = Number(values[i]);
  return out;
}

/** Convert int64 labels to numbers; values beyond 2^53 cannot be compared with numbers. */
function bigintLabelsToNumbers(values: bigint[], name: string): Float64Array {
  const out = new Float64Array(values.length);
  for (let i = 0; i < out.length; i++) {
    const v = values[i] as bigint;
    const asNumber = Number(v);
    if (!Number.isSafeInteger(asNumber)) {
      throw new InvalidParameterError(
        `${name} holds the int64 label ${String(v)}, which is outside the safe integer range and cannot be compared with numeric labels`,
        name,
        v
      );
    }
    out[i] = asNumber;
  }
  return out;
}

type LabelPair = { kind: LabelKind; trueLabels: DenseLabels; predLabels: DenseLabels };

/**
 * Read both label vectors on one common footing. When an int64 tensor is paired with a
 * numeric one, the int64 values become numbers (they must be safe integers, and the
 * numeric side must be integer valued).
 */
function readLabelPair(yTrue: Tensor, yPred: Tensor): LabelPair {
  const kind = assertComparableLabelTypes(yTrue, yPred);
  let trueLabels = denseLabels(yTrue, "yTrue");
  let predLabels = denseLabels(yPred, "yPred");
  if (labelKind(yTrue) !== labelKind(yPred)) {
    const numericSide = trueLabels instanceof Float64Array ? trueLabels : predLabels;
    for (let i = 0; i < numericSide.length; i++) {
      if (!Number.isInteger(numericSide[i])) {
        throw new DTypeError(
          `Numeric labels compared with int64 labels must be integers; found ${String(numericSide[i])}`
        );
      }
    }
    if (!(trueLabels instanceof Float64Array)) {
      trueLabels = bigintLabelsToNumbers(trueLabels as bigint[], "yTrue");
    }
    if (!(predLabels instanceof Float64Array)) {
      predLabels = bigintLabelsToNumbers(predLabels as bigint[], "yPred");
    }
  }
  return { kind, trueLabels, predLabels };
}

/** Whether the two label arrays together hold more than two distinct values. */
function hasMoreThanTwoClasses(a: DenseLabels, b: DenseLabels): boolean {
  const seen: Label[] = [];
  for (const arr of [a, b]) {
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i] as Label;
      if (!seen.includes(v)) {
        seen.push(v);
        if (seen.length > 2) return true;
      }
    }
  }
  return false;
}

function isMulticlass(yTrue: Tensor, yPred: Tensor): boolean {
  if (yTrue.dtype === "string" || yPred.dtype === "string") return true;
  const { trueLabels, predLabels } = readLabelPair(yTrue, yPred);
  return hasMoreThanTwoClasses(trueLabels, predLabels);
}

type EncodedLabels = {
  /** Distinct labels, sorted ascending. */
  classes: Label[];
  /** Index into `classes` for every sample of yTrue (-1: not in `classes`). */
  trueIdx: Int32Array;
  /** Index into `classes` for every sample of yPred (-1: not in `classes`). */
  predIdx: Int32Array;
};

/**
 * Map both label vectors onto integer class indices. With `fixedClasses` the
 * class list is given (samples outside it map to -1); otherwise it is the
 * sorted union of the labels found in both vectors.
 */
function encodeLabels(
  yTrue: Tensor,
  yPred: Tensor,
  fixedClasses?: readonly Label[]
): EncodedLabels {
  const { trueLabels: tl, predLabels: pl } = readLabelPair(yTrue, yPred);
  const n = tl.length;
  const trueIdx = new Int32Array(n);
  const predIdx = new Int32Array(n);

  if (fixedClasses !== undefined) {
    const index = new Map<Label, number>();
    for (let i = 0; i < fixedClasses.length; i++) index.set(fixedClasses[i] as Label, i);
    for (let i = 0; i < n; i++) {
      trueIdx[i] = index.get(tl[i] as Label) ?? -1;
      predIdx[i] = index.get(pl[i] as Label) ?? -1;
    }
    return { classes: [...fixedClasses], trueIdx, predIdx };
  }

  const index = new Map<Label, number>();
  const firstSeen: Label[] = [];
  for (let i = 0; i < n; i++) {
    const a = tl[i] as Label;
    let ia = index.get(a);
    if (ia === undefined) {
      ia = firstSeen.length;
      index.set(a, ia);
      firstSeen.push(a);
    }
    trueIdx[i] = ia;
    const b = pl[i] as Label;
    let ib = index.get(b);
    if (ib === undefined) {
      ib = firstSeen.length;
      index.set(b, ib);
      firstSeen.push(b);
    }
    predIdx[i] = ib;
  }

  const k = firstSeen.length;
  const order = Array.from({ length: k }, (_, i) => i);
  order.sort((x, y) => compareLabels(firstSeen[x] as Label, firstSeen[y] as Label));
  const rank = new Int32Array(k);
  const classes: Label[] = new Array<Label>(k);
  for (let r = 0; r < k; r++) {
    const original = order[r] as number;
    rank[original] = r;
    classes[r] = firstSeen[original] as Label;
  }
  for (let i = 0; i < n; i++) {
    trueIdx[i] = rank[trueIdx[i] as number] as number;
    predIdx[i] = rank[predIdx[i] as number] as number;
  }
  return { classes, trueIdx, predIdx };
}

type ClassCounts = {
  classes: Label[];
  tp: Float64Array;
  fp: Float64Array;
  fn: Float64Array;
  /** Number (or total weight) of true samples per class. */
  support: Float64Array;
  totalTp: number;
  totalFp: number;
  totalFn: number;
};

/**
 * Per-class true/false positive and false negative counts over the union of labels, or
 * over `fixedClasses` when given. With `weights` every sample adds its weight instead of 1.
 */
function classCounts(
  yTrue: Tensor,
  yPred: Tensor,
  weights?: Float64Array,
  fixedClasses?: readonly Label[]
): ClassCounts {
  const { classes, trueIdx, predIdx } = encodeLabels(yTrue, yPred, fixedClasses);
  const k = classes.length;
  const tp = new Float64Array(k);
  const fp = new Float64Array(k);
  const fn = new Float64Array(k);
  const support = new Float64Array(k);
  for (let i = 0; i < trueIdx.length; i++) {
    const a = trueIdx[i] as number;
    const b = predIdx[i] as number;
    const w = weights === undefined ? 1 : (weights[i] as number);
    if (a >= 0) support[a] = (support[a] as number) + w;
    if (a === b) {
      if (a >= 0) tp[a] = (tp[a] as number) + w;
    } else {
      if (b >= 0) fp[b] = (fp[b] as number) + w;
      if (a >= 0) fn[a] = (fn[a] as number) + w;
    }
  }
  let totalTp = 0;
  let totalFp = 0;
  let totalFn = 0;
  for (let c = 0; c < k; c++) {
    totalTp += tp[c] as number;
    totalFp += fp[c] as number;
    totalFn += fn[c] as number;
  }
  return { classes, tp, fp, fn, support, totalTp, totalFp, totalFn };
}

/** Confusion counts for validated 0/1 labels (positive class = 1). */
function binaryCounts(
  yTrue: Tensor,
  yPred: Tensor,
  weights?: Float64Array
): { tp: number; fp: number; fn: number; tn: number } {
  const t = binaryLabels(yTrue, "yTrue");
  const p = binaryLabels(yPred, "yPred");
  let tp = 0;
  let fp = 0;
  let fn = 0;
  let tn = 0;
  for (let i = 0; i < t.length; i++) {
    const w = weights === undefined ? 1 : (weights[i] as number);
    if (t[i] === 1) {
      if (p[i] === 1) tp += w;
      else fn += w;
    } else if (p[i] === 1) fp += w;
    else tn += w;
  }
  return { tp, fp, fn, tn };
}

/** Number of positions where the two label vectors agree. */
function countMatches(yTrue: Tensor, yPred: Tensor): number {
  const { trueLabels: t, predLabels: p } = readLabelPair(yTrue, yPred);
  let matches = 0;
  for (let i = 0; i < t.length; i++) {
    if (t[i] === p[i]) matches++;
  }
  return matches;
}

/**
 * A score from raw (tp, fp, fn) counts. Returns NaN when its denominator is zero, which
 * the caller replaces by the zero-division value.
 */
type CountScore = (tp: number, fp: number, fn: number) => number;

const precisionFromCounts: CountScore = (tp, fp) => (tp + fp === 0 ? Number.NaN : tp / (tp + fp));
const recallFromCounts: CountScore = (tp, _fp, fn) => (tp + fn === 0 ? Number.NaN : tp / (tp + fn));
const jaccardFromCounts: CountScore = (tp, fp, fn) =>
  tp + fp + fn === 0 ? Number.NaN : tp / (tp + fp + fn);

/**
 * F-beta from raw counts: (1 + b²)·TP / ((1 + b²)·TP + b²·FN + FP). Equal to the
 * harmonic-mean definition but never divides by a rounded precision or recall.
 */
function fbetaFromCounts(betaSq: number): CountScore {
  return (tp, fp, fn) => {
    const denominator = (1 + betaSq) * tp + betaSq * fn + fp;
    return denominator === 0 ? Number.NaN : ((1 + betaSq) * tp) / denominator;
  };
}

function definedOrZero(value: number): number {
  return Number.isNaN(value) ? 0 : value;
}

function parseAveragedArg(arg: AveragedArg): AveragedMetricOptions {
  if (arg === undefined) return {};
  if (arg === null) return { average: null };
  if (typeof arg === "object") return arg;
  return { average: arg };
}

function resolveZeroDivision(value: number | "warn" | undefined, fallback: number): number {
  if (value === undefined) return fallback;
  if (value === "warn") return 0;
  if (typeof value === "number" && (value === 0 || value === 1 || Number.isNaN(value))) {
    return value;
  }
  throw new InvalidParameterError(
    `Invalid zeroDivision parameter: ${String(value)}. Must be 0, 1, NaN or 'warn'`,
    "zeroDivision",
    value
  );
}

function resolveAverage(
  yTrue: Tensor,
  yPred: Tensor,
  average: Average | null | undefined,
  hasLabels: boolean
): Average | null {
  if (average === undefined) {
    return hasLabels || isMulticlass(yTrue, yPred) ? "weighted" : "binary";
  }
  if (average !== null && !AVERAGE_MODES.includes(average)) {
    throw new InvalidParameterError(
      `Invalid average parameter: ${String(average)}. Must be one of: 'binary', 'micro', 'macro', 'weighted', or null`,
      "average",
      average
    );
  }
  return average;
}

/**
 * Shared driver for precision, recall, F1, F-beta and Jaccard. `score` maps raw
 * (tp, fp, fn) counts to the metric, so every averaging mode goes through the
 * same code and the per-class values are computed from one pass over the data.
 * Undefined scores (NaN) become `zeroDivision`; NaN stays out of macro and weighted means.
 */
function averagedScore(
  yTrue: Tensor,
  yPred: Tensor,
  arg: AveragedArg,
  score: CountScore,
  defaultZeroDivision: number
): number | number[] {
  const options = parseAveragedArg(arg);
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");
  const zeroDivision = resolveZeroDivision(options.zeroDivision, defaultZeroDivision);
  const mode = resolveAverage(yTrue, yPred, options.average, options.labels !== undefined);
  const weights = readSampleWeight(options.sampleWeight, yTrue.size);
  const defined = (value: number): number => (Number.isNaN(value) ? zeroDivision : value);

  if (yTrue.size === 0 && options.labels === undefined) {
    return mode === null ? [] : zeroDivision;
  }

  if (mode === "binary") {
    if (yTrue.dtype === "string" || yPred.dtype === "string") {
      throw new InvalidParameterError(
        "Binary average requires numeric labels (0/1). Use 'macro', 'micro', or 'weighted' for string labels.",
        "average",
        mode
      );
    }
    const { tp, fp, fn } = binaryCounts(yTrue, yPred, weights);
    return defined(score(tp, fp, fn));
  }

  const fixed =
    options.labels === undefined
      ? undefined
      : normalizeLabelList(options.labels, assertComparableLabelTypes(yTrue, yPred));
  const counts = classCounts(yTrue, yPred, weights, fixed);
  if (mode === "micro") return defined(score(counts.totalTp, counts.totalFp, counts.totalFn));

  const k = counts.classes.length;
  const perClass = new Array<number>(k);
  for (let i = 0; i < k; i++) {
    perClass[i] = defined(
      score(counts.tp[i] as number, counts.fp[i] as number, counts.fn[i] as number)
    );
  }
  if (mode === null) return perClass;

  if (k === 0) return zeroDivision;
  let sum = 0;
  let totalWeight = 0;
  for (let i = 0; i < k; i++) {
    const value = perClass[i] as number;
    if (Number.isNaN(value)) continue;
    const weight = mode === "macro" ? 1 : (counts.support[i] as number);
    sum += value * weight;
    totalWeight += weight;
  }
  if (mode === "macro") return totalWeight === 0 ? Number.NaN : sum / totalWeight;
  return totalWeight === 0 ? zeroDivision : sum / totalWeight;
}

/**
 * Calculates the accuracy classification score.
 *
 * Accuracy is the fraction of predictions that match the true labels.
 * It's the most intuitive performance measure but can be misleading
 * for imbalanced datasets.
 *
 * **Formula**: accuracy = (correct predictions) / (total predictions)
 *
 * **Time Complexity**: O(n) where n is the number of samples
 * **Space Complexity**: O(n) for the dense copies of the label vectors
 *
 * With `options.sampleWeight` each sample counts with its weight:
 * Σ w_i * [y_i == p_i] / Σ w_i.
 *
 * int64 labels may be compared with integer-valued numeric labels.
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated targets as returned by a classifier
 * @param options - Optional `sampleWeight`, one weight per sample
 * @returns Accuracy score in range [0, 1], where 1 is perfect accuracy. Returns 0 for empty input.
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors, or
 *   `sampleWeight` has the wrong length
 * @throws {DTypeError} If yTrue and yPred use incompatible label types
 * @throws {InvalidParameterError} If `sampleWeight` sums to zero
 * @throws {DataValidationError} If numeric labels contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { accuracy } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([0, 1, 1, 0, 1]);
 * const yPred = tensor([0, 1, 0, 0, 1]);
 * const acc = accuracy(yTrue, yPred); // 0.8 (4 out of 5 correct)
 * accuracy(yTrue, yPred, { sampleWeight: [1, 1, 3, 1, 1] }); // 0.5714285714285714
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function accuracy(
  yTrue: Tensor,
  yPred: Tensor,
  options: WeightedMetricOptions = {}
): number {
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");
  const weights = readSampleWeight(options.sampleWeight, yTrue.size);
  if (yTrue.size === 0) return 0;
  if (weights === undefined) return countMatches(yTrue, yPred) / yTrue.size;

  const { trueLabels: t, predLabels: p } = readLabelPair(yTrue, yPred);
  const total = nonZeroWeightSum(weights);
  let matched = 0;
  for (let i = 0; i < t.length; i++) {
    if (t[i] === p[i]) matched += weights[i] as number;
  }
  return matched / total;
}

/**
 * Calculates the precision classification score.
 *
 * Precision is the ratio of true positives to all positive predictions.
 * It answers: "Of all samples predicted as positive, how many are actually positive?"
 * High precision means low false positive rate.
 *
 * **Formula**: precision = TP / (TP + FP)
 *
 * When a class is never predicted (TP + FP = 0) its precision is reported as 0.
 *
 * **Time Complexity**: O(n) for binary, O(n + k) for multiclass where k is the number of classes
 * **Space Complexity**: O(n + k)
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated targets as returned by a classifier
 * @param average - Averaging strategy: 'binary', 'micro', 'macro', 'weighted', or null.
 *   When omitted, 'binary' is used for at most two distinct labels and 'weighted' otherwise.
 *   - 'binary': Calculate metrics for the positive class (label 1) only; labels must be 0 or 1
 *   - 'micro': Calculate metrics globally by counting total TP, FP, FN
 *   - 'macro': Calculate metrics for each class, return unweighted mean
 *   - 'weighted': Calculate metrics for each class, return mean weighted by support
 *   - null: Return array of scores for each class, in ascending label order
 *
 *   Instead of the string an options object {@link AveragedMetricOptions} can be passed:
 *   `{ average, labels, zeroDivision, sampleWeight }`.
 * @returns Precision score(s) in range [0, 1]
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If yTrue and yPred use incompatible label types
 * @throws {DataValidationError} If numeric labels contain NaN or infinite values
 * @throws {InvalidParameterError} If average is invalid, or 'binary' is used with labels other
 *   than 0/1 (including string labels)
 *
 * @example
 * ```ts
 * import { precision } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * // Binary classification
 * const yTrue = tensor([0, 1, 1, 0, 1]);
 * const yPred = tensor([0, 1, 0, 0, 1]);
 * const prec = precision(yTrue, yPred); // 1.0 (2 TP, 0 FP)
 *
 * // Multiclass
 * const yTrueMulti = tensor([0, 1, 2, 0, 1, 2]);
 * const yPredMulti = tensor([0, 2, 1, 0, 0, 1]);
 * const precMacro = precision(yTrueMulti, yPredMulti, 'macro');
 *
 * // Options object: classes to score, zero-division value and sample weights
 * const perClass = precision(yTrueMulti, yPredMulti, {
 *   average: null,
 *   labels: [2, 1, 0],
 *   zeroDivision: 1,
 *   sampleWeight: [1, 2, 1, 1, 1, 1],
 * }); // [0, 0, 0.6667]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function precision(yTrue: Tensor, yPred: Tensor): number;
export function precision(
  yTrue: Tensor,
  yPred: Tensor,
  average: "binary" | "micro" | "macro" | "weighted"
): number;
export function precision(yTrue: Tensor, yPred: Tensor, average: null): number[];
export function precision(
  yTrue: Tensor,
  yPred: Tensor,
  options: AveragedMetricOptions & { readonly average: null }
): number[];
export function precision(
  yTrue: Tensor,
  yPred: Tensor,
  options: AveragedMetricOptions & { readonly average?: Average }
): number;
export function precision(
  yTrue: Tensor,
  yPred: Tensor,
  average?: Average | null | AveragedMetricOptions
): number | number[];
export function precision(
  yTrue: Tensor,
  yPred: Tensor,
  average?: Average | null | AveragedMetricOptions
): number | number[] {
  return averagedScore(yTrue, yPred, average, precisionFromCounts, 0);
}

/**
 * Calculates the recall classification score (sensitivity, true positive rate).
 *
 * Recall is the ratio of true positives to all actual positive samples.
 * It answers: "Of all actual positive samples, how many did we correctly identify?"
 * High recall means low false negative rate.
 *
 * **Formula**: recall = TP / (TP + FN)
 *
 * When a class has no true samples (TP + FN = 0) its recall is reported as 0.
 *
 * **Time Complexity**: O(n) for binary, O(n + k) for multiclass where k is the number of classes
 * **Space Complexity**: O(n + k)
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated targets as returned by a classifier
 * @param average - Averaging strategy: 'binary', 'micro', 'macro', 'weighted', or null.
 *   When omitted, 'binary' is used for at most two distinct labels and 'weighted' otherwise.
 *   See {@link precision} for the meaning of each mode. An options object
 *   {@link AveragedMetricOptions} (`{ average, labels, zeroDivision, sampleWeight }`) is
 *   accepted as well.
 * @returns Recall score(s) in range [0, 1]
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If yTrue and yPred use incompatible label types
 * @throws {DataValidationError} If numeric labels contain NaN or infinite values
 * @throws {InvalidParameterError} If average is invalid, or 'binary' is used with labels other
 *   than 0/1 (including string labels)
 *
 * @example
 * ```ts
 * import { recall } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([0, 1, 1, 0, 1]);
 * const yPred = tensor([0, 1, 0, 0, 1]);
 * const rec = recall(yTrue, yPred); // 0.667 (2 out of 3 positives found)
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function recall(yTrue: Tensor, yPred: Tensor): number;
export function recall(
  yTrue: Tensor,
  yPred: Tensor,
  average: "binary" | "micro" | "macro" | "weighted"
): number;
export function recall(yTrue: Tensor, yPred: Tensor, average: null): number[];
export function recall(
  yTrue: Tensor,
  yPred: Tensor,
  options: AveragedMetricOptions & { readonly average: null }
): number[];
export function recall(
  yTrue: Tensor,
  yPred: Tensor,
  options: AveragedMetricOptions & { readonly average?: Average }
): number;
export function recall(
  yTrue: Tensor,
  yPred: Tensor,
  average?: Average | null | AveragedMetricOptions
): number | number[];
export function recall(
  yTrue: Tensor,
  yPred: Tensor,
  average?: Average | null | AveragedMetricOptions
): number | number[] {
  return averagedScore(yTrue, yPred, average, recallFromCounts, 0);
}

/**
 * Calculates the F1 score (harmonic mean of precision and recall).
 *
 * F1 score is the harmonic mean of precision and recall, providing a single
 * metric that balances both concerns. It's especially useful when you need
 * to balance false positives and false negatives.
 *
 * **Formula**: F1 = 2 * (precision * recall) / (precision + recall) = 2TP / (2TP + FP + FN)
 *
 * For 'macro' and 'weighted' averaging the F1 of each class is computed first and
 * then averaged, which differs from the F1 of the averaged precision and recall.
 * Classes with no true and no predicted samples score 0.
 *
 * **Time Complexity**: O(n) for binary, O(n + k) for multiclass
 * **Space Complexity**: O(n + k)
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated targets as returned by a classifier
 * @param average - Averaging strategy: 'binary', 'micro', 'macro', 'weighted', or null.
 *   When omitted, 'binary' is used for at most two distinct labels and 'weighted' otherwise.
 *   An options object {@link AveragedMetricOptions}
 *   (`{ average, labels, zeroDivision, sampleWeight }`) is also accepted.
 * @returns F1 score(s) in range [0, 1]
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If yTrue and yPred use incompatible label types
 * @throws {DataValidationError} If numeric labels contain NaN or infinite values
 * @throws {InvalidParameterError} If average is invalid, or 'binary' is used with labels other
 *   than 0/1 (including string labels)
 *
 * @example
 * ```ts
 * import { f1Score } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([0, 1, 1, 0, 1]);
 * const yPred = tensor([0, 1, 0, 0, 1]);
 * const f1 = f1Score(yTrue, yPred); // 0.8
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function f1Score(yTrue: Tensor, yPred: Tensor): number;
export function f1Score(
  yTrue: Tensor,
  yPred: Tensor,
  average: "binary" | "micro" | "macro" | "weighted"
): number;
export function f1Score(yTrue: Tensor, yPred: Tensor, average: null): number[];
export function f1Score(
  yTrue: Tensor,
  yPred: Tensor,
  options: AveragedMetricOptions & { readonly average: null }
): number[];
export function f1Score(
  yTrue: Tensor,
  yPred: Tensor,
  options: AveragedMetricOptions & { readonly average?: Average }
): number;
export function f1Score(
  yTrue: Tensor,
  yPred: Tensor,
  average?: Average | null | AveragedMetricOptions
): number | number[];
export function f1Score(
  yTrue: Tensor,
  yPred: Tensor,
  average?: Average | null | AveragedMetricOptions
): number | number[] {
  return averagedScore(yTrue, yPred, average, fbetaFromCounts(1), 0);
}

/**
 * Calculates the F-beta score.
 *
 * F-beta score is a weighted harmonic mean of precision and recall, where
 * beta controls the trade-off between precision and recall.
 * - beta < 1: More weight on precision
 * - beta = 1: Equal weight (equivalent to F1 score)
 * - beta > 1: More weight on recall
 *
 * **Formula**: F_beta = (1 + beta²) * (precision * recall) / (beta² * precision + recall)
 *
 * With beta = 0 the score equals precision.
 *
 * **Time Complexity**: O(n) for binary, O(n + k) for multiclass
 * **Space Complexity**: O(n + k)
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated targets as returned by a classifier
 * @param beta - Weight of recall vs precision (beta > 1 favors recall, beta < 1 favors precision).
 *   Must be a non-negative finite number.
 * @param average - Averaging strategy: 'binary', 'micro', 'macro', 'weighted', or null.
 *   When omitted, 'binary' is used for at most two distinct labels and 'weighted' otherwise,
 *   the same rule as {@link f1Score}. An options object {@link AveragedMetricOptions}
 *   (`{ average, labels, zeroDivision, sampleWeight }`) is accepted as well.
 * @returns F-beta score(s) in range [0, 1]
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If yTrue and yPred use incompatible label types
 * @throws {DataValidationError} If numeric labels contain NaN or infinite values
 * @throws {InvalidParameterError} If beta or average is invalid, or 'binary' is used with labels
 *   other than 0/1 (including string labels)
 *
 * @example
 * ```ts
 * import { fbetaScore } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([0, 1, 1, 0, 1]);
 * const yPred = tensor([0, 1, 0, 0, 1]);
 * const fb2 = fbetaScore(yTrue, yPred, 2); // Favors recall
 * const fb05 = fbetaScore(yTrue, yPred, 0.5); // Favors precision
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function fbetaScore(yTrue: Tensor, yPred: Tensor, beta: number): number;
export function fbetaScore(
  yTrue: Tensor,
  yPred: Tensor,
  beta: number,
  average: "binary" | "micro" | "macro" | "weighted"
): number;
export function fbetaScore(yTrue: Tensor, yPred: Tensor, beta: number, average: null): number[];
export function fbetaScore(
  yTrue: Tensor,
  yPred: Tensor,
  beta: number,
  options: AveragedMetricOptions & { readonly average: null }
): number[];
export function fbetaScore(
  yTrue: Tensor,
  yPred: Tensor,
  beta: number,
  options: AveragedMetricOptions & { readonly average?: Average }
): number;
export function fbetaScore(
  yTrue: Tensor,
  yPred: Tensor,
  beta: number,
  average?: Average | null | AveragedMetricOptions
): number | number[];
export function fbetaScore(
  yTrue: Tensor,
  yPred: Tensor,
  beta: number,
  average?: Average | null | AveragedMetricOptions
): number | number[] {
  if (!Number.isFinite(beta) || beta < 0) {
    throw new InvalidParameterError("beta must be a non-negative finite number", "beta", beta);
  }
  const betaSq = beta * beta;
  // beta so large that beta² overflows: the score is the recall to within rounding.
  const score = Number.isFinite(betaSq) ? fbetaFromCounts(betaSq) : recallFromCounts;
  return averagedScore(yTrue, yPred, average, score, 0);
}

/**
 * Options for {@link confusionMatrix}.
 */
export type ConfusionMatrixOptions = {
  /**
   * Labels that index the rows and columns, in the order given. Samples whose true
   * or predicted label is not listed are ignored. Labels that never occur produce
   * all-zero rows and columns. Defaults to the sorted union of labels in yTrue and yPred.
   * For int64 tensors, plain integers are converted to bigint; when int64 labels are paired
   * with numeric ones, bigint values are converted to numbers.
   */
  readonly labels?: ReadonlyArray<number | string | bigint>;
  /**
   * Normalize the counts over the true labels (`"true"`, each row sums to 1),
   * the predicted labels (`"pred"`, each column sums to 1) or all samples (`"all"`).
   * Rows or columns that sum to zero stay zero. Default: no normalization (raw counts).
   */
  readonly normalize?: "true" | "pred" | "all" | null;
  /**
   * One weight per sample. Each cell then holds the summed weight of its samples instead
   * of their count (the result is float64 either way).
   */
  readonly sampleWeight?: SampleWeightInput;
};

function normalizeLabelList(
  labels: ReadonlyArray<number | string | bigint>,
  kind: LabelKind
): Label[] {
  if (labels.length === 0) {
    throw new InvalidParameterError("labels must contain at least one label", "labels", labels);
  }
  const out: Label[] = [];
  const seen = new Set<Label>();
  for (const raw of labels) {
    let label: Label = raw;
    if (kind === "numeric" && typeof raw === "bigint") {
      if (!Number.isSafeInteger(Number(raw))) {
        throw new InvalidParameterError(
          `labels holds ${String(raw)}, which is outside the safe integer range and cannot be compared with numeric labels`,
          "labels",
          raw
        );
      }
      label = Number(raw);
    }
    if (kind === "int64" && typeof raw === "number") {
      if (!Number.isSafeInteger(raw)) {
        throw new InvalidParameterError(
          `labels must be integers for int64 tensors; found ${String(raw)}`,
          "labels",
          raw
        );
      }
      label = BigInt(raw);
    }
    const expected = kind === "string" ? "string" : kind === "int64" ? "bigint" : "number";
    if (typeof label !== expected) {
      throw new DTypeError(`labels must be ${expected} values to match the label tensors`);
    }
    if (typeof label === "number" && !Number.isFinite(label)) {
      throw new InvalidParameterError("labels must be finite numbers", "labels", label);
    }
    if (seen.has(label)) {
      throw new InvalidParameterError(
        `labels must be unique; found ${String(label)} more than once`,
        "labels",
        labels
      );
    }
    seen.add(label);
    out.push(label);
  }
  return out;
}

/**
 * Computes the confusion matrix to evaluate classification accuracy.
 *
 * A confusion matrix is a table showing the counts of correct and incorrect
 * predictions broken down by each class. Rows represent true labels,
 * columns represent predicted labels. Classes are listed in ascending order
 * (strings by code unit, like NumPy), unless `options.labels` fixes the order.
 *
 * **Time Complexity**: O(n + k²) where n is number of samples, k is number of classes
 * **Space Complexity**: O(n + k²)
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated targets as returned by a classifier
 * @param options - Optional `labels` (row/column order and subset), `normalize` and `sampleWeight`
 * @returns Confusion matrix as a float64 tensor of shape [n_classes, n_classes]
 *
 * int64 labels may be paired with integer-valued numeric labels (the int64 values must be
 * safe integers); they are then compared as numbers.
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors, or
 *   `sampleWeight` has the wrong length
 * @throws {DTypeError} If yTrue and yPred use incompatible label types, or `labels` does not
 *   match the label type
 * @throws {DataValidationError} If numeric labels contain NaN or infinite values
 * @throws {InvalidParameterError} If `labels` is empty or has duplicates, or `normalize` is invalid
 *
 * @example
 * ```ts
 * import { confusionMatrix } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([0, 1, 1, 0, 1]);
 * const yPred = tensor([0, 1, 0, 0, 1]);
 * const cm = confusionMatrix(yTrue, yPred);
 * // [[2, 0],
 * //  [1, 2]]
 *
 * const rowRates = confusionMatrix(yTrue, yPred, { normalize: "true" });
 * // [[1, 0],
 * //  [1/3, 2/3]]
 *
 * const weighted = confusionMatrix(yTrue, yPred, { sampleWeight: [1, 1, 2, 1, 1] });
 * // [[2, 0],
 * //  [2, 2]]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function confusionMatrix(
  yTrue: Tensor,
  yPred: Tensor,
  options: ConfusionMatrixOptions = {}
): Tensor {
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");
  const kind = assertComparableLabelTypes(yTrue, yPred);
  const weights = readSampleWeight(options.sampleWeight, yTrue.size);

  const normalize = options.normalize ?? null;
  if (normalize !== null && normalize !== "true" && normalize !== "pred" && normalize !== "all") {
    throw new InvalidParameterError(
      `Invalid normalize parameter: ${String(normalize)}. Must be one of: 'true', 'pred', 'all', or null`,
      "normalize",
      normalize
    );
  }
  const fixed = options.labels === undefined ? undefined : normalizeLabelList(options.labels, kind);

  if (yTrue.size === 0 && fixed === undefined) {
    return tensor([], { dtype: "float64" }).reshape([0, 0]);
  }

  const { classes, trueIdx, predIdx } = encodeLabels(yTrue, yPred, fixed);
  const k = classes.length;
  const out = new Float64Array(k * k);
  for (let i = 0; i < trueIdx.length; i++) {
    const r = trueIdx[i] as number;
    const c = predIdx[i] as number;
    if (r < 0 || c < 0) continue;
    out[r * k + c] =
      (out[r * k + c] as number) + (weights === undefined ? 1 : (weights[i] as number));
  }

  if (normalize !== null) {
    const rowSums = new Float64Array(k);
    const colSums = new Float64Array(k);
    let total = 0;
    for (let r = 0; r < k; r++) {
      for (let c = 0; c < k; c++) {
        const v = out[r * k + c] as number;
        rowSums[r] = (rowSums[r] as number) + v;
        colSums[c] = (colSums[c] as number) + v;
        total += v;
      }
    }
    for (let r = 0; r < k; r++) {
      for (let c = 0; c < k; c++) {
        const denominator =
          normalize === "true"
            ? (rowSums[r] as number)
            : normalize === "pred"
              ? (colSums[c] as number)
              : total;
        out[r * k + c] = denominator === 0 ? 0 : (out[r * k + c] as number) / denominator;
      }
    }
  }

  return TensorClass.fromTypedArray({
    data: out,
    shape: [k, k],
    dtype: "float64",
    device: yTrue.device,
  });
}

/**
 * Generates a text classification report showing main classification metrics.
 *
 * The report lists precision, recall, F1-score and support for each class,
 * followed by accuracy and the macro and weighted averages. Only binary
 * labels (0 or 1) are supported. int64 labels may be paired with numeric labels.
 *
 * **Time Complexity**: O(n)
 * **Space Complexity**: O(n)
 *
 * @param yTrue - Ground truth (correct) binary target values (0 or 1)
 * @param yPred - Estimated binary targets as returned by a classifier (0 or 1)
 * @returns Formatted string report with per-class and aggregate classification metrics
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DataValidationError} If labels contain NaN or infinite values
 * @throws {InvalidParameterError} If labels are strings or are not binary (0 or 1)
 *
 * @example
 * ```ts
 * import { classificationReport } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([0, 1, 1, 0, 1]);
 * const yPred = tensor([0, 1, 0, 0, 1]);
 * console.log(classificationReport(yTrue, yPred));
 * // Classification Report:
 * // Class         Precision   Recall      F1-Score    Support
 * // ---------------------------------------------------------
 * // 0             0.6667      1.0000      0.8000      2
 * // 1             1.0000      0.6667      0.8000      3
 * //
 * // Accuracy                              0.8000      5
 * // Macro Avg     0.8333      0.8333      0.8000      5
 * // Weighted Avg  0.8667      0.8000      0.8000      5
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function classificationReport(yTrue: Tensor, yPred: Tensor): string {
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");

  if (yTrue.size === 0) return "Classification Report:\n  (empty)";

  if (yTrue.dtype === "string" || yPred.dtype === "string") {
    throw new InvalidParameterError(
      "classificationReport requires binary numeric labels (0 or 1)",
      "yTrue"
    );
  }
  // Validates that every label is 0 or 1.
  binaryLabels(yTrue, "yTrue");
  binaryLabels(yPred, "yPred");

  const counts = classCounts(yTrue, yPred);
  const classes = counts.classes;
  const n = yTrue.size;
  const f1FromCounts = fbetaFromCounts(1);
  const acc = counts.totalTp / n;

  // The label column must also fit the summary rows ("Weighted Avg" is the longest).
  const maxLabelLen = Math.max(...classes.map((c) => String(c).length), "Weighted Avg".length);
  const colWidth = maxLabelLen + 2;

  let report = "Classification Report:\n";
  report +=
    "Class".padEnd(colWidth) +
    "Precision".padEnd(12) +
    "Recall".padEnd(12) +
    "F1-Score".padEnd(12) +
    "Support\n";
  report += `${"-".repeat(colWidth + 36 + 7)}\n`;

  let totalSupport = 0;
  let weightedPrec = 0;
  let weightedRec = 0;
  let weightedF1 = 0;
  let macroPrec = 0;
  let macroRec = 0;
  let macroF1 = 0;

  for (let i = 0; i < classes.length; i++) {
    const tp = counts.tp[i] as number;
    const fp = counts.fp[i] as number;
    const fn = counts.fn[i] as number;
    const s = counts.support[i] as number;
    // An undefined score (zero denominator) is reported as 0.
    const p = definedOrZero(precisionFromCounts(tp, fp, fn));
    const r = definedOrZero(recallFromCounts(tp, fp, fn));
    const f1 = definedOrZero(f1FromCounts(tp, fp, fn));

    totalSupport += s;
    weightedPrec += p * s;
    weightedRec += r * s;
    weightedF1 += f1 * s;
    macroPrec += p;
    macroRec += r;
    macroF1 += f1;

    report +=
      String(classes[i]).padEnd(colWidth) +
      p.toFixed(4).padEnd(12) +
      r.toFixed(4).padEnd(12) +
      f1.toFixed(4).padEnd(12) +
      String(s) +
      "\n";
  }

  report += "\n";

  const nClasses = classes.length;
  macroPrec /= nClasses;
  macroRec /= nClasses;
  macroF1 /= nClasses;
  weightedPrec = totalSupport === 0 ? 0 : weightedPrec / totalSupport;
  weightedRec = totalSupport === 0 ? 0 : weightedRec / totalSupport;
  weightedF1 = totalSupport === 0 ? 0 : weightedF1 / totalSupport;

  report +=
    "Accuracy".padEnd(colWidth) +
    "".padEnd(12) +
    "".padEnd(12) +
    acc.toFixed(4).padEnd(12) +
    String(totalSupport) +
    "\n";
  report +=
    "Macro Avg".padEnd(colWidth) +
    macroPrec.toFixed(4).padEnd(12) +
    macroRec.toFixed(4).padEnd(12) +
    macroF1.toFixed(4).padEnd(12) +
    String(totalSupport) +
    "\n";
  report +=
    "Weighted Avg".padEnd(colWidth) +
    weightedPrec.toFixed(4).padEnd(12) +
    weightedRec.toFixed(4).padEnd(12) +
    weightedF1.toFixed(4).padEnd(12) +
    String(totalSupport);

  return report;
}

function argsortAscending(values: Float64Array): Int32Array {
  const n = values.length;
  const out = new Int32Array(n);
  if (n <= SMALL_SORT_THRESHOLD) {
    const idx = Array.from({ length: n }, (_, i) => i);
    idx.sort((a, b) => (values[a] as number) - (values[b] as number));
    out.set(idx);
    return out;
  }
  radixArgsortF64(values, out);
  return out;
}

type Sweep = {
  /** Distinct score thresholds in descending order. */
  thresholds: Float64Array;
  /** Cumulative true positives among samples with score >= threshold. */
  tps: Float64Array;
  /** Cumulative false positives among samples with score >= threshold. */
  fps: Float64Array;
  nPos: number;
  nNeg: number;
};

/**
 * Validate (yTrue, yScore) for the ranking metrics and sweep the score
 * thresholds from high to low. Tied scores form a single step.
 */
function rankedSweep(yTrue: Tensor, yScore: Tensor): Sweep {
  return sweepScores(binaryLabels(yTrue, "yTrue"), denseScores(yScore, "yScore"));
}

/**
 * Sweep the score thresholds from high to low over validated 0/1 labels. With `weights`
 * the running counts and the class totals are weighted sums.
 */
function sweepScores(labels: Uint8Array, scores: Float64Array, weights?: Float64Array): Sweep {
  const n = labels.length;
  const order = argsortAscending(scores);
  const thresholds = new Float64Array(n);
  const tps = new Float64Array(n);
  const fps = new Float64Array(n);
  let steps = 0;
  let tp = 0;
  let fp = 0;
  let pos = n - 1;
  while (pos >= 0) {
    const threshold = scores[order[pos] as number] as number;
    while (pos >= 0 && scores[order[pos] as number] === threshold) {
      const sample = order[pos] as number;
      const w = weights === undefined ? 1 : (weights[sample] as number);
      if (labels[sample] === 1) tp += w;
      else fp += w;
      pos--;
    }
    thresholds[steps] = threshold;
    tps[steps] = tp;
    fps[steps] = fp;
    steps++;
  }
  return {
    thresholds: thresholds.subarray(0, steps),
    tps: tps.subarray(0, steps),
    fps: fps.subarray(0, steps),
    nPos: tp,
    nNeg: fp,
  };
}

function emptyCurve(): [Tensor, Tensor, Tensor] {
  const empty = () => tensor([], { dtype: "float64" });
  return [empty(), empty(), empty()];
}

/**
 * ROC curve data.
 *
 * Computes Receiver Operating Characteristic (ROC) curve for binary classification.
 * The ROC curve shows the trade-off between true positive rate and false positive rate
 * at various threshold settings.
 *
 * **Returns**: [fpr, tpr, thresholds]
 * - fpr: False positive rates
 * - tpr: True positive rates
 * - thresholds: Decision thresholds (in descending order). The first entry is
 *   `Infinity` and belongs to the starting point (0, 0).
 *
 * **Edge Cases**:
 * - Returns empty tensors if yTrue contains only one class
 * - Handles tied scores by grouping them at the same threshold
 * - No intermediate points are dropped (every distinct score is a threshold)
 *
 * @param yTrue - Ground truth binary labels (must be 0 or 1)
 * @param yScore - Target scores (higher score = more likely positive class)
 * @returns Tuple of [fpr, tpr, thresholds] float64 tensors
 *
 * @throws {ShapeError} If yTrue and yScore have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If inputs are string tensors
 * @throws {InvalidParameterError} If yTrue contains non-binary values
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { rocCurve } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([0, 0, 1, 1]);
 * const yScore = tensor([0.1, 0.4, 0.35, 0.8]);
 * const [fpr, tpr, thresholds] = rocCurve(yTrue, yScore);
 * // fpr: [0, 0, 0.5, 0.5, 1], tpr: [0, 0.5, 0.5, 1, 1]
 * // thresholds: [Infinity, 0.8, 0.4, 0.35, 0.1]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function rocCurve(yTrue: Tensor, yScore: Tensor): [Tensor, Tensor, Tensor] {
  assertSameSizeVectors(yTrue, yScore, "yTrue", "yScore");
  if (yTrue.size === 0) return emptyCurve();

  const { thresholds, tps, fps, nPos, nNeg } = rankedSweep(yTrue, yScore);
  if (nPos === 0 || nNeg === 0) return emptyCurve();

  const m = thresholds.length;
  const fpr = new Float64Array(m + 1);
  const tpr = new Float64Array(m + 1);
  const thr = new Float64Array(m + 1);
  thr[0] = Infinity;
  for (let i = 0; i < m; i++) {
    fpr[i + 1] = (fps[i] as number) / nNeg;
    tpr[i + 1] = (tps[i] as number) / nPos;
    thr[i + 1] = thresholds[i] as number;
  }
  return [
    tensor(fpr, { dtype: "float64" }),
    tensor(tpr, { dtype: "float64" }),
    tensor(thr, { dtype: "float64" }),
  ];
}

/**
 * Final area under a threshold sweep with the trapezoidal rule. Tied scores are one
 * step, so the result is the probability that a random positive outranks a random
 * negative, with ties counting one half. 0.5 when a class has no (weighted) samples.
 */
function aucFromSweep(sweep: Sweep): number {
  const { tps, fps, nPos, nNeg } = sweep;
  if (nPos === 0 || nNeg === 0) return 0.5;

  // Trapezoid areas in units of 1/(2 * nPos * nNeg): for unweighted data every term is an
  // exact integer, so the only rounding happens in the final division.
  let doubledArea = 0;
  let prevTp = 0;
  let prevFp = 0;
  for (let i = 0; i < tps.length; i++) {
    const tp = tps[i] as number;
    const fp = fps[i] as number;
    doubledArea += (fp - prevFp) * (tp + prevTp);
    prevTp = tp;
    prevFp = fp;
  }
  return doubledArea / (2 * nPos * nNeg);
}

/**
 * Options for {@link rocAucScore}.
 *
 * @example
 * ```ts
 * import { rocAucScore } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([0, 1, 2, 1]);
 * const proba = tensor([
 *   [0.8, 0.1, 0.1],
 *   [0.2, 0.7, 0.1],
 *   [0.1, 0.2, 0.7],
 *   [0.3, 0.4, 0.3],
 * ]);
 * rocAucScore(yTrue, proba, { multiClass: "ovo", average: "weighted" });
 * ```
 */
export type RocAucOptions = {
  /**
   * How a 2-D score matrix is turned into one number: `"ovr"` (one class against the
   * rest) or `"ovo"` (every pair of classes, Hand and Till 2001). Used only when
   * `yScore` has two or more columns. Default: `"ovr"` (scikit-learn requires you to
   * choose; Deepbox picks the common one).
   */
  readonly multiClass?: "ovr" | "ovo";
  /**
   * Averaging over the classes (or class pairs): `"macro"` (default), `"weighted"` (by
   * class prevalence), `"micro"` (`"ovr"` only, all one-vs-rest problems pooled) or
   * `null` for the per-class AUCs (`"ovr"` only). Ignored for 1-D binary scores, where
   * `null` returns a one-element array.
   */
  readonly average?: "micro" | "macro" | "weighted" | null;
  /**
   * Class labels in the order of the columns of a 2-D `yScore`, in ascending order.
   * Defaults to the sorted distinct labels of yTrue. Ignored for 1-D scores.
   */
  readonly labels?: ReadonlyArray<number | string | bigint>;
  /**
   * One weight per sample. Not supported with `multiClass: "ovo"`, like scikit-learn.
   */
  readonly sampleWeight?: SampleWeightInput;
};

const ROC_AUC_AVERAGES: readonly (string | null)[] = ["micro", "macro", "weighted", null];

/**
 * Index every value on the sorted class list: `fixed` when given (a value outside it is an
 * error), otherwise the sorted distinct values.
 */
function encodeClasses(
  values: DenseLabels,
  fixed: readonly Label[] | undefined,
  name: string
): { classes: Label[]; idx: Int32Array } {
  const n = values.length;
  const idx = new Int32Array(n);
  if (fixed !== undefined) {
    const index = new Map<Label, number>();
    for (let i = 0; i < fixed.length; i++) index.set(fixed[i] as Label, i);
    for (let i = 0; i < n; i++) {
      const found = index.get(values[i] as Label);
      if (found === undefined) {
        throw new InvalidParameterError(
          `${name} contains the label ${String(values[i])}, which is not in labels`,
          name,
          values[i]
        );
      }
      idx[i] = found;
    }
    return { classes: [...fixed], idx };
  }
  const distinct = [...new Set<Label>(Array.from(values as ArrayLike<Label>))].sort(compareLabels);
  const index = new Map<Label, number>();
  for (let i = 0; i < distinct.length; i++) index.set(distinct[i] as Label, i);
  for (let i = 0; i < n; i++) idx[i] = index.get(values[i] as Label) as number;
  return { classes: distinct, idx };
}

/** Normalize a `labels` option that indexes the columns of a score matrix. */
function columnLabels(
  labels: ReadonlyArray<number | string | bigint>,
  kind: LabelKind,
  columns: number
): Label[] {
  const classes = normalizeLabelList(labels, kind);
  for (let i = 1; i < classes.length; i++) {
    if (compareLabels(classes[i - 1] as Label, classes[i] as Label) >= 0) {
      throw new InvalidParameterError(
        "labels must be in ascending order; the columns of the score matrix follow the sorted labels",
        "labels",
        labels
      );
    }
  }
  if (classes.length !== columns) {
    throw new InvalidParameterError(
      `labels has ${classes.length} entries but the score matrix has ${columns} columns`,
      "labels",
      labels
    );
  }
  return classes;
}

function multiclassRocAuc(
  yTrue: Tensor,
  yScore: Tensor,
  options: RocAucOptions,
  average: "micro" | "macro" | "weighted" | null,
  multiClass: "ovr" | "ovo"
): number | number[] {
  assertVectorLike(yTrue, "yTrue");
  const n = yScore.shape[0] as number;
  const k = yScore.shape[1] as number;
  if (yTrue.size !== n) {
    throw new ShapeError(
      `yTrue (size ${yTrue.size}) and yScore (${n} rows) must have the same number of samples`
    );
  }
  const weights = readSampleWeight(options.sampleWeight, n);
  if (multiClass === "ovo") {
    if (weights !== undefined) {
      throw new InvalidParameterError(
        "sampleWeight is not supported for multiClass 'ovo'",
        "sampleWeight",
        undefined
      );
    }
    if (average === null || average === "micro") {
      throw new InvalidParameterError(
        "average must be 'macro' or 'weighted' for multiClass 'ovo'",
        "average",
        average
      );
    }
  }
  if (n === 0) return average === null ? new Array<number>(k).fill(0.5) : 0.5;

  const trueLabels = denseLabels(yTrue, "yTrue");
  const scores = denseScores(yScore, "yScore");
  for (let i = 0; i < n; i++) {
    let rowSum = 0;
    for (let c = 0; c < k; c++) rowSum += scores[i * k + c] as number;
    if (Math.abs(1 - rowSum) > 1e-8 + 1e-5 * Math.abs(rowSum)) {
      throw new InvalidParameterError(
        "Target scores need to be probabilities for multiclass roc auc, i.e. each row of yScore must sum to 1",
        "yScore",
        rowSum
      );
    }
  }

  const fixed =
    options.labels === undefined ? undefined : columnLabels(options.labels, labelKind(yTrue), k);
  const { classes, idx } = encodeClasses(trueLabels, fixed, "yTrue");
  if (classes.length !== k) {
    throw new InvalidParameterError(
      `yTrue has ${classes.length} distinct classes but yScore has ${k} columns; pass labels to name the columns`,
      "yScore",
      k
    );
  }

  if (multiClass === "ovo") return ovoRocAuc(idx, scores, n, k, average === "weighted");

  if (average === "micro") {
    const flatLabels = new Uint8Array(n * k);
    const flatWeights = weights === undefined ? undefined : new Float64Array(n * k);
    for (let i = 0; i < n; i++) {
      flatLabels[i * k + (idx[i] as number)] = 1;
      if (flatWeights !== undefined && weights !== undefined) {
        flatWeights.fill(weights[i] as number, i * k, (i + 1) * k);
      }
    }
    return aucFromSweep(sweepScores(flatLabels, scores, flatWeights));
  }

  const perClass = new Array<number>(k);
  const prevalence = new Array<number>(k);
  const column = new Float64Array(n);
  const positive = new Uint8Array(n);
  for (let c = 0; c < k; c++) {
    let positives = 0;
    for (let i = 0; i < n; i++) {
      const isClass = idx[i] === c ? 1 : 0;
      positive[i] = isClass;
      positives += isClass;
      column[i] = scores[i * k + c] as number;
    }
    if (positives === 0 || positives === n) {
      throw new InvalidParameterError(
        `ROC AUC is not defined for the class ${String(classes[c])}: yTrue needs samples inside and outside of it`,
        "yTrue",
        classes[c]
      );
    }
    const sweep = sweepScores(positive, column, weights);
    perClass[c] = aucFromSweep(sweep);
    prevalence[c] = sweep.nPos;
  }
  if (average === null) return perClass;
  if (average === "macro") {
    let sum = 0;
    for (let c = 0; c < k; c++) sum += perClass[c] as number;
    return sum / k;
  }
  let weightedSum = 0;
  let totalPrevalence = 0;
  for (let c = 0; c < k; c++) {
    const w = prevalence[c] as number;
    if (w === 0) continue;
    weightedSum += (perClass[c] as number) * w;
    totalPrevalence += w;
  }
  return totalPrevalence === 0 ? 0 : weightedSum / totalPrevalence;
}

/** One-vs-one multiclass AUC of Hand and Till (2001) over the classes present in yTrue. */
function ovoRocAuc(
  idx: Int32Array,
  scores: Float64Array,
  n: number,
  k: number,
  weighted: boolean
): number {
  const members: number[][] = Array.from({ length: k }, () => []);
  for (let i = 0; i < n; i++) (members[idx[i] as number] as number[]).push(i);
  const present: number[] = [];
  for (let c = 0; c < k; c++) if ((members[c] as number[]).length > 0) present.push(c);
  if (present.length < 2) {
    throw new InvalidParameterError(
      "multiClass 'ovo' needs at least two classes in yTrue",
      "yTrue",
      present.length
    );
  }

  let sum = 0;
  let totalPrevalence = 0;
  for (let x = 0; x < present.length; x++) {
    for (let y = x + 1; y < present.length; y++) {
      const a = present[x] as number;
      const b = present[y] as number;
      const rowsA = members[a] as number[];
      const rowsB = members[b] as number[];
      const rows = rowsA.concat(rowsB);
      const m = rows.length;
      const labelsA = new Uint8Array(m);
      const labelsB = new Uint8Array(m);
      const scoresA = new Float64Array(m);
      const scoresB = new Float64Array(m);
      for (let r = 0; r < m; r++) {
        const row = rows[r] as number;
        if (r < rowsA.length) labelsA[r] = 1;
        else labelsB[r] = 1;
        scoresA[r] = scores[row * k + a] as number;
        scoresB[r] = scores[row * k + b] as number;
      }
      const pair =
        (aucFromSweep(sweepScores(labelsA, scoresA)) +
          aucFromSweep(sweepScores(labelsB, scoresB))) /
        2;
      const prevalence = weighted ? m / n : 1;
      sum += pair * prevalence;
      totalPrevalence += prevalence;
    }
  }
  return sum / totalPrevalence;
}

/**
 * Area Under ROC Curve (AUC-ROC).
 *
 * Computes the Area Under the Receiver Operating Characteristic Curve with the
 * trapezoidal rule. Tied scores are handled as one step, so the result equals the
 * probability that a random positive is ranked above a random negative, counting
 * ties as one half.
 *
 * **Binary**: `yTrue` holds 0/1 labels and `yScore` is a 1-D score vector (or a column
 * vector).
 *
 * **Multiclass**: `yScore` is a matrix of shape [n_samples, n_classes] whose columns
 * follow the sorted class labels and whose rows sum to 1 (probabilities), as in
 * scikit-learn. `options.multiClass` picks `"ovr"` (one class against the rest, the
 * default) or `"ovo"` (every pair of classes); `options.average` is `"macro"`
 * (default), `"weighted"`, `"micro"` (`"ovr"` only) or `null` (per-class AUCs, `"ovr"`
 * only). Labels may be numbers, strings or int64 values. A class that has no sample, or
 * all the samples, in `yTrue` has no ROC curve and throws in `"ovr"` mode.
 *
 * **Range**: [0, 1], where 1 is perfect and 0.5 is random.
 *
 * **Time Complexity**: O(n log n) binary, O(k n log n) multiclass "ovr"
 * **Space Complexity**: O(n)
 *
 * @param yTrue - Ground truth labels: 0/1 for a 1-D score, any class labels for a score matrix
 * @param yScore - Target scores (higher score = more likely positive class), 1-D or a
 *   [n_samples, n_classes] probability matrix
 * @param options - Optional `multiClass`, `average`, `labels` and `sampleWeight`
 *   (see {@link RocAucOptions})
 * @returns AUC score in range [0, 1] (an array with `average: null`). Returns 0.5 for empty
 *   input or when a binary yTrue has only one class (the ROC curve is undefined there).
 *
 * @throws {ShapeError} If yTrue and yScore have different sizes, are not 1D/column vectors
 *   (or a 2-D score matrix), or `sampleWeight` has the wrong length
 * @throws {DTypeError} If inputs are string tensors (except multiclass yTrue labels)
 * @throws {InvalidParameterError} If binary yTrue contains non-binary values, an option is
 *   invalid, the score rows do not sum to 1, the number of classes does not match the
 *   columns, or a class has no ROC curve
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { rocAucScore } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([0, 0, 1, 1]);
 * const yScore = tensor([0.1, 0.4, 0.35, 0.8]);
 * const auc = rocAucScore(yTrue, yScore); // 0.75
 *
 * // Multiclass: one probability column per class
 * const labels = tensor([0, 1, 2, 2, 1, 0]);
 * const proba = tensor([
 *   [0.7, 0.2, 0.1],
 *   [0.2, 0.6, 0.2],
 *   [0.1, 0.3, 0.6],
 *   [0.3, 0.3, 0.4],
 *   [0.5, 0.4, 0.1],
 *   [0.2, 0.5, 0.3],
 * ]);
 * rocAucScore(labels, proba, { multiClass: "ovr" }); // 0.8541666666666666
 * rocAucScore(labels, proba, { multiClass: "ovr", average: null }); // [0.6875, 0.875, 1]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function rocAucScore(
  yTrue: Tensor,
  yScore: Tensor,
  options: RocAucOptions & { readonly average: null }
): number[];
export function rocAucScore(
  yTrue: Tensor,
  yScore: Tensor,
  options?: RocAucOptions & { readonly average?: "micro" | "macro" | "weighted" }
): number;
export function rocAucScore(
  yTrue: Tensor,
  yScore: Tensor,
  options?: RocAucOptions
): number | number[];
export function rocAucScore(
  yTrue: Tensor,
  yScore: Tensor,
  options: RocAucOptions = {}
): number | number[] {
  const average = options.average === undefined ? "macro" : options.average;
  if (!ROC_AUC_AVERAGES.includes(average)) {
    throw new InvalidParameterError(
      `Invalid average parameter: ${String(average)}. Must be one of: 'micro', 'macro', 'weighted', or null`,
      "average",
      average
    );
  }
  const multiClass = options.multiClass ?? "ovr";
  if (multiClass !== "ovr" && multiClass !== "ovo") {
    throw new InvalidParameterError(
      `Invalid multiClass parameter: ${String(multiClass)}. Must be 'ovr' or 'ovo'`,
      "multiClass",
      multiClass
    );
  }
  if (yScore.ndim > 2) {
    throw new ShapeError(`yScore must be 1D or 2D; got shape [${yScore.shape.join(", ")}]`);
  }
  if (yScore.ndim === 2 && (yScore.shape[1] ?? 0) >= 2) {
    return multiclassRocAuc(yTrue, yScore, options, average, multiClass);
  }

  assertSameSizeVectors(yTrue, yScore, "yTrue", "yScore");
  const weights = readSampleWeight(options.sampleWeight, yTrue.size);
  const auc =
    yTrue.size === 0
      ? 0.5
      : aucFromSweep(
          sweepScores(binaryLabels(yTrue, "yTrue"), denseScores(yScore, "yScore"), weights)
        );
  return average === null ? [auc] : auc;
}

/**
 * Precision-Recall curve.
 *
 * Computes precision-recall pairs for different probability thresholds.
 * Useful for evaluating classifiers on imbalanced datasets where ROC curves
 * may be overly optimistic.
 *
 * **Returns**: [precision, recall, thresholds] as a tuple of tensors
 * - precision: Precision values at each threshold
 * - recall: Recall values at each threshold
 * - thresholds: Decision thresholds (in descending order). The first entry is
 *   `Infinity` and belongs to the starting point (precision 1, recall 0).
 *
 * Unlike scikit-learn, the three arrays have the same length and are ordered by
 * descending threshold (increasing recall).
 *
 * **Time Complexity**: O(n log n) due to sorting
 * **Space Complexity**: O(n)
 *
 * @param yTrue - Ground truth binary labels (0 or 1)
 * @param yScore - Target scores (higher score = more likely positive class)
 * @returns Tuple of [precision, recall, thresholds] float64 tensors; empty tensors when
 *   yTrue has no positive sample
 *
 * @throws {ShapeError} If yTrue and yScore have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If inputs are string tensors
 * @throws {InvalidParameterError} If yTrue contains non-binary values
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { precisionRecallCurve } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([0, 0, 1, 1]);
 * const yScore = tensor([0.1, 0.4, 0.35, 0.8]);
 * const [prec, rec, thresh] = precisionRecallCurve(yTrue, yScore);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function precisionRecallCurve(yTrue: Tensor, yScore: Tensor): [Tensor, Tensor, Tensor] {
  assertSameSizeVectors(yTrue, yScore, "yTrue", "yScore");
  if (yTrue.size === 0) return emptyCurve();

  const { thresholds, tps, fps, nPos } = rankedSweep(yTrue, yScore);
  if (nPos === 0) return emptyCurve();

  const m = thresholds.length;
  const prec = new Float64Array(m + 1);
  const rec = new Float64Array(m + 1);
  const thr = new Float64Array(m + 1);
  prec[0] = 1;
  thr[0] = Infinity;
  for (let i = 0; i < m; i++) {
    const tp = tps[i] as number;
    prec[i + 1] = tp / (tp + (fps[i] as number));
    rec[i + 1] = tp / nPos;
    thr[i + 1] = thresholds[i] as number;
  }
  return [
    tensor(prec, { dtype: "float64" }),
    tensor(rec, { dtype: "float64" }),
    tensor(thr, { dtype: "float64" }),
  ];
}

/**
 * Average precision score.
 *
 * Computes the average precision (AP) from prediction scores. AP summarizes
 * a precision-recall curve as the weighted mean of precisions achieved at
 * each threshold, with the increase in recall from the previous threshold
 * used as the weight.
 *
 * **Range**: [0, 1], where 1 is perfect.
 *
 * **Time Complexity**: O(n log n) due to sorting
 * **Space Complexity**: O(n)
 *
 * @param yTrue - Ground truth binary labels (0 or 1)
 * @param yScore - Target scores (higher score = more likely positive class)
 * @returns Average precision score in range [0, 1]. Returns 0 for empty input or when yTrue
 *   has no positive sample.
 *
 * @throws {ShapeError} If yTrue and yScore have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If inputs are string tensors
 * @throws {InvalidParameterError} If yTrue contains non-binary values
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { averagePrecisionScore } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([0, 0, 1, 1]);
 * const yScore = tensor([0.1, 0.4, 0.35, 0.8]);
 * const ap = averagePrecisionScore(yTrue, yScore); // 0.8333...
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function averagePrecisionScore(yTrue: Tensor, yScore: Tensor): number {
  assertSameSizeVectors(yTrue, yScore, "yTrue", "yScore");
  if (yTrue.size === 0) return 0;

  const { tps, fps, nPos } = rankedSweep(yTrue, yScore);
  if (nPos === 0) return 0;

  let weighted = 0;
  let prevTp = 0;
  for (let i = 0; i < tps.length; i++) {
    const tp = tps[i] as number;
    const gained = tp - prevTp;
    if (gained > 0) weighted += (gained * tp) / (tp + (fps[i] as number));
    prevTp = tp;
  }
  return weighted / nPos;
}

/** Machine epsilon that scikit-learn uses to clip probabilities for each dtype. */
function clipEpsilon(dtype: Tensor["dtype"]): number {
  switch (dtype) {
    case "float32":
      return 2 ** -23;
    case "float16":
      return 2 ** -10;
    case "bfloat16":
      return 2 ** -7;
    default:
      return Number.EPSILON;
  }
}

/**
 * Options for {@link logLoss}.
 *
 * @example
 * ```ts
 * import { logLoss } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * // yTrue holds only classes 0 and 2, so `labels` names the three probability columns.
 * const yTrue = tensor([0, 2, 2]);
 * const proba = tensor([
 *   [0.6, 0.3, 0.1],
 *   [0.1, 0.2, 0.7],
 *   [0.2, 0.2, 0.6],
 * ]);
 * logLoss(yTrue, proba, { labels: [0, 1, 2], normalize: false });
 * ```
 */
export type LogLossOptions = WeightedMetricOptions & {
  /**
   * Class labels in the order of the columns of a 2-D `yPred`, in ascending order (for a
   * 1-D `yPred`: exactly two labels, the second is the positive class). Defaults to the
   * sorted distinct labels of yTrue (0 and 1 for a 1-D `yPred`). Required when yTrue does
   * not contain every class.
   */
  readonly labels?: ReadonlyArray<number | string | bigint>;
  /** Return the mean loss (default) or, with `false`, the sum of the (weighted) losses. */
  readonly normalize?: boolean;
};

/**
 * Log loss (logistic loss, cross-entropy loss).
 *
 * Measures the performance of a classification model where the prediction is a probability
 * value between 0 and 1. Lower log loss indicates better predictions.
 *
 * **Binary**: `yPred` is a 1-D vector of positive-class probabilities and `yTrue` holds 0/1
 * labels. Loss: -1/n * Σ(y * log(p) + (1 - y) * log(1 - p)).
 *
 * **Multiclass**: `yPred` is a matrix of shape [n_samples, n_classes] of class
 * probabilities whose columns follow the sorted class labels (a two-column matrix is the
 * binary case). Loss: -1/n * Σ log(p[i, y_i]). Rows are not renormalized. `options.labels`
 * names the columns when yTrue does not contain every class. Labels may be numbers,
 * strings or int64 values.
 *
 * **Edge Cases**:
 * - Predictions are clipped to [eps, 1 - eps] to avoid log(0), where eps is the machine
 *   epsilon of yPred's dtype (2.2e-16 for float64, 1.2e-7 for float32), as in scikit-learn
 * - Returns 0 for empty inputs
 *
 * @param yTrue - Ground truth labels
 * @param yPred - Predicted probabilities: 1-D positive-class probabilities, or a
 *   [n_samples, n_classes] matrix (each value in [0, 1])
 * @param options - Optional `labels`, `sampleWeight` and `normalize` (see {@link LogLossOptions})
 * @returns Log loss value (lower is better, 0 is perfect)
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes, are not 1D/column vectors
 *   (or a 2-D probability matrix), or `sampleWeight` has the wrong length
 * @throws {DTypeError} If yPred is a string tensor, or yTrue is a string tensor for a 1-D yPred
 * @throws {InvalidParameterError} If a 1-D yTrue is not binary or yPred is outside [0, 1], the
 *   number of classes does not match the columns of yPred, `labels` is invalid, or
 *   `sampleWeight` sums to zero
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { logLoss } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([0, 1, 1, 0]);
 * const yPred = tensor([0.1, 0.9, 0.8, 0.2], { dtype: "float64" });
 * const loss = logLoss(yTrue, yPred); // 0.164252...
 *
 * // Multiclass probabilities, one column per class
 * const labels = tensor([0, 1, 2]);
 * const proba = tensor(
 *   [
 *     [0.7, 0.2, 0.1],
 *     [0.2, 0.6, 0.2],
 *     [0.1, 0.3, 0.6],
 *   ],
 *   { dtype: "float64" }
 * );
 * logLoss(labels, proba); // 0.4594420638235713
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function logLoss(yTrue: Tensor, yPred: Tensor, options: LogLossOptions = {}): number {
  if (yPred.ndim > 2) {
    throw new ShapeError(`yPred must be 1D or 2D; got shape [${yPred.shape.join(", ")}]`);
  }
  const normalize = options.normalize ?? true;
  if (typeof normalize !== "boolean") {
    throw new InvalidParameterError("normalize must be a boolean", "normalize", normalize);
  }
  const columns = yPred.ndim === 2 ? (yPred.shape[1] as number) : 1;
  const multiclass = columns >= 2;
  if (multiclass) {
    assertVectorLike(yTrue, "yTrue");
    if (yTrue.size !== (yPred.shape[0] as number)) {
      throw new ShapeError(
        `yTrue (size ${yTrue.size}) and yPred (${yPred.shape[0]} rows) must have the same number of samples`
      );
    }
  } else {
    assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");
  }

  const n = yTrue.size;
  const weights = readSampleWeight(options.sampleWeight, n);
  if (n === 0) return 0;

  const probs = denseScores(yPred, "yPred");
  const eps = clipEpsilon(yPred.dtype);
  const upper = 1 - eps;
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i] as number;
    if (p < 0 || p > 1) {
      throw new InvalidParameterError(
        `yPred must contain probabilities in range [0, 1], found ${String(p)} at index ${i}`,
        "yPred",
        p
      );
    }
  }

  // Column of each sample's true class (multiclass), or the 0/1 label (binary).
  let trueClass: ArrayLike<number>;
  if (multiclass) {
    const fixed =
      options.labels === undefined
        ? undefined
        : columnLabels(options.labels, labelKind(yTrue), columns);
    const encoded = encodeClasses(denseLabels(yTrue, "yTrue"), fixed, "yTrue");
    if (encoded.classes.length !== columns) {
      throw new InvalidParameterError(
        `yTrue has ${encoded.classes.length} distinct classes but yPred has ${columns} columns; pass labels to name the columns`,
        "yPred",
        columns
      );
    }
    trueClass = encoded.idx;
  } else if (options.labels === undefined) {
    trueClass = binaryLabels(yTrue, "yTrue");
  } else {
    const classes = columnLabels(options.labels, labelKind(yTrue), 2);
    trueClass = encodeClasses(denseLabels(yTrue, "yTrue"), classes, "yTrue").idx;
  }

  const terms = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let term: number;
    if (multiclass) {
      const p = probs[i * columns + (trueClass[i] as number)] as number;
      term = -Math.log(p < eps ? eps : p > upper ? upper : p);
    } else {
      const p = probs[i] as number;
      const clipped = p < eps ? eps : p > upper ? upper : p;
      term = trueClass[i] === 1 ? -Math.log(clipped) : -Math.log1p(-clipped);
    }
    terms[i] = weights === undefined ? term : term * (weights[i] as number);
  }

  const sum = compensatedSum(terms);
  if (!normalize) return sum;
  return sum / (weights === undefined ? n : nonZeroWeightSum(weights));
}

/**
 * Hamming loss.
 *
 * Computes the fraction of labels that are incorrectly predicted.
 * For single-label classification this equals 1 - accuracy.
 *
 * **Formula**: hamming_loss = (incorrect predictions) / (total predictions)
 *
 * **Range**: [0, 1], where 0 is perfect.
 *
 * **Time Complexity**: O(n)
 * **Space Complexity**: O(n)
 *
 * @param yTrue - Ground truth target values
 * @param yPred - Estimated targets as returned by a classifier
 * @returns Hamming loss in range [0, 1]. Returns 0 for empty input.
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If yTrue and yPred use incompatible label types
 * @throws {DataValidationError} If numeric labels contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { hammingLoss } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([0, 1, 1, 0, 1]);
 * const yPred = tensor([0, 1, 0, 0, 1]);
 * const loss = hammingLoss(yTrue, yPred); // 0.2
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function hammingLoss(yTrue: Tensor, yPred: Tensor): number {
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");
  if (yTrue.size === 0) return 0;
  return (yTrue.size - countMatches(yTrue, yPred)) / yTrue.size;
}

/**
 * Jaccard similarity score (Intersection over Union).
 *
 * Computes the Jaccard similarity coefficient between two label sets.
 * Also known as the Jaccard index or Intersection over Union (IoU).
 *
 * **Formula**: jaccard = TP / (TP + FP + FN)
 *
 * Averaging works as for {@link f1Score}: 'binary' scores the positive class 1,
 * 'micro' pools the counts, 'macro' and 'weighted' average the per-class scores and
 * null returns one score per class. When `average` is omitted, 'binary' is used for at
 * most two distinct labels and 'weighted' otherwise.
 *
 * **Edge Cases**:
 * - If a score has no positives in either vector (TP + FP + FN = 0), it is `zeroDivision`,
 *   which defaults to 1 here: two empty sets are identical. Empty input also returns that
 *   value. (scikit-learn returns 0 in this case unless `zero_division` is set.)
 *
 * **Range**: [0, 1], where 1 is perfect.
 *
 * **Time Complexity**: O(n) for binary, O(n + k) for multiclass
 * **Space Complexity**: O(n + k)
 *
 * @param yTrue - Ground truth labels
 * @param yPred - Predicted labels
 * @param average - Averaging strategy ('binary', 'micro', 'macro', 'weighted' or null), or an
 *   options object {@link AveragedMetricOptions}
 *   (`{ average, labels, zeroDivision, sampleWeight }`)
 * @returns Jaccard score in range [0, 1], or one score per class with `average: null`
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If yTrue and yPred use incompatible label types
 * @throws {DataValidationError} If numeric labels contain NaN or infinite values
 * @throws {InvalidParameterError} If average is invalid, or 'binary' is used with labels other
 *   than 0/1 (including string labels)
 *
 * @example
 * ```ts
 * import { jaccardScore } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([0, 1, 1, 0, 1]);
 * const yPred = tensor([0, 1, 0, 0, 1]);
 * const score = jaccardScore(yTrue, yPred); // 0.667
 *
 * const multiTrue = tensor([0, 1, 2, 2, 1, 0]);
 * const multiPred = tensor([0, 2, 2, 2, 1, 1]);
 * jaccardScore(multiTrue, multiPred, "macro"); // 0.5
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function jaccardScore(yTrue: Tensor, yPred: Tensor): number;
export function jaccardScore(yTrue: Tensor, yPred: Tensor, average: Average): number;
export function jaccardScore(yTrue: Tensor, yPred: Tensor, average: null): number[];
export function jaccardScore(
  yTrue: Tensor,
  yPred: Tensor,
  options: AveragedMetricOptions & { readonly average: null }
): number[];
export function jaccardScore(
  yTrue: Tensor,
  yPred: Tensor,
  options: AveragedMetricOptions & { readonly average?: Average }
): number;
export function jaccardScore(
  yTrue: Tensor,
  yPred: Tensor,
  average?: Average | null | AveragedMetricOptions
): number | number[];
export function jaccardScore(
  yTrue: Tensor,
  yPred: Tensor,
  average?: Average | null | AveragedMetricOptions
): number | number[] {
  return averagedScore(yTrue, yPred, average, jaccardFromCounts, 1);
}

/**
 * Matthews correlation coefficient (MCC).
 *
 * Computes the Matthews correlation coefficient, a balanced measure that
 * can be used even if the classes are of very different sizes. MCC is
 * considered one of the best metrics for classification.
 *
 * **Binary formula**: MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
 *
 * **Multiclass**: the generalization from the confusion matrix (Gorodkin 2004),
 * identical to scikit-learn's `matthews_corrcoef`:
 * MCC = (c*s - Σ t_k p_k) / sqrt((s² - Σ p_k²)(s² - Σ t_k²)) with c the number of correct
 * predictions, s the number of samples and t_k, p_k the true and predicted counts of
 * class k. Labels may be numbers, strings or int64 values.
 *
 * Returns 0 when the coefficient is undefined (a denominator term is zero, for example
 * when only one class occurs).
 *
 * **Range**: [-1, 1], where 1 is perfect, 0 is random, -1 is inverse.
 *
 * **Time Complexity**: O(n + k)
 * **Space Complexity**: O(n + k)
 *
 * @param yTrue - Ground truth labels
 * @param yPred - Predicted labels (same label type as yTrue)
 * @param options - Optional `sampleWeight`, one weight per sample
 * @returns MCC score in range [-1, 1]
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors, or
 *   `sampleWeight` has the wrong length
 * @throws {DTypeError} If yTrue and yPred use incompatible label types
 * @throws {DataValidationError} If numeric labels contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { matthewsCorrcoef } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([0, 1, 1, 0, 1]);
 * const yPred = tensor([0, 1, 0, 0, 1]);
 * const mcc = matthewsCorrcoef(yTrue, yPred); // ~0.667
 *
 * const multiTrue = tensor([0, 1, 2, 2, 1, 0]);
 * const multiPred = tensor([0, 1, 2, 1, 1, 2]);
 * matthewsCorrcoef(multiTrue, multiPred); // 0.5222329678670935
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function matthewsCorrcoef(
  yTrue: Tensor,
  yPred: Tensor,
  options: WeightedMetricOptions = {}
): number {
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");
  const weights = readSampleWeight(options.sampleWeight, yTrue.size);
  if (yTrue.size === 0) return 0;

  const { classes, trueIdx, predIdx } = encodeLabels(yTrue, yPred);
  const k = classes.length;
  const matrix = new Float64Array(k * k);
  for (let i = 0; i < trueIdx.length; i++) {
    const cell = (trueIdx[i] as number) * k + (predIdx[i] as number);
    matrix[cell] = (matrix[cell] as number) + (weights === undefined ? 1 : (weights[i] as number));
  }

  if (k === 2) {
    const tn = matrix[0] as number;
    const fp = matrix[1] as number;
    const fn = matrix[2] as number;
    const tp = matrix[3] as number;
    const denominator = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
    if (denominator === 0) return 0;
    return Math.max(-1, Math.min(1, (tp * tn - fp * fn) / denominator));
  }

  const trueCount = new Float64Array(k);
  const predCount = new Float64Array(k);
  let correct = 0;
  let total = 0;
  for (let a = 0; a < k; a++) {
    for (let b = 0; b < k; b++) {
      const v = matrix[a * k + b] as number;
      trueCount[a] = (trueCount[a] as number) + v;
      predCount[b] = (predCount[b] as number) + v;
      total += v;
      if (a === b) correct += v;
    }
  }
  let sumTrueTimesPred = 0;
  let sumTrueSquares = 0;
  let sumPredSquares = 0;
  for (let c = 0; c < k; c++) {
    const t = trueCount[c] as number;
    const p = predCount[c] as number;
    sumTrueTimesPred += t * p;
    sumTrueSquares += t * t;
    sumPredSquares += p * p;
  }
  const covTruePred = correct * total - sumTrueTimesPred;
  const covPredPred = total * total - sumPredSquares;
  const covTrueTrue = total * total - sumTrueSquares;
  const denominator = Math.sqrt(covTrueTrue * covPredPred);
  if (denominator === 0 || Number.isNaN(denominator)) return 0;
  return Math.max(-1, Math.min(1, covTruePred / denominator));
}

/**
 * Cohen's kappa score.
 *
 * Computes Cohen's kappa, a statistic that measures inter-annotator agreement.
 * Unlike plain percent agreement it corrects for the agreement expected by chance.
 *
 * **Formula**: kappa = (p_o - p_e) / (1 - p_e)
 * - p_o: observed agreement
 * - p_e: expected agreement by chance
 *
 * With `weights` the disagreement between classes i and j (in ascending label order)
 * is weighted by |i - j| (`"linear"`) or (i - j)² (`"quadratic"`), which suits ordinal labels.
 *
 * **Range**: [-1, 1], where 1 is perfect agreement, 0 is chance, <0 is worse than chance.
 * When both vectors hold the same single class the score is 1.
 *
 * **Time Complexity**: O(n) unweighted, O(n + k²) weighted
 * **Space Complexity**: O(n + k) unweighted, O(n + k²) weighted
 *
 * @param yTrue - Ground truth labels (numeric, int64 or string)
 * @param yPred - Predicted labels (same label type as yTrue)
 * @param weights - Optional disagreement weighting: `"linear"`, `"quadratic"` or null (default)
 * @returns Kappa score in range [-1, 1]
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If yTrue and yPred use incompatible label types
 * @throws {DataValidationError} If numeric labels contain NaN or infinite values
 * @throws {InvalidParameterError} If weights is not 'linear', 'quadratic' or null
 *
 * @example
 * ```ts
 * import { cohenKappaScore } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([0, 1, 1, 0, 1]);
 * const yPred = tensor([0, 1, 0, 0, 1]);
 * const kappa = cohenKappaScore(yTrue, yPred); // ~0.615
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function cohenKappaScore(
  yTrue: Tensor,
  yPred: Tensor,
  weights: "linear" | "quadratic" | null = null
): number {
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");
  if (weights !== null && weights !== "linear" && weights !== "quadratic") {
    throw new InvalidParameterError(
      `Invalid weights parameter: ${String(weights)}. Must be one of: 'linear', 'quadratic', or null`,
      "weights",
      weights
    );
  }

  const n = yTrue.size;
  if (n === 0) return 0;

  const { classes, trueIdx, predIdx } = encodeLabels(yTrue, yPred);
  const k = classes.length;
  const trueCount = new Float64Array(k);
  const predCount = new Float64Array(k);

  if (weights === null) {
    let agree = 0;
    for (let i = 0; i < n; i++) {
      const a = trueIdx[i] as number;
      const b = predIdx[i] as number;
      if (a === b) agree++;
      trueCount[a] = (trueCount[a] as number) + 1;
      predCount[b] = (predCount[b] as number) + 1;
    }
    const po = agree / n;
    let pe = 0;
    for (let c = 0; c < k; c++) {
      pe += ((trueCount[c] as number) / n) * ((predCount[c] as number) / n);
    }
    const denom = 1 - pe;
    if (denom === 0) return po === 1 ? 1 : 0;
    return (po - pe) / denom;
  }

  const observed = new Float64Array(k * k);
  for (let i = 0; i < n; i++) {
    const a = trueIdx[i] as number;
    const b = predIdx[i] as number;
    observed[a * k + b] = (observed[a * k + b] as number) + 1;
    trueCount[a] = (trueCount[a] as number) + 1;
    predCount[b] = (predCount[b] as number) + 1;
  }
  let weightedObserved = 0;
  let weightedExpected = 0;
  for (let a = 0; a < k; a++) {
    for (let b = 0; b < k; b++) {
      const distance = Math.abs(a - b);
      const w = weights === "linear" ? distance : distance * distance;
      weightedObserved += w * (observed[a * k + b] as number);
      weightedExpected += (w * (trueCount[a] as number) * (predCount[b] as number)) / n;
    }
  }
  if (weightedExpected === 0) return weightedObserved === 0 ? 1 : 0;
  return 1 - weightedObserved / weightedExpected;
}

/**
 * Balanced accuracy score.
 *
 * Computes the balanced accuracy, which is the macro-averaged recall over the
 * classes that occur in yTrue. It is useful for imbalanced datasets where regular
 * accuracy can be misleading.
 *
 * **Formula**: balanced_accuracy = (1/n_classes) * Σ(recall_per_class)
 *
 * **Range**: [0, 1], where 1 is perfect.
 *
 * **Time Complexity**: O(n + k)
 * **Space Complexity**: O(n + k)
 *
 * @param yTrue - Ground truth labels (numeric, int64 or string)
 * @param yPred - Predicted labels (same label type as yTrue)
 * @returns Balanced accuracy score in range [0, 1]. Returns 0 for empty input.
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If yTrue and yPred use incompatible label types
 * @throws {DataValidationError} If numeric labels contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { balancedAccuracyScore } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([0, 0, 0, 0, 1]); // Imbalanced
 * const yPred = tensor([0, 0, 0, 0, 0]); // Predicts all 0
 * const bacc = balancedAccuracyScore(yTrue, yPred); // 0.5 (not 0.8!)
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function balancedAccuracyScore(yTrue: Tensor, yPred: Tensor): number {
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");
  if (yTrue.size === 0) return 0;

  const { classes, tp, support } = classCounts(yTrue, yPred);
  let sumRecall = 0;
  let classCount = 0;
  for (let i = 0; i < classes.length; i++) {
    const s = support[i] as number;
    if (s === 0) continue;
    sumRecall += (tp[i] as number) / s;
    classCount++;
  }
  return classCount === 0 ? 0 : sumRecall / classCount;
}
