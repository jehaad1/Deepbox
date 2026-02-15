import {
  getBigIntElement,
  getNumericElement,
  getStringElement,
  isNumericTypedArray,
  isTypedArray,
  type NumericTypedArray,
} from "../core";
import { DTypeError, InvalidParameterError } from "../core/errors";
import { type Tensor, tensor } from "../ndarray";
import {
  assertFiniteNumber,
  assertSameSizeVectors,
  createFlatOffsetter,
  type FlatOffsetter,
} from "./_internal";

function getNumericLabelData(t: Tensor): NumericTypedArray {
  if (t.dtype === "string") {
    throw new DTypeError("metrics do not support string labels");
  }
  if (t.dtype === "int64") {
    throw new DTypeError("metrics do not support int64 tensors");
  }

  const data = t.data;
  if (!isTypedArray(data) || !isNumericTypedArray(data)) {
    throw new DTypeError("metrics require numeric tensors");
  }
  return data;
}

function readNumericLabel(
  data: NumericTypedArray,
  offsetter: FlatOffsetter,
  index: number,
  name: string
) {
  const value = getNumericElement(data, offsetter(index));
  assertFiniteNumber(value, name, `index ${index}`);
  return value;
}

function ensureBinaryValue(value: number, name: string, index: number): void {
  if (value !== 0 && value !== 1) {
    throw new InvalidParameterError(
      `${name} must contain only binary values (0 or 1); found ${String(value)} at index ${index}`,
      name,
      value
    );
  }
}

function isMulticlass(yTrue: Tensor, yPred: Tensor): boolean {
  if (yTrue.dtype === "string" || yPred.dtype === "string") return true;
  const yTrueData = getNumericLabelData(yTrue);
  const yPredData = getNumericLabelData(yPred);
  const trueOffset = createFlatOffsetter(yTrue);
  const predOffset = createFlatOffsetter(yPred);
  const unique = new Set<number>();
  for (let i = 0; i < yTrue.size; i++) {
    unique.add(getNumericElement(yTrueData, trueOffset(i)));
    unique.add(getNumericElement(yPredData, predOffset(i)));
    if (unique.size > 2) return true;
  }
  return unique.size > 2;
}

function assertBinaryLabels(yTrue: Tensor, yPred: Tensor): void {
  if (yTrue.dtype === "string" || yPred.dtype === "string") {
    throw new InvalidParameterError(
      "classificationReport requires binary numeric labels (0 or 1)",
      "yTrue"
    );
  }

  const yTrueData = getNumericLabelData(yTrue);
  const yPredData = getNumericLabelData(yPred);
  const trueOffset = createFlatOffsetter(yTrue);
  const predOffset = createFlatOffsetter(yPred);

  for (let i = 0; i < yTrue.size; i++) {
    const trueVal = readNumericLabel(yTrueData, trueOffset, i, "yTrue");
    const predVal = readNumericLabel(yPredData, predOffset, i, "yPred");
    ensureBinaryValue(trueVal, "yTrue", i);
    ensureBinaryValue(predVal, "yPred", i);
  }
}

type LabelKind = "string" | "int64" | "numeric";

function assertComparableLabelTypes(yTrue: Tensor, yPred: Tensor): LabelKind {
  const trueKind =
    yTrue.dtype === "string" ? "string" : yTrue.dtype === "int64" ? "int64" : "numeric";
  const predKind =
    yPred.dtype === "string" ? "string" : yPred.dtype === "int64" ? "int64" : "numeric";

  if (trueKind !== predKind) {
    throw new DTypeError("yTrue and yPred must use compatible label types");
  }

  return trueKind;
}

function readComparableLabel(t: Tensor, offsetter: FlatOffsetter, index: number, name: string) {
  const offset = offsetter(index);
  const data = t.data;

  if (Array.isArray(data)) {
    return getStringElement(data, offset);
  }

  if (data instanceof BigInt64Array) {
    return getBigIntElement(data, offset);
  }

  if (!isTypedArray(data) || !isNumericTypedArray(data)) {
    throw new DTypeError(`${name} must be numeric or string labels`);
  }

  const value = getNumericElement(data, offset);
  assertFiniteNumber(value, name, `index ${index}`);
  return value;
}

type ClassStats = {
  tp: number;
  fp: number;
  fn: number;
  support: number;
};

function buildClassStats(yTrue: Tensor, yPred: Tensor) {
  assertComparableLabelTypes(yTrue, yPred);
  const trueOffset = createFlatOffsetter(yTrue);
  const predOffset = createFlatOffsetter(yPred);

  const stats = new Map<number | string | bigint, ClassStats>();
  let totalTp = 0;
  let totalFp = 0;
  let totalFn = 0;

  for (let i = 0; i < yTrue.size; i++) {
    const trueVal = readComparableLabel(yTrue, trueOffset, i, "yTrue");
    const predVal = readComparableLabel(yPred, predOffset, i, "yPred");

    let trueStats = stats.get(trueVal);
    if (!trueStats) {
      trueStats = { tp: 0, fp: 0, fn: 0, support: 0 };
      stats.set(trueVal, trueStats);
    }

    let predStats = stats.get(predVal);
    if (!predStats) {
      predStats = { tp: 0, fp: 0, fn: 0, support: 0 };
      stats.set(predVal, predStats);
    }

    trueStats.support += 1;

    if (trueVal === predVal) {
      trueStats.tp += 1;
      totalTp += 1;
    } else {
      predStats.fp += 1;
      trueStats.fn += 1;
      totalFp += 1;
      totalFn += 1;
    }
  }

  const classes = Array.from(stats.keys()).sort((a, b) => {
    if (typeof a === "number" && typeof b === "number") return a - b;
    if (typeof a === "string" && typeof b === "string") return a.localeCompare(b);
    if (typeof a === "bigint" && typeof b === "bigint") return a === b ? 0 : a < b ? -1 : 1;
    return String(a).localeCompare(String(b));
  });
  return { classes, stats, totalTp, totalFp, totalFn };
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
 * **Space Complexity**: O(1)
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated targets as returned by a classifier
 * @returns Accuracy score in range [0, 1], where 1 is perfect accuracy
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If yTrue and yPred use incompatible label types
 * @throws {DataValidationError} If numeric labels contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { accuracy, tensor } from 'deepbox/core';
 *
 * const yTrue = tensor([0, 1, 1, 0, 1]);
 * const yPred = tensor([0, 1, 0, 0, 1]);
 * const acc = accuracy(yTrue, yPred); // 0.8 (4 out of 5 correct)
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function accuracy(yTrue: Tensor, yPred: Tensor): number {
  // Validate input tensors have same size
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");

  // Handle empty input - return 0 for undefined accuracy
  if (yTrue.size === 0) return 0;

  assertComparableLabelTypes(yTrue, yPred);
  const trueOffset = createFlatOffsetter(yTrue);
  const predOffset = createFlatOffsetter(yPred);

  // Count correct predictions
  let correct = 0;
  for (let i = 0; i < yTrue.size; i++) {
    const trueVal = readComparableLabel(yTrue, trueOffset, i, "yTrue");
    const predVal = readComparableLabel(yPred, predOffset, i, "yPred");
    if (trueVal === predVal) {
      correct++;
    }
  }

  // Return fraction of correct predictions
  return correct / yTrue.size;
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
 * **Time Complexity**: O(n) for binary, O(n*k) for multiclass where k is number of classes
 * **Space Complexity**: O(k) for multiclass
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated targets as returned by a classifier
 * @param average - Averaging strategy: 'binary', 'micro', 'macro', 'weighted', or null
 *   - 'binary': Calculate metrics for positive class only (default)
 *   - 'micro': Calculate metrics globally by counting total TP, FP, FN
 *   - 'macro': Calculate metrics for each class, return unweighted mean
 *   - 'weighted': Calculate metrics for each class, return weighted mean by support
 *   - null: Return array of scores for each class
 * @returns Precision score(s) in range [0, 1]
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If labels are not numeric (string or bigint)
 * @throws {DataValidationError} If labels contain NaN or infinite values
 * @throws {InvalidParameterError} If average is invalid or binary labels are not in {0,1}
 *
 * @example
 * ```ts
 * import { precision, tensor } from 'deepbox/core';
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
  average?: "binary" | "micro" | "macro" | "weighted" | null
): number | number[] {
  if (average === undefined) {
    average = isMulticlass(yTrue, yPred) ? "weighted" : "binary";
  }
  // Validate input tensors have same size
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");

  // Handle empty input
  if (yTrue.size === 0) return average === null ? [] : 0;

  // Binary classification (assumes positive class is 1)
  if (average === "binary") {
    if (yTrue.dtype === "string" || yPred.dtype === "string") {
      throw new InvalidParameterError(
        "Binary average requires numeric labels (0/1). Use 'macro', 'micro', or 'weighted' for string labels."
      );
    }
    const yTrueData = getNumericLabelData(yTrue);
    const yPredData = getNumericLabelData(yPred);
    const trueOffset = createFlatOffsetter(yTrue);
    const predOffset = createFlatOffsetter(yPred);

    let tp = 0;
    let fp = 0;

    for (let i = 0; i < yTrue.size; i++) {
      const trueVal = readNumericLabel(yTrueData, trueOffset, i, "yTrue");
      const predVal = readNumericLabel(yPredData, predOffset, i, "yPred");
      ensureBinaryValue(trueVal, "yTrue", i);
      ensureBinaryValue(predVal, "yPred", i);

      if (predVal === 1) {
        if (trueVal === 1) {
          tp++;
        } else {
          fp++;
        }
      }
    }

    return tp + fp === 0 ? 0 : tp / (tp + fp);
  }

  const { classes, stats, totalTp, totalFp } = buildClassStats(yTrue, yPred);

  const precisions: number[] = [];
  const supports: number[] = [];
  for (const cls of classes) {
    const classStats = stats.get(cls);
    const tp = classStats?.tp ?? 0;
    const fp = classStats?.fp ?? 0;
    const support = classStats?.support ?? 0;
    precisions.push(tp + fp === 0 ? 0 : tp / (tp + fp));
    supports.push(support);
  }

  if (average === null) {
    return precisions;
  }

  if (average === "micro") {
    return totalTp + totalFp === 0 ? 0 : totalTp / (totalTp + totalFp);
  }

  if (average === "macro") {
    const sum = precisions.reduce((acc, val) => acc + val, 0);
    return precisions.length === 0 ? 0 : sum / precisions.length;
  }

  if (average === "weighted") {
    let weightedSum = 0;
    let totalSupport = 0;

    for (let i = 0; i < precisions.length; i++) {
      weightedSum += (precisions[i] ?? 0) * (supports[i] ?? 0);
      totalSupport += supports[i] ?? 0;
    }

    return totalSupport === 0 ? 0 : weightedSum / totalSupport;
  }

  throw new InvalidParameterError(
    `Invalid average parameter: ${average}. Must be one of: 'binary', 'micro', 'macro', 'weighted', or null`,
    "average",
    average
  );
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
 * **Time Complexity**: O(n) for binary, O(n*k) for multiclass where k is number of classes
 * **Space Complexity**: O(k) for multiclass
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated targets as returned by a classifier
 * @param average - Averaging strategy: 'binary', 'micro', 'macro', 'weighted', or null
 * @returns Recall score(s) in range [0, 1]
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If labels are not numeric (string or bigint)
 * @throws {DataValidationError} If labels contain NaN or infinite values
 * @throws {InvalidParameterError} If average is invalid or binary labels are not in {0,1}
 *
 * @example
 * ```ts
 * import { recall, tensor } from 'deepbox/core';
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
  average?: "binary" | "micro" | "macro" | "weighted" | null
): number | number[] {
  if (average === undefined) {
    average = isMulticlass(yTrue, yPred) ? "weighted" : "binary";
  }
  // Validate input tensors have same size
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");

  // Handle empty input
  if (yTrue.size === 0) return average === null ? [] : 0;

  // Binary classification (assumes positive class is 1)
  if (average === "binary") {
    if (yTrue.dtype === "string" || yPred.dtype === "string") {
      throw new InvalidParameterError(
        "Binary average requires numeric labels (0/1). Use 'macro', 'micro', or 'weighted' for string labels."
      );
    }
    const yTrueData = getNumericLabelData(yTrue);
    const yPredData = getNumericLabelData(yPred);
    const trueOffset = createFlatOffsetter(yTrue);
    const predOffset = createFlatOffsetter(yPred);

    let tp = 0;
    let fn = 0;

    for (let i = 0; i < yTrue.size; i++) {
      const trueVal = readNumericLabel(yTrueData, trueOffset, i, "yTrue");
      const predVal = readNumericLabel(yPredData, predOffset, i, "yPred");
      ensureBinaryValue(trueVal, "yTrue", i);
      ensureBinaryValue(predVal, "yPred", i);

      if (trueVal === 1) {
        if (predVal === 1) {
          tp++;
        } else {
          fn++;
        }
      }
    }

    return tp + fn === 0 ? 0 : tp / (tp + fn);
  }

  const { classes, stats, totalTp, totalFn } = buildClassStats(yTrue, yPred);

  const recalls: number[] = [];
  const supports: number[] = [];
  for (const cls of classes) {
    const classStats = stats.get(cls);
    const tp = classStats?.tp ?? 0;
    const fn = classStats?.fn ?? 0;
    const support = classStats?.support ?? 0;
    recalls.push(tp + fn === 0 ? 0 : tp / (tp + fn));
    supports.push(support);
  }

  if (average === null) {
    return recalls;
  }

  if (average === "micro") {
    return totalTp + totalFn === 0 ? 0 : totalTp / (totalTp + totalFn);
  }

  if (average === "macro") {
    const sum = recalls.reduce((acc, val) => acc + val, 0);
    return recalls.length === 0 ? 0 : sum / recalls.length;
  }

  if (average === "weighted") {
    let weightedSum = 0;
    let totalSupport = 0;

    for (let i = 0; i < recalls.length; i++) {
      weightedSum += (recalls[i] ?? 0) * (supports[i] ?? 0);
      totalSupport += supports[i] ?? 0;
    }

    return totalSupport === 0 ? 0 : weightedSum / totalSupport;
  }

  throw new InvalidParameterError(
    `Invalid average parameter: ${average}. Must be one of: 'binary', 'micro', 'macro', 'weighted', or null`,
    "average",
    average
  );
}

/**
 * Calculates the F1 score (harmonic mean of precision and recall).
 *
 * F1 score is the harmonic mean of precision and recall, providing a single
 * metric that balances both concerns. It's especially useful when you need
 * to balance false positives and false negatives.
 *
 * **Formula**: F1 = 2 * (precision * recall) / (precision + recall)
 *
 * **Time Complexity**: O(n) for binary, O(n*k) for multiclass
 * **Space Complexity**: O(k) for multiclass
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated targets as returned by a classifier
 * @param average - Averaging strategy: 'binary', 'micro', 'macro', 'weighted', or null
 * @returns F1 score(s) in range [0, 1]
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If labels are not numeric (string or bigint)
 * @throws {DataValidationError} If labels contain NaN or infinite values
 * @throws {InvalidParameterError} If average is invalid or binary labels are not in {0,1}
 *
 * @example
 * ```ts
 * import { f1Score, tensor } from 'deepbox/core';
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
  average: { average: "binary" | "micro" | "macro" | "weighted" }
): number;
export function f1Score(
  yTrue: Tensor,
  yPred: Tensor,
  average?:
    | "binary"
    | "micro"
    | "macro"
    | "weighted"
    | null
    | { average: "binary" | "micro" | "macro" | "weighted" }
): number | number[] {
  if (average !== null && typeof average === "object") {
    average = average.average;
  }
  if (average === undefined) {
    average = isMulticlass(yTrue, yPred) ? "weighted" : "binary";
  }
  // For binary and micro, computing F1 from scalar P and R is correct.
  // For macro and weighted, we must compute per-class F1 first, then average,
  // because the harmonic mean is non-linear: avg(F1_i) ≠ F1(avg(P_i), avg(R_i)).
  if (average === "binary" || average === "micro") {
    const p = precision(yTrue, yPred, average);
    const r = recall(yTrue, yPred, average);
    return p + r === 0 ? 0 : (2 * p * r) / (p + r);
  }

  // Compute per-class precision and recall
  const prec = precision(yTrue, yPred, null);
  const rec = recall(yTrue, yPred, null);

  // Compute per-class F1 scores
  const f1Scores: number[] = [];
  for (let i = 0; i < prec.length; i++) {
    const p = prec[i] ?? 0;
    const r = rec[i] ?? 0;
    f1Scores.push(p + r === 0 ? 0 : (2 * p * r) / (p + r));
  }

  if (average === null) {
    return f1Scores;
  }

  if (f1Scores.length === 0) return 0;

  if (average === "macro") {
    const sum = f1Scores.reduce((acc, val) => acc + val, 0);
    return sum / f1Scores.length;
  }

  if (average === "weighted") {
    const { classes, stats } = buildClassStats(yTrue, yPred);
    let weightedSum = 0;
    let totalSupport = 0;
    for (let i = 0; i < f1Scores.length; i++) {
      const cls = classes[i];
      const support = cls !== undefined ? (stats.get(cls)?.support ?? 0) : 0;
      weightedSum += (f1Scores[i] ?? 0) * support;
      totalSupport += support;
    }
    return totalSupport === 0 ? 0 : weightedSum / totalSupport;
  }

  throw new InvalidParameterError(
    `Invalid average parameter: ${average}. Must be one of: 'binary', 'micro', 'macro', 'weighted', or null`,
    "average",
    average
  );
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
 * **Time Complexity**: O(n) for binary, O(n*k) for multiclass
 * **Space Complexity**: O(k) for multiclass
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated targets as returned by a classifier
 * @param beta - Weight of recall vs precision (beta > 1 favors recall, beta < 1 favors precision)
 * @param average - Averaging strategy: 'binary', 'micro', 'macro', 'weighted', or null
 * @returns F-beta score(s) in range [0, 1]
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If labels are not numeric (string or bigint)
 * @throws {DataValidationError} If labels contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { fbetaScore, tensor } from 'deepbox/core';
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
  average: "binary" | "micro" | "macro" | "weighted" | null = "binary"
) {
  if (!Number.isFinite(beta) || beta <= 0) {
    throw new InvalidParameterError("beta must be a positive finite number", "beta", beta);
  }

  const betaSq = beta * beta;

  // For binary and micro, computing fbeta from scalar P and R is correct.
  // For macro and weighted, we must compute per-class fbeta first, then average,
  // because the weighted harmonic mean is non-linear.
  if (average === "binary" || average === "micro") {
    const p = precision(yTrue, yPred, average);
    const r = recall(yTrue, yPred, average);
    return p + r === 0 ? 0 : ((1 + betaSq) * p * r) / (betaSq * p + r);
  }

  // Compute per-class precision and recall
  const prec = precision(yTrue, yPred, null);
  const rec = recall(yTrue, yPred, null);

  // Compute per-class fbeta scores
  const fbetaScores: number[] = [];
  for (let i = 0; i < prec.length; i++) {
    const p = prec[i] ?? 0;
    const r = rec[i] ?? 0;
    fbetaScores.push(p + r === 0 ? 0 : ((1 + betaSq) * p * r) / (betaSq * p + r));
  }

  if (average === null) {
    return fbetaScores;
  }

  if (fbetaScores.length === 0) return 0;

  if (average === "macro") {
    const sum = fbetaScores.reduce((acc, val) => acc + val, 0);
    return sum / fbetaScores.length;
  }

  if (average === "weighted") {
    const { classes, stats } = buildClassStats(yTrue, yPred);
    let weightedSum = 0;
    let totalSupport = 0;
    for (let i = 0; i < fbetaScores.length; i++) {
      const cls = classes[i];
      const support = cls !== undefined ? (stats.get(cls)?.support ?? 0) : 0;
      weightedSum += (fbetaScores[i] ?? 0) * support;
      totalSupport += support;
    }
    return totalSupport === 0 ? 0 : weightedSum / totalSupport;
  }

  throw new InvalidParameterError(
    `Invalid average parameter: ${average}. Must be one of: 'binary', 'micro', 'macro', 'weighted', or null`,
    "average",
    average
  );
}

/**
 * Computes the confusion matrix to evaluate classification accuracy.
 *
 * A confusion matrix is a table showing the counts of correct and incorrect
 * predictions broken down by each class. Rows represent true labels,
 * columns represent predicted labels.
 *
 * **Time Complexity**: O(n + k²) where n is number of samples, k is number of classes
 * **Space Complexity**: O(k²)
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated targets as returned by a classifier
 * @returns Confusion matrix as a 2D tensor of shape [n_classes, n_classes]
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If labels are not numeric (string or bigint)
 * @throws {DataValidationError} If labels contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { confusionMatrix, tensor } from 'deepbox/core';
 *
 * const yTrue = tensor([0, 1, 1, 0, 1]);
 * const yPred = tensor([0, 1, 0, 0, 1]);
 * const cm = confusionMatrix(yTrue, yPred);
 * // [[2, 0],
 * //  [1, 2]]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function confusionMatrix(yTrue: Tensor, yPred: Tensor): Tensor {
  // Validate input tensors have same size
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");
  assertComparableLabelTypes(yTrue, yPred);

  if (yTrue.size === 0) {
    return tensor([]).reshape([0, 0]);
  }

  const trueOffset = createFlatOffsetter(yTrue);
  const predOffset = createFlatOffsetter(yPred);

  // Collect all unique labels from both true and predicted values
  const labelSet = new Set<number | string | bigint>();
  for (let i = 0; i < yTrue.size; i++) {
    labelSet.add(readComparableLabel(yTrue, trueOffset, i, "yTrue"));
    labelSet.add(readComparableLabel(yPred, predOffset, i, "yPred"));
  }

  // Sort labels for consistent ordering
  const labels = Array.from(labelSet).sort((a, b) => {
    if (typeof a === "number" && typeof b === "number") return a - b;
    if (typeof a === "string" && typeof b === "string") return a.localeCompare(b);
    if (typeof a === "bigint" && typeof b === "bigint") return a === b ? 0 : a < b ? -1 : 1;
    return String(a).localeCompare(String(b));
  });

  // Create mapping from label to matrix index
  const labelToIndex = new Map<number | string | bigint, number>();
  for (let i = 0; i < labels.length; i++) {
    const label = labels[i];
    if (label === undefined) continue;
    labelToIndex.set(label, i);
  }

  // Initialize confusion matrix with zeros
  const nClasses = labels.length;
  const matrix = Array.from({ length: nClasses }, () => new Array<number>(nClasses).fill(0));

  // Populate confusion matrix by counting predictions
  for (let i = 0; i < yTrue.size; i++) {
    const trueLabel = readComparableLabel(yTrue, trueOffset, i, "yTrue");
    const predLabel = readComparableLabel(yPred, predOffset, i, "yPred");

    // Get matrix indices for this true/pred label pair
    const r = labelToIndex.get(trueLabel);
    const c = labelToIndex.get(predLabel);
    if (r === undefined || c === undefined) continue;

    // Increment count at [true_label, pred_label]
    const row = matrix[r];
    if (row) row[c] = (row[c] ?? 0) + 1;
  }

  // Convert to tensor and return
  return tensor(matrix);
}

/**
 * Generates a text classification report showing main classification metrics.
 *
 * Provides a comprehensive summary including precision, recall, F1-score,
 * and accuracy for the classification task.
 *
 * **Time Complexity**: O(n * k) where n is the number of samples and k is the number of classes
 * **Space Complexity**: O(k) where k is the number of classes
 *
 * @param yTrue - Ground truth (correct) binary target values (0 or 1)
 * @param yPred - Estimated binary targets as returned by a classifier (0 or 1)
 * @returns Formatted string report with per-class and aggregate classification metrics
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If labels are not numeric (string or int64)
 * @throws {DataValidationError} If labels contain NaN or infinite values
 * @throws {InvalidParameterError} If labels are not binary (0 or 1)
 *
 * @example
 * ```ts
 * import { classificationReport, tensor } from 'deepbox/core';
 *
 * const yTrue = tensor([0, 1, 1, 0, 1]);
 * const yPred = tensor([0, 1, 0, 0, 1]);
 * console.log(classificationReport(yTrue, yPred));
 * // Classification Report:
 * //   Precision: 1.0000
 * //   Recall:    0.6667
 * //   F1-Score:  0.8000
 * //   Accuracy:  0.8000
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function classificationReport(yTrue: Tensor, yPred: Tensor): string {
  // Validate input tensors have same size
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");

  // Handle empty input
  if (yTrue.size === 0) return "Classification Report:\n  (empty)";

  assertBinaryLabels(yTrue, yPred);

  // Calculate per-class metrics
  const { classes, stats } = buildClassStats(yTrue, yPred);
  const precs = precision(yTrue, yPred, null);
  const recs = recall(yTrue, yPred, null);
  const f1s = f1Score(yTrue, yPred, null);
  const acc = accuracy(yTrue, yPred);

  // Determine width for class names
  const maxClassLen = Math.max(...classes.map((c) => String(c).length), "Class".length);
  const colWidth = Math.max(12, maxClassLen + 2);

  // Header
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

  for (const [i, cls] of classes.entries()) {
    const p = precs[i] ?? 0;
    const r = recs[i] ?? 0;
    const f1 = f1s[i] ?? 0;
    const s = stats.get(cls)?.support ?? 0;

    totalSupport += s;
    weightedPrec += p * s;
    weightedRec += r * s;
    weightedF1 += f1 * s;
    macroPrec += p;
    macroRec += r;
    macroF1 += f1;

    report +=
      String(cls).padEnd(colWidth) +
      p.toFixed(4).padEnd(12) +
      r.toFixed(4).padEnd(12) +
      f1.toFixed(4).padEnd(12) +
      String(s) +
      "\n";
  }

  report += "\n";

  // Averages

  const nClasses = classes.length;
  if (nClasses > 0) {
    macroPrec /= nClasses;
    macroRec /= nClasses;
    macroF1 /= nClasses;
  } else {
    macroPrec = 0;
    macroRec = 0;
    macroF1 = 0;
  }

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
 * - thresholds: Decision thresholds (in descending order)
 *
 * **Edge Cases**:
 * - Returns empty tensors if yTrue contains only one class
 * - Handles tied scores by grouping them at the same threshold
 *
 * @param yTrue - Ground truth binary labels (must be 0 or 1)
 * @param yScore - Target scores (higher score = more likely positive class)
 * @returns Tuple of [fpr, tpr, thresholds] tensors
 *
 * @throws {ShapeError} If yTrue and yScore have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If inputs are not numeric (string or int64)
 * @throws {InvalidParameterError} If yTrue contains non-binary values
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function rocCurve(yTrue: Tensor, yScore: Tensor): [Tensor, Tensor, Tensor] {
  assertSameSizeVectors(yTrue, yScore, "yTrue", "yScore");

  const n = yTrue.size;

  if (n === 0) return [tensor([]), tensor([]), tensor([])];

  const yTrueData = getNumericLabelData(yTrue);
  const yScoreData = getNumericLabelData(yScore);
  const trueOffset = createFlatOffsetter(yTrue);
  const scoreOffset = createFlatOffsetter(yScore);

  // Create pairs of (score, label)
  const pairs: Array<{ score: number; label: number }> = [];
  let nPos = 0;
  let nNeg = 0;
  for (let i = 0; i < n; i++) {
    const label = readNumericLabel(yTrueData, trueOffset, i, "yTrue");
    ensureBinaryValue(label, "yTrue", i);
    const score = readNumericLabel(yScoreData, scoreOffset, i, "yScore");
    pairs.push({ score, label });
    if (label === 1) nPos++;
    else nNeg++;
  }

  // Sort by score descending
  pairs.sort((a, b) => b.score - a.score);

  if (nPos === 0 || nNeg === 0) return [tensor([]), tensor([]), tensor([])];

  const fpr = [0];
  const tpr = [0];
  const thresholds = [Infinity];

  let tp = 0;
  let fp = 0;
  let idx = 0;
  while (idx < pairs.length) {
    const threshold = pairs[idx]?.score ?? 0;

    // Consume all samples with the same score as one step.
    while (idx < pairs.length && (pairs[idx]?.score ?? 0) === threshold) {
      const label = pairs[idx]?.label ?? 0;
      if (label === 1) tp++;
      else fp++;
      idx++;
    }

    fpr.push(fp / nNeg);
    tpr.push(tp / nPos);
    thresholds.push(threshold);
  }

  return [tensor(fpr), tensor(tpr), tensor(thresholds)];
}

/**
 * Area Under ROC Curve (AUC-ROC).
 *
 * Computes the Area Under the Receiver Operating Characteristic Curve.
 * AUC represents the probability that a randomly chosen positive sample
 * is ranked higher than a randomly chosen negative sample.
 *
 * **Range**: [0, 1], where 1 is perfect and 0.5 is random.
 *
 * **Time Complexity**: O(n log n) due to sorting
 * **Space Complexity**: O(n)
 *
 * @param yTrue - Ground truth binary labels (must be 0 or 1)
 * @param yScore - Target scores (higher score = more likely positive class)
 * @returns AUC score in range [0, 1], or 0.5 if ROC curve cannot be computed
 *
 * @throws {ShapeError} If yTrue and yScore have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If inputs are not numeric (string or int64)
 * @throws {InvalidParameterError} If yTrue contains non-binary values
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { rocAucScore, tensor } from 'deepbox/metrics';
 *
 * const yTrue = tensor([0, 0, 1, 1]);
 * const yScore = tensor([0.1, 0.4, 0.35, 0.8]);
 * const auc = rocAucScore(yTrue, yScore); // ~0.75
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function rocAucScore(yTrue: Tensor, yScore: Tensor): number {
  const curves = rocCurve(yTrue, yScore);
  const fprT = curves[0];
  const tprT = curves[1];
  if (!fprT || !tprT || fprT.size === 0 || tprT.size === 0) return 0.5;

  const fprData = getNumericLabelData(fprT);
  const tprData = getNumericLabelData(tprT);
  const fprOffset = createFlatOffsetter(fprT);
  const tprOffset = createFlatOffsetter(tprT);

  let auc = 0;
  let prevX = 0;
  let prevY = 0;
  for (let i = 1; i < fprT.size; i++) {
    const x = readNumericLabel(fprData, fprOffset, i, "fpr");
    const y = readNumericLabel(tprData, tprOffset, i, "tpr");
    auc += (x - prevX) * ((y + prevY) / 2);
    prevX = x;
    prevY = y;
  }

  return auc;
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
 * - thresholds: Decision thresholds (in descending order)
 *
 * **Time Complexity**: O(n log n) due to sorting
 * **Space Complexity**: O(n)
 *
 * @param yTrue - Ground truth binary labels (0 or 1)
 * @param yScore - Target scores (higher score = more likely positive class)
 * @returns Tuple of [precision, recall, thresholds] tensors
 *
 * @throws {ShapeError} If yTrue and yScore have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If inputs are not numeric (string or int64)
 * @throws {InvalidParameterError} If yTrue contains non-binary values
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { precisionRecallCurve, tensor } from 'deepbox/metrics';
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

  const n = yTrue.size;

  if (n === 0) return [tensor([]), tensor([]), tensor([])];

  const yTrueData = getNumericLabelData(yTrue);
  const yScoreData = getNumericLabelData(yScore);
  const trueOffset = createFlatOffsetter(yTrue);
  const scoreOffset = createFlatOffsetter(yScore);

  // Create pairs of (score, label)
  const pairs: Array<{ score: number; label: number }> = [];
  let nPos = 0;
  for (let i = 0; i < n; i++) {
    const label = readNumericLabel(yTrueData, trueOffset, i, "yTrue");
    ensureBinaryValue(label, "yTrue", i);
    const score = readNumericLabel(yScoreData, scoreOffset, i, "yScore");
    pairs.push({ score, label });
    if (label === 1) nPos++;
  }

  // Sort by score descending
  pairs.sort((a, b) => b.score - a.score);

  if (nPos === 0) return [tensor([]), tensor([]), tensor([])];

  const prec = [1];
  const rec = [0];
  const thresholds = [Infinity];

  let tp = 0;
  let fp = 0;
  let idx = 0;
  while (idx < pairs.length) {
    const threshold = pairs[idx]?.score ?? 0;
    while (idx < pairs.length && (pairs[idx]?.score ?? 0) === threshold) {
      const label = pairs[idx]?.label ?? 0;
      if (label === 1) tp++;
      else fp++;
      idx++;
    }

    const precisionVal = tp + fp === 0 ? 1 : tp / (tp + fp);
    const recallVal = tp / nPos;
    prec.push(precisionVal);
    rec.push(recallVal);
    thresholds.push(threshold);
  }

  return [tensor(prec), tensor(rec), tensor(thresholds)];
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
 * @returns Average precision score in range [0, 1]
 *
 * @throws {ShapeError} If yTrue and yScore have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If inputs are not numeric (string or int64)
 * @throws {InvalidParameterError} If yTrue contains non-binary values
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { averagePrecisionScore, tensor } from 'deepbox/metrics';
 *
 * const yTrue = tensor([0, 0, 1, 1]);
 * const yScore = tensor([0.1, 0.4, 0.35, 0.8]);
 * const ap = averagePrecisionScore(yTrue, yScore);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function averagePrecisionScore(yTrue: Tensor, yScore: Tensor): number {
  const curves = precisionRecallCurve(yTrue, yScore);
  const precT = curves[0];
  const recT = curves[1];
  if (!precT || !recT || precT.size === 0 || recT.size === 0) return 0;

  const precData = getNumericLabelData(precT);
  const recData = getNumericLabelData(recT);
  const precOffset = createFlatOffsetter(precT);
  const recOffset = createFlatOffsetter(recT);

  let ap = 0;
  let prevRecall = readNumericLabel(recData, recOffset, 0, "recall");
  for (let i = 1; i < recT.size; i++) {
    const recall = readNumericLabel(recData, recOffset, i, "recall");
    const precision = readNumericLabel(precData, precOffset, i, "precision");
    const deltaRecall = recall - prevRecall;
    if (deltaRecall > 0) {
      ap += deltaRecall * precision;
    }
    prevRecall = recall;
  }

  return ap;
}

/**
 * Log loss (logistic loss, cross-entropy loss).
 *
 * Measures the performance of a classification model where the prediction is a probability
 * value between 0 and 1. Lower log loss indicates better predictions.
 *
 * **Formula**: -1/n * Σ(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
 *
 * **Edge Cases**:
 * - Predictions are clipped to [1e-15, 1-1e-15] to avoid log(0)
 * - Returns 0 for empty inputs
 *
 * @param yTrue - Ground truth binary labels (0 or 1)
 * @param yPred - Predicted probabilities (must be in range [0, 1])
 * @returns Log loss value (lower is better, 0 is perfect)
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If inputs are not numeric (string or int64)
 * @throws {InvalidParameterError} If yTrue is not binary or yPred is outside [0, 1]
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function logLoss(yTrue: Tensor, yPred: Tensor): number {
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");

  if (yTrue.size === 0) return 0;

  const yTrueData = getNumericLabelData(yTrue);
  const yPredData = getNumericLabelData(yPred);
  const trueOffset = createFlatOffsetter(yTrue);
  const predOffset = createFlatOffsetter(yPred);

  const eps = 1e-15;
  let loss = 0;

  for (let i = 0; i < yTrue.size; i++) {
    const trueVal = readNumericLabel(yTrueData, trueOffset, i, "yTrue");
    ensureBinaryValue(trueVal, "yTrue", i);
    const predRaw = readNumericLabel(yPredData, predOffset, i, "yPred");
    if (predRaw < 0 || predRaw > 1) {
      throw new InvalidParameterError(
        `yPred must contain probabilities in range [0, 1], found ${String(predRaw)} at index ${i}`,
        "yPred",
        predRaw
      );
    }
    const predVal = Math.max(eps, Math.min(1 - eps, predRaw));
    loss -= trueVal * Math.log(predVal) + (1 - trueVal) * Math.log(1 - predVal);
  }

  return loss / yTrue.size;
}

/**
 * Hamming loss.
 *
 * Computes the fraction of labels that are incorrectly predicted.
 * For binary classification, this equals 1 - accuracy.
 *
 * **Formula**: hamming_loss = (incorrect predictions) / (total predictions)
 *
 * **Range**: [0, 1], where 0 is perfect.
 *
 * **Time Complexity**: O(n)
 * **Space Complexity**: O(1)
 *
 * @param yTrue - Ground truth target values
 * @param yPred - Estimated targets as returned by a classifier
 * @returns Hamming loss in range [0, 1]
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If yTrue and yPred use incompatible label types
 * @throws {DataValidationError} If numeric labels contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { hammingLoss, tensor } from 'deepbox/metrics';
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

  assertComparableLabelTypes(yTrue, yPred);
  const trueOffset = createFlatOffsetter(yTrue);
  const predOffset = createFlatOffsetter(yPred);

  let errors = 0;
  for (let i = 0; i < yTrue.size; i++) {
    const trueVal = readComparableLabel(yTrue, trueOffset, i, "yTrue");
    const predVal = readComparableLabel(yPred, predOffset, i, "yPred");
    if (trueVal !== predVal) {
      errors++;
    }
  }

  return errors / yTrue.size;
}

/**
 * Jaccard similarity score (Intersection over Union).
 *
 * Computes the Jaccard similarity coefficient between two binary label sets.
 * Also known as the Jaccard index or Intersection over Union (IoU).
 *
 * **Formula**: jaccard = TP / (TP + FP + FN)
 *
 * **Edge Cases**:
 * - If both yTrue and yPred contain no positive labels (TP + FP + FN = 0),
 *   returns 1 to reflect perfect similarity of empty sets.
 *
 * **Range**: [0, 1], where 1 is perfect.
 *
 * **Time Complexity**: O(n)
 * **Space Complexity**: O(1)
 *
 * @param yTrue - Ground truth binary labels (0 or 1)
 * @param yPred - Predicted binary labels (0 or 1)
 * @returns Jaccard score in range [0, 1]
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If labels are not numeric (string or bigint)
 * @throws {DataValidationError} If labels contain NaN or infinite values
 * @throws {InvalidParameterError} If labels are not binary (0 or 1)
 *
 * @example
 * ```ts
 * import { jaccardScore, tensor } from 'deepbox/metrics';
 *
 * const yTrue = tensor([0, 1, 1, 0, 1]);
 * const yPred = tensor([0, 1, 0, 0, 1]);
 * const score = jaccardScore(yTrue, yPred); // 0.667
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function jaccardScore(yTrue: Tensor, yPred: Tensor): number {
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");

  if (yTrue.size === 0) return 1;

  const yTrueData = getNumericLabelData(yTrue);
  const yPredData = getNumericLabelData(yPred);
  const trueOffset = createFlatOffsetter(yTrue);
  const predOffset = createFlatOffsetter(yPred);

  let tp = 0,
    fp = 0,
    fn = 0;

  for (let i = 0; i < yTrue.size; i++) {
    const trueVal = readNumericLabel(yTrueData, trueOffset, i, "yTrue");
    const predVal = readNumericLabel(yPredData, predOffset, i, "yPred");
    ensureBinaryValue(trueVal, "yTrue", i);
    ensureBinaryValue(predVal, "yPred", i);

    if (trueVal === 1 && predVal === 1) tp++;
    else if (trueVal === 0 && predVal === 1) fp++;
    else if (trueVal === 1 && predVal === 0) fn++;
  }

  return tp + fp + fn === 0 ? 1 : tp / (tp + fp + fn);
}

/**
 * Matthews correlation coefficient (MCC).
 *
 * Computes the Matthews correlation coefficient, a balanced measure that
 * can be used even if the classes are of very different sizes. MCC is
 * considered one of the best metrics for binary classification.
 *
 * **Formula**: MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
 *
 * **Range**: [-1, 1], where 1 is perfect, 0 is random, -1 is inverse.
 *
 * **Time Complexity**: O(n)
 * **Space Complexity**: O(1)
 *
 * @param yTrue - Ground truth binary labels (0 or 1)
 * @param yPred - Predicted binary labels (0 or 1)
 * @returns MCC score in range [-1, 1]
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If labels are not numeric (string or bigint)
 * @throws {DataValidationError} If labels contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { matthewsCorrcoef, tensor } from 'deepbox/metrics';
 *
 * const yTrue = tensor([0, 1, 1, 0, 1]);
 * const yPred = tensor([0, 1, 0, 0, 1]);
 * const mcc = matthewsCorrcoef(yTrue, yPred); // ~0.667
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function matthewsCorrcoef(yTrue: Tensor, yPred: Tensor): number {
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");

  if (yTrue.size === 0) return 0;

  const yTrueData = getNumericLabelData(yTrue);
  const yPredData = getNumericLabelData(yPred);
  const trueOffset = createFlatOffsetter(yTrue);
  const predOffset = createFlatOffsetter(yPred);

  let tp = 0,
    tn = 0,
    fp = 0,
    fn = 0;

  for (let i = 0; i < yTrue.size; i++) {
    const trueVal = readNumericLabel(yTrueData, trueOffset, i, "yTrue");
    const predVal = readNumericLabel(yPredData, predOffset, i, "yPred");
    ensureBinaryValue(trueVal, "yTrue", i);
    ensureBinaryValue(predVal, "yPred", i);

    if (trueVal === 1 && predVal === 1) tp++;
    else if (trueVal === 0 && predVal === 0) tn++;
    else if (trueVal === 0 && predVal === 1) fp++;
    else if (trueVal === 1 && predVal === 0) fn++;
  }

  const numerator = tp * tn - fp * fn;
  const denominator = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));

  return denominator === 0 ? 0 : numerator / denominator;
}

/**
 * Cohen's kappa score.
 *
 * Computes Cohen's kappa, a statistic that measures inter-annotator agreement.
 * It is generally thought to be a more robust measure than simple percent
 * agreement since it takes into account the possibility of agreement by chance.
 *
 * **Formula**: kappa = (p_o - p_e) / (1 - p_e)
 * - p_o: observed agreement
 * - p_e: expected agreement by chance
 *
 * **Range**: [-1, 1], where 1 is perfect agreement, 0 is chance, <0 is worse than chance.
 *
 * **Time Complexity**: O(n)
 * **Space Complexity**: O(k) where k is number of classes
 *
 * @param yTrue - Ground truth labels
 * @param yPred - Predicted labels
 * @returns Kappa score in range [-1, 1]
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If labels are not numeric (string or bigint)
 * @throws {DataValidationError} If labels contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { cohenKappaScore, tensor } from 'deepbox/metrics';
 *
 * const yTrue = tensor([0, 1, 1, 0, 1]);
 * const yPred = tensor([0, 1, 0, 0, 1]);
 * const kappa = cohenKappaScore(yTrue, yPred); // ~0.615
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-classification | Deepbox Classification Metrics}
 */
export function cohenKappaScore(yTrue: Tensor, yPred: Tensor): number {
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");

  const n = yTrue.size;

  if (n === 0) return 0;

  const yTrueData = getNumericLabelData(yTrue);
  const yPredData = getNumericLabelData(yPred);
  const trueOffset = createFlatOffsetter(yTrue);
  const predOffset = createFlatOffsetter(yPred);

  // Observed agreement and class marginals
  let po = 0;
  const trueCount = new Map<number, number>();
  const predCount = new Map<number, number>();

  for (let i = 0; i < n; i++) {
    const t = readNumericLabel(yTrueData, trueOffset, i, "yTrue");
    const p = readNumericLabel(yPredData, predOffset, i, "yPred");
    if (t === p) {
      po++;
    }
    trueCount.set(t, (trueCount.get(t) ?? 0) + 1);
    predCount.set(p, (predCount.get(p) ?? 0) + 1);
  }

  po /= n;

  let pe = 0;
  const allClasses = new Set([...trueCount.keys(), ...predCount.keys()]);
  for (const c of allClasses) {
    const trueProb = (trueCount.get(c) ?? 0) / n;
    const predProb = (predCount.get(c) ?? 0) / n;
    pe += trueProb * predProb;
  }

  const denom = 1 - pe;
  if (denom === 0) return po === 1 ? 1 : 0;
  return (po - pe) / denom;
}

/**
 * Balanced accuracy score.
 *
 * Computes the balanced accuracy, which is the macro-averaged recall.
 * It is useful for imbalanced datasets where regular accuracy can be misleading.
 *
 * **Formula**: balanced_accuracy = (1/n_classes) * Σ(recall_per_class)
 *
 * **Range**: [0, 1], where 1 is perfect.
 *
 * **Time Complexity**: O(n * k) where k is number of classes
 * **Space Complexity**: O(k)
 *
 * @param yTrue - Ground truth labels
 * @param yPred - Predicted labels
 * @returns Balanced accuracy score in range [0, 1]
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If labels are not numeric (string or bigint)
 * @throws {DataValidationError} If labels contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { balancedAccuracyScore, tensor } from 'deepbox/metrics';
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

  const { classes, stats } = buildClassStats(yTrue, yPred);
  let sumRecall = 0;
  let classCount = 0;

  for (const cls of classes) {
    const classStats = stats.get(cls);
    const support = classStats?.support ?? 0;
    if (support === 0) continue;
    const tp = classStats?.tp ?? 0;
    const fn = classStats?.fn ?? 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    sumRecall += recall;
    classCount++;
  }

  return classCount === 0 ? 0 : sumRecall / classCount;
}
