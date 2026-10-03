/**
 * Additional metrics: SMAPE, MSLE, Brier score, hinge loss, zero-one loss, top-k accuracy,
 * deviance scores, ranking losses and the DET curve.
 *
 * @module metrics/extra
 * @see {@link https://deepbox.dev/docs/metrics-clustering | Deepbox Metrics}
 */

import { InvalidParameterError, ShapeError } from "../core/errors";
import type { Tensor } from "../ndarray";
import {
  compensatedSum,
  createFlatOffsetter,
  nonZeroWeightSum,
  readFiniteFloat64,
  readSampleWeight,
  type WeightedMetricOptions,
} from "./_internal";

function validateSameLength(yTrue: Tensor, yPred: Tensor, name: string): void {
  if (yTrue.ndim !== 1) {
    throw new ShapeError(`${name}: yTrue must be 1D; got ndim=${yTrue.ndim}`);
  }
  if (yPred.ndim !== 1) {
    throw new ShapeError(`${name}: yPred must be 1D; got ndim=${yPred.ndim}`);
  }
  if (yTrue.size !== yPred.size) {
    throw new ShapeError(
      `${name}: yTrue and yPred must have same length; got ${yTrue.size} vs ${yPred.size}`
    );
  }
}

/**
 * Symmetric Mean Absolute Percentage Error (SMAPE).
 *
 * SMAPE = (1/n) * Σ |y_true - y_pred| / ((|y_true| + |y_pred|) / 2)
 * Returns value in [0, 2]. Commonly multiplied by 100 for percentage.
 * Samples where both values are zero count as a zero error.
 *
 * @param yTrue - True values of shape (n_samples,)
 * @param yPred - Predicted values of shape (n_samples,)
 * @returns SMAPE score in [0, 2]
 *
 * @throws {ShapeError} If inputs are not 1D or have different lengths
 * @throws {DTypeError} If an input is a string tensor
 * @throws {DataValidationError} If an input contains NaN or infinite values
 *
 * @example
 * ```ts
 * import { smape } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * smape(tensor([100, 200]), tensor([110, 180])); // 0.1002506265664160
 * ```
 */
export function smape(yTrue: Tensor, yPred: Tensor): number {
  validateSameLength(yTrue, yPred, "smape");
  const n = yTrue.size;
  if (n === 0) return 0;

  const yt = readFiniteFloat64(yTrue, "yTrue");
  const yp = readFiniteFloat64(yPred, "yPred");

  const terms = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const t = yt[i] as number;
    const p = yp[i] as number;
    const denom = Math.abs(t) / 2 + Math.abs(p) / 2;
    if (denom === 0) continue;
    terms[i] = Math.abs(t - p) / denom;
  }
  return compensatedSum(terms) / n;
}

/**
 * Mean Squared Logarithmic Error (MSLE).
 *
 * MSLE = (1/n) * Σ (log(1 + y_true) - log(1 + y_pred))²
 * Only valid for non-negative values. With `options.sampleWeight` the mean is
 * weighted: Σ w_i * term_i / Σ w_i, as in scikit-learn.
 *
 * @param yTrue - True non-negative values of shape (n_samples,)
 * @param yPred - Predicted non-negative values of shape (n_samples,)
 * @param options - Optional `sampleWeight`, one weight per sample
 * @returns MSLE score
 *
 * @throws {ShapeError} If inputs are not 1D, have different lengths, or `sampleWeight` has the
 *   wrong length
 * @throws {InvalidParameterError} If a value is negative, or the weights sum to zero
 * @throws {DataValidationError} If an input contains NaN or infinite values
 *
 * @example
 * ```ts
 * import { meanSquaredLogError } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([3, 5, 2.5, 7]);
 * const yPred = tensor([2.5, 5, 4, 8]);
 * meanSquaredLogError(yTrue, yPred); // 0.03973012298459379
 * meanSquaredLogError(yTrue, yPred, { sampleWeight: [1, 2, 3, 4] }); // 0.04549730536710696
 * ```
 */
export function meanSquaredLogError(
  yTrue: Tensor,
  yPred: Tensor,
  options: WeightedMetricOptions = {}
): number {
  validateSameLength(yTrue, yPred, "meanSquaredLogError");
  const n = yTrue.size;
  const weights = readSampleWeight(options.sampleWeight, n);
  if (n === 0) return 0;

  const yt = readFiniteFloat64(yTrue, "yTrue");
  const yp = readFiniteFloat64(yPred, "yPred");

  const terms = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const t = yt[i] as number;
    const p = yp[i] as number;
    if (t < 0 || p < 0) {
      throw new InvalidParameterError(
        "meanSquaredLogError requires non-negative values",
        "y",
        t < 0 ? t : p
      );
    }
    const diff = Math.log1p(t) - Math.log1p(p);
    terms[i] = diff * diff;
  }
  if (weights === undefined) return compensatedSum(terms) / n;

  const total = nonZeroWeightSum(weights);
  for (let i = 0; i < n; i++) terms[i] = (terms[i] as number) * (weights[i] as number);
  return compensatedSum(terms) / total;
}

/** Whether a sorted label set is one of {0, 1}, {-1, 1}, {0}, {-1} or {1}. */
function isIndicatorLabeling(labels: readonly number[]): boolean {
  const first = labels[0];
  const second = labels[1];
  if (labels.length === 1) return first === 0 || first === 1 || first === -1;
  return second === 1 && (first === 0 || first === -1);
}

/**
 * Brier score loss for probability predictions.
 *
 * Brier = (1/n) * Σ (1[y_true = posLabel] - y_prob)²
 * Lower is better. `yTrue` holds at most two distinct labels; `yProb` holds the
 * predicted probability of the positive label.
 *
 * @param yTrue - True binary labels of shape (n_samples,)
 * @param yProb - Predicted probabilities of shape (n_samples,), each in [0, 1]
 * @param posLabel - Label treated as positive. Defaults to 1 for {0, 1} and {-1, 1}
 *   labels (and a lone 0, 1 or -1), otherwise to the largest label.
 * @returns Brier score in [0, 1]
 *
 * @throws {ShapeError} If inputs are not 1D or have different lengths
 * @throws {InvalidParameterError} If yTrue has more than two distinct labels or a
 *   probability lies outside [0, 1]
 * @throws {DataValidationError} If an input contains NaN or infinite values
 *
 * @example
 * ```ts
 * import { brierScoreLoss } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * brierScoreLoss(tensor([0, 1, 1, 0]), tensor([0.1, 0.9, 0.8, 0.2])); // 0.025
 * ```
 */
export function brierScoreLoss(yTrue: Tensor, yProb: Tensor, posLabel?: number): number {
  validateSameLength(yTrue, yProb, "brierScoreLoss");
  const n = yTrue.size;
  if (n === 0) return 0;

  const yt = readFiniteFloat64(yTrue, "yTrue");
  const yp = readFiniteFloat64(yProb, "yProb");

  for (let i = 0; i < n; i++) {
    const p = yp[i] as number;
    if (p < 0 || p > 1) {
      throw new InvalidParameterError("yProb must contain probabilities in [0, 1]", "yProb", p);
    }
  }

  const labels = [...new Set(yt)].sort((a, b) => a - b);
  if (labels.length > 2) {
    throw new InvalidParameterError(
      `brierScoreLoss requires binary yTrue; found ${labels.length} distinct labels`,
      "yTrue",
      labels.length
    );
  }
  // Like scikit-learn: {0, 1}, {-1, 1} and a lone 0, 1 or -1 mean "positive is 1";
  // any other labeling takes its largest label as the positive class.
  const positive =
    posLabel ?? (isIndicatorLabeling(labels) ? 1 : (labels[labels.length - 1] as number));

  const terms = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const diff = ((yt[i] as number) === positive ? 1 : 0) - (yp[i] as number);
    terms[i] = diff * diff;
  }
  return compensatedSum(terms) / n;
}

/**
 * Average hinge loss (for SVM evaluation).
 *
 * hinge_loss = (1/n) * Σ max(0, 1 - y_true * y_decision)
 *
 * `yTrue` holds the two classes; the larger label is the positive class (+1) and the
 * smaller one the negative class (-1), so {-1, +1} and {0, 1} labels both work.
 *
 * @param yTrue - True binary labels of shape (n_samples,)
 * @param yDecision - Decision function values of shape (n_samples,)
 * @returns Mean hinge loss
 *
 * @throws {ShapeError} If inputs are not 1D or have different lengths
 * @throws {InvalidParameterError} If yTrue has more than two distinct labels, or a single
 *   label other than -1 or +1
 * @throws {DataValidationError} If an input contains NaN or infinite values
 *
 * @example
 * ```ts
 * import { hingeLoss } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * hingeLoss(tensor([0, 1, 1, 0]), tensor([-1, 2, 0.2, 0.3])); // 0.525
 * ```
 */
export function hingeLoss(yTrue: Tensor, yDecision: Tensor): number {
  validateSameLength(yTrue, yDecision, "hingeLoss");
  const n = yTrue.size;
  if (n === 0) return 0;

  const yt = readFiniteFloat64(yTrue, "yTrue");
  const yd = readFiniteFloat64(yDecision, "yDecision");

  const labels = [...new Set(yt)].sort((a, b) => a - b);
  if (labels.length > 2) {
    throw new InvalidParameterError(
      `hingeLoss requires binary yTrue; found ${labels.length} distinct labels`,
      "yTrue",
      labels.length
    );
  }
  let positive: number;
  if (labels.length === 2) {
    positive = labels[1] as number;
  } else {
    const only = labels[0] as number;
    if (only !== -1 && only !== 1) {
      throw new InvalidParameterError(
        `yTrue holds the single label ${only}; hingeLoss needs both classes or labels in {-1, +1}`,
        "yTrue",
        only
      );
    }
    positive = 1;
  }

  const terms = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const sign = (yt[i] as number) === positive ? 1 : -1;
    terms[i] = Math.max(0, 1 - sign * (yd[i] as number));
  }
  return compensatedSum(terms) / n;
}

/** Read class labels of any non-string or string tensor as comparable primitives. */
function readLabels(t: Tensor, name: string): Array<number | string> {
  if (t.dtype === "string") {
    const offsetter = createFlatOffsetter(t);
    const data = t.data as string[];
    const out: string[] = new Array<string>(t.size);
    for (let i = 0; i < out.length; i++) out[i] = data[offsetter(i)] as string;
    return out;
  }
  return Array.from(readFiniteFloat64(t, name));
}

/**
 * Zero-one classification loss (fraction of misclassifications).
 *
 * @param yTrue - True labels of shape (n_samples,); numeric, bool or string
 * @param yPred - Predicted labels of shape (n_samples,)
 * @param options - Optional parameters
 * @param options.normalize - Return the fraction of misclassified samples (default: true)
 *   or their count (false)
 * @returns Fraction of misclassifications in [0, 1], or their count when `normalize` is false
 *
 * @throws {ShapeError} If inputs are not 1D or have different lengths
 * @throws {DataValidationError} If a numeric input contains NaN or infinite values
 *
 * @example
 * ```ts
 * import { zeroOneLoss } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * zeroOneLoss(tensor([0, 1, 2, 3]), tensor([0, 1, 0, 3])); // 0.25
 * zeroOneLoss(tensor([0, 1, 2, 3]), tensor([0, 1, 0, 3]), { normalize: false }); // 1
 * ```
 */
export function zeroOneLoss(
  yTrue: Tensor,
  yPred: Tensor,
  options?: { normalize?: boolean }
): number {
  validateSameLength(yTrue, yPred, "zeroOneLoss");
  const n = yTrue.size;
  const normalize = options?.normalize ?? true;
  if (n === 0) return 0;

  const yt = readLabels(yTrue, "yTrue");
  const yp = readLabels(yPred, "yPred");

  let wrong = 0;
  for (let i = 0; i < n; i++) {
    if (yt[i] !== yp[i]) wrong++;
  }
  return normalize ? wrong / n : wrong;
}

/**
 * Top-K accuracy score.
 *
 * A prediction is correct if the true label is among the K classes with the highest
 * scores. Ties are resolved like scikit-learn: among equal scores the class with the
 * larger index ranks first.
 *
 * @param yTrue - True labels of shape (n_samples,), integer class indices
 * @param yScore - Predicted scores of shape (n_samples, n_classes)
 * @param k - Number of top predictions to consider (default: 5, capped at n_classes)
 * @returns Top-K accuracy in [0, 1]
 *
 * @throws {ShapeError} If yTrue is not 1D, yScore is not 2D, or the sample counts differ
 * @throws {InvalidParameterError} If k is not an integer in [1, n_classes], or a label is not
 *   an integer class index in [0, n_classes)
 * @throws {DataValidationError} If an input contains NaN or infinite values
 *
 * @example
 * ```ts
 * import { topKAccuracyScore } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([0, 1, 2]);
 * const yScore = tensor([
 *   [0.1, 0.5, 0.4],
 *   [0.1, 0.8, 0.1],
 *   [0.1, 0.1, 0.8],
 * ]);
 * topKAccuracyScore(yTrue, yScore, 2); // 2 / 3
 * ```
 */
export function topKAccuracyScore(yTrue: Tensor, yScore: Tensor, k?: number): number {
  if (yTrue.ndim !== 1) {
    throw new ShapeError(`topKAccuracyScore: yTrue must be 1D; got ndim=${yTrue.ndim}`);
  }
  if (yScore.ndim !== 2) {
    throw new ShapeError(`topKAccuracyScore: yScore must be 2D; got ndim=${yScore.ndim}`);
  }
  const n = yTrue.size;
  const nClasses = yScore.shape[1] ?? 0;
  if ((yScore.shape[0] ?? 0) !== n) {
    throw new ShapeError(`topKAccuracyScore: yTrue and yScore must have same n_samples`);
  }
  const topK = k ?? Math.min(5, nClasses);
  if (!Number.isInteger(topK) || topK < 1 || topK > nClasses) {
    throw new InvalidParameterError(
      `k must be an integer in [1, ${nClasses}]; got ${String(k)}`,
      "k",
      k
    );
  }
  if (n === 0) return 0;

  const labels = readFiniteFloat64(yTrue, "yTrue");
  const scores = readFiniteFloat64(yScore, "yScore");

  let correct = 0;
  for (let i = 0; i < n; i++) {
    const label = labels[i] as number;
    if (!Number.isInteger(label) || label < 0 || label >= nClasses) {
      throw new InvalidParameterError(
        `yTrue must contain class indices in [0, ${nClasses}); got ${label} at index ${i}`,
        "yTrue",
        label
      );
    }
    const row = i * nClasses;
    const trueScore = scores[row + label] as number;
    // Number of classes that rank ahead of the true class.
    let ahead = 0;
    for (let j = 0; j < nClasses; j++) {
      const s = scores[row + j] as number;
      if (s > trueScore || (s === trueScore && j > label)) ahead++;
    }
    if (ahead < topK) correct++;
  }

  return correct / n;
}

/**
 * Evaluates f(u) = (1 + u) * log(1 + u) - u without cancellation for small |u|.
 * The Poisson unit deviance is mu * f((y - mu) / mu).
 */
function poissonKernel(u: number): number {
  if (Math.abs(u) >= 0.1) return (1 + u) * Math.log1p(u) - u;
  // f(u) = sum_{k>=2} (-1)^k u^k / (k (k - 1))
  let sum = 0;
  let pow = u * u;
  let sign = 1;
  for (let k = 2; k < 60; k++) {
    const term = (sign * pow) / (k * (k - 1));
    sum += term;
    if (Math.abs(term) <= 1e-17 * Math.abs(sum)) break;
    pow *= u;
    sign = -sign;
  }
  return sum;
}

/**
 * Evaluates g(u) = u - log(1 + u) without cancellation for small |u|.
 * The Gamma unit deviance is g((y - mu) / mu).
 */
function gammaKernel(u: number): number {
  if (Math.abs(u) >= 0.1) return u - Math.log1p(u);
  // g(u) = sum_{k>=2} (-1)^k u^k / k
  let sum = 0;
  let pow = u * u;
  let sign = 1;
  for (let k = 2; k < 60; k++) {
    const term = (sign * pow) / k;
    sum += term;
    if (Math.abs(term) <= 1e-17 * Math.abs(sum)) break;
    pow *= u;
    sign = -sign;
  }
  return sum;
}

/** Half the Poisson unit deviance: y log(y / mu) - y + mu. */
function halfPoissonDeviance(y: number, mu: number): number {
  if (y === 0) return mu;
  const u = (y - mu) / mu;
  // For a huge ratio y / mu the kernel overflows although the deviance is finite.
  if (!(Math.abs(u) < 1e100)) return y * (Math.log(y) - Math.log(mu)) - y + mu;
  return mu * poissonKernel(u);
}

/** Half the Gamma unit deviance: log(mu / y) + y / mu - 1. */
function halfGammaDeviance(y: number, mu: number): number {
  const u = (y - mu) / mu;
  // y / mu beyond the double range: the deviance exceeds it as well.
  return Number.isFinite(u) ? gammaKernel(u) : Number.POSITIVE_INFINITY;
}

/**
 * Mean Poisson deviance regression loss.
 *
 * Poisson deviance = (2/n) * Σ (y_true * log(y_true / y_pred) - y_true + y_pred)
 * Requires y_pred > 0 and y_true >= 0. The terms are evaluated without the
 * cancellation a direct evaluation suffers when predictions are close to the targets.
 *
 * @param yTrue - True non-negative target values of shape (n_samples,)
 * @param yPred - Predicted positive values of shape (n_samples,)
 * @returns Mean Poisson deviance (lower is better, 0 is perfect)
 *
 * @throws {ShapeError} If inputs are not 1D or have different lengths
 * @throws {InvalidParameterError} If yTrue has a negative value or yPred a non-positive value
 * @throws {DataValidationError} If an input contains NaN or infinite values
 */
export function meanPoissonDeviance(yTrue: Tensor, yPred: Tensor): number {
  validateSameLength(yTrue, yPred, "meanPoissonDeviance");
  const n = yTrue.size;
  if (n === 0) return 0;

  const yt = readFiniteFloat64(yTrue, "yTrue");
  const yp = readFiniteFloat64(yPred, "yPred");

  const terms = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const t = yt[i] as number;
    const p = yp[i] as number;
    if (t < 0) {
      throw new InvalidParameterError("meanPoissonDeviance requires y_true >= 0", "yTrue", t);
    }
    if (p <= 0) {
      throw new InvalidParameterError("meanPoissonDeviance requires y_pred > 0", "yPred", p);
    }
    terms[i] = 2 * halfPoissonDeviance(t, p);
  }
  return compensatedSum(terms) / n;
}

/**
 * Mean Gamma deviance regression loss.
 *
 * Gamma deviance = (2/n) * Σ (log(y_pred / y_true) + y_true / y_pred - 1)
 * Requires y_pred > 0 and y_true > 0. The terms are evaluated without the
 * cancellation a direct evaluation suffers when predictions are close to the targets.
 *
 * @param yTrue - True positive target values of shape (n_samples,)
 * @param yPred - Predicted positive values of shape (n_samples,)
 * @returns Mean Gamma deviance (lower is better, 0 is perfect)
 *
 * @throws {ShapeError} If inputs are not 1D or have different lengths
 * @throws {InvalidParameterError} If a value is not positive
 * @throws {DataValidationError} If an input contains NaN or infinite values
 */
export function meanGammaDeviance(yTrue: Tensor, yPred: Tensor): number {
  validateSameLength(yTrue, yPred, "meanGammaDeviance");
  const n = yTrue.size;
  if (n === 0) return 0;

  const yt = readFiniteFloat64(yTrue, "yTrue");
  const yp = readFiniteFloat64(yPred, "yPred");

  const terms = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const t = yt[i] as number;
    const p = yp[i] as number;
    if (t <= 0) {
      throw new InvalidParameterError("meanGammaDeviance requires y_true > 0", "yTrue", t);
    }
    if (p <= 0) {
      throw new InvalidParameterError("meanGammaDeviance requires y_pred > 0", "yPred", p);
    }
    terms[i] = 2 * halfGammaDeviance(t, p);
  }
  return compensatedSum(terms) / n;
}

/** One-vs-rest confusion counts of a single class label. */
export type MultilabelConfusionMatrixEntry = {
  label: number;
  tn: number;
  fp: number;
  fn: number;
  tp: number;
};

/**
 * Compute a one-vs-rest confusion matrix for each class label.
 *
 * For each label, returns the counts of a 2x2 confusion matrix:
 * [[TN, FP], [FN, TP]], treating that label as the positive class.
 *
 * @param yTrue - True labels of shape (n_samples,)
 * @param yPred - Predicted labels of shape (n_samples,)
 * @param labels - Optional label values to report, in the order returned. Defaults to
 *   the sorted union of the labels in `yTrue` and `yPred`.
 * @returns Array of { label, tn, fp, fn, tp } for each label
 *
 * @throws {ShapeError} If inputs are not 1D or have different lengths
 * @throws {InvalidParameterError} If `labels` contains a non-finite value
 * @throws {DataValidationError} If an input contains NaN or infinite values
 *
 * @example
 * ```ts
 * import { multilabelConfusionMatrix } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * multilabelConfusionMatrix(tensor([0, 1, 2, 1]), tensor([0, 2, 2, 1]), [2, 0, 1]);
 * // [{ label: 2, tn: 2, fp: 1, fn: 0, tp: 1 }, { label: 0, tn: 3, fp: 0, fn: 0, tp: 1 }, ...]
 * ```
 */
export function multilabelConfusionMatrix(
  yTrue: Tensor,
  yPred: Tensor,
  labels?: number[]
): MultilabelConfusionMatrixEntry[] {
  validateSameLength(yTrue, yPred, "multilabelConfusionMatrix");
  const n = yTrue.size;

  const yt = readFiniteFloat64(yTrue, "yTrue");
  const yp = readFiniteFloat64(yPred, "yPred");

  let labelList: number[];
  if (labels) {
    for (const l of labels) {
      if (!Number.isFinite(l)) {
        throw new InvalidParameterError("labels must contain finite numbers", "labels", l);
      }
    }
    labelList = [...new Set(labels)];
  } else {
    const seen = new Set<number>(yt);
    for (let i = 0; i < n; i++) seen.add(yp[i] as number);
    labelList = [...seen].sort((a, b) => a - b);
  }

  const position = new Map<number, number>();
  labelList.forEach((l, idx) => {
    position.set(l, idx);
  });
  const nLabels = labelList.length;
  const tp = new Float64Array(nLabels);
  const trueCount = new Float64Array(nLabels);
  const predCount = new Float64Array(nLabels);
  for (let i = 0; i < n; i++) {
    const ti = position.get(yt[i] as number);
    const pi = position.get(yp[i] as number);
    if (ti !== undefined) trueCount[ti]! += 1;
    if (pi !== undefined) predCount[pi]! += 1;
    if (ti !== undefined && ti === pi) tp[ti]! += 1;
  }

  return labelList.map((label, idx) => {
    const hit = tp[idx] as number;
    const fp = (predCount[idx] as number) - hit;
    const fn = (trueCount[idx] as number) - hit;
    return { label, tn: n - hit - fp - fn, fp, fn, tp: hit };
  });
}

/** False positive and false negative rates of the DET curve at each threshold. */
export type DetCurveResult = { fpr: number[]; fnr: number[]; thresholds: number[] };

/**
 * Detection Error Tradeoff (DET) curve.
 *
 * Returns false positive rate (FPR) and false negative rate (FNR) at each distinct
 * score threshold, with thresholds in decreasing order. A sample is predicted
 * positive when its score is greater than or equal to the threshold. Unlike ROC
 * curves, both axes represent error rates.
 *
 * Returns empty arrays when `yTrue` is empty or contains only one class.
 *
 * @param yTrue - True binary labels (0 or 1) of shape (n_samples,)
 * @param yScore - Target scores of shape (n_samples,)
 * @returns `{ fpr, fnr, thresholds }` arrays of equal length
 *
 * @throws {ShapeError} If inputs are not 1D or have different lengths
 * @throws {InvalidParameterError} If yTrue holds a label other than 0 or 1
 * @throws {DataValidationError} If an input contains NaN or infinite values
 *
 * @example
 * ```ts
 * import { detCurve } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const { fpr, fnr, thresholds } = detCurve(tensor([0, 0, 1, 1]), tensor([0.1, 0.4, 0.35, 0.8]));
 * // thresholds: [0.8, 0.4, 0.35, 0.1]
 * ```
 */
export function detCurve(yTrue: Tensor, yScore: Tensor): DetCurveResult {
  validateSameLength(yTrue, yScore, "detCurve");
  const n = yTrue.size;
  if (n === 0) return { fpr: [], fnr: [], thresholds: [] };

  const yt = readFiniteFloat64(yTrue, "yTrue");
  const ys = readFiniteFloat64(yScore, "yScore");

  let nPos = 0;
  let nNeg = 0;
  for (let i = 0; i < n; i++) {
    const label = yt[i] as number;
    if (label !== 0 && label !== 1) {
      throw new InvalidParameterError("detCurve requires binary labels (0 or 1)", "yTrue", label);
    }
    if (label === 1) nPos++;
    else nNeg++;
  }

  if (nPos === 0 || nNeg === 0) return { fpr: [], fnr: [], thresholds: [] };

  // Sort sample indices by score, highest first.
  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => (ys[b] as number) - (ys[a] as number));

  const fpr: number[] = [];
  const fnr: number[] = [];
  const thresholds: number[] = [];

  let tp = 0;
  let fp = 0;
  let idx = 0;
  while (idx < n) {
    const threshold = ys[order[idx] as number] as number;

    // Consume all samples with the same score
    while (idx < n && ys[order[idx] as number] === threshold) {
      if (yt[order[idx] as number] === 1) tp++;
      else fp++;
      idx++;
    }

    fpr.push(fp / nNeg);
    fnr.push(1 - tp / nPos);
    thresholds.push(threshold);
  }

  return { fpr, fnr, thresholds };
}

/**
 * D² score for Tweedie deviance regression.
 *
 * D² = 1 - deviance(y_true, y_pred) / deviance(y_true, y_mean)
 *
 * Generalizes R² to Tweedie distributions. The `power` parameter selects
 * the distribution family:
 * - power=0: Normal (equivalent to R²)
 * - power=1: Poisson
 * - power=2: Gamma
 * - power=3: Inverse Gaussian
 *
 * Domain: y_pred must be positive for every power other than 0. For power in [1, 2)
 * y_true must be non-negative, and for power >= 2 it must be positive.
 *
 * @param yTrue - True values of shape (n_samples,)
 * @param yPred - Predicted values of shape (n_samples,)
 * @param power - Tweedie power parameter (default: 0)
 * @returns D² score (1 is perfect, 0 is baseline)
 *
 * @throws {ShapeError} If inputs have different lengths or are not 1D
 * @throws {InvalidParameterError} If power is in (0, 1), the inputs are empty, or values lie
 *   outside the domain of the chosen power
 * @throws {DataValidationError} If an input contains NaN or infinite values
 */
export function d2TweedieScore(yTrue: Tensor, yPred: Tensor, power: number = 0): number {
  validateSameLength(yTrue, yPred, "d2TweedieScore");
  const n = yTrue.size;
  if (n === 0) {
    throw new InvalidParameterError("d2TweedieScore requires at least one sample", "yTrue", n);
  }
  if (!Number.isFinite(power)) {
    throw new InvalidParameterError("power must be finite", "power", power);
  }
  if (power > 0 && power < 1) {
    throw new InvalidParameterError(
      "power must be <= 0 or >= 1 for Tweedie deviance",
      "power",
      power
    );
  }

  const yt = readFiniteFloat64(yTrue, "yTrue");
  const yp = readFiniteFloat64(yPred, "yPred");

  const devModel = tweedieDeviance(yt, yp, power, n);

  const yMean = compensatedSum(yt) / n;
  if (power !== 0 && yMean <= 0) {
    throw new InvalidParameterError(
      `yTrue must have a positive mean for Tweedie power=${power}`,
      "yTrue",
      yMean
    );
  }
  const devNull = tweedieDeviance(yt, new Float64Array(n).fill(yMean), power, n);

  if (devNull === 0) {
    return devModel === 0 ? 1.0 : 0.0;
  }

  return 1 - devModel / devNull;
}

/**
 * Half the Tweedie unit deviance for a power outside {0, 1, 2}:
 * y^q / ((1 - p) q) - y mu^(1 - p) / (1 - p) + mu^q / q with q = 2 - p.
 *
 * With u = (y - mu) / mu this equals mu^q * ((1 + u)^q - 1 - q u) / (q (q - 1)), whose
 * Taylor series avoids the cancellation of the direct form when y is close to mu.
 */
function halfTweedieDeviance(y: number, mu: number, p: number): number {
  const q = 2 - p;
  const u = (y - mu) / mu;
  if (Math.abs(u) < 0.1) {
    // sum_{k>=2} c_k u^k with c_2 = 1/2 and c_(k+1) = c_k (q - k) / (k + 1)
    let coef = 0.5;
    let pow = u * u;
    let sum = coef * pow;
    for (let k = 2; k < 60; k++) {
      coef *= (q - k) / (k + 1);
      pow *= u;
      const term = coef * pow;
      sum += term;
      if (Math.abs(term) <= 1e-17 * Math.abs(sum)) break;
    }
    return mu ** q * sum;
  }
  // For power < 0 negative targets are clipped at 0 in the first term.
  const term1 = y > 0 ? y ** q / ((1 - p) * q) : 0;
  const term2 = (y * mu ** (1 - p)) / (1 - p);
  const term3 = mu ** q / q;
  return term1 - term2 + term3;
}

/** Total (not mean) Tweedie deviance, validating the domain of `power`. */
function tweedieDeviance(
  yTrue: Float64Array,
  yPred: Float64Array,
  power: number,
  n: number
): number {
  const terms = new Float64Array(n);
  if (power === 0) {
    // Normal: (y - mu)^2
    for (let i = 0; i < n; i++) {
      const diff = (yTrue[i] as number) - (yPred[i] as number);
      terms[i] = diff * diff;
    }
  } else if (power === 1) {
    // Poisson: 2 * (y*log(y/mu) - (y - mu))
    for (let i = 0; i < n; i++) {
      const y = yTrue[i] as number;
      const mu = yPred[i] as number;
      if (mu <= 0) {
        throw new InvalidParameterError(
          "yPred must be positive for Tweedie power=1 (Poisson)",
          "yPred",
          mu
        );
      }
      if (y < 0) {
        throw new InvalidParameterError(
          "yTrue must be non-negative for Tweedie power=1 (Poisson)",
          "yTrue",
          y
        );
      }
      terms[i] = 2 * halfPoissonDeviance(y, mu);
    }
  } else if (power === 2) {
    // Gamma: 2 * (-log(y/mu) + (y-mu)/mu)
    for (let i = 0; i < n; i++) {
      const y = yTrue[i] as number;
      const mu = yPred[i] as number;
      if (y <= 0 || mu <= 0) {
        throw new InvalidParameterError(
          "yTrue and yPred must be positive for Tweedie power=2 (Gamma)",
          "yTrue/yPred",
          { y, mu }
        );
      }
      terms[i] = 2 * halfGammaDeviance(y, mu);
    }
  } else {
    // General Tweedie: 2 * (y^(2-p)/((1-p)*(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p))
    const p = power;
    for (let i = 0; i < n; i++) {
      const y = yTrue[i] as number;
      const mu = yPred[i] as number;
      if (mu <= 0) {
        throw new InvalidParameterError(
          `yPred must be positive for Tweedie power=${p}`,
          "yPred",
          mu
        );
      }
      if (p > 1 && p < 2 && y < 0) {
        throw new InvalidParameterError(
          `yTrue must be non-negative for Tweedie power=${p}`,
          "yTrue",
          y
        );
      }
      if (p > 2 && y <= 0) {
        throw new InvalidParameterError(
          `yTrue must be positive for Tweedie power=${p}`,
          "yTrue",
          y
        );
      }
      terms[i] = 2 * halfTweedieDeviance(y, mu, p);
    }
  }
  return compensatedSum(terms);
}

/**
 * Mean pinball loss (quantile loss).
 *
 * For a given quantile alpha:
 * L(y, q) = alpha * max(y - q, 0) + (1 - alpha) * max(q - y, 0)
 *
 * The pinball loss is the average of the quantile-specific losses.
 * It is minimized by the alpha-quantile of the distribution.
 *
 * @param yTrue - True values of shape (n_samples,)
 * @param yPred - Predicted quantile values of shape (n_samples,)
 * @param alpha - Quantile level in (0, 1), default 0.5 (median)
 * @returns Mean pinball loss (non-negative)
 *
 * @throws {ShapeError} If inputs have different lengths or are not 1D
 * @throws {InvalidParameterError} If alpha is not in (0, 1)
 * @throws {DataValidationError} If an input contains NaN or infinite values
 */
export function meanPinballLoss(yTrue: Tensor, yPred: Tensor, alpha: number = 0.5): number {
  validateSameLength(yTrue, yPred, "meanPinballLoss");
  if (!Number.isFinite(alpha) || alpha <= 0 || alpha >= 1) {
    throw new InvalidParameterError("alpha must be in (0, 1)", "alpha", alpha);
  }
  const n = yTrue.size;
  if (n === 0) return 0;

  const yt = readFiniteFloat64(yTrue, "yTrue");
  const yp = readFiniteFloat64(yPred, "yPred");

  const terms = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const diff = (yt[i] as number) - (yp[i] as number);
    terms[i] = diff >= 0 ? alpha * diff : (1 - alpha) * -diff;
  }
  return compensatedSum(terms) / n;
}

/** Row-major [n_samples, n_labels] data read from a 2D tensor. */
function read2D(
  t: Tensor,
  name: string,
  fn: string
): { data: Float64Array; nSamples: number; nLabels: number } {
  if (t.ndim !== 2) {
    throw new ShapeError(`${fn}: ${name} must be 2D; got ndim=${t.ndim}`);
  }
  return {
    data: readFiniteFloat64(t, name),
    nSamples: t.shape[0] ?? 0,
    nLabels: t.shape[1] ?? 0,
  };
}

/** Read an indicator matrix and a score matrix of the same shape. */
function readIndicatorAndScores(
  yTrue: Tensor,
  yScore: Tensor,
  fn: string
): { indicator: Float64Array; scores: Float64Array; nSamples: number; nLabels: number } {
  const truth = read2D(yTrue, "yTrue", fn);
  const score = read2D(yScore, "yScore", fn);
  if (truth.nSamples !== score.nSamples || truth.nLabels !== score.nLabels) {
    throw new ShapeError(
      `${fn}: shape mismatch [${truth.nSamples},${truth.nLabels}] vs [${score.nSamples},${score.nLabels}]`
    );
  }
  for (let i = 0; i < truth.data.length; i++) {
    const v = truth.data[i] as number;
    if (v !== 0 && v !== 1) {
      throw new InvalidParameterError(
        `${fn}: yTrue must be a binary indicator matrix (0 or 1); got ${v}`,
        "yTrue",
        v
      );
    }
  }
  return {
    indicator: truth.data,
    scores: score.data,
    nSamples: truth.nSamples,
    nLabels: truth.nLabels,
  };
}

/**
 * Coverage error for multi-label ranking.
 *
 * Compute how far we need to go through the ranked scores to cover all
 * true labels. The best value is equal to the average number of labels
 * per sample.
 *
 * coverage(y, f) = (1/n) * Σ_i max_{j: y_ij=1} rank_ij
 *
 * where rank_ij is the number of labels scored at least as high as label j in
 * sample i (highest score = rank 1). Labels that tie with a true label count
 * against the ranking, as in scikit-learn. Samples without a true label contribute 0.
 *
 * @param yTrue - Binary label indicators of shape (n_samples, n_labels)
 * @param yScore - Target scores of shape (n_samples, n_labels)
 * @returns Coverage error (lower is better, minimum = avg number of true labels)
 *
 * @throws {ShapeError} If an input is not 2D or the shapes differ
 * @throws {InvalidParameterError} If yTrue holds a value other than 0 or 1
 * @throws {DataValidationError} If an input contains NaN or infinite values
 *
 * @example
 * ```ts
 * import { coverageError } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * coverageError(tensor([[1, 0, 0], [0, 1, 1]]), tensor([[0.5, 0.5, 0.5], [0.2, 0.2, 0.2]])); // 3
 * ```
 */
export function coverageError(yTrue: Tensor, yScore: Tensor): number {
  const { indicator, scores, nSamples, nLabels } = readIndicatorAndScores(
    yTrue,
    yScore,
    "coverageError"
  );
  if (nSamples === 0) return 0;

  let totalCoverage = 0;
  for (let i = 0; i < nSamples; i++) {
    const row = i * nLabels;

    // Lowest score among the true labels; coverage counts every label at or above it.
    let minRelevant = Number.POSITIVE_INFINITY;
    for (let j = 0; j < nLabels; j++) {
      if (indicator[row + j] === 1) {
        const s = scores[row + j] as number;
        if (s < minRelevant) minRelevant = s;
      }
    }
    if (minRelevant === Number.POSITIVE_INFINITY) continue;

    let covered = 0;
    for (let j = 0; j < nLabels; j++) {
      if ((scores[row + j] as number) >= minRelevant) covered++;
    }
    totalCoverage += covered;
  }

  return totalCoverage / nSamples;
}

/**
 * Label ranking loss for multi-label classification.
 *
 * Computes the average fraction of label pairs that are incorrectly ordered.
 * A pair (i, j) is incorrectly ordered if y_true[i]=1, y_true[j]=0 but
 * score[i] <= score[j]; tied scores therefore count as errors. Samples with no
 * true label or with every label true contribute 0.
 *
 * Best value is 0.
 *
 * @param yTrue - Binary label indicators of shape (n_samples, n_labels)
 * @param yScore - Target scores of shape (n_samples, n_labels)
 * @returns Label ranking loss in [0, 1]
 *
 * @throws {ShapeError} If an input is not 2D or the shapes differ
 * @throws {InvalidParameterError} If yTrue holds a value other than 0 or 1
 * @throws {DataValidationError} If an input contains NaN or infinite values
 *
 * @example
 * ```ts
 * import { labelRankingLoss } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * labelRankingLoss(tensor([[1, 1, 0]]), tensor([[0.9, 0.1, 0.5]])); // 0.5
 * ```
 */
export function labelRankingLoss(yTrue: Tensor, yScore: Tensor): number {
  const { indicator, scores, nSamples, nLabels } = readIndicatorAndScores(
    yTrue,
    yScore,
    "labelRankingLoss"
  );
  if (nSamples === 0) return 0;

  let totalLoss = 0;
  const negatives = new Float64Array(nLabels);
  const positives = new Float64Array(nLabels);

  for (let i = 0; i < nSamples; i++) {
    const row = i * nLabels;
    let nPos = 0;
    let nNeg = 0;
    for (let j = 0; j < nLabels; j++) {
      const s = scores[row + j] as number;
      if (indicator[row + j] === 1) positives[nPos++] = s;
      else negatives[nNeg++] = s;
    }
    if (nPos === 0 || nNeg === 0) continue;

    // Count (positive, negative) pairs with score_pos <= score_neg using the sorted negatives.
    const sortedNeg = negatives.subarray(0, nNeg).sort();
    let nIncorrect = 0;
    for (let p = 0; p < nPos; p++) {
      const s = positives[p] as number;
      // First index with a negative score >= s.
      let lo = 0;
      let hi = nNeg;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if ((sortedNeg[mid] as number) < s) lo = mid + 1;
        else hi = mid;
      }
      nIncorrect += nNeg - lo;
    }

    totalLoss += nIncorrect / (nPos * nNeg);
  }

  return totalLoss / nSamples;
}
