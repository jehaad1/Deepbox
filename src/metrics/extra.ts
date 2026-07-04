/**
 * Additional metrics: SMAPE, MSLE, Brier score, hinge loss, zero-one loss, top-k accuracy.
 *
 * @module metrics/extra
 * @see {@link https://deepbox.dev/docs/metrics-clustering | Deepbox Metrics}
 */

import { InvalidParameterError, ShapeError } from "../core";
import type { Tensor } from "../ndarray";

function extractArray(t: Tensor): number[] {
  const arr: number[] = [];
  for (let i = 0; i < t.size; i++) {
    arr.push(Number(t.data[t.offset + i]));
  }
  return arr;
}

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
 *
 * @param yTrue - True values of shape (n_samples,)
 * @param yPred - Predicted values of shape (n_samples,)
 * @returns SMAPE score in [0, 2]
 */
export function smape(yTrue: Tensor, yPred: Tensor): number {
  validateSameLength(yTrue, yPred, "smape");
  const n = yTrue.size;
  if (n === 0) return 0;

  const yt = extractArray(yTrue);
  const yp = extractArray(yPred);

  let sum = 0;
  for (let i = 0; i < n; i++) {
    const denom = (Math.abs(yt[i]!) + Math.abs(yp[i]!)) / 2;
    if (denom === 0) continue;
    sum += Math.abs(yt[i]! - yp[i]!) / denom;
  }
  return sum / n;
}

/**
 * Mean Squared Logarithmic Error (MSLE).
 *
 * MSLE = (1/n) * Σ (log(1 + y_true) - log(1 + y_pred))²
 * Only valid for non-negative values.
 *
 * @param yTrue - True non-negative values of shape (n_samples,)
 * @param yPred - Predicted non-negative values of shape (n_samples,)
 * @returns MSLE score
 */
export function meanSquaredLogError(yTrue: Tensor, yPred: Tensor): number {
  validateSameLength(yTrue, yPred, "meanSquaredLogError");
  const n = yTrue.size;
  if (n === 0) return 0;

  const yt = extractArray(yTrue);
  const yp = extractArray(yPred);

  let sum = 0;
  for (let i = 0; i < n; i++) {
    if (yt[i]! < 0 || yp[i]! < 0) {
      throw new InvalidParameterError(
        "meanSquaredLogError requires non-negative values",
        "y",
        yt[i]! < 0 ? yt[i]! : yp[i]!
      );
    }
    const diff = Math.log1p(yt[i]!) - Math.log1p(yp[i]!);
    sum += diff * diff;
  }
  return sum / n;
}

/**
 * Brier score loss for probability predictions.
 *
 * Brier = (1/n) * Σ (y_true - y_prob)²
 * Lower is better. y_true should be binary (0 or 1), y_prob should be probabilities.
 *
 * @param yTrue - True binary labels of shape (n_samples,)
 * @param yProb - Predicted probabilities of shape (n_samples,)
 * @returns Brier score in [0, 1]
 */
export function brierScoreLoss(yTrue: Tensor, yProb: Tensor): number {
  validateSameLength(yTrue, yProb, "brierScoreLoss");
  const n = yTrue.size;
  if (n === 0) return 0;

  const yt = extractArray(yTrue);
  const yp = extractArray(yProb);

  let sum = 0;
  for (let i = 0; i < n; i++) {
    const diff = yt[i]! - yp[i]!;
    sum += diff * diff;
  }
  return sum / n;
}

/**
 * Average hinge loss (for SVM evaluation).
 *
 * hinge_loss = (1/n) * Σ max(0, 1 - y_true * y_decision)
 * y_true should be in {-1, +1}.
 *
 * @param yTrue - True labels in {-1, +1} of shape (n_samples,)
 * @param yDecision - Decision function values of shape (n_samples,)
 * @returns Mean hinge loss
 */
export function hingeLoss(yTrue: Tensor, yDecision: Tensor): number {
  validateSameLength(yTrue, yDecision, "hingeLoss");
  const n = yTrue.size;
  if (n === 0) return 0;

  const yt = extractArray(yTrue);
  const yd = extractArray(yDecision);

  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += Math.max(0, 1 - yt[i]! * yd[i]!);
  }
  return sum / n;
}

/**
 * Zero-one classification loss (fraction of misclassifications).
 *
 * @param yTrue - True labels of shape (n_samples,)
 * @param yPred - Predicted labels of shape (n_samples,)
 * @returns Fraction of misclassifications in [0, 1]
 */
export function zeroOneLoss(yTrue: Tensor, yPred: Tensor): number {
  validateSameLength(yTrue, yPred, "zeroOneLoss");
  const n = yTrue.size;
  if (n === 0) return 0;

  const yt = extractArray(yTrue);
  const yp = extractArray(yPred);

  let wrong = 0;
  for (let i = 0; i < n; i++) {
    if (yt[i] !== yp[i]) wrong++;
  }
  return wrong / n;
}

/**
 * Top-K accuracy score.
 *
 * A prediction is correct if the true label is among the top K predicted classes.
 *
 * @param yTrue - True labels of shape (n_samples,), integer class indices
 * @param yScore - Predicted scores of shape (n_samples, n_classes)
 * @param k - Number of top predictions to consider (default: 5)
 * @returns Top-K accuracy in [0, 1]
 */
export function topKAccuracyScore(yTrue: Tensor, yScore: Tensor, k = 5): number {
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
  if (k < 1 || k > nClasses) {
    throw new InvalidParameterError(`k must be in [1, ${nClasses}]; got ${k}`, "k", k);
  }

  let correct = 0;
  for (let i = 0; i < n; i++) {
    const trueLabel = Number(yTrue.data[yTrue.offset + i]);
    // Get scores for this sample and find top K indices
    const scores: Array<{ idx: number; score: number }> = [];
    for (let j = 0; j < nClasses; j++) {
      scores.push({
        idx: j,
        score: Number(yScore.data[yScore.offset + i * nClasses + j]),
      });
    }
    scores.sort((a, b) => b.score - a.score);
    for (let t = 0; t < k; t++) {
      if (scores[t]!.idx === trueLabel) {
        correct++;
        break;
      }
    }
  }

  return correct / n;
}

/**
 * Mean Poisson deviance regression loss.
 *
 * Poisson deviance = (2/n) * Σ (y_true * log(y_true / y_pred) - y_true + y_pred)
 * Requires y_pred > 0 and y_true >= 0.
 *
 * @param yTrue - True non-negative target values of shape (n_samples,)
 * @param yPred - Predicted positive values of shape (n_samples,)
 * @returns Mean Poisson deviance (lower is better, 0 is perfect)
 */
export function meanPoissonDeviance(yTrue: Tensor, yPred: Tensor): number {
  validateSameLength(yTrue, yPred, "meanPoissonDeviance");
  const n = yTrue.size;
  if (n === 0) return 0;

  const yt = extractArray(yTrue);
  const yp = extractArray(yPred);

  let sum = 0;
  for (let i = 0; i < n; i++) {
    const t = yt[i]!;
    const p = yp[i]!;
    if (t < 0) {
      throw new InvalidParameterError("meanPoissonDeviance requires y_true >= 0", "yTrue", t);
    }
    if (p <= 0) {
      throw new InvalidParameterError("meanPoissonDeviance requires y_pred > 0", "yPred", p);
    }
    // deviance_i = 2 * (y_true * log(y_true / y_pred) - y_true + y_pred)
    // When y_true == 0: deviance_i = 2 * y_pred
    if (t === 0) {
      sum += 2 * p;
    } else {
      sum += 2 * (t * Math.log(t / p) - t + p);
    }
  }
  return sum / n;
}

/**
 * Mean Gamma deviance regression loss.
 *
 * Gamma deviance = (2/n) * Σ (log(y_pred / y_true) + y_true / y_pred - 1)
 * Requires y_pred > 0 and y_true > 0.
 *
 * @param yTrue - True positive target values of shape (n_samples,)
 * @param yPred - Predicted positive values of shape (n_samples,)
 * @returns Mean Gamma deviance (lower is better, 0 is perfect)
 */
export function meanGammaDeviance(yTrue: Tensor, yPred: Tensor): number {
  validateSameLength(yTrue, yPred, "meanGammaDeviance");
  const n = yTrue.size;
  if (n === 0) return 0;

  const yt = extractArray(yTrue);
  const yp = extractArray(yPred);

  let sum = 0;
  for (let i = 0; i < n; i++) {
    const t = yt[i]!;
    const p = yp[i]!;
    if (t <= 0) {
      throw new InvalidParameterError("meanGammaDeviance requires y_true > 0", "yTrue", t);
    }
    if (p <= 0) {
      throw new InvalidParameterError("meanGammaDeviance requires y_pred > 0", "yPred", p);
    }
    sum += 2 * (Math.log(p / t) + t / p - 1);
  }
  return sum / n;
}

/**
 * Compute a confusion matrix for each class (multilabel).
 *
 * For each label, returns a 2x2 confusion matrix:
 * [[TN, FP], [FN, TP]]
 *
 * @param yTrue - True labels of shape (n_samples,)
 * @param yPred - Predicted labels of shape (n_samples,)
 * @param labels - Optional array of label values to include
 * @returns Array of { label, tn, fp, fn, tp } for each unique label
 */
export function multilabelConfusionMatrix(
  yTrue: Tensor,
  yPred: Tensor,
  labels?: number[]
): Array<{ label: number; tn: number; fp: number; fn: number; tp: number }> {
  validateSameLength(yTrue, yPred, "multilabelConfusionMatrix");
  const n = yTrue.size;

  const yt = extractArray(yTrue);
  const yp = extractArray(yPred);

  // Determine labels
  const labelSet = new Set<number>();
  if (labels) {
    for (const l of labels) labelSet.add(l);
  } else {
    for (let i = 0; i < n; i++) {
      labelSet.add(yt[i]!);
      labelSet.add(yp[i]!);
    }
  }
  const sortedLabels = [...labelSet].sort((a, b) => a - b);

  const result: Array<{
    label: number;
    tn: number;
    fp: number;
    fn: number;
    tp: number;
  }> = [];
  for (const label of sortedLabels) {
    let tp = 0,
      fp = 0,
      fn = 0,
      tn = 0;
    for (let i = 0; i < n; i++) {
      const isTrue = yt[i] === label;
      const isPred = yp[i] === label;
      if (isTrue && isPred) tp++;
      else if (!isTrue && isPred) fp++;
      else if (isTrue && !isPred) fn++;
      else tn++;
    }
    result.push({ label, tn, fp, fn, tp });
  }
  return result;
}

/**
 * Detection Error Tradeoff (DET) curve.
 *
 * Returns false positive rate (FPR) and false negative rate (FNR) at various thresholds.
 * Unlike ROC curves, both axes represent error rates.
 *
 * @param yTrue - True binary labels (0 or 1) of shape (n_samples,)
 * @param yScore - Target scores of shape (n_samples,)
 * @returns [fpr, fnr, thresholds] arrays
 */
export function detCurve(
  yTrue: Tensor,
  yScore: Tensor
): { fpr: number[]; fnr: number[]; thresholds: number[] } {
  validateSameLength(yTrue, yScore, "detCurve");
  const n = yTrue.size;
  if (n === 0) return { fpr: [], fnr: [], thresholds: [] };

  const yt = extractArray(yTrue);
  const ys = extractArray(yScore);

  // Count positives and negatives
  let nPos = 0;
  let nNeg = 0;
  const pairs: Array<{ score: number; label: number }> = [];
  for (let i = 0; i < n; i++) {
    const label = yt[i]!;
    if (label !== 0 && label !== 1) {
      throw new InvalidParameterError("detCurve requires binary labels (0 or 1)", "yTrue", label);
    }
    pairs.push({ score: ys[i]!, label });
    if (label === 1) nPos++;
    else nNeg++;
  }

  if (nPos === 0 || nNeg === 0) return { fpr: [], fnr: [], thresholds: [] };

  // Sort by score descending
  pairs.sort((a, b) => b.score - a.score);

  const fpr: number[] = [];
  const fnr: number[] = [];
  const thresholds: number[] = [];

  let tp = 0;
  let fp = 0;
  let idx = 0;
  while (idx < pairs.length) {
    const threshold = pairs[idx]!.score;

    // Consume all samples with the same score
    while (idx < pairs.length && pairs[idx]!.score === threshold) {
      if (pairs[idx]!.label === 1) tp++;
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
 * @param yTrue - True values of shape (n_samples,)
 * @param yPred - Predicted values of shape (n_samples,)
 * @param power - Tweedie power parameter (default: 0)
 * @returns D² score (1 is perfect, 0 is baseline)
 *
 * @throws {ShapeError} If inputs have different lengths or are not 1D
 * @throws {InvalidParameterError} If power is in (0, 1) or inputs are empty
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

  const yt = extractArray(yTrue);
  const yp = extractArray(yPred);

  // Compute mean of yTrue for the null model
  let sum = 0;
  for (let i = 0; i < n; i++) sum += yt[i]!;
  const yMean = sum / n;

  const devModel = tweedieDeviance(yt, yp, power, n);
  const yMeanArr = new Array<number>(n).fill(yMean);
  const devNull = tweedieDeviance(yt, yMeanArr, power, n);

  if (devNull === 0) {
    return devModel === 0 ? 1.0 : 0.0;
  }

  return 1 - devModel / devNull;
}

function tweedieDeviance(yTrue: number[], yPred: number[], power: number, n: number): number {
  let deviance = 0;
  if (power === 0) {
    // Normal: (y - mu)^2
    for (let i = 0; i < n; i++) {
      const diff = yTrue[i]! - yPred[i]!;
      deviance += diff * diff;
    }
  } else if (power === 1) {
    // Poisson: 2 * (y*log(y/mu) - (y - mu))
    for (let i = 0; i < n; i++) {
      const y = yTrue[i]!;
      const mu = yPred[i]!;
      if (mu <= 0) {
        throw new InvalidParameterError(
          "yPred must be positive for Tweedie power=1 (Poisson)",
          "yPred",
          mu
        );
      }
      const term = y > 0 ? y * Math.log(y / mu) : 0;
      deviance += 2 * (term - (y - mu));
    }
  } else if (power === 2) {
    // Gamma: 2 * (-log(y/mu) + (y-mu)/mu)
    for (let i = 0; i < n; i++) {
      const y = yTrue[i]!;
      const mu = yPred[i]!;
      if (y <= 0 || mu <= 0) {
        throw new InvalidParameterError(
          "yTrue and yPred must be positive for Tweedie power=2 (Gamma)",
          "yTrue/yPred",
          { y, mu }
        );
      }
      deviance += 2 * (-Math.log(y / mu) + (y - mu) / mu);
    }
  } else {
    // General Tweedie: 2 * (y^(2-p)/((1-p)*(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p))
    const p = power;
    for (let i = 0; i < n; i++) {
      const y = yTrue[i]!;
      const mu = yPred[i]!;
      if (mu <= 0) {
        throw new InvalidParameterError(
          `yPred must be positive for Tweedie power=${p}`,
          "yPred",
          mu
        );
      }
      const term1 = y > 0 ? y ** (2 - p) / ((1 - p) * (2 - p)) : 0;
      const term2 = (y * mu ** (1 - p)) / (1 - p);
      const term3 = mu ** (2 - p) / (2 - p);
      deviance += 2 * (term1 - term2 + term3);
    }
  }
  return deviance;
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
 */
export function meanPinballLoss(yTrue: Tensor, yPred: Tensor, alpha: number = 0.5): number {
  validateSameLength(yTrue, yPred, "meanPinballLoss");
  if (!Number.isFinite(alpha) || alpha <= 0 || alpha >= 1) {
    throw new InvalidParameterError("alpha must be in (0, 1)", "alpha", alpha);
  }
  const n = yTrue.size;
  if (n === 0) return 0;

  const yt = extractArray(yTrue);
  const yp = extractArray(yPred);

  let totalLoss = 0;
  for (let i = 0; i < n; i++) {
    const diff = yt[i]! - yp[i]!;
    if (diff >= 0) {
      totalLoss += alpha * diff;
    } else {
      totalLoss += (1 - alpha) * -diff;
    }
  }

  return totalLoss / n;
}

function extract2D(t: Tensor): {
  data: number[][];
  nSamples: number;
  nLabels: number;
} {
  if (t.ndim !== 2) {
    throw new ShapeError(`Expected 2D tensor; got ndim=${t.ndim}`);
  }
  const nSamples = t.shape[0] ?? 0;
  const nLabels = t.shape[1] ?? 0;
  const data: number[][] = [];
  for (let i = 0; i < nSamples; i++) {
    const row: number[] = [];
    for (let j = 0; j < nLabels; j++) {
      row.push(Number(t.data[t.offset + i * (t.strides[0] ?? 0) + j * (t.strides[1] ?? 0)]));
    }
    data.push(row);
  }
  return { data, nSamples, nLabels };
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
 * where rank_ij is the rank of score f_ij (highest score = rank 1).
 *
 * @param yTrue - Binary label indicators of shape (n_samples, n_labels)
 * @param yScore - Target scores of shape (n_samples, n_labels)
 * @returns Coverage error (lower is better, minimum = avg number of true labels)
 */
export function coverageError(yTrue: Tensor, yScore: Tensor): number {
  const { data: trueData, nSamples, nLabels } = extract2D(yTrue);
  const { data: scoreData, nSamples: nS2, nLabels: nL2 } = extract2D(yScore);

  if (nSamples !== nS2 || nLabels !== nL2) {
    throw new ShapeError(
      `coverageError: shape mismatch [${nSamples},${nLabels}] vs [${nS2},${nL2}]`
    );
  }
  if (nSamples === 0) return 0;

  let totalCoverage = 0;

  for (let i = 0; i < nSamples; i++) {
    const scores = scoreData[i]!;
    const labels = trueData[i]!;

    // Rank scores: highest score gets rank 1
    // Create index array sorted by score descending
    const indices = Array.from({ length: nLabels }, (_, k) => k);
    indices.sort((a, b) => (scores[b] ?? 0) - (scores[a] ?? 0));

    // Assign ranks (1-based)
    const ranks = new Array<number>(nLabels);
    for (let r = 0; r < nLabels; r++) {
      ranks[indices[r]!] = r + 1;
    }

    // Find max rank among true labels
    let maxRank = 0;
    for (let j = 0; j < nLabels; j++) {
      if ((labels[j] ?? 0) > 0) {
        const rank = ranks[j] ?? 0;
        if (rank > maxRank) maxRank = rank;
      }
    }
    totalCoverage += maxRank;
  }

  return totalCoverage / nSamples;
}

/**
 * Label ranking loss for multi-label classification.
 *
 * Computes the average number of label pairs that are incorrectly ordered.
 * A pair (i, j) is incorrectly ordered if y_true[i]=1, y_true[j]=0 but
 * score[i] <= score[j].
 *
 * Best value is 0.
 *
 * @param yTrue - Binary label indicators of shape (n_samples, n_labels)
 * @param yScore - Target scores of shape (n_samples, n_labels)
 * @returns Label ranking loss in [0, 1]
 */
export function labelRankingLoss(yTrue: Tensor, yScore: Tensor): number {
  const { data: trueData, nSamples, nLabels } = extract2D(yTrue);
  const { data: scoreData, nSamples: nS2, nLabels: nL2 } = extract2D(yScore);

  if (nSamples !== nS2 || nLabels !== nL2) {
    throw new ShapeError(
      `labelRankingLoss: shape mismatch [${nSamples},${nLabels}] vs [${nS2},${nL2}]`
    );
  }
  if (nSamples === 0) return 0;

  let totalLoss = 0;

  for (let i = 0; i < nSamples; i++) {
    const scores = scoreData[i]!;
    const labels = trueData[i]!;

    // Separate positive and negative label indices
    const posIndices: number[] = [];
    const negIndices: number[] = [];
    for (let j = 0; j < nLabels; j++) {
      if ((labels[j] ?? 0) > 0) {
        posIndices.push(j);
      } else {
        negIndices.push(j);
      }
    }

    const nPos = posIndices.length;
    const nNeg = negIndices.length;
    if (nPos === 0 || nNeg === 0) continue;

    // Count incorrectly ordered pairs
    let nIncorrect = 0;
    for (const p of posIndices) {
      for (const n of negIndices) {
        if ((scores[p] ?? 0) <= (scores[n] ?? 0)) {
          nIncorrect++;
        }
      }
    }

    totalLoss += nIncorrect / (nPos * nNeg);
  }

  return totalLoss / nSamples;
}
