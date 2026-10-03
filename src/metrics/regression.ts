import { isNumericTypedArray, isTypedArray } from "../core";
import { DTypeError, InvalidParameterError } from "../core/errors";
import type { Tensor } from "../ndarray";
import {
  assertSameSizeVectors,
  compensatedSum,
  denseFloat64,
  nonZeroWeightSum,
  readSampleWeight,
  type WeightedMetricOptions,
} from "./_internal";

function assertNumericRegressionTensor(t: Tensor, name: string): void {
  if (t.dtype === "string") {
    throw new DTypeError(`${name} must be numeric tensors`);
  }
  if (t.dtype === "int64") {
    throw new DTypeError(`${name} must be numeric tensors (int64 not supported)`);
  }

  const data = t.data;
  if (!isTypedArray(data) || !isNumericTypedArray(data)) {
    throw new DTypeError(`${name} must be numeric tensors`);
  }
}

/** Dense Float64 copy of a vector-like tensor; every value must be finite. */
function readVector(t: Tensor, name: string): Float64Array {
  return denseFloat64(t, name, true);
}

/**
 * Sum of squared deviations from the mean, exactly 0 when every value is equal (the
 * rounded mean of equal values can be off by an ulp, which would leave a tiny non-zero
 * sum). Overwrites `a` with the squared deviations.
 */
function sumSquaredDeviations(a: Float64Array): number {
  const n = a.length;
  const first = a[0] as number;
  let constant = true;
  for (let i = 1; i < n; i++) {
    if (a[i] !== first) {
      constant = false;
      break;
    }
  }
  if (constant) return 0;

  const mean = compensatedSum(a) / n;
  for (let i = 0; i < n; i++) {
    const d = (a[i] as number) - mean;
    a[i] = d * d;
  }
  return compensatedSum(a);
}

/**
 * Weighted sum of squared deviations from the weighted mean (`total` is the sum of the
 * weights), exactly 0 when every value with a non-zero weight is equal.
 */
function weightedSumSquaredDeviations(a: Float64Array, w: Float64Array, total: number): number {
  const n = a.length;
  let reference = 0;
  let seen = false;
  let constant = true;
  for (let i = 0; i < n; i++) {
    if (w[i] === 0) continue;
    const v = a[i] as number;
    if (!seen) {
      reference = v;
      seen = true;
    } else if (v !== reference) {
      constant = false;
      break;
    }
  }
  if (constant) return 0;

  const terms = new Float64Array(n);
  for (let i = 0; i < n; i++) terms[i] = (a[i] as number) * (w[i] as number);
  const mean = compensatedSum(terms) / total;
  for (let i = 0; i < n; i++) {
    const d = (a[i] as number) - mean;
    terms[i] = (w[i] as number) * d * d;
  }
  return compensatedSum(terms);
}

/** Mean of `terms` (overwritten when weighted): plain, or weighted by `w`. */
function averageTerms(terms: Float64Array, w: Float64Array | undefined): number {
  if (w === undefined) return compensatedSum(terms) / terms.length;
  const total = nonZeroWeightSum(w);
  for (let i = 0; i < terms.length; i++) terms[i] = (terms[i] as number) * (w[i] as number);
  return compensatedSum(terms) / total;
}

/**
 * Validate a (yTrue, yPred) pair and return private Float64 copies. The copies are
 * owned by the caller, so metrics may overwrite them with intermediate terms.
 */
function readPair(
  yTrue: Tensor,
  yPred: Tensor,
  options: WeightedMetricOptions = {}
): { t: Float64Array; p: Float64Array; w: Float64Array | undefined } {
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");
  assertNumericRegressionTensor(yTrue, "yTrue");
  assertNumericRegressionTensor(yPred, "yPred");
  const w = readSampleWeight(options.sampleWeight, yTrue.size);
  return { t: readVector(yTrue, "yTrue"), p: readVector(yPred, "yPred"), w };
}

/**
 * Calculate Mean Squared Error (MSE).
 *
 * Measures the average squared difference between predictions and actual values.
 * MSE is sensitive to outliers due to squaring the errors.
 *
 * **Formula**: MSE = (1/n) * Σ(y_true - y_pred)²
 *
 * **Time Complexity**: O(n) where n is the number of samples
 * **Space Complexity**: O(1)
 *
 * With `options.sampleWeight` the mean is weighted: Σ w_i * (y_i - p_i)² / Σ w_i.
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated target values
 * @param options - Optional `sampleWeight`, one weight per sample
 * @returns MSE value (always non-negative, 0 is perfect)
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors, or
 *   `sampleWeight` has the wrong length
 * @throws {DTypeError} If yTrue or yPred is non-numeric or int64
 * @throws {InvalidParameterError} If `sampleWeight` sums to zero
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { mse } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([3, -0.5, 2, 7]);
 * const yPred = tensor([2.5, 0.0, 2, 8]);
 * const error = mse(yTrue, yPred);  // 0.375
 * const weighted = mse(yTrue, yPred, { sampleWeight: [1, 2, 3, 4] });  // 0.475
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-regression | Deepbox Regression Metrics}
 */
export function mse(yTrue: Tensor, yPred: Tensor, options: WeightedMetricOptions = {}): number {
  const { t, p, w } = readPair(yTrue, yPred, options);
  const n = t.length;
  if (n === 0) return 0;

  for (let i = 0; i < n; i++) {
    const diff = (t[i] as number) - (p[i] as number);
    p[i] = diff * diff;
  }
  return averageTerms(p, w);
}

/**
 * Calculate Root Mean Squared Error (RMSE).
 *
 * Square root of MSE, expressed in the same units as the target variable.
 * RMSE is more interpretable than MSE as it's in the original scale.
 *
 * **Formula**: RMSE = √(MSE) = √((1/n) * Σ(y_true - y_pred)²)
 *
 * **Time Complexity**: O(n) where n is the number of samples
 * **Space Complexity**: O(1)
 *
 * With `options.sampleWeight` the underlying MSE is weighted, see {@link mse}.
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated target values
 * @param options - Optional `sampleWeight`, one weight per sample
 * @returns RMSE value (always non-negative, 0 is perfect)
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors, or
 *   `sampleWeight` has the wrong length
 * @throws {DTypeError} If yTrue or yPred is non-numeric or int64
 * @throws {InvalidParameterError} If `sampleWeight` sums to zero
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { rmse } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([3, -0.5, 2, 7]);
 * const yPred = tensor([2.5, 0.0, 2, 8]);
 * const error = rmse(yTrue, yPred);  // √0.375 ≈ 0.612
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-regression | Deepbox Regression Metrics}
 */
export function rmse(yTrue: Tensor, yPred: Tensor, options: WeightedMetricOptions = {}): number {
  return Math.sqrt(mse(yTrue, yPred, options));
}

/**
 * Calculate Mean Absolute Error (MAE).
 *
 * Measures the average absolute difference between predictions and actual values.
 * MAE is less sensitive to outliers than MSE.
 *
 * **Formula**: MAE = (1/n) * Σ|y_true - y_pred|
 *
 * **Time Complexity**: O(n) where n is the number of samples
 * **Space Complexity**: O(1)
 *
 * With `options.sampleWeight` the mean is weighted: Σ w_i * |y_i - p_i| / Σ w_i.
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated target values
 * @param options - Optional `sampleWeight`, one weight per sample
 * @returns MAE value (always non-negative, 0 is perfect)
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors, or
 *   `sampleWeight` has the wrong length
 * @throws {DTypeError} If yTrue or yPred is non-numeric or int64
 * @throws {InvalidParameterError} If `sampleWeight` sums to zero
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { mae } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([3, -0.5, 2, 7]);
 * const yPred = tensor([2.5, 0.0, 2, 8]);
 * const error = mae(yTrue, yPred);  // 0.5
 * const weighted = mae(yTrue, yPred, { sampleWeight: [1, 2, 3, 4] });  // 0.55
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-regression | Deepbox Regression Metrics}
 */
export function mae(yTrue: Tensor, yPred: Tensor, options: WeightedMetricOptions = {}): number {
  const { t, p, w } = readPair(yTrue, yPred, options);
  const n = t.length;
  if (n === 0) return 0;

  for (let i = 0; i < n; i++) {
    p[i] = Math.abs((t[i] as number) - (p[i] as number));
  }
  return averageTerms(p, w);
}

/**
 * Calculate R² (coefficient of determination) score.
 *
 * Represents the proportion of variance in the target variable that is
 * explained by the model. R² of 1 indicates perfect predictions, 0 indicates
 * the model is no better than predicting the mean, and negative values indicate
 * the model is worse than predicting the mean.
 *
 * **Formula**: R² = 1 - (SS_res / SS_tot)
 * - SS_res = Σ(y_true - y_pred)² (residual sum of squares)
 * - SS_tot = Σ(y_true - mean(y_true))² (total sum of squares)
 *
 * **Time Complexity**: O(n) where n is the number of samples
 * **Space Complexity**: O(1)
 *
 * With `options.sampleWeight` both sums are weighted and SS_tot is taken around the
 * weighted mean of y_true.
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated target values
 * @param options - Optional `sampleWeight`, one weight per sample
 * @returns R² score (1 is perfect, 0 is baseline, negative is worse than baseline)
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors, or
 *   `sampleWeight` has the wrong length
 * @throws {DTypeError} If yTrue or yPred is non-numeric or int64
 * @throws {InvalidParameterError} If inputs are empty or `sampleWeight` sums to zero
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { r2Score } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([3, -0.5, 2, 7]);
 * const yPred = tensor([2.5, 0.0, 2, 8]);
 * const score = r2Score(yTrue, yPred);  // 0.9486081370449679
 * const weighted = r2Score(yTrue, yPred, { sampleWeight: [1, 2, 3, 4] });  // 0.9459613196814562
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-regression | Deepbox Regression Metrics}
 */
export function r2Score(yTrue: Tensor, yPred: Tensor, options: WeightedMetricOptions = {}): number {
  const { t, p, w } = readPair(yTrue, yPred, options);
  const n = t.length;
  if (n === 0) {
    throw new InvalidParameterError("r2Score requires at least one sample", "yTrue", n);
  }

  const total = w === undefined ? n : nonZeroWeightSum(w);
  for (let i = 0; i < n; i++) {
    const dRes = (t[i] as number) - (p[i] as number);
    p[i] = w === undefined ? dRes * dRes : (w[i] as number) * dRes * dRes;
  }
  const ssRes = compensatedSum(p);
  const ssTot =
    w === undefined ? sumSquaredDeviations(t) : weightedSumSquaredDeviations(t, w, total);

  // Constant targets (ssTot = 0): there is no variance to explain, so a perfect
  // fit scores 1 and anything else 0 (scikit-learn's force_finite behavior).
  if (ssTot === 0) {
    return ssRes === 0 ? 1.0 : 0.0;
  }

  return 1 - ssRes / ssTot;
}

/**
 * Calculate Adjusted R² score.
 *
 * R² adjusted for the number of features in the model. Penalizes the addition
 * of features that don't improve the model. More appropriate than R² when
 * comparing models with different numbers of features.
 *
 * **Formula**: Adjusted R² = 1 - ((1 - R²) * (n - 1)) / (n - p - 1)
 * - n = number of samples
 * - p = number of features
 *
 * **Time Complexity**: O(n) where n is the number of samples
 * **Space Complexity**: O(1)
 *
 * **Constraints**: Requires n > p + 1 (more samples than features + 1)
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated target values
 * @param nFeatures - Number of features (predictors) used in the model
 * @returns Adjusted R² score
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If yTrue or yPred is non-numeric or int64
 * @throws {InvalidParameterError} If nFeatures is not a non-negative integer, n <= p + 1, or inputs are empty
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { adjustedR2Score } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([3, -0.5, 2, 7]);
 * const yPred = tensor([2.5, 0.0, 2, 8]);
 * const score = adjustedR2Score(yTrue, yPred, 2);  // 0.8458244111349038
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-regression | Deepbox Regression Metrics}
 */
export function adjustedR2Score(yTrue: Tensor, yPred: Tensor, nFeatures: number): number {
  if (!Number.isFinite(nFeatures) || !Number.isInteger(nFeatures) || nFeatures < 0) {
    throw new InvalidParameterError(
      "nFeatures must be a non-negative integer",
      "nFeatures",
      nFeatures
    );
  }

  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");
  const n = yTrue.size;
  const p = nFeatures;

  // Validate sufficient samples
  if (n <= p + 1) {
    throw new InvalidParameterError(
      `Adjusted R² requires n > p + 1 (samples > features + 1). Got n=${n}, p=${p}`,
      "nFeatures",
      nFeatures
    );
  }

  const r2 = r2Score(yTrue, yPred);

  return 1 - ((1 - r2) * (n - 1)) / (n - p - 1);
}

/**
 * Calculate Mean Absolute Percentage Error (MAPE).
 *
 * Measures the average absolute percentage difference between predictions
 * and actual values. Expressed as a percentage, making it scale-independent
 * and easy to interpret.
 *
 * **Formula**: MAPE = (100/m) * Σ|((y_true - y_pred) / y_true)|
 * where m is the number of non-zero targets.
 *
 * **Time Complexity**: O(n) where n is the number of samples
 * **Space Complexity**: O(1)
 *
 * **Difference from scikit-learn:** `mape` returns a percentage (32.7 means 32.7 %),
 * while scikit-learn's `mean_absolute_percentage_error` returns a fraction (0.327).
 * `mape` also skips zero values in yTrue instead of guarding with an epsilon, and
 * returns 0 if all targets are zero. For the scikit-learn result, including
 * `sampleWeight` support, use {@link meanAbsolutePercentageError}.
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated target values
 * @returns MAPE value as percentage (0 is perfect, lower is better)
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If yTrue or yPred is non-numeric or int64
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { mape } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([3, -0.5, 2, 7]);
 * const yPred = tensor([2.5, 0.0, 2, 8]);
 * const error = mape(yTrue, yPred);  // 32.73809523809524 (percent)
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-regression | Deepbox Regression Metrics}
 */
export function mape(yTrue: Tensor, yPred: Tensor): number {
  const { t, p } = readPair(yTrue, yPred);
  const n = t.length;
  if (n === 0) return 0;

  let nonZeroCount = 0;
  for (let i = 0; i < n; i++) {
    const trueVal = t[i] as number;
    if (trueVal !== 0) {
      p[nonZeroCount++] = Math.abs((trueVal - (p[i] as number)) / trueVal);
    }
  }

  if (nonZeroCount === 0) {
    return 0;
  }

  return (compensatedSum(p.subarray(0, nonZeroCount)) / nonZeroCount) * 100;
}

/**
 * Calculate the Mean Absolute Percentage Error with scikit-learn's definition.
 *
 * **Formula**: MAPE = (1/n) * Σ |y_true - y_pred| / max(|y_true|, ε)
 * where ε = 2.220446049250313e-16 (the float64 machine epsilon). A zero target
 * therefore produces a very large term instead of being skipped.
 *
 * The result is a fraction (0.25 means 25 %), exactly like scikit-learn's
 * `mean_absolute_percentage_error`. {@link mape} returns a percentage and skips zero
 * targets instead; use this function for scikit-learn compatible numbers.
 *
 * With `options.sampleWeight` the mean is weighted: Σ w_i * term_i / Σ w_i.
 *
 * **Time Complexity**: O(n) where n is the number of samples
 * **Space Complexity**: O(n)
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated target values
 * @param options - Optional `sampleWeight`, one weight per sample
 * @returns MAPE as a fraction (0 is perfect). Returns 0 for empty input.
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors, or
 *   `sampleWeight` has the wrong length
 * @throws {DTypeError} If yTrue or yPred is non-numeric or int64
 * @throws {InvalidParameterError} If `sampleWeight` sums to zero
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { meanAbsolutePercentageError } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([3, -0.5, 2, 7]);
 * const yPred = tensor([2.5, 0.0, 2, 8]);
 * meanAbsolutePercentageError(yTrue, yPred); // 0.3273809523809524
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-regression | Deepbox Regression Metrics}
 */
export function meanAbsolutePercentageError(
  yTrue: Tensor,
  yPred: Tensor,
  options: WeightedMetricOptions = {}
): number {
  const { t, p, w } = readPair(yTrue, yPred, options);
  const n = t.length;
  if (n === 0) return 0;

  for (let i = 0; i < n; i++) {
    const trueVal = t[i] as number;
    p[i] = Math.abs((p[i] as number) - trueVal) / Math.max(Math.abs(trueVal), Number.EPSILON);
  }
  return averageTerms(p, w);
}

/**
 * Calculate Median Absolute Error (MedAE).
 *
 * Measures the median of absolute differences between predictions and actual values.
 * Less sensitive to outliers than MAE or MSE because it uses the median instead of the mean.
 *
 * **Formula**: MedAE = median(|y_true - y_pred|)
 *
 * **Time Complexity**: O(n log n) due to sorting for median calculation
 * **Space Complexity**: O(n) for storing error array
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated target values
 * @returns Median absolute error (always non-negative, 0 is perfect)
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If yTrue or yPred is non-numeric or int64
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { medianAbsoluteError } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([3, -0.5, 2, 7]);
 * const yPred = tensor([2.5, 0.0, 2, 8]);
 * const error = medianAbsoluteError(yTrue, yPred);  // 0.5
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-regression | Deepbox Regression Metrics}
 */
export function medianAbsoluteError(yTrue: Tensor, yPred: Tensor): number {
  const { t, p: errors } = readPair(yTrue, yPred);
  const n = t.length;
  if (n === 0) return 0;

  for (let i = 0; i < n; i++) {
    errors[i] = Math.abs((t[i] as number) - (errors[i] as number));
  }

  // The median needs only the middle order statistic(s), so quickselect
  // (O(n) average) beats a full O(n log n) sort. quickselectF64 partially
  // partitions `errors` in place around the requested rank.
  const mid = n >> 1;
  if (n % 2 !== 0) {
    return quickselectF64(errors, mid);
  }
  const hi = quickselectF64(errors, mid);
  // The lower median is the max of the left partition, now in [0, mid).
  let lo = errors[0] as number;
  for (let i = 1; i < mid; i++) {
    const v = errors[i] as number;
    if (v > lo) lo = v;
  }
  return lo + (hi - lo) / 2;
}

/**
 * In-place quickselect: returns the value that would sit at sorted index `k`,
 * partitioning `a` so entries < result are left of `k` and entries > result
 * are right. Median-of-three pivot; O(n) average.
 */
function quickselectF64(a: Float64Array, k: number): number {
  let lo = 0;
  let hi = a.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    // Median-of-three pivot, so sorted or adversarial input does not degrade the sort.
    const x = a[lo] as number;
    const y = a[mid] as number;
    const z = a[hi] as number;
    let pivot: number;
    if (x < y) pivot = y < z ? y : x < z ? z : x;
    else pivot = x < z ? x : y < z ? z : y;
    let i = lo;
    let j = hi;
    while (i <= j) {
      while ((a[i] as number) < pivot) i++;
      while ((a[j] as number) > pivot) j--;
      if (i <= j) {
        const tmp = a[i] as number;
        a[i] = a[j] as number;
        a[j] = tmp;
        i++;
        j--;
      }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else break;
  }
  return a[k] as number;
}

/**
 * Calculate maximum residual error.
 *
 * Returns the maximum absolute difference between predictions and actual values.
 * Useful for identifying the worst-case prediction error.
 *
 * **Formula**: max_error = max(|y_true - y_pred|)
 *
 * **Time Complexity**: O(n) where n is the number of samples
 * **Space Complexity**: O(1)
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated target values
 * @returns Maximum absolute error (always non-negative, 0 is perfect)
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If yTrue or yPred is non-numeric or int64
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { maxError } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([3, -0.5, 2, 7]);
 * const yPred = tensor([2.5, 0.0, 2, 8]);
 * const error = maxError(yTrue, yPred);  // 1.0 (worst prediction)
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-regression | Deepbox Regression Metrics}
 */
export function maxError(yTrue: Tensor, yPred: Tensor): number {
  const { t, p } = readPair(yTrue, yPred);
  let maxErr = 0;
  for (let i = 0; i < t.length; i++) {
    const diff = Math.abs((t[i] as number) - (p[i] as number));
    if (diff > maxErr) maxErr = diff;
  }

  return maxErr;
}

/**
 * Calculate explained variance score.
 *
 * Measures the proportion of variance in the target variable that is explained
 * by the model. Similar to R² but uses variance instead of sum of squares.
 * Best possible score is 1.0, lower values are worse.
 *
 * **Formula**: explained_variance = 1 - Var(y_true - y_pred) / Var(y_true)
 *
 * **Time Complexity**: O(n) where n is the number of samples
 * **Space Complexity**: O(1)
 *
 * With `options.sampleWeight` both variances are weighted (weighted means and
 * weighted sums of squares).
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated target values
 * @param options - Optional `sampleWeight`, one weight per sample
 * @returns Explained variance score (1.0 is perfect, lower is worse)
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors, or
 *   `sampleWeight` has the wrong length
 * @throws {DTypeError} If yTrue or yPred is non-numeric or int64
 * @throws {InvalidParameterError} If inputs are empty or `sampleWeight` sums to zero
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { explainedVarianceScore } from 'deepbox/metrics';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const yTrue = tensor([3, -0.5, 2, 7]);
 * const yPred = tensor([2.5, 0.0, 2, 8]);
 * const score = explainedVarianceScore(yTrue, yPred);  // 0.9571734475374732
 * const weighted = explainedVarianceScore(yTrue, yPred, { sampleWeight: [1, 2, 3, 4] }); // 0.9689988623435722
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-regression | Deepbox Regression Metrics}
 */
export function explainedVarianceScore(
  yTrue: Tensor,
  yPred: Tensor,
  options: WeightedMetricOptions = {}
): number {
  const { t, p, w } = readPair(yTrue, yPred, options);
  const n = t.length;
  if (n === 0) {
    throw new InvalidParameterError(
      "explainedVarianceScore requires at least one sample",
      "yTrue",
      n
    );
  }

  // p now holds the residuals y_true - y_pred.
  for (let i = 0; i < n; i++) p[i] = (t[i] as number) - (p[i] as number);
  let varResidual: number;
  let varTrue: number;
  if (w === undefined) {
    varResidual = sumSquaredDeviations(p);
    varTrue = sumSquaredDeviations(t);
  } else {
    const total = nonZeroWeightSum(w);
    varResidual = weightedSumSquaredDeviations(p, w, total);
    varTrue = weightedSumSquaredDeviations(t, w, total);
  }

  // Constant targets (varTrue = 0): a perfect fit scores 1, anything else 0.
  if (varTrue === 0) {
    return varResidual === 0 ? 1.0 : 0.0;
  }

  return 1 - varResidual / varTrue;
}

/**
 * Alias of {@link mse}.
 *
 * @see {@link https://deepbox.dev/docs/metrics-regression | Deepbox Regression Metrics}
 */
export const meanSquaredError: typeof mse = mse;

/**
 * Alias of {@link rmse}.
 *
 * @see {@link https://deepbox.dev/docs/metrics-regression | Deepbox Regression Metrics}
 */
export const rootMeanSquaredError: typeof rmse = rmse;

/**
 * Alias of {@link mae}.
 *
 * @see {@link https://deepbox.dev/docs/metrics-regression | Deepbox Regression Metrics}
 */
export const meanAbsoluteError: typeof mae = mae;
