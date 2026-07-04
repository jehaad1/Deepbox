import {
  getNumericElement,
  isNumericTypedArray,
  isTypedArray,
  type NumericTypedArray,
} from "../core";
import { DTypeError, InvalidParameterError } from "../core/errors";
import type { Tensor } from "../ndarray";
import {
  assertFiniteNumber,
  assertSameSizeVectors,
  createFlatOffsetter,
  denseFloat64,
  type FlatOffsetter,
} from "./_internal";

function getNumericRegressionData(t: Tensor, name: string): NumericTypedArray {
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

  return data;
}

function readNumeric(
  data: NumericTypedArray,
  offsetter: FlatOffsetter,
  index: number,
  name: string
) {
  const value = getNumericElement(data, offsetter(index));
  assertFiniteNumber(value, name, `index ${index}`);
  return value;
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
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated target values
 * @returns MSE value (always non-negative, 0 is perfect)
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If yTrue or yPred is non-numeric or int64
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { mse, tensor } from 'deepbox/metrics';
 *
 * const yTrue = tensor([3, -0.5, 2, 7]);
 * const yPred = tensor([2.5, 0.0, 2, 8]);
 * const error = mse(yTrue, yPred);  // 0.375
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-regression | Deepbox Regression Metrics}
 */
export function mse(yTrue: Tensor, yPred: Tensor): number {
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");
  const yTrueData = getNumericRegressionData(yTrue, "yTrue");
  const yPredData = getNumericRegressionData(yPred, "yPred");

  if (yTrue.size === 0) return 0;

  const trueOffset = createFlatOffsetter(yTrue);
  const predOffset = createFlatOffsetter(yPred);

  let sumSquaredError = 0;
  for (let i = 0; i < yTrue.size; i++) {
    const diff =
      readNumeric(yTrueData, trueOffset, i, "yTrue") -
      readNumeric(yPredData, predOffset, i, "yPred");
    sumSquaredError += diff * diff;
  }

  return sumSquaredError / yTrue.size;
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
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated target values
 * @returns RMSE value (always non-negative, 0 is perfect)
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If yTrue or yPred is non-numeric or int64
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { rmse, tensor } from 'deepbox/metrics';
 *
 * const yTrue = tensor([3, -0.5, 2, 7]);
 * const yPred = tensor([2.5, 0.0, 2, 8]);
 * const error = rmse(yTrue, yPred);  // √0.375 ≈ 0.612
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-regression | Deepbox Regression Metrics}
 */
export function rmse(yTrue: Tensor, yPred: Tensor): number {
  return Math.sqrt(mse(yTrue, yPred));
}

/**
 * Calculate Mean Absolute Error (MAE).
 *
 * Measures the average absolute difference between predictions and actual values.
 * MAE is more robust to outliers than MSE.
 *
 * **Formula**: MAE = (1/n) * Σ|y_true - y_pred|
 *
 * **Time Complexity**: O(n) where n is the number of samples
 * **Space Complexity**: O(1)
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated target values
 * @returns MAE value (always non-negative, 0 is perfect)
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If yTrue or yPred is non-numeric or int64
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { mae, tensor } from 'deepbox/metrics';
 *
 * const yTrue = tensor([3, -0.5, 2, 7]);
 * const yPred = tensor([2.5, 0.0, 2, 8]);
 * const error = mae(yTrue, yPred);  // 0.5
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-regression | Deepbox Regression Metrics}
 */
export function mae(yTrue: Tensor, yPred: Tensor): number {
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");
  const yTrueData = getNumericRegressionData(yTrue, "yTrue");
  const yPredData = getNumericRegressionData(yPred, "yPred");

  if (yTrue.size === 0) return 0;

  const trueOffset = createFlatOffsetter(yTrue);
  const predOffset = createFlatOffsetter(yPred);

  let sumAbsError = 0;
  for (let i = 0; i < yTrue.size; i++) {
    const diff =
      readNumeric(yTrueData, trueOffset, i, "yTrue") -
      readNumeric(yPredData, predOffset, i, "yPred");
    sumAbsError += Math.abs(diff);
  }

  return sumAbsError / yTrue.size;
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
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated target values
 * @returns R² score (1 is perfect, 0 is baseline, negative is worse than baseline)
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If yTrue or yPred is non-numeric or int64
 * @throws {InvalidParameterError} If inputs are empty
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { r2Score, tensor } from 'deepbox/metrics';
 *
 * const yTrue = tensor([3, -0.5, 2, 7]);
 * const yPred = tensor([2.5, 0.0, 2, 8]);
 * const score = r2Score(yTrue, yPred);  // Close to 1 for good fit
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-regression | Deepbox Regression Metrics}
 */
export function r2Score(yTrue: Tensor, yPred: Tensor): number {
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");
  getNumericRegressionData(yTrue, "yTrue");
  getNumericRegressionData(yPred, "yPred");
  if (yTrue.size === 0) {
    throw new InvalidParameterError("r2Score requires at least one sample", "yTrue", yTrue.size);
  }

  const t = denseFloat64(yTrue, "yTrue");
  const p = denseFloat64(yPred, "yPred");
  const n = t.length;

  let sumTrue = 0;
  for (let i = 0; i < n; i++) sumTrue += t[i] as number;
  const mean = sumTrue / n;

  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    const trueVal = t[i] as number;
    const dRes = trueVal - (p[i] as number);
    const dTot = trueVal - mean;
    ssRes += dRes * dRes;
    ssTot += dTot * dTot;
  }

  // Handle constant targets (ssTot = 0)
  // When all true values are identical, return 0.0 (no variance to explain)
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
 * import { adjustedR2Score, tensor } from 'deepbox/metrics';
 *
 * const yTrue = tensor([3, -0.5, 2, 7]);
 * const yPred = tensor([2.5, 0.0, 2, 8]);
 * const score = adjustedR2Score(yTrue, yPred, 2);  // Adjusted for 2 features
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
 * **Important**: Zero values in yTrue are skipped. If all targets are zero,
 * this function returns 0.
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
 * import { mape, tensor } from 'deepbox/metrics';
 *
 * const yTrue = tensor([3, -0.5, 2, 7]);
 * const yPred = tensor([2.5, 0.0, 2, 8]);
 * const error = mape(yTrue, yPred);  // Percentage error
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-regression | Deepbox Regression Metrics}
 */
export function mape(yTrue: Tensor, yPred: Tensor): number {
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");
  const yTrueData = getNumericRegressionData(yTrue, "yTrue");
  const yPredData = getNumericRegressionData(yPred, "yPred");
  if (yTrue.size === 0) return 0;

  const trueOffset = createFlatOffsetter(yTrue);
  const predOffset = createFlatOffsetter(yPred);

  let sumPercentError = 0;
  let nonZeroCount = 0;
  for (let i = 0; i < yTrue.size; i++) {
    const trueVal = readNumeric(yTrueData, trueOffset, i, "yTrue");
    const predVal = readNumeric(yPredData, predOffset, i, "yPred");
    if (trueVal !== 0) {
      sumPercentError += Math.abs((trueVal - predVal) / trueVal);
      nonZeroCount++;
    }
  }

  if (nonZeroCount === 0) {
    return 0;
  }

  return (sumPercentError / nonZeroCount) * 100;
}

/**
 * Calculate Median Absolute Error (MedAE).
 *
 * Measures the median of absolute differences between predictions and actual values.
 * More robust to outliers than MAE or MSE as it uses the median instead of mean.
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
 * import { medianAbsoluteError, tensor } from 'deepbox/metrics';
 *
 * const yTrue = tensor([3, -0.5, 2, 7]);
 * const yPred = tensor([2.5, 0.0, 2, 8]);
 * const error = medianAbsoluteError(yTrue, yPred);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-regression | Deepbox Regression Metrics}
 */
export function medianAbsoluteError(yTrue: Tensor, yPred: Tensor): number {
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");
  getNumericRegressionData(yTrue, "yTrue");
  getNumericRegressionData(yPred, "yPred");

  if (yTrue.size === 0) return 0;

  const t = denseFloat64(yTrue, "yTrue");
  const p = denseFloat64(yPred, "yPred");
  const n = t.length;
  const errors = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    errors[i] = Math.abs((t[i] as number) - (p[i] as number));
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
  return (lo + hi) / 2;
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
    // Median-of-three pivot for robustness against sorted/adversarial input.
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
 * import { maxError, tensor } from 'deepbox/metrics';
 *
 * const yTrue = tensor([3, -0.5, 2, 7]);
 * const yPred = tensor([2.5, 0.0, 2, 8]);
 * const error = maxError(yTrue, yPred);  // 1.0 (worst prediction)
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-regression | Deepbox Regression Metrics}
 */
export function maxError(yTrue: Tensor, yPred: Tensor): number {
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");
  getNumericRegressionData(yTrue, "yTrue");
  getNumericRegressionData(yPred, "yPred");

  const t = denseFloat64(yTrue, "yTrue");
  const p = denseFloat64(yPred, "yPred");
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
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated target values
 * @returns Explained variance score (1.0 is perfect, lower is worse)
 *
 * @throws {ShapeError} If yTrue and yPred have different sizes or are not 1D/column vectors
 * @throws {DTypeError} If yTrue or yPred is non-numeric or int64
 * @throws {InvalidParameterError} If inputs are empty
 * @throws {DataValidationError} If inputs contain NaN or infinite values
 *
 * @example
 * ```ts
 * import { explainedVarianceScore, tensor } from 'deepbox/metrics';
 *
 * const yTrue = tensor([3, -0.5, 2, 7]);
 * const yPred = tensor([2.5, 0.0, 2, 8]);
 * const score = explainedVarianceScore(yTrue, yPred);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/metrics-regression | Deepbox Regression Metrics}
 */
export function explainedVarianceScore(yTrue: Tensor, yPred: Tensor): number {
  assertSameSizeVectors(yTrue, yPred, "yTrue", "yPred");
  getNumericRegressionData(yTrue, "yTrue");
  getNumericRegressionData(yPred, "yPred");
  if (yTrue.size === 0) {
    throw new InvalidParameterError(
      "explainedVarianceScore requires at least one sample",
      "yTrue",
      yTrue.size
    );
  }

  const t = denseFloat64(yTrue, "yTrue");
  const p = denseFloat64(yPred, "yPred");
  const n = t.length;

  let sumTrue = 0;
  let sumResidual = 0;
  for (let i = 0; i < n; i++) {
    const trueVal = t[i] as number;
    sumTrue += trueVal;
    sumResidual += trueVal - (p[i] as number);
  }
  const meanTrue = sumTrue / n;
  const meanResidual = sumResidual / n;

  let varResidual = 0;
  let varTrue = 0;
  for (let i = 0; i < n; i++) {
    const trueVal = t[i] as number;
    const residual = trueVal - (p[i] as number);
    const dr = residual - meanResidual;
    const dt = trueVal - meanTrue;
    varResidual += dr * dr;
    varTrue += dt * dt;
  }

  // Handle constant targets (varTrue = 0)
  // When all true values are identical, return 0.0 (no variance to explain)
  if (varTrue === 0) {
    return varResidual === 0 ? 1.0 : 0.0;
  }

  return 1 - varResidual / varTrue;
}
