/**
 * Internal validation utilities for ML models.
 * This file is not exported from the public API.
 *
 * @internal
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, DTypeError, ShapeError } from "../core";
import type { Tensor } from "../ndarray";
import { isDenseLayout } from "../ndarray/tensor/strides";

/**
 * Throw if a tensor is not laid out contiguously in row-major order.
 *
 * ML routines read `tensor.data` directly starting at `tensor.offset`, which is
 * only valid when logical element `i` lives at `offset + i`. Transposed or strided views
 * are rejected; a view that only differs in the strides of size-1 axes (for example the
 * transpose of a `[1, n]` row) is dense and accepted, as in NumPy.
 *
 * @param t - Tensor to check
 * @param name - Argument name used in the error message
 * @throws {DataValidationError} If the tensor is a non-contiguous view
 *
 * @internal
 */
export function assertContiguous(t: Tensor, name: string): void {
  if (!isDenseLayout(t.shape, t.strides)) {
    throw new DataValidationError(
      `${name} must be contiguous in row-major order; materialize a contiguous tensor before passing to ML routines`
    );
  }
}

/**
 * Throw if a tensor does not hold real numbers (string and complex tensors).
 *
 * @internal
 */
function assertNumericDType(t: Tensor, name: string): void {
  if (t.dtype === "string" || t.dtype === "complex64" || t.dtype === "complex128") {
    throw new DTypeError(`${name} must have a real numeric dtype; got dtype "${t.dtype}"`);
  }
}

/**
 * Throw if any element of a contiguous tensor is NaN or +/-Infinity.
 *
 * BigInt (`int64`) tensors cannot hold non-finite values and are accepted as is.
 *
 * @internal
 */
function assertAllFinite(t: Tensor, name: string): void {
  const data = t.data;
  if (data instanceof BigInt64Array) return;
  const end = t.offset + t.size;
  for (let i = t.offset; i < end; i++) {
    if (!Number.isFinite(data[i])) {
      throw new DataValidationError(`${name} contains non-finite values (NaN or Inf)`);
    }
  }
}

/**
 * Read a contiguous numeric tensor as a flat `Float64Array` in row-major order.
 *
 * When the tensor is already backed by a `Float64Array` the returned array is a
 * view of the tensor's own memory (no copy), so callers must treat it as
 * read-only. Other dtypes (including `int64`) are converted into a new array.
 *
 * @param t - Contiguous numeric tensor
 * @param name - Argument name used in error messages (default: "tensor")
 * @returns Flat array of `t.size` values
 * @throws {DataValidationError} If the tensor is a non-contiguous view
 * @throws {DTypeError} If the tensor does not hold real numbers
 *
 * @internal
 */
export function toFloat64View(t: Tensor, name = "tensor"): Float64Array {
  assertContiguous(t, name);
  assertNumericDType(t, name);
  const data = t.data;
  if (data instanceof Float64Array) {
    return data.subarray(t.offset, t.offset + t.size);
  }
  const out = new Float64Array(t.size);
  if (data instanceof BigInt64Array) {
    for (let i = 0; i < out.length; i++) out[i] = Number(data[t.offset + i]);
  } else {
    for (let i = 0; i < out.length; i++) out[i] = data[t.offset + i] as number;
  }
  return out;
}

/**
 * Percentile of an ascending-sorted array with linear interpolation between
 * order statistics, using the same arithmetic as `numpy.percentile`.
 *
 * @param sorted - Values sorted in ascending order (non-empty)
 * @param percent - Percentile in [0, 100]
 *
 * @internal
 */
export function percentileSorted(sorted: ArrayLike<number>, percent: number): number {
  const n = sorted.length;
  const t = percent / 100;
  const virtual = (n - 1) * t;
  const lo = Math.max(0, Math.min(n - 1, Math.floor(virtual)));
  const hi = Math.min(n - 1, lo + 1);
  const gamma = virtual - lo;
  const a = sorted[lo] as number;
  const b = sorted[hi] as number;
  const diff = b - a;
  return gamma >= 0.5 ? b - diff * (1 - gamma) : a + diff * gamma;
}

/**
 * Validate inputs for supervised learning fit methods.
 *
 * Checks:
 * - X is 2D, y is 1D
 * - X and y have matching number of samples
 * - No empty data (at least 1 sample and 1 feature)
 * - No NaN or Inf values
 *
 * @param X - Feature matrix of shape (n_samples, n_features)
 * @param y - Target vector of shape (n_samples,)
 * @throws {ShapeError} If dimensions are invalid
 * @throws {DataValidationError} If data contains invalid values
 *
 * @internal
 */
export function validateFitInputs(X: Tensor, y: Tensor): void {
  // Check dimensions
  if (X.ndim !== 2) {
    throw new ShapeError(`X must be 2-dimensional; got ndim=${X.ndim}`);
  }
  if (y.ndim !== 1) {
    throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
  }
  assertContiguous(X, "X");
  assertContiguous(y, "y");
  assertNumericDType(X, "X");
  assertNumericDType(y, "y");

  // Check for empty data
  const nSamples = X.shape[0] ?? 0;
  const nFeatures = X.shape[1] ?? 0;

  if (nSamples === 0) {
    throw new DataValidationError("X must have at least one sample");
  }
  if (nFeatures === 0) {
    throw new DataValidationError("X must have at least one feature");
  }

  // Check shape match
  if (nSamples !== y.shape[0]) {
    throw new ShapeError(
      `X and y must have the same number of samples; got X.shape[0]=${nSamples}, y.shape[0]=${y.shape[0]}`
    );
  }

  assertAllFinite(X, "X");
  assertAllFinite(y, "y");
}

/**
 * Validate inputs for unsupervised learning fit methods.
 *
 * Checks:
 * - X is 2D
 * - No empty data (at least 1 sample and 1 feature)
 * - No NaN or Inf values
 *
 * @param X - Feature matrix of shape (n_samples, n_features)
 * @throws {ShapeError} If dimensions are invalid
 * @throws {DataValidationError} If data contains invalid values
 *
 * @internal
 */
export function validateUnsupervisedFitInputs(X: Tensor): void {
  // Check dimensions
  if (X.ndim !== 2) {
    throw new ShapeError(`X must be 2-dimensional; got ndim=${X.ndim}`);
  }
  assertContiguous(X, "X");
  assertNumericDType(X, "X");

  // Check for empty data
  const nSamples = X.shape[0] ?? 0;
  const nFeatures = X.shape[1] ?? 0;

  if (nSamples === 0) {
    throw new DataValidationError("X must have at least one sample");
  }
  if (nFeatures === 0) {
    throw new DataValidationError("X must have at least one feature");
  }

  assertAllFinite(X, "X");
}

/**
 * Validate inputs for prediction methods.
 *
 * Checks:
 * - X is 2D
 * - X has correct number of features
 * - No NaN or Inf values
 *
 * @param X - Feature matrix of shape (n_samples, n_features)
 * @param nFeaturesExpected - Expected number of features from training
 * @param modelName - Name of the model (for error messages)
 * @throws {ShapeError} If dimensions are invalid
 * @throws {DataValidationError} If data contains invalid values
 * @throws {DTypeError} If X is not numeric
 *
 * @internal
 */
export function validatePredictInputs(
  X: Tensor,
  nFeaturesExpected: number,
  modelName: string
): void {
  // Check dimensions
  if (X.ndim !== 2) {
    throw new ShapeError(`X must be 2-dimensional; got ndim=${X.ndim}`);
  }
  assertContiguous(X, "X");
  assertNumericDType(X, "X");

  // Check feature count
  const nFeatures = X.shape[1] ?? 0;
  if (nFeatures !== nFeaturesExpected) {
    throw new ShapeError(
      `X has ${nFeatures} features but ${modelName} was fitted with ${nFeaturesExpected} features`
    );
  }

  assertAllFinite(X, "X");
}
