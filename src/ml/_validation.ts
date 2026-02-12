/**
 * Internal validation utilities for ML models.
 * This file is not exported from the public API.
 *
 * @internal
 */

import { DataValidationError, ShapeError } from "../core";
import type { Tensor } from "../ndarray";
import { isContiguous } from "../ndarray/tensor/strides";

export function assertContiguous(t: Tensor, name: string): void {
  if (!isContiguous(t.shape, t.strides)) {
    throw new DataValidationError(
      `${name} must be contiguous in row-major order; materialize a contiguous tensor before passing to ML routines`
    );
  }
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

  // Check for NaN/Inf in X
  for (let i = 0; i < X.size; i++) {
    const val = X.data[X.offset + i] ?? 0;
    if (!Number.isFinite(val)) {
      throw new DataValidationError("X contains non-finite values (NaN or Inf)");
    }
  }

  // Check for NaN/Inf in y
  for (let i = 0; i < y.size; i++) {
    const val = y.data[y.offset + i] ?? 0;
    if (!Number.isFinite(val)) {
      throw new DataValidationError("y contains non-finite values (NaN or Inf)");
    }
  }
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

  // Check for empty data
  const nSamples = X.shape[0] ?? 0;
  const nFeatures = X.shape[1] ?? 0;

  if (nSamples === 0) {
    throw new DataValidationError("X must have at least one sample");
  }
  if (nFeatures === 0) {
    throw new DataValidationError("X must have at least one feature");
  }

  // Check for NaN/Inf in X
  for (let i = 0; i < X.size; i++) {
    const val = X.data[X.offset + i] ?? 0;
    if (!Number.isFinite(val)) {
      throw new DataValidationError("X contains non-finite values (NaN or Inf)");
    }
  }
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

  // Check feature count
  const nFeatures = X.shape[1] ?? 0;
  if (nFeatures !== nFeaturesExpected) {
    throw new ShapeError(
      `X has ${nFeatures} features but ${modelName} was fitted with ${nFeaturesExpected} features`
    );
  }

  // Check for NaN/Inf
  for (let i = 0; i < X.size; i++) {
    const val = X.data[X.offset + i] ?? 0;
    if (!Number.isFinite(val)) {
      throw new DataValidationError("X contains non-finite values (NaN or Inf)");
    }
  }
}
