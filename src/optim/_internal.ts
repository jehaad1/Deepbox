/**
 * Internal utilities for optimizer implementations.
 * This module is not part of the public API.
 *
 * @internal
 */

import {
  DeepboxError,
  DTypeError,
  IndexError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../core";
import type { GradTensor } from "../ndarray";

/**
 * Supported floating-point typed array types for optimizer parameters.
 */
export type FloatTypedArray = Float32Array | Float64Array;

function isFloatTypedArray(value: unknown): value is FloatTypedArray {
  return value instanceof Float32Array || value instanceof Float64Array;
}

/**
 * Safely access an array element with bounds checking.
 *
 * @param array - Array to access
 * @param index - Index to access
 * @param context - Context string for error messages
 * @returns The value at the index
 * @throws {IndexError} If index is out of bounds
 * @throws {DeepboxError} If value is unexpectedly undefined
 */
export function safeArrayAccess<T>(array: ArrayLike<T>, index: number, context: string): T {
  if (index < 0 || index >= array.length) {
    throw new IndexError(`Index ${index} out of bounds [0, ${array.length}) in ${context}`, {
      index,
      validRange: [0, array.length - 1],
    });
  }
  const value = array[index];
  if (value === undefined) {
    throw new DeepboxError(`Unexpected undefined at index ${index} in ${context}`);
  }
  return value;
}

/**
 * Validates that a numeric value is finite and non-negative.
 *
 * @param name - Name of the parameter being validated
 * @param value - Value to validate
 * @throws {InvalidParameterError} If value is not finite or is negative
 */
export function assertFiniteNonNegative(name: string, value: number): void {
  if (!Number.isFinite(value) || value < 0) {
    throw new InvalidParameterError(`Invalid ${name}: ${value}`, name, value);
  }
}

/**
 * Validates that a numeric value is finite and positive (> 0).
 *
 * @param name - Name of the parameter being validated
 * @param value - Value to validate
 * @throws {InvalidParameterError} If value is not finite or is not positive
 */
export function assertFinitePositive(name: string, value: number): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new InvalidParameterError(`Invalid ${name}: ${value} (must be > 0)`, name, value);
  }
}

/**
 * Validates that a numeric value is finite.
 *
 * @param name - Name of the parameter being validated
 * @param value - Value to validate
 * @throws {InvalidParameterError} If value is not finite
 */
export function assertFinite(name: string, value: number): void {
  if (!Number.isFinite(value)) {
    throw new InvalidParameterError(`Invalid ${name}: ${value}`, name, value);
  }
}

/**
 * Validates that a value is in the range [min, max).
 *
 * @param name - Name of the parameter being validated
 * @param value - Value to validate
 * @param min - Minimum value (inclusive)
 * @param max - Maximum value (exclusive)
 * @throws {InvalidParameterError} If value is out of range
 */
export function assertInRange(name: string, value: number, min: number, max: number): void {
  if (!Number.isFinite(value) || value < min || value >= max) {
    throw new InvalidParameterError(
      `Invalid ${name}: ${value} (must be in range [${min}, ${max}))`,
      name,
      value
    );
  }
}

/**
 * Validates that a parameter has a gradient and returns gradient information.
 *
 * @param param - Parameter to validate
 * @param optimizerName - Name of the optimizer for error messages
 * @returns Object containing gradient data, offset, parameter data, and offset
 * @throws {InvalidParameterError} If parameter doesn't require gradients
 * @throws {NotFittedError} If parameter has no gradient
 * @throws {DTypeError} If parameter or gradient has unsupported dtype
 * @throws {ShapeError} If gradient shape doesn't match parameter shape
 */
export function assertHasGradFloat(
  param: GradTensor,
  optimizerName: string
): {
  grad: FloatTypedArray;
  gradOffset: number;
  param: FloatTypedArray;
  paramOffset: number;
} {
  if (!param.requiresGrad) {
    throw new InvalidParameterError(
      "Cannot optimize a parameter with requiresGrad=false",
      "requiresGrad",
      false
    );
  }

  const g = param.grad;
  if (!g) {
    throw new NotFittedError(
      "Cannot optimize a parameter without a gradient. Did you forget backward()?"
    );
  }

  const paramData = param.tensor.data;
  const gradData = g.data;

  if (!isFloatTypedArray(paramData) || !isFloatTypedArray(gradData)) {
    throw new DTypeError(
      `${optimizerName} optimizer supports float32 and float64 parameters and gradients only`
    );
  }

  if (paramData.constructor !== gradData.constructor) {
    throw new DTypeError(
      `${optimizerName} optimizer requires parameter and gradient dtypes to match`
    );
  }

  if (param.tensor.size !== g.size) {
    throw new ShapeError(
      `Gradient shape must match parameter shape (param: ${param.tensor.size}, grad: ${g.size})`
    );
  }

  return {
    grad: gradData,
    gradOffset: g.offset,
    param: paramData,
    paramOffset: param.tensor.offset,
  };
}

/**
 * Validates that a state buffer has the correct size.
 *
 * @param buffer - State buffer to validate
 * @param expectedSize - Expected size
 * @param bufferName - Name of the buffer for error messages
 * @throws {DeepboxError} If buffer size doesn't match expected size
 */
export function assertBufferSize(
  buffer: ArrayLike<number>,
  expectedSize: number,
  bufferName: string
): void {
  if (buffer.length !== expectedSize) {
    throw new DeepboxError(
      `State buffer size mismatch for ${bufferName}: expected ${expectedSize}, got ${buffer.length}`
    );
  }
}
