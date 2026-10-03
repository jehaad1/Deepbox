/**
 * Internal utilities for optimizer implementations.
 * This module is not part of the public API.
 *
 * @internal
 * @see {@link https://deepbox.dev/docs/optim-optimizers | Deepbox documentation}
 */

import {
  DeepboxError,
  DTypeError,
  IndexError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../core";
import {
  add,
  addScalar,
  type GradTensor,
  mulScalar,
  neg,
  relu,
  sub,
  type Tensor,
  tensor,
  where,
} from "../ndarray";
import { offsetFromFlatIndex } from "../ndarray/tensor/strides";

/**
 * Replace a parameter's underlying storage tensor in place. Optimizers that
 * run their update on a device (where the parameter buffer is opaque GPU
 * memory) compose the new parameter with device tensor ops and swap it in via
 * this helper, mirroring how `nn.Module.to` moves parameters. The `tensor` /
 * `_grad` fields are declared `readonly`, so the write goes through
 * `Reflect.set`.
 */
export function replaceParamStorage(
  param: GradTensor,
  field: "tensor" | "_grad",
  value: Tensor
): void {
  if (!Reflect.set(param, field, value)) {
    throw new DeepboxError(`optimizer: failed to update parameter ${field} on device`);
  }
}

/**
 * Device analogue of `Math.sign`, composed from device-dispatched ops so it can
 * run on opaque accelerator memory (the exported `sign` op is host-only). Returns
 * +1 where `t > 0`, -1 where `t < 0`, and 0 where `t === 0`, matching
 * `Math.sign` on finite values.
 *
 * @internal
 */
export function deviceSign(t: Tensor): Tensor {
  const one = tensor(1);
  const zero = tensor(0);
  // relu(t) is nonzero exactly where t > 0; relu(-t) exactly where t < 0.
  const pos = where(relu(t), one, zero);
  const negPart = where(relu(neg(t)), one, zero);
  return sub(pos, negPart);
}

/**
 * Device analogue of `Math.max(t, c)` for a scalar `c`, composed from
 * device-dispatched ops (the exported `maximum` op is host-only). Uses the
 * identity `max(t, c) = c + relu(t - c)`, which holds for finite values up to
 * one rounding of the final addition.
 *
 * @internal
 */
export function deviceMaxScalar(t: Tensor, c: number): Tensor {
  return addScalar(relu(addScalar(t, -c)), c);
}

/**
 * Device analogue of `Math.min(t, c)` for a scalar `c`, composed from
 * device-dispatched ops. Uses `min(t, c) = c - relu(c - t)`.
 *
 * @internal
 */
export function deviceMinScalar(t: Tensor, c: number): Tensor {
  return addScalar(mulScalar(relu(addScalar(neg(t), c)), -1), c);
}

/**
 * Device analogue of element-wise `maximum(a, b)`, composed from
 * device-dispatched ops via `max(a, b) = a + relu(b - a)` (equal to the true
 * maximum for finite values up to one rounding of the final addition).
 *
 * @internal
 */
export function deviceMaxTensor(a: Tensor, b: Tensor): Tensor {
  return add(a, relu(sub(b, a)));
}

/**
 * Supported floating-point typed array types for optimizer parameters.
 */
export type FloatTypedArray = Float32Array | Float64Array;

/**
 * True when the elements of a tensor are laid out densely in row-major order
 * (dimensions of size 1 may carry any stride). Optimizers update flat typed
 * arrays as `offset + i`, which is only right for such a layout.
 */
function isDenseRowMajor(shape: readonly number[], strides: readonly number[]): boolean {
  if (shape.length !== strides.length) return false;
  let expected = 1;
  for (let axis = shape.length - 1; axis >= 0; axis--) {
    const dim = shape[axis] ?? 1;
    if (dim === 0) return true;
    if (dim === 1) continue;
    if (strides[axis] !== expected) return false;
    expected *= dim;
  }
  return true;
}

function sameShape(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((dim, i) => dim === b[i]);
}

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
    throw new InvalidParameterError(`Invalid ${name}: ${value} (must be >= 0)`, name, value);
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
    throw new InvalidParameterError(`Invalid ${name}: ${value} (must be finite)`, name, value);
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
 * Validates that a parameter has a gradient and returns flat views of both.
 *
 * The parameter must be stored densely (row-major, no strides) because optimizers
 * update it in place as `offset + i`. A strided gradient is copied into a dense
 * buffer first (reported with `gradOffset` 0), so the gradient may be any view.
 *
 * @param param - Parameter to validate
 * @param optimizerName - Name of the optimizer for error messages
 * @returns Object containing gradient data, offset, parameter data, and offset
 * @throws {InvalidParameterError} If parameter doesn't require gradients
 * @throws {NotFittedError} If parameter has no gradient
 * @throws {DTypeError} If parameter or gradient has unsupported dtype
 * @throws {ShapeError} If the gradient shape differs from the parameter shape, or the
 *   parameter is a non-contiguous view
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

  if (!sameShape(param.tensor.shape, g.shape)) {
    throw new ShapeError(
      `Gradient shape must match parameter shape (param: [${param.tensor.shape}], grad: [${g.shape}])`
    );
  }

  if (!isDenseRowMajor(param.tensor.shape, param.tensor.strides)) {
    throw new ShapeError(
      `${optimizerName} optimizer requires a contiguous parameter (got a strided view with shape [${param.tensor.shape}])`
    );
  }

  if (!isDenseRowMajor(g.shape, g.strides)) {
    const dense = new (gradData.constructor as new (length: number) => FloatTypedArray)(g.size);
    const logicalStrides: number[] = new Array(g.shape.length);
    let running = 1;
    for (let axis = g.shape.length - 1; axis >= 0; axis--) {
      logicalStrides[axis] = running;
      running *= g.shape[axis] ?? 1;
    }
    for (let flat = 0; flat < g.size; flat++) {
      dense[flat] = gradData[offsetFromFlatIndex(flat, logicalStrides, g.strides, g.offset)] ?? 0;
    }
    return { grad: dense, gradOffset: 0, param: paramData, paramOffset: param.tensor.offset };
  }

  return {
    grad: gradData,
    gradOffset: g.offset,
    param: paramData,
    paramOffset: param.tensor.offset,
  };
}

/**
 * Validates that a parameter is stored as a dense float32 or float64 host buffer and returns
 * its flat view. Unlike {@link assertHasGradFloat} it does not look at the gradient, so it
 * also accepts a parameter whose gradient is `null`.
 *
 * @param param - Parameter to validate
 * @param optimizerName - Name of the optimizer for error messages
 * @returns The parameter data and its offset
 * @throws {InvalidParameterError} If the parameter doesn't require gradients
 * @throws {DTypeError} If the parameter has an unsupported dtype
 * @throws {ShapeError} If the parameter is a non-contiguous view
 */
export function assertFloatParam(
  param: GradTensor,
  optimizerName: string
): { param: FloatTypedArray; paramOffset: number } {
  if (!param.requiresGrad) {
    throw new InvalidParameterError(
      "Cannot optimize a parameter with requiresGrad=false",
      "requiresGrad",
      false
    );
  }
  const paramData = param.tensor.data;
  if (!isFloatTypedArray(paramData)) {
    throw new DTypeError(`${optimizerName} optimizer supports float32 and float64 parameters only`);
  }
  if (!isDenseRowMajor(param.tensor.shape, param.tensor.strides)) {
    throw new ShapeError(
      `${optimizerName} optimizer requires a contiguous parameter (got a strided view with shape [${param.tensor.shape}])`
    );
  }
  return { param: paramData, paramOffset: param.tensor.offset };
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
