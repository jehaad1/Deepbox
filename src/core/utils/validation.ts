import { DataValidationError } from "../errors/validation";
import type { Shape } from "../types/common";
import { DEVICES, type Device, isDevice } from "../types/device";
import { DTYPES, type DType, isDType } from "../types/dtype";

/**
 * Calculate the total number of elements from a shape.
 *
 * @param shape - The shape to calculate size from
 * @param name - Parameter name for error messages (default: 'shape')
 * @returns Total number of elements
 * @throws {DataValidationError} If shape is invalid or size exceeds safe integer range
 */
export function shapeToSize(shape: Shape): number;
export function shapeToSize(shape: unknown, name?: string): number;
export function shapeToSize(shape: unknown, name = "shape"): number {
  const validated = validateShape(shape, name);
  let size = 1;

  for (const dim of validated) {
    if (dim === 0) return 0;

    if (size > Number.MAX_SAFE_INTEGER / dim) {
      throw new DataValidationError(
        `${name} is too large; total size exceeds ${Number.MAX_SAFE_INTEGER}`
      );
    }
    size *= dim;
  }
  return size;
}

/**
 * Validate that a shape is well-formed.
 *
 * Checks that:
 * - Value is an array
 * - All elements are finite non-negative integers
 * - All elements are safe integers
 *
 * @param shape - The shape to validate
 * @param name - Parameter name for error messages (default: 'shape')
 * @returns The validated shape
 * @throws {DataValidationError} If shape is invalid
 */
export function validateShape(shape: Shape, name?: string): Shape;
export function validateShape(shape: unknown, name?: string): Shape;
export function validateShape(shape: unknown, name = "shape"): Shape {
  if (!Array.isArray(shape)) {
    throw new DataValidationError(`${name} must be an array of integers`);
  }

  const validated: number[] = [];
  for (let i = 0; i < shape.length; i++) {
    const dim: unknown = shape[i];

    if (typeof dim !== "number" || !Number.isFinite(dim) || !Number.isInteger(dim)) {
      throw new DataValidationError(
        `${name}[${i}] must be a finite integer; received ${String(dim)}`
      );
    }
    if (dim < 0) {
      throw new DataValidationError(`${name}[${i}] must be >= 0; received ${dim}`);
    }
    if (!Number.isSafeInteger(dim)) {
      throw new DataValidationError(
        `${name}[${i}] must be a safe integer (<= ${Number.MAX_SAFE_INTEGER}); received ${dim}`
      );
    }
    validated.push(dim);
  }

  return validated;
}

/**
 * Validate that a `dtype` is supported.
 *
 * @param dtype - The data type to validate
 * @param name - The parameter name for error messages
 * @returns The validated dtype (typed as {@link DType})
 * @throws {DataValidationError} If dtype is not valid
 */
export function validateDtype(dtype: DType, name?: string): DType;
export function validateDtype(dtype: unknown, name?: string): DType;
export function validateDtype(dtype: unknown, name = "dtype"): DType {
  if (!isDType(dtype)) {
    throw new DataValidationError(
      `${name} must be one of [${DTYPES.join(", ")}]; received ${String(dtype)}`
    );
  }
  return dtype;
}

/**
 * Validate that a `device` is supported.
 *
 * @param device - The device to validate
 * @param name - The parameter name for error messages
 * @returns The validated device (typed as {@link Device})
 * @throws {DataValidationError} If device is not valid
 */
export function validateDevice(device: Device, name?: string): Device;
export function validateDevice(device: unknown, name?: string): Device;
export function validateDevice(device: unknown, name = "device"): Device {
  if (!isDevice(device)) {
    throw new DataValidationError(
      `${name} must be one of [${DEVICES.join(", ")}]; received ${String(device)}`
    );
  }
  return device;
}

/**
 * Validate that a value is positive (> 0).
 *
 * @param value - The value to validate
 * @param name - The parameter name for error messages
 * @throws {DataValidationError} If value is not positive
 */
export function validatePositive(value: number, name: string): void {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new DataValidationError(`${name} must be a finite number; received ${String(value)}`);
  }
  if (value <= 0) {
    throw new DataValidationError(`${name} must be positive (> 0); received ${value}`);
  }
}

/**
 * Validate that a value is non-negative (>= 0).
 *
 * @param value - The value to validate
 * @param name - The parameter name for error messages
 * @throws {DataValidationError} If value is negative
 */
export function validateNonNegative(value: number, name: string): void {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new DataValidationError(`${name} must be a finite number; received ${String(value)}`);
  }
  if (value < 0) {
    throw new DataValidationError(`${name} must be non-negative (>= 0); received ${value}`);
  }
}

/**
 * Validate that a value is within a specified range [min, max].
 *
 * @param value - The value to validate
 * @param min - The minimum allowed value (inclusive)
 * @param max - The maximum allowed value (inclusive)
 * @param name - The parameter name for error messages
 * @throws {DataValidationError} If value is out of range or bounds are invalid
 */
export function validateRange(value: number, min: number, max: number, name: string): void {
  if (typeof min !== "number" || !Number.isFinite(min)) {
    throw new DataValidationError(`${name} min must be a finite number; received ${String(min)}`);
  }
  if (typeof max !== "number" || !Number.isFinite(max)) {
    throw new DataValidationError(`${name} max must be a finite number; received ${String(max)}`);
  }
  if (min > max) {
    throw new DataValidationError(`${name} min must be <= max; received min=${min}, max=${max}`);
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new DataValidationError(`${name} must be a finite number; received ${String(value)}`);
  }
  if (value < min || value > max) {
    throw new DataValidationError(`${name} must be in range [${min}, ${max}]; received ${value}`);
  }
}

/**
 * Validate that a value is a safe integer.
 *
 * @param value - The value to validate
 * @param name - The parameter name for error messages
 * @throws {DataValidationError} If value is not a safe integer
 */
export function validateInteger(value: number, name: string): void {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new DataValidationError(`${name} must be a finite number; received ${String(value)}`);
  }
  if (!Number.isInteger(value)) {
    throw new DataValidationError(`${name} must be an integer; received ${value}`);
  }
  if (!Number.isSafeInteger(value)) {
    throw new DataValidationError(
      `${name} must be a safe integer (<= ${Number.MAX_SAFE_INTEGER}); received ${value}`
    );
  }
}

/**
 * Validate that a value is one of the allowed options.
 *
 * @param value - The value to validate
 * @param options - The array of allowed values
 * @param name - The parameter name for error messages
 * @throws {DataValidationError} If value is not in options
 */
export function validateOneOf<T extends string>(
  value: unknown,
  options: readonly T[],
  name: string
): asserts value is T {
  if (typeof value !== "string") {
    throw new DataValidationError(`${name} must be a string; received ${typeof value}`);
  }
  let found = false;
  for (const opt of options) {
    if (opt === value) {
      found = true;
      break;
    }
  }
  if (!found) {
    throw new DataValidationError(
      `${name} must be one of [${options.join(", ")}]; received ${String(value)}`
    );
  }
}

/**
 * Validate that a value is an array.
 *
 * @param arr - The value to validate
 * @param name - The parameter name for error messages
 * @throws {DataValidationError} If value is not an array
 */
export function validateArray(arr: unknown, name: string): asserts arr is unknown[] {
  if (!Array.isArray(arr)) {
    throw new DataValidationError(`${name} must be an array; received ${typeof arr}`);
  }
}
