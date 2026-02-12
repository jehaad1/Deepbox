/**
 * Type-safe typed array accessor utilities.
 *
 * These utilities provide safe access to TypedArray elements without
 * requiring unsafe type assertions. They handle the `noUncheckedIndexedAccess`
 * TypeScript configuration properly.
 *
 * Element accessors for TypedArray storage throw {@link IndexError} on
 * out-of-bounds access to surface stride/index bugs immediately rather
 * than silently returning zero.
 */

import { IndexError } from "../errors/indexError";
import { DataValidationError } from "../errors/validation";
import type { TypedArray } from "../types/common";

const MAX_SAFE_BIGINT = BigInt(Number.MAX_SAFE_INTEGER);
const MIN_SAFE_BIGINT = BigInt(Number.MIN_SAFE_INTEGER);

/**
 * Numeric typed arrays (excludes BigInt64Array).
 */
export type NumericTypedArray = Float32Array | Float64Array | Int32Array | Uint8Array;

/**
 * Get an element from a numeric typed array at the specified index.
 *
 * @param arr - The numeric typed array
 * @param index - The index to access
 * @returns The element at the index
 * @throws {IndexError} If the index is out of bounds
 */
export function getNumericElement(arr: NumericTypedArray, index: number): number {
  const value = arr[index];
  if (value === undefined) {
    throw new IndexError(
      `Index ${index} is out of bounds for typed array of length ${arr.length}`,
      { index, validRange: [0, arr.length - 1] }
    );
  }
  return value;
}

/**
 * Get an element from a BigInt64Array at the specified index.
 *
 * @param arr - The BigInt64Array
 * @param index - The index to access
 * @returns The element at the index
 * @throws {IndexError} If the index is out of bounds
 */
export function getBigIntElement(arr: BigInt64Array, index: number): bigint {
  const value = arr[index];
  if (value === undefined) {
    throw new IndexError(
      `Index ${index} is out of bounds for BigInt64Array of length ${arr.length}`,
      { index, validRange: [0, arr.length - 1] }
    );
  }
  return value;
}

/**
 * Get an element from a TypedArray, returning as number.
 * For BigInt64Array, converts to number (throws if out of safe integer range).
 *
 * @param arr - The typed array
 * @param index - The index to access
 * @returns The element at the index as a number
 * @throws {IndexError} If the index is out of bounds
 * @throws {DataValidationError} If a bigint value exceeds the safe integer range
 */
export function getElementAsNumber(arr: TypedArray, index: number): number {
  if (arr instanceof BigInt64Array) {
    const value = arr[index];
    if (value === undefined) {
      throw new IndexError(
        `Index ${index} is out of bounds for BigInt64Array of length ${arr.length}`,
        { index, validRange: [0, arr.length - 1] }
      );
    }
    if (value > MAX_SAFE_BIGINT || value < MIN_SAFE_BIGINT) {
      throw new DataValidationError(
        `BigInt value at index ${index} exceeds safe integer range; received ${value.toString()}`
      );
    }
    return Number(value);
  }
  const value = arr[index];
  if (value === undefined) {
    throw new IndexError(
      `Index ${index} is out of bounds for typed array of length ${arr.length}`,
      { index, validRange: [0, arr.length - 1] }
    );
  }
  return value;
}

/**
 * Get an element from a shape array at the specified index.
 * Returns 1 if the index is out of bounds (used in broadcasting where
 * missing leading dimensions are treated as size 1).
 *
 * @param shape - The shape array
 * @param index - The index to access
 * @returns The dimension at the index, or 1 if out of bounds
 */
export function getShapeDim(shape: readonly number[], index: number): number {
  const value = shape[index];
  return value !== undefined ? value : 1;
}

/**
 * Get an element from a number array at the specified index.
 * Returns the provided default value if the index is out of bounds.
 *
 * @param arr - The number array
 * @param index - The index to access
 * @param defaultValue - The default value to return if out of bounds (default: 0)
 * @returns The element at the index, or the default value if out of bounds
 */
export function getArrayElement(arr: readonly number[], index: number, defaultValue = 0): number {
  const value = arr[index];
  return value !== undefined ? value : defaultValue;
}

/**
 * Safely get a string from a string array at the specified index.
 * Returns empty string if the index is out of bounds.
 *
 * @param arr - The string array
 * @param index - The index to access
 * @returns The string at the index, or empty string if out of bounds
 */
export function getStringElement(arr: readonly string[], index: number): string {
  const value = arr[index];
  return value !== undefined ? value : "";
}

/**
 * Create a readonly number array from mutable array with proper typing.
 * This avoids the need for `as Shape` assertions.
 *
 * @param arr - The mutable number array
 * @returns The same array typed as readonly
 */
export function asReadonlyArray<T extends number[]>(arr: T): readonly number[] {
  return arr;
}

/**
 * Type guard to check if a typed array is a BigInt64Array.
 *
 * @param arr - The typed array to check
 * @returns True if the array is a BigInt64Array
 */
export function isBigInt64Array(arr: TypedArray): arr is BigInt64Array {
  return arr instanceof BigInt64Array;
}

/**
 * Type guard to check if a typed array is a numeric (non-BigInt) array.
 *
 * @param arr - The typed array to check
 * @returns True if the array is a numeric typed array
 */
export function isNumericTypedArray(arr: TypedArray): arr is NumericTypedArray {
  return !(arr instanceof BigInt64Array);
}
