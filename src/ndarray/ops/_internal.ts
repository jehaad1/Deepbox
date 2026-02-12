/**
 * Shared internal helpers for ndarray/ops.
 *
 * These utilities are used across multiple ops modules (arithmetic, comparison,
 * logical, math, activation, trigonometry, reduction). Centralising them here
 * eliminates the ~30+ copy-pasted copies that previously lived in each file.
 *
 * @internal
 */

import {
  DataValidationError,
  DTypeError,
  getBigIntElement,
  getNumericElement,
  type TypedArray,
} from "../../core";
import { offsetFromFlatIndex } from "../tensor/strides";
import { isBigIntArray } from "../tensor/Tensor";

/**
 * Resolve the physical buffer offset for a logical flat index.
 *
 * For contiguous tensors this is a simple addition; for strided views it
 * falls back to the general `offsetFromFlatIndex` computation.
 */
export function flatOffset(
  flat: number,
  offset: number,
  contiguous: boolean,
  logicalStrides: readonly number[],
  strides: readonly number[]
): number {
  return contiguous ? offset + flat : offsetFromFlatIndex(flat, logicalStrides, strides, offset);
}

/**
 * Convert a BigInt to a Number, throwing if the value exceeds
 * `Number.MAX_SAFE_INTEGER`.
 */
export function bigintToNumberSafe(v: bigint): number {
  const max = BigInt(Number.MAX_SAFE_INTEGER);
  const min = -max;
  if (v > max || v < min) {
    throw new DataValidationError("int64 value is too large to safely convert to number");
  }
  return Number(v);
}

/**
 * Read an element from a TypedArray as a `number`, converting BigInt
 * values via `bigintToNumberSafe`.
 */
export function readAsNumberSafe(data: TypedArray, idx: number): number {
  if (isBigIntArray(data)) {
    return bigintToNumberSafe(getBigIntElement(data, idx));
  }
  return getNumericElement(data, idx);
}

/**
 * Read an element from a TypedArray, preserving its original type
 * (`bigint` or `number`).
 */
export function readElement(data: TypedArray, idx: number): bigint | number {
  if (isBigIntArray(data)) {
    return getBigIntElement(data, idx);
  }
  return getNumericElement(data, idx);
}

/**
 * Read an element as `number`, converting BigInt via `Number()` without
 * the safe-integer check.  Use when the caller tolerates precision loss
 * (e.g. comparison ops that only need ordering).
 */
export function readAsNumber(data: TypedArray, idx: number): number {
  if (isBigIntArray(data)) {
    return Number(getBigIntElement(data, idx));
  }
  return getNumericElement(data, idx);
}

/**
 * Assert that tensor data is a numeric TypedArray (not string[]).
 *
 * @param opName - Operation name for the error message
 */
export function requireNumericData(data: TypedArray | string[], opName: string): TypedArray {
  if (Array.isArray(data)) {
    throw new DTypeError(`${opName} is not implemented for string dtype`);
  }
  return data;
}
