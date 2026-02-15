/**
 * Supported data types for tensors.
 *
 * - `float32`: 32-bit floating point (single precision)
 * - `float64`: 64-bit floating point (double precision)
 * - `int32`: 32-bit signed integer
 * - `int64`: 64-bit signed integer (BigInt)
 * - `uint8`: 8-bit unsigned integer
 * - `bool`: Boolean values (stored as uint8)
 * - `string`: String values (limited support)
 *
 * @example
 * ```ts
 * import type { DType } from 'deepbox/core';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const dtype: DType = 'float32';
 * const x = tensor([1, 2, 3], { dtype });
 * ```
 *
 * @see {@link https://deepbox.dev/docs/core-types | Deepbox Core Types}
 */
export type DType = "float32" | "float64" | "int32" | "int64" | "uint8" | "bool" | "string";

/**
 * Numeric DTypes whose JavaScript element type is `number`.
 * Excludes `int64` (BigInt) and `string`.
 */
export type ScalarDType = "float32" | "float64" | "int32" | "uint8" | "bool";

/**
 * Maps a DType to its JavaScript element type.
 *
 * - `string` → `string`
 * - `int64`  → `bigint`
 * - all others → `number`
 */
export type ElementOf<D extends DType> = D extends "string"
  ? string
  : D extends "int64"
    ? bigint
    : number;

/**
 * Array of all supported data types.
 *
 * Use this constant for validation or UI selection.
 *
 * @example
 * ```ts
 * import { DTYPES } from 'deepbox/core';
 *
 * console.log(DTYPES);
 * // ['float32', 'float64', 'int32', 'int64', 'uint8', 'bool', 'string']
 * ```
 */
export const DTYPES: readonly DType[] = [
  "float32",
  "float64",
  "int32",
  "int64",
  "uint8",
  "bool",
  "string",
];

/**
 * Type guard to check if a value is a valid DType.
 *
 * @param value - The value to check
 * @returns True if value is a valid DType, false otherwise
 *
 * @example
 * ```ts
 * import { isDType } from 'deepbox/core';
 *
 * if (isDType('float32')) {
 *   console.log('Valid dtype');
 * }
 *
 * isDType('float128');  // false
 * isDType('int32');     // true
 * ```
 */
export function isDType(value: unknown): value is DType {
  if (typeof value !== "string") {
    return false;
  }
  for (const d of DTYPES) {
    if (d === value) {
      return true;
    }
  }
  return false;
}
