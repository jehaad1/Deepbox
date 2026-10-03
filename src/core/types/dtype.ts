/**
 * Supported data types for tensors.
 *
 * - `float16`: 16-bit floating point (half precision, IEEE 754)
 * - `bfloat16`: 16-bit Brain Floating Point (same exponent as float32)
 * - `float32`: 32-bit floating point (single precision)
 * - `float64`: 64-bit floating point (double precision)
 * - `int32`: 32-bit signed integer
 * - `int64`: 64-bit signed integer (BigInt)
 * - `uint8`: 8-bit unsigned integer
 * - `bool`: Boolean values (stored as uint8)
 * - `complex64`: Reserved. Complex number with float32 real and imaginary parts.
 *   Tensors cannot be created with this dtype yet (a `DTypeError` is thrown);
 *   use `Complex64Array` for complex data.
 * - `complex128`: Reserved. Complex number with float64 real and imaginary parts.
 *   Tensors cannot be created with this dtype yet (a `DTypeError` is thrown);
 *   use `Complex128Array` for complex data.
 * - `string`: String values (limited support; backed by a `string[]`)
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
export type DType =
  | "float16"
  | "bfloat16"
  | "float32"
  | "float64"
  | "int32"
  | "int64"
  | "uint8"
  | "bool"
  | "complex64"
  | "complex128"
  | "string";

/**
 * Numeric DTypes whose JavaScript element type is `number`.
 * Excludes `int64` (BigInt), `complex64`/`complex128`, and `string`.
 */
export type ScalarDType =
  | "float16"
  | "bfloat16"
  | "float32"
  | "float64"
  | "int32"
  | "uint8"
  | "bool";

/**
 * Maps a DType to its JavaScript element type.
 *
 * - `string` → `string`
 * - `int64`  → `bigint`
 * - all others → `number` (the complex dtypes are reserved and have no tensor storage yet)
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
 * // ['float16', 'bfloat16', 'float32', 'float64', 'int32', 'int64', 'uint8', 'bool',
 * //  'complex64', 'complex128', 'string']
 * ```
 */
export const DTYPES: readonly DType[] = [
  "float16",
  "bfloat16",
  "float32",
  "float64",
  "int32",
  "int64",
  "uint8",
  "bool",
  "complex64",
  "complex128",
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
