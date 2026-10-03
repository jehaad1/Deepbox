/**
 * @see {@link https://deepbox.dev/docs/core-utils | Utilities, serialization & parallelism}
 */

import { DTypeError } from "../errors/dtype";
import type { DType } from "../types/index";

/** All DType values except "string". Used when an operation requires numeric data. */
export type NumericDType = Exclude<DType, "string">;

/**
 * Ensure a dtype is numeric (non-string).
 *
 * @param dtype - Data type identifier
 * @param context - Context string for error messages
 * @returns The same dtype narrowed to numeric types
 * @throws {DTypeError} If dtype is 'string'
 */
export function ensureNumericDType(dtype: DType, context = "operation"): NumericDType {
  if (dtype === "string") {
    throw new DTypeError(`${context} does not support string dtype`);
  }
  return dtype;
}

/**
 * Get TypedArray constructor for a given DType.
 *
 * Maps Deepbox dtype strings to JavaScript TypedArray constructors.
 * Used internally for allocating tensor storage.
 *
 * **Mapping:**
 * - `float16`, `bfloat16`, `float32`, `complex64` → `Float32Array` (the working component
 *   type; half-precision tensors store through a software-emulated wrapper over
 *   `Uint16Array`; the complex dtypes are reserved, tensors cannot use them yet, and
 *   their component type is listed here for completeness)
 * - `float64`, `complex128` → `Float64Array`
 * - `int32` → `Int32Array`
 * - `int64` → `BigInt64Array`
 * - `uint8` → `Uint8Array`
 * - `bool` → `Uint8Array` (1 byte per boolean)
 * - `string` → Not supported (throws error)
 *
 * @param dtype - Data type identifier
 * @returns TypedArray constructor for the given dtype
 * @throws {DTypeError} If dtype is 'string' (strings are stored in a plain array)
 *
 * @example
 * ```ts
 * const Ctor = dtypeToTypedArrayCtor('float32');
 * const arr = new Ctor(10); // Float32Array with 10 elements
 * ```
 */
export function dtypeToTypedArrayCtor(
  dtype: DType
):
  | Float32ArrayConstructor
  | Float64ArrayConstructor
  | Int32ArrayConstructor
  | BigInt64ArrayConstructor
  | Uint8ArrayConstructor {
  switch (dtype) {
    case "float16":
    case "bfloat16":
    case "float32":
    case "complex64":
      return Float32Array;
    case "float64":
    case "complex128":
      return Float64Array;
    case "int32":
      return Int32Array;
    case "int64":
      return BigInt64Array;
    case "uint8":
      return Uint8Array;
    case "bool":
      return Uint8Array;
    case "string":
      throw new DTypeError("string dtype is not supported for TypedArray storage");
    default: {
      const _exhaustive: never = dtype;
      throw new DTypeError(`Unsupported dtype: ${String(_exhaustive)}`);
    }
  }
}

/**
 * Pick the dtype for numeric data that is built without an explicit dtype.
 *
 * Follows the global default dtype when every value survives it unchanged:
 * floating-point defaults always do, integer defaults need integral values in range, and
 * `bool` needs values that are all 0 or 1. Otherwise (including the `string` and complex
 * defaults, which cannot hold plain numeric data) it returns `float32`, so no value is
 * silently truncated, wrapped or collapsed to `true`.
 *
 * @internal
 */
export function resolveLosslessDType(configured: DType, data: ArrayLike<number>): DType {
  let min = 0;
  let max = 0;
  switch (configured) {
    case "float16":
    case "bfloat16":
    case "float32":
    case "float64":
      return configured;
    case "bool":
      min = 0;
      max = 1;
      break;
    case "uint8":
      min = 0;
      max = 255;
      break;
    case "int32":
      min = -2147483648;
      max = 2147483647;
      break;
    case "int64":
      min = Number.MIN_SAFE_INTEGER;
      max = Number.MAX_SAFE_INTEGER;
      break;
    default:
      return "float32";
  }
  for (let i = 0; i < data.length; i++) {
    const v = data[i] as number;
    if (!Number.isInteger(v) || v < min || v > max) return "float32";
  }
  return configured;
}

/** Promotion rank of the non-float, non-complex dtypes. */
const INTEGER_RANK: Readonly<Record<string, number>> = { bool: 0, uint8: 1, int32: 2, int64: 3 };

/** Promotion rank of the float dtypes: the two 16-bit types share the lowest rank. */
const FLOAT_RANK: Readonly<Record<string, number>> = {
  float16: 1,
  bfloat16: 1,
  float32: 2,
  float64: 3,
};

/**
 * True for `float16`, `bfloat16`, `float32` and `float64`.
 *
 * @param dtype - Data type identifier
 *
 * @example
 * ```ts
 * isFloatDType("float32"); // true
 * isFloatDType("int32"); // false
 * ```
 */
export function isFloatDType(dtype: DType): boolean {
  return dtype === "float16" || dtype === "bfloat16" || dtype === "float32" || dtype === "float64";
}

/**
 * Result dtype for operations that produce fractional values (exp, sqrt, mean, softmax, ...).
 *
 * Float dtypes are kept as they are; integer and bool input computes in `float32`,
 * Deepbox's default float dtype.
 *
 * @param dtype - Input dtype
 * @returns `dtype` itself for float dtypes, otherwise `"float32"`
 *
 * @example
 * ```ts
 * toFloatDType("float64"); // "float64"
 * toFloatDType("int32"); // "float32"
 * ```
 */
export function toFloatDType(dtype: DType): "float16" | "bfloat16" | "float32" | "float64" {
  if (dtype === "float16" || dtype === "bfloat16" || dtype === "float64") return dtype;
  return "float32";
}

/**
 * Result dtype of a tensor-tensor binary operation (PyTorch-style type promotion).
 *
 * The order is `bool < uint8 < int32 < int64 < float16, bfloat16 < float32 < float64`.
 * Within one category the wider type wins, an integer combined with a float gives the float
 * type, and `float16` combined with `bfloat16` gives `float32`. A complex dtype absorbs real
 * dtypes (`complex64` with `float64` gives `complex128`). `string` never promotes.
 *
 * @param a - First dtype
 * @param b - Second dtype
 * @returns The common dtype both operands convert to
 * @throws {DTypeError} If either dtype is `string`
 *
 * @example
 * ```ts
 * promoteTypes("int32", "float32"); // "float32"
 * promoteTypes("uint8", "int64"); // "int64"
 * promoteTypes("float16", "bfloat16"); // "float32"
 * ```
 */
export function promoteTypes(a: DType, b: DType): DType {
  if (a === "string" || b === "string") {
    throw new DTypeError(`Cannot promote dtypes ${a} and ${b}: string does not promote`);
  }
  if (a === b) return a;

  const aComplex = a === "complex64" || a === "complex128";
  const bComplex = b === "complex64" || b === "complex128";
  if (aComplex || bComplex) {
    if (a === "complex128" || b === "complex128" || a === "float64" || b === "float64") {
      return "complex128";
    }
    return "complex64";
  }

  const aFloat = FLOAT_RANK[a];
  const bFloat = FLOAT_RANK[b];
  if (aFloat !== undefined && bFloat !== undefined) {
    if (aFloat === bFloat) return aFloat === 1 ? "float32" : a;
    return aFloat > bFloat ? a : b;
  }
  if (aFloat !== undefined) return a;
  if (bFloat !== undefined) return b;

  const aInt = INTEGER_RANK[a] ?? 0;
  const bInt = INTEGER_RANK[b] ?? 0;
  return aInt >= bInt ? a : b;
}
