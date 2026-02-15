import { DTypeError } from "../errors/dtype";
import type { DType } from "../types/index";

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
 * - `float32` → `Float32Array`
 * - `float64` → `Float64Array`
 * - `int32` → `Int32Array`
 * - `int64` → `BigInt64Array`
 * - `uint8` → `Uint8Array`
 * - `bool` → `Uint8Array` (1 byte per boolean)
 * - `string` → Not supported (throws error)
 *
 * @param dtype - Data type identifier
 * @returns TypedArray constructor for the given dtype
 * @throws {DTypeError} If dtype is 'string' (not yet supported)
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
    case "float32":
      return Float32Array;
    case "float64":
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
