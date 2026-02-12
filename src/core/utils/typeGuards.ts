import type { TypedArray } from "../types/common";

/**
 * Type guard to check if a value is one of the supported TypedArray types.
 *
 * Returns true only for the exact TypedArray subclasses in the Deepbox
 * {@link TypedArray} union: Float32Array, Float64Array, Int32Array,
 * BigInt64Array, and Uint8Array. Returns false for unsupported typed arrays
 * (e.g. Uint16Array, Int16Array), DataView, and regular arrays.
 *
 * @param value - The value to check
 * @returns True if value is a supported TypedArray, false otherwise
 *
 * @example
 * ```ts
 * import { isTypedArray } from 'deepbox/core';
 *
 * isTypedArray(new Float32Array(10));  // true
 * isTypedArray(new Uint16Array(10));   // false (unsupported)
 * isTypedArray([1, 2, 3]);             // false
 * isTypedArray(new DataView(new ArrayBuffer(10)));  // false
 * ```
 */
export function isTypedArray(value: unknown): value is TypedArray {
  return (
    value instanceof Float32Array ||
    value instanceof Float64Array ||
    value instanceof Int32Array ||
    value instanceof BigInt64Array ||
    value instanceof Uint8Array
  );
}
