/**
 * @see {@link https://deepbox.dev/docs/core-utils | Utilities, serialization & parallelism}
 */

import type { TypedArray } from "../types/common";

/**
 * Type guard to check if a value is one of the supported TypedArray types.
 *
 * Returns true for instances of the TypedArray classes in the Deepbox
 * {@link TypedArray} union: Float32Array, Float64Array, Int32Array,
 * BigInt64Array, and Uint8Array (subclasses such as Node.js `Buffer` included).
 * Returns false for unsupported typed arrays (e.g. Uint16Array, Int16Array,
 * Uint8ClampedArray), DataView, and regular arrays. The check uses
 * `instanceof`, so arrays created in another realm (iframe, vm context) are
 * not recognised.
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
