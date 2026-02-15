/**
 * Represents the dimensions of a multi-dimensional array (tensor).
 *
 * Each element represents the size of that dimension.
 * For example, [2, 3, 4] represents a 3D array with dimensions 2×3×4.
 *
 * @example
 * ```ts
 * const shape1D: Shape = [5];           // 1D array with 5 elements
 * const shape2D: Shape = [3, 4];        // 2D array (matrix) 3×4
 * const shape3D: Shape = [2, 3, 4];     // 3D array 2×3×4
 * const scalar: Shape = [];             // Scalar (0-dimensional)
 * ```
 */
export type Shape = readonly number[];

/**
 * Union type of all TypedArray types supported by Deepbox.
 *
 * TypedArrays provide efficient storage and manipulation of numeric data
 * in JavaScript, offering better performance than regular arrays for
 * numerical computations.
 *
 * @see {@link https://deepbox.dev/docs/core-types | Deepbox Core Types}
 *
 * @example
 * ```ts
 * import type { TypedArray } from 'deepbox/core';
 *
 * function processArray(arr: TypedArray): number {
 *   let sum = 0;
 *   for (let i = 0; i < arr.length; i++) {
 *     sum += Number(arr[i]);
 *   }
 *   return sum;
 * }
 * ```
 */
export type TypedArray = Float32Array | Float64Array | Int32Array | BigInt64Array | Uint8Array;

/**
 * Backing storage for a tensor.
 *
 * Numeric tensors are backed by TypedArrays.
 * String tensors are backed by a string array.
 */
export type TensorStorage = TypedArray | string[];

/**
 * Axis identifier.
 *
 * Can be an integer index (0, 1, ...) or a string alias:
 * - `0` aliases: "index", "rows"
 * - `1` aliases: "columns"
 */
export type Axis = number | "index" | "rows" | "columns";
