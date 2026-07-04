/**
 * Internal utilities for the stats package.
 *
 * This module contains internal helper functions used by the stats package.
 * Functions are exported for use by other stats modules but are not part
 * of the stable public API exported from `src/stats/index.ts`.
 *
 * Some functions (particularly CDFs and special functions like `normalCdf`,
 * `studentTCdf`, `logGamma`, etc.) may be promoted to the public API in
 * future versions if there is user demand.
 *
 * @internal
 * @module stats/_internal
 * @see {@link https://deepbox.dev/docs/stats-descriptive | Deepbox documentation}
 */

import type { Axis, Shape } from "../core";
import {
  DTypeError,
  getElementAsNumber,
  InvalidParameterError,
  normalizeAxis,
  ShapeError,
  validateShape,
} from "../core";

import type { NumericTypedArray } from "../core/utils/typed_array_access";
import { Tensor } from "../ndarray";
import { isContiguous } from "../ndarray/tensor/strides";

/**
 * Sum `raw[off .. off+n)`. The `instanceof` split gives V8 a concrete element
 * type at each load site: `Tensor.data` is a wide typed-array union, so a
 * single shared loop over it stays megamorphic and ~4-5x slower.
 */
function contiguousSum(raw: NumericTypedArray, off: number, n: number): number {
  let s = 0;
  if (raw instanceof Float64Array) {
    for (let i = 0; i < n; i++) s += raw[off + i] as number;
  } else if (raw instanceof Float32Array) {
    for (let i = 0; i < n; i++) s += raw[off + i] as number;
  } else {
    for (let i = 0; i < n; i++) s += raw[off + i] as number;
  }
  return s;
}

/**
 * Copy `raw[off .. off+n)` into a fresh `Float64Array`. The `instanceof` split
 * keeps each copy loop monomorphic (see {@link contiguousSum}).
 */
export function copyContiguousToF64(raw: NumericTypedArray, off: number, n: number): Float64Array {
  const out = new Float64Array(n);
  if (raw instanceof Float64Array) {
    out.set(raw.subarray(off, off + n));
  } else if (raw instanceof Float32Array) {
    for (let i = 0; i < n; i++) out[i] = raw[off + i] as number;
  } else {
    for (let i = 0; i < n; i++) out[i] = raw[off + i] as number;
  }
  return out;
}

/**
 * In-place quickselect: reorders `arr` so `arr[k]` holds the k-th smallest
 * value and returns it. O(n) average (median-of-three pivot). Mutates `arr`.
 */
export function quickSelectF64(arr: Float64Array, k: number): number {
  let lo = 0;
  let hi = arr.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    const a = arr[lo] as number;
    const b = arr[mid] as number;
    const c = arr[hi] as number;
    let pivotIdx: number;
    if ((a <= b && b <= c) || (c <= b && b <= a)) pivotIdx = mid;
    else if ((b <= a && a <= c) || (c <= a && a <= b)) pivotIdx = lo;
    else pivotIdx = hi;
    const pivotVal = arr[pivotIdx] as number;
    arr[pivotIdx] = arr[hi] as number;
    arr[hi] = pivotVal;
    let storeIdx = lo;
    for (let i = lo; i < hi; i++) {
      if ((arr[i] as number) < pivotVal) {
        const tmp = arr[storeIdx] as number;
        arr[storeIdx] = arr[i] as number;
        arr[i] = tmp;
        storeIdx++;
      }
    }
    arr[hi] = arr[storeIdx] as number;
    arr[storeIdx] = pivotVal;
    if (storeIdx === k) return pivotVal;
    if (storeIdx < k) lo = storeIdx + 1;
    else hi = storeIdx - 1;
  }
  return arr[lo] as number;
}

/**
 * Median of a mutable `Float64Array` via O(n) quickselect (NaN propagates, per
 * NumPy semantics). Mutates `arr`.
 */
export function quickMedianF64(arr: Float64Array): number {
  const n = arr.length;
  for (let i = 0; i < n; i++) {
    if (Number.isNaN(arr[i] as number)) return Number.NaN;
  }
  const mid = Math.floor(n / 2);
  if (n % 2 === 1) return quickSelectF64(arr, mid);
  // Even length: upper is the k=mid selection; after partitioning, the lower
  // middle value is the max of arr[0..mid-1].
  const upper = quickSelectF64(arr, mid);
  let lower = arr[0] as number;
  for (let i = 1; i < mid; i++) {
    const v = arr[i] as number;
    if (v > lower) lower = v;
  }
  return (lower + upper) / 2;
}

/**
 * Sum of squared deviations (M2) over `raw[off .. off+n)` via the textbook
 * two-pass method — compute the mean, then sum `(x - mean)^2`. This is the
 * numerically stable form NumPy/SciPy use for a full array; it avoids Welford's
 * per-element division (~3x faster) while agreeing to ~1e-14 relative. Both
 * passes are monomorphic per typed-array kind (see {@link contiguousSum}).
 */
function contiguousM2(raw: NumericTypedArray, off: number, n: number): number {
  const mean = contiguousSum(raw, off, n) / n;
  let m2 = 0;
  if (raw instanceof Float64Array) {
    for (let i = 0; i < n; i++) {
      const d = (raw[off + i] as number) - mean;
      m2 += d * d;
    }
  } else if (raw instanceof Float32Array) {
    for (let i = 0; i < n; i++) {
      const d = (raw[off + i] as number) - mean;
      m2 += d * d;
    }
  } else {
    for (let i = 0; i < n; i++) {
      const d = (raw[off + i] as number) - mean;
      m2 += d * d;
    }
  }
  return m2;
}

/**
 * Type representing axis specification for reduction operations.
 * Can be a single axis number/alias or an array of axis numbers/aliases.
 *
 * @example
 * ```ts
 * const axis1: AxisLike = 0;        // Single axis
 * const axis2: AxisLike = [0, 1];   // Multiple axes
 * const axis3: AxisLike = "rows";   // Alias
 * ```
 */
export type AxisLike = Axis | readonly Axis[];

/**
 * Normalizes axis specification to a sorted array of non-negative axis indices.
 *
 * Converts negative indices to positive, validates bounds, removes duplicates,
 * and returns a sorted array. Returns empty array if axis is undefined.
 *
 * @param axis - Axis specification (single number, array, or undefined)
 * @param ndim - Number of dimensions in the tensor
 * @returns Sorted array of unique, non-negative axis indices
 * @throws {RangeError} If any axis is out of bounds for the given ndim
 *
 * @example
 * ```ts
 * normalizeAxes([-1], 3);      // Returns [2]
 * normalizeAxes([1, 0, 1], 3); // Returns [0, 1] (sorted, deduplicated)
 * normalizeAxes(undefined, 3); // Returns []
 * ```
 */
export function normalizeAxes(axis: AxisLike | undefined, ndim: number): readonly number[] {
  if (axis === undefined) return [];
  const axesInput: Axis[] = Array.isArray(axis) ? [...axis] : [axis];

  const seen = new Set<number>();
  const result: number[] = [];

  for (const ax of axesInput) {
    const norm = normalizeAxis(ax, ndim);
    if (!seen.has(norm)) {
      seen.add(norm);
      result.push(norm);
    }
  }

  // Sort for consistency with previous behavior
  return result.sort((a, b) => a - b);
}

/**
 * Computes the output shape after reduction along specified axes.
 *
 * When keepdims=true, reduced dimensions become 1.
 * When keepdims=false, reduced dimensions are removed entirely.
 *
 * @param shape - Original tensor shape
 * @param axes - Axes to reduce over (must be normalized)
 * @param keepdims - Whether to keep reduced dimensions as size 1
 * @returns New shape after reduction
 * @throws {ShapeError} If shape dimensions are invalid
 *
 * @example
 * ```ts
 * reducedShape([3, 4, 5], [1], false);    // [3, 5]
 * reducedShape([3, 4, 5], [1], true);     // [3, 1, 5]
 * reducedShape([3, 4, 5], [], false);     // []
 * reducedShape([3, 4, 5], [], true);      // [1, 1, 1]
 * ```
 */
export function reducedShape(shape: Shape, axes: readonly number[], keepdims: boolean): Shape {
  if (axes.length === 0) {
    return keepdims ? new Array<number>(shape.length).fill(1) : [];
  }

  const reduce = new Set(axes);
  const out: number[] = [];

  for (let i = 0; i < shape.length; i++) {
    const d = shape[i];
    if (d === undefined) throw new ShapeError("Internal error: missing shape dimension");

    if (reduce.has(i)) {
      if (keepdims) out.push(1);
    } else {
      out.push(d);
    }
  }

  validateShape(out);
  return out;
}

/**
 * Computes row-major (C-order) strides for a given shape.
 *
 * Strides define how many elements to skip in memory to move one position
 * along each dimension. Computed in row-major order where the last dimension
 * is contiguous in memory.
 *
 * @param shape - Tensor shape
 * @returns Array of strides for each dimension
 *
 * @example
 * ```ts
 * computeStrides([3, 4, 5]); // Returns [20, 5, 1]
 * computeStrides([2, 3]);    // Returns [3, 1]
 * ```
 */
export function computeStrides(shape: readonly number[]): readonly number[] {
  const strides = new Array<number>(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i] ?? 0;
  }
  return strides;
}

/**
 * Asserts that two tensors have the same total number of elements.
 *
 * Used for operations that require element-wise correspondence between tensors,
 * regardless of their shapes (e.g., correlation between flattened arrays).
 *
 * @param a - First tensor
 * @param b - Second tensor
 * @param name - Name of the calling function (for error messages)
 * @throws {InvalidParameterError} If tensors have different sizes
 *
 * @example
 * ```ts
 * assertSameSize(tensor([1, 2, 3]), tensor([4, 5, 6]), "pearsonr"); // OK
 * assertSameSize(tensor([1, 2]), tensor([3, 4, 5]), "pearsonr");    // Throws
 * ```
 */
export function assertSameSize(a: Tensor, b: Tensor, name: string): void {
  if (a.size !== b.size) {
    throw new InvalidParameterError(
      `${name}: tensors must have the same number of elements; got ${a.size} and ${b.size}`,
      "size",
      { a: a.size, b: b.size }
    );
  }
}

/**
 * Extracts a numeric value from a tensor at a specific memory offset.
 *
 * Handles type conversion from bigint to number and validates dtype.
 * This is a low-level accessor used by reduction operations.
 *
 * @param t - Tensor to read from
 * @param offset - Memory offset in the underlying data array
 * @returns Numeric value at the offset
 * @throws {DTypeError} If tensor has string dtype
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3]);
 * getNumberAt(t, t.offset + 1); // Returns 2
 * ```
 */
export function getNumberAt(t: Tensor, offset: number): number {
  if (t.dtype === "string" || Array.isArray(t.data)) {
    throw new DTypeError("operation not supported for string dtype");
  }

  return getElementAsNumber(t.data, offset);
}

/**
 * Assigns average ranks to values with tie tracking.
 *
 * Ranks are 1-indexed. Ties receive the average of their rank positions.
 * Returns the sum of (t^3 - t) over tied groups, which is used for tie correction.
 *
 * @param values - Input values to rank
 * @returns Object containing ranks and tie sum
 */
export function rankData(values: Float64Array): {
  ranks: Float64Array;
  tieSum: number;
} {
  const n = values.length;
  const ranks = new Float64Array(n);
  if (n === 0) return { ranks, tieSum: 0 };

  // Sort an index array (Int32) rather than an array of {v, i} objects: this
  // avoids n heap allocations and keeps the comparator reading a monomorphic
  // typed array. Ordering is identical to a stable value sort. (A radix argsort
  // was tried here but its per-call buffer allocation makes it slower than the
  // comparator until n is far larger than typical ranking inputs.)
  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => (values[a] as number) - (values[b] as number));
  let tieSum = 0;

  for (let i = 0; i < n; ) {
    const vi = values[order[i] as number] as number;
    let j = i + 1;
    while (j < n && (values[order[j] as number] as number) === vi) {
      j++;
    }
    const t = j - i;
    const avgRank = (i + 1 + j) / 2;
    for (let k = i; k < j; k++) {
      ranks[order[k] as number] = avgRank;
    }
    if (t > 1) tieSum += t * t * t - t;
    i = j;
  }

  return { ranks, tieSum };
}

/**
 * Iterates over all elements in a tensor, calling a function with offset and index.
 *
 * Handles arbitrary tensor shapes and strides, iterating in row-major order.
 * This is the core iteration primitive used by all reduction operations.
 *
 * @param t - Tensor to iterate over
 * @param fn - Callback function receiving (offset, index) for each element
 *
 * @example
 * ```ts
 * const t = tensor([[1, 2], [3, 4]]);
 * forEachIndexOffset(t, (offset, idx) => {
 *   console.log(`idx=${idx}, value=${t.data[offset]}`);
 * });
 * // Outputs: idx=[0,0], value=1
 * //          idx=[0,1], value=2
 * //          idx=[1,0], value=3
 * //          idx=[1,1], value=4
 * ```
 */
export function forEachIndexOffset(
  t: Tensor,
  fn: (offset: number, idx: readonly number[]) => void
): void {
  if (t.size === 0) return;

  if (t.ndim === 0) {
    fn(t.offset, []);
    return;
  }

  const shape = t.shape;
  const strides = t.strides;
  const idx = new Array<number>(t.ndim).fill(0);
  let offset = t.offset;

  while (true) {
    fn(offset, idx);

    // Odometer increment from last axis.
    let axis = t.ndim - 1;
    for (;;) {
      idx[axis] = (idx[axis] ?? 0) + 1;
      offset += strides[axis] ?? 0;

      const dim = shape[axis] ?? 0;
      if ((idx[axis] ?? 0) < dim) break;

      // carry
      offset -= (idx[axis] ?? 0) * (strides[axis] ?? 0);
      idx[axis] = 0;
      axis--;
      if (axis < 0) return;
    }
  }
}

/**
 * Computes the arithmetic mean along specified axes.
 *
 * Uses a simple sum-based approach for numerical stability.
 * This is an internal function used by the public mean() API.
 *
 * @param t - Input tensor
 * @param axis - Axis or axes to reduce over (undefined means all)
 * @param keepdims - Whether to keep reduced dimensions as size 1
 * @returns Tensor containing mean values
 * @throws {InvalidParameterError} If tensor is empty or reduction over empty axis
 *
 * @example
 * ```ts
 * const t = tensor([[1, 2], [3, 4]]);
 * reduceMean(t, undefined, false); // Returns tensor([2.5])
 * reduceMean(t, 0, false);         // Returns tensor([2, 3])
 * ```
 */
export function reduceMean(t: Tensor, axis: AxisLike | undefined, keepdims: boolean): Tensor {
  const axes = normalizeAxes(axis, t.ndim);
  if (axes.length === 0) {
    // Full reduction.
    if (t.size === 0) {
      throw new InvalidParameterError("mean() requires at least one element", "size", t.size);
    }

    let sum = 0;
    const raw = t.data;
    if (
      !Array.isArray(raw) &&
      !(raw instanceof BigInt64Array) &&
      isContiguous(t.shape, t.strides)
    ) {
      // Contiguous numeric fast path: a monomorphic typed-array loop instead of
      // a per-element closure + dispatched accessor (the generic path is ~20x
      // slower on flat arrays, e.g. mean over 10K elements).
      sum = contiguousSum(raw, t.offset, t.size);
    } else {
      forEachIndexOffset(t, (off) => {
        sum += getNumberAt(t, off);
      });
    }

    const out = new Float64Array(1);
    out[0] = sum / t.size;

    const outShape = keepdims ? new Array<number>(t.ndim).fill(1) : [];
    return Tensor.fromTypedArray({
      data: out,
      shape: outShape,
      dtype: "float64",
      device: t.device,
    });
  }

  const outShape = reducedShape(t.shape, axes, keepdims);
  const outStrides = computeStrides(outShape);
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const sums = new Float64Array(outSize);

  const reduce = new Set<number>(axes);
  const reduceCount = axes.reduce((acc, ax) => acc * (t.shape[ax] ?? 0), 1);
  if (reduceCount === 0) {
    throw new InvalidParameterError(
      "mean() reduction over empty axis is undefined",
      "reduceCount",
      reduceCount
    );
  }

  forEachIndexOffset(t, (off, idx) => {
    let outFlat = 0;
    if (keepdims) {
      // outShape has same rank as input.
      for (let i = 0; i < t.ndim; i++) {
        const s = outStrides[i] ?? 0;
        const v = reduce.has(i) ? 0 : (idx[i] ?? 0);
        outFlat += v * s;
      }
    } else {
      let oi = 0;
      for (let i = 0; i < t.ndim; i++) {
        if (reduce.has(i)) continue;
        outFlat += (idx[i] ?? 0) * (outStrides[oi] ?? 0);
        oi++;
      }
    }

    sums[outFlat] = (sums[outFlat] ?? 0) + getNumberAt(t, off);
  });

  for (let i = 0; i < sums.length; i++) {
    sums[i] = (sums[i] ?? 0) / reduceCount;
  }

  return Tensor.fromTypedArray({
    data: sums,
    shape: outShape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Computes variance along specified axes using Welford's online algorithm.
 *
 * Welford's algorithm provides numerical stability for variance computation
 * by avoiding catastrophic cancellation that occurs with the naive two-pass method.
 *
 * @param t - Input tensor
 * @param axis - Axis or axes to reduce over (undefined means all)
 * @param keepdims - Whether to keep reduced dimensions as size 1
 * @param ddof - Delta degrees of freedom (0 for population, 1 for sample variance)
 * @returns Tensor containing variance values
 * @throws {InvalidParameterError} If tensor is empty, ddof >= sample size, or reduction over empty axis
 * @throws {DTypeError} If tensor has string dtype
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5]);
 * reduceVariance(t, undefined, false, 1); // Sample variance
 * reduceVariance(t, undefined, false, 0); // Population variance
 * ```
 */
export function reduceVariance(
  t: Tensor,
  axis: AxisLike | undefined,
  keepdims: boolean,
  ddof: number
): Tensor {
  const axes = normalizeAxes(axis, t.ndim);

  if (t.dtype === "string") {
    throw new DTypeError("variance() not supported for string dtype");
  }

  if (axes.length === 0) {
    if (t.size === 0) {
      throw new InvalidParameterError("variance() requires at least one element", "size", t.size);
    }
    if (ddof < 0) {
      throw new InvalidParameterError("ddof must be non-negative", "ddof", ddof);
    }
    if (t.size <= ddof) {
      throw new InvalidParameterError(
        `ddof=${ddof} >= size=${t.size}, variance undefined`,
        "ddof",
        ddof
      );
    }

    // Welford's online algorithm for numerically stable variance computation
    // Maintains running mean and sum of squared deviations (M2)
    let mean = 0; // Running mean
    let m2 = 0; // Sum of squared deviations from mean
    let n = t.size; // Count of elements processed
    const raw = t.data;
    if (
      !Array.isArray(raw) &&
      !(raw instanceof BigInt64Array) &&
      isContiguous(t.shape, t.strides)
    ) {
      // Contiguous numeric fast path: stable two-pass M2 in monomorphic
      // typed-array loops (no per-element closure or dispatched accessor, and
      // not megamorphic across dtypes).
      m2 = contiguousM2(raw, t.offset, n);
    } else {
      n = 0;
      forEachIndexOffset(t, (off) => {
        const x = getNumberAt(t, off);
        n++;
        const delta = x - mean; // Deviation from old mean
        mean += delta / n; // Update mean incrementally
        const delta2 = x - mean; // Deviation from new mean
        m2 += delta * delta2; // Update M2 (numerically stable)
      });
    }

    const out = new Float64Array(1);
    out[0] = m2 / (n - ddof);

    const outShape = keepdims ? new Array<number>(t.ndim).fill(1) : [];
    return Tensor.fromTypedArray({
      data: out,
      shape: outShape,
      dtype: "float64",
      device: t.device,
    });
  }

  const outShape = reducedShape(t.shape, axes, keepdims);
  const outStrides = computeStrides(outShape);
  const outSize = outShape.reduce((a, b) => a * b, 1);

  const reduce = new Set<number>(axes);
  const reduceCount = axes.reduce((acc, ax) => acc * (t.shape[ax] ?? 0), 1);
  if (reduceCount === 0) {
    throw new InvalidParameterError(
      "variance() reduction over empty axis is undefined",
      "reduceCount",
      reduceCount
    );
  }
  if (ddof < 0) {
    throw new InvalidParameterError("ddof must be non-negative", "ddof", ddof);
  }
  if (reduceCount <= ddof) {
    throw new InvalidParameterError(
      `ddof=${ddof} >= reduced size=${reduceCount}, variance undefined`,
      "ddof",
      ddof
    );
  }

  const means = new Float64Array(outSize);
  const m2s = new Float64Array(outSize);
  const counts = new Int32Array(outSize);

  forEachIndexOffset(t, (off, idx) => {
    let outFlat = 0;
    if (keepdims) {
      for (let i = 0; i < t.ndim; i++) {
        const s = outStrides[i] ?? 0;
        const v = reduce.has(i) ? 0 : (idx[i] ?? 0);
        outFlat += v * s;
      }
    } else {
      let oi = 0;
      for (let i = 0; i < t.ndim; i++) {
        if (reduce.has(i)) continue;
        outFlat += (idx[i] ?? 0) * (outStrides[oi] ?? 0);
        oi++;
      }
    }

    const x = getNumberAt(t, off);
    const n = (counts[outFlat] ?? 0) + 1;
    counts[outFlat] = n;

    const mean = means[outFlat] ?? 0;
    const delta = x - mean;
    const nextMean = mean + delta / n;
    means[outFlat] = nextMean;
    const delta2 = x - nextMean;
    m2s[outFlat] = (m2s[outFlat] ?? 0) + delta * delta2;
  });

  const out = new Float64Array(outSize);
  for (let i = 0; i < outSize; i++) {
    const n = counts[i] ?? 0;
    out[i] = (m2s[i] ?? 0) / (n - ddof);
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "float64",
    device: t.device,
  });
}

// ---- Special functions for distribution CDFs ----

/**
 * Lanczos approximation coefficients for gamma function (g=7, n=9).
 * These constants provide high-precision approximation of the gamma function.
 */
const LANCZOS_COEFFS: readonly number[] = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.3234287776531,
  -176.6150291621406, 12.507343278686905, -0.13857109526572012, 0.000009984369578019572,
  1.5056327351493116e-7,
];

/**
 * Computes the natural logarithm of the gamma function: ln(Γ(z)).
 *
 * Uses Lanczos approximation for z >= 0.5 and reflection formula for z < 0.5.
 * The gamma function extends the factorial to real and complex numbers:
 * Γ(n) = (n-1)! for positive integers n.
 *
 * @param z - Input value (real number)
 * @returns Natural logarithm of gamma function at z
 *
 * @example
 * ```ts
 * logGamma(5);    // Returns ln(4!) = ln(24) ≈ 3.178
 * logGamma(0.5);  // Returns ln(√π) ≈ 0.572
 * ```
 */
export function logGamma(z: number): number {
  if (z < 0.5) {
    // Use reflection formula: Γ(z)Γ(1-z) = π/sin(πz)
    // Therefore: ln(Γ(z)) = ln(π) - ln(sin(πz)) - ln(Γ(1-z))
    return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
  }

  // Lanczos approximation for z >= 0.5
  z -= 1; // Shift z for Lanczos formula
  let x = LANCZOS_COEFFS[0] ?? 0; // Start with first coefficient
  // Sum the series: x = c0 + c1/(z+1) + c2/(z+2) + ... + c8/(z+8)
  for (let i = 1; i < LANCZOS_COEFFS.length; i++) {
    x += (LANCZOS_COEFFS[i] ?? 0) / (z + i);
  }

  const t = z + 7.5; // g + 0.5 where g=7
  // Final Lanczos formula: ln(Γ(z+1)) = 0.5*ln(2π) + (z+0.5)*ln(t) - t + ln(x)
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

/**
 * Computes the digamma function ψ(x) = d/dx ln(Γ(x)) for x > 0.
 *
 * Uses the recurrence ψ(x) = ψ(x+1) − 1/x to shift the argument above 6,
 * then an asymptotic series. Accurate to ~1e-12 for x > 0.
 *
 * @param x - Input value (must be > 0)
 * @returns ψ(x)
 *
 * @internal
 */
export function digamma(x: number): number {
  let result = 0;
  let v = x;
  // Shift argument up using ψ(v) = ψ(v+1) - 1/v until v >= 6.
  while (v < 6) {
    result -= 1 / v;
    v += 1;
  }
  // Asymptotic expansion for large v.
  const inv = 1 / v;
  const inv2 = inv * inv;
  result +=
    Math.log(v) - 0.5 * inv - inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 * (1 / 252 - inv2 / 240)));
  return result;
}

/**
 * Evaluates continued fraction for incomplete beta function.
 *
 * Uses Lentz's algorithm for evaluating continued fractions.
 * This is a helper function for regularizedIncompleteBeta.
 *
 * @param a - First shape parameter
 * @param b - Second shape parameter
 * @param x - Evaluation point in [0, 1]
 * @returns Continued fraction value
 */
function betacf(a: number, b: number, x: number): number {
  const MAX_ITER = 200; // Maximum iterations for convergence
  const EPS = 3e-14; // Convergence threshold
  const FPMIN = 1e-300; // Minimum floating point value to prevent division by zero

  // Precompute common terms
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;

  // Initialize Lentz's algorithm
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN; // Prevent division by zero
  d = 1 / d;
  let h = d; // Accumulated result

  // Iterate using modified Lentz's method
  for (let m = 1; m <= MAX_ITER; m++) {
    const m2 = 2 * m;

    // Even step of continued fraction
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    h *= d * c;

    // Odd step of continued fraction
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;

    // Check for convergence
    if (Math.abs(del - 1.0) < EPS) break;
  }

  return h;
}

/**
 * Computes the regularized incomplete beta function I_x(a, b).
 *
 * The regularized incomplete beta function is defined as:
 * I_x(a, b) = B(x; a, b) / B(a, b)
 * where B(x; a, b) is the incomplete beta function and B(a, b) is the beta function.
 *
 * Used in computing CDFs for beta, F, and t distributions.
 *
 * @param a - First shape parameter (must be > 0)
 * @param b - Second shape parameter (must be > 0)
 * @param x - Evaluation point in [0, 1]
 * @returns Value of I_x(a, b) in [0, 1]
 * @throws {InvalidParameterError} If parameters are outside their valid ranges
 *
 * @example
 * ```ts
 * regularizedIncompleteBeta(2, 3, 0.5); // Returns ~0.6875
 * ```
 */
export function regularizedIncompleteBeta(a: number, b: number, x: number): number {
  if (!Number.isFinite(a) || a <= 0) {
    throw new InvalidParameterError("a must be > 0", "a", a);
  }
  if (!Number.isFinite(b) || b <= 0) {
    throw new InvalidParameterError("b must be > 0", "b", b);
  }
  // Validate input range
  if (x < 0 || x > 1) {
    throw new InvalidParameterError("x must be in [0,1]", "x", x);
  }
  // Handle boundary cases
  if (x === 0) return 0;
  if (x === 1) return 1;

  // Compute ln(B(x; a, b)) = ln(Γ(a+b)) - ln(Γ(a)) - ln(Γ(b)) + a*ln(x) + b*ln(1-x)
  const lnBt = logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x);
  const bt = Math.exp(lnBt);

  // Use symmetry relation to ensure x is in the more stable region
  if (x < (a + 1) / (a + b + 2)) {
    // Direct evaluation
    return (bt * betacf(a, b, x)) / a;
  }
  // Use symmetry: I_x(a,b) = 1 - I_(1-x)(b,a)
  return 1 - (bt * betacf(b, a, 1 - x)) / b;
}

/**
 * Computes the regularized lower incomplete gamma function P(s, x).
 *
 * P(s, x) = γ(s, x) / Γ(s) where γ(s, x) is the lower incomplete gamma function.
 * This represents the CDF of the gamma distribution.
 *
 * @param s - Shape parameter (must be > 0)
 * @param x - Evaluation point (must be >= 0)
 * @returns Value of P(s, x) in [0, 1]
 * @throws {InvalidParameterError} If parameters are outside their valid ranges
 */
function regularizedLowerIncompleteGamma(s: number, x: number): number {
  // Validate input
  if (!Number.isFinite(s) || s <= 0) {
    throw new InvalidParameterError("s must be > 0", "s", s);
  }
  if (x < 0) throw new InvalidParameterError("x must be >= 0", "x", x);
  if (x === 0) return 0;

  const ITMAX = 200; // Maximum iterations
  const EPS = 3e-14; // Convergence threshold
  const FPMIN = 1e-300; // Minimum floating point value

  if (x < s + 1) {
    // Use series representation for x < s + 1 (more stable)
    // P(s,x) = e^(-x) * x^s / Γ(s) * Σ(x^n / Γ(s+n+1))
    let sum = 1 / s;
    let del = sum;
    let ap = s;

    for (let n = 1; n <= ITMAX; n++) {
      ap += 1;
      del *= x / ap;
      sum += del;
      if (Math.abs(del) < Math.abs(sum) * EPS) break; // Converged
    }

    // Multiply by normalization factor
    return sum * Math.exp(-x + s * Math.log(x) - logGamma(s));
  }

  // Use continued fraction for Q(s, x) = 1 - P(s, x) when x >= s + 1
  // This is more numerically stable in this region
  let b = x + 1 - s;
  let c = 1 / FPMIN;
  let d = 1 / b;
  let h = d;

  // Evaluate continued fraction using Lentz's algorithm
  for (let i = 1; i <= ITMAX; i++) {
    const an = -i * (i - s);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    c = b + an / c;
    if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break; // Converged
  }

  // Compute Q(s, x) and return P(s, x) = 1 - Q(s, x)
  const q = Math.exp(-x + s * Math.log(x) - logGamma(s)) * h;
  return 1 - q;
}

/** Chebyshev coefficients for the erfc fit (Numerical Recipes, erfcc). */
const ERFC_COEFFS: readonly number[] = [
  -1.26551223, 1.00002368, 0.37409196, 0.09678418, -0.18628806, 0.27886807, -1.13520398, 1.48851587,
  -0.82215223, 0.17087277,
];

/**
 * Computes the complementary error function erfc(x) = 1 - erf(x).
 *
 * Uses the rational Chebyshev approximation of `t·exp(z²)·erfc(z)` from
 * Numerical Recipes (Press et al.), with fractional error below ~1.2e-7
 * across the full real line — far more accurate than single-term
 * approximations and adequate for p-value computation.
 *
 * @param x - Input value
 * @returns erfc(x) in [0, 2]
 *
 * @internal
 */
export function erfc(x: number): number {
  if (x === 0) return 1;
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  // Horner evaluation of the polynomial in t.
  let poly = ERFC_COEFFS[ERFC_COEFFS.length - 1] ?? 0;
  for (let i = ERFC_COEFFS.length - 2; i >= 0; i--) {
    poly = (ERFC_COEFFS[i] ?? 0) + t * poly;
  }
  const tau = t * Math.exp(-z * z + poly);
  return x >= 0 ? tau : 2 - tau;
}

/**
 * Computes the error function erf(x).
 *
 * @param x - Input value
 * @returns erf(x) in [-1, 1]
 *
 * @internal
 */
export function erf(x: number): number {
  return 1 - erfc(x);
}

/**
 * Computes the cumulative distribution function (CDF) of the standard normal distribution.
 *
 * Uses the relation Φ(x) = ½·erfc(−x/√2) with a high-accuracy erfc
 * (Cody/Numerical-Recipes rational Chebyshev fit), giving relative error
 * below ~1e-7 over the full real line — far tighter than single-term
 * approximations and adequate for p-value computation.
 *
 * @param x - Input value
 * @returns Probability P(X <= x) where X ~ N(0, 1)
 *
 * @example
 * ```ts
 * normalCdf(0);     // Returns 0.5
 * normalCdf(1.96);  // Returns ~0.975 (95th percentile)
 * normalCdf(-1.96); // Returns ~0.025 (5th percentile)
 * ```
 */
export function normalCdf(x: number): number {
  // Φ(x) = 0.5 * erfc(-x/√2)
  return 0.5 * erfc(-x / Math.SQRT2);
}

/**
 * Computes the inverse (quantile) of the standard normal CDF, Φ⁻¹(p).
 *
 * Uses Peter Acklam's rational approximation followed by one Halley
 * refinement step against the high-accuracy {@link normalCdf}, achieving
 * full double precision (relative error ~1e-15) over (0, 1).
 *
 * @param p - Probability in [0, 1]
 * @returns z such that Φ(z) = p
 *
 * @example
 * ```ts
 * normalPpf(0.5);   // Returns 0
 * normalPpf(0.975); // Returns ~1.95996
 * ```
 *
 * @internal
 */
export function normalPpf(p: number): number {
  if (Number.isNaN(p) || p < 0 || p > 1) return NaN;
  if (p === 0) return -Infinity;
  if (p === 1) return Infinity;

  // Acklam's rational approximation coefficients.
  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2,
    -3.066479806614716e1, 2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1,
    -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734,
    4.374664141464968, 2.938163982698783,
  ];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];

  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  let z: number;

  if (p < pLow) {
    const q = Math.sqrt(-2 * Math.log(p));
    z =
      (((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) /
      ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1);
  } else if (p <= pHigh) {
    const q = p - 0.5;
    const r = q * q;
    z =
      ((((((a[0]! * r + a[1]!) * r + a[2]!) * r + a[3]!) * r + a[4]!) * r + a[5]!) * q) /
      (((((b[0]! * r + b[1]!) * r + b[2]!) * r + b[3]!) * r + b[4]!) * r + 1);
  } else {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    z =
      -(((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) /
      ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1);
  }

  // One Halley refinement step for full double precision.
  const e = normalCdf(z) - p;
  const u = e * Math.sqrt(2 * Math.PI) * Math.exp((z * z) / 2);
  z = z - u / (1 + (z * u) / 2);
  return z;
}

/**
 * Computes the cumulative distribution function (CDF) of Student's t-distribution.
 *
 * Uses the relationship between t-distribution and incomplete beta function:
 * F_t(t; ν) = 0.5 * I_x(ν/2, 1/2) where x = ν/(ν + t²)
 *
 * @param t - t-statistic value
 * @param df - Degrees of freedom (must be > 0)
 * @returns Probability P(T <= t) where T ~ t(df)
 * @throws {InvalidParameterError} If df <= 0
 *
 * @example
 * ```ts
 * studentTCdf(0, 10);      // Returns 0.5 (symmetric at 0)
 * studentTCdf(2.228, 10);  // Returns ~0.975 (95th percentile for df=10)
 * ```
 */
export function studentTCdf(t: number, df: number): number {
  // Validate degrees of freedom
  if (!Number.isFinite(df) || df <= 0) {
    throw new InvalidParameterError("df must be > 0", "df", df);
  }
  // Handle infinite t-values
  if (Number.isNaN(t)) return NaN;
  if (!Number.isFinite(t)) return t < 0 ? 0 : 1;

  // Transform to incomplete beta function parameter
  const x = df / (df + t * t);
  const a = df / 2;
  const b = 0.5;
  const ib = regularizedIncompleteBeta(a, b, x);
  const p = 0.5 * ib;
  // Use symmetry of t-distribution around 0
  return t >= 0 ? 1 - p : p;
}

/**
 * Computes the cumulative distribution function (CDF) of the chi-square distribution.
 *
 * Uses the relationship: χ²(x; k) = P(k/2, x/2)
 * where P is the regularized lower incomplete gamma function.
 *
 * @param x - Chi-square statistic (must be >= 0)
 * @param k - Degrees of freedom (must be > 0)
 * @returns Probability P(X <= x) where X ~ χ²(k)
 * @throws {InvalidParameterError} If k <= 0
 *
 * @example
 * ```ts
 * chiSquareCdf(3.841, 1);  // Returns ~0.95 (95th percentile for df=1)
 * chiSquareCdf(0, 5);      // Returns 0
 * ```
 */
export function chiSquareCdf(x: number, k: number): number {
  // Validate degrees of freedom
  if (!Number.isFinite(k) || k <= 0) {
    throw new InvalidParameterError("degrees of freedom must be > 0", "k", k);
  }
  if (Number.isNaN(x)) return NaN;
  if (x === Infinity) return 1;
  // Chi-square is non-negative
  if (x <= 0) return 0;
  // Use gamma CDF relationship: χ²(k) is Gamma(k/2, 2)
  return regularizedLowerIncompleteGamma(k / 2, x / 2);
}

/**
 * Computes the cumulative distribution function (CDF) of the F-distribution.
 *
 * Uses the relationship between F-distribution and incomplete beta function:
 * F(x; d₁, d₂) = I_y(d₁/2, d₂/2) where y = (d₁*x)/(d₁*x + d₂)
 *
 * @param x - F-statistic value (must be >= 0)
 * @param dfn - Numerator degrees of freedom (must be > 0)
 * @param dfd - Denominator degrees of freedom (must be > 0)
 * @returns Probability P(F <= x) where F ~ F(dfn, dfd)
 * @throws {InvalidParameterError} If dfn <= 0 or dfd <= 0
 *
 * @example
 * ```ts
 * fCdf(4.0, 5, 10);  // Returns F-distribution CDF at x=4
 * fCdf(0, 5, 10);    // Returns 0
 * ```
 */
export function fCdf(x: number, dfn: number, dfd: number): number {
  // Validate degrees of freedom
  if (!Number.isFinite(dfn) || dfn <= 0) {
    throw new InvalidParameterError("degrees of freedom (dfn) must be > 0", "dfn", dfn);
  }
  if (!Number.isFinite(dfd) || dfd <= 0) {
    throw new InvalidParameterError("degrees of freedom (dfd) must be > 0", "dfd", dfd);
  }
  if (Number.isNaN(x)) return NaN;
  if (x === Infinity) return 1;
  // F-statistic is non-negative
  if (x <= 0) return 0;

  // Transform to incomplete beta parameter
  const xx = (dfn * x) / (dfn * x + dfd);
  return regularizedIncompleteBeta(dfn / 2, dfd / 2, xx);
}
