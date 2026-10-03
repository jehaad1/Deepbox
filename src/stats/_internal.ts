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

/** Block length for the two-level summation used by {@link contiguousSum}. */
const SUM_BLOCK = 256;

/**
 * Sum `raw[off .. off+n)`. The `instanceof` split gives V8 a concrete element
 * type at each load site: `Tensor.data` is a wide typed-array union, so a
 * single shared loop over it stays megamorphic and ~4-5x slower.
 *
 * The sum is accumulated in blocks of {@link SUM_BLOCK} elements whose partial
 * sums are then added together (a two-level pairwise scheme, like NumPy's
 * pairwise summation). Rounding error grows with `n / SUM_BLOCK` rather than
 * with `n`, and inputs shorter than one block are summed exactly as a plain
 * left-to-right loop.
 */
function contiguousSum(raw: NumericTypedArray, off: number, n: number): number {
  let total = 0;
  if (raw instanceof Float64Array) {
    for (let start = 0; start < n; start += SUM_BLOCK) {
      const end = Math.min(n, start + SUM_BLOCK);
      let s = 0;
      for (let i = off + start; i < off + end; i++) s += raw[i] as number;
      total += s;
    }
  } else if (raw instanceof Float32Array) {
    for (let start = 0; start < n; start += SUM_BLOCK) {
      const end = Math.min(n, start + SUM_BLOCK);
      let s = 0;
      for (let i = off + start; i < off + end; i++) s += raw[i] as number;
      total += s;
    }
  } else {
    for (let start = 0; start < n; start += SUM_BLOCK) {
      const end = Math.min(n, start + SUM_BLOCK);
      let s = 0;
      for (let i = off + start; i < off + end; i++) s += raw[i] as number;
      total += s;
    }
  }
  return total;
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
 * value and returns it. O(n) average (median-of-three pivot, three-way
 * partition so runs of equal values, e.g. a constant or binary column, do not
 * degrade to O(n^2)). Mutates `arr`. After the call every element left of `k`
 * is `<= arr[k]` and every element right of `k` is `>= arr[k]`.
 *
 * `k` must be an integer in `[0, arr.length)`. NaN values are not ordered;
 * callers that can see NaN must check for it first (see {@link quickMedianF64}).
 */
export function quickSelectF64(arr: Float64Array, k: number): number {
  let lo = 0;
  let hi = arr.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    const a = arr[lo] as number;
    const b = arr[mid] as number;
    const c = arr[hi] as number;
    let pivotVal: number;
    if ((a <= b && b <= c) || (c <= b && b <= a)) pivotVal = b;
    else if ((b <= a && a <= c) || (c <= a && a <= b)) pivotVal = a;
    else pivotVal = c;

    // Three-way (Dutch national flag) partition around pivotVal:
    // [lo, lt) < pivot, [lt, gt] == pivot, (gt, hi] > pivot.
    let lt = lo;
    let gt = hi;
    let i = lo;
    while (i <= gt) {
      const v = arr[i] as number;
      if (v < pivotVal) {
        arr[i] = arr[lt] as number;
        arr[lt] = v;
        lt++;
        i++;
      } else if (v > pivotVal) {
        arr[i] = arr[gt] as number;
        arr[gt] = v;
        gt--;
      } else {
        i++;
      }
    }
    if (k < lt) hi = lt - 1;
    else if (k > gt) lo = gt + 1;
    else return pivotVal;
  }
  return arr[lo] as number;
}

/**
 * Median of a mutable `Float64Array` via O(n) quickselect (NaN propagates, per
 * NumPy semantics). Mutates `arr`.
 */
export function quickMedianF64(arr: Float64Array): number {
  const n = arr.length;
  if (n === 0) return Number.NaN;
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
 * two-pass method: compute the mean, then sum `(x - mean)^2`. This is the
 * numerically stable form NumPy/SciPy use for a full array; it avoids Welford's
 * per-element division (~3x faster) while agreeing to ~1e-14 relative. Both
 * passes are monomorphic per typed-array kind and use the blocked summation of
 * {@link contiguousSum}.
 */
function contiguousM2(raw: NumericTypedArray, off: number, n: number): number {
  const mean = contiguousSum(raw, off, n) / n;
  let m2 = 0;
  if (raw instanceof Float64Array) {
    for (let start = 0; start < n; start += SUM_BLOCK) {
      const end = Math.min(n, start + SUM_BLOCK);
      let s = 0;
      for (let i = off + start; i < off + end; i++) {
        const d = (raw[i] as number) - mean;
        s += d * d;
      }
      m2 += s;
    }
  } else if (raw instanceof Float32Array) {
    for (let start = 0; start < n; start += SUM_BLOCK) {
      const end = Math.min(n, start + SUM_BLOCK);
      let s = 0;
      for (let i = off + start; i < off + end; i++) {
        const d = (raw[i] as number) - mean;
        s += d * d;
      }
      m2 += s;
    }
  } else {
    for (let start = 0; start < n; start += SUM_BLOCK) {
      const end = Math.min(n, start + SUM_BLOCK);
      let s = 0;
      for (let i = off + start; i < off + end; i++) {
        const d = (raw[i] as number) - mean;
        s += d * d;
      }
      m2 += s;
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
 * @throws {InvalidParameterError} If any axis is not an integer, is an unknown alias, or is out
 *   of bounds for the given ndim
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
 * NaN values sort after every other value (including +Infinity), in input
 * order, and each NaN gets its own rank; NaN is never counted as a tie.
 * Callers that need NaN to propagate should check for it before ranking.
 *
 * @param values - Input values to rank
 * @returns Object containing ranks and tie sum
 *
 * @example
 * ```ts
 * rankData(new Float64Array([10, 20, 10, 30]));
 * // { ranks: Float64Array [1.5, 3, 1.5, 4], tieSum: 6 }
 * ```
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
  order.sort((a, b) => {
    const va = values[a] as number;
    const vb = values[b] as number;
    if (va < vb) return -1;
    if (va > vb) return 1;
    if (va === vb) return 0;
    // At least one NaN: NaN goes last, and two NaNs keep their input order.
    const aNaN = Number.isNaN(va);
    const bNaN = Number.isNaN(vb);
    if (aNaN === bNaN) return a - b;
    return aNaN ? 1 : -1;
  });
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
 * For each input axis, the stride that axis contributes to a flat index into
 * the (row-major) reduction output; 0 for reduced axes. Lets the reduction
 * loops compute an output slot with one multiply-add per axis and no set
 * lookups.
 */
function outputAxisStrides(
  ndim: number,
  axes: readonly number[],
  outShape: Shape,
  keepdims: boolean
): readonly number[] {
  const outStrides = computeStrides(outShape);
  const reduce = new Set<number>(axes);
  const contrib = new Array<number>(ndim).fill(0);
  let oi = 0;
  for (let i = 0; i < ndim; i++) {
    if (reduce.has(i)) {
      if (keepdims) oi++;
      continue;
    }
    contrib[i] = outStrides[oi] ?? 0;
    oi++;
  }
  return contrib;
}

/**
 * Computes the arithmetic mean along specified axes.
 *
 * Contiguous numeric tensors reduced over all axes are summed with blocked
 * (pairwise-style) accumulation; other layouts use a plain running sum in
 * double precision. Always returns a `float64` tensor.
 * This is an internal function used by the public mean() API.
 *
 * @param t - Input tensor
 * @param axis - Axis or axes to reduce over (undefined means all)
 * @param keepdims - Whether to keep reduced dimensions as size 1
 * @returns Tensor containing mean values
 * @throws {InvalidParameterError} If tensor is empty or reduction over empty axis
 * @throws {DTypeError} If tensor has string dtype
 *
 * @example
 * ```ts
 * const t = tensor([[1, 2], [3, 4]]);
 * reduceMean(t, undefined, false); // Returns scalar tensor(2.5)
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
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const sums = new Float64Array(outSize);

  const reduceCount = axes.reduce((acc, ax) => acc * (t.shape[ax] ?? 0), 1);
  if (reduceCount === 0) {
    throw new InvalidParameterError(
      "mean() reduction over empty axis is undefined",
      "reduceCount",
      reduceCount
    );
  }

  const contrib = outputAxisStrides(t.ndim, axes, outShape, keepdims);
  const ndim = t.ndim;
  forEachIndexOffset(t, (off, idx) => {
    let outFlat = 0;
    for (let i = 0; i < ndim; i++) outFlat += (idx[i] as number) * (contrib[i] as number);
    sums[outFlat] = (sums[outFlat] as number) + getNumberAt(t, off);
  });

  for (let i = 0; i < sums.length; i++) {
    sums[i] = (sums[i] as number) / reduceCount;
  }

  return Tensor.fromTypedArray({
    data: sums,
    shape: outShape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Computes variance along specified axes.
 *
 * Contiguous numeric tensors reduced over all axes use a two-pass
 * (mean, then squared deviations) algorithm; every other case uses Welford's
 * online update. Both avoid the catastrophic cancellation of the naive
 * `E[x^2] - E[x]^2` formula. Always returns a `float64` tensor.
 *
 * @param t - Input tensor
 * @param axis - Axis or axes to reduce over (undefined means all)
 * @param keepdims - Whether to keep reduced dimensions as size 1
 * @param ddof - Delta degrees of freedom (0 for population, 1 for sample variance)
 * @returns Tensor containing variance values
 * @throws {InvalidParameterError} If tensor is empty, ddof is negative or not finite,
 *   ddof >= sample size, or the reduction is over an empty axis
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
    if (!Number.isFinite(ddof) || ddof < 0) {
      throw new InvalidParameterError("ddof must be a non-negative finite number", "ddof", ddof);
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
  const outSize = outShape.reduce((a, b) => a * b, 1);

  const reduceCount = axes.reduce((acc, ax) => acc * (t.shape[ax] ?? 0), 1);
  if (reduceCount === 0) {
    throw new InvalidParameterError(
      "variance() reduction over empty axis is undefined",
      "reduceCount",
      reduceCount
    );
  }
  if (!Number.isFinite(ddof) || ddof < 0) {
    throw new InvalidParameterError("ddof must be a non-negative finite number", "ddof", ddof);
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
  const counts = new Float64Array(outSize);

  const contrib = outputAxisStrides(t.ndim, axes, outShape, keepdims);
  const ndim = t.ndim;
  forEachIndexOffset(t, (off, idx) => {
    let outFlat = 0;
    for (let i = 0; i < ndim; i++) outFlat += (idx[i] as number) * (contrib[i] as number);

    const x = getNumberAt(t, off);
    const n = (counts[outFlat] as number) + 1;
    counts[outFlat] = n;

    const mean = means[outFlat] as number;
    const delta = x - mean;
    const nextMean = mean + delta / n;
    means[outFlat] = nextMean;
    const delta2 = x - nextMean;
    m2s[outFlat] = (m2s[outFlat] as number) + delta * delta2;
  });

  const out = new Float64Array(outSize);
  for (let i = 0; i < outSize; i++) {
    const n = counts[i] as number;
    out[i] = (m2s[i] as number) / (n - ddof);
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "float64",
    device: t.device,
  });
}

// ---- Special functions for distribution CDFs ----

const LN_SQRT_2PI = 0.9189385332046727; // 0.5 * ln(2π)

/**
 * Lanczos approximation coefficients for gamma function (g=7, n=9).
 * These constants provide high-precision approximation of the gamma function.
 */
const LANCZOS_COEFFS: readonly number[] = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.3234287776531,
  -176.6150291621406, 12.507343278686905, -0.13857109526572012, 0.000009984369578019572,
  1.5056327351493116e-7,
];

/** ζ(k) - 1 for k = 2..20, the coefficients of the Taylor series of ln Γ(1 + x). */
const ZETA_MINUS_ONE: readonly number[] = [
  0.6449340668482264, 0.2020569031595943, 0.08232323371113819, 0.03692775514336993,
  0.01734306198444914, 0.008349277381922827, 0.00407735619794434, 0.0020083928260822143,
  0.0009945751278180853, 0.0004941886041194645, 0.0002460865533080483, 0.00012271334757848915,
  6.124813505870483e-5, 3.058823630702049e-5, 1.528225940865187e-5, 7.637197637899763e-6,
  3.81729326499984e-6, 1.908212716553939e-6, 9.539620338727962e-7,
];

/**
 * x (1 - γ) + Σ_{k>=2} (ζ(k) - 1) (-x)^k / k, for |x| <= 1/4 (terms shrink like
 * (|x|/2)^k). By DLMF 5.7.3 this equals ln Γ(2 + x) and ln Γ(1 + x) + ln(1 + x).
 */
function logGammaSeries(x: number): number {
  let sum = 0;
  let pow = x * x; // (-x)^2
  for (let i = 0; i < ZETA_MINUS_ONE.length; i++) {
    sum += ((ZETA_MINUS_ONE[i] as number) * pow) / (i + 2);
    pow *= -x;
  }
  return x * 0.42278433509846713 + sum;
}

/** ln Γ(1 + x) for |x| <= 1/4. */
function logGammaOnePlus(x: number): number {
  return logGammaSeries(x) - Math.log1p(x);
}

/**
 * Computes the natural logarithm of the absolute value of the gamma function,
 * ln|Γ(z)|.
 *
 * Uses the Lanczos approximation for z >= 0.5 and the reflection formula for
 * z < 0.5. The gamma function extends the factorial to real numbers:
 * Γ(n) = (n-1)! for positive integers n. For negative non-integer z the sign of
 * Γ(z) is discarded, matching `scipy.special.gammaln`.
 *
 * Returns `Infinity` at the poles (0, -1, -2, ...) and for `±Infinity`, and `NaN`
 * for `NaN`.
 *
 * @param z - Input value (real number)
 * @returns ln|Γ(z)|
 *
 * @example
 * ```ts
 * logGamma(5);    // Returns ln(4!) = ln(24) ≈ 3.178
 * logGamma(0.5);  // Returns ln(√π) ≈ 0.572
 * logGamma(-0.5); // Returns ln|Γ(-0.5)| ≈ 1.2655
 * ```
 */
export function logGamma(z: number): number {
  if (Number.isNaN(z)) return Number.NaN;
  if (z === Number.POSITIVE_INFINITY || z === Number.NEGATIVE_INFINITY)
    return Number.POSITIVE_INFINITY;
  if (z === 1 || z === 2) return 0;
  if (z <= 0 && Number.isInteger(z)) return Number.POSITIVE_INFINITY;

  // Near the zeros of ln Γ at 1 and 2 the Lanczos sum keeps only an absolute
  // error of ~1e-16, which is a large relative error. A Taylor series is exact
  // to the last digit there.
  if (z > 0.75 && z < 1.25) return logGammaOnePlus(z - 1);
  if (z > 1.75 && z < 2.25) return logGammaSeries(z - 2);

  if (z < 0.5) {
    // Reflection formula: Γ(z)Γ(1-z) = π/sin(πz), so
    // ln|Γ(z)| = ln(π) - ln|sin(πz)| - ln|Γ(1-z)|. Reduce z to the nearest
    // integer first so sin(πz) keeps full precision for large |z|.
    const r = z - Math.round(z);
    return Math.log(Math.PI) - Math.log(Math.abs(Math.sin(Math.PI * r))) - logGamma(1 - z);
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
  return LN_SQRT_2PI + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

/**
 * Remainder of Stirling's series, ln Γ(z) - [(z - 1/2) ln z - z + ln(2π)/2],
 * for z >= 10 (absolute error below 1e-16 there).
 */
function stirlingCorrection(z: number): number {
  const inv = 1 / z;
  const inv2 = inv * inv;
  return (
    inv *
    (1 / 12 -
      inv2 *
        (1 / 360 -
          inv2 * (1 / 1260 - inv2 * (1 / 1680 - inv2 * (1 / 1188 - (inv2 * 691) / 360360)))))
  );
}

/**
 * Natural logarithm of the beta function, ln B(a, b) = lnΓ(a) + lnΓ(b) - lnΓ(a+b),
 * for a, b > 0.
 *
 * When either argument is large, the three `logGamma` terms are huge and nearly
 * cancel, so a plain difference loses roughly `log10(a + b)` digits. In that
 * regime the Stirling series is combined analytically so only small
 * quantities are subtracted.
 *
 * @internal
 */
export function logBeta(a: number, b: number): number {
  const hi = Math.max(a, b);
  const lo = Math.min(a, b);
  if (hi < 10) return logGamma(a) + logGamma(b) - logGamma(a + b);
  const s = a + b;
  if (lo >= 10) {
    return (
      -(a - 0.5) * Math.log1p(b / a) -
      (b - 0.5) * Math.log1p(a / b) -
      0.5 * Math.log(s) +
      LN_SQRT_2PI +
      stirlingCorrection(a) +
      stirlingCorrection(b) -
      stirlingCorrection(s)
    );
  }
  // lo < 10 <= hi: lnΓ(lo) + [lnΓ(hi) - lnΓ(hi + lo)].
  return (
    logGamma(lo) -
    (hi - 0.5) * Math.log1p(lo / hi) -
    lo * Math.log(s) +
    lo +
    stirlingCorrection(hi) -
    stirlingCorrection(s)
  );
}

/**
 * Computes the digamma function ψ(x) = d/dx ln(Γ(x)).
 *
 * Uses the recurrence ψ(x) = ψ(x+1) − 1/x to shift the argument above 10,
 * then an asymptotic series (relative error below ~1e-15). Negative arguments
 * use the reflection formula ψ(x) = ψ(1−x) − π/tan(πx). Returns `NaN` at
 * negative integers and for `-Infinity`, `-Infinity` at `+0` and `Infinity` at `-0`
 * (matching `scipy.special.digamma`).
 *
 * @param x - Input value
 * @returns ψ(x)
 *
 * @internal
 */
export function digamma(x: number): number {
  if (Number.isNaN(x) || x === Number.NEGATIVE_INFINITY) return Number.NaN;
  if (x === Number.POSITIVE_INFINITY) return Number.POSITIVE_INFINITY;
  if (x === 0) return -1 / x;
  if (x < 0) {
    if (Number.isInteger(x)) return Number.NaN;
    const r = x - Math.round(x);
    return digamma(1 - x) - Math.PI / Math.tan(Math.PI * r);
  }
  let result = 0;
  let v = x;
  // Shift argument up using ψ(v) = ψ(v+1) - 1/v until v >= 10.
  while (v < 10) {
    result -= 1 / v;
    v += 1;
  }
  // Asymptotic expansion for large v:
  // ln v - 1/(2v) - Σ B_2k / (2k v^2k).
  const inv = 1 / v;
  const inv2 = inv * inv;
  result +=
    Math.log(v) -
    0.5 * inv -
    inv2 *
      (1 / 12 -
        inv2 *
          (1 / 120 -
            inv2 * (1 / 252 - inv2 * (1 / 240 - inv2 * (1 / 132 - (inv2 * 691) / 32760)))));
  return result;
}

/**
 * Evaluates continued fraction for incomplete beta function.
 *
 * Uses Lentz's algorithm for evaluating continued fractions.
 * This is a helper function for regularizedIncompleteBeta. The iteration
 * count needed grows like `sqrt(max(a, b))`, so the cap scales with it.
 *
 * @param a - First shape parameter
 * @param b - Second shape parameter
 * @param x - Evaluation point in (0, 1)
 * @param y - `1 - x`, supplied separately because `1 - qab * x / qap` below
 *   cancels badly when `x` is close to 1 (it loses about `log10(a)` digits)
 * @returns Continued fraction value
 */
function betacf(a: number, b: number, x: number, y: number): number {
  const MAX_ITER = Math.min(1e6, Math.ceil(300 + 30 * Math.sqrt(Math.max(a, b)))); // Maximum iterations
  const EPS = 1e-15; // Convergence threshold
  const FPMIN = 1e-300; // Minimum floating point value to prevent division by zero

  // Precompute common terms
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;

  // Initialize Lentz's algorithm
  let c = 1;
  // 1 - (a + b) x / (a + 1), written with y = 1 - x when x is near 1.
  let d = x > 0.5 ? (1 - b + qab * y) / qap : 1 - (qab * x) / qap;
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
 * ln[x^a (1-x)^b / B(a, b)], the prefactor of the incomplete beta continued
 * fraction. `y` must equal `1 - x` and is passed separately so callers that
 * know it accurately (e.g. `y = t²/(df+t²)`) do not lose it to rounding.
 *
 * For a, b >= 10 the terms are regrouped around the distribution's centre
 * (`x ≈ a/(a+b)`) so only small logarithms are summed; the direct form would
 * subtract numbers of size `a·ln x` and lose ~`log10(a)` digits.
 */
function betaPowerLog(a: number, b: number, x: number, y: number): number {
  if (a >= 10 && b >= 10) {
    const s = a + b;
    const r1 = (x * s) / a;
    const r2 = (y * s) / b;
    const e = x * s - a; // = -(y*s - b)
    const l1 = Math.abs(r1 - 1) < 0.5 ? Math.log1p(e / a) : Math.log(r1);
    const l2 = Math.abs(r2 - 1) < 0.5 ? Math.log1p(-e / b) : Math.log(r2);
    return (
      a * l1 +
      b * l2 +
      0.5 * (Math.log(a) + Math.log(b) - Math.log(s)) -
      LN_SQRT_2PI -
      stirlingCorrection(a) -
      stirlingCorrection(b) +
      stirlingCorrection(s)
    );
  }
  // ln x and ln y are taken from whichever of x, y is the exactly-known small
  // one (log1p of the other) so a near-1 argument does not lose digits.
  const lnX = x < 0.5 ? Math.log(x) : Math.log1p(-y);
  const lnY = y < 0.5 ? Math.log(y) : Math.log1p(-x);
  return a * lnX + b * lnY - logBeta(a, b);
}

/**
 * I_x(a, b) given both `x` and `y = 1 - x` (a, b already validated, 0 < x < 1).
 * The result is accurate in whichever tail it is small, because the
 * continued fraction is always evaluated on the smaller side.
 */
function incompleteBetaXY(a: number, b: number, x: number, y: number): number {
  if (x <= 0) return 0;
  if (y <= 0) return 1;

  const bt = Math.exp(betaPowerLog(a, b, x, y));

  // Use symmetry relation to ensure x is in the more stable region
  let value: number;
  if (x < (a + 1) / (a + b + 2)) {
    // Direct evaluation
    value = (bt * betacf(a, b, x, y)) / a;
  } else {
    // Use symmetry: I_x(a,b) = 1 - I_(1-x)(b,a)
    value = 1 - (bt * betacf(b, a, y, x)) / b;
  }
  return value < 0 ? 0 : value > 1 ? 1 : value;
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
 * Relative error is about 1e-15 for moderate shape parameters. When one shape
 * parameter is very large (above about 1e4) and `x` is within `1 / a` of 1, the
 * continued fraction loses roughly `log10(a)` digits, so the error grows to about
 * `1e-16 * a` (for example 1e-11 at a = 1e5).
 *
 * @param a - First shape parameter (must be > 0)
 * @param b - Second shape parameter (must be > 0)
 * @param x - Evaluation point in [0, 1]
 * @returns Value of I_x(a, b) in [0, 1] (`NaN` if `x` is `NaN`)
 * @throws {InvalidParameterError} If parameters are outside their valid ranges
 *
 * @example
 * ```ts
 * regularizedIncompleteBeta(2, 3, 0.5); // Returns 0.6875
 * ```
 */
export function regularizedIncompleteBeta(a: number, b: number, x: number): number {
  if (!Number.isFinite(a) || a <= 0) {
    throw new InvalidParameterError("a must be > 0", "a", a);
  }
  if (!Number.isFinite(b) || b <= 0) {
    throw new InvalidParameterError("b must be > 0", "b", b);
  }
  if (Number.isNaN(x)) return Number.NaN;
  // Validate input range
  if (x < 0 || x > 1) {
    throw new InvalidParameterError("x must be in [0,1]", "x", x);
  }
  // Handle boundary cases
  if (x === 0) return 0;
  if (x === 1) return 1;

  return incompleteBetaXY(a, b, x, 1 - x);
}

/**
 * Regularized incomplete gamma functions for s > 0 and finite x > 0, returned
 * as the pair `{ p, q }` = `{ P(s, x), Q(s, x) }` with P + Q = 1. The smaller of
 * the two is computed directly (series for x < s + 1, continued fraction
 * otherwise) and the other is its complement, so both keep full relative
 * accuracy in their own tail where it matters.
 */
function incompleteGammaPQ(s: number, x: number): { p: number; q: number } {
  const ITMAX = Math.min(1e6, Math.ceil(300 + 30 * Math.sqrt(s))); // Maximum iterations
  const EPS = 1e-16; // Convergence threshold
  const FPMIN = 1e-300; // Minimum floating point value

  // ln(x^s e^-x / Γ(s)). For s >= 10 use Stirling so that x ≈ s does not
  // subtract huge, nearly equal numbers.
  let lnPrefix: number;
  if (s >= 10) {
    // ln(1 + d) is taken as log(x / s) once d is far from 0: there 1 + d would
    // inherit the rounding error of d, which is large relative to a small x / s.
    const d = (x - s) / s;
    const lnRatio = Math.abs(d) < 0.5 ? Math.log1p(d) : Math.log(x / s);
    lnPrefix = s * (lnRatio - d) + 0.5 * Math.log(s) - LN_SQRT_2PI - stirlingCorrection(s);
  } else {
    lnPrefix = -x + s * Math.log(x) - logGamma(s);
  }
  const prefix = Math.exp(lnPrefix);

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

    const p = Math.min(1, sum * prefix);
    return { p, q: 1 - p };
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
    if (Math.abs(del - 1) < 1e-15) break; // Converged
  }

  const q = Math.min(1, prefix * h);
  return { p: 1 - q, q };
}

const SQRT_PI = 1.7724538509055159;

/**
 * e^(-x²) with the square split into an exactly representable part and a small
 * remainder, so the rounding error of `x*x` (relative error ~x²·ε in the
 * result) does not leak into the answer for large x.
 */
function expNegSquare(x: number): number {
  const hi = Math.round(x * 16) / 16; // few mantissa bits, so hi*hi is exact
  return Math.exp(-hi * hi) * Math.exp(-(x - hi) * (x + hi));
}

/** e^(-x²/2), split like {@link expNegSquare}. */
function expNegHalfSquare(x: number): number {
  const hi = Math.round(x * 16) / 16;
  return Math.exp(-0.5 * hi * hi) * Math.exp(-0.5 * (x - hi) * (x + hi));
}

/** Beyond this |x|, erfc(|x|) underflows to 0 in double precision. */
const ERFC_UNDERFLOW = 27.3;
/** Below this |x| erf is summed as a power series; above it erfc uses a continued fraction. */
const ERF_SERIES_LIMIT = 1;

/**
 * erf(ax) for 0 <= ax < {@link ERF_SERIES_LIMIT}, from the all-positive series
 * erf(x) = 2/√π · e^(-x²) · Σ 2^n x^(2n+1) / (2n+1)!!. `expNeg` is e^(-ax²),
 * supplied by the caller so it can be formed without rounding `ax*ax`.
 */
function erfSeries(ax: number, expNeg: number): number {
  const x2 = ax * ax;
  let term = ax;
  let sum = ax;
  for (let n = 0; n < 200; n++) {
    term *= (2 * x2) / (2 * n + 3);
    sum += term;
    if (term < sum * 1e-17) break;
  }
  return (2 / SQRT_PI) * expNeg * sum;
}

/**
 * erfc(ax) for ax >= {@link ERF_SERIES_LIMIT} via the Laplace continued fraction
 * erfc(x) = e^(-x²)/√π · 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + ...)))), evaluated
 * with the modified Lentz method. `expNeg` is e^(-ax²), supplied by the caller.
 */
function erfcContinuedFraction(ax: number, expNeg: number): number {
  const TINY = 1e-300;
  let f = ax;
  let c = f;
  let d = 0;
  for (let k = 1; k < 500; k++) {
    const a = k / 2;
    d = ax + a * d;
    if (d === 0) d = TINY;
    c = ax + a / c;
    if (c === 0) c = TINY;
    d = 1 / d;
    const delta = c * d;
    f *= delta;
    if (Math.abs(delta - 1) < 1e-16) break;
  }
  return expNeg / (SQRT_PI * f);
}

/**
 * Computes the complementary error function erfc(x) = 1 - erf(x).
 *
 * Accurate to about 3e-15 relative error over the real line, including the far
 * tails (where `1 - erf(x)` would lose all precision). For |x| < 1 it
 * uses a power series for erf; beyond that a continued fraction for erfc
 * directly.
 *
 * @param x - Input value
 * @returns erfc(x) in [0, 2]
 *
 * @internal
 */
export function erfc(x: number): number {
  if (Number.isNaN(x)) return Number.NaN;
  const ax = Math.abs(x);
  if (ax < ERF_SERIES_LIMIT) {
    const e = erfSeries(ax, expNegSquare(ax));
    return x < 0 ? 1 + e : 1 - e;
  }
  const tail = ax > ERFC_UNDERFLOW ? 0 : erfcContinuedFraction(ax, expNegSquare(ax));
  return x > 0 ? tail : 2 - tail;
}

/**
 * Computes the error function erf(x), accurate to about 1e-15 relative error
 * (including tiny |x|, where `1 - erfc(x)` would cancel).
 *
 * @param x - Input value
 * @returns erf(x) in [-1, 1]
 *
 * @internal
 */
export function erf(x: number): number {
  if (Number.isNaN(x)) return Number.NaN;
  const ax = Math.abs(x);
  const v =
    ax < ERF_SERIES_LIMIT
      ? erfSeries(ax, expNegSquare(ax))
      : 1 - (ax > ERFC_UNDERFLOW ? 0 : erfcContinuedFraction(ax, expNegSquare(ax)));
  return x < 0 ? -v : v;
}

/**
 * P(X > ax) = ½·erfc(ax/√2) for X ~ N(0, 1) and ax >= 0. The exponential is
 * formed from `ax` itself (not from the rounded `ax/√2`), which keeps the
 * relative error near 1e-15 even for ax ≈ 30 instead of ~ax²·1e-16.
 */
function normalUpperTail(ax: number): number {
  const z = ax / Math.SQRT2;
  const expNeg = expNegHalfSquare(ax);
  if (z < ERF_SERIES_LIMIT) return 0.5 * (1 - erfSeries(z, expNeg));
  if (z > ERFC_UNDERFLOW) return 0;
  return 0.5 * erfcContinuedFraction(z, expNeg);
}

/**
 * Computes the cumulative distribution function (CDF) of the standard normal distribution.
 *
 * Uses the relation Φ(x) = ½·erfc(−x/√2) with a double-precision erfc
 * (relative error about 1e-15, including the lower tail).
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
  if (Number.isNaN(x)) return Number.NaN;
  // Φ(x) = 0.5 * erfc(-x/√2); evaluated through the upper tail of |x|.
  const tail = normalUpperTail(Math.abs(x));
  return x < 0 ? tail : 1 - tail;
}

/**
 * Survival function of the standard normal distribution, P(X > x) = Φ(−x).
 *
 * Unlike `1 - normalCdf(x)`, this keeps full relative precision for large `x`
 * (e.g. p-values of 1e-20).
 *
 * @param x - Input value
 * @returns Probability P(X > x) where X ~ N(0, 1)
 *
 * @internal
 */
export function normalSf(x: number): number {
  if (Number.isNaN(x)) return Number.NaN;
  const tail = normalUpperTail(Math.abs(x));
  return x < 0 ? 1 - tail : tail;
}

// Acklam's rational approximation coefficients.
const PPF_A: readonly number[] = [
  -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2,
  -3.066479806614716e1, 2.506628277459239,
];
const PPF_B: readonly number[] = [
  -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1,
  -1.328068155288572e1,
];
const PPF_C: readonly number[] = [
  -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734,
  4.374664141464968, 2.938163982698783,
];
const PPF_D: readonly number[] = [
  7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416,
];

/**
 * Computes the inverse (quantile) of the standard normal CDF, Φ⁻¹(p).
 *
 * Uses Peter Acklam's rational approximation followed by one Halley
 * refinement step against {@link normalCdf}, which gives close to full double
 * precision (relative error about 1e-15) over (0, 1). Upper-half
 * probabilities use the symmetry Φ⁻¹(p) = −Φ⁻¹(1−p).
 *
 * @param p - Probability in [0, 1]
 * @returns z such that Φ(z) = p (`NaN` outside [0, 1])
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
  if (Number.isNaN(p) || p < 0 || p > 1) return Number.NaN;
  if (p === 0) return Number.NEGATIVE_INFINITY;
  if (p === 1) return Number.POSITIVE_INFINITY;
  if (p === 0.5) return 0;
  // 1 - p is exact for p in [0.5, 1], so the upper half loses nothing.
  if (p > 0.5) return -normalPpf(1 - p);

  const a = PPF_A as number[];
  const b = PPF_B as number[];
  const c = PPF_C as number[];
  const d = PPF_D as number[];

  const pLow = 0.02425;
  let z: number;

  if (p < pLow) {
    const q = Math.sqrt(-2 * Math.log(p));
    z =
      (((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) /
      ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1);
  } else {
    const q = p - 0.5;
    const r = q * q;
    z =
      ((((((a[0]! * r + a[1]!) * r + a[2]!) * r + a[3]!) * r + a[4]!) * r + a[5]!) * q) /
      (((((b[0]! * r + b[1]!) * r + b[2]!) * r + b[3]!) * r + b[4]!) * r + 1);
  }

  // One Halley refinement step for full double precision. For p below ~1e-309
  // exp(z²/2) overflows; the Acklam estimate is kept there.
  // In the central branch Φ(z) - p is formed as ½·erf(z/√2) - (p - ½), where
  // p - ½ is exact and erf has no cancellation, so tiny quantiles keep their
  // relative precision (Φ(z) itself only has an absolute error of ~1e-16).
  const e = p < pLow ? normalCdf(z) - p : 0.5 * erf(z / Math.SQRT2) - (p - 0.5);
  const u = e * Math.sqrt(2 * Math.PI) * Math.exp((z * z) / 2);
  if (Number.isFinite(u)) z = z - u / (1 + (z * u) / 2);
  return z;
}

/**
 * P(|T| > |t|) for T ~ t(df): the two-sided tail, I_x(df/2, 1/2) with
 * x = df/(df+t²). Both x and 1−x are formed without cancellation.
 */
function studentTTwoSided(t: number, df: number): number {
  const tt = t * t;
  if (!Number.isFinite(tt)) return 0;
  const denom = df + tt;
  return incompleteBetaXY(df / 2, 0.5, df / denom, tt / denom);
}

/**
 * Computes the cumulative distribution function (CDF) of Student's t-distribution.
 *
 * Uses the relationship between t-distribution and incomplete beta function:
 * F_t(t; ν) = 0.5 * I_x(ν/2, 1/2) where x = ν/(ν + t²)
 *
 * @param t - t-statistic value
 * @param df - Degrees of freedom (must be > 0; `Infinity` gives the standard normal)
 * @returns Probability P(T <= t) where T ~ t(df)
 * @throws {InvalidParameterError} If df is not greater than 0 (or is NaN)
 *
 * @example
 * ```ts
 * studentTCdf(0, 10);      // Returns 0.5 (symmetric at 0)
 * studentTCdf(2.228, 10);  // Returns ~0.975 (95th percentile for df=10)
 * ```
 */
export function studentTCdf(t: number, df: number): number {
  // Validate degrees of freedom (written so that NaN is rejected as well)
  if (!(df > 0)) {
    throw new InvalidParameterError("df must be > 0", "df", df);
  }
  // Handle infinite t-values
  if (Number.isNaN(t)) return Number.NaN;
  if (!Number.isFinite(t)) return t < 0 ? 0 : 1;
  // The limit of t(df) as df -> Infinity is the standard normal distribution.
  if (df === Number.POSITIVE_INFINITY) return normalCdf(t);

  const p = 0.5 * studentTTwoSided(t, df);
  // Use symmetry of t-distribution around 0
  return t >= 0 ? 1 - p : p;
}

/**
 * Survival function of Student's t-distribution, P(T > t).
 *
 * Unlike `1 - studentTCdf(t, df)`, this keeps full relative precision for
 * large `t` (tiny upper-tail probabilities).
 *
 * @param t - t-statistic value
 * @param df - Degrees of freedom (must be > 0)
 * @returns Probability P(T > t) where T ~ t(df)
 * @throws {InvalidParameterError} If df <= 0
 *
 * @internal
 */
export function studentTSf(t: number, df: number): number {
  return studentTCdf(-t, df);
}

/**
 * Two-sided p-value for a correlation coefficient `r` with `df = n - 2` degrees
 * of freedom, P(|T| >= |t|) for t = r·sqrt(df/(1−r²)).
 *
 * Computed directly as I_{1−r²}(df/2, 1/2), with 1−r² formed as (1−|r|)(1+|r|),
 * so it stays accurate when |r| is close to 1 and for p-values far below 1e-16.
 * `|r| >= 1` gives 0 and `r = 0` gives 1. `r` is clamped to [-1, 1].
 *
 * @param r - Correlation coefficient
 * @param df - Degrees of freedom (must be > 0)
 * @returns Two-sided p-value in [0, 1] (`NaN` if `r` is `NaN`)
 * @throws {InvalidParameterError} If df <= 0
 *
 * @internal
 */
export function correlationPValue(r: number, df: number): number {
  if (!Number.isFinite(df) || df <= 0) {
    throw new InvalidParameterError("df must be > 0", "df", df);
  }
  if (Number.isNaN(r)) return Number.NaN;
  const ar = Math.min(1, Math.abs(r));
  if (ar === 1) return 0;
  if (ar === 0) return 1;
  return incompleteBetaXY(df / 2, 0.5, (1 - ar) * (1 + ar), ar * ar);
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
  if (Number.isNaN(x)) return Number.NaN;
  if (x === Number.POSITIVE_INFINITY) return 1;
  // Chi-square is non-negative
  if (x <= 0) return 0;
  // Use gamma CDF relationship: χ²(k) is Gamma(k/2, 2)
  return incompleteGammaPQ(k / 2, x / 2).p;
}

/**
 * Survival function of the chi-square distribution, P(X > x).
 *
 * Unlike `1 - chiSquareCdf(x, k)`, this keeps full relative precision in the
 * upper tail (p-values far below 1e-16).
 *
 * @param x - Chi-square statistic
 * @param k - Degrees of freedom (must be > 0)
 * @returns Probability P(X > x) where X ~ χ²(k)
 * @throws {InvalidParameterError} If k <= 0
 *
 * @internal
 */
export function chiSquareSf(x: number, k: number): number {
  if (!Number.isFinite(k) || k <= 0) {
    throw new InvalidParameterError("degrees of freedom must be > 0", "k", k);
  }
  if (Number.isNaN(x)) return Number.NaN;
  if (x === Number.POSITIVE_INFINITY) return 0;
  if (x <= 0) return 1;
  return incompleteGammaPQ(k / 2, x / 2).q;
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
  if (Number.isNaN(x)) return Number.NaN;
  if (x === Number.POSITIVE_INFINITY) return 1;
  // F-statistic is non-negative
  if (x <= 0) return 0;

  // Transform to incomplete beta parameter
  const num = dfn * x;
  if (!Number.isFinite(num)) return 1;
  const denom = num + dfd;
  return incompleteBetaXY(dfn / 2, dfd / 2, num / denom, dfd / denom);
}

/**
 * Survival function of the F-distribution, P(F > x).
 *
 * Unlike `1 - fCdf(x, dfn, dfd)`, this keeps full relative precision in the
 * upper tail (small p-values).
 *
 * @param x - F-statistic value
 * @param dfn - Numerator degrees of freedom (must be > 0)
 * @param dfd - Denominator degrees of freedom (must be > 0)
 * @returns Probability P(F > x) where F ~ F(dfn, dfd)
 * @throws {InvalidParameterError} If dfn <= 0 or dfd <= 0
 *
 * @internal
 */
export function fSf(x: number, dfn: number, dfd: number): number {
  if (!Number.isFinite(dfn) || dfn <= 0) {
    throw new InvalidParameterError("degrees of freedom (dfn) must be > 0", "dfn", dfn);
  }
  if (!Number.isFinite(dfd) || dfd <= 0) {
    throw new InvalidParameterError("degrees of freedom (dfd) must be > 0", "dfd", dfd);
  }
  if (Number.isNaN(x)) return Number.NaN;
  if (x === Number.POSITIVE_INFINITY) return 0;
  if (x <= 0) return 1;

  const num = dfn * x;
  if (!Number.isFinite(num)) return 0;
  const denom = num + dfd;
  // 1 - I_xx(dfn/2, dfd/2) = I_yy(dfd/2, dfn/2)
  return incompleteBetaXY(dfd / 2, dfn / 2, dfd / denom, num / denom);
}
