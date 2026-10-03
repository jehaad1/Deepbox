/**
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import { type Axis, DTypeError, dtypeToTypedArrayCtor, normalizeAxis } from "../../core";
import { computeStrides, Tensor } from "../tensor/Tensor";
import { RADIX_SORT_THRESHOLD, radixArgsortF64, radixSortF64 } from "./radix";

/** Largest lane length for which `(key << 21) | index` is exact in a double. */
const PACK_MAX_LANE = 2097152;

/**
 * Visit every 1-D lane of `t` along `ax`.
 *
 * `inBase` is the physical buffer offset of the lane's first element in `t`
 * (view offset included) and `outBase` is the matching flat offset in a dense
 * row-major result of the same shape. Lanes are visited in row-major order of
 * the remaining axes. Empty tensors have no lanes; a 0-d tensor has one.
 */
function forEachLane(
  t: Tensor,
  ax: number,
  visit: (inBase: number, outBase: number) => void
): void {
  if (t.size === 0) return;
  const shape = t.shape;
  const rank = shape.length;
  if (rank === 0) {
    visit(t.offset, 0);
    return;
  }
  const strides = t.strides;
  const logical = computeStrides(shape);

  const outerDims: number[] = [];
  let total = 1;
  for (let d = 0; d < rank; d++) {
    if (d === ax) continue;
    outerDims.push(d);
    total *= shape[d] ?? 1;
  }

  const counter = new Array<number>(outerDims.length).fill(0);
  let inBase = t.offset;
  let outBase = 0;
  for (let n = 0; n < total; n++) {
    visit(inBase, outBase);
    for (let j = outerDims.length - 1; j >= 0; j--) {
      const d = outerDims[j] as number;
      const dim = shape[d] ?? 1;
      const c = (counter[j] as number) + 1;
      inBase += strides[d] ?? 0;
      outBase += logical[d] ?? 0;
      if (c < dim) {
        counter[j] = c;
        break;
      }
      inBase -= (strides[d] ?? 0) * dim;
      outBase -= (logical[d] ?? 0) * dim;
      counter[j] = 0;
    }
  }
}

/** Equality used to find runs of tied values: all NaNs tie, `-0` ties with `+0`. */
function sameValue(a: number | bigint, b: number | bigint): boolean {
  return (
    a === b ||
    (typeof a === "number" && typeof b === "number" && Number.isNaN(a) && Number.isNaN(b))
  );
}

/**
 * Write a lane of ascending-stable indices to `out`.
 *
 * For descending order, runs of tied values keep their original index order
 * (a stable descending sort, like `torch.argsort(descending=True, stable=True)`
 * or `np.argsort(-a, kind="stable")`); NaNs come first.
 */
function writeIndexLane(
  asc: ArrayLike<number>,
  vals: ArrayLike<number | bigint>,
  n: number,
  descending: boolean,
  out: Int32Array,
  outBase: number,
  outStride: number
): void {
  if (!descending) {
    for (let k = 0; k < n; k++) out[outBase + k * outStride] = asc[k] as number;
    return;
  }
  let w = 0;
  let end = n;
  while (end > 0) {
    let start = end - 1;
    const v = vals[asc[start] as number] as number | bigint;
    while (start > 0 && sameValue(vals[asc[start - 1] as number] as number | bigint, v)) start--;
    for (let p = start; p < end; p++) {
      out[outBase + w * outStride] = asc[p] as number;
      w++;
    }
    end = start;
  }
}

/**
 * Sort values along a given axis, returning a new tensor of the same shape and dtype.
 *
 * NaNs are treated as larger than every other value: they end up last for
 * ascending order and first for descending order (NumPy / PyTorch behavior).
 * `-0` and `+0` compare equal. The input is never modified. A 0-d tensor is
 * treated as a single element (axis 0 or -1), as in PyTorch.
 *
 * Complexity is O(L log L) per lane of length L, with a linear-time radix sort
 * for long float lanes.
 *
 * @param t - Input tensor (any numeric dtype, including int64)
 * @param axis - Axis to sort along (default: -1, the last axis)
 * @param descending - Sort from largest to smallest (default: false)
 * @throws {DTypeError} If the tensor has string dtype
 * @throws {InvalidParameterError} If `axis` is out of range
 *
 * @example
 * ```ts
 * import { tensor, sort } from "deepbox/ndarray";
 *
 * sort(tensor([3, 1, 2])).toArray(); // [1, 2, 3]
 * sort(tensor([[3, 1], [2, 5]]), 0).toArray(); // [[2, 1], [3, 5]]
 * ```
 */
export function sort(t: Tensor, axis: Axis | undefined = -1, descending = false): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("sort is not implemented for string dtype");
  }

  const ax = normalizeAxis(axis ?? -1, Math.max(t.ndim, 1));
  const axisLen = t.ndim === 0 ? 1 : (t.shape[ax] ?? 1);
  const inStride = t.ndim === 0 ? 1 : (t.strides[ax] ?? 1);
  const outStride = t.ndim === 0 ? 1 : (computeStrides(t.shape)[ax] ?? 1);

  const Ctor = dtypeToTypedArrayCtor(t.dtype);
  const out = new Ctor(t.size);
  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("sort is not implemented for string dtype");
  }

  if (data instanceof BigInt64Array && out instanceof BigInt64Array) {
    // TypedArray.prototype.sort without a comparator is a numeric ascending sort.
    const lane = new BigInt64Array(axisLen);
    forEachLane(t, ax, (inBase, outBase) => {
      for (let k = 0; k < axisLen; k++) lane[k] = data[inBase + k * inStride] as bigint;
      lane.sort();
      if (descending) lane.reverse();
      if (outStride === 1) {
        out.set(lane, outBase);
      } else {
        for (let k = 0; k < axisLen; k++) out[outBase + k * outStride] = lane[k] as bigint;
      }
    });
  } else if (!(data instanceof BigInt64Array) && !(out instanceof BigInt64Array)) {
    // Float64 lane buffer + comparator-free sort: TypedArray.prototype.sort is
    // numeric ascending with NaNs last (the NumPy order) and ~10x faster than a
    // comparator sort. Every supported element type is exactly representable
    // as a double, so the round trip is lossless.
    const lane = new Float64Array(axisLen);
    const useRadix = axisLen >= RADIX_SORT_THRESHOLD;
    forEachLane(t, ax, (inBase, outBase) => {
      if (inStride === 1) {
        lane.set(data.subarray(inBase, inBase + axisLen));
      } else {
        for (let k = 0; k < axisLen; k++) lane[k] = data[inBase + k * inStride] as number;
      }
      if (useRadix) radixSortF64(lane);
      else lane.sort();
      if (descending) lane.reverse();
      if (outStride === 1) {
        out.set(lane, outBase);
      } else {
        for (let k = 0; k < axisLen; k++) out[outBase + k * outStride] = lane[k] as number;
      }
    });
  } else {
    throw new DTypeError(`sort is not implemented for dtype ${t.dtype}`);
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: t.dtype,
    device: t.device,
  });
}

/**
 * Return the indices that would sort the tensor along a given axis.
 *
 * The sort is stable: equal values keep their original relative order, also
 * for `descending = true` (ties are not reversed). NaNs are treated as larger
 * than every other value, so they are last for ascending and first for
 * descending order. The result has dtype `int32` and the same shape as `t`.
 *
 * @param t - Input tensor (any numeric dtype, including int64)
 * @param axis - Axis to sort along (default: -1, the last axis)
 * @param descending - Order from largest to smallest (default: false)
 * @throws {DTypeError} If the tensor has string dtype
 * @throws {InvalidParameterError} If `axis` is out of range
 *
 * @example
 * ```ts
 * import { tensor, argsort } from "deepbox/ndarray";
 *
 * argsort(tensor([3, 1, 2])).toArray(); // [1, 2, 0]
 * argsort(tensor([1, 2, 2, 3]), -1, true).toArray(); // [3, 1, 2, 0]
 * ```
 */
export function argsort(t: Tensor, axis: Axis | undefined = -1, descending = false): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("argsort is not implemented for string dtype");
  }

  const ax = normalizeAxis(axis ?? -1, Math.max(t.ndim, 1));
  const axisLen = t.ndim === 0 ? 1 : (t.shape[ax] ?? 1);
  const inStride = t.ndim === 0 ? 1 : (t.strides[ax] ?? 1);
  const outStride = t.ndim === 0 ? 1 : (computeStrides(t.shape)[ax] ?? 1);

  const out = new Int32Array(t.size);
  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("argsort is not implemented for string dtype");
  }

  const asc = new Int32Array(axisLen);

  if (data instanceof BigInt64Array) {
    const vals = new BigInt64Array(axisLen);
    const order = new Array<number>(axisLen);
    forEachLane(t, ax, (inBase, outBase) => {
      for (let k = 0; k < axisLen; k++) {
        vals[k] = data[inBase + k * inStride] as bigint;
        order[k] = k;
      }
      order.sort((a, b) => {
        const va = vals[a] as bigint;
        const vb = vals[b] as bigint;
        return va < vb ? -1 : va > vb ? 1 : 0;
      });
      writeIndexLane(order, vals, axisLen, descending, out, outBase, outStride);
    });
  } else if (
    (data instanceof Float32Array || data instanceof Int32Array || data instanceof Uint8Array) &&
    axisLen <= PACK_MAX_LANE
  ) {
    // Encode each value as an order-preserving uint32 and pack
    // `key * 2^21 + index` into a Float64Array (exact: key < 2^32, index < 2^21),
    // then use the comparator-free typed sort. NaNs encode above +inf, and the
    // index in the low bits keeps the sort stable.
    const vals = new Float64Array(axisLen);
    const packed = new Float64Array(axisLen);
    const f32Scratch = new Float32Array(1);
    const u32Scratch = new Uint32Array(f32Scratch.buffer);
    const isFloat = data instanceof Float32Array;
    forEachLane(t, ax, (inBase, outBase) => {
      for (let k = 0; k < axisLen; k++) {
        const v = data[inBase + k * inStride] as number;
        vals[k] = v;
        let enc: number;
        if (isFloat) {
          if (Number.isNaN(v)) {
            enc = 4294967295; // above +inf: NaN sorts last
          } else if (v === 0) {
            // -0 and +0 share one key so they compare equal.
            enc = 0x80000000;
          } else {
            f32Scratch[0] = v;
            const bits = (u32Scratch[0] as number) >>> 0;
            enc = bits & 0x80000000 ? ~bits >>> 0 : (bits | 0x80000000) >>> 0;
          }
        } else {
          enc = v + 2147483648; // int32 / uint8 -> order-preserving offset
        }
        packed[k] = enc * PACK_MAX_LANE + k;
      }
      packed.sort();
      for (let k = 0; k < axisLen; k++) {
        const key = packed[k] as number;
        asc[k] = key - Math.floor(key / PACK_MAX_LANE) * PACK_MAX_LANE;
      }
      writeIndexLane(asc, vals, axisLen, descending, out, outBase, outStride);
    });
  } else {
    const vals = new Float64Array(axisLen);
    const useRadix = axisLen >= RADIX_SORT_THRESHOLD;
    const order = useRadix ? null : new Array<number>(axisLen);
    forEachLane(t, ax, (inBase, outBase) => {
      for (let k = 0; k < axisLen; k++) vals[k] = data[inBase + k * inStride] as number;
      if (order === null) {
        radixArgsortF64(vals, asc);
        writeIndexLane(asc, vals, axisLen, descending, out, outBase, outStride);
      } else {
        for (let k = 0; k < axisLen; k++) order[k] = k;
        order.sort((a, b) => compareNumbersNanLast(vals[a] as number, vals[b] as number));
        writeIndexLane(order, vals, axisLen, descending, out, outBase, outStride);
      }
    });
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: "int32",
    device: t.device,
  });
}

/**
 * NaN-aware comparator matching NumPy: NaN sorts to the end. A plain
 * `a - b` comparator returns NaN for NaN operands, which is undefined
 * behavior for Array.prototype.sort and leaves the array unsorted.
 */
function compareNumbersNanLast(a: number, b: number): number {
  const aNaN = Number.isNaN(a);
  const bNaN = Number.isNaN(b);
  if (aNaN) return bNaN ? 0 : 1;
  if (bNaN) return -1;
  return a < b ? -1 : a > b ? 1 : 0;
}
