/**
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import {
  type Axis,
  DTypeError,
  dtypeToTypedArrayCtor,
  getBigIntElement,
  getNumericElement,
  normalizeAxis,
} from "../../core";
import { computeStrides, Tensor } from "../tensor/Tensor";
import { readNumericContiguous } from "./_internal";
import { RADIX_SORT_THRESHOLD, radixArgsortF64, radixSortF64 } from "./radix";

/**
 * Compute the physical buffer offset for a multi-dimensional coordinate.
 */
function physicalOffset(coord: number[], strides: readonly number[], offset: number): number {
  let out = offset;
  for (let i = 0; i < coord.length; i++) {
    out += (coord[i] ?? 0) * (strides[i] ?? 0);
  }
  return out;
}

/**
 * Iterate over all "outer" coordinate tuples — every combination of indices
 * for all dimensions *except* the sort axis.  For each tuple we yield a
 * coordinate array whose `axis` element is 0 (caller will vary it).
 */
function* outerCoords(shape: readonly number[], axis: number): Generator<number[]> {
  const rank = shape.length;
  if (rank === 0) {
    yield [];
    return;
  }
  // Build the list of dims to iterate (everything except axis)
  const outerDims: number[] = [];
  for (let d = 0; d < rank; d++) {
    if (d !== axis) outerDims.push(d);
  }

  const total = outerDims.reduce((n, d) => n * (shape[d] ?? 1), 1);
  const coord = new Array<number>(rank).fill(0);

  for (let flat = 0; flat < total; flat++) {
    // Unravel `flat` into outer coordinates
    let rem = flat;
    for (let i = outerDims.length - 1; i >= 0; i--) {
      const d = outerDims[i] ?? 0;
      const dim = shape[d] ?? 1;
      coord[d] = rem % dim;
      rem = Math.floor(rem / dim);
    }
    coord[axis] = 0;
    yield coord;
  }
}

/**
 * Sort values along a given axis.
 *
 * Supports tensors of any dimensionality.  Default axis is -1 (last).
 *
 * Performance:
 * - O(N log N) where N is the total number of elements.
 */
export function sort(t: Tensor, axis: Axis | undefined = -1, descending = false): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("sort is not implemented for string dtype");
  }

  const ax = normalizeAxis(axis ?? -1, t.ndim);
  const axisLen = t.shape[ax] ?? 1;
  const logicalStrides = computeStrides(t.shape);

  const Ctor = dtypeToTypedArrayCtor(t.dtype);
  const out = new Ctor(t.size);

  // 1-D fast path: sort a dense copy directly, skipping the per-element
  // coordinate machinery of the generic N-D path.
  if (t.ndim <= 1 && !(out instanceof BigInt64Array)) {
    const src = readNumericContiguous(t);
    if (src) {
      const lane = out instanceof Float64Array ? out : new Float64Array(t.size);
      if (lane !== src) lane.set(src as Float64Array);
      if (t.size >= RADIX_SORT_THRESHOLD) {
        radixSortF64(lane);
      } else {
        lane.sort();
      }
      if (descending) lane.reverse();
      if (lane !== out) out.set(lane);
      return Tensor.fromTypedArray({
        data: out,
        shape: t.shape,
        dtype: t.dtype,
        device: t.device,
      });
    }
  }

  if (t.data instanceof BigInt64Array) {
    const bigintData = t.data;
    const slice = new Array<bigint>(axisLen);

    for (const baseCoord of outerCoords(t.shape, ax)) {
      // Extract 1D slice along axis
      for (let k = 0; k < axisLen; k++) {
        baseCoord[ax] = k;
        const off = physicalOffset(baseCoord, t.strides, t.offset);
        slice[k] = getBigIntElement(bigintData, off);
      }
      slice.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
      if (descending) slice.reverse();

      // Write back
      if (!(out instanceof BigInt64Array)) break; // type guard
      for (let k = 0; k < axisLen; k++) {
        baseCoord[ax] = k;
        const outFlat = flatFromCoord(baseCoord, logicalStrides);
        out[outFlat] = slice[k] ?? 0n;
      }
    }
  } else {
    const numericData = t.data;
    if (Array.isArray(numericData)) {
      throw new DTypeError("sort is not implemented for string dtype");
    }
    // Typed lane buffer + comparator-free sort: TypedArray.prototype.sort
    // is numeric ascending with NaNs last, exactly matching
    // compareNumbersNanLast (~10x faster than the comparator sort).
    const lane = new Float64Array(axisLen);

    for (const baseCoord of outerCoords(t.shape, ax)) {
      for (let k = 0; k < axisLen; k++) {
        baseCoord[ax] = k;
        const off = physicalOffset(baseCoord, t.strides, t.offset);
        lane[k] = getNumericElement(numericData, off);
      }
      if (axisLen >= RADIX_SORT_THRESHOLD) {
        radixSortF64(lane);
      } else {
        lane.sort();
      }

      if (out instanceof BigInt64Array) break; // type guard
      if (descending) {
        for (let k = 0; k < axisLen; k++) {
          baseCoord[ax] = k;
          const outFlat = flatFromCoord(baseCoord, logicalStrides);
          out[outFlat] = lane[axisLen - 1 - k] ?? 0;
        }
      } else {
        for (let k = 0; k < axisLen; k++) {
          baseCoord[ax] = k;
          const outFlat = flatFromCoord(baseCoord, logicalStrides);
          out[outFlat] = lane[k] ?? 0;
        }
      }
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: t.dtype,
    device: t.device,
  });
}

/**
 * Return indices that would sort the tensor along a given axis.
 *
 * Supports tensors of any dimensionality.  Default axis is -1 (last).
 *
 * Performance:
 * - O(N log N) where N is the total number of elements.
 */
export function argsort(t: Tensor, axis: Axis | undefined = -1, descending = false): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("argsort is not implemented for string dtype");
  }

  const ax = normalizeAxis(axis ?? -1, t.ndim);
  const axisLen = t.shape[ax] ?? 1;
  const logicalStrides = computeStrides(t.shape);

  const out = new Int32Array(t.size);

  // 1-D fast path: stable radix argsort straight into the output buffer.
  if (t.ndim <= 1 && !(t.data instanceof BigInt64Array) && !Array.isArray(t.data)) {
    const src = readNumericContiguous(t);
    if (src && t.size >= RADIX_SORT_THRESHOLD) {
      const lane = src instanceof Float64Array ? src : Float64Array.from(src as Float32Array);
      radixArgsortF64(lane, out);
      if (descending) out.reverse();
      return Tensor.fromTypedArray({
        data: out,
        shape: t.shape,
        dtype: "int32",
        device: t.device,
      });
    }
  }

  const idxBuf = Array.from({ length: axisLen }, (_, i) => i);

  if (t.data instanceof BigInt64Array) {
    const bigintData = t.data;
    const vals = new Array<bigint>(axisLen);

    for (const baseCoord of outerCoords(t.shape, ax)) {
      for (let k = 0; k < axisLen; k++) {
        baseCoord[ax] = k;
        const off = physicalOffset(baseCoord, t.strides, t.offset);
        vals[k] = getBigIntElement(bigintData, off);
      }
      // Reset indices
      for (let k = 0; k < axisLen; k++) idxBuf[k] = k;
      idxBuf.sort((a, b) => {
        const va = vals[a] ?? 0n,
          vb = vals[b] ?? 0n;
        return va < vb ? -1 : va > vb ? 1 : 0;
      });
      if (descending) idxBuf.reverse();

      for (let k = 0; k < axisLen; k++) {
        baseCoord[ax] = k;
        const outFlat = flatFromCoord(baseCoord, logicalStrides);
        out[outFlat] = idxBuf[k] ?? 0;
      }
    }
  } else {
    const numericData = t.data;
    if (Array.isArray(numericData)) {
      throw new DTypeError("argsort is not implemented for string dtype");
    }
    const vals = new Array<number>(axisLen);

    // Fast path for float32/int32/uint8/bool lanes: encode each value into
    // an order-preserving uint32 and pack (key << 32) | index into a
    // Float64Array (exact for axisLen <= 2^21), then use the comparator-free
    // typed sort. NaNs encode above +inf, matching NumPy's NaN-last order;
    // the index in the low bits keeps the sort stable.
    const packable =
      (t.dtype === "float32" || t.dtype === "int32" || t.dtype === "uint8" || t.dtype === "bool") &&
      axisLen <= 2097152;

    if (packable) {
      const packed = new Float64Array(axisLen);
      const f32Scratch = new Float32Array(1);
      const u32Scratch = new Uint32Array(f32Scratch.buffer);
      const isFloat = t.dtype === "float32";
      // key = enc * 2^21 + index fits doubles exactly: enc < 2^32, index < 2^21.
      const IDX_RANGE = 2097152;
      for (const baseCoord of outerCoords(t.shape, ax)) {
        for (let k = 0; k < axisLen; k++) {
          baseCoord[ax] = k;
          const off = physicalOffset(baseCoord, t.strides, t.offset);
          const v = getNumericElement(numericData, off);
          let enc: number;
          if (isFloat) {
            if (Number.isNaN(v)) {
              enc = 4294967295; // above +inf: NaN sorts last
            } else if (v === 0) {
              // Normalize -0 and +0 to one key so signed zeros compare equal
              // (matches numpy and the comparator fallback); the index
              // tiebreaker then keeps their order stable.
              enc = 0x80000000;
            } else {
              f32Scratch[0] = v;
              const bits = u32Scratch[0]! >>> 0;
              enc = bits & 0x80000000 ? ~bits >>> 0 : (bits | 0x80000000) >>> 0;
            }
          } else {
            enc = v + 2147483648; // int32 -> order-preserving offset
          }
          packed[k] = enc * IDX_RANGE + k;
        }
        packed.sort();
        for (let k = 0; k < axisLen; k++) {
          const key = packed[descending ? axisLen - 1 - k : k]!;
          const idx = key - Math.floor(key / IDX_RANGE) * IDX_RANGE;
          baseCoord[ax] = k;
          const outFlat = flatFromCoord(baseCoord, logicalStrides);
          out[outFlat] = idx;
        }
      }
    } else {
      for (const baseCoord of outerCoords(t.shape, ax)) {
        for (let k = 0; k < axisLen; k++) {
          baseCoord[ax] = k;
          const off = physicalOffset(baseCoord, t.strides, t.offset);
          vals[k] = getNumericElement(numericData, off);
        }
        for (let k = 0; k < axisLen; k++) idxBuf[k] = k;
        idxBuf.sort((a, b) => compareNumbersNanLast(vals[a] ?? 0, vals[b] ?? 0));
        if (descending) idxBuf.reverse();

        for (let k = 0; k < axisLen; k++) {
          baseCoord[ax] = k;
          const outFlat = flatFromCoord(baseCoord, logicalStrides);
          out[outFlat] = idxBuf[k] ?? 0;
        }
      }
    }
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

/** Convert a coordinate array to a flat index using logical (row-major) strides. */
function flatFromCoord(coord: number[], strides: readonly number[]): number {
  let flat = 0;
  for (let i = 0; i < coord.length; i++) {
    flat += (coord[i] ?? 0) * (strides[i] ?? 0);
  }
  return flat;
}
