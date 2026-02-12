import {
  type Axis,
  DTypeError,
  dtypeToTypedArrayCtor,
  getBigIntElement,
  getNumericElement,
  normalizeAxis,
} from "../../core";
import { computeStrides, Tensor } from "../tensor/Tensor";

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
    const slice = new Array<number>(axisLen);

    for (const baseCoord of outerCoords(t.shape, ax)) {
      for (let k = 0; k < axisLen; k++) {
        baseCoord[ax] = k;
        const off = physicalOffset(baseCoord, t.strides, t.offset);
        slice[k] = getNumericElement(numericData, off);
      }
      slice.sort((a, b) => a - b);
      if (descending) slice.reverse();

      if (out instanceof BigInt64Array) break; // type guard
      for (let k = 0; k < axisLen; k++) {
        baseCoord[ax] = k;
        const outFlat = flatFromCoord(baseCoord, logicalStrides);
        out[outFlat] = slice[k] ?? 0;
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

    for (const baseCoord of outerCoords(t.shape, ax)) {
      for (let k = 0; k < axisLen; k++) {
        baseCoord[ax] = k;
        const off = physicalOffset(baseCoord, t.strides, t.offset);
        vals[k] = getNumericElement(numericData, off);
      }
      for (let k = 0; k < axisLen; k++) idxBuf[k] = k;
      idxBuf.sort((a, b) => (vals[a] ?? 0) - (vals[b] ?? 0));
      if (descending) idxBuf.reverse();

      for (let k = 0; k < axisLen; k++) {
        baseCoord[ax] = k;
        const outFlat = flatFromCoord(baseCoord, logicalStrides);
        out[outFlat] = idxBuf[k] ?? 0;
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

/** Convert a coordinate array to a flat index using logical (row-major) strides. */
function flatFromCoord(coord: number[], strides: readonly number[]): number {
  let flat = 0;
  for (let i = 0; i < coord.length; i++) {
    flat += (coord[i] ?? 0) * (strides[i] ?? 0);
  }
  return flat;
}
