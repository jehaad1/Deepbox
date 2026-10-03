/**
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import {
  type Axis,
  DeepboxError,
  DTypeError,
  dtypeToTypedArrayCtor,
  getBigIntElement,
  getNumericElement,
  IndexError,
  InvalidParameterError,
  normalizeAxis,
  shapeToSize,
} from "../../core";
import type { SliceRange } from "./slice_helpers";
import { Tensor } from "./Tensor";

export type { SliceRange } from "./slice_helpers";

/**
 * Slice a tensor.
 *
 * Each argument selects along one leading axis: a number picks a single index
 * and drops that axis; a `{ start, end, step }` range keeps the axis (NumPy
 * slicing semantics, negative bounds count from the end, `step` may be
 * negative). Axes without an argument are kept whole. The result is a copy.
 *
 * @param t - Input tensor
 * @param ranges - One index or range per leading axis
 * @returns Sliced tensor
 * @throws {ShapeError} If more ranges than dimensions are given
 * @throws {IndexError} If a single index is out of bounds
 * @throws {InvalidParameterError} If a step is zero or a bound is not an integer
 *
 * @example
 * ```ts
 * const t = tensor([[1, 2, 3], [4, 5, 6]]);
 * slice(t, { start: 0, end: 1 });  // [[1, 2, 3]]
 * slice(t, 0, { start: 1 });       // [2, 3]
 * ```
 */
export function slice(t: Tensor, ...ranges: SliceRange[]): Tensor {
  // Same semantics as the method form; Tensor.slice has the optimized
  // row-blocked copy path.
  return t.slice(...ranges);
}

/**
 * Storage offsets of every element of a (sub)shape, in row-major order.
 */
function enumerateOffsets(shape: readonly number[], strides: readonly number[]): Float64Array {
  const count = shapeToSize(shape);
  const out = new Float64Array(count);
  const ndim = shape.length;
  const coord = new Array<number>(ndim).fill(0);
  let off = 0;
  for (let i = 0; i < count; i++) {
    out[i] = off;
    for (let d = ndim - 1; d >= 0; d--) {
      const c = (coord[d] ?? 0) + 1;
      off += strides[d] ?? 0;
      if (c < (shape[d] ?? 0)) {
        coord[d] = c;
        break;
      }
      off -= c * (strides[d] ?? 0);
      coord[d] = 0;
    }
  }
  return out;
}

/**
 * Copy `src` elements selected by (outer x index x inner) into `dst`.
 */
function gatherCopy<V>(
  src: ArrayLike<V>,
  dst: { [index: number]: V },
  base: number,
  outerOffsets: Float64Array,
  innerOffsets: Float64Array,
  idx: Float64Array,
  axisStride: number
): void {
  const n = idx.length;
  const inner = innerOffsets.length;
  let pos = 0;
  for (let o = 0; o < outerOffsets.length; o++) {
    const outerBase = base + (outerOffsets[o] ?? 0);
    for (let k = 0; k < n; k++) {
      const rowBase = outerBase + (idx[k] ?? 0) * axisStride;
      for (let j = 0; j < inner; j++) {
        dst[pos++] = src[rowBase + (innerOffsets[j] ?? 0)] as V;
      }
    }
  }
}

/**
 * Gather values along an axis specified by indices.
 *
 * The result has the shape of `t` with the size of `axis` replaced by
 * `indices.size`, i.e. `out[..., k, ...] = t[..., indices[k], ...]`. This is
 * `torch.index_select(t, axis, indices)` (and `numpy.take` for in-bounds
 * indices); negative indices are not wrapped and raise an {@link IndexError}.
 *
 * @param t - Input tensor (host tensor of any dtype, including string)
 * @param indices - 1D tensor of integer indices into `axis`
 * @param axis - Axis along which to gather (negative counts from the end)
 * @returns Gathered tensor (copy) with the dtype of `t`
 * @throws {DTypeError} If `indices` is a string tensor
 * @throws {InvalidParameterError} If `indices` is not 1D or holds non-integers
 * @throws {IndexError} If an index is outside `[0, size of axis)`
 *
 * @example
 * ```ts
 * const t = tensor([[1, 2], [3, 4], [5, 6]]);
 * const indices = tensor([0, 2]);
 * const result = gather(t, indices, 0);  // [[1, 2], [5, 6]]
 * ```
 */
export function gather(t: Tensor, indices: Tensor, axis: Axis): Tensor {
  const ax = normalizeAxis(axis, t.ndim);

  if (indices.dtype === "string") {
    throw new DTypeError("gather() requires numeric indices tensor");
  }
  if (indices.ndim !== 1) {
    throw new InvalidParameterError(
      "gather() requires a 1D indices tensor",
      "indices",
      indices.shape
    );
  }

  const n = indices.size;
  const axisSize = t.shape[ax] ?? 1;

  // Read and validate every index once, before touching the data.
  const idx = new Float64Array(n);
  const idxData = indices.data;
  if (Array.isArray(idxData)) {
    throw new DTypeError("gather() requires numeric indices tensor");
  }
  const idxStride = indices.strides[0] ?? 1;
  for (let k = 0; k < n; k++) {
    const pos = indices.offset + k * idxStride;
    let idxVal: number;
    if (idxData instanceof BigInt64Array) {
      const value = getBigIntElement(idxData, pos);
      idxVal = Number(value);
      if (!Number.isSafeInteger(idxVal)) {
        throw new InvalidParameterError(
          `gather() index ${value} exceeds safe integer range`,
          "indices",
          value
        );
      }
    } else {
      idxVal = getNumericElement(idxData, pos);
    }
    if (!Number.isInteger(idxVal)) {
      throw new InvalidParameterError(
        `gather() index ${idxVal} is not an integer`,
        "indices",
        idxVal
      );
    }
    if (idxVal < 0 || idxVal >= axisSize) {
      throw new IndexError(
        `index ${idxVal} is out of bounds for axis ${axis} with size ${axisSize}`
      );
    }
    idx[k] = idxVal;
  }

  const outShape = [...t.shape];
  outShape[ax] = n;
  const outSize = shapeToSize(outShape);

  // View the input as (outer dims) x (gathered axis) x (inner dims) and
  // precompute the storage offsets of the outer and inner coordinates.
  const outerOffsets = enumerateOffsets(t.shape.slice(0, ax), t.strides.slice(0, ax));
  const innerOffsets = enumerateOffsets(t.shape.slice(ax + 1), t.strides.slice(ax + 1));
  const axisStride = t.strides[ax] ?? 0;

  const tData = t.data;
  if (t.dtype === "string") {
    if (!Array.isArray(tData)) {
      throw new DeepboxError("Internal error: string tensor has non-array data");
    }
    const out = new Array<string>(outSize).fill("");
    gatherCopy(tData, out, t.offset, outerOffsets, innerOffsets, idx, axisStride);
    return Tensor.fromStringArray({ data: out, shape: outShape, device: t.device });
  }

  if (Array.isArray(tData)) {
    throw new DeepboxError("Internal error: numeric tensor has array data");
  }
  const Ctor = dtypeToTypedArrayCtor(t.dtype);
  const out = new Ctor(outSize);
  if (tData instanceof BigInt64Array && out instanceof BigInt64Array) {
    gatherCopy(tData, out, t.offset, outerOffsets, innerOffsets, idx, axisStride);
  } else if (!(tData instanceof BigInt64Array) && !(out instanceof BigInt64Array)) {
    gatherCopy(tData, out, t.offset, outerOffsets, innerOffsets, idx, axisStride);
  } else {
    throw new DeepboxError("Internal error: gather source and output storage differ");
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: t.dtype,
    device: t.device,
  });
}
