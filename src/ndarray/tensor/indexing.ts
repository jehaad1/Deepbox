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
  ShapeError,
  shapeToSize,
} from "../../core";
import { normalizeRange, type SliceRange } from "./sliceHelpers";
import { offsetFromFlatIndex } from "./strides";
import { computeStrides, Tensor } from "./Tensor";

export type { SliceRange } from "./sliceHelpers";

/**
 * Slice a tensor.
 *
 * Examples:
 * - `slice(t, { start: 0, end: 2 })` on a 1D tensor keeps the first 2 elements.
 * - `slice(t, 0, { start: 1 })` on a 2D tensor selects row 0 and columns from 1.
 */
export function slice(t: Tensor, ...ranges: SliceRange[]): Tensor {
  const ndim = t.ndim;

  if (ranges.length > ndim) {
    throw new ShapeError(`Too many indices for tensor: got ${ranges.length}, expected <= ${ndim}`);
  }

  const normalized = new Array<{ start: number; end: number; step: number }>(ndim);
  const outShape: number[] = [];

  for (let axis = 0; axis < ndim; axis++) {
    const dim = t.shape[axis] ?? 0;
    const range = ranges[axis] ?? { start: 0, end: dim, step: 1 };
    const nr = normalizeRange(range, dim);
    normalized[axis] = nr;

    // If the user passed a number, that dimension is squeezed out.
    if (typeof range !== "number") {
      const len =
        nr.step > 0
          ? Math.max(0, Math.ceil((nr.end - nr.start) / nr.step))
          : Math.max(0, Math.ceil((nr.start - nr.end) / -nr.step));
      outShape.push(len);
    }
  }

  // If all axes were indexed by numbers, the result is a scalar (0D) tensor.
  const outSize = outShape.length === 0 ? 1 : outShape.reduce((a, b) => a * b, 1);
  const out =
    t.dtype === "string"
      ? new Array<string>(outSize)
      : new (dtypeToTypedArrayCtor(t.dtype))(outSize);

  // Iterate over output indices and map back to input indices.
  const outStrides = new Array<number>(outShape.length);
  let stride = 1;
  for (let i = outShape.length - 1; i >= 0; i--) {
    outStrides[i] = stride;
    stride *= outShape[i] ?? 0;
  }

  const outNdim = outShape.length;

  for (let outFlat = 0; outFlat < outSize; outFlat++) {
    // Convert flat to multi-index.
    let rem = outFlat;
    const outIdx = new Array<number>(outNdim);
    for (let i = 0; i < outNdim; i++) {
      const s = outStrides[i] ?? 1;
      outIdx[i] = Math.floor(rem / s);
      rem %= s;
    }

    // Map to input multi-index.
    const inIdx = new Array<number>(ndim);
    let outAxis = 0;
    for (let axis = 0; axis < ndim; axis++) {
      const r = ranges[axis];
      const nr = normalized[axis];
      if (nr === undefined) {
        throw new DeepboxError("Internal error: missing normalized slice range");
      }

      if (typeof r === "number") {
        inIdx[axis] = nr.start;
      } else {
        inIdx[axis] = nr.start + (outIdx[outAxis] ?? 0) * nr.step;
        outAxis++;
      }
    }

    // Compute input flat offset.
    let inFlat = t.offset;
    for (let axis = 0; axis < ndim; axis++) {
      inFlat += (inIdx[axis] ?? 0) * (t.strides[axis] ?? 0);
    }

    if (Array.isArray(out) && Array.isArray(t.data)) {
      out[outFlat] = t.data[inFlat] ?? "";
    } else if (out instanceof BigInt64Array && t.data instanceof BigInt64Array) {
      out[outFlat] = getBigIntElement(t.data, inFlat);
    } else if (
      !Array.isArray(out) &&
      !(out instanceof BigInt64Array) &&
      !Array.isArray(t.data) &&
      !(t.data instanceof BigInt64Array)
    ) {
      out[outFlat] = getNumericElement(t.data, inFlat);
    }
  }

  if (Array.isArray(out)) {
    return Tensor.fromStringArray({
      data: out,
      shape: outShape.length === 0 ? [] : outShape,
      device: t.device,
    });
  }

  if (t.dtype === "string") {
    throw new DeepboxError("Internal error: string dtype but non-array data");
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: outShape.length === 0 ? [] : outShape,
    dtype: t.dtype,
    device: t.device,
  });
}

/**
 * Gather values along an axis specified by indices.
 *
 * @param t - Input tensor
 * @param indices - Indices to gather
 * @param axis - Axis along which to gather
 * @returns Gathered tensor
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

  const outShape = [...t.shape];
  outShape[ax] = indices.size;

  const outSize = shapeToSize(outShape);
  const axisSize = t.shape[ax] ?? 1;

  const outStrides = new Array<number>(outShape.length);
  let stride = 1;
  for (let i = outShape.length - 1; i >= 0; i--) {
    outStrides[i] = stride;
    stride *= outShape[i] ?? 1;
  }

  const indicesLogicalStrides = computeStrides(indices.shape);

  const readIndexValue = (flat: number): number => {
    const idxOffset = offsetFromFlatIndex(
      flat,
      indicesLogicalStrides,
      indices.strides,
      indices.offset
    );
    if (indices.data instanceof BigInt64Array) {
      const value = getBigIntElement(indices.data, idxOffset);
      const num = Number(value);
      if (!Number.isSafeInteger(num)) {
        throw new InvalidParameterError(
          `gather() index ${value} exceeds safe integer range`,
          "indices",
          value
        );
      }
      return num;
    }
    const numericData = indices.data;
    if (Array.isArray(numericData)) {
      throw new DTypeError("gather() requires numeric indices tensor");
    }
    return getNumericElement(numericData, idxOffset);
  };

  if (t.dtype === "string") {
    const out = new Array<string>(outSize);
    const tData = t.data;
    if (!Array.isArray(tData)) {
      throw new DeepboxError("Internal error: string tensor has non-array data");
    }

    for (let outFlat = 0; outFlat < outSize; outFlat++) {
      let rem = outFlat;
      const outIdx = new Array<number>(outShape.length);
      for (let i = 0; i < outShape.length; i++) {
        const s = outStrides[i] ?? 1;
        outIdx[i] = Math.floor(rem / s);
        rem %= s;
      }

      const idxVal = readIndexValue(outIdx[ax] ?? 0);
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

      const inIdx = outIdx.slice();
      inIdx[ax] = idxVal;

      let inOffset = t.offset;
      for (let i = 0; i < t.ndim; i++) {
        inOffset += (inIdx[i] ?? 0) * (t.strides[i] ?? 0);
      }
      out[outFlat] = tData[inOffset] ?? "";
    }

    return Tensor.fromStringArray({
      data: out,
      shape: outShape,
      device: t.device,
    });
  }

  const Ctor = dtypeToTypedArrayCtor(t.dtype);
  const out = new Ctor(outSize);
  const tData = t.data;
  if (Array.isArray(tData)) {
    throw new DeepboxError("Internal error: numeric tensor has array data");
  }

  for (let outFlat = 0; outFlat < outSize; outFlat++) {
    let rem = outFlat;
    const outIdx = new Array<number>(outShape.length);
    for (let i = 0; i < outShape.length; i++) {
      const s = outStrides[i] ?? 1;
      outIdx[i] = Math.floor(rem / s);
      rem %= s;
    }

    const idxVal = readIndexValue(outIdx[ax] ?? 0);
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

    const inIdx = outIdx.slice();
    inIdx[ax] = idxVal;
    let inOffset = t.offset;
    for (let i = 0; i < t.ndim; i++) {
      inOffset += (inIdx[i] ?? 0) * (t.strides[i] ?? 0);
    }
    if (t.data instanceof BigInt64Array) {
      if (out instanceof BigInt64Array) {
        out[outFlat] = getBigIntElement(t.data, inOffset);
      }
    } else {
      if (
        !Array.isArray(out) &&
        !(out instanceof BigInt64Array) &&
        !Array.isArray(tData) &&
        !(tData instanceof BigInt64Array)
      ) {
        out[outFlat] = getNumericElement(tData, inOffset);
      }
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: t.dtype,
    device: t.device,
  });
}
