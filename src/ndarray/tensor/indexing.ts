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
import { offsetFromFlatIndex } from "./strides";
import { computeStrides, Tensor } from "./Tensor";

export type { SliceRange } from "./slice_helpers";

/**
 * Slice a tensor.
 *
 * Examples:
 * - `slice(t, { start: 0, end: 2 })` on a 1D tensor keeps the first 2 elements.
 * - `slice(t, 0, { start: 1 })` on a 2D tensor selects row 0 and columns from 1.
 */
export function slice(t: Tensor, ...ranges: SliceRange[]): Tensor {
  // Same semantics as the method form; Tensor.slice has the optimized
  // row-blocked copy path.
  return t.slice(...ranges);
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
