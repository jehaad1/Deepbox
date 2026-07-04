/**
 * Tensor insert and delete operations.
 *
 * - insert: Insert values into a tensor along an axis
 * - delete_: Remove elements from a tensor along an axis
 *
 * @module ndarray/ops/insert_delete
 */

import type { Axis, DType, TypedArray } from "../../core";
import {
  DeepboxError,
  dtypeToTypedArrayCtor,
  getBigIntElement,
  getNumericElement,
  InvalidParameterError,
  normalizeAxis,
  shapeToSize,
} from "../../core";
import { computeStrides, Tensor } from "../tensor/Tensor";
import { readNumericContiguous } from "./_internal";

/**
 * Insert values into a tensor along an axis before given indices.
 *
 * If `axis` is undefined the input is flattened first.
 *
 * **Complexity**: O(n + m) where n is input size and m is values size
 *
 * @param t - Input tensor
 * @param indices - Index or array of indices before which values are inserted
 * @param values - Values to insert (scalar number or Tensor)
 * @param axis - Axis along which to insert. If undefined, tensor is flattened.
 * @returns New tensor with values inserted
 *
 * @example
 * ```ts
 * const a = tensor([1, 2, 3, 4]);
 * insert(a, 2, 99);          // [1, 2, 99, 3, 4]
 * insert(a, [1, 3], 99);     // [1, 99, 2, 3, 99, 4]
 *
 * const b = tensor([[1, 2], [3, 4]]);
 * insert(b, 1, tensor([5, 6]), 0);  // [[1, 2], [5, 6], [3, 4]]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function insert(
  t: Tensor,
  indices: number | number[],
  values: number | Tensor,
  axis?: Axis
): Tensor {
  // If no axis, flatten first
  if (axis === undefined) {
    const flat = t.flatten();
    return insert(flat, indices, values, 0);
  }

  const ax = normalizeAxis(axis, t.ndim);
  const axisSize = t.shape[ax] ?? 0;

  // Normalize indices to array
  const idxArr = typeof indices === "number" ? [indices] : [...indices];

  // Validate and normalize indices (negative indices count from the end,
  // matching NumPy and this module's delete())
  for (let i = 0; i < idxArr.length; i++) {
    let idx = idxArr[i];
    if (idx === undefined || !Number.isInteger(idx)) {
      throw new InvalidParameterError(
        `insert index must be an integer; received ${String(idx)}`,
        "indices",
        indices
      );
    }
    if (idx < 0) idx += axisSize;
    if (idx < 0 || idx > axisSize) {
      throw new InvalidParameterError(
        `insert index ${idxArr[i]} is out of bounds for axis size ${axisSize}`,
        "indices",
        indices
      );
    }
    idxArr[i] = idx;
  }

  // Sort indices stably while keeping each index paired with the position of
  // its value in the caller-supplied order (NumPy pairs values with the
  // indices as given, not with the sorted indices).
  const order = idxArr.map((idx, valuePos) => ({ idx, valuePos }));
  order.sort((a, b) => a.idx - b.idx);
  const nInsert = order.length;

  // Build output shape
  const outShape = [...t.shape];
  outShape[ax] = axisSize + nInsert;

  // Allocate output
  const outSize = shapeToSize(outShape);
  const isString = t.dtype === "string";
  const outData = isString
    ? new Array<string>(outSize)
    : new (dtypeToTypedArrayCtor(t.dtype))(outSize);

  const outStrides = computeStrides(outShape);

  // Prepare source/dest buffers
  let stringOut: string[] | undefined;
  let bigIntOut: BigInt64Array | undefined;
  let numericOut: Exclude<TypedArray, BigInt64Array> | undefined;

  if (Array.isArray(outData)) {
    stringOut = outData;
  } else if (outData instanceof BigInt64Array) {
    bigIntOut = outData;
  } else {
    numericOut = outData;
  }

  let stringSrc: readonly string[] | undefined;
  let numericSrc: Exclude<TypedArray, BigInt64Array> | undefined;
  if (Array.isArray(t.data)) {
    stringSrc = t.data;
  } else if (!(t.data instanceof BigInt64Array)) {
    numericSrc = t.data;
  }

  // Build mapping: output axis position → {source: srcIdx} or {inserted: valueIdx}
  const outAxisSize = axisSize + nInsert;
  const isInserted = new Array<boolean>(outAxisSize).fill(false);
  const insertionValueIdx = new Array<number>(outAxisSize).fill(-1);

  // Mark insertion positions (shift by number of prior insertions)
  for (let i = 0; i < nInsert; i++) {
    const entry = order[i];
    const idx = (entry?.idx ?? 0) + i;
    isInserted[idx] = true;
    insertionValueIdx[idx] = entry?.valuePos ?? 0;
  }

  // Build source axis mapping for non-inserted positions
  const sourceAxisIdx = new Array<number>(outAxisSize).fill(-1);
  let srcPos = 0;
  for (let outPos = 0; outPos < outAxisSize; outPos++) {
    if (!isInserted[outPos]) {
      sourceAxisIdx[outPos] = srcPos;
      srcPos++;
    }
  }

  // Get insertion values
  const valTensor = typeof values === "number" ? null : values;
  const valScalar = typeof values === "number" ? values : NaN;

  // Fast path: 1-D numeric insert (the axis === undefined case always lands
  // here after flatten) — copy the segments between insertion points with
  // subarray/set instead of decoding coordinates per element.
  if (t.ndim === 1 && numericOut) {
    const src = readNumericContiguous(t);
    const valSrc = valTensor === null ? null : readNumericContiguous(valTensor);
    if (src && (valTensor === null || (valSrc && valTensor.ndim <= 1))) {
      let srcPos = 0;
      let outPos = 0;
      for (let i = 0; i < nInsert; i++) {
        const entry = order[i] as { idx: number; valuePos: number };
        numericOut.set(src.subarray(srcPos, entry.idx), outPos);
        outPos += entry.idx - srcPos;
        srcPos = entry.idx;
        if (valSrc) {
          // 1-D values pair with the caller-supplied insertion order;
          // size-1 values broadcast (same rule as the generic path).
          numericOut[outPos++] = valSrc[valSrc.length === 1 ? 0 : entry.valuePos] as number;
        } else {
          numericOut[outPos++] = valScalar;
        }
      }
      numericOut.set(src.subarray(srcPos), outPos);
      return Tensor.fromTypedArray({
        data: numericOut,
        shape: outShape,
        dtype: t.dtype as Exclude<DType, "string">,
        device: t.device,
      });
    }
  }

  // Fill output
  for (let flatIdx = 0; flatIdx < outSize; flatIdx++) {
    // Convert flat index to coordinates in output
    let rem = flatIdx;
    const outCoords = new Array<number>(t.ndim);
    for (let d = 0; d < t.ndim; d++) {
      const stride = outStrides[d] ?? 1;
      outCoords[d] = Math.floor(rem / stride);
      rem -= (outCoords[d] ?? 0) * stride;
    }

    const outAxisCoord = outCoords[ax] ?? 0;

    if (isInserted[outAxisCoord]) {
      // Inserted position — read from values
      if (valTensor) {
        const vIdx = insertionValueIdx[outAxisCoord] ?? 0;
        let valOffset: number;
        if (valTensor.ndim === 0) {
          valOffset = valTensor.offset;
        } else if (valTensor.ndim === 1 && t.ndim === 1) {
          // 1D input, 1D values: index by insertion index
          const vCoord = valTensor.size === 1 ? 0 : vIdx;
          valOffset = valTensor.offset + vCoord * (valTensor.strides[0] ?? 1);
        } else if (valTensor.ndim === 1) {
          // 1D values inserted into nD tensor: index by the trailing non-axis coord
          // Find the first non-axis dimension to use as the value index
          let coordForVal = 0;
          for (let d = 0; d < t.ndim; d++) {
            if (d !== ax) {
              coordForVal = outCoords[d] ?? 0;
              break;
            }
          }
          const vCoord = valTensor.size === 1 ? 0 : coordForVal;
          valOffset = valTensor.offset + vCoord * (valTensor.strides[0] ?? 1);
        } else {
          valOffset = valTensor.offset;
          let vd = 0;
          for (let d = 0; d < t.ndim; d++) {
            if (d === ax) {
              const vCoord = (valTensor.shape[vd] ?? 1) === 1 ? 0 : vIdx;
              valOffset += vCoord * (valTensor.strides[vd] ?? 0);
              vd++;
            } else if (vd < valTensor.ndim) {
              valOffset += (outCoords[d] ?? 0) * (valTensor.strides[vd] ?? 0);
              vd++;
            }
          }
        }
        if (stringOut && Array.isArray(valTensor.data)) {
          stringOut[flatIdx] = valTensor.data[valOffset] ?? "";
        } else if (bigIntOut && valTensor.data instanceof BigInt64Array) {
          bigIntOut[flatIdx] = getBigIntElement(valTensor.data, valOffset);
        } else if (
          numericOut &&
          !Array.isArray(valTensor.data) &&
          !(valTensor.data instanceof BigInt64Array)
        ) {
          numericOut[flatIdx] = getNumericElement(valTensor.data, valOffset);
        }
      } else {
        // Scalar value
        if (numericOut) {
          numericOut[flatIdx] = valScalar;
        } else if (bigIntOut) {
          bigIntOut[flatIdx] = BigInt(Math.trunc(valScalar));
        }
      }
    } else {
      // Source position — read from input tensor
      const inCoords = [...outCoords];
      inCoords[ax] = sourceAxisIdx[outAxisCoord] ?? 0;

      let srcOffset = t.offset;
      for (let d = 0; d < t.ndim; d++) {
        srcOffset += (inCoords[d] ?? 0) * (t.strides[d] ?? 0);
      }

      if (stringOut && stringSrc) {
        stringOut[flatIdx] = stringSrc[srcOffset] ?? "";
      } else if (bigIntOut && t.data instanceof BigInt64Array) {
        bigIntOut[flatIdx] = getBigIntElement(t.data, srcOffset);
      } else if (numericOut && numericSrc) {
        numericOut[flatIdx] = getNumericElement(numericSrc, srcOffset);
      }
    }
  }

  if (Array.isArray(outData)) {
    return Tensor.fromStringArray({
      data: outData,
      shape: outShape,
      device: t.device,
    });
  }

  if (t.dtype === "string") {
    throw new DeepboxError("Internal error: string dtype but non-array data");
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: outShape,
    dtype: t.dtype,
    device: t.device,
  });
}

/**
 * Delete elements from a tensor along an axis.
 *
 * If `axis` is undefined the input is flattened first.
 * Uses trailing underscore to avoid collision with the JS `delete` keyword.
 *
 * **Complexity**: O(n) where n is input size
 *
 * @param t - Input tensor
 * @param indices - Index or array of indices of elements to delete
 * @param axis - Axis along which to delete. If undefined, tensor is flattened.
 * @returns New tensor with specified elements removed
 *
 * @example
 * ```ts
 * const a = tensor([1, 2, 3, 4, 5]);
 * delete_(a, 2);           // [1, 2, 4, 5]
 * delete_(a, [0, 3]);      // [2, 3, 5]
 *
 * const b = tensor([[1, 2], [3, 4], [5, 6]]);
 * delete_(b, 1, 0);        // [[1, 2], [5, 6]]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function delete_(t: Tensor, indices: number | number[], axis?: Axis): Tensor {
  // If no axis, flatten first
  if (axis === undefined) {
    const flat = t.flatten();
    return delete_(flat, indices, 0);
  }

  const ax = normalizeAxis(axis, t.ndim);
  const axisSize = t.shape[ax] ?? 0;

  // Normalize indices to sorted unique set
  const rawIndices = typeof indices === "number" ? [indices] : [...indices];
  const deleteSet = new Set<number>();
  for (const idx of rawIndices) {
    const normalized = idx < 0 ? idx + axisSize : idx;
    if (!Number.isInteger(normalized) || normalized < 0 || normalized >= axisSize) {
      throw new InvalidParameterError(
        `delete index ${idx} is out of bounds for axis size ${axisSize}`,
        "indices",
        indices
      );
    }
    deleteSet.add(normalized);
  }

  const nDelete = deleteSet.size;
  const newAxisSize = axisSize - nDelete;

  // Build output shape
  const outShape = [...t.shape];
  outShape[ax] = newAxisSize;

  // Allocate output
  const outSize = shapeToSize(outShape);
  const isString = t.dtype === "string";
  const outData = isString
    ? new Array<string>(outSize)
    : new (dtypeToTypedArrayCtor(t.dtype))(outSize);

  const outStrides = computeStrides(outShape);

  // Prepare buffers
  let stringOut: string[] | undefined;
  let bigIntOut: BigInt64Array | undefined;
  let numericOut: Exclude<TypedArray, BigInt64Array> | undefined;

  if (Array.isArray(outData)) {
    stringOut = outData;
  } else if (outData instanceof BigInt64Array) {
    bigIntOut = outData;
  } else {
    numericOut = outData;
  }

  let stringSrc: readonly string[] | undefined;
  let numericSrc: Exclude<TypedArray, BigInt64Array> | undefined;
  if (Array.isArray(t.data)) {
    stringSrc = t.data;
  } else if (!(t.data instanceof BigInt64Array)) {
    numericSrc = t.data;
  }

  // Build mapping from output axis position → source axis position (skipping deleted)
  const outToSrcAxis = new Array<number>(newAxisSize);
  let outIdx = 0;
  for (let s = 0; s < axisSize; s++) {
    if (!deleteSet.has(s)) {
      outToSrcAxis[outIdx] = s;
      outIdx++;
    }
  }

  // Fast path: numeric data — each kept axis position is a contiguous slab
  // of `inner` elements in the dense layout; copy slabs instead of decoding
  // coordinates per element.
  const denseSrc = numericOut ? readNumericContiguous(t) : null;
  if (denseSrc && numericOut && outSize > 0) {
    let inner = 1;
    for (let d = ax + 1; d < t.ndim; d++) inner *= t.shape[d] ?? 0;
    const slab = axisSize * inner;
    const outer = inner === 0 ? 0 : outSize / (newAxisSize * inner);
    let pos = 0;
    for (let o = 0; o < outer; o++) {
      const srcBase = o * slab;
      for (let j = 0; j < newAxisSize; j++) {
        const start = srcBase + (outToSrcAxis[j] as number) * inner;
        if (inner >= 16) {
          numericOut.set(denseSrc.subarray(start, start + inner), pos);
          pos += inner;
        } else {
          for (let k = 0; k < inner; k++) numericOut[pos++] = denseSrc[start + k] as number;
        }
      }
    }
    return Tensor.fromTypedArray({
      data: numericOut,
      shape: outShape,
      dtype: t.dtype as Exclude<DType, "string">,
      device: t.device,
    });
  }

  // Fill output
  for (let flatIdx = 0; flatIdx < outSize; flatIdx++) {
    // Convert flat index to coordinates in output
    let rem = flatIdx;
    const outCoords = new Array<number>(t.ndim);
    for (let d = 0; d < t.ndim; d++) {
      const stride = outStrides[d] ?? 1;
      outCoords[d] = Math.floor(rem / stride);
      rem -= (outCoords[d] ?? 0) * stride;
    }

    // Map output axis coordinate back to source axis coordinate
    const outAxisCoord = outCoords[ax] ?? 0;
    const srcAxisCoord = outToSrcAxis[outAxisCoord] ?? 0;

    // Build source coordinates
    const inCoords = [...outCoords];
    inCoords[ax] = srcAxisCoord;

    let srcOffset = t.offset;
    for (let d = 0; d < t.ndim; d++) {
      srcOffset += (inCoords[d] ?? 0) * (t.strides[d] ?? 0);
    }

    if (stringOut && stringSrc) {
      stringOut[flatIdx] = stringSrc[srcOffset] ?? "";
    } else if (bigIntOut && t.data instanceof BigInt64Array) {
      bigIntOut[flatIdx] = getBigIntElement(t.data, srcOffset);
    } else if (numericOut && numericSrc) {
      numericOut[flatIdx] = getNumericElement(numericSrc, srcOffset);
    }
  }

  if (Array.isArray(outData)) {
    return Tensor.fromStringArray({
      data: outData,
      shape: outShape,
      device: t.device,
    });
  }

  if (t.dtype === "string") {
    throw new DeepboxError("Internal error: string dtype but non-array data");
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: outShape,
    dtype: t.dtype,
    device: t.device,
  });
}
