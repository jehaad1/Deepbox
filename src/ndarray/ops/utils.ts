/**
 * Tensor utility operations.
 *
 * This module provides utility functions for tensor creation and manipulation:
 * - where: Conditional element selection
 * - zeros_like / ones_like / empty_like / full_like: Create tensors matching shape
 * - clone / detach / contiguous: Tensor copying and memory layout
 * - diag / diagonal: Diagonal extraction and construction
 * - triu / tril: Triangular extraction
 * - flip / fliplr / flipud: Tensor reversal
 * - roll: Circular shift
 * - unique: Return unique elements
 * - searchsorted: Binary search in sorted array
 * - pad: Pad tensor with values
 * - moveaxis / swapaxes: Rearrange axes
 * - broadcast_to: Explicit broadcasting
 * - scatter: Scatter values at indices
 * - copy: Explicit deep copy
 * - booleanIndex / fancyIndex: Advanced indexing
 * - nansum / nanmean / nanstd / nanmin / nanmax: NaN-aware reductions
 * - histogram / bincount: Binning operations
 *
 * All operations maintain type safety and proper error handling.
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import type { Axis, DType, Shape, TypedArray } from "../../core";
import {
  DeepboxError,
  DTypeError,
  dtypeToTypedArrayCtor,
  getBigIntElement,
  getNumericElement,
  InvalidParameterError,
  normalizeAxis,
  ShapeError,
  shapeToSize,
} from "../../core";
import { transpose } from "../tensor/shape";
import { isContiguous } from "../tensor/strides";
import { computeStrides, isBigIntArray, Tensor } from "../tensor/Tensor";
import { flatOffset, readAsNumber, readNumericContiguous, requireNumericData } from "./_internal";
import { dispatchTernary } from "./device_dispatch";

// ─── Internal type-safe helpers ─────────────────────────────────────────────

/**
 * Assert dtype is not "string" and return a narrowed type.
 * This avoids using `as` casts when constructing Tensor.fromTypedArray.
 */
function ensureNonStringDType(dtype: DType): Exclude<DType, "string"> {
  if (dtype === "string") {
    throw new DTypeError("Expected numeric dtype, got string");
  }
  return dtype;
}

/**
 * Write a single element from `src[srcIdx]` into `out[idx]`,
 * handling BigInt64Array vs numeric typed arrays without `as` casts.
 */
function writeElement(out: TypedArray, idx: number, src: TypedArray, srcIdx: number): void {
  if (out instanceof BigInt64Array) {
    if (src instanceof BigInt64Array) {
      out[idx] = getBigIntElement(src, srcIdx);
    }
  } else if (!(src instanceof BigInt64Array)) {
    out[idx] = getNumericElement(src, srcIdx);
  }
}

/**
 * Fill a TypedArray with a numeric value, handling BigInt64Array correctly.
 */
function fillTypedArray(arr: TypedArray, value: number): void {
  if (arr instanceof BigInt64Array) {
    arr.fill(BigInt(value));
  } else {
    arr.fill(value);
  }
}

/**
 * Materialize a contiguous flat buffer from any tensor (numeric only).
 */
function materializeNumeric(t: Tensor): TypedArray {
  const data = requireNumericData(t.data, "materializeNumeric");
  const size = t.size;
  if (size === 0) {
    return new (dtypeToTypedArrayCtor(t.dtype))(0);
  }
  const contig = isContiguous(t.shape, t.strides);
  if (contig && t.offset === 0 && data.length === size) {
    return data.slice();
  }
  const logicalStrides = computeStrides(t.shape);
  if (isBigIntArray(data)) {
    const out = new BigInt64Array(size);
    for (let i = 0; i < size; i++) {
      const off = flatOffset(i, t.offset, contig, logicalStrides, t.strides);
      out[i] = getBigIntElement(data, off);
    }
    return out;
  }
  const Ctor = dtypeToTypedArrayCtor(t.dtype);
  const out = new Ctor(size);
  if (out instanceof BigInt64Array) {
    return out;
  }
  for (let i = 0; i < size; i++) {
    const off = flatOffset(i, t.offset, contig, logicalStrides, t.strides);
    out[i] = getNumericElement(data, off);
  }
  return out;
}

// ─── Broadcasting helpers ───────────────────────────────────────────────────

function broadcastShapeTwo(a: Shape, b: Shape): number[] {
  const maxLen = Math.max(a.length, b.length);
  const result: number[] = [];
  for (let i = 0; i < maxLen; i++) {
    const dimA = i < a.length ? (a[a.length - 1 - i] ?? 1) : 1;
    const dimB = i < b.length ? (b[b.length - 1 - i] ?? 1) : 1;
    if (dimA === dimB) {
      result.unshift(dimA);
    } else if (dimA === 1) {
      result.unshift(dimB);
    } else if (dimB === 1) {
      result.unshift(dimA);
    } else {
      throw ShapeError.mismatch(a, b, "broadcast");
    }
  }
  return result;
}

function broadcastShapeThree(a: Shape, b: Shape, c: Shape): number[] {
  return broadcastShapeTwo(broadcastShapeTwo(a, b), c);
}

function flatToCoords(flat: number, strides: readonly number[], ndim: number): number[] {
  const coords = new Array<number>(ndim);
  let rem = flat;
  for (let d = 0; d < ndim; d++) {
    const s = strides[d] ?? 1;
    coords[d] = Math.floor(rem / s);
    rem -= (coords[d] ?? 0) * s;
  }
  return coords;
}

function broadcastOffset(
  coords: number[],
  shape: Shape,
  strides: readonly number[],
  offset: number,
  outNdim: number
): number {
  let off = offset;
  const rankDiff = outNdim - shape.length;
  for (let d = 0; d < shape.length; d++) {
    const dim = shape[d] ?? 1;
    const coord = coords[d + rankDiff] ?? 0;
    off += (dim === 1 ? 0 : coord) * (strides[d] ?? 0);
  }
  return off;
}

// ─── where ──────────────────────────────────────────────────────────────────

/**
 * Conditional element selection: returns elements from `x` where `condition`
 * is true, and from `y` where `condition` is false.
 *
 * All three tensors must be broadcastable to a common shape.
 * The condition tensor is evaluated as boolean (0 = false, nonzero = true).
 *
 * **Complexity**: O(n) where n is the number of elements in the output
 *
 * @param condition - Boolean condition tensor
 * @param x - Values where condition is true
 * @param y - Values where condition is false
 * @returns Tensor with selected elements
 *
 * @example
 * ```ts
 * const cond = tensor([1, 0, 1, 0], { dtype: 'bool' });
 * const x = tensor([10, 20, 30, 40]);
 * const y = tensor([1, 2, 3, 4]);
 * const result = where(cond, x, y); // [10, 2, 30, 4]
 * ```
 */
export function where(condition: Tensor, x: Tensor, y: Tensor): Tensor {
  if (condition.dtype === "string") {
    throw new DTypeError("where condition must be a numeric tensor, not string");
  }
  if (x.dtype === "string" || y.dtype === "string") {
    throw new DTypeError("where does not support string dtype for x or y");
  }
  if (x.dtype !== y.dtype) {
    throw new DTypeError(
      `where requires x and y to have the same dtype; got ${x.dtype} and ${y.dtype}`
    );
  }

  if (condition.device !== "cpu" || x.device !== "cpu" || y.device !== "cpu") {
    const onDevice = dispatchTernary("where", condition, x, y);
    if (onDevice) return onDevice;
  }

  const outShape = broadcastShapeThree(condition.shape, x.shape, y.shape);
  const outSize = shapeToSize(outShape);
  const dtype = ensureNonStringDType(x.dtype);
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const outData = new Ctor(outSize);
  const outStrides = computeStrides(outShape);

  const condData = requireNumericData(condition.data, "where");
  const xData = requireNumericData(x.data, "where");
  const yData = requireNumericData(y.data, "where");

  // Fast path: no broadcasting — all three inputs already share the output
  // shape and are contiguous. Index the buffers directly instead of decoding
  // coordinates and computing three broadcast offsets per element.
  const nd = outShape.length;
  const sameShape = (s: Shape): boolean => {
    if (s.length !== nd) return false;
    for (let d = 0; d < nd; d++) if (s[d] !== outShape[d]) return false;
    return true;
  };
  if (
    sameShape(condition.shape) &&
    sameShape(x.shape) &&
    sameShape(y.shape) &&
    isContiguous(condition.shape, condition.strides) &&
    isContiguous(x.shape, x.strides) &&
    isContiguous(y.shape, y.strides)
  ) {
    const co = condition.offset;
    const xo = x.offset;
    const yo = y.offset;
    if (
      condData instanceof Float64Array &&
      xData instanceof Float64Array &&
      yData instanceof Float64Array &&
      outData instanceof Float64Array
    ) {
      const cd = condData;
      const xf = xData;
      const yf = yData;
      const od = outData;
      for (let i = 0; i < outSize; i++) {
        od[i] = (cd[co + i] as number) !== 0 ? (xf[xo + i] as number) : (yf[yo + i] as number);
      }
    } else {
      for (let i = 0; i < outSize; i++) {
        if (readAsNumber(condData, co + i) !== 0) {
          writeElement(outData, i, xData, xo + i);
        } else {
          writeElement(outData, i, yData, yo + i);
        }
      }
    }
    return Tensor.fromTypedArray({
      data: outData,
      shape: outShape,
      dtype,
      device: x.device,
    });
  }

  for (let i = 0; i < outSize; i++) {
    const coords = flatToCoords(i, outStrides, outShape.length);
    const condOff = broadcastOffset(
      coords,
      condition.shape,
      condition.strides,
      condition.offset,
      outShape.length
    );
    const xOff = broadcastOffset(coords, x.shape, x.strides, x.offset, outShape.length);
    const yOff = broadcastOffset(coords, y.shape, y.strides, y.offset, outShape.length);

    const condVal = readAsNumber(condData, condOff);
    const srcOff = condVal !== 0 ? xOff : yOff;
    const srcData = condVal !== 0 ? xData : yData;
    writeElement(outData, i, srcData, srcOff);
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: outShape,
    dtype,
    device: x.device,
  });
}

// ─── zeros_like / ones_like / empty_like / full_like ────────────────────────

/**
 * Create a tensor of zeros with the same shape and dtype as the input tensor.
 *
 * @param t - Reference tensor
 * @returns New tensor filled with zeros
 *
 * @example
 * ```ts
 * const a = tensor([[1, 2], [3, 4]]);
 * const z = zeros_like(a); // [[0, 0], [0, 0]]
 * ```
 * @deprecated Prefer {@link zerosLike}.
 */
export function zeros_like(t: Tensor): Tensor {
  if (t.dtype === "string") {
    const data = new Array<string>(t.size).fill("");
    return Tensor.fromStringArray({
      data,
      shape: [...t.shape],
      device: t.device,
    });
  }
  const dtype = ensureNonStringDType(t.dtype);
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const data = new Ctor(t.size);
  return Tensor.fromTypedArray({
    data,
    shape: [...t.shape],
    dtype,
    device: t.device,
  });
}

/**
 * Create a tensor of ones with the same shape and dtype as the input tensor.
 *
 * @param t - Reference tensor
 * @returns New tensor filled with ones
 *
 * @example
 * ```ts
 * const a = tensor([[1, 2], [3, 4]]);
 * const o = ones_like(a); // [[1, 1], [1, 1]]
 * ```
 * @deprecated Prefer {@link onesLike}.
 */
export function ones_like(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("ones_like is not defined for string dtype");
  }
  const dtype = ensureNonStringDType(t.dtype);
  const size = t.size;
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const data = new Ctor(size);
  if (data instanceof BigInt64Array) {
    data.fill(1n);
  } else {
    data.fill(1);
  }
  return Tensor.fromTypedArray({
    data,
    shape: [...t.shape],
    dtype,
    device: t.device,
  });
}

/**
 * Create an uninitialized tensor with the same shape and dtype as the input tensor.
 * In practice, the buffer is zero-filled (TypedArrays are zero-initialized).
 *
 * @param t - Reference tensor
 * @returns New tensor (zero-initialized)
 * @deprecated Prefer {@link emptyLike}.
 */
export function empty_like(t: Tensor): Tensor {
  return zeros_like(t);
}

/**
 * Create a tensor filled with a specified value, matching the shape and dtype of the input.
 *
 * @param t - Reference tensor
 * @param fillValue - Value to fill the tensor with
 * @returns New tensor filled with the specified value
 *
 * @example
 * ```ts
 * const a = tensor([[1, 2], [3, 4]]);
 * const f = full_like(a, 7); // [[7, 7], [7, 7]]
 * ```
 * @deprecated Prefer {@link fullLike}.
 */
export function full_like(t: Tensor, fillValue: number | bigint): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("full_like is not defined for string dtype");
  }
  const dtype = ensureNonStringDType(t.dtype);
  const size = t.size;
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const data = new Ctor(size);
  if (data instanceof BigInt64Array) {
    const val = typeof fillValue === "bigint" ? fillValue : BigInt(fillValue);
    data.fill(val);
  } else {
    const val = typeof fillValue === "bigint" ? Number(fillValue) : fillValue;
    data.fill(val);
  }
  return Tensor.fromTypedArray({
    data,
    shape: [...t.shape],
    dtype,
    device: t.device,
  });
}

// ─── clone / detach / contiguous / copy ─────────────────────────────────────

/**
 * Create a deep copy of a tensor with its own data buffer.
 *
 * The returned tensor shares no data with the original.
 *
 * @param t - Input tensor
 * @returns Deep copy of the tensor
 *
 * @example
 * ```ts
 * const a = tensor([1, 2, 3]);
 * const b = clone(a);
 * ```
 */
export function clone(t: Tensor): Tensor {
  if (t.dtype === "string") {
    const data = t.data;
    if (!Array.isArray(data)) {
      throw new DeepboxError("Internal error: string dtype but non-array data");
    }
    const size = t.size;
    const contig = isContiguous(t.shape, t.strides);
    const logicalStrides = computeStrides(t.shape);
    const out = new Array<string>(size);
    for (let i = 0; i < size; i++) {
      const off = flatOffset(i, t.offset, contig, logicalStrides, t.strides);
      out[i] = data[off] ?? "";
    }
    return Tensor.fromStringArray({
      data: out,
      shape: [...t.shape],
      device: t.device,
    });
  }
  const dtype = ensureNonStringDType(t.dtype);
  const outData = materializeNumeric(t);
  return Tensor.fromTypedArray({
    data: outData,
    shape: [...t.shape],
    dtype,
    device: t.device,
  });
}

/**
 * Create a copy of a tensor detached from any computation graph.
 * Equivalent to clone for non-gradient tensors.
 *
 * @param t - Input tensor
 * @returns Detached copy of the tensor
 */
export function detach(t: Tensor): Tensor {
  return clone(t);
}

/**
 * Return a contiguous tensor. If the tensor is already contiguous, returns itself.
 * Otherwise, materializes a new contiguous copy.
 *
 * @param t - Input tensor
 * @returns Contiguous tensor
 */
export function contiguous(t: Tensor): Tensor {
  if (isContiguous(t.shape, t.strides) && t.offset === 0) {
    return t;
  }
  return clone(t);
}

/**
 * Create an explicit deep copy of a tensor (alias for clone).
 *
 * @param t - Input tensor
 * @returns Deep copy of the tensor
 */
export function copy(t: Tensor): Tensor {
  return clone(t);
}

// ─── diag / diagonal ────────────────────────────────────────────────────────

/**
 * Extract a diagonal or construct a diagonal matrix.
 *
 * - If input is 1-D, returns a 2-D square matrix with the input as diagonal.
 * - If input is 2-D, returns the diagonal elements as a 1-D tensor.
 *
 * @param t - Input tensor (1-D or 2-D)
 * @param k - Diagonal offset (0 = main, positive = above, negative = below)
 * @returns Diagonal tensor
 *
 * @example
 * ```ts
 * diag(tensor([1, 2, 3]));     // [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
 * diag(tensor([[1, 2], [3, 4]])); // [1, 4]
 * ```
 */
export function diag(t: Tensor, k = 0): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("diag is not defined for string dtype");
  }
  if (t.ndim === 1) {
    return constructDiagonal(t, k);
  }
  if (t.ndim === 2) {
    return diagonal(t, k);
  }
  throw new ShapeError(`diag requires 1-D or 2-D input; got ${t.ndim}-D`);
}

/**
 * Extract the k-th diagonal from a 2-D tensor.
 *
 * @param t - Input 2-D tensor
 * @param k - Diagonal offset (0 = main, positive = above, negative = below)
 * @returns 1-D tensor of diagonal elements
 */
export function diagonal(t: Tensor, k = 0): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("diagonal is not defined for string dtype");
  }
  if (t.ndim !== 2) {
    throw new ShapeError(`diagonal requires 2-D input; got ${t.ndim}-D`);
  }

  const dtype = ensureNonStringDType(t.dtype);
  const rows = t.shape[0] ?? 0;
  const cols = t.shape[1] ?? 0;

  let startRow: number;
  let startCol: number;
  if (k >= 0) {
    startRow = 0;
    startCol = k;
  } else {
    startRow = -k;
    startCol = 0;
  }

  const diagLen = Math.max(0, Math.min(rows - startRow, cols - startCol));
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const outData = new Ctor(diagLen);
  const data = requireNumericData(t.data, "diagonal");

  for (let i = 0; i < diagLen; i++) {
    const r = startRow + i;
    const c = startCol + i;
    const off = t.offset + r * (t.strides[0] ?? 0) + c * (t.strides[1] ?? 0);
    writeElement(outData, i, data, off);
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: [diagLen],
    dtype,
    device: t.device,
  });
}

function constructDiagonal(t: Tensor, k: number): Tensor {
  const dtype = ensureNonStringDType(t.dtype);
  const n = t.size;
  const absK = Math.abs(k);
  const size = n + absK;
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const outData = new Ctor(size * size);
  const data = requireNumericData(t.data, "diag");

  const contig = isContiguous(t.shape, t.strides);
  const logicalStrides = computeStrides(t.shape);

  for (let i = 0; i < n; i++) {
    const srcOff = flatOffset(i, t.offset, contig, logicalStrides, t.strides);
    const r = k >= 0 ? i : i + absK;
    const c = k >= 0 ? i + absK : i;
    const dstOff = r * size + c;
    writeElement(outData, dstOff, data, srcOff);
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: [size, size],
    dtype,
    device: t.device,
  });
}

// ─── triu / tril ────────────────────────────────────────────────────────────

/**
 * Return the upper triangular part of a 2-D tensor, with elements
 * below the k-th diagonal zeroed.
 *
 * @param t - Input 2-D tensor
 * @param k - Diagonal offset (0 = main diagonal)
 * @returns Upper triangular tensor
 *
 * @example
 * ```ts
 * triu(tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]));
 * // [[1, 2, 3], [0, 5, 6], [0, 0, 9]]
 * ```
 */
export function triu(t: Tensor, k = 0): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("triu is not defined for string dtype");
  }
  if (t.ndim !== 2) {
    throw new ShapeError(`triu requires 2-D input; got ${t.ndim}-D`);
  }

  const dtype = ensureNonStringDType(t.dtype);
  const rows = t.shape[0] ?? 0;
  const cols = t.shape[1] ?? 0;
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const outData = new Ctor(rows * cols);
  const data = requireNumericData(t.data, "triu");

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (c >= r + k) {
        const srcOff = t.offset + r * (t.strides[0] ?? 0) + c * (t.strides[1] ?? 0);
        const dstOff = r * cols + c;
        writeElement(outData, dstOff, data, srcOff);
      }
    }
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: [rows, cols],
    dtype,
    device: t.device,
  });
}

/**
 * Return the lower triangular part of a 2-D tensor, with elements
 * above the k-th diagonal zeroed.
 *
 * @param t - Input 2-D tensor
 * @param k - Diagonal offset (0 = main diagonal)
 * @returns Lower triangular tensor
 *
 * @example
 * ```ts
 * tril(tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]));
 * // [[1, 0, 0], [4, 5, 0], [7, 8, 9]]
 * ```
 */
export function tril(t: Tensor, k = 0): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("tril is not defined for string dtype");
  }
  if (t.ndim !== 2) {
    throw new ShapeError(`tril requires 2-D input; got ${t.ndim}-D`);
  }

  const dtype = ensureNonStringDType(t.dtype);
  const rows = t.shape[0] ?? 0;
  const cols = t.shape[1] ?? 0;
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const outData = new Ctor(rows * cols);
  const data = requireNumericData(t.data, "tril");

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (c <= r + k) {
        const srcOff = t.offset + r * (t.strides[0] ?? 0) + c * (t.strides[1] ?? 0);
        const dstOff = r * cols + c;
        writeElement(outData, dstOff, data, srcOff);
      }
    }
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: [rows, cols],
    dtype,
    device: t.device,
  });
}

// ─── flip / fliplr / flipud ─────────────────────────────────────────────────

/**
 * Reverse the order of elements along the given axes.
 *
 * @param t - Input tensor
 * @param axes - Axes along which to flip. If undefined, flip all axes.
 * @returns Flipped tensor
 *
 * @example
 * ```ts
 * flip(tensor([1, 2, 3]));     // [3, 2, 1]
 * flip(tensor([[1, 2], [3, 4]]), [1]); // [[2, 1], [4, 3]]
 * ```
 */
export function flip(t: Tensor, axes?: number[]): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("flip is not defined for string dtype");
  }
  const dtype = ensureNonStringDType(t.dtype);

  const flipAxes = axes ?? Array.from({ length: t.ndim }, (_, i) => i);
  const normalizedAxes = flipAxes.map((ax) => normalizeAxis(ax, t.ndim));

  const outSize = t.size;
  const ndim = t.ndim;
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const outData = new Ctor(outSize);
  const data = requireNumericData(t.data, "flip");

  // Fast path: contiguous source. Walk the output in row-major order with an
  // allocation-free odometer that tracks the source offset incrementally, and
  // read/write the typed arrays directly. Avoids the per-element coords array
  // and indirect writeElement call of the generic path (numpy-competitive).
  if (outSize > 0 && ndim > 0 && isContiguous(t.shape, t.strides)) {
    const src = computeStrides(t.shape);
    const shapeArr = new Int32Array(ndim);
    const step = new Int32Array(ndim);
    const coord = new Int32Array(ndim);
    const flipped = new Uint8Array(ndim);
    for (const ax of normalizedAxes) flipped[ax] = 1;
    let srcOff = t.offset;
    for (let d = 0; d < ndim; d++) {
      const dim = t.shape[d] ?? 0;
      const st = src[d] ?? 0;
      shapeArr[d] = dim;
      if (flipped[d]) {
        step[d] = -st;
        srcOff += (dim - 1) * st;
      } else {
        step[d] = st;
      }
    }
    for (let i = 0; i < outSize; i++) {
      outData[i] = data[srcOff] as never;
      for (let d = ndim - 1; d >= 0; d--) {
        const s = step[d] as number;
        const dim = shapeArr[d] as number;
        srcOff += s;
        const c = (coord[d] as number) + 1;
        if (c < dim) {
          coord[d] = c;
          break;
        }
        coord[d] = 0;
        srcOff -= s * dim;
      }
    }
    return Tensor.fromTypedArray({
      data: outData,
      shape: [...t.shape],
      dtype,
      device: t.device,
    });
  }

  const outStrides = computeStrides(t.shape);
  for (let i = 0; i < outSize; i++) {
    const coords = flatToCoords(i, outStrides, ndim);

    for (const ax of normalizedAxes) {
      const dim = t.shape[ax] ?? 0;
      coords[ax] = dim - 1 - (coords[ax] ?? 0);
    }

    let srcOff = t.offset;
    for (let d = 0; d < ndim; d++) {
      srcOff += (coords[d] ?? 0) * (t.strides[d] ?? 0);
    }

    writeElement(outData, i, data, srcOff);
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: [...t.shape],
    dtype,
    device: t.device,
  });
}

/**
 * Flip a 2-D tensor left-right (reverse columns).
 *
 * @param t - Input 2-D tensor
 * @returns Horizontally flipped tensor
 * @deprecated Prefer {@link flipLr}.
 */
export function fliplr(t: Tensor): Tensor {
  if (t.ndim < 2) {
    throw new ShapeError(`fliplr requires at least 2-D input; got ${t.ndim}-D`);
  }
  return flip(t, [1]);
}

/**
 * Flip a 2-D tensor up-down (reverse rows).
 *
 * @param t - Input 2-D tensor
 * @returns Vertically flipped tensor
 * @deprecated Prefer {@link flipUd}.
 */
export function flipud(t: Tensor): Tensor {
  if (t.ndim < 1) {
    throw new ShapeError(`flipud requires at least 1-D input; got ${t.ndim}-D`);
  }
  return flip(t, [0]);
}

// ─── roll ───────────────────────────────────────────────────────────────────

/**
 * Roll tensor elements along the given axis.
 * Elements that roll beyond the last position are re-introduced at the first.
 *
 * @param t - Input tensor
 * @param shift - Number of places to shift (positive = toward end)
 * @param axis - Axis along which to roll (default: flatten, roll, reshape)
 * @returns Rolled tensor
 *
 * @example
 * ```ts
 * roll(tensor([1, 2, 3, 4, 5]), 2);  // [4, 5, 1, 2, 3]
 * ```
 */
export function roll(t: Tensor, shift: number, axis?: Axis): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("roll is not defined for string dtype");
  }
  const dtype = ensureNonStringDType(t.dtype);

  if (axis === undefined) {
    const flatSize = t.size;
    if (flatSize === 0) return clone(t);
    const s = ((shift % flatSize) + flatSize) % flatSize;
    const Ctor = dtypeToTypedArrayCtor(dtype);
    const outData = new Ctor(flatSize);
    const data = requireNumericData(t.data, "roll");
    const contig = isContiguous(t.shape, t.strides);
    const logicalStrides = computeStrides(t.shape);

    // Fast path: a flat roll of a contiguous buffer is two slab copies —
    // out[(i+s)%N] = data[i] ⇒ tail moves to front, head to back. Uses
    // TypedArray.set (memcpy) instead of per-element indirection.
    if (contig && t.offset === 0 && data.length === flatSize) {
      if (s === 0) {
        outData.set(data as never);
      } else {
        (outData as TypedArray).set((data as TypedArray).subarray(flatSize - s) as never, 0);
        (outData as TypedArray).set((data as TypedArray).subarray(0, flatSize - s) as never, s);
      }
      return Tensor.fromTypedArray({
        data: outData,
        shape: [...t.shape],
        dtype,
        device: t.device,
      });
    }

    for (let i = 0; i < flatSize; i++) {
      const srcIdx = flatOffset(i, t.offset, contig, logicalStrides, t.strides);
      const dstIdx = (i + s) % flatSize;
      writeElement(outData, dstIdx, data, srcIdx);
    }

    return Tensor.fromTypedArray({
      data: outData,
      shape: [...t.shape],
      dtype,
      device: t.device,
    });
  }

  const ax = normalizeAxis(axis, t.ndim);
  const axDim = t.shape[ax] ?? 0;
  if (axDim === 0) return clone(t);
  const s = ((shift % axDim) + axDim) % axDim;

  const outSize = t.size;
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const outData = new Ctor(outSize);
  const data = requireNumericData(t.data, "roll");
  const outStrides = computeStrides(t.shape);

  for (let i = 0; i < outSize; i++) {
    const coords = flatToCoords(i, outStrides, t.ndim);
    const outCoord = coords[ax] ?? 0;
    const srcCoord = (((outCoord - s) % axDim) + axDim) % axDim;
    coords[ax] = srcCoord;

    let srcOff = t.offset;
    for (let d = 0; d < t.ndim; d++) {
      srcOff += (coords[d] ?? 0) * (t.strides[d] ?? 0);
    }

    writeElement(outData, i, data, srcOff);
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: [...t.shape],
    dtype,
    device: t.device,
  });
}

// ─── pad ────────────────────────────────────────────────────────────────────

type PadMode = "constant" | "reflect" | "replicate" | "circular";

/**
 * Pad a tensor with values.
 *
 * @param t - Input tensor
 * @param padWidth - Padding for each axis as [[before0, after0], [before1, after1], ...]
 * @param mode - Padding mode: 'constant', 'reflect', 'replicate', 'circular'
 * @param constantValue - Fill value for constant mode (default: 0)
 * @returns Padded tensor
 *
 * @example
 * ```ts
 * pad(tensor([1, 2, 3]), [[2, 1]]); // [0, 0, 1, 2, 3, 0]
 * ```
 */
export function pad(
  t: Tensor,
  padWidth: ReadonlyArray<readonly [number, number]>,
  mode: PadMode = "constant",
  constantValue = 0
): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("pad is not defined for string dtype");
  }
  const dtype = ensureNonStringDType(t.dtype);

  if (padWidth.length !== t.ndim) {
    throw new InvalidParameterError(
      `padWidth length ${padWidth.length} must match tensor ndim ${t.ndim}`,
      "padWidth",
      padWidth
    );
  }

  for (let d = 0; d < padWidth.length; d++) {
    const pw = padWidth[d];
    if (pw?.length !== 2) {
      throw new InvalidParameterError(
        `padWidth[${d}] must be [before, after]`,
        "padWidth",
        padWidth
      );
    }
    if (pw[0] < 0 || pw[1] < 0) {
      throw new InvalidParameterError("padWidth values must be non-negative", "padWidth", padWidth);
    }
  }

  const outShape = t.shape.map((dim, d) => {
    const pw = padWidth[d];
    return (dim ?? 0) + (pw?.[0] ?? 0) + (pw?.[1] ?? 0);
  });

  const outSize = shapeToSize(outShape);
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const outData = new Ctor(outSize);
  const data = requireNumericData(t.data, "pad");
  const outStrides = computeStrides(outShape);

  if (mode === "constant") {
    fillTypedArray(outData, constantValue);
  }

  for (let i = 0; i < outSize; i++) {
    const outCoords = flatToCoords(i, outStrides, t.ndim);

    let valid = true;
    const inCoords = new Array<number>(t.ndim);
    for (let d = 0; d < t.ndim; d++) {
      const before = padWidth[d]?.[0] ?? 0;
      const dim = t.shape[d] ?? 0;
      let c = (outCoords[d] ?? 0) - before;

      if (c < 0 || c >= dim) {
        if (mode === "constant") {
          valid = false;
          break;
        } else if (mode === "replicate") {
          c = Math.max(0, Math.min(dim - 1, c));
        } else if (mode === "reflect") {
          if (dim <= 1) {
            c = 0;
          } else {
            while (c < 0 || c >= dim) {
              if (c < 0) c = -c;
              if (c >= dim) c = 2 * dim - 2 - c;
            }
          }
        } else if (mode === "circular") {
          c = ((c % dim) + dim) % dim;
        }
      }
      inCoords[d] = c;
    }

    if (!valid) continue;

    let srcOff = t.offset;
    for (let d = 0; d < t.ndim; d++) {
      srcOff += (inCoords[d] ?? 0) * (t.strides[d] ?? 0);
    }

    writeElement(outData, i, data, srcOff);
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: outShape,
    dtype,
    device: t.device,
  });
}

// ─── moveaxis / swapaxes ────────────────────────────────────────────────────

/**
 * Move axes of a tensor to new positions.
 *
 * @param t - Input tensor
 * @param source - Original positions of axes to move
 * @param destination - Destination positions for each axis
 * @returns Tensor with moved axes
 */
export function moveaxis(
  t: Tensor,
  source: number | number[],
  destination: number | number[]
): Tensor {
  const src = typeof source === "number" ? [source] : source;
  const dst = typeof destination === "number" ? [destination] : destination;

  if (src.length !== dst.length) {
    throw new InvalidParameterError(
      `source and destination must have the same length; got ${src.length} and ${dst.length}`,
      "source",
      source
    );
  }

  const ndim = t.ndim;
  const normSrc = src.map((s) => normalizeAxis(s, ndim));
  const normDst = dst.map((d) => normalizeAxis(d, ndim));

  const order = new Array<number>(ndim).fill(-1);
  const remaining: number[] = [];

  for (let i = 0; i < normSrc.length; i++) {
    order[normDst[i] ?? 0] = normSrc[i] ?? 0;
  }

  const usedSrc = new Set(normSrc);
  for (let i = 0; i < ndim; i++) {
    if (!usedSrc.has(i)) {
      remaining.push(i);
    }
  }

  let remIdx = 0;
  for (let i = 0; i < ndim; i++) {
    if (order[i] === -1) {
      order[i] = remaining[remIdx] ?? 0;
      remIdx++;
    }
  }

  return permuteAxes(t, order);
}

/**
 * Swap two axes of a tensor.
 *
 * @param t - Input tensor
 * @param axis1 - First axis
 * @param axis2 - Second axis
 * @returns Tensor with swapped axes
 */
export function swapaxes(t: Tensor, axis1: number, axis2: number): Tensor {
  const ndim = t.ndim;
  const ax1 = normalizeAxis(axis1, ndim);
  const ax2 = normalizeAxis(axis2, ndim);
  const order = Array.from({ length: ndim }, (_, i) => i);
  order[ax1] = ax2;
  order[ax2] = ax1;
  return permuteAxes(t, order);
}

function permuteAxes(t: Tensor, order: number[]): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("permuteAxes is not defined for string dtype");
  }
  const dtype = ensureNonStringDType(t.dtype);

  // Axis permutation is a pure stride/shape reorder — return a zero-copy view
  // (same as `transpose`), sharing the buffer instead of gathering elements.
  const outShape = order.map((ax) => t.shape[ax] ?? 0);
  const outStrides = order.map((ax) => t.strides[ax] ?? 0);

  if (t.isDeviceTensor) {
    return t.view(outShape, outStrides, t.offset);
  }

  const data = requireNumericData(t.data, "permuteAxes");
  return Tensor.fromTypedArray({
    data,
    shape: outShape,
    dtype,
    device: t.device,
    offset: t.offset,
    strides: outStrides,
  });
}

// ─── broadcast_to ───────────────────────────────────────────────────────────

/**
 * Broadcast a tensor to a target shape.
 *
 * @param t - Input tensor
 * @param shape - Target shape
 * @returns Broadcast tensor (new copy)
 * @deprecated Prefer {@link broadcastTo}.
 */
export function broadcast_to(t: Tensor, shape: Shape): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("broadcast_to is not defined for string dtype");
  }
  const dtype = ensureNonStringDType(t.dtype);

  if (shape.length < t.ndim) {
    throw new ShapeError(
      `Cannot broadcast shape [${t.shape}] to [${shape}]: target has fewer dimensions`
    );
  }

  const rankDiff = shape.length - t.ndim;
  for (let i = 0; i < t.ndim; i++) {
    const srcDim = t.shape[i] ?? 1;
    const tgtDim = shape[i + rankDiff] ?? 1;
    if (srcDim !== 1 && srcDim !== tgtDim) {
      throw new ShapeError(
        `Cannot broadcast shape [${t.shape}] to [${shape}]: incompatible at dimension ${i}`
      );
    }
  }

  // Zero-copy view: broadcast dimensions (size-1 or newly-added leading axes)
  // get stride 0 so they alias the same element, exactly like NumPy's
  // broadcast_to. Non-broadcast axes keep the source stride.
  const outStrides = new Array<number>(shape.length);
  for (let i = 0; i < shape.length; i++) {
    if (i < rankDiff) {
      outStrides[i] = 0;
    } else {
      const srcDim = t.shape[i - rankDiff] ?? 1;
      outStrides[i] = srcDim === 1 ? 0 : (t.strides[i - rankDiff] ?? 0);
    }
  }

  const data = requireNumericData(t.data, "broadcast_to");
  return Tensor.fromTypedArray({
    data,
    shape: [...shape],
    dtype,
    device: t.device,
    offset: t.offset,
    strides: outStrides,
  });
}

// ─── scatter ────────────────────────────────────────────────────────────────

/**
 * Scatter values into a tensor at specified indices along an axis.
 * This is the inverse of gather.
 *
 * @param t - Input tensor
 * @param dim - Dimension along which to scatter
 * @param index - Index tensor with same shape as src
 * @param src - Source values tensor
 * @returns New tensor with scattered values
 */
export function scatter(t: Tensor, dim: number, index: Tensor, src: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("scatter is not defined for string dtype");
  }
  const ax = normalizeAxis(dim, t.ndim);

  const result = clone(t);
  const outData = requireNumericData(result.data, "scatter");
  const srcData = requireNumericData(src.data, "scatter");
  const idxData = requireNumericData(index.data, "scatter");

  const idxContiguous = isContiguous(index.shape, index.strides);
  const idxLogical = computeStrides(index.shape);
  const srcContiguous = isContiguous(src.shape, src.strides);
  const srcLogical = computeStrides(src.shape);

  const outStrides = computeStrides(result.shape);
  const idxStrides = computeStrides(index.shape);

  for (let i = 0; i < index.size; i++) {
    const coords = flatToCoords(i, idxStrides, index.ndim);
    const idxOff = flatOffset(i, index.offset, idxContiguous, idxLogical, index.strides);
    const srcOff = flatOffset(i, src.offset, srcContiguous, srcLogical, src.strides);

    const targetIdx = readAsNumber(idxData, idxOff);
    coords[ax] = Math.round(targetIdx);

    let outOff = result.offset;
    for (let d = 0; d < result.ndim; d++) {
      outOff += (coords[d] ?? 0) * (outStrides[d] ?? 0);
    }

    writeElement(outData, outOff, srcData, srcOff);
  }

  return result;
}

// ─── NaN-aware reductions ───────────────────────────────────────────────────

/**
 * Sum of elements, ignoring NaN values.
 */
export function nansum(t: Tensor, axis?: Axis): Tensor {
  if (axis === undefined) {
    const f = nanSumCountFlat(t, "nansum");
    if (f) return nanScalar(f.sum, t.device);
  }
  return nanReduce(t, axis, "nansum", (vals) => {
    let s = 0;
    for (const v of vals) {
      if (!Number.isNaN(v)) s += v;
    }
    return s;
  });
}

/**
 * Mean of elements, ignoring NaN values.
 */
export function nanmean(t: Tensor, axis?: Axis): Tensor {
  if (axis === undefined) {
    const f = nanSumCountFlat(t, "nanmean");
    if (f) return nanScalar(f.count === 0 ? Number.NaN : f.sum / f.count, t.device);
  }
  return nanReduce(t, axis, "nanmean", (vals) => {
    let s = 0;
    let count = 0;
    for (const v of vals) {
      if (!Number.isNaN(v)) {
        s += v;
        count++;
      }
    }
    return count === 0 ? NaN : s / count;
  });
}

/**
 * Standard deviation of elements, ignoring NaN values.
 */
export function nanstd(t: Tensor, axis?: Axis): Tensor {
  // Population std (ddof=0), matching both `std` and NumPy's nanstd default.
  // Two-pass computation avoids the catastrophic cancellation of E[x²]−mean².
  return nanReduce(t, axis, "nanstd", (vals) => {
    let s = 0;
    let count = 0;
    for (const v of vals) {
      if (!Number.isNaN(v)) {
        s += v;
        count++;
      }
    }
    if (count === 0) return NaN;
    const mean = s / count;
    let ss = 0;
    for (const v of vals) {
      if (!Number.isNaN(v)) {
        const d = v - mean;
        ss += d * d;
      }
    }
    return Math.sqrt(ss / count);
  });
}

/**
 * Minimum of elements, ignoring NaN values.
 */
export function nanmin(t: Tensor, axis?: Axis): Tensor {
  if (axis === undefined) {
    const fast = nanExtremeFlat(t, false, "nanmin");
    if (fast) return fast;
  }
  return nanReduce(t, axis, "nanmin", (vals) => {
    let m = Infinity;
    let seen = false;
    for (const v of vals) {
      if (!Number.isNaN(v)) {
        seen = true;
        if (v < m) m = v;
      }
    }
    // Track "any non-NaN seen" separately: a slice of all +Infinity must
    // return Infinity, not NaN.
    return seen ? m : NaN;
  });
}

/**
 * Maximum of elements, ignoring NaN values.
 */
export function nanmax(t: Tensor, axis?: Axis): Tensor {
  if (axis === undefined) {
    const fast = nanExtremeFlat(t, true, "nanmax");
    if (fast) return fast;
  }
  return nanReduce(t, axis, "nanmax", (vals) => {
    let m = -Infinity;
    let seen = false;
    for (const v of vals) {
      if (!Number.isNaN(v)) {
        seen = true;
        if (v > m) m = v;
      }
    }
    return seen ? m : NaN;
  });
}

/**
 * Streaming NaN-ignoring sum + non-NaN count over the whole (flattened) tensor,
 * scanning the contiguous typed array directly with a narrowed load site (no
 * per-element `number[]` push). Returns null when the input is not a simple
 * contiguous numeric buffer (caller falls back to the generic reducer).
 */
function nanSumCountFlat(t: Tensor, opName: string): { sum: number; count: number } | null {
  if (t.dtype === "string") {
    throw new DTypeError(`${opName} is not defined for string dtype`);
  }
  if (!isContiguous(t.shape, t.strides)) return null;
  const data = requireNumericData(t.data, opName);
  if (isBigIntArray(data)) return null;
  const n = t.size;
  const start = t.offset;
  const end = start + n;
  let sum = 0;
  let count = 0;
  if (data instanceof Float64Array) {
    const src = data;
    for (let i = start; i < end; i++) {
      const v = src[i] as number;
      if (!Number.isNaN(v)) {
        sum += v;
        count++;
      }
    }
  } else {
    const src = data;
    for (let i = start; i < end; i++) {
      const v = src[i] as number;
      if (!Number.isNaN(v)) {
        sum += v;
        count++;
      }
    }
  }
  return { sum, count };
}

function nanScalar(value: number, device: Tensor["device"]): Tensor {
  return Tensor.fromTypedArray({
    data: new Float64Array([value]),
    shape: [],
    dtype: "float64",
    device,
  });
}

/**
 * Streaming NaN-ignoring min/max over the whole (flattened) tensor. Avoids the
 * generic path's per-element `number[]` push (O(n) boxing) by scanning the
 * contiguous typed array directly with a narrowed load site. Returns null when
 * the input is not a simple contiguous numeric buffer (caller falls back).
 */
function nanExtremeFlat(t: Tensor, isMax: boolean, opName: string): Tensor | null {
  if (t.dtype === "string") {
    throw new DTypeError(`${opName} is not defined for string dtype`);
  }
  if (!isContiguous(t.shape, t.strides)) return null;
  const data = requireNumericData(t.data, opName);
  if (isBigIntArray(data)) return null;
  const n = t.size;
  const start = t.offset;
  const end = start + n;
  let m = isMax ? -Infinity : Infinity;
  let seen = false;
  if (data instanceof Float64Array) {
    const src = data;
    for (let i = start; i < end; i++) {
      const v = src[i] as number;
      if (!Number.isNaN(v)) {
        seen = true;
        if (isMax ? v > m : v < m) m = v;
      }
    }
  } else {
    const src = data;
    for (let i = start; i < end; i++) {
      const v = src[i] as number;
      if (!Number.isNaN(v)) {
        seen = true;
        if (isMax ? v > m : v < m) m = v;
      }
    }
  }
  const outData = new Float64Array([seen ? m : NaN]);
  return Tensor.fromTypedArray({
    data: outData,
    shape: [],
    dtype: "float64",
    device: t.device,
  });
}

function nanReduce(
  t: Tensor,
  axis: Axis | undefined,
  opName: string,
  reducer: (vals: number[]) => number
): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError(`${opName} is not defined for string dtype`);
  }

  const data = requireNumericData(t.data, opName);
  const contig = isContiguous(t.shape, t.strides);
  const logicalStrides = computeStrides(t.shape);

  if (axis === undefined) {
    const vals: number[] = [];
    for (let i = 0; i < t.size; i++) {
      const off = flatOffset(i, t.offset, contig, logicalStrides, t.strides);
      vals.push(readAsNumber(data, off));
    }
    const result = reducer(vals);
    const outData = new Float64Array([result]);
    return Tensor.fromTypedArray({
      data: outData,
      shape: [],
      dtype: "float64",
      device: t.device,
    });
  }

  const ax = normalizeAxis(axis, t.ndim);
  const axDim = t.shape[ax] ?? 0;
  const outShape = t.shape.filter((_, i) => i !== ax);
  const outSize = shapeToSize(outShape.length === 0 ? [1] : outShape);
  const outData = new Float64Array(outSize);
  const outStrides = computeStrides(outShape.length === 0 ? [1] : outShape);

  for (let i = 0; i < outSize; i++) {
    const outCoords = flatToCoords(i, outStrides, outShape.length);

    const vals: number[] = [];
    for (let a = 0; a < axDim; a++) {
      const fullCoords: number[] = [];
      let oc = 0;
      for (let d = 0; d < t.ndim; d++) {
        if (d === ax) {
          fullCoords.push(a);
        } else {
          fullCoords.push(outCoords[oc] ?? 0);
          oc++;
        }
      }

      let off = t.offset;
      for (let d = 0; d < t.ndim; d++) {
        off += (fullCoords[d] ?? 0) * (t.strides[d] ?? 0);
      }
      vals.push(readAsNumber(data, off));
    }

    outData[i] = reducer(vals);
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: outShape.length === 0 ? [] : outShape,
    dtype: "float64",
    device: t.device,
  });
}

// ─── unique ─────────────────────────────────────────────────────────────────

/**
 * Return the unique elements of a tensor, sorted in ascending order.
 *
 * @param t - Input tensor
 * @param returnCounts - If true, also return the count of each unique element
 * @returns Object with `values` tensor and optionally `counts` tensor
 *
 * @example
 * ```ts
 * const { values, counts } = unique(tensor([3, 1, 2, 1, 3]), true);
 * // values: [1, 2, 3], counts: [2, 1, 2]
 * ```
 */
export function unique(t: Tensor, returnCounts?: boolean): { values: Tensor; counts?: Tensor } {
  if (t.dtype === "string") {
    throw new DTypeError("unique is not defined for string dtype");
  }
  const dtype = ensureNonStringDType(t.dtype);

  const data = requireNumericData(t.data, "unique");
  const contig = isContiguous(t.shape, t.strides);
  const logicalStrides = computeStrides(t.shape);

  const vals: number[] = [];
  for (let i = 0; i < t.size; i++) {
    const off = flatOffset(i, t.offset, contig, logicalStrides, t.strides);
    vals.push(readAsNumber(data, off));
  }

  // NaN-aware sort (NaN last, NumPy convention); a plain `a - b` comparator
  // returns NaN for NaN operands and leaves the array unsorted.
  vals.sort((a, b) => {
    const aNaN = Number.isNaN(a);
    const bNaN = Number.isNaN(b);
    if (aNaN) return bNaN ? 0 : 1;
    if (bNaN) return -1;
    return a < b ? -1 : a > b ? 1 : 0;
  });

  const uniqueVals: number[] = [];
  const countVals: number[] = [];

  for (let i = 0; i < vals.length; ) {
    const v = vals[i] ?? 0;
    let count = 1;
    // NaN !== NaN, so group NaNs explicitly (NumPy keeps a single NaN entry)
    if (Number.isNaN(v)) {
      while (i + count < vals.length && Number.isNaN(vals[i + count] ?? 0)) {
        count++;
      }
    } else {
      while (i + count < vals.length && (vals[i + count] ?? 0) === v) {
        count++;
      }
    }
    uniqueVals.push(v);
    countVals.push(count);
    i += count;
  }

  const Ctor = dtypeToTypedArrayCtor(dtype);
  const outData = new Ctor(uniqueVals.length);
  if (outData instanceof BigInt64Array) {
    for (let i = 0; i < uniqueVals.length; i++) {
      outData[i] = BigInt(uniqueVals[i] ?? 0);
    }
  } else {
    for (let i = 0; i < uniqueVals.length; i++) {
      outData[i] = uniqueVals[i] ?? 0;
    }
  }

  const values = Tensor.fromTypedArray({
    data: outData,
    shape: [uniqueVals.length],
    dtype,
    device: t.device,
  });

  if (returnCounts) {
    const countsData = new Float64Array(countVals.length);
    for (let i = 0; i < countVals.length; i++) {
      countsData[i] = countVals[i] ?? 0;
    }
    const counts = Tensor.fromTypedArray({
      data: countsData,
      shape: [countVals.length],
      dtype: "float64",
      device: t.device,
    });
    return { values, counts };
  }

  return { values };
}

// ─── searchsorted ───────────────────────────────────────────────────────────

/**
 * Find indices where elements should be inserted to maintain order.
 *
 * @param sortedArr - 1-D sorted tensor
 * @param values - Values to insert
 * @param side - 'left' (default) or 'right'
 * @returns Tensor of insertion indices
 */
export function searchsorted(
  sortedArr: Tensor,
  values: Tensor,
  side: "left" | "right" = "left"
): Tensor {
  if (sortedArr.dtype === "string" || values.dtype === "string") {
    throw new DTypeError("searchsorted is not defined for string dtype");
  }
  if (sortedArr.ndim !== 1) {
    throw new ShapeError(`searchsorted requires 1-D sorted array; got ${sortedArr.ndim}-D`);
  }

  const arrData = requireNumericData(sortedArr.data, "searchsorted");
  const arrContiguous = isContiguous(sortedArr.shape, sortedArr.strides);
  const arrLogical = computeStrides(sortedArr.shape);
  const arrLen = sortedArr.size;

  const valData = requireNumericData(values.data, "searchsorted");
  const valContiguous = isContiguous(values.shape, values.strides);
  const valLogical = computeStrides(values.shape);

  const outData = new Float64Array(values.size);

  for (let i = 0; i < values.size; i++) {
    const valOff = flatOffset(i, values.offset, valContiguous, valLogical, values.strides);
    const v = readAsNumber(valData, valOff);

    let lo = 0;
    let hi = arrLen;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      const midOff = flatOffset(
        mid,
        sortedArr.offset,
        arrContiguous,
        arrLogical,
        sortedArr.strides
      );
      const midVal = readAsNumber(arrData, midOff);
      if (side === "left" ? midVal < v : midVal <= v) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    outData[i] = lo;
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: [...values.shape],
    dtype: "float64",
    device: sortedArr.device,
  });
}

// ─── histogram / bincount ───────────────────────────────────────────────────

/**
 * Compute the histogram of a 1-D tensor.
 *
 * @param t - Input 1-D tensor
 * @param bins - Number of equal-width bins
 * @param range - [min, max] range for the bins
 * @returns Object with `counts` and `binEdges` tensors
 */
export function histogram(
  t: Tensor,
  bins = 10,
  range?: readonly [number, number]
): { counts: Tensor; binEdges: Tensor } {
  if (t.dtype === "string") {
    throw new DTypeError("histogram is not defined for string dtype");
  }
  if (!Number.isInteger(bins) || bins <= 0) {
    throw new InvalidParameterError(`bins must be a positive integer; got ${bins}`, "bins", bins);
  }

  const data = requireNumericData(t.data, "histogram");
  const contig = isContiguous(t.shape, t.strides);
  const logicalStrides = computeStrides(t.shape);

  const vals: number[] = [];
  for (let i = 0; i < t.size; i++) {
    const off = flatOffset(i, t.offset, contig, logicalStrides, t.strides);
    vals.push(readAsNumber(data, off));
  }

  let lo: number;
  let hi: number;
  if (range) {
    lo = range[0];
    hi = range[1];
  } else {
    lo = Infinity;
    hi = -Infinity;
    for (const v of vals) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!Number.isFinite(lo)) lo = 0;
    if (!Number.isFinite(hi)) hi = 1;
    if (lo === hi) {
      lo -= 0.5;
      hi += 0.5;
    }
  }

  const width = (hi - lo) / bins;
  const counts = new Float64Array(bins);
  const edges = new Float64Array(bins + 1);

  for (let i = 0; i <= bins; i++) {
    edges[i] = lo + i * width;
  }

  for (const v of vals) {
    if (v < lo || v > hi) continue;
    let idx = Math.floor((v - lo) / width);
    if (idx === bins) idx = bins - 1;
    if (idx >= 0 && idx < bins) {
      counts[idx] = (counts[idx] ?? 0) + 1;
    }
  }

  return {
    counts: Tensor.fromTypedArray({
      data: counts,
      shape: [bins],
      dtype: "float64",
      device: t.device,
    }),
    binEdges: Tensor.fromTypedArray({
      data: edges,
      shape: [bins + 1],
      dtype: "float64",
      device: t.device,
    }),
  };
}

/**
 * Count number of occurrences of each value in a non-negative integer tensor.
 *
 * @param t - Input tensor of non-negative integers
 * @param minlength - Minimum output length
 * @returns Tensor of counts
 */
export function bincount(t: Tensor, minlength = 0): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("bincount is not defined for string dtype");
  }

  const data = requireNumericData(t.data, "bincount");
  const contig = isContiguous(t.shape, t.strides);
  const logicalStrides = computeStrides(t.shape);

  let maxVal = minlength - 1;
  for (let i = 0; i < t.size; i++) {
    const off = flatOffset(i, t.offset, contig, logicalStrides, t.strides);
    const v = readAsNumber(data, off);
    if (v < 0 || !Number.isInteger(v)) {
      throw new InvalidParameterError(
        `bincount requires non-negative integer values; got ${v}`,
        "t",
        v
      );
    }
    if (v > maxVal) maxVal = v;
  }

  const counts = new Float64Array(maxVal + 1);
  for (let i = 0; i < t.size; i++) {
    const off = flatOffset(i, t.offset, contig, logicalStrides, t.strides);
    const v = Math.round(readAsNumber(data, off));
    counts[v] = (counts[v] ?? 0) + 1;
  }

  return Tensor.fromTypedArray({
    data: counts,
    shape: [maxVal + 1],
    dtype: "float64",
    device: t.device,
  });
}

// ─── Boolean indexing / masking ─────────────────────────────────────────────

/**
 * Select elements from a tensor using a boolean mask.
 * Returns a 1-D tensor of elements where the mask is true (nonzero).
 *
 * @param t - Input tensor
 * @param mask - Boolean mask tensor (same shape as t)
 * @returns 1-D tensor of selected elements
 *
 * @example
 * ```ts
 * const a = tensor([1, 2, 3, 4, 5]);
 * const m = tensor([1, 0, 1, 0, 1], { dtype: 'bool' });
 * booleanIndex(a, m); // [1, 3, 5]
 * ```
 */
export function booleanIndex(t: Tensor, mask: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("booleanIndex is not defined for string dtype");
  }
  if (mask.dtype === "string") {
    throw new DTypeError("booleanIndex mask must be numeric");
  }
  if (t.size !== mask.size) {
    throw new ShapeError(`Tensor size ${t.size} does not match mask size ${mask.size}`);
  }
  const dtype = ensureNonStringDType(t.dtype);

  const tData = requireNumericData(t.data, "booleanIndex");
  const maskData = requireNumericData(mask.data, "booleanIndex");
  const tContiguous = isContiguous(t.shape, t.strides);
  const tLogical = computeStrides(t.shape);
  const maskContiguous = isContiguous(mask.shape, mask.strides);
  const maskLogical = computeStrides(mask.shape);

  let count = 0;
  for (let i = 0; i < mask.size; i++) {
    const mOff = flatOffset(i, mask.offset, maskContiguous, maskLogical, mask.strides);
    if (readAsNumber(maskData, mOff) !== 0) count++;
  }

  const Ctor = dtypeToTypedArrayCtor(dtype);
  const outData = new Ctor(count);
  let idx = 0;

  for (let i = 0; i < t.size; i++) {
    const mOff = flatOffset(i, mask.offset, maskContiguous, maskLogical, mask.strides);
    if (readAsNumber(maskData, mOff) !== 0) {
      const tOff = flatOffset(i, t.offset, tContiguous, tLogical, t.strides);
      writeElement(outData, idx, tData, tOff);
      idx++;
    }
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: [count],
    dtype,
    device: t.device,
  });
}

// ─── Fancy indexing ─────────────────────────────────────────────────────────

/**
 * Select elements from a tensor using integer index arrays.
 *
 * @param t - Input tensor
 * @param indices - 1-D tensor of indices
 * @param axis - Axis along which to index (default: 0)
 * @returns Tensor with selected elements
 */
export function fancyIndex(t: Tensor, indices: Tensor, axis: Axis = 0): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("fancyIndex is not defined for string dtype");
  }
  if (indices.dtype === "string") {
    throw new DTypeError("fancyIndex indices must be numeric");
  }
  const dtype = ensureNonStringDType(t.dtype);

  const ax = normalizeAxis(axis, t.ndim);
  const axDim = t.shape[ax] ?? 0;

  const idxData = requireNumericData(indices.data, "fancyIndex");
  const idxContiguous = isContiguous(indices.shape, indices.strides);
  const idxLogical = computeStrides(indices.shape);
  const numIndices = indices.size;

  const outShape = [...t.shape];
  outShape[ax] = numIndices;
  const outSize = shapeToSize(outShape);
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const outData = new Ctor(outSize);
  const tData = requireNumericData(t.data, "fancyIndex");
  const outStrides = computeStrides(outShape);

  for (let i = 0; i < outSize; i++) {
    const coords = flatToCoords(i, outStrides, t.ndim);
    const idxPos = coords[ax] ?? 0;
    const idxOff = flatOffset(idxPos, indices.offset, idxContiguous, idxLogical, indices.strides);
    let actualIdx = Math.round(readAsNumber(idxData, idxOff));
    if (actualIdx < 0) actualIdx += axDim;

    coords[ax] = actualIdx;

    let srcOff = t.offset;
    for (let d = 0; d < t.ndim; d++) {
      srcOff += (coords[d] ?? 0) * (t.strides[d] ?? 0);
    }

    writeElement(outData, i, tData, srcOff);
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: outShape,
    dtype,
    device: t.device,
  });
}

/**
 * Rotate a 2-D tensor by 90 degrees counter-clockwise.
 *
 * @param t - Input 2-D tensor
 * @param k - Number of 90° rotations (default 1). Negative values rotate clockwise.
 */
export function rot90(t: Tensor, k = 1): Tensor {
  if (t.ndim !== 2) {
    throw new ShapeError(`rot90 requires a 2-D tensor, got ${t.ndim}-D`);
  }
  // Normalize k to [0,3]
  k = ((k % 4) + 4) % 4;
  if (k === 0) return clone(t);

  let result = t;
  for (let i = 0; i < k; i++) {
    // 90° CCW: transpose then flipud
    result = flipud(transpose(result));
  }
  return result;
}

/**
 * Ensure the input is at least 1-D.
 * Scalars become shape [1]; higher-dimensional tensors pass through unchanged.
 * @deprecated Prefer {@link atleast1d}.
 */
export function atleast_1d(t: Tensor): Tensor {
  if (t.ndim >= 1) return t;
  // Scalar → [1]
  return t.reshape([1]);
}

/**
 * Ensure the input is at least 2-D.
 * Scalars become shape [1,1]; 1-D tensors become shape [1,n]; others pass through.
 * @deprecated Prefer {@link atleast2d}.
 */
export function atleast_2d(t: Tensor): Tensor {
  if (t.ndim >= 2) return t;
  if (t.ndim === 0) return t.reshape([1, 1]);
  // 1-D → [1, n]
  return t.reshape([1, t.shape[0]!]);
}

/**
 * Compute the cross product of two 3-element vectors.
 *
 * Both inputs must be 1-D tensors of length 3.
 */
export function cross(a: Tensor, b: Tensor): Tensor {
  if (a.ndim !== 1 || a.shape[0] !== 3) {
    throw new ShapeError(`cross requires 1-D tensors of length 3; a has shape [${a.shape}]`);
  }
  if (b.ndim !== 1 || b.shape[0] !== 3) {
    throw new ShapeError(`cross requires 1-D tensors of length 3; b has shape [${b.shape}]`);
  }

  const aData = requireNumericData(a.data, "cross");
  const bData = requireNumericData(b.data, "cross");
  const aStride = a.strides[0] ?? 1;
  const bStride = b.strides[0] ?? 1;
  const a0 = readAsNumber(aData, a.offset);
  const a1 = readAsNumber(aData, a.offset + aStride);
  const a2 = readAsNumber(aData, a.offset + 2 * aStride);
  const b0 = readAsNumber(bData, b.offset);
  const b1 = readAsNumber(bData, b.offset + bStride);
  const b2 = readAsNumber(bData, b.offset + 2 * bStride);

  return Tensor.fromTypedArray({
    data: new Float64Array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0]),
    shape: [3],
    dtype: "float64",
    device: a.device,
  });
}

/**
 * Return coordinate matrices from coordinate vectors.
 *
 * Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector
 * fields over N-D grids, given one-dimensional coordinate arrays x1, x2, ..., xn.
 *
 * Supports `"xy"` (default, Cartesian) and `"ij"` (matrix) indexing.
 *
 * @param tensors - 1-D coordinate tensors
 * @param indexing - Indexing mode: `"xy"` (default) or `"ij"`
 * @returns Array of N-D tensors, one for each input
 *
 * @example
 * ```ts
 * const x = tensor([1, 2, 3]);
 * const y = tensor([4, 5]);
 * const [X, Y] = meshgrid(x, y);
 * // X shape: [2, 3], Y shape: [2, 3]
 * ```
 */
export function meshgrid(
  ...args: [...Tensor[], ...[{ readonly indexing?: "xy" | "ij" }]] | Tensor[]
): Tensor[] {
  let indexing: "xy" | "ij" = "xy";
  let tensors: Tensor[];

  // Check if last arg is options
  const lastArg = args[args.length - 1];
  if (
    lastArg !== undefined &&
    !(lastArg instanceof Tensor) &&
    typeof lastArg === "object" &&
    lastArg !== null &&
    "indexing" in lastArg
  ) {
    const opts = lastArg as { readonly indexing?: "xy" | "ij" };
    if (opts.indexing !== undefined) indexing = opts.indexing;
    tensors = args.slice(0, -1) as Tensor[];
  } else {
    tensors = args as Tensor[];
  }

  if (tensors.length === 0) return [];

  // Validate all inputs are 1-D
  for (const t of tensors) {
    if (t.ndim !== 1) {
      throw new ShapeError(`meshgrid requires 1-D input tensors; got ${t.ndim}-D`);
    }
  }

  const ndim = tensors.length;
  const sizes = tensors.map((t) => t.shape[0] ?? 0);

  // For "xy" indexing, swap first two dimensions
  let orderedTensors = tensors;
  let orderedSizes = sizes;
  if (indexing === "xy" && ndim >= 2) {
    orderedTensors = [tensors[1]!, tensors[0]!, ...tensors.slice(2)];
    orderedSizes = [sizes[1]!, sizes[0]!, ...sizes.slice(2)];
  }

  // Output shape
  const outShape = [...orderedSizes];

  const result: Tensor[] = [];
  const totalSize = outShape.reduce((a, b) => a * b, 1);
  for (let dim = 0; dim < ndim; dim++) {
    const t = orderedTensors[dim]!;
    const srcData = requireNumericData(t.data, "meshgrid");
    const outData = new Float64Array(totalSize);

    // Along `dim`, each coordinate value repeats for `inner` consecutive
    // elements, and that pattern of length `dimSize * inner` tiles the
    // whole output. Fill the first tile with `fill()`, then replicate it
    // with doubling `copyWithin` instead of decoding every flat index.
    const dimSize = outShape[dim] ?? 0;
    const srcStride = t.strides[0] ?? 0;
    let inner = 1;
    for (let d = dim + 1; d < outShape.length; d++) inner *= outShape[d] ?? 0;
    const tile = dimSize * inner;
    if (totalSize > 0 && tile > 0) {
      for (let j = 0; j < dimSize; j++) {
        const v = readAsNumber(srcData, t.offset + j * srcStride);
        outData.fill(v, j * inner, (j + 1) * inner);
      }
      for (let filled = tile; filled < totalSize; filled *= 2) {
        outData.copyWithin(filled, 0, Math.min(filled, totalSize - filled));
      }
    }

    result.push(
      Tensor.fromTypedArray({
        data: outData,
        shape: [...outShape],
        dtype: "float64",
        device: t.device,
      })
    );
  }

  // For "xy" indexing, swap first two results back
  if (indexing === "xy" && ndim >= 2) {
    const swapped = [result[1]!, result[0]!, ...result.slice(2)];
    return swapped;
  }

  return result;
}

/**
 * Select elements along a dimension using an index tensor.
 *
 * Returns a new tensor that indexes the input tensor along dimension `dim`
 * using the entries in `index` (a 1-D LongTensor/int32 Tensor).
 *
 * Equivalent to PyTorch's `torch.index_select(input, dim, index)`.
 *
 * @param input - The input tensor
 * @param dim - The dimension to index along
 * @param index - 1-D tensor of indices to select
 * @returns A new tensor with the selected elements
 *
 * @example
 * ```ts
 * const x = tensor([[1, 2, 3], [4, 5, 6]]);
 * const idx = tensor([0, 2], { dtype: 'int32' });
 * const result = index_select(x, 1, idx); // [[1, 3], [4, 6]]
 * ```
 * @deprecated Prefer {@link indexSelect}.
 */
export function index_select(input: Tensor, dim: number, index: Tensor): Tensor {
  if (index.ndim !== 1) {
    throw new ShapeError(`index_select requires a 1-D index tensor; got ${index.ndim}-D`);
  }

  const ndim = input.ndim;
  const resolvedDim = dim < 0 ? ndim + dim : dim;
  if (resolvedDim < 0 || resolvedDim >= ndim) {
    throw new ShapeError(`index_select dim ${dim} out of range for ${ndim}-D tensor`);
  }

  const nIdx = index.shape[0] ?? 0;
  const idxData = requireNumericData(index.data, "index_select");

  // Build output shape: same as input but dim has size nIdx
  const outShape = [...input.shape];
  outShape[resolvedDim] = nIdx;

  const totalSize = outShape.reduce((a, b) => a * b, 1);
  const srcData = requireNumericData(input.data, "index_select");
  const outStrides = computeStrides(outShape);

  const outData = new Float64Array(totalSize);

  // Resolve and bounds-check the gather indices once (honoring the index
  // tensor's stride), instead of re-decoding them per output element.
  const dimSize = input.shape[resolvedDim] ?? 0;
  const idxStride = index.strides[0] ?? 1;
  const mapped = new Int32Array(nIdx);
  for (let j = 0; j < nIdx; j++) {
    const mi = Math.round(readAsNumber(idxData, index.offset + j * idxStride));
    if (mi < 0 || mi >= dimSize) {
      throw new ShapeError(
        `index_select index ${mi} is out of bounds for dimension ${resolvedDim} of size ${dimSize}`
      );
    }
    mapped[j] = mi;
  }

  const src = readNumericContiguous(input);
  if (src && totalSize > 0) {
    // Dense layout: every selected index is a contiguous slab of `inner`
    // elements, copied with subarray/set.
    let inner = 1;
    for (let d = resolvedDim + 1; d < ndim; d++) inner *= input.shape[d] ?? 0;
    const slab = dimSize * inner;
    const outer = inner === 0 ? 0 : totalSize / (nIdx * inner);
    let pos = 0;
    for (let o = 0; o < outer; o++) {
      const srcBase = o * slab;
      for (let j = 0; j < nIdx; j++) {
        const start = srcBase + (mapped[j] as number) * inner;
        if (inner >= 16) {
          outData.set(src.subarray(start, start + inner), pos);
          pos += inner;
        } else {
          for (let k = 0; k < inner; k++) outData[pos++] = src[start + k] as number;
        }
      }
    }
  } else if (totalSize > 0) {
    for (let i = 0; i < totalSize; i++) {
      // Decompose flat index into multi-index for output
      let rem = i;
      const multiIdx: number[] = [];
      for (let d = 0; d < ndim; d++) {
        const s = outStrides[d] ?? 1;
        multiIdx.push(Math.floor(rem / s));
        rem = rem % s;
      }

      const mappedIdx = mapped[multiIdx[resolvedDim] ?? 0] as number;

      // Build source flat index
      let srcFlat = input.offset;
      for (let d = 0; d < ndim; d++) {
        const idx = d === resolvedDim ? mappedIdx : (multiIdx[d] ?? 0);
        srcFlat += idx * (input.strides[d] ?? 0);
      }

      outData[i] = readAsNumber(srcData, srcFlat);
    }
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: outShape,
    dtype: "float64",
    device: input.device,
  });
}

/**
 * Test whether each element of `a` is in `values`.
 *
 * Returns a boolean tensor of the same shape as `a`.
 *
 * @param a - Input tensor
 * @param values - Values to test against (as a 1-D tensor or number array)
 */
export function isin(a: Tensor, values: Tensor | number[]): Tensor {
  const valArr = Array.isArray(values)
    ? values
    : (() => {
        const v: number[] = [];
        const vData = requireNumericData(values.data, "isin");
        const cont = isContiguous(values.shape, values.strides);
        const logStrides = computeStrides(values.shape);
        for (let i = 0; i < values.size; i++) {
          const off = flatOffset(i, values.offset, cont, logStrides, values.strides);
          v.push(readAsNumber(vData, off));
        }
        return v;
      })();
  const valSet = new Set(valArr);

  const out = new Uint8Array(a.size);
  const srcC = readNumericContiguous(a);
  if (srcC) {
    for (let i = 0; i < a.size; i++) {
      out[i] = valSet.has(srcC[i] as number) ? 1 : 0;
    }
  } else {
    const srcData = requireNumericData(a.data, "isin");
    const cont = isContiguous(a.shape, a.strides);
    const logStrides = computeStrides(a.shape);
    for (let i = 0; i < a.size; i++) {
      const off = flatOffset(i, a.offset, cont, logStrides, a.strides);
      out[i] = valSet.has(readAsNumber(srcData, off)) ? 1 : 0;
    }
  }
  return Tensor.fromTypedArray({
    data: out,
    shape: [...a.shape],
    dtype: "bool",
    device: a.device,
  });
}

// ---------------------------------------------------------------------------
// Canonical camelCase aliases
//
// Deepbox is converging on camelCase across its public surface to match the
// ml/metrics/estimator APIs. The snake_case spellings above mirror NumPy and
// remain exported (and are marked `@deprecated`) for backward compatibility;
// these aliases are the recommended names and refer to the exact same function.
// ---------------------------------------------------------------------------

/** Canonical camelCase alias of {@link broadcast_to}. */
export const broadcastTo = broadcast_to;
/** Canonical camelCase alias of {@link atleast_1d}. */
export const atleast1d = atleast_1d;
/** Canonical camelCase alias of {@link atleast_2d}. */
export const atleast2d = atleast_2d;
/** Canonical camelCase alias of {@link zeros_like}. */
export const zerosLike = zeros_like;
/** Canonical camelCase alias of {@link ones_like}. */
export const onesLike = ones_like;
/** Canonical camelCase alias of {@link empty_like}. */
export const emptyLike = empty_like;
/** Canonical camelCase alias of {@link full_like}. */
export const fullLike = full_like;
/** Canonical camelCase alias of {@link index_select}. */
export const indexSelect = index_select;
/** Canonical camelCase alias of {@link fliplr}. */
export const flipLr = fliplr;
/** Canonical camelCase alias of {@link flipud}. */
export const flipUd = flipud;
