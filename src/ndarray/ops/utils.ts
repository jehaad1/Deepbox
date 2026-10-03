/**
 * Tensor utility operations.
 *
 * This module provides utility functions for tensor creation and manipulation:
 * - where: Conditional element selection
 * - zeros_like / ones_like / empty_like / full_like: Create tensors matching shape
 * - clone / detach / contiguous / copy: Tensor copying and memory layout
 * - diag / diagonal: Diagonal extraction and construction
 * - triu / tril: Triangular extraction
 * - flip / fliplr / flipud / rot90: Tensor reversal and rotation
 * - roll: Circular shift
 * - unique: Unique elements, with optional index, inverse, counts and axis
 * - searchsorted: Binary search in sorted array
 * - pad: Pad tensor with values
 * - moveaxis / swapaxes: Rearrange axes
 * - broadcast_to: Explicit broadcasting
 * - atleast_1d / atleast_2d: Promote tensors to a minimum rank
 * - scatter: Scatter values at indices
 * - booleanIndex / fancyIndex / index_select: Advanced indexing
 * - isin: Membership test
 * - cross: Batched cross product of 3-element vectors
 * - takeAlongAxis / putAlongAxis: Index along an axis with per-axis index tensors
 * - nonzero / argwhere / countNonzero: Locate and count nonzero elements
 * - meshgrid: Coordinate grids from coordinate vectors
 * - nansum / nanmean / nanstd / nanmin / nanmax: NaN-aware reductions
 * - histogram / bincount: Binning operations
 *
 * Functions that return a new tensor never alias their input buffer unless the
 * documentation says the result is a view (`moveaxis`, `swapaxes`,
 * `broadcast_to`, `atleast_1d` / `atleast_2d`, and `contiguous` on an already
 * contiguous tensor).
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import type { Axis, DType, Shape, TypedArray } from "../../core";
import {
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
import { promoteTypes } from "../../core/utils/dtype_utils";
import { transpose } from "../tensor/shape";
import { isContiguous } from "../tensor/strides";
import { computeStrides, Tensor } from "../tensor/Tensor";
import {
  readAsNumber,
  readElement,
  readNumbers,
  requireNumericData,
  roundHalfResult,
} from "./_internal";
import { dispatchTernary } from "./device_dispatch";
import { RADIX_SORT_THRESHOLD, radixArgsortF64 } from "./radix";

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
 * Convert a finite number to a BigInt by truncating toward zero.
 */
function numberToBigInt(value: number): bigint {
  if (!Number.isFinite(value)) {
    throw new InvalidParameterError(
      `Cannot store the non-finite value ${value} in an int64 tensor`,
      "value",
      value
    );
  }
  return BigInt(Math.trunc(value));
}

/**
 * Write a single element from `src[srcIdx]` into `out[idx]`. int64 buffers
 * (BigInt) and number buffers are converted into each other (truncating
 * toward zero) instead of silently dropping the write.
 */
function writeElement(out: TypedArray, idx: number, src: TypedArray, srcIdx: number): void {
  if (out instanceof BigInt64Array) {
    out[idx] =
      src instanceof BigInt64Array
        ? getBigIntElement(src, srcIdx)
        : numberToBigInt(getNumericElement(src, srcIdx));
  } else {
    out[idx] =
      src instanceof BigInt64Array
        ? Number(getBigIntElement(src, srcIdx))
        : getNumericElement(src, srcIdx);
  }
}

/**
 * Fill a TypedArray (or a range of it) with a value, converting between
 * number and BigInt as the buffer requires.
 */
function fillRange(arr: TypedArray, value: number | bigint, start?: number, end?: number): void {
  if (arr instanceof BigInt64Array) {
    arr.fill(typeof value === "bigint" ? value : numberToBigInt(value), start, end);
  } else {
    arr.fill(typeof value === "bigint" ? Number(value) : value, start, end);
  }
}

/**
 * Convert a user supplied fill value into the value stored for `dtype`:
 * booleans become 0/1 (NaN counts as true, like NumPy), int64 gets a BigInt and
 * the integer dtypes reject NaN and infinities.
 */
function fillValueFor(
  dtype: Exclude<DType, "string">,
  value: number | bigint,
  fnName: string
): number | bigint {
  if (dtype === "int64") {
    if (typeof value === "bigint") return value;
    if (!Number.isFinite(value)) {
      throw new InvalidParameterError(
        `${fnName} cannot store ${value} in an int64 tensor`,
        "fillValue",
        value
      );
    }
    return BigInt(Math.trunc(value));
  }
  const n = typeof value === "bigint" ? Number(value) : value;
  if (dtype === "bool") return n !== 0 ? 1 : 0;
  if ((dtype === "int32" || dtype === "uint8") && !Number.isFinite(n)) {
    throw new InvalidParameterError(
      `${fnName} cannot store ${n} in a ${dtype} tensor`,
      "fillValue",
      value
    );
  }
  return n;
}

/**
 * Copy elements in row-major output order. Output element `i` of a tensor with
 * the given shape is read from `src[start + sum(coord[d] * steps[d])]`. Steps may be
 * zero (broadcast) or negative (reversed axes). Walks the source with an
 * allocation-free odometer.
 */
function strideCopy<V>(
  src: ArrayLike<V>,
  out: { [i: number]: V },
  shape: readonly number[],
  steps: readonly number[],
  start: number
): void {
  const ndim = shape.length;
  const size = shapeToSize(shape as Shape);
  if (size === 0) return;
  if (ndim === 0) {
    out[0] = src[start] as V;
    return;
  }
  const inner = shape[ndim - 1] as number;
  const innerStep = steps[ndim - 1] as number;
  const outer = size / inner;
  const coords = new Int32Array(ndim);
  let base = start;
  let pos = 0;
  for (let o = 0; o < outer; o++) {
    let idx = base;
    for (let i = 0; i < inner; i++) {
      out[pos++] = src[idx] as V;
      idx += innerStep;
    }
    for (let d = ndim - 2; d >= 0; d--) {
      const step = steps[d] as number;
      const dim = shape[d] as number;
      const c = (coords[d] as number) + 1;
      base += step;
      if (c < dim) {
        coords[d] = c;
        break;
      }
      coords[d] = 0;
      base -= step * dim;
    }
  }
}

/** {@link strideCopy} for buffers whose element type is only known at runtime. */
function copyTyped(
  src: TypedArray,
  out: TypedArray,
  shape: readonly number[],
  steps: readonly number[],
  start: number
): void {
  if (src instanceof BigInt64Array && out instanceof BigInt64Array) {
    strideCopy<bigint>(src, out, shape, steps, start);
  } else if (!(src instanceof BigInt64Array) && !(out instanceof BigInt64Array)) {
    strideCopy<number>(src, out, shape, steps, start);
  } else {
    throw new DTypeError("Internal error: cannot mix int64 and non-int64 buffers");
  }
}

/**
 * Gather `out[c0, ..., cn] = src[maps[0][c0], ..., maps[n][cn]]` in row-major
 * output order. A map entry of -1 leaves the pre-filled output value untouched.
 */
function gatherByAxisMaps<V>(
  src: ArrayLike<V>,
  srcOffset: number,
  srcStrides: readonly number[],
  outShape: readonly number[],
  maps: ReadonlyArray<ArrayLike<number>>,
  out: { [i: number]: V }
): void {
  const nd = outShape.length;
  const size = shapeToSize(outShape as Shape);
  if (size === 0) return;
  if (nd === 0) {
    out[0] = src[srcOffset] as V;
    return;
  }
  const offs: Float64Array[] = [];
  for (let d = 0; d < nd; d++) {
    const dim = outShape[d] as number;
    const stride = srcStrides[d] as number;
    const map = maps[d] as ArrayLike<number>;
    const a = new Float64Array(dim);
    for (let c = 0; c < dim; c++) {
      const s = map[c] as number;
      a[c] = s < 0 ? -1 : s * stride;
    }
    offs.push(a);
  }
  const last = nd - 1;
  const lastDim = outShape[last] as number;
  const lastOffs = offs[last] as Float64Array;
  const outer = size / lastDim;
  const coords = new Int32Array(nd);
  let pos = 0;
  for (let o = 0; o < outer; o++) {
    let base = srcOffset;
    let skip = false;
    for (let d = 0; d < last; d++) {
      const v = (offs[d] as Float64Array)[coords[d] as number] as number;
      if (v < 0) {
        skip = true;
        break;
      }
      base += v;
    }
    if (skip) {
      pos += lastDim;
    } else {
      for (let c = 0; c < lastDim; c++) {
        const v = lastOffs[c] as number;
        if (v >= 0) out[pos] = src[base + v] as V;
        pos++;
      }
    }
    for (let d = last - 1; d >= 0; d--) {
      const c = (coords[d] as number) + 1;
      if (c < (outShape[d] as number)) {
        coords[d] = c;
        break;
      }
      coords[d] = 0;
    }
  }
}

/** {@link gatherByAxisMaps} for buffers whose element type is only known at runtime. */
function gatherTyped(
  src: TypedArray,
  srcOffset: number,
  srcStrides: readonly number[],
  outShape: readonly number[],
  maps: ReadonlyArray<ArrayLike<number>>,
  out: TypedArray
): void {
  if (src instanceof BigInt64Array && out instanceof BigInt64Array) {
    gatherByAxisMaps<bigint>(src, srcOffset, srcStrides, outShape, maps, out);
  } else if (!(src instanceof BigInt64Array) && !(out instanceof BigInt64Array)) {
    gatherByAxisMaps<number>(src, srcOffset, srcStrides, outShape, maps, out);
  } else {
    throw new DTypeError("Internal error: cannot mix int64 and non-int64 buffers");
  }
}

/** Copy `src[start, end)` into `out` at `dstPos` (memcpy for typed arrays). */
function copyRange(
  out: TypedArray,
  src: TypedArray,
  start: number,
  end: number,
  dstPos: number
): void {
  if (out instanceof BigInt64Array && src instanceof BigInt64Array) {
    out.set(src.subarray(start, end), dstPos);
  } else if (!(out instanceof BigInt64Array) && !(src instanceof BigInt64Array)) {
    out.set(src.subarray(start, end), dstPos);
  } else {
    throw new DTypeError("Internal error: cannot mix int64 and non-int64 buffers");
  }
}

/**
 * Copy a tensor's logical elements into a new contiguous buffer (numeric only).
 */
function materializeNumeric(t: Tensor): TypedArray {
  const data = requireNumericData(t.data, "materializeNumeric");
  const size = t.size;
  if (size === 0) {
    return new (dtypeToTypedArrayCtor(t.dtype))(0);
  }
  if (isContiguous(t.shape, t.strides)) {
    return data.slice(t.offset, t.offset + size);
  }
  const out = new (dtypeToTypedArrayCtor(t.dtype))(size);
  copyTyped(data, out, t.shape, t.strides, t.offset);
  return out;
}

/**
 * Logical row-major view of a tensor's elements. Contiguous tensors are
 * returned without copying (the result may alias the tensor's buffer and must
 * not be written to); strided views are gathered into a new buffer.
 */
function logicalData(t: Tensor, opName: string): TypedArray {
  const data = requireNumericData(t.data, opName);
  const size = t.size;
  if (isContiguous(t.shape, t.strides)) {
    return t.offset === 0 && data.length === size ? data : data.subarray(t.offset, t.offset + size);
  }
  return materializeNumeric(t);
}

/**
 * Normalize an axis or list of axes. `undefined` selects every axis. Duplicate
 * axes are rejected (as NumPy does for `flip` and the reductions).
 */
function resolveAxes(
  axis: Axis | readonly Axis[] | undefined,
  ndim: number,
  opName: string
): number[] {
  if (axis === undefined) return Array.from({ length: ndim }, (_, i) => i);
  const list: readonly Axis[] =
    typeof axis === "number" || typeof axis === "string" ? [axis] : axis;
  const seen = new Set<number>();
  const out: number[] = [];
  for (const a of list) {
    const n = normalizeAxis(a, ndim);
    if (seen.has(n)) {
      throw new InvalidParameterError(`${opName} received the repeated axis ${a}`, "axis", axis);
    }
    seen.add(n);
    out.push(n);
  }
  return out;
}

/** Offsets of every coordinate combination of the given axes, in row-major order. */
function axisOffsets(shape: readonly number[], strides: readonly number[]): Float64Array {
  const n = shapeToSize(shape as Shape);
  const out = new Float64Array(n);
  if (n === 0) return out;
  const nd = shape.length;
  const coords = new Int32Array(nd);
  let off = 0;
  for (let i = 0; i < n; i++) {
    out[i] = off;
    for (let d = nd - 1; d >= 0; d--) {
      const step = strides[d] as number;
      const dim = shape[d] as number;
      const c = (coords[d] as number) + 1;
      off += step;
      if (c < dim) {
        coords[d] = c;
        break;
      }
      coords[d] = 0;
      off -= step * dim;
    }
  }
  return out;
}

/**
 * Read an index tensor as validated integers for an axis of size `axisDim`.
 * Negative values count from the end when `allowNegative` is set.
 */
function resolveIndices(
  index: Tensor,
  axisDim: number,
  opName: string,
  allowNegative: boolean
): Int32Array {
  const raw = readNumbers(index, opName, false);
  const n = index.size;
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const v = raw[i] as number;
    if (!Number.isInteger(v)) {
      throw new InvalidParameterError(`${opName} indices must be integers; got ${v}`, "index", v);
    }
    const k = allowNegative && v < 0 ? v + axisDim : v;
    if (k < 0 || k >= axisDim) {
      throw new IndexError(`${opName} index ${v} is out of bounds for an axis of size ${axisDim}`, {
        index: v,
        validRange: [allowNegative ? -axisDim : 0, axisDim - 1],
      });
    }
    out[i] = k;
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

/** Strides of `shape` as seen from `outShape`: broadcast axes get stride 0. */
function broadcastStrides(
  shape: Shape,
  strides: readonly number[],
  outShape: readonly number[]
): number[] {
  const rankDiff = outShape.length - shape.length;
  const out = new Array<number>(outShape.length).fill(0);
  for (let d = 0; d < shape.length; d++) {
    out[d + rankDiff] = (shape[d] ?? 1) === 1 ? 0 : (strides[d] ?? 0);
  }
  return out;
}

// ─── where ──────────────────────────────────────────────────────────────────

/**
 * Conditional element selection: returns elements from `x` where `condition`
 * is true, and from `y` where `condition` is false.
 *
 * All three tensors must be broadcastable to a common shape. The condition is
 * evaluated as boolean: 0 is false, every other value (including NaN) is true.
 * `x` and `y` must have the same dtype.
 *
 * **Complexity**: O(n) where n is the number of elements in the output
 *
 * @param condition - Boolean condition tensor
 * @param x - Values where condition is true
 * @param y - Values where condition is false
 * @returns Tensor with selected elements
 * @throws {DTypeError} If any input has string dtype, or `x` and `y` differ in dtype
 * @throws {ShapeError} If the shapes cannot be broadcast together
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

  const condData = requireNumericData(condition.data, "where");
  const xData = requireNumericData(x.data, "where");
  const yData = requireNumericData(y.data, "where");

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
    // No broadcasting and all three inputs are dense: index the buffers
    // directly instead of tracking three broadcast offsets per element.
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
    } else if (
      !(condData instanceof BigInt64Array) &&
      !(xData instanceof BigInt64Array) &&
      !(yData instanceof BigInt64Array) &&
      !(outData instanceof BigInt64Array)
    ) {
      for (let i = 0; i < outSize; i++) {
        outData[i] =
          (condData[co + i] as number) !== 0
            ? (xData[xo + i] as number)
            : (yData[yo + i] as number);
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
  } else {
    const cs = broadcastStrides(condition.shape, condition.strides, outShape);
    const xs = broadcastStrides(x.shape, x.strides, outShape);
    const ys = broadcastStrides(y.shape, y.strides, outShape);
    const coord = new Int32Array(nd);
    let co = condition.offset;
    let xo = x.offset;
    let yo = y.offset;
    const numeric =
      !(condData instanceof BigInt64Array) &&
      !(xData instanceof BigInt64Array) &&
      !(yData instanceof BigInt64Array) &&
      !(outData instanceof BigInt64Array);
    for (let i = 0; i < outSize; i++) {
      if (numeric) {
        outData[i] = (condData[co] as number) !== 0 ? (xData[xo] as number) : (yData[yo] as number);
      } else if (readAsNumber(condData, co) !== 0) {
        writeElement(outData, i, xData, xo);
      } else {
        writeElement(outData, i, yData, yo);
      }
      for (let d = nd - 1; d >= 0; d--) {
        const dim = outShape[d] as number;
        const c = (coord[d] as number) + 1;
        const cStep = cs[d] as number;
        const xStep = xs[d] as number;
        const yStep = ys[d] as number;
        co += cStep;
        xo += xStep;
        yo += yStep;
        if (c < dim) {
          coord[d] = c;
          break;
        }
        coord[d] = 0;
        co -= cStep * dim;
        xo -= xStep * dim;
        yo -= yStep * dim;
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

// ─── zeros_like / ones_like / empty_like / full_like ────────────────────────

/** Options shared by the `*_like` creation functions. */
export interface LikeOptions {
  /** Result dtype. Defaults to the dtype of the reference tensor. */
  readonly dtype?: DType;
}

/**
 * Create a tensor of zeros with the same shape as the input tensor.
 *
 * @param t - Reference tensor
 * @param options - `dtype` overrides the result dtype (default: the dtype of `t`)
 * @returns New tensor filled with zeros (empty strings for string dtype)
 *
 * @example
 * ```ts
 * const a = tensor([[1, 2], [3, 4]]);
 * const z = zerosLike(a); // [[0, 0], [0, 0]]
 * ```
 * @deprecated Prefer {@link zerosLike}.
 */
export function zeros_like(t: Tensor, options?: LikeOptions): Tensor {
  const target = options?.dtype ?? t.dtype;
  if (target === "string") {
    const data = new Array<string>(t.size).fill("");
    return Tensor.fromStringArray({
      data,
      shape: [...t.shape],
      device: t.device,
    });
  }
  const dtype = ensureNonStringDType(target);
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
 * Create a tensor of ones with the same shape as the input tensor.
 *
 * @param t - Reference tensor
 * @param options - `dtype` overrides the result dtype (default: the dtype of `t`)
 * @returns New tensor filled with ones
 * @throws {DTypeError} If the resulting dtype is string
 *
 * @example
 * ```ts
 * const a = tensor([[1, 2], [3, 4]]);
 * const o = onesLike(a); // [[1, 1], [1, 1]]
 * ```
 * @deprecated Prefer {@link onesLike}.
 */
export function ones_like(t: Tensor, options?: LikeOptions): Tensor {
  const target = options?.dtype ?? t.dtype;
  if (target === "string") {
    throw new DTypeError("ones_like is not defined for string dtype");
  }
  const dtype = ensureNonStringDType(target);
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const data = new Ctor(t.size);
  fillRange(data, dtype === "int64" ? 1n : 1);
  return Tensor.fromTypedArray({
    data,
    shape: [...t.shape],
    dtype,
    device: t.device,
  });
}

/**
 * Create a zero-filled tensor with the same shape as the input tensor.
 * JavaScript typed arrays are always zero-initialized, so unlike NumPy's
 * `empty_like` the contents are well defined.
 *
 * @param t - Reference tensor
 * @param options - `dtype` overrides the result dtype (default: the dtype of `t`)
 * @returns New tensor (zero-initialized)
 * @deprecated Prefer {@link emptyLike}.
 */
export function empty_like(t: Tensor, options?: LikeOptions): Tensor {
  return zeros_like(t, options);
}

/**
 * Create a tensor filled with a specified value, matching the shape of the input.
 *
 * The value is converted to the result dtype: `bool` stores 1 for any non-zero
 * value (NaN included), integer dtypes truncate toward zero and reject NaN and
 * infinities, and `int64` stores a BigInt.
 *
 * @param t - Reference tensor
 * @param fillValue - Value to fill the tensor with
 * @param options - `dtype` overrides the result dtype (default: the dtype of `t`)
 * @returns New tensor filled with the specified value
 * @throws {DTypeError} If the resulting dtype is string
 * @throws {InvalidParameterError} If a non-finite value is stored in an integer dtype
 *
 * @example
 * ```ts
 * const a = tensor([[1, 2], [3, 4]]);
 * const f = fullLike(a, 7); // [[7, 7], [7, 7]]
 * ```
 * @deprecated Prefer {@link fullLike}.
 */
export function full_like(t: Tensor, fillValue: number | bigint, options?: LikeOptions): Tensor {
  const target = options?.dtype ?? t.dtype;
  if (target === "string") {
    throw new DTypeError("full_like is not defined for string dtype");
  }
  const dtype = ensureNonStringDType(target);
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const data = new Ctor(t.size);
  fillRange(data, fillValueFor(dtype, fillValue, "full_like"));
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
 * The returned tensor shares no data with the original and is always
 * contiguous with offset 0, even when the input is a strided view.
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
    let out: string[];
    if (isContiguous(t.shape, t.strides)) {
      out = data.slice(t.offset, t.offset + size);
    } else {
      out = new Array<string>(size);
      strideCopy<string>(data, out, t.shape, t.strides, t.offset);
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
 * Equivalent to {@link clone}: plain tensors carry no graph.
 *
 * @param t - Input tensor
 * @returns Detached copy of the tensor
 */
export function detach(t: Tensor): Tensor {
  return clone(t);
}

/**
 * Return a contiguous tensor. A tensor that is already contiguous and starts at
 * offset 0 is returned as is (no copy); any other tensor is materialized into a
 * new contiguous copy.
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
 * Create an explicit deep copy of a tensor (alias for {@link clone}).
 *
 * @param t - Input tensor
 * @returns Deep copy of the tensor
 */
export function copy(t: Tensor): Tensor {
  return clone(t);
}

// ─── diag / diagonal ────────────────────────────────────────────────────────

function assertIntegerOffset(k: number, fnName: string): void {
  if (!Number.isInteger(k)) {
    throw new InvalidParameterError(`${fnName} offset k must be an integer; got ${k}`, "k", k);
  }
}

/**
 * Extract a diagonal or construct a diagonal matrix.
 *
 * - If input is 1-D, returns a 2-D square matrix with the input on the k-th diagonal.
 * - If input is 2-D, returns a copy of the k-th diagonal as a 1-D tensor.
 *
 * @param t - Input tensor (1-D or 2-D)
 * @param k - Diagonal offset (0 = main, positive = above, negative = below)
 * @returns Diagonal tensor
 * @throws {ShapeError} If the input is not 1-D or 2-D
 * @throws {InvalidParameterError} If `k` is not an integer
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
  assertIntegerOffset(k, "diag");
  if (t.ndim === 1) {
    return constructDiagonal(t, k);
  }
  if (t.ndim === 2) {
    return diagonal(t, k);
  }
  throw new ShapeError(`diag requires 1-D or 2-D input; got ${t.ndim}-D`);
}

/**
 * Extract the k-th diagonal from a 2-D tensor. The result is a copy.
 *
 * @param t - Input 2-D tensor
 * @param k - Diagonal offset (0 = main, positive = above, negative = below)
 * @returns 1-D tensor of diagonal elements (empty if `|k|` exceeds the matrix size)
 * @throws {ShapeError} If the input is not 2-D
 * @throws {InvalidParameterError} If `k` is not an integer
 */
export function diagonal(t: Tensor, k = 0): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("diagonal is not defined for string dtype");
  }
  if (t.ndim !== 2) {
    throw new ShapeError(`diagonal requires 2-D input; got ${t.ndim}-D`);
  }
  assertIntegerOffset(k, "diagonal");

  const dtype = ensureNonStringDType(t.dtype);
  const rows = t.shape[0] ?? 0;
  const cols = t.shape[1] ?? 0;
  const rowStride = t.strides[0] ?? 0;
  const colStride = t.strides[1] ?? 0;

  const startRow = k >= 0 ? 0 : -k;
  const startCol = k >= 0 ? k : 0;

  const diagLen = Math.max(0, Math.min(rows - startRow, cols - startCol));
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const outData = new Ctor(diagLen);
  const data = requireNumericData(t.data, "diagonal");

  copyTyped(
    data,
    outData,
    [diagLen],
    [rowStride + colStride],
    t.offset + startRow * rowStride + startCol * colStride
  );

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
  const outData = new Ctor(shapeToSize([size, size]));
  const data = requireNumericData(t.data, "diag");
  const stride = t.strides[0] ?? 1;

  for (let i = 0; i < n; i++) {
    const srcOff = t.offset + i * stride;
    const r = k >= 0 ? i : i + absK;
    const c = k >= 0 ? i + absK : i;
    writeElement(outData, r * size + c, data, srcOff);
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
 * Shared implementation of {@link triu} and {@link tril}. Operates on the last
 * two axes; any leading axes are treated as a batch.
 */
function triangle(t: Tensor, k: number, upper: boolean, fnName: string): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError(`${fnName} is not defined for string dtype`);
  }
  if (t.ndim < 2) {
    throw new ShapeError(`${fnName} requires at least 2-D input; got ${t.ndim}-D`);
  }
  assertIntegerOffset(k, fnName);

  const dtype = ensureNonStringDType(t.dtype);
  const ndim = t.ndim;
  const rows = t.shape[ndim - 2] ?? 0;
  const cols = t.shape[ndim - 1] ?? 0;
  const rowStride = t.strides[ndim - 2] ?? 0;
  const colStride = t.strides[ndim - 1] ?? 0;
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const outData = new Ctor(t.size);
  const data = requireNumericData(t.data, fnName);

  const lead = t.shape.slice(0, ndim - 2);
  const bases = axisOffsets(lead, t.strides.slice(0, ndim - 2));
  const matSize = rows * cols;

  for (let b = 0; b < bases.length; b++) {
    const srcBase = t.offset + (bases[b] as number);
    const dstBase = b * matSize;
    for (let r = 0; r < rows; r++) {
      // triu keeps c >= r + k, tril keeps c <= r + k.
      const cStart = upper ? Math.max(0, r + k) : 0;
      const cEnd = upper ? cols : Math.min(cols, r + k + 1);
      for (let c = cStart; c < cEnd; c++) {
        writeElement(
          outData,
          dstBase + r * cols + c,
          data,
          srcBase + r * rowStride + c * colStride
        );
      }
    }
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: [...t.shape],
    dtype,
    device: t.device,
  });
}

/**
 * Return the upper triangular part of a tensor, with elements below the k-th
 * diagonal zeroed. For tensors with more than two axes the last two axes are
 * treated as the matrix and the leading axes as a batch.
 *
 * @param t - Input tensor with at least 2 dimensions
 * @param k - Diagonal offset (0 = main diagonal)
 * @returns Upper triangular tensor
 * @throws {ShapeError} If the input has fewer than 2 dimensions
 * @throws {InvalidParameterError} If `k` is not an integer
 *
 * @example
 * ```ts
 * triu(tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]));
 * // [[1, 2, 3], [0, 5, 6], [0, 0, 9]]
 * ```
 */
export function triu(t: Tensor, k = 0): Tensor {
  return triangle(t, k, true, "triu");
}

/**
 * Return the lower triangular part of a tensor, with elements above the k-th
 * diagonal zeroed. For tensors with more than two axes the last two axes are
 * treated as the matrix and the leading axes as a batch.
 *
 * @param t - Input tensor with at least 2 dimensions
 * @param k - Diagonal offset (0 = main diagonal)
 * @returns Lower triangular tensor
 * @throws {ShapeError} If the input has fewer than 2 dimensions
 * @throws {InvalidParameterError} If `k` is not an integer
 *
 * @example
 * ```ts
 * tril(tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]));
 * // [[1, 0, 0], [4, 5, 0], [7, 8, 9]]
 * ```
 */
export function tril(t: Tensor, k = 0): Tensor {
  return triangle(t, k, false, "tril");
}

// ─── flip / fliplr / flipud ─────────────────────────────────────────────────

/**
 * Reverse the order of elements along the given axes. The result is a copy.
 *
 * @param t - Input tensor
 * @param axes - Axis or axes along which to flip. If undefined, flip all axes.
 * @returns Flipped tensor
 * @throws {InvalidParameterError} If an axis is out of range or listed twice
 *
 * @example
 * ```ts
 * flip(tensor([1, 2, 3]));     // [3, 2, 1]
 * flip(tensor([[1, 2], [3, 4]]), [1]); // [[2, 1], [4, 3]]
 * ```
 */
export function flip(t: Tensor, axes?: Axis | readonly Axis[]): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("flip is not defined for string dtype");
  }
  const dtype = ensureNonStringDType(t.dtype);
  const ndim = t.ndim;
  const flipAxes = resolveAxes(axes, ndim, "flip");

  const Ctor = dtypeToTypedArrayCtor(dtype);
  const outData = new Ctor(t.size);
  const data = requireNumericData(t.data, "flip");

  // Flipping an axis walks it backwards: start at its last element and use a
  // negated stride. Works for any strided source without per-element indexing.
  const flipped = new Uint8Array(ndim);
  for (const ax of flipAxes) flipped[ax] = 1;
  const steps = new Array<number>(ndim);
  let start = t.offset;
  for (let d = 0; d < ndim; d++) {
    const dim = t.shape[d] ?? 0;
    const stride = t.strides[d] ?? 0;
    if (flipped[d] === 1 && dim > 0) {
      steps[d] = -stride;
      start += (dim - 1) * stride;
    } else {
      steps[d] = stride;
    }
  }
  copyTyped(data, outData, t.shape, steps, start);

  return Tensor.fromTypedArray({
    data: outData,
    shape: [...t.shape],
    dtype,
    device: t.device,
  });
}

/**
 * Flip a tensor left-right (reverse the order of columns, axis 1).
 *
 * @param t - Input tensor with at least 2 dimensions
 * @returns Horizontally flipped tensor
 * @throws {ShapeError} If the input has fewer than 2 dimensions
 * @deprecated Prefer {@link flipLr}.
 */
export function fliplr(t: Tensor): Tensor {
  if (t.ndim < 2) {
    throw new ShapeError(`fliplr requires at least 2-D input; got ${t.ndim}-D`);
  }
  return flip(t, [1]);
}

/**
 * Flip a tensor up-down (reverse the order of rows, axis 0).
 *
 * @param t - Input tensor with at least 1 dimension
 * @returns Vertically flipped tensor
 * @throws {ShapeError} If the input is a scalar
 * @deprecated Prefer {@link flipUd}.
 */
export function flipud(t: Tensor): Tensor {
  if (t.ndim < 1) {
    throw new ShapeError(`flipud requires at least 1-D input; got ${t.ndim}-D`);
  }
  return flip(t, [0]);
}

// ─── roll ───────────────────────────────────────────────────────────────────

function assertIntegerShift(shift: number): void {
  if (!Number.isInteger(shift)) {
    throw new InvalidParameterError(`roll shift must be an integer; got ${shift}`, "shift", shift);
  }
}

/**
 * Roll `outer` independent blocks of `axDim * inner` elements along their
 * middle axis by `s` positions (0 <= s < axDim), using block copies.
 */
function rollBlocks(
  src: TypedArray,
  out: TypedArray,
  outer: number,
  axDim: number,
  inner: number,
  s: number
): void {
  const block = axDim * inner;
  const split = (axDim - s) * inner;
  for (let o = 0; o < outer; o++) {
    const base = o * block;
    copyRange(out, src, base + split, base + block, base);
    copyRange(out, src, base, base + split, base + s * inner);
  }
}

function rollAxis(t: Tensor, ax: number, shift: number): Tensor {
  const dtype = ensureNonStringDType(t.dtype);
  const axDim = t.shape[ax] ?? 0;
  if (axDim === 0 || t.size === 0) return clone(t);
  const s = ((shift % axDim) + axDim) % axDim;
  const src = logicalData(t, "roll");
  const out = new (dtypeToTypedArrayCtor(dtype))(t.size);
  let inner = 1;
  for (let d = ax + 1; d < t.ndim; d++) inner *= t.shape[d] ?? 1;
  rollBlocks(src, out, t.size / (axDim * inner), axDim, inner, s);
  return Tensor.fromTypedArray({ data: out, shape: [...t.shape], dtype, device: t.device });
}

/**
 * Roll tensor elements along the given axis.
 * Elements that roll beyond the last position are re-introduced at the first.
 * The result is a copy.
 *
 * With several axes, `shift` is either one number applied to every axis or a
 * list with one shift per axis (NumPy semantics). Without `axis` the tensor is
 * rolled as if flattened and keeps its shape.
 *
 * @param t - Input tensor
 * @param shift - Number of places to shift (positive = toward end)
 * @param axis - Axis or axes along which to roll (default: roll the flattened tensor)
 * @returns Rolled tensor
 * @throws {InvalidParameterError} If a shift is not an integer, or `shift` and `axis` lengths disagree
 *
 * @example
 * ```ts
 * roll(tensor([1, 2, 3, 4, 5]), 2);  // [4, 5, 1, 2, 3]
 * ```
 */
export function roll(
  t: Tensor,
  shift: number | readonly number[],
  axis?: Axis | readonly Axis[]
): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("roll is not defined for string dtype");
  }
  const dtype = ensureNonStringDType(t.dtype);
  const shifts: readonly number[] = typeof shift === "number" ? [shift] : shift;
  for (const s of shifts) assertIntegerShift(s);

  if (axis === undefined) {
    if (shifts.length !== 1) {
      throw new InvalidParameterError(
        "roll needs a single shift when axis is omitted",
        "shift",
        shift
      );
    }
    const total = t.size;
    if (total === 0) return clone(t);
    const s = (((shifts[0] ?? 0) % total) + total) % total;
    const src = logicalData(t, "roll");
    const out = new (dtypeToTypedArrayCtor(dtype))(total);
    rollBlocks(src, out, 1, total, 1, s);
    return Tensor.fromTypedArray({ data: out, shape: [...t.shape], dtype, device: t.device });
  }

  const axes: readonly Axis[] =
    typeof axis === "number" || typeof axis === "string" ? [axis] : axis;
  const count = Math.max(shifts.length, axes.length);
  if (
    (shifts.length !== count && shifts.length !== 1) ||
    (axes.length !== count && axes.length !== 1)
  ) {
    throw new InvalidParameterError(
      `roll shift and axis must have the same length; got ${shifts.length} and ${axes.length}`,
      "shift",
      shift
    );
  }

  // Shifts along a repeated axis add up, as in NumPy.
  const totals = new Array<number>(t.ndim).fill(0);
  for (let i = 0; i < count; i++) {
    const ax = normalizeAxis(axes[axes.length === 1 ? 0 : i] ?? 0, t.ndim);
    totals[ax] = (totals[ax] ?? 0) + (shifts[shifts.length === 1 ? 0 : i] ?? 0);
  }

  let result: Tensor | null = null;
  for (let ax = 0; ax < t.ndim; ax++) {
    const total = totals[ax] ?? 0;
    const dim = t.shape[ax] ?? 0;
    if (dim === 0 || total % dim === 0) continue;
    result = rollAxis(result ?? t, ax, total);
  }
  return result ?? clone(t);
}

// ─── pad ────────────────────────────────────────────────────────────────────

/**
 * Padding modes accepted by {@link pad}.
 *
 * - `"constant"`: fill with a constant value
 * - `"reflect"`: mirror the tensor without repeating the edge value
 * - `"symmetric"`: mirror the tensor including the edge value
 * - `"replicate"` (alias `"edge"`): repeat the edge value
 * - `"circular"` (alias `"wrap"`): wrap around to the opposite side
 */
export type PadMode =
  | "constant"
  | "reflect"
  | "symmetric"
  | "replicate"
  | "edge"
  | "circular"
  | "wrap";

/** Padding amounts accepted by {@link pad}. */
export type PadWidth =
  | number
  | readonly [number, number]
  | ReadonlyArray<readonly [number, number]>;

type ResolvedPadMode = "constant" | "reflect" | "symmetric" | "replicate" | "circular";

function resolvePadMode(mode: PadMode): ResolvedPadMode {
  switch (mode) {
    case "constant":
    case "reflect":
    case "symmetric":
    case "replicate":
    case "circular":
      return mode;
    case "edge":
      return "replicate";
    case "wrap":
      return "circular";
    default:
      throw new InvalidParameterError(
        `Unknown pad mode '${String(mode)}'; expected 'constant', 'reflect', 'symmetric', ` +
          "'replicate' (or 'edge') or 'circular' (or 'wrap')",
        "mode",
        mode
      );
  }
}

function normalizePadWidth(padWidth: PadWidth, ndim: number): Array<[number, number]> {
  const checkPair = (before: unknown, after: unknown, label: string): [number, number] => {
    if (
      typeof before !== "number" ||
      typeof after !== "number" ||
      !Number.isInteger(before) ||
      !Number.isInteger(after)
    ) {
      throw new InvalidParameterError(
        `${label} must be [before, after] integers`,
        "padWidth",
        padWidth
      );
    }
    if (before < 0 || after < 0) {
      throw new InvalidParameterError("padWidth values must be non-negative", "padWidth", padWidth);
    }
    return [before, after];
  };

  if (typeof padWidth === "number") {
    const pair = checkPair(padWidth, padWidth, "padWidth");
    return Array.from({ length: ndim }, () => [pair[0], pair[1]]);
  }
  if (typeof padWidth[0] === "number") {
    if (padWidth.length !== 2) {
      throw new InvalidParameterError("padWidth must be [before, after]", "padWidth", padWidth);
    }
    const pair = checkPair(padWidth[0], padWidth[1], "padWidth");
    return Array.from({ length: ndim }, () => [pair[0], pair[1]]);
  }
  const pairs = padWidth as ReadonlyArray<readonly [number, number]>;
  if (pairs.length !== ndim && pairs.length !== 1) {
    throw new InvalidParameterError(
      `padWidth length ${pairs.length} must match tensor ndim ${ndim}`,
      "padWidth",
      padWidth
    );
  }
  const out: Array<[number, number]> = [];
  for (let d = 0; d < ndim; d++) {
    const pw = pairs[pairs.length === 1 ? 0 : d];
    if (pw === undefined || pw.length !== 2) {
      throw new InvalidParameterError(
        `padWidth[${d}] must be [before, after]`,
        "padWidth",
        padWidth
      );
    }
    out.push(checkPair(pw[0], pw[1], `padWidth[${d}]`));
  }
  return out;
}

/** Source coordinate for output coordinate `c` (already shifted by the leading pad); -1 means fill. */
function padSourceIndex(c: number, dim: number, mode: ResolvedPadMode): number {
  if (c >= 0 && c < dim) return c;
  switch (mode) {
    case "constant":
      return -1;
    case "replicate":
      return c < 0 ? 0 : dim - 1;
    case "circular":
      return ((c % dim) + dim) % dim;
    case "reflect": {
      if (dim === 1) return 0;
      const period = 2 * (dim - 1);
      const m = ((c % period) + period) % period;
      return m < dim ? m : period - m;
    }
    case "symmetric": {
      const period = 2 * dim;
      const m = ((c % period) + period) % period;
      return m < dim ? m : period - 1 - m;
    }
  }
}

/**
 * Pad a tensor.
 *
 * @param t - Input tensor
 * @param padWidth - Padding as `[[before0, after0], [before1, after1], ...]` with one
 *   pair per axis. A single pair, or a single-element list of pairs, applies to every
 *   axis, and a number pads every side of every axis by that amount.
 * @param mode - Padding mode: `'constant'`, `'reflect'`, `'symmetric'`, `'replicate'`
 *   (alias `'edge'`) or `'circular'` (alias `'wrap'`)
 * @param constantValue - Fill value for constant mode (default: 0). Converted to the
 *   dtype of `t` (booleans store 1 for any non-zero value)
 * @returns Padded tensor
 * @throws {InvalidParameterError} If the mode is unknown, a pad width is negative or not an
 *   integer, or a non-constant mode would extend an empty axis
 *
 * @example
 * ```ts
 * pad(tensor([1, 2, 3]), [[2, 1]]); // [0, 0, 1, 2, 3, 0]
 * ```
 */
export function pad(
  t: Tensor,
  padWidth: PadWidth,
  mode: PadMode = "constant",
  constantValue: number | bigint = 0
): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("pad is not defined for string dtype");
  }
  const dtype = ensureNonStringDType(t.dtype);
  const resolvedMode = resolvePadMode(mode);
  const widths = normalizePadWidth(padWidth, t.ndim);

  const outShape = t.shape.map((dim, d) => {
    const pw = widths[d] as [number, number];
    return dim + pw[0] + pw[1];
  });

  if (resolvedMode !== "constant") {
    for (let d = 0; d < t.ndim; d++) {
      const pw = widths[d] as [number, number];
      if (t.shape[d] === 0 && pw[0] + pw[1] > 0) {
        throw new InvalidParameterError(
          `Cannot pad the empty axis ${d} with mode '${mode}'; use mode 'constant'`,
          "mode",
          mode
        );
      }
    }
  }

  const outSize = shapeToSize(outShape);
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const outData = new Ctor(outSize);
  const data = requireNumericData(t.data, "pad");

  if (resolvedMode === "constant") {
    const fill = fillValueFor(dtype, constantValue, "pad");
    if (fill !== 0 && fill !== 0n) fillRange(outData, fill);
  }

  const maps = outShape.map((outDim, d) => {
    const before = (widths[d] as [number, number])[0];
    const dim = t.shape[d] ?? 0;
    const map = new Int32Array(outDim);
    for (let c = 0; c < outDim; c++) map[c] = padSourceIndex(c - before, dim, resolvedMode);
    return map;
  });
  gatherTyped(data, t.offset, t.strides, outShape, maps, outData);

  return Tensor.fromTypedArray({
    data: outData,
    shape: outShape,
    dtype,
    device: t.device,
  });
}

// ─── moveaxis / swapaxes ────────────────────────────────────────────────────

/**
 * Move axes of a tensor to new positions. Other axes keep their relative order.
 * The result is a zero-copy view that shares the input's buffer.
 *
 * @param t - Input tensor
 * @param source - Original positions of axes to move
 * @param destination - Destination positions for each axis
 * @returns Tensor view with moved axes
 * @throws {InvalidParameterError} If the lists differ in length, an axis is out of range,
 *   or an axis is repeated in `source` or in `destination`
 *
 * @example
 * ```ts
 * moveaxis(zeros([2, 3, 4]), 0, -1).shape; // [3, 4, 2]
 * ```
 */
export function moveaxis(
  t: Tensor,
  source: number | readonly number[],
  destination: number | readonly number[]
): Tensor {
  const src: readonly number[] = typeof source === "number" ? [source] : source;
  const dst: readonly number[] = typeof destination === "number" ? [destination] : destination;

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
  if (new Set(normSrc).size !== normSrc.length) {
    throw new InvalidParameterError("repeated axis in source", "source", source);
  }
  if (new Set(normDst).size !== normDst.length) {
    throw new InvalidParameterError("repeated axis in destination", "destination", destination);
  }

  const order = new Array<number>(ndim).fill(-1);
  for (let i = 0; i < normSrc.length; i++) {
    order[normDst[i] ?? 0] = normSrc[i] ?? 0;
  }

  const usedSrc = new Set(normSrc);
  const remaining: number[] = [];
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
 * Swap two axes of a tensor. The result is a zero-copy view that shares the
 * input's buffer.
 *
 * @param t - Input tensor
 * @param axis1 - First axis
 * @param axis2 - Second axis
 * @returns Tensor view with swapped axes
 * @throws {InvalidParameterError} If an axis is out of range
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
  // Axis permutation is a pure stride/shape reorder, so it returns a zero-copy
  // view (like `transpose`) that shares the buffer instead of gathering elements.
  const outShape = order.map((ax) => t.shape[ax] ?? 0);
  const outStrides = order.map((ax) => t.strides[ax] ?? 0);

  if (t.isDeviceTensor || t.dtype === "string") {
    return t.view(outShape, outStrides, t.offset);
  }

  const dtype = ensureNonStringDType(t.dtype);
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
 * The result is a zero-copy view: broadcast axes get stride 0, so all elements
 * along them alias the same buffer entry (like NumPy's `broadcast_to`). Call
 * {@link clone} to get independent memory.
 *
 * @param t - Input tensor
 * @param shape - Target shape
 * @returns Broadcast view of the tensor
 * @throws {ShapeError} If the target has fewer dimensions or an axis cannot be broadcast
 * @throws {InvalidParameterError} If the target shape has a negative or non-integer entry
 * @deprecated Prefer {@link broadcastTo}.
 */
export function broadcast_to(t: Tensor, shape: Shape): Tensor {
  for (const dim of shape) {
    if (!Number.isInteger(dim) || dim < 0) {
      throw new InvalidParameterError(
        `broadcast_to shape must contain non-negative integers; got [${shape}]`,
        "shape",
        shape
      );
    }
  }
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
        `Cannot broadcast shape [${t.shape}] to [${shape}]: ` +
          `axis ${i + rankDiff} has size ${tgtDim} but the input has size ${srcDim}`
      );
    }
  }

  // Broadcast dimensions (size-1 or newly-added leading axes) get stride 0;
  // the other axes keep the source stride.
  const outStrides = new Array<number>(shape.length);
  for (let i = 0; i < shape.length; i++) {
    if (i < rankDiff) {
      outStrides[i] = 0;
    } else {
      const srcDim = t.shape[i - rankDiff] ?? 1;
      outStrides[i] = srcDim === 1 ? 0 : (t.strides[i - rankDiff] ?? 0);
    }
  }

  if (t.dtype === "string") {
    const sdata = t.data;
    if (!Array.isArray(sdata)) {
      throw new DeepboxError("Internal error: string dtype but non-array data");
    }
    return Tensor.fromStringArray({
      data: sdata,
      shape: [...shape],
      device: t.device,
      offset: t.offset,
      strides: outStrides,
    });
  }

  const dtype = ensureNonStringDType(t.dtype);
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
 * Scatter values into a copy of a tensor at specified indices along an axis.
 * This is the inverse of gather: for a 3-D tensor and `dim = 0` the result
 * satisfies `out[index[i][j][k]][j][k] = src[i][j][k]`.
 *
 * `t`, `index` and `src` must have the same number of dimensions, `index` must
 * not be larger than `src` along any axis, and not larger than `t` along every
 * axis except `dim`. Negative indices count from the end of the axis. When an
 * index repeats, the last write wins. Values of `src` are converted to the
 * dtype of `t`.
 *
 * @param t - Input tensor (left unchanged)
 * @param dim - Dimension along which to scatter
 * @param index - Integer-valued index tensor
 * @param src - Source values tensor
 * @returns New tensor with scattered values
 * @throws {ShapeError} If the shapes are inconsistent
 * @throws {IndexError} If an index is out of bounds for `dim`
 * @throws {InvalidParameterError} If an index is not an integer
 */
export function scatter(t: Tensor, dim: Axis, index: Tensor, src: Tensor): Tensor {
  if (t.dtype === "string" || src.dtype === "string") {
    throw new DTypeError("scatter is not defined for string dtype");
  }
  if (index.dtype === "string") {
    throw new DTypeError("scatter index must be numeric");
  }
  const ndim = t.ndim;
  const ax = normalizeAxis(dim, ndim);
  if (index.ndim !== ndim || src.ndim !== ndim) {
    throw new ShapeError(
      `scatter requires t, index and src to have the same number of dimensions; ` +
        `got ${ndim}, ${index.ndim} and ${src.ndim}`
    );
  }
  for (let d = 0; d < ndim; d++) {
    const idxDim = index.shape[d] ?? 0;
    if (idxDim > (src.shape[d] ?? 0)) {
      throw new ShapeError(
        `scatter index shape [${index.shape}] must not exceed src shape [${src.shape}]`
      );
    }
    if (d !== ax && idxDim > (t.shape[d] ?? 0)) {
      throw new ShapeError(
        `scatter index shape [${index.shape}] must not exceed the shape [${t.shape}] of t ` +
          `outside axis ${ax}`
      );
    }
  }

  const result = clone(t);
  if (index.size === 0) return result;

  const outData = requireNumericData(result.data, "scatter");
  const srcData = requireNumericData(src.data, "scatter");
  const idxVals = readNumbers(index, "scatter", false);
  const axDim = t.shape[ax] ?? 0;
  const outSteps = [...computeStrides(result.shape)];
  const axStride = outSteps[ax] ?? 0;
  outSteps[ax] = 0;
  const srcSteps = src.strides;
  const iterShape = index.shape;
  const boolOut = t.dtype === "bool" && outData instanceof Uint8Array ? outData : null;
  const coord = new Int32Array(ndim);
  let outBase = 0;
  let srcOff = src.offset;

  const total = index.size;
  for (let i = 0; i < total; i++) {
    const raw = idxVals[i] as number;
    if (!Number.isInteger(raw)) {
      throw new InvalidParameterError(`scatter indices must be integers; got ${raw}`, "index", raw);
    }
    const k = raw < 0 ? raw + axDim : raw;
    if (k < 0 || k >= axDim) {
      throw new IndexError(
        `scatter index ${raw} is out of bounds for axis ${ax} with size ${axDim}`,
        { index: raw, validRange: [-axDim, axDim - 1] }
      );
    }
    const dst = outBase + k * axStride;
    if (boolOut) {
      boolOut[dst] = readAsNumber(srcData, srcOff) !== 0 ? 1 : 0;
    } else {
      writeElement(outData, dst, srcData, srcOff);
    }
    for (let d = ndim - 1; d >= 0; d--) {
      const dimSize = iterShape[d] as number;
      const oStep = outSteps[d] as number;
      const sStep = srcSteps[d] as number;
      const c = (coord[d] as number) + 1;
      outBase += oStep;
      srcOff += sStep;
      if (c < dimSize) {
        coord[d] = c;
        break;
      }
      coord[d] = 0;
      outBase -= oStep * dimSize;
      srcOff -= sStep * dimSize;
    }
  }

  return result;
}

// ─── NaN-aware reductions ───────────────────────────────────────────────────

/** Count of non-NaN values seen by the last {@link nanSumCompensated} call. */
let lastNanCount = 0;

/**
 * Sum of the non-NaN values of `a[0, n)` using Neumaier compensated summation,
 * so the result does not drift with the number of terms. Sets `lastNanCount`.
 */
function nanSumCompensated(a: ArrayLike<number>, n: number): number {
  let sum = 0;
  let comp = 0;
  let count = 0;
  for (let i = 0; i < n; i++) {
    const v = a[i] as number;
    if (Number.isNaN(v)) continue;
    count++;
    const next = sum + v;
    comp += Math.abs(sum) >= Math.abs(v) ? sum - next + v : v - next + sum;
    sum = next;
  }
  lastNanCount = count;
  // Once the sum is infinite or NaN the compensation term is meaningless.
  return Number.isFinite(sum) ? sum + comp : sum;
}

type NanKernel = (vals: ArrayLike<number>, n: number) => number;

/**
 * Shared driver of the NaN-aware reductions. Always returns float64; the
 * kernel reduces `vals[0, n)` to one number.
 */
function nanReduce(
  t: Tensor,
  axis: Axis | readonly Axis[] | undefined,
  keepdims: boolean,
  opName: string,
  kernel: NanKernel
): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError(`${opName} is not defined for string dtype`);
  }
  const vals = readNumbers(t, opName, false);
  const ndim = t.ndim;
  const redAxes = resolveAxes(axis, ndim, opName).sort((a, b) => a - b);
  const isReduced = new Uint8Array(ndim);
  for (const ax of redAxes) isReduced[ax] = 1;
  const keptAxes: number[] = [];
  for (let d = 0; d < ndim; d++) if (isReduced[d] === 0) keptAxes.push(d);

  const outShape: number[] = [];
  for (let d = 0; d < ndim; d++) {
    if (isReduced[d] === 0) outShape.push(t.shape[d] ?? 0);
    else if (keepdims) outShape.push(1);
  }

  let out: Float64Array;
  if (keptAxes.length === 0) {
    out = new Float64Array([kernel(vals, t.size)]);
  } else {
    const logical = computeStrides(t.shape);
    const keptOffsets = axisOffsets(
      keptAxes.map((d) => t.shape[d] ?? 0),
      keptAxes.map((d) => logical[d] ?? 0)
    );
    const redShape = redAxes.map((d) => t.shape[d] ?? 0);
    const redOffsets = axisOffsets(
      redShape,
      redAxes.map((d) => logical[d] ?? 0)
    );
    const redCount = redOffsets.length;
    out = new Float64Array(keptOffsets.length);
    const lastAxisOnly = redAxes.length === 1 && redAxes[0] === ndim - 1;
    if (lastAxisOnly) {
      // The reduced axis is contiguous in memory: reduce it in place.
      for (let j = 0; j < out.length; j++) {
        const base = keptOffsets[j] as number;
        out[j] = kernel(vals.subarray(base, base + redCount), redCount);
      }
    } else {
      const buf = new Float64Array(redCount);
      for (let j = 0; j < out.length; j++) {
        const base = keptOffsets[j] as number;
        for (let k = 0; k < redCount; k++)
          buf[k] = vals[base + (redOffsets[k] as number)] as number;
        out[j] = kernel(buf, redCount);
      }
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Sum of elements, treating NaN as zero. The sum of an empty or all-NaN slice
 * is 0. Uses compensated summation. The result is always float64.
 *
 * @param t - Input tensor
 * @param axis - Axis or axes to reduce (default: all)
 * @param keepdims - Keep reduced axes as size-1 dimensions
 * @returns float64 tensor of sums
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If an axis is out of range or repeated
 */
export function nansum(t: Tensor, axis?: Axis | readonly Axis[], keepdims = false): Tensor {
  return nanReduce(t, axis, keepdims, "nansum", (vals, n) => nanSumCompensated(vals, n));
}

/**
 * Mean of elements, ignoring NaN. A slice without non-NaN values gives NaN.
 * The result is always float64.
 *
 * @param t - Input tensor
 * @param axis - Axis or axes to reduce (default: all)
 * @param keepdims - Keep reduced axes as size-1 dimensions
 * @returns float64 tensor of means
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If an axis is out of range or repeated
 */
export function nanmean(t: Tensor, axis?: Axis | readonly Axis[], keepdims = false): Tensor {
  return nanReduce(t, axis, keepdims, "nanmean", (vals, n) => {
    const sum = nanSumCompensated(vals, n);
    return lastNanCount === 0 ? Number.NaN : sum / lastNanCount;
  });
}

/**
 * Standard deviation of elements, ignoring NaN. Computed in two passes, so it
 * does not suffer from the cancellation of `E[x^2] - mean^2`. A slice with
 * `count - ddof <= 0` non-NaN values gives NaN. The result is always float64.
 *
 * @param t - Input tensor
 * @param axis - Axis or axes to reduce (default: all)
 * @param keepdims - Keep reduced axes as size-1 dimensions
 * @param ddof - Delta degrees of freedom; the divisor is `count - ddof` (default: 0,
 *   population standard deviation, like NumPy)
 * @returns float64 tensor of standard deviations
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If an axis is out of range or repeated, or `ddof` is negative or not finite
 */
export function nanstd(
  t: Tensor,
  axis?: Axis | readonly Axis[],
  keepdims = false,
  ddof = 0
): Tensor {
  if (!Number.isFinite(ddof) || ddof < 0) {
    throw new InvalidParameterError(
      `ddof must be a non-negative finite number; got ${ddof}`,
      "ddof",
      ddof
    );
  }
  return nanReduce(t, axis, keepdims, "nanstd", (vals, n) => {
    const sum = nanSumCompensated(vals, n);
    const count = lastNanCount;
    if (count - ddof <= 0) return Number.NaN;
    const mean = sum / count;
    let ss = 0;
    for (let i = 0; i < n; i++) {
      const v = vals[i] as number;
      if (!Number.isNaN(v)) {
        const d = v - mean;
        ss += d * d;
      }
    }
    return Math.sqrt(ss / (count - ddof));
  });
}

function nanExtreme(vals: ArrayLike<number>, n: number, isMax: boolean): number {
  let m = isMax ? -Infinity : Infinity;
  let seen = false;
  for (let i = 0; i < n; i++) {
    const v = vals[i] as number;
    if (!Number.isNaN(v)) {
      seen = true;
      if (isMax ? v > m : v < m) m = v;
    }
  }
  // Track "any non-NaN seen" separately: a slice of all +Infinity must return
  // Infinity, not NaN.
  return seen ? m : Number.NaN;
}

/**
 * Minimum of elements, ignoring NaN. A slice without non-NaN values (including
 * an empty slice) gives NaN. The result is always float64.
 *
 * @param t - Input tensor
 * @param axis - Axis or axes to reduce (default: all)
 * @param keepdims - Keep reduced axes as size-1 dimensions
 * @returns float64 tensor of minima
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If an axis is out of range or repeated
 */
export function nanmin(t: Tensor, axis?: Axis | readonly Axis[], keepdims = false): Tensor {
  return nanReduce(t, axis, keepdims, "nanmin", (vals, n) => nanExtreme(vals, n, false));
}

/**
 * Maximum of elements, ignoring NaN. A slice without non-NaN values (including
 * an empty slice) gives NaN. The result is always float64.
 *
 * @param t - Input tensor
 * @param axis - Axis or axes to reduce (default: all)
 * @param keepdims - Keep reduced axes as size-1 dimensions
 * @returns float64 tensor of maxima
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If an axis is out of range or repeated
 */
export function nanmax(t: Tensor, axis?: Axis | readonly Axis[], keepdims = false): Tensor {
  return nanReduce(t, axis, keepdims, "nanmax", (vals, n) => nanExtreme(vals, n, true));
}

// ─── unique ─────────────────────────────────────────────────────────────────

/** Options of {@link unique}. */
export interface UniqueOptions {
  /** Also return the index of the first occurrence of each unique value (int32). */
  readonly returnIndex?: boolean;
  /** Also return the indices that rebuild the input from the unique values (int32). */
  readonly returnInverse?: boolean;
  /** Also return how many times each unique value occurs (int32). */
  readonly returnCounts?: boolean;
  /**
   * Find unique sub-tensors along this axis instead of unique elements of the
   * flattened input.
   */
  readonly axis?: Axis;
}

/** Result of {@link unique} when called with an options object. */
export interface UniqueResult {
  /** Sorted unique values (or sub-tensors), same dtype as the input. */
  values: Tensor;
  /** First-occurrence index of each unique value; present with `returnIndex`. */
  indices?: Tensor;
  /** Position of each input element (or sub-tensor) in `values`; present with `returnInverse`. */
  inverse?: Tensor;
  /** Number of occurrences of each unique value; present with `returnCounts`. */
  counts?: Tensor;
}

/** {@link UniqueResult} with the fields requested through the literal options made required. */
export type UniqueOutput<O extends UniqueOptions> = UniqueResult &
  (O extends { readonly returnIndex: true } ? { indices: Tensor } : unknown) &
  (O extends { readonly returnInverse: true } ? { inverse: Tensor } : unknown) &
  (O extends { readonly returnCounts: true } ? { counts: Tensor } : unknown);

/** Three-way comparison with NaN last and `-0` equal to `+0`. */
function compareNumbers(x: number, y: number): number {
  if (x < y) return -1;
  if (x > y) return 1;
  if (x === y) return 0;
  const xNaN = Number.isNaN(x);
  const yNaN = Number.isNaN(y);
  return xNaN === yNaN ? 0 : xNaN ? 1 : -1;
}

function makeIndexTensor(values: ArrayLike<number>, shape: readonly number[], t: Tensor): Tensor {
  return Tensor.fromTypedArray({
    data: Int32Array.from(values),
    shape: [...shape],
    dtype: "int32",
    device: t.device,
  });
}

/** Stable argsort of `n` items under `compare(i, j)`; ties keep their input order. */
function stableOrder(n: number, compare: (i: number, j: number) => number): Int32Array {
  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((i, j) => compare(i, j));
  return order;
}

/**
 * Unique elements of the flattened tensor with first-occurrence indices, the inverse
 * map and counts. Large non-int64 inputs are ordered with the radix argsort.
 */
function uniqueFlat(
  t: Tensor,
  dtype: Exclude<DType, "string">,
  wantInverse: boolean
): {
  values: TypedArray;
  count: number;
  first: Int32Array;
  inverse: Int32Array;
  counts: Int32Array;
} {
  const src = materializeNumeric(t);
  const n = src.length;
  let order: Int32Array;
  let same: (i: number, j: number) => boolean;
  if (src instanceof BigInt64Array) {
    order = stableOrder(n, (i, j) => {
      const x = src[i] as bigint;
      const y = src[j] as bigint;
      return x < y ? -1 : x > y ? 1 : 0;
    });
    same = (i, j) => src[i] === src[j];
  } else {
    if (n >= RADIX_SORT_THRESHOLD) {
      order = new Int32Array(n);
      radixArgsortF64(src instanceof Float64Array ? src : Float64Array.from(src), order);
    } else {
      order = stableOrder(n, (i, j) => compareNumbers(src[i] as number, src[j] as number));
    }
    same = (i, j) => compareNumbers(src[i] as number, src[j] as number) === 0;
  }

  const first = new Int32Array(n);
  const counts = new Int32Array(n);
  const inverse = new Int32Array(wantInverse ? n : 0);
  let k = 0;
  for (let p = 0; p < n; p++) {
    const i = order[p] as number;
    if (p === 0 || !same(i, order[p - 1] as number)) {
      first[k] = i;
      counts[k] = 1;
      k++;
    } else {
      counts[k - 1] = (counts[k - 1] as number) + 1;
    }
    if (wantInverse) inverse[i] = k - 1;
  }

  const Ctor = dtypeToTypedArrayCtor(dtype);
  const values = new Ctor(k);
  for (let u = 0; u < k; u++) writeElement(values, u, src, first[u] as number);
  return { values, count: k, first, inverse, counts };
}

/** Unique sub-tensors along `axis`, in lexicographic order. */
function uniqueAlongAxis(
  t: Tensor,
  dtype: Exclude<DType, "string">,
  axis: Axis
): { values: Tensor; first: Int32Array; inverse: Int32Array; counts: Int32Array } {
  const ax = normalizeAxis(axis, t.ndim);
  const n = t.shape[ax] ?? 0;
  const rest = t.shape.filter((_, d) => d !== ax);
  const rowLen = shapeToSize(rest);
  const src = materializeNumeric(moveaxis(t, ax, 0));

  const compareRows =
    src instanceof BigInt64Array
      ? (i: number, j: number): number => {
          for (let c = 0; c < rowLen; c++) {
            const x = src[i * rowLen + c] as bigint;
            const y = src[j * rowLen + c] as bigint;
            if (x !== y) return x < y ? -1 : 1;
          }
          return 0;
        }
      : (i: number, j: number): number => {
          for (let c = 0; c < rowLen; c++) {
            const r = compareNumbers(src[i * rowLen + c] as number, src[j * rowLen + c] as number);
            if (r !== 0) return r;
          }
          return 0;
        };
  const order = stableOrder(n, compareRows);

  const first = new Int32Array(n);
  const counts = new Int32Array(n);
  const inverse = new Int32Array(n);
  let k = 0;
  for (let p = 0; p < n; p++) {
    const i = order[p] as number;
    if (p === 0 || compareRows(i, order[p - 1] as number) !== 0) {
      first[k] = i;
      counts[k] = 1;
      k++;
    } else {
      counts[k - 1] = (counts[k - 1] as number) + 1;
    }
    inverse[i] = k - 1;
  }

  const out = new (dtypeToTypedArrayCtor(dtype))(k * rowLen);
  for (let u = 0; u < k; u++) {
    const base = (first[u] as number) * rowLen;
    for (let c = 0; c < rowLen; c++) writeElement(out, u * rowLen + c, src, base + c);
  }
  const stacked = Tensor.fromTypedArray({
    data: out,
    shape: [k, ...rest],
    dtype,
    device: t.device,
  });
  return {
    values: contiguous(moveaxis(stacked, 0, ax)),
    first: first.slice(0, k),
    inverse,
    counts: counts.slice(0, k),
  };
}

/**
 * Return the unique elements of a tensor, sorted in ascending order.
 *
 * Without options the tensor is flattened. NaN values collapse into a single
 * trailing NaN entry and `-0` equals `+0`, as in NumPy. int64 values are
 * compared exactly. The `values` tensor has the dtype of `t`.
 *
 * Called with a boolean (the original form), the boolean selects `counts`,
 * returned as a float64 tensor. Called with an options object, the result
 * follows `np.unique`:
 *
 * - `returnIndex`: `indices`, the index of the first occurrence of each unique
 *   value in the flattened input
 * - `returnInverse`: `inverse`, the position of every input element in
 *   `values`; with no `axis` it has the shape of `t`, with an `axis` it is
 *   1-D with one entry per sub-tensor
 * - `returnCounts`: `counts`, the number of occurrences of each unique value
 * - `axis`: find unique sub-tensors (rows for axis 0) instead of elements.
 *   Sub-tensors are ordered lexicographically; NaN entries compare equal to
 *   each other and sort last. `indices` then refers to positions along `axis`
 *
 * In the options form `indices`, `inverse` and `counts` are int32 tensors.
 *
 * @param t - Input tensor
 * @param returnCountsOrOptions - `true` to also return float64 counts, or an options object
 * @returns Object with `values` and the requested extra tensors
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If `axis` is out of range for `t`
 *
 * @example
 * ```ts
 * const { values, counts } = unique(tensor([3, 1, 2, 1, 3]), true);
 * // values: [1, 2, 3], counts: [2, 1, 2]
 *
 * const r = unique(tensor([3, 1, 2, 1, 3]), { returnIndex: true, returnInverse: true });
 * // r.values: [1, 2, 3], r.indices: [1, 2, 0], r.inverse: [2, 0, 1, 0, 2]
 *
 * unique(tensor([[1, 2], [1, 2], [0, 5]]), { axis: 0 }).values; // [[0, 5], [1, 2]]
 * ```
 */
export function unique(t: Tensor, returnCounts?: boolean): { values: Tensor; counts?: Tensor };
export function unique<const O extends UniqueOptions>(t: Tensor, options: O): UniqueOutput<O>;
export function unique(t: Tensor, returnCountsOrOptions?: boolean | UniqueOptions): UniqueResult {
  if (t.dtype === "string") {
    throw new DTypeError("unique is not defined for string dtype");
  }
  const dtype = ensureNonStringDType(t.dtype);
  const options: UniqueOptions =
    typeof returnCountsOrOptions === "object" && returnCountsOrOptions !== null
      ? returnCountsOrOptions
      : {};
  const legacy = typeof returnCountsOrOptions !== "object";
  const wantIndex = options.returnIndex === true;
  const wantInverse = options.returnInverse === true;
  const wantCounts = legacy ? returnCountsOrOptions === true : options.returnCounts === true;

  if (options.axis !== undefined) {
    const r = uniqueAlongAxis(t, dtype, options.axis);
    const result: UniqueResult = { values: r.values };
    if (wantIndex) result.indices = makeIndexTensor(r.first, [r.first.length], t);
    if (wantInverse) result.inverse = makeIndexTensor(r.inverse, [r.inverse.length], t);
    if (wantCounts) result.counts = makeIndexTensor(r.counts, [r.counts.length], t);
    return result;
  }

  if (wantIndex || wantInverse) {
    const r = uniqueFlat(t, dtype, wantInverse);
    const result: UniqueResult = {
      values: Tensor.fromTypedArray({
        data: r.values,
        shape: [r.count],
        dtype,
        device: t.device,
      }),
    };
    if (wantIndex) result.indices = makeIndexTensor(r.first.subarray(0, r.count), [r.count], t);
    if (wantInverse) result.inverse = makeIndexTensor(r.inverse, t.shape, t);
    if (wantCounts) result.counts = makeIndexTensor(r.counts.subarray(0, r.count), [r.count], t);
    return result;
  }

  // Native typed-array sort is numeric, ascending, and puts NaN last.
  const sorted = materializeNumeric(t);
  sorted.sort();

  const n = sorted.length;
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const vals = new Ctor(n);
  const counts = new Float64Array(n);
  let k = 0;
  if (sorted instanceof BigInt64Array && vals instanceof BigInt64Array) {
    for (let i = 0; i < n; ) {
      const v = sorted[i] as bigint;
      let j = i + 1;
      while (j < n && sorted[j] === v) j++;
      vals[k] = v;
      counts[k++] = j - i;
      i = j;
    }
  } else if (!(sorted instanceof BigInt64Array) && !(vals instanceof BigInt64Array)) {
    for (let i = 0; i < n; ) {
      const v = sorted[i] as number;
      let j = i + 1;
      if (Number.isNaN(v)) {
        // NaN !== NaN, so group the trailing NaNs explicitly.
        while (j < n && Number.isNaN(sorted[j] as number)) j++;
      } else {
        while (j < n && sorted[j] === v) j++;
      }
      vals[k] = v;
      counts[k++] = j - i;
      i = j;
    }
  }

  const values = Tensor.fromTypedArray({
    data: vals.slice(0, k),
    shape: [k],
    dtype,
    device: t.device,
  });

  if (wantCounts) {
    const result: UniqueResult = { values };
    result.counts = legacy
      ? Tensor.fromTypedArray({
          data: counts.slice(0, k),
          shape: [k],
          dtype: "float64",
          device: t.device,
        })
      : makeIndexTensor(counts.subarray(0, k), [k], t);
    return result;
  }

  return { values };
}

// ─── searchsorted ───────────────────────────────────────────────────────────

/**
 * Find indices where elements should be inserted to maintain order.
 *
 * NaN sorts after every other value (NumPy convention), so searching for NaN
 * returns the length of the array unless the array itself ends in NaNs. The
 * array is not checked for being sorted. The indices are returned as int32.
 *
 * @param sortedArr - 1-D sorted tensor
 * @param values - Values to insert (any shape)
 * @param side - `'left'` (default): first suitable index, `'right'`: last suitable index
 * @returns int32 tensor of insertion indices with the shape of `values`
 * @throws {ShapeError} If `sortedArr` is not 1-D
 * @throws {InvalidParameterError} If `side` is not `'left'` or `'right'`
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
  if (side !== "left" && side !== "right") {
    throw new InvalidParameterError(
      `side must be 'left' or 'right'; got '${String(side)}'`,
      "side",
      side
    );
  }

  const arr = readNumbers(sortedArr, "searchsorted", false);
  const vals = readNumbers(values, "searchsorted", false);
  const arrLen = sortedArr.size;
  const n = values.size;
  const outData = new Int32Array(n);
  const left = side === "left";

  for (let i = 0; i < n; i++) {
    const v = vals[i] as number;
    const vIsNaN = Number.isNaN(v);
    let lo = 0;
    let hi = arrLen;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      const m = arr[mid] as number;
      // left: go right while m < v; right: go right while !(v < m).
      // "<" is extended so that NaN is greater than every number.
      const goRight = left
        ? m < v || (vIsNaN && !Number.isNaN(m))
        : !(v < m || (Number.isNaN(m) && !vIsNaN));
      if (goRight) {
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
    dtype: "int32",
    device: sortedArr.device,
  });
}

// ─── histogram / bincount ───────────────────────────────────────────────────

/** Options of {@link histogram}. */
export interface HistogramOptions {
  /** Return the probability density instead of counts (the histogram integrates to 1). */
  readonly density?: boolean;
  /** Weight of each sample; must have the same shape as the input. */
  readonly weights?: Tensor;
}

/**
 * Compute the histogram of a tensor (flattened).
 *
 * Bins are half-open `[left, right)` except the last one, which also includes
 * its right edge. NaN samples and samples outside the range are ignored. When
 * no range is given the bins span the data minimum to maximum (0 to 1 for
 * empty data); a range with equal ends is widened by 0.5 on each side.
 *
 * @param t - Input tensor
 * @param bins - Number of equal-width bins (default: 10), or a non-decreasing
 *   list of bin edges
 * @param range - `[min, max]` range for equal-width bins (ignored when `bins` is a list of edges)
 * @param options - `density` and `weights`
 * @returns Object with float64 `counts` (or densities) and `binEdges` tensors
 * @throws {InvalidParameterError} If `bins` or `range` is invalid, or the data range is not finite
 * @throws {ShapeError} If `weights` does not match the shape of `t`
 *
 * @example
 * ```ts
 * const { counts, binEdges } = histogram(tensor([1, 2, 2, 3, 10]), 3, [0, 12]);
 * // counts: [4, 0, 1], binEdges: [0, 4, 8, 12]
 * ```
 */
export function histogram(
  t: Tensor,
  bins: number | readonly number[] = 10,
  range?: readonly [number, number],
  options?: HistogramOptions
): { counts: Tensor; binEdges: Tensor } {
  if (t.dtype === "string") {
    throw new DTypeError("histogram is not defined for string dtype");
  }

  const vals = readNumbers(t, "histogram", false);
  const n = t.size;
  let weights: ArrayLike<number> | null = null;
  const weightTensor = options?.weights;
  if (weightTensor !== undefined) {
    if (
      weightTensor.ndim !== t.ndim ||
      weightTensor.shape.some((dim, d) => dim !== (t.shape[d] ?? 0))
    ) {
      throw ShapeError.mismatch(t.shape, weightTensor.shape, "histogram weights");
    }
    weights = readNumbers(weightTensor, "histogram", false);
  }

  let edges: Float64Array;
  let numBins: number;
  let counts: Float64Array;
  if (typeof bins === "number") {
    if (!Number.isInteger(bins) || bins <= 0) {
      throw new InvalidParameterError(`bins must be a positive integer; got ${bins}`, "bins", bins);
    }
    numBins = bins;
    let lo: number;
    let hi: number;
    if (range) {
      lo = range[0];
      hi = range[1];
      if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
        throw new InvalidParameterError("range must contain finite numbers", "range", range);
      }
      if (lo > hi) {
        throw new InvalidParameterError(
          `range[0] must not exceed range[1]; got [${lo}, ${hi}]`,
          "range",
          range
        );
      }
    } else {
      lo = Infinity;
      hi = -Infinity;
      let seen = false;
      for (let i = 0; i < n; i++) {
        const v = vals[i] as number;
        if (Number.isNaN(v)) continue;
        seen = true;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      if (!seen) {
        lo = 0;
        hi = 1;
      } else if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
        throw new InvalidParameterError(
          `autodetected range [${lo}, ${hi}] is not finite; pass a finite range`,
          "range",
          range
        );
      }
    }
    if (lo === hi) {
      lo -= 0.5;
      hi += 0.5;
    }
    const span = hi - lo;
    if (!Number.isFinite(span)) {
      throw new InvalidParameterError("range is too wide to split into bins", "range", range);
    }
    const step = span / numBins;
    edges = new Float64Array(numBins + 1);
    for (let i = 0; i < numBins; i++) edges[i] = lo + i * step;
    edges[numBins] = hi;

    counts = new Float64Array(numBins);
    for (let i = 0; i < n; i++) {
      const v = vals[i] as number;
      if (!(v >= lo && v <= hi)) continue;
      // Same bin computation as NumPy: estimate from the position, then fix
      // up rounding errors against the actual edges.
      let idx = Math.trunc(((v - lo) / span) * numBins);
      if (idx >= numBins) idx = numBins - 1;
      if (v < (edges[idx] as number)) idx -= 1;
      if (v >= (edges[idx + 1] as number) && idx !== numBins - 1) idx += 1;
      counts[idx] = (counts[idx] as number) + (weights ? (weights[i] as number) : 1);
    }
  } else {
    if (bins.length < 2) {
      throw new InvalidParameterError(
        "bins must contain at least 2 edges when given as a list",
        "bins",
        bins
      );
    }
    edges = Float64Array.from(bins);
    for (let i = 0; i < edges.length; i++) {
      const e = edges[i] as number;
      if (!Number.isFinite(e) || (i > 0 && e < (edges[i - 1] as number))) {
        throw new InvalidParameterError(
          "bins must be finite and increase monotonically",
          "bins",
          bins
        );
      }
    }
    numBins = edges.length - 1;
    const lo = edges[0] as number;
    const hi = edges[numBins] as number;
    counts = new Float64Array(numBins);
    for (let i = 0; i < n; i++) {
      const v = vals[i] as number;
      if (!(v >= lo && v <= hi)) continue;
      // Last edge <= v, so a sample on a shared edge lands in the later bin.
      let a = 0;
      let b = numBins + 1;
      while (a < b) {
        const mid = (a + b) >>> 1;
        if ((edges[mid] as number) <= v) a = mid + 1;
        else b = mid;
      }
      const idx = Math.min(a - 1, numBins - 1);
      counts[idx] = (counts[idx] as number) + (weights ? (weights[i] as number) : 1);
    }
  }

  if (options?.density) {
    let total = 0;
    for (let i = 0; i < numBins; i++) total += counts[i] as number;
    for (let i = 0; i < numBins; i++) {
      const width = (edges[i + 1] as number) - (edges[i] as number);
      counts[i] = (counts[i] as number) / width / total;
    }
  }

  return {
    counts: Tensor.fromTypedArray({
      data: counts,
      shape: [numBins],
      dtype: "float64",
      device: t.device,
    }),
    binEdges: Tensor.fromTypedArray({
      data: edges,
      shape: [numBins + 1],
      dtype: "float64",
      device: t.device,
    }),
  };
}

/**
 * Count the occurrences of each value in a tensor of non-negative integers
 * (flattened). The result is a float64 tensor of length
 * `max(max(t) + 1, minlength)`, or the sum of `weights` per value when given.
 *
 * @param t - Input tensor of non-negative integers
 * @param minlength - Minimum output length (default: 0)
 * @param weights - Optional weight per element, same number of elements as `t`
 * @returns Tensor of counts
 * @throws {InvalidParameterError} If `t` has a negative or non-integer value, or `minlength` is invalid
 * @throws {ShapeError} If `weights` does not have the same size as `t`
 */
export function bincount(t: Tensor, minlength = 0, weights?: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("bincount is not defined for string dtype");
  }
  if (!Number.isInteger(minlength) || minlength < 0) {
    throw new InvalidParameterError(
      `minlength must be a non-negative integer; got ${minlength}`,
      "minlength",
      minlength
    );
  }
  if (weights !== undefined && weights.size !== t.size) {
    throw new ShapeError(
      `bincount weights size ${weights.size} does not match input size ${t.size}`
    );
  }

  const vals = readNumbers(t, "bincount", false);
  const w = weights ? readNumbers(weights, "bincount", false) : null;
  const n = t.size;

  let maxVal = minlength - 1;
  for (let i = 0; i < n; i++) {
    const v = vals[i] as number;
    if (v < 0 || !Number.isInteger(v)) {
      throw new InvalidParameterError(
        `bincount requires non-negative integer values; got ${v}`,
        "t",
        v
      );
    }
    if (v > maxVal) maxVal = v;
  }
  if (maxVal + 1 > 0x7fffffff) {
    throw new InvalidParameterError(
      `bincount value ${maxVal} is too large to allocate a count array`,
      "t",
      maxVal
    );
  }

  let counts: Float64Array;
  try {
    counts = new Float64Array(maxVal + 1);
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
    throw new InvalidParameterError(
      `bincount cannot allocate ${maxVal + 1} counts; the largest value or minlength is too big`,
      "t",
      maxVal
    );
  }
  for (let i = 0; i < n; i++) {
    const v = vals[i] as number;
    counts[v] = (counts[v] as number) + (w ? (w[i] as number) : 1);
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
 * Returns a 1-D tensor of elements where the mask is true (nonzero, NaN included).
 *
 * The mask is matched to `t` element by element in row-major order, so it must
 * have the same number of elements as `t` (normally the same shape).
 *
 * @param t - Input tensor
 * @param mask - Boolean mask tensor (same shape as t)
 * @returns 1-D tensor of selected elements, in row-major order
 * @throws {ShapeError} If `t` and `mask` have a different number of elements
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

  const maskVals = readNumbers(mask, "booleanIndex", false);
  const n = mask.size;
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (maskVals[i] !== 0) count++;
  }

  const Ctor = dtypeToTypedArrayCtor(dtype);
  const outData = new Ctor(count);
  const tData = logicalData(t, "booleanIndex");
  let idx = 0;
  for (let i = 0; i < n && idx < count; i++) {
    if (maskVals[i] !== 0) {
      writeElement(outData, idx, tData, i);
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
 * Select elements from a tensor using integer index arrays (like `np.take`).
 *
 * The result has the shape of `t` with `axis` replaced by the shape of
 * `indices`. A 0-d `indices` tensor selects a single entry and keeps the axis
 * with size 1. Negative indices count from the end of the axis.
 *
 * @param t - Input tensor
 * @param indices - Integer-valued index tensor (bool tensors are rejected; use {@link booleanIndex})
 * @param axis - Axis along which to index (default: 0)
 * @returns Tensor with selected elements, same dtype as `t`
 * @throws {IndexError} If an index is out of bounds for `axis`
 * @throws {InvalidParameterError} If an index is not an integer
 * @throws {DTypeError} If `indices` has bool dtype
 */
export function fancyIndex(t: Tensor, indices: Tensor, axis: Axis = 0): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("fancyIndex is not defined for string dtype");
  }
  if (indices.dtype === "string") {
    throw new DTypeError("fancyIndex indices must be numeric");
  }
  if (indices.dtype === "bool") {
    throw new DTypeError("fancyIndex indices must be integers; use booleanIndex for boolean masks");
  }
  const dtype = ensureNonStringDType(t.dtype);

  const ax = normalizeAxis(axis, t.ndim);
  const axDim = t.shape[ax] ?? 0;
  const idx = resolveIndices(indices, axDim, "fancyIndex", true);

  const flatShape = [...t.shape];
  flatShape[ax] = idx.length;
  const Ctor = dtypeToTypedArrayCtor(dtype);
  const outData = new Ctor(shapeToSize(flatShape));
  const tData = requireNumericData(t.data, "fancyIndex");

  const maps = flatShape.map((dim, d) => {
    if (d === ax) return idx;
    const identity = new Int32Array(dim);
    for (let i = 0; i < dim; i++) identity[i] = i;
    return identity;
  });
  gatherTyped(tData, t.offset, t.strides, flatShape, maps, outData);

  const flat = Tensor.fromTypedArray({
    data: outData,
    shape: flatShape,
    dtype,
    device: t.device,
  });
  if (indices.ndim < 2) return flat;
  return flat.reshape([...t.shape.slice(0, ax), ...indices.shape, ...t.shape.slice(ax + 1)]);
}

// ─── Along-axis indexing and nonzero ────────────────────────────────────────

/**
 * Visit every position of `shape` in row-major order. Three offsets advance in
 * step with the position, each by its own per-axis steps (a step may be zero to
 * broadcast an axis). `visit` receives the three offsets and the flat position.
 */
function walkSteps(
  shape: readonly number[],
  stepsA: readonly number[],
  stepsB: readonly number[],
  stepsC: readonly number[],
  startA: number,
  startB: number,
  startC: number,
  visit: (a: number, b: number, c: number, pos: number) => void
): void {
  const nd = shape.length;
  const size = shapeToSize(shape as Shape);
  if (size === 0) return;
  if (nd === 0) {
    visit(startA, startB, startC, 0);
    return;
  }
  const inner = shape[nd - 1] as number;
  const innerA = stepsA[nd - 1] as number;
  const innerB = stepsB[nd - 1] as number;
  const innerC = stepsC[nd - 1] as number;
  const outer = size / inner;
  const coords = new Int32Array(nd);
  let a = startA;
  let b = startB;
  let c = startC;
  let pos = 0;
  for (let o = 0; o < outer; o++) {
    let ia = a;
    let ib = b;
    let ic = c;
    for (let i = 0; i < inner; i++) {
      visit(ia, ib, ic, pos++);
      ia += innerA;
      ib += innerB;
      ic += innerC;
    }
    for (let d = nd - 2; d >= 0; d--) {
      const dim = shape[d] as number;
      const sa = stepsA[d] as number;
      const sb = stepsB[d] as number;
      const sc = stepsC[d] as number;
      const next = (coords[d] as number) + 1;
      a += sa;
      b += sb;
      c += sc;
      if (next < dim) {
        coords[d] = next;
        break;
      }
      coords[d] = 0;
      a -= sa * dim;
      b -= sb * dim;
      c -= sc * dim;
    }
  }
}

/** Validate an integer index against an axis of size `axDim`; negative values count from the end. */
function resolveAlongIndex(raw: number, axDim: number, name: string): number {
  if (!Number.isInteger(raw)) {
    throw new InvalidParameterError(`${name} indices must be integers; got ${raw}`, "indices", raw);
  }
  const k = raw < 0 ? raw + axDim : raw;
  if (k < 0 || k >= axDim) {
    throw new IndexError(`${name} index ${raw} is out of bounds for an axis of size ${axDim}`, {
      index: raw,
      validRange: [-axDim, axDim - 1],
    });
  }
  return k;
}

/** Reject index tensors that cannot hold integer positions. */
function assertIndexDType(indices: Tensor, name: string): void {
  if (indices.dtype === "string") {
    throw new DTypeError(`${name} indices must be numeric`);
  }
  if (indices.dtype === "bool") {
    throw new DTypeError(`${name} indices must be integers, not bool`);
  }
}

/**
 * Layout shared by {@link takeAlongAxis} and {@link putAlongAxis}: every axis
 * except `ax` broadcasts between the array and the indices, and the axis `ax`
 * has the extent of `indices`.
 */
function planAlongAxis(
  arrShape: readonly number[],
  arrStrides: readonly number[],
  idxShape: readonly number[],
  ax: number,
  name: string
): {
  shape: number[];
  arrSteps: number[];
  idxSteps: number[];
  axStride: number;
  axDim: number;
} {
  if (arrShape.length !== idxShape.length) {
    throw new ShapeError(
      `${name} requires arr and indices to have the same number of dimensions; ` +
        `got ${arrShape.length} and ${idxShape.length}`
    );
  }
  const idxLogical = computeStrides(idxShape as Shape);
  const shape: number[] = [];
  const arrSteps: number[] = [];
  const idxSteps: number[] = [];
  for (let d = 0; d < arrShape.length; d++) {
    const ad = arrShape[d] as number;
    const id = idxShape[d] as number;
    if (d === ax) {
      shape.push(id);
      arrSteps.push(0);
      idxSteps.push(idxLogical[d] as number);
      continue;
    }
    let dim: number;
    if (ad === id) dim = ad;
    else if (ad === 1) dim = id;
    else if (id === 1) dim = ad;
    else {
      throw new ShapeError(
        `${name} cannot broadcast arr shape [${arrShape}] with indices shape [${idxShape}] ` +
          `outside axis ${ax}`
      );
    }
    shape.push(dim);
    arrSteps.push(ad === 1 && dim !== 1 ? 0 : (arrStrides[d] as number));
    idxSteps.push(id === 1 && dim !== 1 ? 0 : (idxLogical[d] as number));
  }
  return {
    shape,
    arrSteps,
    idxSteps,
    axStride: arrStrides[ax] as number,
    axDim: arrShape[ax] as number,
  };
}

/** Per-axis steps that read a tensor of `shape` as if it had the broadcast shape `outShape`. */
function broadcastStepsTo(
  shape: readonly number[],
  strides: readonly number[],
  outShape: readonly number[],
  name: string
): number[] {
  if (shape.length > outShape.length) {
    throw new ShapeError(
      `${name} cannot broadcast values of shape [${shape}] to the index shape [${outShape}]`
    );
  }
  const lead = outShape.length - shape.length;
  const steps = new Array<number>(outShape.length).fill(0);
  for (let d = 0; d < shape.length; d++) {
    const dim = shape[d] as number;
    const target = outShape[d + lead] as number;
    if (dim !== target && dim !== 1) {
      throw new ShapeError(
        `${name} cannot broadcast values of shape [${shape}] to the index shape [${outShape}]`
      );
    }
    steps[d + lead] = dim === 1 ? 0 : (strides[d] as number);
  }
  return steps;
}

function takeLoop<V>(
  src: ArrayLike<V>,
  srcStart: number,
  out: { [i: number]: V },
  idx: ArrayLike<number>,
  plan: ReturnType<typeof planAlongAxis>,
  name: string
): void {
  const zeros = new Array<number>(plan.shape.length).fill(0);
  walkSteps(plan.shape, plan.idxSteps, plan.arrSteps, zeros, 0, srcStart, 0, (i, a, _c, pos) => {
    const k = resolveAlongIndex(idx[i] as number, plan.axDim, name);
    out[pos] = src[a + k * plan.axStride] as V;
  });
}

/**
 * Take values from `arr` along an axis at the positions given by `indices`
 * (like `np.take_along_axis`). It is the companion of {@link argsort}, `argmax`
 * and similar functions that return per-axis positions.
 *
 * `arr` and `indices` must have the same number of dimensions. Every axis
 * except `axis` broadcasts between the two (a size-1 axis stretches), and the
 * result has the length of `indices` along `axis`:
 * `out[i][j][k] = arr[i][indices[i][j][k]][k]` for `axis = 1`. Negative indices
 * count from the end of the axis. Integer-valued float indices are accepted.
 *
 * With `axis = null`, `arr` is flattened first and `indices` must be 1-D.
 *
 * @param arr - Source tensor
 * @param indices - Integer positions along `axis` (same number of dimensions as `arr`)
 * @param axis - Axis to index along (default: -1), or `null` to use the flattened `arr`
 * @returns New tensor with the dtype of `arr` and the broadcast shape described above
 * @throws {ShapeError} If the dimensions differ or the other axes do not broadcast
 * @throws {IndexError} If an index is out of bounds for `axis`
 * @throws {InvalidParameterError} If an index is not an integer or `axis` is out of range
 * @throws {DTypeError} If `arr` or `indices` has string dtype, or `indices` has bool dtype
 *
 * @example
 * ```ts
 * const a = tensor([[10, 30, 20], [60, 40, 50]]);
 * const order = argsort(a, 1); // [[0, 2, 1], [1, 2, 0]]
 * takeAlongAxis(a, order, 1); // [[10, 20, 30], [40, 50, 60]]
 * ```
 */
export function takeAlongAxis(arr: Tensor, indices: Tensor, axis: Axis | null = -1): Tensor {
  const name = "takeAlongAxis";
  if (arr.dtype === "string") {
    throw new DTypeError(`${name} is not defined for string dtype`);
  }
  assertIndexDType(indices, name);
  const dtype = ensureNonStringDType(arr.dtype);

  let data: TypedArray;
  let shape: readonly number[];
  let strides: readonly number[];
  let start: number;
  let ax: number;
  if (axis === null) {
    data = logicalData(arr, name);
    shape = [arr.size];
    strides = [1];
    start = 0;
    ax = 0;
  } else {
    data = requireNumericData(arr.data, name);
    shape = arr.shape;
    strides = arr.strides;
    start = arr.offset;
    ax = normalizeAxis(axis, arr.ndim);
  }

  const plan = planAlongAxis(shape, strides, indices.shape, ax, name);
  const out = new (dtypeToTypedArrayCtor(dtype))(shapeToSize(plan.shape as Shape));
  const idx = readNumbers(indices, name, false);
  if (data instanceof BigInt64Array && out instanceof BigInt64Array) {
    takeLoop<bigint>(data, start, out, idx, plan, name);
  } else if (!(data instanceof BigInt64Array) && !(out instanceof BigInt64Array)) {
    takeLoop<number>(data, start, out, idx, plan, name);
  } else {
    throw new DTypeError("Internal error: cannot mix int64 and non-int64 buffers");
  }
  return Tensor.fromTypedArray({ data: out, shape: plan.shape, dtype, device: arr.device });
}

/**
 * Put `values` into a copy of `arr` along an axis at the positions given by
 * `indices` (like `np.put_along_axis`, with the same index and broadcasting
 * rules as {@link takeAlongAxis}).
 *
 * Unlike NumPy, which modifies `arr` in place and returns nothing, this
 * function leaves `arr` unchanged and returns the updated copy, as
 * {@link scatter} does. `values` broadcasts to the broadcast shape of the
 * indices; a plain number or bigint is accepted and converted to the dtype of
 * `arr`. When an index repeats, the last value written wins (row-major order).
 *
 * With `axis = null`, `arr` is flattened for the operation, `indices` must be
 * 1-D, and the result has the shape of `arr`.
 *
 * @param arr - Tensor to update (not modified)
 * @param indices - Integer positions along `axis` (same number of dimensions as `arr`)
 * @param values - Values to write: a tensor broadcastable to the index shape, or a scalar
 * @param axis - Axis to index along (default: -1), or `null` to use the flattened `arr`
 * @returns New tensor with the shape and dtype of `arr`
 * @throws {ShapeError} If the dimensions differ, the other axes do not broadcast, or `values`
 *   does not broadcast to the index shape
 * @throws {IndexError} If an index is out of bounds for `axis`
 * @throws {InvalidParameterError} If an index is not an integer or `axis` is out of range
 * @throws {DTypeError} If a tensor has string dtype, or `indices` has bool dtype
 *
 * @example
 * ```ts
 * const a = tensor([[10, 30, 20], [60, 40, 50]]);
 * const idx = tensor([[0], [2]]);
 * putAlongAxis(a, idx, 99, 1); // [[99, 30, 20], [60, 40, 99]]
 * ```
 */
export function putAlongAxis(
  arr: Tensor,
  indices: Tensor,
  values: Tensor | number | bigint,
  axis: Axis | null = -1
): Tensor {
  const name = "putAlongAxis";
  if (arr.dtype === "string") {
    throw new DTypeError(`${name} is not defined for string dtype`);
  }
  assertIndexDType(indices, name);
  const dtype = ensureNonStringDType(arr.dtype);

  let valueTensor: Tensor;
  if (values instanceof Tensor) {
    valueTensor = values;
  } else {
    const scalar = new (dtypeToTypedArrayCtor(dtype))(1);
    fillRange(scalar, fillValueFor(dtype, values, name));
    valueTensor = Tensor.fromTypedArray({
      data: scalar,
      shape: [],
      dtype,
      device: arr.device,
    });
  }
  if (valueTensor.dtype === "string") {
    throw new DTypeError(`${name} values must be numeric`);
  }

  const result = clone(arr);
  const shape = axis === null ? [arr.size] : arr.shape;
  const ax = axis === null ? 0 : normalizeAxis(axis, arr.ndim);
  const plan = planAlongAxis(shape, computeStrides(shape as Shape), indices.shape, ax, name);
  const valSteps = broadcastStepsTo(valueTensor.shape, valueTensor.strides, plan.shape, name);
  if (shapeToSize(plan.shape as Shape) === 0) return result;

  const out = requireNumericData(result.data, name);
  const src = requireNumericData(valueTensor.data, name);
  const idx = readNumbers(indices, name, false);
  const boolOut = dtype === "bool";
  walkSteps(
    plan.shape,
    plan.idxSteps,
    plan.arrSteps,
    valSteps,
    0,
    0,
    valueTensor.offset,
    (i, a, v) => {
      const k = resolveAlongIndex(idx[i] as number, plan.axDim, name);
      const dst = a + k * plan.axStride;
      if (boolOut) out[dst] = readAsNumber(src, v) !== 0 ? 1 : 0;
      else writeElement(out, dst, src, v);
    }
  );
  return roundHalfResult(result);
}

/** Row-major coordinates of the nonzero elements, `ndim` entries per element. */
function collectNonzero(
  t: Tensor,
  name: string
): { coords: Int32Array; count: number; ndim: number } {
  const vals = readNumbers(t, name, false);
  const n = t.size;
  const ndim = t.ndim;
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (vals[i] !== 0) count++;
  }
  const coords = new Int32Array(count * ndim);
  if (count === 0 || ndim === 0) return { coords, count, ndim };

  const shape = t.shape;
  const cur = new Int32Array(ndim);
  let w = 0;
  for (let i = 0; i < n; i++) {
    if (vals[i] !== 0) {
      coords.set(cur, w);
      w += ndim;
    }
    for (let d = ndim - 1; d >= 0; d--) {
      const next = (cur[d] as number) + 1;
      if (next < (shape[d] as number)) {
        cur[d] = next;
        break;
      }
      cur[d] = 0;
    }
  }
  return { coords, count, ndim };
}

function assertNonzeroDType(t: Tensor, name: string): void {
  if (t.dtype === "string") {
    throw new DTypeError(`${name} is not defined for string dtype`);
  }
}

/**
 * Indices of the nonzero elements, one int32 tensor per dimension
 * (like `np.nonzero`). NaN counts as nonzero. The indices are in row-major
 * order, so `nonzero(t)` can be used directly to index `t` element by element.
 *
 * @param t - Input tensor
 * @returns Array with `t.ndim` int32 tensors of shape `[count]`; entry `d` holds the
 *   index along axis `d` of every nonzero element
 * @throws {ShapeError} If `t` is 0-d (use {@link argwhere} or reshape to 1-D first)
 * @throws {DTypeError} If `t` has string dtype
 *
 * @example
 * ```ts
 * const [rows, cols] = nonzero(tensor([[0, 2], [3, 0]]));
 * // rows: [0, 1], cols: [1, 0]
 * ```
 */
export function nonzero(t: Tensor): Tensor[] {
  assertNonzeroDType(t, "nonzero");
  if (t.ndim === 0) {
    throw new ShapeError(
      "nonzero is not defined for 0-d tensors; use argwhere or reshape the tensor to 1-D"
    );
  }
  const { coords, count, ndim } = collectNonzero(t, "nonzero");
  const out: Tensor[] = [];
  for (let d = 0; d < ndim; d++) {
    const col = new Int32Array(count);
    for (let i = 0; i < count; i++) col[i] = coords[i * ndim + d] as number;
    out.push(
      Tensor.fromTypedArray({ data: col, shape: [count], dtype: "int32", device: t.device })
    );
  }
  return out;
}

/**
 * Indices of the nonzero elements as one int32 matrix (like `np.argwhere`).
 * Row `i` holds the coordinates of the `i`-th nonzero element in row-major
 * order. NaN counts as nonzero. A 0-d tensor gives shape `[1, 0]` when it is
 * nonzero and `[0, 0]` otherwise.
 *
 * @param t - Input tensor
 * @returns int32 tensor of shape `[count, t.ndim]`
 * @throws {DTypeError} If `t` has string dtype
 *
 * @example
 * ```ts
 * argwhere(tensor([[0, 2], [3, 0]])); // [[0, 1], [1, 0]]
 * ```
 */
export function argwhere(t: Tensor): Tensor {
  assertNonzeroDType(t, "argwhere");
  const { coords, count, ndim } = collectNonzero(t, "argwhere");
  return Tensor.fromTypedArray({
    data: coords,
    shape: [count, ndim],
    dtype: "int32",
    device: t.device,
  });
}

/**
 * Count the nonzero elements (like `np.count_nonzero`). NaN counts as nonzero.
 *
 * @param t - Input tensor
 * @param axis - Axis or axes to count along (default: all axes)
 * @param keepdims - Keep the counted axes as size-1 dimensions
 * @returns int32 tensor of counts; 0-d when every axis is counted without `keepdims`
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If an axis is out of range or repeated
 *
 * @example
 * ```ts
 * const a = tensor([[1, 0, 3], [0, 0, 5]]);
 * countNonzero(a); // 3
 * countNonzero(a, 0); // [1, 0, 2]
 * ```
 */
export function countNonzero(t: Tensor, axis?: Axis | readonly Axis[], keepdims = false): Tensor {
  assertNonzeroDType(t, "countNonzero");
  const axes = new Set(resolveAxes(axis, t.ndim, "countNonzero"));
  const outShape: number[] = [];
  for (let d = 0; d < t.ndim; d++) {
    if (!axes.has(d)) outShape.push(t.shape[d] as number);
    else if (keepdims) outShape.push(1);
  }
  const outLogical = computeStrides(outShape as Shape);
  const steps = new Array<number>(t.ndim).fill(0);
  let o = 0;
  for (let d = 0; d < t.ndim; d++) {
    if (!axes.has(d)) steps[d] = outLogical[o++] as number;
    else if (keepdims) o++;
  }

  const vals = readNumbers(t, "countNonzero", false);
  const out = new Int32Array(shapeToSize(outShape as Shape));
  const zeros = new Array<number>(t.ndim).fill(0);
  walkSteps(t.shape, steps, zeros, zeros, 0, 0, 0, (dst, _b, _c, pos) => {
    if (vals[pos] !== 0) out[dst] = (out[dst] as number) + 1;
  });
  return Tensor.fromTypedArray({ data: out, shape: outShape, dtype: "int32", device: t.device });
}

/**
 * Rotate a 2-D tensor by 90 degrees counter-clockwise, `k` times.
 *
 * @param t - Input 2-D tensor
 * @param k - Number of 90° rotations (default 1). Negative values rotate clockwise.
 * @returns Rotated copy of the tensor
 * @throws {ShapeError} If the input is not 2-D
 * @throws {InvalidParameterError} If `k` is not an integer
 */
export function rot90(t: Tensor, k = 1): Tensor {
  if (t.ndim !== 2) {
    throw new ShapeError(`rot90 requires a 2-D tensor, got ${t.ndim}-D`);
  }
  if (!Number.isInteger(k)) {
    throw new InvalidParameterError(`rot90 k must be an integer; got ${k}`, "k", k);
  }
  switch (((k % 4) + 4) % 4) {
    case 1:
      return flip(transpose(t), [0]);
    case 2:
      return flip(t, [0, 1]);
    case 3:
      return flip(transpose(t), [1]);
    default:
      return clone(t);
  }
}

/**
 * Ensure the input is at least 1-D.
 * Scalars become shape [1]; higher-dimensional tensors pass through unchanged
 * (the same tensor is returned).
 * @deprecated Prefer {@link atleast1d}.
 */
export function atleast_1d(t: Tensor): Tensor {
  if (t.ndim >= 1) return t;
  return t.reshape([1]);
}

/**
 * Ensure the input is at least 2-D.
 * Scalars become shape [1,1]; 1-D tensors become shape [1,n]; others pass through
 * unchanged (the same tensor is returned).
 * @deprecated Prefer {@link atleast2d}.
 */
export function atleast_2d(t: Tensor): Tensor {
  if (t.ndim >= 2) return t;
  if (t.ndim === 0) return t.reshape([1, 1]);
  return t.reshape([1, t.shape[0] ?? 0]);
}

/** Options of {@link cross}. */
export interface CrossOptions {
  /** Axis that holds the three vector components (default: -1). */
  readonly axis?: Axis;
}

/**
 * Result dtype of {@link cross}: the promoted dtype of the inputs. Float inputs
 * keep their float dtype (float16 with bfloat16 gives float32); uint8 stays
 * uint8, int32 and int64 stay integer, and bool counts as int32.
 */
function crossDType(
  a: Exclude<DType, "string">,
  b: Exclude<DType, "string">
): "float16" | "bfloat16" | "float32" | "float64" | "int32" | "int64" | "uint8" {
  const promoted = promoteTypes(a, b);
  return promoted === "bool" ? "int32" : (promoted as ReturnType<typeof crossDType>);
}

/**
 * Compute the cross product of 3-element vectors (like `np.cross`).
 *
 * The vectors lie along `axis` (default: the last axis), which must have length
 * 3 in both inputs. All other axes are batch axes that broadcast against each
 * other under NumPy rules, so `[N, 3]` with `[3]` or `[N, 1, 3]` with `[M, 3]`
 * work. The result has the broadcast batch shape and the vectors along the same
 * `axis` position.
 *
 * The result dtype is the promoted dtype of the inputs: float32 inputs give
 * float32, float64 gives float64 (float16 with bfloat16 gives float32). Integer
 * inputs give integer results (uint8, int32, or int64) that wrap on overflow, as
 * in NumPy; bool counts as int32.
 *
 * @param a - First tensor with `a.shape[axis] === 3`
 * @param b - Second tensor with `b.shape[axis] === 3`
 * @param options - `{ axis }`, or the axis number itself (default: -1)
 * @returns Cross product `a x b`
 * @throws {ShapeError} If an input is 0-d, the cross axis does not have length 3, or the
 *   batch axes do not broadcast
 * @throws {InvalidParameterError} If `axis` is out of range
 * @throws {DTypeError} If an input has string dtype
 *
 * @example
 * ```ts
 * cross(tensor([1, 0, 0]), tensor([0, 1, 0])); // [0, 0, 1]
 *
 * // Batched: every row of `a` crossed with the same vector.
 * cross(tensor([[1, 0, 0], [0, 1, 0]]), tensor([0, 0, 1])); // [[0, -1, 0], [1, 0, 0]]
 *
 * // Vectors stored along axis 0.
 * cross(tensor([[1, 0], [0, 1], [0, 0]]), tensor([[0, 0], [0, 0], [1, 1]]), { axis: 0 });
 * ```
 */
export function cross(a: Tensor, b: Tensor, options?: CrossOptions | Axis): Tensor {
  if (a.dtype === "string" || b.dtype === "string") {
    throw new DTypeError("cross is not defined for string dtype");
  }
  const axis: Axis =
    typeof options === "object" && options !== null ? (options.axis ?? -1) : (options ?? -1);
  if (a.ndim === 0 || b.ndim === 0) {
    throw new ShapeError("cross requires tensors with at least 1 dimension");
  }
  const axA = normalizeAxis(axis, a.ndim);
  const axB = normalizeAxis(axis, b.ndim);
  if (a.shape[axA] !== 3) {
    throw new ShapeError(
      `cross requires length 3 along the cross axis; a has shape [${a.shape}] and axis ${axA}`
    );
  }
  if (b.shape[axB] !== 3) {
    throw new ShapeError(
      `cross requires length 3 along the cross axis; b has shape [${b.shape}] and axis ${axB}`
    );
  }

  const batchShapeA = a.shape.filter((_, d) => d !== axA);
  const batchStridesA = a.strides.filter((_, d) => d !== axA);
  const batchShapeB = b.shape.filter((_, d) => d !== axB);
  const batchStridesB = b.strides.filter((_, d) => d !== axB);
  const batch = broadcastShapeTwo(batchShapeA, batchShapeB);
  const stepsA = broadcastStrides(batchShapeA, batchStridesA, batch);
  const stepsB = broadcastStrides(batchShapeB, batchStridesB, batch);
  const compA = a.strides[axA] as number;
  const compB = b.strides[axB] as number;
  const outAxis = normalizeAxis(axis, batch.length + 1);

  const dtype = crossDType(ensureNonStringDType(a.dtype), ensureNonStringDType(b.dtype));
  const ad = requireNumericData(a.data, "cross");
  const bd = requireNumericData(b.data, "cross");
  const outSize = shapeToSize(batch) * 3;
  const outLogical = computeStrides([...batch, 3]);
  const stepsO = batch.map((_, d) => outLogical[d] as number);

  let out: TypedArray;
  if (dtype === "int64") {
    const res = new BigInt64Array(outSize);
    const rd = (data: TypedArray, i: number): bigint => BigInt(readElement(data, i));
    walkSteps(batch, stepsA, stepsB, stepsO, a.offset, b.offset, 0, (ai, bi, oi) => {
      const a0 = rd(ad, ai);
      const a1 = rd(ad, ai + compA);
      const a2 = rd(ad, ai + 2 * compA);
      const b0 = rd(bd, bi);
      const b1 = rd(bd, bi + compB);
      const b2 = rd(bd, bi + 2 * compB);
      res[oi] = BigInt.asIntN(64, a1 * b2 - a2 * b1);
      res[oi + 1] = BigInt.asIntN(64, a2 * b0 - a0 * b2);
      res[oi + 2] = BigInt.asIntN(64, a0 * b1 - a1 * b0);
    });
    out = res;
  } else if (dtype === "int32" || dtype === "uint8") {
    const res = dtype === "int32" ? new Int32Array(outSize) : new Uint8Array(outSize);
    walkSteps(batch, stepsA, stepsB, stepsO, a.offset, b.offset, 0, (ai, bi, oi) => {
      const a0 = readAsNumber(ad, ai);
      const a1 = readAsNumber(ad, ai + compA);
      const a2 = readAsNumber(ad, ai + 2 * compA);
      const b0 = readAsNumber(bd, bi);
      const b1 = readAsNumber(bd, bi + compB);
      const b2 = readAsNumber(bd, bi + 2 * compB);
      res[oi] = Math.imul(a1, b2) - Math.imul(a2, b1);
      res[oi + 1] = Math.imul(a2, b0) - Math.imul(a0, b2);
      res[oi + 2] = Math.imul(a0, b1) - Math.imul(a1, b0);
    });
    out = res;
  } else {
    const res = dtype === "float64" ? new Float64Array(outSize) : new Float32Array(outSize);
    walkSteps(batch, stepsA, stepsB, stepsO, a.offset, b.offset, 0, (ai, bi, oi) => {
      const a0 = readAsNumber(ad, ai);
      const a1 = readAsNumber(ad, ai + compA);
      const a2 = readAsNumber(ad, ai + 2 * compA);
      const b0 = readAsNumber(bd, bi);
      const b1 = readAsNumber(bd, bi + compB);
      const b2 = readAsNumber(bd, bi + 2 * compB);
      res[oi] = a1 * b2 - a2 * b1;
      res[oi + 1] = a2 * b0 - a0 * b2;
      res[oi + 2] = a0 * b1 - a1 * b0;
    });
    out = res;
  }

  const result = roundHalfResult(
    Tensor.fromTypedArray({ data: out, shape: [...batch, 3], dtype, device: a.device })
  );
  return outAxis === batch.length ? result : contiguous(moveaxis(result, -1, outAxis));
}

/**
 * Return coordinate matrices from coordinate vectors.
 *
 * Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector
 * fields over N-D grids, given one-dimensional coordinate arrays x1, x2, ..., xn.
 * Each output keeps the dtype of its input vector.
 *
 * Supports `"xy"` (default, Cartesian) and `"ij"` (matrix) indexing. With `"xy"`
 * the first two output axes are swapped: the outputs have shape
 * `[len(x2), len(x1), len(x3), ...]`.
 *
 * @param args - 1-D coordinate tensors, optionally followed by `{ indexing }`
 * @returns Array of N-D tensors, one for each input
 * @throws {ShapeError} If an input is not 1-D
 * @throws {InvalidParameterError} If `indexing` is neither `"xy"` nor `"ij"`
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

  // A trailing plain object (anything that is not a Tensor) holds the options.
  const lastArg = args[args.length - 1];
  if (lastArg !== undefined && lastArg !== null && !(lastArg instanceof Tensor)) {
    const opts = lastArg as { readonly indexing?: "xy" | "ij" };
    if (opts.indexing !== undefined) indexing = opts.indexing;
    tensors = args.slice(0, -1) as Tensor[];
  } else {
    tensors = args as Tensor[];
  }

  if (indexing !== "xy" && indexing !== "ij") {
    throw new InvalidParameterError(
      `meshgrid indexing must be 'xy' or 'ij'; got '${String(indexing)}'`,
      "indexing",
      indexing
    );
  }
  if (tensors.length === 0) return [];

  for (const t of tensors) {
    if (t.ndim !== 1) {
      throw new ShapeError(`meshgrid requires 1-D input tensors; got ${t.ndim}-D`);
    }
  }

  const ndim = tensors.length;
  const swapXY = indexing === "xy" && ndim >= 2;
  const sizes = tensors.map((t) => t.shape[0] ?? 0);
  // Output axis a holds the coordinates of input `axisOf[a]`.
  const axisOf = Array.from({ length: ndim }, (_, i) => i);
  if (swapXY) {
    axisOf[0] = 1;
    axisOf[1] = 0;
  }
  const outShape = axisOf.map((i) => sizes[i] ?? 0);
  const totalSize = shapeToSize(outShape);

  const result: Tensor[] = [];
  for (let i = 0; i < ndim; i++) {
    const t = tensors[i] as Tensor;
    const dtype = ensureNonStringDType(t.dtype);
    const srcData = requireNumericData(t.data, "meshgrid");
    const outData = new (dtypeToTypedArrayCtor(dtype))(totalSize);

    // Along output axis `dim`, each coordinate value repeats for `inner`
    // consecutive elements, and that pattern of length `dimSize * inner` tiles
    // the whole output. Fill the first tile with `fill()`, then replicate it
    // with doubling `copyWithin` instead of decoding every flat index.
    const dim = axisOf.indexOf(i);
    const dimSize = outShape[dim] ?? 0;
    const srcStride = t.strides[0] ?? 0;
    let inner = 1;
    for (let d = dim + 1; d < ndim; d++) inner *= outShape[d] ?? 0;
    const tile = dimSize * inner;
    if (totalSize > 0 && tile > 0) {
      for (let j = 0; j < dimSize; j++) {
        fillRange(
          outData,
          readElement(srcData, t.offset + j * srcStride),
          j * inner,
          (j + 1) * inner
        );
      }
      for (let filled = tile; filled < totalSize; filled *= 2) {
        outData.copyWithin(filled, 0, Math.min(filled, totalSize - filled));
      }
    }

    result.push(
      Tensor.fromTypedArray({
        data: outData,
        shape: [...outShape],
        dtype,
        device: t.device,
      })
    );
  }

  return result;
}

/**
 * Select elements along a dimension using an index tensor.
 *
 * Returns a new tensor that indexes the input tensor along dimension `dim`
 * using the entries in `index` (a 1-D integer tensor). The result keeps the
 * dtype of `input`. Negative indices are not accepted.
 *
 * Equivalent to PyTorch's `torch.index_select(input, dim, index)`.
 *
 * @param input - The input tensor
 * @param dim - The dimension to index along
 * @param index - 1-D tensor of indices to select
 * @returns A new tensor with the selected elements
 * @throws {ShapeError} If `index` is not 1-D
 * @throws {IndexError} If an index is outside `[0, size)` for `dim`
 * @throws {InvalidParameterError} If `dim` is out of range or an index is not an integer
 *
 * @example
 * ```ts
 * const x = tensor([[1, 2, 3], [4, 5, 6]]);
 * const idx = tensor([0, 2], { dtype: 'int32' });
 * const result = indexSelect(x, 1, idx); // [[1, 3], [4, 6]]
 * ```
 * @deprecated Prefer {@link indexSelect}.
 */
export function index_select(input: Tensor, dim: Axis, index: Tensor): Tensor {
  if (input.dtype === "string") {
    throw new DTypeError("index_select is not defined for string dtype");
  }
  if (index.dtype === "string") {
    throw new DTypeError("index_select index must be numeric");
  }
  if (index.ndim !== 1) {
    throw new ShapeError(`index_select requires a 1-D index tensor; got ${index.ndim}-D`);
  }
  const dtype = ensureNonStringDType(input.dtype);

  const ndim = input.ndim;
  const resolvedDim = normalizeAxis(dim, ndim);
  const dimSize = input.shape[resolvedDim] ?? 0;
  const mapped = resolveIndices(index, dimSize, "index_select", false);
  const nIdx = mapped.length;

  const outShape = [...input.shape];
  outShape[resolvedDim] = nIdx;
  const totalSize = shapeToSize(outShape);
  const outData = new (dtypeToTypedArrayCtor(dtype))(totalSize);
  const srcData = requireNumericData(input.data, "index_select");

  if (totalSize > 0) {
    let inner = 1;
    for (let d = resolvedDim + 1; d < ndim; d++) inner *= input.shape[d] ?? 0;
    if (inner >= 16 && isContiguous(input.shape, input.strides)) {
      // Dense layout with long slabs: every selected index is a contiguous run
      // of `inner` elements, copied as a block.
      const slab = dimSize * inner;
      const outer = totalSize / (nIdx * inner);
      let pos = 0;
      for (let o = 0; o < outer; o++) {
        const srcBase = input.offset + o * slab;
        for (let j = 0; j < nIdx; j++) {
          const start = srcBase + (mapped[j] as number) * inner;
          copyRange(outData, srcData, start, start + inner, pos);
          pos += inner;
        }
      }
    } else {
      const maps = outShape.map((size, d) => {
        if (d === resolvedDim) return mapped;
        const identity = new Int32Array(size);
        for (let i = 0; i < size; i++) identity[i] = i;
        return identity;
      });
      gatherTyped(srcData, input.offset, input.strides, outShape, maps, outData);
    }
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: outShape,
    dtype,
    device: input.device,
  });
}

/**
 * Test whether each element of `a` is in `values`.
 *
 * Returns a boolean tensor of the same shape as `a`. NaN is never a member
 * (NaN never equals NaN, as in NumPy). String tensors can be tested against a
 * string tensor or a list of strings; int64 values are compared exactly.
 *
 * @param a - Input tensor
 * @param values - Values to test against (a tensor or an array)
 * @param invert - If true, return the negation (`a` not in `values`)
 * @returns Boolean tensor with the shape of `a`
 * @throws {DTypeError} If string and numeric inputs are mixed
 *
 * @example
 * ```ts
 * isin(tensor([1, 2, 3, 4]), [2, 4]); // [0, 1, 0, 1]
 * ```
 */
export function isin(
  a: Tensor,
  values: Tensor | readonly number[] | readonly string[],
  invert = false
): Tensor {
  const hit = invert ? 0 : 1;
  const miss = invert ? 1 : 0;
  const n = a.size;
  const out = new Uint8Array(n);

  // An empty list is compatible with every dtype.
  const valueIsString =
    values instanceof Tensor
      ? values.dtype === "string"
      : values.length === 0
        ? a.dtype === "string"
        : typeof values[0] === "string";
  if (valueIsString !== (a.dtype === "string")) {
    throw new DTypeError("isin cannot compare string values with numeric values");
  }

  if (a.dtype === "string") {
    const list = values instanceof Tensor ? clone(values).data : values;
    const set = new Set<string>(list as readonly string[]);
    const src = clone(a).data as string[];
    for (let i = 0; i < n; i++) out[i] = set.has(src[i] as string) ? hit : miss;
  } else if (a.dtype === "int64") {
    // Compare as BigInt so values beyond 2^53 are not merged.
    const set = new Set<bigint>();
    if (values instanceof Tensor) {
      const vv = logicalData(values, "isin");
      for (let i = 0; i < vv.length; i++) {
        const v = vv[i] as number | bigint;
        if (typeof v === "bigint") set.add(v);
        else if (Number.isInteger(v)) set.add(BigInt(v));
      }
    } else {
      for (const v of values as readonly number[]) if (Number.isInteger(v)) set.add(BigInt(v));
    }
    const src = logicalData(a, "isin") as BigInt64Array;
    for (let i = 0; i < n; i++) out[i] = set.has(src[i] as bigint) ? hit : miss;
  } else {
    const set = new Set<number>();
    if (values instanceof Tensor) {
      const vv = readNumbers(values, "isin", false);
      for (let i = 0; i < values.size; i++) {
        const v = vv[i] as number;
        if (!Number.isNaN(v)) set.add(v);
      }
    } else {
      for (const v of values as readonly number[]) if (!Number.isNaN(v)) set.add(v);
    }
    const src = readNumbers(a, "isin", false);
    for (let i = 0; i < n; i++) out[i] = set.has(src[i] as number) ? hit : miss;
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

/**
 * Broadcast a tensor to a target shape.
 *
 * The result is a zero-copy view: broadcast axes get stride 0, so all elements
 * along them alias the same buffer entry (like NumPy's `broadcast_to`). Call
 * {@link clone} to get independent memory.
 *
 * @param t - Input tensor
 * @param shape - Target shape
 * @returns Broadcast view of the tensor
 * @throws {ShapeError} If the target has fewer dimensions or an axis cannot be broadcast
 * @throws {InvalidParameterError} If the target shape has a negative or non-integer entry
 */
export const broadcastTo = broadcast_to;
/**
 * Ensure the input is at least 1-D.
 * Scalars become shape [1]; higher-dimensional tensors pass through unchanged
 * (the same tensor is returned).
 */
export const atleast1d = atleast_1d;
/**
 * Ensure the input is at least 2-D.
 * Scalars become shape [1,1]; 1-D tensors become shape [1,n]; others pass through
 * unchanged (the same tensor is returned).
 */
export const atleast2d = atleast_2d;
/**
 * Create a tensor of zeros with the same shape as the input tensor.
 *
 * @param t - Reference tensor
 * @param options - `dtype` overrides the result dtype (default: the dtype of `t`)
 * @returns New tensor filled with zeros (empty strings for string dtype)
 *
 * @example
 * ```ts
 * const a = tensor([[1, 2], [3, 4]]);
 * const z = zerosLike(a); // [[0, 0], [0, 0]]
 * ```
 */
export const zerosLike = zeros_like;
/**
 * Create a tensor of ones with the same shape as the input tensor.
 *
 * @param t - Reference tensor
 * @param options - `dtype` overrides the result dtype (default: the dtype of `t`)
 * @returns New tensor filled with ones
 * @throws {DTypeError} If the resulting dtype is string
 *
 * @example
 * ```ts
 * const a = tensor([[1, 2], [3, 4]]);
 * const o = onesLike(a); // [[1, 1], [1, 1]]
 * ```
 */
export const onesLike = ones_like;
/**
 * Create a zero-filled tensor with the same shape as the input tensor.
 * JavaScript typed arrays are always zero-initialized, so unlike NumPy's
 * `empty_like` the contents are well defined.
 *
 * @param t - Reference tensor
 * @param options - `dtype` overrides the result dtype (default: the dtype of `t`)
 * @returns New tensor (zero-initialized)
 */
export const emptyLike = empty_like;
/**
 * Create a tensor filled with a specified value, matching the shape of the input.
 *
 * The value is converted to the result dtype: `bool` stores 1 for any non-zero
 * value (NaN included), integer dtypes truncate toward zero and reject NaN and
 * infinities, and `int64` stores a BigInt.
 *
 * @param t - Reference tensor
 * @param fillValue - Value to fill the tensor with
 * @param options - `dtype` overrides the result dtype (default: the dtype of `t`)
 * @returns New tensor filled with the specified value
 * @throws {DTypeError} If the resulting dtype is string
 * @throws {InvalidParameterError} If a non-finite value is stored in an integer dtype
 *
 * @example
 * ```ts
 * const a = tensor([[1, 2], [3, 4]]);
 * const f = fullLike(a, 7); // [[7, 7], [7, 7]]
 * ```
 */
export const fullLike = full_like;
/**
 * Select elements along a dimension using an index tensor.
 *
 * Returns a new tensor that indexes the input tensor along dimension `dim`
 * using the entries in `index` (a 1-D integer tensor). The result keeps the
 * dtype of `input`. Negative indices are not accepted.
 *
 * Equivalent to PyTorch's `torch.index_select(input, dim, index)`.
 *
 * @param input - The input tensor
 * @param dim - The dimension to index along
 * @param index - 1-D tensor of indices to select
 * @returns A new tensor with the selected elements
 * @throws {ShapeError} If `index` is not 1-D
 * @throws {IndexError} If an index is outside `[0, size)` for `dim`
 * @throws {InvalidParameterError} If `dim` is out of range or an index is not an integer
 *
 * @example
 * ```ts
 * const x = tensor([[1, 2, 3], [4, 5, 6]]);
 * const idx = tensor([0, 2], { dtype: 'int32' });
 * const result = indexSelect(x, 1, idx); // [[1, 3], [4, 6]]
 * ```
 */
export const indexSelect = index_select;
/**
 * Flip a tensor left-right (reverse the order of columns, axis 1).
 *
 * @param t - Input tensor with at least 2 dimensions
 * @returns Horizontally flipped tensor
 * @throws {ShapeError} If the input has fewer than 2 dimensions
 */
export const flipLr = fliplr;
/**
 * Flip a tensor up-down (reverse the order of rows, axis 0).
 *
 * @param t - Input tensor with at least 1 dimension
 * @returns Vertically flipped tensor
 * @throws {ShapeError} If the input is a scalar
 */
export const flipUd = flipud;
