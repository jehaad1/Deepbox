/**
 * Tensor manipulation operations.
 *
 * This module provides functions for joining, splitting and repeating tensors:
 * - concatenate: Join tensors along an existing axis
 * - stack: Join tensors along a new axis
 * - split: Split a tensor into several sub-tensors
 * - tile: Repeat a tensor along each axis
 * - repeat: Repeat individual elements
 *
 * Padding and flipping live in `ops/utils`. Every operation returns a new
 * tensor that owns its data (nothing aliases the inputs) and accepts
 * non-contiguous views, which are gathered once before the block copies.
 * These operations run on host memory; a tensor on a kernel device (such as
 * `webgpu`) raises a `DeviceError` asking for `await t.cpu()` first.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */

import type { Axis, Device, DType, TypedArray } from "../../core";
import {
  DeepboxError,
  DeviceError,
  DTypeError,
  dtypeToTypedArrayCtor,
  InvalidParameterError,
  normalizeAxis,
  ShapeError,
  shapeToSize,
} from "../../core";
import { isContiguous } from "../tensor/strides";
import { computeStrides, Tensor } from "../tensor/Tensor";
import { readNumericContiguous } from "./_internal";

// ─── Internal helpers ─────────────────────────────────────────────────────────

/** Backing storage of a tensor: a typed array, or a plain array for strings. */
type Storage = TypedArray | string[];

/** Element access that does not care about the concrete storage class. */
type Indexable = { [index: number]: unknown };

/** Ranges shorter than this are copied with a plain loop instead of `set`/`copyWithin`. */
const SMALL_RANGE = 24;

/**
 * Reject tensors that live in kernel-device memory. These operations copy
 * blocks of host memory and have no device kernel, so the caller gets an
 * explicit hint instead of the generic synchronous-access error.
 */
function assertHostTensors(op: string, tensors: readonly Tensor[]): void {
  for (const t of tensors) {
    if (t.isDeviceTensor) {
      throw new DeviceError(
        `${op} is not available on device "${t.device}". ` +
          "Move the tensor to the CPU first with `await t.cpu()`."
      );
    }
  }
}

function allocate(dtype: DType, size: number): Storage {
  if (dtype === "string") return new Array<string>(size).fill("");
  const Ctor = dtypeToTypedArrayCtor(dtype);
  return new Ctor(size);
}

function build(data: Storage, shape: number[], dtype: DType, device: Device): Tensor {
  if (Array.isArray(data)) {
    return Tensor.fromStringArray({ data, shape, device });
  }
  if (dtype === "string") {
    throw new DeepboxError("Internal error: string dtype but non-array data");
  }
  return Tensor.fromTypedArray({ data, shape, dtype, device });
}

/**
 * Logical elements of `t` in row-major order, as `data[start .. start + t.size)`.
 *
 * Contiguous tensors are returned as they are (`owned` is false, the buffer
 * belongs to the tensor); strided views are gathered into a fresh dense buffer
 * (`owned` is true).
 */
function toDense(t: Tensor): { data: Storage; start: number; owned: boolean } {
  const data = t.data as Storage;
  if (t.size === 0 || isContiguous(t.shape, t.strides)) {
    return { data, start: t.offset, owned: false };
  }
  if (!Array.isArray(data) && !(data instanceof BigInt64Array)) {
    const dense = readNumericContiguous(t);
    if (dense === null) throw new DeepboxError("Internal error: expected numeric data");
    return { data: dense, start: 0, owned: true };
  }
  const out = allocate(t.dtype, t.size) as Indexable;
  const src = data as Indexable;
  const shape = t.shape;
  const strides = t.strides;
  const ndim = shape.length;
  const inner = shape[ndim - 1] ?? 1;
  const innerStride = strides[ndim - 1] ?? 0;
  const rows = t.size / inner;
  const coords = new Array<number>(ndim).fill(0);
  let base = t.offset;
  let pos = 0;
  for (let r = 0; r < rows; r++) {
    let idx = base;
    for (let i = 0; i < inner; i++) {
      out[pos++] = src[idx];
      idx += innerStride;
    }
    for (let d = ndim - 2; d >= 0; d--) {
      coords[d] = (coords[d] ?? 0) + 1;
      base += strides[d] ?? 0;
      if ((coords[d] ?? 0) < (shape[d] ?? 0)) break;
      base -= (strides[d] ?? 0) * (shape[d] ?? 0);
      coords[d] = 0;
    }
  }
  return { data: out as Storage, start: 0, owned: true };
}

/** Copy `src[s .. s + len)` into `dst[d .. d + len)`; both have the same dtype. */
function copyRange(src: Storage, s: number, dst: Storage, d: number, len: number): void {
  if (len <= 0) return;
  if (len < SMALL_RANGE || Array.isArray(src)) {
    const a = src as Indexable;
    const b = dst as Indexable;
    for (let i = 0; i < len; i++) b[d + i] = a[s + i];
    return;
  }
  (dst as Float64Array).set((src as Float64Array).subarray(s, s + len), d);
}

/** Copy `buf[from .. from + len)` to `buf[to .. to + len)`; the ranges must not overlap. */
function copyWithinRange(buf: Storage, from: number, to: number, len: number): void {
  if (len <= 0) return;
  if (len < SMALL_RANGE || Array.isArray(buf)) {
    const b = buf as Indexable;
    for (let i = 0; i < len; i++) b[to + i] = b[from + i];
    return;
  }
  (buf as Float64Array).copyWithin(to, from, from + len);
}

/**
 * Given `buf[pos .. pos + blockLen)`, write `reps - 1` further copies of that
 * block right after it. Doubles the filled region each step, so the number of
 * native copies is logarithmic in `reps`.
 */
function replicate(buf: Storage, pos: number, blockLen: number, reps: number): void {
  if (blockLen <= 0) return;
  let filled = 1;
  while (filled < reps) {
    const copies = Math.min(filled, reps - filled);
    copyWithinRange(buf, pos, pos + filled * blockLen, copies * blockLen);
    filled += copies;
  }
}

/** Write `src[s]` into `dst[d .. d + count)`. */
function fillFrom(src: Storage, s: number, dst: Storage, d: number, count: number): void {
  if (count <= 0) return;
  if (count < 8 || Array.isArray(dst)) {
    const a = src as Indexable;
    const b = dst as Indexable;
    const v = a[s];
    for (let i = 0; i < count; i++) b[d + i] = v;
    return;
  }
  if (dst instanceof BigInt64Array) {
    dst.fill((src as BigInt64Array)[s] as bigint, d, d + count);
  } else {
    (dst as Float64Array).fill((src as Float64Array)[s] as number, d, d + count);
  }
}

function productOf(shape: readonly number[], from: number, to: number): number {
  let p = 1;
  for (let d = from; d < to; d++) p *= shape[d] ?? 1;
  return p;
}

// ─── concatenate / stack ──────────────────────────────────────────────────────

/**
 * Concatenate tensors along an existing axis.
 *
 * All tensors must have the same number of dimensions, the same dtype and the
 * same shape except along `axis`. Mixed dtypes are rejected rather than
 * promoted; convert with `astype` first. The result is always a new tensor,
 * also when a single tensor is passed.
 *
 * **Complexity**: O(n) where n is total number of elements
 *
 * @param tensors - Array of tensors to concatenate
 * @param axis - Axis along which to concatenate (default: 0)
 * @returns Concatenated tensor
 * @throws {InvalidParameterError} If `tensors` is empty or `axis` is out of range
 * @throws {ShapeError} If ndim or the non-concatenation dimensions differ
 * @throws {DTypeError} If the dtypes differ
 *
 * @example
 * ```ts
 * const a = tensor([[1, 2], [3, 4]]);
 * const b = tensor([[5, 6]]);
 * const c = concatenate([a, b], 0);  // [[1, 2], [3, 4], [5, 6]]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function concatenate(tensors: Tensor[], axis: Axis = 0): Tensor {
  if (tensors.length === 0) {
    throw new InvalidParameterError("concatenate requires at least one tensor", "tensors");
  }
  assertHostTensors("concatenate", tensors);

  const first = tensors[0];
  if (!first) throw new DeepboxError("Unexpected: tensor at index 0 is undefined");
  const ndim = first.ndim;
  const dtype = first.dtype;
  const ax = normalizeAxis(axis, ndim);

  for (let i = 1; i < tensors.length; i++) {
    const t = tensors[i];
    if (!t) throw new DeepboxError(`Unexpected: tensor at index ${i} is undefined`);
    if (t.ndim !== ndim) {
      throw new ShapeError(`All tensors must have same ndim; got ${ndim} and ${t.ndim}`);
    }
    if (t.dtype !== dtype) {
      throw new DTypeError(`All tensors must have same dtype; got ${dtype} and ${t.dtype}`);
    }
  }

  for (let i = 1; i < tensors.length; i++) {
    const t = tensors[i];
    if (!t) throw new DeepboxError(`Unexpected: tensor at index ${i} is undefined`);
    for (let d = 0; d < ndim; d++) {
      if (d !== ax && first.shape[d] !== t.shape[d]) {
        throw ShapeError.mismatch(
          first.shape,
          t.shape,
          `concatenate: shapes must match except on axis ${axis}`
        );
      }
    }
  }

  const outShape = [...first.shape];
  let total = 0;
  for (const t of tensors) total += t.shape[ax] ?? 0;
  outShape[ax] = total;

  // View every tensor as [outer, axisLen, inner]; each (tensor, outer) pair is
  // one contiguous block in both the source and the output.
  const outer = productOf(outShape, 0, ax);
  const inner = productOf(outShape, ax + 1, ndim);
  const rowLen = total * inner;

  if (tensors.length === 1) {
    const { data, start, owned } = toDense(first);
    let copy = data;
    if (!owned) {
      copy = allocate(dtype, first.size);
      copyRange(data, start, copy, 0, first.size);
    }
    return build(copy, outShape, dtype, first.device);
  }

  const out = allocate(dtype, shapeToSize(outShape));
  let along = 0;
  for (const t of tensors) {
    const axisLen = t.shape[ax] ?? 0;
    const len = axisLen * inner;
    if (len > 0 && outer > 0) {
      const { data, start } = toDense(t);
      if (outer === 1) {
        copyRange(data, start, out, along * inner, len);
      } else {
        for (let o = 0; o < outer; o++) {
          copyRange(data, start + o * len, out, o * rowLen + along * inner, len);
        }
      }
    }
    along += axisLen;
  }
  return build(out, outShape, dtype, first.device);
}

/**
 * Stack tensors along a new axis.
 *
 * All tensors must have exactly the same shape and dtype. The result has one
 * more dimension than the inputs, with size `tensors.length` at `axis`.
 *
 * **Complexity**: O(n) where n is total number of elements
 *
 * @param tensors - Array of tensors to stack
 * @param axis - Position of the new axis in the result, `-(ndim + 1)` to `ndim` (default: 0)
 * @returns Stacked tensor
 * @throws {InvalidParameterError} If `tensors` is empty or `axis` is out of range
 * @throws {ShapeError} If the shapes differ
 * @throws {DTypeError} If the dtypes differ
 *
 * @example
 * ```ts
 * const a = tensor([1, 2, 3]);
 * const b = tensor([4, 5, 6]);
 * const c = stack([a, b], 0);  // [[1, 2, 3], [4, 5, 6]]
 * const d = stack([a, b], 1);  // [[1, 4], [2, 5], [3, 6]]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function stack(tensors: Tensor[], axis: Axis = 0): Tensor {
  if (tensors.length === 0) {
    throw new InvalidParameterError("stack requires at least one tensor", "tensors");
  }
  assertHostTensors("stack", tensors);

  const first = tensors[0];
  if (!first) {
    throw new DeepboxError("Unexpected: first tensor is undefined");
  }
  const ndim = first.ndim;
  const dtype = first.dtype;

  // The result has ndim + 1 dimensions, so the new axis ranges over [-ndim-1, ndim].
  const ax = normalizeAxis(axis, ndim + 1);

  for (let i = 1; i < tensors.length; i++) {
    const t = tensors[i];
    if (!t) throw new DeepboxError(`Unexpected: tensor at index ${i} is undefined`);
    if (t.ndim !== ndim) {
      throw new ShapeError(`All tensors must have same ndim; got ${ndim} and ${t.ndim}`);
    }
    if (t.dtype !== dtype) {
      throw new DTypeError(`All tensors must have same dtype; got ${dtype} and ${t.dtype}`);
    }
    for (let d = 0; d < ndim; d++) {
      if (first.shape[d] !== t.shape[d]) {
        throw ShapeError.mismatch(first.shape, t.shape, "stack: all tensors must have same shape");
      }
    }
  }

  const outShape = [...first.shape];
  outShape.splice(ax, 0, tensors.length);

  // View the output as [outer, count, inner]: tensor k contributes one block of
  // `inner` elements to every outer index.
  const count = tensors.length;
  const outer = productOf(first.shape, 0, ax);
  const inner = productOf(first.shape, ax, ndim);
  const out = allocate(dtype, shapeToSize(outShape));

  if (inner > 0 && outer > 0) {
    for (let k = 0; k < count; k++) {
      const t = tensors[k];
      if (!t) throw new DeepboxError(`Unexpected: tensor at index ${k} is undefined`);
      const { data, start } = toDense(t);
      if (outer === 1) {
        copyRange(data, start, out, k * inner, inner);
      } else {
        for (let o = 0; o < outer; o++) {
          copyRange(data, start + o * inner, out, (o * count + k) * inner, inner);
        }
      }
    }
  }
  return build(out, outShape, dtype, first.device);
}

// ─── split ────────────────────────────────────────────────────────────────────

/**
 * Split tensor into multiple sub-tensors along an axis.
 *
 * If `indices_or_sections` is an integer, the tensor is split into that many
 * equal parts (the axis length must be divisible by it). If it is an array, it
 * lists the indices at which to split; the indices must be integers in
 * `[0, axisLength]` in non-decreasing order, and the result has one more part
 * than there are indices. The parts are copies, not views.
 *
 * **Complexity**: O(n) where n is total number of elements
 *
 * @param t - Input tensor
 * @param indices_or_sections - Number of sections or array of split indices
 * @param axis - Axis along which to split (default: 0)
 * @returns Array of sub-tensors
 * @throws {InvalidParameterError} If the sections or indices are invalid, or `axis` is out of range
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5, 6]);
 * const parts = split(t, 3);  // [tensor([1, 2]), tensor([3, 4]), tensor([5, 6])]
 * const parts2 = split(t, [2, 4]);  // [tensor([1, 2]), tensor([3, 4]), tensor([5, 6])]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function split(t: Tensor, indices_or_sections: number | number[], axis: Axis = 0): Tensor[] {
  assertHostTensors("split", [t]);
  const ax = normalizeAxis(axis, t.ndim);
  const axisSize = t.shape[ax] ?? 0;

  let splitPoints: number[];
  if (typeof indices_or_sections === "number") {
    const numSections = indices_or_sections;
    if (!Number.isInteger(numSections) || numSections <= 0) {
      throw new InvalidParameterError(
        `indices_or_sections must be a positive integer; received ${numSections}`,
        "indices_or_sections",
        numSections
      );
    }
    if (axisSize % numSections !== 0) {
      throw new InvalidParameterError(
        `axis dimension ${axisSize} not divisible by ${numSections} equal sections`,
        "indices_or_sections",
        numSections
      );
    }
    const sectionSize = axisSize / numSections;
    splitPoints = [];
    for (let i = 1; i < numSections; i++) {
      splitPoints.push(i * sectionSize);
    }
  } else {
    splitPoints = [...indices_or_sections];
    let prev = 0;
    for (let i = 0; i < splitPoints.length; i++) {
      const idx = splitPoints[i];
      if (idx === undefined || !Number.isInteger(idx)) {
        throw new InvalidParameterError(
          `split index must be an integer; received ${String(idx)}`,
          "indices_or_sections",
          indices_or_sections
        );
      }
      if (idx < 0 || idx > axisSize) {
        throw new InvalidParameterError(
          `split index ${idx} is out of bounds for axis size ${axisSize}`,
          "indices_or_sections",
          indices_or_sections
        );
      }
      if (i > 0 && idx < prev) {
        throw new InvalidParameterError(
          "split indices must be non-decreasing",
          "indices_or_sections",
          indices_or_sections
        );
      }
      prev = idx;
    }
  }

  const boundaries = [0, ...splitPoints, axisSize];
  const outer = productOf(t.shape, 0, ax);
  const inner = productOf(t.shape, ax + 1, t.ndim);
  const dense = t.size > 0 ? toDense(t) : null;

  const result: Tensor[] = [];
  for (let i = 0; i < boundaries.length - 1; i++) {
    const start = boundaries[i] ?? 0;
    const end = boundaries[i + 1] ?? axisSize;
    const subShape = [...t.shape];
    subShape[ax] = end - start;

    const out = allocate(t.dtype, shapeToSize(subShape));
    const len = (end - start) * inner;
    if (dense && len > 0) {
      for (let o = 0; o < outer; o++) {
        copyRange(dense.data, dense.start + (o * axisSize + start) * inner, out, o * len, len);
      }
    }
    result.push(build(out, subShape, t.dtype, t.device));
  }
  return result;
}

// ─── tile / repeat ────────────────────────────────────────────────────────────

/**
 * Repeat tensor along axes by tiling.
 *
 * Constructs a tensor by repeating the input the given number of times along
 * each axis. If `reps` has fewer entries than the tensor has dimensions it is
 * padded with ones on the left; if the tensor has fewer dimensions than `reps`
 * has entries, the tensor is treated as having leading axes of size 1. A
 * repetition count of 0 produces an empty tensor.
 *
 * **Complexity**: O(n * product(reps)) where n is input size
 *
 * @param t - Input tensor
 * @param reps - Number of repetitions along each axis
 * @returns Tiled tensor
 * @throws {InvalidParameterError} If `reps` is empty or contains a negative or non-integer value
 *
 * @example
 * ```ts
 * const t = tensor([[1, 2], [3, 4]]);
 * const tiled = tile(t, [2, 3]);
 * // [[1, 2, 1, 2, 1, 2],
 * //  [3, 4, 3, 4, 3, 4],
 * //  [1, 2, 1, 2, 1, 2],
 * //  [3, 4, 3, 4, 3, 4]]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function tile(t: Tensor, reps: number[]): Tensor {
  assertHostTensors("tile", [t]);
  if (reps.length === 0) {
    throw new InvalidParameterError("reps must have at least one element", "reps");
  }
  for (let i = 0; i < reps.length; i++) {
    const rep = reps[i];
    if (rep === undefined || !Number.isInteger(rep) || rep < 0) {
      throw new InvalidParameterError(
        `reps[${i}] must be a non-negative integer; received ${String(rep)}`,
        "reps",
        reps
      );
    }
  }

  // Align shape and reps on their trailing dimensions.
  const ndim = Math.max(t.ndim, reps.length);
  const inShape = new Array<number>(ndim).fill(1);
  const repCounts = new Array<number>(ndim).fill(1);
  for (let i = 0; i < t.ndim; i++) {
    inShape[ndim - t.ndim + i] = t.shape[i] ?? 1;
  }
  for (let i = 0; i < reps.length; i++) {
    repCounts[ndim - reps.length + i] = reps[i] ?? 1;
  }

  const outShape = inShape.map((s, i) => s * (repCounts[i] ?? 1));
  const outSize = shapeToSize(outShape);
  const out = allocate(t.dtype, outSize);

  if (outSize > 0) {
    const { data, start } = toDense(t);
    const inStrides = computeStrides(inShape);
    const outStrides = computeStrides(outShape);

    // Fill the first tile of each dimension recursively, then replicate it.
    const fill = (dim: number, srcPos: number, dstPos: number): void => {
      const n = inShape[dim] ?? 1;
      const times = repCounts[dim] ?? 1;
      if (dim === ndim - 1) {
        copyRange(data, srcPos, out, dstPos, n);
        replicate(out, dstPos, n, times);
        return;
      }
      const inStride = inStrides[dim] ?? 1;
      const outStride = outStrides[dim] ?? 1;
      for (let i = 0; i < n; i++) {
        fill(dim + 1, srcPos + i * inStride, dstPos + i * outStride);
      }
      replicate(out, dstPos, n * outStride, times);
    };
    fill(0, start, 0);
  }
  return build(out, outShape, t.dtype, t.device);
}

/**
 * Repeat elements of a tensor along an axis.
 *
 * Each element is repeated consecutively. `repeats` is either one count used
 * for every element, or an array with one count per element along the axis
 * (a one-element array is broadcast). Without `axis` the tensor is flattened
 * first and the result is 1-D.
 *
 * **Complexity**: O(n * repeats) where n is input size
 *
 * @param t - Input tensor
 * @param repeats - Number of times to repeat each element, or one count per element
 * @param axis - Axis along which to repeat (default: flatten first)
 * @returns Tensor with repeated elements
 * @throws {InvalidParameterError} If a count is negative or not an integer, the array length
 *   does not match the axis length, or `axis` is out of range
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3]);
 * repeat(t, 2);           // [1, 1, 2, 2, 3, 3]
 * repeat(t, [0, 2, 1]);   // [2, 2, 3]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function repeat(t: Tensor, repeats: number | readonly number[], axis?: Axis): Tensor {
  assertHostTensors("repeat", [t]);
  const checkCount = (value: number | undefined, label: string): number => {
    if (value === undefined || !Number.isInteger(value) || value < 0) {
      throw new InvalidParameterError(
        `${label} must be a non-negative integer; received ${String(value)}`,
        "repeats",
        repeats
      );
    }
    return value;
  };

  let scalar = 0;
  let list: readonly number[] | null = null;
  if (typeof repeats === "number") {
    scalar = checkCount(repeats, "repeats");
  } else {
    for (let i = 0; i < repeats.length; i++) checkCount(repeats[i], `repeats[${i}]`);
    list = repeats;
  }

  const ax = axis === undefined ? -1 : normalizeAxis(axis, t.ndim);
  const n = ax < 0 ? t.size : (t.shape[ax] ?? 0);

  // Expand to one count per element along the repeated axis (or per element when flattened).
  let counts: readonly number[] | null = null;
  if (list !== null) {
    if (list.length === 1) {
      scalar = list[0] ?? 0;
    } else if (list.length === n) {
      counts = list;
    } else {
      throw new InvalidParameterError(
        `repeats has length ${list.length} but ${ax < 0 ? "the flattened tensor has" : `axis ${ax} has`} ` +
          `${n} elements; pass a single count or one count per element`,
        "repeats",
        repeats
      );
    }
  }
  const countAt = (i: number): number => (counts === null ? scalar : (counts[i] ?? 0));

  // Output position along the repeated axis where element i starts.
  const starts = new Array<number>(n + 1);
  starts[0] = 0;
  for (let i = 0; i < n; i++) starts[i + 1] = (starts[i] ?? 0) + countAt(i);
  const axisTotal = starts[n] ?? 0;

  if (ax < 0) {
    const out = allocate(t.dtype, axisTotal);
    if (axisTotal > 0) {
      const { data, start } = toDense(t);
      for (let i = 0; i < n; i++) {
        fillFrom(data, start + i, out, starts[i] ?? 0, countAt(i));
      }
    }
    return build(out, [axisTotal], t.dtype, t.device);
  }

  const outShape = [...t.shape];
  outShape[ax] = axisTotal;
  const outer = productOf(t.shape, 0, ax);
  const inner = productOf(t.shape, ax + 1, t.ndim);
  const out = allocate(t.dtype, shapeToSize(outShape));

  if (axisTotal > 0 && inner > 0 && outer > 0) {
    const { data, start } = toDense(t);
    for (let o = 0; o < outer; o++) {
      for (let i = 0; i < n; i++) {
        const c = countAt(i);
        if (c === 0) continue;
        const srcPos = start + (o * n + i) * inner;
        const dstPos = (o * axisTotal + (starts[i] ?? 0)) * inner;
        if (inner === 1) {
          fillFrom(data, srcPos, out, dstPos, c);
        } else {
          copyRange(data, srcPos, out, dstPos, inner);
          replicate(out, dstPos, inner, c);
        }
      }
    }
  }
  return build(out, outShape, t.dtype, t.device);
}
