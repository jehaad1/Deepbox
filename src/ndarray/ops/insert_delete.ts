/**
 * Tensor insert and delete operations.
 *
 * - insert: Insert values into a tensor along an axis
 * - delete_: Remove elements from a tensor along an axis
 *
 * @module ndarray/ops/insert_delete
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */

import type { Axis, DType } from "../../core";
import {
  DTypeError,
  dtypeToTypedArrayCtor,
  InvalidParameterError,
  normalizeAxis,
  ShapeError,
  shapeToSize,
} from "../../core";
import { isContiguous } from "../tensor/strides";
import { computeStrides, Tensor } from "../tensor/Tensor";
import { flatOffset, readNumericContiguous } from "./_internal";

/** Any buffer a tensor can be stored in. */
type Buffer = Float32Array | Float64Array | Int32Array | Uint8Array | BigInt64Array | string[];

/** A run of consecutive positions along the working axis. */
interface Run {
  /** Copy from the inserted-values buffer (true) or from the source buffer (false). */
  readonly fromValues: boolean;
  readonly outStart: number;
  /** First source index (or first value slot when `fromValues`). */
  readonly start: number;
  readonly len: number;
}

/** Row-major logical elements of `t` in a zero-based buffer (may alias the tensor's storage). */
function denseElements(t: Tensor): Buffer {
  const numeric = readNumericContiguous(t);
  if (numeric) return numeric;
  const data = t.data as BigInt64Array | string[];
  const size = t.size;
  if (isContiguous(t.shape, t.strides)) {
    return Array.isArray(data)
      ? data.slice(t.offset, t.offset + size)
      : data.subarray(t.offset, t.offset + size);
  }
  const logicalStrides = computeStrides(t.shape);
  if (Array.isArray(data)) {
    const out = new Array<string>(size);
    for (let i = 0; i < size; i++) {
      out[i] = data[flatOffset(i, t.offset, false, logicalStrides, t.strides)] as string;
    }
    return out;
  }
  const out = new BigInt64Array(size);
  for (let i = 0; i < size; i++) {
    out[i] = data[flatOffset(i, t.offset, false, logicalStrides, t.strides)] as bigint;
  }
  return out;
}

function allocate(dtype: DType, size: number): Buffer {
  if (dtype === "string") return new Array<string>(size).fill("");
  return new (dtypeToTypedArrayCtor(dtype))(size);
}

/** Copy `len` elements; both buffers have the same element type. */
function copyRange(out: Buffer, outPos: number, src: Buffer, srcPos: number, len: number): void {
  if (Array.isArray(out)) {
    const from = src as string[];
    for (let i = 0; i < len; i++) out[outPos + i] = from[srcPos + i] as string;
  } else {
    (out as Float64Array).set((src as Float64Array).subarray(srcPos, srcPos + len), outPos);
  }
}

function wrapResult(data: Buffer, shape: number[], t: Tensor): Tensor {
  if (Array.isArray(data)) {
    return Tensor.fromStringArray({ data, shape, device: t.device });
  }
  return Tensor.fromTypedArray({
    data,
    shape,
    dtype: t.dtype as Exclude<DType, "string">,
    device: t.device,
  });
}

/** Fill `out` for each outer block from the source and (optionally) value buffers by runs. */
function assemble(
  out: Buffer,
  src: Buffer,
  values: Buffer | null,
  runs: readonly Run[],
  outer: number,
  inner: number,
  axisSize: number,
  outAxisSize: number,
  numnew: number
): void {
  for (let o = 0; o < outer; o++) {
    for (const run of runs) {
      const dst = (o * outAxisSize + run.outStart) * inner;
      if (run.fromValues) {
        if (values) {
          copyRange(out, dst, values, (o * numnew + run.start) * inner, run.len * inner);
        }
      } else {
        copyRange(out, dst, src, (o * axisSize + run.start) * inner, run.len * inner);
      }
    }
  }
}

function normalizeIndexList(
  indices: number | readonly number[],
  axisSize: number,
  allowEnd: boolean,
  op: string
): number[] {
  const list = typeof indices === "number" ? [indices] : [...indices];
  const limit = allowEnd ? axisSize : axisSize - 1;
  return list.map((raw) => {
    if (typeof raw !== "number" || !Number.isInteger(raw)) {
      throw new InvalidParameterError(
        `${op} index must be an integer; received ${String(raw)}`,
        "indices",
        indices
      );
    }
    const idx = raw < 0 ? raw + axisSize : raw;
    if (idx < 0 || idx > limit) {
      throw new InvalidParameterError(
        `${op} index ${raw} is out of bounds for axis size ${axisSize}`,
        "indices",
        indices
      );
    }
    return idx;
  });
}

/** Value operand prepared for broadcasting against the insertion slab. */
interface ValueOperand {
  readonly shape: number[];
  readonly strides: number[];
  readonly offset: number;
  readonly tensor: Tensor;
}

/** Left-pad to `ndim` dimensions (stride 0 on padding), as NumPy's `ndmin`. */
function prepareValues(values: Tensor, ndim: number, ax: number, moveFront: boolean): ValueOperand {
  // Extra leading dimensions are allowed when they have size 1 (NumPy broadcasting).
  const extra = Math.max(0, values.ndim - ndim);
  for (let d = 0; d < extra; d++) {
    if (values.shape[d] !== 1) {
      throw new ShapeError(
        `insert: values have ${values.ndim} dimensions but the tensor has ${ndim}; ` +
          "values must broadcast to the insertion slab"
      );
    }
  }
  const pad = Math.max(0, ndim - values.ndim);
  let shape = [...new Array<number>(pad).fill(1), ...values.shape.slice(extra)];
  let strides = [...new Array<number>(pad).fill(0), ...values.strides.slice(extra)];
  if (moveFront && ax !== 0) {
    // A single insertion index takes the values' leading axis as the insertion axis.
    const s0 = shape[0] as number;
    const st0 = strides[0] as number;
    shape = [...shape.slice(1, ax + 1), s0, ...shape.slice(ax + 1)];
    strides = [...strides.slice(1, ax + 1), st0, ...strides.slice(ax + 1)];
  }
  return { shape, strides, offset: values.offset, tensor: values };
}

/**
 * Insert values into a tensor along an axis before given indices.
 *
 * Follows `numpy.insert`. If `axis` is undefined the input is flattened first.
 * Negative indices count from the end. With several indices, the i-th value
 * (along `axis`) is inserted before the i-th index as given, and `values` are
 * broadcast over the other dimensions. With a single index, every value along
 * the insertion axis is inserted there in order, so
 * `insert(tensor([1, 2, 3]), 1, tensor([9, 8]))` gives `[1, 9, 8, 2, 3]`.
 *
 * The result has the dtype of `t`; inserted values are converted to it.
 *
 * **Complexity**: O(n + m) where n is input size and m is the size of the inserted data
 *
 * @param t - Input tensor
 * @param indices - Index or array of indices before which values are inserted
 * @param values - Values to insert: a number, a string (string tensors only), or a Tensor
 * @param axis - Axis along which to insert. If undefined, tensor is flattened.
 * @returns New tensor with values inserted
 * @throws {InvalidParameterError} If an index is not an integer in `[-n, n]` for axis size `n`
 * @throws {ShapeError} If `values` cannot be broadcast to the insertion slab
 * @throws {DTypeError} If string and numeric data are mixed
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
  values: number | string | Tensor,
  axis?: Axis
): Tensor {
  if (axis === undefined) {
    return insert(t.flatten(), indices, values, 0);
  }

  const ax = normalizeAxis(axis, t.ndim);
  const axisSize = t.shape[ax] as number;
  const isString = t.dtype === "string";

  // ---- Validate the value type against the tensor dtype ----
  if (isString) {
    const ok =
      typeof values === "string" || (typeof values !== "number" && values.dtype === "string");
    if (!ok) {
      throw new DTypeError("insert: a string tensor can only receive string values");
    }
  } else if (
    typeof values === "string" ||
    (typeof values !== "number" && values.dtype === "string")
  ) {
    throw new DTypeError(`insert: cannot insert string values into a ${t.dtype} tensor`);
  }

  const idx = normalizeIndexList(indices, axisSize, true, "insert");
  const singleIndex = idx.length === 1;

  const operand =
    typeof values === "object"
      ? prepareValues(values, t.ndim, ax, typeof indices === "number")
      : null;

  // ---- Number of inserted slots along the axis ----
  const numnew = singleIndex && operand ? (operand.shape[ax] as number) : idx.length;
  if (operand && numnew > 0) {
    for (let d = 0; d < t.ndim; d++) {
      const vs = operand.shape[d] as number;
      const target = d === ax ? numnew : (t.shape[d] as number);
      if (vs !== 1 && vs !== target) {
        throw new ShapeError(
          `insert: values of shape [${operand.tensor.shape}] cannot be broadcast to the insertion ` +
            `slab (dimension ${d} has size ${vs}, expected ${target} or 1)`
        );
      }
    }
  }

  // ---- Layout of the output axis ----
  const outAxisSize = axisSize + numnew;
  const outShape = [...t.shape];
  outShape[ax] = outAxisSize;

  // slotAt[p] = value slot written at output axis position p, or -1 for source data.
  const slotAt = new Array<number>(outAxisSize).fill(-1);
  // valueCoord[slot] = coordinate along the values' insertion axis feeding that slot.
  const valueCoord = new Array<number>(numnew).fill(0);
  const vAxisSize = operand ? (operand.shape[ax] as number) : 1;
  if (singleIndex) {
    const at = idx[0] as number;
    for (let s = 0; s < numnew; s++) {
      slotAt[at + s] = s;
      valueCoord[s] = vAxisSize === 1 ? 0 : s;
    }
  } else {
    // Pair each value with its index as given, but write in sorted index order.
    const order = idx.map((at, valuePos) => ({ at, valuePos }));
    order.sort((a, b) => a.at - b.at || a.valuePos - b.valuePos);
    for (let s = 0; s < order.length; s++) {
      const entry = order[s] as { at: number; valuePos: number };
      slotAt[entry.at + s] = s;
      valueCoord[s] = vAxisSize === 1 ? 0 : entry.valuePos;
    }
  }

  const runs: Run[] = [];
  let srcPos = 0;
  for (let p = 0; p < outAxisSize; p++) {
    const slot = slotAt[p] as number;
    const fromValues = slot >= 0;
    const start = fromValues ? slot : srcPos++;
    const last = runs[runs.length - 1];
    if (last && last.fromValues === fromValues && last.start + last.len === start) {
      runs[runs.length - 1] = { ...last, len: last.len + 1 };
    } else {
      runs.push({ fromValues, outStart: p, start, len: 1 });
    }
  }

  const outer = shapeToSize(t.shape.slice(0, ax));
  const inner = shapeToSize(t.shape.slice(ax + 1));
  const outSize = shapeToSize(outShape);
  const out = allocate(t.dtype, outSize);
  if (outSize === 0) return wrapResult(out, outShape, t);

  const valuesBuf =
    numnew > 0 ? buildValues(t, values, operand, ax, outer, inner, numnew, valueCoord) : null;
  assemble(out, denseElements(t), valuesBuf, runs, outer, inner, axisSize, outAxisSize, numnew);
  return wrapResult(out, outShape, t);
}

/**
 * Materialize the inserted data in slot order as a dense `[outer, numnew, inner]`
 * buffer of the target dtype.
 */
function buildValues(
  t: Tensor,
  values: number | string | Tensor,
  operand: ValueOperand | null,
  ax: number,
  outer: number,
  inner: number,
  numnew: number,
  valueCoord: readonly number[]
): Buffer {
  const total = outer * numnew * inner;
  const out = allocate(t.dtype, total);

  const toTarget = (raw: number | bigint | string): number | bigint | string => {
    if (typeof raw === "string") return raw;
    if (t.dtype === "int64") {
      if (typeof raw === "bigint") return raw;
      if (!Number.isFinite(raw)) {
        throw new InvalidParameterError(
          `insert: cannot insert ${raw} into an int64 tensor`,
          "values",
          values
        );
      }
      return BigInt(Math.trunc(raw));
    }
    if (t.dtype === "bool") return raw !== 0 && raw !== 0n ? 1 : 0;
    return typeof raw === "bigint" ? Number(raw) : raw;
  };

  if (operand === null) {
    const fillValue = toTarget(values as number | string);
    const flat = out as unknown as (number | bigint | string)[];
    for (let i = 0; i < total; i++) flat[i] = fillValue;
    return out;
  }

  const src = operand.tensor.data as ArrayLike<number | bigint | string>;
  const vShape = operand.shape;
  const vStrides = operand.strides;
  const stride = (d: number): number => ((vShape[d] as number) === 1 ? 0 : (vStrides[d] as number));

  // Offsets contributed by the dimensions before / after the insertion axis.
  const outerOffsets = new Array<number>(outer).fill(0);
  for (let o = 0; o < outer; o++) {
    let rem = o;
    let off = 0;
    for (let d = ax - 1; d >= 0; d--) {
      const size = t.shape[d] as number;
      off += (rem % size) * stride(d);
      rem = Math.floor(rem / size);
    }
    outerOffsets[o] = off;
  }
  const innerOffsets = new Array<number>(inner).fill(0);
  for (let i = 0; i < inner; i++) {
    let rem = i;
    let off = 0;
    for (let d = t.ndim - 1; d > ax; d--) {
      const size = t.shape[d] as number;
      off += (rem % size) * stride(d);
      rem = Math.floor(rem / size);
    }
    innerOffsets[i] = off;
  }
  const axisStride = stride(ax);

  const dst = out as unknown as (number | bigint | string)[];
  let pos = 0;
  for (let o = 0; o < outer; o++) {
    const oBase = operand.offset + (outerOffsets[o] as number);
    for (let k = 0; k < numnew; k++) {
      const kBase = oBase + (valueCoord[k] as number) * axisStride;
      for (let i = 0; i < inner; i++) {
        dst[pos++] = toTarget(src[kBase + (innerOffsets[i] as number)] as number | bigint | string);
      }
    }
  }
  return out;
}

/**
 * Delete elements from a tensor along an axis.
 *
 * Follows `numpy.delete`. If `axis` is undefined the input is flattened first.
 * Negative indices count from the end and repeated indices are deleted once.
 * Uses trailing underscore to avoid collision with the JS `delete` keyword.
 *
 * **Complexity**: O(n) where n is input size
 *
 * @param t - Input tensor
 * @param indices - Index or array of indices of elements to delete
 * @param axis - Axis along which to delete. If undefined, tensor is flattened.
 * @returns New tensor with specified elements removed
 * @throws {InvalidParameterError} If an index is not an integer in `[-n, n - 1]` for axis size `n`
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
  if (axis === undefined) {
    return delete_(t.flatten(), indices, 0);
  }

  const ax = normalizeAxis(axis, t.ndim);
  const axisSize = t.shape[ax] as number;
  const deleteSet = new Set(normalizeIndexList(indices, axisSize, false, "delete"));

  const newAxisSize = axisSize - deleteSet.size;
  const outShape = [...t.shape];
  outShape[ax] = newAxisSize;

  // Runs of consecutive kept source positions.
  const runs: Run[] = [];
  let outPos = 0;
  for (let s = 0; s < axisSize; s++) {
    if (deleteSet.has(s)) continue;
    const last = runs[runs.length - 1];
    if (last && last.start + last.len === s) {
      runs[runs.length - 1] = { ...last, len: last.len + 1 };
    } else {
      runs.push({ fromValues: false, outStart: outPos, start: s, len: 1 });
    }
    outPos++;
  }

  const outer = shapeToSize(t.shape.slice(0, ax));
  const inner = shapeToSize(t.shape.slice(ax + 1));
  const outSize = shapeToSize(outShape);
  const out = allocate(t.dtype, outSize);
  if (outSize > 0) {
    assemble(out, denseElements(t), null, runs, outer, inner, axisSize, newAxisSize, 0);
  }
  return wrapResult(out, outShape, t);
}
