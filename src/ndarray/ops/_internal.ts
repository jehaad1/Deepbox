/**
 * Shared internal helpers for ndarray/ops.
 *
 * These utilities are used across multiple ops modules (arithmetic, comparison,
 * logical, math, activation, trigonometry, reduction). Centralising them here
 * eliminates the ~30+ copy-pasted copies that previously lived in each file.
 *
 * @internal
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import {
  DataValidationError,
  DTypeError,
  getBigIntElement,
  getNumericElement,
  type TypedArray,
} from "../../core";
import type { NumericTypedArray } from "../../core/utils/typed_array_access";
import { isContiguous, offsetFromFlatIndex } from "../tensor/strides";
import { isBigIntArray, type Tensor } from "../tensor/Tensor";

/**
 * Resolve the physical buffer offset for a logical flat index.
 *
 * For contiguous tensors this is a simple addition; for strided views it
 * falls back to the general `offsetFromFlatIndex` computation.
 */
export function flatOffset(
  flat: number,
  offset: number,
  contiguous: boolean,
  logicalStrides: readonly number[],
  strides: readonly number[]
): number {
  return contiguous ? offset + flat : offsetFromFlatIndex(flat, logicalStrides, strides, offset);
}

/**
 * Materialize a tensor's logical elements as a zero-based numeric typed
 * array for read-only access, or `null` for string/int64 tensors.
 *
 * Contiguous tensors return a `subarray` view (no copy); strided views are
 * gathered in a single stride-walking pass. Hot loops can then index `[i]`
 * directly instead of paying a per-element `flatOffset`/`getNumericElement`
 * call, which V8 cannot keep on its fast path across module boundaries.
 *
 * The returned array may alias the tensor's buffer — callers must not
 * write to it.
 */
export function readNumericContiguous(t: Tensor): NumericTypedArray | null {
  const data = t.data;
  if (Array.isArray(data) || data instanceof BigInt64Array) return null;
  const size = t.size;
  if (size === 0) return data.subarray(0, 0);
  const shape = t.shape;
  const strides = t.strides;
  if (isContiguous(shape, strides)) {
    return t.offset === 0 && data.length === size ? data : data.subarray(t.offset, t.offset + size);
  }
  const Ctor = data.constructor as new (n: number) => Exclude<NumericTypedArray, BigInt64Array>;
  const out = new Ctor(size);
  const ndim = shape.length;
  const inner = shape[ndim - 1] ?? 1;
  const innerStride = strides[ndim - 1] ?? 0;
  const outer = inner === 0 ? 0 : size / inner;
  const coords = new Array<number>(Math.max(0, ndim - 1)).fill(0);
  let base = t.offset;
  let pos = 0;
  for (let b = 0; b < outer; b++) {
    let idx = base;
    for (let i = 0; i < inner; i++) {
      out[pos++] = data[idx] as number;
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
  return out;
}

/**
 * Convert a BigInt to a Number, throwing if the value exceeds
 * `Number.MAX_SAFE_INTEGER`.
 */
export function bigintToNumberSafe(v: bigint): number {
  const max = BigInt(Number.MAX_SAFE_INTEGER);
  const min = -max;
  if (v > max || v < min) {
    throw new DataValidationError("int64 value is too large to safely convert to number");
  }
  return Number(v);
}

/**
 * Read an element from a TypedArray as a `number`, converting BigInt
 * values via `bigintToNumberSafe`.
 */
export function readAsNumberSafe(data: TypedArray, idx: number): number {
  if (isBigIntArray(data)) {
    return bigintToNumberSafe(getBigIntElement(data, idx));
  }
  return getNumericElement(data, idx);
}

/**
 * Read an element from a TypedArray, preserving its original type
 * (`bigint` or `number`).
 */
export function readElement(data: TypedArray, idx: number): bigint | number {
  if (isBigIntArray(data)) {
    return getBigIntElement(data, idx);
  }
  return getNumericElement(data, idx);
}

/**
 * Read an element as `number`, converting BigInt via `Number()` without
 * the safe-integer check.  Use when the caller tolerates precision loss
 * (e.g. comparison ops that only need ordering).
 */
export function readAsNumber(data: TypedArray, idx: number): number {
  if (isBigIntArray(data)) {
    return Number(getBigIntElement(data, idx));
  }
  return getNumericElement(data, idx);
}

/**
 * Assert that tensor data is a numeric TypedArray (not string[]).
 *
 * @param opName - Operation name for the error message
 */
export function requireNumericData(data: TypedArray | string[], opName: string): TypedArray {
  if (Array.isArray(data)) {
    throw new DTypeError(`${opName} is not implemented for string dtype`);
  }
  return data;
}
