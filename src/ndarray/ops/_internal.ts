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
  type Shape,
  type TypedArray,
} from "../../core";
import type { NumericTypedArray } from "../../core/utils/typed_array_access";
import { roundToBFloat16, roundToFloat16 } from "../tensor/float16";
import { isContiguous, offsetFromFlatIndex } from "../tensor/strides";
import { computeStrides, isBigIntArray, Tensor } from "../tensor/Tensor";

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
 * The returned array may alias the tensor's buffer, so callers must not
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

const MAX_SAFE_BIGINT = BigInt(Number.MAX_SAFE_INTEGER);
const MIN_SAFE_BIGINT = -MAX_SAFE_BIGINT;

/**
 * Convert a BigInt to a Number, throwing if the magnitude exceeds
 * `Number.MAX_SAFE_INTEGER`.
 *
 * @throws {DataValidationError} If the value cannot be represented exactly
 */
export function bigintToNumberSafe(v: bigint): number {
  if (v > MAX_SAFE_BIGINT || v < MIN_SAFE_BIGINT) {
    throw new DataValidationError(
      `int64 value ${v.toString()} is too large to safely convert to number ` +
        "(magnitude exceeds 2^53 - 1)"
    );
  }
  return Number(v);
}

/**
 * Read a tensor's logical elements as numbers in row-major order.
 *
 * Numeric tensors go through {@link readNumericContiguous} (zero-copy when the
 * tensor is contiguous). int64 tensors are converted into a new `Float64Array`;
 * with `exact` (the default) a value outside the safe integer range throws via
 * {@link bigintToNumberSafe}, otherwise it is rounded like `Number(value)`. The
 * result may alias the tensor's buffer, so callers must not write to it.
 *
 * @param t - Source tensor
 * @param opName - Operation name for the error message
 * @param exact - Reject int64 values that a double cannot hold exactly (default: true)
 * @throws {DTypeError} If the tensor has string dtype
 * @throws {DataValidationError} If `exact` is set and an int64 value is outside the safe integer range
 */
export function readNumbers(t: Tensor, opName: string, exact = true): NumericTypedArray {
  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError(`${opName} is not defined for string dtype`);
  }
  if (data instanceof BigInt64Array) {
    const size = t.size;
    const out = new Float64Array(size);
    const logicalStrides = computeStrides(t.shape);
    const contiguous = isContiguous(t.shape, t.strides);
    for (let i = 0; i < size; i++) {
      const off = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const v = getBigIntElement(data, off);
      out[i] = exact ? bigintToNumberSafe(v) : Number(v);
    }
    return out;
  }
  const src = readNumericContiguous(t);
  if (src === null) {
    throw new DTypeError(`${opName} is not defined for string dtype`);
  }
  return src;
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

/**
 * Snap the values of a freshly computed half-precision result onto the float16 / bfloat16
 * grid. Half-precision tensors keep float32 host storage, so an op that computes in float32
 * would otherwise leave values that the dtype cannot represent. Other dtypes, device tensors
 * and strided views are returned unchanged.
 *
 * Call this only on tensors the op just allocated: the buffer is rounded in place.
 */
export function roundHalfResult(t: Tensor): Tensor {
  const dtype = t.dtype;
  if (dtype !== "float16" && dtype !== "bfloat16") return t;
  if (t.isDeviceTensor) return t;
  const data = t.data;
  if (!(data instanceof Float32Array) || !isContiguous(t.shape, t.strides)) return t;
  const round = dtype === "float16" ? roundToFloat16 : roundToBFloat16;
  const end = t.offset + t.size;
  for (let i = t.offset; i < end; i++) data[i] = round(data[i] as number);
  return t;
}

/** The float dtypes a result tensor can have. */
export type FloatDType = "float16" | "bfloat16" | "float32" | "float64";

/** Host storage of a float tensor (half-precision dtypes use `Float32Array`). */
export type FloatArray = Float32Array | Float64Array;

/**
 * Allocate zero-filled host storage for a float dtype: `Float64Array` for `float64`,
 * `Float32Array` for the three narrower float dtypes.
 */
export function allocFloat(dtype: FloatDType, size: number): FloatArray {
  return dtype === "float64" ? new Float64Array(size) : new Float32Array(size);
}

/**
 * Wrap freshly computed float storage in a tensor. Values are computed in float64 and
 * narrowed by the typed-array store; half-precision results are then snapped onto their
 * float16 / bfloat16 grid.
 */
export function floatResult(
  out: FloatArray,
  shape: Shape,
  dtype: FloatDType,
  device: Tensor["device"]
): Tensor {
  return roundHalfResult(Tensor.fromTypedArray({ data: out, shape, dtype, device }));
}
