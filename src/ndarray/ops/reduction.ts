/**
 * Reductions and scans over tensor axes: sum, mean, prod, std, variance, min,
 * max, median, argmax, argmin, cumsum, cumprod, diff, any and all, plus the
 * NaN-ignoring variants nanvar, nanmedian, nanprod, nanargmin, nanargmax,
 * nancumsum and nanquantile.
 *
 * Output dtypes: float input keeps its float dtype (float16, bfloat16, float32,
 * float64), accumulating in float64 and rounding once at the end. Integer input to
 * an operation with fractional results (mean, variance, std, median, quantiles)
 * computes in float32. Integer-preserving operations (sum, prod, cumsum, cumprod,
 * min, max, diff) keep int32 and int64; uint8 and bool give int32. Operations that
 * return indices (argmax, argmin, nanargmax, nanargmin) return int32.
 *
 * Axis arguments accept a single axis (negative values count from the end), a
 * list of axes, or the aliases `"index"`, `"rows"` and `"columns"`. Reductions
 * over several axes at once are computed jointly, so `variance(t, [0, 1])` is
 * the variance of all elements of a matrix and not a variance of variances.
 *
 * @module ndarray/ops/reduction
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
import type { Axis, DType, Shape } from "../../core";
import {
  DataValidationError,
  DTypeError,
  InvalidParameterError,
  normalizeAxes,
  normalizeAxis,
} from "../../core";
import { toFloatDType } from "../../core/utils/dtype_utils";
import type { NumericTypedArray } from "../../core/utils/typed_array_access";
import { roundToBFloat16, roundToFloat16 } from "../tensor/float16";
import { isContiguous } from "../tensor/strides";
import { computeStrides, Tensor } from "../tensor/Tensor";
import { flatOffset, readNumbers, roundHalfResult } from "./_internal";
import { dispatchReduce } from "./device_dispatch";

const INT64_MIN = -(1n << 63n);
const INT64_MAX = (1n << 63n) - 1n;
const INT32_MIN = -2147483648;
const INT32_MAX = 2147483647;

type NumericDType = Exclude<DType, "string">;

/** Axis argument accepted by the reductions: one axis or a list of axes. */
type AxisArg = Axis | Axis[];

function ensureNumericTensor(t: Tensor, op: string): asserts t is Tensor<Shape, NumericDType> {
  if (t.dtype === "string") {
    throw new DTypeError(`${op}() not supported for string dtype`);
  }
  if (t.dtype === "complex64" || t.dtype === "complex128") {
    throw new DTypeError(`${op}() not supported for complex dtypes`);
  }
}

/** Output dtype of `sum`: int64 stays int64, other integers widen to int32, floats are kept. */
function outDtypeForSum(inDtype: Tensor["dtype"]): NumericDType {
  if (inDtype === "int64") return "int64";
  if (
    inDtype === "float16" ||
    inDtype === "bfloat16" ||
    inDtype === "float32" ||
    inDtype === "float64"
  ) {
    return inDtype;
  }
  if (inDtype === "string") throw new DTypeError("sum is not defined for string dtype");
  return "int32";
}

/** Shape of a scalar result: `[]`, or all ones when dimensions are kept. */
function scalarShape(ndim: number, keepdims: boolean): Shape {
  return keepdims ? new Array<number>(ndim).fill(1) : [];
}

function scalarTensor(
  value: number | bigint,
  dtype: NumericDType,
  ndim: number,
  keepdims: boolean,
  device: Tensor["device"]
): Tensor {
  const shape = scalarShape(ndim, keepdims);
  if (dtype === "int64") {
    const data = new BigInt64Array(1);
    data[0] = BigInt(value);
    return Tensor.fromTypedArray({ data, shape, dtype, device });
  }
  const data = newTyped(dtype, 1);
  data[0] = Number(value);
  return Tensor.fromTypedArray({ data, shape, dtype, device });
}

/** Allocate a zero-filled non-BigInt typed array for `dtype`. */
function newTyped(dtype: NumericDType, size: number): NumericTypedArray {
  switch (dtype) {
    case "float64":
    case "complex128":
      return new Float64Array(size);
    case "int32":
      return new Int32Array(size);
    case "uint8":
    case "bool":
      return new Uint8Array(size);
    case "int64":
      throw new DTypeError("int64 data is not a number array");
    default:
      return new Float32Array(size);
  }
}

// ---------------------------------------------------------------------------
// Axis planning
// ---------------------------------------------------------------------------

/**
 * Physical offsets for walking a reduction.
 *
 * Output element `o` reduces the input elements at `bases[o] + redOffsets[r]`
 * for `r` in `0..redCount-1`. Output elements follow row-major order over the
 * kept dimensions.
 */
interface ReducePlan {
  readonly outShape: number[];
  readonly outSize: number;
  readonly bases: Float64Array;
  readonly redOffsets: Float64Array;
  readonly redCount: number;
}

/** Offsets of every index combination of `dims`, in row-major order. */
function enumerateOffsets(
  dims: readonly number[],
  strides: readonly number[],
  start: number,
  count: number
): Float64Array {
  const out = new Float64Array(count);
  if (count === 0) return out;
  const nd = dims.length;
  if (nd === 0) {
    out[0] = start;
    return out;
  }
  const coords = new Array<number>(nd).fill(0);
  let off = start;
  for (let i = 0; i < count; i++) {
    out[i] = off;
    for (let d = nd - 1; d >= 0; d--) {
      const c = (coords[d] as number) + 1;
      off += strides[d] as number;
      if (c < (dims[d] as number)) {
        coords[d] = c;
        break;
      }
      off -= (strides[d] as number) * (dims[d] as number);
      coords[d] = 0;
    }
  }
  return out;
}

function planReduction(
  shape: readonly number[],
  strides: readonly number[],
  offset: number,
  axes: readonly number[],
  keepdims: boolean
): ReducePlan {
  const reduced = new Array<boolean>(shape.length).fill(false);
  for (const a of axes) reduced[a] = true;
  const keptDims: number[] = [];
  const keptStrides: number[] = [];
  const redDims: number[] = [];
  const redStrides: number[] = [];
  const outShape: number[] = [];
  for (let i = 0; i < shape.length; i++) {
    const dim = shape[i] as number;
    const stride = strides[i] as number;
    if (reduced[i]) {
      redDims.push(dim);
      redStrides.push(stride);
      if (keepdims) outShape.push(1);
    } else {
      keptDims.push(dim);
      keptStrides.push(stride);
      outShape.push(dim);
    }
  }
  const outSize = keptDims.reduce((p, d) => p * d, 1);
  const redCount = redDims.reduce((p, d) => p * d, 1);
  return {
    outShape,
    outSize,
    bases: enumerateOffsets(keptDims, keptStrides, offset, outSize),
    redOffsets: enumerateOffsets(redDims, redStrides, 0, redCount),
    redCount,
  };
}

/**
 * Resolve an axis argument. Returns `null` for a full reduction (no axis, or
 * every axis listed), otherwise the normalized (non-negative, unique) axes in
 * the order given.
 */
function resolveAxes(axis: AxisArg | undefined, ndim: number): number[] | null {
  if (axis === undefined) return null;
  const axes = normalizeAxes(axis, ndim);
  return axes.length === ndim ? null : axes;
}

interface NumberView {
  readonly data: NumericTypedArray;
  readonly strides: readonly number[];
  readonly offset: number;
}

/**
 * Strided read-only view of the tensor's values as numbers. int64 tensors are
 * converted to a contiguous float64 copy (rounding values beyond 2^53 like
 * `Number(v)`).
 */
function numberView(t: Tensor, op: string): NumberView {
  if (t.data instanceof BigInt64Array) {
    return { data: readNumbers(t, op, false), strides: computeStrides(t.shape), offset: 0 };
  }
  if (Array.isArray(t.data)) {
    throw new DTypeError(`${op}() not supported for string dtype`);
  }
  return { data: t.data, strides: t.strides, offset: t.offset };
}

/**
 * Reduction plan together with the buffer it indexes. The plan must use the
 * strides and offset of the returned view, which differ from the tensor's own
 * for int64 inputs (those are copied into a contiguous float64 buffer).
 */
function planWithView(
  t: Tensor,
  axes: readonly number[],
  keepdims: boolean,
  op: string
): { readonly plan: ReducePlan; readonly data: NumericTypedArray } {
  const view = numberView(t, op);
  return {
    plan: planReduction(t.shape, view.strides, view.offset, axes, keepdims),
    data: view.data,
  };
}

// ---------------------------------------------------------------------------
// Pairwise summation
// ---------------------------------------------------------------------------

const PAIRWISE_BLOCK = 128;

/**
 * Pairwise sum of `n` values starting at `lo`, using the same blocking as
 * NumPy (up to 8 values are added left to right, up to 128 use eight
 * independent accumulators, larger ranges are split in two), so results agree
 * with `np.sum` bit for bit. Rounding error grows as O(log n) instead of O(n).
 */
function pairwiseRange(src: ArrayLike<number>, lo: number, n: number): number {
  if (n < 8) {
    let res = -0;
    for (let i = 0; i < n; i++) res += src[lo + i] as number;
    return res;
  }
  if (n <= PAIRWISE_BLOCK) {
    let r0 = src[lo] as number;
    let r1 = src[lo + 1] as number;
    let r2 = src[lo + 2] as number;
    let r3 = src[lo + 3] as number;
    let r4 = src[lo + 4] as number;
    let r5 = src[lo + 5] as number;
    let r6 = src[lo + 6] as number;
    let r7 = src[lo + 7] as number;
    let i = 8;
    for (; i < n - (n % 8); i += 8) {
      const k = lo + i;
      r0 += src[k] as number;
      r1 += src[k + 1] as number;
      r2 += src[k + 2] as number;
      r3 += src[k + 3] as number;
      r4 += src[k + 4] as number;
      r5 += src[k + 5] as number;
      r6 += src[k + 6] as number;
      r7 += src[k + 7] as number;
    }
    let res = r0 + r1 + (r2 + r3) + (r4 + r5 + (r6 + r7));
    for (; i < n; i++) res += src[lo + i] as number;
    return res;
  }
  let n2 = n >>> 1;
  n2 -= n2 % 8;
  return pairwiseRange(src, lo, n2) + pairwiseRange(src, lo + n2, n - n2);
}

/** Sum of `src[lo..hi)`; 0 for an empty range. */
function pairwiseSum(src: ArrayLike<number>, lo: number, hi: number): number {
  return hi <= lo ? 0 : pairwiseRange(src, lo, hi - lo);
}

/** Pairwise sum of `(src[lo + i] - center)^2` for `i` in `0..n-1`, blocked like {@link pairwiseRange}. */
function pairwiseSqRange(src: ArrayLike<number>, lo: number, n: number, center: number): number {
  if (n < 8) {
    let res = -0;
    for (let i = 0; i < n; i++) {
      const d = (src[lo + i] as number) - center;
      res += d * d;
    }
    return res;
  }
  if (n <= PAIRWISE_BLOCK) {
    const sq = (k: number): number => {
      const d = (src[k] as number) - center;
      return d * d;
    };
    let r0 = sq(lo);
    let r1 = sq(lo + 1);
    let r2 = sq(lo + 2);
    let r3 = sq(lo + 3);
    let r4 = sq(lo + 4);
    let r5 = sq(lo + 5);
    let r6 = sq(lo + 6);
    let r7 = sq(lo + 7);
    let i = 8;
    for (; i < n - (n % 8); i += 8) {
      const k = lo + i;
      r0 += sq(k);
      r1 += sq(k + 1);
      r2 += sq(k + 2);
      r3 += sq(k + 3);
      r4 += sq(k + 4);
      r5 += sq(k + 5);
      r6 += sq(k + 6);
      r7 += sq(k + 7);
    }
    let res = r0 + r1 + (r2 + r3) + (r4 + r5 + (r6 + r7));
    for (; i < n; i++) res += sq(lo + i);
    return res;
  }
  let n2 = n >>> 1;
  n2 -= n2 % 8;
  return pairwiseSqRange(src, lo, n2, center) + pairwiseSqRange(src, lo + n2, n - n2, center);
}

/** Sum of squared deviations `(src[i] - center)^2` over `src[lo..hi)`, pairwise. */
function pairwiseSumSq(src: ArrayLike<number>, lo: number, hi: number, center: number): number {
  return hi <= lo ? 0 : pairwiseSqRange(src, lo, hi - lo, center);
}

// ---------------------------------------------------------------------------
// sum / mean / prod
// ---------------------------------------------------------------------------

/**
 * Sum of array elements over the given axes.
 *
 * Output dtype:
 * - `int64` stays `int64`
 * - `float16`, `bfloat16`, `float32` and `float64` keep their dtype
 * - `int32` stays `int32`; `uint8` and `bool` promote to `int32`
 *
 * Integer sums that leave the output range throw instead of wrapping around.
 * The sum of an empty reduction is 0. Floating-point sums use pairwise
 * accumulation in float64 and round once to the output dtype, so the result can
 * differ from a left-to-right sum in the last bits.
 *
 * @param t - Input tensor
 * @param axis - Axis or list of axes to reduce. If omitted, all elements are summed
 * @param keepdims - If true, reduced axes stay as size-1 dimensions
 * @returns Tensor of sums
 * @throws {DTypeError} If `t` has string dtype
 * @throws {DataValidationError} If an int32 or int64 sum overflows
 * @throws {InvalidParameterError} If an axis is out of range or listed twice
 *
 * @example
 * ```ts
 * import { sum, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([[1, 2], [3, 4]]);
 * sum(x);          // 10
 * sum(x, 0);       // [4, 6]
 * sum(x, [0, 1]);  // 10
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function sum(t: Tensor, axis?: AxisArg, keepdims = false): Tensor {
  return roundHalfResult(sumImpl(t, axis, keepdims));
}

function sumImpl(t: Tensor, axis: AxisArg | undefined, keepdims: boolean): Tensor {
  ensureNumericTensor(t, "sum");

  if (t.device !== "cpu") {
    const onDevice = dispatchReduce("sum", t, axis, keepdims);
    if (onDevice) return onDevice;
  }

  const outDtype = outDtypeForSum(t.dtype);
  const axes = resolveAxes(axis, t.ndim);

  if (axes === null) {
    if (t.data instanceof BigInt64Array) {
      const logicalStrides = computeStrides(t.shape);
      const contiguous = isContiguous(t.shape, t.strides);
      let acc = 0n;
      for (let i = 0; i < t.size; i++) {
        acc += t.data[flatOffset(i, t.offset, contiguous, logicalStrides, t.strides)] as bigint;
      }
      if (acc < INT64_MIN || acc > INT64_MAX) {
        throw new DataValidationError("int64 sum overflow");
      }
      return scalarTensor(acc, "int64", t.ndim, keepdims, t.device);
    }
    const src = readNumbers(t, "sum");
    const acc = pairwiseSum(src, 0, src.length);
    if (outDtype === "int32" && (acc > INT32_MAX || acc < INT32_MIN)) {
      throw new DataValidationError("int32 sum overflow");
    }
    return scalarTensor(acc, outDtype, t.ndim, keepdims, t.device);
  }

  const plan = planReduction(t.shape, t.strides, t.offset, axes, keepdims);
  const { bases, redOffsets, redCount, outSize } = plan;

  if (t.data instanceof BigInt64Array) {
    const data = t.data;
    const out = new BigInt64Array(outSize);
    for (let o = 0; o < outSize; o++) {
      const base = bases[o] as number;
      let acc = 0n;
      for (let r = 0; r < redCount; r++) acc += data[base + (redOffsets[r] as number)] as bigint;
      if (acc < INT64_MIN || acc > INT64_MAX) {
        throw new DataValidationError("int64 sum overflow");
      }
      out[o] = acc;
    }
    return Tensor.fromTypedArray({
      data: out,
      shape: plan.outShape,
      dtype: "int64",
      device: t.device,
    });
  }

  const { data } = numberView(t, "sum");
  const out = newTyped(outDtype, outSize);
  const checkInt32 = outDtype === "int32";
  for (let o = 0; o < outSize; o++) {
    const base = bases[o] as number;
    let acc = 0;
    for (let r = 0; r < redCount; r++) acc += data[base + (redOffsets[r] as number)] as number;
    if (checkInt32 && (acc > INT32_MAX || acc < INT32_MIN)) {
      throw new DataValidationError("int32 sum overflow");
    }
    out[o] = acc;
  }
  return Tensor.fromTypedArray({
    data: out,
    shape: plan.outShape,
    dtype: outDtype,
    device: t.device,
  });
}

/**
 * Arithmetic mean over the given axes.
 *
 * Float input keeps its dtype and integer or bool input gives `float32`. Sums
 * are accumulated in float64, so integer inputs cannot overflow, and the result
 * is rounded once. The mean of an empty reduction is NaN.
 *
 * @param t - Input tensor
 * @param axis - Axis or list of axes to reduce. If omitted, the mean of all elements
 * @param keepdims - If true, reduced axes stay as size-1 dimensions
 * @returns Tensor of means; NaN where the reduction is empty
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If an axis is out of range or listed twice
 *
 * @example
 * ```ts
 * import { mean, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([[1, 2], [3, 4]]);
 * mean(x);     // 2.5
 * mean(x, 0);  // [2, 3]
 * mean(x, 1);  // [1.5, 3.5]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function mean(t: Tensor, axis?: AxisArg, keepdims = false): Tensor {
  return roundHalfResult(meanImpl(t, axis, keepdims));
}

function meanImpl(t: Tensor, axis: AxisArg | undefined, keepdims: boolean): Tensor {
  ensureNumericTensor(t, "mean");

  if (t.device !== "cpu") {
    const onDevice = dispatchReduce("mean", t, axis, keepdims);
    if (onDevice) return onDevice;
  }

  const axes = resolveAxes(axis, t.ndim);
  const outDtype = toFloatDType(t.dtype);

  if (axes === null) {
    if (t.size === 0) {
      return scalarTensor(Number.NaN, outDtype, t.ndim, keepdims, t.device);
    }
    const src = readNumbers(t, "mean", false);
    return scalarTensor(
      pairwiseSum(src, 0, src.length) / t.size,
      outDtype,
      t.ndim,
      keepdims,
      t.device
    );
  }

  const { plan, data } = planWithView(t, axes, keepdims, "mean");
  const { bases, redOffsets, redCount, outSize } = plan;
  const out = newTyped(outDtype, outSize);
  if (redCount === 0) {
    out.fill(Number.NaN);
  } else {
    for (let o = 0; o < outSize; o++) {
      const base = bases[o] as number;
      let acc = 0;
      for (let r = 0; r < redCount; r++) acc += data[base + (redOffsets[r] as number)] as number;
      out[o] = acc / redCount;
    }
  }
  return Tensor.fromTypedArray({
    data: out,
    shape: plan.outShape,
    dtype: outDtype,
    device: t.device,
  });
}

/** Round to the nearest representable value of a half-precision dtype. */
function roundForDType(dtype: NumericDType, x: number): number {
  if (dtype === "float16") return roundToFloat16(x);
  if (dtype === "bfloat16") return roundToBFloat16(x);
  return x;
}

/**
 * Product of array elements over the given axes.
 *
 * Output dtype:
 * - `int64` stays `int64`
 * - `int32`, `uint8` and `bool` produce `int32`
 * - `float16`, `bfloat16`, `float32` and `float64` keep their dtype
 *
 * Integer products that leave the output range throw instead of wrapping
 * around. The product of an empty reduction is 1.
 *
 * @param t - Input tensor
 * @param axis - Axis or list of axes to reduce. If omitted, all elements are multiplied
 * @param keepdims - If true, reduced axes stay as size-1 dimensions
 * @returns Tensor of products
 * @throws {DTypeError} If `t` has string dtype
 * @throws {DataValidationError} If an int32 or int64 product overflows
 * @throws {InvalidParameterError} If an axis is out of range or listed twice
 *
 * @example
 * ```ts
 * import { tensor, prod } from 'deepbox/ndarray';
 *
 * prod(tensor([1, 2, 3, 4]));  // 24
 * prod(tensor([[1, 2], [3, 4]]), 0);  // [3, 8]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function prod(t: Tensor, axis?: AxisArg, keepdims = false): Tensor {
  ensureNumericTensor(t, "prod");

  const dtype: NumericDType = t.dtype === "uint8" || t.dtype === "bool" ? "int32" : t.dtype;
  const integer = dtype === "int32";
  const axes = resolveAxes(axis, t.ndim);

  if (axes === null) {
    if (t.data instanceof BigInt64Array) {
      const logicalStrides = computeStrides(t.shape);
      const contiguous = isContiguous(t.shape, t.strides);
      let acc = 1n;
      for (let i = 0; i < t.size; i++) {
        acc *= t.data[flatOffset(i, t.offset, contiguous, logicalStrides, t.strides)] as bigint;
      }
      if (acc < INT64_MIN || acc > INT64_MAX) {
        throw new DataValidationError("int64 prod overflow");
      }
      return scalarTensor(acc, "int64", t.ndim, keepdims, t.device);
    }
    const src = readNumbers(t, "prod");
    let acc = 1;
    for (let i = 0; i < src.length; i++) {
      const v = src[i] as number;
      acc *= v;
      // A zero makes an integer product exactly 0 even if the running value
      // had already overflowed the double range.
      if (integer && v === 0) {
        acc = 0;
        break;
      }
    }
    if (integer && !(acc <= INT32_MAX && acc >= INT32_MIN)) {
      throw new DataValidationError("int32 prod overflow");
    }
    return scalarTensor(roundForDType(dtype, acc), dtype, t.ndim, keepdims, t.device);
  }

  const plan = planReduction(t.shape, t.strides, t.offset, axes, keepdims);
  const { bases, redOffsets, redCount, outSize } = plan;

  if (t.data instanceof BigInt64Array) {
    const data = t.data;
    const out = new BigInt64Array(outSize);
    for (let o = 0; o < outSize; o++) {
      const base = bases[o] as number;
      let acc = 1n;
      for (let r = 0; r < redCount; r++) acc *= data[base + (redOffsets[r] as number)] as bigint;
      if (acc < INT64_MIN || acc > INT64_MAX) {
        throw new DataValidationError("int64 prod overflow");
      }
      out[o] = acc;
    }
    return Tensor.fromTypedArray({
      data: out,
      shape: plan.outShape,
      dtype: "int64",
      device: t.device,
    });
  }

  const { data } = numberView(t, "prod");
  const out = newTyped(dtype, outSize);
  for (let o = 0; o < outSize; o++) {
    const base = bases[o] as number;
    let acc = 1;
    for (let r = 0; r < redCount; r++) {
      const v = data[base + (redOffsets[r] as number)] as number;
      acc *= v;
      if (integer && v === 0) {
        acc = 0;
        break;
      }
    }
    if (integer && !(acc <= INT32_MAX && acc >= INT32_MIN)) {
      throw new DataValidationError("int32 prod overflow");
    }
    out[o] = roundForDType(dtype, acc);
  }
  return Tensor.fromTypedArray({
    data: out,
    shape: plan.outShape,
    dtype,
    device: t.device,
  });
}

// ---------------------------------------------------------------------------
// variance / std
// ---------------------------------------------------------------------------

/**
 * Variance over the given axes.
 *
 * Computes `sum((x - mean(x))^2) / (N - ddof)` with a two-pass algorithm in
 * float64. Float input keeps its dtype and integer or bool input gives
 * `float32`. This throws when the reduction has no elements or when
 * `ddof >= N`, rather than returning NaN or Infinity.
 *
 * @param t - Input tensor
 * @param axis - Axis or list of axes to reduce. If omitted, the variance of all elements
 * @param keepdims - If true, reduced axes stay as size-1 dimensions
 * @param ddof - Delta degrees of freedom; 0 for population variance (default), 1 for the sample variance
 * @returns Tensor of variances
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If `t` is empty, `ddof` is negative or not finite, `ddof >= N`, or an axis is invalid
 *
 * @example
 * ```ts
 * import { tensor, variance } from 'deepbox/ndarray';
 *
 * const t = tensor([1, 2, 3, 4, 5]);
 * variance(t);                       // 2
 * variance(t, undefined, false, 1);  // 2.5
 *
 * const t2 = tensor([[1, 2], [3, 4]]);
 * variance(t2, 0);  // [1, 1]
 * variance(t2, 1);  // [0.25, 0.25]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function variance(t: Tensor, axis?: AxisArg, keepdims = false, ddof = 0): Tensor {
  return roundHalfResult(varianceKernel(t, axis, keepdims, ddof, false));
}

/**
 * Shared kernel of {@link variance} and {@link std}: computes in float64 and, for `std`,
 * takes the square root before the single rounding to the output dtype.
 */
function varianceKernel(
  t: Tensor,
  axis: AxisArg | undefined,
  keepdims: boolean,
  ddof: number,
  takeSqrt: boolean
): Tensor {
  const name = takeSqrt ? "std" : "variance";
  ensureNumericTensor(t, name);
  const outDtype = toFloatDType(t.dtype);
  const finish = takeSqrt ? Math.sqrt : (v: number): number => v;

  if (!Number.isFinite(ddof) || ddof < 0) {
    throw new InvalidParameterError("ddof must be non-negative and finite", "ddof", ddof);
  }

  // Need at least one element
  if (t.size === 0) {
    throw new InvalidParameterError(`${name}() requires at least one element`, "t");
  }

  const axes = resolveAxes(axis, t.ndim);

  if (axes === null) {
    if (t.size <= ddof) {
      throw new InvalidParameterError(
        `ddof=${ddof} >= size=${t.size}, ${name} undefined`,
        "ddof",
        ddof
      );
    }
    const src = readNumbers(t, name, false);
    const n = src.length;
    const meanValue = pairwiseSum(src, 0, n) / n;
    const sumSquaredDev = pairwiseSumSq(src, 0, n, meanValue);
    return scalarTensor(finish(sumSquaredDev / (n - ddof)), outDtype, t.ndim, keepdims, t.device);
  }

  const { plan, data } = planWithView(t, axes, keepdims, name);
  const { bases, redOffsets, redCount, outSize } = plan;
  if (redCount <= ddof) {
    throw new InvalidParameterError(
      `ddof=${ddof} >= axis size=${redCount}, ${name} undefined`,
      "ddof",
      ddof
    );
  }

  const out = newTyped(outDtype, outSize);
  for (let o = 0; o < outSize; o++) {
    const base = bases[o] as number;
    let sum = 0;
    for (let r = 0; r < redCount; r++) sum += data[base + (redOffsets[r] as number)] as number;
    const meanValue = sum / redCount;
    let sumSquaredDev = 0;
    for (let r = 0; r < redCount; r++) {
      const d = (data[base + (redOffsets[r] as number)] as number) - meanValue;
      sumSquaredDev += d * d;
    }
    out[o] = finish(sumSquaredDev / (redCount - ddof));
  }
  return Tensor.fromTypedArray({
    data: out,
    shape: plan.outShape,
    dtype: outDtype,
    device: t.device,
  });
}

/**
 * Standard deviation over the given axes: the square root of
 * {@link variance}.
 *
 * @param t - Input tensor
 * @param axis - Axis or list of axes to reduce. If omitted, the standard deviation of all elements
 * @param keepdims - If true, reduced axes stay as size-1 dimensions
 * @param ddof - Delta degrees of freedom; 0 for population (default), 1 for sample standard deviation
 * @returns Tensor of standard deviations (float dtypes are kept, integer and bool input gives `float32`)
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If `t` is empty, `ddof` is negative or not finite, `ddof >= N`, or an axis is invalid
 *
 * @example
 * ```ts
 * import { tensor, std } from 'deepbox/ndarray';
 *
 * const t = tensor([1, 2, 3, 4, 5]);
 * std(t);                       // 1.414...
 * std(t, undefined, false, 1);  // 1.581...
 *
 * const t2 = tensor([[1, 2], [3, 4]]);
 * std(t2, 0);  // [1, 1]
 * std(t2, 1);  // [0.5, 0.5]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function std(t: Tensor, axis?: AxisArg, keepdims = false, ddof = 0): Tensor {
  return roundHalfResult(varianceKernel(t, axis, keepdims, ddof, true));
}

// ---------------------------------------------------------------------------
// min / max
// ---------------------------------------------------------------------------

/**
 * Shared implementation of `min` and `max`. NaN propagates (NumPy semantics)
 * and `Math.min`/`Math.max` also order -0 before +0.
 */
function extremum(
  t: Tensor,
  axis: AxisArg | undefined,
  keepdims: boolean,
  isMin: boolean,
  name: "min" | "max"
): Tensor {
  const dtype = t.dtype as NumericDType;
  const axes = resolveAxes(axis, t.ndim);

  if (axes === null) {
    if (t.size === 0) {
      throw new InvalidParameterError(`${name}() requires at least one element`, "t");
    }
    if (t.data instanceof BigInt64Array) {
      const logicalStrides = computeStrides(t.shape);
      const contiguous = isContiguous(t.shape, t.strides);
      const data = t.data;
      let best = data[flatOffset(0, t.offset, contiguous, logicalStrides, t.strides)] as bigint;
      for (let i = 1; i < t.size; i++) {
        const v = data[flatOffset(i, t.offset, contiguous, logicalStrides, t.strides)] as bigint;
        if (isMin ? v < best : v > best) best = v;
      }
      return scalarTensor(best, "int64", t.ndim, keepdims, t.device);
    }
    const src = readNumbers(t, name);
    const n = src.length;
    // Two independent lanes keep the scalar min/max off a single dependency chain.
    let m0 = src[0] as number;
    let m1 = n > 1 ? (src[1] as number) : m0;
    let i = 2;
    if (isMin) {
      for (; i + 2 <= n; i += 2) {
        m0 = Math.min(m0, src[i] as number);
        m1 = Math.min(m1, src[i + 1] as number);
      }
      for (; i < n; i++) m0 = Math.min(m0, src[i] as number);
    } else {
      for (; i + 2 <= n; i += 2) {
        m0 = Math.max(m0, src[i] as number);
        m1 = Math.max(m1, src[i + 1] as number);
      }
      for (; i < n; i++) m0 = Math.max(m0, src[i] as number);
    }
    return scalarTensor(
      isMin ? Math.min(m0, m1) : Math.max(m0, m1),
      dtype,
      t.ndim,
      keepdims,
      t.device
    );
  }

  const plan = planReduction(t.shape, t.strides, t.offset, axes, keepdims);
  const { bases, redOffsets, redCount, outSize } = plan;
  if (redCount === 0) {
    throw new InvalidParameterError(`${name}() requires at least one element`, "t");
  }

  if (t.data instanceof BigInt64Array) {
    const data = t.data;
    const out = new BigInt64Array(outSize);
    for (let o = 0; o < outSize; o++) {
      const base = bases[o] as number;
      let best = data[base + (redOffsets[0] as number)] as bigint;
      for (let r = 1; r < redCount; r++) {
        const v = data[base + (redOffsets[r] as number)] as bigint;
        if (isMin ? v < best : v > best) best = v;
      }
      out[o] = best;
    }
    return Tensor.fromTypedArray({
      data: out,
      shape: plan.outShape,
      dtype: "int64",
      device: t.device,
    });
  }

  const { data } = numberView(t, name);
  const out = newTyped(dtype, outSize);
  for (let o = 0; o < outSize; o++) {
    const base = bases[o] as number;
    let best = data[base + (redOffsets[0] as number)] as number;
    if (isMin) {
      for (let r = 1; r < redCount; r++) {
        best = Math.min(best, data[base + (redOffsets[r] as number)] as number);
      }
    } else {
      for (let r = 1; r < redCount; r++) {
        best = Math.max(best, data[base + (redOffsets[r] as number)] as number);
      }
    }
    out[o] = best;
  }
  return Tensor.fromTypedArray({
    data: out,
    shape: plan.outShape,
    dtype,
    device: t.device,
  });
}

/**
 * Minimum value over the given axes.
 *
 * NaN propagates: the minimum of any slice containing NaN is NaN. The output
 * keeps the input dtype.
 *
 * @param t - Input tensor
 * @param axis - Axis or list of axes to reduce. If omitted, the minimum of all elements
 * @param keepdims - If true, reduced axes stay as size-1 dimensions
 * @returns Tensor of minimum values
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If the reduction has no elements, or an axis is invalid
 *
 * @example
 * ```ts
 * import { tensor, min } from 'deepbox/ndarray';
 *
 * min(tensor([3, 1, 4, 1, 5]));  // 1
 * min(tensor([[1, 5], [3, 2]]), 0);  // [1, 2]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function min(t: Tensor, axis?: AxisArg, keepdims = false): Tensor {
  ensureNumericTensor(t, "min");

  if (t.device !== "cpu") {
    const onDevice = dispatchReduce("min", t, axis, keepdims);
    if (onDevice) return onDevice;
  }
  return extremum(t, axis, keepdims, true, "min");
}

/**
 * Maximum value over the given axes.
 *
 * NaN propagates: the maximum of any slice containing NaN is NaN. The output
 * keeps the input dtype.
 *
 * @param t - Input tensor
 * @param axis - Axis or list of axes to reduce. If omitted, the maximum of all elements
 * @param keepdims - If true, reduced axes stay as size-1 dimensions
 * @returns Tensor of maximum values
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If the reduction has no elements, or an axis is invalid
 *
 * @example
 * ```ts
 * import { tensor, max } from 'deepbox/ndarray';
 *
 * max(tensor([3, 1, 4, 1, 5]));  // 5
 * max(tensor([[1, 5], [3, 2]]), 1);  // [5, 3]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function max(t: Tensor, axis?: AxisArg, keepdims = false): Tensor {
  ensureNumericTensor(t, "max");

  if (t.device !== "cpu") {
    const onDevice = dispatchReduce("max", t, axis, keepdims);
    if (onDevice) return onDevice;
  }
  return extremum(t, axis, keepdims, false, "max");
}

// ---------------------------------------------------------------------------
// median
// ---------------------------------------------------------------------------

/**
 * Find the k-th smallest element (0-based) with an in-place quickselect that
 * uses a three-way partition, so runs of equal values cost O(n) rather than
 * O(n^2). Afterwards every element left of `k` is <= the returned value.
 * The array must not contain NaN.
 */
function quickSelect(arr: Float64Array, k: number): number {
  let lo = 0;
  let hi = arr.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    const a = arr[lo] as number;
    const b = arr[mid] as number;
    const c = arr[hi] as number;
    // Median-of-three pivot
    let pivot: number;
    if ((a <= b && b <= c) || (c <= b && b <= a)) pivot = b;
    else if ((b <= a && a <= c) || (c <= a && a <= b)) pivot = a;
    else pivot = c;

    // Partition into [< pivot][== pivot][> pivot]
    let lt = lo;
    let i = lo;
    let gt = hi;
    while (i <= gt) {
      const v = arr[i] as number;
      if (v < pivot) {
        arr[i] = arr[lt] as number;
        arr[lt] = v;
        lt++;
        i++;
      } else if (v > pivot) {
        arr[i] = arr[gt] as number;
        arr[gt] = v;
        gt--;
      } else {
        i++;
      }
    }

    if (k < lt) hi = lt - 1;
    else if (k > gt) lo = gt + 1;
    else return pivot;
  }
  return arr[lo] as number;
}

/**
 * Median of a scratch buffer, which is reordered in place. NaN propagates
 * (NumPy semantics); quickselect comparisons with NaN would otherwise
 * partition arbitrarily and return a finite value.
 */
function medianOf(arr: Float64Array): number {
  const n = arr.length;
  for (let i = 0; i < n; i++) {
    if (Number.isNaN(arr[i] as number)) return Number.NaN;
  }
  const mid = n >>> 1;
  const upper = quickSelect(arr, mid);
  if (n % 2 === 1) return upper;
  // After selecting index `mid`, everything left of it is <= upper, so the
  // other middle value is the maximum of arr[0..mid-1].
  let lower = arr[0] as number;
  for (let i = 1; i < mid; i++) {
    if ((arr[i] as number) > lower) lower = arr[i] as number;
  }
  const s = lower + upper;
  if (Number.isFinite(s) || !Number.isFinite(lower) || !Number.isFinite(upper)) return s / 2;
  // Both middle values are finite but their sum overflowed.
  return lower / 2 + upper / 2;
}

/**
 * Median over the given axes.
 *
 * For an even number of elements this is the mean of the two middle values.
 * NaN propagates: the median of any slice containing NaN is NaN. Float input
 * keeps its dtype and integer or bool input gives `float32`. Uses quickselect, so
 * it takes O(n) average time per slice.
 *
 * @param t - Input tensor
 * @param axis - Axis or list of axes to reduce. If omitted, the median of all elements
 * @param keepdims - If true, reduced axes stay as size-1 dimensions
 * @returns Tensor of medians
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If the reduction has no elements, or an axis is invalid
 *
 * @example
 * ```ts
 * import { tensor, median } from 'deepbox/ndarray';
 *
 * median(tensor([1, 3, 5, 7, 9]));  // 5
 * median(tensor([1, 2, 3, 4]));     // 2.5
 *
 * const t3 = tensor([[1, 3], [2, 4]]);
 * median(t3, 0);  // [1.5, 3.5]
 * median(t3, 1);  // [2, 3]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function median(t: Tensor, axis?: AxisArg, keepdims = false): Tensor {
  return roundHalfResult(medianImpl(t, axis, keepdims));
}

function medianImpl(t: Tensor, axis: AxisArg | undefined, keepdims: boolean): Tensor {
  ensureNumericTensor(t, "median");
  const outDtype = toFloatDType(t.dtype);

  // Need at least one element
  if (t.size === 0) {
    throw new InvalidParameterError("median() requires at least one element", "t");
  }

  const axes = resolveAxes(axis, t.ndim);

  if (axes === null) {
    // readNumbers may alias the tensor's buffer, so copy before reordering.
    const scratch = Float64Array.from(readNumbers(t, "median", false));
    return scalarTensor(medianOf(scratch), outDtype, t.ndim, keepdims, t.device);
  }

  const { plan, data } = planWithView(t, axes, keepdims, "median");
  const { bases, redOffsets, redCount, outSize } = plan;
  const out = newTyped(outDtype, outSize);
  const scratch = new Float64Array(redCount);
  for (let o = 0; o < outSize; o++) {
    const base = bases[o] as number;
    for (let r = 0; r < redCount; r++)
      scratch[r] = data[base + (redOffsets[r] as number)] as number;
    out[o] = medianOf(scratch);
  }
  return Tensor.fromTypedArray({
    data: out,
    shape: plan.outShape,
    dtype: outDtype,
    device: t.device,
  });
}

// ---------------------------------------------------------------------------
// cumsum / cumprod
// ---------------------------------------------------------------------------

/**
 * Shared implementation of `cumsum` and `cumprod`.
 *
 * With no axis the input is flattened in row-major order. int64 stays int64
 * and int32, uint8 and bool give int32; both throw on overflow. Float dtypes are
 * kept: the scan accumulates in float64 and every element is rounded once.
 */
function cumulative(t: Tensor, axis: Axis | undefined, isSum: boolean, name: string): Tensor {
  return roundHalfResult(cumulativeImpl(t, axis, isSum, name));
}

/** Result dtype of the cumulative scans: floats and int64 are kept, other integers give int32. */
function outDtypeForCumulative(dtype: Tensor["dtype"]): NumericDType {
  return dtype === "int64" || toFloatDType(dtype) === dtype ? (dtype as NumericDType) : "int32";
}

function cumulativeImpl(t: Tensor, axis: Axis | undefined, isSum: boolean, name: string): Tensor {
  ensureNumericTensor(t, name);

  // Layout of the scan: `bases[j]` is the physical offset of the first element of
  // scan line `j`, lines are `n` long and spaced `axisStride` apart. Output is
  // contiguous in `outShape`; line `j` starts at `outBase(j)` with step `inner`.
  let outShape: number[];
  let n: number;
  let inner: number;
  let axisStride: number;
  let bases: ArrayLike<number>;
  let source: NumericTypedArray | BigInt64Array;

  if (axis === undefined) {
    outShape = [t.size];
    n = t.size;
    inner = 1;
    axisStride = 1;
    bases = [0];
    if (t.data instanceof BigInt64Array) {
      const flat = new BigInt64Array(t.size);
      const logicalStrides = computeStrides(t.shape);
      const contiguous = isContiguous(t.shape, t.strides);
      for (let i = 0; i < t.size; i++) {
        flat[i] = t.data[flatOffset(i, t.offset, contiguous, logicalStrides, t.strides)] as bigint;
      }
      source = flat;
    } else {
      source = readNumbers(t, name);
    }
  } else {
    const ax = normalizeAxis(axis, t.ndim);
    outShape = [...t.shape];
    n = t.shape[ax] as number;
    inner = 1;
    for (let i = ax + 1; i < t.ndim; i++) inner *= t.shape[i] as number;
    axisStride = t.strides[ax] as number;
    bases = planReduction(t.shape, t.strides, t.offset, [ax], false).bases;
    if (Array.isArray(t.data)) {
      throw new DTypeError(`${name}() not supported for string dtype`);
    }
    source = t.data;
  }

  const lines = bases.length;
  const outSize = t.size;

  if (source instanceof BigInt64Array) {
    const data = source;
    const out = new BigInt64Array(outSize);
    const label = isSum ? "int64 cumsum overflow" : "int64 cumprod overflow";
    for (let j = 0; j < lines; j++) {
      const outer = Math.floor(j / inner);
      const outBase = outer * n * inner + (j - outer * inner);
      const base = bases[j] as number;
      let acc = isSum ? 0n : 1n;
      for (let k = 0; k < n; k++) {
        const v = data[base + k * axisStride] as bigint;
        acc = isSum ? acc + v : acc * v;
        if (acc < INT64_MIN || acc > INT64_MAX) throw new DataValidationError(label);
        out[outBase + k * inner] = acc;
      }
    }
    return Tensor.fromTypedArray({ data: out, shape: outShape, dtype: "int64", device: t.device });
  }

  const data = source;
  const outDtype = outDtypeForCumulative(t.dtype);
  const out = newTyped(outDtype, outSize);
  const checkInt32 = outDtype === "int32";
  for (let j = 0; j < lines; j++) {
    const outer = Math.floor(j / inner);
    const outBase = outer * n * inner + (j - outer * inner);
    const base = bases[j] as number;
    let acc = isSum ? 0 : 1;
    for (let k = 0; k < n; k++) {
      const v = data[base + k * axisStride] as number;
      acc = isSum ? acc + v : acc * v;
      if (checkInt32 && !(acc <= INT32_MAX && acc >= INT32_MIN)) {
        throw new DataValidationError(`int32 ${name} overflow`);
      }
      out[outBase + k * inner] = acc;
    }
  }
  return Tensor.fromTypedArray({ data: out, shape: outShape, dtype: outDtype, device: t.device });
}

/**
 * Cumulative sum along an axis.
 *
 * Each output element is the sum of all elements up to and including it along
 * the axis. If `axis` is omitted the input is flattened first, so the result
 * is 1-D (NumPy semantics). int64 stays int64 and int32, uint8 and bool give
 * int32, both throwing on overflow; float dtypes are kept (accumulated in
 * float64, rounded once per element).
 *
 * @param t - Input tensor
 * @param axis - Axis to accumulate along; omit to accumulate over the flattened tensor
 * @returns Tensor of cumulative sums; same shape as `t` when `axis` is given, shape `[t.size]` otherwise
 * @throws {DTypeError} If `t` has string dtype
 * @throws {DataValidationError} If an integer sum overflows
 * @throws {InvalidParameterError} If `axis` is out of range
 *
 * @example
 * ```ts
 * import { tensor, cumsum } from 'deepbox/ndarray';
 *
 * cumsum(tensor([1, 2, 3, 4]));  // [1, 3, 6, 10]
 * cumsum(tensor([[1, 2], [3, 4]]), 0);  // [[1, 2], [4, 6]]
 * cumsum(tensor([[1, 2], [3, 4]]));     // [1, 3, 6, 10]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function cumsum(t: Tensor, axis?: Axis): Tensor {
  return cumulative(t, axis, true, "cumsum");
}

/**
 * Cumulative product along an axis.
 *
 * Each output element is the product of all elements up to and including it
 * along the axis. If `axis` is omitted the input is flattened first, so the
 * result is 1-D (NumPy semantics). int64 stays int64 and int32, uint8 and bool
 * give int32, both throwing on overflow; float dtypes are kept (accumulated in
 * float64, rounded once per element).
 *
 * @param t - Input tensor
 * @param axis - Axis to accumulate along; omit to accumulate over the flattened tensor
 * @returns Tensor of cumulative products; same shape as `t` when `axis` is given, shape `[t.size]` otherwise
 * @throws {DTypeError} If `t` has string dtype
 * @throws {DataValidationError} If an integer product overflows
 * @throws {InvalidParameterError} If `axis` is out of range
 *
 * @example
 * ```ts
 * import { tensor, cumprod } from 'deepbox/ndarray';
 *
 * cumprod(tensor([1, 2, 3, 4]));  // [1, 2, 6, 24]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function cumprod(t: Tensor, axis?: Axis): Tensor {
  return cumulative(t, axis, false, "cumprod");
}

// ---------------------------------------------------------------------------
// diff
// ---------------------------------------------------------------------------

/** One first-order difference along `ax`; the axis shrinks by one (never below 0). */
function diffOnce(input: Tensor, ax: number): Tensor {
  return roundHalfResult(diffOnceImpl(input, ax));
}

function diffOnceImpl(input: Tensor, ax: number): Tensor {
  const outDtype = outDtypeForCumulative(input.dtype);
  const len = input.shape[ax] as number;
  const outShape = [...input.shape];
  outShape[ax] = Math.max(len - 1, 0);
  const outLen = outShape[ax] as number;
  const outSize = outShape.reduce((p, d) => p * d, 1);
  const isBig = input.data instanceof BigInt64Array;

  if (outSize === 0) {
    return isBig
      ? Tensor.fromTypedArray({
          data: new BigInt64Array(0),
          shape: outShape,
          dtype: "int64",
          device: input.device,
        })
      : Tensor.fromTypedArray({
          data: newTyped(outDtype, 0),
          shape: outShape,
          dtype: outDtype,
          device: input.device,
        });
  }

  const plan = planReduction(input.shape, input.strides, input.offset, [ax], false);
  const bases = plan.bases;
  let inner = 1;
  for (let i = ax + 1; i < input.ndim; i++) inner *= input.shape[i] as number;
  const axisStride = input.strides[ax] as number;

  if (input.data instanceof BigInt64Array) {
    const data = input.data;
    const out = new BigInt64Array(outSize);
    for (let j = 0; j < bases.length; j++) {
      const outer = Math.floor(j / inner);
      const outBase = outer * outLen * inner + (j - outer * inner);
      const base = bases[j] as number;
      let prev = data[base] as bigint;
      for (let k = 0; k < outLen; k++) {
        const cur = data[base + (k + 1) * axisStride] as bigint;
        const d = cur - prev;
        if (d < INT64_MIN || d > INT64_MAX) throw new DataValidationError("int64 diff overflow");
        out[outBase + k * inner] = d;
        prev = cur;
      }
    }
    return Tensor.fromTypedArray({
      data: out,
      shape: outShape,
      dtype: "int64",
      device: input.device,
    });
  }

  const { data } = numberView(input, "diff");
  const out = newTyped(outDtype, outSize);
  const checkInt32 = outDtype === "int32";
  for (let j = 0; j < bases.length; j++) {
    const outer = Math.floor(j / inner);
    const outBase = outer * outLen * inner + (j - outer * inner);
    const base = bases[j] as number;
    let prev = data[base] as number;
    for (let k = 0; k < outLen; k++) {
      const cur = data[base + (k + 1) * axisStride] as number;
      const d = cur - prev;
      if (checkInt32 && (d > INT32_MAX || d < INT32_MIN)) {
        throw new DataValidationError("int32 diff overflow");
      }
      out[outBase + k * inner] = d;
      prev = cur;
    }
  }
  return Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: outDtype,
    device: input.device,
  });
}

/**
 * n-th discrete difference along an axis.
 *
 * The first difference is `out[i] = a[i + 1] - a[i]` along `axis`; higher
 * orders apply it repeatedly. The axis shrinks by `n` (down to length 0).
 * int64 stays int64 and int32, uint8 and bool give int32, both throwing on
 * overflow; float dtypes are kept.
 *
 * @param t - Input tensor
 * @param n - Number of times to difference; a non-negative integer (default 1)
 * @param axis - Axis to difference along (default: last axis)
 * @returns Tensor of differences; with `n = 0` the input itself
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If `n` is not a non-negative integer or `axis` is out of range
 *
 * @example
 * ```ts
 * import { tensor, diff } from 'deepbox/ndarray';
 *
 * diff(tensor([1, 3, 6, 10]));     // [2, 3, 4]
 * diff(tensor([1, 3, 6, 10]), 2);  // [1, 1]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function diff(t: Tensor, n = 1, axis: Axis = -1): Tensor {
  ensureNumericTensor(t, "diff");

  if (!Number.isInteger(n) || n < 0) {
    throw new InvalidParameterError("n must be a non-negative integer", "n", n);
  }

  const ax = normalizeAxis(axis, t.ndim);

  if (n === 0) {
    return t;
  }

  let result: Tensor = t;
  for (let i = 0; i < n; i++) {
    result = diffOnce(result, ax);
    // Once the axis is empty every further difference is empty too.
    if ((result.shape[ax] as number) === 0) break;
  }
  return result;
}

// ---------------------------------------------------------------------------
// any / all
// ---------------------------------------------------------------------------

/**
 * Shared implementation of `any` and `all`. A value is truthy when it is not
 * zero, so NaN counts as true (as in NumPy).
 */
function truthReduction(
  t: Tensor,
  axis: AxisArg | undefined,
  keepdims: boolean,
  isAny: boolean,
  name: "any" | "all"
): Tensor {
  ensureNumericTensor(t, name);
  const axes = resolveAxes(axis, t.ndim);
  // `any` looks for a truthy value, `all` for a falsy one; both stop at the first hit.
  const stopWhenZero = !isAny;

  if (axes === null) {
    const src = readNumbers(t, name, false);
    let result = !isAny;
    for (let i = 0; i < src.length; i++) {
      const isZero = src[i] === 0;
      if (isZero === stopWhenZero) {
        result = isAny;
        break;
      }
    }
    return scalarTensor(result ? 1 : 0, "bool", t.ndim, keepdims, t.device);
  }

  const { plan, data } = planWithView(t, axes, keepdims, name);
  const { bases, redOffsets, redCount, outSize } = plan;
  const out = new Uint8Array(outSize);
  for (let o = 0; o < outSize; o++) {
    const base = bases[o] as number;
    let result = !isAny;
    for (let r = 0; r < redCount; r++) {
      const isZero = data[base + (redOffsets[r] as number)] === 0;
      if (isZero === stopWhenZero) {
        result = isAny;
        break;
      }
    }
    out[o] = result ? 1 : 0;
  }
  return Tensor.fromTypedArray({
    data: out,
    shape: plan.outShape,
    dtype: "bool",
    device: t.device,
  });
}

/**
 * Test whether any element is true (non-zero) over the given axes.
 *
 * NaN counts as true. The result is a bool tensor, and `any` of an empty
 * reduction is false.
 *
 * @param t - Input tensor
 * @param axis - Axis or list of axes to reduce. If omitted, tests all elements
 * @param keepdims - If true, reduced axes stay as size-1 dimensions
 * @returns Bool tensor
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If an axis is out of range or listed twice
 *
 * @example
 * ```ts
 * any(tensor([0, 0, 1, 0]));  // true
 * any(tensor([0, 0, 0]));     // false
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function any(t: Tensor, axis?: AxisArg, keepdims = false): Tensor {
  return truthReduction(t, axis, keepdims, true, "any");
}

/**
 * Test whether all elements are true (non-zero) over the given axes.
 *
 * NaN counts as true. The result is a bool tensor, and `all` of an empty
 * reduction is true.
 *
 * @param t - Input tensor
 * @param axis - Axis or list of axes to reduce. If omitted, tests all elements
 * @param keepdims - If true, reduced axes stay as size-1 dimensions
 * @returns Bool tensor
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If an axis is out of range or listed twice
 *
 * @example
 * ```ts
 * all(tensor([1, 2, 3]));  // true
 * all(tensor([1, 0, 3]));  // false
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function all(t: Tensor, axis?: AxisArg, keepdims = false): Tensor {
  return truthReduction(t, axis, keepdims, false, "all");
}

// ---------------------------------------------------------------------------
// argmax / argmin
// ---------------------------------------------------------------------------

/**
 * Index (relative to `start`, in steps of `stride`) of the first maximum or minimum of `count`
 * numbers. With `ignoreNaN` false the first NaN wins immediately (NumPy semantics); with
 * `ignoreNaN` true NaN is skipped and -1 is returned for an all-NaN slice.
 */
function bestIndexNumbers(
  src: ArrayLike<number>,
  start: number,
  stride: number,
  count: number,
  isMax: boolean,
  ignoreNaN: boolean
): number {
  let best = -1;
  let bestVal = 0;
  let pos = start;
  for (let k = 0; k < count; k++, pos += stride) {
    const v = src[pos] as number;
    if (Number.isNaN(v)) {
      if (ignoreNaN) continue;
      return k;
    }
    if (best === -1 || (isMax ? v > bestVal : v < bestVal)) {
      best = k;
      bestVal = v;
    }
  }
  return best;
}

/** First index of the maximum or minimum of `count` int64 values (exact comparison). */
function bestIndexBigInts(
  src: BigInt64Array,
  start: number,
  stride: number,
  count: number,
  isMax: boolean
): number {
  let best = 0;
  let bestVal = src[start] as bigint;
  let pos = start + stride;
  for (let k = 1; k < count; k++, pos += stride) {
    const v = src[pos] as bigint;
    if (isMax ? v > bestVal : v < bestVal) {
      best = k;
      bestVal = v;
    }
  }
  return best;
}

/** Shared implementation of argmax, argmin, nanargmax and nanargmin. */
function argExtremum(
  t: Tensor,
  axis: Axis | undefined,
  keepdims: boolean,
  isMax: boolean,
  ignoreNaN: boolean,
  name: string
): Tensor {
  ensureNumericTensor(t, name);
  const emptyMessage = `${name}() requires at least one element`;
  const allNaN = (): never => {
    throw new DataValidationError(`${name}(): all-NaN slice encountered`);
  };

  if (axis === undefined) {
    if (t.size === 0) throw new InvalidParameterError(emptyMessage, "t");
    let index: number;
    if (t.data instanceof BigInt64Array) {
      const logicalStrides = computeStrides(t.shape);
      const contiguous = isContiguous(t.shape, t.strides);
      const data = t.data;
      let bestVal = data[flatOffset(0, t.offset, contiguous, logicalStrides, t.strides)] as bigint;
      index = 0;
      for (let i = 1; i < t.size; i++) {
        const v = data[flatOffset(i, t.offset, contiguous, logicalStrides, t.strides)] as bigint;
        if (isMax ? v > bestVal : v < bestVal) {
          bestVal = v;
          index = i;
        }
      }
    } else {
      const src = readNumbers(t, name);
      index = bestIndexNumbers(src, 0, 1, src.length, isMax, ignoreNaN);
      if (index === -1) allNaN();
    }
    return scalarTensor(index, "int32", t.ndim, keepdims, t.device);
  }

  const ax = normalizeAxis(axis, t.ndim);
  if (t.data instanceof BigInt64Array) {
    const plan = planReduction(t.shape, t.strides, t.offset, [ax], keepdims);
    if (plan.redCount === 0) throw new InvalidParameterError(emptyMessage, "t");
    const out = new Int32Array(plan.outSize);
    const stride = t.strides[ax] as number;
    for (let o = 0; o < plan.outSize; o++) {
      out[o] = bestIndexBigInts(t.data, plan.bases[o] as number, stride, plan.redCount, isMax);
    }
    return Tensor.fromTypedArray({
      data: out,
      shape: plan.outShape,
      dtype: "int32",
      device: t.device,
    });
  }

  const view = numberView(t, name);
  const plan = planReduction(t.shape, view.strides, view.offset, [ax], keepdims);
  if (plan.redCount === 0) throw new InvalidParameterError(emptyMessage, "t");
  const out = new Int32Array(plan.outSize);
  const stride = view.strides[ax] as number;
  for (let o = 0; o < plan.outSize; o++) {
    const index = bestIndexNumbers(
      view.data,
      plan.bases[o] as number,
      stride,
      plan.redCount,
      isMax,
      ignoreNaN
    );
    if (index === -1) allNaN();
    out[o] = index;
  }
  return Tensor.fromTypedArray({
    data: out,
    shape: plan.outShape,
    dtype: "int32",
    device: t.device,
  });
}

/**
 * Index of the maximum value along an axis (NumPy `argmax`).
 *
 * The first occurrence wins on ties. NaN propagates: the result is the index of the first NaN
 * in the slice. Without `axis` the tensor is flattened and the index refers to the flattened
 * (row-major) order. The result is always `int32`. int64 values are compared exactly.
 *
 * @param t - Input tensor
 * @param axis - Axis to search along; omit to search the flattened tensor
 * @param keepdims - If true, the reduced axis stays as a size-1 dimension (all axes when `axis`
 *   is omitted)
 * @returns `int32` tensor of indices
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If the searched slice is empty or `axis` is out of range
 *
 * @example
 * ```ts
 * import { argmax, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([[1, 9, 3], [7, 2, 8]]);
 * argmax(x);     // 1 (flattened index)
 * argmax(x, 1);  // [1, 2]
 * argmax(x, 0);  // [1, 0, 1]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function argmax(t: Tensor, axis?: Axis, keepdims = false): Tensor {
  return argExtremum(t, axis, keepdims, true, false, "argmax");
}

/**
 * Index of the minimum value along an axis (NumPy `argmin`).
 *
 * The first occurrence wins on ties. NaN propagates: the result is the index of the first NaN
 * in the slice. Without `axis` the tensor is flattened and the index refers to the flattened
 * (row-major) order. The result is always `int32`. int64 values are compared exactly.
 *
 * @param t - Input tensor
 * @param axis - Axis to search along; omit to search the flattened tensor
 * @param keepdims - If true, the reduced axis stays as a size-1 dimension (all axes when `axis`
 *   is omitted)
 * @returns `int32` tensor of indices
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If the searched slice is empty or `axis` is out of range
 *
 * @example
 * ```ts
 * import { argmin, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([[4, 9, 3], [7, 2, 8]]);
 * argmin(x);     // 4 (flattened index)
 * argmin(x, 1);  // [2, 1]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function argmin(t: Tensor, axis?: Axis, keepdims = false): Tensor {
  return argExtremum(t, axis, keepdims, false, false, "argmin");
}

/**
 * Index of the maximum value along an axis, ignoring NaN (NumPy `nanargmax`).
 *
 * The first occurrence wins on ties. The result is always `int32`.
 *
 * @param t - Input tensor
 * @param axis - Axis to search along; omit to search the flattened tensor
 * @param keepdims - If true, the reduced axis stays as a size-1 dimension
 * @returns `int32` tensor of indices
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If the searched slice is empty or `axis` is out of range
 * @throws {DataValidationError} If a slice contains only NaN (NumPy raises here too)
 *
 * @example
 * ```ts
 * import { nanargmax, tensor } from 'deepbox/ndarray';
 *
 * nanargmax(tensor([NaN, 3, 7, NaN]));  // 2
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function nanargmax(t: Tensor, axis?: Axis, keepdims = false): Tensor {
  return argExtremum(t, axis, keepdims, true, true, "nanargmax");
}

/**
 * Index of the minimum value along an axis, ignoring NaN (NumPy `nanargmin`).
 *
 * The first occurrence wins on ties. The result is always `int32`.
 *
 * @param t - Input tensor
 * @param axis - Axis to search along; omit to search the flattened tensor
 * @param keepdims - If true, the reduced axis stays as a size-1 dimension
 * @returns `int32` tensor of indices
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If the searched slice is empty or `axis` is out of range
 * @throws {DataValidationError} If a slice contains only NaN (NumPy raises here too)
 *
 * @example
 * ```ts
 * import { nanargmin, tensor } from 'deepbox/ndarray';
 *
 * nanargmin(tensor([NaN, 3, 1, NaN]));  // 2
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function nanargmin(t: Tensor, axis?: Axis, keepdims = false): Tensor {
  return argExtremum(t, axis, keepdims, false, true, "nanargmin");
}

// ---------------------------------------------------------------------------
// NaN-ignoring reductions
// ---------------------------------------------------------------------------

/** Plan for reducing `axis` (a single axis, a list, or every axis when omitted). */
function planNanReduction(
  t: Tensor,
  axis: AxisArg | undefined,
  keepdims: boolean,
  op: string
): { readonly plan: ReducePlan; readonly data: NumericTypedArray } {
  const axes = resolveAxes(axis, t.ndim);
  const all = axes ?? Array.from({ length: t.ndim }, (_, i) => i);
  return planWithView(t, all, keepdims, op);
}

/**
 * Variance ignoring NaN (NumPy `nanvar`).
 *
 * Computes `sum((x - mean(x))^2) / (count - ddof)` over the non-NaN values of each slice, in
 * float64. A slice with `count - ddof <= 0` non-NaN values (including an empty or all-NaN slice)
 * gives NaN. Float input keeps its dtype and integer or bool input gives `float32`.
 *
 * @param t - Input tensor
 * @param axis - Axis or list of axes to reduce. If omitted, all elements
 * @param keepdims - If true, reduced axes stay as size-1 dimensions
 * @param ddof - Delta degrees of freedom (default: 0, population variance)
 * @returns Tensor of variances
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If `ddof` is negative or not finite, or an axis is invalid
 *
 * @example
 * ```ts
 * import { nanvar, tensor } from 'deepbox/ndarray';
 *
 * nanvar(tensor([1, NaN, 3]));  // 1
 * nanvar(tensor([[1, NaN], [3, 5]]), 1);  // [0, 1]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function nanvar(t: Tensor, axis?: AxisArg, keepdims = false, ddof = 0): Tensor {
  ensureNumericTensor(t, "nanvar");
  if (!Number.isFinite(ddof) || ddof < 0) {
    throw new InvalidParameterError("ddof must be non-negative and finite", "ddof", ddof);
  }
  const outDtype = toFloatDType(t.dtype);
  const { plan, data } = planNanReduction(t, axis, keepdims, "nanvar");
  const { bases, redOffsets, redCount, outSize } = plan;
  const out = newTyped(outDtype, outSize);
  for (let o = 0; o < outSize; o++) {
    const base = bases[o] as number;
    let sum = 0;
    let count = 0;
    for (let r = 0; r < redCount; r++) {
      const v = data[base + (redOffsets[r] as number)] as number;
      if (!Number.isNaN(v)) {
        sum += v;
        count++;
      }
    }
    if (count - ddof <= 0) {
      out[o] = Number.NaN;
      continue;
    }
    const meanValue = sum / count;
    let sumSquaredDev = 0;
    for (let r = 0; r < redCount; r++) {
      const v = data[base + (redOffsets[r] as number)] as number;
      if (!Number.isNaN(v)) {
        const d = v - meanValue;
        sumSquaredDev += d * d;
      }
    }
    out[o] = sumSquaredDev / (count - ddof);
  }
  return roundHalfResult(
    Tensor.fromTypedArray({ data: out, shape: plan.outShape, dtype: outDtype, device: t.device })
  );
}

/**
 * Median ignoring NaN (NumPy `nanmedian`).
 *
 * Each slice uses only its non-NaN values; a slice without any (including an empty slice)
 * gives NaN. Float input keeps its dtype and integer or bool input gives `float32`.
 *
 * @param t - Input tensor
 * @param axis - Axis or list of axes to reduce. If omitted, all elements
 * @param keepdims - If true, reduced axes stay as size-1 dimensions
 * @returns Tensor of medians
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If an axis is out of range or listed twice
 *
 * @example
 * ```ts
 * import { nanmedian, tensor } from 'deepbox/ndarray';
 *
 * nanmedian(tensor([1, NaN, 3, 10]));  // 3
 * nanmedian(tensor([[1, NaN], [3, 5]]), 1);  // [1, 4]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function nanmedian(t: Tensor, axis?: AxisArg, keepdims = false): Tensor {
  ensureNumericTensor(t, "nanmedian");
  const outDtype = toFloatDType(t.dtype);
  const { plan, data } = planNanReduction(t, axis, keepdims, "nanmedian");
  const { bases, redOffsets, redCount, outSize } = plan;
  const out = newTyped(outDtype, outSize);
  const scratch = new Float64Array(redCount);
  for (let o = 0; o < outSize; o++) {
    const base = bases[o] as number;
    let count = 0;
    for (let r = 0; r < redCount; r++) {
      const v = data[base + (redOffsets[r] as number)] as number;
      if (!Number.isNaN(v)) scratch[count++] = v;
    }
    out[o] = count === 0 ? Number.NaN : medianOf(scratch.subarray(0, count));
  }
  return roundHalfResult(
    Tensor.fromTypedArray({ data: out, shape: plan.outShape, dtype: outDtype, device: t.device })
  );
}

/**
 * Product treating NaN as one (NumPy `nanprod`).
 *
 * The product of an empty or all-NaN slice is 1. The dtype follows {@link prod}: float dtypes
 * are kept, `int32` and `int64` stay, `uint8` and `bool` give `int32`.
 *
 * @param t - Input tensor
 * @param axis - Axis or list of axes to reduce. If omitted, all elements
 * @param keepdims - If true, reduced axes stay as size-1 dimensions
 * @returns Tensor of products
 * @throws {DTypeError} If `t` has string dtype
 * @throws {DataValidationError} If an integer product overflows
 * @throws {InvalidParameterError} If an axis is out of range or listed twice
 *
 * @example
 * ```ts
 * import { nanprod, tensor } from 'deepbox/ndarray';
 *
 * nanprod(tensor([2, NaN, 4]));  // 8
 * nanprod(tensor([[2, NaN], [3, 5]]), 0);  // [6, 5]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function nanprod(t: Tensor, axis?: AxisArg, keepdims = false): Tensor {
  ensureNumericTensor(t, "nanprod");
  // Integer and bool tensors cannot hold NaN.
  if (toFloatDType(t.dtype) !== t.dtype) return prod(t, axis, keepdims);
  const { plan, data } = planNanReduction(t, axis, keepdims, "nanprod");
  const { bases, redOffsets, redCount, outSize } = plan;
  const out = newTyped(t.dtype as NumericDType, outSize);
  for (let o = 0; o < outSize; o++) {
    const base = bases[o] as number;
    let acc = 1;
    for (let r = 0; r < redCount; r++) {
      const v = data[base + (redOffsets[r] as number)] as number;
      if (!Number.isNaN(v)) acc *= v;
    }
    out[o] = acc;
  }
  return roundHalfResult(
    Tensor.fromTypedArray({ data: out, shape: plan.outShape, dtype: t.dtype, device: t.device })
  );
}

/**
 * Cumulative sum treating NaN as zero (NumPy `nancumsum`).
 *
 * Without `axis` the tensor is flattened first. The dtype follows {@link cumsum}: float dtypes
 * are kept, `int32` and `int64` stay, `uint8` and `bool` give `int32`.
 *
 * @param t - Input tensor
 * @param axis - Axis to accumulate along; omit to accumulate over the flattened tensor
 * @returns Tensor of cumulative sums
 * @throws {DTypeError} If `t` has string dtype
 * @throws {DataValidationError} If an integer sum overflows
 * @throws {InvalidParameterError} If `axis` is out of range
 *
 * @example
 * ```ts
 * import { nancumsum, tensor } from 'deepbox/ndarray';
 *
 * nancumsum(tensor([1, NaN, 3]));  // [1, 1, 4]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function nancumsum(t: Tensor, axis?: Axis): Tensor {
  ensureNumericTensor(t, "nancumsum");
  if (toFloatDType(t.dtype) !== t.dtype) return cumsum(t, axis);
  const src = readNumbers(t, "nancumsum");
  const clean = newTyped(t.dtype as NumericDType, src.length);
  for (let i = 0; i < src.length; i++) {
    const v = src[i] as number;
    clean[i] = Number.isNaN(v) ? 0 : v;
  }
  return cumsum(
    Tensor.fromTypedArray({ data: clean, shape: t.shape, dtype: t.dtype, device: t.device }),
    axis
  );
}

/**
 * Linear interpolation between neighbouring order statistics, with NumPy's rounding behaviour
 * for finite values. An infinite neighbour gives that infinity (the limit of the formula)
 * instead of the NaN that `inf - inf` would produce; opposite infinities give NaN.
 */
function lerpQuantile(a: number, b: number, t: number): number {
  if (t === 0 || a === b) return a;
  if (t === 1) return b;
  if (!Number.isFinite(a) || !Number.isFinite(b)) {
    if (!Number.isFinite(a) && !Number.isFinite(b)) return Number.NaN;
    return Number.isFinite(a) ? b : a;
  }
  const diff = b - a;
  return t >= 0.5 ? b - diff * (1 - t) : a + diff * t;
}

/**
 * Quantiles ignoring NaN, with linear interpolation (NumPy `nanquantile`, `method="linear"`).
 *
 * Each slice uses only its non-NaN values; a slice without any (including an empty slice)
 * gives NaN. With a scalar `q` the result has the reduced shape; with an array `q` a leading
 * axis of length `q.length` is added, as in NumPy. Float input keeps its dtype and integer or
 * bool input gives `float32`. An infinite value interpolates to that infinity where NumPy
 * would return NaN from `inf - inf` (the same rule as `quantile` in the stats module).
 *
 * @param t - Input tensor
 * @param q - Quantile or list of quantiles in `[0, 1]`
 * @param axis - Axis or list of axes to reduce. If omitted, all elements
 * @param keepdims - If true, reduced axes stay as size-1 dimensions
 * @returns Tensor of quantiles
 * @throws {DTypeError} If `t` has string dtype
 * @throws {InvalidParameterError} If a quantile is outside `[0, 1]` or not finite, or an axis is invalid
 *
 * @example
 * ```ts
 * import { nanquantile, tensor } from 'deepbox/ndarray';
 *
 * nanquantile(tensor([1, NaN, 3, 5]), 0.5);        // 3
 * nanquantile(tensor([1, NaN, 3, 5]), [0.25, 1]);  // [2, 5]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function nanquantile(
  t: Tensor,
  q: number | readonly number[],
  axis?: AxisArg,
  keepdims = false
): Tensor {
  ensureNumericTensor(t, "nanquantile");
  const qs: readonly number[] = typeof q === "number" ? [q] : q;
  for (const v of qs) {
    if (typeof v !== "number" || !Number.isFinite(v) || v < 0 || v > 1) {
      throw new InvalidParameterError("q must be in [0, 1]", "q", v);
    }
  }
  const outDtype = toFloatDType(t.dtype);
  const { plan, data } = planNanReduction(t, axis, keepdims, "nanquantile");
  const { bases, redOffsets, redCount, outSize } = plan;
  const nq = qs.length;
  const out = newTyped(outDtype, nq * outSize);
  const scratch = new Float64Array(redCount);
  for (let o = 0; o < outSize; o++) {
    const base = bases[o] as number;
    let count = 0;
    for (let r = 0; r < redCount; r++) {
      const v = data[base + (redOffsets[r] as number)] as number;
      if (!Number.isNaN(v)) scratch[count++] = v;
    }
    const sorted = scratch.subarray(0, count);
    sorted.sort();
    for (let qi = 0; qi < nq; qi++) {
      const qv = qs[qi] as number;
      let result: number;
      if (count === 0) {
        result = Number.NaN;
      } else {
        const virtual = (count - 1) * qv;
        if (virtual >= count - 1) {
          result = sorted[count - 1] as number;
        } else {
          const lo = Math.floor(virtual);
          result = lerpQuantile(sorted[lo] as number, sorted[lo + 1] as number, virtual - lo);
        }
      }
      out[qi * outSize + o] = result;
    }
  }
  const shape = typeof q === "number" ? plan.outShape : [nq, ...plan.outShape];
  return roundHalfResult(
    Tensor.fromTypedArray({ data: out, shape, dtype: outDtype, device: t.device })
  );
}
