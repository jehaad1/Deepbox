/**
 * Element-wise arithmetic: add, sub, mul, div, floorDiv, mod, pow, neg, abs,
 * sign, reciprocal, maximum, minimum, clip and the scalar helpers.
 *
 * Binary ops follow NumPy broadcasting and promote their operands like PyTorch:
 * `bool < uint8 < int32 < int64 < float16, bfloat16 < float32 < float64`, the wider
 * type of one category wins, an integer with a float gives the float type, and
 * `float16` with `bfloat16` gives `float32`. Number scalars never upcast a tensor
 * (an integer tensor combined with a fractional scalar becomes `float32`). Float
 * results keep the float dtype of the inputs; integer input to an op with
 * fractional results (`div`, `reciprocal`) gives `float32`. Integer dtypes wrap
 * on overflow like their NumPy counterparts.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */

import {
  type DType,
  DTypeError,
  dtypeToTypedArrayCtor,
  getBigIntElement,
  getNumericElement,
  InvalidParameterError,
  type Shape,
} from "../../core";
import { isFloatDType, toFloatDType } from "../../core/utils/dtype_utils";
import type { NumericTypedArray } from "../../core/utils/typed_array_access";
import { isContiguous } from "../tensor/strides";
import { computeStrides, isBigIntArray, Tensor } from "../tensor/Tensor";
import { flatOffset, readNumbers, readNumericContiguous, roundHalfResult } from "./_internal";
import {
  broadcastApply,
  ensureBroadcastableScalar,
  ensureNumericDType,
  getBroadcastShape,
  isScalar,
  promoteOperands,
} from "./broadcast";
import { dispatchBinary, dispatchUnary } from "./device_dispatch";

type NumericDType = Exclude<DType, "string">;
type NumberTypedArray = Float32Array | Float64Array | Int32Array | Uint8Array;

/** Result dtype of true division and other ops that must produce fractions. */
function outDtypeForTrueDiv(dtype: NumericDType): NumericDType {
  return toFloatDType(dtype);
}

function shapeSize(shape: Shape): number {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

function getOutShape(a: Tensor, b: Tensor): Shape {
  if (isScalar(a)) return b.shape;
  if (isScalar(b)) return a.shape;
  return getBroadcastShape(a.shape, b.shape);
}

/**
 * Check if two tensors are eligible for the contiguous fast path:
 * same shape, both contiguous, both non-BigInt numeric.
 * Returns zero-based numeric typed arrays if eligible, or null otherwise.
 */
function fastPathArrays(
  a: Tensor,
  b: Tensor
): { aArr: NumericTypedArray; bArr: NumericTypedArray; size: number } | null {
  if (a.size === 0) return null;
  if (a.dtype === "int64" || b.dtype === "int64") return null;
  const aData = a.data;
  const bData = b.data;
  if (Array.isArray(aData) || Array.isArray(bData)) return null;
  if (aData instanceof BigInt64Array || bData instanceof BigInt64Array) return null;
  // Same shape check
  if (a.ndim !== b.ndim) return null;
  for (let i = 0; i < a.ndim; i++) {
    if (a.shape[i] !== b.shape[i]) return null;
  }
  if (!isContiguous(a.shape, a.strides) || !isContiguous(b.shape, b.strides)) return null;
  // Slice offset views once so the hot loops index from 0 (V8 eliminates
  // bounds checks for plain [i] indexing).
  const size = a.size;
  const aArr = (
    a.offset === 0 ? aData : aData.subarray(a.offset, a.offset + size)
  ) as NumericTypedArray;
  const bArr = (
    b.offset === 0 ? bData : bData.subarray(b.offset, b.offset + size)
  ) as NumericTypedArray;
  return { aArr, bArr, size };
}

/**
 * Strided same-shape fast path for float/int32/uint8 operands:
 * walks both tensors with an incremental odometer (no per-element closures
 * or div/mod), keeping transposed/sliced views within ~2-4x of contiguous
 * speed instead of the generic broadcast machinery's ~30x.
 */
function stridedBinaryNumeric(
  op: "add" | "sub" | "mul" | "div",
  a: Tensor,
  b: Tensor
): Tensor | null {
  const dtype = a.dtype;
  if (a.size === 0 || dtype === "bool" || dtype === "int64" || dtype === "string") return null;
  if (a.ndim !== b.ndim || a.ndim === 0) return null;
  for (let i = 0; i < a.ndim; i++) {
    if (a.shape[i] !== b.shape[i]) return null;
  }
  const aData = a.data;
  const bData = b.data;
  if (Array.isArray(aData) || Array.isArray(bData)) return null;
  if (aData instanceof BigInt64Array || bData instanceof BigInt64Array) return null;

  const outDtype = op === "div" ? outDtypeForTrueDiv(dtype) : dtype;
  if (op === "div" && outDtype !== dtype) return null;
  // int32 products can exceed 2^53, where a double multiply loses the low
  // bits that wrap-around semantics depend on; Math.imul keeps them exact.
  const int32Mul = op === "mul" && dtype === "int32";

  const ndim = a.ndim;
  const shape = a.shape;
  const aStrides = a.strides;
  const bStrides = b.strides;
  const inner = shape[ndim - 1] ?? 0;
  const aS = aStrides[ndim - 1] ?? 0;
  const bS = bStrides[ndim - 1] ?? 0;
  const outerSize = a.size / (inner === 0 ? 1 : inner);

  const Ctor = dtypeToTypedArrayCtor(outDtype);
  const out = new Ctor(a.size) as NumberTypedArray;

  const coords = new Array<number>(ndim - 1).fill(0);
  let aBase = a.offset;
  let bBase = b.offset;
  let outPos = 0;

  for (let block = 0; block < outerSize; block++) {
    let aIdx = aBase;
    let bIdx = bBase;
    if (op === "add") {
      for (let i = 0; i < inner; i++) {
        out[outPos++] = (aData[aIdx] as number) + (bData[bIdx] as number);
        aIdx += aS;
        bIdx += bS;
      }
    } else if (op === "sub") {
      for (let i = 0; i < inner; i++) {
        out[outPos++] = (aData[aIdx] as number) - (bData[bIdx] as number);
        aIdx += aS;
        bIdx += bS;
      }
    } else if (int32Mul) {
      for (let i = 0; i < inner; i++) {
        out[outPos++] = Math.imul(aData[aIdx] as number, bData[bIdx] as number);
        aIdx += aS;
        bIdx += bS;
      }
    } else if (op === "mul") {
      for (let i = 0; i < inner; i++) {
        out[outPos++] = (aData[aIdx] as number) * (bData[bIdx] as number);
        aIdx += aS;
        bIdx += bS;
      }
    } else {
      for (let i = 0; i < inner; i++) {
        out[outPos++] = (aData[aIdx] as number) / (bData[bIdx] as number);
        aIdx += aS;
        bIdx += bS;
      }
    }
    // Odometer increment over the outer dimensions.
    for (let d = ndim - 2; d >= 0; d--) {
      coords[d] = (coords[d] ?? 0) + 1;
      aBase += aStrides[d] ?? 0;
      bBase += bStrides[d] ?? 0;
      if ((coords[d] ?? 0) < (shape[d] ?? 0)) break;
      aBase -= (aStrides[d] ?? 0) * (shape[d] ?? 0);
      bBase -= (bStrides[d] ?? 0) * (shape[d] ?? 0);
      coords[d] = 0;
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: a.shape,
    dtype: outDtype,
    device: a.device,
  });
}

/**
 * Apply a number-valued binary function with broadcasting. Same-shape
 * contiguous operands take a tight loop; everything else walks the broadcast
 * offsets. Operands must not be int64 or string (see {@link asNumberTensor}).
 */
function mapBinaryNumbers(
  a: Tensor,
  b: Tensor,
  outShape: Shape,
  outDtype: NumericDType,
  f: (x: number, y: number) => number
): Tensor {
  const aData = a.data;
  const bData = b.data;
  if (Array.isArray(aData) || Array.isArray(bData)) {
    throw new DTypeError("Binary arithmetic is not defined for string dtype");
  }
  if (aData instanceof BigInt64Array || bData instanceof BigInt64Array) {
    throw new DTypeError("Internal error: int64 operand reached a number kernel");
  }
  const Ctor = dtypeToTypedArrayCtor(outDtype);
  const out = new Ctor(shapeSize(outShape)) as NumberTypedArray;
  const result = Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: outDtype,
    device: a.device,
  });

  const fp = fastPathArrays(a, b);
  if (fp) {
    const { aArr, bArr, size } = fp;
    for (let i = 0; i < size; i++) {
      out[i] = f(aArr[i] as number, bArr[i] as number);
    }
    return result;
  }

  broadcastApply(a, b, result, (offA, offB, offOut) => {
    out[offOut] = f(aData[offA] as number, bData[offB] as number);
  });
  return result;
}

/** Apply a bigint-valued binary function with broadcasting to two int64 tensors. */
function mapBinaryBigInt(
  a: Tensor,
  b: Tensor,
  outShape: Shape,
  f: (x: bigint, y: bigint) => bigint
): Tensor {
  const aData = a.data;
  const bData = b.data;
  if (!(aData instanceof BigInt64Array) || !(bData instanceof BigInt64Array)) {
    throw new DTypeError("Internal error: non-int64 operand reached a bigint kernel");
  }
  const out = new BigInt64Array(shapeSize(outShape));
  const result = Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "int64",
    device: a.device,
  });
  broadcastApply(a, b, result, (offA, offB, offOut) => {
    out[offOut] = f(getBigIntElement(aData, offA), getBigIntElement(bData, offB));
  });
  return result;
}

/**
 * Return `t` unchanged unless it is int64, in which case return a float64
 * copy. Lets mixed number/bigint ops (true division, negative-exponent pow)
 * share the number kernels.
 */
function asNumberTensor(t: Tensor, opName: string): Tensor {
  if (!(t.data instanceof BigInt64Array)) return t;
  return Tensor.fromTypedArray({
    data: readNumbers(t, opName) as Float64Array,
    shape: t.shape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Floor division for bigint values (rounds toward -Infinity).
 *
 * JavaScript bigint division truncates toward zero. Deepbox floor division
 * rounds toward -Infinity, so negative values with a remainder need
 * an additional decrement. Division by zero yields 0, like NumPy's integer
 * `floor_divide`.
 */
function floorDivBigInt(a: bigint, b: bigint): bigint {
  if (b === 0n) return 0n;
  const q = a / b;
  const r = a % b;
  const rPositive = r > 0n;
  const bPositive = b > 0n;
  if (r !== 0n && rPositive !== bPositive) {
    return q - 1n;
  }
  return q;
}

/** Modulo for bigint values: the result takes the sign of the divisor; `x % 0` is 0. */
function modBigInt(a: bigint, b: bigint): bigint {
  if (b === 0n) return 0n;
  const r = a % b;
  return r !== 0n && r > 0n !== b > 0n ? r + b : r;
}

/** Sign of zero follows `copysign(0, x)`: -0 for negative x and for -0, else +0. */
function zeroWithSignOf(x: number): number {
  return x < 0 || Object.is(x, -0) ? -0 : 0;
}

/**
 * Floating-point floor division, ported from NumPy's `npy_divmod`.
 *
 * `Math.floor(a / b)` is wrong when the rounded quotient lands on an integer
 * although the exact quotient is just below it: 1 // 0.1 is 9, not 10. The
 * quotient is rebuilt from the exact `fmod` remainder instead.
 */
function floorDivFloat(a: number, b: number): number {
  // Fast path: a rounded quotient that is not an integer has the same floor as
  // the exact quotient (rounding is monotone and integers are representable),
  // so only integer-valued, infinite or NaN quotients need the exact remainder.
  // Quotients of magnitude 2^50 or more are excluded: their spacing is 0.25 or
  // wider, and NumPy's remainder-based result can differ from floor(a / b) there.
  const q = a / b;
  const fl = Math.floor(q);
  if (q !== fl && Math.abs(q) < 1125899906842624) return fl;
  if (b === 0) return q;
  const m = a % b;
  let div = (a - m) / b;
  if (m !== 0 && b < 0 !== m < 0) {
    div -= 1;
  }
  if (div !== 0) {
    const whole = Math.floor(div);
    return div - whole > 0.5 ? whole + 1 : whole;
  }
  return zeroWithSignOf(a / b);
}

/**
 * Single-precision variant of {@link floorDivFloat}. NumPy evaluates float32
 * `floor_divide` with float32 intermediates, which changes the snapped quotient
 * once it exceeds about 2^21, so the intermediates are rounded with `fround`.
 */
function floorDivFloat32(a: number, b: number): number {
  const q = Math.fround(a / b);
  const fl = Math.floor(q);
  if (q !== fl && Math.abs(q) < 2097152) return fl;
  if (b === 0) return q;
  const m = a % b;
  let div = Math.fround(Math.fround(a - m) / b);
  if (m !== 0 && b < 0 !== m < 0) {
    div = Math.fround(div - 1);
  }
  if (div !== 0) {
    const whole = Math.floor(div);
    return Math.fround(div - whole) > 0.5 ? whole + 1 : whole;
  }
  return zeroWithSignOf(a / b);
}

/**
 * Floating-point modulo with the sign of the divisor (Python `%`, NumPy `mod`).
 *
 * Uses the exact `fmod` remainder. The textbook `a - floor(a / b) * b` loses
 * every digit once `a / b` exceeds 2^53 (`mod(1e17, 3)` would give 0, not 1).
 */
function modFloat(a: number, b: number): number {
  if (b === 0) return Number.NaN;
  const m = a % b;
  if (m === 0) return zeroWithSignOf(b);
  return b < 0 !== m < 0 ? m + b : m;
}

/** Integer floor division; division by zero yields 0 like NumPy. */
function floorDivInt(a: number, b: number): number {
  return b === 0 ? 0 : Math.floor(a / b);
}

/** Integer modulo with the sign of the divisor; `x % 0` is 0 like NumPy. */
function modInt(a: number, b: number): number {
  if (b === 0) return 0;
  const m = a % b;
  return m !== 0 && b < 0 !== m < 0 ? m + b : m;
}

function hasNegativeExponent(t: Tensor): boolean {
  if (t.size === 0) return false;
  const data = t.data;
  if (data instanceof BigInt64Array) {
    const logicalStrides = computeStrides(t.shape);
    const contiguous = isContiguous(t.shape, t.strides);
    for (let i = 0; i < t.size; i++) {
      const offset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      if (getBigIntElement(data, offset) < 0n) return true;
    }
    return false;
  }
  const src = readNumericContiguous(t);
  if (src === null) return false; // string tensor
  for (let i = 0; i < src.length; i++) {
    if ((src[i] as number) < 0) return true;
  }
  return false;
}

/**
 * Element-wise addition.
 *
 * The operands are promoted to a common dtype (see the module notes). Shapes
 * follow NumPy broadcasting (a 0-d tensor broadcasts against anything). For
 * two `bool` tensors the result is the logical OR, as in NumPy. Integer dtypes
 * wrap on overflow.
 *
 * @param a - First tensor
 * @param b - Second tensor
 * @returns Tensor containing the element-wise sum, of the promoted dtype
 * @throws {DTypeError} If either tensor has `string` dtype
 * @throws {ShapeError} If the shapes do not broadcast
 *
 * @example
 * ```ts
 * import { add, tensor } from 'deepbox/ndarray';
 *
 * add(tensor([1, 2, 3]), tensor([10, 20, 30]));  // [11, 22, 33]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function add(a: Tensor, b: Tensor): Tensor {
  return roundHalfResult(addCore(a, b));
}

function addCore(a0: Tensor, b0: Tensor): Tensor {
  const [a, b] = promoteOperands(a0, b0, "add");
  ensureBroadcastableScalar(a, b);

  if (a.device !== "cpu" || b.device !== "cpu") {
    const onDevice = dispatchBinary("add", a, b);
    if (onDevice) return onDevice;
  }

  // bool + bool is logical OR (NumPy semantics); a raw sum would store
  // out-of-range values like 2 in a bool tensor.
  const isBool = a.dtype === "bool";

  // Fast path: contiguous same-shape numeric tensors
  const fp = fastPathArrays(a, b);
  if (fp) {
    const Ctor = dtypeToTypedArrayCtor(a.dtype);
    const out = new Ctor(fp.size);
    const { aArr, bArr } = fp;
    if (isBool) {
      for (let i = 0; i < fp.size; i++) {
        out[i] = (aArr[i] as number) !== 0 || (bArr[i] as number) !== 0 ? 1 : 0;
      }
    } else {
      for (let i = 0; i < fp.size; i++) {
        out[i] = (aArr[i] as number) + (bArr[i] as number);
      }
    }
    return Tensor.fromTypedArray({
      data: out,
      shape: a.shape,
      dtype: a.dtype,
      device: a.device,
    });
  }

  if (!isBool) {
    const strided = stridedBinaryNumeric("add", a, b);
    if (strided) return strided;
  }

  // Handle broadcasting
  const outShape = getOutShape(a, b);
  const outSize = shapeSize(outShape);

  const Ctor = dtypeToTypedArrayCtor(a.dtype);
  const out = new Ctor(outSize);
  const result = Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
  });

  if (isBigIntArray(out) && isBigIntArray(a.data) && isBigIntArray(b.data)) {
    const aData = a.data;
    const bData = b.data;
    broadcastApply(a, b, result, (offA, offB, offOut) => {
      out[offOut] = getBigIntElement(aData, offA) + getBigIntElement(bData, offB);
    });
  } else if (!isBigIntArray(out) && !isBigIntArray(a.data) && !isBigIntArray(b.data)) {
    const aData = a.data;
    const bData = b.data;
    if (isBool) {
      broadcastApply(a, b, result, (offA, offB, offOut) => {
        out[offOut] =
          getNumericElement(aData, offA) !== 0 || getNumericElement(bData, offB) !== 0 ? 1 : 0;
      });
    } else {
      broadcastApply(a, b, result, (offA, offB, offOut) => {
        out[offOut] = getNumericElement(aData, offA) + getNumericElement(bData, offB);
      });
    }
  }

  return result;
}

/**
 * Element-wise subtraction.
 *
 * Computes a - b element by element. The operands are promoted to a common
 * dtype and their shapes must broadcast. Integer dtypes wrap on overflow.
 *
 * @param a - First tensor
 * @param b - Second tensor
 * @returns Tensor containing element-wise difference, of the promoted dtype
 * @throws {DTypeError} If the promoted dtype is `bool`, or either tensor is `string`
 * @throws {ShapeError} If the shapes do not broadcast
 *
 * @example
 * ```ts
 * import { sub, tensor } from 'deepbox/ndarray';
 *
 * const a = tensor([5, 6, 7]);
 * const b = tensor([1, 2, 3]);
 * const result = sub(a, b);  // [4, 4, 4]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function sub(a: Tensor, b: Tensor): Tensor {
  return roundHalfResult(subCore(a, b));
}

function subCore(a0: Tensor, b0: Tensor): Tensor {
  const [a, b] = promoteOperands(a0, b0, "sub");
  ensureBroadcastableScalar(a, b);

  if (a.device !== "cpu" || b.device !== "cpu") {
    const onDevice = dispatchBinary("sub", a, b);
    if (onDevice) return onDevice;
  }

  if (a.dtype === "bool") {
    // true - true would store -1/255 in a bool tensor; NumPy raises here too.
    throw new DTypeError(
      "sub is not supported for bool tensors; cast to a numeric dtype or use logicalXor"
    );
  }

  // Fast path: contiguous same-shape numeric tensors
  const fp = fastPathArrays(a, b);
  if (fp) {
    const Ctor = dtypeToTypedArrayCtor(a.dtype);
    const out = new Ctor(fp.size);
    const { aArr, bArr } = fp;
    for (let i = 0; i < fp.size; i++) {
      out[i] = (aArr[i] as number) - (bArr[i] as number);
    }
    return Tensor.fromTypedArray({
      data: out,
      shape: a.shape,
      dtype: a.dtype,
      device: a.device,
    });
  }

  const strided = stridedBinaryNumeric("sub", a, b);
  if (strided) return strided;

  const outShape = getOutShape(a, b);
  const outSize = shapeSize(outShape);

  const Ctor = dtypeToTypedArrayCtor(a.dtype);
  const out = new Ctor(outSize);
  const result = Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
  });

  if (isBigIntArray(out) && isBigIntArray(a.data) && isBigIntArray(b.data)) {
    const aData = a.data;
    const bData = b.data;
    broadcastApply(a, b, result, (offA, offB, offOut) => {
      out[offOut] = getBigIntElement(aData, offA) - getBigIntElement(bData, offB);
    });
  } else if (!isBigIntArray(out) && !isBigIntArray(a.data) && !isBigIntArray(b.data)) {
    const aData = a.data;
    const bData = b.data;
    broadcastApply(a, b, result, (offA, offB, offOut) => {
      out[offOut] = getNumericElement(aData, offA) - getNumericElement(bData, offB);
    });
  }

  return result;
}

/**
 * Element-wise multiplication.
 *
 * Computes a * b element by element. The operands are promoted to a common
 * dtype and their shapes must broadcast. Integer dtypes wrap on overflow
 * (int32 products are computed exactly modulo 2^32).
 *
 * @param a - First tensor
 * @param b - Second tensor
 * @returns Tensor containing element-wise product, of the promoted dtype
 * @throws {DTypeError} If either tensor has `string` dtype
 * @throws {ShapeError} If the shapes do not broadcast
 *
 * @example
 * ```ts
 * import { mul, tensor } from 'deepbox/ndarray';
 *
 * const a = tensor([2, 3, 4]);
 * const b = tensor([5, 6, 7]);
 * const result = mul(a, b);  // [10, 18, 28]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function mul(a: Tensor, b: Tensor): Tensor {
  return roundHalfResult(mulCore(a, b));
}

function mulCore(a0: Tensor, b0: Tensor): Tensor {
  const [a, b] = promoteOperands(a0, b0, "mul");
  ensureBroadcastableScalar(a, b);

  if (a.device !== "cpu" || b.device !== "cpu") {
    const onDevice = dispatchBinary("mul", a, b);
    if (onDevice) return onDevice;
  }

  const int32 = a.dtype === "int32";

  // Fast path: contiguous same-shape numeric tensors
  const fp = fastPathArrays(a, b);
  if (fp) {
    const Ctor = dtypeToTypedArrayCtor(a.dtype);
    const out = new Ctor(fp.size);
    const { aArr, bArr } = fp;
    if (int32) {
      for (let i = 0; i < fp.size; i++) {
        out[i] = Math.imul(aArr[i] as number, bArr[i] as number);
      }
    } else {
      for (let i = 0; i < fp.size; i++) {
        out[i] = (aArr[i] as number) * (bArr[i] as number);
      }
    }
    return Tensor.fromTypedArray({
      data: out,
      shape: a.shape,
      dtype: a.dtype,
      device: a.device,
    });
  }

  const strided = stridedBinaryNumeric("mul", a, b);
  if (strided) return strided;

  const outShape = getOutShape(a, b);
  const outSize = shapeSize(outShape);

  const Ctor = dtypeToTypedArrayCtor(a.dtype);
  const out = new Ctor(outSize);
  const result = Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: a.dtype,
    device: a.device,
  });

  if (isBigIntArray(out) && isBigIntArray(a.data) && isBigIntArray(b.data)) {
    const aData = a.data;
    const bData = b.data;
    broadcastApply(a, b, result, (offA, offB, offOut) => {
      out[offOut] = getBigIntElement(aData, offA) * getBigIntElement(bData, offB);
    });
  } else if (!isBigIntArray(out) && !isBigIntArray(a.data) && !isBigIntArray(b.data)) {
    const aData = a.data;
    const bData = b.data;
    if (int32) {
      broadcastApply(a, b, result, (offA, offB, offOut) => {
        out[offOut] = Math.imul(getNumericElement(aData, offA), getNumericElement(bData, offB));
      });
    } else {
      broadcastApply(a, b, result, (offA, offB, offOut) => {
        out[offOut] = getNumericElement(aData, offA) * getNumericElement(bData, offB);
      });
    }
  }

  return result;
}

/**
 * Element-wise true division.
 *
 * Computes a / b element by element. The operands are promoted to a common
 * dtype first. Float dtypes keep their dtype; integer and bool operands produce
 * `float32`, as in PyTorch. Division by zero follows IEEE 754 (`Infinity`,
 * `-Infinity` or `NaN`).
 *
 * @param a - Numerator tensor
 * @param b - Denominator tensor
 * @returns Tensor containing element-wise quotient
 * @throws {DTypeError} If either tensor has `string` dtype
 * @throws {ShapeError} If the shapes do not broadcast
 *
 * @example
 * ```ts
 * import { div, tensor } from 'deepbox/ndarray';
 *
 * const a = tensor([10, 20, 30]);
 * const b = tensor([2, 4, 5]);
 * const result = div(a, b);  // [5, 5, 6]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function div(a: Tensor, b: Tensor): Tensor {
  return roundHalfResult(divCore(a, b));
}

function divCore(a0: Tensor, b0: Tensor): Tensor {
  const [a, b] = promoteOperands(a0, b0, "div");
  ensureBroadcastableScalar(a, b);

  if (a.device !== "cpu" || b.device !== "cpu") {
    const onDevice = dispatchBinary("div", a, b);
    if (onDevice) return onDevice;
  }

  // Fast path: contiguous same-shape float tensors (no dtype promotion needed)
  const outDtype = outDtypeForTrueDiv(a.dtype);
  const fp = fastPathArrays(a, b);
  if (fp && outDtype === a.dtype) {
    const Ctor = dtypeToTypedArrayCtor(outDtype);
    const out = new Ctor(fp.size);
    const { aArr, bArr } = fp;
    for (let i = 0; i < fp.size; i++) {
      out[i] = (aArr[i] as number) / (bArr[i] as number);
    }
    return Tensor.fromTypedArray({
      data: out,
      shape: a.shape,
      dtype: outDtype,
      device: a.device,
    });
  }

  const strided = stridedBinaryNumeric("div", a, b);
  if (strided) return strided;

  return mapBinaryNumbers(
    asNumberTensor(a, "div"),
    asNumberTensor(b, "div"),
    getOutShape(a, b),
    outDtype,
    divNumbers
  );
}

function divNumbers(x: number, y: number): number {
  return x / y;
}

/**
 * Shared implementation of {@link addScalar} and {@link mulScalar}.
 *
 * - float dtypes keep their dtype; for float32/float16/bfloat16 the scalar is
 *   rounded to float32 first, which matches NumPy and PyTorch (`x * 0.1` in
 *   float32 multiplies by `fround(0.1)`).
 * - int32/uint8/int64 keep their dtype for integer scalars (wrapping on
 *   overflow). A non-integer scalar gives a float32 result (a number scalar
 *   never upcasts a tensor to float64).
 * - bool tensors become int32 (integer scalar) or float32.
 */
function scalarArithmetic(t: Tensor, s: number, op: "add" | "mul", name: string): Tensor {
  ensureNumericDType(t, name);
  if (typeof s !== "number") {
    throw new InvalidParameterError(
      `${name}: scalar must be a number; received ${typeof s}`,
      "s",
      s
    );
  }
  if (t.device !== "cpu") {
    const scalarT = Tensor.fromTypedArray({
      data: new Float64Array([s]),
      shape: [],
      dtype: "float64",
      device: "cpu",
    });
    const onDevice = dispatchBinary(op, t, scalarT);
    if (onDevice) return onDevice;
  }

  const size = t.size;
  const dtype = t.dtype;

  if (isFloatDType(dtype)) {
    const sv = dtype === "float64" ? s : Math.fround(s);
    const src = readNumbers(t, name);
    const Ctor = dtypeToTypedArrayCtor(dtype);
    const out = new Ctor(size) as Float32Array | Float64Array;
    if (op === "add") {
      for (let i = 0; i < size; i++) out[i] = (src[i] as number) + sv;
    } else {
      for (let i = 0; i < size; i++) out[i] = (src[i] as number) * sv;
    }
    return Tensor.fromTypedArray({ data: out, shape: t.shape, dtype, device: t.device });
  }

  if (!Number.isInteger(s)) {
    // Integer or bool tensor with a fractional scalar: both sides convert to
    // float32 first (a number scalar never upcasts a tensor), as PyTorch does.
    const src = readNumbers(t, name, false);
    const out = new Float32Array(size);
    const sv = Math.fround(s);
    if (op === "add") {
      for (let i = 0; i < size; i++) out[i] = Math.fround(src[i] as number) + sv;
    } else {
      for (let i = 0; i < size; i++) out[i] = Math.fround(src[i] as number) * sv;
    }
    return Tensor.fromTypedArray({ data: out, shape: t.shape, dtype: "float32", device: t.device });
  }

  if (dtype === "int64") {
    const data = t.data;
    if (!(data instanceof BigInt64Array)) {
      throw new DTypeError(`${name}: int64 tensor without BigInt storage`);
    }
    const scalar = BigInt(s);
    const out = new BigInt64Array(size);
    const logicalStrides = computeStrides(t.shape);
    const contiguous = isContiguous(t.shape, t.strides);
    for (let i = 0; i < size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const v = getBigIntElement(data, srcOffset);
      out[i] = op === "add" ? v + scalar : v * scalar;
    }
    return Tensor.fromTypedArray({ data: out, shape: t.shape, dtype, device: t.device });
  }

  // int32 / uint8 / bool: wrap-around arithmetic on the scalar's low 32 bits
  // (exactly what NumPy does after casting the scalar to the tensor dtype).
  const outDtype: NumericDType = dtype === "bool" ? "int32" : dtype;
  const src = readNumbers(t, name);
  const Ctor = dtypeToTypedArrayCtor(outDtype);
  const out = new Ctor(size) as Int32Array | Uint8Array;
  const si = s | 0;
  if (op === "add") {
    for (let i = 0; i < size; i++) out[i] = (src[i] as number) + si;
  } else {
    for (let i = 0; i < size; i++) out[i] = Math.imul(src[i] as number, si);
  }
  return Tensor.fromTypedArray({ data: out, shape: t.shape, dtype: outDtype, device: t.device });
}

/**
 * Add a scalar value to all elements of a tensor.
 *
 * Float tensors keep their dtype. Integer tensors keep theirs for integer
 * scalars (wrapping on overflow); a fractional scalar gives a `float32` result,
 * as in PyTorch. `bool` tensors produce `int32` (or `float32`).
 *
 * @param t - Input tensor
 * @param s - Scalar value to add
 * @returns New tensor with scalar added to all elements
 * @throws {DTypeError} For `string` tensors
 *
 * @example
 * ```ts
 * import { addScalar, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([1, 2, 3]);
 * const result = addScalar(x, 10);  // [11, 12, 13]
 * ```
 */
export function addScalar(t: Tensor, s: number): Tensor {
  return roundHalfResult(addScalarCore(t, s));
}

function addScalarCore(t: Tensor, s: number): Tensor {
  return scalarArithmetic(t, s, "add", "addScalar");
}

/**
 * Multiply all elements of a tensor by a scalar value.
 *
 * Dtype rules match {@link addScalar}.
 *
 * @param t - Input tensor
 * @param s - Scalar multiplier
 * @returns New tensor with all elements multiplied by scalar
 * @throws {DTypeError} For `string` tensors
 *
 * @example
 * ```ts
 * import { mulScalar, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([1, 2, 3]);
 * const result = mulScalar(x, 10);  // [10, 20, 30]
 * ```
 */
export function mulScalar(t: Tensor, s: number): Tensor {
  return roundHalfResult(mulScalarCore(t, s));
}

function mulScalarCore(t: Tensor, s: number): Tensor {
  return scalarArithmetic(t, s, "mul", "mulScalar");
}

/**
 * Element-wise floor division.
 *
 * Computes the largest integer less than or equal to the quotient, the same
 * as NumPy's `floor_divide` (Python `//`): `floorDiv(1, 0.1)` is 9 because
 * 0.1 is slightly larger than one tenth. Integer division by zero yields 0.
 * The operands are promoted to a common dtype.
 *
 * @param a - Dividend
 * @param b - Divisor
 * @returns Tensor of the promoted dtype
 * @throws {DTypeError} If either tensor has `string` dtype
 * @throws {ShapeError} If the shapes do not broadcast
 *
 * @example
 * ```ts
 * import { floorDiv, tensor } from 'deepbox/ndarray';
 *
 * const a = tensor([7, 8, 9]);
 * const b = tensor([3, 3, 3]);
 * const result = floorDiv(a, b);  // [2, 2, 3]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function floorDiv(a: Tensor, b: Tensor): Tensor {
  return roundHalfResult(floorDivCore(a, b));
}

function floorDivCore(a0: Tensor, b0: Tensor): Tensor {
  const [a, b] = promoteOperands(a0, b0, "floorDiv");
  ensureBroadcastableScalar(a, b);

  const outShape = getOutShape(a, b);
  if (a.dtype === "int64") return mapBinaryBigInt(a, b, outShape, floorDivBigInt);
  const kernel =
    a.dtype === "float64" ? floorDivFloat : isFloatDType(a.dtype) ? floorDivFloat32 : floorDivInt;
  return mapBinaryNumbers(a, b, outShape, a.dtype, kernel);
}

/**
 * Element-wise modulo (remainder).
 *
 * The result takes the sign of the divisor, as in NumPy's `mod` and Python's
 * `%`, and satisfies `a == floorDiv(a, b) * b + mod(a, b)`. Integer `x % 0`
 * yields 0; float `x % 0` yields NaN. The operands are promoted to a common
 * dtype.
 *
 * @param a - Dividend
 * @param b - Divisor
 * @returns Tensor of the promoted dtype
 * @throws {DTypeError} If either tensor has `string` dtype
 * @throws {ShapeError} If the shapes do not broadcast
 *
 * @example
 * ```ts
 * import { mod, tensor } from 'deepbox/ndarray';
 *
 * const a = tensor([7, -8, 9]);
 * const b = tensor([3, 3, 3]);
 * const result = mod(a, b);  // [1, 1, 0]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function mod(a: Tensor, b: Tensor): Tensor {
  return roundHalfResult(modCore(a, b));
}

function modCore(a0: Tensor, b0: Tensor): Tensor {
  const [a, b] = promoteOperands(a0, b0, "mod");
  ensureBroadcastableScalar(a, b);

  const outShape = getOutShape(a, b);
  if (a.dtype === "int64") return mapBinaryBigInt(a, b, outShape, modBigInt);
  return mapBinaryNumbers(a, b, outShape, a.dtype, isFloatDType(a.dtype) ? modFloat : modInt);
}

/**
 * IEEE-754 / NumPy-compatible pow. JS `**` returns NaN for 1 ** Infinity and
 * (-1) ** ±Infinity, where IEEE 754 (and NumPy) define the result as 1.
 */
function powIEEE(base: number, exp: number): number {
  if (base === 1) return 1; // pow(1, anything) = 1, including NaN/Infinity
  if (base === -1 && !Number.isFinite(exp) && !Number.isNaN(exp)) return 1;
  return base ** exp;
}

/**
 * Integer power by squaring with 32-bit wrap-around (exact modulo 2^32, which
 * is what NumPy's int32 `power` returns). The exponent must be >= 0.
 */
function powInt32(base: number, exp: number): number {
  let result = 1;
  let b = base | 0;
  let e = exp;
  while (e > 0) {
    if (e & 1) result = Math.imul(result, b);
    e = Math.floor(e / 2);
    if (e > 0) b = Math.imul(b, b);
  }
  return result;
}

/** Integer power by squaring with 64-bit wrap-around; the exponent must be >= 0. */
function powBigInt(base: bigint, exp: bigint): bigint {
  let result = 1n;
  let b = BigInt.asIntN(64, base);
  let e = exp;
  while (e > 0n) {
    if (e & 1n) result = BigInt.asIntN(64, result * b);
    e >>= 1n;
    if (e > 0n) b = BigInt.asIntN(64, b * b);
  }
  return result;
}

/**
 * Float power with one scalar exponent, specialised for the common exponents.
 * `0.5` uses `Math.sqrt`, which (like NumPy's scalar-power shortcut) returns -0
 * for -0 and NaN for negative inputs.
 */
function powFloatScalarExponent(
  src: NumericTypedArray,
  out: Float32Array | Float64Array,
  e: number
): void {
  const size = src.length;
  if (e === 2) {
    for (let i = 0; i < size; i++) {
      const v = src[i] as number;
      out[i] = v * v;
    }
  } else if (e === 1) {
    for (let i = 0; i < size; i++) out[i] = src[i] as number;
  } else if (e === 0.5) {
    for (let i = 0; i < size; i++) out[i] = Math.sqrt(src[i] as number);
  } else if (e === -1) {
    for (let i = 0; i < size; i++) out[i] = 1 / (src[i] as number);
  } else if (Number.isFinite(e)) {
    // JS ** matches IEEE 754 pow for finite exponents; the powIEEE special
    // cases only differ for infinite/NaN exponents.
    for (let i = 0; i < size; i++) out[i] = (src[i] as number) ** e;
  } else {
    for (let i = 0; i < size; i++) out[i] = powIEEE(src[i] as number, e);
  }
}

/**
 * Element-wise power.
 *
 * Raises elements of the first tensor to powers from the second tensor, with
 * IEEE-754 special cases (`pow(1, NaN)` is 1, `pow(x, 0)` is 1). The operands
 * are promoted to a common dtype. Integer powers are computed exactly and wrap
 * on overflow; if any int32/int64 exponent is negative the whole result is
 * `float32` (NumPy and PyTorch would raise an error there).
 *
 * @param a - Base
 * @param b - Exponent
 * @returns Tensor of the promoted dtype (`float32` for negative integer exponents)
 * @throws {DTypeError} If either tensor has `string` dtype
 * @throws {ShapeError} If the shapes do not broadcast
 *
 * @example
 * ```ts
 * import { pow, tensor } from 'deepbox/ndarray';
 *
 * const a = tensor([2, 3, 4]);
 * const b = tensor([2, 3, 2]);
 * const result = pow(a, b);  // [4, 27, 16]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function pow(a: Tensor, b: Tensor): Tensor {
  return roundHalfResult(powCore(a, b));
}

function powCore(a0: Tensor, b0: Tensor): Tensor {
  const [a, b] = promoteOperands(a0, b0, "pow");
  ensureBroadcastableScalar(a, b);

  if (a.device !== "cpu" || b.device !== "cpu") {
    const onDevice = dispatchBinary("pow", a, b);
    if (onDevice) return onDevice;
  }

  // Fast path: float base with a scalar exponent: tight loops specialized
  // for the common exponents instead of a per-element broadcast closure.
  if ((a.dtype === "float32" || a.dtype === "float64") && isScalar(b) && a.size > 0) {
    const src = readNumericContiguous(a);
    const expArr = readNumericContiguous(b);
    if (src && expArr) {
      const Ctor = dtypeToTypedArrayCtor(a.dtype);
      const out = new Ctor(src.length) as Float32Array | Float64Array;
      powFloatScalarExponent(src, out, expArr[0] as number);
      return Tensor.fromTypedArray({
        data: out,
        shape: a.shape,
        dtype: a.dtype,
        device: a.device,
      });
    }
  }

  const outShape = getOutShape(a, b);

  if (a.dtype === "int64") {
    if (!hasNegativeExponent(b)) return mapBinaryBigInt(a, b, outShape, powBigInt);
    return mapBinaryNumbers(
      asNumberTensor(a, "pow"),
      asNumberTensor(b, "pow"),
      outShape,
      "float32",
      powIEEE
    );
  }
  if (a.dtype === "int32" && hasNegativeExponent(b)) {
    return mapBinaryNumbers(a, b, outShape, "float32", powIEEE);
  }
  if (a.dtype === "int32" || a.dtype === "uint8" || a.dtype === "bool") {
    return mapBinaryNumbers(a, b, outShape, a.dtype, powInt32);
  }
  return mapBinaryNumbers(a, b, outShape, a.dtype, powIEEE);
}

/**
 * Element-wise negation.
 *
 * Returns -x for each element. `uint8` values wrap modulo 256. `bool` tensors
 * are rejected (NumPy does too); use `logicalNot` instead.
 *
 * @param t - Input tensor
 * @returns Tensor of the same dtype and shape
 * @throws {DTypeError} For `bool` and `string` tensors
 *
 * @example
 * ```ts
 * import { neg, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([1, -2, 3]);
 * const result = neg(x);  // [-1, 2, -3]
 * ```
 */
export function neg(t: Tensor): Tensor {
  ensureNumericDType(t, "neg");

  if (t.device !== "cpu") {
    const onDevice = dispatchUnary("neg", t);
    if (onDevice) return onDevice;
  }

  if (t.dtype === "bool") {
    throw new DTypeError(
      "neg is not supported for bool tensors; use logicalNot or cast to a numeric dtype"
    );
  }

  const Ctor = dtypeToTypedArrayCtor(t.dtype);
  const out = new Ctor(t.size);

  if (isBigIntArray(out) && isBigIntArray(t.data)) {
    const logicalStrides = computeStrides(t.shape);
    const contiguous = isContiguous(t.shape, t.strides);
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = -getBigIntElement(t.data, srcOffset);
    }
  } else if (!isBigIntArray(out)) {
    const src = readNumbers(t, "neg");
    for (let i = 0; i < t.size; i++) {
      out[i] = -(src[i] as number);
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: t.dtype,
    device: t.device,
  });
}

/**
 * Element-wise absolute value.
 *
 * Returns |x| for each element. The most negative int32/int64 value has no
 * positive counterpart and wraps to itself, as in NumPy.
 *
 * @param t - Input tensor
 * @returns Tensor of the same dtype and shape
 * @throws {DTypeError} For `string` tensors
 *
 * @example
 * ```ts
 * import { abs, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([-1, 2, -3]);
 * const result = abs(x);  // [1, 2, 3]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function abs(t: Tensor): Tensor {
  ensureNumericDType(t, "abs");

  if (t.device !== "cpu") {
    const onDevice = dispatchUnary("abs", t);
    if (onDevice) return onDevice;
  }

  const Ctor = dtypeToTypedArrayCtor(t.dtype);
  const out = new Ctor(t.size);

  if (isBigIntArray(out) && isBigIntArray(t.data)) {
    const logicalStrides = computeStrides(t.shape);
    const contiguous = isContiguous(t.shape, t.strides);
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = getBigIntElement(t.data, srcOffset);
      out[i] = val < 0n ? -val : val;
    }
  } else if (!isBigIntArray(out)) {
    const src = readNumbers(t, "abs");
    for (let i = 0; i < t.size; i++) {
      out[i] = Math.abs(src[i] as number);
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: t.dtype,
    device: t.device,
  });
}

/**
 * Element-wise sign function.
 *
 * Returns -1, 0, or 1 depending on the sign of each element. Zeros (including
 * -0) map to 0 and NaN stays NaN.
 *
 * @param t - Input tensor
 * @returns Tensor of the same dtype and shape
 * @throws {DTypeError} For `string` tensors
 *
 * @example
 * ```ts
 * import { sign, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([-5, 0, 3]);
 * const result = sign(x);  // [-1, 0, 1]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function sign(t: Tensor): Tensor {
  ensureNumericDType(t, "sign");

  if (t.device !== "cpu") {
    const onDevice = dispatchUnary("sign", t);
    if (onDevice) return onDevice;
  }

  const Ctor = dtypeToTypedArrayCtor(t.dtype);
  const out = new Ctor(t.size);

  if (isBigIntArray(out) && isBigIntArray(t.data)) {
    const logicalStrides = computeStrides(t.shape);
    const contiguous = isContiguous(t.shape, t.strides);
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = getBigIntElement(t.data, srcOffset);
      out[i] = val < 0n ? -1n : val > 0n ? 1n : 0n;
    }
  } else if (!isBigIntArray(out)) {
    const src = readNumbers(t, "sign");
    for (let i = 0; i < t.size; i++) {
      const v = src[i] as number;
      out[i] = v > 0 ? 1 : v < 0 ? -1 : v === 0 ? 0 : v;
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: t.dtype,
    device: t.device,
  });
}

/**
 * Element-wise reciprocal.
 *
 * Returns 1/x for each element. Float dtypes are kept; integer and bool inputs
 * produce `float32` (NumPy would truncate to integers); `1/0` is `Infinity`.
 *
 * @param t - Input tensor
 * @returns Tensor of the float dtype matching the input
 * @throws {DTypeError} For `string` tensors
 *
 * @example
 * ```ts
 * import { reciprocal, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([2, 4, 8]);
 * const result = reciprocal(x);  // [0.5, 0.25, 0.125]
 * ```
 */
export function reciprocal(t: Tensor): Tensor {
  return roundHalfResult(reciprocalCore(t));
}

function reciprocalCore(t: Tensor): Tensor {
  ensureNumericDType(t, "reciprocal");

  if (t.device !== "cpu") {
    const onDevice = dispatchUnary("reciprocal", t);
    if (onDevice) return onDevice;
  }

  const outDtype = outDtypeForTrueDiv(t.dtype);
  const Ctor = dtypeToTypedArrayCtor(outDtype);
  const out = new Ctor(t.size) as NumberTypedArray;
  const src = readNumbers(t, "reciprocal");
  for (let i = 0; i < t.size; i++) {
    out[i] = 1 / (src[i] as number);
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: outDtype,
    device: t.device,
  });
}

/**
 * Element-wise maximum of two tensors.
 *
 * NaN propagates: if either element is NaN the result is NaN, as in NumPy's
 * `maximum` (not `fmax`).
 *
 * @param a - First tensor
 * @param b - Second tensor
 * @returns Tensor of the promoted dtype with the broadcast shape
 * @throws {DTypeError} If either tensor has `string` dtype
 * @throws {ShapeError} If the shapes do not broadcast
 *
 * @example
 * ```ts
 * import { maximum, tensor } from 'deepbox/ndarray';
 *
 * const a = tensor([1, 5, 3]);
 * const b = tensor([4, 2, 6]);
 * const result = maximum(a, b);  // [4, 5, 6]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function maximum(a0: Tensor, b0: Tensor): Tensor {
  const [a, b] = promoteOperands(a0, b0, "maximum");
  ensureBroadcastableScalar(a, b);

  if (a.device !== "cpu" || b.device !== "cpu") {
    const onDevice = dispatchBinary("maximum", a, b);
    if (onDevice) return onDevice;
  }

  // Fast path: contiguous same-shape numeric tensors (mirrors add/sub/mul/div).
  // Math.max propagates NaN like NumPy's np.maximum.
  const fp = fastPathArrays(a, b);
  if (fp) {
    const Ctor = dtypeToTypedArrayCtor(a.dtype);
    const fpOut = new Ctor(fp.size);
    const { aArr, bArr, size } = fp;
    for (let i = 0; i < size; i++) {
      fpOut[i] = Math.max(aArr[i] as number, bArr[i] as number);
    }
    return Tensor.fromTypedArray({
      data: fpOut,
      shape: a.shape,
      dtype: a.dtype,
      device: a.device,
    });
  }

  const outShape = getOutShape(a, b);
  if (a.dtype === "int64") return mapBinaryBigInt(a, b, outShape, (x, y) => (x > y ? x : y));
  return mapBinaryNumbers(a, b, outShape, a.dtype, Math.max);
}

/**
 * Element-wise minimum of two tensors.
 *
 * NaN propagates: if either element is NaN the result is NaN (NumPy's
 * `minimum`).
 *
 * @param a - First tensor
 * @param b - Second tensor
 * @returns Tensor of the promoted dtype with the broadcast shape
 * @throws {DTypeError} If either tensor has `string` dtype
 * @throws {ShapeError} If the shapes do not broadcast
 *
 * @example
 * ```ts
 * import { minimum, tensor } from 'deepbox/ndarray';
 *
 * const a = tensor([1, 5, 3]);
 * const b = tensor([4, 2, 6]);
 * const result = minimum(a, b);  // [1, 2, 3]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function minimum(a0: Tensor, b0: Tensor): Tensor {
  const [a, b] = promoteOperands(a0, b0, "minimum");
  ensureBroadcastableScalar(a, b);

  if (a.device !== "cpu" || b.device !== "cpu") {
    const onDevice = dispatchBinary("minimum", a, b);
    if (onDevice) return onDevice;
  }

  // Fast path: contiguous same-shape numeric tensors (mirrors add/sub/mul/div).
  // Math.min propagates NaN like NumPy's np.minimum.
  const fp = fastPathArrays(a, b);
  if (fp) {
    const Ctor = dtypeToTypedArrayCtor(a.dtype);
    const fpOut = new Ctor(fp.size);
    const { aArr, bArr, size } = fp;
    for (let i = 0; i < size; i++) {
      fpOut[i] = Math.min(aArr[i] as number, bArr[i] as number);
    }
    return Tensor.fromTypedArray({
      data: fpOut,
      shape: a.shape,
      dtype: a.dtype,
      device: a.device,
    });
  }

  const outShape = getOutShape(a, b);
  if (a.dtype === "int64") return mapBinaryBigInt(a, b, outShape, (x, y) => (x < y ? x : y));
  return mapBinaryNumbers(a, b, outShape, a.dtype, Math.min);
}

const TWO_POW_63 = 2 ** 63;

/** Whether `x` is an integer that the integer dtype can hold. */
function boundFitsIntegerDType(dtype: DType, x: number): boolean {
  if (!Number.isInteger(x)) return false;
  switch (dtype) {
    case "int32":
      return x >= -2147483648 && x <= 2147483647;
    case "uint8":
      return x >= 0 && x <= 255;
    case "bool":
      return x >= 0 && x <= 1;
    case "int64":
      return x >= -TWO_POW_63 && x < TWO_POW_63;
    default:
      return false;
  }
}

/**
 * Clip (limit) values in tensor.
 *
 * Given an interval, values outside the interval are clipped to interval
 * edges. Either bound may be omitted. NaN elements stay NaN, and a NaN bound
 * turns every element into NaN (NumPy semantics).
 *
 * Integer tensors keep their dtype when the bounds are integers inside the
 * dtype's range. An integer bound outside the range widens the result (to
 * `int32` for `uint8` and `bool` tensors when the bounds fit it, otherwise to
 * `int64`); a fractional or NaN bound gives `float32` (a number scalar never
 * upcasts a tensor to `float64`). Infinite bounds are treated as no bound.
 *
 * @param t - Input tensor
 * @param min - Lower bound. If undefined, no lower clipping
 * @param max - Upper bound. If undefined, no upper clipping
 * @returns New tensor of the same shape
 * @throws {InvalidParameterError} If `min > max`
 * @throws {DTypeError} For `string` tensors
 *
 * @example
 * ```ts
 * import { clip, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([1, 2, 3, 4, 5]);
 * const result = clip(x, 2, 4);  // [2, 2, 3, 4, 4]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function clip(t: Tensor, min?: number, max?: number): Tensor {
  return roundHalfResult(clipCore(t, min, max));
}

function clipCore(t: Tensor, min?: number, max?: number): Tensor {
  ensureNumericDType(t, "clip");
  if (min !== undefined && max !== undefined && min > max) {
    throw new InvalidParameterError(`clip: min (${min}) must be <= max (${max})`, "min/max", {
      min,
      max,
    });
  }
  // An infinite bound on the open side never changes a value.
  const lo = min === Number.NEGATIVE_INFINITY ? undefined : min;
  const hi = max === Number.POSITIVE_INFINITY ? undefined : max;

  if (t.isDeviceTensor) {
    // Device tensors clip through the maximum/minimum kernels with 0-d
    // scalar operands (uploaded by the dispatcher).
    const scalarLike = (v: number): Tensor =>
      Tensor.fromTypedArray({
        data: new Float32Array([v]),
        shape: [],
        dtype: t.dtype as "float32",
        device: "cpu",
      });
    let r: Tensor = t;
    if (lo !== undefined) r = maximum(r, scalarLike(lo));
    if (hi !== undefined) r = minimum(r, scalarLike(hi));
    if (r === t) {
      const copied = dispatchUnary("copy", t);
      if (copied) return copied;
    }
    return r;
  }

  const size = t.size;

  if (isFloatDType(t.dtype)) {
    const src = readNumbers(t, "clip");
    const Ctor = dtypeToTypedArrayCtor(t.dtype);
    const out = new Ctor(size) as Float32Array | Float64Array;
    // Branchless Math.max/Math.min chain (V8 emits float min/max ops, no
    // data-dependent branches); NaN passes through both.
    const l = lo ?? Number.NEGATIVE_INFINITY;
    const h = hi ?? Number.POSITIVE_INFINITY;
    for (let i = 0; i < size; i++) {
      out[i] = Math.min(Math.max(src[i] as number, l), h);
    }
    return Tensor.fromTypedArray({ data: out, shape: t.shape, dtype: t.dtype, device: t.device });
  }

  const fits =
    (lo === undefined || boundFitsIntegerDType(t.dtype, lo)) &&
    (hi === undefined || boundFitsIntegerDType(t.dtype, hi));

  if (!fits) {
    const src = readNumbers(t, "clip", false);
    const l = lo ?? Number.NEGATIVE_INFINITY;
    const h = hi ?? Number.POSITIVE_INFINITY;
    // An integer bound beyond the dtype's range widens the result: to int32 when the bounds fit
    // it (uint8 and bool tensors), otherwise to int64.
    if (
      t.dtype !== "int64" &&
      (lo === undefined || boundFitsIntegerDType("int64", lo)) &&
      (hi === undefined || boundFitsIntegerDType("int64", hi))
    ) {
      const toInt32 =
        t.dtype !== "int32" &&
        (lo === undefined || boundFitsIntegerDType("int32", lo)) &&
        (hi === undefined || boundFitsIntegerDType("int32", hi));
      if (toInt32) {
        const narrow = new Int32Array(size);
        for (let i = 0; i < size; i++) {
          narrow[i] = Math.min(Math.max(src[i] as number, l), h);
        }
        return Tensor.fromTypedArray({
          data: narrow,
          shape: t.shape,
          dtype: "int32",
          device: t.device,
        });
      }
      const wide = new BigInt64Array(size);
      const minVal = lo !== undefined ? BigInt(lo) : undefined;
      const maxVal = hi !== undefined ? BigInt(hi) : undefined;
      for (let i = 0; i < size; i++) {
        let val = BigInt(src[i] as number);
        if (minVal !== undefined && val < minVal) val = minVal;
        if (maxVal !== undefined && val > maxVal) val = maxVal;
        wide[i] = val;
      }
      return Tensor.fromTypedArray({
        data: wide,
        shape: t.shape,
        dtype: "int64",
        device: t.device,
      });
    }
    // A fractional or NaN bound gives float32: a number scalar never upcasts to float64.
    const out = new Float32Array(size);
    const lf = Math.fround(l);
    const hf = Math.fround(h);
    for (let i = 0; i < size; i++) {
      out[i] = Math.min(Math.max(Math.fround(src[i] as number), lf), hf);
    }
    return Tensor.fromTypedArray({ data: out, shape: t.shape, dtype: "float32", device: t.device });
  }

  if (t.dtype === "int64") {
    const data = t.data;
    if (!(data instanceof BigInt64Array)) {
      throw new DTypeError("clip: int64 tensor without BigInt storage");
    }
    const minVal = lo !== undefined ? BigInt(lo) : undefined;
    const maxVal = hi !== undefined ? BigInt(hi) : undefined;
    const out = new BigInt64Array(size);
    const logicalStrides = computeStrides(t.shape);
    const contiguous = isContiguous(t.shape, t.strides);
    for (let i = 0; i < size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      let val = getBigIntElement(data, srcOffset);
      if (minVal !== undefined && val < minVal) val = minVal;
      if (maxVal !== undefined && val > maxVal) val = maxVal;
      out[i] = val;
    }
    return Tensor.fromTypedArray({ data: out, shape: t.shape, dtype: t.dtype, device: t.device });
  }

  // int32 / uint8 / bool: bounds are integers inside the dtype's range, so
  // every clipped value stays representable.
  const src = readNumbers(t, "clip");
  const Ctor = dtypeToTypedArrayCtor(t.dtype);
  const out = new Ctor(size) as Int32Array | Uint8Array;
  const l = lo ?? Number.NEGATIVE_INFINITY;
  const h = hi ?? Number.POSITIVE_INFINITY;
  for (let i = 0; i < size; i++) {
    out[i] = Math.min(Math.max(src[i] as number, l), h);
  }
  return Tensor.fromTypedArray({ data: out, shape: t.shape, dtype: t.dtype, device: t.device });
}
