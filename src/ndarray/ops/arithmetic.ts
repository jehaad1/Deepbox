import {
  type DType,
  DTypeError,
  dtypeToTypedArrayCtor,
  getBigIntElement,
  getNumericElement,
  InvalidParameterError,
  type Shape,
} from "../../core";
import type { NumericTypedArray } from "../../core/utils/typedArrayAccess";
import { isContiguous } from "../tensor/strides";
import { computeStrides, isBigIntArray, Tensor } from "../tensor/Tensor";
import { flatOffset } from "./_internal";
import {
  broadcastApply,
  ensureBroadcastableScalar,
  ensureNumericDType,
  ensureSameDType,
  getBroadcastShape,
  isScalar,
} from "./broadcast";

/**
 * Check if two tensors are eligible for the contiguous fast path:
 * same shape, both contiguous, both at offset 0, both non-BigInt numeric.
 * Returns the raw numeric typed arrays if eligible, or null otherwise.
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
  // Contiguous with zero offset
  if (a.offset !== 0 || b.offset !== 0) return null;
  if (!isContiguous(a.shape, a.strides) || !isContiguous(b.shape, b.strides)) return null;
  return { aArr: aData, bArr: bData, size: a.size };
}

function outDtypeForTrueDiv(dtype: Exclude<DType, "string">): Exclude<DType, "string"> {
  if (dtype === "float32" || dtype === "float64") return dtype;
  return "float64";
}

/**
 * Floor division for bigint values (rounds toward -Infinity).
 *
 * JavaScript bigint division truncates toward zero. Deepbox floor division
 * rounds toward -Infinity, so negative values with a remainder need
 * an additional decrement.
 */
function floorDivBigInt(a: bigint, b: bigint): bigint {
  const q = a / b;
  const r = a % b;
  const rPositive = r > 0n;
  const bPositive = b > 0n;
  if (r !== 0n && rPositive !== bPositive) {
    return q - 1n;
  }
  return q;
}

function floorDivNumber(a: number, b: number): number {
  return Math.floor(a / b);
}

/**
 * Modulo: remainder has the same sign as divisor.
 */
function modNumber(a: number, b: number): number {
  return a - floorDivNumber(a, b) * b;
}

function modBigInt(a: bigint, b: bigint): bigint {
  return a - floorDivBigInt(a, b) * b;
}

function hasNegativeExponent(t: Tensor): boolean {
  if (t.size === 0) return false;
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);
  if (t.data instanceof BigInt64Array) {
    for (let i = 0; i < t.size; i++) {
      const offset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      if (getBigIntElement(t.data, offset) < 0n) return true;
    }
    return false;
  }
  const numericData = t.data;
  if (Array.isArray(numericData)) {
    // String tensor, effectively no negative numeric exponents
    return false;
  }
  for (let i = 0; i < t.size; i++) {
    const offset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
    if (getNumericElement(numericData, offset) < 0) return true;
  }
  return false;
}

/**
 * Element-wise addition.
 *
 * Supported:
 * - Standard broadcasting semantics (including scalar and dimension-1 broadcast)
 */
export function add(a: Tensor, b: Tensor): Tensor {
  ensureNumericDType(a, "add");
  ensureNumericDType(b, "add");
  ensureSameDType(a, b);
  ensureBroadcastableScalar(a, b);

  // Fast path: contiguous same-shape numeric tensors
  const fp = fastPathArrays(a, b);
  if (fp) {
    const Ctor = dtypeToTypedArrayCtor(a.dtype);
    const out = new Ctor(fp.size);
    const aArr = fp.aArr;
    const bArr = fp.bArr;
    for (let i = 0; i < fp.size; i++) {
      out[i] = (aArr[i] as number) + (bArr[i] as number);
    }
    return Tensor.fromTypedArray({
      data: out,
      shape: a.shape,
      dtype: a.dtype,
      device: a.device,
    });
  }

  // Handle broadcasting
  const aIsScalar = isScalar(a);
  const bIsScalar = isScalar(b);
  const outShape: Shape = aIsScalar
    ? b.shape
    : bIsScalar
      ? a.shape
      : getBroadcastShape(a.shape, b.shape);
  const outSize = outShape.reduce((acc, dim) => acc * dim, 1);

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
    broadcastApply(a, b, result, (offA, offB, offOut) => {
      out[offOut] = getNumericElement(aData, offA) + getNumericElement(bData, offB);
    });
  }

  return result;
}

function getOutShape(a: Tensor, b: Tensor): Shape {
  if (isScalar(a)) return b.shape;
  if (isScalar(b)) return a.shape;
  return getBroadcastShape(a.shape, b.shape);
}

/**
 * Element-wise subtraction.
 *
 * Computes a - b element by element.
 *
 * Broadcasting: supports standard Deepbox-style broadcasting.
 *
 * @param a - First tensor
 * @param b - Second tensor
 * @returns Tensor containing element-wise difference
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
  ensureNumericDType(a, "sub");
  ensureNumericDType(b, "sub");
  ensureSameDType(a, b);
  ensureBroadcastableScalar(a, b);

  // Fast path: contiguous same-shape numeric tensors
  const fp = fastPathArrays(a, b);
  if (fp) {
    const Ctor = dtypeToTypedArrayCtor(a.dtype);
    const out = new Ctor(fp.size);
    const aArr = fp.aArr;
    const bArr = fp.bArr;
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

  const outShape = getOutShape(a, b);
  const outSize = outShape.reduce((acc, dim) => acc * dim, 1);

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
 * Computes a * b element by element.
 *
 * Broadcasting: supports standard Deepbox-style broadcasting.
 *
 * @param a - First tensor
 * @param b - Second tensor
 * @returns Tensor containing element-wise product
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
  ensureNumericDType(a, "mul");
  ensureNumericDType(b, "mul");
  ensureSameDType(a, b);
  ensureBroadcastableScalar(a, b);

  // Fast path: contiguous same-shape numeric tensors
  const fp = fastPathArrays(a, b);
  if (fp) {
    const Ctor = dtypeToTypedArrayCtor(a.dtype);
    const out = new Ctor(fp.size);
    const aArr = fp.aArr;
    const bArr = fp.bArr;
    for (let i = 0; i < fp.size; i++) {
      out[i] = (aArr[i] as number) * (bArr[i] as number);
    }
    return Tensor.fromTypedArray({
      data: out,
      shape: a.shape,
      dtype: a.dtype,
      device: a.device,
    });
  }

  const outShape = getOutShape(a, b);
  const outSize = outShape.reduce((acc, dim) => acc * dim, 1);

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
    broadcastApply(a, b, result, (offA, offB, offOut) => {
      out[offOut] = getNumericElement(aData, offA) * getNumericElement(bData, offB);
    });
  }

  return result;
}

/**
 * Element-wise division.
 *
 * Computes a / b element by element.
 *
 * Broadcasting: supports standard Deepbox-style broadcasting.
 *
 * @param a - Numerator tensor
 * @param b - Denominator tensor
 * @returns Tensor containing element-wise quotient
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
  ensureNumericDType(a, "div");
  ensureNumericDType(b, "div");
  ensureSameDType(a, b);
  ensureBroadcastableScalar(a, b);

  // Fast path: contiguous same-shape float tensors (no dtype promotion needed)
  const outDtype = outDtypeForTrueDiv(a.dtype);
  const fp = fastPathArrays(a, b);
  if (fp && outDtype === a.dtype) {
    const Ctor = dtypeToTypedArrayCtor(outDtype);
    const out = new Ctor(fp.size);
    const aArr = fp.aArr;
    const bArr = fp.bArr;
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

  const outShape = getOutShape(a, b);
  const outSize = outShape.reduce((acc, dim) => acc * dim, 1);

  const Ctor = dtypeToTypedArrayCtor(outDtype);
  const out = new Ctor(outSize);
  const result = Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: outDtype,
    device: a.device,
  });

  if (isBigIntArray(out) && isBigIntArray(a.data) && isBigIntArray(b.data)) {
    const aData = a.data;
    const bData = b.data;
    broadcastApply(a, b, result, (offA, offB, offOut) => {
      out[offOut] = getBigIntElement(aData, offA) / getBigIntElement(bData, offB);
    });
  } else if (!isBigIntArray(out) && isBigIntArray(a.data) && isBigIntArray(b.data)) {
    const aData = a.data;
    const bData = b.data;
    broadcastApply(a, b, result, (offA, offB, offOut) => {
      out[offOut] = Number(getBigIntElement(aData, offA)) / Number(getBigIntElement(bData, offB));
    });
  } else if (!isBigIntArray(out) && !isBigIntArray(a.data) && !isBigIntArray(b.data)) {
    const aData = a.data;
    const bData = b.data;
    broadcastApply(a, b, result, (offA, offB, offOut) => {
      out[offOut] = getNumericElement(aData, offA) / getNumericElement(bData, offB);
    });
  }

  return result;
}

/**
 * Add a scalar value to all elements of a tensor.
 *
 * @param t - Input tensor
 * @param s - Scalar value to add
 * @returns New tensor with scalar added to all elements
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
  ensureNumericDType(t, "addScalar");
  const Ctor = dtypeToTypedArrayCtor(t.dtype);
  const out = new Ctor(t.size);
  const contiguous = isContiguous(t.shape, t.strides);

  if (isBigIntArray(out) && isBigIntArray(t.data)) {
    const scalar = BigInt(Math.trunc(s));
    const logicalStrides = computeStrides(t.shape);
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = getBigIntElement(t.data, srcOffset) + scalar;
    }
  } else if (!isBigIntArray(out) && !isBigIntArray(t.data)) {
    const numData = t.data;
    if (!Array.isArray(numData)) {
      // Fast path: contiguous zero-offset
      if (contiguous && t.offset === 0) {
        for (let i = 0; i < t.size; i++) {
          out[i] = (numData[i] as number) + s;
        }
      } else {
        const logicalStrides = computeStrides(t.shape);
        for (let i = 0; i < t.size; i++) {
          const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
          out[i] = getNumericElement(numData, srcOffset) + s;
        }
      }
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
 * Multiply all elements of a tensor by a scalar value.
 *
 * @param t - Input tensor
 * @param s - Scalar multiplier
 * @returns New tensor with all elements multiplied by scalar
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
  ensureNumericDType(t, "mulScalar");
  const Ctor = dtypeToTypedArrayCtor(t.dtype);
  const out = new Ctor(t.size);
  const contiguous = isContiguous(t.shape, t.strides);

  if (isBigIntArray(out) && isBigIntArray(t.data)) {
    const scalar = BigInt(Math.trunc(s));
    const logicalStrides = computeStrides(t.shape);
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = getBigIntElement(t.data, srcOffset) * scalar;
    }
  } else if (!isBigIntArray(out) && !isBigIntArray(t.data)) {
    const numData = t.data;
    if (!Array.isArray(numData)) {
      // Fast path: contiguous zero-offset
      if (contiguous && t.offset === 0) {
        for (let i = 0; i < t.size; i++) {
          out[i] = (numData[i] as number) * s;
        }
      } else {
        const logicalStrides = computeStrides(t.shape);
        for (let i = 0; i < t.size; i++) {
          const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
          out[i] = getNumericElement(numData, srcOffset) * s;
        }
      }
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
 * Element-wise floor division.
 *
 * Computes the largest integer less than or equal to the quotient.
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
  ensureNumericDType(a, "floorDiv");
  ensureNumericDType(b, "floorDiv");
  ensureSameDType(a, b);
  ensureBroadcastableScalar(a, b);

  const outShape = getOutShape(a, b);
  const outSize = outShape.reduce((acc, dim) => acc * dim, 1);

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
      out[offOut] = floorDivBigInt(getBigIntElement(aData, offA), getBigIntElement(bData, offB));
    });
  } else if (!isBigIntArray(out) && !isBigIntArray(a.data) && !isBigIntArray(b.data)) {
    const aData = a.data;
    const bData = b.data;
    broadcastApply(a, b, result, (offA, offB, offOut) => {
      out[offOut] = floorDivNumber(getNumericElement(aData, offA), getNumericElement(bData, offB));
    });
  }

  return result;
}

/**
 * Element-wise modulo operation.
 *
 * Returns remainder of division.
 *
 * @example
 * ```ts
 * import { mod, tensor } from 'deepbox/ndarray';
 *
 * const a = tensor([7, 8, 9]);
 * const b = tensor([3, 3, 3]);
 * const result = mod(a, b);  // [1, 2, 0]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function mod(a: Tensor, b: Tensor): Tensor {
  ensureNumericDType(a, "mod");
  ensureNumericDType(b, "mod");
  ensureSameDType(a, b);
  ensureBroadcastableScalar(a, b);

  const outShape = getOutShape(a, b);
  const outSize = outShape.reduce((acc, dim) => acc * dim, 1);

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
      out[offOut] = modBigInt(getBigIntElement(aData, offA), getBigIntElement(bData, offB));
    });
  } else if (!isBigIntArray(out) && !isBigIntArray(a.data) && !isBigIntArray(b.data)) {
    const aData = a.data;
    const bData = b.data;
    broadcastApply(a, b, result, (offA, offB, offOut) => {
      out[offOut] = modNumber(getNumericElement(aData, offA), getNumericElement(bData, offB));
    });
  }

  return result;
}

/**
 * Element-wise power.
 *
 * Raises elements of first tensor to powers from second tensor.
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
  ensureNumericDType(a, "pow");
  ensureNumericDType(b, "pow");
  ensureSameDType(a, b);
  ensureBroadcastableScalar(a, b);

  const outShape = getOutShape(a, b);
  const outSize = outShape.reduce((acc, dim) => acc * dim, 1);

  const promoteToFloat = (a.dtype === "int32" || a.dtype === "int64") && hasNegativeExponent(b);
  const outDtype = promoteToFloat ? "float64" : a.dtype;
  const Ctor = dtypeToTypedArrayCtor(outDtype);
  const out = new Ctor(outSize);
  const result = Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: outDtype,
    device: a.device,
  });

  if (outDtype === "float64" || outDtype === "float32") {
    const aData = a.data;
    const bData = b.data;

    // Use specific branches for best performance to avoid repeated checks
    if (isBigIntArray(aData)) {
      if (isBigIntArray(bData)) {
        broadcastApply(a, b, result, (offA, offB, offOut) => {
          out[offOut] =
            Number(getBigIntElement(aData, offA)) ** Number(getBigIntElement(bData, offB));
        });
      } else {
        broadcastApply(a, b, result, (offA, offB, offOut) => {
          out[offOut] = Number(getBigIntElement(aData, offA)) ** getNumericElement(bData, offB);
        });
      }
    } else {
      if (isBigIntArray(bData)) {
        broadcastApply(a, b, result, (offA, offB, offOut) => {
          out[offOut] = getNumericElement(aData, offA) ** Number(getBigIntElement(bData, offB));
        });
      } else {
        broadcastApply(a, b, result, (offA, offB, offOut) => {
          out[offOut] = getNumericElement(aData, offA) ** getNumericElement(bData, offB);
        });
      }
    }
  } else if (isBigIntArray(out) && isBigIntArray(a.data) && isBigIntArray(b.data)) {
    const aData = a.data;
    const bData = b.data;
    broadcastApply(a, b, result, (offA, offB, offOut) => {
      out[offOut] = getBigIntElement(aData, offA) ** getBigIntElement(bData, offB);
    });
  } else if (!isBigIntArray(out) && !isBigIntArray(a.data) && !isBigIntArray(b.data)) {
    const aData = a.data;
    const bData = b.data;
    broadcastApply(a, b, result, (offA, offB, offOut) => {
      out[offOut] = getNumericElement(aData, offA) ** getNumericElement(bData, offB);
    });
  }

  return result;
}

/**
 * Element-wise negation.
 *
 * Returns -x for each element.
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
  const Ctor = dtypeToTypedArrayCtor(t.dtype);
  const out = new Ctor(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  if (isBigIntArray(out) && isBigIntArray(t.data)) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = -getBigIntElement(t.data, srcOffset);
    }
  } else if (!isBigIntArray(out) && !isBigIntArray(t.data)) {
    const numData = t.data;
    if (!Array.isArray(numData) && contiguous && t.offset === 0) {
      for (let i = 0; i < t.size; i++) {
        out[i] = -(numData[i] as number);
      }
    } else {
      for (let i = 0; i < t.size; i++) {
        const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
        out[i] = -getNumericElement(t.data, srcOffset);
      }
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
 * Returns |x| for each element.
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
  const Ctor = dtypeToTypedArrayCtor(t.dtype);
  const out = new Ctor(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  if (isBigIntArray(out) && isBigIntArray(t.data)) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = getBigIntElement(t.data, srcOffset);
      out[i] = val < 0n ? -val : val;
    }
  } else if (!isBigIntArray(out) && !isBigIntArray(t.data)) {
    const numData = t.data;
    if (!Array.isArray(numData)) {
      // Fast path: contiguous zero-offset — direct TypedArray loop
      if (contiguous && t.offset === 0) {
        for (let i = 0; i < t.size; i++) {
          out[i] = Math.abs(numData[i] as number);
        }
      } else {
        for (let i = 0; i < t.size; i++) {
          const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
          out[i] = Math.abs(getNumericElement(numData, srcOffset));
        }
      }
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
 * Returns -1, 0, or 1 depending on sign of each element.
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
  const Ctor = dtypeToTypedArrayCtor(t.dtype);
  const out = new Ctor(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  if (isBigIntArray(out) && isBigIntArray(t.data)) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = getBigIntElement(t.data, srcOffset);
      out[i] = val < 0n ? -1n : val > 0n ? 1n : 0n;
    }
  } else if (!isBigIntArray(out) && !isBigIntArray(t.data)) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = Math.sign(getNumericElement(t.data, srcOffset));
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
 * Returns 1/x for each element.
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
  ensureNumericDType(t, "reciprocal");
  const outDtype = outDtypeForTrueDiv(t.dtype);
  const Ctor = dtypeToTypedArrayCtor(outDtype);
  const out = new Ctor(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  if (isBigIntArray(out) && isBigIntArray(t.data)) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = getBigIntElement(t.data, srcOffset);
      out[i] = val === 0n ? 0n : 1n / val;
    }
  } else if (!isBigIntArray(out) && isBigIntArray(t.data)) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = 1 / Number(getBigIntElement(t.data, srcOffset));
    }
  } else if (!isBigIntArray(out) && !isBigIntArray(t.data)) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = 1 / getNumericElement(t.data, srcOffset);
    }
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
export function maximum(a: Tensor, b: Tensor): Tensor {
  ensureNumericDType(a, "maximum");
  ensureNumericDType(b, "maximum");
  ensureSameDType(a, b);
  ensureBroadcastableScalar(a, b);

  const outShape = getOutShape(a, b);
  const outSize = outShape.reduce((acc, dim) => acc * dim, 1);

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
      const ax = getBigIntElement(aData, offA);
      const bx = getBigIntElement(bData, offB);
      out[offOut] = ax > bx ? ax : bx;
    });
  } else if (!isBigIntArray(out) && !isBigIntArray(a.data) && !isBigIntArray(b.data)) {
    const aData = a.data;
    const bData = b.data;
    broadcastApply(a, b, result, (offA, offB, offOut) => {
      out[offOut] = Math.max(getNumericElement(aData, offA), getNumericElement(bData, offB));
    });
  }

  return result;
}

/**
 * Element-wise minimum of two tensors.
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
export function minimum(a: Tensor, b: Tensor): Tensor {
  ensureNumericDType(a, "minimum");
  ensureNumericDType(b, "minimum");
  ensureSameDType(a, b);
  ensureBroadcastableScalar(a, b);

  const outShape = getOutShape(a, b);
  const outSize = outShape.reduce((acc, dim) => acc * dim, 1);

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
      const ax = getBigIntElement(aData, offA);
      const bx = getBigIntElement(bData, offB);
      out[offOut] = ax < bx ? ax : bx;
    });
  } else if (!isBigIntArray(out) && !isBigIntArray(a.data) && !isBigIntArray(b.data)) {
    const aData = a.data;
    const bData = b.data;
    if (Array.isArray(aData) || Array.isArray(bData)) {
      throw new DTypeError("String tensors not supported in minimum");
    }
    broadcastApply(a, b, result, (offA, offB, offOut) => {
      out[offOut] = Math.min(getNumericElement(aData, offA), getNumericElement(bData, offB));
    });
  }

  return result;
}

/**
 * Clip (limit) values in tensor.
 *
 * Given an interval, values outside the interval are clipped to interval edges.
 *
 * **Parameters**:
 * @param t - Input tensor
 * @param min - Minimum value. If undefined, no lower clipping
 * @param max - Maximum value. If undefined, no upper clipping
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
  ensureNumericDType(t, "clip");
  if (min !== undefined && max !== undefined && min > max) {
    throw new InvalidParameterError(`clip: min (${min}) must be <= max (${max})`, "min/max", {
      min,
      max,
    });
  }

  const Ctor = dtypeToTypedArrayCtor(t.dtype);
  const out = new Ctor(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  if (isBigIntArray(out) && isBigIntArray(t.data)) {
    const minVal = min !== undefined ? BigInt(Math.trunc(min)) : undefined;
    const maxVal = max !== undefined ? BigInt(Math.trunc(max)) : undefined;

    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      let val = getBigIntElement(t.data, srcOffset);
      if (minVal !== undefined && val < minVal) val = minVal;
      if (maxVal !== undefined && val > maxVal) val = maxVal;
      out[i] = val;
    }
  } else if (!isBigIntArray(out) && !isBigIntArray(t.data)) {
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      let val = getNumericElement(t.data, srcOffset);
      if (min !== undefined && val < min) val = min;
      if (max !== undefined && val > max) val = max;
      out[i] = val;
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: t.dtype,
    device: t.device,
  });
}
