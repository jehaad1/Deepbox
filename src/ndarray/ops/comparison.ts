/**
 * Element-wise comparison and closeness tests: equal, notEqual, greater,
 * greaterEqual, less, lessEqual, isclose, allclose, arrayEqual, isnan, isinf
 * and isfinite.
 *
 * Comparisons broadcast like arithmetic ops and accept operands of different
 * numeric dtypes without converting them first: values are compared as numbers
 * (an int64 tensor is compared with a float tensor exactly, without rounding the
 * integers to doubles). The result is always `bool`.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import {
  DeepboxError,
  DTypeError,
  getBigIntElement,
  InvalidParameterError,
  type Shape,
} from "../../core";
import type { NumericTypedArray } from "../../core/utils/typed_array_access";
import { isContiguous } from "../tensor/strides";
import { computeStrides, isBigIntArray, Tensor } from "../tensor/Tensor";
import {
  flatOffset,
  readAsNumber,
  readElement,
  readNumericContiguous,
  requireNumericData,
} from "./_internal";
import {
  broadcastApply,
  ensureBroadcastableScalar,
  getBroadcastShape,
  isScalar,
} from "./broadcast";

type CompareOp = "eq" | "neq" | "gt" | "ge" | "lt" | "le";

/**
 * Compare a bigint with a number (or two values of the same kind) exactly,
 * following IEEE semantics for NaN and infinities: NaN is unequal to
 * everything, and no integer is larger than +Infinity.
 */
function compareMixed(a: bigint | number, b: bigint | number, op: CompareOp): boolean {
  if (typeof a === "bigint" && typeof b === "bigint") {
    switch (op) {
      case "eq":
        return a === b;
      case "neq":
        return a !== b;
      case "gt":
        return a > b;
      case "ge":
        return a >= b;
      case "lt":
        return a < b;
      case "le":
        return a <= b;
    }
  }

  if (typeof a === "number" && typeof b === "number") {
    switch (op) {
      case "eq":
        return a === b;
      case "neq":
        return a !== b;
      case "gt":
        return a > b;
      case "ge":
        return a >= b;
      case "lt":
        return a < b;
      case "le":
        return a <= b;
    }
  }

  // Mixed
  let big: bigint;
  let num: number;
  let bigIsA: boolean;

  if (typeof a === "bigint") {
    big = a;
    if (typeof b !== "number") throw new DeepboxError("Internal error: expected number");
    num = b;
    bigIsA = true;
  } else {
    num = a;
    if (typeof b !== "bigint") throw new DeepboxError("Internal error: expected bigint");
    big = b;
    bigIsA = false;
  }

  if (Number.isNaN(num)) return op === "neq";
  if (num === Infinity) {
    if (op === "eq") return false;
    if (op === "neq") return true;
    // big < Inf -> True
    if (bigIsA) return op === "lt" || op === "le";
    return op === "gt" || op === "ge";
  }
  if (num === -Infinity) {
    if (op === "eq") return false;
    if (op === "neq") return true;
    // big > -Inf -> True
    if (bigIsA) return op === "gt" || op === "ge";
    return op === "lt" || op === "le";
  }

  if (Number.isInteger(num)) {
    const numBig = BigInt(num);
    const A = bigIsA ? big : numBig;
    const B = bigIsA ? numBig : big;
    switch (op) {
      case "eq":
        return A === B;
      case "neq":
        return A !== B;
      case "gt":
        return A > B;
      case "ge":
        return A >= B;
      case "lt":
        return A < B;
      case "le":
        return A <= B;
    }
  }

  // Non-integer number
  if (op === "eq") return false;
  if (op === "neq") return true;

  const floorVal = BigInt(Math.floor(num));
  // big > num <-> big > floorVal
  // big < num <-> big <= floorVal

  if (bigIsA) {
    // A=big, B=num
    if (op === "gt" || op === "ge") return big > floorVal;
    return big <= floorVal;
  }
  // A=num, B=big
  // num > big <-> floorVal >= big
  // num < big <-> floorVal < big
  if (op === "gt" || op === "ge") return floorVal >= big;
  return floorVal < big;
}

function broadcastOutShape(a: Tensor, b: Tensor): Shape {
  if (isScalar(a)) return b.shape;
  if (isScalar(b)) return a.shape;
  return getBroadcastShape(a.shape, b.shape);
}

function sameShape(a: Tensor, b: Tensor): boolean {
  if (a.ndim !== b.ndim) return false;
  for (let d = 0; d < a.ndim; d++) {
    if (a.shape[d] !== b.shape[d]) return false;
  }
  return true;
}

function runComparison(a: Tensor, b: Tensor, op: CompareOp): Tensor {
  if (a.dtype === "string" || b.dtype === "string") {
    throw new DTypeError(`${op} for string dtype is not implemented`);
  }

  ensureBroadcastableScalar(a, b);

  const aData = requireNumericData(a.data, op);
  const bData = requireNumericData(b.data, op);

  const outShape = broadcastOutShape(a, b);
  const outSize = outShape.reduce((acc, dim) => acc * dim, 1);

  const out = new Uint8Array(outSize);
  const result = Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "bool",
    device: a.device,
  });
  if (outSize === 0) return result;

  // BigInt operands (int64) need exact mixed bigint/number comparison.
  if (aData instanceof BigInt64Array || bData instanceof BigInt64Array) {
    broadcastApply(a, b, result, (offA, offB, offOut) => {
      out[offOut] = compareMixed(readElement(aData, offA), readElement(bData, offB), op) ? 1 : 0;
    });
    return result;
  }

  // Same-shape operands (contiguous or strided) compare in tight per-op loops;
  // the generic closure path costs ~15x on small tensors.
  if (sameShape(a, b)) {
    const aArr = readNumericContiguous(a) as NumericTypedArray;
    const bArr = readNumericContiguous(b) as NumericTypedArray;
    const n = outSize;
    switch (op) {
      case "eq":
        for (let i = 0; i < n; i++) out[i] = (aArr[i] as number) === (bArr[i] as number) ? 1 : 0;
        break;
      case "neq":
        for (let i = 0; i < n; i++) out[i] = (aArr[i] as number) !== (bArr[i] as number) ? 1 : 0;
        break;
      case "gt":
        for (let i = 0; i < n; i++) out[i] = (aArr[i] as number) > (bArr[i] as number) ? 1 : 0;
        break;
      case "ge":
        for (let i = 0; i < n; i++) out[i] = (aArr[i] as number) >= (bArr[i] as number) ? 1 : 0;
        break;
      case "lt":
        for (let i = 0; i < n; i++) out[i] = (aArr[i] as number) < (bArr[i] as number) ? 1 : 0;
        break;
      case "le":
        for (let i = 0; i < n; i++) out[i] = (aArr[i] as number) <= (bArr[i] as number) ? 1 : 0;
        break;
    }
    return result;
  }

  // Broadcasting: the op switch sits outside the closure so each callback is
  // a single comparison on raw buffer offsets.
  switch (op) {
    case "eq":
      broadcastApply(a, b, result, (oa, ob, oo) => {
        out[oo] = (aData[oa] as number) === (bData[ob] as number) ? 1 : 0;
      });
      break;
    case "neq":
      broadcastApply(a, b, result, (oa, ob, oo) => {
        out[oo] = (aData[oa] as number) !== (bData[ob] as number) ? 1 : 0;
      });
      break;
    case "gt":
      broadcastApply(a, b, result, (oa, ob, oo) => {
        out[oo] = (aData[oa] as number) > (bData[ob] as number) ? 1 : 0;
      });
      break;
    case "ge":
      broadcastApply(a, b, result, (oa, ob, oo) => {
        out[oo] = (aData[oa] as number) >= (bData[ob] as number) ? 1 : 0;
      });
      break;
    case "lt":
      broadcastApply(a, b, result, (oa, ob, oo) => {
        out[oo] = (aData[oa] as number) < (bData[ob] as number) ? 1 : 0;
      });
      break;
    case "le":
      broadcastApply(a, b, result, (oa, ob, oo) => {
        out[oo] = (aData[oa] as number) <= (bData[ob] as number) ? 1 : 0;
      });
      break;
  }

  return result;
}

/**
 * Element-wise equality.
 *
 * NaN is not equal to anything, including itself. int64 tensors compare
 * exactly against any numeric dtype.
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @returns `bool` tensor with the broadcast shape
 * @throws {DTypeError} For `string` tensors
 * @throws {ShapeError} If the shapes do not broadcast
 */
export function equal(a: Tensor, b: Tensor): Tensor {
  return runComparison(a, b, "eq");
}

/**
 * Element-wise inequality (not equal) comparison.
 *
 * Returns a boolean tensor where each element is true (1) if the
 * corresponding elements in a and b are not equal, false (0) otherwise.
 * NaN is unequal to everything, including itself.
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @returns `bool` tensor with the broadcast shape
 * @throws {DTypeError} For `string` tensors
 * @throws {ShapeError} If the shapes do not broadcast
 */
export function notEqual(a: Tensor, b: Tensor): Tensor {
  return runComparison(a, b, "neq");
}

/**
 * Element-wise greater than comparison (a > b).
 *
 * Returns a boolean tensor where each element is true (1) if the
 * corresponding element in a is greater than the element in b. Comparisons
 * with NaN are false.
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @returns `bool` tensor with the broadcast shape
 * @throws {DTypeError} For `string` tensors
 * @throws {ShapeError} If the shapes do not broadcast
 */
export function greater(a: Tensor, b: Tensor): Tensor {
  return runComparison(a, b, "gt");
}

/**
 * Element-wise greater than or equal comparison (a >= b).
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @returns `bool` tensor with the broadcast shape
 * @throws {DTypeError} For `string` tensors
 * @throws {ShapeError} If the shapes do not broadcast
 */
export function greaterEqual(a: Tensor, b: Tensor): Tensor {
  return runComparison(a, b, "ge");
}

/**
 * Element-wise less than comparison (a < b).
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @returns `bool` tensor with the broadcast shape
 * @throws {DTypeError} For `string` tensors
 * @throws {ShapeError} If the shapes do not broadcast
 */
export function less(a: Tensor, b: Tensor): Tensor {
  return runComparison(a, b, "lt");
}

/**
 * Element-wise less than or equal comparison (a <= b).
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @returns `bool` tensor with the broadcast shape
 * @throws {DTypeError} For `string` tensors
 * @throws {ShapeError} If the shapes do not broadcast
 */
export function lessEqual(a: Tensor, b: Tensor): Tensor {
  return runComparison(a, b, "le");
}

/**
 * NumPy-compatible scalar closeness: NaN is never close to anything (unless
 * `equalNan` is set and both are NaN), and infinities are close only to an
 * infinity of the same sign (the tolerance formula would otherwise yield an
 * infinite threshold or a NaN diff).
 */
function scalarIsClose(
  ax: number,
  bx: number,
  rtol: number,
  atol: number,
  equalNan: boolean
): boolean {
  if (Number.isNaN(ax) || Number.isNaN(bx)) return equalNan && Number.isNaN(ax) && Number.isNaN(bx);
  if (!Number.isFinite(ax) || !Number.isFinite(bx)) return ax === bx;
  return Math.abs(ax - bx) <= atol + rtol * Math.abs(bx);
}

function validateTolerances(rtol: number, atol: number, fn: string): void {
  if (typeof rtol !== "number" || !(rtol >= 0)) {
    throw new InvalidParameterError(
      `${fn}: rtol must be a non-negative number; received ${String(rtol)}`,
      "rtol",
      rtol
    );
  }
  if (typeof atol !== "number" || !(atol >= 0)) {
    throw new InvalidParameterError(
      `${fn}: atol must be a non-negative number; received ${String(atol)}`,
      "atol",
      atol
    );
  }
}

/**
 * Element-wise test for approximate equality within tolerance.
 *
 * Returns true where: |a - b| <= (atol + rtol * |b|)
 *
 * The formula is not symmetric: `rtol` scales with `b`. NaN is never close
 * (unless `equalNan` is true and both are NaN), and an infinity is only close
 * to the same infinity. int64 values are compared after conversion to double.
 *
 * Useful for floating-point comparisons where exact equality is unreliable.
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @param rtol - Relative tolerance (default: 1e-5)
 * @param atol - Absolute tolerance (default: 1e-8)
 * @param equalNan - Treat NaN as close to NaN (default: false)
 * @returns `bool` tensor with the broadcast shape
 * @throws {InvalidParameterError} If `rtol` or `atol` is negative or NaN
 * @throws {DTypeError} For `string` tensors
 * @throws {ShapeError} If the shapes do not broadcast
 */
export function isclose(
  a: Tensor,
  b: Tensor,
  rtol: number = 1e-5,
  atol: number = 1e-8,
  equalNan: boolean = false
): Tensor {
  if (a.dtype === "string" || b.dtype === "string") {
    throw new DTypeError("isclose for string dtype is not implemented");
  }
  validateTolerances(rtol, atol, "isclose");

  ensureBroadcastableScalar(a, b);

  const outShape = broadcastOutShape(a, b);
  const outSize = outShape.reduce((acc, dim) => acc * dim, 1);

  const out = new Uint8Array(outSize);
  const result = Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "bool",
    device: a.device,
  });

  const aData = requireNumericData(a.data, "isclose");
  const bData = requireNumericData(b.data, "isclose");

  if (isBigIntArray(aData) || isBigIntArray(bData)) {
    broadcastApply(a, b, result, (offA, offB, offOut) => {
      const ax = readAsNumber(aData, offA);
      const bx = readAsNumber(bData, offB);
      out[offOut] = scalarIsClose(ax, bx, rtol, atol, equalNan) ? 1 : 0;
    });
    return result;
  }

  broadcastApply(a, b, result, (offA, offB, offOut) => {
    out[offOut] = scalarIsClose(aData[offA] as number, bData[offB] as number, rtol, atol, equalNan)
      ? 1
      : 0;
  });

  return result;
}

/**
 * Test whether all corresponding elements are close within tolerance.
 *
 * Returns a single boolean (not a tensor) indicating if ALL elements pass
 * the closeness test of {@link isclose}. Stops at the first mismatch.
 * Tensors whose shapes cannot broadcast are never close, so the result is
 * `false` instead of an error. Empty broadcast results are vacuously close.
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @param rtol - Relative tolerance (default: 1e-5)
 * @param atol - Absolute tolerance (default: 1e-8)
 * @param equalNan - Treat NaN as close to NaN (default: false)
 * @returns True if all elements are close, false otherwise
 * @throws {InvalidParameterError} If `rtol` or `atol` is negative or NaN
 * @throws {DTypeError} For `string` tensors
 */
export function allclose(
  a: Tensor,
  b: Tensor,
  rtol: number = 1e-5,
  atol: number = 1e-8,
  equalNan: boolean = false
): boolean {
  if (a.dtype === "string" || b.dtype === "string") {
    throw new DTypeError("allclose is not defined for string dtype");
  }
  validateTolerances(rtol, atol, "allclose");

  const aData = requireNumericData(a.data, "allclose");
  const bData = requireNumericData(b.data, "allclose");

  let outShape: Shape;
  if (isScalar(a)) {
    outShape = b.shape;
  } else if (isScalar(b)) {
    outShape = a.shape;
  } else {
    try {
      outShape = getBroadcastShape(a.shape, b.shape);
    } catch {
      return false;
    }
  }

  const outSize = outShape.reduce((acc, dim) => acc * dim, 1);
  if (outSize === 0) {
    return true;
  }

  // Scalar case
  if (outShape.length === 0) {
    return scalarIsClose(
      readAsNumber(aData, a.offset),
      readAsNumber(bData, b.offset),
      rtol,
      atol,
      equalNan
    );
  }

  // Setup broadcast strides
  const rank = outShape.length;
  const stridesA = new Array<number>(rank).fill(0);
  const stridesB = new Array<number>(rank).fill(0);

  const rankDiffA = rank - a.ndim;
  for (let i = rankDiffA; i < rank; i++) {
    if ((a.shape[i - rankDiffA] ?? 1) > 1) {
      stridesA[i] = a.strides[i - rankDiffA] ?? 0;
    }
  }

  const rankDiffB = rank - b.ndim;
  for (let i = rankDiffB; i < rank; i++) {
    if ((b.shape[i - rankDiffB] ?? 1) > 1) {
      stridesB[i] = b.strides[i - rankDiffB] ?? 0;
    }
  }

  // Odometer walk over the broadcast shape
  const idx = new Array<number>(rank).fill(0);
  let offA = a.offset;
  let offB = b.offset;

  for (;;) {
    if (
      !scalarIsClose(readAsNumber(aData, offA), readAsNumber(bData, offB), rtol, atol, equalNan)
    ) {
      return false;
    }

    let axis = rank - 1;
    for (;;) {
      const nextIdx = (idx[axis] ?? 0) + 1;
      idx[axis] = nextIdx;

      const strideA = stridesA[axis] ?? 0;
      const strideB = stridesB[axis] ?? 0;

      offA += strideA;
      offB += strideB;

      if (nextIdx < (outShape[axis] ?? 0)) {
        break;
      }

      // Carry: rewind this axis and move to the next outer one
      offA -= strideA * nextIdx;
      offB -= strideB * nextIdx;
      idx[axis] = 0;
      axis--;

      if (axis < 0) return true; // All passed
    }
  }
}

/**
 * Test for exact array equality (shape, dtype, and all values).
 *
 * Returns a single boolean indicating if tensors are identical. Unlike NumPy's
 * `array_equal`, differing dtypes are never equal. NaN is unequal to itself
 * unless `equalNan` is true.
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @param equalNan - Treat NaN elements at the same position as equal (default: false)
 * @returns True if all elements are equal, false otherwise
 */
export function arrayEqual(a: Tensor, b: Tensor, equalNan: boolean = false): boolean {
  // Check shape match
  if (!sameShape(a, b)) {
    return false;
  }

  // Check dtype match
  if (a.dtype !== b.dtype) {
    return false;
  }

  if (a.dtype === "string") {
    const aStr = a.data;
    const bStr = b.data;
    if (Array.isArray(aStr) && Array.isArray(bStr)) {
      const aLogicalStrides = computeStrides(a.shape);
      const bLogicalStrides = computeStrides(b.shape);
      const aContiguous = isContiguous(a.shape, a.strides);
      const bContiguous = isContiguous(b.shape, b.strides);
      for (let i = 0; i < a.size; i++) {
        const aOffset = flatOffset(i, a.offset, aContiguous, aLogicalStrides, a.strides);
        const bOffset = flatOffset(i, b.offset, bContiguous, bLogicalStrides, b.strides);
        if (aStr[aOffset] !== bStr[bOffset]) {
          return false;
        }
      }
    }
    return true;
  }

  const aData = requireNumericData(a.data, "arrayEqual");
  const bData = requireNumericData(b.data, "arrayEqual");

  if (isBigIntArray(aData) && isBigIntArray(bData)) {
    const aLogicalStrides = computeStrides(a.shape);
    const bLogicalStrides = computeStrides(b.shape);
    const aContiguous = isContiguous(a.shape, a.strides);
    const bContiguous = isContiguous(b.shape, b.strides);
    for (let i = 0; i < a.size; i++) {
      const aOffset = flatOffset(i, a.offset, aContiguous, aLogicalStrides, a.strides);
      const bOffset = flatOffset(i, b.offset, bContiguous, bLogicalStrides, b.strides);
      if (getBigIntElement(aData, aOffset) !== getBigIntElement(bData, bOffset)) {
        return false;
      }
    }
  } else if (!isBigIntArray(aData) && !isBigIntArray(bData)) {
    const aArr = readNumericContiguous(a) as NumericTypedArray;
    const bArr = readNumericContiguous(b) as NumericTypedArray;
    for (let i = 0; i < a.size; i++) {
      const x = aArr[i] as number;
      const y = bArr[i] as number;
      if (x !== y && !(equalNan && Number.isNaN(x) && Number.isNaN(y))) {
        return false;
      }
    }
  }

  return true;
}

/** Whether the dtype can hold NaN or infinity (integer and bool tensors cannot). */
function isFloatingDType(dtype: Tensor["dtype"]): boolean {
  return dtype === "float16" || dtype === "bfloat16" || dtype === "float32" || dtype === "float64";
}

/**
 * Shared implementation of isnan/isinf/isfinite. Integer and bool tensors
 * never hold NaN or infinity, so they are answered without scanning.
 */
function classifyFloats(
  t: Tensor,
  name: string,
  predicate: (v: number) => boolean,
  nonFloatResult: 0 | 1
): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError(`${name} for string dtype is not supported`);
  }

  requireNumericData(t.data, name);
  const out = new Uint8Array(t.size);

  if (!isFloatingDType(t.dtype)) {
    out.fill(nonFloatResult);
  } else {
    const src = readNumericContiguous(t) as NumericTypedArray;
    for (let i = 0; i < t.size; i++) {
      out[i] = predicate(src[i] as number) ? 1 : 0;
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: "bool",
    device: t.device,
  });
}

/**
 * Element-wise test for NaN (Not a Number) values.
 *
 * Returns true (1) for NaN elements, false (0) otherwise. Integer and bool
 * tensors never contain NaN.
 *
 * @param t - Input tensor
 * @returns `bool` tensor with the same shape
 * @throws {DTypeError} For `string` tensors
 */
export function isnan(t: Tensor): Tensor {
  return classifyFloats(t, "isnan", Number.isNaN, 0);
}

/**
 * Element-wise test for infinity (+Inf or -Inf).
 *
 * Returns true (1) for infinite elements, false (0) otherwise.
 * Note: NaN is NOT considered infinite.
 *
 * @param t - Input tensor
 * @returns `bool` tensor with the same shape
 * @throws {DTypeError} For `string` tensors
 */
export function isinf(t: Tensor): Tensor {
  return classifyFloats(t, "isinf", (v) => v === Infinity || v === -Infinity, 0);
}

/**
 * Element-wise test for finite values (not NaN, not Inf).
 *
 * Returns true (1) for finite elements, false (0) for NaN or Inf. Integer and
 * bool tensors are always finite.
 *
 * @param t - Input tensor
 * @returns `bool` tensor with the same shape
 * @throws {DTypeError} For `string` tensors
 */
export function isfinite(t: Tensor): Tensor {
  return classifyFloats(t, "isfinite", Number.isFinite, 1);
}
