import {
  DeepboxError,
  DTypeError,
  getBigIntElement,
  getNumericElement,
  type Shape,
} from "../../core";
import { isContiguous } from "../tensor/strides";
import { computeStrides, isBigIntArray, Tensor } from "../tensor/Tensor";
import { flatOffset, readAsNumber, readElement, requireNumericData } from "./_internal";
import {
  broadcastApply,
  ensureBroadcastableScalar,
  getBroadcastShape,
  isScalar,
} from "./broadcast";

function compareMixed(
  a: bigint | number,
  b: bigint | number,
  op: "eq" | "neq" | "gt" | "ge" | "lt" | "le"
): boolean {
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
  } else {
    // A=num, B=big
    // num > big <-> floorVal >= big
    // num < big <-> floorVal < big
    if (op === "gt" || op === "ge") return floorVal >= big;
    return floorVal < big;
  }
}

function runComparison(a: Tensor, b: Tensor, op: "eq" | "neq" | "gt" | "ge" | "lt" | "le"): Tensor {
  if (a.dtype === "string" || b.dtype === "string") {
    throw new DTypeError(`${op} for string dtype is not implemented`);
  }

  ensureBroadcastableScalar(a, b);

  const aData = requireNumericData(a.data, op);
  const bData = requireNumericData(b.data, op);

  const aIsScalar = isScalar(a);
  const bIsScalar = isScalar(b);
  const outShape: Shape = aIsScalar
    ? b.shape
    : bIsScalar
      ? a.shape
      : getBroadcastShape(a.shape, b.shape);
  const outSize = outShape.reduce((acc, dim) => acc * dim, 1);

  const out = new Uint8Array(outSize);
  const result = Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "bool",
    device: a.device,
  });

  // compareMixed handles mixed types (BigInt/Number).
  // We rely on broadcastApply to handle iteration.
  // Note: aData and bData are TypedArrays. Accessing by index returns number or bigint.

  broadcastApply(a, b, result, (offA, offB, offOut) => {
    // Direct access. TS doesn't know exact type but compareMixed accepts bigint|number.
    const valA = readElement(aData, offA);
    const valB = readElement(bData, offB);
    out[offOut] = compareMixed(valA, valB, op) ? 1 : 0;
  });

  return result;
}

/**
 * Element-wise equality.
 *
 * Output dtype:
 * - `bool`
 */
export function equal(a: Tensor, b: Tensor): Tensor {
  return runComparison(a, b, "eq");
}

/**
 * Element-wise inequality (not equal) comparison.
 *
 * Returns a boolean tensor where each element is true (1) if the
 * corresponding elements in a and b are not equal, false (0) otherwise.
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @returns Boolean tensor of same shape as inputs
 */
export function notEqual(a: Tensor, b: Tensor): Tensor {
  return runComparison(a, b, "neq");
}

/**
 * Element-wise greater than comparison (a > b).
 *
 * Returns a boolean tensor where each element is true (1) if the
 * corresponding element in a is greater than the element in b.
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @returns Boolean tensor with comparison results
 */
export function greater(a: Tensor, b: Tensor): Tensor {
  return runComparison(a, b, "gt");
}

/**
 * Element-wise greater than or equal comparison (a >= b).
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @returns Boolean tensor with comparison results
 */
export function greaterEqual(a: Tensor, b: Tensor): Tensor {
  return runComparison(a, b, "ge");
}

/**
 * Element-wise less than comparison (a < b).
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @returns Boolean tensor with comparison results
 */
export function less(a: Tensor, b: Tensor): Tensor {
  return runComparison(a, b, "lt");
}

/**
 * Element-wise less than or equal comparison (a <= b).
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @returns Boolean tensor with comparison results
 */
export function lessEqual(a: Tensor, b: Tensor): Tensor {
  return runComparison(a, b, "le");
}

/**
 * Element-wise test for approximate equality within tolerance.
 *
 * Returns true where: |a - b| <= (atol + rtol * |b|)
 *
 * Useful for floating-point comparisons where exact equality is unreliable.
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @param rtol - Relative tolerance (default: 1e-5)
 * @param atol - Absolute tolerance (default: 1e-8)
 * @returns Boolean tensor with closeness test results
 */
export function isclose(a: Tensor, b: Tensor, rtol: number = 1e-5, atol: number = 1e-8): Tensor {
  if (a.dtype === "string" || b.dtype === "string") {
    throw new DTypeError("isclose for string dtype is not implemented");
  }

  ensureBroadcastableScalar(a, b);

  const aIsScalar = isScalar(a);
  const bIsScalar = isScalar(b);
  const outShape: Shape = aIsScalar
    ? b.shape
    : bIsScalar
      ? a.shape
      : getBroadcastShape(a.shape, b.shape);
  const outSize = outShape.reduce((acc, dim) => acc * dim, 1);

  const out = new Uint8Array(outSize);
  const result = Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "bool",
    device: a.device,
  });

  // Only handle numeric types (not BigInt for floating-point comparison)
  const aData = requireNumericData(a.data, "isclose");
  const bData = requireNumericData(b.data, "isclose");

  if (isBigIntArray(aData) || isBigIntArray(bData)) {
    throw new DTypeError("isclose not supported for BigInt tensors");
  }

  broadcastApply(a, b, result, (offA, offB, offOut) => {
    // We know it's numeric
    const ax = readAsNumber(aData, offA);
    const bx = readAsNumber(bData, offB);

    // Check if within tolerance: |a - b| <= atol + rtol * |b|
    const diff = Math.abs(ax - bx);
    const threshold = atol + rtol * Math.abs(bx);
    out[offOut] = diff <= threshold ? 1 : 0;
  });

  return result;
}

/**
 * Test whether all corresponding elements are close within tolerance.
 *
 * Returns a single boolean (not a tensor) indicating if ALL elements pass
 * the closeness test.
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @param rtol - Relative tolerance (default: 1e-5)
 * @param atol - Absolute tolerance (default: 1e-8)
 * @returns True if all elements are close, false otherwise
 */
export function allclose(a: Tensor, b: Tensor, rtol: number = 1e-5, atol: number = 1e-8): boolean {
  if (a.dtype === "string" || b.dtype === "string") {
    throw new DTypeError("allclose is not defined for string dtype");
  }

  const aData = requireNumericData(a.data, "allclose");
  const bData = requireNumericData(b.data, "allclose");

  if (isBigIntArray(aData) || isBigIntArray(bData)) {
    throw new DTypeError("allclose not supported for BigInt tensors");
  }

  const aIsScalar = isScalar(a);
  const bIsScalar = isScalar(b);
  let outShape: Shape;
  if (aIsScalar) {
    outShape = b.shape;
  } else if (bIsScalar) {
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

  // Handle scalar case efficiently
  if (outShape.length === 0) {
    const ax = readAsNumber(aData, a.offset);
    const bx = readAsNumber(bData, b.offset);
    const diff = Math.abs(ax - bx);
    const threshold = atol + rtol * Math.abs(bx);
    return diff <= threshold;
  }

  // Setup broadcast strides
  const rank = outShape.length;
  const stridesA = new Array<number>(rank).fill(0);
  const stridesB = new Array<number>(rank).fill(0);

  const rankDiffA = rank - a.ndim;
  for (let i = 0; i < rank; i++) {
    if (i >= rankDiffA) {
      const dim = a.shape[i - rankDiffA] ?? 1;
      if (dim > 1) {
        stridesA[i] = a.strides[i - rankDiffA] ?? 0;
      }
    }
  }

  const rankDiffB = rank - b.ndim;
  for (let i = 0; i < rank; i++) {
    if (i >= rankDiffB) {
      const dim = b.shape[i - rankDiffB] ?? 1;
      if (dim > 1) {
        stridesB[i] = b.strides[i - rankDiffB] ?? 0;
      }
    }
  }

  // Iterate
  const idx = new Array<number>(rank).fill(0);
  let offA = a.offset;
  let offB = b.offset;

  while (true) {
    // Check element
    // Direct access as number
    const ax = readAsNumber(aData, offA);
    const bx = readAsNumber(bData, offB);

    const diff = Math.abs(ax - bx);
    const threshold = atol + rtol * Math.abs(bx);
    if (diff > threshold) {
      return false;
    }

    // Odometer increment
    let axis = rank - 1;
    for (;;) {
      const currentIdx = idx[axis] ?? 0;
      const nextIdx = currentIdx + 1;
      idx[axis] = nextIdx;

      const strideA = stridesA[axis];
      const strideB = stridesB[axis];
      const dim = outShape[axis];

      if (strideA === undefined || strideB === undefined || dim === undefined) {
        // Should not happen
        return true;
      }

      offA += strideA;
      offB += strideB;

      if (nextIdx < dim) {
        break;
      }

      // Carry
      const count = nextIdx;
      offA -= strideA * count;
      offB -= strideB * count;
      idx[axis] = 0;
      axis--;

      if (axis < 0) return true; // All passed
    }
  }
}

/**
 * Test for exact array equality (shape, dtype, and all values).
 *
 * Returns a single boolean indicating if tensors are identical.
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @returns True if arrays are exactly equal, false otherwise
 */
export function arrayEqual(a: Tensor, b: Tensor): boolean {
  // Check shape match
  if (a.ndim !== b.ndim || a.size !== b.size) {
    return false;
  }

  for (let i = 0; i < a.ndim; i++) {
    if (a.shape[i] !== b.shape[i]) {
      return false;
    }
  }

  // Check dtype match
  if (a.dtype !== b.dtype) {
    return false;
  }

  if (a.dtype === "string") {
    if (Array.isArray(a.data) && Array.isArray(b.data)) {
      const aLogicalStrides = computeStrides(a.shape);
      const bLogicalStrides = computeStrides(b.shape);
      const aContiguous = isContiguous(a.shape, a.strides);
      const bContiguous = isContiguous(b.shape, b.strides);
      for (let i = 0; i < a.size; i++) {
        const aOffset = flatOffset(i, a.offset, aContiguous, aLogicalStrides, a.strides);
        const bOffset = flatOffset(i, b.offset, bContiguous, bLogicalStrides, b.strides);
        if (a.data[aOffset] !== b.data[bOffset]) {
          return false;
        }
      }
    }
    return true;
  }

  const aData = requireNumericData(a.data, "arrayEqual");
  const bData = requireNumericData(b.data, "arrayEqual");
  const aLogicalStrides = computeStrides(a.shape);
  const bLogicalStrides = computeStrides(b.shape);
  const aContiguous = isContiguous(a.shape, a.strides);
  const bContiguous = isContiguous(b.shape, b.strides);

  // Check all values
  if (isBigIntArray(aData) && isBigIntArray(bData)) {
    for (let i = 0; i < a.size; i++) {
      const aOffset = flatOffset(i, a.offset, aContiguous, aLogicalStrides, a.strides);
      const bOffset = flatOffset(i, b.offset, bContiguous, bLogicalStrides, b.strides);
      if (getBigIntElement(aData, aOffset) !== getBigIntElement(bData, bOffset)) {
        return false;
      }
    }
  } else if (!isBigIntArray(aData) && !isBigIntArray(bData)) {
    for (let i = 0; i < a.size; i++) {
      const aOffset = flatOffset(i, a.offset, aContiguous, aLogicalStrides, a.strides);
      const bOffset = flatOffset(i, b.offset, bContiguous, bLogicalStrides, b.strides);
      if (getNumericElement(aData, aOffset) !== getNumericElement(bData, bOffset)) {
        return false;
      }
    }
  }

  return true;
}

/**
 * Element-wise test for NaN (Not a Number) values.
 *
 * Returns true (1) for NaN elements, false (0) otherwise.
 *
 * @param t - Input tensor
 * @returns Boolean tensor with NaN test results
 */
export function isnan(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("isnan for string dtype is not supported");
  }

  const out = new Uint8Array(t.size);

  const data = requireNumericData(t.data, "isnan");
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  // BigInt cannot be NaN
  if (isBigIntArray(data)) {
    // All zeros for BigInt
    out.fill(0);
  } else {
    // Check each element for NaN
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = getNumericElement(data, srcOffset);
      out[i] = Number.isNaN(val) ? 1 : 0;
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
 * Element-wise test for infinity (+Inf or -Inf).
 *
 * Returns true (1) for infinite elements, false (0) otherwise.
 * Note: NaN is NOT considered infinite.
 *
 * @param t - Input tensor
 * @returns Boolean tensor with infinity test results
 */
export function isinf(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("isinf for string dtype is not supported");
  }

  const out = new Uint8Array(t.size);

  const data = requireNumericData(t.data, "isinf");
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  // BigInt cannot be infinite
  if (isBigIntArray(data)) {
    out.fill(0);
  } else {
    // Check each element for infinity (not NaN)
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = getNumericElement(data, srcOffset);
      // Infinite if not finite and not NaN
      out[i] = !Number.isFinite(val) && !Number.isNaN(val) ? 1 : 0;
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
 * Element-wise test for finite values (not NaN, not Inf).
 *
 * Returns true (1) for finite elements, false (0) for NaN or Inf.
 *
 * @param t - Input tensor
 * @returns Boolean tensor with finite test results
 */
export function isfinite(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("isfinite for string dtype is not supported");
  }

  const out = new Uint8Array(t.size);

  const data = requireNumericData(t.data, "isfinite");
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  // BigInt is always finite
  if (isBigIntArray(data)) {
    out.fill(1);
  } else {
    // Check each element for finiteness
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = getNumericElement(data, srcOffset);
      out[i] = Number.isFinite(val) ? 1 : 0;
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: "bool",
    device: t.device,
  });
}
