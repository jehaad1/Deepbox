import {
  DTypeError,
  getBigIntElement,
  getNumericElement,
  type Shape,
  type TypedArray,
} from "../../core";
import { isContiguous } from "../tensor/strides";
import { computeStrides, isBigIntArray, Tensor } from "../tensor/Tensor";
import { flatOffset } from "./_internal";
import {
  broadcastApply,
  ensureBroadcastableScalar,
  getBroadcastShape,
  isScalar,
} from "./broadcast";

function requireTypedArray(t: Tensor): TypedArray {
  if (t.dtype === "string" || Array.isArray(t.data)) {
    throw new DTypeError("logical operations are not implemented for string dtype");
  }
  return t.data;
}

function isTruthy(data: TypedArray, offset: number): boolean {
  if (isBigIntArray(data)) {
    return getBigIntElement(data, offset) !== 0n;
  }
  return getNumericElement(data, offset) !== 0;
}

/**
 * Element-wise logical AND.
 *
 * Returns true (1) where both inputs are non-zero (truthy), false (0) otherwise.
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @returns Boolean tensor with AND results
 *
 * @example
 * ```ts
 * const a = tensor([1, 0, 1]);
 * const b = tensor([1, 1, 0]);
 * logicalAnd(a, b);  // tensor([1, 0, 0])
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function logicalAnd(a: Tensor, b: Tensor): Tensor {
  if (a.dtype === "string" || b.dtype === "string") {
    throw new DTypeError("logical operations are not implemented for string dtype");
  }
  ensureBroadcastableScalar(a, b);

  const outShape: Shape = isScalar(a)
    ? b.shape
    : isScalar(b)
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

  const aData = requireTypedArray(a);
  const bData = requireTypedArray(b);

  broadcastApply(a, b, result, (offA, offB, offOut) => {
    const ax = isTruthy(aData, offA);
    const bx = isTruthy(bData, offB);
    out[offOut] = ax && bx ? 1 : 0;
  });

  return result;
}

/**
 * Element-wise logical OR.
 *
 * Returns true (1) where at least one input is non-zero (truthy).
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @returns Boolean tensor with OR results
 *
 * @example
 * ```ts
 * const a = tensor([1, 0, 0]);
 * const b = tensor([1, 1, 0]);
 * logicalOr(a, b);  // tensor([1, 1, 0])
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function logicalOr(a: Tensor, b: Tensor): Tensor {
  if (a.dtype === "string" || b.dtype === "string") {
    throw new DTypeError("logical operations are not implemented for string dtype");
  }
  // Ensure tensors are compatible (same size or one is scalar)
  ensureBroadcastableScalar(a, b);

  // Determine output shape (use non-scalar shape)
  const outShape: Shape = isScalar(a)
    ? b.shape
    : isScalar(b)
      ? a.shape
      : getBroadcastShape(a.shape, b.shape);
  const outSize = outShape.reduce((acc, dim) => acc * dim, 1);

  // Create output array for boolean results
  const out = new Uint8Array(outSize);
  const result = Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "bool",
    device: a.device,
  });

  const aData = requireTypedArray(a);
  const bData = requireTypedArray(b);

  // Element-wise OR operation
  broadcastApply(a, b, result, (offA, offB, offOut) => {
    const ax = isTruthy(aData, offA);
    const bx = isTruthy(bData, offB);
    out[offOut] = ax || bx ? 1 : 0;
  });

  return result;
}

/**
 * Element-wise logical XOR (exclusive OR).
 *
 * Returns true (1) where exactly one input is non-zero (true but not both).
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @returns Boolean tensor with XOR results
 *
 * @example
 * ```ts
 * const a = tensor([1, 0, 1, 0]);
 * const b = tensor([1, 1, 0, 0]);
 * logicalXor(a, b);  // tensor([0, 1, 1, 0])
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function logicalXor(a: Tensor, b: Tensor): Tensor {
  if (a.dtype === "string" || b.dtype === "string") {
    throw new DTypeError("logical operations are not implemented for string dtype");
  }
  ensureBroadcastableScalar(a, b);

  const outShape: Shape = isScalar(a)
    ? b.shape
    : isScalar(b)
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

  const aData = requireTypedArray(a);
  const bData = requireTypedArray(b);

  // Element-wise XOR: true if exactly one is true
  broadcastApply(a, b, result, (offA, offB, offOut) => {
    const ax = isTruthy(aData, offA);
    const bx = isTruthy(bData, offB);
    out[offOut] = (ax && !bx) || (!ax && bx) ? 1 : 0;
  });

  return result;
}

/**
 * Element-wise logical NOT.
 *
 * Returns true (1) for zero elements, false (0) for non-zero elements.
 * Inverts the truthiness of each element.
 *
 * @param t - Input tensor
 * @returns Boolean tensor with NOT results
 *
 * @example
 * ```ts
 * const t = tensor([1, 0, 5, 0]);
 * logicalNot(t);  // tensor([0, 1, 0, 1])
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function logicalNot(t: Tensor): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("logicalNot is not implemented for string dtype");
  }

  const out = new Uint8Array(t.size);
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);
  const data = requireTypedArray(t);

  // Element-wise NOT operation
  for (let i = 0; i < t.size; i++) {
    // Invert: 0 becomes 1, non-zero becomes 0
    const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
    out[i] = isTruthy(data, srcOffset) ? 0 : 1;
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: "bool",
    device: t.device,
  });
}
