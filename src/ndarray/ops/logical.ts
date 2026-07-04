import {
  DTypeError,
  getBigIntElement,
  getNumericElement,
  type Shape,
  type TypedArray,
} from "../../core";
import { isContiguous } from "../tensor/strides";
import { computeStrides, isBigIntArray, Tensor } from "../tensor/Tensor";
import { flatOffset, readNumericContiguous } from "./_internal";
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
/**
 * Same-shape contiguous fast path for binary logical ops: tight loops over
 * zero-based views (the generic closure path costs ~15x on small tensors).
 * Returns null when ineligible.
 */
function fastLogicalBinary(
  a: Tensor,
  b: Tensor,
  out: Uint8Array,
  result: Tensor,
  op: "and" | "or" | "xor"
): Tensor | null {
  const aData = a.data;
  const bData = b.data;
  if (Array.isArray(aData) || Array.isArray(bData)) return null;
  if (aData instanceof BigInt64Array || bData instanceof BigInt64Array) return null;
  if (a.ndim !== b.ndim) return null;
  for (let d = 0; d < a.ndim; d++) {
    if (a.shape[d] !== b.shape[d]) return null;
  }
  if (!isContiguous(a.shape, a.strides) || !isContiguous(b.shape, b.strides)) return null;

  const aArr = a.offset === 0 ? aData : aData.subarray(a.offset, a.offset + a.size);
  const bArr = b.offset === 0 ? bData : bData.subarray(b.offset, b.offset + b.size);
  const n = a.size;
  if (op === "and") {
    for (let i = 0; i < n; i++)
      out[i] = (aArr[i] as number) !== 0 && (bArr[i] as number) !== 0 ? 1 : 0;
  } else if (op === "or") {
    for (let i = 0; i < n; i++)
      out[i] = (aArr[i] as number) !== 0 || (bArr[i] as number) !== 0 ? 1 : 0;
  } else {
    for (let i = 0; i < n; i++)
      out[i] = ((aArr[i] as number) !== 0) !== ((bArr[i] as number) !== 0) ? 1 : 0;
  }
  return result;
}

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

  const fast = fastLogicalBinary(a, b, out, result, "and");
  if (fast) return fast;

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

  const fast = fastLogicalBinary(a, b, out, result, "or");
  if (fast) return fast;

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

  const fast = fastLogicalBinary(a, b, out, result, "xor");
  if (fast) return fast;

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
  const data = requireTypedArray(t);

  const src = readNumericContiguous(t);
  if (src) {
    // Invert: 0 becomes 1, non-zero becomes 0
    for (let i = 0; i < t.size; i++) {
      out[i] = (src[i] as number) === 0 ? 1 : 0;
    }
  } else {
    const logicalStrides = computeStrides(t.shape);
    const contiguous = isContiguous(t.shape, t.strides);
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      out[i] = isTruthy(data, srcOffset) ? 0 : 1;
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: "bool",
    device: t.device,
  });
}
