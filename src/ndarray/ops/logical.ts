/**
 * Element-wise logical operations on tensors (`logicalAnd`, `logicalOr`, `logicalXor`,
 * `logicalNot`). Any non-zero value, including NaN, is true; the result is a bool tensor.
 * The two operands of a binary operation may have different numeric dtypes.
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */

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

type LogicalOp = "and" | "or" | "xor";

function applyLogical(op: LogicalOp, x: boolean, y: boolean): number {
  if (op === "and") return x && y ? 1 : 0;
  if (op === "or") return x || y ? 1 : 0;
  return x !== y ? 1 : 0;
}

/**
 * Tight-loop paths for the common cases: both operands contiguous with the same
 * shape, or one of them a 0-D scalar. The generic closure path costs ~15x on small
 * tensors. Returns false when the inputs are not eligible (int64 data, strided views,
 * general broadcasting).
 */
function fastLogicalBinary(a: Tensor, b: Tensor, out: Uint8Array, op: LogicalOp): boolean {
  const aData = a.data;
  const bData = b.data;
  if (Array.isArray(aData) || Array.isArray(bData)) return false;
  if (aData instanceof BigInt64Array || bData instanceof BigInt64Array) return false;

  const aScalar = isScalar(a);
  const bScalar = isScalar(b);
  if (!aScalar && !bScalar) {
    if (a.ndim !== b.ndim) return false;
    for (let d = 0; d < a.ndim; d++) {
      if (a.shape[d] !== b.shape[d]) return false;
    }
  }
  if (!aScalar && !isContiguous(a.shape, a.strides)) return false;
  if (!bScalar && !isContiguous(b.shape, b.strides)) return false;

  const n = out.length;
  const aArr = aScalar ? aData : aData.subarray(a.offset, a.offset + a.size);
  const bArr = bScalar ? bData : bData.subarray(b.offset, b.offset + b.size);
  const aBase = aScalar ? a.offset : 0;
  const bBase = bScalar ? b.offset : 0;
  const aStep = aScalar ? 0 : 1;
  const bStep = bScalar ? 0 : 1;
  if (op === "and") {
    for (let i = 0; i < n; i++) {
      out[i] =
        (aArr[aBase + i * aStep] as number) !== 0 && (bArr[bBase + i * bStep] as number) !== 0
          ? 1
          : 0;
    }
  } else if (op === "or") {
    for (let i = 0; i < n; i++) {
      out[i] =
        (aArr[aBase + i * aStep] as number) !== 0 || (bArr[bBase + i * bStep] as number) !== 0
          ? 1
          : 0;
    }
  } else {
    for (let i = 0; i < n; i++) {
      out[i] =
        ((aArr[aBase + i * aStep] as number) !== 0) !== ((bArr[bBase + i * bStep] as number) !== 0)
          ? 1
          : 0;
    }
  }
  return true;
}

function logicalBinary(a: Tensor, b: Tensor, op: LogicalOp): Tensor {
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

  if (fastLogicalBinary(a, b, out, op)) return result;

  broadcastApply(a, b, result, (offA, offB, offOut) => {
    out[offOut] = applyLogical(op, isTruthy(aData, offA), isTruthy(bData, offB));
  });

  return result;
}

/**
 * Element-wise logical AND.
 *
 * Returns true (1) where both inputs are non-zero (truthy), false (0) otherwise.
 * Inputs broadcast against each other (NumPy rules).
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @returns Boolean tensor with AND results
 * @throws {DTypeError} If an input has string dtype
 * @throws {ShapeError} If the shapes cannot be broadcast
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
  return logicalBinary(a, b, "and");
}

/**
 * Element-wise logical OR.
 *
 * Returns true (1) where at least one input is non-zero (truthy).
 * Inputs broadcast against each other (NumPy rules).
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @returns Boolean tensor with OR results
 * @throws {DTypeError} If an input has string dtype
 * @throws {ShapeError} If the shapes cannot be broadcast
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
  return logicalBinary(a, b, "or");
}

/**
 * Element-wise logical XOR (exclusive OR).
 *
 * Returns true (1) where exactly one input is non-zero (true but not both).
 * Inputs broadcast against each other (NumPy rules).
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @returns Boolean tensor with XOR results
 * @throws {DTypeError} If an input has string dtype
 * @throws {ShapeError} If the shapes cannot be broadcast
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
  return logicalBinary(a, b, "xor");
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
