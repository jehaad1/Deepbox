/**
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import { type DType, DTypeError, getArrayElement, type Shape, ShapeError } from "../../core";
import { promoteTypes } from "../../core/utils/dtype_utils";
import type { Tensor } from "../tensor/Tensor";

/** A tensor whose dtype is not `string`. */
export type NumericTensor = Tensor<Shape, Exclude<DType, "string">>;

/**
 * True for 0-d tensors holding one element (shape `[]`).
 *
 * Tensors of shape `[1]`, `[1, 1]`, ... are not scalars here; they take part in
 * ordinary broadcasting.
 */
export function isScalar(t: Tensor): boolean {
  return t.ndim === 0 && t.size === 1;
}

/**
 * Assert that a tensor has a numeric (non-string) dtype.
 *
 * @param t - Tensor to check
 * @param op - Operation name used in the error message
 * @throws {DTypeError} If the tensor has string dtype
 */
export function ensureNumericDType(
  t: Tensor,
  op: string
): asserts t is Tensor<Shape, Exclude<DType, "string">> {
  if (t.dtype === "string") {
    throw new DTypeError(`${op} is not defined for string dtype`);
  }
}

/**
 * Assert that two tensors share one dtype.
 *
 * Binary element-wise ops no longer need this because they promote their operands with
 * {@link promoteOperands}; it stays for callers that require identical dtypes.
 *
 * @throws {DTypeError} If the dtypes differ
 */
export function ensureSameDType(a: Tensor, b: Tensor): void {
  if (a.dtype !== b.dtype) {
    throw new DTypeError(`DType mismatch: ${a.dtype} vs ${b.dtype}`);
  }
}

/**
 * Convert the operands of a binary op to their common dtype (PyTorch-style promotion).
 *
 * `bool < uint8 < int32 < int64 < float16, bfloat16 < float32 < float64`; an operand that
 * already has the common dtype is returned as it is (no copy). Both operands must be
 * numeric, and a tensor that lives in device memory cannot be converted.
 *
 * @param a - First operand
 * @param b - Second operand
 * @param op - Operation name used in the error message
 * @returns The two operands, both with the common dtype
 * @throws {DTypeError} If either tensor has string dtype, or a tensor in device memory would
 *   need a dtype conversion
 *
 * @example
 * ```ts
 * const [x, y] = promoteOperands(tensor([1, 2], { dtype: "int32" }), tensor([0.5, 1.5]), "add");
 * x.dtype; // "float32"
 * ```
 */
export function promoteOperands(a: Tensor, b: Tensor, op: string): [NumericTensor, NumericTensor] {
  ensureNumericDType(a, op);
  ensureNumericDType(b, op);
  if (a.dtype === b.dtype) return [a, b];
  const target = promoteTypes(a.dtype, b.dtype);
  if ((a.isDeviceTensor && a.dtype !== target) || (b.isDeviceTensor && b.dtype !== target)) {
    throw new DTypeError(
      `${op}: cannot promote dtype ${a.dtype} with ${b.dtype} for a tensor in device memory; ` +
        "move it to the CPU with `await t.cpu()` or give both operands the same dtype"
    );
  }
  const x: Tensor = a.dtype === target ? a : a.astype(target);
  const y: Tensor = b.dtype === target ? b : b.astype(target);
  ensureNumericDType(x, op);
  ensureNumericDType(y, op);
  return [x, y];
}

/**
 * Assert that two tensors can be combined element-wise (either is a 0-d
 * scalar, or their shapes broadcast).
 *
 * @throws {ShapeError} If the shapes are not broadcast-compatible
 */
export function ensureBroadcastableScalar(a: Tensor, b: Tensor): void {
  if (!isScalar(a) && !isScalar(b) && !canBroadcast(a.shape, b.shape)) {
    throw ShapeError.mismatch(a.shape, b.shape, "broadcast");
  }
}

/**
 * Check whether two shapes are broadcast-compatible under NumPy rules: shapes
 * are aligned at the trailing axis and each pair of sizes must be equal or
 * contain a 1. A size-0 axis only pairs with 0 or 1.
 */
export function canBroadcast(shapeA: Shape, shapeB: Shape): boolean {
  const maxLen = Math.max(shapeA.length, shapeB.length);
  for (let i = 0; i < maxLen; i++) {
    const dimA = getArrayElement(shapeA, shapeA.length - 1 - i, 1);
    const dimB = getArrayElement(shapeB, shapeB.length - 1 - i, 1);
    if (dimA !== dimB && dimA !== 1 && dimB !== 1) {
      return false;
    }
  }
  return true;
}

/**
 * Compute the broadcast result shape of two shapes under NumPy rules.
 *
 * @throws {ShapeError} If the shapes are not broadcast-compatible
 */
export function getBroadcastShape(shapeA: Shape, shapeB: Shape): Shape {
  const maxLen = Math.max(shapeA.length, shapeB.length);
  const result = new Array<number>(maxLen);
  for (let i = 0; i < maxLen; i++) {
    const dimA = getArrayElement(shapeA, shapeA.length - 1 - i, 1);
    const dimB = getArrayElement(shapeB, shapeB.length - 1 - i, 1);
    if (dimA === dimB || dimB === 1) {
      result[maxLen - 1 - i] = dimA;
    } else if (dimA === 1) {
      result[maxLen - 1 - i] = dimB;
    } else {
      throw ShapeError.mismatch(shapeA, shapeB, "broadcast");
    }
  }
  return result;
}

/**
 * Iterates over two broadcasted tensors and an output tensor efficiently.
 *
 * This avoids expensive index calculations (division/modulo) inside the inner loop
 * by maintaining running offsets for all tensors.
 *
 * Inputs may be strided views; size-1 axes are walked with stride 0. The output
 * tensor must have the broadcast shape of `a` and `b`. The callback receives
 * physical buffer offsets (view offset included), not logical flat indices.
 *
 * @param a - First input tensor
 * @param b - Second input tensor
 * @param out - Output tensor (must have broadcasted shape)
 * @param op - Callback to execute for each element: (offsetA, offsetB, offsetOut)
 */
export function broadcastApply(
  a: Tensor,
  b: Tensor,
  out: Tensor,
  op: (offA: number, offB: number, offOut: number) => void
): void {
  if (out.size === 0) return;

  // Handle scalar case (rank 0)
  if (out.ndim === 0) {
    op(a.offset, b.offset, out.offset);
    return;
  }

  // Pre-compute strides for A and B relative to Output shape
  const outShape = out.shape;
  const outStrides = out.strides;
  const rank = outShape.length;

  const stridesA = new Array<number>(rank).fill(0);
  const stridesB = new Array<number>(rank).fill(0);

  const rankDiffA = rank - a.ndim;
  for (let i = 0; i < rank; i++) {
    if (i >= rankDiffA) {
      const dim = a.shape[i - rankDiffA] ?? 1;
      // If dim is 1, stride is 0 (broadcast). Otherwise, use actual stride.
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

  // Loop state
  const idx = new Array<number>(rank).fill(0);
  let offA = a.offset;
  let offB = b.offset;
  let offOut = out.offset;

  while (true) {
    op(offA, offB, offOut);

    // Odometer increment
    let axis = rank - 1;
    for (;;) {
      const currentIdx = idx[axis];
      if (currentIdx === undefined) {
        // Should never happen if rank matches idx length
        return;
      }
      const nextIdx = currentIdx + 1;
      idx[axis] = nextIdx;

      const strideA = stridesA[axis];
      const strideB = stridesB[axis];
      const strideOut = outStrides[axis];
      const dimOut = outShape[axis];

      if (strideA === undefined || strideB === undefined || dimOut === undefined) {
        // Should never happen if shapes/strides are valid
        return;
      }

      offA += strideA;
      offB += strideB;
      offOut += strideOut ?? 0;

      if (nextIdx < dimOut) {
        break;
      }

      // Carry: reset index and offset for this axis
      // nextIdx is now dimOut
      const count = nextIdx;
      offA -= strideA * count;
      offB -= strideB * count;
      offOut -= (strideOut ?? 0) * count;
      idx[axis] = 0;
      axis--;

      if (axis < 0) return;
    }
  }
}
