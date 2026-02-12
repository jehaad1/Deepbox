import { type DType, DTypeError, getArrayElement, type Shape, ShapeError } from "../../core";
import type { Tensor } from "../tensor/Tensor";

export function isScalar(t: Tensor): boolean {
  return t.ndim === 0 && t.size === 1;
}

export function ensureNumericDType(
  t: Tensor,
  op: string
): asserts t is Tensor<Shape, Exclude<DType, "string">> {
  if (t.dtype === "string") {
    throw new DTypeError(`${op} is not defined for string dtype`);
  }
}

export function ensureSameDType(a: Tensor, b: Tensor): void {
  if (a.dtype !== b.dtype) {
    throw new DTypeError(`DType mismatch: ${a.dtype} vs ${b.dtype}`);
  }
}

export function ensureBroadcastableScalar(a: Tensor, b: Tensor): void {
  if (!isScalar(a) && !isScalar(b) && !canBroadcast(a.shape, b.shape)) {
    throw ShapeError.mismatch(a.shape, b.shape, "broadcast");
  }
}

export function canBroadcast(shapeA: Shape, shapeB: Shape): boolean {
  const maxLen = Math.max(shapeA.length, shapeB.length);
  for (let i = 0; i < maxLen; i++) {
    const dimA = getArrayElement(shapeA, shapeA.length - 1 - i, 1);
    const dimB = getArrayElement(shapeB, shapeB.length - 1 - i, 1);
    if (dimA === dimB || dimA === 1 || dimB === 1) {
      continue;
    }
    // Explicitly reject broadcasting 0 with anything other than 1 or 0,
    // which is covered by the condition above.
    if (dimA === 0 || dimB === 0) {
      return false;
    }
    return false;
  }
  return true;
}

export function getBroadcastShape(shapeA: Shape, shapeB: Shape): Shape {
  const maxLen = Math.max(shapeA.length, shapeB.length);
  const result: number[] = [];
  for (let i = 0; i < maxLen; i++) {
    const dimA = getArrayElement(shapeA, shapeA.length - 1 - i, 1);
    const dimB = getArrayElement(shapeB, shapeB.length - 1 - i, 1);
    if (dimA === dimB) {
      result.unshift(dimA);
    } else if (dimA === 1) {
      result.unshift(dimB);
    } else if (dimB === 1) {
      result.unshift(dimA);
    } else if (dimA === 0 || dimB === 0) {
      // If one dimension is 0, the broadcasted dimension is 0.
      // This is only allowed if the other is 1 (handled above) or they are equal.
      // If we are here, it's an invalid broadcast like (0, 2), which should fail.
      throw ShapeError.mismatch(shapeA, shapeB, "broadcast");
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
