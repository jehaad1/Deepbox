import type { Shape } from "../../core";

export function isContiguous(shape: Shape, strides: readonly number[]): boolean {
  if (shape.length !== strides.length) return false;
  // Check strides match row-major layout without allocating computeStrides
  let expected = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    if (strides[i] !== expected) return false;
    expected *= shape[i] ?? 1;
  }
  return true;
}

export function offsetFromFlatIndex(
  flat: number,
  logicalStrides: readonly number[],
  strides: readonly number[],
  offset: number
): number {
  let rem = flat;
  let out = offset;
  for (let axis = 0; axis < logicalStrides.length; axis++) {
    const stride = logicalStrides[axis] ?? 1;
    const coord = Math.floor(rem / stride);
    rem -= coord * stride;
    out += coord * (strides[axis] ?? 0);
  }
  return out;
}

export function broadcastOffsetFromFlatIndex(
  flat: number,
  outShape: Shape,
  outStrides: readonly number[],
  inShape: Shape,
  inStrides: readonly number[],
  inOffset: number
): number {
  if (inShape.length === 0) {
    return inOffset;
  }

  const rankDiff = outShape.length - inShape.length;
  let rem = flat;
  let offset = inOffset;

  for (let axis = 0; axis < outShape.length; axis++) {
    const stride = outStrides[axis] ?? 1;
    const coord = Math.floor(rem / stride);
    rem -= coord * stride;

    if (axis >= rankDiff) {
      const inDim = inShape[axis - rankDiff] ?? 1;
      if (inDim !== 1) {
        offset += coord * (inStrides[axis - rankDiff] ?? 0);
      }
    }
  }

  return offset;
}
