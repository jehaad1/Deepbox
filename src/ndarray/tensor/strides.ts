/**
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import type { Shape } from "../../core";

/**
 * Whether a strided layout is exactly row-major and gap-free, i.e. logical
 * element `i` lives at `offset + i`.
 *
 * The check is strict: every dimension's stride must equal the row-major
 * stride, including dimensions of size 1, so views such as a transposed column
 * vector count as non-contiguous. Kernels that only need to read the elements
 * in memory order (the ML input validation, `reshape`, `astype`) use
 * {@link isDenseLayout} instead, which ignores the strides of size-1 axes.
 *
 * @param shape - Logical shape
 * @param strides - Element strides, one per dimension
 */
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

/**
 * Whether logical element `i` lives at `offset + i`, ignoring the stride of
 * every dimension of size 1 (NumPy's `C_CONTIGUOUS` rule). A tensor with no
 * elements is always dense.
 *
 * This is weaker than {@link isContiguous}: the view of shape `[3, 1]` and
 * strides `[1, 3]` that `transpose` produces from a `[1, 3]` row is dense,
 * because the stride of the size-1 axis is never used. `reshape` and `astype`
 * use it to avoid copying such views.
 *
 * @param shape - Logical shape
 * @param strides - Element strides, one per dimension
 */
export function isDenseLayout(shape: Shape, strides: readonly number[]): boolean {
  if (shape.length !== strides.length) return false;
  if (shape.includes(0)) return true;
  let expected = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    const dim = shape[i] ?? 1;
    if (dim !== 1 && strides[i] !== expected) return false;
    expected *= dim;
  }
  return true;
}

/**
 * Convert a row-major logical flat index into a storage offset.
 *
 * @param flat - Logical flat index (row-major)
 * @param logicalStrides - Contiguous strides of the logical shape
 * @param strides - Actual element strides of the tensor
 * @param offset - Storage offset of the first element
 */
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

/**
 * Storage offset of an input element for a flat index into the broadcast
 * output shape. Dimensions of size 1 in the input (and missing leading
 * dimensions) contribute no offset.
 *
 * @param flat - Logical flat index into the output (row-major)
 * @param outShape - Broadcast output shape
 * @param outStrides - Contiguous strides of `outShape`
 * @param inShape - Input shape (must be broadcastable to `outShape`)
 * @param inStrides - Actual element strides of the input
 * @param inOffset - Storage offset of the input's first element
 */
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
