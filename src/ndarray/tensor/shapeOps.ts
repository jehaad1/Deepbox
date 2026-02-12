import {
  type Axis,
  type DType,
  InvalidParameterError,
  normalizeAxes,
  type Shape,
  ShapeError,
  validateShape,
} from "../../core";

import { Tensor } from "./Tensor";

type NumericDType = Exclude<DType, "string">;

function isStringTensor(t: Tensor): t is Tensor<Shape, "string"> {
  return t.dtype === "string";
}

function isNumericTensor(t: Tensor): t is Tensor<Shape, NumericDType> {
  return t.dtype !== "string";
}

/**
 * Remove single-dimensional entries from the shape.
 *
 * Returns a view of the tensor with all dimensions of size 1 removed.
 * If axis is specified, only removes dimensions at those positions.
 *
 * **Complexity**: O(ndim) - only manipulates shape metadata, no data copy
 *
 * **Parameters**:
 * @param t - Input tensor
 * @param axis - Axis to squeeze. If undefined, squeeze all axes of size 1
 *
 * **Returns**: Tensor with squeezed dimensions (view, no copy)
 *
 * @example
 * ```ts
 * import { squeeze, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([[[1], [2], [3]]]);
  // shape: (1, 3, 1)
 * const y = squeeze(x);                  // shape: (3,)
 * const z = squeeze(x, 2);               // shape: (1, 3)
 * ```
 *
 * @throws {Error} If axis is specified and dimension is not 1
 *
 * @see {@link https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html | NumPy squeeze}
 */
export function squeeze(t: Tensor, axis?: Axis | Axis[]): Tensor {
  let axesToSqueeze: Set<number>;

  if (axis === undefined) {
    // Squeeze all dimensions of size 1
    axesToSqueeze = new Set();
    for (let i = 0; i < t.ndim; i++) {
      if (t.shape[i] === 1) {
        axesToSqueeze.add(i);
      }
    }
  } else {
    // Squeeze only specified axes
    const axes = normalizeAxes(axis, t.ndim);
    axesToSqueeze = new Set();

    for (const ax of axes) {
      // Validate dimension is 1
      if (t.shape[ax] !== 1) {
        throw new ShapeError(`Cannot squeeze axis ${ax} with dimension ${t.shape[ax]} (must be 1)`);
      }

      axesToSqueeze.add(ax);
    }
  }

  // Build new shape and strides, excluding squeezed dimensions
  const newShape: number[] = [];
  const newStrides: number[] = [];

  for (let i = 0; i < t.ndim; i++) {
    if (!axesToSqueeze.has(i)) {
      const dim = t.shape[i];
      const stride = t.strides[i];
      if (dim === undefined || stride === undefined) {
        throw new ShapeError("Internal error: missing dimension or stride");
      }
      newShape.push(dim);
      newStrides.push(stride);
    }
  }

  // If all dimensions are squeezed, the result is a scalar (0D) tensor.
  // Represent scalars with shape [] (ndim=0), consistent with the rest of ndarray.
  if (newShape.length > 0) {
    validateShape(newShape);
  }

  // Return view with new shape
  if (isStringTensor(t)) {
    return Tensor.fromStringArray({
      data: t.data,
      shape: newShape,
      device: t.device,
      offset: t.offset,
      strides: newStrides,
    });
  }

  if (!isNumericTensor(t)) {
    throw new ShapeError("squeeze is not defined for string dtype");
  }

  return Tensor.fromTypedArray({
    data: t.data,
    shape: newShape,
    dtype: t.dtype,
    device: t.device,
    offset: t.offset,
    strides: newStrides,
  });
}

/**
 * Expand the shape by inserting a new axis.
 *
 * Returns a view of the tensor with a new dimension of size 1 inserted
 * at the specified position.
 *
 * **Complexity**: O(ndim) - only manipulates shape metadata, no data copy
 *
 * **Parameters**:
 * @param t - Input tensor
 * @param axis - Position where new axis is placed (can be negative)
 *
 * **Returns**: Tensor with expanded dimensions (view, no copy)
 *
 * @example
 * ```ts
 * import { unsqueeze, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([1, 2, 3]);     // shape: (3,)
 * const y = unsqueeze(x, 0);       // shape: (1, 3)
 * const z = unsqueeze(x, 1);       // shape: (3, 1)
 * const w = unsqueeze(x, -1);      // shape: (3, 1)
 * ```
 *
 * @see {@link https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html | NumPy expand_dims}
 */
export function unsqueeze(t: Tensor, axis: number): Tensor {
  // New ndim will be one more than current
  const newNdim = t.ndim + 1;

  // Normalize negative axis
  // Allow axis from -newNdim to newNdim (inclusive)
  const normalizedAxis = axis < 0 ? newNdim + axis : axis;

  // Validate axis is in valid range [0, newNdim]
  if (normalizedAxis < 0 || normalizedAxis > newNdim) {
    throw new InvalidParameterError(
      `axis ${axis} is out of bounds for result with ${newNdim} dimensions`,
      "axis",
      axis
    );
  }

  // Build new shape and strides with dimension of size 1 inserted
  const newShape: number[] = [];
  const newStrides: number[] = [];

  // Copy dimensions before insertion point
  for (let i = 0; i < normalizedAxis; i++) {
    const dim = t.shape[i];
    const stride = t.strides[i];
    if (dim === undefined || stride === undefined) {
      throw new ShapeError("Internal error: missing dimension or stride");
    }
    newShape.push(dim);
    newStrides.push(stride);
  }

  // Insert new dimension of size 1
  // Stride can be any value since dimension is 1 (no effect)
  newShape.push(1);
  // Use a stride that preserves contiguity when the source is contiguous.
  // For size-1 dims, any stride is valid, but this avoids breaking reshape/isContiguous.
  const insertedStride =
    normalizedAxis < t.ndim ? (t.strides[normalizedAxis] ?? 0) * (t.shape[normalizedAxis] ?? 1) : 1;
  newStrides.push(insertedStride);

  // Copy remaining dimensions after insertion point
  for (let i = normalizedAxis; i < t.ndim; i++) {
    const dim = t.shape[i];
    const stride = t.strides[i];
    if (dim === undefined || stride === undefined) {
      throw new ShapeError("Internal error: missing dimension or stride");
    }
    newShape.push(dim);
    newStrides.push(stride);
  }

  validateShape(newShape);

  // Return view with new shape
  if (isStringTensor(t)) {
    return Tensor.fromStringArray({
      data: t.data,
      shape: newShape,
      device: t.device,
      offset: t.offset,
      strides: newStrides,
    });
  }

  if (!isNumericTensor(t)) {
    throw new ShapeError("unsqueeze is not defined for string dtype");
  }

  return Tensor.fromTypedArray({
    data: t.data,
    shape: newShape,
    dtype: t.dtype,
    device: t.device,
    offset: t.offset,
    strides: newStrides,
  });
}

/**
 * Alias for unsqueeze.
 */
export const expandDims = unsqueeze;
