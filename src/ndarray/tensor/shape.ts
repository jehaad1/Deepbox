/**
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import {
  type DType,
  InvalidParameterError,
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
 * Give a tensor a new shape with the same number of elements.
 *
 * Returns a view that shares memory with `t` when its layout is contiguous.
 * Otherwise (for example after `transpose`) the elements are copied into
 * row-major order first, so writes to the result do not affect `t`.
 *
 * @param t - Input tensor
 * @param rawShape - Target shape. One dimension may be `-1`, in which case it
 *   is inferred from the remaining dimensions.
 * @returns Tensor with shape `rawShape`
 * @throws {ShapeError} If the shape has more than one `-1`, `-1` cannot be
 *   inferred, or the element count does not match
 * @throws {DataValidationError} If a dimension is negative (other than a
 *   single `-1`), fractional or not finite
 *
 * @example
 * ```ts
 * const x = tensor([[1, 2, 3], [4, 5, 6]]);
 * reshape(x, [3, 2]);   // [[1, 2], [3, 4], [5, 6]]
 * reshape(x, [-1]);     // [1, 2, 3, 4, 5, 6]
 * ```
 */
export function reshape(t: Tensor, rawShape: Shape): Tensor {
  // One implementation for both entry points: Tensor.reshape handles the -1
  // inference, device tensors, strings and the copy of non-contiguous views.
  return t.reshape(rawShape);
}

/**
 * Flatten a tensor to 1D in row-major order.
 *
 * Returns a view for contiguous input and a copy otherwise (see {@link reshape}).
 *
 * @param t - Input tensor
 * @returns 1D tensor with `t.size` elements
 */
export function flatten(t: Tensor): Tensor {
  return reshape(t, [t.size]);
}

/**
 * Transpose tensor dimensions.
 *
 * Reverses or permutes the axes of a tensor. The result is a view that shares
 * memory with the input.
 *
 * @param t - Input tensor
 * @param axes - Permutation of axes. If undefined, reverses all axes
 * @returns Transposed tensor (view)
 * @throws {ShapeError} If `axes` does not have one entry per dimension
 * @throws {InvalidParameterError} If `axes` contains a non-integer, an
 *   out-of-range axis or a duplicate
 *
 * @example
 * ```ts
 * import { transpose, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([[1, 2], [3, 4]]);  // shape: (2, 2)
 * const y = transpose(x);              // shape: (2, 2), values: [[1, 3], [2, 4]]
 *
 * const z = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);  // shape: (2, 2, 2)
 * const w = transpose(z, [2, 0, 1]);   // shape: (2, 2, 2), axes permuted
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-shape | Deepbox Shape & Indexing}
 */
export function transpose(t: Tensor, axes?: readonly number[]): Tensor {
  let axesArr: number[];

  if (axes === undefined) {
    // Reverse the axes order for default transpose
    // Create array [ndim-1, ndim-2, ..., 1, 0]
    // Example: for ndim=3, creates [2, 1, 0]
    axesArr = [];
    for (let i = t.ndim - 1; i >= 0; i--) {
      axesArr.push(i);
    }
  } else {
    axesArr = [...axes];

    // Validate axes
    if (axesArr.length !== t.ndim) {
      throw new ShapeError(`axes must have length ${t.ndim}, got ${axesArr.length}`);
    }

    const seen = new Set<number>();
    const normalized: number[] = [];
    for (const axis of axesArr) {
      if (!Number.isInteger(axis)) {
        throw new InvalidParameterError(`axes must be integers; received ${axis}`, "axes", axis);
      }
      const norm = axis < 0 ? t.ndim + axis : axis;
      if (norm < 0 || norm >= t.ndim) {
        throw new InvalidParameterError(
          `axis ${axis} out of range for ${t.ndim}D tensor`,
          "axes",
          axis
        );
      }
      if (seen.has(norm)) {
        throw new InvalidParameterError(`duplicate axis ${axis}`, "axes", axis);
      }
      seen.add(norm);
      normalized.push(norm);
    }
    axesArr = normalized;
  }

  // Compute new shape and strides
  const newShape: number[] = new Array<number>(t.ndim);
  const newStrides: number[] = new Array<number>(t.ndim);

  for (let i = 0; i < t.ndim; i++) {
    const axis = axesArr[i];
    if (axis === undefined) {
      throw new ShapeError("Internal error: missing axis");
    }
    const dim = t.shape[axis];
    const stride = t.strides[axis];
    if (dim === undefined || stride === undefined) {
      throw new ShapeError("Internal error: missing dimension or stride");
    }
    newShape[i] = dim;
    newStrides[i] = stride;
  }

  validateShape(newShape);

  if (t.isDeviceTensor) {
    return t.view(newShape, newStrides, t.offset);
  }

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
    throw new ShapeError("transpose is not defined for string dtype");
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
