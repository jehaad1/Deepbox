import {
  type DType,
  InvalidParameterError,
  type Shape,
  ShapeError,
  shapeToSize,
  validateShape,
} from "../../core";
import { isContiguous } from "./strides";
import { computeStrides, Tensor } from "./Tensor";

type NumericDType = Exclude<DType, "string">;

function isStringTensor(t: Tensor): t is Tensor<Shape, "string"> {
  return t.dtype === "string";
}

function isNumericTensor(t: Tensor): t is Tensor<Shape, NumericDType> {
  return t.dtype !== "string";
}

/**
 * Change shape (view) without copying.
 *
 * Notes:
 * - Currently only supports contiguous tensors.
 * - In the future, reshape should support more view cases using strides.
 */
export function reshape(t: Tensor, newShape: Shape): Tensor {
  validateShape(newShape);
  const newSize = shapeToSize(newShape);
  if (newSize !== t.size) {
    throw new ShapeError(`Cannot reshape tensor of size ${t.size} to shape [${newShape}]`);
  }
  if (!isContiguous(t.shape, t.strides)) {
    throw new ShapeError("reshape requires a contiguous tensor");
  }

  if (isStringTensor(t)) {
    return Tensor.fromStringArray({
      data: t.data,
      shape: newShape,
      device: t.device,
      offset: t.offset,
      strides: computeStrides(newShape),
    });
  }

  if (!isNumericTensor(t)) {
    throw new ShapeError("reshape is not defined for string dtype");
  }

  return Tensor.fromTypedArray({
    data: t.data,
    shape: newShape,
    dtype: t.dtype,
    device: t.device,
    offset: t.offset,
    strides: computeStrides(newShape),
  });
}

/**
 * Flatten to 1D.
 */
export function flatten(t: Tensor): Tensor {
  return reshape(t, [t.size]);
}

/**
 * Transpose tensor dimensions.
 *
 * Reverses or permutes the axes of a tensor.
 *
 * @param t - Input tensor
 * @param axes - Permutation of axes. If undefined, reverses all axes
 * @returns Transposed tensor
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
 * @see {@link https://numpy.org/doc/stable/reference/generated/numpy.transpose.html | NumPy transpose}
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
