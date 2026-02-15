import {
  type DType,
  dtypeToTypedArrayCtor,
  InvalidParameterError,
  type Shape,
  ShapeError,
  shapeToSize,
  type TypedArray,
  validateShape,
} from "../../core";
import { isContiguous, offsetFromFlatIndex } from "./strides";
import { computeStrides, Tensor } from "./Tensor";

type NumericDType = Exclude<DType, "string">;

/**
 * Resolve a shape that may contain a single -1 dimension.
 * Replaces -1 with the inferred size based on total element count.
 */
function resolveInferredShape(newShape: Shape, totalSize: number): Shape {
  let inferIdx = -1;
  let known = 1;
  for (let i = 0; i < newShape.length; i++) {
    const d = newShape[i];
    if (d === undefined) continue;
    if (d === -1) {
      if (inferIdx !== -1) {
        throw new ShapeError("Only one dimension can be -1 in reshape");
      }
      inferIdx = i;
    } else {
      known *= d;
    }
  }
  if (inferIdx === -1) return newShape;
  if (known === 0 || totalSize % known !== 0) {
    throw new ShapeError(
      `Cannot infer dimension for shape [${newShape}] with total size ${totalSize}`
    );
  }
  const resolved = [...newShape];
  resolved[inferIdx] = totalSize / known;
  return resolved;
}

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
export function reshape(t: Tensor, rawShape: Shape): Tensor {
  const newShape = resolveInferredShape(rawShape, t.size);
  validateShape(newShape);
  const newSize = shapeToSize(newShape);
  if (newSize !== t.size) {
    throw new ShapeError(`Cannot reshape tensor of size ${t.size} to shape [${newShape}]`);
  }

  const contiguous = isContiguous(t.shape, t.strides);

  if (isStringTensor(t)) {
    if (!contiguous) {
      const logicalStrides = computeStrides(t.shape);
      const out = new Array<string>(t.size);
      const data = t.data as string[];
      for (let i = 0; i < t.size; i++) {
        const off = offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
        out[i] = data[off] ?? "";
      }
      return Tensor.fromStringArray({
        data: out,
        shape: newShape,
        device: t.device,
      });
    }
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

  if (!contiguous) {
    const Ctor = dtypeToTypedArrayCtor(t.dtype);
    const logicalStrides = computeStrides(t.shape);
    const data = t.data as TypedArray;
    if (data instanceof BigInt64Array) {
      const out = new BigInt64Array(t.size);
      for (let i = 0; i < t.size; i++) {
        const off = offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
        out[i] = data[off] ?? 0n;
      }
      return Tensor.fromTypedArray({
        data: out,
        shape: newShape,
        dtype: t.dtype,
        device: t.device,
      });
    }
    const out = new Ctor(t.size);
    if (!(out instanceof BigInt64Array)) {
      const numData = data as Exclude<TypedArray, BigInt64Array>;
      for (let i = 0; i < t.size; i++) {
        const off = offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
        out[i] = numData[off] ?? 0;
      }
    }
    return Tensor.fromTypedArray({
      data: out,
      shape: newShape,
      dtype: t.dtype,
      device: t.device,
    });
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
