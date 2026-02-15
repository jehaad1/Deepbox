/**
 * Tensor manipulation operations.
 *
 * This module provides functions for manipulating tensor structure and content:
 * - concatenate: Join tensors along an axis
 * - stack: Stack tensors along a new axis
 * - split: Split tensor into multiple sub-tensors
 * - tile: Repeat tensor along axes
 * - repeat: Repeat elements along an axis
 * - pad: Add padding to tensor
 * - flip: Reverse tensor along axes
 *
 * All operations maintain type safety and proper error handling.
 */

import type { Axis, TypedArray } from "../../core";
import {
  DeepboxError,
  DTypeError,
  dtypeToTypedArrayCtor,
  getBigIntElement,
  getNumericElement,
  InvalidParameterError,
  normalizeAxis,
  ShapeError,
  shapeToSize,
} from "../../core";
import { isContiguous, offsetFromFlatIndex } from "../tensor/strides";
import { computeStrides, Tensor } from "../tensor/Tensor";

/**
 * Concatenate tensors along an existing axis.
 *
 * All tensors must have the same shape except in the concatenation dimension.
 * The output dtype is determined by the first tensor.
 *
 * **Complexity**: O(n) where n is total number of elements
 *
 * @param tensors - Array of tensors to concatenate
 * @param axis - Axis along which to concatenate (default: 0)
 * @returns Concatenated tensor
 *
 * @example
 * ```ts
 * const a = tensor([[1, 2], [3, 4]]);
 * const b = tensor([[5, 6]]);
 * const c = concatenate([a, b], 0);  // [[1, 2], [3, 4], [5, 6]]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function concatenate(tensors: Tensor[], axis: Axis = 0): Tensor {
  // Validate input: need at least one tensor
  if (tensors.length === 0) {
    throw new InvalidParameterError("concatenate requires at least one tensor", "tensors");
  }

  // Single tensor: return copy
  if (tensors.length === 1) {
    const t = tensors[0];
    if (!t) throw new DeepboxError("Unexpected: tensor at index 0 is undefined");
    if (t.dtype === "string") {
      const data = t.data;
      if (!Array.isArray(data)) throw new DeepboxError("Internal error: expected string array");
      return Tensor.fromStringArray({
        data: [...data],
        shape: t.shape,
        device: t.device,
      });
    }
    const data = t.data;
    if (Array.isArray(data)) throw new DeepboxError("Internal error: expected typed array");
    return Tensor.fromTypedArray({
      data: data.slice(),
      shape: t.shape,
      dtype: t.dtype,
      device: t.device,
    });
  }

  // Get reference tensor for validation
  const first = tensors[0];
  if (!first) throw new DeepboxError("Unexpected: first tensor is undefined");
  const ndim = first.ndim;
  const dtype = first.dtype;

  // Normalize axis to positive index
  const ax = normalizeAxis(axis, ndim);

  // Validate all tensors have same ndim and dtype
  for (let i = 1; i < tensors.length; i++) {
    const t = tensors[i];
    if (!t) throw new DeepboxError(`Unexpected: tensor at index ${i} is undefined`);
    if (t.ndim !== ndim) {
      throw new ShapeError(`All tensors must have same ndim; got ${ndim} and ${t.ndim}`);
    }
    if (t.dtype !== dtype) {
      throw new DTypeError(`All tensors must have same dtype; got ${dtype} and ${t.dtype}`);
    }
  }

  // Validate shapes match except on concatenation axis
  for (let i = 1; i < tensors.length; i++) {
    const t = tensors[i];
    if (!t) throw new DeepboxError(`Unexpected: tensor at index ${i} is undefined`);
    for (let d = 0; d < ndim; d++) {
      if (d !== ax && first.shape[d] !== t.shape[d]) {
        throw ShapeError.mismatch(
          first.shape,
          t.shape,
          `concatenate: shapes must match except on axis ${axis}`
        );
      }
    }
  }

  // Calculate output shape: sum along concatenation axis
  const outShape = [...first.shape];
  let totalAlongAxis = first.shape[ax] ?? 0;
  for (let i = 1; i < tensors.length; i++) {
    const t = tensors[i];
    if (!t) throw new DeepboxError(`Unexpected: tensor at index ${i} is undefined`);
    totalAlongAxis += t.shape[ax] ?? 0;
  }
  outShape[ax] = totalAlongAxis;

  // Allocate output buffer
  const outSize = shapeToSize(outShape);
  const isString = dtype === "string";
  const outData = isString
    ? new Array<string>(outSize)
    : new (dtypeToTypedArrayCtor(dtype))(outSize);

  // Compute output strides for efficient indexing
  const outStrides = computeStrides(outShape);

  // Copy data from each tensor
  let offsetAlongAxis = 0;

  // Prepare output buffers
  let stringOut: string[] | undefined;
  let bigIntOut: BigInt64Array | undefined;
  let numericOut: Exclude<TypedArray, BigInt64Array> | undefined;

  if (Array.isArray(outData)) {
    stringOut = outData;
  } else if (outData instanceof BigInt64Array) {
    bigIntOut = outData;
  } else {
    numericOut = outData;
  }

  for (const tensor of tensors) {
    const t = tensor;
    if (!t) throw new DeepboxError("Unexpected: tensor is undefined");
    const tSize = t.size;
    const tLogicalStrides = computeStrides(t.shape);

    // Prepare source buffers
    let stringSrc: readonly string[] | undefined;
    let numericSrc: Exclude<TypedArray, BigInt64Array> | undefined;

    if (Array.isArray(t.data)) {
      stringSrc = t.data;
    } else if (!(t.data instanceof BigInt64Array)) {
      numericSrc = t.data;
    }

    // Iterate through all elements in current tensor
    for (let flatIdx = 0; flatIdx < tSize; flatIdx++) {
      // Convert flat index to coordinates in source tensor
      let rem = flatIdx;
      const coords = new Array<number>(ndim);
      for (let d = 0; d < ndim; d++) {
        const stride = tLogicalStrides[d] ?? 1;
        coords[d] = Math.floor(rem / stride);
        rem -= (coords[d] ?? 0) * stride;
      }

      // Compute source offset using actual strides
      let srcOffset = t.offset;
      for (let d = 0; d < ndim; d++) {
        srcOffset += (coords[d] ?? 0) * (t.strides[d] ?? 0);
      }

      // Adjust coordinate along concatenation axis
      coords[ax] = (coords[ax] ?? 0) + offsetAlongAxis;

      // Convert coordinates to flat index in output
      let outIdx = 0;
      for (let d = 0; d < ndim; d++) {
        outIdx += (coords[d] ?? 0) * (outStrides[d] ?? 1);
      }

      // Copy element
      if (stringOut && stringSrc) {
        stringOut[outIdx] = stringSrc[srcOffset] ?? "";
      } else if (bigIntOut && t.data instanceof BigInt64Array) {
        bigIntOut[outIdx] = getBigIntElement(t.data, srcOffset);
      } else if (numericOut && numericSrc) {
        numericOut[outIdx] = getNumericElement(numericSrc, srcOffset);
      }
    }

    // Update offset for next tensor
    offsetAlongAxis += t.shape[ax] ?? 0;
  }

  if (Array.isArray(outData)) {
    return Tensor.fromStringArray({
      data: outData,
      shape: outShape,
      device: first.device,
    });
  }

  if (dtype === "string") {
    throw new DeepboxError("Internal error: string dtype but non-array data");
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: outShape,
    dtype,
    device: first.device,
  });
}

/**
 * Stack tensors along a new axis.
 *
 * All tensors must have exactly the same shape.
 * Creates a new dimension at the specified axis.
 *
 * **Complexity**: O(n) where n is total number of elements
 *
 * @param tensors - Array of tensors to stack
 * @param axis - Axis along which to stack (default: 0)
 * @returns Stacked tensor
 *
 * @example
 * ```ts
 * const a = tensor([1, 2, 3]);
 * const b = tensor([4, 5, 6]);
 * const c = stack([a, b], 0);  // [[1, 2, 3], [4, 5, 6]]
 * const d = stack([a, b], 1);  // [[1, 4], [2, 5], [3, 6]]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function stack(tensors: Tensor[], axis: Axis = 0): Tensor {
  // Validate input
  if (tensors.length === 0) {
    throw new InvalidParameterError("stack requires at least one tensor", "tensors");
  }

  const first = tensors[0];
  if (!first) {
    throw new DeepboxError("Unexpected: first tensor is undefined");
  }
  const ndim = first.ndim;
  const dtype = first.dtype;

  // Normalize axis: can be from -ndim-1 to ndim (inclusive)
  // We use ndim + 1 because the output tensor has one more dimension
  const ax = normalizeAxis(axis, ndim + 1);

  // Validate all tensors have identical shape and dtype
  for (let i = 1; i < tensors.length; i++) {
    const t = tensors[i];
    if (!t) throw new DeepboxError(`Unexpected: tensor at index ${i} is undefined`);
    if (t.ndim !== ndim) {
      throw new ShapeError(`All tensors must have same ndim; got ${ndim} and ${t.ndim}`);
    }
    if (t.dtype !== dtype) {
      throw new DTypeError(`All tensors must have same dtype; got ${dtype} and ${t.dtype}`);
    }
    for (let d = 0; d < ndim; d++) {
      if (first.shape[d] !== t.shape[d]) {
        throw ShapeError.mismatch(first.shape, t.shape, "stack: all tensors must have same shape");
      }
    }
  }

  // Build output shape: insert new dimension at axis
  const outShape: number[] = [];
  for (let d = 0; d < ax; d++) {
    outShape.push(first.shape[d] ?? 0);
  }
  outShape.push(tensors.length); // New dimension with size = number of tensors
  for (let d = ax; d < ndim; d++) {
    outShape.push(first.shape[d] ?? 0);
  }

  // Allocate output buffer
  const outSize = shapeToSize(outShape);
  const isString = dtype === "string";
  const outData = isString
    ? new Array<string>(outSize)
    : new (dtypeToTypedArrayCtor(dtype))(outSize);

  // Compute strides
  const outStrides = computeStrides(outShape);
  const elemSize = first.size;

  // Prepare output buffers
  let stringOut: string[] | undefined;
  let bigIntOut: BigInt64Array | undefined;
  let numericOut: Exclude<TypedArray, BigInt64Array> | undefined;

  if (Array.isArray(outData)) {
    stringOut = outData;
  } else if (outData instanceof BigInt64Array) {
    bigIntOut = outData;
  } else {
    numericOut = outData;
  }

  // Copy each tensor into the output
  for (let tensorIdx = 0; tensorIdx < tensors.length; tensorIdx++) {
    const t = tensors[tensorIdx];
    if (!t) throw new DeepboxError(`Unexpected: tensor at index ${tensorIdx} is undefined`);

    const tLogicalStrides = computeStrides(t.shape);

    let stringSrc: readonly string[] | undefined;
    let numericSrc: Exclude<TypedArray, BigInt64Array> | undefined;
    if (Array.isArray(t.data)) {
      stringSrc = t.data;
    } else if (!(t.data instanceof BigInt64Array)) {
      numericSrc = t.data;
    }

    // Iterate through all elements in current tensor
    for (let flatIdx = 0; flatIdx < elemSize; flatIdx++) {
      // Convert flat index to coordinates in source tensor
      let rem = flatIdx;
      const srcCoords = new Array<number>(ndim);
      for (let d = 0; d < ndim; d++) {
        const stride = tLogicalStrides[d] ?? 1;
        srcCoords[d] = Math.floor(rem / stride);
        rem -= (srcCoords[d] ?? 0) * stride;
      }

      // Compute source offset using actual strides
      let srcOffset = t.offset;
      for (let d = 0; d < ndim; d++) {
        srcOffset += (srcCoords[d] ?? 0) * (t.strides[d] ?? 0);
      }

      // Convert to flat index in output (insert tensorIdx at axis)
      let outIdx = 0;
      for (let d = 0; d < ax; d++) {
        outIdx += (srcCoords[d] ?? 0) * (outStrides[d] ?? 1);
      }
      outIdx += tensorIdx * (outStrides[ax] ?? 1);
      for (let d = ax; d < ndim; d++) {
        outIdx += (srcCoords[d] ?? 0) * (outStrides[d + 1] ?? 1);
      }

      // Copy element
      if (stringOut && stringSrc) {
        stringOut[outIdx] = stringSrc[srcOffset] ?? "";
      } else if (bigIntOut && t.data instanceof BigInt64Array) {
        bigIntOut[outIdx] = getBigIntElement(t.data, srcOffset);
      } else if (numericOut && numericSrc) {
        numericOut[outIdx] = getNumericElement(numericSrc, srcOffset);
      }
    }
  }

  if (Array.isArray(outData)) {
    return Tensor.fromStringArray({
      data: outData,
      shape: outShape,
      device: first.device,
    });
  }

  if (dtype === "string") {
    throw new DeepboxError("Internal error: string dtype but non-array data");
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: outShape,
    dtype,
    device: first.device,
  });
}

/**
 * Split tensor into multiple sub-tensors along an axis.
 *
 * If indices_or_sections is an integer, the tensor is split into that many
 * equal parts (axis dimension must be divisible).
 * If it's an array, it specifies the indices where to split.
 *
 * **Complexity**: O(n) where n is total number of elements
 *
 * @param t - Input tensor
 * @param indices_or_sections - Number of sections or array of split indices
 * @param axis - Axis along which to split (default: 0)
 * @returns Array of sub-tensors
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3, 4, 5, 6]);
 * const parts = split(t, 3);  // [tensor([1, 2]), tensor([3, 4]), tensor([5, 6])]
 * const parts2 = split(t, [2, 4]);  // [tensor([1, 2]), tensor([3, 4]), tensor([5, 6])]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function split(t: Tensor, indices_or_sections: number | number[], axis: Axis = 0): Tensor[] {
  // Normalize axis
  const ax = normalizeAxis(axis, t.ndim);

  const axisSize = t.shape[ax] ?? 0;

  // Determine split points
  let splitPoints: number[];
  if (typeof indices_or_sections === "number") {
    // Equal splits
    const numSections = indices_or_sections;
    if (!Number.isInteger(numSections) || numSections <= 0) {
      throw new InvalidParameterError(
        `indices_or_sections must be a positive integer; received ${numSections}`,
        "indices_or_sections",
        numSections
      );
    }
    if (axisSize % numSections !== 0) {
      throw new InvalidParameterError(
        `axis dimension ${axisSize} not divisible by ${numSections} equal sections`,
        "indices_or_sections",
        numSections
      );
    }
    const sectionSize = axisSize / numSections;
    splitPoints = [];
    for (let i = 1; i < numSections; i++) {
      splitPoints.push(i * sectionSize);
    }
  } else {
    // Split at specified indices
    splitPoints = [...indices_or_sections];
    let prev = 0;
    for (let i = 0; i < splitPoints.length; i++) {
      const idx = splitPoints[i];
      if (idx === undefined || !Number.isInteger(idx)) {
        throw new InvalidParameterError(
          `split index must be an integer; received ${String(idx)}`,
          "indices_or_sections",
          indices_or_sections
        );
      }
      if (idx < 0 || idx > axisSize) {
        throw new InvalidParameterError(
          `split index ${idx} is out of bounds for axis size ${axisSize}`,
          "indices_or_sections",
          indices_or_sections
        );
      }
      if (i > 0 && idx < prev) {
        throw new InvalidParameterError(
          "split indices must be non-decreasing",
          "indices_or_sections",
          indices_or_sections
        );
      }
      prev = idx;
    }
  }

  // Add boundaries
  const boundaries = [0, ...splitPoints, axisSize];

  // Prepare source buffers
  let stringSrc: readonly string[] | undefined;
  let numericSrc: Exclude<TypedArray, BigInt64Array> | undefined;
  if (Array.isArray(t.data)) {
    stringSrc = t.data;
  } else if (!(t.data instanceof BigInt64Array)) {
    numericSrc = t.data;
  }

  // Create sub-tensors
  const result: Tensor[] = [];
  for (let i = 0; i < boundaries.length - 1; i++) {
    const start = boundaries[i] ?? 0;
    const end = boundaries[i + 1] ?? axisSize;
    const size = end - start;

    // Build shape for this sub-tensor
    const subShape = [...t.shape];
    subShape[ax] = size;

    // Allocate buffer
    const subSize = shapeToSize(subShape);
    const isString = t.dtype === "string";
    const subData = isString
      ? new Array<string>(subSize)
      : new (dtypeToTypedArrayCtor(t.dtype))(subSize);

    // Prepare output buffers
    let stringOut: string[] | undefined;
    let bigIntOut: BigInt64Array | undefined;
    let numericOut: Exclude<TypedArray, BigInt64Array> | undefined;

    if (Array.isArray(subData)) {
      stringOut = subData;
    } else if (subData instanceof BigInt64Array) {
      bigIntOut = subData;
    } else {
      numericOut = subData;
    }

    // Copy elements
    const subStrides = computeStrides(subShape);

    for (let flatIdx = 0; flatIdx < subSize; flatIdx++) {
      // Convert flat index to coordinates in sub-tensor
      let rem = flatIdx;
      const coords = new Array<number>(t.ndim);
      for (let d = 0; d < t.ndim; d++) {
        const stride = subStrides[d] ?? 1;
        coords[d] = Math.floor(rem / stride);
        rem -= (coords[d] ?? 0) * stride;
      }

      // Adjust coordinate along split axis
      coords[ax] = (coords[ax] ?? 0) + start;

      // Convert to flat index in source tensor
      let srcIdx = t.offset;
      for (let d = 0; d < t.ndim; d++) {
        srcIdx += (coords[d] ?? 0) * (t.strides[d] ?? 0);
      }

      // Copy element
      if (stringOut && stringSrc) {
        stringOut[flatIdx] = stringSrc[srcIdx] ?? "";
      } else if (bigIntOut && t.data instanceof BigInt64Array) {
        bigIntOut[flatIdx] = getBigIntElement(t.data, srcIdx);
      } else if (numericOut && numericSrc) {
        numericOut[flatIdx] = getNumericElement(numericSrc, srcIdx);
      }
    }

    if (Array.isArray(subData)) {
      result.push(
        Tensor.fromStringArray({
          data: subData,
          shape: subShape,
          device: t.device,
        })
      );
    } else {
      if (t.dtype === "string") {
        throw new DeepboxError("Internal error: string dtype but non-array data");
      }
      result.push(
        Tensor.fromTypedArray({
          data: subData,
          shape: subShape,
          dtype: t.dtype,
          device: t.device,
        })
      );
    }
  }

  return result;
}

/**
 * Repeat tensor along axes by tiling.
 *
 * Constructs a tensor by repeating the input tensor the specified number
 * of times along each axis.
 *
 * **Complexity**: O(n * product(reps)) where n is input size
 *
 * @param t - Input tensor
 * @param reps - Number of repetitions along each axis
 * @returns Tiled tensor
 *
 * @example
 * ```ts
 * const t = tensor([[1, 2], [3, 4]]);
 * const tiled = tile(t, [2, 3]);
 * // [[1, 2, 1, 2, 1, 2],
 * //  [3, 4, 3, 4, 3, 4],
 * //  [1, 2, 1, 2, 1, 2],
 * //  [3, 4, 3, 4, 3, 4]]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function tile(t: Tensor, reps: number[]): Tensor {
  // Validate reps
  if (reps.length === 0) {
    throw new InvalidParameterError("reps must have at least one element", "reps");
  }
  for (let i = 0; i < reps.length; i++) {
    const rep = reps[i];
    if (rep === undefined || !Number.isInteger(rep) || rep < 0) {
      throw new InvalidParameterError(
        `reps[${i}] must be a non-negative integer; received ${String(rep)}`,
        "reps",
        reps
      );
    }
  }

  // Adjust dimensions if needed
  const ndim = Math.max(t.ndim, reps.length);
  const inShape = new Array<number>(ndim).fill(1);
  const repCounts = new Array<number>(ndim).fill(1);

  // Fill from the right (trailing dimensions)
  for (let i = 0; i < t.ndim; i++) {
    inShape[ndim - t.ndim + i] = t.shape[i] ?? 1;
  }
  for (let i = 0; i < reps.length; i++) {
    repCounts[ndim - reps.length + i] = reps[i] ?? 1;
  }

  // Calculate output shape
  const outShape = inShape.map((s, i) => s * (repCounts[i] ?? 1));

  // Allocate output
  const outSize = shapeToSize(outShape);
  const outData =
    t.dtype === "string"
      ? new Array<string>(outSize)
      : new (dtypeToTypedArrayCtor(t.dtype))(outSize);

  // Compute strides
  const outStrides = computeStrides(outShape);

  // Prepare buffers
  let stringOut: string[] | undefined;
  let bigIntOut: BigInt64Array | undefined;
  let numericOut: Exclude<TypedArray, BigInt64Array> | undefined;

  if (Array.isArray(outData)) {
    stringOut = outData;
  } else if (outData instanceof BigInt64Array) {
    bigIntOut = outData;
  } else {
    numericOut = outData;
  }

  let stringSrc: readonly string[] | undefined;
  let numericSrc: Exclude<TypedArray, BigInt64Array> | undefined;
  if (Array.isArray(t.data)) {
    stringSrc = t.data;
  } else if (!(t.data instanceof BigInt64Array)) {
    numericSrc = t.data;
  }

  // Fill output by repeating input
  for (let flatIdx = 0; flatIdx < outSize; flatIdx++) {
    // Convert flat index to coordinates in output
    let rem = flatIdx;
    const outCoords = new Array<number>(ndim);
    for (let d = 0; d < ndim; d++) {
      const stride = outStrides[d] ?? 1;
      outCoords[d] = Math.floor(rem / stride);
      rem -= (outCoords[d] ?? 0) * stride;
    }

    // Map to input coordinates using modulo
    const inCoords = outCoords.map((c, i) => c % (inShape[i] ?? 1));

    // Convert to flat index in input (accounting for original shape)
    let srcIdx = t.offset;
    for (let d = 0; d < t.ndim; d++) {
      const inCoord = inCoords[ndim - t.ndim + d] ?? 0;
      srcIdx += inCoord * (t.strides[d] ?? 0);
    }

    // Copy element
    if (stringOut && stringSrc) {
      stringOut[flatIdx] = stringSrc[srcIdx] ?? "";
    } else if (bigIntOut && t.data instanceof BigInt64Array) {
      bigIntOut[flatIdx] = getBigIntElement(t.data, srcIdx);
    } else if (numericOut && numericSrc) {
      numericOut[flatIdx] = getNumericElement(numericSrc, srcIdx);
    }
  }

  if (Array.isArray(outData)) {
    return Tensor.fromStringArray({
      data: outData,
      shape: outShape,
      device: t.device,
    });
  }

  if (t.dtype === "string") {
    throw new DeepboxError("Internal error: string dtype but non-array data");
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: outShape,
    dtype: t.dtype,
    device: t.device,
  });
}

/**
 * Repeat elements of a tensor along an axis.
 *
 * Each element is repeated the specified number of times.
 *
 * **Complexity**: O(n * repeats) where n is input size
 *
 * @param t - Input tensor
 * @param repeats - Number of times to repeat each element
 * @param axis - Axis along which to repeat (default: flatten first)
 * @returns Tensor with repeated elements
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3]);
 * const r = repeat(t, 2);  // [1, 1, 2, 2, 3, 3]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function repeat(t: Tensor, repeats: number, axis?: Axis): Tensor {
  if (!Number.isInteger(repeats) || repeats < 0) {
    throw new InvalidParameterError(
      `repeats must be a non-negative integer; received ${repeats}`,
      "repeats",
      repeats
    );
  }
  // If no axis specified, flatten and repeat
  if (axis === undefined) {
    const flatSize = t.size * repeats;
    const outData =
      t.dtype === "string"
        ? new Array<string>(flatSize)
        : new (dtypeToTypedArrayCtor(t.dtype))(flatSize);

    // Copy each element 'repeats' times
    const logicalStrides = computeStrides(t.shape);
    const contiguous = isContiguous(t.shape, t.strides);
    let outIdx = 0;

    // Prepare buffers
    let stringOut: string[] | undefined;
    let bigIntOut: BigInt64Array | undefined;
    let numericOut: Exclude<TypedArray, BigInt64Array> | undefined;

    if (Array.isArray(outData)) {
      stringOut = outData;
    } else if (outData instanceof BigInt64Array) {
      bigIntOut = outData;
    } else {
      numericOut = outData;
    }

    let stringSrc: readonly string[] | undefined;
    let numericSrc: Exclude<TypedArray, BigInt64Array> | undefined;
    if (Array.isArray(t.data)) {
      stringSrc = t.data;
    } else if (!(t.data instanceof BigInt64Array)) {
      numericSrc = t.data;
    }

    for (let i = 0; i < t.size; i++) {
      const srcIdx = contiguous
        ? t.offset + i
        : offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset);
      for (let r = 0; r < repeats; r++) {
        if (stringOut && stringSrc) {
          stringOut[outIdx++] = stringSrc[srcIdx] ?? "";
        } else if (bigIntOut && t.data instanceof BigInt64Array) {
          bigIntOut[outIdx++] = getBigIntElement(t.data, srcIdx);
        } else if (numericOut && numericSrc) {
          numericOut[outIdx++] = getNumericElement(numericSrc, srcIdx);
        }
      }
    }

    if (Array.isArray(outData)) {
      return Tensor.fromStringArray({
        data: outData,
        shape: [flatSize],
        device: t.device,
      });
    }

    if (t.dtype === "string") {
      throw new DeepboxError("Internal error: string dtype but non-array data");
    }

    return Tensor.fromTypedArray({
      data: outData,
      shape: [flatSize],
      dtype: t.dtype,
      device: t.device,
    });
  }

  // Repeat along specified axis
  const ax = normalizeAxis(axis, t.ndim);

  // Calculate output shape
  const outShape = [...t.shape];
  outShape[ax] = (t.shape[ax] ?? 0) * repeats;

  // Allocate output
  const outSize = shapeToSize(outShape);
  const outData =
    t.dtype === "string"
      ? new Array<string>(outSize)
      : new (dtypeToTypedArrayCtor(t.dtype))(outSize);

  // Compute strides
  const outStrides = computeStrides(outShape);

  // Prepare buffers
  let stringOut: string[] | undefined;
  let bigIntOut: BigInt64Array | undefined;
  let numericOut: Exclude<TypedArray, BigInt64Array> | undefined;

  if (Array.isArray(outData)) {
    stringOut = outData;
  } else if (outData instanceof BigInt64Array) {
    bigIntOut = outData;
  } else {
    numericOut = outData;
  }

  let stringSrc: readonly string[] | undefined;
  let numericSrc: Exclude<TypedArray, BigInt64Array> | undefined;
  if (Array.isArray(t.data)) {
    stringSrc = t.data;
  } else if (!(t.data instanceof BigInt64Array)) {
    numericSrc = t.data;
  }

  // Fill output
  for (let flatIdx = 0; flatIdx < outSize; flatIdx++) {
    // Convert to coordinates in output
    let rem = flatIdx;
    const outCoords = new Array<number>(t.ndim);
    for (let d = 0; d < t.ndim; d++) {
      const stride = outStrides[d] ?? 1;
      outCoords[d] = Math.floor(rem / stride);
      rem -= (outCoords[d] ?? 0) * stride;
    }

    // Map to input coordinates: divide by repeats on the repeat axis
    const inCoords = [...outCoords];
    inCoords[ax] = Math.floor((outCoords[ax] ?? 0) / repeats);

    // Convert to flat index in input
    let srcIdx = t.offset;
    for (let d = 0; d < t.ndim; d++) {
      srcIdx += (inCoords[d] ?? 0) * (t.strides[d] ?? 0);
    }

    // Copy element
    if (stringOut && stringSrc) {
      stringOut[flatIdx] = stringSrc[srcIdx] ?? "";
    } else if (bigIntOut && t.data instanceof BigInt64Array) {
      bigIntOut[flatIdx] = getBigIntElement(t.data, srcIdx);
    } else if (numericOut && numericSrc) {
      numericOut[flatIdx] = getNumericElement(numericSrc, srcIdx);
    }
  }

  if (Array.isArray(outData)) {
    return Tensor.fromStringArray({
      data: outData,
      shape: outShape,
      device: t.device,
    });
  }

  if (t.dtype === "string") {
    throw new DeepboxError("Internal error: string dtype but non-array data");
  }

  return Tensor.fromTypedArray({
    data: outData,
    shape: outShape,
    dtype: t.dtype,
    device: t.device,
  });
}
