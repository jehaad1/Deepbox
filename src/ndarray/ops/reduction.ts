import type { Axis, DType, Shape, TypedArray } from "../../core";
import {
  DataValidationError,
  DTypeError,
  dtypeToTypedArrayCtor,
  getArrayElement,
  getBigIntElement,
  getNumericElement,
  getShapeDim,
  InvalidParameterError,
  normalizeAxes,
  normalizeAxis,
  ShapeError,
  validateShape,
} from "../../core";
import { isContiguous } from "../tensor/strides";
import { computeStrides, Tensor } from "../tensor/Tensor";
import { bigintToNumberSafe, flatOffset } from "./_internal";
import { mulScalar } from "./arithmetic";

const INT64_MIN = -(1n << 63n);
const INT64_MAX = (1n << 63n) - 1n;
type NumericDType = Exclude<DType, "string">;

function ensureNumericTensor(t: Tensor, op: string): asserts t is Tensor<Shape, NumericDType> {
  if (t.dtype === "string") {
    throw new DTypeError(`${op}() not supported for string dtype`);
  }
}

function expectBigIntData(data: Tensor["data"], op: string): BigInt64Array {
  if (Array.isArray(data)) {
    throw new DTypeError(`${op} not supported for string dtype`);
  }
  if (!(data instanceof BigInt64Array)) {
    throw new DTypeError(`${op} requires int64 dtype`);
  }
  return data;
}

function expectNumericData(data: Tensor["data"], op: string): Exclude<TypedArray, BigInt64Array> {
  if (Array.isArray(data)) {
    throw new DTypeError(`${op} not supported for string dtype`);
  }
  if (data instanceof BigInt64Array) {
    throw new DTypeError(`${op} requires non-int64 dtype`);
  }
  return data;
}

function outDtypeForSum(inDtype: Tensor["dtype"]): NumericDType {
  if (inDtype === "int64") return "int64";
  if (inDtype === "float32" || inDtype === "float64") return "float64";
  if (inDtype === "string") throw new DTypeError("sum is not defined for string dtype");
  return "int32";
}

/**
 * Sum reduction.
 *
 * Supported (initial foundation):
 * - `axis` omitted: sum all elements to a scalar tensor
 * - `axis` provided: supports any axis
 *
 * Output dtype:
 * - `int64` stays `int64`
 * - `float32` and `float64` promote to `float64`
 * - `int32`/`uint8`/`bool` promote to `int32`
 * - `string` is unsupported
 */
export function sum(t: Tensor, axis?: Axis, keepdims = false): Tensor {
  ensureNumericTensor(t, "sum");

  const outDtype = outDtypeForSum(t.dtype);

  if (axis === undefined) {
    const logicalStrides = computeStrides(t.shape);
    const contiguous = isContiguous(t.shape, t.strides);
    if (t.dtype === "int64") {
      const bigIntData = expectBigIntData(t.data, "sum");
      let acc = 0n;
      for (let i = 0; i < t.size; i++) {
        const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
        acc += getBigIntElement(bigIntData, srcOffset);
      }
      if (acc < INT64_MIN || acc > INT64_MAX) {
        throw new DataValidationError("int64 sum overflow");
      }
      const out = new BigInt64Array(1);
      out[0] = acc;
      const outShape: Shape = keepdims ? new Array<number>(t.ndim).fill(1) : [];
      return Tensor.fromTypedArray({
        data: out,
        shape: outShape,
        dtype: "int64",
        device: t.device,
      });
    }

    let acc = 0;
    if (t.data instanceof BigInt64Array) {
      for (let i = 0; i < t.size; i++) {
        const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
        acc += Number(getBigIntElement(t.data, srcOffset));
      }
    } else {
      const numericData = t.data;
      if (Array.isArray(numericData)) {
        throw new DTypeError("sum not supported for string dtype");
      }
      // Fast path: contiguous zero-offset — direct TypedArray loop
      if (contiguous && t.offset === 0) {
        for (let i = 0; i < t.size; i++) {
          acc += numericData[i] as number;
        }
      } else {
        for (let i = 0; i < t.size; i++) {
          const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
          acc += getNumericElement(numericData, srcOffset);
        }
      }
    }

    const out = outDtype === "float64" ? new Float64Array(1) : new Int32Array(1);
    out[0] = acc;
    const outShape: Shape = keepdims ? new Array<number>(t.ndim).fill(1) : [];

    return Tensor.fromTypedArray({
      data: out,
      shape: outShape,
      dtype: outDtype,
      device: t.device,
    });
  }

  const ax = normalizeAxis(axis, t.ndim);
  const axisDim = getShapeDim(t.shape, ax);

  // Empty-axis reduction: return zeros (additive identity) with correct output shape
  if (axisDim === 0) {
    const outShapeArr: number[] = [];
    for (let i = 0; i < t.ndim; i++) {
      if (i === ax) {
        if (keepdims) outShapeArr.push(1);
      } else {
        outShapeArr.push(getShapeDim(t.shape, i));
      }
    }
    validateShape(outShapeArr);
    const outSize = outShapeArr.reduce((a, b) => a * b, 1);
    const OutCtor = dtypeToTypedArrayCtor(outDtype);
    const out = new OutCtor(outSize);
    // Typed arrays are zero-initialized, which is the additive identity
    return Tensor.fromTypedArray({
      data: out,
      shape: outShapeArr,
      dtype: outDtype,
      device: t.device,
    });
  }

  const outShapeArr: number[] = [];
  for (let i = 0; i < t.ndim; i++) {
    if (i === ax) {
      if (keepdims) outShapeArr.push(1);
    } else {
      outShapeArr.push(getShapeDim(t.shape, i));
    }
  }
  validateShape(outShapeArr);

  const outSize = outShapeArr.reduce((a, b) => a * b, 1);
  const OutCtor = dtypeToTypedArrayCtor(outDtype);
  const out = new OutCtor(outSize);

  // Precompute output strides for converting outFlat -> outIdx
  const outStrides = new Array<number>(outShapeArr.length);
  let stride = 1;
  for (let i = outShapeArr.length - 1; i >= 0; i--) {
    outStrides[i] = stride;
    stride *= getArrayElement(outShapeArr, i);
  }

  for (let outFlat = 0; outFlat < outSize; outFlat++) {
    let rem = outFlat;
    const outIdx = new Array<number>(outShapeArr.length);
    for (let i = 0; i < outShapeArr.length; i++) {
      const s = getArrayElement(outStrides, i, 1);
      outIdx[i] = Math.floor(rem / s);
      rem %= s;
    }

    // Map outIdx back to an input base index (with reduced axis set to 0).
    const inIdx = new Array<number>(t.ndim);
    if (keepdims) {
      for (let i = 0; i < t.ndim; i++) {
        if (i === ax) {
          inIdx[i] = 0;
        } else {
          inIdx[i] = getArrayElement(outIdx, i);
        }
      }
    } else {
      let outAxis = 0;
      for (let i = 0; i < t.ndim; i++) {
        if (i === ax) {
          inIdx[i] = 0;
        } else {
          inIdx[i] = getArrayElement(outIdx, outAxis);
          outAxis++;
        }
      }
    }

    let baseOffset = t.offset;
    for (let i = 0; i < t.ndim; i++) {
      baseOffset += getArrayElement(inIdx, i) * getArrayElement(t.strides, i);
    }

    const axisStride = getArrayElement(t.strides, ax);
    if (outDtype === "int64") {
      const bigIntData = expectBigIntData(t.data, "sum");
      if (!(out instanceof BigInt64Array)) {
        throw new DTypeError("sum output dtype mismatch");
      }
      const bigIntOut = out;
      let acc = 0n;
      for (let k = 0; k < axisDim; k++) {
        acc += getBigIntElement(bigIntData, baseOffset + k * axisStride);
      }
      if (acc < INT64_MIN || acc > INT64_MAX) {
        throw new DataValidationError("int64 sum overflow");
      }
      bigIntOut[outFlat] = acc;
    } else {
      let acc = 0;
      if (t.data instanceof BigInt64Array) {
        for (let k = 0; k < axisDim; k++) {
          acc += Number(getBigIntElement(t.data, baseOffset + k * axisStride));
        }
      } else {
        const numericData = expectNumericData(t.data, "sum");
        for (let k = 0; k < axisDim; k++) {
          acc += getNumericElement(numericData, baseOffset + k * axisStride);
        }
      }

      if (out instanceof BigInt64Array) {
        throw new DTypeError("sum output dtype mismatch");
      }
      const numericOut = out;
      numericOut[outFlat] = acc;
    }
  }

  const finalShape: Shape = outShapeArr;
  return Tensor.fromTypedArray({
    data: out,
    shape: finalShape,
    dtype: outDtype,
    device: t.device,
  });
}

/**
 * Compute the arithmetic mean along the specified axis.
 *
 * Returns NaN for empty reductions (when the number of elements along the
 * reduction axis is 0), matching NumPy's behavior of returning NaN with a
 * "mean of empty slice" warning.
 *
 * @param t - Input tensor
 * @param axis - Axis along which to compute mean. If undefined, compute over all elements
 * @param keepdims - If true, keep reduced dimensions as size 1
 * @returns Tensor containing mean values. NaN for empty reductions.
 *
 * @example
 * ```ts
 * import { mean, tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([[1, 2], [3, 4]]);
 * const m1 = mean(x);           // 2.5
 * const m2 = mean(x, 0);        // [2, 3]
 * const m3 = mean(x, 1);        // [1.5, 3.5]
 * ```
 *
 * @see {@link https://numpy.org/doc/stable/reference/generated/numpy.mean.html | NumPy mean}
 */
export function mean(t: Tensor, axis?: Axis, keepdims = false): Tensor {
  ensureNumericTensor(t, "mean");

  // Empty full-reduction: return NaN scalar (matches NumPy behavior)
  if (axis === undefined && t.size === 0) {
    const out = new Float64Array(1);
    out[0] = NaN;
    const outShape: Shape = keepdims ? new Array<number>(t.ndim).fill(1) : [];
    return Tensor.fromTypedArray({
      data: out,
      shape: outShape,
      dtype: "float64",
      device: t.device,
    });
  }

  // Empty axis-reduction: return NaN-filled tensor (matches NumPy behavior)
  if (axis !== undefined) {
    const ax = normalizeAxis(axis, t.ndim);
    const axisDim = getShapeDim(t.shape, ax);
    if (axisDim === 0) {
      const outShapeArr: number[] = [];
      for (let i = 0; i < t.ndim; i++) {
        if (i === ax) {
          if (keepdims) outShapeArr.push(1);
        } else {
          outShapeArr.push(getShapeDim(t.shape, i));
        }
      }
      validateShape(outShapeArr);
      const outSize = outShapeArr.reduce((a, b) => a * b, 1);
      const out = new Float64Array(outSize);
      out.fill(NaN);
      return Tensor.fromTypedArray({
        data: out,
        shape: outShapeArr,
        dtype: "float64",
        device: t.device,
      });
    }
  }

  // Compute the sum along the specified axis (or all elements if axis is undefined)
  const total = sum(t, axis, keepdims);

  // Ensure floating output for integer/bool/uint8 inputs to avoid truncation.
  let totalFloat = total;
  if (total.dtype !== "float32" && total.dtype !== "float64") {
    const data = new Float64Array(total.size);
    const totalData = total.data;
    if (Array.isArray(totalData)) {
      throw new DTypeError("mean not supported for string dtype");
    }
    if (totalData instanceof BigInt64Array) {
      for (let i = 0; i < total.size; i++) {
        data[i] = bigintToNumberSafe(getBigIntElement(totalData, total.offset + i));
      }
    } else {
      for (let i = 0; i < total.size; i++) {
        data[i] = getNumericElement(totalData, total.offset + i);
      }
    }
    totalFloat = Tensor.fromTypedArray({
      data,
      shape: total.shape,
      dtype: "float64",
      device: total.device,
    });
  }

  if (axis === undefined) {
    // Divide total by the total number of elements to get the mean
    return mulScalar(totalFloat, 1 / t.size);
  }

  // Normalize negative axis indices to positive
  const ax = normalizeAxis(axis, t.ndim);
  const axisDim = getShapeDim(t.shape, ax);

  // Divide total by the size of the reduced axis to get the mean
  return mulScalar(totalFloat, 1 / axisDim);
}

/**
 * Product of array elements along axis.
 *
 * Multiplies all elements together. Returns 1 for empty arrays.
 * Supports full reduction and reduction over one or more axes.
 *
 * **Complexity**: O(n) where n is the number of elements
 *
 * Output dtype:
 * - Same as input dtype (int32, float64, int64)
 * - Potential for overflow with large products
 *
 * @param t - Input tensor
 * @param axis - Axis or axes along which to compute the product
 * @param keepdims - If true, keep reduced dimensions as size 1
 * @returns Scalar tensor containing the product
 *
 * @example
 * ```ts
 * import { tensor, prod } from 'deepbox/ndarray';
 *
 * const t = tensor([1, 2, 3, 4]);
 * prod(t);  // tensor(24) - 1*2*3*4
 *
 * const t2 = tensor([2.5, 4.0]);
 * prod(t2);  // tensor(10.0)
 * ```
 *
 * @see {@link https://numpy.org/doc/stable/reference/generated/numpy.prod.html | NumPy prod}
 */
export function prod(t: Tensor, axis?: Axis | Axis[], keepdims = false): Tensor {
  ensureNumericTensor(t, "prod");

  const prodAxis = (
    input: Tensor<Shape, NumericDType>,
    ax: number,
    keep: boolean
  ): Tensor<Shape, NumericDType> => {
    const axisDim = getShapeDim(input.shape, ax);

    const outShapeArr: number[] = [];
    for (let i = 0; i < input.ndim; i++) {
      if (i === ax) {
        if (keep) outShapeArr.push(1);
      } else {
        outShapeArr.push(getShapeDim(input.shape, i));
      }
    }
    validateShape(outShapeArr);

    const outSize = outShapeArr.reduce((a, b) => a * b, 1);

    let outData: TypedArray | BigInt64Array;
    if (input.data instanceof BigInt64Array) {
      outData = new BigInt64Array(outSize);
    } else {
      const Ctor = dtypeToTypedArrayCtor(input.dtype);
      outData = new Ctor(outSize);
    }

    const outStrides = new Array<number>(outShapeArr.length);
    let stride = 1;
    for (let i = outShapeArr.length - 1; i >= 0; i--) {
      outStrides[i] = stride;
      stride *= getArrayElement(outShapeArr, i);
    }

    for (let outFlat = 0; outFlat < outSize; outFlat++) {
      let rem = outFlat;
      const outIdx = new Array<number>(outShapeArr.length);
      for (let i = 0; i < outShapeArr.length; i++) {
        const s = getArrayElement(outStrides, i, 1);
        outIdx[i] = Math.floor(rem / s);
        rem %= s;
      }

      const inIdx = new Array<number>(input.ndim);
      if (keep) {
        for (let i = 0; i < input.ndim; i++) {
          inIdx[i] = i === ax ? 0 : getArrayElement(outIdx, i);
        }
      } else {
        let outAxis = 0;
        for (let i = 0; i < input.ndim; i++) {
          if (i === ax) {
            inIdx[i] = 0;
          } else {
            inIdx[i] = getArrayElement(outIdx, outAxis);
            outAxis++;
          }
        }
      }

      let baseOffset = input.offset;
      for (let i = 0; i < input.ndim; i++) {
        baseOffset += getArrayElement(inIdx, i) * getArrayElement(input.strides, i);
      }

      const axisStride = getArrayElement(input.strides, ax);
      if (input.data instanceof BigInt64Array) {
        let acc = 1n;
        if (axisDim > 0) {
          for (let k = 0; k < axisDim; k++) {
            acc *= getBigIntElement(input.data, baseOffset + k * axisStride);
          }
        }
        if (acc < INT64_MIN || acc > INT64_MAX) {
          throw new DataValidationError("int64 prod overflow");
        }
        if (!(outData instanceof BigInt64Array)) {
          throw new DTypeError("prod output dtype mismatch");
        }
        outData[outFlat] = acc;
      } else {
        let acc = 1;
        if (axisDim > 0) {
          const numericData = input.data;
          if (Array.isArray(numericData)) {
            throw new DTypeError("prod not supported for string dtype");
          }
          for (let k = 0; k < axisDim; k++) {
            acc *= getNumericElement(numericData, baseOffset + k * axisStride);
          }
        }
        if (outData instanceof BigInt64Array) {
          throw new DTypeError("prod output dtype mismatch");
        }
        outData[outFlat] = acc;
      }
    }

    if (outData instanceof BigInt64Array) {
      return Tensor.fromTypedArray({
        data: outData,
        shape: outShapeArr,
        dtype: "int64",
        device: input.device,
      });
    }
    return Tensor.fromTypedArray({
      data: outData,
      shape: outShapeArr,
      dtype: input.dtype,
      device: input.device,
    });
  };

  if (axis !== undefined) {
    const axes = normalizeAxes(axis, t.ndim);
    const sorted = keepdims
      ? axes.slice().sort((a, b) => a - b)
      : axes.slice().sort((a, b) => b - a);
    let result: Tensor<Shape, NumericDType> = t;
    for (const ax of sorted) {
      result = prodAxis(result, ax, keepdims);
    }
    return result;
  }

  // Handle empty tensor - return 1 (multiplicative identity)
  if (t.size === 0) {
    const emptyShape: Shape = [];
    if (t.dtype === "int64") {
      const out = new BigInt64Array(1);
      out[0] = 1n;
      return Tensor.fromTypedArray({
        data: out,
        shape: emptyShape,
        dtype: "int64",
        device: t.device,
      });
    }
    const Ctor = dtypeToTypedArrayCtor(t.dtype);
    const out = new Ctor(1);
    out[0] = 1;
    return Tensor.fromTypedArray({
      data: out,
      shape: emptyShape,
      dtype: t.dtype,
      device: t.device,
    });
  }

  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  // Handle BigInt separately (int64 dtype)
  if (t.data instanceof BigInt64Array) {
    let acc = 1n;

    // Multiply all elements
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      acc *= getBigIntElement(t.data, srcOffset);
    }

    // Check for overflow (simplified check)
    if (acc < INT64_MIN || acc > INT64_MAX) {
      throw new DataValidationError("int64 prod overflow");
    }

    const out = new BigInt64Array(1);
    out[0] = acc;
    const outShape: Shape = keepdims ? new Array<number>(t.ndim).fill(1) : [];

    return Tensor.fromTypedArray({
      data: out,
      shape: outShape,
      dtype: "int64",
      device: t.device,
    });
  }

  // Handle numeric types (float64, int32, etc.)
  const numericData = t.data;
  if (Array.isArray(numericData)) {
    throw new DTypeError("prod not supported for string dtype");
  }
  let acc = 1;

  // Multiply all elements
  for (let i = 0; i < t.size; i++) {
    const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
    acc *= getNumericElement(numericData, srcOffset);
  }

  // Create output array based on input dtype
  const Ctor = dtypeToTypedArrayCtor(t.dtype);
  const out = new Ctor(1);
  if (t.dtype === "int32") {
    out[0] = Math.floor(acc);
  } else {
    out[0] = acc;
  }

  const outShape: Shape = keepdims ? new Array<number>(t.ndim).fill(1) : [];
  return Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: t.dtype,
    device: t.device,
  });
}

/**
 * Standard deviation along axis.
 *
 * Computes the standard deviation, a measure of the spread of a distribution.
 * Uses the formula: sqrt(variance)
 *
 * **Complexity**: O(n) where n is the number of elements (requires 2 passes)
 *
 * @param t - Input tensor
 * @param axis - Axis or axes along which to compute std. If undefined, compute over all elements
 * @param keepdims - If true, keep reduced dimensions as size 1
 * @param ddof - Delta degrees of freedom (default 0 for population, 1 for sample)
 * @returns Tensor containing the standard deviation values
 *
 * @example
 * ```ts
 * import { tensor, std } from 'deepbox/ndarray';
 *
 * const t = tensor([1, 2, 3, 4, 5]);
 * std(t);  // tensor(1.414...) - population std
 * std(t, undefined, false, 1);  // sample std
 *
 * const t2 = tensor([[1, 2], [3, 4]]);
 * std(t2, 0);  // [1, 1] - std along rows
 * std(t2, 1);  // [0.5, 0.5] - std along columns
 * ```
 *
 * @see {@link https://numpy.org/doc/stable/reference/generated/numpy.std.html | NumPy std}
 */
export function std(t: Tensor, axis?: Axis, keepdims = false, ddof = 0): Tensor {
  // Compute variance first, then take square root
  const varianceTensor = variance(t, axis, keepdims, ddof);

  // Apply sqrt element-wise to variance
  const varianceData = expectNumericData(varianceTensor.data, "std");
  const out = new Float64Array(varianceTensor.size);

  for (let i = 0; i < varianceTensor.size; i++) {
    out[i] = Math.sqrt(getNumericElement(varianceData, varianceTensor.offset + i));
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: varianceTensor.shape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Variance along axis.
 *
 * Computes the variance, the average of squared deviations from the mean.
 * Formula: Σ(x - mean)² / (N - ddof)
 *
 * **Complexity**: O(n) where n is the number of elements (requires 2 passes: one for mean, one for variance)
 *
 * @param t - Input tensor
 * @param axis - Axis along which to compute variance. If undefined, compute over all elements
 * @param keepdims - If true, keep reduced dimensions as size 1
 * @param ddof - Delta degrees of freedom (default 0 for population, 1 for sample)
 * @returns Tensor containing the variance values
 *
 * @example
 * ```ts
 * import { tensor, variance } from 'deepbox/ndarray';
 *
 * const t = tensor([1, 2, 3, 4, 5]);
 * variance(t);  // tensor(2.0) - population variance
 * variance(t, undefined, false, 1);  // tensor(2.5) - sample variance
 *
 * const t2 = tensor([[1, 2], [3, 4]]);
 * variance(t2, 0);  // [1, 1] - variance along rows
 * variance(t2, 1);  // [0.25, 0.25] - variance along columns
 * ```
 *
 * @see {@link https://numpy.org/doc/stable/reference/generated/numpy.var.html | NumPy var}
 */
export function variance(t: Tensor, axis?: Axis, keepdims = false, ddof = 0): Tensor {
  ensureNumericTensor(t, "variance");

  if (ddof < 0) {
    throw new InvalidParameterError("ddof must be non-negative", "ddof", ddof);
  }

  // Need at least one element
  if (t.size === 0) {
    throw new InvalidParameterError("variance() requires at least one element", "t");
  }

  // Full reduction case
  if (axis === undefined) {
    const logicalStrides = computeStrides(t.shape);
    const contiguous = isContiguous(t.shape, t.strides);
    // Check ddof validity
    if (t.size <= ddof) {
      throw new InvalidParameterError(
        `ddof=${ddof} >= size=${t.size}, variance undefined`,
        "ddof",
        ddof
      );
    }

    // Pass 1: Compute mean
    let meanValue = 0;
    if (t.data instanceof BigInt64Array) {
      let sum = 0;
      for (let i = 0; i < t.size; i++) {
        const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
        sum += bigintToNumberSafe(getBigIntElement(t.data, srcOffset));
      }
      meanValue = sum / t.size;
    } else {
      const numericData = t.data;
      if (Array.isArray(numericData)) {
        throw new DTypeError("variance not supported for string dtype");
      }
      let sum = 0;
      for (let i = 0; i < t.size; i++) {
        const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
        sum += getNumericElement(numericData, srcOffset);
      }
      meanValue = sum / t.size;
    }

    // Pass 2: Compute sum of squared deviations
    let sumSquaredDev = 0;
    if (t.data instanceof BigInt64Array) {
      for (let i = 0; i < t.size; i++) {
        const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
        const val = bigintToNumberSafe(getBigIntElement(t.data, srcOffset));
        const deviation = val - meanValue;
        sumSquaredDev += deviation * deviation;
      }
    } else {
      const numericData = t.data;
      if (Array.isArray(numericData)) {
        throw new DTypeError("variance not supported for string dtype");
      }
      for (let i = 0; i < t.size; i++) {
        const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
        const val = getNumericElement(numericData, srcOffset);
        const deviation = val - meanValue;
        sumSquaredDev += deviation * deviation;
      }
    }

    // Compute variance: divide by (N - ddof)
    const varianceValue = sumSquaredDev / (t.size - ddof);

    const out = new Float64Array(1);
    out[0] = varianceValue;
    const outShape: Shape = keepdims ? new Array<number>(t.ndim).fill(1) : [];

    return Tensor.fromTypedArray({
      data: out,
      shape: outShape,
      dtype: "float64",
      device: t.device,
    });
  }

  // Axis-wise reduction
  const ax = normalizeAxis(axis, t.ndim);
  const axisDim = getShapeDim(t.shape, ax);

  if (axisDim === 0) {
    const outShapeArr: number[] = [];
    for (let i = 0; i < t.ndim; i++) {
      if (i === ax) {
        if (keepdims) outShapeArr.push(1);
      } else {
        outShapeArr.push(getShapeDim(t.shape, i));
      }
    }
    validateShape(outShapeArr);
    const outSize = outShapeArr.reduce((a, b) => a * b, 1);
    const out = new Float64Array(outSize);
    out.fill(NaN);
    return Tensor.fromTypedArray({
      data: out,
      shape: outShapeArr,
      dtype: "float64",
      device: t.device,
    });
  }

  // Check ddof validity for axis reduction
  if (axisDim <= ddof) {
    throw new InvalidParameterError(
      `ddof=${ddof} >= axis size=${axisDim}, variance undefined`,
      "ddof",
      ddof
    );
  }

  // Build output shape
  const outShapeArr: number[] = [];
  for (let i = 0; i < t.ndim; i++) {
    if (i === ax) {
      if (keepdims) outShapeArr.push(1);
    } else {
      outShapeArr.push(getShapeDim(t.shape, i));
    }
  }
  validateShape(outShapeArr);

  const outSize = outShapeArr.reduce((a, b) => a * b, 1);
  const out = new Float64Array(outSize);

  // Precompute output strides
  const outStrides = new Array<number>(outShapeArr.length);
  let stride = 1;
  for (let i = outShapeArr.length - 1; i >= 0; i--) {
    outStrides[i] = stride;
    stride *= getArrayElement(outShapeArr, i);
  }

  // For each output position, compute variance along the axis
  for (let outFlat = 0; outFlat < outSize; outFlat++) {
    // Convert flat to multi-index
    let rem = outFlat;
    const outIdx = new Array<number>(outShapeArr.length);
    for (let i = 0; i < outShapeArr.length; i++) {
      const s = getArrayElement(outStrides, i, 1);
      outIdx[i] = Math.floor(rem / s);
      rem %= s;
    }

    // Map outIdx back to an input base index (with reduced axis set to 0)
    const inIdx = new Array<number>(t.ndim);
    if (keepdims) {
      for (let i = 0; i < t.ndim; i++) {
        inIdx[i] = i === ax ? 0 : getArrayElement(outIdx, i);
      }
    } else {
      let outAxis = 0;
      for (let i = 0; i < t.ndim; i++) {
        if (i === ax) {
          inIdx[i] = 0;
        } else {
          inIdx[i] = getArrayElement(outIdx, outAxis);
          outAxis++;
        }
      }
    }

    let baseOffset = t.offset;
    for (let i = 0; i < t.ndim; i++) {
      baseOffset += getArrayElement(inIdx, i) * getArrayElement(t.strides, i);
    }

    const axisStride = getArrayElement(t.strides, ax);

    // Pass 1: Compute mean along axis
    let meanValue = 0;
    if (t.data instanceof BigInt64Array) {
      let sum = 0;
      for (let k = 0; k < axisDim; k++) {
        sum += bigintToNumberSafe(getBigIntElement(t.data, baseOffset + k * axisStride));
      }
      meanValue = sum / axisDim;
    } else {
      const numericData = t.data;
      if (Array.isArray(numericData)) {
        throw new DTypeError("variance not supported for string dtype");
      }
      let sum = 0;
      for (let k = 0; k < axisDim; k++) {
        sum += getNumericElement(numericData, baseOffset + k * axisStride);
      }
      meanValue = sum / axisDim;
    }

    // Pass 2: Compute sum of squared deviations
    let sumSquaredDev = 0;
    if (t.data instanceof BigInt64Array) {
      for (let k = 0; k < axisDim; k++) {
        const val = bigintToNumberSafe(getBigIntElement(t.data, baseOffset + k * axisStride));
        const deviation = val - meanValue;
        sumSquaredDev += deviation * deviation;
      }
    } else {
      const numericData = t.data;
      if (Array.isArray(numericData)) {
        throw new DTypeError("variance not supported for string dtype");
      }
      for (let k = 0; k < axisDim; k++) {
        const val = getNumericElement(numericData, baseOffset + k * axisStride);
        const deviation = val - meanValue;
        sumSquaredDev += deviation * deviation;
      }
    }

    // Compute variance: divide by (axisDim - ddof)
    out[outFlat] = sumSquaredDev / (axisDim - ddof);
  }

  const finalShape: Shape = outShapeArr;
  return Tensor.fromTypedArray({
    data: out,
    shape: finalShape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Minimum value along axis.
 *
 * Finds the smallest element in the tensor, optionally along one or more axes.
 *
 * **Complexity**: O(n) where n is the number of elements
 *
 * @param t - Input tensor
 * @param axis - Axis or axes along which to compute the minimum
 * @param keepdims - If true, keep reduced dimensions as size 1
 * @returns Scalar tensor containing the minimum value
 *
 * @example
 * ```ts
 * import { tensor, min } from 'deepbox/ndarray';
 *
 * const t = tensor([3, 1, 4, 1, 5]);
 * min(t);  // tensor(1)
 * ```
 *
 * @see {@link https://numpy.org/doc/stable/reference/generated/numpy.amin.html | NumPy amin}
 */
export function min(t: Tensor, axis?: Axis | Axis[], keepdims = false): Tensor {
  ensureNumericTensor(t, "min");

  const minAxis = (
    input: Tensor<Shape, NumericDType>,
    ax: number,
    keep: boolean
  ): Tensor<Shape, NumericDType> => {
    const axisDim = getShapeDim(input.shape, ax);
    if (axisDim === 0) {
      throw new InvalidParameterError("min() requires at least one element", "t");
    }

    const outShapeArr: number[] = [];
    for (let i = 0; i < input.ndim; i++) {
      if (i === ax) {
        if (keep) outShapeArr.push(1);
      } else {
        outShapeArr.push(getShapeDim(input.shape, i));
      }
    }
    validateShape(outShapeArr);

    const outSize = outShapeArr.reduce((a, b) => a * b, 1);
    let outData: TypedArray | BigInt64Array;
    if (input.data instanceof BigInt64Array) {
      outData = new BigInt64Array(outSize);
    } else {
      const Ctor = dtypeToTypedArrayCtor(input.dtype);
      outData = new Ctor(outSize);
    }

    const outStrides = new Array<number>(outShapeArr.length);
    let stride = 1;
    for (let i = outShapeArr.length - 1; i >= 0; i--) {
      outStrides[i] = stride;
      stride *= getArrayElement(outShapeArr, i);
    }

    for (let outFlat = 0; outFlat < outSize; outFlat++) {
      let rem = outFlat;
      const outIdx = new Array<number>(outShapeArr.length);
      for (let i = 0; i < outShapeArr.length; i++) {
        const s = getArrayElement(outStrides, i, 1);
        outIdx[i] = Math.floor(rem / s);
        rem %= s;
      }

      const inIdx = new Array<number>(input.ndim);
      if (keep) {
        for (let i = 0; i < input.ndim; i++) {
          inIdx[i] = i === ax ? 0 : getArrayElement(outIdx, i);
        }
      } else {
        let outAxis = 0;
        for (let i = 0; i < input.ndim; i++) {
          if (i === ax) {
            inIdx[i] = 0;
          } else {
            inIdx[i] = getArrayElement(outIdx, outAxis);
            outAxis++;
          }
        }
      }

      let baseOffset = input.offset;
      for (let i = 0; i < input.ndim; i++) {
        baseOffset += getArrayElement(inIdx, i) * getArrayElement(input.strides, i);
      }

      const axisStride = getArrayElement(input.strides, ax);
      if (input.data instanceof BigInt64Array) {
        let minVal = getBigIntElement(input.data, baseOffset);
        for (let k = 1; k < axisDim; k++) {
          const val = getBigIntElement(input.data, baseOffset + k * axisStride);
          if (val < minVal) minVal = val;
        }
        if (!(outData instanceof BigInt64Array)) {
          throw new DTypeError("min output dtype mismatch");
        }
        outData[outFlat] = minVal;
      } else {
        const numericData = input.data;
        if (Array.isArray(numericData)) {
          throw new DTypeError("min not supported for string dtype");
        }
        let minVal = getNumericElement(numericData, baseOffset);
        for (let k = 1; k < axisDim; k++) {
          const val = getNumericElement(numericData, baseOffset + k * axisStride);
          if (val < minVal || Number.isNaN(val)) minVal = val;
        }
        if (outData instanceof BigInt64Array) {
          throw new DTypeError("min output dtype mismatch");
        }
        outData[outFlat] = minVal;
      }
    }

    if (outData instanceof BigInt64Array) {
      return Tensor.fromTypedArray({
        data: outData,
        shape: outShapeArr,
        dtype: "int64",
        device: input.device,
      });
    }
    return Tensor.fromTypedArray({
      data: outData,
      shape: outShapeArr,
      dtype: input.dtype,
      device: input.device,
    });
  };

  if (axis !== undefined) {
    const axes = normalizeAxes(axis, t.ndim);
    const sorted = keepdims
      ? axes.slice().sort((a, b) => a - b)
      : axes.slice().sort((a, b) => b - a);
    let result: Tensor<Shape, NumericDType> = t;
    for (const ax of sorted) {
      result = minAxis(result, ax, keepdims);
    }
    return result;
  }

  // Need at least one element
  if (t.size === 0) {
    throw new InvalidParameterError("min() requires at least one element", "t");
  }

  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  // Handle BigInt separately (int64 dtype)
  if (t.data instanceof BigInt64Array) {
    let minVal = getBigIntElement(
      t.data,
      flatOffset(0, t.offset, contiguous, logicalStrides, t.strides)
    );

    // Find minimum value
    for (let i = 1; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = getBigIntElement(t.data, srcOffset);
      if (val < minVal) {
        minVal = val;
      }
    }

    const out = new BigInt64Array(1);
    out[0] = minVal;
    const outShape: Shape = keepdims ? new Array<number>(t.ndim).fill(1) : [];

    return Tensor.fromTypedArray({
      data: out,
      shape: outShape,
      dtype: "int64",
      device: t.device,
    });
  }

  // Handle numeric types (float64, int32, etc.)
  const numericData = expectNumericData(t.data, "min");

  let minVal: number;
  // Fast path: contiguous zero-offset — direct TypedArray loop
  if (contiguous && t.offset === 0) {
    minVal = numericData[0] as number;
    for (let i = 1; i < t.size; i++) {
      const val = numericData[i] as number;
      if (val < minVal || Number.isNaN(val)) {
        minVal = val;
      }
    }
  } else {
    minVal = getNumericElement(
      numericData,
      flatOffset(0, t.offset, contiguous, logicalStrides, t.strides)
    );
    // Find minimum value (propagate NaN to match NumPy amin behavior)
    for (let i = 1; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = getNumericElement(numericData, srcOffset);
      if (val < minVal || Number.isNaN(val)) {
        minVal = val;
      }
    }
  }

  // Create output array based on input dtype
  const Ctor = dtypeToTypedArrayCtor(t.dtype);
  const out = new Ctor(1);
  out[0] = minVal;

  const outShape: Shape = keepdims ? new Array<number>(t.ndim).fill(1) : [];
  return Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: t.dtype,
    device: t.device,
  });
}

/**
 * Maximum value along axis.
 *
 * Finds the largest element in the tensor, optionally along one or more axes.
 *
 * **Complexity**: O(n) where n is the number of elements
 *
 * @param t - Input tensor
 * @param axis - Axis or axes along which to compute the maximum
 * @param keepdims - If true, keep reduced dimensions as size 1
 * @returns Scalar tensor containing the maximum value
 *
 * @example
 * ```ts
 * import { tensor, max } from 'deepbox/ndarray';
 *
 * const t = tensor([3, 1, 4, 1, 5]);
 * max(t);  // tensor(5)
 * ```
 *
 * @see {@link https://numpy.org/doc/stable/reference/generated/numpy.amax.html | NumPy amax}
 */
export function max(t: Tensor, axis?: Axis | Axis[], keepdims = false): Tensor {
  ensureNumericTensor(t, "max");

  const maxAxis = (
    input: Tensor<Shape, NumericDType>,
    ax: number,
    keep: boolean
  ): Tensor<Shape, NumericDType> => {
    const axisDim = getShapeDim(input.shape, ax);
    if (axisDim === 0) {
      throw new InvalidParameterError("max() requires at least one element", "t");
    }

    const outShapeArr: number[] = [];
    for (let i = 0; i < input.ndim; i++) {
      if (i === ax) {
        if (keep) outShapeArr.push(1);
      } else {
        outShapeArr.push(getShapeDim(input.shape, i));
      }
    }
    validateShape(outShapeArr);

    const outSize = outShapeArr.reduce((a, b) => a * b, 1);
    let outData: TypedArray | BigInt64Array;
    if (input.data instanceof BigInt64Array) {
      outData = new BigInt64Array(outSize);
    } else {
      const Ctor = dtypeToTypedArrayCtor(input.dtype);
      outData = new Ctor(outSize);
    }

    const outStrides = new Array<number>(outShapeArr.length);
    let stride = 1;
    for (let i = outShapeArr.length - 1; i >= 0; i--) {
      outStrides[i] = stride;
      stride *= getArrayElement(outShapeArr, i);
    }

    for (let outFlat = 0; outFlat < outSize; outFlat++) {
      let rem = outFlat;
      const outIdx = new Array<number>(outShapeArr.length);
      for (let i = 0; i < outShapeArr.length; i++) {
        const s = getArrayElement(outStrides, i, 1);
        outIdx[i] = Math.floor(rem / s);
        rem %= s;
      }

      const inIdx = new Array<number>(input.ndim);
      if (keep) {
        for (let i = 0; i < input.ndim; i++) {
          inIdx[i] = i === ax ? 0 : getArrayElement(outIdx, i);
        }
      } else {
        let outAxis = 0;
        for (let i = 0; i < input.ndim; i++) {
          if (i === ax) {
            inIdx[i] = 0;
          } else {
            inIdx[i] = getArrayElement(outIdx, outAxis);
            outAxis++;
          }
        }
      }

      let baseOffset = input.offset;
      for (let i = 0; i < input.ndim; i++) {
        baseOffset += getArrayElement(inIdx, i) * getArrayElement(input.strides, i);
      }

      const axisStride = getArrayElement(input.strides, ax);
      if (input.data instanceof BigInt64Array) {
        let maxVal = getBigIntElement(input.data, baseOffset);
        for (let k = 1; k < axisDim; k++) {
          const val = getBigIntElement(input.data, baseOffset + k * axisStride);
          if (val > maxVal) maxVal = val;
        }
        if (!(outData instanceof BigInt64Array)) {
          throw new DTypeError("max output dtype mismatch");
        }
        outData[outFlat] = maxVal;
      } else {
        const numericData = expectNumericData(input.data, "max");
        let maxVal = getNumericElement(numericData, baseOffset);
        for (let k = 1; k < axisDim; k++) {
          const val = getNumericElement(numericData, baseOffset + k * axisStride);
          if (val > maxVal || Number.isNaN(val)) maxVal = val;
        }
        if (outData instanceof BigInt64Array) {
          throw new DTypeError("max output dtype mismatch");
        }
        outData[outFlat] = maxVal;
      }
    }

    if (outData instanceof BigInt64Array) {
      return Tensor.fromTypedArray({
        data: outData,
        shape: outShapeArr,
        dtype: "int64",
        device: input.device,
      });
    }
    return Tensor.fromTypedArray({
      data: outData,
      shape: outShapeArr,
      dtype: input.dtype,
      device: input.device,
    });
  };

  if (axis !== undefined) {
    const axes = normalizeAxes(axis, t.ndim);
    const sorted = keepdims
      ? axes.slice().sort((a, b) => a - b)
      : axes.slice().sort((a, b) => b - a);
    let result: Tensor<Shape, NumericDType> = t;
    for (const ax of sorted) {
      result = maxAxis(result, ax, keepdims);
    }
    return result;
  }

  // Need at least one element
  if (t.size === 0) {
    throw new InvalidParameterError("max() requires at least one element", "t");
  }

  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  // Handle BigInt separately (int64 dtype)
  if (t.data instanceof BigInt64Array) {
    let maxVal = getBigIntElement(
      t.data,
      flatOffset(0, t.offset, contiguous, logicalStrides, t.strides)
    );

    // Find maximum value
    for (let i = 1; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = getBigIntElement(t.data, srcOffset);
      if (val > maxVal) {
        maxVal = val;
      }
    }

    const out = new BigInt64Array(1);
    out[0] = maxVal;
    const outShape: Shape = keepdims ? new Array<number>(t.ndim).fill(1) : [];

    return Tensor.fromTypedArray({
      data: out,
      shape: outShape,
      dtype: "int64",
      device: t.device,
    });
  }

  // Handle numeric types (float64, int32, etc.)
  const numericData = expectNumericData(t.data, "max");

  let maxVal: number;
  // Fast path: contiguous zero-offset — direct TypedArray loop
  if (contiguous && t.offset === 0) {
    maxVal = numericData[0] as number;
    for (let i = 1; i < t.size; i++) {
      const val = numericData[i] as number;
      if (val > maxVal || Number.isNaN(val)) {
        maxVal = val;
      }
    }
  } else {
    maxVal = getNumericElement(
      numericData,
      flatOffset(0, t.offset, contiguous, logicalStrides, t.strides)
    );
    // Find maximum value (propagate NaN to match NumPy amax behavior)
    for (let i = 1; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      const val = getNumericElement(numericData, srcOffset);
      if (val > maxVal || Number.isNaN(val)) {
        maxVal = val;
      }
    }
  }

  // Create output array based on input dtype
  const Ctor = dtypeToTypedArrayCtor(t.dtype);
  const out = new Ctor(1);
  out[0] = maxVal;

  const outShape: Shape = keepdims ? new Array<number>(t.ndim).fill(1) : [];
  return Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: t.dtype,
    device: t.device,
  });
}

/**
 * Median value along axis.
 *
 * Computes the median (middle value) of the data.
 * For even-sized arrays, returns the average of the two middle values.
 *
 * **Complexity**: O(n log n) due to sorting per output element
 *
 * @param t - Input tensor
 * @param axis - Axis along which to compute median. If undefined, compute over all elements
 * @param keepdims - If true, keep reduced dimensions as size 1
 * @returns Tensor containing the median values
 *
 * @example
 * ```ts
 * import { tensor, median } from 'deepbox/ndarray';
 *
 * const t = tensor([1, 3, 5, 7, 9]);
 * median(t);  // tensor(5) - middle value
 *
 * const t2 = tensor([1, 2, 3, 4]);
 * median(t2);  // tensor(2.5) - average of 2 and 3
 *
 * const t3 = tensor([[1, 3], [2, 4]]);
 * median(t3, 0);  // [1.5, 3.5] - median along rows
 * median(t3, 1);  // [2, 3] - median along columns
 * ```
 *
 * Performance:
 * - This implementation copies and sorts values (O(n log n) time, O(n) memory).
 *
 * @see {@link https://numpy.org/doc/stable/reference/generated/numpy.median.html | NumPy median}
 */
export function median(t: Tensor, axis?: Axis, keepdims = false): Tensor {
  ensureNumericTensor(t, "median");

  // Need at least one element
  if (t.size === 0) {
    throw new InvalidParameterError("median() requires at least one element", "t");
  }

  // Full reduction case
  if (axis === undefined) {
    const logicalStrides = computeStrides(t.shape);
    const contiguous = isContiguous(t.shape, t.strides);
    // Copy data to avoid mutating original (must not use in-place sort)
    const sorted: number[] = [];

    if (t.data instanceof BigInt64Array) {
      for (let i = 0; i < t.size; i++) {
        const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
        sorted.push(Number(getBigIntElement(t.data, srcOffset)));
      }
    } else {
      const numericData = expectNumericData(t.data, "median");
      for (let i = 0; i < t.size; i++) {
        const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
        sorted.push(getNumericElement(numericData, srcOffset));
      }
    }

    sorted.sort((a, b) => a - b);

    let medianValue: number;
    const mid = Math.floor(sorted.length / 2);

    if (sorted.length % 2 === 0) {
      medianValue = (getArrayElement(sorted, mid - 1) + getArrayElement(sorted, mid)) / 2;
    } else {
      medianValue = getArrayElement(sorted, mid);
    }

    const out = new Float64Array(1);
    out[0] = medianValue;
    const outShape: Shape = keepdims ? new Array<number>(t.ndim).fill(1) : [];

    return Tensor.fromTypedArray({
      data: out,
      shape: outShape,
      dtype: "float64",
      device: t.device,
    });
  }

  // Axis-wise reduction
  const ax = normalizeAxis(axis, t.ndim);
  const axisDim = getShapeDim(t.shape, ax);

  if (axisDim === 0) {
    throw new ShapeError("Internal error: missing axis dimension");
  }

  // Build output shape
  const outShapeArr: number[] = [];
  for (let i = 0; i < t.ndim; i++) {
    if (i === ax) {
      if (keepdims) outShapeArr.push(1);
    } else {
      outShapeArr.push(getShapeDim(t.shape, i));
    }
  }
  validateShape(outShapeArr);

  const outSize = outShapeArr.reduce((a, b) => a * b, 1);
  const out = new Float64Array(outSize);

  // Precompute output strides
  const outStrides = new Array<number>(outShapeArr.length);
  let stride = 1;
  for (let i = outShapeArr.length - 1; i >= 0; i--) {
    outStrides[i] = stride;
    stride *= getArrayElement(outShapeArr, i);
  }

  // For each output position, compute median along the axis
  for (let outFlat = 0; outFlat < outSize; outFlat++) {
    // Convert flat to multi-index
    let rem = outFlat;
    const outIdx = new Array<number>(outShapeArr.length);
    for (let i = 0; i < outShapeArr.length; i++) {
      const s = getArrayElement(outStrides, i, 1);
      outIdx[i] = Math.floor(rem / s);
      rem %= s;
    }

    // Map outIdx back to an input base index (with reduced axis set to 0)
    const inIdx = new Array<number>(t.ndim);
    if (keepdims) {
      for (let i = 0; i < t.ndim; i++) {
        inIdx[i] = i === ax ? 0 : getArrayElement(outIdx, i);
      }
    } else {
      let outAxis = 0;
      for (let i = 0; i < t.ndim; i++) {
        if (i === ax) {
          inIdx[i] = 0;
        } else {
          inIdx[i] = getArrayElement(outIdx, outAxis);
          outAxis++;
        }
      }
    }

    let baseOffset = t.offset;
    for (let i = 0; i < t.ndim; i++) {
      baseOffset += getArrayElement(inIdx, i) * getArrayElement(t.strides, i);
    }

    const axisStride = getArrayElement(t.strides, ax);

    // Collect values along axis
    const sorted: number[] = [];
    if (t.data instanceof BigInt64Array) {
      for (let k = 0; k < axisDim; k++) {
        sorted.push(Number(getBigIntElement(t.data, baseOffset + k * axisStride)));
      }
    } else {
      const numericData = expectNumericData(t.data, "median");
      for (let k = 0; k < axisDim; k++) {
        sorted.push(getNumericElement(numericData, baseOffset + k * axisStride));
      }
    }

    // Sort and compute median
    sorted.sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);

    if (sorted.length % 2 === 0) {
      out[outFlat] = (getArrayElement(sorted, mid - 1) + getArrayElement(sorted, mid)) / 2;
    } else {
      out[outFlat] = getArrayElement(sorted, mid);
    }
  }

  const finalShape: Shape = outShapeArr;
  return Tensor.fromTypedArray({
    data: out,
    shape: finalShape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Cumulative sum along axis.
 *
 * Returns an array of the same shape where each element is the sum of all
 * previous elements (inclusive) along the specified axis.
 *
 * **Complexity**: O(n) where n is the number of elements
 *
 * @param t - Input tensor
 * @param axis - Axis along which to compute cumulative sum. If undefined, operates on the flattened array.
 * @returns Tensor of same shape with cumulative sums
 *
 * @example
 * ```ts
 * import { tensor, cumsum } from 'deepbox/ndarray';
 *
 * const t = tensor([1, 2, 3, 4]);
 * cumsum(t);  // tensor([1, 3, 6, 10])
 * ```
 *
 * @see {@link https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html | NumPy cumsum}
 */
export function cumsum(t: Tensor, axis?: Axis): Tensor {
  // String dtype not supported
  if (t.dtype === "string") {
    throw new DTypeError("cumsum() not supported for string dtype");
  }

  if (axis !== undefined) {
    const ax = normalizeAxis(axis, t.ndim);
    const axisDim = getShapeDim(t.shape, ax);

    const outStrides = new Array<number>(t.ndim);
    let stride = 1;
    for (let i = t.ndim - 1; i >= 0; i--) {
      outStrides[i] = stride;
      stride *= getShapeDim(t.shape, i);
    }

    const baseShape: number[] = [];
    for (let i = 0; i < t.ndim; i++) {
      if (i !== ax) baseShape.push(getShapeDim(t.shape, i));
    }
    const baseSize = baseShape.reduce((a, b) => a * b, 1);

    if (t.data instanceof BigInt64Array) {
      const out = new BigInt64Array(t.size);
      for (let baseFlat = 0; baseFlat < baseSize; baseFlat++) {
        let rem = baseFlat;
        const baseIdx = new Array<number>(baseShape.length);
        for (let i = 0; i < baseShape.length; i++) {
          const s = baseShape.slice(i + 1).reduce((a, b) => a * b, 1);
          baseIdx[i] = baseShape.length === 0 ? 0 : Math.floor(rem / s);
          rem = baseShape.length === 0 ? 0 : rem % s;
        }

        const inIdx = new Array<number>(t.ndim);
        let outAxis = 0;
        for (let i = 0; i < t.ndim; i++) {
          if (i === ax) {
            inIdx[i] = 0;
          } else {
            inIdx[i] = baseIdx[outAxis] ?? 0;
            outAxis++;
          }
        }

        let inBaseOffset = t.offset;
        let outBaseOffset = 0;
        for (let i = 0; i < t.ndim; i++) {
          inBaseOffset += getArrayElement(inIdx, i) * getArrayElement(t.strides, i);
          outBaseOffset += getArrayElement(inIdx, i) * getArrayElement(outStrides, i);
        }

        const axisStride = getArrayElement(t.strides, ax);
        const outAxisStride = getArrayElement(outStrides, ax);
        let acc = 0n;
        for (let k = 0; k < axisDim; k++) {
          acc += getBigIntElement(t.data, inBaseOffset + k * axisStride);
          out[outBaseOffset + k * outAxisStride] = acc;
        }
      }

      return Tensor.fromTypedArray({
        data: out,
        shape: t.shape,
        dtype: "int64",
        device: t.device,
      });
    }

    const numericData = expectNumericData(t.data, "cumsum");
    const out = new Float64Array(t.size);
    for (let baseFlat = 0; baseFlat < baseSize; baseFlat++) {
      let rem = baseFlat;
      const baseIdx = new Array<number>(baseShape.length);
      for (let i = 0; i < baseShape.length; i++) {
        const s = baseShape.slice(i + 1).reduce((a, b) => a * b, 1);
        baseIdx[i] = baseShape.length === 0 ? 0 : Math.floor(rem / s);
        rem = baseShape.length === 0 ? 0 : rem % s;
      }

      const inIdx = new Array<number>(t.ndim);
      let outAxis = 0;
      for (let i = 0; i < t.ndim; i++) {
        if (i === ax) {
          inIdx[i] = 0;
        } else {
          inIdx[i] = baseIdx[outAxis] ?? 0;
          outAxis++;
        }
      }

      let inBaseOffset = t.offset;
      let outBaseOffset = 0;
      for (let i = 0; i < t.ndim; i++) {
        inBaseOffset += getArrayElement(inIdx, i) * getArrayElement(t.strides, i);
        outBaseOffset += getArrayElement(inIdx, i) * getArrayElement(outStrides, i);
      }

      const axisStride = getArrayElement(t.strides, ax);
      const outAxisStride = getArrayElement(outStrides, ax);
      let acc = 0;
      for (let k = 0; k < axisDim; k++) {
        acc += getNumericElement(numericData, inBaseOffset + k * axisStride);
        out[outBaseOffset + k * outAxisStride] = acc;
      }
    }

    return Tensor.fromTypedArray({
      data: out,
      shape: t.shape,
      dtype: "float64",
      device: t.device,
    });
  }

  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  // Handle BigInt separately (int64 dtype)
  if (t.data instanceof BigInt64Array) {
    const out = new BigInt64Array(t.size);
    let acc = 0n;

    // Compute cumulative sum
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      acc += getBigIntElement(t.data, srcOffset);
      out[i] = acc;
    }

    return Tensor.fromTypedArray({
      data: out,
      shape: t.shape,
      dtype: "int64",
      device: t.device,
    });
  }

  // Handle numeric types (float64, int32, etc.)
  // Always use float64 for cumsum to avoid overflow
  const numericData = expectNumericData(t.data, "cumsum");
  const out = new Float64Array(t.size);
  let acc = 0;

  // Compute cumulative sum
  for (let i = 0; i < t.size; i++) {
    const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
    acc += getNumericElement(numericData, srcOffset);
    out[i] = acc;
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Cumulative product along axis.
 *
 * Returns an array of the same shape where each element is the product of all
 * previous elements (inclusive) along the specified axis.
 *
 * **Complexity**: O(n) where n is the number of elements
 *
 * @param t - Input tensor
 * @param axis - Axis along which to compute cumulative product. If undefined, operates on the flattened array.
 * @returns Tensor of same shape with cumulative products
 *
 * @example
 * ```ts
 * import { tensor, cumprod } from 'deepbox/ndarray';
 *
 * const t = tensor([1, 2, 3, 4]);
 * cumprod(t);  // tensor([1, 2, 6, 24])
 * ```
 *
 * @see {@link https://numpy.org/doc/stable/reference/generated/numpy.cumprod.html | NumPy cumprod}
 */
export function cumprod(t: Tensor, axis?: Axis): Tensor {
  // String dtype not supported
  if (t.dtype === "string") {
    throw new DTypeError("cumprod() not supported for string dtype");
  }

  if (axis !== undefined) {
    const ax = normalizeAxis(axis, t.ndim);
    const axisDim = getShapeDim(t.shape, ax);

    const outStrides = new Array<number>(t.ndim);
    let stride = 1;
    for (let i = t.ndim - 1; i >= 0; i--) {
      outStrides[i] = stride;
      stride *= getShapeDim(t.shape, i);
    }

    const baseShape: number[] = [];
    for (let i = 0; i < t.ndim; i++) {
      if (i !== ax) baseShape.push(getShapeDim(t.shape, i));
    }
    const baseSize = baseShape.reduce((a, b) => a * b, 1);

    if (t.data instanceof BigInt64Array) {
      const out = new BigInt64Array(t.size);
      for (let baseFlat = 0; baseFlat < baseSize; baseFlat++) {
        let rem = baseFlat;
        const baseIdx = new Array<number>(baseShape.length);
        for (let i = 0; i < baseShape.length; i++) {
          const s = baseShape.slice(i + 1).reduce((a, b) => a * b, 1);
          baseIdx[i] = baseShape.length === 0 ? 0 : Math.floor(rem / s);
          rem = baseShape.length === 0 ? 0 : rem % s;
        }

        const inIdx = new Array<number>(t.ndim);
        let outAxis = 0;
        for (let i = 0; i < t.ndim; i++) {
          if (i === ax) {
            inIdx[i] = 0;
          } else {
            inIdx[i] = baseIdx[outAxis] ?? 0;
            outAxis++;
          }
        }

        let inBaseOffset = t.offset;
        let outBaseOffset = 0;
        for (let i = 0; i < t.ndim; i++) {
          inBaseOffset += getArrayElement(inIdx, i) * getArrayElement(t.strides, i);
          outBaseOffset += getArrayElement(inIdx, i) * getArrayElement(outStrides, i);
        }

        const axisStride = getArrayElement(t.strides, ax);
        const outAxisStride = getArrayElement(outStrides, ax);
        let acc = 1n;
        for (let k = 0; k < axisDim; k++) {
          acc *= getBigIntElement(t.data, inBaseOffset + k * axisStride);
          out[outBaseOffset + k * outAxisStride] = acc;
        }
      }

      return Tensor.fromTypedArray({
        data: out,
        shape: t.shape,
        dtype: "int64",
        device: t.device,
      });
    }

    const numericData = expectNumericData(t.data, "cumprod");
    const out = new Float64Array(t.size);
    for (let baseFlat = 0; baseFlat < baseSize; baseFlat++) {
      let rem = baseFlat;
      const baseIdx = new Array<number>(baseShape.length);
      for (let i = 0; i < baseShape.length; i++) {
        const s = baseShape.slice(i + 1).reduce((a, b) => a * b, 1);
        baseIdx[i] = baseShape.length === 0 ? 0 : Math.floor(rem / s);
        rem = baseShape.length === 0 ? 0 : rem % s;
      }

      const inIdx = new Array<number>(t.ndim);
      let outAxis = 0;
      for (let i = 0; i < t.ndim; i++) {
        if (i === ax) {
          inIdx[i] = 0;
        } else {
          inIdx[i] = baseIdx[outAxis] ?? 0;
          outAxis++;
        }
      }

      let inBaseOffset = t.offset;
      let outBaseOffset = 0;
      for (let i = 0; i < t.ndim; i++) {
        inBaseOffset += getArrayElement(inIdx, i) * getArrayElement(t.strides, i);
        outBaseOffset += getArrayElement(inIdx, i) * getArrayElement(outStrides, i);
      }

      const axisStride = getArrayElement(t.strides, ax);
      const outAxisStride = getArrayElement(outStrides, ax);
      let acc = 1;
      for (let k = 0; k < axisDim; k++) {
        acc *= getNumericElement(numericData, inBaseOffset + k * axisStride);
        out[outBaseOffset + k * outAxisStride] = acc;
      }
    }

    return Tensor.fromTypedArray({
      data: out,
      shape: t.shape,
      dtype: "float64",
      device: t.device,
    });
  }

  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  // Handle BigInt separately (int64 dtype)
  if (t.data instanceof BigInt64Array) {
    const out = new BigInt64Array(t.size);
    let acc = 1n;

    // Compute cumulative product
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      acc *= getBigIntElement(t.data, srcOffset);
      out[i] = acc;
    }

    return Tensor.fromTypedArray({
      data: out,
      shape: t.shape,
      dtype: "int64",
      device: t.device,
    });
  }

  // Handle numeric types (float64, int32, etc.)
  // Always use float64 for cumprod to avoid overflow
  const numericData = expectNumericData(t.data, "cumprod");
  const out = new Float64Array(t.size);
  let acc = 1;

  // Compute cumulative product
  for (let i = 0; i < t.size; i++) {
    const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
    acc *= getNumericElement(numericData, srcOffset);
    out[i] = acc;
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: t.shape,
    dtype: "float64",
    device: t.device,
  });
}

/**
 * Calculate differences along axis.
 *
 * Computes the n-th discrete difference along the given axis.
 * The first difference is given by out[i] = a[i+1] - a[i] along the flattened array.
 * Computes differences along the specified axis (default: last axis).
 *
 * **Complexity**: O(n) where n is the number of elements
 *
 * @param t - Input tensor
 * @param n - Number of times to take difference (default 1)
 * @param axis - Axis along which to compute differences (default: last axis)
 * @returns Tensor with differences (size reduced by n along the given axis)
 *
 * @example
 * ```ts
 * import { tensor, diff } from 'deepbox/ndarray';
 *
 * const t = tensor([1, 3, 6, 10]);
 * diff(t);  // tensor([2, 3, 4]) - differences between consecutive elements
 * ```
 *
 * @see {@link https://numpy.org/doc/stable/reference/generated/numpy.diff.html | NumPy diff}
 */
export function diff(t: Tensor, n = 1, axis = -1): Tensor {
  // String dtype not supported
  if (t.dtype === "string") {
    throw new DTypeError("diff() not supported for string dtype");
  }

  if (n < 0) {
    throw new InvalidParameterError("n must be >= 0", "n", n);
  }

  if (n === 0) {
    return t;
  }

  const ax = normalizeAxis(axis, t.ndim);

  const diffOnce = (input: Tensor, axisIdx: number): Tensor => {
    const axisDim = getShapeDim(input.shape, axisIdx);
    const outShape = [...input.shape];
    outShape[axisIdx] = Math.max(axisDim - 1, 0);
    validateShape(outShape);
    const outSize = outShape.reduce((a, b) => a * b, 1);

    if (input.data instanceof BigInt64Array) {
      const outData = new BigInt64Array(outSize);
      if (outSize === 0) {
        return Tensor.fromTypedArray({
          data: outData,
          shape: outShape,
          dtype: "int64",
          device: input.device,
        });
      }

      const outStrides = new Array<number>(outShape.length);
      let stride = 1;
      for (let i = outShape.length - 1; i >= 0; i--) {
        outStrides[i] = stride;
        stride *= outShape[i] ?? 1;
      }
      const inStrides = input.strides;

      for (let outFlat = 0; outFlat < outSize; outFlat++) {
        let rem = outFlat;
        const outIdx = new Array<number>(input.ndim);
        for (let i = 0; i < input.ndim; i++) {
          const stride = outStrides[i] ?? 1;
          outIdx[i] = Math.floor(rem / stride);
          rem -= (outIdx[i] ?? 0) * stride;
        }

        const currIdx = outIdx[axisIdx] ?? 0;
        const nextIdx = currIdx + 1;
        let baseOffset = input.offset;
        for (let i = 0; i < input.ndim; i++) {
          baseOffset += (outIdx[i] ?? 0) * (inStrides[i] ?? 1);
        }

        const axisStride = inStrides[axisIdx] ?? 1;
        const curr = getBigIntElement(input.data, baseOffset);
        const next = getBigIntElement(input.data, baseOffset + (nextIdx - currIdx) * axisStride);
        outData[outFlat] = next - curr;
      }

      return Tensor.fromTypedArray({
        data: outData,
        shape: outShape,
        dtype: "int64",
        device: input.device,
      });
    }

    const outData = new Float64Array(outSize);
    if (outSize === 0) {
      return Tensor.fromTypedArray({
        data: outData,
        shape: outShape,
        dtype: "float64",
        device: input.device,
      });
    }

    const outStrides = new Array<number>(outShape.length);
    let stride = 1;
    for (let i = outShape.length - 1; i >= 0; i--) {
      outStrides[i] = stride;
      stride *= outShape[i] ?? 1;
    }
    const inStrides = input.strides;
    const numericData = expectNumericData(input.data, "diff");

    for (let outFlat = 0; outFlat < outSize; outFlat++) {
      let rem = outFlat;
      const outIdx = new Array<number>(input.ndim);
      for (let i = 0; i < input.ndim; i++) {
        const stride = outStrides[i] ?? 1;
        outIdx[i] = Math.floor(rem / stride);
        rem -= (outIdx[i] ?? 0) * stride;
      }

      const currIdx = outIdx[axisIdx] ?? 0;
      const nextIdx = currIdx + 1;
      let baseOffset = input.offset;
      for (let i = 0; i < input.ndim; i++) {
        baseOffset += (outIdx[i] ?? 0) * (inStrides[i] ?? 1);
      }

      const axisStride = inStrides[axisIdx] ?? 1;
      const curr = getNumericElement(numericData, baseOffset);
      const next = getNumericElement(numericData, baseOffset + (nextIdx - currIdx) * axisStride);
      outData[outFlat] = next - curr;
    }

    return Tensor.fromTypedArray({
      data: outData,
      shape: outShape,
      dtype: "float64",
      device: input.device,
    });
  };

  let result = t;
  for (let i = 0; i < n; i++) {
    result = diffOnce(result, ax);
  }
  return result;
}

/**
 * Test whether any array element evaluates to True (non-zero).
 *
 * Returns true if at least one element is non-zero, false if all are zero.
 * Supports reduction over one or more axes.
 *
 * **Complexity**: O(n) - checks each element until finding non-zero or reaching end
 *
 * @param t - Input tensor
 * @param axis - Axis or axes along which to compute the reduction
 * @param keepdims - If true, keep reduced dimensions as size 1
 * @returns Scalar tensor (true=1, false=0)
 *
 * @example
 * ```ts
 * const t = tensor([0, 0, 1, 0]);
 * any(t);  // tensor(1) - at least one non-zero
 *
 * const t2 = tensor([0, 0, 0]);
 * any(t2);  // tensor(0) - all zeros
 * ```
 *
 * @see {@link https://numpy.org/doc/stable/reference/generated/numpy.any.html | NumPy any}
 */
export function any(t: Tensor, axis?: number | number[], keepdims = false): Tensor {
  ensureNumericTensor(t, "any");

  const anyAxis = (input: Tensor, ax: number, keep: boolean): Tensor => {
    const axisDim = getShapeDim(input.shape, ax);
    const outShapeArr: number[] = [];
    for (let i = 0; i < input.ndim; i++) {
      if (i === ax) {
        if (keep) outShapeArr.push(1);
      } else {
        outShapeArr.push(getShapeDim(input.shape, i));
      }
    }
    validateShape(outShapeArr);

    const outSize = outShapeArr.reduce((a, b) => a * b, 1);
    const outData = new Uint8Array(outSize);

    const outStrides = new Array<number>(outShapeArr.length);
    let stride = 1;
    for (let i = outShapeArr.length - 1; i >= 0; i--) {
      outStrides[i] = stride;
      stride *= getArrayElement(outShapeArr, i);
    }

    for (let outFlat = 0; outFlat < outSize; outFlat++) {
      let rem = outFlat;
      const outIdx = new Array<number>(outShapeArr.length);
      for (let i = 0; i < outShapeArr.length; i++) {
        const s = getArrayElement(outStrides, i, 1);
        outIdx[i] = Math.floor(rem / s);
        rem %= s;
      }

      const inIdx = new Array<number>(input.ndim);
      if (keep) {
        for (let i = 0; i < input.ndim; i++) {
          inIdx[i] = i === ax ? 0 : getArrayElement(outIdx, i);
        }
      } else {
        let outAxis = 0;
        for (let i = 0; i < input.ndim; i++) {
          if (i === ax) {
            inIdx[i] = 0;
          } else {
            inIdx[i] = getArrayElement(outIdx, outAxis);
            outAxis++;
          }
        }
      }

      if (axisDim === 0) {
        outData[outFlat] = 0;
        continue;
      }

      let baseOffset = input.offset;
      for (let i = 0; i < input.ndim; i++) {
        baseOffset += getArrayElement(inIdx, i) * getArrayElement(input.strides, i);
      }

      const axisStride = getArrayElement(input.strides, ax);
      let result = 0;
      if (input.data instanceof BigInt64Array) {
        for (let k = 0; k < axisDim; k++) {
          if (getBigIntElement(input.data, baseOffset + k * axisStride) !== 0n) {
            result = 1;
            break;
          }
        }
      } else {
        const numericData = expectNumericData(input.data, "any");
        for (let k = 0; k < axisDim; k++) {
          if (getNumericElement(numericData, baseOffset + k * axisStride) !== 0) {
            result = 1;
            break;
          }
        }
      }
      outData[outFlat] = result;
    }

    return Tensor.fromTypedArray({
      data: outData,
      shape: outShapeArr,
      dtype: "bool",
      device: input.device,
    });
  };

  if (axis !== undefined) {
    const axes = normalizeAxes(axis, t.ndim);
    const sorted = keepdims
      ? axes.slice().sort((a, b) => a - b)
      : axes.slice().sort((a, b) => b - a);
    let result: Tensor = t;
    for (const ax of sorted) {
      result = anyAxis(result, ax, keepdims);
    }
    return result;
  }

  // Check if any element is non-zero (truthy)
  let result = false;
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  if (t.data instanceof BigInt64Array) {
    // BigInt path: check for any non-zero bigint
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      if (getBigIntElement(t.data, srcOffset) !== 0n) {
        result = true;
        break; // Early exit optimization - found non-zero
      }
    }
  } else {
    // Number path: check for any non-zero number
    const numericData = expectNumericData(t.data, "any");
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      if (getNumericElement(numericData, srcOffset) !== 0) {
        result = true;
        break; // Early exit optimization
      }
    }
  }

  // Return scalar boolean tensor (0 or 1)
  const out = new Uint8Array(1);
  out[0] = result ? 1 : 0;
  const outShape: Shape = keepdims ? new Array<number>(t.ndim).fill(1) : [];

  return Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "bool",
    device: t.device,
  });
}

/**
 * Test whether all array elements evaluate to True (non-zero).
 *
 * Returns true only if all elements are non-zero, false if any are zero.
 * Supports reduction over one or more axes.
 *
 * **Complexity**: O(n) - checks each element until finding zero or reaching end
 *
 * @param t - Input tensor
 * @param axis - Axis or axes along which to compute the reduction
 * @param keepdims - If true, keep reduced dimensions as size 1
 * @returns Scalar tensor (true=1, false=0)
 *
 * @example
 * ```ts
 * const t = tensor([1, 2, 3]);
 * all(t);  // tensor(1) - all non-zero
 *
 * const t2 = tensor([1, 0, 3]);
 * all(t2);  // tensor(0) - has a zero
 * ```
 *
 * @see {@link https://numpy.org/doc/stable/reference/generated/numpy.all.html | NumPy all}
 */
export function all(t: Tensor, axis?: number | number[], keepdims = false): Tensor {
  ensureNumericTensor(t, "all");

  const allAxis = (input: Tensor, ax: number, keep: boolean): Tensor => {
    const axisDim = getShapeDim(input.shape, ax);
    const outShapeArr: number[] = [];
    for (let i = 0; i < input.ndim; i++) {
      if (i === ax) {
        if (keep) outShapeArr.push(1);
      } else {
        outShapeArr.push(getShapeDim(input.shape, i));
      }
    }
    validateShape(outShapeArr);

    const outSize = outShapeArr.reduce((a, b) => a * b, 1);
    const outData = new Uint8Array(outSize);

    const outStrides = new Array<number>(outShapeArr.length);
    let stride = 1;
    for (let i = outShapeArr.length - 1; i >= 0; i--) {
      outStrides[i] = stride;
      stride *= getArrayElement(outShapeArr, i);
    }

    for (let outFlat = 0; outFlat < outSize; outFlat++) {
      let rem = outFlat;
      const outIdx = new Array<number>(outShapeArr.length);
      for (let i = 0; i < outShapeArr.length; i++) {
        const s = getArrayElement(outStrides, i, 1);
        outIdx[i] = Math.floor(rem / s);
        rem %= s;
      }

      const inIdx = new Array<number>(input.ndim);
      if (keep) {
        for (let i = 0; i < input.ndim; i++) {
          inIdx[i] = i === ax ? 0 : getArrayElement(outIdx, i);
        }
      } else {
        let outAxis = 0;
        for (let i = 0; i < input.ndim; i++) {
          if (i === ax) {
            inIdx[i] = 0;
          } else {
            inIdx[i] = getArrayElement(outIdx, outAxis);
            outAxis++;
          }
        }
      }

      if (axisDim === 0) {
        outData[outFlat] = 1;
        continue;
      }

      let baseOffset = input.offset;
      for (let i = 0; i < input.ndim; i++) {
        baseOffset += getArrayElement(inIdx, i) * getArrayElement(input.strides, i);
      }

      const axisStride = getArrayElement(input.strides, ax);
      let result = 1;
      if (input.data instanceof BigInt64Array) {
        for (let k = 0; k < axisDim; k++) {
          if (getBigIntElement(input.data, baseOffset + k * axisStride) === 0n) {
            result = 0;
            break;
          }
        }
      } else {
        const numericData = expectNumericData(input.data, "all");
        for (let k = 0; k < axisDim; k++) {
          if (getNumericElement(numericData, baseOffset + k * axisStride) === 0) {
            result = 0;
            break;
          }
        }
      }
      outData[outFlat] = result;
    }

    return Tensor.fromTypedArray({
      data: outData,
      shape: outShapeArr,
      dtype: "bool",
      device: input.device,
    });
  };

  if (axis !== undefined) {
    const axes = normalizeAxes(axis, t.ndim);
    const sorted = keepdims
      ? axes.slice().sort((a, b) => a - b)
      : axes.slice().sort((a, b) => b - a);
    let result: Tensor = t;
    for (const ax of sorted) {
      result = allAxis(result, ax, keepdims);
    }
    return result;
  }

  // Check if all elements are non-zero (truthy)
  let result = true;
  const logicalStrides = computeStrides(t.shape);
  const contiguous = isContiguous(t.shape, t.strides);

  if (t.data instanceof BigInt64Array) {
    // BigInt path: check for any zero bigint
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      if (getBigIntElement(t.data, srcOffset) === 0n) {
        result = false;
        break; // Early exit optimization - found zero
      }
    }
  } else {
    // Number path: check for any zero number
    const numericData = expectNumericData(t.data, "all");
    for (let i = 0; i < t.size; i++) {
      const srcOffset = flatOffset(i, t.offset, contiguous, logicalStrides, t.strides);
      if (getNumericElement(numericData, srcOffset) === 0) {
        result = false;
        break; // Early exit optimization
      }
    }
  }

  // Return scalar boolean tensor (0 or 1)
  const out = new Uint8Array(1);
  out[0] = result ? 1 : 0;
  const outShape: Shape = keepdims ? new Array<number>(t.ndim).fill(1) : [];

  return Tensor.fromTypedArray({
    data: out,
    shape: outShape,
    dtype: "bool",
    device: t.device,
  });
}
