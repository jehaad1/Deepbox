/**
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { DTypeError, ShapeError } from "../../core";
import type { AnyTensor } from "../../ndarray";

type NumericArray = Float32Array | Float64Array | Int32Array | Uint8Array;

function isNumericArray(data: unknown): data is NumericArray {
  return (
    data instanceof Float64Array ||
    data instanceof Float32Array ||
    data instanceof Int32Array ||
    data instanceof Uint8Array
  );
}

function readElement(data: NumericArray, index: number): number {
  const v = data[index];
  if (v === undefined) {
    throw new ShapeError(`Tensor view reads index ${index} beyond its storage (${data.length})`);
  }
  return v;
}

/**
 * Converts a 1D tensor to Float64Array. The result is always a fresh copy, so callers may keep
 * or mutate it without affecting the tensor. `int64` values are converted to the nearest double.
 * @throws {ShapeError} If the tensor is not 1D.
 * @throws {DTypeError} If the tensor holds strings.
 * @internal
 */
export function tensorToFloat64Vector1D(t: AnyTensor): Float64Array {
  // Extract underlying tensor from GradTensor if needed
  const tensor = "tensor" in t ? t.tensor : t;

  if (tensor.ndim !== 1) {
    throw new ShapeError(`Expected a 1D tensor; received ndim=${tensor.ndim}`);
  }
  if (tensor.dtype === "string") throw new DTypeError("Plotting does not support string tensors");

  const n = tensor.shape[0] ?? 0;
  const stride = tensor.strides[0] ?? 0;
  const out = new Float64Array(n);
  const base = tensor.offset;
  const data: unknown = tensor.data;

  if (isNumericArray(data)) {
    if (stride === 1 && base + n <= data.length) {
      out.set(data.subarray(base, base + n));
      return out;
    }
    for (let i = 0; i < n; i++) {
      out[i] = readElement(data, base + i * stride);
    }
    return out;
  }

  const values = data as ArrayLike<unknown>;
  for (let i = 0; i < n; i++) {
    out[i] = safeConvertToNumber(values[base + i * stride]);
  }

  return out;
}

/**
 * Converts a 2D tensor to a row-major Float64Array. The result is always a fresh copy.
 * @throws {ShapeError} If the tensor is not 2D.
 * @throws {DTypeError} If the tensor holds strings.
 * @internal
 */
export function tensorToFloat64Matrix2D(t: AnyTensor): {
  readonly rows: number;
  readonly cols: number;
  readonly data: Float64Array;
} {
  // Extract underlying tensor from GradTensor if needed
  const tensor = "tensor" in t ? t.tensor : t;

  if (tensor.ndim !== 2) {
    throw new ShapeError(`Expected a 2D tensor; received ndim=${tensor.ndim}`);
  }
  if (tensor.dtype === "string") throw new DTypeError("Plotting does not support string tensors");

  const rows = tensor.shape[0] ?? 0;
  const cols = tensor.shape[1] ?? 0;
  const strideRow = tensor.strides[0] ?? 0;
  const strideCol = tensor.strides[1] ?? 0;
  const out = new Float64Array(rows * cols);
  const base = tensor.offset;
  const data: unknown = tensor.data;

  if (isNumericArray(data)) {
    if (strideCol === 1 && strideRow === cols && base + rows * cols <= data.length) {
      out.set(data.subarray(base, base + rows * cols));
      return { rows, cols, data: out };
    }
    for (let i = 0; i < rows; i++) {
      const rowBase = base + i * strideRow;
      for (let j = 0; j < cols; j++) {
        out[i * cols + j] = readElement(data, rowBase + j * strideCol);
      }
    }
    return { rows, cols, data: out };
  }

  const values = data as ArrayLike<unknown>;
  for (let i = 0; i < rows; i++) {
    const rowBase = base + i * strideRow;
    for (let j = 0; j < cols; j++) {
      out[i * cols + j] = safeConvertToNumber(values[rowBase + j * strideCol]);
    }
  }

  return { rows, cols, data: out };
}

/**
 * Safely converts a tensor data value to number with proper type checking.
 * @internal
 */
function safeConvertToNumber(value: unknown): number {
  if (typeof value === "number") {
    return value;
  }
  if (typeof value === "bigint") {
    return Number(value);
  }
  throw new DTypeError(`Cannot convert ${typeof value} to number for plotting`);
}
