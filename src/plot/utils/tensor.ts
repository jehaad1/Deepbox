import { DTypeError, ShapeError } from "../../core";
import type { AnyTensor } from "../../ndarray";

/**
 * Converts a 1D tensor to Float64Array.
 * @internal
 */
export function tensorToFloat64Vector1D(t: AnyTensor): Float64Array {
  // Extract underlying tensor from GradTensor if needed
  const tensor = "tensor" in t ? t.tensor : t;

  if (tensor.ndim !== 1) throw new ShapeError("Expected a 1D tensor");
  if (tensor.dtype === "string") throw new DTypeError("Plotting does not support string tensors");

  const n = tensor.shape[0] ?? 0;
  const stride = tensor.strides[0] ?? 0;
  const out = new Float64Array(n);
  const base = tensor.offset;

  for (let i = 0; i < n; i++) {
    const v = tensor.data[base + i * stride];
    out[i] = safeConvertToNumber(v);
  }

  return out;
}

/**
 * Converts a 2D tensor to Float64Array.
 * @internal
 */
export function tensorToFloat64Matrix2D(t: AnyTensor): {
  readonly rows: number;
  readonly cols: number;
  readonly data: Float64Array;
} {
  // Extract underlying tensor from GradTensor if needed
  const tensor = "tensor" in t ? t.tensor : t;

  if (tensor.ndim !== 2) throw new ShapeError("Expected a 2D tensor");
  if (tensor.dtype === "string") throw new DTypeError("Plotting does not support string tensors");

  const rows = tensor.shape[0] ?? 0;
  const cols = tensor.shape[1] ?? 0;
  const strideRow = tensor.strides[0] ?? 0;
  const strideCol = tensor.strides[1] ?? 0;
  const out = new Float64Array(rows * cols);

  if (strideCol === 1 && strideRow === cols) {
    const start = tensor.offset;
    const end = start + rows * cols;
    for (let i = 0, j = start; j < end; i++, j++) {
      const v = tensor.data[j];
      out[i] = safeConvertToNumber(v);
    }
    return { rows, cols, data: out };
  }

  const base = tensor.offset;
  for (let i = 0; i < rows; i++) {
    const rowBase = base + i * strideRow;
    for (let j = 0; j < cols; j++) {
      const v = tensor.data[rowBase + j * strideCol];
      out[i * cols + j] = safeConvertToNumber(v);
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
