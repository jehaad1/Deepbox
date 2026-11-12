/**
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { DTypeError, ShapeError } from "../../core";
import { type AnyTensor, GradTensor, Tensor } from "../../ndarray";

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

/** A 2D grid of numbers: a tensor, or nested arrays or typed arrays. */
export type GridInput = AnyTensor | ArrayLike<ArrayLike<number>>;

/** A 1D list of numbers: a tensor, an array or a typed array. */
export type VectorInput = AnyTensor | ArrayLike<number>;

function isAnyTensor(value: unknown): value is AnyTensor {
  return value instanceof Tensor || value instanceof GradTensor;
}

/**
 * Copies a 1D tensor or array-like into a fresh Float64Array.
 * @throws {ShapeError} If a tensor is not 1D.
 * @throws {DTypeError} If a tensor holds strings.
 * @internal
 */
export function toFloat64Vector(value: VectorInput): Float64Array {
  return isAnyTensor(value) ? tensorToFloat64Vector1D(value) : Float64Array.from(value);
}

/**
 * Copies the x, y and z grids of a 3D surface or wireframe into rectangular Float64Array rows,
 * accepting 2D tensors, nested arrays or arrays of typed arrays.
 * @param fn - Name of the calling plot function, used in error messages
 * @throws {ShapeError} If a tensor is not 2D, a grid is empty or ragged, or the grids differ in shape.
 * @throws {DTypeError} If a tensor holds strings.
 * @internal
 */
export function toFloat64Grids(
  fn: string,
  xGrid: GridInput,
  yGrid: GridInput,
  zGrid: GridInput
): [Float64Array[], Float64Array[], Float64Array[]] {
  const rowsOf = (grid: GridInput): ArrayLike<ArrayLike<number>> => {
    if (!isAnyTensor(grid)) return grid;
    const { rows, cols, data } = tensorToFloat64Matrix2D(grid);
    return Array.from({ length: rows }, (_, r) => data.subarray(r * cols, (r + 1) * cols));
  };
  const x = rowsOf(xGrid);
  const y = rowsOf(yGrid);
  const z = rowsOf(zGrid);
  const rows = z.length;
  if (rows === 0 || x.length !== rows || y.length !== rows) {
    throw new ShapeError(
      `${fn}: xGrid, yGrid and zGrid must have the same, non-zero number of rows; received ${x.length}, ${y.length}, ${z.length}`
    );
  }
  const cols = z[0]?.length ?? 0;
  if (cols === 0) {
    throw new ShapeError(`${fn}: grids must have at least one column`);
  }
  const copy = (grid: ArrayLike<ArrayLike<number>>, name: string): Float64Array[] => {
    const out: Float64Array[] = [];
    for (let r = 0; r < rows; r++) {
      const row = grid[r];
      if (row === undefined || row.length !== cols) {
        throw new ShapeError(
          `${fn}: ${name} must be a rectangular ${rows} x ${cols} grid; row ${r} has length ${row?.length ?? 0}`
        );
      }
      out.push(Float64Array.from(row));
    }
    return out;
  };
  return [copy(x, "xGrid"), copy(y, "yGrid"), copy(z, "zGrid")];
}
