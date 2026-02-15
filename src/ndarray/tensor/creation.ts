import type { Device, DType, ScalarDType, Shape, TypedArray } from "../../core";
import {
  DataValidationError,
  DeepboxError,
  DTypeError,
  getConfig,
  getNumericElement,
  InvalidParameterError,
  isTypedArray,
  shapeToSize,
  validateShape,
} from "../../core";

import { dtypeToTypedArrayCtor, Tensor } from "./Tensor";

/**
 * Recursive type for nested number arrays.
 *
 * Used to represent multi-dimensional data in JavaScript arrays.
 *
 * @example
 * ```ts
 * const scalar: NestedArray = 5;
 * const vector: NestedArray = [1, 2, 3];
 * const matrix: NestedArray = [[1, 2], [3, 4]];
 * const tensor3d: NestedArray = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
 * ```
 */
export type NestedArray = number | boolean | NestedArray[];

export type StringNestedArray = string | StringNestedArray[];

export type TensorCreateOptions = {
  readonly dtype?: DType;
  readonly device?: Device;
};

type NumericDType = Exclude<DType, "string">;

function ensureNumericDType(dtype: DType, op: string): NumericDType {
  if (dtype === "string") {
    throw new DTypeError(`${op} does not support string dtype`);
  }
  return dtype;
}

function inferShapeFromNestedArray(data: unknown): Shape {
  const shape: number[] = [];

  let cursor: unknown = data;
  while (Array.isArray(cursor)) {
    shape.push(cursor.length);
    if (cursor.length === 0) {
      return shape;
    }
    cursor = cursor[0];
  }

  return shape;
}

function validateRegularStringNestedArray(data: unknown, shape: Shape, depth = 0): void {
  if (depth === shape.length) {
    if (typeof data !== "string") {
      throw new DataValidationError("string tensor data leaf values must be strings");
    }
    return;
  }

  if (!Array.isArray(data)) {
    throw new DataValidationError(
      "string tensor data must be a nested array with consistent shape"
    );
  }

  const expectedLen = shape[depth] ?? 0;
  if (data.length !== expectedLen) {
    throw new DataValidationError(
      `Ragged tensor: expected length ${expectedLen} at depth ${depth}, got ${data.length}`
    );
  }

  for (const item of data) {
    validateRegularStringNestedArray(item, shape, depth + 1);
  }
}

function inferShapeFromStringNestedArray(data: unknown): Shape {
  return inferShapeFromNestedArray(data);
}

function validateRegularNestedArray(data: unknown, shape: Shape, depth = 0): void {
  if (depth === shape.length) {
    if (typeof data !== "number" && typeof data !== "boolean") {
      throw new DataValidationError("tensor data leaf values must be numbers or booleans");
    }
    return;
  }

  if (!Array.isArray(data)) {
    throw new DataValidationError("tensor data must be a nested array with consistent shape");
  }

  const expectedLen = shape[depth] ?? 0;
  if (data.length !== expectedLen) {
    throw new DataValidationError(
      `Ragged tensor: expected length ${expectedLen} at depth ${depth}, got ${data.length}`
    );
  }

  for (const item of data) {
    validateRegularNestedArray(item, shape, depth + 1);
  }
}

function flattenNestedArray(data: unknown, out: number[]): void {
  if (Array.isArray(data)) {
    for (const item of data) {
      flattenNestedArray(item, out);
    }
    return;
  }

  if (typeof data === "boolean") {
    out.push(data ? 1 : 0);
    return;
  }

  if (typeof data !== "number") {
    throw new DataValidationError("Expected number");
  }

  out.push(data);
}

function flattenStringNestedArray(data: unknown, out: string[]): void {
  if (Array.isArray(data)) {
    for (const item of data) {
      flattenStringNestedArray(item, out);
    }
    return;
  }

  if (typeof data !== "string") {
    throw new DataValidationError("Expected string");
  }

  out.push(data);
}

function coerceNumberToTypedArrayValue(dtype: DType, value: number): number | bigint {
  switch (dtype) {
    case "int64":
      if (!Number.isFinite(value) || !Number.isInteger(value)) {
        throw new DTypeError(`int64 tensor values must be finite integers; received ${value}`);
      }
      return BigInt(value);
    case "bool":
      return value ? 1 : 0;
    default:
      return value;
  }
}

function inferDTypeFromInput(data: NestedArray | StringNestedArray): DType {
  let cursor: NestedArray | StringNestedArray | undefined = data;
  while (Array.isArray(cursor)) {
    if (cursor.length === 0) {
      return "float32";
    }
    cursor = cursor[0];
  }
  if (typeof cursor === "string") return "string";
  if (typeof cursor === "boolean") return "bool";
  return "float32";
}

function inferDTypeFromTypedArray(data: TypedArray): DType {
  if (data instanceof Float32Array) return "float32";
  if (data instanceof Float64Array) return "float64";
  if (data instanceof Int32Array) return "int32";
  if (data instanceof BigInt64Array) return "int64";
  return "uint8";
}

function isTypedArrayCompatibleWithDType(data: TypedArray, dtype: DType): boolean {
  if (dtype === "string") return false;
  if (dtype === "float32") return data instanceof Float32Array;
  if (dtype === "float64") return data instanceof Float64Array;
  if (dtype === "int32") return data instanceof Int32Array;
  if (dtype === "int64") return data instanceof BigInt64Array;
  if (dtype === "uint8" || dtype === "bool") return data instanceof Uint8Array;
  return false;
}

/**
 * Create a tensor from nested arrays or TypedArray.
 *
 * This is the primary function for creating tensors. It accepts:
 * - Nested JavaScript arrays (e.g., [[1, 2], [3, 4]])
 * - TypedArrays (e.g., Float32Array)
 * - Scalars (single numbers)
 *
 * Time complexity: O(n) where n is total number of elements.
 * Space complexity: O(n) for data storage.
 *
 * @param data - Input data as nested array or TypedArray
 * @param opts - Creation options (dtype, device)
 * @returns New tensor
 *
 * @throws {TypeError} If data has inconsistent shape (ragged arrays)
 * @throws {DTypeError} If dtype is incompatible with data
 *
 * @example
 * ```ts
 * import { tensor } from 'deepbox/ndarray';
 *
 * // From nested arrays
 * const t1 = tensor([[1, 2, 3], [4, 5, 6]]);
 *
 * // Specify dtype
 * const t2 = tensor([1, 2, 3], { dtype: 'int32' });
 *
 * // From TypedArray
 * const data = new Float32Array([1, 2, 3, 4]);
 * const t3 = tensor(data);
 *
 * // Scalar
 * const t4 = tensor(42);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox Tensor Creation}
 */
export function tensor(data: NestedArray, opts?: TensorCreateOptions): Tensor<Shape, ScalarDType>;
export function tensor(
  data: NestedArray | StringNestedArray | TypedArray,
  opts?: TensorCreateOptions
): Tensor;
export function tensor(
  data: NestedArray | StringNestedArray | TypedArray,
  opts: TensorCreateOptions = {}
): Tensor {
  const config = getConfig();
  const device = opts.device ?? config.defaultDevice;

  if (isTypedArray(data)) {
    const inferred = inferDTypeFromTypedArray(data);
    if (opts.dtype !== undefined && !isTypedArrayCompatibleWithDType(data, opts.dtype)) {
      throw new DTypeError(
        `TypedArray ${data.constructor.name} is not compatible with dtype ${opts.dtype}`
      );
    }
    const dtype = opts.dtype ?? inferred;
    const numericDtype = ensureNumericDType(dtype, "tensor");
    return Tensor.fromTypedArray({
      data,
      shape: [data.length],
      dtype: numericDtype,
      device,
    });
  }

  const inferred = opts.dtype === undefined ? inferDTypeFromInput(data) : undefined;
  const dtype = opts.dtype ?? inferred ?? config.defaultDtype;

  if (dtype === "string") {
    const shape = inferShapeFromStringNestedArray(data);
    validateShape(shape);
    validateRegularStringNestedArray(data, shape);

    const flat: string[] = [];
    flattenStringNestedArray(data, flat);

    if (shapeToSize(shape) !== flat.length) {
      throw new DeepboxError("Internal error: flattened size mismatch");
    }

    return Tensor.fromStringArray({ data: flat, shape, device });
  }

  const numericDtype = ensureNumericDType(dtype, "tensor");
  const shape = inferShapeFromNestedArray(data);
  validateShape(shape);
  validateRegularNestedArray(data, shape);

  const size = shapeToSize(shape);
  const Ctor = dtypeToTypedArrayCtor(numericDtype);
  const typed = new Ctor(size);

  // Fast path: flatten directly into TypedArray for non-BigInt dtypes
  if (!(typed instanceof BigInt64Array) && numericDtype !== "int64" && numericDtype !== "bool") {
    let idx = 0;
    const flattenDirect = (arr: unknown): void => {
      if (Array.isArray(arr)) {
        for (let i = 0; i < arr.length; i++) {
          flattenDirect(arr[i]);
        }
      } else if (typeof arr === "number") {
        typed[idx++] = arr;
      } else if (typeof arr === "boolean") {
        typed[idx++] = arr ? 1 : 0;
      } else {
        throw new DataValidationError("Expected number");
      }
    };
    flattenDirect(data);
  } else {
    const flat: number[] = [];
    flattenNestedArray(data, flat);

    if (typed instanceof BigInt64Array) {
      for (let i = 0; i < flat.length; i++) {
        const v = flat[i];
        if (v === undefined) {
          throw new DeepboxError("Internal error: missing flattened value");
        }
        const coerced = coerceNumberToTypedArrayValue(numericDtype, v);
        typed[i] = typeof coerced === "bigint" ? coerced : BigInt(coerced);
      }
    } else {
      for (let i = 0; i < flat.length; i++) {
        const v = flat[i];
        if (v === undefined) {
          throw new DeepboxError("Internal error: missing flattened value");
        }
        const coerced = coerceNumberToTypedArrayValue(numericDtype, v);
        typed[i] = typeof coerced === "number" ? coerced : Number(coerced);
      }
    }
  }

  return Tensor.fromTypedArray({
    data: typed,
    shape,
    dtype: numericDtype,
    device,
  });
}

/**
 * All zeros.
 */
export function zeros(shape: Shape, opts: TensorCreateOptions = {}): Tensor {
  const config = getConfig();
  const dtype = opts.dtype ?? config.defaultDtype;
  const device = opts.device ?? config.defaultDevice;
  if (dtype === "string") {
    return Tensor.zeros(shape, { dtype, device });
  }
  const numericDtype = ensureNumericDType(dtype, "zeros");
  return Tensor.zeros(shape, { dtype: numericDtype, device });
}

/**
 * All ones.
 */
export function ones(shape: Shape, opts: TensorCreateOptions = {}): Tensor {
  validateShape(shape);
  const config = getConfig();
  const dtype = opts.dtype ?? config.defaultDtype;
  const device = opts.device ?? config.defaultDevice;

  const size = shapeToSize(shape);

  if (dtype === "string") {
    const data = new Array<string>(size).fill("1");
    return Tensor.fromStringArray({ data, shape, device });
  }

  const numericDtype = ensureNumericDType(dtype, "ones");
  const Ctor = dtypeToTypedArrayCtor(numericDtype);
  const data = new Ctor(size);
  if (data instanceof BigInt64Array) {
    data.fill(1n);
  } else {
    data.fill(1);
  }
  return Tensor.fromTypedArray({ data, shape, dtype: numericDtype, device });
}

/**
 * Fill with a scalar value.
 */
export function empty(shape: Shape, opts: TensorCreateOptions = {}): Tensor {
  validateShape(shape);
  const config = getConfig();
  const dtype = opts.dtype ?? config.defaultDtype;
  const device = opts.device ?? config.defaultDevice;

  const size = shapeToSize(shape);
  if (dtype === "string") {
    const data = new Array<string>(size);
    return Tensor.fromStringArray({ data, shape, device });
  }
  const numericDtype = ensureNumericDType(dtype, "empty");
  const Ctor = dtypeToTypedArrayCtor(numericDtype);
  const data = new Ctor(size);

  return Tensor.fromTypedArray({ data, shape, dtype: numericDtype, device });
}

export function full(shape: Shape, value: number | string, opts: TensorCreateOptions = {}): Tensor {
  const t = zeros(shape, opts);

  if (t.dtype === "string") {
    if (typeof value !== "string") {
      throw new DTypeError(`Expected string fill value for dtype string; received ${typeof value}`);
    }
    if (!Array.isArray(t.data)) {
      throw new DTypeError("string dtype requires string[] backing data");
    }
    t.data.fill(value);
    return t;
  }

  if (typeof value !== "number") {
    throw new DTypeError(
      `Expected number fill value for dtype ${t.dtype}; received ${typeof value}`
    );
  }

  if (!isTypedArray(t.data)) {
    throw new DTypeError("numeric dtype requires TypedArray backing data");
  }
  if (t.data instanceof BigInt64Array) {
    t.data.fill(BigInt(value));
  } else {
    t.data.fill(Number(value));
  }

  return t;
}

/**
 * Range.
 */
export function arange(
  start: number,
  stop?: number,
  step = 1,
  opts: TensorCreateOptions = {}
): Tensor {
  const config = getConfig();
  const dtype = opts.dtype ?? config.defaultDtype;
  const device = opts.device ?? config.defaultDevice;
  const numericDtype = ensureNumericDType(dtype, "arange");

  const actualStop = stop ?? start;
  const actualStart = stop === undefined ? 0 : start;

  if (step === 0) {
    throw new InvalidParameterError("step must be non-zero", "step", step);
  }

  if (numericDtype === "int64") {
    if (!Number.isFinite(actualStart) || !Number.isInteger(actualStart)) {
      throw new InvalidParameterError(
        `start must be a finite integer for int64 arange; received ${actualStart}`,
        "start",
        actualStart
      );
    }
    if (!Number.isFinite(actualStop) || !Number.isInteger(actualStop)) {
      throw new InvalidParameterError(
        `stop must be a finite integer for int64 arange; received ${actualStop}`,
        "stop",
        actualStop
      );
    }
    if (!Number.isFinite(step) || !Number.isInteger(step)) {
      throw new InvalidParameterError(
        `step must be a finite integer for int64 arange; received ${step}`,
        "step",
        step
      );
    }
  }

  const length = Math.max(0, Math.ceil((actualStop - actualStart) / step));
  const Ctor = dtypeToTypedArrayCtor(numericDtype);
  const data = new Ctor(length);

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < length; i++) {
      data[i] = BigInt(Math.trunc(actualStart + i * step));
    }
  } else {
    for (let i = 0; i < length; i++) {
      data[i] = actualStart + i * step;
    }
  }

  return Tensor.fromTypedArray({
    data,
    shape: [length],
    dtype: numericDtype,
    device,
  });
}

/**
 * Evenly spaced numbers over a specified interval.
 *
 * Returns `num` evenly spaced samples, calculated over the interval [start, stop].
 *
 * **Algorithm**: Linear interpolation
 *
 * **Parameters**:
 * @param start - Starting value of the sequence
 * @param stop - End value of the sequence
 * @param num - Number of samples to generate (default: 50)
 * @param endpoint - If true, stop is the last sample. Otherwise, it is not included (default: true)
 * @param opts - Tensor options (dtype, device)
 *
 * **Returns**: Tensor of shape (num,)
 *
 * @example
 * ```ts
 * import { linspace } from 'deepbox/ndarray';
 *
 * const x = linspace(0, 10, 5);
 * // [0, 2.5, 5, 7.5, 10]
 *
 * const y = linspace(0, 10, 5, false);
 * // [0, 2, 4, 6, 8]
 * ```
 *
 * @throws {RangeError} If num < 0
 *
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox Tensor Creation}
 */
export function linspace(
  start: number,
  stop: number,
  num = 50,
  endpoint = true,
  opts: TensorCreateOptions = {}
): Tensor {
  if (num < 0) {
    throw new InvalidParameterError("num must be non-negative", "num", num);
  }

  if (num === 0) {
    return zeros([0], opts);
  }

  if (num === 1) {
    return tensor([start], opts);
  }

  const config = getConfig();
  const dtype = opts.dtype ?? config.defaultDtype;
  const device = opts.device ?? config.defaultDevice;
  const numericDtype = ensureNumericDType(dtype, "linspace");

  // Calculate the step size based on endpoint flag
  // If endpoint=true, divide range by (num-1) to include stop value
  // If endpoint=false, divide range by num to exclude stop value
  const step = endpoint ? (stop - start) / (num - 1) : (stop - start) / num;

  const Ctor = dtypeToTypedArrayCtor(numericDtype);
  const data = new Ctor(num);

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < num; i++) {
      data[i] = BigInt(Math.trunc(start + i * step));
    }
  } else {
    for (let i = 0; i < num; i++) {
      data[i] = start + i * step;
    }
  }

  return Tensor.fromTypedArray({
    data,
    shape: [num],
    dtype: numericDtype,
    device,
  });
}

/**
 * Numbers spaced evenly on a log scale.
 *
 * In linear space, the sequence starts at base^start and ends with base^stop.
 *
 * **Parameters**:
 * @param start - base^start is the starting value
 * @param stop - base^stop is the final value
 * @param num - Number of samples to generate (default: 50)
 * @param base - The base of the log space (default: 10)
 * @param endpoint - If true, stop is the last sample (default: true)
 * @param opts - Tensor options
 *
 * **Returns**: Tensor of shape (num,)
 *
 * @example
 * ```ts
 * import { logspace } from 'deepbox/ndarray';
 *
 * const x = logspace(0, 3, 4);
 * // [1, 10, 100, 1000]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox Tensor Creation}
 */
export function logspace(
  start: number,
  stop: number,
  num = 50,
  base = 10,
  endpoint = true,
  opts: TensorCreateOptions = {}
): Tensor {
  // Generate evenly spaced exponents
  const exponents = linspace(start, stop, num, endpoint, {
    ...opts,
    dtype: "float64",
  });

  const config = getConfig();
  const dtype = opts.dtype ?? config.defaultDtype;
  const device = opts.device ?? config.defaultDevice;
  const numericDtype = ensureNumericDType(dtype, "logspace");

  const Ctor = dtypeToTypedArrayCtor(numericDtype);
  const data = new Ctor(num);

  // exponents is always numeric (not BigInt64Array)
  const expData = exponents.data;
  if (Array.isArray(expData) || expData instanceof BigInt64Array) {
    throw new DeepboxError("Internal error: logspace exponents expected to be numeric typed array");
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < num; i++) {
      // Get exponent from linspace result
      const exponent = getNumericElement(expData, exponents.offset + i);
      // Calculate base^exponent and convert to BigInt
      const value = base ** exponent;
      data[i] = BigInt(Math.round(value));
    }
  } else {
    for (let i = 0; i < num; i++) {
      // Get exponent from linspace result
      const exponent = getNumericElement(expData, exponents.offset + i);
      // Calculate base^exponent
      data[i] = base ** exponent;
    }
  }

  return Tensor.fromTypedArray({
    data,
    shape: [num],
    dtype: numericDtype,
    device,
  });
}

/**
 * Numbers spaced evenly on a log scale (geometric progression).
 *
 * Each output value is a constant multiple of the previous.
 *
 * **Parameters**:
 * @param start - Starting value of the sequence
 * @param stop - Final value of the sequence
 * @param num - Number of samples (default: 50)
 * @param endpoint - If true, stop is the last sample (default: true)
 * @param opts - Tensor options
 *
 * **Returns**: Tensor of shape (num,)
 *
 * @example
 * ```ts
 * import { geomspace } from 'deepbox/ndarray';
 *
 * const x = geomspace(1, 1000, 4);
 * // [1, 10, 100, 1000]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox Tensor Creation}
 */
export function geomspace(
  start: number,
  stop: number,
  num = 50,
  endpoint = true,
  opts: TensorCreateOptions = {}
): Tensor {
  if (start === 0 || stop === 0) {
    throw new InvalidParameterError(
      "geomspace requires start and stop to be non-zero",
      "start",
      start
    );
  }

  if ((start > 0 && stop < 0) || (start < 0 && stop > 0)) {
    throw new InvalidParameterError(
      "geomspace requires start and stop to have the same sign",
      "start",
      start
    );
  }

  // Convert start and stop to log scale (natural logarithm)
  // Use absolute values to handle negative numbers properly
  const logStart = Math.log(Math.abs(start));
  const logStop = Math.log(Math.abs(stop));

  // Generate evenly spaced values in log scale (force float intermediates)
  const logValues = linspace(logStart, logStop, num, endpoint, {
    ...opts,
    dtype: "float64",
  });

  const config = getConfig();
  const dtype = opts.dtype ?? config.defaultDtype;
  const device = opts.device ?? config.defaultDevice;
  const numericDtype = ensureNumericDType(dtype, "geomspace");

  const Ctor = dtypeToTypedArrayCtor(numericDtype);
  const data = new Ctor(num);

  const sign = start < 0 ? -1 : 1;

  // logValues is always numeric (not BigInt64Array)
  const logData = logValues.data;
  if (Array.isArray(logData) || logData instanceof BigInt64Array) {
    throw new DeepboxError(
      "Internal error: geomspace logValues expected to be numeric typed array"
    );
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < num; i++) {
      const logVal = getNumericElement(logData, logValues.offset + i);
      data[i] = BigInt(Math.trunc(sign * Math.exp(logVal)));
    }
  } else {
    for (let i = 0; i < num; i++) {
      const logVal = getNumericElement(logData, logValues.offset + i);
      data[i] = sign * Math.exp(logVal);
    }
  }

  return Tensor.fromTypedArray({
    data,
    shape: [num],
    dtype: numericDtype,
    device,
  });
}

/**
 * Identity matrix.
 *
 * Returns a 2D tensor with ones on the diagonal and zeros elsewhere.
 *
 * **Parameters**:
 * @param n - Number of rows
 * @param m - Number of columns (default: n, making it square)
 * @param k - Index of the diagonal (default: 0, main diagonal)
 * @param opts - Tensor options
 *
 * **Returns**: Tensor of shape (n, m)
 *
 * @example
 * ```ts
 * import { eye } from 'deepbox/ndarray';
 *
 * const I = eye(3);
 * // [[1, 0, 0],
 * //  [0, 1, 0],
 * //  [0, 0, 1]]
 *
 * const A = eye(3, 4, 1);
 * // [[0, 1, 0, 0],
 * //  [0, 0, 1, 0],
 * //  [0, 0, 0, 1]]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox Tensor Creation}
 */
export function eye(n: number, m?: number, k = 0, opts: TensorCreateOptions = {}): Tensor {
  const cols = m ?? n;
  const matrix = zeros([n, cols], opts);

  // Set diagonal elements to 1
  if (matrix.data instanceof BigInt64Array) {
    for (let i = 0; i < n; i++) {
      const j = i + k; // Column index, offset by k
      if (j >= 0 && j < cols) {
        // Calculate flat index for row i, column j
        // In row-major order: index = row * num_cols + col
        const index = i * cols + j;
        matrix.data[index] = 1n;
      }
    }
  } else {
    for (let i = 0; i < n; i++) {
      const j = i + k; // Column index, offset by k
      if (j >= 0 && j < cols) {
        // Calculate flat index for row i, column j
        // In row-major order: index = row * num_cols + col
        const index = i * cols + j;
        matrix.data[index] = 1;
      }
    }
  }

  return matrix;
}

/**
 * Return a tensor filled with random samples from a standard normal distribution.
 *
 * @param shape - Shape of the output tensor
 * @param opts - Additional tensor options
 *
 * @example
 * ```ts
 * import { randn } from 'deepbox/ndarray';
 *
 * const x = randn([2, 3]);
 * // Random values from N(0, 1)
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox Tensor Creation}
 */
export function randn(shape: Shape, opts: TensorCreateOptions = {}): Tensor {
  const config = getConfig();
  const dtype = opts.dtype ?? config.defaultDtype;
  const device = opts.device ?? config.defaultDevice;
  const numericDtype = ensureNumericDType(dtype, "randn");

  const shapeArr = Array.isArray(shape) ? shape : [shape];
  validateShape(shapeArr);
  const size = shapeArr.reduce((a, b) => a * b, 1);

  const Ctor = dtypeToTypedArrayCtor(numericDtype);
  const data = new Ctor(size);

  const sampleUnitOpen = (): number => {
    const u = Math.random();
    return u > 0 ? u : Number.MIN_VALUE;
  };

  // Box-Muller transform for generating normal distribution
  if (data instanceof BigInt64Array) {
    for (let i = 0; i < size; i += 2) {
      const u1 = sampleUnitOpen();
      const u2 = Math.random();

      const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);

      data[i] = BigInt(Math.trunc(z0));
      if (i + 1 < size) {
        data[i + 1] = BigInt(Math.trunc(z1));
      }
    }
  } else {
    for (let i = 0; i < size; i += 2) {
      // Generate two uniform random numbers between 0 and 1
      // These will be transformed using Box-Muller algorithm
      const u1 = sampleUnitOpen();
      const u2 = Math.random();

      // Box-Muller transform to convert uniform to normal distribution
      // z0 and z1 are independent standard normal random variables
      const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      const z1 = Math.sqrt(-2 * Math.log(u1)) * Math.sin(2 * Math.PI * u2);

      data[i] = z0;
      if (i + 1 < size) {
        data[i + 1] = z1;
      }
    }
  }

  return Tensor.fromTypedArray({
    data,
    shape: shapeArr,
    dtype: numericDtype,
    device,
  });
}
