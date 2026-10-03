import type { Device, DType, Shape, TypedArray } from "../../core";
import {
  DataValidationError,
  DeepboxError,
  DTypeError,
  getConfig,
  InvalidParameterError,
  isTypedArray,
  MemoryError,
  shapeToSize,
  validateShape,
} from "../../core";
import { __fillNormal, __normalRandom } from "../../random/random";

import { roundToBFloat16, roundToFloat16 } from "./float16";
import { dtypeToTypedArrayCtor, Tensor, toInt64 } from "./Tensor";

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

/** Recursive type for nested string arrays used in string tensor creation. */
export type StringNestedArray = string | StringNestedArray[];

/** Options for the {@link tensor} creation function. */
export type TensorCreateOptions = {
  readonly dtype?: DType;
  readonly device?: Device;
};

type NumericDType = Exclude<DType, "string">;

function ensureNumericDType(dtype: DType, op: string): NumericDType {
  if (dtype === "string") {
    throw new DTypeError(`${op} does not support string dtype`);
  }
  if (dtype === "complex64" || dtype === "complex128") {
    throw new DTypeError(
      `${op} does not support dtype ${dtype} yet; use Complex64Array / Complex128Array for complex data`
    );
  }
  return dtype;
}

/**
 * Rounding function for the half-precision dtypes, `null` for every other
 * dtype. Half-precision tensors keep float32 host storage, so values must be
 * snapped onto the float16 / bfloat16 grid when they are created.
 */
function halfRounder(dtype: DType): ((x: number) => number) | null {
  if (dtype === "float16") return roundToFloat16;
  if (dtype === "bfloat16") return roundToBFloat16;
  return null;
}

/** How fractional values are converted when the target dtype is an integer. */
type IntRounding = "trunc" | "floor";

/**
 * Allocate a typed array of `dtype` and fill element `i` with `fn(i)`,
 * converting like the dtype requires: integers are truncated (or floored),
 * int64 rejects non-finite values, bool maps nonzero to 1, and float16 /
 * bfloat16 values are rounded to the half-precision grid.
 */
function fillFromFn(
  dtype: NumericDType,
  size: number,
  fn: (i: number) => number,
  op: string,
  rounding: IntRounding = "trunc"
): TypedArray {
  const Ctor = dtypeToTypedArrayCtor(dtype);
  assertAllocatable([size], size, Ctor.BYTES_PER_ELEMENT);
  const data = new Ctor(size);
  if (data instanceof BigInt64Array) {
    for (let i = 0; i < size; i++) {
      const v = fn(i);
      data[i] = toInt64(rounding === "floor" ? Math.floor(v) : v, op);
    }
  } else if (dtype === "bool") {
    for (let i = 0; i < size; i++) data[i] = fn(i) !== 0 ? 1 : 0;
  } else if (rounding === "floor" && (dtype === "int32" || dtype === "uint8")) {
    for (let i = 0; i < size; i++) data[i] = Math.floor(fn(i));
  } else {
    const round = halfRounder(dtype);
    if (round) {
      for (let i = 0; i < size; i++) data[i] = round(fn(i));
    } else {
      for (let i = 0; i < size; i++) data[i] = fn(i);
    }
  }
  return data;
}

/** Validate a sample count and return the generator of the evenly spaced values. */
function linspaceValues(
  start: number,
  stop: number,
  num: number,
  endpoint: boolean
): (i: number) => number {
  if (!Number.isInteger(num) || num < 0) {
    throw new InvalidParameterError(
      `num must be a non-negative integer; received ${String(num)}`,
      "num",
      num
    );
  }
  const div = endpoint ? num - 1 : num;
  const delta = stop - start;
  const step = div > 0 ? delta / div : 0;
  return (i: number): number => {
    if (endpoint && num > 1 && i === num - 1) return stop;
    if (div <= 0) return start;
    // A denormal step underflows to 0; scale the index instead (as NumPy does).
    return step === 0 ? (i / div) * delta + start : i * step + start;
  };
}

/**
 * Per-tensor allocation ceiling, in bytes.
 *
 * JavaScript engines cap the size of a single `ArrayBuffer` (and therefore any
 * `TypedArray` backing a tensor) at roughly 2 GiB. Exceeding it throws an
 * opaque `RangeError: Invalid typed array length` from the engine with no
 * indication of which shape or dtype was at fault. We compute the requested
 * byte size up-front and raise a clear {@link MemoryError} instead.
 *
 * At this ceiling a single tensor can hold up to ~500M `float32`/`int32` or
 * ~250M `float64`/`int64` elements.
 */
const MAX_TENSOR_BYTES = 0x8000_0000; // 2 GiB (2 ** 31)

/**
 * Guard against oversized typed-array allocations before they are attempted.
 *
 * Runs in O(1): the element count is already computed for allocation, so this
 * only multiplies by the dtype byte width and compares against the ceiling. It
 * adds no per-element overhead and does not affect normally-sized tensors.
 *
 * @param shape - Requested tensor shape (used only for the error message).
 * @param size - Total element count (product of `shape`).
 * @param bytesPerElement - Byte width of the backing dtype (e.g. 4 for float32).
 *
 * @throws {MemoryError} If the resulting byte size exceeds {@link MAX_TENSOR_BYTES}.
 */
function assertAllocatable(shape: Shape, size: number, bytesPerElement: number): void {
  const byteSize = size * bytesPerElement;
  if (byteSize <= MAX_TENSOR_BYTES) {
    return;
  }

  const shapeStr = `[${shape.map((d) => d.toLocaleString("en-US")).join(", ")}]`;
  const wideDtypeHint =
    bytesPerElement >= 8 ? " use a narrower dtype such as float32 (halves the footprint)," : "";
  throw new MemoryError(
    `Cannot allocate tensor of shape ${shapeStr}: ` +
      `${size.toLocaleString("en-US")} elements × ${bytesPerElement} bytes = ` +
      `${byteSize.toLocaleString("en-US")} bytes, which exceeds the per-tensor ceiling of ` +
      `${MAX_TENSOR_BYTES.toLocaleString("en-US")} bytes (~2 GiB, the JavaScript ArrayBuffer limit). ` +
      `Split the data into smaller chunks,${wideDtypeHint} or process it in batches.`,
    { requestedBytes: byteSize, availableBytes: MAX_TENSOR_BYTES }
  );
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
      return toInt64(value, "int64 tensor value");
    case "bool":
      // Nonzero (NaN included) is true, matching NumPy and Tensor.astype.
      return value !== 0 ? 1 : 0;
    default:
      return value;
  }
}

function inferDTypeFromInput(data: NestedArray | StringNestedArray, fallback: DType): DType {
  // Numbers use the configured default dtype, unless that is a dtype nested
  // numeric data cannot be stored in (string, complex).
  const numeric =
    fallback === "string" || fallback === "complex64" || fallback === "complex128"
      ? "float32"
      : fallback;
  let cursor: NestedArray | StringNestedArray | undefined = data;
  while (Array.isArray(cursor)) {
    if (cursor.length === 0) {
      return numeric;
    }
    cursor = cursor[0];
  }
  if (typeof cursor === "string") return "string";
  if (typeof cursor === "boolean") return "bool";
  return numeric;
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
  if (dtype === "float32" || dtype === "float16" || dtype === "bfloat16") {
    return data instanceof Float32Array;
  }
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
 * Nested arrays are always copied. A TypedArray is wrapped without copying,
 * so the tensor and the array share memory (except for float16 / bfloat16,
 * where the values are rounded into a new array). Without `opts.dtype` the
 * dtype is inferred: booleans give `bool`, strings give `string`, and numbers
 * use the configured default dtype (`float32` unless changed with `setConfig`).
 * An integer dtype truncates fractional values; `bool` maps nonzero (NaN
 * included) to true.
 *
 * The default `float32` rounds decimal input to single precision, so
 * `tensor([0.8]).toArray()` holds `0.800000011920929`. Pass `{ dtype: "float64" }`
 * when exact decimals matter, for example probabilities that are compared with
 * NumPy or scikit-learn references, encoder inputs, or plot data.
 *
 * Time complexity: O(n) where n is total number of elements.
 * Space complexity: O(n) for nested arrays, O(1) for TypedArray input.
 *
 * @param data - Input data as nested array or TypedArray
 * @param opts - Creation options (dtype, device)
 * @returns New tensor
 *
 * @throws {DataValidationError} If data has inconsistent shape (ragged arrays) or invalid leaves
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
 * // Keep decimals exact (the default float32 would round 0.8)
 * const t2b = tensor([0.8, 0.2], { dtype: 'float64' });
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
export function tensor(data: NestedArray, opts?: TensorCreateOptions): Tensor;
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
    const round = halfRounder(numericDtype);
    return Tensor.fromTypedArray({
      data: round ? Float32Array.from(data as Float32Array, round) : data,
      shape: [data.length],
      dtype: numericDtype,
      device,
    });
  }

  const dtype = opts.dtype ?? inferDTypeFromInput(data, config.defaultDtype);

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
  assertAllocatable(shape, size, Ctor.BYTES_PER_ELEMENT);
  const typed = new Ctor(size);

  // Fast path: flatten directly into TypedArray for non-BigInt dtypes
  if (!(typed instanceof BigInt64Array) && numericDtype !== "int64" && numericDtype !== "bool") {
    let idx = 0;
    const round = halfRounder(numericDtype);
    const flattenDirect = (arr: unknown): void => {
      if (Array.isArray(arr)) {
        for (let i = 0; i < arr.length; i++) {
          flattenDirect(arr[i]);
        }
      } else if (typeof arr === "number") {
        typed[idx++] = round ? round(arr) : arr;
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
 * Tensor of zeros.
 *
 * With `dtype: "string"` every element is the empty string.
 *
 * @param shape - Shape of the tensor
 * @param opts - dtype (default: the configured default dtype) and device
 *
 * @example
 * ```ts
 * zeros([2, 3]); // 2x3 tensor of 0
 * ```
 */
export function zeros(shape: Shape, opts: TensorCreateOptions = {}): Tensor {
  const config = getConfig();
  const dtype = opts.dtype ?? config.defaultDtype;
  const device = opts.device ?? config.defaultDevice;
  if (dtype === "string") {
    return Tensor.zeros(shape, { dtype, device });
  }
  const numericDtype = ensureNumericDType(dtype, "zeros");
  assertAllocatable(
    shape,
    shapeToSize(shape),
    dtypeToTypedArrayCtor(numericDtype).BYTES_PER_ELEMENT
  );
  return Tensor.zeros(shape, { dtype: numericDtype, device });
}

/**
 * Tensor of ones.
 *
 * With `dtype: "string"` every element is the text `"1"`.
 *
 * @param shape - Shape of the tensor
 * @param opts - dtype (default: the configured default dtype) and device
 *
 * @example
 * ```ts
 * ones([2, 2]); // [[1, 1], [1, 1]]
 * ```
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
  assertAllocatable(shape, size, Ctor.BYTES_PER_ELEMENT);
  const data = new Ctor(size);
  if (data instanceof BigInt64Array) {
    data.fill(1n);
  } else {
    data.fill(1);
  }
  return Tensor.fromTypedArray({ data, shape, dtype: numericDtype, device });
}

/**
 * Tensor with unspecified initial contents.
 *
 * JavaScript always zero-initializes memory, so numeric tensors come back
 * filled with 0 and string tensors with empty strings. Prefer {@link zeros}
 * when the initial value matters to readers of your code.
 *
 * @param shape - Shape of the tensor
 * @param opts - dtype (default: the configured default dtype) and device
 */
export function empty(shape: Shape, opts: TensorCreateOptions = {}): Tensor {
  validateShape(shape);
  const config = getConfig();
  const dtype = opts.dtype ?? config.defaultDtype;
  const device = opts.device ?? config.defaultDevice;

  const size = shapeToSize(shape);
  if (dtype === "string") {
    const data = new Array<string>(size).fill("");
    return Tensor.fromStringArray({ data, shape, device });
  }
  const numericDtype = ensureNumericDType(dtype, "empty");
  const Ctor = dtypeToTypedArrayCtor(numericDtype);
  assertAllocatable(shape, size, Ctor.BYTES_PER_ELEMENT);
  const data = new Ctor(size);

  return Tensor.fromTypedArray({ data, shape, dtype: numericDtype, device });
}

/**
 * Tensor filled with one value.
 *
 * Numeric dtypes take a number (or a boolean, stored as 1/0); `string`
 * takes a string. The value is converted like `tensor()` does: integer dtypes
 * truncate toward zero, `bool` maps nonzero to true, float16 / bfloat16 round
 * to half precision, and int64 requires a finite integer.
 *
 * @param shape - Shape of the tensor
 * @param value - Fill value
 * @param opts - dtype (default: the configured default dtype) and device
 * @throws {DTypeError} If `value` does not fit the dtype
 *
 * @example
 * ```ts
 * full([2, 2], 7);                       // [[7, 7], [7, 7]]
 * full([3], 5, { dtype: "bool" });       // [1, 1, 1]
 * full([2], "x", { dtype: "string" });   // ["x", "x"]
 * ```
 */
export function full(
  shape: Shape,
  value: number | string | boolean,
  opts: TensorCreateOptions = {}
): Tensor {
  validateShape(shape);
  const config = getConfig();
  const dtype = opts.dtype ?? config.defaultDtype;
  const device = opts.device ?? config.defaultDevice;
  const size = shapeToSize(shape);

  if (dtype === "string") {
    if (typeof value !== "string") {
      throw new DTypeError(`Expected string fill value for dtype string; received ${typeof value}`);
    }
    return Tensor.fromStringArray({ data: new Array<string>(size).fill(value), shape, device });
  }

  const numericDtype = ensureNumericDType(dtype, "full");
  if (typeof value !== "number" && typeof value !== "boolean") {
    throw new DTypeError(
      `Expected number fill value for dtype ${numericDtype}; received ${typeof value}`
    );
  }
  const num = typeof value === "boolean" ? (value ? 1 : 0) : value;

  const Ctor = dtypeToTypedArrayCtor(numericDtype);
  assertAllocatable(shape, size, Ctor.BYTES_PER_ELEMENT);
  const data = new Ctor(size);
  if (data instanceof BigInt64Array) {
    data.fill(BigInt(coerceNumberToTypedArrayValue("int64", num)));
  } else {
    data.fill(
      Number(coerceNumberToTypedArrayValue(numericDtype, halfRounder(numericDtype)?.(num) ?? num))
    );
  }
  return Tensor.fromTypedArray({ data, shape, dtype: numericDtype, device });
}

/**
 * Evenly spaced values within a half-open interval, like NumPy's `arange`.
 *
 * Call as `arange(stop)` for `0, 1, ..., stop - 1`, or
 * `arange(start, stop, step)`. The length is `ceil((stop - start) / step)`
 * (0 when the range is empty); element `i` is `start + i * step`. The default
 * dtype is the configured default dtype, not an integer dtype.
 *
 * @param start - Start of the interval, or the stop value when `stop` is omitted
 * @param stop - End of the interval (exclusive)
 * @param step - Spacing between values (default: 1, must be non-zero)
 * @param opts - dtype and device
 * @throws {InvalidParameterError} If `step` is zero or any argument is not finite
 *
 * @example
 * ```ts
 * arange(5);             // [0, 1, 2, 3, 4]
 * arange(1, 2, 0.25);    // [1, 1.25, 1.5, 1.75]
 * arange(3, 0, -1);      // [3, 2, 1]
 * ```
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
  for (const [name, v] of [
    ["start", actualStart],
    ["stop", actualStop],
    ["step", step],
  ] as const) {
    if (!Number.isFinite(v)) {
      throw new InvalidParameterError(`${name} must be finite; received ${v}`, name, v);
    }
  }

  if (numericDtype === "int64") {
    if (!Number.isInteger(actualStart)) {
      throw new InvalidParameterError(
        `start must be a finite integer for int64 arange; received ${actualStart}`,
        "start",
        actualStart
      );
    }
    if (!Number.isInteger(actualStop)) {
      throw new InvalidParameterError(
        `stop must be a finite integer for int64 arange; received ${actualStop}`,
        "stop",
        actualStop
      );
    }
    if (!Number.isInteger(step)) {
      throw new InvalidParameterError(
        `step must be a finite integer for int64 arange; received ${step}`,
        "step",
        step
      );
    }
  }

  const length = Math.max(0, Math.ceil((actualStop - actualStart) / step));
  const data = fillFromFn(numericDtype, length, (i) => actualStart + i * step, "arange");

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
 * Returns `num` evenly spaced samples over `[start, stop]`. With
 * `endpoint: true` the last sample is exactly `stop`. Integer dtypes floor the
 * samples (NumPy behavior), so `linspace(-5, 5, 4, true, { dtype: "int32" })`
 * is `[-5, -2, 1, 5]`.
 *
 * @param start - Starting value of the sequence
 * @param stop - End value of the sequence
 * @param num - Number of samples to generate (default: 50)
 * @param endpoint - If true, stop is the last sample. Otherwise, it is not included (default: true)
 * @param opts - Tensor options (dtype, device)
 * @returns Tensor of shape (num,)
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
 * @throws {InvalidParameterError} If `num` is not a non-negative integer
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
  const value = linspaceValues(start, stop, num, endpoint);
  const config = getConfig();
  const dtype = opts.dtype ?? config.defaultDtype;
  const device = opts.device ?? config.defaultDevice;
  const numericDtype = ensureNumericDType(dtype, "linspace");

  const data = fillFromFn(numericDtype, num, value, "linspace", "floor");

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
 * Integer dtypes truncate the values toward zero.
 *
 * @param start - base^start is the starting value
 * @param stop - base^stop is the final value
 * @param num - Number of samples to generate (default: 50)
 * @param base - The base of the log space (default: 10)
 * @param endpoint - If true, stop is the last sample (default: true)
 * @param opts - Tensor options
 * @returns Tensor of shape (num,)
 *
 * @example
 * ```ts
 * import { logspace } from 'deepbox/ndarray';
 *
 * const x = logspace(0, 3, 4);
 * // [1, 10, 100, 1000]
 * ```
 *
 * @throws {InvalidParameterError} If `num` is not a non-negative integer
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
  const exponent = linspaceValues(start, stop, num, endpoint);

  const config = getConfig();
  const dtype = opts.dtype ?? config.defaultDtype;
  const device = opts.device ?? config.defaultDevice;
  const numericDtype = ensureNumericDType(dtype, "logspace");

  const data = fillFromFn(numericDtype, num, (i) => base ** exponent(i), "logspace");

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
 * Each output value is a constant multiple of the previous. The first value
 * is exactly `start` and, with `endpoint: true`, the last is exactly `stop`.
 * Integer dtypes truncate the values toward zero.
 *
 * @param start - Starting value of the sequence (non-zero, same sign as `stop`)
 * @param stop - Final value of the sequence
 * @param num - Number of samples (default: 50)
 * @param endpoint - If true, stop is the last sample (default: true)
 * @param opts - Tensor options
 * @returns Tensor of shape (num,)
 *
 * @example
 * ```ts
 * import { geomspace } from 'deepbox/ndarray';
 *
 * const x = geomspace(1, 1000, 4);
 * // [1, 10, 100, 1000]
 * ```
 *
 * @throws {InvalidParameterError} If `start` or `stop` is zero, non-finite or of opposite sign,
 *   or `num` is not a non-negative integer
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
  if (!Number.isFinite(start) || !Number.isFinite(stop)) {
    throw new InvalidParameterError(
      `geomspace requires finite start and stop; received start=${start}, stop=${stop}`,
      "start",
      start
    );
  }

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

  // Interpolate the base-10 logarithms of the magnitudes, then restore the sign.
  const logValue = linspaceValues(
    Math.log10(Math.abs(start)),
    Math.log10(Math.abs(stop)),
    num,
    endpoint
  );

  const config = getConfig();
  const dtype = opts.dtype ?? config.defaultDtype;
  const device = opts.device ?? config.defaultDevice;
  const numericDtype = ensureNumericDType(dtype, "geomspace");

  const sign = start < 0 ? -1 : 1;
  const data = fillFromFn(
    numericDtype,
    num,
    (i) => {
      // The endpoints are exact; a log/exp round trip would drift by an ulp.
      if (i === 0) return start;
      if (endpoint && i === num - 1) return stop;
      return sign * 10 ** logValue(i);
    },
    "geomspace"
  );

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
 * @param n - Number of rows
 * @param m - Number of columns (default: n, making it square)
 * @param k - Index of the diagonal (default: 0, main diagonal; positive is above)
 * @param opts - Tensor options
 * @returns Tensor of shape (n, m)
 * @throws {InvalidParameterError} If `k` is not an integer
 * @throws {DTypeError} If the dtype is string or complex
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
  if (!Number.isInteger(k)) {
    throw new InvalidParameterError(`k must be an integer; received ${String(k)}`, "k", k);
  }
  const cols = m ?? n;
  const shape: Shape = [n, cols];
  const size = shapeToSize(shape);

  const config = getConfig();
  const dtype = opts.dtype ?? config.defaultDtype;
  const device = opts.device ?? config.defaultDevice;
  const numericDtype = ensureNumericDType(dtype, "eye");

  const Ctor = dtypeToTypedArrayCtor(numericDtype);
  assertAllocatable(shape, size, Ctor.BYTES_PER_ELEMENT);
  const data = new Ctor(size);

  // Row i holds its one at column i + k; keep only rows where that is in range.
  const first = Math.max(0, -k);
  const last = Math.min(n, cols - k);
  if (data instanceof BigInt64Array) {
    for (let i = first; i < last; i++) data[i * cols + i + k] = 1n;
  } else {
    for (let i = first; i < last; i++) data[i * cols + i + k] = 1;
  }

  return Tensor.fromTypedArray({ data, shape, dtype: numericDtype, device });
}

/**
 * Return a tensor filled with random samples from a standard normal distribution.
 *
 * Uses the library's seeded generator, so `setSeed` makes the output
 * reproducible. Integer dtypes truncate the samples toward zero.
 *
 * @param shape - Shape of the output tensor
 * @param opts - Additional tensor options
 * @throws {DTypeError} If the dtype is bool, string or complex
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
  if (numericDtype === "bool") {
    throw new DTypeError("randn does not support bool dtype; sample a float dtype and compare");
  }

  const shapeArr = Array.isArray(shape) ? shape : [shape];
  validateShape(shapeArr);
  const size = shapeToSize(shapeArr);

  const Ctor = dtypeToTypedArrayCtor(numericDtype);
  assertAllocatable(shapeArr, size, Ctor.BYTES_PER_ELEMENT);
  const data = new Ctor(size);

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < size; i++) {
      data[i] = BigInt(Math.trunc(__normalRandom()));
    }
  } else if (data instanceof Float64Array || data instanceof Float32Array) {
    __fillNormal(data, size);
    const round = halfRounder(numericDtype);
    if (round) {
      for (let i = 0; i < size; i++) data[i] = round(data[i] as number);
    }
  } else {
    for (let i = 0; i < size; i++) {
      data[i] = __normalRandom();
    }
  }

  return Tensor.fromTypedArray({
    data,
    shape: shapeArr,
    dtype: numericDtype,
    device,
  });
}
