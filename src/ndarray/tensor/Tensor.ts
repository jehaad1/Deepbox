import type { Device, DType, ElementOf, Shape, TensorLike, TypedArray } from "../../core";
import {
  DeepboxError,
  DTypeError,
  dtypeToTypedArrayCtor,
  getBigIntElement,
  getNumericElement,
  IndexError,
  InvalidParameterError,
  ShapeError,
  shapeToSize,
  validateShape,
} from "../../core";
import { normalizeRange, type SliceRange } from "./sliceHelpers";
import { isContiguous, offsetFromFlatIndex } from "./strides";

export type TensorOptions = {
  readonly dtype: DType;
  readonly device: Device;
};

type TensorData<D extends DType> = D extends "string" ? string[] : TypedArray;

function isTypedArrayForDType(data: TypedArray, dtype: DType): boolean {
  if (dtype === "string") return false;
  if (dtype === "float32") return data instanceof Float32Array;
  if (dtype === "float64") return data instanceof Float64Array;
  if (dtype === "int32") return data instanceof Int32Array;
  if (dtype === "int64") return data instanceof BigInt64Array;
  if (dtype === "uint8" || dtype === "bool") return data instanceof Uint8Array;
  return false;
}

function assertTypedArrayForDType(data: TypedArray, dtype: DType): void {
  if (!isTypedArrayForDType(data, dtype)) {
    throw new DTypeError(
      `TypedArray ${data.constructor.name} does not match dtype ${dtype}; ` +
        "provide matching dtype or convert the data first."
    );
  }
}

function validateStrides(
  shape: Shape,
  strides: readonly number[],
  offset: number,
  dataLength: number
): void {
  if (strides.length !== shape.length) {
    throw new ShapeError(
      `strides length ${strides.length} does not match shape length ${shape.length}`
    );
  }

  for (const stride of strides) {
    if (!Number.isInteger(stride)) {
      throw new InvalidParameterError(
        `stride must be an integer; received ${String(stride)}`,
        "strides",
        stride
      );
    }
    if (stride < 0) {
      throw new InvalidParameterError(
        `stride must be >= 0; received ${String(stride)}`,
        "strides",
        stride
      );
    }
  }

  if (offset < 0) {
    throw new InvalidParameterError(`offset must be >= 0; received ${offset}`, "offset", offset);
  }

  if (shapeToSize(shape) === 0) {
    if (offset > dataLength) {
      throw new ShapeError(`offset ${offset} is out of bounds for buffer length ${dataLength}`);
    }
    return;
  }

  let maxOffset = offset;
  for (let i = 0; i < shape.length; i++) {
    const dim = shape[i] ?? 0;
    const stride = strides[i] ?? 0;
    if (dim > 0) {
      maxOffset += (dim - 1) * stride;
    }
  }

  if (maxOffset >= dataLength) {
    throw new ShapeError(
      `Data length ${dataLength} is too small for shape [${shape}] with strides [${strides}] and offset ${offset}`
    );
  }
}

/**
 * Compute memory strides for row-major layout.
 *
 * Strides determine the step size in the underlying buffer for each dimension.
 * For row-major (C-order), the last dimension has stride 1.
 *
 * Time complexity: O(n) where n is number of dimensions.
 *
 * @param shape - Tensor shape
 * @returns Array of stride values
 *
 * @example
 * ```ts
 * computeStrides([2, 3, 4]); // [12, 4, 1]
 * // To access element [i, j, k]: offset + i*12 + j*4 + k*1
 * ```
 */
export function computeStrides(shape: Shape): readonly number[] {
  const strides = new Array<number>(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i] ?? 0;
  }
  return strides;
}

export { dtypeToTypedArrayCtor } from "../../core";

/**
 * Type guard to check if TypedArray is BigInt64Array.
 *
 * @param arr - TypedArray to check
 * @returns True if array is BigInt64Array
 */
export function isBigIntArray(arr: TypedArray): arr is BigInt64Array {
  return arr instanceof BigInt64Array;
}

/**
 * Multi-dimensional array (tensor) with typed storage.
 *
 * Core data structure for numerical computing. Supports:
 * - N-dimensional arrays with any shape
 * - Multiple data types (float32, float64, int32, etc.)
 * - Memory-efficient strided views
 * - Device abstraction (CPU, WebGPU, WASM)
 *
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox Tensors}
 *
 * @typeParam S - Shape type (readonly number array)
 * @typeParam D - Data type (DType literal)
 *
 * @example
 * ```ts
 * import { tensor } from 'deepbox/ndarray';
 *
 * // Create from nested arrays
 * const t = tensor([[1, 2, 3], [4, 5, 6]]);
 * console.log(t.shape);  // [2, 3]
 * console.log(t.dtype);  // 'float32'
 *
 * // Access properties
 * console.log(t.size);   // 6
 * console.log(t.ndim);   // 2
 * ```
 */
export class Tensor<S extends Shape = Shape, D extends DType = DType> implements TensorLike<S, D> {
  readonly shape: S;
  readonly dtype: D;
  readonly device: Device;
  readonly data: TensorData<D>;
  readonly strides: readonly number[];
  readonly offset: number;
  readonly size: number;
  readonly ndim: number;

  private constructor(args: {
    readonly data: TensorData<D>;
    readonly shape: S;
    readonly dtype: D;
    readonly device: Device;
    readonly strides: readonly number[];
    readonly offset: number;
  }) {
    this.data = args.data;
    this.shape = args.shape;
    this.dtype = args.dtype;
    this.device = args.device;
    this.strides = args.strides;
    this.offset = args.offset;

    this.ndim = this.shape.length;
    this.size = shapeToSize(this.shape);
  }

  private isStringTensor(): this is Tensor<S, "string"> {
    return this.dtype === "string";
  }

  private isNumericTensor(): this is Tensor<S, Exclude<DType, "string">> {
    return this.dtype !== "string";
  }

  static fromTypedArray<S extends Shape, D extends Exclude<DType, "string">>(args: {
    readonly data: TensorData<D>;
    readonly shape: S;
    readonly dtype: D;
    readonly device: Device;
    readonly offset?: number;
    readonly strides?: readonly number[];
  }): Tensor<S, D> {
    validateShape(args.shape);
    const offset = args.offset ?? 0;
    const strides = args.strides ?? computeStrides(args.shape);
    assertTypedArrayForDType(args.data, args.dtype);
    validateStrides(args.shape, strides, offset, args.data.length);

    return new Tensor<S, D>({
      data: args.data,
      shape: args.shape,
      dtype: args.dtype,
      device: args.device,
      offset,
      strides,
    });
  }

  static fromStringArray<S extends Shape>(args: {
    readonly data: string[];
    readonly shape: S;
    readonly device?: Device;
    readonly offset?: number;
    readonly strides?: readonly number[];
  }): Tensor<S, "string"> {
    validateShape(args.shape);
    const offset = args.offset ?? 0;
    const strides = args.strides ?? computeStrides(args.shape);
    validateStrides(args.shape, strides, offset, args.data.length);

    return new Tensor({
      data: args.data,
      shape: args.shape,
      dtype: "string",
      device: args.device ?? "cpu",
      offset,
      strides,
    });
  }

  static zeros<S extends Shape>(
    shape: S,
    opts: TensorOptions & { readonly dtype: "string" }
  ): Tensor<S, "string">;
  static zeros<S extends Shape, D extends Exclude<DType, "string">>(
    shape: S,
    opts: TensorOptions & { readonly dtype: D }
  ): Tensor<S, D>;
  static zeros<S extends Shape>(
    shape: S,
    opts: TensorOptions & { readonly dtype: DType }
  ): Tensor<S, DType> {
    validateShape(shape);
    const size = shapeToSize(shape);
    if (opts.dtype === "string") {
      const data = new Array<string>(size);
      data.fill("");
      return Tensor.fromStringArray({
        data,
        shape,
        device: opts.device,
      });
    }

    const Ctor = dtypeToTypedArrayCtor(opts.dtype);
    const data = new Ctor(size);
    return Tensor.fromTypedArray({
      data,
      shape,
      dtype: opts.dtype,
      device: opts.device,
    });
  }

  /**
   * Create a view sharing the same underlying data.
   *
   * Note: This does not copy data. Mutations (if exposed in the future) would be shared.
   */
  view<S2 extends Shape>(
    this: Tensor<S, "string">,
    shape: S2,
    strides?: readonly number[],
    offset?: number
  ): Tensor<S2, "string">;
  view<S2 extends Shape>(
    this: Tensor<S, Exclude<DType, "string">>,
    shape: S2,
    strides?: readonly number[],
    offset?: number
  ): Tensor<S2, Exclude<DType, "string">>;
  view<S2 extends Shape>(
    this: Tensor<S, DType>,
    shape: S2,
    strides?: readonly number[],
    offset?: number
  ): Tensor<S2, DType>;
  view<S2 extends Shape>(
    shape: S2,
    strides?: readonly number[],
    offset = this.offset
  ): Tensor<S2, DType> {
    validateShape(shape);
    if (shapeToSize(shape) !== this.size) {
      throw ShapeError.mismatch([this.size], [shapeToSize(shape)], "view");
    }
    if (this.isStringTensor()) {
      // Safe: this branch only executes when dtype is string, so D is "string".
      return Tensor.fromStringArray({
        data: this.data,
        shape,
        device: this.device,
        offset,
        strides: strides ?? computeStrides(shape),
      });
    }

    if (!this.isNumericTensor()) {
      throw new DTypeError("view is not defined for string dtype");
    }

    return Tensor.fromTypedArray({
      data: this.data,
      shape,
      dtype: this.dtype,
      device: this.device,
      offset,
      strides: strides ?? computeStrides(shape),
    });
  }

  /**
   * Reshape the tensor to a new shape without copying data.
   *
   * Returns a new tensor with the specified shape, sharing the same underlying data.
   * The total number of elements must remain the same.
   * Requires a contiguous tensor; non-contiguous views will throw.
   *
   * This is a convenience method that wraps the standalone `reshape` function,
   * providing a more intuitive API for tensor slicing and indexing.
   *
   * @param newShape - The desired shape for the tensor
   * @returns A new tensor with the specified shape
   * @throws {ShapeError} If the new shape is incompatible with the tensor's size
   *
   * @example
   * ```ts
   * const t = tensor([1, 2, 3, 4, 5, 6]);
   * const reshaped = t.reshape([2, 3]);
   * console.log(reshaped.shape); // [2, 3]
   *
   * const matrix = tensor([[1, 2], [3, 4]]);
   * const flat = matrix.reshape([4]);
   * console.log(flat.shape); // [4]
   * ```
   *
   * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox Tensor Creation}
   * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox Tensors}
   */
  reshape<S2 extends Shape>(this: Tensor<S, "string">, newShape: S2): Tensor<S2, "string">;
  reshape<S2 extends Shape>(
    this: Tensor<S, Exclude<DType, "string">>,
    newShape: S2
  ): Tensor<S2, Exclude<DType, "string">>;
  reshape<S2 extends Shape>(this: Tensor<S, DType>, newShape: S2): Tensor<S2, DType>;
  reshape<S2 extends Shape>(newShape: S2): Tensor<S2, DType> {
    validateShape(newShape);
    const newSize = shapeToSize(newShape);
    if (newSize !== this.size) {
      throw new ShapeError(`Cannot reshape tensor of size ${this.size} to shape [${newShape}]`);
    }

    const contiguous = isContiguous(this.shape, this.strides);

    if (this.isStringTensor()) {
      if (!contiguous) {
        const logicalStrides = computeStrides(this.shape);
        const out = new Array<string>(this.size);
        const data = this.data as string[];
        for (let i = 0; i < this.size; i++) {
          const off = offsetFromFlatIndex(i, logicalStrides, this.strides, this.offset);
          out[i] = data[off] ?? "";
        }
        return Tensor.fromStringArray({
          data: out,
          shape: newShape,
          device: this.device,
        });
      }
      return Tensor.fromStringArray({
        data: this.data,
        shape: newShape,
        device: this.device,
        offset: this.offset,
        strides: computeStrides(newShape),
      });
    }

    if (!this.isNumericTensor()) {
      throw new DTypeError("reshape is not defined for string dtype");
    }

    if (!contiguous) {
      const logicalStrides = computeStrides(this.shape);
      const data = this.data;
      if (data instanceof BigInt64Array) {
        const out = new BigInt64Array(this.size);
        for (let i = 0; i < this.size; i++) {
          const off = offsetFromFlatIndex(i, logicalStrides, this.strides, this.offset);
          out[i] = data[off] ?? 0n;
        }
        return new Tensor({
          data: out as TensorData<D>,
          shape: newShape,
          dtype: this.dtype,
          device: this.device,
          strides: computeStrides(newShape),
          offset: 0,
        });
      }
      const Ctor = dtypeToTypedArrayCtor(this.dtype);
      const out = new Ctor(this.size);
      if (!(out instanceof BigInt64Array)) {
        const numData = data as Exclude<TypedArray, BigInt64Array>;
        for (let i = 0; i < this.size; i++) {
          const off = offsetFromFlatIndex(i, logicalStrides, this.strides, this.offset);
          out[i] = numData[off] ?? 0;
        }
      }
      return new Tensor({
        data: out as TensorData<D>,
        shape: newShape,
        dtype: this.dtype,
        device: this.device,
        strides: computeStrides(newShape),
        offset: 0,
      });
    }

    return Tensor.fromTypedArray({
      data: this.data,
      shape: newShape,
      dtype: this.dtype,
      device: this.device,
      offset: this.offset,
      strides: computeStrides(newShape),
    });
  }

  /**
   * Flatten the tensor to a 1-dimensional array.
   *
   * Returns a new 1D tensor containing all elements, sharing the same underlying data.
   *
   * @returns A 1D tensor with shape [size]
   *
   * @example
   * ```ts
   * const matrix = tensor([[1, 2, 3], [4, 5, 6]]);
   * const flat = matrix.flatten();
   * console.log(flat.shape); // [6]
   * ```
   */
  flatten(this: Tensor<S, "string">): Tensor<[number], "string">;
  flatten(this: Tensor<S, Exclude<DType, "string">>): Tensor<[number], Exclude<DType, "string">>;
  flatten(this: Tensor<S, DType>): Tensor<[number], DType>;
  flatten(): Tensor<[number], DType> {
    return this.reshape([this.size]);
  }

  /**
   * Slice this tensor along one or more axes.
   *
   * @param ranges - Per-axis slice specifications (number, or {start?, end?, step?})
   * @returns New tensor with the sliced data
   *
   * @example
   * ```ts
   * const t = tensor([[1, 2, 3], [4, 5, 6]]);
   * t.slice(0);           // tensor([1, 2, 3])
   * t.slice({ start: 0, end: 1 }, { start: 1 }); // tensor([[2, 3]])
   * ```
   */
  slice(...ranges: SliceRange[]): Tensor {
    const ndim = this.ndim;
    if (ranges.length > ndim) {
      throw new ShapeError(
        `Too many indices for tensor: got ${ranges.length}, expected <= ${ndim}`
      );
    }

    const normalized = new Array<{ start: number; end: number; step: number }>(ndim);
    const outShape: number[] = [];

    for (let axis = 0; axis < ndim; axis++) {
      const dim = this.shape[axis] ?? 0;
      const range = ranges[axis] ?? { start: 0, end: dim, step: 1 };
      const nr = normalizeRange(range, dim);
      normalized[axis] = nr;
      if (typeof range !== "number") {
        const len =
          nr.step > 0
            ? Math.max(0, Math.ceil((nr.end - nr.start) / nr.step))
            : Math.max(0, Math.ceil((nr.start - nr.end) / -nr.step));
        outShape.push(len);
      }
    }

    const outSize = outShape.length === 0 ? 1 : outShape.reduce((a, b) => a * b, 1);
    const out =
      this.dtype === "string"
        ? new Array<string>(outSize)
        : new (dtypeToTypedArrayCtor(this.dtype))(outSize);

    const outStrides = new Array<number>(outShape.length);
    let stride = 1;
    for (let i = outShape.length - 1; i >= 0; i--) {
      outStrides[i] = stride;
      stride *= outShape[i] ?? 0;
    }

    const outNdim = outShape.length;
    for (let outFlat = 0; outFlat < outSize; outFlat++) {
      let rem = outFlat;
      const outIdx = new Array<number>(outNdim);
      for (let i = 0; i < outNdim; i++) {
        const s = outStrides[i] ?? 1;
        outIdx[i] = Math.floor(rem / s);
        rem %= s;
      }

      const inIdx = new Array<number>(ndim);
      let outAxis = 0;
      for (let axis = 0; axis < ndim; axis++) {
        const r = ranges[axis];
        const nr = normalized[axis];
        if (nr === undefined) {
          throw new DeepboxError("Internal error: missing normalized slice range");
        }
        if (typeof r === "number") {
          inIdx[axis] = nr.start;
        } else {
          inIdx[axis] = nr.start + (outIdx[outAxis] ?? 0) * nr.step;
          outAxis++;
        }
      }

      let inFlat = this.offset;
      for (let axis = 0; axis < ndim; axis++) {
        inFlat += (inIdx[axis] ?? 0) * (this.strides[axis] ?? 0);
      }

      if (Array.isArray(out) && Array.isArray(this.data)) {
        out[outFlat] = this.data[inFlat] ?? "";
      } else if (out instanceof BigInt64Array && this.data instanceof BigInt64Array) {
        out[outFlat] = getBigIntElement(this.data, inFlat);
      } else if (
        !Array.isArray(out) &&
        !(out instanceof BigInt64Array) &&
        !Array.isArray(this.data) &&
        !(this.data instanceof BigInt64Array)
      ) {
        out[outFlat] = getNumericElement(this.data, inFlat);
      }
    }

    if (Array.isArray(out)) {
      return Tensor.fromStringArray({
        data: out,
        shape: outShape.length === 0 ? [] : outShape,
        device: this.device,
      });
    }

    if (this.dtype === "string") {
      throw new DeepboxError("Internal error: string dtype but non-array data");
    }

    return Tensor.fromTypedArray({
      data: out as TypedArray,
      shape: outShape.length === 0 ? [] : outShape,
      dtype: this.dtype as Exclude<DType, "string">,
      device: this.device,
    });
  }

  at(...indices: number[]): ElementOf<D> {
    if (indices.length !== this.ndim) {
      throw new ShapeError(
        `Expected ${this.ndim} indices for a ${this.ndim}D tensor; received ${indices.length}`
      );
    }

    let flat = this.offset;
    for (let axis = 0; axis < this.ndim; axis++) {
      const dim = this.shape[axis] ?? 0;
      const stride = this.strides[axis] ?? 0;
      const raw = indices[axis] ?? 0;
      const idx = raw < 0 ? dim + raw : raw;

      if (!Number.isInteger(idx)) {
        throw new InvalidParameterError(
          `index for axis ${axis} must be an integer; received ${String(raw)}`,
          `indices[${axis}]`,
          raw
        );
      }
      if (idx < 0 || idx >= dim) {
        throw new IndexError(`index ${raw} is out of bounds for dimension of size ${dim}`);
      }

      flat += idx * stride;
    }

    const v = this.data[flat];
    if (v === undefined) {
      throw new DeepboxError("Internal error: computed flat index is out of bounds");
    }
    return v as ElementOf<D>;
  }

  toArray(): unknown {
    const recur = (axis: number, baseOffset: number): unknown => {
      if (axis === this.ndim) {
        const v = this.data[baseOffset];
        if (v === undefined) {
          throw new DeepboxError("Internal error: computed flat index is out of bounds");
        }
        return v;
      }

      const dim = this.shape[axis] ?? 0;
      const stride = this.strides[axis] ?? 0;
      const out = new Array<unknown>(dim);
      for (let i = 0; i < dim; i++) {
        out[i] = recur(axis + 1, baseOffset + i * stride);
      }
      return out;
    };

    return recur(0, this.offset);
  }

  /**
   * Return a human-readable string representation of this tensor.
   *
   * Scalars print as a bare value, 1-D tensors as a bracketed list, and
   * higher-rank tensors use nested brackets with newline separators.
   * Large dimensions are summarized with an ellipsis.
   *
   * @param maxElements - Maximum number of elements to display per
   *   dimension before summarizing (default: 6).
   * @returns Formatted string representation
   *
   * @example
   * ```ts
   * const t = tensor([1, 2, 3]);
   * t.toString(); // "tensor([1, 2, 3], dtype=float32)"
   * ```
   */
  toString(maxElements = 6): string {
    const formatValue = (v: unknown): string => {
      if (typeof v === "bigint") return v.toString();
      if (typeof v === "number") {
        if (Number.isInteger(v) && Math.abs(v) < 1e15) return v.toString();
        return v.toPrecision(4);
      }
      if (typeof v === "string") return JSON.stringify(v);
      return String(v);
    };

    const formatArray = (arr: unknown, depth: number): string => {
      if (!Array.isArray(arr)) return formatValue(arr);

      const len = arr.length;
      if (len === 0) return "[]";

      const half = Math.floor(maxElements / 2);
      const pad = " ".repeat(depth + 7);

      let items: string[];
      if (len <= maxElements) {
        items = arr.map((el) => formatArray(el, depth + 1));
      } else {
        const head = arr.slice(0, half).map((el) => formatArray(el, depth + 1));
        const tail = arr.slice(len - half).map((el) => formatArray(el, depth + 1));
        items = [...head, "...", ...tail];
      }

      if (depth === 0 && this.ndim === 1) {
        return `[${items.join(", ")}]`;
      }
      if (!Array.isArray(arr[0])) {
        return `[${items.join(", ")}]`;
      }
      return `[${items.join(`\n${pad}`)}]`;
    };

    // Scalar (0-D)
    if (this.ndim === 0) {
      const v = this.data[this.offset];
      return `tensor(${formatValue(v)}, dtype=${this.dtype})`;
    }

    const nested = this.toArray();
    const body = formatArray(nested, 0);
    return `tensor(${body}, dtype=${this.dtype})`;
  }
}
