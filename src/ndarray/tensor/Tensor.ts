import type { Axis, Device, DType, ElementOf, Shape, TensorLike, TypedArray } from "../../core";
import {
  DeepboxError,
  DeviceError,
  DTypeError,
  dtypeToTypedArrayCtor,
  getBigIntElement,
  IndexError,
  InvalidParameterError,
  ShapeError,
  shapeToSize,
  validateShape,
} from "../../core";
import type { DeviceBuffer, KernelBackend } from "../../core/backend/kernels";
import {
  ensureBackendAvailable,
  getKernelBackend,
  requireKernelBackend,
} from "../../core/backend/registry";
import { matmul as matmulOp } from "../linalg/basic";
import { dot as dotOp } from "../linalg/index";
import {
  elu as eluOp,
  type GeluApproximation,
  type GeluOptions,
  gelu as geluOp,
  hardtanh as hardtanhOp,
  leakyRelu as leakyReluOp,
  logSigmoid as logSigmoidOp,
  logSoftmax as logSoftmaxOp,
  mish as mishOp,
  relu as reluOp,
  selu as seluOp,
  sigmoid as sigmoidOp,
  softmax as softmaxOp,
  softplus as softplusOp,
  swish as swishOp,
  tanhshrink as tanhshrinkOp,
} from "../ops/activation";
import {
  abs as absOp,
  add as addOp,
  addScalar as addScalarOp,
  clip as clipOp,
  div as divOp,
  maximum as maximumOp,
  minimum as minimumOp,
  mul as mulOp,
  mulScalar as mulScalarOp,
  neg as negOp,
  pow as powOp,
  sub as subOp,
} from "../ops/arithmetic";
import {
  equal as equalOp,
  greaterEqual as greaterEqualOp,
  greater as greaterOp,
  isnan as isnanOp,
  lessEqual as lessEqualOp,
  less as lessOp,
  notEqual as notEqualOp,
} from "../ops/comparison";
import {
  ceil as ceilOp,
  expm1 as expm1Op,
  exp as expOp,
  floor as floorOp,
  log1p as log1pOp,
  log as logOp,
  round as roundOp,
  sqrt as sqrtOp,
  square as squareOp,
} from "../ops/math";
import {
  all as allOp,
  any as anyOp,
  argmax as argmaxOp,
  argmin as argminOp,
  cumsum as cumsumOp,
  max as maxOp,
  mean as meanOp,
  min as minOp,
  prod as prodOp,
  std as stdOp,
  sum as sumOp,
  variance as varianceOp,
} from "../ops/reduction";
import { argsort as argsortOp, sort as sortOp } from "../ops/sorting";
import { cos as cosOp, sin as sinOp, tanh as tanhOp, tan as tanOp } from "../ops/trigonometry";
import { clone as cloneOp, flip as flipOp } from "../ops/utils";
import { roundToBFloat16, roundToFloat16 } from "./float16";
import { gather as gatherOp } from "./indexing";
import { transpose as transposeOp } from "./shape";
import { squeeze as squeezeOp, unsqueeze as unsqueezeOp } from "./shape_ops";
import { normalizeRange, type SliceRange } from "./slice_helpers";
import { isDenseLayout } from "./strides";

/** Options for tensor dtype and device configuration. */
export type TensorOptions = {
  readonly dtype: DType;
  readonly device: Device;
};

type TensorData<D extends DType> = D extends "string" ? string[] : TypedArray;

function isTypedArrayForDType(data: TypedArray, dtype: DType): boolean {
  if (dtype === "string") return false;
  // Half-precision tensors keep float32 host storage (the numeric values);
  // the dtype label distinguishes them and drives device packing / rounding.
  if (dtype === "float16" || dtype === "bfloat16") return data instanceof Float32Array;
  if (dtype === "float32") return data instanceof Float32Array;
  if (dtype === "float64") return data instanceof Float64Array;
  if (dtype === "int32") return data instanceof Int32Array;
  if (dtype === "int64") return data instanceof BigInt64Array;
  if (dtype === "uint8" || dtype === "bool") return data instanceof Uint8Array;
  return false;
}

function assertTypedArrayForDType(data: TypedArray, dtype: DType): void {
  if (dtype === "complex64" || dtype === "complex128") {
    throw new DTypeError(
      `Tensors do not support dtype ${dtype} yet; ` +
        "use Complex64Array / Complex128Array for complex data."
    );
  }
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

  if (!Number.isInteger(offset)) {
    throw new InvalidParameterError(
      `offset must be an integer; received ${String(offset)}`,
      "offset",
      offset
    );
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
 * Reference-counted owner of a {@link DeviceBuffer}.
 *
 * Multiple tensors (views) can share one device allocation; the buffer is
 * released on the owning backend once every holder has released it,
 * either explicitly via `Tensor.dispose()` or lazily when a tensor is
 * garbage collected.
 *
 * @internal
 */
export class DeviceBufferOwner {
  private refs = 0;
  private freed = false;

  constructor(
    readonly buffer: DeviceBuffer,
    readonly backend: KernelBackend
  ) {}

  acquire(): void {
    if (this.freed) {
      throw new DeviceError("Cannot acquire a device buffer that has already been freed");
    }
    this.refs++;
  }

  release(): void {
    if (this.freed) return;
    this.refs--;
    if (this.refs <= 0) {
      this.freed = true;
      this.backend.free(this.buffer);
    }
  }

  get isFreed(): boolean {
    return this.freed;
  }
}

/** Best-effort GC hook so unreachable device tensors release their memory. */
const deviceBufferFinalizer = new FinalizationRegistry<DeviceBufferOwner>((owner) => {
  owner.release();
});

/**
 * Dtypes with device kernel support. float32 is the default; float16 and
 * bfloat16 are half-precision device dtypes (float16 requires a backend with
 * the `shader-f16` feature, the backend throws a clear DeviceError at upload
 * time if the hardware lacks it). Other dtypes stay on the host.
 */
function assertDeviceDType(
  dtype: DType,
  device: Device
): asserts dtype is "float32" | "float16" | "bfloat16" {
  if (dtype !== "float32" && dtype !== "float16" && dtype !== "bfloat16") {
    throw new DTypeError(
      `Device "${device}" supports float32, float16 and bfloat16 tensors only; ` +
        `received dtype ${dtype}. ` +
        `Convert with astype('float32') before moving the tensor to ${device}.`
    );
  }
}

/**
 * Buffer offsets of every element of a strided view in logical (row-major)
 * order. An odometer walk replaces a per-element division chain, so copying a
 * non-contiguous view costs O(size) instead of O(size * ndim).
 */
function stridedOffsets(shape: Shape, strides: readonly number[], offset: number): Float64Array {
  const size = shapeToSize(shape);
  const out = new Float64Array(size);
  const ndim = shape.length;
  if (size === 0) return out;
  if (ndim === 0) {
    out[0] = offset;
    return out;
  }
  const last = ndim - 1;
  const lastDim = shape[last] ?? 1;
  const lastStride = strides[last] ?? 0;
  const index = new Array<number>(ndim).fill(0);
  let base = offset;
  let pos = 0;
  while (pos < size) {
    for (let j = 0; j < lastDim; j++) out[pos++] = base + j * lastStride;
    let axis = last - 1;
    while (axis >= 0) {
      const stride = strides[axis] ?? 0;
      const dim = shape[axis] ?? 1;
      index[axis] = (index[axis] ?? 0) + 1;
      base += stride;
      if ((index[axis] ?? 0) < dim) break;
      base -= stride * dim;
      index[axis] = 0;
      axis--;
    }
    if (axis < 0) break;
  }
  return out;
}

const INT64_LIMIT = 2 ** 63;

/**
 * Convert a finite number to an int64 BigInt, truncating toward zero.
 * Throws instead of letting the BigInt64Array store silently wrap values
 * outside `[-2^63, 2^63)`.
 *
 * @internal
 */
export function toInt64(value: number, context: string): bigint {
  if (!Number.isFinite(value)) {
    throw new DTypeError(`${context}: cannot convert non-finite value ${value} to int64`);
  }
  const t = Math.trunc(value);
  if (t >= INT64_LIMIT || t < -INT64_LIMIT) {
    throw new DTypeError(`${context}: value ${value} is outside the int64 range`);
  }
  return BigInt(t);
}

/** Copy a readonly axis list into the mutable form the reduction ops take. */
function axisArg(axis: Axis | readonly Axis[] | undefined): Axis | Axis[] | undefined {
  return axis === undefined || !Array.isArray(axis) ? (axis as Axis | undefined) : [...axis];
}

/**
 * Turn the right-hand side of a fluent binary method into a tensor.
 *
 * A tensor is returned as it is. A number becomes a 0-d tensor that follows the
 * scalar rules of `addScalar`: float tensors keep their dtype, integer tensors
 * keep theirs for whole numbers (`bool` becomes `int32`), and a fractional,
 * NaN or infinite number gives `float32`. The number therefore never upcasts
 * the tensor, as in PyTorch.
 *
 * @internal Shared with `GradTensor`; not part of the public API.
 */
export function scalarOperand(t: Tensor, other: number | Tensor, op: string): Tensor {
  if (typeof other !== "number") {
    if (other instanceof Tensor) return other;
    throw new InvalidParameterError(
      `${op}: operand must be a number or a Tensor; received ${typeof other}`,
      "other",
      other
    );
  }
  const make = (data: TypedArray, dtype: Exclude<DType, "string">): Tensor =>
    Tensor.fromTypedArray({ data, shape: [], dtype, device: "cpu" });
  const dtype = t.dtype;
  if (dtype === "float64") return make(new Float64Array([other]), "float64");
  if (dtype === "float32") return make(new Float32Array([other]), "float32");
  if (dtype === "float16") return make(new Float32Array([roundToFloat16(other)]), "float16");
  if (dtype === "bfloat16") return make(new Float32Array([roundToBFloat16(other)]), "bfloat16");
  if (!Number.isInteger(other) || dtype === "string") {
    return make(new Float32Array([other]), "float32");
  }
  const fitsInt32 = other >= -(2 ** 31) && other < 2 ** 31;
  if (dtype === "uint8" && other >= 0 && other <= 255) {
    return make(new Uint8Array([other]), "uint8");
  }
  if (dtype !== "int64" && fitsInt32) return make(new Int32Array([other]), "int32");
  // Whole numbers outside the dtype range widen the scalar instead of wrapping,
  // so comparisons stay exact.
  return make(new BigInt64Array([toInt64(other, op)]), "int64");
}

/**
 * Shortest decimal text that parses back to `value` after rounding with
 * `round` (the dtype's own rounding), e.g. a float32 `0.1` prints as "0.1"
 * rather than "0.10000000149011612".
 */
function shortestRoundTripString(value: number, round: (x: number) => number): string {
  if (!Number.isFinite(value) || value === 0) return String(value);
  for (let precision = 1; precision < 9; precision++) {
    const candidate = Number(value.toPrecision(precision));
    if (round(candidate) === value) return String(candidate);
  }
  return String(Number(value.toPrecision(9)));
}

/**
 * Replace a single `-1` dimension of a requested shape with the size implied
 * by the element count. Shapes without `-1` are returned unchanged.
 */
function resolveInferredShape(shape: Shape, totalSize: number): Shape {
  if (!Array.isArray(shape)) return shape;
  let inferIdx = -1;
  let known = 1;
  for (let i = 0; i < shape.length; i++) {
    const d = shape[i];
    if (d === -1) {
      if (inferIdx !== -1) {
        throw new ShapeError("Only one dimension can be -1 in reshape");
      }
      inferIdx = i;
    } else if (typeof d === "number" && Number.isInteger(d) && d >= 0) {
      known *= d;
    } else {
      // Let validateShape report the malformed dimension.
      return shape;
    }
  }
  if (inferIdx === -1) return shape;
  if (known === 0 || totalSize % known !== 0) {
    throw new ShapeError(
      `Cannot infer dimension for shape [${shape}] with total size ${totalSize}`
    );
  }
  const resolved = [...shape];
  resolved[inferIdx] = totalSize / known;
  return resolved;
}

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
 * - Device placement (`cpu`, `webgpu`, `wasm`): with a registered
 *   `WebGpuBackend`, tensors on `webgpu` store their data in GPU memory and
 *   the accelerated op set (element-wise arithmetic, activations, matmul,
 *   full reductions) executes on the GPU, so move data with `await t.to(device)`
 *   / `await t.cpu()`. With a registered `WasmBackend`, `wasm` tensors keep
 *   zero-copy host storage and accelerate eligible float32 arithmetic with
 *   SIMD. Ops a device cannot execute throw `DeviceError` with a transfer
 *   hint instead of silently computing elsewhere (see Devices & execution on
 *   DeepboxDocs).
 *
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox Tensors}
 * @see {@link https://deepbox.dev/docs/devices-and-execution | Devices & execution}
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
  readonly strides: readonly number[];
  readonly offset: number;
  readonly size: number;
  readonly ndim: number;

  /**
   * Underlying host storage (TypedArray, or `string[]` for string dtype).
   *
   * Host tensors expose this as a plain own property (no accessor overhead
   * in hot loops). Tensors whose data lives in device memory (e.g. `webgpu`)
   * get a per-instance throwing getter instead: device buffers cannot be
   * read synchronously, so transfer first with `await t.cpu()`.
   *
   * @throws {DeviceError} If the tensor's storage is on a non-CPU device
   */
  declare readonly data: TensorData<D>;
  /** Device storage owner; `null` when the tensor lives in host memory. */
  private readonly bufferOwner: DeviceBufferOwner | null;
  private disposed = false;

  private constructor(args: {
    readonly data?: TensorData<D>;
    readonly bufferOwner?: DeviceBufferOwner;
    readonly shape: S;
    readonly dtype: D;
    readonly device: Device;
    readonly strides: readonly number[];
    readonly offset: number;
  }) {
    this.bufferOwner = args.bufferOwner ?? null;
    if ((args.data === undefined) === (this.bufferOwner === null)) {
      throw new DeepboxError(
        "Internal error: Tensor requires exactly one of host data or a device buffer"
      );
    }
    if (args.data !== undefined) {
      // Own data property: hot loops read it at plain-field speed.
      (this as { data: TensorData<D> }).data = args.data;
    } else {
      const device = args.device;
      Object.defineProperty(this, "data", {
        get(): never {
          throw new DeviceError(
            `Cannot access data of a tensor stored on device "${device}". ` +
              `This operation is not accelerated on ${device}; ` +
              "move the tensor to the CPU first with `await t.cpu()`. " +
              "See https://deepbox.dev/docs/devices-and-execution for the set of device-accelerated ops."
          );
        },
        enumerable: false,
        configurable: false,
      });
    }
    this.shape = args.shape;
    this.dtype = args.dtype;
    this.device = args.device;
    this.strides = args.strides;
    this.offset = args.offset;

    this.ndim = this.shape.length;
    this.size = shapeToSize(this.shape);

    if (this.bufferOwner) {
      this.bufferOwner.acquire();
      deviceBufferFinalizer.register(this, this.bufferOwner, this);
    }
  }

  /**
   * `true` when the tensor's storage lives in device memory
   * (its `data` cannot be read synchronously).
   */
  get isDeviceTensor(): boolean {
    return this.bufferOwner !== null;
  }

  /**
   * The device buffer backing this tensor, or `null` for host tensors.
   *
   * Advanced API for custom backend integrations.
   */
  get deviceBuffer(): DeviceBuffer | null {
    this.assertAlive();
    return this.bufferOwner?.buffer ?? null;
  }

  /**
   * Shared buffer owner for view creation and dispatch.
   *
   * @internal
   */
  get __bufferOwner(): DeviceBufferOwner | null {
    this.assertAlive();
    return this.bufferOwner;
  }

  /**
   * Release this tensor's device memory reference immediately.
   *
   * Device buffers are also released automatically when tensors are garbage
   * collected, but explicit disposal is deterministic and recommended for
   * large buffers. Host (CPU) tensors ignore this call. Views share the
   * underlying allocation; it is freed when the last holder releases it.
   * Calling `dispose()` again is a no-op. Using a disposed tensor (reshaping,
   * slicing, moving, or passing it to an op) throws `DeviceError`, even while
   * other views still keep the allocation alive.
   */
  dispose(): void {
    if (this.disposed || !this.bufferOwner) return;
    this.disposed = true;
    deviceBufferFinalizer.unregister(this);
    this.bufferOwner.release();
  }

  private assertAlive(): void {
    if (this.disposed) {
      throw new DeviceError(
        `Cannot use a tensor that was disposed (device: ${this.device}); ` +
          "create a new tensor or keep a reference to a live view."
      );
    }
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
    const device = ensureBackendAvailable(args.device, "tensor device");
    const offset = args.offset ?? 0;
    const strides = args.strides ?? computeStrides(args.shape);
    assertTypedArrayForDType(args.data, args.dtype);
    validateStrides(args.shape, strides, offset, args.data.length);

    // Devices with kernel backends own their tensors' memory: upload the
    // host data instead of mislabeling host storage with a device name.
    // Host-accelerator devices (e.g. `wasm`) keep zero-copy host storage.
    const kernelBackend = device === "cpu" ? null : getKernelBackend(device);
    if (kernelBackend) {
      assertDeviceDType(args.dtype, device);
      const buffer = kernelBackend.upload(new Float32Array(args.data as Float32Array), args.dtype);
      const owner = new DeviceBufferOwner(buffer, kernelBackend);
      return new Tensor<S, D>({
        bufferOwner: owner,
        shape: args.shape,
        dtype: args.dtype,
        device,
        offset,
        strides,
      });
    }

    return new Tensor<S, D>({
      data: args.data,
      shape: args.shape,
      dtype: args.dtype,
      device,
      offset,
      strides,
    });
  }

  /**
   * Create a tensor over existing device memory.
   *
   * The tensor takes shared ownership of the buffer: pass a fresh
   * {@link DeviceBufferOwner} for newly allocated memory, or an existing
   * owner to create a view over the same allocation.
   *
   * Advanced API for backend integrations and the internal dispatch layer.
   */
  static fromDeviceBuffer<S extends Shape>(args: {
    readonly owner: DeviceBufferOwner;
    readonly shape: S;
    readonly device: Device;
    readonly offset?: number;
    readonly strides?: readonly number[];
  }): Tensor<S, "float32"> {
    validateShape(args.shape);
    const offset = args.offset ?? 0;
    const strides = args.strides ?? computeStrides(args.shape);
    validateStrides(args.shape, strides, offset, args.owner.buffer.size);

    // The buffer's dtype (f16/bf16/f32) rounds-trips onto the tensor so device
    // op results keep their half-precision label; absent means float32. The
    // static return type stays float32 (device tensors are read back via
    // `await t.cpu()`, which reconstructs the correct host dtype).
    const dtype = (args.owner.buffer.dtype ?? "float32") as "float32";
    return new Tensor<S, "float32">({
      bufferOwner: args.owner,
      shape: args.shape,
      dtype,
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
    const device = ensureBackendAvailable(args.device ?? "cpu", "tensor device");
    const offset = args.offset ?? 0;
    const strides = args.strides ?? computeStrides(args.shape);
    validateStrides(args.shape, strides, offset, args.data.length);

    return new Tensor({
      data: args.data,
      shape: args.shape,
      dtype: "string",
      device,
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
   * This is a raw reinterpretation: it does not copy, and without explicit
   * `strides` the new shape is read as a contiguous row-major layout starting
   * at `offset` (default: this tensor's offset). Writes to the shared buffer
   * are visible through every view. For a layout-aware reshape of arbitrary
   * (including non-contiguous) tensors use {@link Tensor.reshape}.
   *
   * @param shape - Shape of the view; its size must equal `this.size`.
   * @param strides - Element strides of the view (default: row-major for `shape`).
   * @param offset - Offset into the buffer (default: this tensor's offset).
   * @throws {ShapeError} If the sizes differ or the view reaches outside the buffer.
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
    this.assertAlive();
    validateShape(shape);
    if (shapeToSize(shape) !== this.size) {
      throw ShapeError.mismatch([this.size], [shapeToSize(shape)], "view");
    }
    if (this.bufferOwner) {
      return Tensor.fromDeviceBuffer({
        owner: this.bufferOwner,
        shape,
        device: this.device,
        offset,
        strides: strides ?? computeStrides(shape),
      });
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
   * Reshape the tensor to a new shape.
   *
   * Contiguous tensors return a view that shares the underlying data. A
   * non-contiguous view (for example the result of `transpose`) is copied
   * into logical (row-major) order first. The total number of elements must
   * stay the same.
   *
   * One dimension may be `-1`; it is inferred from the remaining dimensions,
   * like NumPy's `reshape`.
   *
   * @param newShape - The desired shape for the tensor
   * @returns A tensor with the specified shape
   * @throws {ShapeError} If the new shape is incompatible with the tensor's size
   *
   * @example
   * ```ts
   * const t = tensor([1, 2, 3, 4, 5, 6]);
   * const reshaped = t.reshape([2, 3]);
   * console.log(reshaped.shape); // [2, 3]
   *
   * const inferred = t.reshape([3, -1]);
   * console.log(inferred.shape); // [3, 2]
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
  reshape<S2 extends Shape>(rawShape: S2): Tensor<S2, DType> {
    this.assertAlive();
    const newShape = resolveInferredShape(rawShape, this.size) as S2;
    validateShape(newShape);
    const newSize = shapeToSize(newShape);
    if (newSize !== this.size) {
      throw new ShapeError(`Cannot reshape tensor of size ${this.size} to shape [${newShape}]`);
    }

    const contiguous = isDenseLayout(this.shape, this.strides);

    if (this.bufferOwner) {
      if (contiguous) {
        return Tensor.fromDeviceBuffer({
          owner: this.bufferOwner,
          shape: newShape,
          device: this.device,
          offset: this.offset,
          strides: computeStrides(newShape),
        });
      }
      // Non-contiguous device view: materialize on-device with a copy kernel.
      const backend = requireKernelBackend(this.device, "reshape");
      const out = backend.unary("copy", this.bufferOwner.buffer, {
        shape: this.shape,
        strides: this.strides,
        offset: this.offset,
      });
      return Tensor.fromDeviceBuffer({
        owner: new DeviceBufferOwner(out, backend),
        shape: newShape,
        device: this.device,
      });
    }

    if (this.isStringTensor()) {
      if (!contiguous) {
        const offsets = stridedOffsets(this.shape, this.strides, this.offset);
        const out = new Array<string>(this.size);
        const data = this.data as string[];
        for (let i = 0; i < this.size; i++) {
          out[i] = data[offsets[i] as number] ?? "";
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
      const offsets = stridedOffsets(this.shape, this.strides, this.offset);
      const data = this.data;
      if (data instanceof BigInt64Array) {
        const out = new BigInt64Array(this.size);
        for (let i = 0; i < this.size; i++) {
          out[i] = data[offsets[i] as number] ?? 0n;
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
          out[i] = numData[offsets[i] as number] ?? 0;
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

  // ---------------------------------------------------------------------------
  // Fluent method surface. Every method delegates to the functional op of the
  // same name, so dtype promotion, broadcasting and errors are identical to
  // calling `add(t, other)`, `sum(t, axis)` and so on. The ops are only looked
  // up when a method is called, never while the module is evaluated.
  // ---------------------------------------------------------------------------

  /**
   * Element-wise sum with a tensor or a number (functional form: `add`).
   *
   * A number never upcasts the tensor, see `addScalar`.
   *
   * @param other - Tensor (broadcast) or number
   * @returns New tensor
   *
   * @example
   * ```ts
   * tensor([1, 2, 3]).add(10); // [11, 12, 13]
   * ```
   */
  add(other: number | Tensor): Tensor {
    return typeof other === "number" ? addScalarOp(this, other) : addOp(this, other);
  }

  /**
   * Element-wise difference with a tensor or a number (functional form: `sub`).
   *
   * @param other - Tensor (broadcast) or number
   * @returns New tensor
   *
   * @example
   * ```ts
   * tensor([5, 7]).sub(tensor([1, 2])); // [4, 5]
   * ```
   */
  sub(other: number | Tensor): Tensor {
    // Subtracting a number is adding its negation in IEEE arithmetic, and the
    // scalar kernel avoids the broadcast path.
    if (
      typeof other === "number" &&
      (this.dtype === "float32" || this.dtype === "float64") &&
      this.device === "cpu"
    ) {
      return addScalarOp(this, -other);
    }
    return subOp(this, scalarOperand(this, other, "sub"));
  }

  /**
   * Element-wise product with a tensor or a number (functional form: `mul`).
   *
   * @param other - Tensor (broadcast) or number
   * @returns New tensor
   *
   * @example
   * ```ts
   * tensor([1, 2, 3]).mul(2); // [2, 4, 6]
   * ```
   */
  mul(other: number | Tensor): Tensor {
    return typeof other === "number" ? mulScalarOp(this, other) : mulOp(this, other);
  }

  /**
   * Element-wise true division by a tensor or a number (functional form: `div`).
   *
   * @param other - Tensor (broadcast) or number
   * @returns New tensor; integer input gives `float32`
   *
   * @example
   * ```ts
   * tensor([1, 2, 3]).div(2); // [0.5, 1, 1.5]
   * ```
   */
  div(other: number | Tensor): Tensor {
    return divOp(this, scalarOperand(this, other, "div"));
  }

  /**
   * Element-wise power (functional form: `pow`).
   *
   * @param exponent - Tensor (broadcast) or number
   * @returns New tensor
   *
   * @example
   * ```ts
   * tensor([1, 2, 3]).pow(2); // [1, 4, 9]
   * ```
   */
  pow(exponent: number | Tensor): Tensor {
    return powOp(this, scalarOperand(this, exponent, "pow"));
  }

  /**
   * Element-wise negation (functional form: `neg`).
   *
   * @example
   * ```ts
   * tensor([1, -2]).neg(); // [-1, 2]
   * ```
   */
  neg(): Tensor {
    return negOp(this);
  }

  /**
   * Element-wise absolute value (functional form: `abs`).
   *
   * @example
   * ```ts
   * tensor([-1, 2]).abs(); // [1, 2]
   * ```
   */
  abs(): Tensor {
    return absOp(this);
  }

  /**
   * Element-wise exponential (functional form: `exp`).
   *
   * @example
   * ```ts
   * tensor([0, 1]).exp(); // [1, 2.718...]
   * ```
   */
  exp(): Tensor {
    return expOp(this);
  }

  /**
   * Element-wise natural logarithm (functional form: `log`).
   *
   * @example
   * ```ts
   * tensor([1, Math.E]).log(); // [0, 1]
   * ```
   */
  log(): Tensor {
    return logOp(this);
  }

  /**
   * Element-wise square root (functional form: `sqrt`).
   *
   * @example
   * ```ts
   * tensor([4, 9]).sqrt(); // [2, 3]
   * ```
   */
  sqrt(): Tensor {
    return sqrtOp(this);
  }

  /**
   * Element-wise square (functional form: `square`).
   *
   * @example
   * ```ts
   * tensor([2, 3]).square(); // [4, 9]
   * ```
   */
  square(): Tensor {
    return squareOp(this);
  }

  /**
   * Element-wise sine (functional form: `sin`).
   *
   * @example
   * ```ts
   * tensor([0, Math.PI / 2]).sin(); // [0, 1]
   * ```
   */
  sin(): Tensor {
    return sinOp(this);
  }

  /**
   * Element-wise cosine (functional form: `cos`).
   *
   * @example
   * ```ts
   * tensor([0]).cos(); // [1]
   * ```
   */
  cos(): Tensor {
    return cosOp(this);
  }

  /**
   * Element-wise hyperbolic tangent (functional form: `tanh`).
   *
   * @example
   * ```ts
   * tensor([0]).tanh(); // [0]
   * ```
   */
  tanh(): Tensor {
    return tanhOp(this);
  }

  /**
   * Element-wise logistic sigmoid (functional form: `sigmoid`).
   *
   * @example
   * ```ts
   * tensor([0]).sigmoid(); // [0.5]
   * ```
   */
  sigmoid(): Tensor {
    return sigmoidOp(this);
  }

  /**
   * Element-wise rectified linear unit (functional form: `relu`).
   *
   * @example
   * ```ts
   * tensor([-1, 2]).relu(); // [0, 2]
   * ```
   */
  relu(): Tensor {
    return reluOp(this);
  }

  /**
   * Element-wise `exp(x) - 1`, accurate for small `x` (functional form: `expm1`).
   *
   * @example
   * ```ts
   * tensor([0, 1]).expm1(); // [0, 1.718...]
   * ```
   */
  expm1(): Tensor {
    return expm1Op(this);
  }

  /**
   * Element-wise `log(1 + x)`, accurate for small `x` (functional form: `log1p`).
   *
   * @example
   * ```ts
   * tensor([0, 1]).log1p(); // [0, 0.693...]
   * ```
   */
  log1p(): Tensor {
    return log1pOp(this);
  }

  /**
   * Element-wise tangent (functional form: `tan`).
   *
   * @example
   * ```ts
   * tensor([0]).tan(); // [0]
   * ```
   */
  tan(): Tensor {
    return tanOp(this);
  }

  /**
   * Element-wise larger of two tensors, or of a tensor and a number
   * (functional form: `maximum`). NaN propagates, as in `numpy.maximum`.
   *
   * @param other - Tensor (broadcast) or number
   *
   * @example
   * ```ts
   * tensor([1, 5, 3]).maximum(tensor([4, 2, 6])); // [4, 5, 6]
   * ```
   */
  maximum(other: number | Tensor): Tensor {
    return maximumOp(this, scalarOperand(this, other, "maximum"));
  }

  /**
   * Element-wise smaller of two tensors, or of a tensor and a number
   * (functional form: `minimum`). NaN propagates, as in `numpy.minimum`.
   *
   * @param other - Tensor (broadcast) or number
   *
   * @example
   * ```ts
   * tensor([1, 5, 3]).minimum(tensor([4, 2, 6])); // [1, 2, 3]
   * ```
   */
  minimum(other: number | Tensor): Tensor {
    return minimumOp(this, scalarOperand(this, other, "minimum"));
  }

  /**
   * Leaky ReLU (functional form: `leakyRelu`).
   *
   * @param negativeSlope - Slope for negative inputs (default 0.01)
   *
   * @example
   * ```ts
   * tensor([-2, 3]).leakyRelu(0.1); // [-0.2, 3]
   * ```
   */
  leakyRelu(negativeSlope = 0.01): Tensor {
    return leakyReluOp(this, negativeSlope);
  }

  /**
   * Exponential linear unit (functional form: `elu`).
   *
   * @param alpha - Scale of the negative part (default 1)
   *
   * @example
   * ```ts
   * tensor([-1, 2]).elu(); // [-0.632..., 2]
   * ```
   */
  elu(alpha = 1.0): Tensor {
    return eluOp(this, alpha);
  }

  /**
   * Scaled exponential linear unit (functional form: `selu`).
   *
   * @example
   * ```ts
   * tensor([1, 0]).selu(); // [1.0507..., 0]
   * ```
   */
  selu(): Tensor {
    return seluOp(this);
  }

  /**
   * Gaussian error linear unit (functional form: `gelu`).
   *
   * @param options - `{ approximate?: "tanh" | "none" }` (default `"tanh"`), or the bare string
   *
   * @example
   * ```ts
   * tensor([1]).gelu({ approximate: "none" }); // [0.841...]
   * ```
   */
  gelu(options: GeluApproximation | GeluOptions = {}): Tensor {
    return geluOp(this, options);
  }

  /**
   * Softplus, `log(1 + exp(x))` (functional form: `softplus`).
   *
   * @example
   * ```ts
   * tensor([0]).softplus(); // [0.693...]
   * ```
   */
  softplus(): Tensor {
    return softplusOp(this);
  }

  /**
   * Mish, `x * tanh(softplus(x))` (functional form: `mish`).
   *
   * @example
   * ```ts
   * tensor([0, 1]).mish(); // [0, 0.865...]
   * ```
   */
  mish(): Tensor {
    return mishOp(this);
  }

  /**
   * Swish (SiLU), `x * sigmoid(x)` (functional form: `swish`).
   *
   * @example
   * ```ts
   * tensor([0, 1]).swish(); // [0, 0.731...]
   * ```
   */
  swish(): Tensor {
    return swishOp(this);
  }

  /**
   * Log of the logistic function, `log(sigmoid(x))` (functional form: `logSigmoid`).
   *
   * @example
   * ```ts
   * tensor([0]).logSigmoid(); // [-0.693...]
   * ```
   */
  logSigmoid(): Tensor {
    return logSigmoidOp(this);
  }

  /**
   * Clamp to `[minVal, maxVal]` (functional form: `hardtanh`).
   *
   * @param minVal - Lower bound (default -1)
   * @param maxVal - Upper bound (default 1)
   *
   * @example
   * ```ts
   * tensor([-2, 0.5, 3]).hardtanh(); // [-1, 0.5, 1]
   * ```
   */
  hardtanh(minVal = -1, maxVal = 1): Tensor {
    return hardtanhOp(this, minVal, maxVal);
  }

  /**
   * `x - tanh(x)` (functional form: `tanhshrink`).
   *
   * @example
   * ```ts
   * tensor([0]).tanhshrink(); // [0]
   * ```
   */
  tanhshrink(): Tensor {
    return tanhshrinkOp(this);
  }

  /**
   * Softmax along an axis (functional form: `softmax`).
   *
   * @param axis - Axis to normalise (default -1)
   *
   * @example
   * ```ts
   * tensor([[1, 1]]).softmax(); // [[0.5, 0.5]]
   * ```
   */
  softmax(axis: Axis = -1): Tensor {
    return softmaxOp(this, axis);
  }

  /**
   * Log-softmax along an axis (functional form: `logSoftmax`).
   *
   * @param axis - Axis to normalise (default -1)
   *
   * @example
   * ```ts
   * tensor([[0, 0]]).logSoftmax(); // [[-0.693..., -0.693...]]
   * ```
   */
  logSoftmax(axis: Axis = -1): Tensor {
    return logSoftmaxOp(this, axis);
  }

  /**
   * Sum of elements over one axis, several axes, or all of them (functional form: `sum`).
   *
   * @param axis - Axis or axes to reduce (default: all)
   * @param keepdims - Keep reduced axes with size 1
   *
   * @example
   * ```ts
   * tensor([[1, 2], [3, 4]]).sum(0); // [4, 6]
   * ```
   */
  sum(axis?: Axis | readonly Axis[], keepdims = false): Tensor {
    return sumOp(this, axisArg(axis), keepdims);
  }

  /**
   * Mean of elements (functional form: `mean`).
   *
   * @param axis - Axis or axes to reduce (default: all)
   * @param keepdims - Keep reduced axes with size 1
   *
   * @example
   * ```ts
   * tensor([[1, 2], [3, 4]]).mean(1); // [1.5, 3.5]
   * ```
   */
  mean(axis?: Axis | readonly Axis[], keepdims = false): Tensor {
    return meanOp(this, axisArg(axis), keepdims);
  }

  /**
   * Maximum of elements (functional form: `max`).
   *
   * @param axis - Axis or axes to reduce (default: all)
   * @param keepdims - Keep reduced axes with size 1
   *
   * @example
   * ```ts
   * tensor([[1, 5], [3, 2]]).max(0); // [3, 5]
   * ```
   */
  max(axis?: Axis | readonly Axis[], keepdims = false): Tensor {
    return maxOp(this, axisArg(axis), keepdims);
  }

  /**
   * Minimum of elements (functional form: `min`).
   *
   * @param axis - Axis or axes to reduce (default: all)
   * @param keepdims - Keep reduced axes with size 1
   *
   * @example
   * ```ts
   * tensor([[1, 5], [3, 2]]).min(0); // [1, 2]
   * ```
   */
  min(axis?: Axis | readonly Axis[], keepdims = false): Tensor {
    return minOp(this, axisArg(axis), keepdims);
  }

  /**
   * Product of elements (functional form: `prod`).
   *
   * @param axis - Axis or axes to reduce (default: all)
   * @param keepdims - Keep reduced axes with size 1
   *
   * @example
   * ```ts
   * tensor([1, 2, 3, 4]).prod(); // 24
   * ```
   */
  prod(axis?: Axis | readonly Axis[], keepdims = false): Tensor {
    return prodOp(this, axisArg(axis), keepdims);
  }

  /**
   * Standard deviation (functional form: `std`).
   *
   * @param axis - Axis or axes to reduce (default: all)
   * @param keepdims - Keep reduced axes with size 1
   * @param ddof - Delta degrees of freedom (default 0, population)
   *
   * @example
   * ```ts
   * tensor([1, 2, 3, 4]).std(); // 1.118...
   * ```
   */
  std(axis?: Axis | readonly Axis[], keepdims = false, ddof = 0): Tensor {
    return stdOp(this, axisArg(axis), keepdims, ddof);
  }

  /**
   * Variance (functional form: `variance`).
   *
   * @param axis - Axis or axes to reduce (default: all)
   * @param keepdims - Keep reduced axes with size 1
   * @param ddof - Delta degrees of freedom (default 0, population)
   *
   * @example
   * ```ts
   * tensor([1, 2, 3, 4]).var(); // 1.25
   * ```
   */
  var(axis?: Axis | readonly Axis[], keepdims = false, ddof = 0): Tensor {
    return varianceOp(this, axisArg(axis), keepdims, ddof);
  }

  /**
   * Index of the maximum (functional form: `argmax`).
   *
   * @param axis - Axis to search (default: over the flattened tensor)
   * @param keepdims - Keep the reduced axis with size 1
   *
   * @example
   * ```ts
   * tensor([1, 9, 3]).argmax(); // 1
   * ```
   */
  argmax(axis?: Axis, keepdims = false): Tensor {
    return argmaxOp(this, axis, keepdims);
  }

  /**
   * Index of the minimum (functional form: `argmin`).
   *
   * @param axis - Axis to search (default: over the flattened tensor)
   * @param keepdims - Keep the reduced axis with size 1
   *
   * @example
   * ```ts
   * tensor([4, 1, 3]).argmin(); // 1
   * ```
   */
  argmin(axis?: Axis, keepdims = false): Tensor {
    return argminOp(this, axis, keepdims);
  }

  /**
   * Cumulative sum (functional form: `cumsum`).
   *
   * @param axis - Axis to accumulate along (default: flattened)
   *
   * @example
   * ```ts
   * tensor([1, 2, 3]).cumsum(); // [1, 3, 6]
   * ```
   */
  cumsum(axis?: Axis): Tensor {
    return cumsumOp(this, axis);
  }

  /**
   * Matrix product with another tensor (functional form: `matmul`).
   *
   * @param other - Right operand
   *
   * @example
   * ```ts
   * tensor([[1, 2], [3, 4]]).matmul(tensor([[1, 0], [0, 1]])); // [[1, 2], [3, 4]]
   * ```
   */
  matmul(other: Tensor): Tensor {
    return matmulOp(this, other);
  }

  /**
   * Dot product (functional form: `dot`).
   *
   * @param other - Right operand
   *
   * @example
   * ```ts
   * tensor([1, 2, 3]).dot(tensor([4, 5, 6])); // 32
   * ```
   */
  dot(other: Tensor): Tensor {
    return dotOp(this, other);
  }

  /**
   * Permute the axes (functional form: `transpose`). Returns a view.
   *
   * @param axes - New axis order (default: reverse the axes)
   *
   * @example
   * ```ts
   * tensor([[1, 2, 3]]).transpose().shape; // [3, 1]
   * ```
   */
  transpose(axes?: readonly number[]): Tensor {
    return transposeOp(this, axes);
  }

  /**
   * Axes reversed, as `transpose()` with no argument (like `numpy.ndarray.T`).
   *
   * @example
   * ```ts
   * tensor([[1, 2, 3]]).T.shape; // [3, 1]
   * ```
   */
  get T(): Tensor {
    return transposeOp(this);
  }

  /**
   * Remove axes of size 1 (functional form: `squeeze`). Returns a view.
   *
   * @param axis - Axis or axes to remove (default: all size-1 axes)
   *
   * @example
   * ```ts
   * tensor([[1, 2]]).squeeze().shape; // [2]
   * ```
   */
  squeeze(axis?: Axis | readonly Axis[]): Tensor {
    return squeezeOp(this, axis);
  }

  /**
   * Insert an axis of size 1 (functional form: `unsqueeze`). Returns a view.
   *
   * @param axis - Position of the new axis; negative values count from the end
   *
   * @example
   * ```ts
   * tensor([1, 2]).unsqueeze(0).shape; // [1, 2]
   * ```
   */
  unsqueeze(axis: number): Tensor {
    return unsqueezeOp(this, axis);
  }

  /**
   * Select entries along an axis with a 1-D index tensor (functional form: `gather`).
   * Equivalent to `torch.index_select(t, axis, indices)`.
   *
   * @param indices - 1-D integer indices
   * @param axis - Axis to select along
   *
   * @example
   * ```ts
   * tensor([[1, 2], [3, 4], [5, 6]]).gather(tensor([0, 2]), 0); // [[1, 2], [5, 6]]
   * ```
   */
  gather(indices: Tensor, axis: Axis): Tensor {
    return gatherOp(this, indices, axis);
  }

  /**
   * Return the tensor itself. A plain `Tensor` never tracks gradients, so this exists for
   * parity with `GradTensor.detach()` (like `torch.Tensor.detach`).
   *
   * @example
   * ```ts
   * const x = tensor([1, 2]);
   * x.detach() === x; // true
   * ```
   */
  detach(): this {
    return this;
  }

  /**
   * Always `false`: a plain `Tensor` does not track gradients. Present so code that
   * receives a `Tensor | GradTensor` (for example `model.forward(x)`) can read it
   * without narrowing, as in PyTorch.
   */
  get requiresGrad(): false {
    return false;
  }

  /** Always `null`: a plain `Tensor` never holds a gradient. See {@link requiresGrad}. */
  get grad(): null {
    return null;
  }

  /**
   * Throws, because a plain `Tensor` is not part of a computation graph.
   *
   * It exists so `loss.backward()` type-checks when `loss` is a `Tensor | GradTensor`.
   * A loss computed inside `noGrad()`, or from a model with no trainable parameters,
   * is a plain `Tensor`, and calling this tells you so.
   *
   * @throws {DeepboxError} Always
   */
  backward(_grad?: Tensor): void {
    throw new DeepboxError(
      "backward() was called on a tensor that does not track gradients. " +
        "Compute the loss outside noGrad() from a model with trainable parameters, " +
        "or wrap inputs with parameter() to track them."
    );
  }

  /**
   * Copy of the tensor with its own contiguous storage (functional form: `clone`).
   *
   * @example
   * ```ts
   * const b = tensor([1, 2]).clone();
   * ```
   */
  clone(): Tensor {
    return cloneOp(this);
  }

  /**
   * Set every element to one value, in place, and return the tensor (like
   * `torch.Tensor.fill_`). The value is converted to the tensor dtype the way
   * a typed array store does (`bool` stores 0 or 1). Works on strided views.
   *
   * @param value - Number, boolean or bigint for numeric dtypes; string for `string` dtype
   * @returns This tensor
   * @throws {DeviceError} For tensors in device memory
   * @throws {DTypeError} If the value type does not fit the dtype
   *
   * @example
   * ```ts
   * const t = tensor([1, 2, 3]);
   * t.fill(7); // t is now [7, 7, 7]
   * ```
   */
  fill(value: number | bigint | boolean | string): this {
    this.assertAlive();
    if (this.bufferOwner) {
      throw new DeviceError(
        `fill is not supported on device tensors; call await t.cpu() first (device: ${this.device})`
      );
    }
    const data = this.data;
    const offsets = isDenseLayout(this.shape, this.strides)
      ? null
      : stridedOffsets(this.shape, this.strides, this.offset);
    const size = this.size;
    const base = this.offset;

    if (Array.isArray(data)) {
      if (typeof value !== "string") {
        throw new DTypeError(
          `fill: a string tensor needs a string value; received ${typeof value}`
        );
      }
      for (let i = 0; i < size; i++)
        data[offsets === null ? base + i : (offsets[i] as number)] = value;
      return this;
    }
    if (typeof value === "string") {
      throw new DTypeError(`fill: a ${this.dtype} tensor needs a numeric value; received a string`);
    }

    if (data instanceof BigInt64Array) {
      const v =
        typeof value === "bigint"
          ? BigInt.asIntN(64, value)
          : toInt64(typeof value === "boolean" ? Number(value) : value, "fill");
      for (let i = 0; i < size; i++) data[offsets === null ? base + i : (offsets[i] as number)] = v;
      return this;
    }

    let num = typeof value === "number" ? value : Number(value);
    if (this.dtype === "bool") num = num !== 0 ? 1 : 0;
    else if (this.dtype === "float16") num = roundToFloat16(num);
    else if (this.dtype === "bfloat16") num = roundToBFloat16(num);
    for (let i = 0; i < size; i++) data[offsets === null ? base + i : (offsets[i] as number)] = num;
    return this;
  }

  /**
   * Limit values to `[min, max]` (functional form: `clip`). Either bound may be omitted.
   *
   * @param min - Lower bound
   * @param max - Upper bound
   *
   * @example
   * ```ts
   * tensor([-2, 0.5, 3]).clip(0, 1); // [0, 0.5, 1]
   * ```
   */
  clip(min?: number, max?: number): Tensor {
    return clipOp(this, min, max);
  }

  /**
   * Element-wise equality, giving a `bool` tensor (functional form: `equal`).
   *
   * @param other - Tensor (broadcast) or number
   *
   * @example
   * ```ts
   * tensor([1, 2, 3]).eq(2); // [false, true, false]
   * ```
   */
  eq(other: number | Tensor): Tensor {
    return equalOp(this, scalarOperand(this, other, "eq"));
  }

  /**
   * Element-wise inequality (functional form: `notEqual`).
   *
   * @param other - Tensor (broadcast) or number
   *
   * @example
   * ```ts
   * tensor([1, 2, 3]).ne(2); // [true, false, true]
   * ```
   */
  ne(other: number | Tensor): Tensor {
    return notEqualOp(this, scalarOperand(this, other, "ne"));
  }

  /**
   * Element-wise greater-than (functional form: `greater`).
   *
   * @param other - Tensor (broadcast) or number
   *
   * @example
   * ```ts
   * tensor([1, 2, 3]).gt(2); // [false, false, true]
   * ```
   */
  gt(other: number | Tensor): Tensor {
    return greaterOp(this, scalarOperand(this, other, "gt"));
  }

  /**
   * Element-wise greater-or-equal (functional form: `greaterEqual`).
   *
   * @param other - Tensor (broadcast) or number
   *
   * @example
   * ```ts
   * tensor([1, 2, 3]).ge(2); // [false, true, true]
   * ```
   */
  ge(other: number | Tensor): Tensor {
    return greaterEqualOp(this, scalarOperand(this, other, "ge"));
  }

  /**
   * Element-wise less-than (functional form: `less`).
   *
   * @param other - Tensor (broadcast) or number
   *
   * @example
   * ```ts
   * tensor([1, 2, 3]).lt(2); // [true, false, false]
   * ```
   */
  lt(other: number | Tensor): Tensor {
    return lessOp(this, scalarOperand(this, other, "lt"));
  }

  /**
   * Element-wise less-or-equal (functional form: `lessEqual`).
   *
   * @param other - Tensor (broadcast) or number
   *
   * @example
   * ```ts
   * tensor([1, 2, 3]).le(2); // [true, true, false]
   * ```
   */
  le(other: number | Tensor): Tensor {
    return lessEqualOp(this, scalarOperand(this, other, "le"));
  }

  /**
   * Element-wise NaN test, giving a `bool` tensor (functional form: `isnan`).
   *
   * @example
   * ```ts
   * tensor([1, NaN]).isnan(); // [false, true]
   * ```
   */
  isnan(): Tensor {
    return isnanOp(this);
  }

  /**
   * True where any element is non-zero (functional form: `any`).
   *
   * @param axis - Axis or axes to reduce (default: all)
   * @param keepdims - Keep reduced axes with size 1
   *
   * @example
   * ```ts
   * tensor([0, 1]).any(); // true
   * ```
   */
  any(axis?: Axis | readonly Axis[], keepdims = false): Tensor {
    return anyOp(this, axisArg(axis), keepdims);
  }

  /**
   * True where all elements are non-zero (functional form: `all`).
   *
   * @param axis - Axis or axes to reduce (default: all)
   * @param keepdims - Keep reduced axes with size 1
   *
   * @example
   * ```ts
   * tensor([1, 0]).all(); // false
   * ```
   */
  all(axis?: Axis | readonly Axis[], keepdims = false): Tensor {
    return allOp(this, axisArg(axis), keepdims);
  }

  /**
   * Round half to even (functional form: `round`).
   *
   * @param decimals - Digits after the decimal point (negative rounds to tens, hundreds, ...)
   *
   * @example
   * ```ts
   * tensor([0.5, 1.5, 2.5]).round(); // [0, 2, 2]
   * ```
   */
  round(decimals = 0): Tensor {
    return roundOp(this, decimals);
  }

  /**
   * Round down (functional form: `floor`).
   *
   * @example
   * ```ts
   * tensor([1.7, -1.2]).floor(); // [1, -2]
   * ```
   */
  floor(): Tensor {
    return floorOp(this);
  }

  /**
   * Round up (functional form: `ceil`).
   *
   * @example
   * ```ts
   * tensor([1.2, -1.7]).ceil(); // [2, -1]
   * ```
   */
  ceil(): Tensor {
    return ceilOp(this);
  }

  /**
   * Sorted copy along an axis (functional form: `sort`).
   *
   * @param axis - Axis to sort along (default -1)
   * @param descending - Sort from largest to smallest
   *
   * @example
   * ```ts
   * tensor([3, 1, 2]).sort(); // [1, 2, 3]
   * ```
   */
  sort(axis: Axis | undefined = -1, descending = false): Tensor {
    return sortOp(this, axis, descending);
  }

  /**
   * Indices that would sort the tensor along an axis (functional form: `argsort`).
   *
   * @param axis - Axis to sort along (default -1)
   * @param descending - Sort from largest to smallest
   *
   * @example
   * ```ts
   * tensor([3, 1, 2]).argsort(); // [1, 2, 0]
   * ```
   */
  argsort(axis: Axis | undefined = -1, descending = false): Tensor {
    return argsortOp(this, axis, descending);
  }

  /**
   * Reverse the order of elements along axes (functional form: `flip`).
   *
   * @param axes - Axis or axes to flip (default: all)
   *
   * @example
   * ```ts
   * tensor([1, 2, 3]).flip(); // [3, 2, 1]
   * ```
   */
  flip(axes?: Axis | readonly Axis[]): Tensor {
    return flipOp(this, axes);
  }

  /**
   * Move the tensor to another device.
   *
   * - `cpu` → kernel device (e.g. `webgpu`): uploads the data to device
   *   memory. Requires a `float32`, `float16` or `bfloat16` tensor (convert
   *   with `astype` first).
   * - kernel device → `cpu`: downloads the data. This waits for all pending
   *   device work, which is why `to` is asynchronous (WebGPU readback cannot
   *   block the JavaScript thread, which is the honest equivalent of PyTorch's
   *   synchronizing `.cpu()`).
   * - `cpu` ↔ host-accelerator device (`wasm`): zero-copy relabel; both
   *   devices share host memory.
   *
   * Returns `this` when the tensor is already on the target device.
   *
   * @param device - Target device
   * @returns Promise resolving to a tensor on the target device
   * @throws {DeviceError} If the target device has no available backend
   * @throws {DTypeError} If a tensor of another dtype is moved to a kernel device
   *
   * @example
   * ```ts
   * const g = await tensor([1, 2, 3]).to('webgpu');
   * const back = await g.cpu();
   * ```
   */
  async to(device: Device): Promise<Tensor<S, DType>> {
    this.assertAlive();
    ensureBackendAvailable(device, "tensor device");
    if (device === this.device) return this;

    if (this.bufferOwner) {
      const host = await this.bufferOwner.backend.download(this.bufferOwner.buffer);
      // `download` returns float32 values (f16/bf16 already unpacked/rounded);
      // reconstruct with the tensor's own dtype so half-precision round-trips.
      const cpuTensor = Tensor.fromTypedArray({
        data: host,
        shape: this.shape,
        dtype: this.dtype as Exclude<DType, "string">,
        device: "cpu",
        offset: this.offset,
        strides: this.strides,
      }) as Tensor<S, DType>;
      if (device === "cpu") return cpuTensor;
      return cpuTensor.to(device);
    }

    if (this.isStringTensor()) {
      if (getKernelBackend(device)) {
        throw new DTypeError(
          `Device "${device}" supports float32, float16 and bfloat16 tensors only; received string`
        );
      }
      return Tensor.fromStringArray({
        data: this.data as string[],
        shape: this.shape,
        device,
        offset: this.offset,
        strides: this.strides,
      }) as Tensor<S, DType>;
    }

    return Tensor.fromTypedArray({
      data: this.data as TypedArray,
      shape: this.shape,
      dtype: this.dtype as Exclude<DType, "string">,
      device,
      offset: this.offset,
      strides: this.strides,
    }) as Tensor<S, DType>;
  }

  /**
   * Move the tensor to the CPU. Shorthand for `to('cpu')`.
   *
   * @returns Promise resolving to a CPU tensor
   */
  cpu(): Promise<Tensor<S, DType>> {
    return this.to("cpu");
  }

  /**
   * Slice this tensor along one or more axes.
   *
   * A number selects a single index and drops that axis; `{ start, end, step }`
   * keeps the axis (negative `start`/`end` count from the end, a negative
   * `step` walks backwards). Axes without a range are kept whole. Host
   * tensors return a copy, so the result is always contiguous; device tensors
   * return a zero-copy strided view. To get a non-contiguous host tensor (for
   * example to test strided code paths) use `transpose` or
   * `Tensor.fromTypedArray({ strides })` instead of `slice`.
   *
   * @param ranges - Per-axis slice specifications (number, or {start?, end?, step?})
   * @returns New tensor with the sliced data
   * @throws {ShapeError} If more ranges than dimensions are given
   * @throws {IndexError} If a single-index range is out of bounds
   * @throws {InvalidParameterError} If an index, bound or step is not an integer
   *
   * @example
   * ```ts
   * const t = tensor([[1, 2, 3], [4, 5, 6]]);
   * t.slice(0);           // tensor([1, 2, 3])
   * t.slice({ start: 0, end: 1 }, { start: 1 }); // tensor([[2, 3]])
   * ```
   */
  slice(...ranges: SliceRange[]): Tensor {
    this.assertAlive();
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

    if (this.bufferOwner) {
      // Device tensors slice as zero-copy strided views.
      let viewOffset = this.offset;
      const viewStrides: number[] = [];
      for (let axis = 0; axis < ndim; axis++) {
        const nr = normalized[axis];
        if (nr === undefined) {
          throw new DeepboxError("Internal error: missing normalized slice range");
        }
        if (nr.step < 0) {
          throw new DeviceError(
            `Negative-step slicing is not supported on device "${this.device}". ` +
              "Move the tensor to the CPU first with `await t.cpu()`."
          );
        }
        viewOffset += nr.start * (this.strides[axis] ?? 0);
        if (typeof ranges[axis] !== "number") {
          viewStrides.push((this.strides[axis] ?? 0) * nr.step);
        }
      }
      // An empty slice reads nothing; keep its offset inside the buffer.
      if (outShape.includes(0)) viewOffset = this.offset;
      return Tensor.fromDeviceBuffer({
        owner: this.bufferOwner,
        shape: outShape.length === 0 ? [] : outShape,
        device: this.device,
        offset: viewOffset,
        strides: viewStrides,
      });
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
    const data = this.data;
    const strides = this.strides;
    // Precompute per-output-axis input strides and the fixed base offset so
    // the copy loop is pure integer arithmetic (no per-element type checks).
    let base = this.offset;
    const axisStep: number[] = [];
    let outAxisInit = 0;
    for (let axis = 0; axis < ndim; axis++) {
      const nr = normalized[axis];
      if (nr === undefined) {
        throw new DeepboxError("Internal error: missing normalized slice range");
      }
      base += nr.start * (strides[axis] ?? 0);
      if (typeof ranges[axis] !== "number") {
        axisStep[outAxisInit++] = nr.step * (strides[axis] ?? 0);
      }
    }

    const outIdx = new Array<number>(outNdim).fill(0);
    const stringData = Array.isArray(data) ? data : null;
    const bigData = data instanceof BigInt64Array ? data : null;
    const numData = !stringData && !bigData ? (data as Exclude<TypedArray, BigInt64Array>) : null;

    // Fast path: unit step on the innermost axis: copy whole rows with
    // subarray/set (memcpy) instead of one element per odometer tick.
    if (
      numData &&
      !Array.isArray(out) &&
      !(out instanceof BigInt64Array) &&
      outNdim > 0 &&
      axisStep[outNdim - 1] === 1 &&
      (outShape[outNdim - 1] ?? 0) > 0 &&
      outSize > 0
    ) {
      const innerLen = outShape[outNdim - 1] as number;
      const outerSize = outSize / innerLen;
      const idx = new Array<number>(outNdim - 1).fill(0);
      let inBase = base;
      let outPos = 0;
      for (let b = 0; b < outerSize; b++) {
        out.set(numData.subarray(inBase, inBase + innerLen), outPos);
        outPos += innerLen;
        for (let d = outNdim - 2; d >= 0; d--) {
          idx[d] = (idx[d] ?? 0) + 1;
          inBase += axisStep[d] ?? 0;
          if ((idx[d] ?? 0) < (outShape[d] ?? 0)) break;
          inBase -= (axisStep[d] ?? 0) * (outShape[d] ?? 0);
          idx[d] = 0;
        }
      }
      return Tensor.fromTypedArray({
        data: out as TypedArray,
        shape: outShape.length === 0 ? [] : outShape,
        dtype: this.dtype as Exclude<DType, "string">,
        device: this.device,
      });
    }

    let inFlat = base;
    for (let outFlat = 0; outFlat < outSize; outFlat++) {
      if (stringData && Array.isArray(out)) {
        out[outFlat] = stringData[inFlat] ?? "";
      } else if (bigData && out instanceof BigInt64Array) {
        out[outFlat] = getBigIntElement(bigData, inFlat);
      } else if (numData && !Array.isArray(out) && !(out instanceof BigInt64Array)) {
        out[outFlat] = numData[inFlat] ?? 0;
      }
      // Odometer increment (last axis fastest).
      for (let d = outNdim - 1; d >= 0; d--) {
        outIdx[d] = (outIdx[d] ?? 0) + 1;
        inFlat += axisStep[d] ?? 0;
        if ((outIdx[d] ?? 0) < (outShape[d] ?? 0)) break;
        inFlat -= (axisStep[d] ?? 0) * (outShape[d] ?? 0);
        outIdx[d] = 0;
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

  /**
   * Read a single element. Provide one index per dimension; negative indices
   * count from the end of that dimension.
   *
   * @throws {ShapeError} If the number of indices differs from `ndim`
   * @throws {IndexError} If an index is out of bounds
   * @throws {InvalidParameterError} If an index is not an integer
   *
   * @example
   * ```ts
   * const t = tensor([[1, 2], [3, 4]]);
   * t.at(1, 0);   // 3
   * t.at(-1, -1); // 4
   * ```
   */
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

  /**
   * Return a new tensor with the data converted to the given dtype.
   *
   * Mirrors NumPy's `ndarray.astype` (and `GradTensor.astype`):
   * - numeric -> numeric: value-preserving cast (float -> int truncates via
   *   the typed-array store; -> bool maps every nonzero value, NaN included,
   *   to 1)
   * - numeric -> float16 / bfloat16 rounds to the nearest representable
   *   half-precision value (ties to even, overflow to Infinity)
   * - numeric -> int64 truncates toward zero and throws on NaN/Infinity
   * - string -> numeric parses with `Number()` (unparseable -> NaN); string ->
   *   int64 parses integer literals exactly and throws on anything else
   * - numeric -> string uses the shortest text that round-trips the value
   *   within its dtype (a float32 `0.1` becomes `"0.1"`)
   *
   * Returns `this` unchanged when the dtype already matches. Tensors that
   * live in device memory must be moved to the CPU first (`await t.cpu()`).
   *
   * @throws {DeviceError} If the tensor lives in device memory
   * @throws {DTypeError} If the target is a complex dtype, or a value cannot
   *   be represented (non-finite -> int64)
   *
   * @example
   * ```ts
   * const t = tensor([1.7, -2.3]);          // float32
   * t.astype("int32").toArray();            // [1, -2]
   * t.astype("float64").dtype;              // "float64"
   * ```
   */
  astype(dtype: DType): Tensor {
    this.assertAlive();
    if (this.dtype === dtype) return this;
    if (this.bufferOwner) {
      throw new DeviceError(
        `astype is not supported on device tensors; call await t.cpu() first (device: ${this.device})`
      );
    }
    if (dtype === "complex64" || dtype === "complex128") {
      throw new DTypeError(
        `astype: tensors do not support dtype ${dtype} yet; ` +
          "use Complex64Array / Complex128Array for complex data."
      );
    }

    const contiguous = isDenseLayout(this.shape, this.strides);
    const src = this.data as TypedArray | string[];
    const offsets = contiguous ? null : stridedOffsets(this.shape, this.strides, this.offset);
    const readOffset = (i: number): number =>
      offsets === null ? this.offset + i : (offsets[i] as number);

    if (dtype === "string") {
      const out = new Array<string>(this.size);
      if (src instanceof BigInt64Array) {
        for (let i = 0; i < this.size; i++)
          out[i] = getBigIntElement(src, readOffset(i)).toString();
      } else if (
        this.dtype === "float32" ||
        this.dtype === "float16" ||
        this.dtype === "bfloat16"
      ) {
        const round =
          this.dtype === "float32"
            ? Math.fround
            : this.dtype === "float16"
              ? roundToFloat16
              : roundToBFloat16;
        for (let i = 0; i < this.size; i++) {
          out[i] = shortestRoundTripString((src as Float32Array)[readOffset(i)] as number, round);
        }
      } else {
        for (let i = 0; i < this.size; i++) {
          const v = (src as Exclude<TypedArray, BigInt64Array> | string[])[readOffset(i)];
          out[i] = String(v);
        }
      }
      return Tensor.fromStringArray({ data: out, shape: this.shape, device: this.device });
    }

    if (dtype === "float16" || dtype === "bfloat16") {
      // Round straight from the source value: going through a float32 store
      // first would round twice (e.g. 65519.999 -> 65520 -> Infinity).
      const round = dtype === "float16" ? roundToFloat16 : roundToBFloat16;
      const half = new Float32Array(this.size);
      const values = src as ArrayLike<number | bigint | string>;
      for (let i = 0; i < this.size; i++) {
        half[i] = round(Number(values[readOffset(i)]));
      }
      return Tensor.fromTypedArray({ data: half, shape: this.shape, dtype, device: this.device });
    }

    const Ctor = dtypeToTypedArrayCtor(dtype);
    const out = new Ctor(this.size);
    const toBool = dtype === "bool";

    if (out instanceof BigInt64Array) {
      if (src instanceof BigInt64Array) {
        for (let i = 0; i < this.size; i++) out[i] = getBigIntElement(src, readOffset(i));
      } else if (Array.isArray(src)) {
        for (let i = 0; i < this.size; i++) {
          const text = src[readOffset(i)] ?? "";
          if (/^\s*[+-]?\d+\s*$/.test(text)) {
            const exact = BigInt(text.trim());
            if (exact !== BigInt.asIntN(64, exact)) {
              throw new DTypeError(
                `astype: string ${JSON.stringify(text)} is outside the int64 range`
              );
            }
            out[i] = exact;
            continue;
          }
          const v = Number(text);
          if (!Number.isFinite(v)) {
            throw new DTypeError(`astype: cannot convert string ${JSON.stringify(text)} to int64`);
          }
          out[i] = toInt64(v, "astype");
        }
      } else {
        for (let i = 0; i < this.size; i++) {
          out[i] = toInt64(src[readOffset(i)] as number, "astype");
        }
      }
    } else if (src instanceof BigInt64Array) {
      for (let i = 0; i < this.size; i++) {
        const v = getBigIntElement(src, readOffset(i));
        out[i] = toBool ? (v !== 0n ? 1 : 0) : Number(v);
      }
    } else if (Array.isArray(src)) {
      for (let i = 0; i < this.size; i++) {
        const v = Number(src[readOffset(i)]);
        out[i] = toBool ? (v !== 0 && !Number.isNaN(v) ? 1 : 0) : v;
      }
    } else if (contiguous && !toBool) {
      // Native element-wise conversion (same rules as per-element assignment).
      out.set(src.subarray(this.offset, this.offset + this.size));
    } else {
      for (let i = 0; i < this.size; i++) {
        const v = src[readOffset(i)] as number;
        out[i] = toBool ? (v !== 0 ? 1 : 0) : v;
      }
    }

    return Tensor.fromTypedArray({
      data: out,
      shape: this.shape,
      dtype,
      device: this.device,
    });
  }

  /**
   * Extract the value of a single-element tensor as a JS scalar.
   *
   * Mirrors NumPy's `ndarray.item()` / PyTorch's `Tensor.item()`: works for
   * 0-D tensors and any shape with exactly one element.
   *
   * @throws {ShapeError} When the tensor has more than one element.
   *
   * @example
   * ```ts
   * sum(tensor([1, 2, 3])).item(); // 6
   * ```
   */
  item(): ElementOf<D> {
    if (this.size !== 1) {
      throw new ShapeError(`item() requires a single-element tensor; got ${this.size} elements`);
    }
    const data = this.data;
    const v = data[this.offset];
    if (v === undefined) {
      throw new DeepboxError("Internal error: item() offset out of bounds");
    }
    return v as ElementOf<D>;
  }

  /**
   * Convert to nested JavaScript arrays (a bare scalar for 0-D tensors).
   * Elements keep their storage type: `number`, `bigint` for int64, `string`
   * for string tensors. Views are read in logical order.
   *
   * @example
   * ```ts
   * tensor([[1, 2], [3, 4]]).toArray(); // [[1, 2], [3, 4]]
   * ```
   */
  toArray(): unknown {
    const data = this.data;
    const shape = this.shape;
    const strides = this.strides;
    const ndim = this.ndim;
    const recur = (axis: number, baseOffset: number): unknown => {
      const dim = shape[axis] ?? 0;
      const stride = strides[axis] ?? 0;
      const out = new Array<unknown>(dim);
      if (axis === ndim - 1) {
        for (let i = 0; i < dim; i++) {
          const v = data[baseOffset + i * stride];
          if (v === undefined) {
            throw new DeepboxError("Internal error: computed flat index is out of bounds");
          }
          out[i] = v;
        }
        return out;
      }
      for (let i = 0; i < dim; i++) {
        out[i] = recur(axis + 1, baseOffset + i * stride);
      }
      return out;
    };

    if (ndim === 0) {
      const v = data[this.offset];
      if (v === undefined) {
        throw new DeepboxError("Internal error: computed flat index is out of bounds");
      }
      return v;
    }
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

    // Device tensors cannot be read synchronously; show metadata instead.
    if (this.bufferOwner) {
      return `tensor(<${this.device}>, shape=[${this.shape.join(", ")}], dtype=${this.dtype})`;
    }

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
