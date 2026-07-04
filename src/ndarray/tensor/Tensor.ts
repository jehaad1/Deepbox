import type { Device, DType, ElementOf, Shape, TensorLike, TypedArray } from "../../core";
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
import { normalizeRange, type SliceRange } from "./slice_helpers";
import { isContiguous, offsetFromFlatIndex } from "./strides";

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
 * Reference-counted owner of a {@link DeviceBuffer}.
 *
 * Multiple tensors (views) can share one device allocation; the buffer is
 * released on the owning backend once every holder has released it —
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
 * the `shader-f16` feature — the backend throws a clear DeviceError at upload
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
 *   full reductions) executes on the GPU — move data with `await t.to(device)`
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
   * read synchronously — transfer first with `await t.cpu()`.
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
    return this.bufferOwner?.buffer ?? null;
  }

  /**
   * Shared buffer owner for view creation and dispatch.
   *
   * @internal
   */
  get __bufferOwner(): DeviceBufferOwner | null {
    return this.bufferOwner;
  }

  /**
   * Release this tensor's device memory reference immediately.
   *
   * Device buffers are also released automatically when tensors are garbage
   * collected, but explicit disposal is deterministic and recommended for
   * large buffers. Host (CPU) tensors ignore this call. Views share the
   * underlying allocation; it is freed when the last holder releases it.
   * Using a tensor after disposing it throws.
   */
  dispose(): void {
    if (this.disposed || !this.bufferOwner) return;
    this.disposed = true;
    deviceBufferFinalizer.unregister(this);
    this.bufferOwner.release();
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
   * Move the tensor to another device.
   *
   * - `cpu` → kernel device (e.g. `webgpu`): uploads the data to device
   *   memory. Requires `float32` dtype (convert with `astype` first).
   * - kernel device → `cpu`: downloads the data. This waits for all pending
   *   device work, which is why `to` is asynchronous (WebGPU readback cannot
   *   block the JavaScript thread — the honest equivalent of PyTorch's
   *   synchronizing `.cpu()`).
   * - `cpu` ↔ host-accelerator device (`wasm`): zero-copy relabel; both
   *   devices share host memory.
   *
   * Returns `this` when the tensor is already on the target device.
   *
   * @param device - Target device
   * @returns Promise resolving to a tensor on the target device
   * @throws {DeviceError} If the target device has no available backend
   * @throws {DTypeError} If a non-float32 tensor is moved to a kernel device
   *
   * @example
   * ```ts
   * const g = await tensor([1, 2, 3]).to('webgpu');
   * const back = await g.cpu();
   * ```
   */
  async to(device: Device): Promise<Tensor<S, DType>> {
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
        throw new DTypeError(`Device "${device}" supports float32 tensors only; received string`);
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

    // Fast path: unit step on the innermost axis — copy whole rows with
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
   *   the typed-array store; -> bool maps nonzero to 1)
   * - numeric -> int64 truncates toward zero
   * - string -> numeric parses with `Number()` (unparseable -> NaN)
   * - numeric -> string stringifies each element
   *
   * Returns `this` unchanged when the dtype already matches. Device tensors
   * must be moved to the CPU first (`await t.cpu()`).
   *
   * @example
   * ```ts
   * const t = tensor([1.7, -2.3]);          // float32
   * t.astype("int32").toArray();            // [1, -2]
   * t.astype("float64").dtype;              // "float64"
   * ```
   */
  astype(dtype: DType): Tensor {
    if (this.dtype === dtype) return this;
    if (this.device !== "cpu") {
      throw new DeviceError(
        `astype is not supported on device tensors; call await t.cpu() first (device: ${this.device})`
      );
    }

    const logicalStrides = computeStrides(this.shape);
    const contiguous = isContiguous(this.shape, this.strides);
    const src = this.data as TypedArray | string[];
    const readOffset = (i: number): number =>
      contiguous
        ? this.offset + i
        : offsetFromFlatIndex(i, logicalStrides, this.strides, this.offset);

    if (dtype === "string") {
      const out = new Array<string>(this.size);
      if (src instanceof BigInt64Array) {
        for (let i = 0; i < this.size; i++)
          out[i] = getBigIntElement(src, readOffset(i)).toString();
      } else {
        for (let i = 0; i < this.size; i++) {
          const v = (src as Exclude<TypedArray, BigInt64Array> | string[])[readOffset(i)];
          out[i] = String(v);
        }
      }
      return Tensor.fromStringArray({ data: out, shape: this.shape });
    }

    const Ctor = dtypeToTypedArrayCtor(dtype);
    const out = new Ctor(this.size);
    const toBool = dtype === "bool";

    if (out instanceof BigInt64Array) {
      if (src instanceof BigInt64Array) {
        for (let i = 0; i < this.size; i++) out[i] = getBigIntElement(src, readOffset(i));
      } else if (Array.isArray(src)) {
        for (let i = 0; i < this.size; i++) out[i] = BigInt(Math.trunc(Number(src[readOffset(i)])));
      } else {
        for (let i = 0; i < this.size; i++) {
          const v = src[readOffset(i)];
          if (v === undefined || !Number.isFinite(v)) {
            throw new DTypeError(`astype: cannot convert non-finite value ${v} to int64`);
          }
          out[i] = BigInt(Math.trunc(v));
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
    } else {
      for (let i = 0; i < this.size; i++) {
        const v = src[readOffset(i)] as number;
        out[i] = toBool ? (v !== 0 ? 1 : 0) : v;
      }
    }

    return Tensor.fromTypedArray({ data: out, shape: this.shape, dtype, device: "cpu" });
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
