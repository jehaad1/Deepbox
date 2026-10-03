/**
 * Device kernel contract: the execution interface that accelerated backends implement.
 *
 * A {@link KernelBackend} owns device memory ({@link DeviceBuffer}) and executes
 * a fixed set of tensor kernels on it (float32, plus float16/bfloat16 buffers
 * where the backend supports them). The ndarray dispatch layer routes
 * eligible ops on non-CPU tensors here; everything it cannot express throws a
 * `DeviceError` instead of silently computing on the wrong device.
 *
 * Kernels are layout-aware: every operand arrives with an explicit
 * {@link KernelLayout} (shape / element strides / element offset), so views,
 * transposes, slices and broadcasts execute on-device without materialization.
 * Broadcasting is expressed by the caller as stride-0 dimensions.
 *
 * @module core/backend
 * @see {@link https://deepbox.dev/docs/devices-and-execution | Devices & execution}
 */

import type { Device } from "../types/device";
import type { Backend } from "./Backend";

/**
 * Element type of a {@link DeviceBuffer}.
 *
 * `float32` is the default (an absent/`undefined` `dtype` on a buffer means
 * float32, so all existing code keeps working). `float16` is true on-device
 * IEEE-754 half precision (2 bytes/element, requires the WebGPU `shader-f16`
 * feature). `bfloat16` gives correct bfloat16 numerics to the user via
 * rounding at upload/download while computing on-device in float32, so the
 * WebGPU backend stores it in 4 bytes/element
 * (see {@link https://deepbox.dev/docs/devices-and-execution | Devices & execution}).
 */
export type DeviceDType = "float32" | "float16" | "bfloat16";

/**
 * Opaque handle to memory owned by a {@link KernelBackend}.
 *
 * The handle is created by the backend (`upload`, `fill`, or as a kernel
 * output) and must be released with `free` when no longer referenced.
 * Tensors wrap these handles and free them automatically via a
 * finalization registry, but deterministic release via `Tensor.dispose()`
 * is recommended for large buffers.
 */
export type DeviceBuffer = {
  /** Device that owns this buffer. */
  readonly device: Device;
  /**
   * Size of the buffer's payload in bytes. This is the on-device storage
   * size: 2 bytes per element for `float16`, 4 for `float32`, and 4 for
   * `bfloat16` on backends that store it as float32 (the WebGPU backend).
   */
  readonly byteLength: number;
  /** Number of elements in the buffer. */
  readonly size: number;
  /**
   * Element type of the buffer. Absent/`undefined` means `float32` (so
   * pre-existing float32 buffers are unaffected).
   */
  readonly dtype?: DeviceDType;
};

/**
 * Explicit memory layout of one kernel operand.
 *
 * Strides and offset are measured in elements (not bytes). A dimension
 * broadcast to the output shape is expressed with stride 0.
 */
export type KernelLayout = {
  readonly shape: readonly number[];
  readonly strides: readonly number[];
  readonly offset: number;
};

/** Binary element-wise kernels a backend may execute. */
export type BinaryKernelOp = "add" | "sub" | "mul" | "div" | "pow" | "maximum" | "minimum";

/**
 * Unary element-wise kernels a backend may execute. `copy` materializes a
 * strided view; `step` is the Heaviside step (1 where x > 0, else 0), used
 * by autograd for the relu backward mask.
 */
export type UnaryKernelOp =
  | "copy"
  | "step"
  | "neg"
  | "abs"
  | "exp"
  | "log"
  | "sqrt"
  | "square"
  | "relu"
  | "sigmoid"
  | "tanh"
  | "gelu"
  | "erf"
  | "rsqrt"
  | "reciprocal"
  | "sign"
  | "expm1"
  | "log1p"
  | "softplus";

/** Full-tensor reduction kernels a backend may execute. */
export type ReduceKernelOp = "sum" | "mean" | "max" | "min";

/**
 * Ternary element-wise kernels a backend may execute. `where` selects `a`
 * where the condition (first operand) is non-zero, else `b`.
 */
export type TernaryKernelOp = "where";

/**
 * Geometry of a 2-D convolution unfold/fold (`im2col`/`col2im`), all in
 * elements. The input is `[batch, channels, height, width]`; the unfolded
 * columns are `[batch, outH*outW, channels*kH*kW]`, laid out for a
 * `[channels*kH*kW, outChannels]` weight matmul.
 */
export type Im2ColParams = {
  readonly batch: number;
  readonly channels: number;
  readonly height: number;
  readonly width: number;
  readonly outH: number;
  readonly outW: number;
  readonly kH: number;
  readonly kW: number;
  readonly strideH: number;
  readonly strideW: number;
  readonly padH: number;
  readonly padW: number;
};

/** 2-D pooling kernels a backend may execute. */
export type PoolKernelOp = "max" | "avg";

/**
 * Execution interface for accelerated (non-CPU) backends.
 *
 * Implementations execute kernels over {@link DeviceBuffer} memory.
 * All kernels are synchronous from the caller's perspective: they enqueue
 * device work and return a handle immediately. The only asynchronous
 * operation is `download`, which must wait for in-flight work targeting the
 * buffer to complete before resolving (WebGPU readback is inherently async,
 * which is why `Tensor.to('cpu')` returns a promise).
 *
 * Register an implementation with `registerBackend(device, backend)`; the
 * ndarray dispatch layer picks it up automatically for tensors created on
 * (or moved to) that device.
 */
export interface KernelBackend extends Backend {
  /**
   * Copy host data into new device memory.
   *
   * `data` is always float32 (the host-side numeric values). `dtype` selects
   * the on-device element type: for `float16`/`bfloat16` the values are packed
   * into 2-byte (or bf16-rounded) storage before upload. Absent/`float32`
   * keeps the original float32 behavior exactly.
   */
  upload(data: Float32Array, dtype?: DeviceDType): DeviceBuffer;

  /**
   * Copy device memory back to the host.
   *
   * Resolves after all previously enqueued kernels writing to `buffer`
   * have completed.
   */
  download(buffer: DeviceBuffer): Promise<Float32Array>;

  /** Release device memory. Safe to call once per buffer; must tolerate repeat calls. */
  free(buffer: DeviceBuffer): void;

  /**
   * Allocate a buffer of `size` elements filled with `value`.
   *
   * `dtype` selects the element type (default `float32`); `value` is always a
   * float32 host value.
   */
  fill(value: number, size: number, dtype?: DeviceDType): DeviceBuffer;

  /**
   * Broadcast-aware binary element-wise kernel.
   *
   * `aLayout`/`bLayout` are pre-broadcast to `outShape` by the caller
   * (stride 0 on broadcast dimensions). Returns a contiguous buffer of
   * `prod(outShape)` elements.
   */
  binary(
    op: BinaryKernelOp,
    a: DeviceBuffer,
    aLayout: KernelLayout,
    b: DeviceBuffer,
    bLayout: KernelLayout,
    outShape: readonly number[]
  ): DeviceBuffer;

  /** Strided unary element-wise kernel. Returns a contiguous buffer. */
  unary(op: UnaryKernelOp, x: DeviceBuffer, layout: KernelLayout): DeviceBuffer;

  /**
   * 2-D matrix multiply: `[m, k] @ [k, n] -> [m, n]`.
   *
   * Layouts must be rank-2; strides may describe transposed or sliced views.
   * Returns a contiguous row-major `[m, n]` buffer.
   */
  matmul(
    a: DeviceBuffer,
    aLayout: KernelLayout,
    b: DeviceBuffer,
    bLayout: KernelLayout
  ): DeviceBuffer;

  /**
   * Full reduction over all elements. Returns a buffer holding a single
   * element of the input's dtype. `max`/`min` propagate NaN; `mean` divides
   * the sum by the element count.
   */
  reduce(op: ReduceKernelOp, x: DeviceBuffer, layout: KernelLayout): DeviceBuffer;

  /**
   * Reduction along a single axis. `axis` is a non-negative index into
   * `layout.shape`. Returns a contiguous row-major buffer of the input shape
   * with `axis` removed (`prod(shape) / shape[axis]` elements); the caller
   * reshapes to add a size-1 axis back when `keepdims` is requested.
   * Reducing a zero-length axis is an error.
   */
  reduceAxis(op: ReduceKernelOp, x: DeviceBuffer, layout: KernelLayout, axis: number): DeviceBuffer;

  /**
   * Batched matrix multiply: `[..., m, k] @ [..., k, n] -> [..., m, n]` where
   * the leading batch dimensions match. Each operand arrives with an explicit
   * per-dimension stride (broadcasting a batch dim is a stride-0 dimension),
   * so a shared operand (e.g. one weight matrix across a batch) needs no copy.
   * Returns a contiguous row-major `[batch, m, n]` buffer.
   */
  matmulBatched(
    a: DeviceBuffer,
    aLayout: KernelLayout,
    b: DeviceBuffer,
    bLayout: KernelLayout,
    batch: number,
    m: number,
    k: number,
    n: number
  ): DeviceBuffer;

  /**
   * Broadcast-aware ternary select. `cond`/`a`/`b` are pre-broadcast to
   * `outShape` (stride 0 on broadcast dimensions). Returns a contiguous buffer
   * of `prod(outShape)` elements: `cond != 0 ? a : b` element-wise.
   */
  ternary(
    op: TernaryKernelOp,
    cond: DeviceBuffer,
    condLayout: KernelLayout,
    a: DeviceBuffer,
    aLayout: KernelLayout,
    b: DeviceBuffer,
    bLayout: KernelLayout,
    outShape: readonly number[]
  ): DeviceBuffer;

  /**
   * 2-D convolution unfold. Gathers sliding `kH×kW` windows (with stride,
   * zero padding) from a `[batch, channels, height, width]` input into a
   * contiguous `[batch, outH*outW, channels*kH*kW]` column buffer. Out-of-range
   * (padded) taps are 0.
   */
  im2col(x: DeviceBuffer, layout: KernelLayout, params: Im2ColParams): DeviceBuffer;

  /**
   * Adjoint of {@link im2col}: scatter-adds a contiguous
   * `[batch, outH*outW, channels*kH*kW]` column buffer back into a contiguous
   * `[batch, channels, height, width]` image, summing overlapping window
   * contributions. Used by convolution backward.
   */
  col2im(cols: DeviceBuffer, params: Im2ColParams): DeviceBuffer;

  /**
   * 2-D pooling over a `[batch, channels, height, width]` input, producing a
   * contiguous `[batch, channels, outH, outW]` buffer. `max` propagates NaN
   * and ignores padding; `avg` divides by the number of in-range taps, so
   * padded positions do not count (PyTorch's `count_include_pad=False`).
   */
  pool2d(
    x: DeviceBuffer,
    layout: KernelLayout,
    op: PoolKernelOp,
    params: Im2ColParams
  ): DeviceBuffer;

  /**
   * Backward of {@link pool2d}. Given the pooling input `x` and the upstream
   * gradient `gradOut` (contiguous `[batch, channels, outH, outW]`), returns
   * the input gradient (contiguous NCHW). For `max`, gradient flows only to
   * each window's first-argmax (strict `>`, matching the CPU path); for `avg`,
   * each window's gradient is split evenly across its in-range taps.
   * Atomic-free: one thread per input element sums the contributions of every
   * window that covers it.
   */
  pool2dBackward(
    x: DeviceBuffer,
    xLayout: KernelLayout,
    gradOut: DeviceBuffer,
    op: PoolKernelOp,
    params: Im2ColParams
  ): DeviceBuffer;
}

/** Binary ops a host accelerator may run over contiguous float32 data. */
export type HostBinaryOp = "add" | "sub" | "mul" | "div";

/**
 * Execution interface for host-accelerator backends (e.g. WASM SIMD).
 *
 * Host accelerators share the CPU address space: tensors on their device
 * keep ordinary TypedArray storage, eligible ops run through the
 * accelerator, and everything else uses the normal CPU implementation. That
 * fallback is safe as long as the accelerator's results are bit-identical
 * IEEE-754 float32 arithmetic, which element-wise `add`, `sub`, `mul` and
 * `div` are.
 */
export interface HostAcceleratorBackend extends Backend {
  /**
   * Element-wise `a OP b` over contiguous float32 arrays.
   *
   * @returns The result, or `null` when the accelerator cannot run
   *   (caller falls back to the CPU loop)
   */
  binaryContiguous(op: HostBinaryOp, a: Float32Array, b: Float32Array): Float32Array | null;
}

/**
 * Type guard for the {@link HostAcceleratorBackend} surface.
 *
 * @param backend - Backend to test
 * @returns `true` if the backend exposes host-accelerator kernels
 */
export function isHostAcceleratorBackend(backend: Backend): backend is HostAcceleratorBackend {
  return typeof (backend as Partial<HostAcceleratorBackend>).binaryContiguous === "function";
}

/**
 * Type guard for the {@link KernelBackend} execution surface.
 *
 * @param backend - Backend to test
 * @returns `true` if the backend exposes device kernels
 */
export function isKernelBackend(backend: Backend): backend is KernelBackend {
  const candidate = backend as Partial<KernelBackend>;
  return (
    typeof candidate.upload === "function" &&
    typeof candidate.download === "function" &&
    typeof candidate.free === "function" &&
    typeof candidate.fill === "function" &&
    typeof candidate.binary === "function" &&
    typeof candidate.unary === "function" &&
    typeof candidate.matmul === "function" &&
    typeof candidate.reduce === "function" &&
    typeof candidate.reduceAxis === "function" &&
    typeof candidate.matmulBatched === "function" &&
    typeof candidate.ternary === "function" &&
    typeof candidate.im2col === "function" &&
    typeof candidate.col2im === "function" &&
    typeof candidate.pool2d === "function" &&
    typeof candidate.pool2dBackward === "function"
  );
}
