/**
 * Device dispatch: routes ndarray ops on device tensors to kernel backends.
 *
 * Every wired op calls a `dispatch*` helper first. The helpers return `null`
 * when all operands live in host memory (the caller proceeds with the normal
 * CPU implementation) and a device tensor when the op executed on a kernel
 * backend. Anything a device cannot express throws a `DeviceError` with a
 * transfer hint rather than silently computing somewhere unexpected.
 *
 * Device semantics follow PyTorch:
 * - Mixed-device operands are an error ("expected all tensors to be on the
 *   same device"), except scalar (0-D) host tensors, which are moved to the
 *   device operand's device automatically.
 * - The output lives on the input's device.
 *
 * @module ndarray/ops
 * @see {@link https://deepbox.dev/docs/devices-and-execution | Devices & execution}
 */

import { DeviceError, DTypeError, InvalidParameterError, ShapeError } from "../../core";
import type {
  BinaryKernelOp,
  DeviceBuffer,
  DeviceDType,
  Im2ColParams,
  KernelBackend,
  KernelLayout,
  PoolKernelOp,
  ReduceKernelOp,
  TernaryKernelOp,
  UnaryKernelOp,
} from "../../core/backend/kernels";
import { getHostAccelerator, requireKernelBackend } from "../../core/backend/registry";
import type { Axis } from "../../core/types/common";
import { normalizeAxes } from "../../core/utils/axis";
import { planMatmul } from "../linalg/matmul_plan";
import { isContiguous } from "../tensor/strides";
import { computeStrides, DeviceBufferOwner, Tensor } from "../tensor/Tensor";
import { getBroadcastShape, isScalar } from "./broadcast";

/**
 * Minimum element count for routing through a host accelerator; below this
 * the copy into accelerator memory costs more than the SIMD win.
 */
const HOST_ACCEL_MIN_SIZE = 512;

/**
 * Try to run a binary op through a host accelerator (e.g. WASM SIMD).
 *
 * Only same-shape contiguous float32 tensors are eligible; anything else
 * returns `null` and the caller's CPU implementation runs instead, which is safe
 * because host accelerators produce bit-identical IEEE-754 results over
 * the same host memory.
 */
function tryHostAccelBinary(op: BinaryKernelOp, a: Tensor, b: Tensor): Tensor | null {
  if (op !== "add" && op !== "sub" && op !== "mul" && op !== "div") return null;
  if (a.dtype !== "float32" || b.dtype !== "float32") return null;
  if (a.size < HOST_ACCEL_MIN_SIZE || a.size !== b.size) return null;
  if (a.ndim !== b.ndim) return null;
  for (let i = 0; i < a.ndim; i++) {
    if (a.shape[i] !== b.shape[i]) return null;
  }
  if (!isContiguous(a.shape, a.strides) || !isContiguous(b.shape, b.strides)) return null;

  const accel = getHostAccelerator(a.device !== "cpu" ? a.device : b.device);
  if (!accel) return null;

  const aData = a.data;
  const bData = b.data;
  if (!(aData instanceof Float32Array) || !(bData instanceof Float32Array)) return null;
  const out = accel.binaryContiguous(
    op,
    aData.subarray(a.offset, a.offset + a.size),
    bData.subarray(b.offset, b.offset + b.size)
  );
  if (!out) return null;
  return Tensor.fromTypedArray({
    data: out,
    shape: a.shape,
    dtype: "float32",
    device: a.device !== "cpu" ? a.device : b.device,
  });
}

/** Throw the PyTorch-style mixed-device error. */
function mixedDeviceError(op: string, a: Tensor, b: Tensor): never {
  throw new DeviceError(
    `${op}: expected all tensors to be on the same device, ` +
      `but found "${a.device}" and "${b.device}". ` +
      "Move tensors explicitly with `await t.to(device)`."
  );
}

/** Layout of `t` broadcast to `outShape` (stride 0 on broadcast dimensions). */
function broadcastLayout(t: Tensor, outShape: readonly number[]): KernelLayout {
  const outNdim = outShape.length;
  const strides = new Array<number>(outNdim);
  for (let i = 0; i < outNdim; i++) {
    const inAxis = t.ndim - outNdim + i;
    if (inAxis < 0) {
      strides[i] = 0;
      continue;
    }
    const inDim = t.shape[inAxis] ?? 1;
    const outDim = outShape[i] ?? 1;
    strides[i] = inDim === 1 && outDim > 1 ? 0 : (t.strides[inAxis] ?? 0);
  }
  return { shape: outShape, strides, offset: t.offset };
}

function layoutOf(t: Tensor): KernelLayout {
  return { shape: t.shape, strides: t.strides, offset: t.offset };
}

/** Element type of a device tensor's buffer (`float32` when the buffer does not say). */
function deviceDTypeOf(t: Tensor): DeviceDType {
  return t.__bufferOwner?.buffer.dtype ?? "float32";
}

/**
 * Move a scalar (0-D) host tensor to `backend` so it can participate in a
 * device binary op, mirroring PyTorch's cpu-scalar promotion.
 *
 * The scalar takes the element type of the device operand it is combined with
 * (`dtype`), so `x * 2` works for float16 and bfloat16 tensors instead of
 * tripping the backend's mixed-dtype guard.
 */
function uploadScalar(
  t: Tensor,
  device: Tensor["device"],
  backend: KernelBackend,
  dtype: DeviceDType = "float32"
): Tensor {
  if (t.dtype === "string") {
    throw new DTypeError("cannot combine a string tensor with a device tensor");
  }
  const value = Number(t.data[t.offset] ?? 0);
  const buffer = backend.fill(value, 1, dtype);
  return Tensor.fromDeviceBuffer({
    owner: new DeviceBufferOwner(buffer, backend),
    shape: [],
    device,
    strides: [],
  });
}

/**
 * Route a binary element-wise op when either operand is a device tensor.
 *
 * @returns The device result, or `null` when both operands are host tensors
 * @throws {DeviceError} On mixed devices or unavailable backends
 */
export function dispatchBinary(op: BinaryKernelOp, a: Tensor, b: Tensor): Tensor | null {
  const aDev = a.isDeviceTensor;
  const bDev = b.isDeviceTensor;
  if (!aDev && !bDev) {
    // Host-accelerator devices (wasm) keep host storage; try their SIMD
    // kernels and otherwise let the CPU implementation run.
    if (a.device !== "cpu" || b.device !== "cpu") {
      return tryHostAccelBinary(op, a, b);
    }
    return null;
  }

  let lhs = a;
  let rhs = b;
  if (aDev && !bDev) {
    if (!isScalar(b)) mixedDeviceError(op, a, b);
    const backend = requireKernelBackend(a.device, op);
    rhs = uploadScalar(b, a.device, backend, deviceDTypeOf(a));
  } else if (!aDev && bDev) {
    if (!isScalar(a)) mixedDeviceError(op, a, b);
    const backend = requireKernelBackend(b.device, op);
    lhs = uploadScalar(a, b.device, backend, deviceDTypeOf(b));
  } else if (a.device !== b.device) {
    mixedDeviceError(op, a, b);
  }

  const device = lhs.device;
  const backend = requireKernelBackend(device, op);
  const outShape = getBroadcastShape(lhs.shape, rhs.shape);

  const lhsOwner = lhs.__bufferOwner;
  const rhsOwner = rhs.__bufferOwner;
  if (!lhsOwner || !rhsOwner) {
    throw new DeviceError(`${op}: internal error: device tensor without a device buffer`);
  }

  const out = backend.binary(
    op,
    lhsOwner.buffer,
    broadcastLayout(lhs, outShape),
    rhsOwner.buffer,
    broadcastLayout(rhs, outShape),
    outShape
  );
  return Tensor.fromDeviceBuffer({
    owner: new DeviceBufferOwner(out, backend),
    shape: outShape,
    device,
  });
}

/**
 * Route a unary element-wise op when the operand is a device tensor.
 *
 * @returns The device result, or `null` for host tensors
 */
export function dispatchUnary(op: UnaryKernelOp, t: Tensor): Tensor | null {
  const owner = t.__bufferOwner;
  if (!owner) return null;
  const backend = requireKernelBackend(t.device, op);
  const out = backend.unary(op, owner.buffer, layoutOf(t));
  return Tensor.fromDeviceBuffer({
    owner: new DeviceBufferOwner(out, backend),
    shape: t.shape,
    device: t.device,
  });
}

/**
 * Route a 2-D matmul when either operand is a device tensor.
 *
 * @returns The device result, or `null` for host tensors
 */
export function dispatchMatmul(a: Tensor, b: Tensor): Tensor | null {
  if (!a.isDeviceTensor && !b.isDeviceTensor) return null;
  if (a.device !== b.device) mixedDeviceError("matmul", a, b);
  if (a.ndim !== 2 || b.ndim !== 2) {
    throw new ShapeError("matmul requires 2D tensors");
  }
  if ((a.shape[1] ?? 0) !== (b.shape[0] ?? 0)) {
    throw ShapeError.mismatch(a.shape, b.shape, "matmul");
  }

  const backend = requireKernelBackend(a.device, "matmul");
  const aOwner = a.__bufferOwner;
  const bOwner = b.__bufferOwner;
  if (!aOwner || !bOwner) {
    throw new DeviceError("matmul: internal error: device tensor without a device buffer");
  }

  const out = backend.matmul(aOwner.buffer, layoutOf(a), bOwner.buffer, layoutOf(b));
  return Tensor.fromDeviceBuffer({
    owner: new DeviceBufferOwner(out, backend),
    shape: [a.shape[0] ?? 0, b.shape[1] ?? 0],
    device: a.device,
  });
}

/**
 * Route NumPy-style `dot` when either operand is a device tensor.
 *
 * Shapes follow `numpy.matmul`: 1-D operands are promoted to a row or column
 * vector, and batch dimensions are right-aligned and broadcast (size-1 and
 * missing batch dimensions get stride 0, so no operand is copied). Plain
 * vector and matrix products use the rank-2 matmul kernel through stride
 * tricks, everything with batch dimensions uses the batched matmul kernel.
 * 0-D operands are rejected exactly like the CPU `dot` (use `mul`).
 *
 * @returns The device result, or `null` for host tensors
 */
export function dispatchDot(a: Tensor, b: Tensor): Tensor | null {
  if (!a.isDeviceTensor && !b.isDeviceTensor) return null;
  // Shape rules and error messages are shared with the CPU implementation.
  const plan = planMatmul(a, b, "dot");
  if (a.device !== b.device) mixedDeviceError("dot", a, b);
  const { batchShape, m, k, n } = plan;

  const backend = requireKernelBackend(a.device, "dot");
  const aOwner = a.__bufferOwner;
  const bOwner = b.__bufferOwner;
  if (!aOwner || !bOwner) {
    throw new DeviceError("dot: internal error: device tensor without a device buffer");
  }

  let out: DeviceBuffer;
  if (batchShape.length === 0) {
    const aLayout: KernelLayout = {
      shape: [m, k],
      strides: [plan.aSM, plan.aSK],
      offset: a.offset,
    };
    const bLayout: KernelLayout = {
      shape: [k, n],
      strides: [plan.bSK, plan.bSN],
      offset: b.offset,
    };
    out = backend.matmul(aOwner.buffer, aLayout, bOwner.buffer, bLayout);
  } else {
    const aFull: KernelLayout = {
      shape: [...batchShape, m, k],
      strides: [...plan.aBatchStrides, plan.aSM, plan.aSK],
      offset: a.offset,
    };
    const bFull: KernelLayout = {
      shape: [...batchShape, k, n],
      strides: [...plan.bBatchStrides, plan.bSK, plan.bSN],
      offset: b.offset,
    };
    let batch = 1;
    for (const d of batchShape) batch *= d;
    out = backend.matmulBatched(aOwner.buffer, aFull, bOwner.buffer, bFull, batch, m, k, n);
  }
  return Tensor.fromDeviceBuffer({
    owner: new DeviceBufferOwner(out, backend),
    shape: plan.outShape,
    device: a.device,
  });
}

/**
 * Route a reduction (full or along one/more axes) when the operand is a
 * device tensor.
 *
 * A full reduction (`axis` omitted, or every axis) runs the multi-pass tree
 * reduction. An axis reduction runs the per-output-element axis kernel once
 * per requested axis, reducing the highest axis first so the remaining axis
 * indices stay valid; intermediate device buffers are freed as it goes.
 *
 * @returns The device result tensor, or `null` for host tensors
 */
export function dispatchReduce(
  op: ReduceKernelOp,
  t: Tensor,
  axis: unknown,
  keepdims: boolean
): Tensor | null {
  const owner = t.__bufferOwner;
  if (!owner) return null;
  if (t.size === 0) {
    throw new DeviceError(
      `${op}: reduction of an empty tensor is not supported on device "${t.device}". ` +
        "Move the tensor to the CPU first with `await t.cpu()`."
    );
  }
  const backend = requireKernelBackend(t.device, op);

  // An explicit empty axis list reduces nothing (NumPy `axis=()`), exactly like
  // the CPU path: the result is an element-wise copy, not a full reduction.
  if (Array.isArray(axis) && axis.length === 0) {
    const copied = dispatchUnary("copy", t);
    if (copied) return copied;
  }

  const axes =
    axis === undefined || axis === null ? [] : normalizeAxes(axis as Axis | Axis[], t.ndim);

  // Full reduction: single scalar (multi-pass tree reduce).
  if (axes.length === 0 || axes.length === t.ndim) {
    const out = backend.reduce(op, owner.buffer, layoutOf(t));
    const outShape = keepdims ? t.shape.map(() => 1) : [];
    return Tensor.fromDeviceBuffer({
      owner: new DeviceBufferOwner(out, backend),
      shape: outShape,
      device: t.device,
    });
  }

  // Axis reduction: reduce the highest axis first (descending) so removing a
  // dimension never shifts the index of a still-to-be-reduced lower axis.
  const reducedSet = new Set(axes);
  const sorted = [...reducedSet].sort((a, b) => b - a);
  let curBuffer: DeviceBuffer = owner.buffer;
  let curShape: number[] = [...t.shape];
  let curStrides: number[] = [...t.strides];
  let curOffset = t.offset;
  let owned = false; // curBuffer is an intermediate we allocated and must free

  try {
    for (const ax of sorted) {
      const layout: KernelLayout = { shape: curShape, strides: curStrides, offset: curOffset };
      const next = backend.reduceAxis(op, curBuffer, layout, ax);
      if (owned) backend.free(curBuffer);
      curBuffer = next;
      curShape = curShape.filter((_, d) => d !== ax);
      curStrides = [...computeStrides(curShape)];
      curOffset = 0;
      owned = true;
    }
  } catch (error) {
    // A failing pass must not leak the intermediate buffer of the previous one.
    if (owned) backend.free(curBuffer);
    throw error;
  }

  const finalShape = keepdims ? t.shape.map((d, i) => (reducedSet.has(i) ? 1 : d)) : curShape;

  return Tensor.fromDeviceBuffer({
    owner: new DeviceBufferOwner(curBuffer, backend),
    shape: finalShape,
    device: t.device,
  });
}

/**
 * Validate the window hyperparameters against an NCHW image size and build the
 * shared conv/pool geometry. A kernel that does not fit the padded image is
 * reported here, before any kernel runs, with the sizes involved.
 */
function windowParams(
  dims: readonly [number, number, number, number],
  kernelSize: readonly [number, number],
  stride: readonly [number, number],
  padding: readonly [number, number]
): Im2ColParams {
  const [batch, channels, height, width] = dims;
  const [kH, kW] = kernelSize;
  const [sH, sW] = stride;
  const [pH, pW] = padding;
  const whole = [kH, kW, sH, sW, pH, pW].every((v) => Number.isInteger(v));
  if (!whole || kH < 1 || kW < 1 || sH < 1 || sW < 1 || pH < 0 || pW < 0) {
    throw new InvalidParameterError(
      "kernelSize and stride must be integers >= 1 and padding an integer >= 0",
      "kernelSize",
      { kernelSize, stride, padding }
    );
  }
  const outH = Math.floor((height + 2 * pH - kH) / sH) + 1;
  const outW = Math.floor((width + 2 * pW - kW) / sW) + 1;
  if (outH <= 0 || outW <= 0) {
    throw new InvalidParameterError(
      `invalid output dimensions ${outH}x${outW}; kernel [${kH}, ${kW}] does not fit ` +
        `an input of ${height}x${width} with padding [${pH}, ${pW}]`,
      "output_dimensions",
      { outH, outW }
    );
  }
  return {
    batch,
    channels,
    height,
    width,
    outH,
    outW,
    kH,
    kW,
    strideH: sH,
    strideW: sW,
    padH: pH,
    padW: pW,
  };
}

/** Build the shared conv/pool geometry from a 4-D NCHW input and hyperparams. */
function convParams(
  input: Tensor,
  kernelSize: readonly [number, number],
  stride: readonly [number, number],
  padding: readonly [number, number]
): Im2ColParams {
  if (input.ndim !== 4) {
    throw new ShapeError(
      `expected a 4D [batch, channels, height, width] input, got ${input.ndim}D`
    );
  }
  const dims: [number, number, number, number] = [
    input.shape[0] ?? 0,
    input.shape[1] ?? 0,
    input.shape[2] ?? 0,
    input.shape[3] ?? 0,
  ];
  return windowParams(dims, kernelSize, stride, padding);
}

/**
 * Route a 2-D convolution unfold (`im2col`) on device: a `[B,C,H,W]` input
 * becomes a `[B, outH*outW, C*kH*kW]` column tensor via a gather kernel.
 *
 * @returns The device result, or `null` for host tensors
 */
export function dispatchIm2col(
  input: Tensor,
  kernelSize: [number, number],
  stride: [number, number],
  padding: [number, number]
): Tensor | null {
  const owner = input.__bufferOwner;
  if (!owner) return null;
  const backend = requireKernelBackend(input.device, "im2col");
  const p = convParams(input, kernelSize, stride, padding);
  const out = backend.im2col(owner.buffer, layoutOf(input), p);
  return Tensor.fromDeviceBuffer({
    owner: new DeviceBufferOwner(out, backend),
    shape: [p.batch, p.outH * p.outW, p.channels * p.kH * p.kW],
    device: input.device,
  });
}

/**
 * Route the `col2im` fold (adjoint of `im2col`) on device: scatter-add a
 * `[B, outH*outW, C*kH*kW]` column tensor back into a `[B,C,H,W]` image.
 *
 * @returns The device result, or `null` for host tensors
 */
export function dispatchCol2im(
  cols: Tensor,
  inputShape: readonly number[],
  kernelSize: [number, number],
  stride: [number, number],
  padding: [number, number]
): Tensor | null {
  const owner = cols.__bufferOwner;
  if (!owner) return null;
  const backend = requireKernelBackend(cols.device, "col2im");
  if (inputShape.length !== 4) {
    throw new ShapeError(
      `expected a 4D [batch, channels, height, width] image shape, got ${inputShape.length}D`
    );
  }
  const [batch, channels, height, width] = inputShape as [number, number, number, number];
  const p = windowParams([batch, channels, height, width], kernelSize, stride, padding);
  const out = backend.col2im(owner.buffer, p);
  return Tensor.fromDeviceBuffer({
    owner: new DeviceBufferOwner(out, backend),
    shape: [batch, channels, height, width],
    device: cols.device,
  });
}

/**
 * Route 2-D pooling (`max`/`avg`) on device: `[B,C,H,W]` → `[B,C,outH,outW]`.
 *
 * @returns The device result, or `null` for host tensors
 */
export function dispatchPool2d(
  input: Tensor,
  op: PoolKernelOp,
  kernelSize: [number, number],
  stride: [number, number],
  padding: [number, number]
): Tensor | null {
  const owner = input.__bufferOwner;
  if (!owner) return null;
  const backend = requireKernelBackend(input.device, "pool2d");
  const p = convParams(input, kernelSize, stride, padding);
  const out = backend.pool2d(owner.buffer, layoutOf(input), op, p);
  return Tensor.fromDeviceBuffer({
    owner: new DeviceBufferOwner(out, backend),
    shape: [p.batch, p.channels, p.outH, p.outW],
    device: input.device,
  });
}

/**
 * Route 2-D pooling backward on device: given the pooling input and the
 * upstream gradient (`[B,C,outH,outW]`), returns the input gradient
 * (`[B,C,H,W]`). See {@link dispatchPool2d}.
 *
 * @returns The device result, or `null` for host tensors
 */
export function dispatchPool2dBackward(
  input: Tensor,
  gradOut: Tensor,
  op: PoolKernelOp,
  kernelSize: [number, number],
  stride: [number, number],
  padding: [number, number]
): Tensor | null {
  const inOwner = input.__bufferOwner;
  const goOwner = gradOut.__bufferOwner;
  if (!inOwner || !goOwner) return null;
  const backend = requireKernelBackend(input.device, "pool2dBackward");
  const p = convParams(input, kernelSize, stride, padding);
  const out = backend.pool2dBackward(inOwner.buffer, layoutOf(input), goOwner.buffer, op, p);
  return Tensor.fromDeviceBuffer({
    owner: new DeviceBufferOwner(out, backend),
    shape: input.shape,
    device: input.device,
  });
}

/**
 * Route a broadcast-aware ternary select (`where`) when any operand is a
 * device tensor. Host (0-D scalar) operands are uploaded to the device;
 * mixed non-scalar devices are an error.
 *
 * @returns The device result tensor, or `null` when every operand is a host tensor
 */
export function dispatchTernary(
  op: TernaryKernelOp,
  cond: Tensor,
  a: Tensor,
  b: Tensor
): Tensor | null {
  if (!cond.isDeviceTensor && !a.isDeviceTensor && !b.isDeviceTensor) return null;

  // Resolve the target device from the first device operand.
  const anchor = cond.isDeviceTensor ? cond : a.isDeviceTensor ? a : b;
  const device = anchor.device;
  const backend = requireKernelBackend(device, op);

  // Host scalars among the selected values take the dtype of the device value
  // operand; a host condition stays float32 (it is a 0/1 mask).
  const valueAnchor = a.isDeviceTensor ? a : b.isDeviceTensor ? b : null;
  const valueDType: DeviceDType = valueAnchor ? deviceDTypeOf(valueAnchor) : "float32";

  const operands: Tensor[] = [];
  for (const t of [cond, a, b]) {
    if (t.isDeviceTensor) {
      if (t.device !== device) mixedDeviceError(op, anchor, t);
      operands.push(t);
    } else if (isScalar(t)) {
      operands.push(uploadScalar(t, device, backend, t === cond ? "float32" : valueDType));
    } else {
      throw new DeviceError(
        `${op}: expected all tensors on device "${device}", but found a host tensor. ` +
          "Move it explicitly with `await t.to(device)`."
      );
    }
  }
  const [c, x, y] = operands as [Tensor, Tensor, Tensor];
  const outShape = getBroadcastShape(getBroadcastShape(c.shape, x.shape), y.shape);

  const cOwner = c.__bufferOwner;
  const xOwner = x.__bufferOwner;
  const yOwner = y.__bufferOwner;
  if (!cOwner || !xOwner || !yOwner) {
    throw new DeviceError(`${op}: internal error: device tensor without a device buffer`);
  }

  const out = backend.ternary(
    op,
    cOwner.buffer,
    broadcastLayout(c, outShape),
    xOwner.buffer,
    broadcastLayout(x, outShape),
    yOwner.buffer,
    broadcastLayout(y, outShape),
    outShape
  );
  return Tensor.fromDeviceBuffer({
    owner: new DeviceBufferOwner(out, backend),
    shape: outShape,
    device,
  });
}
