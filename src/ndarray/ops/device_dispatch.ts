/**
 * Device dispatch — routes ndarray ops on device tensors to kernel backends.
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

import { DeviceError, ShapeError } from "../../core";
import type {
  BinaryKernelOp,
  DeviceBuffer,
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
 * returns `null` and the caller's CPU implementation runs instead — safe
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

/**
 * Move a scalar (0-D) host tensor to `backend` so it can participate in a
 * device binary op, mirroring PyTorch's cpu-scalar promotion.
 */
function uploadScalar(t: Tensor, device: Tensor["device"], backend: KernelBackend): Tensor {
  const value = Number(t.data[t.offset] ?? 0);
  const buffer = backend.fill(value, 1);
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
    rhs = uploadScalar(b, a.device, backend);
  } else if (!aDev && bDev) {
    if (!isScalar(a)) mixedDeviceError(op, a, b);
    const backend = requireKernelBackend(b.device, op);
    lhs = uploadScalar(a, b.device, backend);
  } else if (a.device !== b.device) {
    mixedDeviceError(op, a, b);
  }

  const device = lhs.device;
  const backend = requireKernelBackend(device, op);
  const outShape = getBroadcastShape(lhs.shape, rhs.shape);

  const lhsOwner = lhs.__bufferOwner;
  const rhsOwner = rhs.__bufferOwner;
  if (!lhsOwner || !rhsOwner) {
    throw new DeviceError(`${op}: internal error — device tensor without a device buffer`);
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
    throw new DeviceError("matmul: internal error — device tensor without a device buffer");
  }

  const out = backend.matmul(aOwner.buffer, layoutOf(a), bOwner.buffer, layoutOf(b));
  return Tensor.fromDeviceBuffer({
    owner: new DeviceBufferOwner(out, backend),
    shape: [a.shape[0] ?? 0, b.shape[1] ?? 0],
    device: a.device,
  });
}

/**
 * Route a batched (`ndim > 2`) matmul on device, matching the CPU `dot`
 * Case-5 semantics exactly: when both operands carry batch dimensions they
 * must match; when one operand is 2-D it is shared (broadcast, stride 0)
 * across the batch. Output shape is `[...batchShape, m, n]`.
 */
function dispatchBatchedMatmul(a: Tensor, b: Tensor): Tensor {
  const aBatchRank = Math.max(0, a.ndim - 2);
  const bBatchRank = Math.max(0, b.ndim - 2);
  if (a.ndim < 2 || b.ndim < 2) {
    throw new ShapeError(`dot not implemented for shapes ${a.shape} and ${b.shape}`);
  }
  const aBatchShape = a.shape.slice(0, aBatchRank);
  const bBatchShape = b.shape.slice(0, bBatchRank);

  let batchShape: number[];
  if (aBatchRank > 0 && bBatchRank > 0) {
    if (aBatchRank !== bBatchRank || aBatchShape.some((d, i) => d !== bBatchShape[i])) {
      throw new ShapeError(`batch dimensions don't match: [${aBatchShape}] vs [${bBatchShape}]`);
    }
    batchShape = aBatchShape;
  } else if (aBatchRank > 0) {
    batchShape = aBatchShape;
  } else {
    batchShape = bBatchShape;
  }
  const batchNdim = batchShape.length;

  const m = a.shape[a.ndim - 2] ?? 0;
  const k = a.shape[a.ndim - 1] ?? 0;
  const k2 = b.shape[b.ndim - 2] ?? 0;
  const n = b.shape[b.ndim - 1] ?? 0;
  if (k !== k2) throw new ShapeError(`shapes ${a.shape} and ${b.shape} not aligned`);

  // Full [batch..., X, Y] layouts: batch strides come from the operand that
  // has them, or are 0 (shared) for a 2-D operand broadcast across the batch.
  const aBatchStrides =
    aBatchRank > 0 ? a.strides.slice(0, batchNdim) : new Array<number>(batchNdim).fill(0);
  const bBatchStrides =
    bBatchRank > 0 ? b.strides.slice(0, batchNdim) : new Array<number>(batchNdim).fill(0);
  const aFull: KernelLayout = {
    shape: [...batchShape, m, k],
    strides: [...aBatchStrides, a.strides[a.ndim - 2] ?? 0, a.strides[a.ndim - 1] ?? 0],
    offset: a.offset,
  };
  const bFull: KernelLayout = {
    shape: [...batchShape, k, n],
    strides: [...bBatchStrides, b.strides[b.ndim - 2] ?? 0, b.strides[b.ndim - 1] ?? 0],
    offset: b.offset,
  };

  let batch = 1;
  for (const d of batchShape) batch *= d;

  const backend = requireKernelBackend(a.device, "matmul");
  const aOwner = a.__bufferOwner;
  const bOwner = b.__bufferOwner;
  if (!aOwner || !bOwner) {
    throw new DeviceError("matmul: internal error — device tensor without a device buffer");
  }
  const out = backend.matmulBatched(aOwner.buffer, aFull, bOwner.buffer, bFull, batch, m, k, n);
  return Tensor.fromDeviceBuffer({
    owner: new DeviceBufferOwner(out, backend),
    shape: [...batchShape, m, n],
    device: a.device,
  });
}

/**
 * Route NumPy-style `dot` when either operand is a device tensor.
 *
 * Handles 0-D (scalar multiply), 1-D and 2-D operands by mapping them onto
 * the rank-2 matmul kernel with stride tricks, and batched (>2-D) operands
 * through the batched matmul kernel.
 *
 * @returns The device result, or `null` for host tensors
 */
export function dispatchDot(a: Tensor, b: Tensor): Tensor | null {
  if (!a.isDeviceTensor && !b.isDeviceTensor) return null;
  if (a.ndim === 0 || b.ndim === 0) {
    return dispatchBinary("mul", a, b);
  }
  if (a.device !== b.device) mixedDeviceError("dot", a, b);
  if (a.ndim > 2 || b.ndim > 2) {
    return dispatchBatchedMatmul(a, b);
  }

  const m = a.ndim === 1 ? 1 : (a.shape[0] ?? 0);
  const k = a.ndim === 1 ? (a.shape[0] ?? 0) : (a.shape[1] ?? 0);
  const kb = b.ndim === 1 ? (b.shape[0] ?? 0) : (b.shape[0] ?? 0);
  const n = b.ndim === 1 ? 1 : (b.shape[1] ?? 0);
  if (k !== kb) {
    throw new ShapeError(`shapes ${a.shape} and ${b.shape} not aligned`);
  }

  const backend = requireKernelBackend(a.device, "dot");
  const aOwner = a.__bufferOwner;
  const bOwner = b.__bufferOwner;
  if (!aOwner || !bOwner) {
    throw new DeviceError("dot: internal error — device tensor without a device buffer");
  }

  const aLayout: KernelLayout =
    a.ndim === 1
      ? { shape: [1, k], strides: [0, a.strides[0] ?? 0], offset: a.offset }
      : { shape: a.shape, strides: a.strides, offset: a.offset };
  const bLayout: KernelLayout =
    b.ndim === 1
      ? { shape: [k, 1], strides: [b.strides[0] ?? 0, 0], offset: b.offset }
      : { shape: b.shape, strides: b.strides, offset: b.offset };

  const out = backend.matmul(aOwner.buffer, aLayout, bOwner.buffer, bLayout);
  const outShape: number[] = [];
  if (a.ndim === 2) outShape.push(m);
  if (b.ndim === 2) outShape.push(n);
  return Tensor.fromDeviceBuffer({
    owner: new DeviceBufferOwner(out, backend),
    shape: outShape,
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

  const finalShape = keepdims ? t.shape.map((d, i) => (reducedSet.has(i) ? 1 : d)) : curShape;

  return Tensor.fromDeviceBuffer({
    owner: new DeviceBufferOwner(curBuffer, backend),
    shape: finalShape,
    device: t.device,
  });
}

/** Build the shared conv/pool geometry from a 4-D NCHW input and hyperparams. */
function convParams(
  input: Tensor,
  kernelSize: readonly [number, number],
  stride: readonly [number, number],
  padding: readonly [number, number]
): Im2ColParams {
  const batch = input.shape[0] ?? 0;
  const channels = input.shape[1] ?? 0;
  const height = input.shape[2] ?? 0;
  const width = input.shape[3] ?? 0;
  const [kH, kW] = kernelSize;
  const [sH, sW] = stride;
  const [pH, pW] = padding;
  const outH = Math.floor((height + 2 * pH - kH) / sH) + 1;
  const outW = Math.floor((width + 2 * pW - kW) / sW) + 1;
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
  const [batch, channels, height, width] = inputShape as [number, number, number, number];
  const [kH, kW] = kernelSize;
  const [sH, sW] = stride;
  const [pH, pW] = padding;
  const outH = Math.floor((height + 2 * pH - kH) / sH) + 1;
  const outW = Math.floor((width + 2 * pW - kW) / sW) + 1;
  const p: Im2ColParams = {
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
  const device = cond.isDeviceTensor ? cond.device : a.isDeviceTensor ? a.device : b.device;
  const backend = requireKernelBackend(device, op);

  const operands: Tensor[] = [];
  for (const t of [cond, a, b]) {
    if (t.isDeviceTensor) {
      if (t.device !== device) mixedDeviceError(op, t, cond);
      operands.push(t);
    } else if (isScalar(t)) {
      operands.push(uploadScalar(t, device, backend));
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
    throw new DeviceError(`${op}: internal error — device tensor without a device buffer`);
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
