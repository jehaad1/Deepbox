import {
  DeepboxError,
  DeviceError,
  type DType,
  DTypeError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../../core";
import {
  type AnyTensor,
  concatGrad,
  customOp,
  GradTensor,
  im2colGrad,
  parameter,
} from "../../ndarray";
import { readAsNumber, requireNumericData } from "../../ndarray/ops/_internal";
import { dispatchPool2d, dispatchPool2dBackward } from "../../ndarray/ops/device_dispatch";
import { isContiguous, offsetFromFlatIndex } from "../../ndarray/tensor/strides";
import { computeStrides, Tensor } from "../../ndarray/tensor/Tensor";
import { Module } from "../module/Module";
import {
  allPlain,
  type LayerDType,
  resolveLayerDtype,
  settle,
  toGradInput,
  uniformTensor,
} from "./_shared";

/** One output window along a single spatial axis: the input range `[lo, hi)`, clipped to the input. */
interface PoolAxis {
  readonly lo: Int32Array;
  readonly hi: Int32Array;
}

function isFloatDType(dtype: DType): boolean {
  return dtype === "float16" || dtype === "bfloat16" || dtype === "float32" || dtype === "float64";
}

/**
 * Cast `t` to `dtype` when they differ. Layers compute in float64 and then
 * return to the dtype of their inputs, so a float32 graph never picks up
 * float64 nodes (autograd rejects mixing dtypes when it accumulates gradients).
 */
function castTo(t: Tensor, dtype: DType): Tensor {
  return t.dtype === dtype ? t : t.astype(dtype);
}

/**
 * Cast a gradient to the dtype of the tensor it belongs to. Gradients of
 * non-floating tensors stay in float64 because truncating them to an integer
 * dtype would destroy the values.
 */
function gradLike(grad: Tensor, ref: Tensor): Tensor {
  return isFloatDType(ref.dtype) ? castTo(grad, ref.dtype) : grad;
}

/** Differentiable cast of a GradTensor to `dtype` (a no-op when it already has that dtype). */
function asDtype(t: GradTensor, dtype: DType): GradTensor {
  return t.dtype === dtype ? t : t.astype(dtype as Exclude<DType, "string">);
}

/**
 * Output dtype of an op that combines tensors of the given dtypes: float64
 * when any operand is float64, float32 otherwise.
 */
function promoteFloat(...dtypes: DType[]): DType {
  return dtypes.includes("float64") ? "float64" : "float32";
}

/**
 * Materialize any numeric tensor (including non-contiguous views) into a
 * contiguous, logical-order Float64Array. Used by convolution layers whose
 * hand-written forward/backward kernels index elements in row-major order.
 */
function denseFloat64(t: Tensor): Float64Array {
  if (t.isDeviceTensor) {
    throw new DeviceError(
      `This layer runs on the host and cannot read a tensor stored on device "${t.device}"; ` +
        "move it back with `await tensor.cpu()` first"
    );
  }
  const out = new Float64Array(t.size);
  const data = requireNumericData(t.data, "conv");
  const contig = isContiguous(t.shape, t.strides);
  const logical = computeStrides(t.shape);
  for (let i = 0; i < t.size; i++) {
    const off = contig ? t.offset + i : offsetFromFlatIndex(i, logical, t.strides, t.offset);
    out[i] = readAsNumber(data, off);
  }
  return out;
}

function floatTensor(
  data: Float64Array,
  shape: readonly number[],
  device: Tensor["device"],
  dtype: DType
): Tensor {
  return castTo(
    Tensor.fromTypedArray({ data, shape: [...shape], dtype: "float64", device }),
    dtype
  );
}

/** Input window `[lo, hi)` of a sliding kernel at output index `o`, clipped to `[0, inSize)`. */
function slidingAxis(
  inSize: number,
  outSize: number,
  kernel: number,
  stride: number,
  padding: number
): PoolAxis {
  const lo = new Int32Array(outSize);
  const hi = new Int32Array(outSize);
  for (let o = 0; o < outSize; o++) {
    const start = o * stride - padding;
    lo[o] = Math.max(0, start);
    hi[o] = Math.min(inSize, start + kernel);
  }
  return { lo, hi };
}

/**
 * Adaptive pooling window along one axis: `[floor(o * in / out), ceil((o + 1) * in / out))`.
 * Neighbouring windows overlap when `in` is not a multiple of `out`, and no window is
 * empty when `in < out` (the same rule PyTorch uses).
 */
function adaptiveAxis(inSize: number, outSize: number): PoolAxis {
  const lo = new Int32Array(outSize);
  const hi = new Int32Array(outSize);
  for (let o = 0; o < outSize; o++) {
    lo[o] = Math.floor((o * inSize) / outSize);
    hi[o] = Math.floor(((o + 1) * inSize + outSize - 1) / outSize);
  }
  return { lo, hi };
}

/** Axis used for the spatial dimensions a lower-rank pool does not have. */
const UNIT_AXIS: PoolAxis = { lo: new Int32Array([0]), hi: new Int32Array([1]) };

/**
 * Differentiable pooling over a `(planes, D, H, W)` volume. 1-D and 2-D pooling
 * pass `UNIT_AXIS` for the missing axes. Each output window is the Cartesian
 * product of one `[lo, hi)` range per axis, so no per-window index lists are built.
 *
 * - `max` ignores padding (the window is clipped to the input, i.e. -inf padding)
 *   and routes the gradient to the first maximum; a NaN in the window wins and
 *   propagates, as in PyTorch.
 * - `avg` divides by `fixedDivisor` when given (count_include_pad), by the product of
 *   `extents` (the padded window lengths per axis) when given, otherwise by the number of
 *   in-range elements.
 *
 * Floating inputs keep their dtype; other dtypes produce float64.
 */
function pool3d(
  input: GradTensor,
  mode: "max" | "avg",
  planes: number,
  inDims: readonly [number, number, number],
  axes: readonly [PoolAxis, PoolAxis, PoolAxis],
  outShape: readonly number[],
  fixedDivisor: number | null,
  extents?: readonly [Int32Array, Int32Array, Int32Array]
): GradTensor {
  const [inD, inH, inW] = inDims;
  const [aD, aH, aW] = axes;
  const outD = aD.lo.length;
  const outH = aH.lo.length;
  const outW = aW.lo.length;
  const inPlane = inD * inH * inW;
  const outSize = planes * outD * outH * outW;
  const x = denseFloat64(input.tensor);
  const out = new Float64Array(outSize);
  const argmax = mode === "max" ? new Int32Array(outSize).fill(-1) : null;

  const windowCount = (od: number, oh: number, ow: number): number =>
    extents
      ? (extents[0][od] ?? 0) * (extents[1][oh] ?? 0) * (extents[2][ow] ?? 0)
      : ((aD.hi[od] ?? 0) - (aD.lo[od] ?? 0)) *
        ((aH.hi[oh] ?? 0) - (aH.lo[oh] ?? 0)) *
        ((aW.hi[ow] ?? 0) - (aW.lo[ow] ?? 0));

  let o = 0;
  for (let p = 0; p < planes; p++) {
    const base = p * inPlane;
    for (let od = 0; od < outD; od++) {
      const d0 = aD.lo[od] ?? 0;
      const d1 = aD.hi[od] ?? 0;
      for (let oh = 0; oh < outH; oh++) {
        const h0 = aH.lo[oh] ?? 0;
        const h1 = aH.hi[oh] ?? 0;
        for (let ow = 0; ow < outW; ow++, o++) {
          const w0 = aW.lo[ow] ?? 0;
          const w1 = aW.hi[ow] ?? 0;
          if (argmax) {
            let best = Number.NEGATIVE_INFINITY;
            let bestIdx = -1;
            for (let id = d0; id < d1; id++) {
              for (let ih = h0; ih < h1; ih++) {
                const row = base + (id * inH + ih) * inW;
                for (let iw = w0; iw < w1; iw++) {
                  const v = x[row + iw] as number;
                  if (bestIdx < 0 || v > best || Number.isNaN(v)) {
                    best = v;
                    bestIdx = row + iw;
                  }
                }
              }
            }
            out[o] = bestIdx < 0 ? 0 : best;
            argmax[o] = bestIdx;
          } else {
            let s = 0;
            for (let id = d0; id < d1; id++) {
              for (let ih = h0; ih < h1; ih++) {
                const row = base + (id * inH + ih) * inW;
                for (let iw = w0; iw < w1; iw++) s += x[row + iw] as number;
              }
            }
            const d = fixedDivisor ?? windowCount(od, oh, ow);
            out[o] = d === 0 ? 0 : s / d;
          }
        }
      }
    }
  }

  const inShape = input.tensor.shape;
  const device = input.tensor.device;
  const outDtype = isFloatDType(input.tensor.dtype) ? input.tensor.dtype : "float64";
  const outTensor = floatTensor(out, outShape, device, outDtype);

  return customOp(outTensor, [
    [
      input,
      (g: Tensor): Tensor => {
        const go = denseFloat64(g);
        const gi = new Float64Array(input.tensor.size);
        if (argmax) {
          for (let k = 0; k < outSize; k++) {
            const idx = argmax[k] ?? -1;
            if (idx >= 0) gi[idx]! += go[k] ?? 0;
          }
        } else {
          let k = 0;
          for (let p = 0; p < planes; p++) {
            const base = p * inPlane;
            for (let od = 0; od < outD; od++) {
              for (let oh = 0; oh < outH; oh++) {
                for (let ow = 0; ow < outW; ow++, k++) {
                  const d = fixedDivisor ?? windowCount(od, oh, ow);
                  if (d === 0) continue;
                  const share = (go[k] ?? 0) / d;
                  for (let id = aD.lo[od] ?? 0; id < (aD.hi[od] ?? 0); id++) {
                    for (let ih = aH.lo[oh] ?? 0; ih < (aH.hi[oh] ?? 0); ih++) {
                      const row = base + (id * inH + ih) * inW;
                      for (let iw = aW.lo[ow] ?? 0; iw < (aW.hi[ow] ?? 0); iw++) {
                        gi[row + iw]! += share;
                      }
                    }
                  }
                }
              }
            }
          }
        }
        return gradLike(floatTensor(gi, inShape, device, "float64"), input.tensor);
      },
    ],
  ]);
}

/**
 * On-device 2-D pooling with a regular kernel/stride/padding. Runs the pool
 * kernel forward and wires the matching pool-backward kernel through
 * `customOp`, so the whole op stays resident on the device. Semantics match
 * {@link pool3d}: `max` excludes padding and routes the gradient to each
 * window's first-argmax; `avg` divides by the in-range tap count.
 */
function devicePool2d(
  input: GradTensor,
  mode: "max" | "avg",
  kernel: [number, number],
  stride: [number, number],
  padding: [number, number]
): GradTensor {
  const outTensor = dispatchPool2d(input.tensor, mode, kernel, stride, padding);
  if (!outTensor) {
    throw new DeepboxError("devicePool2d: input is not a device tensor");
  }
  return customOp(outTensor, [
    [
      input,
      (g: Tensor): Tensor => {
        const gi = dispatchPool2dBackward(input.tensor, g, mode, kernel, stride, padding);
        if (!gi) {
          throw new DeepboxError("devicePool2d: pooling backward is not available on device");
        }
        return gi;
      },
    ],
  ]);
}

function checkInteger(name: string, value: number, min: 0 | 1): number {
  if (!Number.isInteger(value) || value < min) {
    throw new InvalidParameterError(
      `${name} must be a ${min === 1 ? "positive" : "non-negative"} integer`,
      name,
      value
    );
  }
  return value;
}

function normalizePair(
  name: string,
  value: number | [number, number],
  allowZero: boolean,
  description: string
): [number, number] {
  const arr = typeof value === "number" ? [value, value] : value;
  const first = arr[0];
  const second = arr[1];
  if (
    arr.length !== 2 ||
    first === undefined ||
    second === undefined ||
    !Number.isInteger(first) ||
    !Number.isInteger(second) ||
    (allowZero ? first < 0 || second < 0 : first <= 0 || second <= 0)
  ) {
    throw new InvalidParameterError(`${name} must be ${description}`, name, value);
  }
  return [first, second];
}

function normalizeTriple(
  name: string,
  value: number | [number, number, number],
  allowZero: boolean,
  description: string
): [number, number, number] {
  const arr = typeof value === "number" ? [value, value, value] : value;
  const a = arr[0];
  const b = arr[1];
  const c = arr[2];
  if (
    arr.length !== 3 ||
    a === undefined ||
    b === undefined ||
    c === undefined ||
    !Number.isInteger(a) ||
    !Number.isInteger(b) ||
    !Number.isInteger(c) ||
    (allowZero ? a < 0 || b < 0 || c < 0 : a <= 0 || b <= 0 || c <= 0)
  ) {
    throw new InvalidParameterError(`${name} must be ${description}`, name, value);
  }
  return [a, b, c];
}

/**
 * A pooling window made only of padding has no values to reduce, so PyTorch
 * requires the padding to be at most half the kernel size.
 */
function checkPoolPadding(
  layer: string,
  kernel: readonly number[],
  padding: readonly number[]
): void {
  for (let i = 0; i < kernel.length; i++) {
    const k = kernel[i] ?? 1;
    const p = padding[i] ?? 0;
    if (p > Math.floor(k / 2)) {
      throw new InvalidParameterError(
        `${layer}: padding must be at most half of kernelSize (kernelSize=${k}, padding=${p})`,
        "padding",
        padding.length === 1 ? p : [...padding]
      );
    }
  }
}

/**
 * Throws a ShapeError if a computed output size is not positive. `sizes` are the output
 * sizes per axis.
 */
function checkOutputSizes(layer: string, sizes: readonly number[]): void {
  for (const s of sizes) {
    if (s <= 0) {
      throw new ShapeError(
        `${layer}: computed output size ${sizes.join("x")} is not positive; check kernelSize, stride and padding against the input size`
      );
    }
  }
}

/** Number of window positions of a sliding kernel along one axis. */
function slidingOutSize(inSize: number, kernel: number, stride: number, padding: number): number {
  return Math.floor((inSize + 2 * padding - kernel) / stride) + 1;
}

/**
 * Number of pooling windows along one axis. With `ceilMode` the last partial window is kept
 * (PyTorch's `ceil_mode`), unless it would start in the right padding.
 */
function poolOutSize(
  inSize: number,
  kernel: number,
  stride: number,
  padding: number,
  ceilMode: boolean
): number {
  if (!ceilMode) return slidingOutSize(inSize, kernel, stride, padding);
  let out = Math.floor((inSize + 2 * padding - kernel + stride - 1) / stride) + 1;
  if ((out - 1) * stride >= inSize + padding) out--;
  return out;
}

/**
 * Length of each pooling window along one axis, clipped to the padded input (so padding
 * counts but the overhang of a `ceilMode` window does not). It is the per-axis factor of
 * PyTorch's `count_include_pad` divisor.
 */
function paddedExtent(
  inSize: number,
  outSize: number,
  kernel: number,
  stride: number,
  padding: number
): Int32Array {
  const ext = new Int32Array(outSize);
  for (let o = 0; o < outSize; o++) {
    const start = o * stride - padding;
    ext[o] = Math.min(start + kernel, inSize + padding) - start;
  }
  return ext;
}

/** Extent used for the spatial dimensions a lower-rank pool does not have. */
const UNIT_EXTENT = new Int32Array([1]);

/** Rejects `ceilMode` for tensors that live on a device, where the pooling kernels do not support it. */
function rejectDeviceCeilMode(layer: string): never {
  throw new DeviceError(
    `${layer} with ceilMode runs on the host and cannot read a device tensor; ` +
      "move it back with `await tensor.cpu()` first"
  );
}

/** Output size of a transposed convolution along one axis. */
function transposedOutSize(
  inSize: number,
  kernel: number,
  stride: number,
  padding: number,
  outputPadding: number
): number {
  return (inSize - 1) * stride - 2 * padding + kernel + outputPadding;
}

/** PyTorch requires `outputPadding < stride` so the extra size stays within one stride step. */
function checkOutputPadding(
  outputPadding: readonly number[],
  stride: readonly number[],
  value: unknown
): void {
  for (let i = 0; i < outputPadding.length; i++) {
    const op = outputPadding[i] ?? 0;
    if (op > 0 && op >= (stride[i] ?? 1)) {
      throw new InvalidParameterError(
        `outputPadding must be smaller than stride (outputPadding=${op}, stride=${stride[i]})`,
        "outputPadding",
        value
      );
    }
  }
}

/**
 * Valid kernel taps `[lo, hi)` for output position `o` of a dilated kernel, so that
 * `o * stride + k * dilation - padBefore` lies inside `[0, inSize)`.
 */
function dilatedTapRange(
  o: number,
  stride: number,
  padBefore: number,
  dilation: number,
  kernel: number,
  inSize: number
): [number, number] {
  const start = o * stride - padBefore;
  const lo = Math.max(0, Math.ceil(-start / dilation));
  const hi = Math.min(kernel, Math.floor((inSize - 1 - start) / dilation) + 1);
  return [lo, hi];
}

function rejectString(dtype: DType, layer: string): void {
  if (dtype === "string") {
    throw new DTypeError(`${layer} does not support string dtype`);
  }
}

/** Reject a string input, then hand the tensor back so the call can be chained. */
function requireNumeric(t: GradTensor, layer: string): GradTensor {
  rejectString(t.dtype, layer);
  return t;
}

/** Padding option of the convolution layers: an amount, or `"same"` / `"valid"`. */
type ConvPadding<T> = T | "same" | "valid";

/** Check that `groups` is a positive integer dividing both channel counts. */
function checkGroups(inChannels: number, outChannels: number, groups: number): number {
  checkInteger("groups", groups, 1);
  if (inChannels % groups !== 0) {
    throw new InvalidParameterError(
      `inChannels (${inChannels}) must be divisible by groups (${groups})`,
      "groups",
      groups
    );
  }
  if (outChannels % groups !== 0) {
    throw new InvalidParameterError(
      `outChannels (${outChannels}) must be divisible by groups (${groups})`,
      "groups",
      groups
    );
  }
  return groups;
}

/**
 * Resolve the padding option into the zeros added before and after each spatial axis.
 *
 * - an amount (number or tuple, checked by `normalize`) pads both sides by that amount;
 * - `"valid"` means no padding;
 * - `"same"` pads so that the output has the input size (stride 1 only). The total
 *   `dilation * (kernel - 1)` is split with the smaller half in front, so an odd total puts the
 *   extra zero at the end (as PyTorch does).
 */
function resolveConvPadding<T>(
  layer: string,
  padding: ConvPadding<T>,
  normalize: (value: T) => number[],
  kernel: readonly number[],
  stride: readonly number[],
  dilation: readonly number[]
): { before: number[]; after: number[] } {
  if (padding === "valid") {
    return { before: kernel.map(() => 0), after: kernel.map(() => 0) };
  }
  if (padding === "same") {
    if (stride.some((s) => s !== 1)) {
      throw new InvalidParameterError(
        `${layer}: padding "same" is not supported for strided convolutions; use stride 1`,
        "padding",
        padding
      );
    }
    const before: number[] = [];
    const after: number[] = [];
    for (let i = 0; i < kernel.length; i++) {
      const total = (dilation[i] ?? 1) * ((kernel[i] ?? 1) - 1);
      const half = Math.floor(total / 2);
      before.push(half);
      after.push(total - half);
    }
    return { before, after };
  }
  const amount = normalize(padding);
  return { before: amount, after: [...amount] };
}

/** Size of the kernel footprint of a dilated kernel along one axis. */
function effectiveKernel(kernel: number, dilation: number): number {
  return dilation * (kernel - 1) + 1;
}

/** Number of window positions of a dilated kernel with separate padding before and after. */
function convOutSize(
  inSize: number,
  kernel: number,
  stride: number,
  dilation: number,
  before: number,
  after: number
): number {
  return Math.floor((inSize + before + after - effectiveKernel(kernel, dilation)) / stride) + 1;
}

/** 0/1 matrix that spreads a flattened `kH x kW` kernel onto its dilated `kEffH x kEffW` grid. */
function dilationMatrix(
  kernel: readonly [number, number],
  dilation: readonly [number, number],
  dtype: LayerDType,
  device: Tensor["device"]
): GradTensor {
  const [kH, kW] = kernel;
  const [dH, dW] = dilation;
  const kEffW = effectiveKernel(kW, dW);
  const kEffArea = effectiveKernel(kH, dH) * kEffW;
  const data =
    dtype === "float64"
      ? new Float64Array(kH * kW * kEffArea)
      : new Float32Array(kH * kW * kEffArea);
  for (let i = 0; i < kH; i++) {
    for (let j = 0; j < kW; j++) {
      data[(i * kW + j) * kEffArea + i * dH * kEffW + j * dW] = 1;
    }
  }
  return GradTensor.fromTensor(
    Tensor.fromTypedArray({ data, shape: [kH * kW, kEffArea], dtype, device }),
    { requiresGrad: false }
  );
}

/** Geometry of a 2-D convolution (a 1-D convolution is a 2-D one with height 1). */
interface Conv2dGeometry {
  readonly kernel: readonly [number, number];
  readonly stride: readonly [number, number];
  readonly dilation: readonly [number, number];
  readonly before: readonly [number, number];
  readonly after: readonly [number, number];
  readonly groups: number;
}

/**
 * Differentiable 2-D cross-correlation of `input` `(B, C, H, W)` with `weight`
 * `(O, C / groups, kH, kW)` through im2col and a matrix product.
 *
 * - Dilation spreads the kernel onto its dilated grid with a constant 0/1 matrix, so the
 *   gradient of the weight flows through it.
 * - Groups run one product per channel group and join the outputs on the channel axis.
 * - Padding that differs between the two sides (`"same"` with an even dilated kernel) is
 *   applied as the larger amount on both sides, and the surplus leading outputs are cropped.
 */
function conv2dForward(
  input: GradTensor,
  weight: GradTensor,
  bias: GradTensor | undefined,
  geo: Conv2dGeometry
): GradTensor {
  const batch = input.shape[0] ?? 0;
  const inH = input.shape[2] ?? 0;
  const inW = input.shape[3] ?? 0;
  const outC = weight.shape[0] ?? 0;
  const groupIn = weight.shape[1] ?? 0;
  const [kH, kW] = geo.kernel;
  const [dH, dW] = geo.dilation;
  const kEffH = effectiveKernel(kH, dH);
  const kEffW = effectiveKernel(kW, dW);
  const padH = Math.max(geo.before[0], geo.after[0]);
  const padW = Math.max(geo.before[1], geo.after[1]);
  const groups = geo.groups;
  const groupOut = outC / groups;

  // (outC, groupIn * kEffH * kEffW)
  let weightFlat: GradTensor;
  if (dH === 1 && dW === 1) {
    weightFlat = weight.reshape([outC, groupIn * kH * kW]);
  } else {
    const spread = dilationMatrix(
      geo.kernel,
      geo.dilation,
      weight.dtype as LayerDType,
      weight.device
    );
    weightFlat = weight
      .reshape([outC * groupIn, kH * kW])
      .matmul(spread)
      .reshape([outC, groupIn * kEffH * kEffW]);
  }

  const parts: GradTensor[] = [];
  for (let g = 0; g < groups; g++) {
    const x =
      groups === 1 ? input : input.slice({}, { start: g * groupIn, end: (g + 1) * groupIn });
    const w =
      groups === 1
        ? weightFlat
        : weightFlat.slice({ start: g * groupOut, end: (g + 1) * groupOut });
    // (B, outPixels, groupIn * kEffH * kEffW) @ (.., groupOut) -> (B, outPixels, groupOut)
    const cols = im2colGrad(x, [kEffH, kEffW], [geo.stride[0], geo.stride[1]], [padH, padW]);
    parts.push(cols.matmul(w.transpose()));
  }
  const out = parts.length === 1 ? (parts[0] as GradTensor) : concatGrad(parts, 2);

  const fullH = convOutSize(inH, kH, geo.stride[0], dH, padH, padH);
  const fullW = convOutSize(inW, kW, geo.stride[1], dW, padW, padW);
  let result = out.transpose([0, 2, 1]).reshape([batch, outC, fullH, fullW]);

  // Crop the surplus leading outputs of an uneven "same" padding.
  const leadH = padH - geo.before[0];
  const leadW = padW - geo.before[1];
  if (leadH > 0 || leadW > 0) {
    const outH = convOutSize(inH, kH, geo.stride[0], dH, geo.before[0], geo.after[0]);
    const outW = convOutSize(inW, kW, geo.stride[1], dW, geo.before[1], geo.after[1]);
    result = result.slice(
      {},
      {},
      { start: leadH, end: leadH + outH },
      { start: leadW, end: leadW + outW }
    );
  }

  if (bias) {
    result = result.add(bias.reshape([1, outC, 1, 1]));
  }
  return result;
}

/**
 * 1D Convolutional Layer.
 *
 * Applies a 1D cross-correlation over an input of shape `(batch, inChannels, length)`
 * and returns `(batch, outChannels, outLength)` with
 * `outLength = floor((length + 2 * padding - dilation * (kernelSize - 1) - 1) / stride) + 1`.
 *
 * Weights have shape `(outChannels, inChannels / groups, kernelSize)`. As in PyTorch, weights
 * and bias are drawn from `U(-1/sqrt(fanIn), 1/sqrt(fanIn))` with
 * `fanIn = inChannels / groups * kernelSize`.
 *
 * `dilation` spaces out the kernel taps, `groups` splits the channels into independent
 * groups (`groups = inChannels` gives a depthwise convolution), and `padding` accepts
 * `"valid"` (no padding) or `"same"` (output length equals input length; stride 1 only).
 *
 * The layer computes in its parameter dtype (`float32` unless `dtype` or the global default
 * dtype says otherwise) and casts the input to it. A `GradTensor` input gives a `GradTensor`;
 * a plain `Tensor` input gives a `GradTensor` that tracks the weights while they require grad
 * and gradient tracking is on, and a plain `Tensor` otherwise (inside `noGrad()` or with
 * frozen weights).
 *
 * @example
 * ```ts
 * import { Conv1d } from 'deepbox/nn';
 *
 * const conv = new Conv1d(16, 33, 3); // in_channels=16, out_channels=33, kernel_size=3
 * const same = new Conv1d(16, 32, 5, { padding: 'same', dilation: 2, groups: 4 });
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-layers | Deepbox Layers}
 */
export class Conv1d extends Module {
  private readonly inChannels: number;
  private readonly outChannels: number;
  private readonly kernelSize: number;
  private readonly stride: number;
  private readonly dilation: number;
  private readonly groups: number;
  private readonly padBefore: number;
  private readonly padAfter: number;
  private readonly paddingLabel: string;
  private readonly useBias: boolean;
  private readonly layerDtype: LayerDType;

  private weight_?: GradTensor;
  private bias_?: GradTensor;

  /**
   * @param inChannels - Number of input channels
   * @param outChannels - Number of output channels
   * @param kernelSize - Size of the convolution kernel
   * @param options.stride - Step between windows (default 1)
   * @param options.padding - Zeros added to both ends of the input (default 0), or `"valid"`
   *   (no padding) or `"same"` (keep the length; needs `stride` 1)
   * @param options.dilation - Spacing between kernel taps (default 1)
   * @param options.groups - Number of channel groups (default 1); must divide `inChannels` and
   *   `outChannels`
   * @param options.bias - Add a learnable bias (default true)
   * @param options.dtype - Dtype of the parameters (default: the global default dtype)
   * @throws {InvalidParameterError} If a size, stride, dilation, groups or padding is not valid
   */
  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number,
    options: {
      readonly stride?: number;
      readonly padding?: ConvPadding<number>;
      readonly dilation?: number;
      readonly groups?: number;
      readonly bias?: boolean;
      readonly dtype?: LayerDType;
    } = {}
  ) {
    super();

    checkInteger("inChannels", inChannels, 1);
    checkInteger("outChannels", outChannels, 1);
    checkInteger("kernelSize", kernelSize, 1);
    const stride = checkInteger("stride", options.stride ?? 1, 1);
    const dilation = checkInteger("dilation", options.dilation ?? 1, 1);
    const groups = checkGroups(inChannels, outChannels, options.groups ?? 1);
    const padOption = options.padding ?? 0;
    const resolved = resolveConvPadding(
      "Conv1d",
      padOption,
      (value: number) => [checkInteger("padding", value, 0)],
      [kernelSize],
      [stride],
      [dilation]
    );

    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = kernelSize;
    this.stride = stride;
    this.dilation = dilation;
    this.groups = groups;
    this.padBefore = resolved.before[0] ?? 0;
    this.padAfter = resolved.after[0] ?? 0;
    this.paddingLabel = typeof padOption === "string" ? `"${padOption}"` : String(this.padBefore);
    this.useBias = options.bias ?? true;
    this.layerDtype = resolveLayerDtype(options.dtype);

    this.initializeParameters();
  }

  private initializeParameters(): void {
    // PyTorch default: weight and bias ~ U(-1/sqrt(fanIn), 1/sqrt(fanIn)).
    const groupIn = this.inChannels / this.groups;
    const bound = 1 / Math.sqrt(groupIn * this.kernelSize);
    const opts = { dtype: this.layerDtype };
    this.weight_ = parameter(
      uniformTensor([this.outChannels, groupIn, this.kernelSize], bound, opts)
    );
    this.registerParameter("weight", this.weight_);

    if (this.useBias) {
      this.bias_ = parameter(uniformTensor([this.outChannels], bound, opts));
      this.registerParameter("bias", this.bias_);
    }
  }

  /**
   * @param x - Input of shape `(batch, inChannels, length)`
   * @returns Output of shape `(batch, outChannels, outLength)`
   * @throws {ShapeError} If the input is not 3-D, its channel count does not match, or the
   *   (dilated) kernel does not fit the padded input
   * @throws {DTypeError} If the input has string dtype
   */
  forward(x: GradTensor): GradTensor;
  forward(x: Tensor): AnyTensor;
  forward(x: AnyTensor): AnyTensor;
  forward(x: AnyTensor): AnyTensor {
    return settle(this.run(x), allPlain(x));
  }

  private run(x: AnyTensor): GradTensor {
    // Convert to GradTensor if needed
    const input = toGradInput(x);

    rejectString(input.dtype, "Conv1d");

    // Input shape: (batch, in_channels, length)
    if (input.ndim !== 3) {
      throw new ShapeError(`Conv1d expects 3D input (batch, channels, length), got ${input.ndim}D`);
    }

    const batch = input.shape[0] ?? 0;
    const inC = input.shape[1] ?? 0;
    const inL = input.shape[2] ?? 0;

    if (inC !== this.inChannels) {
      throw new ShapeError(`Expected ${this.inChannels} input channels, got ${inC}`);
    }

    const storedWeight = this.weight_;
    if (!storedWeight) throw new NotFittedError("Weight not initialized");

    // A kernel larger than the padded input is a shape error
    checkOutputSizes("Conv1d", [
      convOutSize(inL, this.kernelSize, this.stride, this.dilation, this.padBefore, this.padAfter),
    ]);

    // The layer computes in its parameter dtype: the input is cast to it.
    const dtype = storedWeight.dtype;

    // 1D convolution as a 2D one with height 1: (B, C, L) -> (B, C, 1, L)
    const input2d = asDtype(input, dtype).reshape([batch, inC, 1, inL]);
    const out = conv2dForward(
      input2d,
      storedWeight.reshape([this.outChannels, this.inChannels / this.groups, 1, this.kernelSize]),
      this.useBias ? this.bias_ : undefined,
      {
        kernel: [1, this.kernelSize],
        stride: [1, this.stride],
        dilation: [1, this.dilation],
        before: [0, this.padBefore],
        after: [0, this.padAfter],
        groups: this.groups,
      }
    );
    return out.reshape([batch, this.outChannels, out.shape[3] ?? 0]);
  }

  /** Learnable kernel of shape `(outChannels, inChannels / groups, kernelSize)`. */
  get weight(): GradTensor {
    if (!this.weight_) {
      throw new NotFittedError("Weight not initialized");
    }
    return this.weight_;
  }

  /** Learnable bias of shape `(outChannels)`, or `undefined` when the layer was built with `bias: false`. */
  get bias(): GradTensor | undefined {
    return this.bias_;
  }

  override toString(): string {
    const extra =
      (this.dilation === 1 ? "" : `, dilation=${this.dilation}`) +
      (this.groups === 1 ? "" : `, groups=${this.groups}`);
    return `Conv1d(${this.inChannels}, ${this.outChannels}, kernel_size=${this.kernelSize}, stride=${this.stride}, padding=${this.paddingLabel}${extra}, bias=${this.useBias})`;
  }
}

/**
 * 2D Convolutional Layer.
 *
 * Applies a 2D cross-correlation over an input of shape `(batch, inChannels, height, width)`
 * and returns `(batch, outChannels, outH, outW)` with
 * `outH = floor((height + 2 * padH - dilH * (kH - 1) - 1) / strideH) + 1` (and likewise for
 * the width).
 *
 * Weights have shape `(outChannels, inChannels / groups, kH, kW)`. As in PyTorch, weights and
 * bias are drawn from `U(-1/sqrt(fanIn), 1/sqrt(fanIn))` with
 * `fanIn = inChannels / groups * kH * kW`.
 *
 * `dilation` spaces out the kernel taps, `groups` splits the channels into independent
 * groups (`groups = inChannels` gives a depthwise convolution), and `padding` accepts
 * `"valid"` (no padding) or `"same"` (output size equals input size; stride 1 only).
 *
 * The layer computes in its parameter dtype (`float32` unless `dtype` or the global default
 * dtype says otherwise) and casts the input to it. A `GradTensor` input gives a `GradTensor`;
 * a plain `Tensor` input gives a `GradTensor` that tracks the weights while they require grad
 * and gradient tracking is on, and a plain `Tensor` otherwise (inside `noGrad()` or with
 * frozen weights).
 *
 * @example
 * ```ts
 * import { Conv2d } from 'deepbox/nn';
 *
 * const conv = new Conv2d(3, 64, 3); // RGB to 64 channels, 3x3 kernel
 * const depthwise = new Conv2d(32, 32, 3, { padding: 'same', groups: 32 });
 * const dilated = new Conv2d(8, 8, 3, { dilation: 2, padding: 2 });
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-layers | Deepbox Layers}
 */
export class Conv2d extends Module {
  private readonly inChannels: number;
  private readonly outChannels: number;
  private readonly kernelSize: [number, number];
  private readonly stride: [number, number];
  private readonly dilation: [number, number];
  private readonly groups: number;
  private readonly padBefore: [number, number];
  private readonly padAfter: [number, number];
  private readonly paddingLabel: string;
  private readonly useBias: boolean;
  private readonly layerDtype: LayerDType;

  private weight_?: GradTensor;
  private bias_?: GradTensor;

  /**
   * @param inChannels - Number of input channels
   * @param outChannels - Number of output channels
   * @param kernelSize - Kernel size, one integer or `[kH, kW]`
   * @param options.stride - Step between windows (default 1)
   * @param options.padding - Zeros added to each side of the input (default 0), or `"valid"`
   *   (no padding) or `"same"` (keep the size; needs `stride` 1)
   * @param options.dilation - Spacing between kernel taps, one integer or `[dH, dW]` (default 1)
   * @param options.groups - Number of channel groups (default 1); must divide `inChannels` and
   *   `outChannels`
   * @param options.bias - Add a learnable bias (default true)
   * @param options.dtype - Dtype of the parameters (default: the global default dtype)
   * @throws {InvalidParameterError} If a size, stride, dilation, groups or padding is not valid
   */
  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number | [number, number],
    options: {
      readonly stride?: number | [number, number];
      readonly padding?: ConvPadding<number | [number, number]>;
      readonly dilation?: number | [number, number];
      readonly groups?: number;
      readonly bias?: boolean;
      readonly dtype?: LayerDType;
    } = {}
  ) {
    super();
    checkInteger("inChannels", inChannels, 1);
    checkInteger("outChannels", outChannels, 1);
    const kernelArr = normalizePair(
      "kernelSize",
      kernelSize,
      false,
      "a positive integer or a tuple of two positive integers"
    );

    const strideArr = normalizePair(
      "stride",
      options.stride ?? 1,
      false,
      "a positive integer or a tuple of two positive integers"
    );

    const dilationArr = normalizePair(
      "dilation",
      options.dilation ?? 1,
      false,
      "a positive integer or a tuple of two positive integers"
    );
    const groups = checkGroups(inChannels, outChannels, options.groups ?? 1);

    const padOption = options.padding ?? 0;
    const resolved = resolveConvPadding(
      "Conv2d",
      padOption,
      (value: number | [number, number]) =>
        normalizePair(
          "padding",
          value,
          true,
          "a non-negative integer or a tuple of two non-negative integers"
        ),
      kernelArr,
      strideArr,
      dilationArr
    );

    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = kernelArr;
    this.stride = strideArr;
    this.dilation = dilationArr;
    this.groups = groups;
    this.padBefore = [resolved.before[0] ?? 0, resolved.before[1] ?? 0];
    this.padAfter = [resolved.after[0] ?? 0, resolved.after[1] ?? 0];
    this.paddingLabel =
      typeof padOption === "string" ? `"${padOption}"` : JSON.stringify(this.padBefore);

    this.useBias = options.bias ?? true;
    this.layerDtype = resolveLayerDtype(options.dtype);

    this.initializeParameters();
  }

  private initializeParameters(): void {
    // PyTorch default: weight and bias ~ U(-1/sqrt(fanIn), 1/sqrt(fanIn)).
    const kH = this.kernelSize[0] ?? 1;
    const kW = this.kernelSize[1] ?? 1;
    const groupIn = this.inChannels / this.groups;
    const bound = 1 / Math.sqrt(groupIn * kH * kW);
    const opts = { dtype: this.layerDtype };
    this.weight_ = parameter(uniformTensor([this.outChannels, groupIn, kH, kW], bound, opts));
    this.registerParameter("weight", this.weight_);

    if (this.useBias) {
      this.bias_ = parameter(uniformTensor([this.outChannels], bound, opts));
      this.registerParameter("bias", this.bias_);
    }
  }

  /**
   * @param x - Input of shape `(batch, inChannels, height, width)`
   * @returns Output of shape `(batch, outChannels, outH, outW)`
   * @throws {ShapeError} If the input is not 4-D, its channel count does not match, or the
   *   (dilated) kernel does not fit the padded input
   * @throws {DTypeError} If the input has string dtype
   */
  forward(x: GradTensor): GradTensor;
  forward(x: Tensor): AnyTensor;
  forward(x: AnyTensor): AnyTensor;
  forward(x: AnyTensor): AnyTensor {
    return settle(this.run(x), allPlain(x));
  }

  private run(x: AnyTensor): GradTensor {
    const input = toGradInput(x);

    rejectString(input.dtype, "Conv2d");

    if (input.ndim !== 4) {
      throw new ShapeError(
        `Conv2d expects 4D input (batch, channels, height, width), got ${input.ndim}D`
      );
    }

    const inC = input.shape[1] ?? 0;
    const inH = input.shape[2] ?? 0;
    const inW = input.shape[3] ?? 0;

    if (inC !== this.inChannels) {
      throw new ShapeError(`Expected ${this.inChannels} input channels, got ${inC}`);
    }

    const storedWeight = this.weight_;
    if (!storedWeight) throw new NotFittedError("Weight not initialized");

    // The layer computes in its parameter dtype: the input is cast to it.
    const dtype = storedWeight.dtype;

    const [kH, kW] = this.kernelSize;
    const [sH, sW] = this.stride;
    const [dH, dW] = this.dilation;

    // Output size; a kernel larger than the padded input is a shape error
    const outH = convOutSize(inH, kH, sH, dH, this.padBefore[0], this.padAfter[0]);
    const outW = convOutSize(inW, kW, sW, dW, this.padBefore[1], this.padAfter[1]);
    checkOutputSizes("Conv2d", [outH, outW]);

    return conv2dForward(
      asDtype(input, dtype),
      storedWeight,
      this.useBias ? this.bias_ : undefined,
      {
        kernel: this.kernelSize,
        stride: this.stride,
        dilation: this.dilation,
        before: this.padBefore,
        after: this.padAfter,
        groups: this.groups,
      }
    );
  }

  /** Learnable kernel of shape `(outChannels, inChannels / groups, kH, kW)`. */
  get weight(): GradTensor {
    if (!this.weight_) {
      throw new NotFittedError("Weight not initialized");
    }
    return this.weight_;
  }

  /** Learnable bias of shape `(outChannels)`, or `undefined` when the layer was built with `bias: false`. */
  get bias(): GradTensor | undefined {
    return this.bias_;
  }

  override toString(): string {
    const extra =
      (this.dilation[0] === 1 && this.dilation[1] === 1
        ? ""
        : `, dilation=${JSON.stringify(this.dilation)}`) +
      (this.groups === 1 ? "" : `, groups=${this.groups}`);
    return `Conv2d(${this.inChannels}, ${this.outChannels}, kernel_size=${JSON.stringify(this.kernelSize)}, stride=${JSON.stringify(this.stride)}, padding=${this.paddingLabel}${extra}, bias=${this.useBias})`;
  }
}

/**
 * 2D Max Pooling Layer.
 *
 * Applies a 2D max pooling over an input of shape `(batch, channels, height, width)`.
 * Padding is ignored by the maximum (it behaves like `-Infinity`), a NaN inside a
 * window propagates to the output, and the gradient flows to the first maximum of
 * each window.
 *
 * The stride defaults to the kernel size. As in PyTorch, `padding` may be at most
 * half of `kernelSize`.
 *
 * @example
 * ```ts
 * import { MaxPool2d } from 'deepbox/nn';
 *
 * const pool = new MaxPool2d(2); // 2x2 pooling
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-layers | Deepbox Layers}
 */
export class MaxPool2d extends Module {
  private readonly kernelSizeValue: [number, number];
  private readonly stride: [number, number];
  private readonly padding: [number, number];
  private readonly ceilMode: boolean;

  /**
   * @param kernelSize - Window size, one integer or `[kH, kW]`
   * @param options.stride - Step between windows (default: `kernelSize`)
   * @param options.padding - Implicit padding on each side (default 0)
   * @param options.ceilMode - Round the output size up instead of down (default false)
   * @throws {InvalidParameterError} If a parameter is invalid or padding exceeds half the kernel
   */
  constructor(
    kernelSize: number | [number, number],
    options: {
      readonly stride?: number | [number, number];
      readonly padding?: number | [number, number];
      readonly ceilMode?: boolean;
    } = {}
  ) {
    super();

    const kernelArr = normalizePair(
      "kernelSize",
      kernelSize,
      false,
      "a positive integer or a tuple of two positive integers"
    );
    this.kernelSizeValue = kernelArr;

    const strideArr = normalizePair(
      "stride",
      options.stride ?? kernelSize,
      false,
      "a positive integer or a tuple of two positive integers"
    );
    this.stride = strideArr;

    const paddingArr = normalizePair(
      "padding",
      options.padding ?? 0,
      true,
      "a non-negative integer or a tuple of two non-negative integers"
    );
    this.padding = paddingArr;
    checkPoolPadding("MaxPool2d", kernelArr, paddingArr);
    this.ceilMode = options.ceilMode ?? false;
  }

  /**
   * @param x - Input of shape `(batch, channels, height, width)`
   * @returns Output of shape `(batch, channels, outH, outW)`
   * @throws {ShapeError} If the input is not 4-D or the kernel does not fit
   * @throws {DTypeError} If the input has string dtype
   */
  forward(x: GradTensor): GradTensor;
  forward(x: Tensor): Tensor;
  forward(x: AnyTensor): AnyTensor;
  forward(x: AnyTensor): AnyTensor {
    return settle(this.run(x), allPlain(x));
  }

  private run(x: AnyTensor): GradTensor {
    const input = GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x);

    rejectString(input.dtype, "MaxPool2d");

    if (input.ndim !== 4) {
      throw new ShapeError(
        `MaxPool2d expects 4D input (batch, channels, height, width), got ${input.ndim}D`
      );
    }

    const batch = input.shape[0] ?? 0;
    const channels = input.shape[1] ?? 0;
    const inH = input.shape[2] ?? 0;
    const inW = input.shape[3] ?? 0;

    const [kH, kW] = this.kernelSizeValue;
    const [sH, sW] = this.stride;
    const [pH, pW] = this.padding;

    const outH = poolOutSize(inH, kH, sH, pH, this.ceilMode);
    const outW = poolOutSize(inW, kW, sW, pW, this.ceilMode);
    checkOutputSizes("MaxPool2d", [outH, outW]);

    if (input.tensor.isDeviceTensor) {
      if (this.ceilMode) rejectDeviceCeilMode("MaxPool2d");
      return devicePool2d(input, "max", [kH, kW], [sH, sW], [pH, pW]);
    }

    return pool3d(
      input,
      "max",
      batch * channels,
      [1, inH, inW],
      [UNIT_AXIS, slidingAxis(inH, outH, kH, sH, pH), slidingAxis(inW, outW, kW, sW, pW)],
      [batch, channels, outH, outW],
      null
    );
  }

  override toString(): string {
    return `MaxPool2d(kernel_size=${JSON.stringify(this.kernelSizeValue)}, stride=${JSON.stringify(this.stride)}, padding=${JSON.stringify(this.padding)}${this.ceilMode ? ", ceil_mode=true" : ""})`;
  }
}

/**
 * 2D Average Pooling Layer.
 *
 * Applies a 2D average pooling over an input of shape `(batch, channels, height, width)`.
 * By default padded zeros count towards the average (PyTorch's `count_include_pad=True`);
 * pass `countIncludePad: false` to divide by the number of real elements in each window.
 *
 * The stride defaults to the kernel size. As in PyTorch, `padding` may be at most
 * half of `kernelSize`.
 *
 * @example
 * ```ts
 * import { AvgPool2d } from 'deepbox/nn';
 *
 * const pool = new AvgPool2d(2); // 2x2 pooling
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-layers | Deepbox Layers}
 */
export class AvgPool2d extends Module {
  private readonly kernelSizeValue: [number, number];
  private readonly stride: [number, number];
  private readonly padding: [number, number];
  private readonly countIncludePad: boolean;
  private readonly ceilMode: boolean;

  /**
   * @param kernelSize - Window size, one integer or `[kH, kW]`
   * @param options.stride - Step between windows (default: `kernelSize`)
   * @param options.padding - Implicit zero padding on each side (default 0)
   * @param options.countIncludePad - Count padded zeros in the average (default true)
   * @param options.ceilMode - Round the output size up instead of down (default false). A
   *   partial last window divides by the part of it that lies inside the padded input.
   * @throws {InvalidParameterError} If a parameter is invalid or padding exceeds half the kernel
   */
  constructor(
    kernelSize: number | [number, number],
    options: {
      readonly stride?: number | [number, number];
      readonly padding?: number | [number, number];
      readonly countIncludePad?: boolean;
      readonly ceilMode?: boolean;
    } = {}
  ) {
    super();

    const kernelArr = normalizePair(
      "kernelSize",
      kernelSize,
      false,
      "a positive integer or a tuple of two positive integers"
    );
    this.kernelSizeValue = kernelArr;

    const strideArr = normalizePair(
      "stride",
      options.stride ?? kernelSize,
      false,
      "a positive integer or a tuple of two positive integers"
    );
    this.stride = strideArr;

    const paddingArr = normalizePair(
      "padding",
      options.padding ?? 0,
      true,
      "a non-negative integer or a tuple of two non-negative integers"
    );
    this.padding = paddingArr;
    checkPoolPadding("AvgPool2d", kernelArr, paddingArr);
    this.countIncludePad = options.countIncludePad ?? true;
    this.ceilMode = options.ceilMode ?? false;
  }

  /**
   * @param x - Input of shape `(batch, channels, height, width)`
   * @returns Output of shape `(batch, channels, outH, outW)`
   * @throws {ShapeError} If the input is not 4-D or the kernel does not fit
   * @throws {DTypeError} If the input has string dtype
   */
  forward(x: GradTensor): GradTensor;
  forward(x: Tensor): Tensor;
  forward(x: AnyTensor): AnyTensor;
  forward(x: AnyTensor): AnyTensor {
    return settle(this.run(x), allPlain(x));
  }

  private run(x: AnyTensor): GradTensor {
    const input = GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x);

    rejectString(input.dtype, "AvgPool2d");

    if (input.ndim !== 4) {
      throw new ShapeError(
        `AvgPool2d expects 4D input (batch, channels, height, width), got ${input.ndim}D`
      );
    }

    const batch = input.shape[0] ?? 0;
    const channels = input.shape[1] ?? 0;
    const inH = input.shape[2] ?? 0;
    const inW = input.shape[3] ?? 0;

    const [kH, kW] = this.kernelSizeValue;
    const [sH, sW] = this.stride;
    const [pH, pW] = this.padding;

    // Calculate output dims
    const outH = poolOutSize(inH, kH, sH, pH, this.ceilMode);
    const outW = poolOutSize(inW, kW, sW, pW, this.ceilMode);
    checkOutputSizes("AvgPool2d", [outH, outW]);

    if (this.ceilMode && this.countIncludePad) {
      return pool3d(
        input,
        "avg",
        batch * channels,
        [1, inH, inW],
        [UNIT_AXIS, slidingAxis(inH, outH, kH, sH, pH), slidingAxis(inW, outW, kW, sW, pW)],
        [batch, channels, outH, outW],
        null,
        [UNIT_EXTENT, paddedExtent(inH, outH, kH, sH, pH), paddedExtent(inW, outW, kW, sW, pW)]
      );
    }

    if (!this.countIncludePad) {
      return pool3d(
        input,
        "avg",
        batch * channels,
        [1, inH, inW],
        [UNIT_AXIS, slidingAxis(inH, outH, kH, sH, pH), slidingAxis(inW, outW, kW, sW, pW)],
        [batch, channels, outH, outW],
        null
      );
    }

    // Reshape: (B, C, H, W) -> (B*C, 1, H, W)
    const inputReshaped = input.reshape([batch * channels, 1, inH, inW]);

    // im2col -> (B*C, outPixels, 1 * kH * kW); padded positions are zero, so the mean over
    // the kernel axis divides by the full kernel size.
    const cols = im2colGrad(inputReshaped, [kH, kW], [sH, sW], [pH, pW]);

    // (B*C, outPixels, kH*kW) -> (B*C, outPixels)
    const meanVals = cols.mean(2);

    // Reshape back: (B*C, outH*outW) -> (B, C, outH, outW)
    return meanVals.reshape([batch, channels, outH, outW]);
  }

  override toString(): string {
    return `AvgPool2d(kernel_size=${JSON.stringify(this.kernelSizeValue)}, stride=${JSON.stringify(this.stride)}, padding=${JSON.stringify(this.padding)}${this.ceilMode ? ", ceil_mode=true" : ""})`;
  }
}

/**
 * 2D Transposed Convolution Layer (Deconvolution).
 *
 * Applies a transposed 2D convolution operator over an input image.
 * Used in image generation (GANs), semantic segmentation (U-Net),
 * autoencoders, and super-resolution networks.
 *
 * Output size:
 * ```
 * outH = (inH - 1) * strideH - 2 * padH + kernelH + outputPadH
 * outW = (inW - 1) * strideW - 2 * padW + kernelW + outputPadW
 * ```
 *
 * Weights have shape `(inChannels, outChannels, kH, kW)`. `outputPadding` must be
 * smaller than the stride; it only resolves the output-size ambiguity of strided
 * convolutions and adds no values of its own. As in PyTorch, weights and bias are drawn
 * from `U(-1/sqrt(fanIn), 1/sqrt(fanIn))` with `fanIn = outChannels * kH * kW`.
 *
 * The layer computes in its parameter dtype and casts the input to it. A `GradTensor` input
 * gives a `GradTensor`; a plain `Tensor` input gives a `GradTensor` while the weights
 * require grad and gradient tracking is on, and a plain `Tensor` otherwise.
 *
 * @example
 * ```ts
 * import { ConvTranspose2d } from 'deepbox/nn';
 *
 * const deconv = new ConvTranspose2d(16, 33, 3, { stride: 2, padding: 1 });
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-layers | Deepbox Layers}
 */
export class ConvTranspose2d extends Module {
  /** Hand-written host kernels: the weights stay in host memory under `to(device)`. */
  protected override keepsParametersOnHost(): boolean {
    return true;
  }

  private readonly inChannels: number;
  private readonly outChannels: number;
  private readonly kernelSize: [number, number];
  private readonly stride: [number, number];
  private readonly padding: [number, number];
  private readonly outputPadding: [number, number];
  private readonly useBias: boolean;

  private weight_: GradTensor;
  private bias_: GradTensor | undefined;

  /**
   * @param inChannels - Number of input channels
   * @param outChannels - Number of output channels
   * @param kernelSize - Kernel size, one integer or `[kH, kW]`
   * @param options.stride - Step of the equivalent forward convolution (default 1)
   * @param options.padding - Amount that is removed from each side of the output (default 0)
   * @param options.outputPadding - Extra size added to one side of the output (default 0)
   * @param options.bias - Add a learnable bias (default true)
   * @param options.dtype - Dtype of the parameters (default: the global default dtype)
   * @throws {InvalidParameterError} If a parameter is invalid or `outputPadding >= stride`
   */
  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number | [number, number],
    options: {
      readonly stride?: number | [number, number];
      readonly padding?: number | [number, number];
      readonly outputPadding?: number | [number, number];
      readonly bias?: boolean;
      readonly dtype?: LayerDType;
    } = {}
  ) {
    super();

    checkInteger("inChannels", inChannels, 1);
    checkInteger("outChannels", outChannels, 1);

    this.kernelSize = normalizePair(
      "kernelSize",
      kernelSize,
      false,
      "a positive integer or a tuple of two positive integers"
    );
    this.stride = normalizePair(
      "stride",
      options.stride ?? 1,
      false,
      "a positive integer or a tuple of two positive integers"
    );
    this.padding = normalizePair(
      "padding",
      options.padding ?? 0,
      true,
      "a non-negative integer or a tuple of two non-negative integers"
    );
    this.outputPadding = normalizePair(
      "outputPadding",
      options.outputPadding ?? 0,
      true,
      "a non-negative integer or a tuple of two non-negative integers"
    );
    checkOutputPadding(this.outputPadding, this.stride, options.outputPadding ?? 0);

    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.useBias = options.bias ?? true;

    // Weight shape: (inChannels, outChannels, kH, kW)
    const [kH, kW] = this.kernelSize;
    // PyTorch default: weight and bias ~ U(-1/sqrt(fanIn), 1/sqrt(fanIn)), where fanIn is
    // computed from the weight shape (inChannels, outChannels, kH, kW).
    const bound = 1 / Math.sqrt(outChannels * kH * kW);
    const opts = { dtype: resolveLayerDtype(options.dtype) };
    this.weight_ = parameter(uniformTensor([inChannels, outChannels, kH, kW], bound, opts));
    this.registerParameter("weight", this.weight_);

    if (this.useBias) {
      this.bias_ = parameter(uniformTensor([outChannels], bound, opts));
      this.registerParameter("bias", this.bias_);
    }
  }

  /**
   * @param x - Input of shape `(batch, inChannels, height, width)`
   * @returns Output of shape `(batch, outChannels, outH, outW)`
   * @throws {ShapeError} If the input is not 4-D, the channel count does not match, or the
   *   computed output size is not positive
   * @throws {DTypeError} If the input has string dtype
   */
  forward(x: GradTensor): GradTensor;
  forward(x: Tensor): AnyTensor;
  forward(x: AnyTensor): AnyTensor;
  forward(x: AnyTensor): AnyTensor {
    return settle(this.run(x), allPlain(x));
  }

  private run(x: AnyTensor): GradTensor {
    const input = asDtype(requireNumeric(toGradInput(x), "ConvTranspose2d"), this.weight_.dtype);

    if (input.ndim !== 4) {
      throw new ShapeError(
        `ConvTranspose2d expects 4D input (batch, channels, height, width), got ${input.ndim}D`
      );
    }

    const batch = input.shape[0] ?? 0;
    const inC = input.shape[1] ?? 0;
    const inH = input.shape[2] ?? 0;
    const inW = input.shape[3] ?? 0;

    if (inC !== this.inChannels) {
      throw new ShapeError(`Expected ${this.inChannels} input channels, got ${inC}`);
    }

    const [kH, kW] = this.kernelSize;
    const [sH, sW] = this.stride;
    const [pH, pW] = this.padding;
    const [opH, opW] = this.outputPadding;

    const outH = transposedOutSize(inH, kH, sH, pH, opH);
    const outW = transposedOutSize(inW, kW, sW, pW, opW);
    checkOutputSizes("ConvTranspose2d", [outH, outW]);

    // Transposed convolution: scatter input values through the kernel.
    // Materialize input and weight into contiguous float64 arrays so both the
    // forward scatter and the analytical backward index them uniformly.
    const inputTensor = input.tensor;
    const weightTensor = this.weight_.tensor;
    const biasTensor = this.useBias && this.bias_ ? this.bias_.tensor : null;
    const inFlat = denseFloat64(inputTensor);
    const wFlat = denseFloat64(weightTensor);
    const biasFlat = biasTensor ? denseFloat64(biasTensor) : null;

    const outC = this.outChannels;
    const inPlane = inH * inW;
    const outPlane = outH * outW;
    const kArea = kH * kW;
    const outArr = new Float64Array(batch * outC * outPlane);

    // Loop nest (b, ic, oc, ih, kh, iw, kw): for every output cell the contributions are
    // still added in (ic, ih, iw) order, so the result matches a plain scatter exactly.
    for (let b = 0; b < batch; b++) {
      for (let ic = 0; ic < inC; ic++) {
        const inBase = (b * inC + ic) * inPlane;
        for (let oc = 0; oc < outC; oc++) {
          const wBase = (ic * outC + oc) * kArea;
          const oBase = (b * outC + oc) * outPlane;
          for (let ih = 0; ih < inH; ih++) {
            for (let kh = 0; kh < kH; kh++) {
              const oh = ih * sH - pH + kh;
              if (oh < 0 || oh >= outH) continue;
              const oRow = oBase + oh * outW;
              const wRow = wBase + kh * kW;
              for (let iw = 0; iw < inW; iw++) {
                const inVal = inFlat[inBase + ih * inW + iw] as number;
                const owBase = iw * sW - pW;
                const kwLo = Math.max(0, -owBase);
                const kwHi = Math.min(kW, outW - owBase);
                for (let kw = kwLo; kw < kwHi; kw++) {
                  outArr[oRow + owBase + kw]! += inVal * (wFlat[wRow + kw] as number);
                }
              }
            }
          }
        }
      }
    }

    if (biasFlat) {
      for (let b = 0; b < batch; b++) {
        for (let oc = 0; oc < outC; oc++) {
          const bv = biasFlat[oc] as number;
          const oBase = (b * outC + oc) * outPlane;
          for (let i = 0; i < outPlane; i++) outArr[oBase + i]! += bv;
        }
      }
    }

    const device = inputTensor.device;
    const outDtype = promoteFloat(inputTensor.dtype, weightTensor.dtype);
    const outTensor = floatTensor(outArr, [batch, outC, outH, outW], device, outDtype);

    const grads: Array<[GradTensor, (g: Tensor) => Tensor]> = [];

    grads.push([
      input,
      (g: Tensor): Tensor => {
        const go = denseFloat64(g);
        const gi = new Float64Array(batch * inC * inPlane);
        for (let b = 0; b < batch; b++) {
          for (let ic = 0; ic < inC; ic++) {
            const inBase = (b * inC + ic) * inPlane;
            for (let ih = 0; ih < inH; ih++) {
              for (let iw = 0; iw < inW; iw++) {
                const owBase = iw * sW - pW;
                const kwLo = Math.max(0, -owBase);
                const kwHi = Math.min(kW, outW - owBase);
                let s = 0;
                for (let oc = 0; oc < outC; oc++) {
                  const wBase = (ic * outC + oc) * kArea;
                  const oBase = (b * outC + oc) * outPlane;
                  for (let kh = 0; kh < kH; kh++) {
                    const oh = ih * sH - pH + kh;
                    if (oh < 0 || oh >= outH) continue;
                    const oRow = oBase + oh * outW + owBase;
                    const wRow = wBase + kh * kW;
                    for (let kw = kwLo; kw < kwHi; kw++) {
                      s += (go[oRow + kw] as number) * (wFlat[wRow + kw] as number);
                    }
                  }
                }
                gi[inBase + ih * inW + iw] = s;
              }
            }
          }
        }
        return gradLike(floatTensor(gi, [batch, inC, inH, inW], device, "float64"), inputTensor);
      },
    ]);

    grads.push([
      this.weight_,
      (g: Tensor): Tensor => {
        const go = denseFloat64(g);
        const gw = new Float64Array(inC * outC * kArea);
        for (let b = 0; b < batch; b++) {
          for (let ic = 0; ic < inC; ic++) {
            const inBase = (b * inC + ic) * inPlane;
            for (let oc = 0; oc < outC; oc++) {
              const wBase = (ic * outC + oc) * kArea;
              const oBase = (b * outC + oc) * outPlane;
              for (let ih = 0; ih < inH; ih++) {
                for (let kh = 0; kh < kH; kh++) {
                  const oh = ih * sH - pH + kh;
                  if (oh < 0 || oh >= outH) continue;
                  const oRow = oBase + oh * outW;
                  const wRow = wBase + kh * kW;
                  for (let iw = 0; iw < inW; iw++) {
                    const inVal = inFlat[inBase + ih * inW + iw] as number;
                    const owBase = iw * sW - pW;
                    const kwLo = Math.max(0, -owBase);
                    const kwHi = Math.min(kW, outW - owBase);
                    for (let kw = kwLo; kw < kwHi; kw++) {
                      gw[wRow + kw]! += inVal * (go[oRow + owBase + kw] as number);
                    }
                  }
                }
              }
            }
          }
        }
        return gradLike(floatTensor(gw, [inC, outC, kH, kW], device, "float64"), weightTensor);
      },
    ]);

    if (biasTensor && this.bias_) {
      grads.push([
        this.bias_,
        (g: Tensor): Tensor => {
          const go = denseFloat64(g);
          const gb = new Float64Array(outC);
          for (let b = 0; b < batch; b++) {
            for (let oc = 0; oc < outC; oc++) {
              let s = 0;
              const oBase = (b * outC + oc) * outPlane;
              for (let i = 0; i < outPlane; i++) s += go[oBase + i] as number;
              gb[oc]! += s;
            }
          }
          return gradLike(floatTensor(gb, [outC], device, "float64"), biasTensor);
        },
      ]);
    }

    return customOp(outTensor, grads);
  }

  /** Learnable kernel of shape `(inChannels, outChannels, kH, kW)`. */
  get weight(): GradTensor {
    return this.weight_;
  }

  /** Learnable bias of shape `(outChannels)`, or `undefined` when the layer was built with `bias: false`. */
  get bias(): GradTensor | undefined {
    return this.bias_;
  }

  override toString(): string {
    return `ConvTranspose2d(${this.inChannels}, ${this.outChannels}, kernel_size=${JSON.stringify(this.kernelSize)}, stride=${JSON.stringify(this.stride)}, padding=${JSON.stringify(this.padding)}, output_padding=${JSON.stringify(this.outputPadding)})`;
  }
}

function normalizeAdaptiveSize2d(outputSize: number | [number, number]): [number, number] {
  if (typeof outputSize === "number") {
    if (!Number.isInteger(outputSize) || outputSize <= 0) {
      throw new InvalidParameterError(
        "outputSize must be a positive integer",
        "outputSize",
        outputSize
      );
    }
    return [outputSize, outputSize];
  }
  const [h, w] = outputSize;
  if (outputSize.length !== 2 || !Number.isInteger(h) || h <= 0 || !Number.isInteger(w) || w <= 0) {
    throw new InvalidParameterError(
      "outputSize must be positive integers",
      "outputSize",
      outputSize
    );
  }
  return [h, w];
}

function adaptivePool2d(
  input: GradTensor,
  outputSize: [number, number],
  mode: "max" | "avg",
  layer: string
): GradTensor {
  rejectString(input.dtype, layer);
  if (input.ndim !== 4) {
    throw new ShapeError(
      `${layer} expects 4D input (batch, channels, height, width), got ${input.ndim}D`
    );
  }

  const batch = input.shape[0] ?? 0;
  const channels = input.shape[1] ?? 0;
  const inH = input.shape[2] ?? 0;
  const inW = input.shape[3] ?? 0;
  const [outH, outW] = outputSize;

  if (inH === 0 || inW === 0) {
    throw new ShapeError(
      `${layer} needs a non-empty spatial size, got ${inH}x${inW}; an empty window has nothing to reduce`
    );
  }

  return pool3d(
    input,
    mode,
    batch * channels,
    [1, inH, inW],
    [UNIT_AXIS, adaptiveAxis(inH, outH), adaptiveAxis(inW, outW)],
    [batch, channels, outH, outW],
    null
  );
}

function adaptivePool1d(
  input: GradTensor,
  outL: number,
  mode: "max" | "avg",
  layer: string
): GradTensor {
  rejectString(input.dtype, layer);
  if (input.ndim !== 3) {
    throw new ShapeError(`${layer} expects 3D input (N, C, L); got ${input.ndim}D`);
  }

  const batch = input.shape[0] ?? 0;
  const channels = input.shape[1] ?? 0;
  const inL = input.shape[2] ?? 0;

  if (inL === 0) {
    throw new ShapeError(
      `${layer} needs a non-empty length, got 0; an empty window has nothing to reduce`
    );
  }

  return pool3d(
    input,
    mode,
    batch * channels,
    [1, 1, inL],
    [UNIT_AXIS, UNIT_AXIS, adaptiveAxis(inL, outL)],
    [batch, channels, outL],
    null
  );
}

/**
 * 2D Adaptive Average Pooling.
 *
 * Produces output of specified size regardless of input dimensions. Output cell `o`
 * averages the input range `[floor(o * in / out), ceil((o + 1) * in / out))`, so
 * neighbouring windows overlap when `in` is not a multiple of `out`. Typical use is a
 * global average pool (`[1, 1]`) before a classifier head, so that the head does not depend
 * on the input resolution.
 *
 * @example
 * ```ts
 * import { AdaptiveAvgPool2d } from 'deepbox/nn';
 *
 * const pool = new AdaptiveAvgPool2d([1, 1]); // Global average pooling
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-layers | Deepbox Layers}
 */
export class AdaptiveAvgPool2d extends Module {
  private readonly outputSize: [number, number];

  /**
   * @param outputSize - Output height and width, one integer or `[outH, outW]`
   * @throws {InvalidParameterError} If a size is not a positive integer
   */
  constructor(outputSize: number | [number, number]) {
    super();
    this.outputSize = normalizeAdaptiveSize2d(outputSize);
  }

  /**
   * @param x - Input of shape `(batch, channels, height, width)`
   * @returns Output of shape `(batch, channels, outH, outW)`
   * @throws {ShapeError} If the input is not 4-D
   * @throws {DTypeError} If the input has string dtype
   */
  forward(x: GradTensor): GradTensor;
  forward(x: Tensor): Tensor;
  forward(x: AnyTensor): AnyTensor;
  forward(x: AnyTensor): AnyTensor {
    return settle(this.run(x), allPlain(x));
  }

  private run(x: AnyTensor): GradTensor {
    const input = GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x);
    return adaptivePool2d(input, this.outputSize, "avg", "AdaptiveAvgPool2d");
  }

  override toString(): string {
    return `AdaptiveAvgPool2d(output_size=${JSON.stringify(this.outputSize)})`;
  }
}

/**
 * 2D Adaptive Max Pooling.
 *
 * Produces output of specified size regardless of input dimensions,
 * selecting the maximum value from each adaptive window (the windows are the same
 * as in {@link AdaptiveAvgPool2d}).
 *
 * @example
 * ```ts
 * import { AdaptiveMaxPool2d } from 'deepbox/nn';
 *
 * const pool = new AdaptiveMaxPool2d([1, 1]); // Global max pooling
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-layers | Deepbox Layers}
 */
export class AdaptiveMaxPool2d extends Module {
  private readonly outputSize: [number, number];

  /**
   * @param outputSize - Output height and width, one integer or `[outH, outW]`
   * @throws {InvalidParameterError} If a size is not a positive integer
   */
  constructor(outputSize: number | [number, number]) {
    super();
    this.outputSize = normalizeAdaptiveSize2d(outputSize);
  }

  /**
   * @param x - Input of shape `(batch, channels, height, width)`
   * @returns Output of shape `(batch, channels, outH, outW)`
   * @throws {ShapeError} If the input is not 4-D
   * @throws {DTypeError} If the input has string dtype
   */
  forward(x: GradTensor): GradTensor;
  forward(x: Tensor): Tensor;
  forward(x: AnyTensor): AnyTensor;
  forward(x: AnyTensor): AnyTensor {
    return settle(this.run(x), allPlain(x));
  }

  private run(x: AnyTensor): GradTensor {
    const input = GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x);
    return adaptivePool2d(input, this.outputSize, "max", "AdaptiveMaxPool2d");
  }

  override toString(): string {
    return `AdaptiveMaxPool2d(output_size=${JSON.stringify(this.outputSize)})`;
  }
}

/**
 * 1D Max Pooling.
 *
 * Applies max pooling over a 1D signal (e.g., sequences, audio).
 * Input: (N, C, L) -> Output: (N, C, L_out)
 *
 * The stride defaults to the kernel size. Padding is ignored by the maximum, a NaN
 * inside a window propagates, and `padding` may be at most half of `kernelSize`.
 *
 * @example
 * ```ts
 * import { MaxPool1d } from 'deepbox/nn';
 *
 * const pool = new MaxPool1d(2);
 * ```
 *
 * @category Neural Network Layers
 * @see {@link https://deepbox.dev/docs/nn-layers | Deepbox Layers}
 */
export class MaxPool1d extends Module {
  private readonly kernelSize: number;
  private readonly stride: number;
  private readonly padding: number;
  private readonly ceilMode: boolean;

  /**
   * @param kernelSize - Window size
   * @param options.stride - Step between windows (default: `kernelSize`)
   * @param options.padding - Implicit padding on both ends (default 0)
   * @param options.ceilMode - Round the output length up instead of down (default false)
   * @throws {InvalidParameterError} If a parameter is invalid or padding exceeds half the kernel
   */
  constructor(
    kernelSize: number,
    options: {
      readonly stride?: number;
      readonly padding?: number;
      readonly ceilMode?: boolean;
    } = {}
  ) {
    super();
    checkInteger("kernelSize", kernelSize, 1);
    this.kernelSize = kernelSize;
    this.stride = checkInteger("stride", options.stride ?? kernelSize, 1);
    this.padding = checkInteger("padding", options.padding ?? 0, 0);
    checkPoolPadding("MaxPool1d", [kernelSize], [this.padding]);
    this.ceilMode = options.ceilMode ?? false;
  }

  /**
   * @param input - Input of shape `(N, C, L)`
   * @returns Output of shape `(N, C, outL)`
   * @throws {ShapeError} If the input is not 3-D or the kernel does not fit
   * @throws {DTypeError} If the input has string dtype
   */
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: AnyTensor): AnyTensor {
    return settle(this.run(input), allPlain(input));
  }

  private run(input: AnyTensor): GradTensor {
    const t = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
    const inputTensor = t.tensor;

    rejectString(inputTensor.dtype, "MaxPool1d");
    if (inputTensor.ndim !== 3) {
      throw new ShapeError(`MaxPool1d expects 3D input (N, C, L); got ${inputTensor.ndim}D`);
    }

    const batch = inputTensor.shape[0] ?? 0;
    const channels = inputTensor.shape[1] ?? 0;
    const length = inputTensor.shape[2] ?? 0;
    const outL = poolOutSize(length, this.kernelSize, this.stride, this.padding, this.ceilMode);

    if (outL <= 0) {
      throw new ShapeError("MaxPool1d output length must be positive");
    }

    if (inputTensor.isDeviceTensor) {
      if (this.ceilMode) rejectDeviceCeilMode("MaxPool1d");
      // 1-D pooling is 2-D pooling over a height-1 image: reshape [N,C,L] ->
      // [N,C,1,L], pool with a 1×K window, reshape back. Autograd-aware reshapes.
      const pooled = devicePool2d(
        t.reshape([batch, channels, 1, length]),
        "max",
        [1, this.kernelSize],
        [1, this.stride],
        [0, this.padding]
      );
      return pooled.reshape([batch, channels, outL]);
    }

    return pool3d(
      t,
      "max",
      batch * channels,
      [1, 1, length],
      [UNIT_AXIS, UNIT_AXIS, slidingAxis(length, outL, this.kernelSize, this.stride, this.padding)],
      [batch, channels, outL],
      null
    );
  }

  override toString(): string {
    return `MaxPool1d(kernel_size=${this.kernelSize}, stride=${this.stride}, padding=${this.padding}${this.ceilMode ? ", ceil_mode=true" : ""})`;
  }
}

/**
 * 1D Average Pooling.
 *
 * Applies average pooling over a 1D signal.
 * Input: (N, C, L) -> Output: (N, C, L_out)
 *
 * By default padded zeros count towards the average (PyTorch's `count_include_pad=True`);
 * pass `countIncludePad: false` to divide by the number of real elements in each window.
 * The stride defaults to the kernel size and `padding` may be at most half of `kernelSize`.
 *
 * @example
 * ```ts
 * import { AvgPool1d } from 'deepbox/nn';
 *
 * const pool = new AvgPool1d(2);
 * ```
 *
 * @category Neural Network Layers
 * @see {@link https://deepbox.dev/docs/nn-layers | Deepbox Layers}
 */
export class AvgPool1d extends Module {
  private readonly kernelSize: number;
  private readonly stride: number;
  private readonly padding: number;
  private readonly countIncludePad: boolean;
  private readonly ceilMode: boolean;

  /**
   * @param kernelSize - Window size
   * @param options.stride - Step between windows (default: `kernelSize`)
   * @param options.padding - Implicit zero padding on both ends (default 0)
   * @param options.countIncludePad - Count padded zeros in the average (default true)
   * @param options.ceilMode - Round the output length up instead of down (default false). A
   *   partial last window divides by the part of it that lies inside the padded input.
   * @throws {InvalidParameterError} If a parameter is invalid or padding exceeds half the kernel
   */
  constructor(
    kernelSize: number,
    options: {
      readonly stride?: number;
      readonly padding?: number;
      readonly countIncludePad?: boolean;
      readonly ceilMode?: boolean;
    } = {}
  ) {
    super();
    checkInteger("kernelSize", kernelSize, 1);
    this.kernelSize = kernelSize;
    this.stride = checkInteger("stride", options.stride ?? kernelSize, 1);
    this.padding = checkInteger("padding", options.padding ?? 0, 0);
    checkPoolPadding("AvgPool1d", [kernelSize], [this.padding]);
    this.countIncludePad = options.countIncludePad ?? true;
    this.ceilMode = options.ceilMode ?? false;
  }

  /**
   * @param input - Input of shape `(N, C, L)`
   * @returns Output of shape `(N, C, outL)`
   * @throws {ShapeError} If the input is not 3-D or the kernel does not fit
   * @throws {DTypeError} If the input has string dtype
   */
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: AnyTensor): AnyTensor {
    return settle(this.run(input), allPlain(input));
  }

  private run(input: AnyTensor): GradTensor {
    const t = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
    const inputTensor = t.tensor;

    rejectString(inputTensor.dtype, "AvgPool1d");
    if (inputTensor.ndim !== 3) {
      throw new ShapeError(`AvgPool1d expects 3D input (N, C, L); got ${inputTensor.ndim}D`);
    }

    const batch = inputTensor.shape[0] ?? 0;
    const channels = inputTensor.shape[1] ?? 0;
    const length = inputTensor.shape[2] ?? 0;
    const outL = poolOutSize(length, this.kernelSize, this.stride, this.padding, this.ceilMode);

    if (outL <= 0) {
      throw new ShapeError("AvgPool1d output length must be positive");
    }

    return pool3d(
      t,
      "avg",
      batch * channels,
      [1, 1, length],
      [UNIT_AXIS, UNIT_AXIS, slidingAxis(length, outL, this.kernelSize, this.stride, this.padding)],
      [batch, channels, outL],
      this.countIncludePad && !this.ceilMode ? this.kernelSize : null,
      this.countIncludePad && this.ceilMode
        ? [
            UNIT_EXTENT,
            UNIT_EXTENT,
            paddedExtent(length, outL, this.kernelSize, this.stride, this.padding),
          ]
        : undefined
    );
  }

  override toString(): string {
    return `AvgPool1d(kernel_size=${this.kernelSize}, stride=${this.stride}, padding=${this.padding}${this.ceilMode ? ", ceil_mode=true" : ""})`;
  }
}

/**
 * 1D Adaptive Average Pooling.
 *
 * Produces fixed output length regardless of input size. Output cell `o` averages the
 * input range `[floor(o * L / out), ceil((o + 1) * L / out))`.
 * Input: (N, C, L) -> Output: (N, C, outputSize)
 *
 * @example
 * ```ts
 * import { AdaptiveAvgPool1d } from 'deepbox/nn';
 *
 * const pool = new AdaptiveAvgPool1d(1); // global average over the length axis
 * ```
 *
 * @category Neural Network Layers
 * @see {@link https://deepbox.dev/docs/nn-layers | Deepbox Layers}
 */
export class AdaptiveAvgPool1d extends Module {
  private readonly outputSize: number;

  /**
   * @param outputSize - Output length
   * @throws {InvalidParameterError} If `outputSize` is not a positive integer
   */
  constructor(outputSize: number) {
    super();
    checkInteger("outputSize", outputSize, 1);
    this.outputSize = outputSize;
  }

  /**
   * @param input - Input of shape `(N, C, L)`
   * @returns Output of shape `(N, C, outputSize)`
   * @throws {ShapeError} If the input is not 3-D
   * @throws {DTypeError} If the input has string dtype
   */
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: AnyTensor): AnyTensor {
    return settle(this.run(input), allPlain(input));
  }

  private run(input: AnyTensor): GradTensor {
    const t = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
    return adaptivePool1d(t, this.outputSize, "avg", "AdaptiveAvgPool1d");
  }

  override toString(): string {
    return `AdaptiveAvgPool1d(output_size=${this.outputSize})`;
  }
}

/**
 * 1D Adaptive Max Pooling.
 *
 * Produces fixed output length regardless of input size, taking the maximum of each
 * adaptive window (the windows are the same as in {@link AdaptiveAvgPool1d}).
 * Input: (N, C, L) -> Output: (N, C, outputSize)
 *
 * @example
 * ```ts
 * import { AdaptiveMaxPool1d } from 'deepbox/nn';
 *
 * const pool = new AdaptiveMaxPool1d(1); // global max over the length axis
 * ```
 *
 * @category Neural Network Layers
 * @see {@link https://deepbox.dev/docs/nn-layers | Deepbox Layers}
 */
export class AdaptiveMaxPool1d extends Module {
  private readonly outputSize: number;

  /**
   * @param outputSize - Output length
   * @throws {InvalidParameterError} If `outputSize` is not a positive integer
   */
  constructor(outputSize: number) {
    super();
    checkInteger("outputSize", outputSize, 1);
    this.outputSize = outputSize;
  }

  /**
   * @param input - Input of shape `(N, C, L)`
   * @returns Output of shape `(N, C, outputSize)`
   * @throws {ShapeError} If the input is not 3-D
   * @throws {DTypeError} If the input has string dtype
   */
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: AnyTensor): AnyTensor {
    return settle(this.run(input), allPlain(input));
  }

  private run(input: AnyTensor): GradTensor {
    const t = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
    return adaptivePool1d(t, this.outputSize, "max", "AdaptiveMaxPool1d");
  }

  override toString(): string {
    return `AdaptiveMaxPool1d(output_size=${this.outputSize})`;
  }
}

/**
 * 1D Transposed Convolution Layer (Deconvolution).
 *
 * Applies a transposed 1D convolution over an input signal.
 * Used for upsampling in sequence generation, audio synthesis, etc.
 *
 * Output size: `outL = (inL - 1) * stride - 2 * padding + kernelSize + outputPadding`
 *
 * Weights have shape `(inChannels, outChannels, kernelSize)`. `outputPadding` must be
 * smaller than the stride. As in PyTorch, weights and bias are drawn from
 * `U(-1/sqrt(fanIn), 1/sqrt(fanIn))` with `fanIn = outChannels * kernelSize`.
 *
 * The layer computes in its parameter dtype and casts the input to it. A `GradTensor` input
 * gives a `GradTensor`; a plain `Tensor` input gives a `GradTensor` while the weights
 * require grad and gradient tracking is on, and a plain `Tensor` otherwise.
 *
 * @example
 * ```ts
 * import { ConvTranspose1d } from 'deepbox/nn';
 *
 * const deconv = new ConvTranspose1d(16, 33, 3, { stride: 2 });
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-layers | Deepbox Layers}
 */
export class ConvTranspose1d extends Module {
  /** Hand-written host kernels: the weights stay in host memory under `to(device)`. */
  protected override keepsParametersOnHost(): boolean {
    return true;
  }

  private readonly inChannels: number;
  private readonly outChannels: number;
  private readonly kernelSize: number;
  private readonly stride: number;
  private readonly padding: number;
  private readonly outputPadding: number;
  private readonly useBias: boolean;

  private weight_: GradTensor;
  private bias_: GradTensor | undefined;

  /**
   * @param inChannels - Number of input channels
   * @param outChannels - Number of output channels
   * @param kernelSize - Size of the kernel
   * @param options.stride - Step of the equivalent forward convolution (default 1)
   * @param options.padding - Amount that is removed from each end of the output (default 0)
   * @param options.outputPadding - Extra length added to one end of the output (default 0)
   * @param options.bias - Add a learnable bias (default true)
   * @param options.dtype - Dtype of the parameters (default: the global default dtype)
   * @throws {InvalidParameterError} If a parameter is invalid or `outputPadding >= stride`
   */
  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number,
    options: {
      readonly stride?: number;
      readonly padding?: number;
      readonly outputPadding?: number;
      readonly bias?: boolean;
      readonly dtype?: LayerDType;
    } = {}
  ) {
    super();

    checkInteger("inChannels", inChannels, 1);
    checkInteger("outChannels", outChannels, 1);
    checkInteger("kernelSize", kernelSize, 1);
    const stride = checkInteger("stride", options.stride ?? 1, 1);
    const padding = checkInteger("padding", options.padding ?? 0, 0);
    const outputPadding = checkInteger("outputPadding", options.outputPadding ?? 0, 0);
    checkOutputPadding([outputPadding], [stride], outputPadding);

    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = kernelSize;
    this.stride = stride;
    this.padding = padding;
    this.outputPadding = outputPadding;
    this.useBias = options.bias ?? true;

    // Weight shape: (inChannels, outChannels, kernelSize)
    // PyTorch default: weight and bias ~ U(-1/sqrt(fanIn), 1/sqrt(fanIn)).
    const bound = 1 / Math.sqrt(outChannels * kernelSize);
    const opts = { dtype: resolveLayerDtype(options.dtype) };
    this.weight_ = parameter(uniformTensor([inChannels, outChannels, kernelSize], bound, opts));
    this.registerParameter("weight", this.weight_);

    if (this.useBias) {
      this.bias_ = parameter(uniformTensor([outChannels], bound, opts));
      this.registerParameter("bias", this.bias_);
    }
  }

  /**
   * @param x - Input of shape `(batch, inChannels, length)`
   * @returns Output of shape `(batch, outChannels, outLength)`
   * @throws {ShapeError} If the input is not 3-D, the channel count does not match, or the
   *   computed output length is not positive
   * @throws {DTypeError} If the input has string dtype
   */
  forward(x: GradTensor): GradTensor;
  forward(x: Tensor): AnyTensor;
  forward(x: AnyTensor): AnyTensor;
  forward(x: AnyTensor): AnyTensor {
    return settle(this.run(x), allPlain(x));
  }

  private run(x: AnyTensor): GradTensor {
    const input = asDtype(requireNumeric(toGradInput(x), "ConvTranspose1d"), this.weight_.dtype);

    if (input.ndim !== 3) {
      throw new ShapeError(
        `ConvTranspose1d expects 3D input (batch, channels, length), got ${input.ndim}D`
      );
    }

    const batch = input.shape[0] ?? 0;
    const inC = input.shape[1] ?? 0;
    const inL = input.shape[2] ?? 0;

    if (inC !== this.inChannels) {
      throw new ShapeError(`Expected ${this.inChannels} input channels, got ${inC}`);
    }

    const kS = this.kernelSize;
    const sS = this.stride;
    const pS = this.padding;

    const outL = transposedOutSize(inL, kS, sS, pS, this.outputPadding);
    checkOutputSizes("ConvTranspose1d", [outL]);

    const inputTensor = input.tensor;
    const weightTensor = this.weight_.tensor;
    const biasTensor = this.useBias && this.bias_ ? this.bias_.tensor : null;
    const inFlat = denseFloat64(inputTensor);
    const wFlat = denseFloat64(weightTensor);
    const biasFlat = biasTensor ? denseFloat64(biasTensor) : null;
    const outC = this.outChannels;

    const outArr = new Float64Array(batch * outC * outL);

    // Output cells receive their contributions in (ic, il) order, as in a plain scatter.
    for (let b = 0; b < batch; b++) {
      for (let ic = 0; ic < inC; ic++) {
        const inBase = (b * inC + ic) * inL;
        for (let oc = 0; oc < outC; oc++) {
          const wBase = (ic * outC + oc) * kS;
          const oBase = (b * outC + oc) * outL;
          for (let il = 0; il < inL; il++) {
            const inVal = inFlat[inBase + il] as number;
            const olBase = il * sS - pS;
            const kLo = Math.max(0, -olBase);
            const kHi = Math.min(kS, outL - olBase);
            for (let k = kLo; k < kHi; k++) {
              outArr[oBase + olBase + k]! += inVal * (wFlat[wBase + k] as number);
            }
          }
        }
      }
    }

    if (biasFlat) {
      for (let b = 0; b < batch; b++) {
        for (let oc = 0; oc < outC; oc++) {
          const bv = biasFlat[oc] as number;
          const oBase = (b * outC + oc) * outL;
          for (let ol = 0; ol < outL; ol++) outArr[oBase + ol]! += bv;
        }
      }
    }

    const device = inputTensor.device;
    const outDtype = promoteFloat(inputTensor.dtype, weightTensor.dtype);
    const outTensor = floatTensor(outArr, [batch, outC, outL], device, outDtype);

    const grads: Array<[GradTensor, (g: Tensor) => Tensor]> = [
      [
        input,
        (g: Tensor): Tensor => {
          const go = denseFloat64(g);
          const gi = new Float64Array(batch * inC * inL);
          for (let b = 0; b < batch; b++) {
            for (let ic = 0; ic < inC; ic++) {
              for (let il = 0; il < inL; il++) {
                const olBase = il * sS - pS;
                const kLo = Math.max(0, -olBase);
                const kHi = Math.min(kS, outL - olBase);
                let s = 0;
                for (let oc = 0; oc < outC; oc++) {
                  const wBase = (ic * outC + oc) * kS;
                  const oBase = (b * outC + oc) * outL + olBase;
                  for (let k = kLo; k < kHi; k++) {
                    s += (go[oBase + k] as number) * (wFlat[wBase + k] as number);
                  }
                }
                gi[(b * inC + ic) * inL + il] = s;
              }
            }
          }
          return gradLike(floatTensor(gi, [batch, inC, inL], device, "float64"), inputTensor);
        },
      ],
      [
        this.weight_,
        (g: Tensor): Tensor => {
          const go = denseFloat64(g);
          const gw = new Float64Array(inC * outC * kS);
          for (let b = 0; b < batch; b++) {
            for (let ic = 0; ic < inC; ic++) {
              const inBase = (b * inC + ic) * inL;
              for (let oc = 0; oc < outC; oc++) {
                const wBase = (ic * outC + oc) * kS;
                const oBase = (b * outC + oc) * outL;
                for (let il = 0; il < inL; il++) {
                  const inVal = inFlat[inBase + il] as number;
                  const olBase = il * sS - pS;
                  const kLo = Math.max(0, -olBase);
                  const kHi = Math.min(kS, outL - olBase);
                  for (let k = kLo; k < kHi; k++) {
                    gw[wBase + k]! += inVal * (go[oBase + olBase + k] as number);
                  }
                }
              }
            }
          }
          return gradLike(floatTensor(gw, [inC, outC, kS], device, "float64"), weightTensor);
        },
      ],
    ];

    if (biasTensor && this.bias_) {
      grads.push([
        this.bias_,
        (g: Tensor): Tensor => {
          const go = denseFloat64(g);
          const gb = new Float64Array(outC);
          for (let b = 0; b < batch; b++) {
            for (let oc = 0; oc < outC; oc++) {
              let s = 0;
              const oBase = (b * outC + oc) * outL;
              for (let ol = 0; ol < outL; ol++) s += go[oBase + ol] as number;
              gb[oc]! += s;
            }
          }
          return gradLike(floatTensor(gb, [outC], device, "float64"), biasTensor);
        },
      ]);
    }

    return customOp(outTensor, grads);
  }

  /** Learnable kernel of shape `(inChannels, outChannels, kernelSize)`. */
  get weight(): GradTensor {
    return this.weight_;
  }

  /** Learnable bias of shape `(outChannels)`, or `undefined` when the layer was built with `bias: false`. */
  get bias(): GradTensor | undefined {
    return this.bias_;
  }

  override toString(): string {
    return `ConvTranspose1d(${this.inChannels}, ${this.outChannels}, kernel_size=${this.kernelSize}, stride=${this.stride}, padding=${this.padding}, output_padding=${this.outputPadding})`;
  }
}

/**
 * 3D Convolutional Layer.
 *
 * Applies a 3D cross-correlation over an input signal composed of several input planes.
 * Used for video processing, medical imaging (CT/MRI), and 3D point clouds.
 *
 * Input: (N, C_in, D, H, W) -> Output: (N, C_out, D_out, H_out, W_out), where each output
 * size is `floor((in + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1`.
 *
 * Weights have shape `(outChannels, inChannels / groups, kD, kH, kW)`. As in PyTorch, weights
 * and bias are drawn from `U(-1/sqrt(fanIn), 1/sqrt(fanIn))` with
 * `fanIn = inChannels / groups * kD * kH * kW`.
 *
 * `dilation` spaces out the kernel taps, `groups` splits the channels into independent
 * groups, and `padding` accepts `"valid"` (no padding) or `"same"` (output size equals input
 * size; stride 1 only).
 *
 * The layer computes in its parameter dtype and casts the input to it. A `GradTensor` input
 * gives a `GradTensor`; a plain `Tensor` input gives a `GradTensor` while the weights
 * require grad and gradient tracking is on, and a plain `Tensor` otherwise.
 *
 * @example
 * ```ts
 * import { Conv3d } from 'deepbox/nn';
 *
 * const conv = new Conv3d(1, 16, 3); // 1 input channel, 16 output channels, 3x3x3 kernel
 * const same = new Conv3d(4, 8, 3, { padding: 'same', dilation: 2, groups: 2 });
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-layers | Deepbox Layers}
 */
export class Conv3d extends Module {
  /** Hand-written host kernels: the weights stay in host memory under `to(device)`. */
  protected override keepsParametersOnHost(): boolean {
    return true;
  }

  private readonly inChannels: number;
  private readonly outChannels: number;
  private readonly kernelSize: [number, number, number];
  private readonly stride: [number, number, number];
  private readonly dilation: [number, number, number];
  private readonly groups: number;
  private readonly padBefore: [number, number, number];
  private readonly padAfter: [number, number, number];
  private readonly paddingLabel: string;
  private readonly useBias: boolean;

  private weight_: GradTensor;
  private bias_: GradTensor | undefined;

  /**
   * @param inChannels - Number of input channels
   * @param outChannels - Number of output channels
   * @param kernelSize - Kernel size, one integer or `[kD, kH, kW]`
   * @param options.stride - Step between windows (default 1)
   * @param options.padding - Zeros added to each side of the input (default 0), or `"valid"`
   *   (no padding) or `"same"` (keep the size; needs `stride` 1)
   * @param options.dilation - Spacing between kernel taps, one integer or `[dD, dH, dW]` (default 1)
   * @param options.groups - Number of channel groups (default 1); must divide `inChannels` and
   *   `outChannels`
   * @param options.bias - Add a learnable bias (default true)
   * @param options.dtype - Dtype of the parameters (default: the global default dtype)
   * @throws {InvalidParameterError} If a size, stride, dilation, groups or padding is not valid
   */
  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number | [number, number, number],
    options: {
      readonly stride?: number | [number, number, number];
      readonly padding?: ConvPadding<number | [number, number, number]>;
      readonly dilation?: number | [number, number, number];
      readonly groups?: number;
      readonly bias?: boolean;
      readonly dtype?: LayerDType;
    } = {}
  ) {
    super();
    checkInteger("inChannels", inChannels, 1);
    checkInteger("outChannels", outChannels, 1);

    this.kernelSize = normalizeTriple(
      "kernelSize",
      kernelSize,
      false,
      "a positive integer or a triple of positive integers"
    );
    this.stride = normalizeTriple(
      "stride",
      options.stride ?? 1,
      false,
      "a positive integer or a triple of positive integers"
    );
    this.dilation = normalizeTriple(
      "dilation",
      options.dilation ?? 1,
      false,
      "a positive integer or a triple of positive integers"
    );
    this.groups = checkGroups(inChannels, outChannels, options.groups ?? 1);
    const padOption = options.padding ?? 0;
    const resolved = resolveConvPadding(
      "Conv3d",
      padOption,
      (value: number | [number, number, number]) =>
        normalizeTriple(
          "padding",
          value,
          true,
          "a non-negative integer or a triple of non-negative integers"
        ),
      this.kernelSize,
      this.stride,
      this.dilation
    );
    this.padBefore = [resolved.before[0] ?? 0, resolved.before[1] ?? 0, resolved.before[2] ?? 0];
    this.padAfter = [resolved.after[0] ?? 0, resolved.after[1] ?? 0, resolved.after[2] ?? 0];
    this.paddingLabel =
      typeof padOption === "string" ? `"${padOption}"` : JSON.stringify(this.padBefore);

    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.useBias = options.bias ?? true;

    const [kD, kH, kW] = this.kernelSize;
    // PyTorch default: weight and bias ~ U(-1/sqrt(fanIn), 1/sqrt(fanIn)).
    const groupIn = inChannels / this.groups;
    const bound = 1 / Math.sqrt(groupIn * kD * kH * kW);
    const opts = { dtype: resolveLayerDtype(options.dtype) };
    this.weight_ = parameter(uniformTensor([outChannels, groupIn, kD, kH, kW], bound, opts));
    this.registerParameter("weight", this.weight_);

    if (this.useBias) {
      this.bias_ = parameter(uniformTensor([outChannels], bound, opts));
      this.registerParameter("bias", this.bias_);
    }
  }

  /**
   * @param x - Input of shape `(batch, inChannels, depth, height, width)`
   * @returns Output of shape `(batch, outChannels, outD, outH, outW)`
   * @throws {ShapeError} If the input is not 5-D, the channel count does not match, or the
   *   kernel does not fit the input
   * @throws {DTypeError} If the input has string dtype
   */
  forward(x: GradTensor): GradTensor;
  forward(x: Tensor): AnyTensor;
  forward(x: AnyTensor): AnyTensor;
  forward(x: AnyTensor): AnyTensor {
    return settle(this.run(x), allPlain(x));
  }

  private run(x: AnyTensor): GradTensor {
    const input = asDtype(requireNumeric(toGradInput(x), "Conv3d"), this.weight_.dtype);

    if (input.ndim !== 5) {
      throw new ShapeError(
        `Conv3d expects 5D input (batch, channels, depth, height, width), got ${input.ndim}D`
      );
    }

    const batch = input.shape[0] ?? 0;
    const inC = input.shape[1] ?? 0;
    const inD = input.shape[2] ?? 0;
    const inH = input.shape[3] ?? 0;
    const inW = input.shape[4] ?? 0;

    if (inC !== this.inChannels) {
      throw new ShapeError(`Expected ${this.inChannels} input channels, got ${inC}`);
    }

    const [kD, kH, kW] = this.kernelSize;
    const [sD, sH, sW] = this.stride;
    const [dD, dH, dW] = this.dilation;
    const [pD, pH, pW] = this.padBefore;
    const [aD, aH, aW] = this.padAfter;
    const groups = this.groups;
    const groupIn = inC / groups;
    const groupOut = this.outChannels / groups;

    const outD = convOutSize(inD, kD, sD, dD, pD, aD);
    const outH = convOutSize(inH, kH, sH, dH, pH, aH);
    const outW = convOutSize(inW, kW, sW, dW, pW, aW);
    checkOutputSizes("Conv3d", [outD, outH, outW]);

    const inputTensor = input.tensor;
    const weightTensor = this.weight_.tensor;
    const biasTensor = this.useBias && this.bias_ ? this.bias_.tensor : null;
    const inFlat = denseFloat64(inputTensor);
    const wFlat = denseFloat64(weightTensor);
    const biasFlat = biasTensor ? denseFloat64(biasTensor) : null;
    const outC = this.outChannels;

    const inHW = inH * inW;
    const inVol = inD * inHW;
    const kHW = kH * kW;
    const kVol = kD * kHW;
    const outHW = outH * outW;
    const outVol = outD * outHW;
    const outArr = new Float64Array(batch * outC * outVol);

    // Taps that fall outside the input are skipped by narrowing the kd/kh/kw ranges;
    // the remaining products are summed in the same (ic, kd, kh, kw) order.
    for (let b = 0; b < batch; b++) {
      for (let oc = 0; oc < outC; oc++) {
        let o = (b * outC + oc) * outVol;
        for (let od = 0; od < outD; od++) {
          const [kdLo, kdHi] = dilatedTapRange(od, sD, pD, dD, kD, inD);
          for (let oh = 0; oh < outH; oh++) {
            const [khLo, khHi] = dilatedTapRange(oh, sH, pH, dH, kH, inH);
            for (let ow = 0; ow < outW; ow++, o++) {
              const iw0 = ow * sW - pW;
              const [kwLo, kwHi] = dilatedTapRange(ow, sW, pW, dW, kW, inW);
              let sum = biasFlat ? (biasFlat[oc] as number) : 0;
              const icStart = Math.floor(oc / groupOut) * groupIn;
              for (let c = 0; c < groupIn; c++) {
                const inBase = (b * inC + icStart + c) * inVol;
                const wBase = (oc * groupIn + c) * kVol;
                for (let kd = kdLo; kd < kdHi; kd++) {
                  const id = od * sD + kd * dD - pD;
                  for (let kh = khLo; kh < khHi; kh++) {
                    const ih = oh * sH + kh * dH - pH;
                    const inRow = inBase + id * inHW + ih * inW + iw0;
                    const wRow = wBase + kd * kHW + kh * kW;
                    for (let kw = kwLo; kw < kwHi; kw++) {
                      sum += (inFlat[inRow + kw * dW] as number) * (wFlat[wRow + kw] as number);
                    }
                  }
                }
              }
              outArr[o] = sum;
            }
          }
        }
      }
    }

    const device = inputTensor.device;
    const outDtype = promoteFloat(inputTensor.dtype, weightTensor.dtype);
    const outTensor = floatTensor(outArr, [batch, outC, outD, outH, outW], device, outDtype);

    const grads: Array<[GradTensor, (g: Tensor) => Tensor]> = [
      [
        input,
        (g: Tensor): Tensor => {
          const go = denseFloat64(g);
          const gi = new Float64Array(batch * inC * inVol);
          for (let b = 0; b < batch; b++) {
            for (let oc = 0; oc < outC; oc++) {
              let o = (b * outC + oc) * outVol;
              for (let od = 0; od < outD; od++) {
                const [kdLo, kdHi] = dilatedTapRange(od, sD, pD, dD, kD, inD);
                for (let oh = 0; oh < outH; oh++) {
                  const [khLo, khHi] = dilatedTapRange(oh, sH, pH, dH, kH, inH);
                  for (let ow = 0; ow < outW; ow++, o++) {
                    const gv = go[o] as number;
                    if (gv === 0) continue;
                    const iw0 = ow * sW - pW;
                    const [kwLo, kwHi] = dilatedTapRange(ow, sW, pW, dW, kW, inW);
                    const icStart = Math.floor(oc / groupOut) * groupIn;
                    for (let c = 0; c < groupIn; c++) {
                      const inBase = (b * inC + icStart + c) * inVol;
                      const wBase = (oc * groupIn + c) * kVol;
                      for (let kd = kdLo; kd < kdHi; kd++) {
                        const id = od * sD + kd * dD - pD;
                        for (let kh = khLo; kh < khHi; kh++) {
                          const ih = oh * sH + kh * dH - pH;
                          const inRow = inBase + id * inHW + ih * inW + iw0;
                          const wRow = wBase + kd * kHW + kh * kW;
                          for (let kw = kwLo; kw < kwHi; kw++) {
                            gi[inRow + kw * dW]! += gv * (wFlat[wRow + kw] as number);
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          return gradLike(
            floatTensor(gi, [batch, inC, inD, inH, inW], device, "float64"),
            inputTensor
          );
        },
      ],
      [
        this.weight_,
        (g: Tensor): Tensor => {
          const go = denseFloat64(g);
          const gw = new Float64Array(outC * groupIn * kVol);
          for (let b = 0; b < batch; b++) {
            for (let oc = 0; oc < outC; oc++) {
              let o = (b * outC + oc) * outVol;
              for (let od = 0; od < outD; od++) {
                const [kdLo, kdHi] = dilatedTapRange(od, sD, pD, dD, kD, inD);
                for (let oh = 0; oh < outH; oh++) {
                  const [khLo, khHi] = dilatedTapRange(oh, sH, pH, dH, kH, inH);
                  for (let ow = 0; ow < outW; ow++, o++) {
                    const gv = go[o] as number;
                    if (gv === 0) continue;
                    const iw0 = ow * sW - pW;
                    const [kwLo, kwHi] = dilatedTapRange(ow, sW, pW, dW, kW, inW);
                    const icStart = Math.floor(oc / groupOut) * groupIn;
                    for (let c = 0; c < groupIn; c++) {
                      const inBase = (b * inC + icStart + c) * inVol;
                      const wBase = (oc * groupIn + c) * kVol;
                      for (let kd = kdLo; kd < kdHi; kd++) {
                        const id = od * sD + kd * dD - pD;
                        for (let kh = khLo; kh < khHi; kh++) {
                          const ih = oh * sH + kh * dH - pH;
                          const inRow = inBase + id * inHW + ih * inW + iw0;
                          const wRow = wBase + kd * kHW + kh * kW;
                          for (let kw = kwLo; kw < kwHi; kw++) {
                            gw[wRow + kw]! += gv * (inFlat[inRow + kw * dW] as number);
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          return gradLike(
            floatTensor(gw, [outC, groupIn, kD, kH, kW], device, "float64"),
            weightTensor
          );
        },
      ],
    ];

    if (biasTensor && this.bias_) {
      grads.push([
        this.bias_,
        (g: Tensor): Tensor => {
          const go = denseFloat64(g);
          const gb = new Float64Array(outC);
          for (let b = 0; b < batch; b++) {
            for (let oc = 0; oc < outC; oc++) {
              let s = 0;
              const oBase = (b * outC + oc) * outVol;
              for (let i = 0; i < outVol; i++) s += go[oBase + i] as number;
              gb[oc]! += s;
            }
          }
          return gradLike(floatTensor(gb, [outC], device, "float64"), biasTensor);
        },
      ]);
    }

    return customOp(outTensor, grads);
  }

  /** Learnable kernel of shape `(outChannels, inChannels / groups, kD, kH, kW)`. */
  get weight(): GradTensor {
    return this.weight_;
  }

  /** Learnable bias of shape `(outChannels)`, or `undefined` when the layer was built with `bias: false`. */
  get bias(): GradTensor | undefined {
    return this.bias_;
  }

  override toString(): string {
    const extra =
      (this.dilation.every((d) => d === 1) ? "" : `, dilation=${JSON.stringify(this.dilation)}`) +
      (this.groups === 1 ? "" : `, groups=${this.groups}`);
    return `Conv3d(${this.inChannels}, ${this.outChannels}, kernel_size=${JSON.stringify(this.kernelSize)}, stride=${JSON.stringify(this.stride)}, padding=${this.paddingLabel}${extra})`;
  }
}

/**
 * 3D Max Pooling Layer.
 *
 * Applies max pooling over a 3D input signal (volumetric data).
 * Input: (N, C, D, H, W) -> Output: (N, C, D_out, H_out, W_out)
 *
 * The stride defaults to the kernel size. Padding is ignored by the maximum, a NaN
 * inside a window propagates, and `padding` may be at most half of `kernelSize`.
 *
 * @example
 * ```ts
 * import { MaxPool3d } from 'deepbox/nn';
 *
 * const pool = new MaxPool3d(2); // 2x2x2 pooling
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-layers | Deepbox Layers}
 */
export class MaxPool3d extends Module {
  private readonly kernelSizeValue: [number, number, number];
  private readonly stride: [number, number, number];
  private readonly padding: [number, number, number];
  private readonly ceilMode: boolean;

  /**
   * @param kernelSize - Window size, one integer or `[kD, kH, kW]`
   * @param options.stride - Step between windows (default: `kernelSize`)
   * @param options.padding - Implicit padding on each side (default 0)
   * @param options.ceilMode - Round the output size up instead of down (default false)
   * @throws {InvalidParameterError} If a parameter is invalid or padding exceeds half the kernel
   */
  constructor(
    kernelSize: number | [number, number, number],
    options: {
      readonly stride?: number | [number, number, number];
      readonly padding?: number | [number, number, number];
      readonly ceilMode?: boolean;
    } = {}
  ) {
    super();

    this.kernelSizeValue = normalizeTriple(
      "kernelSize",
      kernelSize,
      false,
      "a positive integer or a triple of positive integers"
    );
    this.stride = normalizeTriple(
      "stride",
      options.stride ?? kernelSize,
      false,
      "a positive integer or a triple of positive integers"
    );
    this.padding = normalizeTriple(
      "padding",
      options.padding ?? 0,
      true,
      "a non-negative integer or a triple of non-negative integers"
    );
    checkPoolPadding("MaxPool3d", this.kernelSizeValue, this.padding);
    this.ceilMode = options.ceilMode ?? false;
  }

  /**
   * @param input - Input of shape `(N, C, D, H, W)`
   * @returns Output of shape `(N, C, outD, outH, outW)`
   * @throws {ShapeError} If the input is not 5-D or the kernel does not fit
   * @throws {DTypeError} If the input has string dtype
   */
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: AnyTensor): AnyTensor {
    return settle(this.run(input), allPlain(input));
  }

  private run(input: AnyTensor): GradTensor {
    const t = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
    const inputTensor = t.tensor;

    rejectString(inputTensor.dtype, "MaxPool3d");
    if (inputTensor.ndim !== 5) {
      throw new ShapeError(`MaxPool3d expects 5D input (N, C, D, H, W); got ${inputTensor.ndim}D`);
    }

    const batch = inputTensor.shape[0] ?? 0;
    const channels = inputTensor.shape[1] ?? 0;
    const inD = inputTensor.shape[2] ?? 0;
    const inH = inputTensor.shape[3] ?? 0;
    const inW = inputTensor.shape[4] ?? 0;

    const [kD, kH, kW] = this.kernelSizeValue;
    const [sD, sH, sW] = this.stride;
    const [pD, pH, pW] = this.padding;

    const outD = poolOutSize(inD, kD, sD, pD, this.ceilMode);
    const outH = poolOutSize(inH, kH, sH, pH, this.ceilMode);
    const outW = poolOutSize(inW, kW, sW, pW, this.ceilMode);

    if (outD <= 0 || outH <= 0 || outW <= 0) {
      throw new ShapeError("MaxPool3d output dimensions must be positive");
    }

    return pool3d(
      t,
      "max",
      batch * channels,
      [inD, inH, inW],
      [
        slidingAxis(inD, outD, kD, sD, pD),
        slidingAxis(inH, outH, kH, sH, pH),
        slidingAxis(inW, outW, kW, sW, pW),
      ],
      [batch, channels, outD, outH, outW],
      null
    );
  }

  override toString(): string {
    return `MaxPool3d(kernel_size=${JSON.stringify(this.kernelSizeValue)}, stride=${JSON.stringify(this.stride)}, padding=${JSON.stringify(this.padding)}${this.ceilMode ? ", ceil_mode=true" : ""})`;
  }
}

/**
 * 3D Average Pooling Layer.
 *
 * Applies average pooling over a 3D input signal (volumetric data).
 * Input: (N, C, D, H, W) -> Output: (N, C, D_out, H_out, W_out)
 *
 * By default padded zeros count towards the average (PyTorch's `count_include_pad=True`);
 * pass `countIncludePad: false` to divide by the number of real elements in each window.
 * The stride defaults to the kernel size and `padding` may be at most half of `kernelSize`.
 *
 * @example
 * ```ts
 * import { AvgPool3d } from 'deepbox/nn';
 *
 * const pool = new AvgPool3d(2); // 2x2x2 pooling
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-layers | Deepbox Layers}
 */
export class AvgPool3d extends Module {
  private readonly kernelSizeValue: [number, number, number];
  private readonly stride: [number, number, number];
  private readonly padding: [number, number, number];
  private readonly countIncludePad: boolean;
  private readonly ceilMode: boolean;

  /**
   * @param kernelSize - Window size, one integer or `[kD, kH, kW]`
   * @param options.stride - Step between windows (default: `kernelSize`)
   * @param options.padding - Implicit zero padding on each side (default 0)
   * @param options.countIncludePad - Count padded zeros in the average (default true)
   * @param options.ceilMode - Round the output size up instead of down (default false). A
   *   partial last window divides by the part of it that lies inside the padded input.
   * @throws {InvalidParameterError} If a parameter is invalid or padding exceeds half the kernel
   */
  constructor(
    kernelSize: number | [number, number, number],
    options: {
      readonly stride?: number | [number, number, number];
      readonly padding?: number | [number, number, number];
      readonly countIncludePad?: boolean;
      readonly ceilMode?: boolean;
    } = {}
  ) {
    super();

    this.kernelSizeValue = normalizeTriple(
      "kernelSize",
      kernelSize,
      false,
      "a positive integer or a triple of positive integers"
    );
    this.stride = normalizeTriple(
      "stride",
      options.stride ?? kernelSize,
      false,
      "a positive integer or a triple of positive integers"
    );
    this.padding = normalizeTriple(
      "padding",
      options.padding ?? 0,
      true,
      "a non-negative integer or a triple of non-negative integers"
    );
    checkPoolPadding("AvgPool3d", this.kernelSizeValue, this.padding);
    this.countIncludePad = options.countIncludePad ?? true;
    this.ceilMode = options.ceilMode ?? false;
  }

  /**
   * @param input - Input of shape `(N, C, D, H, W)`
   * @returns Output of shape `(N, C, outD, outH, outW)`
   * @throws {ShapeError} If the input is not 5-D or the kernel does not fit
   * @throws {DTypeError} If the input has string dtype
   */
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: AnyTensor): AnyTensor {
    return settle(this.run(input), allPlain(input));
  }

  private run(input: AnyTensor): GradTensor {
    const t = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
    const inputTensor = t.tensor;

    rejectString(inputTensor.dtype, "AvgPool3d");
    if (inputTensor.ndim !== 5) {
      throw new ShapeError(`AvgPool3d expects 5D input (N, C, D, H, W); got ${inputTensor.ndim}D`);
    }

    const batch = inputTensor.shape[0] ?? 0;
    const channels = inputTensor.shape[1] ?? 0;
    const inD = inputTensor.shape[2] ?? 0;
    const inH = inputTensor.shape[3] ?? 0;
    const inW = inputTensor.shape[4] ?? 0;

    const [kD, kH, kW] = this.kernelSizeValue;
    const [sD, sH, sW] = this.stride;
    const [pD, pH, pW] = this.padding;

    const outD = poolOutSize(inD, kD, sD, pD, this.ceilMode);
    const outH = poolOutSize(inH, kH, sH, pH, this.ceilMode);
    const outW = poolOutSize(inW, kW, sW, pW, this.ceilMode);

    if (outD <= 0 || outH <= 0 || outW <= 0) {
      throw new ShapeError("AvgPool3d output dimensions must be positive");
    }

    return pool3d(
      t,
      "avg",
      batch * channels,
      [inD, inH, inW],
      [
        slidingAxis(inD, outD, kD, sD, pD),
        slidingAxis(inH, outH, kH, sH, pH),
        slidingAxis(inW, outW, kW, sW, pW),
      ],
      [batch, channels, outD, outH, outW],
      this.countIncludePad && !this.ceilMode ? kD * kH * kW : null,
      this.countIncludePad && this.ceilMode
        ? [
            paddedExtent(inD, outD, kD, sD, pD),
            paddedExtent(inH, outH, kH, sH, pH),
            paddedExtent(inW, outW, kW, sW, pW),
          ]
        : undefined
    );
  }

  override toString(): string {
    return `AvgPool3d(kernel_size=${JSON.stringify(this.kernelSizeValue)}, stride=${JSON.stringify(this.stride)}, padding=${JSON.stringify(this.padding)}${this.ceilMode ? ", ceil_mode=true" : ""})`;
  }
}
