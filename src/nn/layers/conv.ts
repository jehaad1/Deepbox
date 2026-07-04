import {
  DeepboxError,
  DTypeError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../../core";
import {
  type AnyTensor,
  customOp,
  GradTensor,
  im2colGrad,
  mulScalar,
  parameter,
  randn,
} from "../../ndarray";
import { readAsNumber, requireNumericData } from "../../ndarray/ops/_internal";
import { dispatchPool2d, dispatchPool2dBackward } from "../../ndarray/ops/device_dispatch";
import { isContiguous, offsetFromFlatIndex } from "../../ndarray/tensor/strides";
import { computeStrides, Tensor } from "../../ndarray/tensor/Tensor";
import { Module } from "../module/Module";

/**
 * Generic differentiable pooling over precomputed windows.
 *
 * `windows[o]` lists the input logical-flat indices contributing to output
 * element `o` (only in-bounds indices — so max pooling naturally excludes
 * padding, i.e. uses -inf padding semantics rather than 0). For "avg", the
 * divisor defaults to each window's valid-element count; pass `divisor` to
 * override (e.g. full kernel size for PyTorch's count_include_pad=true).
 * Returns a GradTensor whose backward routes gradient to the argmax (max) or
 * distributes it evenly (avg).
 */
function genericPool(
  input: GradTensor,
  outShape: number[],
  windows: number[][],
  mode: "max" | "avg",
  divisor?: number[]
): GradTensor {
  const inDense = denseFloat64(input.tensor);
  const outSize = outShape.reduce((a, b) => a * b, 1);
  const outArr = new Float64Array(outSize);
  const argmax = mode === "max" ? new Int32Array(outSize).fill(-1) : null;

  for (let o = 0; o < outSize; o++) {
    const win = windows[o] ?? [];
    if (mode === "max") {
      let best = Number.NEGATIVE_INFINITY;
      let bestIdx = -1;
      for (const idx of win) {
        const v = inDense[idx] ?? 0;
        if (v > best) {
          best = v;
          bestIdx = idx;
        }
      }
      outArr[o] = win.length === 0 ? 0 : best;
      if (argmax) argmax[o] = bestIdx;
    } else {
      let s = 0;
      for (const idx of win) s += inDense[idx] ?? 0;
      const d = divisor ? (divisor[o] ?? win.length) : win.length;
      outArr[o] = d === 0 ? 0 : s / d;
    }
  }

  const outTensor = Tensor.fromTypedArray({
    data: outArr,
    shape: outShape,
    dtype: "float64",
    device: input.tensor.device,
  });

  return customOp(outTensor, [
    [
      input,
      (g: Tensor): Tensor => {
        const go = denseFloat64(g);
        const gi = new Float64Array(input.tensor.size);
        for (let o = 0; o < outSize; o++) {
          const gv = go[o] ?? 0;
          if (mode === "max") {
            const idx = argmax ? (argmax[o] ?? -1) : -1;
            if (idx >= 0) gi[idx]! += gv;
          } else {
            const win = windows[o] ?? [];
            const d = divisor ? (divisor[o] ?? win.length) : win.length;
            if (d !== 0) {
              const share = gv / d;
              for (const idx of win) gi[idx]! += share;
            }
          }
        }
        return Tensor.fromTypedArray({
          data: gi,
          shape: input.tensor.shape,
          dtype: "float64",
          device: input.tensor.device,
        });
      },
    ],
  ]);
}

/**
 * On-device 2-D pooling with a regular kernel/stride/padding. Runs the pool
 * kernel forward and wires the matching pool-backward kernel through
 * `customOp`, so the whole op stays resident on the device. Semantics match
 * {@link genericPool}: `max` excludes padding and routes the gradient to each
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

/**
 * Materialize any numeric tensor (including non-contiguous views) into a
 * contiguous, logical-order Float64Array. Used by convolution layers whose
 * hand-written forward/backward kernels index elements in row-major order.
 */
function denseFloat64(t: Tensor): Float64Array {
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

/**
 * 1D Convolutional Layer.
 *
 * Applies a 1D convolution over an input signal composed of several input planes.
 *
 * @example
 * ```ts
 * import { Conv1d } from 'deepbox/nn';
 *
 * const conv = new Conv1d(16, 33, 3); // in_channels=16, out_channels=33, kernel_size=3
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-layers | Deepbox Layers}
 */
export class Conv1d extends Module {
  private readonly inChannels: number;
  private readonly outChannels: number;
  private readonly kernelSize: number;
  private readonly stride: number;
  private readonly padding: number;
  private readonly bias: boolean;

  private weight_?: GradTensor;
  private bias_?: GradTensor;

  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number,
    options: {
      readonly stride?: number;
      readonly padding?: number;
      readonly bias?: boolean;
    } = {}
  ) {
    super();

    // Validate parameters
    if (inChannels <= 0 || !Number.isInteger(inChannels)) {
      throw new InvalidParameterError(
        "inChannels must be a positive integer",
        "inChannels",
        inChannels
      );
    }
    if (outChannels <= 0 || !Number.isInteger(outChannels)) {
      throw new InvalidParameterError(
        "outChannels must be a positive integer",
        "outChannels",
        outChannels
      );
    }
    if (kernelSize <= 0 || !Number.isInteger(kernelSize)) {
      throw new InvalidParameterError(
        "kernelSize must be a positive integer",
        "kernelSize",
        kernelSize
      );
    }

    const stride = options.stride ?? 1;
    if (stride <= 0 || !Number.isInteger(stride)) {
      throw new InvalidParameterError("stride must be a positive integer", "stride", stride);
    }

    const padding = options.padding ?? 0;
    if (padding < 0 || !Number.isInteger(padding)) {
      throw new InvalidParameterError("padding must be a non-negative integer", "padding", padding);
    }

    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = kernelSize;
    this.stride = stride;
    this.padding = padding;
    this.bias = options.bias ?? true;

    this.initializeParameters();
  }

  private initializeParameters(): void {
    const k = 1 / Math.sqrt(this.inChannels * this.kernelSize);
    const weight = randn([this.outChannels, this.inChannels, this.kernelSize]);
    this.weight_ = parameter(mulScalar(weight, k));
    this.registerParameter("weight", this.weight_);

    if (this.bias) {
      const biasInit = randn([this.outChannels]);
      this.bias_ = parameter(mulScalar(biasInit, k));
      this.registerParameter("bias", this.bias_);
    }
  }

  forward(x: AnyTensor): GradTensor {
    // Convert to GradTensor if needed
    const input = GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x);

    // Reject string tensors
    if (input.dtype === "string") {
      throw new DTypeError("String tensors are not supported");
    }

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

    const weight = this.weight_;
    if (!weight) throw new NotFittedError("Weight not initialized");

    // 1D Convolution using 2D operations (unsqueeze height dim)
    // Input: (B, C, L) -> (B, C, 1, L)
    const input2d = input.reshape([batch, inC, 1, inL]);

    // Params for im2col
    // Kernel: (1, K)
    const kernelSize: [number, number] = [1, this.kernelSize];
    const stride: [number, number] = [1, this.stride];
    const padding: [number, number] = [0, this.padding];

    // im2col -> (B, outL, C * 1 * K)
    const cols = im2colGrad(input2d, kernelSize, stride, padding);

    // Weights: (outC, inC, K) -> (outC, inC * K)
    // Note: im2col flattens as channels * kH * kW.
    // Our weights are (outC, inC, K). Reshape to (outC, inC * K).
    const weightFlat = weight.reshape([this.outChannels, this.inChannels * this.kernelSize]);

    // Matmul: (B, outL, inC*K) @ (outC, inC*K).T -> (B, outL, outC)
    const out = cols.matmul(weightFlat.transpose());

    // Reshape to (B, outC, outL)
    const outTransposed = out.transpose([0, 2, 1]); // (B, outC, outL)

    if (this.bias && this.bias_) {
      // Bias: (outC) -> (1, outC, 1) broadcast
      const biasReshaped = this.bias_.reshape([1, this.outChannels, 1]);
      return outTransposed.add(biasReshaped);
    }

    return outTransposed;
  }

  get weight(): GradTensor {
    if (!this.weight_) {
      throw new NotFittedError("Weight not initialized");
    }
    return this.weight_;
  }
}

/**
 * 2D Convolutional Layer.
 *
 * Applies a 2D convolution over an input signal composed of several input planes.
 *
 * @example
 * ```ts
 * import { Conv2d } from 'deepbox/nn';
 *
 * const conv = new Conv2d(3, 64, 3); // RGB to 64 channels, 3x3 kernel
 * ```
 *
 * @see {@link https://deepbox.dev/docs/nn-layers | Deepbox Layers}
 */
export class Conv2d extends Module {
  private readonly inChannels: number;
  private readonly outChannels: number;
  private readonly kernelSize: [number, number];
  private readonly stride: [number, number];
  private readonly padding: [number, number];
  private readonly useBias: boolean;

  private weight_?: GradTensor;
  private bias_?: GradTensor;

  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number | [number, number],
    options: {
      readonly stride?: number | [number, number];
      readonly padding?: number | [number, number];
      readonly bias?: boolean;
    } = {}
  ) {
    super();
    if (inChannels <= 0 || !Number.isInteger(inChannels)) {
      throw new InvalidParameterError(
        "inChannels must be a positive integer",
        "inChannels",
        inChannels
      );
    }
    if (outChannels <= 0 || !Number.isInteger(outChannels)) {
      throw new InvalidParameterError(
        "outChannels must be a positive integer",
        "outChannels",
        outChannels
      );
    }
    const kernelArr = normalizePair(
      "kernelSize",
      kernelSize,
      false,
      "a positive integer or a tuple of two positive integers"
    );

    const stride = options.stride ?? 1;
    const strideArr = normalizePair(
      "stride",
      stride,
      false,
      "a positive integer or a tuple of two positive integers"
    );

    const padding = options.padding ?? 0;
    const paddingArr = normalizePair(
      "padding",
      padding,
      true,
      "a non-negative integer or a tuple of two non-negative integers"
    );

    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = kernelArr;
    this.stride = strideArr;
    this.padding = paddingArr;

    this.useBias = options.bias ?? true;

    this.initializeParameters();
  }

  private initializeParameters(): void {
    const kH = this.kernelSize[0] ?? 1;
    const kW = this.kernelSize[1] ?? 1;
    const k = 1 / Math.sqrt(this.inChannels * kH * kW);
    const weight = randn([this.outChannels, this.inChannels, kH, kW]);
    this.weight_ = parameter(mulScalar(weight, k));
    this.registerParameter("weight", this.weight_);

    if (this.useBias) {
      const biasInit = randn([this.outChannels]);
      this.bias_ = parameter(mulScalar(biasInit, k));
      this.registerParameter("bias", this.bias_);
    }
  }

  forward(x: AnyTensor): GradTensor {
    const input = GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x);

    if (input.dtype === "string") {
      throw new DTypeError("String tensors are not supported");
    }

    if (input.ndim !== 4) {
      throw new ShapeError(
        `Conv2d expects 4D input (batch, channels, height, width), got ${input.ndim}D`
      );
    }

    const batch = input.shape[0] ?? 0;
    const inC = input.shape[1] ?? 0;
    const inH = input.shape[2] ?? 0;
    const inW = input.shape[3] ?? 0;

    if (inC !== this.inChannels) {
      throw new ShapeError(`Expected ${this.inChannels} input channels, got ${inC}`);
    }

    const weight = this.weight_;
    if (!weight) throw new NotFittedError("Weight not initialized");

    const [kH, kW] = this.kernelSize;
    const [sH, sW] = this.stride;
    const [pH, pW] = this.padding;

    // im2col -> (B, outH*outW, C*kH*kW)
    const cols = im2colGrad(input, [kH, kW], [sH, sW], [pH, pW]);

    // Calculate output dimensions from input (im2col does this internally but we need it for reshape)
    const outH = Math.floor((inH + 2 * pH - kH) / sH) + 1;
    const outW = Math.floor((inW + 2 * pW - kW) / sW) + 1;

    // Weights: (outC, inC, kH, kW) -> (outC, inC*kH*kW)
    const weightFlat = weight.reshape([this.outChannels, this.inChannels * kH * kW]);

    // Matmul: (B, outPixels, inFeatures) @ (outC, inFeatures).T -> (B, outPixels, outC)
    const out = cols.matmul(weightFlat.transpose());

    // Reshape: (B, outPixels, outC) -> (B, outC, outPixels) -> (B, outC, outH, outW)
    const outTransposed = out.transpose([0, 2, 1]);
    const outReshaped = outTransposed.reshape([batch, this.outChannels, outH, outW]);

    if (this.useBias && this.bias_) {
      // Bias: (outC) -> (1, outC, 1, 1)
      const biasReshaped = this.bias_.reshape([1, this.outChannels, 1, 1]);
      return outReshaped.add(biasReshaped);
    }

    return outReshaped;
  }

  get weight(): GradTensor {
    if (!this.weight_) {
      throw new NotFittedError("Weight not initialized");
    }
    return this.weight_;
  }
}

/**
 * 2D Max Pooling Layer.
 *
 * Applies a 2D max pooling over an input signal.
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

  constructor(
    kernelSize: number | [number, number],
    options: {
      readonly stride?: number | [number, number];
      readonly padding?: number | [number, number];
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
  }

  forward(x: AnyTensor): GradTensor {
    const input = GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x);

    if (input.dtype === "string") {
      throw new DTypeError("String tensors are not supported");
    }

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

    const outH = Math.floor((inH + 2 * pH - kH) / sH) + 1;
    const outW = Math.floor((inW + 2 * pW - kW) / sW) + 1;

    if (input.tensor.isDeviceTensor) {
      return devicePool2d(input, "max", [kH, kW], [sH, sW], [pH, pW]);
    }

    // Enumerate per-output windows of in-bounds input indices. Max pooling
    // over these excludes padding (i.e. -inf padding, matching PyTorch),
    // rather than the im2col path's incorrect 0-padding; genericPool also
    // routes gradients to the argmax in the input's dtype.
    const windows: number[][] = [];
    for (let n = 0; n < batch; n++) {
      for (let c = 0; c < channels; c++) {
        const base = (n * channels + c) * inH * inW;
        for (let oh = 0; oh < outH; oh++) {
          for (let ow = 0; ow < outW; ow++) {
            const win: number[] = [];
            for (let kh = 0; kh < kH; kh++) {
              for (let kw = 0; kw < kW; kw++) {
                const ih = oh * sH + kh - pH;
                const iw = ow * sW + kw - pW;
                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) win.push(base + ih * inW + iw);
              }
            }
            windows.push(win);
          }
        }
      }
    }

    return genericPool(input, [batch, channels, outH, outW], windows, "max");
  }
}

/**
 * 2D Average Pooling Layer.
 *
 * Applies a 2D average pooling over an input signal.
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

  constructor(
    kernelSize: number | [number, number],
    options: {
      readonly stride?: number | [number, number];
      readonly padding?: number | [number, number];
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
  }

  forward(x: AnyTensor): GradTensor {
    const input = GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x);

    if (input.dtype === "string") {
      throw new DTypeError("String tensors are not supported");
    }

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

    // Reshape: (B, C, H, W) -> (B*C, 1, H, W)
    const inputReshaped = input.reshape([batch * channels, 1, inH, inW]);

    // im2col -> (B*C, outPixels, 1 * kH * kW)
    const cols = im2colGrad(inputReshaped, [kH, kW], [sH, sW], [pH, pW]);

    // Mean over kernel window (axis 2)
    // (B*C, outPixels, kH*kW) -> (B*C, outPixels)
    const meanVals = cols.mean(2);

    // Calculate output dims
    const outH = Math.floor((inH + 2 * pH - kH) / sH) + 1;
    const outW = Math.floor((inW + 2 * pW - kW) / sW) + 1;

    // Reshape back: (B*C, outH*outW) -> (B, C, outH, outW)
    return meanVals.reshape([batch, channels, outH, outW]);
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
 * @example
 * ```ts
 * import { ConvTranspose2d } from 'deepbox/nn';
 *
 * const deconv = new ConvTranspose2d(16, 33, 3, { stride: 2, padding: 1 });
 * ```
 */
export class ConvTranspose2d extends Module {
  private readonly inChannels: number;
  private readonly outChannels: number;
  private readonly kernelSize: [number, number];
  private readonly stride: [number, number];
  private readonly padding: [number, number];
  private readonly outputPadding: [number, number];
  private readonly useBias: boolean;

  private weight_: GradTensor;
  private bias_: GradTensor | undefined;

  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number | [number, number],
    options: {
      readonly stride?: number | [number, number];
      readonly padding?: number | [number, number];
      readonly outputPadding?: number | [number, number];
      readonly bias?: boolean;
    } = {}
  ) {
    super();

    if (inChannels <= 0 || !Number.isInteger(inChannels)) {
      throw new InvalidParameterError(
        "inChannels must be a positive integer",
        "inChannels",
        inChannels
      );
    }
    if (outChannels <= 0 || !Number.isInteger(outChannels)) {
      throw new InvalidParameterError(
        "outChannels must be a positive integer",
        "outChannels",
        outChannels
      );
    }

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

    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.useBias = options.bias ?? true;

    // Weight shape: (inChannels, outChannels, kH, kW)
    const [kH, kW] = this.kernelSize;
    const k = 1 / Math.sqrt(outChannels * kH * kW);
    this.weight_ = parameter(mulScalar(randn([inChannels, outChannels, kH, kW]), k));
    this.registerParameter("weight", this.weight_);

    if (this.useBias) {
      this.bias_ = parameter(mulScalar(randn([outChannels]), k));
      this.registerParameter("bias", this.bias_);
    }
  }

  forward(x: AnyTensor): GradTensor {
    const input = GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x);

    if (input.dtype === "string") {
      throw new DTypeError("String tensors are not supported");
    }
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

    const outH = (inH - 1) * sH - 2 * pH + kH + opH;
    const outW = (inW - 1) * sW - 2 * pW + kW + opW;

    // Transposed convolution: scatter input values through the kernel.
    // Materialize input and weight into contiguous float64 arrays so both the
    // forward scatter and the analytical backward index them uniformly.
    const inputTensor = input.tensor;
    const weightTensor = this.weight_.tensor;
    const inFlat = denseFloat64(inputTensor);
    const wFlat = denseFloat64(weightTensor);
    const biasFlat = this.useBias && this.bias_ ? denseFloat64(this.bias_.tensor) : null;

    const outC = this.outChannels;
    const outSize = batch * outC * outH * outW;
    const outArr = new Float64Array(outSize);

    const inIdx = (b: number, ic: number, ih: number, iw: number): number =>
      b * inC * inH * inW + ic * inH * inW + ih * inW + iw;
    const wIdx = (ic: number, oc: number, kh: number, kw: number): number =>
      ic * outC * kH * kW + oc * kH * kW + kh * kW + kw;
    const oIdx = (b: number, oc: number, oh: number, ow: number): number =>
      b * outC * outH * outW + oc * outH * outW + oh * outW + ow;

    for (let b = 0; b < batch; b++) {
      for (let ic = 0; ic < this.inChannels; ic++) {
        for (let ih = 0; ih < inH; ih++) {
          for (let iw = 0; iw < inW; iw++) {
            const inVal = inFlat[inIdx(b, ic, ih, iw)] ?? 0;
            for (let oc = 0; oc < outC; oc++) {
              for (let kh = 0; kh < kH; kh++) {
                for (let kw = 0; kw < kW; kw++) {
                  const oh = ih * sH - pH + kh;
                  const ow = iw * sW - pW + kw;
                  if (oh >= 0 && oh < outH && ow >= 0 && ow < outW) {
                    outArr[oIdx(b, oc, oh, ow)]! += inVal * (wFlat[wIdx(ic, oc, kh, kw)] ?? 0);
                  }
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
          const bv = biasFlat[oc] ?? 0;
          for (let oh = 0; oh < outH; oh++) {
            for (let ow = 0; ow < outW; ow++) outArr[oIdx(b, oc, oh, ow)]! += bv;
          }
        }
      }
    }

    const outTensor = Tensor.fromTypedArray({
      data: outArr,
      shape: [batch, outC, outH, outW],
      dtype: "float64",
      device: inputTensor.device,
    });

    const grads: Array<[GradTensor, (g: Tensor) => Tensor]> = [];

    grads.push([
      input,
      (g: Tensor): Tensor => {
        const go = denseFloat64(g);
        const gi = new Float64Array(batch * inC * inH * inW);
        for (let b = 0; b < batch; b++) {
          for (let ic = 0; ic < this.inChannels; ic++) {
            for (let ih = 0; ih < inH; ih++) {
              for (let iw = 0; iw < inW; iw++) {
                let s = 0;
                for (let oc = 0; oc < outC; oc++) {
                  for (let kh = 0; kh < kH; kh++) {
                    for (let kw = 0; kw < kW; kw++) {
                      const oh = ih * sH - pH + kh;
                      const ow = iw * sW - pW + kw;
                      if (oh >= 0 && oh < outH && ow >= 0 && ow < outW) {
                        s += (go[oIdx(b, oc, oh, ow)] ?? 0) * (wFlat[wIdx(ic, oc, kh, kw)] ?? 0);
                      }
                    }
                  }
                }
                gi[inIdx(b, ic, ih, iw)] = s;
              }
            }
          }
        }
        return Tensor.fromTypedArray({
          data: gi,
          shape: [batch, inC, inH, inW],
          dtype: "float64",
          device: inputTensor.device,
        });
      },
    ]);

    grads.push([
      this.weight_,
      (g: Tensor): Tensor => {
        const go = denseFloat64(g);
        const gw = new Float64Array(this.inChannels * outC * kH * kW);
        for (let b = 0; b < batch; b++) {
          for (let ic = 0; ic < this.inChannels; ic++) {
            for (let ih = 0; ih < inH; ih++) {
              for (let iw = 0; iw < inW; iw++) {
                const inVal = inFlat[inIdx(b, ic, ih, iw)] ?? 0;
                for (let oc = 0; oc < outC; oc++) {
                  for (let kh = 0; kh < kH; kh++) {
                    for (let kw = 0; kw < kW; kw++) {
                      const oh = ih * sH - pH + kh;
                      const ow = iw * sW - pW + kw;
                      if (oh >= 0 && oh < outH && ow >= 0 && ow < outW) {
                        gw[wIdx(ic, oc, kh, kw)]! += inVal * (go[oIdx(b, oc, oh, ow)] ?? 0);
                      }
                    }
                  }
                }
              }
            }
          }
        }
        return Tensor.fromTypedArray({
          data: gw,
          shape: [this.inChannels, outC, kH, kW],
          dtype: "float64",
          device: inputTensor.device,
        });
      },
    ]);

    if (this.useBias && this.bias_) {
      grads.push([
        this.bias_,
        (g: Tensor): Tensor => {
          const go = denseFloat64(g);
          const gb = new Float64Array(outC);
          for (let b = 0; b < batch; b++) {
            for (let oc = 0; oc < outC; oc++) {
              let s = 0;
              for (let oh = 0; oh < outH; oh++) {
                for (let ow = 0; ow < outW; ow++) s += go[oIdx(b, oc, oh, ow)] ?? 0;
              }
              gb[oc]! += s;
            }
          }
          return Tensor.fromTypedArray({
            data: gb,
            shape: [outC],
            dtype: "float64",
            device: inputTensor.device,
          });
        },
      ]);
    }

    return customOp(outTensor, grads);
  }

  get weight(): GradTensor {
    return this.weight_;
  }

  override toString(): string {
    return `ConvTranspose2d(${this.inChannels}, ${this.outChannels}, kernel_size=${this.kernelSize}, stride=${this.stride}, padding=${this.padding})`;
  }
}

/**
 * 2D Adaptive Average Pooling.
 *
 * Produces output of specified size regardless of input dimensions.
 * Required by virtually all modern CNN architectures (ResNet, etc.)
 * to handle variable input sizes.
 *
 * @example
 * ```ts
 * import { AdaptiveAvgPool2d } from 'deepbox/nn';
 *
 * const pool = new AdaptiveAvgPool2d([1, 1]); // Global average pooling
 * ```
 */
export class AdaptiveAvgPool2d extends Module {
  private readonly outputSize: [number, number];

  constructor(outputSize: number | [number, number]) {
    super();

    if (typeof outputSize === "number") {
      if (!Number.isInteger(outputSize) || outputSize <= 0) {
        throw new InvalidParameterError(
          "outputSize must be a positive integer",
          "outputSize",
          outputSize
        );
      }
      this.outputSize = [outputSize, outputSize];
    } else {
      const [h, w] = outputSize;
      if (!Number.isInteger(h) || h <= 0 || !Number.isInteger(w) || w <= 0) {
        throw new InvalidParameterError(
          "outputSize must be positive integers",
          "outputSize",
          outputSize
        );
      }
      this.outputSize = [h, w];
    }
  }

  forward(x: AnyTensor): GradTensor {
    const input = GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x);

    if (input.dtype === "string") {
      throw new DTypeError("String tensors are not supported");
    }
    if (input.ndim !== 4) {
      throw new ShapeError(
        `AdaptiveAvgPool2d expects 4D input (batch, channels, height, width), got ${input.ndim}D`
      );
    }

    const batch = input.shape[0] ?? 0;
    const channels = input.shape[1] ?? 0;
    const inH = input.shape[2] ?? 0;
    const inW = input.shape[3] ?? 0;
    const [outH, outW] = this.outputSize;

    const windows: number[][] = [];
    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        const base = (b * channels + c) * inH * inW;
        for (let oh = 0; oh < outH; oh++) {
          for (let ow = 0; ow < outW; ow++) {
            const hStart = Math.floor((oh * inH) / outH);
            const hEnd = Math.floor(((oh + 1) * inH) / outH);
            const wStart = Math.floor((ow * inW) / outW);
            const wEnd = Math.floor(((ow + 1) * inW) / outW);
            const win: number[] = [];
            for (let ih = hStart; ih < hEnd; ih++) {
              for (let iw = wStart; iw < wEnd; iw++) win.push(base + ih * inW + iw);
            }
            windows.push(win);
          }
        }
      }
    }

    return genericPool(input, [batch, channels, outH, outW], windows, "avg");
  }

  override toString(): string {
    return `AdaptiveAvgPool2d(output_size=${JSON.stringify(this.outputSize)})`;
  }
}

/**
 * 2D Adaptive Max Pooling.
 *
 * Produces output of specified size regardless of input dimensions,
 * selecting the maximum value from each adaptive window.
 *
 * @example
 * ```ts
 * import { AdaptiveMaxPool2d } from 'deepbox/nn';
 *
 * const pool = new AdaptiveMaxPool2d([1, 1]); // Global max pooling
 * ```
 */
export class AdaptiveMaxPool2d extends Module {
  private readonly outputSize: [number, number];

  constructor(outputSize: number | [number, number]) {
    super();

    if (typeof outputSize === "number") {
      if (!Number.isInteger(outputSize) || outputSize <= 0) {
        throw new InvalidParameterError(
          "outputSize must be a positive integer",
          "outputSize",
          outputSize
        );
      }
      this.outputSize = [outputSize, outputSize];
    } else {
      const [h, w] = outputSize;
      if (!Number.isInteger(h) || h <= 0 || !Number.isInteger(w) || w <= 0) {
        throw new InvalidParameterError(
          "outputSize must be positive integers",
          "outputSize",
          outputSize
        );
      }
      this.outputSize = [h, w];
    }
  }

  forward(x: AnyTensor): GradTensor {
    const input = GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x);

    if (input.dtype === "string") {
      throw new DTypeError("String tensors are not supported");
    }
    if (input.ndim !== 4) {
      throw new ShapeError(
        `AdaptiveMaxPool2d expects 4D input (batch, channels, height, width), got ${input.ndim}D`
      );
    }

    const batch = input.shape[0] ?? 0;
    const channels = input.shape[1] ?? 0;
    const inH = input.shape[2] ?? 0;
    const inW = input.shape[3] ?? 0;
    const [outH, outW] = this.outputSize;

    const windows: number[][] = [];
    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        const base = (b * channels + c) * inH * inW;
        for (let oh = 0; oh < outH; oh++) {
          for (let ow = 0; ow < outW; ow++) {
            const hStart = Math.floor((oh * inH) / outH);
            const hEnd = Math.floor(((oh + 1) * inH) / outH);
            const wStart = Math.floor((ow * inW) / outW);
            const wEnd = Math.floor(((ow + 1) * inW) / outW);
            const win: number[] = [];
            for (let ih = hStart; ih < hEnd; ih++) {
              for (let iw = wStart; iw < wEnd; iw++) win.push(base + ih * inW + iw);
            }
            windows.push(win);
          }
        }
      }
    }

    return genericPool(input, [batch, channels, outH, outW], windows, "max");
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
 * @category Neural Network Layers
 */
export class MaxPool1d extends Module {
  private readonly kernelSize: number;
  private readonly stride: number;
  private readonly padding: number;

  constructor(
    kernelSize: number,
    options: { readonly stride?: number; readonly padding?: number } = {}
  ) {
    super();
    if (!Number.isInteger(kernelSize) || kernelSize <= 0) {
      throw new InvalidParameterError(
        "kernelSize must be a positive integer",
        "kernelSize",
        kernelSize
      );
    }
    this.kernelSize = kernelSize;
    this.stride = options.stride ?? kernelSize;
    this.padding = options.padding ?? 0;
  }

  forward(input: AnyTensor): GradTensor {
    const t = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
    const inputTensor = t.tensor;

    if (inputTensor.dtype === "string") {
      throw new DTypeError("MaxPool1d does not support string dtype");
    }
    if (inputTensor.ndim !== 3) {
      throw new ShapeError(`MaxPool1d expects 3D input (N, C, L); got ${inputTensor.ndim}D`);
    }

    const batch = inputTensor.shape[0] ?? 0;
    const channels = inputTensor.shape[1] ?? 0;
    const length = inputTensor.shape[2] ?? 0;
    const outL = Math.floor((length + 2 * this.padding - this.kernelSize) / this.stride) + 1;

    if (outL <= 0) {
      throw new ShapeError("MaxPool1d output length must be positive");
    }

    if (inputTensor.isDeviceTensor) {
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

    const windows: number[][] = [];
    for (let n = 0; n < batch; n++) {
      for (let c = 0; c < channels; c++) {
        const base = (n * channels + c) * length;
        for (let ol = 0; ol < outL; ol++) {
          const win: number[] = [];
          for (let k = 0; k < this.kernelSize; k++) {
            const il = ol * this.stride + k - this.padding;
            if (il >= 0 && il < length) win.push(base + il);
          }
          windows.push(win);
        }
      }
    }

    return genericPool(t, [batch, channels, outL], windows, "max");
  }

  override toString(): string {
    return `MaxPool1d(kernel_size=${this.kernelSize}, stride=${this.stride}, padding=${this.padding})`;
  }
}

/**
 * 1D Average Pooling.
 *
 * Applies average pooling over a 1D signal.
 * Input: (N, C, L) -> Output: (N, C, L_out)
 *
 * @category Neural Network Layers
 */
export class AvgPool1d extends Module {
  private readonly kernelSize: number;
  private readonly stride: number;
  private readonly padding: number;

  constructor(
    kernelSize: number,
    options: { readonly stride?: number; readonly padding?: number } = {}
  ) {
    super();
    if (!Number.isInteger(kernelSize) || kernelSize <= 0) {
      throw new InvalidParameterError(
        "kernelSize must be a positive integer",
        "kernelSize",
        kernelSize
      );
    }
    this.kernelSize = kernelSize;
    this.stride = options.stride ?? kernelSize;
    this.padding = options.padding ?? 0;
  }

  forward(input: AnyTensor): GradTensor {
    const t = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
    const inputTensor = t.tensor;

    if (inputTensor.dtype === "string") {
      throw new DTypeError("AvgPool1d does not support string dtype");
    }
    if (inputTensor.ndim !== 3) {
      throw new ShapeError(`AvgPool1d expects 3D input (N, C, L); got ${inputTensor.ndim}D`);
    }

    const batch = inputTensor.shape[0] ?? 0;
    const channels = inputTensor.shape[1] ?? 0;
    const length = inputTensor.shape[2] ?? 0;
    const outL = Math.floor((length + 2 * this.padding - this.kernelSize) / this.stride) + 1;

    if (outL <= 0) {
      throw new ShapeError("AvgPool1d output length must be positive");
    }

    // count_include_pad=true (PyTorch default): divide by the full kernel
    // size, not the count of in-bounds elements.
    const windows: number[][] = [];
    const divisor: number[] = [];
    for (let n = 0; n < batch; n++) {
      for (let c = 0; c < channels; c++) {
        const base = (n * channels + c) * length;
        for (let ol = 0; ol < outL; ol++) {
          const win: number[] = [];
          for (let k = 0; k < this.kernelSize; k++) {
            const il = ol * this.stride + k - this.padding;
            if (il >= 0 && il < length) win.push(base + il);
          }
          windows.push(win);
          divisor.push(this.kernelSize);
        }
      }
    }

    return genericPool(t, [batch, channels, outL], windows, "avg", divisor);
  }

  override toString(): string {
    return `AvgPool1d(kernel_size=${this.kernelSize}, stride=${this.stride}, padding=${this.padding})`;
  }
}

/**
 * 1D Adaptive Average Pooling.
 *
 * Produces fixed output length regardless of input size.
 * Input: (N, C, L) -> Output: (N, C, outputSize)
 *
 * @category Neural Network Layers
 */
export class AdaptiveAvgPool1d extends Module {
  private readonly outputSize: number;

  constructor(outputSize: number) {
    super();
    if (!Number.isInteger(outputSize) || outputSize <= 0) {
      throw new InvalidParameterError(
        "outputSize must be a positive integer",
        "outputSize",
        outputSize
      );
    }
    this.outputSize = outputSize;
  }

  forward(input: AnyTensor): GradTensor {
    const t = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
    const inputTensor = t.tensor;

    if (inputTensor.dtype === "string") {
      throw new DTypeError("AdaptiveAvgPool1d does not support string dtype");
    }
    if (inputTensor.ndim !== 3) {
      throw new ShapeError(
        `AdaptiveAvgPool1d expects 3D input (N, C, L); got ${inputTensor.ndim}D`
      );
    }

    const batch = inputTensor.shape[0] ?? 0;
    const channels = inputTensor.shape[1] ?? 0;
    const inL = inputTensor.shape[2] ?? 0;
    const outL = this.outputSize;

    const windows: number[][] = [];
    for (let n = 0; n < batch; n++) {
      for (let c = 0; c < channels; c++) {
        const base = (n * channels + c) * inL;
        for (let ol = 0; ol < outL; ol++) {
          const start = Math.floor((ol * inL) / outL);
          const end = Math.floor(((ol + 1) * inL) / outL);
          const win: number[] = [];
          for (let i = start; i < end; i++) win.push(base + i);
          windows.push(win);
        }
      }
    }

    return genericPool(t, [batch, channels, outL], windows, "avg");
  }

  override toString(): string {
    return `AdaptiveAvgPool1d(output_size=${this.outputSize})`;
  }
}

/**
 * 1D Adaptive Max Pooling.
 *
 * Produces fixed output length regardless of input size.
 * Input: (N, C, L) -> Output: (N, C, outputSize)
 *
 * @category Neural Network Layers
 */
export class AdaptiveMaxPool1d extends Module {
  private readonly outputSize: number;

  constructor(outputSize: number) {
    super();
    if (!Number.isInteger(outputSize) || outputSize <= 0) {
      throw new InvalidParameterError(
        "outputSize must be a positive integer",
        "outputSize",
        outputSize
      );
    }
    this.outputSize = outputSize;
  }

  forward(input: AnyTensor): GradTensor {
    const t = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
    const inputTensor = t.tensor;

    if (inputTensor.dtype === "string") {
      throw new DTypeError("AdaptiveMaxPool1d does not support string dtype");
    }
    if (inputTensor.ndim !== 3) {
      throw new ShapeError(
        `AdaptiveMaxPool1d expects 3D input (N, C, L); got ${inputTensor.ndim}D`
      );
    }

    const batch = inputTensor.shape[0] ?? 0;
    const channels = inputTensor.shape[1] ?? 0;
    const inL = inputTensor.shape[2] ?? 0;
    const outL = this.outputSize;

    const windows: number[][] = [];
    for (let n = 0; n < batch; n++) {
      for (let c = 0; c < channels; c++) {
        const base = (n * channels + c) * inL;
        for (let ol = 0; ol < outL; ol++) {
          const start = Math.floor((ol * inL) / outL);
          const end = Math.floor(((ol + 1) * inL) / outL);
          const win: number[] = [];
          for (let i = start; i < end; i++) win.push(base + i);
          windows.push(win);
        }
      }
    }

    return genericPool(t, [batch, channels, outL], windows, "max");
  }

  override toString(): string {
    return `AdaptiveMaxPool1d(output_size=${this.outputSize})`;
  }
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
 * 1D Transposed Convolution Layer (Deconvolution).
 *
 * Applies a transposed 1D convolution over an input signal.
 * Used for upsampling in sequence generation, audio synthesis, etc.
 *
 * Output size: `outL = (inL - 1) * stride - 2 * padding + kernelSize + outputPadding`
 *
 * @example
 * ```ts
 * import { ConvTranspose1d } from 'deepbox/nn';
 *
 * const deconv = new ConvTranspose1d(16, 33, 3, { stride: 2 });
 * ```
 */
export class ConvTranspose1d extends Module {
  private readonly inChannels: number;
  private readonly outChannels: number;
  private readonly kernelSize: number;
  private readonly stride: number;
  private readonly padding: number;
  private readonly outputPadding: number;
  private readonly useBias: boolean;

  private weight_: GradTensor;
  private bias_: GradTensor | undefined;

  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number,
    options: {
      readonly stride?: number;
      readonly padding?: number;
      readonly outputPadding?: number;
      readonly bias?: boolean;
    } = {}
  ) {
    super();

    if (inChannels <= 0 || !Number.isInteger(inChannels)) {
      throw new InvalidParameterError(
        "inChannels must be a positive integer",
        "inChannels",
        inChannels
      );
    }
    if (outChannels <= 0 || !Number.isInteger(outChannels)) {
      throw new InvalidParameterError(
        "outChannels must be a positive integer",
        "outChannels",
        outChannels
      );
    }
    if (kernelSize <= 0 || !Number.isInteger(kernelSize)) {
      throw new InvalidParameterError(
        "kernelSize must be a positive integer",
        "kernelSize",
        kernelSize
      );
    }

    const stride = options.stride ?? 1;
    if (stride <= 0 || !Number.isInteger(stride)) {
      throw new InvalidParameterError("stride must be a positive integer", "stride", stride);
    }

    const padding = options.padding ?? 0;
    if (padding < 0 || !Number.isInteger(padding)) {
      throw new InvalidParameterError("padding must be a non-negative integer", "padding", padding);
    }

    const outputPadding = options.outputPadding ?? 0;
    if (outputPadding < 0 || !Number.isInteger(outputPadding)) {
      throw new InvalidParameterError(
        "outputPadding must be a non-negative integer",
        "outputPadding",
        outputPadding
      );
    }

    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = kernelSize;
    this.stride = stride;
    this.padding = padding;
    this.outputPadding = outputPadding;
    this.useBias = options.bias ?? true;

    // Weight shape: (inChannels, outChannels, kernelSize)
    const k = 1 / Math.sqrt(outChannels * kernelSize);
    this.weight_ = parameter(mulScalar(randn([inChannels, outChannels, kernelSize]), k));
    this.registerParameter("weight", this.weight_);

    if (this.useBias) {
      this.bias_ = parameter(mulScalar(randn([outChannels]), k));
      this.registerParameter("bias", this.bias_);
    }
  }

  forward(x: AnyTensor): GradTensor {
    const input = GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x);

    if (input.dtype === "string") {
      throw new DTypeError("String tensors are not supported");
    }
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
    const opS = this.outputPadding;

    const outL = (inL - 1) * sS - 2 * pS + kS + opS;

    const inputTensor = input.tensor;
    const weightTensor = this.weight_.tensor;
    const inFlat = denseFloat64(inputTensor);
    const wFlat = denseFloat64(weightTensor);
    const biasFlat = this.useBias && this.bias_ ? denseFloat64(this.bias_.tensor) : null;
    const outC = this.outChannels;

    const outArr = new Float64Array(batch * outC * outL);

    const inIdx = (b: number, ic: number, il: number): number => b * inC * inL + ic * inL + il;
    const wIdxF = (ic: number, oc: number, k: number): number => ic * outC * kS + oc * kS + k;
    const oIdx = (b: number, oc: number, ol: number): number => b * outC * outL + oc * outL + ol;

    for (let b = 0; b < batch; b++) {
      for (let ic = 0; ic < this.inChannels; ic++) {
        for (let il = 0; il < inL; il++) {
          const inVal = inFlat[inIdx(b, ic, il)] ?? 0;
          for (let oc = 0; oc < outC; oc++) {
            for (let k = 0; k < kS; k++) {
              const ol = il * sS - pS + k;
              if (ol >= 0 && ol < outL) {
                outArr[oIdx(b, oc, ol)]! += inVal * (wFlat[wIdxF(ic, oc, k)] ?? 0);
              }
            }
          }
        }
      }
    }

    if (biasFlat) {
      for (let b = 0; b < batch; b++) {
        for (let oc = 0; oc < outC; oc++) {
          const bv = biasFlat[oc] ?? 0;
          for (let ol = 0; ol < outL; ol++) outArr[oIdx(b, oc, ol)]! += bv;
        }
      }
    }

    const outTensor = Tensor.fromTypedArray({
      data: outArr,
      shape: [batch, outC, outL],
      dtype: "float64",
      device: inputTensor.device,
    });

    const grads: Array<[GradTensor, (g: Tensor) => Tensor]> = [
      [
        input,
        (g: Tensor): Tensor => {
          const go = denseFloat64(g);
          const gi = new Float64Array(batch * inC * inL);
          for (let b = 0; b < batch; b++) {
            for (let ic = 0; ic < this.inChannels; ic++) {
              for (let il = 0; il < inL; il++) {
                let s = 0;
                for (let oc = 0; oc < outC; oc++) {
                  for (let k = 0; k < kS; k++) {
                    const ol = il * sS - pS + k;
                    if (ol >= 0 && ol < outL) {
                      s += (go[oIdx(b, oc, ol)] ?? 0) * (wFlat[wIdxF(ic, oc, k)] ?? 0);
                    }
                  }
                }
                gi[inIdx(b, ic, il)] = s;
              }
            }
          }
          return Tensor.fromTypedArray({
            data: gi,
            shape: [batch, inC, inL],
            dtype: "float64",
            device: inputTensor.device,
          });
        },
      ],
      [
        this.weight_,
        (g: Tensor): Tensor => {
          const go = denseFloat64(g);
          const gw = new Float64Array(this.inChannels * outC * kS);
          for (let b = 0; b < batch; b++) {
            for (let ic = 0; ic < this.inChannels; ic++) {
              for (let il = 0; il < inL; il++) {
                const inVal = inFlat[inIdx(b, ic, il)] ?? 0;
                for (let oc = 0; oc < outC; oc++) {
                  for (let k = 0; k < kS; k++) {
                    const ol = il * sS - pS + k;
                    if (ol >= 0 && ol < outL) {
                      gw[wIdxF(ic, oc, k)]! += inVal * (go[oIdx(b, oc, ol)] ?? 0);
                    }
                  }
                }
              }
            }
          }
          return Tensor.fromTypedArray({
            data: gw,
            shape: [this.inChannels, outC, kS],
            dtype: "float64",
            device: inputTensor.device,
          });
        },
      ],
    ];

    if (this.useBias && this.bias_) {
      grads.push([
        this.bias_,
        (g: Tensor): Tensor => {
          const go = denseFloat64(g);
          const gb = new Float64Array(outC);
          for (let b = 0; b < batch; b++) {
            for (let oc = 0; oc < outC; oc++) {
              let s = 0;
              for (let ol = 0; ol < outL; ol++) s += go[oIdx(b, oc, ol)] ?? 0;
              gb[oc]! += s;
            }
          }
          return Tensor.fromTypedArray({
            data: gb,
            shape: [outC],
            dtype: "float64",
            device: inputTensor.device,
          });
        },
      ]);
    }

    return customOp(outTensor, grads);
  }

  get weight(): GradTensor {
    return this.weight_;
  }

  override toString(): string {
    return `ConvTranspose1d(${this.inChannels}, ${this.outChannels}, kernel_size=${this.kernelSize}, stride=${this.stride})`;
  }
}

/**
 * 3D Convolutional Layer.
 *
 * Applies a 3D convolution over an input signal composed of several input planes.
 * Used for video processing, medical imaging (CT/MRI), and 3D point clouds.
 *
 * Input: (N, C_in, D, H, W) -> Output: (N, C_out, D_out, H_out, W_out)
 *
 * @example
 * ```ts
 * import { Conv3d } from 'deepbox/nn';
 *
 * const conv = new Conv3d(1, 16, 3); // 1 input channel, 16 output channels, 3x3x3 kernel
 * ```
 */
export class Conv3d extends Module {
  private readonly inChannels: number;
  private readonly outChannels: number;
  private readonly kernelSize: [number, number, number];
  private readonly stride: [number, number, number];
  private readonly padding: [number, number, number];
  private readonly useBias: boolean;

  private weight_: GradTensor;
  private bias_: GradTensor | undefined;

  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number | [number, number, number],
    options: {
      readonly stride?: number | [number, number, number];
      readonly padding?: number | [number, number, number];
      readonly bias?: boolean;
    } = {}
  ) {
    super();
    if (inChannels <= 0 || !Number.isInteger(inChannels)) {
      throw new InvalidParameterError(
        "inChannels must be a positive integer",
        "inChannels",
        inChannels
      );
    }
    if (outChannels <= 0 || !Number.isInteger(outChannels)) {
      throw new InvalidParameterError(
        "outChannels must be a positive integer",
        "outChannels",
        outChannels
      );
    }

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
    this.padding = normalizeTriple(
      "padding",
      options.padding ?? 0,
      true,
      "a non-negative integer or a triple of non-negative integers"
    );

    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.useBias = options.bias ?? true;

    const [kD, kH, kW] = this.kernelSize;
    const k = 1 / Math.sqrt(inChannels * kD * kH * kW);
    this.weight_ = parameter(mulScalar(randn([outChannels, inChannels, kD, kH, kW]), k));
    this.registerParameter("weight", this.weight_);

    if (this.useBias) {
      this.bias_ = parameter(mulScalar(randn([outChannels]), k));
      this.registerParameter("bias", this.bias_);
    }
  }

  forward(x: AnyTensor): GradTensor {
    const input = GradTensor.isGradTensor(x) ? x : GradTensor.fromTensor(x);

    if (input.dtype === "string") {
      throw new DTypeError("String tensors are not supported");
    }
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
    const [pD, pH, pW] = this.padding;

    const outD = Math.floor((inD + 2 * pD - kD) / sD) + 1;
    const outH = Math.floor((inH + 2 * pH - kH) / sH) + 1;
    const outW = Math.floor((inW + 2 * pW - kW) / sW) + 1;

    const inputTensor = input.tensor;
    const weightTensor = this.weight_.tensor;
    const inFlat = denseFloat64(inputTensor);
    const wFlat = denseFloat64(weightTensor);
    const biasFlat = this.useBias && this.bias_ ? denseFloat64(this.bias_.tensor) : null;
    const outC = this.outChannels;

    const outArr = new Float64Array(batch * outC * outD * outH * outW);

    const inIdxF = (b: number, ic: number, id: number, ih: number, iw: number): number =>
      ((b * inC + ic) * inD + id) * inH * inW + ih * inW + iw;
    const wIdxF = (oc: number, ic: number, kd: number, kh: number, kw: number): number =>
      ((oc * this.inChannels + ic) * kD + kd) * kH * kW + kh * kW + kw;
    const oIdxF = (b: number, oc: number, od: number, oh: number, ow: number): number =>
      ((b * outC + oc) * outD + od) * outH * outW + oh * outW + ow;

    for (let b = 0; b < batch; b++) {
      for (let oc = 0; oc < outC; oc++) {
        for (let od = 0; od < outD; od++) {
          for (let oh = 0; oh < outH; oh++) {
            for (let ow = 0; ow < outW; ow++) {
              let sum = biasFlat ? (biasFlat[oc] ?? 0) : 0;
              for (let ic = 0; ic < this.inChannels; ic++) {
                for (let kd = 0; kd < kD; kd++) {
                  for (let kh = 0; kh < kH; kh++) {
                    for (let kw = 0; kw < kW; kw++) {
                      const id = od * sD + kd - pD;
                      const ih = oh * sH + kh - pH;
                      const iw = ow * sW + kw - pW;
                      if (id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                        sum +=
                          (inFlat[inIdxF(b, ic, id, ih, iw)] ?? 0) *
                          (wFlat[wIdxF(oc, ic, kd, kh, kw)] ?? 0);
                      }
                    }
                  }
                }
              }
              outArr[oIdxF(b, oc, od, oh, ow)] = sum;
            }
          }
        }
      }
    }

    const outTensor = Tensor.fromTypedArray({
      data: outArr,
      shape: [batch, outC, outD, outH, outW],
      dtype: "float64",
      device: inputTensor.device,
    });

    const grads: Array<[GradTensor, (g: Tensor) => Tensor]> = [
      [
        input,
        (g: Tensor): Tensor => {
          const go = denseFloat64(g);
          const gi = new Float64Array(batch * inC * inD * inH * inW);
          for (let b = 0; b < batch; b++) {
            for (let oc = 0; oc < outC; oc++) {
              for (let od = 0; od < outD; od++) {
                for (let oh = 0; oh < outH; oh++) {
                  for (let ow = 0; ow < outW; ow++) {
                    const gv = go[oIdxF(b, oc, od, oh, ow)] ?? 0;
                    if (gv === 0) continue;
                    for (let ic = 0; ic < this.inChannels; ic++) {
                      for (let kd = 0; kd < kD; kd++) {
                        for (let kh = 0; kh < kH; kh++) {
                          for (let kw = 0; kw < kW; kw++) {
                            const id = od * sD + kd - pD;
                            const ih = oh * sH + kh - pH;
                            const iw = ow * sW + kw - pW;
                            if (id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                              gi[inIdxF(b, ic, id, ih, iw)]! +=
                                gv * (wFlat[wIdxF(oc, ic, kd, kh, kw)] ?? 0);
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          return Tensor.fromTypedArray({
            data: gi,
            shape: [batch, inC, inD, inH, inW],
            dtype: "float64",
            device: inputTensor.device,
          });
        },
      ],
      [
        this.weight_,
        (g: Tensor): Tensor => {
          const go = denseFloat64(g);
          const gw = new Float64Array(outC * this.inChannels * kD * kH * kW);
          for (let b = 0; b < batch; b++) {
            for (let oc = 0; oc < outC; oc++) {
              for (let od = 0; od < outD; od++) {
                for (let oh = 0; oh < outH; oh++) {
                  for (let ow = 0; ow < outW; ow++) {
                    const gv = go[oIdxF(b, oc, od, oh, ow)] ?? 0;
                    if (gv === 0) continue;
                    for (let ic = 0; ic < this.inChannels; ic++) {
                      for (let kd = 0; kd < kD; kd++) {
                        for (let kh = 0; kh < kH; kh++) {
                          for (let kw = 0; kw < kW; kw++) {
                            const id = od * sD + kd - pD;
                            const ih = oh * sH + kh - pH;
                            const iw = ow * sW + kw - pW;
                            if (id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                              gw[wIdxF(oc, ic, kd, kh, kw)]! +=
                                gv * (inFlat[inIdxF(b, ic, id, ih, iw)] ?? 0);
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          return Tensor.fromTypedArray({
            data: gw,
            shape: [outC, this.inChannels, kD, kH, kW],
            dtype: "float64",
            device: inputTensor.device,
          });
        },
      ],
    ];

    if (this.useBias && this.bias_) {
      grads.push([
        this.bias_,
        (g: Tensor): Tensor => {
          const go = denseFloat64(g);
          const gb = new Float64Array(outC);
          for (let b = 0; b < batch; b++) {
            for (let oc = 0; oc < outC; oc++) {
              let s = 0;
              for (let od = 0; od < outD; od++) {
                for (let oh = 0; oh < outH; oh++) {
                  for (let ow = 0; ow < outW; ow++) s += go[oIdxF(b, oc, od, oh, ow)] ?? 0;
                }
              }
              gb[oc]! += s;
            }
          }
          return Tensor.fromTypedArray({
            data: gb,
            shape: [outC],
            dtype: "float64",
            device: inputTensor.device,
          });
        },
      ]);
    }

    return customOp(outTensor, grads);
  }

  get weight(): GradTensor {
    return this.weight_;
  }

  override toString(): string {
    return `Conv3d(${this.inChannels}, ${this.outChannels}, kernel_size=${JSON.stringify(this.kernelSize)}, stride=${JSON.stringify(this.stride)})`;
  }
}

/**
 * 3D Max Pooling Layer.
 *
 * Applies max pooling over a 3D input signal (volumetric data).
 * Input: (N, C, D, H, W) -> Output: (N, C, D_out, H_out, W_out)
 *
 * @example
 * ```ts
 * import { MaxPool3d } from 'deepbox/nn';
 *
 * const pool = new MaxPool3d(2); // 2x2x2 pooling
 * ```
 */
export class MaxPool3d extends Module {
  private readonly kernelSizeValue: [number, number, number];
  private readonly stride: [number, number, number];
  private readonly padding: [number, number, number];

  constructor(
    kernelSize: number | [number, number, number],
    options: {
      readonly stride?: number | [number, number, number];
      readonly padding?: number | [number, number, number];
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
  }

  forward(input: AnyTensor): GradTensor {
    const t = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
    const inputTensor = t.tensor;

    if (inputTensor.dtype === "string") {
      throw new DTypeError("MaxPool3d does not support string dtype");
    }
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

    const outD = Math.floor((inD + 2 * pD - kD) / sD) + 1;
    const outH = Math.floor((inH + 2 * pH - kH) / sH) + 1;
    const outW = Math.floor((inW + 2 * pW - kW) / sW) + 1;

    if (outD <= 0 || outH <= 0 || outW <= 0) {
      throw new ShapeError("MaxPool3d output dimensions must be positive");
    }

    const windows: number[][] = [];
    for (let n = 0; n < batch; n++) {
      for (let c = 0; c < channels; c++) {
        const base = (n * channels + c) * inD * inH * inW;
        for (let od = 0; od < outD; od++) {
          for (let oh = 0; oh < outH; oh++) {
            for (let ow = 0; ow < outW; ow++) {
              const win: number[] = [];
              for (let kd = 0; kd < kD; kd++) {
                for (let kh = 0; kh < kH; kh++) {
                  for (let kw = 0; kw < kW; kw++) {
                    const id = od * sD + kd - pD;
                    const ih = oh * sH + kh - pH;
                    const iw = ow * sW + kw - pW;
                    if (id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                      win.push(base + (id * inH + ih) * inW + iw);
                    }
                  }
                }
              }
              windows.push(win);
            }
          }
        }
      }
    }

    return genericPool(t, [batch, channels, outD, outH, outW], windows, "max");
  }

  override toString(): string {
    return `MaxPool3d(kernel_size=${JSON.stringify(this.kernelSizeValue)}, stride=${JSON.stringify(this.stride)})`;
  }
}

/**
 * 3D Average Pooling Layer.
 *
 * Applies average pooling over a 3D input signal (volumetric data).
 * Input: (N, C, D, H, W) -> Output: (N, C, D_out, H_out, W_out)
 *
 * @example
 * ```ts
 * import { AvgPool3d } from 'deepbox/nn';
 *
 * const pool = new AvgPool3d(2); // 2x2x2 pooling
 * ```
 */
export class AvgPool3d extends Module {
  private readonly kernelSizeValue: [number, number, number];
  private readonly stride: [number, number, number];
  private readonly padding: [number, number, number];

  constructor(
    kernelSize: number | [number, number, number],
    options: {
      readonly stride?: number | [number, number, number];
      readonly padding?: number | [number, number, number];
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
  }

  forward(input: AnyTensor): GradTensor {
    const t = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
    const inputTensor = t.tensor;

    if (inputTensor.dtype === "string") {
      throw new DTypeError("AvgPool3d does not support string dtype");
    }
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

    const outD = Math.floor((inD + 2 * pD - kD) / sD) + 1;
    const outH = Math.floor((inH + 2 * pH - kH) / sH) + 1;
    const outW = Math.floor((inW + 2 * pW - kW) / sW) + 1;

    if (outD <= 0 || outH <= 0 || outW <= 0) {
      throw new ShapeError("AvgPool3d output dimensions must be positive");
    }

    // count_include_pad=true (PyTorch default): divide by full kernel volume.
    const kernelVol = kD * kH * kW;
    const windows: number[][] = [];
    const divisor: number[] = [];
    for (let n = 0; n < batch; n++) {
      for (let c = 0; c < channels; c++) {
        const base = (n * channels + c) * inD * inH * inW;
        for (let od = 0; od < outD; od++) {
          for (let oh = 0; oh < outH; oh++) {
            for (let ow = 0; ow < outW; ow++) {
              const win: number[] = [];
              for (let kd = 0; kd < kD; kd++) {
                for (let kh = 0; kh < kH; kh++) {
                  for (let kw = 0; kw < kW; kw++) {
                    const id = od * sD + kd - pD;
                    const ih = oh * sH + kh - pH;
                    const iw = ow * sW + kw - pW;
                    if (id >= 0 && id < inD && ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                      win.push(base + (id * inH + ih) * inW + iw);
                    }
                  }
                }
              }
              windows.push(win);
              divisor.push(kernelVol);
            }
          }
        }
      }
    }

    return genericPool(t, [batch, channels, outD, outH, outW], windows, "avg", divisor);
  }

  override toString(): string {
    return `AvgPool3d(kernel_size=${JSON.stringify(this.kernelSizeValue)}, stride=${JSON.stringify(this.stride)})`;
  }
}
