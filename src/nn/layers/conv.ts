import { DTypeError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type AnyTensor, GradTensor, im2colGrad, mulScalar, parameter, randn } from "../../ndarray";
import { Module } from "../module/Module";

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

    // We use im2col for pooling.
    // To handle channels correctly (independent pooling per channel),
    // we treat (batch * channels) as the batch dimension and 1 as channel dimension.
    // Reshape: (B, C, H, W) -> (B*C, 1, H, W)
    const inputReshaped = input.reshape([batch * channels, 1, inH, inW]);

    // im2col -> (B*C, outPixels, 1 * kH * kW)
    const cols = im2colGrad(inputReshaped, [kH, kW], [sH, sW], [pH, pW]);

    // Max over kernel window (axis 2)
    // (B*C, outPixels, kH*kW) -> (B*C, outPixels)
    const maxVals = cols.max(2);

    // Calculate output dims
    const outH = Math.floor((inH + 2 * pH - kH) / sH) + 1;
    const outW = Math.floor((inW + 2 * pW - kW) / sW) + 1;

    // Reshape back: (B*C, outH*outW) -> (B, C, outH, outW)
    return maxVals.reshape([batch, channels, outH, outW]);
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
