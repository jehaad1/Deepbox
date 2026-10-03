/**
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */

import {
  type Axis,
  DTypeError,
  InvalidParameterError,
  normalizeAxis,
  ShapeError,
} from "../../core";
import { type AnyTensor, customOp, GradTensor, parameter, type Tensor } from "../../ndarray";
import { logSoftmax as gradLogSoftmax, softmax as gradSoftmax } from "../../ndarray/autograd";
import { readNumbers } from "../../ndarray/ops/_internal";
import {
  celu as celuOp,
  elu as eluOp,
  type GeluApproximation,
  gelu as geluOp,
  hardshrink as hardshrinkOp,
  hardtanh as hardtanhOp,
  leakyRelu as leakyReluOp,
  logSigmoid as logSigmoidOp,
  logSoftmax as logSoftmaxOp,
  mish as mishOp,
  relu6 as relu6Op,
  relu as reluOp,
  sigmoid as sigmoidOp,
  softmax as softmaxOp,
  softplus as softplusOp,
  softshrink as softshrinkOp,
  swish as swishOp,
  tanhshrink as tanhshrinkOp,
} from "../../ndarray/ops/activation";
import { neg as negOp } from "../../ndarray/ops/arithmetic";
import { tanh as tanhOp } from "../../ndarray/ops/trigonometry";
import { Tensor as TensorClass } from "../../ndarray/tensor/Tensor";
import { Module } from "../module/Module";
import { allPlain, type LayerDType, resolveLayerDtype, settle, toGradInput } from "./_shared";

/** Reject NaN and infinite scalar constructor arguments with a typed error. */
function requireFinite(value: number, name: string): void {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new InvalidParameterError(
      `${name} must be a finite number; received ${String(value)}`,
      name,
      value
    );
  }
}

/**
 * Give `out` the float dtype of `input` (a parameter-free layer keeps the input float
 * dtype). Non-float inputs leave `out` as the operation produced it.
 */
function keepFloat(out: Tensor, input: Tensor): Tensor {
  const dtype = input.dtype;
  const isFloat =
    dtype === "float16" || dtype === "bfloat16" || dtype === "float32" || dtype === "float64";
  return isFloat && out.dtype !== dtype ? out.astype(dtype) : out;
}

/**
 * Apply `fn` to every logical element of `input` and return a tensor of the same shape.
 * The result has the input float dtype (float32 for integer and boolean input). Reads
 * through the tensor's strides, so views and transposed tensors give the same result as
 * their contiguous copies.
 */
function mapToFloat(input: Tensor, name: string, fn: (x: number) => number): Tensor {
  const src = readNumbers(input, name);
  const out = new Float64Array(input.size);
  for (let i = 0; i < out.length; i++) {
    out[i] = fn(src[i] as number);
  }
  const wide = TensorClass.fromTypedArray({
    data: out,
    shape: input.shape,
    dtype: "float64",
    device: input.device,
  });
  // float64 stays float64; every other dtype (half precision, float32, integers, booleans)
  // gives float32 like the tensor operations, and half precision keeps its own dtype.
  const dtype = input.dtype;
  if (dtype === "float64") return wide;
  return wide.astype(dtype === "float16" || dtype === "bfloat16" ? dtype : "float32");
}

/** Clamp x / 6 + 1/2 to [0, 1]; NaN propagates. */
function hardsigmoidScalar(x: number): number {
  return Math.min(1, Math.max(0, x / 6 + 0.5));
}

/** Logistic function that never exponentiates a positive argument. */
function sigmoidScalar(x: number): number {
  if (x >= 0) return 1 / (1 + Math.exp(-x));
  const e = Math.exp(x);
  return e / (1 + e);
}

/** log(1 + exp(x)) without overflow for large x or cancellation for very negative x. */
function softplusScalar(x: number): number {
  return x > 0 ? x + Math.log1p(Math.exp(-x)) : Math.log1p(Math.exp(x));
}

/**
 * Differentiable element-wise op with an explicit derivative. The forward value is
 * `f(x)` and the backward pass multiplies the upstream gradient by `df(x)`, so the
 * result needs one pass each way and no intermediate graph nodes. Float64 input
 * stays float64; every other dtype gives float32.
 */
function unaryGrad(
  input: GradTensor,
  name: string,
  f: (x: number) => number,
  df: (x: number) => number
): GradTensor {
  const x = input.tensor;
  const dtype = x.dtype === "float64" ? "float64" : "float32";
  const src = readNumbers(x, name);
  const n = x.size;
  const out = dtype === "float64" ? new Float64Array(n) : new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = f(src[i] as number);
  }
  const outTensor = TensorClass.fromTypedArray({
    data: out,
    shape: x.shape,
    dtype,
    device: x.device,
  });
  return customOp(outTensor, [
    [
      input,
      (go: Tensor): Tensor => {
        const g = readNumbers(go, name);
        const grad = dtype === "float64" ? new Float64Array(n) : new Float32Array(n);
        for (let i = 0; i < n; i++) {
          grad[i] = (g[i] as number) * df(src[i] as number);
        }
        return TensorClass.fromTypedArray({ data: grad, shape: x.shape, dtype, device: x.device });
      },
    ],
  ]);
}

/**
 * Differentiable softplus(x) = log(1 + exp(beta * x)) / beta with an explicit
 * backward rule (sigmoid(beta * x)). Composing it from `exp` and `log` would
 * overflow to Infinity (and NaN gradients) for inputs above about 88 in float32.
 */
function softplusGrad(input: GradTensor, beta: number): GradTensor {
  return unaryGrad(
    input,
    "Softplus",
    (x) => softplusScalar(beta * x) / beta,
    (x) => sigmoidScalar(beta * x)
  );
}

/** Hardswish(x) = x * hardsigmoid(x). */
function hardswishScalar(x: number): number {
  return x * hardsigmoidScalar(x);
}

/**
 * Derivative of hardsigmoid with PyTorch's convention at the kinks: 1/6 strictly
 * between -3 and 3, and 0 at -3 and 3 and outside.
 */
function hardsigmoidDerivative(x: number): number {
  return x > -3 && x < 3 ? 1 / 6 : 0;
}

/** Derivative of hardswish with PyTorch's convention: 0 up to -3, 1 from 3 on. */
function hardswishDerivative(x: number): number {
  if (x <= -3) return 0;
  if (x < 3) return x / 3 + 0.5;
  return 1;
}

/**
 * Applies the Rectified Linear Unit (ReLU) activation function element-wise.
 *
 * ReLU(x) = max(0, x)
 *
 * @category Neural Network Layers
 */
export class ReLU extends Module {
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) return input.relu();
    return keepFloat(reluOp(input), input);
  }

  override toString(): string {
    return "ReLU()";
  }
}

/**
 * Applies the Sigmoid activation function element-wise.
 *
 * Sigmoid(x) = 1 / (1 + exp(-x))
 *
 * @category Neural Network Layers
 */
export class Sigmoid extends Module {
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) return input.sigmoid();
    return keepFloat(sigmoidOp(input), input);
  }

  override toString(): string {
    return "Sigmoid()";
  }
}

/**
 * Applies the Hyperbolic Tangent (Tanh) activation function element-wise.
 *
 * Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 *
 * @category Neural Network Layers
 */
export class Tanh extends Module {
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) return input.tanh();
    return keepFloat(tanhOp(input), input);
  }

  override toString(): string {
    return "Tanh()";
  }
}

/**
 * Applies the Leaky Rectified Linear Unit (Leaky ReLU) activation.
 *
 * LeakyReLU(x) = x if x > 0, else alpha * x
 *
 * @category Neural Network Layers
 */
export class LeakyReLU extends Module {
  private readonly alpha: number;

  /**
   * @param alpha - Slope for negative inputs (default: 0.01)
   * @throws {InvalidParameterError} If `alpha` is not a finite number
   */
  constructor(alpha = 0.01) {
    super();
    requireFinite(alpha, "alpha");
    this.alpha = alpha;
  }

  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) return input.leakyRelu(this.alpha);
    return keepFloat(leakyReluOp(input, this.alpha), input);
  }

  override toString(): string {
    return `LeakyReLU(alpha=${this.alpha})`;
  }
}

/**
 * Applies the Exponential Linear Unit (ELU) activation.
 *
 * ELU(x) = x if x > 0, else alpha * (exp(x) - 1)
 *
 * @category Neural Network Layers
 */
export class ELU extends Module {
  private readonly alpha: number;

  /**
   * @param alpha - Scale of the negative branch (default: 1.0)
   * @throws {InvalidParameterError} If `alpha` is not a finite number
   */
  constructor(alpha = 1.0) {
    super();
    requireFinite(alpha, "alpha");
    this.alpha = alpha;
  }

  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) return input.elu(this.alpha);
    return keepFloat(eluOp(input, this.alpha), input);
  }

  override toString(): string {
    return `ELU(alpha=${this.alpha})`;
  }
}

/**
 * Applies the Gaussian Error Linear Unit (GELU) activation.
 *
 * GELU(x) = x * Phi(x) where Phi is the CDF of the standard normal distribution.
 *
 * With `approximate: "tanh"` (the default) the value is computed with
 * `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`, which matches PyTorch's
 * `GELU(approximate="tanh")`. With `approximate: "none"` it is the exact erf form
 * `0.5 x (1 + erf(x / sqrt(2)))`, which is PyTorch's default `nn.GELU()`. The two
 * differ by up to about 5e-4.
 *
 * Unlike PyTorch, the default here is `"tanh"`. Pass `{ approximate: "none" }` to get
 * the values of `torch.nn.GELU()`.
 *
 * @example
 * ```ts
 * import { GELU } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const x = tensor([-1, 0, 1]);
 * new GELU().forward(x); // tanh approximation
 * new GELU({ approximate: "none" }).forward(x); // exact, same as torch.nn.GELU()
 * ```
 *
 * @category Neural Network Layers
 */
export class GELU extends Module {
  private readonly approximate: GeluApproximation;

  /**
   * @param options - `{ approximate?: "tanh" | "none" }` (default `"tanh"`). A bare
   *   `"tanh"` or `"none"` string is also accepted.
   * @throws {InvalidParameterError} If `approximate` is neither `"tanh"` nor `"none"`
   */
  constructor(options: GeluApproximation | { readonly approximate?: GeluApproximation } = {}) {
    super();
    const approximate = typeof options === "string" ? options : (options.approximate ?? "tanh");
    if (approximate !== "tanh" && approximate !== "none") {
      throw new InvalidParameterError(
        `approximate must be "tanh" or "none"; received ${String(approximate)}`,
        "approximate",
        approximate
      );
    }
    this.approximate = approximate;
  }

  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) return input.gelu(this.approximate);
    return keepFloat(geluOp(input, { approximate: this.approximate }), input);
  }

  override toString(): string {
    return this.approximate === "tanh" ? "GELU()" : `GELU(approximate="${this.approximate}")`;
  }
}

/**
 * Applies the HardTanh activation function.
 *
 * HardTanh(x) = max(minVal, min(maxVal, x))
 *
 * @category Neural Network Layers
 */
export class Hardtanh extends Module {
  private readonly minVal: number;
  private readonly maxVal: number;

  /**
   * @param minVal - Lower bound (default: -1)
   * @param maxVal - Upper bound (default: 1)
   * @throws {InvalidParameterError} If either bound is NaN or `minVal > maxVal`
   */
  constructor(minVal = -1, maxVal = 1) {
    super();
    if (typeof minVal !== "number" || Number.isNaN(minVal)) {
      throw new InvalidParameterError("minVal must be a number", "minVal", minVal);
    }
    if (typeof maxVal !== "number" || Number.isNaN(maxVal)) {
      throw new InvalidParameterError("maxVal must be a number", "maxVal", maxVal);
    }
    if (minVal > maxVal) {
      throw new InvalidParameterError(
        `minVal (${minVal}) must be <= maxVal (${maxVal})`,
        "minVal",
        minVal
      );
    }
    this.minVal = minVal;
    this.maxVal = maxVal;
  }

  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) return input.hardtanh(this.minVal, this.maxVal);
    return keepFloat(hardtanhOp(input, this.minVal, this.maxVal), input);
  }

  override toString(): string {
    return `Hardtanh(minVal=${this.minVal}, maxVal=${this.maxVal})`;
  }
}

/**
 * Applies the Tanhshrink activation function element-wise.
 *
 * Tanhshrink(x) = x - tanh(x)
 *
 * @category Neural Network Layers
 */
export class Tanhshrink extends Module {
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) return input.tanhshrink();
    return keepFloat(tanhshrinkOp(input), input);
  }

  override toString(): string {
    return "Tanhshrink()";
  }
}

/**
 * Applies the Softmin activation function.
 *
 * Softmin(x_i) = softmax(-x_i)
 *
 * @category Neural Network Layers
 */
export class Softmin extends Module {
  private readonly axis: Axis;

  /**
   * @param axis - Axis to normalize over (default: -1, the last axis)
   */
  constructor(axis: Axis = -1) {
    super();
    this.axis = axis;
  }

  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      const negInput = input.neg();
      return gradSoftmax(negInput, normalizeAxis(this.axis, negInput.ndim));
    }
    return keepFloat(softmaxOp(negOp(input), normalizeAxis(this.axis, input.ndim)), input);
  }

  override toString(): string {
    return `Softmin(axis=${this.axis})`;
  }
}

/**
 * Applies softmax over the channel dimension at every spatial location.
 *
 * Accepts `(C, H, W)` or `(N, C, H, W)` input (as PyTorch's `Softmax2d`) and
 * normalizes along the channel axis (`ndim - 3`), so the values at each pixel sum
 * to 1 across channels.
 *
 * @category Neural Network Layers
 */
export class Softmax2d extends Module {
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    const nd = input.ndim;
    if (nd !== 3 && nd !== 4) {
      throw new ShapeError(
        `Softmax2d expects a 3-D (C, H, W) or 4-D (N, C, H, W) input; got ${nd}-D with shape [${input.shape.join(", ")}]`
      );
    }
    const channelAxis = nd - 3;
    if (GradTensor.isGradTensor(input)) return gradSoftmax(input, channelAxis);
    return keepFloat(softmaxOp(input, channelAxis), input);
  }

  override toString(): string {
    return "Softmax2d()";
  }
}

/**
 * Applies the Softmax activation function.
 *
 * Softmax(x_i) = exp(x_i) / sum(exp(x_j))
 *
 * @category Neural Network Layers
 */
export class Softmax extends Module {
  private readonly axis: Axis;

  constructor(axis: Axis = -1) {
    super();
    // Store axis along which to compute softmax
    // Default -1 means last axis (typical for classification)
    this.axis = axis;
  }

  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      return gradSoftmax(input, normalizeAxis(this.axis, input.tensor.ndim));
    }
    return keepFloat(softmaxOp(input, this.axis), input);
  }

  override toString(): string {
    return `Softmax(axis=${this.axis})`;
  }
}

/**
 * Applies the Log Softmax activation function.
 *
 * LogSoftmax(x_i) = log(exp(x_i) / sum(exp(x_j)))
 *
 * @category Neural Network Layers
 */
export class LogSoftmax extends Module {
  private readonly axis: Axis;

  constructor(axis: Axis = -1) {
    super();
    // Store axis for log-softmax computation
    // More numerically stable than log(softmax(x))
    this.axis = axis;
  }

  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      return gradLogSoftmax(input, normalizeAxis(this.axis, input.tensor.ndim));
    }
    return keepFloat(logSoftmaxOp(input, this.axis), input);
  }

  override toString(): string {
    return `LogSoftmax(axis=${this.axis})`;
  }
}

/**
 * Applies the Softplus activation function.
 *
 * Softplus(x) = log(1 + exp(beta * x)) / beta
 *
 * A smooth approximation of ReLU. It is evaluated without overflow for large
 * inputs, in both the tensor and the autograd path.
 *
 * @category Neural Network Layers
 */
export class Softplus extends Module {
  private readonly beta: number;

  /**
   * @param beta - Sharpness of the transition; larger values approach ReLU (default: 1)
   * @throws {InvalidParameterError} If `beta` is not a positive finite number
   */
  constructor(beta = 1) {
    super();
    if (typeof beta !== "number" || !Number.isFinite(beta) || beta <= 0) {
      throw new InvalidParameterError("beta must be a positive finite number", "beta", beta);
    }
    this.beta = beta;
  }

  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) return softplusGrad(input, this.beta);
    if (this.beta === 1) return keepFloat(softplusOp(input), input);
    const beta = this.beta;
    return mapToFloat(input, "Softplus", (x) => softplusScalar(beta * x) / beta);
  }

  override toString(): string {
    return this.beta === 1 ? "Softplus()" : `Softplus(beta=${this.beta})`;
  }
}

/**
 * Applies the Swish activation function (also known as SiLU).
 *
 * Swish(x) = x * sigmoid(x)
 *
 * @category Neural Network Layers
 */
export class Swish extends Module {
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      // swish(x) = x * sigmoid(x), composed from autograd primitives
      return input.mul(input.sigmoid());
    }
    return keepFloat(swishOp(input), input);
  }

  override toString(): string {
    return "Swish()";
  }
}

/**
 * Applies the Mish activation function.
 *
 * Mish(x) = x * tanh(softplus(x))
 *
 * @category Neural Network Layers
 */
export class Mish extends Module {
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      // mish(x) = x * tanh(softplus(x)), with an overflow-safe softplus
      return input.mul(softplusGrad(input, 1).tanh());
    }
    return keepFloat(mishOp(input), input);
  }

  override toString(): string {
    return "Mish()";
  }
}

/**
 * SiLU (Sigmoid Linear Unit) activation, an alias for Swish.
 *
 * SiLU(x) = x * sigmoid(x)
 *
 * Standard name in PyTorch. Identical to Swish.
 *
 * @category Neural Network Layers
 */
export class SiLU extends Swish {
  override toString(): string {
    return "SiLU()";
  }
}

/**
 * SELU (Scaled Exponential Linear Unit) activation for self-normalizing networks.
 *
 * SELU(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
 *
 * With specific alpha and scale values that enable self-normalization.
 *
 * @category Neural Network Layers
 */
export class SELU extends Module {
  // Fixed constants for self-normalization
  private static readonly ALPHA = 1.6732632423543772;
  private static readonly SCALE = 1.0507009873554805;

  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      // selu(x) = scale * elu(x, alpha)
      const eluResult = input.elu(SELU.ALPHA);
      const scaleT = GradTensor.scalar(SELU.SCALE, {
        dtype: input.dtype === "float64" ? "float64" : "float32",
      });
      return eluResult.mul(scaleT);
    }
    return mapToFloat(input, "SELU", (x) => SELU.SCALE * (x > 0 ? x : SELU.ALPHA * Math.expm1(x)));
  }

  override toString(): string {
    return "SELU()";
  }
}

/**
 * Hardsigmoid activation: piecewise linear approximation of sigmoid.
 *
 * Hardsigmoid(x) = clamp(x/6 + 0.5, 0, 1)
 *
 * Efficient for mobile/embedded (MobileNetV3).
 *
 * @category Neural Network Layers
 */
export class Hardsigmoid extends Module {
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      return unaryGrad(input, "Hardsigmoid", hardsigmoidScalar, hardsigmoidDerivative);
    }
    return mapToFloat(input, "Hardsigmoid", hardsigmoidScalar);
  }

  override toString(): string {
    return "Hardsigmoid()";
  }
}

/**
 * Hardswish activation: piecewise linear approximation of swish.
 *
 * Hardswish(x) = x * Hardsigmoid(x) = x * clamp(x/6 + 0.5, 0, 1)
 *
 * Efficient for mobile/embedded (MobileNetV3).
 *
 * @category Neural Network Layers
 */
export class Hardswish extends Module {
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      return unaryGrad(input, "Hardswish", hardswishScalar, hardswishDerivative);
    }
    return mapToFloat(input, "Hardswish", hardswishScalar);
  }

  override toString(): string {
    return "Hardswish()";
  }
}

/**
 * PReLU (Parametric ReLU) activation with learnable slope parameter.
 *
 * PReLU(x) = max(0, x) + a * min(0, x)
 *
 * Where `a` is learnable. With `numParameters = 1` a single slope is shared by all
 * elements. With `numParameters = C` there is one slope per channel, taken along
 * axis 1 of the input (axis 0 for 1-D input), as in PyTorch.
 *
 * The slope is a parameter named `weight` in the layer dtype (`float32` unless the `dtype`
 * option or the global default dtype says otherwise). The layer computes in that dtype and
 * casts the input to it (differentiably). A `GradTensor` input gives a `GradTensor`; a plain
 * `Tensor` input gives a `GradTensor` while the slope requires grad and gradient tracking
 * is on, and a plain `Tensor` otherwise.
 *
 * @category Neural Network Layers
 */
export class PReLU extends Module {
  /** Hand-written host kernels: the weights stay in host memory under `to(device)`. */
  protected override keepsParametersOnHost(): boolean {
    return true;
  }

  private readonly a: GradTensor;
  private readonly numParameters: number;

  /**
   * @param numParameters - Number of learnable slopes: 1, or the number of channels (default: 1)
   * @param init - Initial value of every slope (default: 0.25)
   * @param options.dtype - Dtype of the slope (default: the global default dtype)
   * @throws {InvalidParameterError} If `numParameters` is not a positive integer or `init` is not finite
   */
  constructor(numParameters = 1, init = 0.25, options: { readonly dtype?: LayerDType } = {}) {
    super();

    if (!Number.isInteger(numParameters) || numParameters <= 0) {
      throw new InvalidParameterError(
        "numParameters must be a positive integer",
        "numParameters",
        numParameters
      );
    }
    requireFinite(init, "init");

    this.numParameters = numParameters;

    // Initialize with constant `init` value, in the layer dtype (like Linear and Conv).
    const dtype = resolveLayerDtype(options.dtype);
    const data =
      dtype === "float64" ? new Float64Array(numParameters) : new Float32Array(numParameters);
    data.fill(init);
    const initTensor = TensorClass.fromTypedArray({
      data,
      shape: [numParameters],
      dtype,
      device: "cpu",
    });
    this.a = parameter(initTensor);
    this.registerParameter("weight", this.a);
  }

  /**
   * @param input - Tensor or GradTensor of any rank (rank >= 1 when `numParameters > 1`)
   * @returns Activated values of the same shape, in the slope dtype
   * @throws {DTypeError} For string tensors
   * @throws {ShapeError} If `numParameters > 1` and the channel axis of `input` does not have
   *   exactly `numParameters` entries
   */
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): AnyTensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: AnyTensor): AnyTensor {
    return settle(this.run(input), allPlain(input));
  }

  private run(input: AnyTensor): GradTensor {
    if (input.dtype === "string") {
      throw new DTypeError("PReLU does not support string dtype");
    }
    let t = toGradInput(input);
    const paramDtype = this.a.dtype;
    if (paramDtype !== "string" && t.dtype !== paramDtype) {
      // Align the input dtype with the learnable parameter so mixed float32/float64
      // inputs work (mirrors Linear's input-cast behavior). astype is differentiable,
      // so gradients still reach whatever produced the input.
      t = t.astype(paramDtype);
    }

    const ndim = t.ndim;
    const n = this.numParameters;
    const slopeShape = new Array<number>(ndim).fill(1);
    if (n > 1) {
      if (ndim === 0) {
        throw new ShapeError(`PReLU with ${n} parameters needs at least a 1-D input; got a scalar`);
      }
      const channelAxis = ndim === 1 ? 0 : 1;
      const channels = t.shape[channelAxis] ?? 0;
      if (channels !== n) {
        throw new ShapeError(
          `PReLU has ${n} parameters but axis ${channelAxis} of the input has size ${channels} ` +
            `(input shape [${t.shape.join(", ")}])`
        );
      }
      slopeShape[channelAxis] = n;
    }
    const slope = this.a.reshape(slopeShape);

    // PReLU(x) = x for x > 0 and a * x otherwise, written as x * (pos + (1 - pos) * a)
    // with a constant 0/1 mask. At x = 0 the gradient with respect to x is `a`, as in PyTorch.
    const xs = readNumbers(t.tensor, "PReLU");
    const dtype = t.dtype === "float64" ? "float64" : "float32";
    const pos = dtype === "float64" ? new Float64Array(xs.length) : new Float32Array(xs.length);
    const neg = dtype === "float64" ? new Float64Array(xs.length) : new Float32Array(xs.length);
    for (let i = 0; i < xs.length; i++) {
      const positive = (xs[i] as number) > 0;
      pos[i] = positive ? 1 : 0;
      neg[i] = positive ? 0 : 1;
    }
    const maskOf = (data: Float32Array | Float64Array): GradTensor =>
      GradTensor.fromTensor(
        TensorClass.fromTypedArray({
          data,
          shape: t.shape,
          dtype,
          device: t.tensor.device,
        }),
        { requiresGrad: false }
      );
    return t.mul(maskOf(pos).add(maskOf(neg).mul(slope)));
  }

  override toString(): string {
    return `PReLU(num_parameters=${this.numParameters})`;
  }
}

/**
 * GLU (Gated Linear Unit) activation.
 *
 * GLU(a, b) = a ⊗ σ(b) where [a, b] = split(input, dim)
 *
 * Splits the input tensor in half along `dim` and applies sigmoid gating.
 * Used in modern NLP architectures.
 *
 * @category Neural Network Layers
 */
export class GLU extends Module {
  private readonly dim: number;

  /**
   * @param dim - Axis to split in half (default: -1, the last axis)
   * @throws {InvalidParameterError} If `dim` is not an integer
   */
  constructor(dim = -1) {
    super();
    if (!Number.isInteger(dim)) {
      throw new InvalidParameterError("dim must be an integer", "dim", dim);
    }
    this.dim = dim;
  }

  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: AnyTensor): AnyTensor {
    return settle(this.run(input), allPlain(input));
  }

  private run(input: AnyTensor): GradTensor {
    const t = toGradInput(input);
    const ndim = t.ndim;
    const resolvedDim = this.dim < 0 ? ndim + this.dim : this.dim;

    if (resolvedDim < 0 || resolvedDim >= ndim) {
      throw new ShapeError(`GLU dim ${this.dim} out of range for ${ndim}-D tensor`);
    }

    const dimSize = t.shape[resolvedDim] ?? 0;
    if (dimSize % 2 !== 0) {
      throw new ShapeError(`GLU requires even size along dim ${resolvedDim}; got ${dimSize}`);
    }

    const halfSize = dimSize / 2;

    // Build slice ranges to select the two halves directly along resolvedDim.
    // For each dimension: keep all elements except along the split dimension,
    // where we select [0, halfSize) for the first half and [halfSize, dimSize)
    // for the second half.
    const firstHalfRanges: { readonly start: number; readonly end: number }[] = [];
    const secondHalfRanges: { readonly start: number; readonly end: number }[] = [];
    for (let i = 0; i < ndim; i++) {
      if (i === resolvedDim) {
        firstHalfRanges.push({ start: 0, end: halfSize });
        secondHalfRanges.push({ start: halfSize, end: dimSize });
      } else {
        const sz = t.shape[i] ?? 0;
        firstHalfRanges.push({ start: 0, end: sz });
        secondHalfRanges.push({ start: 0, end: sz });
      }
    }

    const aHalf = t.slice(...firstHalfRanges);
    const bHalf = t.slice(...secondHalfRanges);

    return aHalf.mul(bHalf.sigmoid());
  }

  override toString(): string {
    return `GLU(dim=${this.dim})`;
  }
}

/**
 * Softsign activation function.
 *
 * Softsign(x) = x / (1 + |x|)
 *
 * @category Neural Network Layers
 */
export class Softsign extends Module {
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      const dtype = input.dtype === "float64" ? "float64" : "float32";
      const one = GradTensor.scalar(1, { dtype });
      return input.div(one.add(input.abs()));
    }
    return mapToFloat(input, "Softsign", (x) => x / (1 + Math.abs(x)));
  }

  override toString(): string {
    return "Softsign()";
  }
}

/**
 * Applies the ReLU6 activation: `min(max(0, x), 6)`.
 *
 * Used by MobileNet. NaN propagates. The gradient is 1 strictly between 0 and 6 and 0
 * elsewhere, including at the two kinks (as in PyTorch).
 *
 * @example
 * ```ts
 * import { ReLU6 } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * new ReLU6().forward(tensor([-2, 3, 8])); // [0, 3, 6]
 * ```
 *
 * @category Neural Network Layers
 */
export class ReLU6 extends Module {
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      return unaryGrad(
        input,
        "ReLU6",
        (x) => Math.min(Math.max(x, 0), 6),
        (x) => (x > 0 && x < 6 ? 1 : 0)
      );
    }
    return keepFloat(relu6Op(input), input);
  }

  override toString(): string {
    return "ReLU6()";
  }
}

/**
 * Applies the LogSigmoid activation: `log(1 / (1 + exp(-x)))`.
 *
 * Computed as `min(x, 0) - log1p(exp(-|x|))`, so it neither overflows for very negative
 * inputs nor loses precision for large positive ones.
 *
 * @example
 * ```ts
 * import { LogSigmoid } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * new LogSigmoid().forward(tensor([-1, 0, 1])); // [-1.3133, -0.6931, -0.3133]
 * ```
 *
 * @category Neural Network Layers
 */
export class LogSigmoid extends Module {
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      return unaryGrad(
        input,
        "LogSigmoid",
        (x) => Math.min(x, 0) - Math.log1p(Math.exp(-Math.abs(x))),
        (x) => sigmoidScalar(-x)
      );
    }
    return keepFloat(logSigmoidOp(input), input);
  }

  override toString(): string {
    return "LogSigmoid()";
  }
}

/**
 * Applies the CELU activation (continuously differentiable ELU).
 *
 * CELU(x) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1))
 *
 * @example
 * ```ts
 * import { CELU } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * new CELU(2).forward(tensor([-1, 0, 1])); // [-0.7869, 0, 1]
 * ```
 *
 * @category Neural Network Layers
 */
export class CELU extends Module {
  private readonly alpha: number;

  /**
   * @param alpha - Scale of the negative branch (default: 1.0); must be finite and non-zero
   * @throws {InvalidParameterError} If `alpha` is zero, NaN or infinite
   */
  constructor(alpha = 1.0) {
    super();
    requireFinite(alpha, "alpha");
    if (alpha === 0) {
      throw new InvalidParameterError("alpha must be non-zero", "alpha", alpha);
    }
    this.alpha = alpha;
  }

  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      const alpha = this.alpha;
      return unaryGrad(
        input,
        "CELU",
        (x) => (x > 0 ? x : alpha * Math.expm1(x / alpha)),
        (x) => (x > 0 ? 1 : Math.exp(x / alpha))
      );
    }
    return keepFloat(celuOp(input, this.alpha), input);
  }

  override toString(): string {
    return `CELU(alpha=${this.alpha})`;
  }
}

/** Validate the `lambd` threshold of the shrink activations. */
function requireLambda(lambd: number): void {
  if (typeof lambd !== "number" || Number.isNaN(lambd) || lambd < 0) {
    throw new InvalidParameterError(
      `lambd must be a non-negative number; received ${String(lambd)}`,
      "lambd",
      lambd
    );
  }
}

/**
 * Applies the Softshrink activation.
 *
 * Softshrink(x) = x - lambd if x > lambd, x + lambd if x < -lambd, otherwise 0.
 *
 * @example
 * ```ts
 * import { Softshrink } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * new Softshrink().forward(tensor([-2, -0.3, 0.3, 2])); // [-1.5, 0, 0, 1.5]
 * ```
 *
 * @category Neural Network Layers
 */
export class Softshrink extends Module {
  private readonly lambd: number;

  /**
   * @param lambd - Threshold (default: 0.5); must be non-negative
   * @throws {InvalidParameterError} If `lambd` is negative or NaN
   */
  constructor(lambd = 0.5) {
    super();
    requireLambda(lambd);
    this.lambd = lambd;
  }

  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      const l = this.lambd;
      return unaryGrad(
        input,
        "Softshrink",
        (x) => (x > l ? x - l : x < -l ? x + l : Number.isNaN(x) ? x : 0),
        (x) => (Math.abs(x) > l ? 1 : 0)
      );
    }
    return keepFloat(softshrinkOp(input, this.lambd), input);
  }

  override toString(): string {
    return `Softshrink(lambd=${this.lambd})`;
  }
}

/**
 * Applies the Hardshrink activation.
 *
 * Hardshrink(x) = x if |x| > lambd, otherwise 0.
 *
 * @example
 * ```ts
 * import { Hardshrink } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * new Hardshrink().forward(tensor([-2, -0.3, 0.3, 2])); // [-2, 0, 0, 2]
 * ```
 *
 * @category Neural Network Layers
 */
export class Hardshrink extends Module {
  private readonly lambd: number;

  /**
   * @param lambd - Threshold (default: 0.5); must be non-negative
   * @throws {InvalidParameterError} If `lambd` is negative or NaN
   */
  constructor(lambd = 0.5) {
    super();
    requireLambda(lambd);
    this.lambd = lambd;
  }

  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      const l = this.lambd;
      return unaryGrad(
        input,
        "Hardshrink",
        (x) => (x >= -l && x <= l ? 0 : x),
        (x) => (Math.abs(x) > l ? 1 : 0)
      );
    }
    return keepFloat(hardshrinkOp(input, this.lambd), input);
  }

  override toString(): string {
    return `Hardshrink(lambd=${this.lambd})`;
  }
}

/**
 * Replaces every element that is not above a threshold with a fixed value.
 *
 * Threshold(x) = x if x > threshold, otherwise `value`. NaN passes through unchanged.
 * The gradient is 1 where `x > threshold` and 0 elsewhere.
 *
 * @example
 * ```ts
 * import { Threshold } from 'deepbox/nn';
 * import { tensor } from 'deepbox/ndarray';
 *
 * new Threshold(0.1, 20).forward(tensor([0, 0.1, 0.5])); // [20, 20, 0.5]
 * ```
 *
 * @category Neural Network Layers
 */
export class Threshold extends Module {
  private readonly threshold: number;
  private readonly value: number;

  /**
   * @param threshold - Values at or below it are replaced; must not be NaN
   * @param value - Replacement value; must not be NaN
   * @throws {InvalidParameterError} If `threshold` or `value` is NaN or not a number
   */
  constructor(threshold: number, value: number) {
    super();
    for (const [name, v] of [
      ["threshold", threshold],
      ["value", value],
    ] as const) {
      if (typeof v !== "number" || Number.isNaN(v)) {
        throw new InvalidParameterError(`${name} must be a number; received ${String(v)}`, name, v);
      }
    }
    this.threshold = threshold;
    this.value = value;
  }

  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: AnyTensor): AnyTensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    const th = this.threshold;
    const value = this.value;
    if (GradTensor.isGradTensor(input)) {
      return unaryGrad(
        input,
        "Threshold",
        (x) => (x <= th ? value : x),
        (x) => (x <= th ? 0 : 1)
      );
    }
    return mapToFloat(input, "Threshold", (x) => (x <= th ? value : x));
  }

  override toString(): string {
    return `Threshold(threshold=${this.threshold}, value=${this.value})`;
  }
}
