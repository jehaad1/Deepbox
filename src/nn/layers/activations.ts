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
import { type AnyTensor, GradTensor, parameter, type Tensor } from "../../ndarray";
import { logSoftmax as gradLogSoftmax, softmax as gradSoftmax } from "../../ndarray/autograd";
import { readAsNumber, requireNumericData } from "../../ndarray/ops/_internal";
import {
  elu as eluOp,
  gelu as geluOp,
  hardtanh as hardtanhOp,
  leakyRelu as leakyReluOp,
  logSoftmax as logSoftmaxOp,
  mish as mishOp,
  relu as reluOp,
  sigmoid as sigmoidOp,
  softmax as softmaxOp,
  softplus as softplusOp,
  swish as swishOp,
  tanhshrink as tanhshrinkOp,
} from "../../ndarray/ops/activation";
import { neg as negOp } from "../../ndarray/ops/arithmetic";
import { tanh as tanhOp } from "../../ndarray/ops/trigonometry";
import { reshape } from "../../ndarray/tensor/shape";
import { Tensor as TensorClass } from "../../ndarray/tensor/Tensor";
import { Module } from "../module/Module";

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
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) return input.relu();
    return reluOp(input);
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
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) return input.sigmoid();
    return sigmoidOp(input);
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
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) return input.tanh();
    return tanhOp(input);
  }

  override toString(): string {
    return "Tanh()";
  }
}

/**
 * Applies the Leaky Rectified Linear Unit (Leaky ReLU) activation.
 *
 * LeakyReLU(x) = max(alpha * x, x)
 *
 * @category Neural Network Layers
 */
export class LeakyReLU extends Module {
  private readonly alpha: number;

  constructor(alpha = 0.01) {
    super();
    this.alpha = alpha;
  }

  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) return input.leakyRelu(this.alpha);
    return leakyReluOp(input, this.alpha);
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

  constructor(alpha = 1.0) {
    super();
    // Store alpha parameter for negative values
    // ELU can produce negative outputs, pushing mean activations closer to zero
    this.alpha = alpha;
  }

  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) return input.elu(this.alpha);
    return eluOp(input, this.alpha);
  }

  override toString(): string {
    return `ELU(alpha=${this.alpha})`;
  }
}

/**
 * Applies the Gaussian Error Linear Unit (GELU) activation.
 *
 * GELU(x) = x * Phi(x) where Phi is the CDF of standard normal distribution
 *
 * @category Neural Network Layers
 */
export class GELU extends Module {
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) return input.gelu();
    return geluOp(input);
  }

  override toString(): string {
    return "GELU()";
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
  private minVal: number;
  private maxVal: number;

  constructor(minVal = -1, maxVal = 1) {
    super();
    this.minVal = minVal;
    this.maxVal = maxVal;
  }

  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) return input.hardtanh(this.minVal, this.maxVal);
    return hardtanhOp(input, this.minVal, this.maxVal);
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
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) return input.tanhshrink();
    return tanhshrinkOp(input);
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
  private axis: number;

  constructor(axis = -1) {
    super();
    this.axis = axis;
  }

  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      const negInput = input.neg();
      return gradSoftmax(negInput, normalizeAxis(this.axis, negInput.ndim));
    }
    const nd = (input as Tensor).ndim;
    return softmaxOp(negOp(input as Tensor), normalizeAxis(this.axis, nd));
  }

  override toString(): string {
    return `Softmin(axis=${this.axis})`;
  }
}

/**
 * Applies the Softmax2d activation function over spatial dimensions.
 *
 * Softmax2d(x) = softmax(reshape(x, [B*C, H*W]), dim=-1) reshaped back
 *
 * @category Neural Network Layers
 */
export class Softmax2d extends Module {
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    // Softmax2d normalizes over the CHANNEL dimension at each spatial
    // location (PyTorch semantics): softmax along axis 1 of [B, C, H*W].
    if (GradTensor.isGradTensor(input)) {
      const shape = input.shape;
      const B = shape[0] ?? 1;
      const C = shape[1] ?? 1;
      const H = shape[2] ?? 1;
      const W = shape[3] ?? 1;
      const flat = input.reshape([B, C, H * W]);
      const sm = gradSoftmax(flat, 1);
      return sm.reshape([B, C, H, W]);
    }
    const t = input as Tensor;
    const shape = t.shape;
    const B = shape[0] ?? 1;
    const C = shape[1] ?? 1;
    const H = shape[2] ?? 1;
    const W = shape[3] ?? 1;
    const flat = reshape(t, [B, C, H * W]);
    const sm = softmaxOp(flat, 1);
    return reshape(sm, [B, C, H, W]);
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
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      return gradSoftmax(input, normalizeAxis(this.axis, input.tensor.ndim));
    }
    return softmaxOp(input, this.axis);
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
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      return gradLogSoftmax(input, normalizeAxis(this.axis, input.tensor.ndim));
    }
    return logSoftmaxOp(input, this.axis);
  }

  override toString(): string {
    return `LogSoftmax(axis=${this.axis})`;
  }
}

/**
 * Applies the Softplus activation function.
 *
 * Softplus(x) = log(1 + exp(x))
 *
 * @category Neural Network Layers
 */
export class Softplus extends Module {
  forward(input: GradTensor): GradTensor;
  forward(input: Tensor): Tensor;
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      // softplus(x) = log(1 + exp(x)), composed from autograd primitives
      const one = GradTensor.scalar(1, {
        dtype: input.dtype === "float64" ? "float64" : "float32",
      });
      return one.add(input.exp()).log();
    }
    return softplusOp(input);
  }

  override toString(): string {
    return "Softplus()";
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
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      // swish(x) = x * sigmoid(x), composed from autograd primitives
      return input.mul(input.sigmoid());
    }
    return swishOp(input);
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
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      // mish(x) = x * tanh(softplus(x)), composed from autograd primitives
      const one = GradTensor.scalar(1, {
        dtype: input.dtype === "float64" ? "float64" : "float32",
      });
      const sp = one.add(input.exp()).log(); // softplus
      return input.mul(sp.tanh());
    }
    return mishOp(input);
  }

  override toString(): string {
    return "Mish()";
  }
}

/**
 * SiLU (Sigmoid Linear Unit) activation — alias for Swish.
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
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      // selu(x) = scale * elu(x, alpha)
      const eluResult = input.elu(SELU.ALPHA);
      const scaleT = GradTensor.scalar(SELU.SCALE, {
        dtype: input.dtype === "float64" ? "float64" : "float32",
      });
      return eluResult.mul(scaleT);
    }
    // Tensor path: manually compute
    const eluResult = eluOp(input, SELU.ALPHA);
    const data = requireNumericData(eluResult.data, "SELU");
    const outData = new Float64Array(eluResult.size);
    for (let i = 0; i < eluResult.size; i++) {
      outData[i] = SELU.SCALE * readAsNumber(data, i);
    }
    return TensorClass.fromTypedArray({
      data: outData,
      shape: eluResult.shape,
      dtype: "float64",
      device: eluResult.device,
    });
  }

  override toString(): string {
    return "SELU()";
  }
}

/**
 * Hardsigmoid activation — piecewise linear approximation of sigmoid.
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
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      const dtype = input.dtype === "float64" ? "float64" : "float32";
      const sixth = GradTensor.scalar(1 / 6, { dtype });
      const half = GradTensor.scalar(0.5, { dtype });
      const shifted = input.mul(sixth).add(half);
      return shifted.clip(0, 1);
    }
    const data = requireNumericData(input.data, "Hardsigmoid");
    const outData = new Float64Array(input.size);
    for (let i = 0; i < input.size; i++) {
      const x = readAsNumber(data, i);
      outData[i] = Math.min(1, Math.max(0, x / 6 + 0.5));
    }
    return TensorClass.fromTypedArray({
      data: outData,
      shape: input.shape,
      dtype: "float64",
      device: input.device,
    });
  }

  override toString(): string {
    return "Hardsigmoid()";
  }
}

/**
 * Hardswish activation — piecewise linear approximation of swish.
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
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      const dtype = input.dtype === "float64" ? "float64" : "float32";
      const sixth = GradTensor.scalar(1 / 6, { dtype });
      const half = GradTensor.scalar(0.5, { dtype });
      const hs = input.mul(sixth).add(half).clip(0, 1);
      return input.mul(hs);
    }
    const data = requireNumericData(input.data, "Hardswish");
    const outData = new Float64Array(input.size);
    for (let i = 0; i < input.size; i++) {
      const x = readAsNumber(data, i);
      const hs = Math.min(1, Math.max(0, x / 6 + 0.5));
      outData[i] = x * hs;
    }
    return TensorClass.fromTypedArray({
      data: outData,
      shape: input.shape,
      dtype: "float64",
      device: input.device,
    });
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
 * Where `a` is a learnable parameter.
 *
 * @category Neural Network Layers
 */
export class PReLU extends Module {
  private a: GradTensor;

  constructor(numParameters = 1, init = 0.25) {
    super();

    if (!Number.isInteger(numParameters) || numParameters <= 0) {
      throw new InvalidParameterError(
        "numParameters must be a positive integer",
        "numParameters",
        numParameters
      );
    }

    // Initialize with constant `init` value. The learnable slope uses the
    // framework's default parameter dtype (float32), consistent with Linear/Conv
    // and `randn`, so the common float32 path needs no cast in forward.
    const data = new Float32Array(numParameters);
    data.fill(init);
    const initTensor = TensorClass.fromTypedArray({
      data,
      shape: [numParameters],
      dtype: "float32",
      device: "cpu",
    });
    this.a = parameter(initTensor);
    this.registerParameter("weight", this.a);
  }

  forward(input: AnyTensor): GradTensor {
    let t = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
    const paramDtype = this.a.tensor.dtype;
    if (t.tensor.dtype !== paramDtype) {
      if (t.tensor.dtype === "string") {
        throw new DTypeError("PReLU does not support string dtype");
      }
      // Align the input dtype with the learnable parameter so mixed float32/float64
      // inputs work (mirrors Linear's input-cast behavior). Flatten first to read
      // logical values in contiguous order, which is correct for strided views.
      const flat = t.tensor.flatten();
      const castData =
        paramDtype === "float64"
          ? new Float64Array(flat.data as ArrayLike<number>)
          : new Float32Array(flat.data as ArrayLike<number>);
      const castTensor = reshape(
        TensorClass.fromTypedArray({
          data: castData,
          shape: [flat.size],
          dtype: paramDtype as "float32" | "float64",
          device: t.tensor.device,
        }),
        t.tensor.shape
      );
      t = GradTensor.fromTensor(castTensor, {
        requiresGrad: GradTensor.isGradTensor(input) ? input.requiresGrad : false,
      });
    }
    // PReLU(x) = max(0,x) + a * min(0,x) = relu(x) - a * relu(-x)
    const pos = t.relu();
    const negated = t.neg();
    const negPart = negated.relu();
    return pos.sub(this.a.mul(negPart));
  }

  override toString(): string {
    return `PReLU(num_parameters=${this.a.tensor.size})`;
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

  constructor(dim = -1) {
    super();
    this.dim = dim;
  }

  forward(input: AnyTensor): GradTensor {
    const t = GradTensor.isGradTensor(input) ? input : GradTensor.fromTensor(input);
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
  forward(input: Tensor | GradTensor): Tensor | GradTensor {
    if (GradTensor.isGradTensor(input)) {
      const dtype = input.dtype === "float64" ? "float64" : "float32";
      const one = GradTensor.scalar(1, { dtype });
      // x / (1 + |x|) — use abs via relu trick: |x| = relu(x) + relu(-x)
      const absX = input.relu().add(input.neg().relu());
      return input.div(one.add(absX));
    }
    const data = requireNumericData(input.data, "Softsign");
    const outData = new Float64Array(input.size);
    for (let i = 0; i < input.size; i++) {
      const x = readAsNumber(data, i);
      outData[i] = x / (1 + Math.abs(x));
    }
    return TensorClass.fromTypedArray({
      data: outData,
      shape: input.shape,
      dtype: "float64",
      device: input.device,
    });
  }

  override toString(): string {
    return "Softsign()";
  }
}
