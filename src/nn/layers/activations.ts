import { type Axis, normalizeAxis } from "../../core";
import { GradTensor, type Tensor } from "../../ndarray";
import { logSoftmax as gradLogSoftmax, softmax as gradSoftmax } from "../../ndarray/autograd";
import {
  elu as eluOp,
  gelu as geluOp,
  leakyRelu as leakyReluOp,
  logSoftmax as logSoftmaxOp,
  mish as mishOp,
  relu as reluOp,
  sigmoid as sigmoidOp,
  softmax as softmaxOp,
  softplus as softplusOp,
  swish as swishOp,
} from "../../ndarray/ops/activation";
import { tanh as tanhOp } from "../../ndarray/ops/trigonometry";
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
    if (input instanceof GradTensor) return input.relu();
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
    if (input instanceof GradTensor) return input.sigmoid();
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
    if (input instanceof GradTensor) return input.tanh();
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
    if (input instanceof GradTensor) return input.leakyRelu(this.alpha);
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
    if (input instanceof GradTensor) return input.elu(this.alpha);
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
    if (input instanceof GradTensor) return input.gelu();
    return geluOp(input);
  }

  override toString(): string {
    return "GELU()";
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
    if (input instanceof GradTensor) {
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
    if (input instanceof GradTensor) {
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
    if (input instanceof GradTensor) {
      return GradTensor.fromTensor(softplusOp(input.tensor), {
        requiresGrad: false,
      });
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
    if (input instanceof GradTensor) {
      return GradTensor.fromTensor(swishOp(input.tensor), {
        requiresGrad: false,
      });
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
    if (input instanceof GradTensor) {
      return GradTensor.fromTensor(mishOp(input.tensor), {
        requiresGrad: false,
      });
    }
    return mishOp(input);
  }

  override toString(): string {
    return "Mish()";
  }
}
