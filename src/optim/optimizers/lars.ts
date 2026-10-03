/**
 * @see {@link https://deepbox.dev/docs/optim-optimizers | Deepbox documentation}
 */

import {
  add,
  addScalar,
  div,
  type GradTensor,
  mul,
  mulScalar,
  sqrt,
  square,
  sub,
  sum,
  type Tensor,
  tensor,
  where,
} from "../../ndarray";
import {
  assertBufferSize,
  assertFinite,
  assertFiniteNonNegative,
  assertFinitePositive,
  assertHasGradFloat,
  assertInRange,
  replaceParamStorage,
} from "../_internal";
import { Optimizer, type ParamGroup } from "../Optimizer";

/**
 * Options for the LARS optimizer.
 *
 * @property lr - Learning rate (step size)
 * @property momentum - Momentum factor
 * @property weightDecay - Weight decay coefficient (L2 penalty)
 * @property eta - LARS trust coefficient
 * @property eps - Small constant for numerical stability
 */
type LARSOptions = {
  lr: number;
  momentum: number;
  weightDecay: number;
  eta: number;
  eps: number;
};

/**
 * State maintained per parameter by LARS.
 *
 * @property momentumBuffer - Momentum buffer
 */
type LARSState = {
  momentumBuffer: Float64Array;
  /** Device momentum buffer (used when the parameter lives on a kernel device). */
  momentumTensor?: Tensor;
};

/**
 * LARS (Layer-wise Adaptive Rate Scaling) optimizer.
 *
 * LARS scales the learning rate per layer based on the ratio of the
 * parameter norm to the gradient norm. This allows training with very
 * large batch sizes (up to 32K) without divergence. Each parameter tensor is
 * treated as one layer:
 *
 * ```
 * localLr = eta * ||theta|| / (||g|| + weightDecay * ||theta|| + eps)   (1 when either norm is zero)
 * m = momentum * m + lr * localLr * (g + weightDecay * theta)
 * theta -= m
 * ```
 *
 * Reference: "Large Batch Training of Convolutional Networks" (You et al., 2017)
 *
 * @example
 * ```ts
 * import { LARS } from 'deepbox/optim';
 *
 * const optimizer = new LARS(model.parameters(), {
 *   lr: 0.1,
 *   momentum: 0.9,
 *   weightDecay: 1e-4,
 *   eta: 0.001,
 * });
 *
 * for (let epoch = 0; epoch < numEpochs; epoch++) {
 *   optimizer.zeroGrad();
 *   // ...
 *   optimizer.step();
 * }
 * ```
 *
 * @category Optimizers
 */
export class LARS extends Optimizer<LARSOptions, LARSState> {
  /**
   * Create a new LARS optimizer.
   *
   * @param params - Parameters to optimize, or an array of parameter groups with per-group options
   * @param options - Hyperparameters
   * @param options.lr - Global learning rate (default: 0.1)
   * @param options.momentum - Momentum factor, in [0, 1) (default: 0.9)
   * @param options.weightDecay - L2 penalty coefficient (default: 1e-4)
   * @param options.eta - Trust coefficient scaling the layer-wise rate (default: 0.001)
   * @param options.eps - Term added to the denominator for numerical stability (default: 1e-8)
   * @throws {InvalidParameterError} If a hyperparameter is out of range
   */
  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<LARSOptions>>,
    options: {
      readonly lr?: number;
      readonly momentum?: number;
      readonly weightDecay?: number;
      readonly eta?: number;
      readonly eps?: number;
    } = {}
  ) {
    const defaults: LARSOptions = {
      lr: options.lr ?? 0.1,
      momentum: options.momentum ?? 0.9,
      weightDecay: options.weightDecay ?? 1e-4,
      eta: options.eta ?? 0.001,
      eps: options.eps ?? 1e-8,
    };

    super(params, defaults);
  }

  protected override validateOptions(options: Readonly<LARSOptions>): void {
    assertFiniteNonNegative("learning rate", options.lr);
    assertInRange("momentum", options.momentum, 0, 1);
    assertFiniteNonNegative("weight_decay value", options.weightDecay);
    assertFinitePositive("eta", options.eta);
    assertFinitePositive("epsilon", options.eps);
  }

  protected isState(state: Record<string, unknown>): state is LARSState {
    return state["momentumBuffer"] instanceof Float64Array;
  }

  /**
   * Perform a single optimization step.
   *
   * A parameter whose gradient is `null` (it took no part in the loss) is skipped, as in PyTorch.
   *
   * @param closure - Optional function that re-evaluates the model and returns the loss
   * @returns The value returned by `closure`, or undefined when no closure is given
   * @throws {InvalidParameterError} If a gradient or parameter value is not finite
   */
  step(closure?: () => number): number | undefined {
    let loss: number | undefined;
    if (closure) {
      loss = closure();
    }

    this.prepareStep("LARS");
    this.countStep();

    for (const group of this.paramGroups) {
      const { lr, momentum, weightDecay, eta, eps } = group.options;

      for (const param of this.trainableParams(group)) {
        // Device path: compose the LARS update (layer-wise trust ratio +
        // momentum) from device-dispatched ops. The local learning rate is a
        // device scalar (norms are full reductions), broadcast into the update.
        if (param.tensor.isDeviceTensor) {
          const g = param.grad;
          if (!g) continue;
          let dstate = this.state.get(param);
          if (!dstate) {
            dstate = { momentumBuffer: new Float64Array(0) };
            this.state.set(param, dstate);
          }
          // dP = g + weightDecay * param
          const dP = weightDecay !== 0 ? add(g, mulScalar(param.tensor, weightDecay)) : g;
          const paramNorm = sqrt(sum(square(param.tensor)));
          const gradNorm = sqrt(sum(square(g)));
          // localLr = (paramNorm>0 && gradNorm>0)
          //   ? eta*paramNorm / (gradNorm + weightDecay*paramNorm + eps) : 1
          const num = mulScalar(paramNorm, eta);
          const den = addScalar(add(gradNorm, mulScalar(paramNorm, weightDecay)), eps);
          const cond = mul(paramNorm, gradNorm);
          const localLr = where(cond, div(num, den), tensor(1));
          // effectiveLr = lr * localLr (device scalar)
          const effLr = mulScalar(localLr, lr);
          const mPrev = dstate.momentumTensor;
          // mNew = momentum * mPrev + effectiveLr * dP; first step mPrev=0.
          const mNew = mPrev ? add(mulScalar(mPrev, momentum), mul(dP, effLr)) : mul(dP, effLr);
          dstate.momentumTensor = mNew;
          replaceParamStorage(param, "tensor", sub(param.tensor, mNew));
          continue;
        }

        const {
          grad,
          gradOffset,
          param: pData,
          paramOffset: pOff,
        } = assertHasGradFloat(param, "LARS");
        const size = param.tensor.size;

        const existing = this.state.get(param);
        const state =
          existing ??
          (() => {
            const next: LARSState = {
              momentumBuffer: new Float64Array(size),
            };
            this.state.set(param, next);
            return next;
          })();

        assertBufferSize(state.momentumBuffer, size, "LARS momentumBuffer");

        // Compute parameter and gradient norms
        let paramNormSq = 0;
        let gradNormSq = 0;
        for (let i = 0; i < size; i++) {
          const pi = pData[pOff + i] as number;
          const gi = grad[gradOffset + i] as number;
          if (!Number.isFinite(pi)) assertFinite("parameter", pi);
          if (!Number.isFinite(gi)) assertFinite("gradient", gi);
          paramNormSq += pi * pi;
          gradNormSq += gi * gi;
        }
        const paramNorm = Math.sqrt(paramNormSq);
        const gradNorm = Math.sqrt(gradNormSq);

        // LARS trust ratio: local_lr = eta * ||w|| / (||g|| + lambda * ||w|| + eps)
        let localLr = 1.0;
        if (paramNorm > 0 && gradNorm > 0) {
          localLr = (eta * paramNorm) / (gradNorm + weightDecay * paramNorm + eps);
        }

        const effectiveLr = lr * localLr;

        // Update with momentum and weight decay
        const momentumBuffer = state.momentumBuffer;
        for (let i = 0; i < size; i++) {
          const gi = grad[gradOffset + i] as number;
          const pi = pData[pOff + i] as number;
          const mi = momentumBuffer[i] as number;

          // Gradient with weight decay
          const dP = gi + weightDecay * pi;

          // Momentum update
          const mNew = momentum * mi + effectiveLr * dP;
          momentumBuffer[i] = mNew;

          // Parameter update
          pData[pOff + i] = pi - mNew;
        }
      }
    }

    return loss;
  }
}
