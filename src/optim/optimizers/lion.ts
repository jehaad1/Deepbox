/**
 * @see {@link https://deepbox.dev/docs/optim-optimizers | Deepbox documentation}
 */

import { add, type GradTensor, mulScalar, sub, type Tensor } from "../../ndarray";
import {
  assertBufferSize,
  assertFinite,
  assertFiniteNonNegative,
  assertHasGradFloat,
  assertInRange,
  deviceSign,
  replaceParamStorage,
} from "../_internal";
import { Optimizer, type ParamGroup } from "../Optimizer";

type LionOptions = {
  lr: number;
  beta1: number;
  beta2: number;
  weightDecay: number;
  maximize: boolean;
};

type LionState = {
  step: number;
  momentum: Float64Array;
  /** Device momentum buffer (used when the parameter lives on a kernel device). */
  momentumTensor?: Tensor;
};

/**
 * Lion (EvoLved Sign Momentum) optimizer.
 *
 * Introduced by Chen et al. (2023) in "Symbolic Discovery of Optimization
 * Algorithms". Lion is a simpler, more memory-efficient alternative to Adam
 * that keeps a single momentum buffer and applies sign-based updates, so every
 * element moves by the same magnitude `lr` regardless of its gradient scale.
 *
 * **Update rule** (per parameter):
 * ```
 * u = sign(beta1 * m + (1 - beta1) * g)
 * theta -= lr * (u + weightDecay * theta)
 * m = beta2 * m + (1 - beta2) * g
 * ```
 *
 * Weight decay is decoupled, as in AdamW. Lion typically needs a 3-10x smaller
 * learning rate than Adam (suggested: 1e-4 to 3e-4, versus Adam's 1e-3) and a
 * correspondingly larger weight decay.
 *
 * @example
 * ```ts
 * import { Lion } from 'deepbox/optim';
 *
 * const optimizer = new Lion(model.parameters(), {
 *   lr: 3e-4,
 *   beta1: 0.9,
 *   beta2: 0.99
 * });
 * ```
 *
 * @see Chen et al. (2023) "Symbolic Discovery of Optimization Algorithms"
 * @category Optimizers
 */
export class Lion extends Optimizer<LionOptions, LionState> {
  /**
   * Create a new Lion optimizer.
   *
   * @param params - Parameters to optimize, or an array of parameter groups with per-group options
   * @param options - Hyperparameters
   * @param options.lr - Learning rate (default: 1e-4)
   * @param options.beta1 - Interpolation factor between momentum and gradient for the update
   *   direction, in [0, 1) (default: 0.9)
   * @param options.beta2 - Decay rate of the momentum buffer, in [0, 1) (default: 0.99)
   * @param options.weightDecay - Decoupled weight decay coefficient (default: 0)
   * @param options.maximize - Maximize the objective instead of minimizing it (default: false)
   * @throws {InvalidParameterError} If a hyperparameter is out of range
   */
  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<LionOptions>>,
    options: {
      readonly lr?: number;
      readonly beta1?: number;
      readonly beta2?: number;
      readonly weightDecay?: number;
      readonly maximize?: boolean;
    } = {}
  ) {
    const defaults = {
      lr: options.lr ?? 1e-4,
      beta1: options.beta1 ?? 0.9,
      beta2: options.beta2 ?? 0.99,
      weightDecay: options.weightDecay ?? 0,
      maximize: options.maximize ?? false,
    };

    super(params, defaults);
  }

  protected override validateOptions(options: Readonly<LionOptions>): void {
    assertFiniteNonNegative("learning rate", options.lr);
    assertInRange("beta1", options.beta1, 0, 1);
    assertInRange("beta2", options.beta2, 0, 1);
    assertFiniteNonNegative("weight_decay value", options.weightDecay);
  }

  protected isState(state: Record<string, unknown>): state is LionState {
    return typeof state["step"] === "number" && state["momentum"] instanceof Float64Array;
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

    this.prepareStep("Lion");
    this.countStep();

    for (const group of this.paramGroups) {
      const { lr, beta1, beta2, weightDecay, maximize } = group.options;

      for (const param of this.trainableParams(group)) {
        // Device path: compose the sign-based Lion update from device-dispatched ops.
        if (param.tensor.isDeviceTensor) {
          const rawGrad = param.grad;
          if (!rawGrad) continue;
          const g = maximize ? mulScalar(rawGrad, -1) : rawGrad;
          let dstate = this.state.get(param);
          if (!dstate) {
            dstate = { step: 0, momentum: new Float64Array(0) };
            this.state.set(param, dstate);
          }
          const mPrev = dstate.momentumTensor;
          // update = sign(beta1 * m + (1 - beta1) * g); first step m=0.
          const inner = mPrev
            ? add(mulScalar(mPrev, beta1), mulScalar(g, 1 - beta1))
            : mulScalar(g, 1 - beta1);
          const update = deviceSign(inner);
          // theta -= lr * (update + weightDecay * theta)
          const step =
            weightDecay !== 0 ? add(update, mulScalar(param.tensor, weightDecay)) : update;
          replaceParamStorage(param, "tensor", sub(param.tensor, mulScalar(step, lr)));
          // m = beta2 * m + (1 - beta2) * g
          dstate.momentumTensor = mPrev
            ? add(mulScalar(mPrev, beta2), mulScalar(g, 1 - beta2))
            : mulScalar(g, 1 - beta2);
          continue;
        }

        const {
          grad: gradData,
          gradOffset,
          param: paramData,
          paramOffset,
        } = assertHasGradFloat(param, "Lion");
        const size = param.tensor.size;

        const existing = this.state.get(param);
        const state =
          existing ??
          (() => {
            const next: LionState = {
              step: 0,
              momentum: new Float64Array(size),
            };
            this.state.set(param, next);
            return next;
          })();

        assertBufferSize(state.momentum, size, "Lion momentum");
        state.step++;

        const momentum = state.momentum;
        for (let i = 0; i < size; i++) {
          const rawGi = gradData[gradOffset + i] as number;
          const p = paramData[paramOffset + i] as number;
          if (!Number.isFinite(rawGi)) assertFinite("gradient", rawGi);
          const g = maximize ? -rawGi : rawGi;
          if (!Number.isFinite(p)) assertFinite("parameter", p);
          const mPrev = momentum[i] as number;

          const update = Math.sign(beta1 * mPrev + (1 - beta1) * g);

          paramData[paramOffset + i] = p - lr * (update + weightDecay * p);

          momentum[i] = beta2 * mPrev + (1 - beta2) * g;
        }
      }
    }

    return loss;
  }
}
