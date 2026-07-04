/**
 * Lion (EvoLved Sign Momentum) optimizer.
 *
 * Introduced by Chen et al. (2023) in "Symbolic Discovery of Optimization
 * Algorithms". Lion is a simpler, more memory-efficient alternative to Adam
 * that uses only a single momentum buffer and sign-based updates.
 *
 * **Update rule** (per parameter):
 * ```
 * u = sign(beta1 * m + (1 - beta1) * g)
 * theta -= lr * (u + weightDecay * theta)
 * m = beta2 * m + (1 - beta2) * g
 * ```
 *
 * Lion typically requires 3-10× smaller learning rates than Adam
 * (suggested: 1e-4 to 3e-4, vs Adam's 1e-3).
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
 * @see {@link https://deepbox.dev/docs/optim-optimizers | Deepbox Optimizers}
 * @category Optimizers
 */

import { InvalidParameterError } from "../../core";
import { add, type GradTensor, mulScalar, sub, type Tensor } from "../../ndarray";
import {
  assertFiniteNonNegative,
  assertFinitePositive,
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
};

type LionState = {
  step: number;
  momentum: Float64Array;
  /** Device momentum buffer (used when the parameter lives on a kernel device). */
  momentumTensor?: Tensor;
};

export class Lion extends Optimizer<LionOptions, LionState> {
  private _stepCount = 0;

  get stepCount(): number {
    return this._stepCount;
  }

  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<LionOptions>>,
    options: {
      readonly lr?: number;
      readonly beta1?: number;
      readonly beta2?: number;
      readonly weightDecay?: number;
    } = {}
  ) {
    const defaults = {
      lr: options.lr ?? 1e-4,
      beta1: options.beta1 ?? 0.9,
      beta2: options.beta2 ?? 0.99,
      weightDecay: options.weightDecay ?? 0,
    };

    super(params, defaults);

    assertFinitePositive("learning rate", defaults.lr);
    assertInRange("beta1", defaults.beta1, 0, 1);
    assertInRange("beta2", defaults.beta2, 0, 1);
    assertFiniteNonNegative("weight_decay value", defaults.weightDecay);
  }

  getLearningRate(groupIdx = 0): number {
    const group = this.paramGroups[groupIdx];
    if (!group) {
      throw new InvalidParameterError(
        `Invalid group index: ${groupIdx} (valid range: [0, ${this.paramGroups.length}))`,
        "groupIdx",
        groupIdx
      );
    }
    return group.options.lr;
  }

  setLearningRate(lr: number): void {
    assertFiniteNonNegative("learning rate", lr);
    for (const group of this.paramGroups) {
      group.options.lr = lr;
    }
  }

  protected isState(state: Record<string, unknown>): state is LionState {
    return typeof state["step"] === "number" && state["momentum"] instanceof Float64Array;
  }

  step(closure?: () => number): number | undefined {
    let loss: number | undefined;

    if (closure) {
      loss = closure();
    }

    this._stepCount++;

    for (const group of this.paramGroups) {
      const { lr, beta1, beta2, weightDecay } = group.options;

      assertFiniteNonNegative("learning rate", lr);
      assertInRange("beta1", beta1, 0, 1);
      assertInRange("beta2", beta2, 0, 1);
      assertFiniteNonNegative("weight_decay value", weightDecay);

      for (const param of group.params) {
        // Device path: compose the sign-based Lion update from device-dispatched ops.
        if (param.tensor.isDeviceTensor) {
          const g = param.grad;
          if (!g) continue;
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

        state.step++;

        for (let i = 0; i < size; i++) {
          const g = gradData[gradOffset + i] ?? 0;
          const mPrev = state.momentum[i] ?? 0;

          const update = Math.sign(beta1 * mPrev + (1 - beta1) * g);
          const p = paramData[paramOffset + i] ?? 0;

          paramData[paramOffset + i] = p - lr * (update + weightDecay * p);

          state.momentum[i] = beta2 * mPrev + (1 - beta2) * g;
        }
      }
    }

    return loss;
  }
}
