import { InvalidParameterError } from "../../core";
import type { GradTensor } from "../../ndarray";
import {
  assertBufferSize,
  assertFinite,
  assertFiniteNonNegative,
  assertFinitePositive,
  assertHasGradFloat,
  safeArrayAccess,
} from "../_internal";
import { Optimizer, type ParamGroup } from "../Optimizer";

type AdagradOptions = {
  lr: number;
  eps: number;
  weightDecay: number;
  lrDecay: number;
};

type AdagradState = {
  step: number;
  sum: Float64Array;
};

/**
 * Adagrad (Adaptive Gradient Algorithm) optimizer.
 *
 * Adagrad adapts the learning rate for each parameter based on the historical
 * sum of squared gradients. Parameters with larger gradients receive smaller
 * effective learning rates, while parameters with smaller gradients receive
 * larger effective learning rates.
 *
 * @example
 * ```ts
 * import { Adagrad } from 'deepbox/optim';
 *
 * const optimizer = new Adagrad(model.parameters(), {
 *   lr: 0.01,
 *   eps: 1e-10
 * });
 *
 * // Training loop
 * for (let epoch = 0; epoch < numEpochs; epoch++) {
 *   optimizer.zeroGrad();
 *   // ...
 *   optimizer.step();
 * }
 * ```
 *
 * @category Optimizers
 */
export class Adagrad extends Optimizer<AdagradOptions, AdagradState> {
  private _stepCount = 0;

  get stepCount(): number {
    return this._stepCount;
  }
  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<AdagradOptions>>,
    options: {
      readonly lr?: number;
      readonly eps?: number;
      readonly weightDecay?: number;
      readonly lrDecay?: number;
    } = {}
  ) {
    const defaults = {
      lr: options.lr ?? 0.01,
      eps: options.eps ?? 1e-10,
      weightDecay: options.weightDecay ?? 0,
      lrDecay: options.lrDecay ?? 0,
    };

    super(params, defaults);

    assertFiniteNonNegative("learning rate", defaults.lr);
    assertFinitePositive("epsilon", defaults.eps);
    assertFiniteNonNegative("weight_decay value", defaults.weightDecay);
    assertFiniteNonNegative("lr_decay", defaults.lrDecay);
  }

  /**
   * Get the current learning rate.
   *
   * @param groupIdx - Parameter group index (default: 0)
   * @returns Current learning rate
   */
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

  /**
   * Set the learning rate for all parameter groups.
   *
   * @param lr - New learning rate
   */
  setLearningRate(lr: number): void {
    assertFiniteNonNegative("learning rate", lr);
    for (const group of this.paramGroups) {
      group.options.lr = lr;
    }
  }

  protected isState(state: Record<string, unknown>): state is AdagradState {
    return typeof state["step"] === "number" && state["sum"] instanceof Float64Array;
  }

  step(closure?: () => number): number | undefined {
    let loss: number | undefined;

    if (closure) {
      loss = closure();
    }

    // Increment global step counter
    this._stepCount++;

    for (const group of this.paramGroups) {
      const { lr, eps, weightDecay, lrDecay } = group.options;

      assertFiniteNonNegative("learning rate", lr);
      assertFinitePositive("epsilon", eps);
      assertFiniteNonNegative("weight_decay value", weightDecay);
      assertFiniteNonNegative("lr_decay", lrDecay);

      for (const param of group.params) {
        const {
          grad: gradData,
          gradOffset: gOff,
          param: pData,
          paramOffset: pOff,
        } = assertHasGradFloat(param, "Adagrad");
        const size = param.tensor.size;

        const existing = this.state.get(param);
        const state =
          existing ??
          (() => {
            const next = {
              step: 0,
              sum: new Float64Array(size),
            };
            this.state.set(param, next);
            return next;
          })();

        // Validate state buffer size
        assertBufferSize(state.sum, size, "Adagrad sum");

        state.step += 1;

        const clr = lr / (1 + (state.step - 1) * lrDecay);

        for (let i = 0; i < size; i++) {
          const gi0 = safeArrayAccess(gradData, gOff + i, "Adagrad gradient");
          const pi = safeArrayAccess(pData, pOff + i, "Adagrad parameter");
          assertFinite("gradient", gi0);
          assertFinite("parameter", pi);

          const gi = weightDecay !== 0 ? gi0 + weightDecay * pi : gi0;

          const sumVal = safeArrayAccess(state.sum, i, "Adagrad sum");
          const sumNew = sumVal + gi * gi;
          state.sum[i] = sumNew;

          const std = Math.sqrt(sumNew) + eps;
          pData[pOff + i] = pi - clr * (gi / std);
        }
      }
    }

    return loss;
  }
}
