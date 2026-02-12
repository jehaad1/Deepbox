import { DeepboxError, InvalidParameterError } from "../../core";
import type { GradTensor } from "../../ndarray";
import {
  assertBufferSize,
  assertFinite,
  assertFiniteNonNegative,
  assertFinitePositive,
  assertHasGradFloat,
  assertInRange,
  safeArrayAccess,
} from "../_internal";
import { Optimizer, type ParamGroup } from "../Optimizer";

type AdamOptions = {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  amsgrad: boolean;
};

type AdamState = {
  step: number;
  expAvg: Float64Array;
  expAvgSq: Float64Array;
  maxExpAvgSq?: Float64Array;
};

/**
 * Adam (Adaptive Moment Estimation) optimizer.
 *
 * Computes adaptive learning rates for each parameter by maintaining
 * running averages of both the gradients and their squared values.
 *
 * @example
 * ```ts
 * import { Adam } from 'deepbox/optim';
 *
 * const optimizer = new Adam(model.parameters(), {
 *   lr: 0.001,
 *   beta1: 0.9,
 *   beta2: 0.999
 * });
 * ```
 *
 * @category Optimizers
 */
export class Adam extends Optimizer<AdamOptions, AdamState> {
  private _stepCount = 0;

  get stepCount(): number {
    return this._stepCount;
  }

  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<AdamOptions>>,
    options: {
      readonly lr?: number;
      readonly beta1?: number;
      readonly beta2?: number;
      readonly eps?: number;
      readonly weightDecay?: number;
      readonly amsgrad?: boolean;
    } = {}
  ) {
    const defaults = {
      lr: options.lr ?? 0.001,
      beta1: options.beta1 ?? 0.9,
      beta2: options.beta2 ?? 0.999,
      eps: options.eps ?? 1e-8,
      weightDecay: options.weightDecay ?? 0,
      amsgrad: options.amsgrad ?? false,
    };

    super(params, defaults);

    assertFiniteNonNegative("learning rate", defaults.lr);
    assertInRange("beta1", defaults.beta1, 0, 1);
    assertInRange("beta2", defaults.beta2, 0, 1);
    assertFinitePositive("epsilon", defaults.eps);
    assertFiniteNonNegative("weight_decay value", defaults.weightDecay);
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

  protected isState(state: Record<string, unknown>): state is AdamState {
    const hasRequired =
      typeof state["step"] === "number" &&
      state["expAvg"] instanceof Float64Array &&
      state["expAvgSq"] instanceof Float64Array;
    if (!hasRequired) return false;
    if (state["maxExpAvgSq"] !== undefined && !(state["maxExpAvgSq"] instanceof Float64Array)) {
      return false;
    }
    return true;
  }

  step(closure?: () => number): number | undefined {
    let loss: number | undefined;

    if (closure) {
      loss = closure();
    }

    this._stepCount++;

    for (const group of this.paramGroups) {
      const { lr, beta1, beta2, eps, weightDecay, amsgrad } = group.options;

      assertFiniteNonNegative("learning rate", lr);
      assertInRange("beta1", beta1, 0, 1);
      assertInRange("beta2", beta2, 0, 1);
      assertFinitePositive("epsilon", eps);
      assertFiniteNonNegative("weight_decay value", weightDecay);

      for (const param of group.params) {
        const {
          grad: gradData,
          gradOffset,
          param: paramData,
          paramOffset,
        } = assertHasGradFloat(param, "Adam");
        const size = param.tensor.size;

        const existing = this.state.get(param);
        const state =
          existing ??
          (() => {
            const next = {
              step: 0,
              expAvg: new Float64Array(size),
              expAvgSq: new Float64Array(size),
              ...(amsgrad ? { maxExpAvgSq: new Float64Array(size) } : {}),
            };
            this.state.set(param, next);
            return next;
          })();

        // Validate state buffer sizes
        assertBufferSize(state.expAvg, size, "Adam expAvg");
        assertBufferSize(state.expAvgSq, size, "Adam expAvgSq");
        if (amsgrad && state.maxExpAvgSq) {
          assertBufferSize(state.maxExpAvgSq, size, "Adam maxExpAvgSq");
        }

        state.step += 1;

        // Bias correction
        const biasCorrection1 = 1 - beta1 ** state.step;
        const biasCorrection2 = 1 - beta2 ** state.step;

        const stepSize = lr / biasCorrection1;

        for (let i = 0; i < size; i++) {
          const gi0 = safeArrayAccess(gradData, gradOffset + i, "Adam gradient");
          const pi = safeArrayAccess(paramData, paramOffset + i, "Adam parameter");
          assertFinite("gradient", gi0);
          assertFinite("parameter", pi);

          // Optional L2 weight decay (classic Adam style)
          const gi = weightDecay !== 0 ? gi0 + weightDecay * pi : gi0;

          const m = safeArrayAccess(state.expAvg, i, "Adam expAvg");
          const v = safeArrayAccess(state.expAvgSq, i, "Adam expAvgSq");

          const mNew = beta1 * m + (1 - beta1) * gi;
          const vNew = beta2 * v + (1 - beta2) * gi * gi;

          state.expAvg[i] = mNew;
          state.expAvgSq[i] = vNew;

          let denomSq = vNew;
          if (amsgrad) {
            const maxBuf = state.maxExpAvgSq;
            if (!maxBuf) {
              throw new DeepboxError("Internal error: AMSGrad enabled but maxExpAvgSq is missing");
            }
            const maxV = Math.max(safeArrayAccess(maxBuf, i, "Adam maxExpAvgSq"), vNew);
            maxBuf[i] = maxV;
            denomSq = maxV;
          }

          const denom = Math.sqrt(denomSq / biasCorrection2) + eps;
          paramData[paramOffset + i] = pi - stepSize * (mNew / denom);
        }
      }
    }

    return loss;
  }
}
