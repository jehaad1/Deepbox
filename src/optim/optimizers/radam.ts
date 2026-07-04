/**
 * @see {@link https://deepbox.dev/docs/optim-optimizers | Deepbox documentation}
 */

import { InvalidParameterError } from "../../core";
import {
  add,
  addScalar,
  div,
  type GradTensor,
  mulScalar,
  sqrt,
  square,
  sub,
  type Tensor,
} from "../../ndarray";
import {
  assertBufferSize,
  assertFinite,
  assertFiniteNonNegative,
  assertFinitePositive,
  assertHasGradFloat,
  assertInRange,
  replaceParamStorage,
  safeArrayAccess,
} from "../_internal";
import { Optimizer, type ParamGroup } from "../Optimizer";

type RAdamOptions = {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
};

type RAdamState = {
  step: number;
  expAvg: Float64Array;
  expAvgSq: Float64Array;
  /** Device state (used when the parameter lives on a kernel device). */
  expAvgTensor?: Tensor;
  expAvgSqTensor?: Tensor;
  deviceStep?: number;
};

/**
 * RAdam (Rectified Adam) optimizer.
 *
 * Fixes Adam's early training variance issue by computing a variance
 * rectification term. When the variance is tractable (high enough SMA),
 * uses the adaptive learning rate; otherwise falls back to SGD with momentum.
 *
 * @example
 * ```ts
 * import { RAdam } from 'deepbox/optim';
 *
 * const optimizer = new RAdam(model.parameters(), { lr: 0.001 });
 * ```
 *
 * Reference: Liu et al., "On the Variance of the Adaptive Learning Rate and Beyond", ICLR 2020.
 *
 * @category Optimizers
 */
export class RAdam extends Optimizer<RAdamOptions, RAdamState> {
  private _stepCount = 0;

  get stepCount(): number {
    return this._stepCount;
  }

  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<RAdamOptions>>,
    options: {
      readonly lr?: number;
      readonly beta1?: number;
      readonly beta2?: number;
      readonly eps?: number;
      readonly weightDecay?: number;
    } = {}
  ) {
    const defaults = {
      lr: options.lr ?? 0.001,
      beta1: options.beta1 ?? 0.9,
      beta2: options.beta2 ?? 0.999,
      eps: options.eps ?? 1e-8,
      weightDecay: options.weightDecay ?? 0,
    };

    super(params, defaults);

    assertFiniteNonNegative("learning rate", defaults.lr);
    assertInRange("beta1", defaults.beta1, 0, 1);
    assertInRange("beta2", defaults.beta2, 0, 1);
    assertFinitePositive("epsilon", defaults.eps);
    assertFiniteNonNegative("weight_decay value", defaults.weightDecay);
  }

  getLearningRate(groupIdx = 0): number {
    const group = this.paramGroups[groupIdx];
    if (!group) {
      throw new InvalidParameterError(`Invalid group index: ${groupIdx}`, "groupIdx", groupIdx);
    }
    return group.options.lr;
  }

  setLearningRate(lr: number): void {
    assertFiniteNonNegative("learning rate", lr);
    for (const group of this.paramGroups) {
      group.options.lr = lr;
    }
  }

  protected isState(state: Record<string, unknown>): state is RAdamState {
    return (
      typeof state["step"] === "number" &&
      state["expAvg"] instanceof Float64Array &&
      state["expAvgSq"] instanceof Float64Array
    );
  }

  step(closure?: () => number): number | undefined {
    let loss: number | undefined;
    if (closure) loss = closure();

    this._stepCount++;

    for (const group of this.paramGroups) {
      const { lr, beta1, beta2, eps, weightDecay } = group.options;

      // Maximum length of the approximated SMA
      const rhoInf = 2 / (1 - beta2) - 1;

      for (const param of group.params) {
        // Device path: compose the RAdam update from device-dispatched ops. The
        // rectification branch depends only on the (host-side) step scalar, so
        // it is selected here identically to the host loop.
        if (param.tensor.isDeviceTensor) {
          const g = param.grad;
          if (!g) continue;
          let dstate = this.state.get(param);
          if (!dstate) {
            dstate = { step: 0, expAvg: new Float64Array(0), expAvgSq: new Float64Array(0) };
            this.state.set(param, dstate);
          }
          const t = (dstate.deviceStep ?? 0) + 1;
          dstate.deviceStep = t;
          const biasCorrection1 = 1 - beta1 ** t;
          const biasCorrection2 = 1 - beta2 ** t;
          const rhoT = rhoInf - (2 * t * beta2 ** t) / biasCorrection2;
          const grad = weightDecay !== 0 ? add(g, mulScalar(param.tensor, weightDecay)) : g;
          const mPrev = dstate.expAvgTensor;
          const vPrev = dstate.expAvgSqTensor;
          const mNew = mPrev
            ? add(mulScalar(mPrev, beta1), mulScalar(grad, 1 - beta1))
            : mulScalar(grad, 1 - beta1);
          const vNew = vPrev
            ? add(mulScalar(vPrev, beta2), mulScalar(square(grad), 1 - beta2))
            : mulScalar(square(grad), 1 - beta2);
          dstate.expAvgTensor = mNew;
          dstate.expAvgSqTensor = vNew;
          const mCorrected = mulScalar(mNew, 1 / biasCorrection1);
          if (rhoT > 5) {
            const vCorrected = mulScalar(vNew, 1 / biasCorrection2);
            const rect = Math.sqrt(
              ((rhoT - 4) * (rhoT - 2) * rhoInf) / ((rhoInf - 4) * (rhoInf - 2) * rhoT)
            );
            const denom = addScalar(sqrt(vCorrected), eps);
            replaceParamStorage(
              param,
              "tensor",
              sub(param.tensor, mulScalar(div(mCorrected, denom), lr * rect))
            );
          } else {
            replaceParamStorage(param, "tensor", sub(param.tensor, mulScalar(mCorrected, lr)));
          }
          continue;
        }

        const {
          grad: gradData,
          gradOffset,
          param: paramData,
          paramOffset,
        } = assertHasGradFloat(param, "RAdam");
        const size = param.tensor.size;

        const existing = this.state.get(param);
        const state =
          existing ??
          (() => {
            const next: RAdamState = {
              step: 0,
              expAvg: new Float64Array(size),
              expAvgSq: new Float64Array(size),
            };
            this.state.set(param, next);
            return next;
          })();

        assertBufferSize(state.expAvg, size, "RAdam expAvg");
        assertBufferSize(state.expAvgSq, size, "RAdam expAvgSq");

        state.step += 1;

        const biasCorrection1 = 1 - beta1 ** state.step;
        const biasCorrection2 = 1 - beta2 ** state.step;

        // SMA for current step
        const rhoT = rhoInf - (2 * state.step * beta2 ** state.step) / biasCorrection2;

        for (let i = 0; i < size; i++) {
          const gi0 = safeArrayAccess(gradData, gradOffset + i, "RAdam gradient");
          const pi = safeArrayAccess(paramData, paramOffset + i, "RAdam parameter");
          assertFinite("gradient", gi0);
          assertFinite("parameter", pi);

          const gi = weightDecay !== 0 ? gi0 + weightDecay * pi : gi0;

          const m = safeArrayAccess(state.expAvg, i, "RAdam expAvg");
          const v = safeArrayAccess(state.expAvgSq, i, "RAdam expAvgSq");

          const mNew = beta1 * m + (1 - beta1) * gi;
          const vNew = beta2 * v + (1 - beta2) * gi * gi;

          state.expAvg[i] = mNew;
          state.expAvgSq[i] = vNew;

          const mCorrected = mNew / biasCorrection1;

          if (rhoT > 5) {
            // Variance is tractable — use adaptive learning rate
            const vCorrected = vNew / biasCorrection2;
            const rect = Math.sqrt(
              ((rhoT - 4) * (rhoT - 2) * rhoInf) / ((rhoInf - 4) * (rhoInf - 2) * rhoT)
            );
            paramData[paramOffset + i] =
              pi - (lr * rect * mCorrected) / (Math.sqrt(vCorrected) + eps);
          } else {
            // Variance not tractable — fall back to SGD with momentum
            paramData[paramOffset + i] = pi - lr * mCorrected;
          }
        }
      }
    }

    return loss;
  }
}
