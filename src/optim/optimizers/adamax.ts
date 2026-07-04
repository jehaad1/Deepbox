/**
 * @see {@link https://deepbox.dev/docs/optim-optimizers | Deepbox documentation}
 */

import { InvalidParameterError } from "../../core";
import {
  abs,
  add,
  addScalar,
  div,
  type GradTensor,
  mulScalar,
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
  deviceMaxTensor,
  replaceParamStorage,
  safeArrayAccess,
} from "../_internal";
import { Optimizer, type ParamGroup } from "../Optimizer";

type AdamaxOptions = {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
};

type AdamaxState = {
  step: number;
  expAvg: Float64Array;
  expInfNorm: Float64Array;
  /** Device state (used when the parameter lives on a kernel device). */
  expAvgTensor?: Tensor;
  expInfNormTensor?: Tensor;
  deviceStep?: number;
};

/**
 * Adamax optimizer — variant of Adam using the infinity norm.
 *
 * Particularly well-suited for embeddings and sparse gradients.
 *
 * Update rule:
 *   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
 *   u_t = max(beta2 * u_{t-1}, |g_t|)
 *   theta_t = theta_{t-1} - (lr / (1 - beta1^t)) * m_t / (u_t + eps)
 *
 * @example
 * ```ts
 * import { Adamax } from 'deepbox/optim';
 *
 * const optimizer = new Adamax(model.parameters(), { lr: 0.002 });
 * ```
 *
 * @category Optimizers
 */
export class Adamax extends Optimizer<AdamaxOptions, AdamaxState> {
  private _stepCount = 0;

  get stepCount(): number {
    return this._stepCount;
  }

  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<AdamaxOptions>>,
    options: {
      readonly lr?: number;
      readonly beta1?: number;
      readonly beta2?: number;
      readonly eps?: number;
      readonly weightDecay?: number;
    } = {}
  ) {
    const defaults = {
      lr: options.lr ?? 0.002,
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

  protected isState(state: Record<string, unknown>): state is AdamaxState {
    return (
      typeof state["step"] === "number" &&
      state["expAvg"] instanceof Float64Array &&
      state["expInfNorm"] instanceof Float64Array
    );
  }

  step(closure?: () => number): number | undefined {
    let loss: number | undefined;
    if (closure) loss = closure();

    this._stepCount++;

    for (const group of this.paramGroups) {
      const { lr, beta1, beta2, eps, weightDecay } = group.options;

      for (const param of group.params) {
        // Device path: compose the Adamax update from device-dispatched ops.
        if (param.tensor.isDeviceTensor) {
          const g = param.grad;
          if (!g) continue;
          let dstate = this.state.get(param);
          if (!dstate) {
            dstate = { step: 0, expAvg: new Float64Array(0), expInfNorm: new Float64Array(0) };
            this.state.set(param, dstate);
          }
          const t = (dstate.deviceStep ?? 0) + 1;
          dstate.deviceStep = t;
          const biasCorrection1 = 1 - beta1 ** t;
          const stepSize = lr / biasCorrection1;
          const grad = weightDecay !== 0 ? add(g, mulScalar(param.tensor, weightDecay)) : g;
          const mPrev = dstate.expAvgTensor;
          const uPrev = dstate.expInfNormTensor;
          const mNew = mPrev
            ? add(mulScalar(mPrev, beta1), mulScalar(grad, 1 - beta1))
            : mulScalar(grad, 1 - beta1);
          // u(t) = max(beta2 * u(t-1), |g|); first step u=0 -> |g|.
          const uNew = uPrev ? deviceMaxTensor(mulScalar(uPrev, beta2), abs(grad)) : abs(grad);
          dstate.expAvgTensor = mNew;
          dstate.expInfNormTensor = uNew;
          const denom = addScalar(uNew, eps);
          replaceParamStorage(
            param,
            "tensor",
            sub(param.tensor, mulScalar(div(mNew, denom), stepSize))
          );
          continue;
        }

        const {
          grad: gradData,
          gradOffset,
          param: paramData,
          paramOffset,
        } = assertHasGradFloat(param, "Adamax");
        const size = param.tensor.size;

        const existing = this.state.get(param);
        const state =
          existing ??
          (() => {
            const next: AdamaxState = {
              step: 0,
              expAvg: new Float64Array(size),
              expInfNorm: new Float64Array(size),
            };
            this.state.set(param, next);
            return next;
          })();

        assertBufferSize(state.expAvg, size, "Adamax expAvg");
        assertBufferSize(state.expInfNorm, size, "Adamax expInfNorm");

        state.step += 1;

        const biasCorrection1 = 1 - beta1 ** state.step;
        const stepSize = lr / biasCorrection1;

        for (let i = 0; i < size; i++) {
          const gi0 = safeArrayAccess(gradData, gradOffset + i, "Adamax gradient");
          const pi = safeArrayAccess(paramData, paramOffset + i, "Adamax parameter");
          assertFinite("gradient", gi0);
          assertFinite("parameter", pi);

          const gi = weightDecay !== 0 ? gi0 + weightDecay * pi : gi0;

          const m = safeArrayAccess(state.expAvg, i, "Adamax expAvg");
          const u = safeArrayAccess(state.expInfNorm, i, "Adamax expInfNorm");

          const mNew = beta1 * m + (1 - beta1) * gi;
          const uNew = Math.max(beta2 * u, Math.abs(gi));

          state.expAvg[i] = mNew;
          state.expInfNorm[i] = uNew;

          paramData[paramOffset + i] = pi - (stepSize * mNew) / (uNew + eps);
        }
      }
    }

    return loss;
  }
}
