/**
 * @see {@link https://deepbox.dev/docs/optim-optimizers | Deepbox documentation}
 */

import { InvalidParameterError } from "../../core";
import { add, type GradTensor, mulScalar, sub, type Tensor } from "../../ndarray";
import {
  assertFinite,
  assertFiniteNonNegative,
  assertHasGradFloat,
  replaceParamStorage,
  safeArrayAccess,
} from "../_internal";
import { Optimizer, type ParamGroup } from "../Optimizer";

type ASGDOptions = {
  lr: number;
  lambda: number;
  alpha: number;
  t0: number;
  weightDecay: number;
};

type ASGDState = {
  step?: number;
  eta?: number;
  mu?: number;
  ax?: Float64Array;
  /** Device running-average buffer (used when the parameter lives on a kernel device). */
  axTensor?: Tensor;
};

/**
 * Averaged Stochastic Gradient Descent (ASGD) optimizer.
 *
 * Implements Polyak-Ruppert averaging: maintains a running average of
 * parameters which often converges better than the last iterate.
 *
 * @example
 * ```ts
 * import { ASGD } from 'deepbox/optim';
 *
 * const optimizer = new ASGD(model.parameters(), {
 *   lr: 0.01,
 *   t0: 1e6,
 * });
 * ```
 *
 * @category Optimizers
 */
export class ASGD extends Optimizer<ASGDOptions, ASGDState> {
  private _stepCount = 0;

  get stepCount(): number {
    return this._stepCount;
  }

  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<ASGDOptions>>,
    options: {
      readonly lr?: number;
      readonly lambda?: number;
      readonly alpha?: number;
      readonly t0?: number;
      readonly weightDecay?: number;
    } = {}
  ) {
    const defaults: ASGDOptions = {
      lr: options.lr ?? 0.01,
      lambda: options.lambda ?? 1e-4,
      alpha: options.alpha ?? 0.75,
      t0: options.t0 ?? 1e6,
      weightDecay: options.weightDecay ?? 0,
    };

    super(params, defaults);

    assertFiniteNonNegative("learning rate", defaults.lr);
    assertFiniteNonNegative("lambda", defaults.lambda);
    assertFiniteNonNegative("alpha", defaults.alpha);
    assertFiniteNonNegative("weight_decay", defaults.weightDecay);
    if (!Number.isFinite(defaults.t0)) {
      throw new InvalidParameterError("t0 must be finite", "t0", defaults.t0);
    }
  }

  protected isState(state: Record<string, unknown>): state is ASGDState {
    if (state["ax"] !== undefined && !(state["ax"] instanceof Float64Array)) {
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
      const { lr, lambda, alpha, t0, weightDecay } = group.options;

      for (const param of group.params) {
        // Device path: compose the ASGD update from device-dispatched ops. The
        // eta/mu/decay scalars are host numbers (from options + step counter),
        // so they are applied identically to the host loop.
        if (param.tensor.isDeviceTensor) {
          const g = param.grad;
          if (!g) continue;
          let dstate = this.state.get(param);
          if (!dstate) {
            dstate = { step: 0, eta: lr, mu: 1 };
            this.state.set(param, dstate);
          }
          const eta = dstate.eta ?? lr;
          const mu = dstate.mu ?? 1;
          // d = g + weightDecay * param
          const d = weightDecay !== 0 ? add(g, mulScalar(param.tensor, weightDecay)) : g;
          // newP = param * (1 - lambda*eta) - eta * d
          const newP = sub(mulScalar(param.tensor, 1 - lambda * eta), mulScalar(d, eta));
          // ax(t) = mu==1 ? newP : prevAx + mu*(newP - prevAx)
          const prevAx = dstate.axTensor ?? param.tensor;
          dstate.axTensor = mu === 1 ? newP : add(prevAx, mulScalar(sub(newP, prevAx), mu));
          replaceParamStorage(param, "tensor", newP);
          // Advance step and recompute eta/mu for the next call.
          dstate.step = (dstate.step ?? 0) + 1;
          const stepNum = dstate.step;
          dstate.eta = lr / (1 + lambda * lr * stepNum) ** alpha;
          dstate.mu = 1 / Math.max(1, stepNum - t0);
          continue;
        }

        const {
          grad: gradData,
          gradOffset,
          param: paramData,
          paramOffset,
        } = assertHasGradFloat(param, "ASGD");
        const size = param.tensor.size;

        let state = this.state.get(param);
        if (!state) {
          state = {
            step: 0,
            eta: lr,
            mu: 1,
          };
          this.state.set(param, state);
        }

        if (!state.ax) {
          state.ax = new Float64Array(size);
          // Initialize ax with current params
          for (let i = 0; i < size; i++) {
            state.ax[i] = safeArrayAccess(paramData, paramOffset + i, "ASGD param");
          }
        }

        // PyTorch ordering: use the eta/mu from the PREVIOUS step (init eta=lr,
        // mu=1), apply the multiplicative decay p *= (1 - lambd*eta) plus the
        // gradient step and averaging, THEN advance the step counter and
        // recompute eta/mu for next time. The prior code recomputed eta with
        // the current step count and omitted the decay term entirely.
        const eta = state.eta ?? lr;
        const mu = state.mu ?? 1;
        const ax = state.ax;

        for (let i = 0; i < size; i++) {
          const gi = safeArrayAccess(gradData, gradOffset + i, "ASGD gradient");
          const pi = safeArrayAccess(paramData, paramOffset + i, "ASGD parameter");
          assertFinite("gradient", gi);
          assertFinite("parameter", pi);

          // Apply weight decay (L2) into the gradient
          let d = gi;
          if (weightDecay !== 0) {
            d = d + weightDecay * pi;
          }

          // Decoupled ASGD decay term, then the SGD update
          const decayed = pi * (1 - lambda * eta);
          const newP = decayed - eta * d;
          paramData[paramOffset + i] = newP;

          // Update running average
          const prevAx = safeArrayAccess(ax, i, "ASGD ax");
          ax[i] = mu === 1 ? newP : prevAx + mu * (newP - prevAx);
        }

        // Advance step and recompute eta/mu for the next call.
        state.step = (state.step ?? 0) + 1;
        const stepNum = state.step;
        state.eta = lr / (1 + lambda * lr * stepNum) ** alpha;
        state.mu = 1 / Math.max(1, stepNum - t0);
      }
    }

    return loss;
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
}
