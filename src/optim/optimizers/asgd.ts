/**
 * @see {@link https://deepbox.dev/docs/optim-optimizers | Deepbox documentation}
 */

import { DeepboxError, DTypeError, InvalidParameterError } from "../../core";
import { add, type GradTensor, mulScalar, reshape, sub, type Tensor, tensor } from "../../ndarray";
import {
  assertFinite,
  assertFiniteNonNegative,
  assertHasGradFloat,
  replaceParamStorage,
} from "../_internal";
import { Optimizer, type ParamGroup } from "../Optimizer";

type ASGDOptions = {
  lr: number;
  lambda: number;
  alpha: number;
  t0: number;
  weightDecay: number;
  maximize: boolean;
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
 * Averaged Stochastic Gradient Descent (ASGD) optimizer (Polyak and Juditsky, 1992).
 *
 * Runs SGD with a decaying step size and keeps a running (Polyak-Ruppert) average
 * of the iterates, which often generalizes better than the last iterate. The average
 * starts after `t0` steps and is available through {@link ASGD.averagedParameters}.
 * The update follows `torch.optim.ASGD`:
 *
 * ```
 * theta = theta * (1 - lambda * eta) - eta * g
 * ax += mu * (theta - ax)
 * eta = lr / (1 + lambda * lr * t)^alpha
 * mu = 1 / max(1, t - t0)
 * ```
 *
 * When `weightDecay` is non-zero, `weightDecay * theta` is added to the gradient
 * (L2 penalty) before the update. With `maximize`, the gradient is negated first.
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
  /**
   * Create a new ASGD optimizer.
   *
   * @param params - Parameters to optimize, or an array of parameter groups with per-group options
   * @param options - Hyperparameters
   * @param options.lr - Initial learning rate (default: 0.01)
   * @param options.lambda - Decay term applied to the parameters (default: 1e-4)
   * @param options.alpha - Exponent of the step-size decay (default: 0.75)
   * @param options.t0 - Step after which averaging begins (default: 1e6)
   * @param options.weightDecay - L2 penalty coefficient (default: 0)
   * @param options.maximize - Maximize the objective instead of minimizing it (default: false)
   * @throws {InvalidParameterError} If a hyperparameter is out of range
   */
  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<ASGDOptions>>,
    options: {
      readonly lr?: number;
      readonly lambda?: number;
      readonly alpha?: number;
      readonly t0?: number;
      readonly weightDecay?: number;
      readonly maximize?: boolean;
    } = {}
  ) {
    const defaults: ASGDOptions = {
      lr: options.lr ?? 0.01,
      lambda: options.lambda ?? 1e-4,
      alpha: options.alpha ?? 0.75,
      t0: options.t0 ?? 1e6,
      weightDecay: options.weightDecay ?? 0,
      maximize: options.maximize ?? false,
    };

    super(params, defaults);
  }

  protected override validateOptions(options: Readonly<ASGDOptions>): void {
    assertFiniteNonNegative("learning rate", options.lr);
    assertFiniteNonNegative("lambda", options.lambda);
    assertFiniteNonNegative("alpha", options.alpha);
    assertFiniteNonNegative("weight_decay", options.weightDecay);
    if (!Number.isFinite(options.t0)) {
      throw new InvalidParameterError("t0 must be finite", "t0", options.t0);
    }
  }

  protected isState(state: Record<string, unknown>): state is ASGDState {
    if (state["ax"] !== undefined && !(state["ax"] instanceof Float64Array)) {
      return false;
    }
    for (const key of ["step", "eta", "mu"] as const) {
      const value = state[key];
      if (value !== undefined && typeof value !== "number") return false;
    }
    return true;
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

    this.prepareStep("ASGD");
    this.countStep();

    for (const group of this.paramGroups) {
      const { lr, lambda, alpha, t0, weightDecay, maximize } = group.options;

      for (const param of this.trainableParams(group)) {
        // Device path: compose the ASGD update from device-dispatched ops. The
        // eta/mu/decay scalars are host numbers (from options + step counter),
        // so they are applied identically to the host loop.
        if (param.tensor.isDeviceTensor) {
          const rawGrad = param.grad;
          if (!rawGrad) continue;
          const g = maximize ? mulScalar(rawGrad, -1) : rawGrad;
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
          // Initialize ax with the current parameters
          state.ax = Float64Array.from(
            paramData.subarray(paramOffset, paramOffset + size) as ArrayLike<number>
          );
        }
        if (state.ax.length !== size) {
          throw new DeepboxError(
            `State buffer size mismatch for ASGD ax: expected ${size}, got ${state.ax.length}`
          );
        }

        // PyTorch ordering: use the eta/mu from the PREVIOUS step (init eta=lr,
        // mu=1), apply the multiplicative decay p *= (1 - lambd*eta) plus the
        // gradient step and averaging, THEN advance the step counter and
        // recompute eta/mu for next time.
        const eta = state.eta ?? lr;
        const mu = state.mu ?? 1;
        const ax = state.ax;
        const decayFactor = 1 - lambda * eta;

        for (let i = 0; i < size; i++) {
          const rawGi = gradData[gradOffset + i] as number;
          const pi = paramData[paramOffset + i] as number;
          if (!Number.isFinite(rawGi)) assertFinite("gradient", rawGi);
          if (!Number.isFinite(pi)) assertFinite("parameter", pi);
          const gi = maximize ? -rawGi : rawGi;

          // Apply weight decay (L2) into the gradient
          const d = weightDecay !== 0 ? gi + weightDecay * pi : gi;

          // Multiplicative ASGD decay term, then the SGD update
          const newP = pi * decayFactor - eta * d;
          paramData[paramOffset + i] = newP;

          // Update running average
          const prevAx = ax[i] as number;
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

  /**
   * Get the running average of each parameter (Polyak-Ruppert averaging).
   *
   * The average starts after `t0` steps; before that it equals the latest iterate.
   * Parameters that have not been stepped yet are returned as a copy of their
   * current value. The returned tensors are copies; the live parameters are not
   * modified.
   *
   * @returns One tensor per parameter, in group order, with the parameter's shape
   */
  averagedParameters(): Tensor[] {
    const result: Tensor[] = [];
    for (const group of this.paramGroups) {
      for (const param of group.params) {
        const state = this.state.get(param);
        const current = param.tensor;
        if (current.isDeviceTensor) {
          result.push(state?.axTensor ?? current);
          continue;
        }
        const dtype = current.dtype === "float32" ? "float32" : "float64";
        const live = current.data;
        if (!(live instanceof Float32Array || live instanceof Float64Array)) {
          throw new DTypeError("ASGD supports float32 and float64 parameters only");
        }
        const source: ArrayLike<number> =
          state?.ax ?? live.subarray(current.offset, current.offset + current.size);
        const data = dtype === "float32" ? Float32Array.from(source) : Float64Array.from(source);
        result.push(reshape(tensor(data, { dtype }), current.shape));
      }
    }
    return result;
  }
}
