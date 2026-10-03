/**
 * @see {@link https://deepbox.dev/docs/optim-optimizers | Deepbox documentation}
 */

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

/**
 * Options for the RAdam optimizer.
 *
 * @property lr - Learning rate
 * @property beta1 - Decay rate of the first moment estimate
 * @property beta2 - Decay rate of the second moment estimate
 * @property eps - Small constant added to the square root of the second moment
 * @property weightDecay - Weight decay coefficient
 * @property decoupledWeightDecay - Apply weight decay directly to the parameters
 *   (`param *= 1 - lr * weightDecay`, as AdamW does) instead of adding it to the gradient
 * @property maximize - Maximize the objective instead of minimizing it
 */
type RAdamOptions = {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  decoupledWeightDecay: boolean;
  maximize: boolean;
};

function validateRAdamOptions(options: Readonly<RAdamOptions>): void {
  assertFiniteNonNegative("learning rate", options.lr);
  assertInRange("beta1", options.beta1, 0, 1);
  assertInRange("beta2", options.beta2, 0, 1);
  assertFinitePositive("epsilon", options.eps);
  assertFiniteNonNegative("weight_decay value", options.weightDecay);
}

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
 * The update follows PyTorch's `torch.optim.RAdam`:
 *
 * ```
 * rhoInf = 2 / (1 - beta2) - 1
 * rhoT   = rhoInf - 2 t beta2^t / (1 - beta2^t)
 * rhoT > 5:  param -= lr * rect * sqrt(1 - beta2^t) * mHat / (sqrt(v) + eps)
 * otherwise: param -= lr * mHat                      (mHat = m / (1 - beta1^t))
 * ```
 *
 * With the default `beta2 = 0.999` the first five steps use the SGD-with-momentum branch.
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
  /**
   * Create a new RAdam optimizer.
   *
   * @param params - Iterable of parameters or parameter groups to optimize
   * @param options - Optimization options
   * @param options.lr - Learning rate (default: 0.001)
   * @param options.beta1 - First moment decay, in [0, 1) (default: 0.9)
   * @param options.beta2 - Second moment decay, in [0, 1) (default: 0.999)
   * @param options.eps - Numerical stability constant (default: 1e-8)
   * @param options.weightDecay - Weight decay coefficient (default: 0)
   * @param options.decoupledWeightDecay - Decay the parameters directly instead of
   *   adding the penalty to the gradient (default: false)
   * @param options.maximize - Maximize the objective instead of minimizing it (default: false)
   * @throws {InvalidParameterError} If a hyperparameter is out of range
   */
  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<RAdamOptions>>,
    options: {
      readonly lr?: number;
      readonly beta1?: number;
      readonly beta2?: number;
      readonly eps?: number;
      readonly weightDecay?: number;
      readonly decoupledWeightDecay?: boolean;
      readonly maximize?: boolean;
    } = {}
  ) {
    const defaults: RAdamOptions = {
      lr: options.lr ?? 0.001,
      beta1: options.beta1 ?? 0.9,
      beta2: options.beta2 ?? 0.999,
      eps: options.eps ?? 1e-8,
      weightDecay: options.weightDecay ?? 0,
      decoupledWeightDecay: options.decoupledWeightDecay ?? false,
      maximize: options.maximize ?? false,
    };

    super(params, defaults);
  }

  protected override validateOptions(options: Readonly<RAdamOptions>): void {
    validateRAdamOptions(options);
  }

  protected isState(state: Record<string, unknown>): state is RAdamState {
    return (
      typeof state["step"] === "number" &&
      state["expAvg"] instanceof Float64Array &&
      state["expAvgSq"] instanceof Float64Array
    );
  }

  /**
   * Perform a single optimization step.
   *
   * A parameter whose gradient is `null` (it took no part in the loss) is skipped, as in PyTorch.
   *
   * @param closure - Optional closure that reevaluates the model and returns the loss
   * @returns Loss value if a closure is provided
   * @throws {InvalidParameterError} If a group option is invalid or a gradient or
   *   parameter value is not finite
   */
  step(closure?: () => number): number | undefined {
    let loss: number | undefined;
    if (closure) loss = closure();

    this.prepareStep("RAdam");
    this.countStep();

    for (const group of this.paramGroups) {
      const { lr, beta1, beta2, eps, weightDecay, decoupledWeightDecay, maximize } = group.options;

      // Maximum length of the approximated SMA
      const rhoInf = 2 / (1 - beta2) - 1;
      const decay = decoupledWeightDecay && weightDecay !== 0 ? 1 - lr * weightDecay : 1;
      const l2 = decoupledWeightDecay ? 0 : weightDecay;

      for (const param of this.trainableParams(group)) {
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
          const signed = maximize ? mulScalar(g, -1) : g;
          const grad = l2 !== 0 ? add(signed, mulScalar(param.tensor, l2)) : signed;
          const base = decay !== 1 ? mulScalar(param.tensor, decay) : param.tensor;
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
            const rect = Math.sqrt(
              ((rhoT - 4) * (rhoT - 2) * rhoInf) / ((rhoInf - 4) * (rhoInf - 2) * rhoT)
            );
            const denom = addScalar(sqrt(vNew), eps);
            replaceParamStorage(
              param,
              "tensor",
              sub(base, mulScalar(div(mCorrected, denom), lr * rect * Math.sqrt(biasCorrection2)))
            );
          } else {
            replaceParamStorage(param, "tensor", sub(base, mulScalar(mCorrected, lr)));
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

        // The rectified branch needs the same scalar for every element.
        const rectified = rhoT > 5;
        const adaptiveStep = rectified
          ? lr *
            Math.sqrt(((rhoT - 4) * (rhoT - 2) * rhoInf) / ((rhoInf - 4) * (rhoInf - 2) * rhoT)) *
            Math.sqrt(biasCorrection2)
          : 0;

        for (let i = 0; i < size; i++) {
          const gi0 = safeArrayAccess(gradData, gradOffset + i, "RAdam gradient");
          const pi = safeArrayAccess(paramData, paramOffset + i, "RAdam parameter");
          assertFinite("gradient", gi0);
          assertFinite("parameter", pi);

          const gSigned = maximize ? -gi0 : gi0;
          const gi = l2 !== 0 ? gSigned + l2 * pi : gSigned;
          const base = decay !== 1 ? pi * decay : pi;

          const m = safeArrayAccess(state.expAvg, i, "RAdam expAvg");
          const v = safeArrayAccess(state.expAvgSq, i, "RAdam expAvgSq");

          const mNew = beta1 * m + (1 - beta1) * gi;
          const vNew = beta2 * v + (1 - beta2) * gi * gi;

          state.expAvg[i] = mNew;
          state.expAvgSq[i] = vNew;

          const mCorrected = mNew / biasCorrection1;

          if (rectified) {
            // Variance is tractable: use the adaptive learning rate.
            paramData[paramOffset + i] =
              base - (adaptiveStep * mCorrected) / (Math.sqrt(vNew) + eps);
          } else {
            // Variance not tractable: fall back to SGD with momentum.
            paramData[paramOffset + i] = base - lr * mCorrected;
          }
        }
      }
    }

    return loss;
  }
}
