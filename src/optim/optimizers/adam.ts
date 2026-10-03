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
  deviceMaxTensor,
  replaceParamStorage,
} from "../_internal";
import { Optimizer, type ParamGroup } from "../Optimizer";

type AdamOptions = {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  amsgrad: boolean;
  maximize: boolean;
};

type AdamState = {
  step: number;
  expAvg: Float64Array;
  expAvgSq: Float64Array;
  maxExpAvgSq?: Float64Array;
  /** Device moment buffers (used when the parameter lives on a kernel device). */
  expAvgTensor?: Tensor;
  expAvgSqTensor?: Tensor;
  maxExpAvgSqTensor?: Tensor;
  deviceStep?: number;
};

/**
 * Adam (Adaptive Moment Estimation) optimizer (Kingma and Ba, 2015).
 *
 * Computes adaptive learning rates for each parameter by maintaining
 * running averages of both the gradients and their squared values. The update
 * follows `torch.optim.Adam`:
 *
 * ```
 * m = beta1 * m + (1 - beta1) * g
 * v = beta2 * v + (1 - beta2) * g^2
 * theta -= (lr / (1 - beta1^t)) * m / (sqrt(v / (1 - beta2^t)) + eps)
 * ```
 *
 * When `weightDecay` is non-zero, `weightDecay * theta` is added to the gradient
 * (classic L2 penalty); use {@link AdamW} for decoupled weight decay. With
 * `amsgrad`, `v` is replaced by its running maximum in the denominator.
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
  /**
   * Create a new Adam optimizer.
   *
   * @param params - Parameters to optimize, or an array of parameter groups with per-group options
   * @param options - Hyperparameters
   * @param options.lr - Learning rate (default: 0.001)
   * @param options.beta1 - Decay rate of the first moment, in [0, 1) (default: 0.9)
   * @param options.beta2 - Decay rate of the second moment, in [0, 1) (default: 0.999)
   * @param options.eps - Term added to the denominator for numerical stability (default: 1e-8)
   * @param options.weightDecay - L2 penalty coefficient (default: 0)
   * @param options.amsgrad - Use the AMSGrad variant (default: false)
   * @param options.maximize - Maximize the objective instead of minimizing it (default: false)
   * @throws {InvalidParameterError} If a hyperparameter is out of range
   */
  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<AdamOptions>>,
    options: {
      readonly lr?: number;
      readonly beta1?: number;
      readonly beta2?: number;
      readonly eps?: number;
      readonly weightDecay?: number;
      readonly amsgrad?: boolean;
      readonly maximize?: boolean;
    } = {}
  ) {
    const defaults = {
      lr: options.lr ?? 0.001,
      beta1: options.beta1 ?? 0.9,
      beta2: options.beta2 ?? 0.999,
      eps: options.eps ?? 1e-8,
      weightDecay: options.weightDecay ?? 0,
      amsgrad: options.amsgrad ?? false,
      maximize: options.maximize ?? false,
    };

    super(params, defaults);
  }

  protected override validateOptions(options: Readonly<AdamOptions>): void {
    assertFiniteNonNegative("learning rate", options.lr);
    assertInRange("beta1", options.beta1, 0, 1);
    assertInRange("beta2", options.beta2, 0, 1);
    assertFinitePositive("epsilon", options.eps);
    assertFiniteNonNegative("weight_decay value", options.weightDecay);
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

    this.prepareStep("Adam");
    this.countStep();

    for (const group of this.paramGroups) {
      const { lr, beta1, beta2, eps, weightDecay, amsgrad, maximize } = group.options;

      for (const param of this.trainableParams(group)) {
        // Device path: keep the entire Adam update resident on the accelerator,
        // composing it from device-dispatched tensor ops (no host readback).
        if (param.tensor.isDeviceTensor) {
          const rawGrad = param.grad;
          if (!rawGrad) continue;
          const g = maximize ? mulScalar(rawGrad, -1) : rawGrad;
          let dstate = this.state.get(param);
          if (!dstate) {
            dstate = { step: 0, expAvg: new Float64Array(0), expAvgSq: new Float64Array(0) };
            this.state.set(param, dstate);
          }
          const t = (dstate.deviceStep ?? 0) + 1;
          dstate.deviceStep = t;
          const gi = weightDecay !== 0 ? add(g, mulScalar(param.tensor, weightDecay)) : g;
          const mPrev = dstate.expAvgTensor;
          const vPrev = dstate.expAvgSqTensor;
          const m = mPrev
            ? add(mulScalar(mPrev, beta1), mulScalar(gi, 1 - beta1))
            : mulScalar(gi, 1 - beta1);
          const v = vPrev
            ? add(mulScalar(vPrev, beta2), mulScalar(square(gi), 1 - beta2))
            : mulScalar(square(gi), 1 - beta2);
          dstate.expAvgTensor = m;
          dstate.expAvgSqTensor = v;
          let denomSq = v;
          if (amsgrad) {
            const maxPrev = dstate.maxExpAvgSqTensor;
            denomSq = maxPrev ? deviceMaxTensor(maxPrev, v) : v;
            dstate.maxExpAvgSqTensor = denomSq;
          }
          const bc1 = 1 - beta1 ** t;
          const bc2 = 1 - beta2 ** t;
          const stepSize = lr / bc1;
          // denom = sqrt(denomSq / bc2) + eps
          const denom = addScalar(sqrt(mulScalar(denomSq, 1 / bc2)), eps);
          const update = mulScalar(div(m, denom), stepSize);
          replaceParamStorage(param, "tensor", sub(param.tensor, update));
          continue;
        }

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
            const next: AdamState = {
              step: 0,
              expAvg: new Float64Array(size),
              expAvgSq: new Float64Array(size),
            };
            this.state.set(param, next);
            return next;
          })();

        // Validate state buffer sizes
        assertBufferSize(state.expAvg, size, "Adam expAvg");
        assertBufferSize(state.expAvgSq, size, "Adam expAvgSq");
        // `amsgrad` may be switched on after state was created (or the state may
        // have been saved without the buffer), so create the maximum lazily.
        let maxBuf: Float64Array | undefined;
        if (amsgrad) {
          maxBuf = state.maxExpAvgSq ??= new Float64Array(size);
          assertBufferSize(maxBuf, size, "Adam maxExpAvgSq");
        }

        state.step += 1;

        // Bias correction
        const biasCorrection1 = 1 - beta1 ** state.step;
        const biasCorrection2 = 1 - beta2 ** state.step;

        const stepSize = lr / biasCorrection1;
        const expAvg = state.expAvg;
        const expAvgSq = state.expAvgSq;

        for (let i = 0; i < size; i++) {
          const rawGi = gradData[gradOffset + i] as number;
          const pi = paramData[paramOffset + i] as number;
          if (!Number.isFinite(rawGi)) assertFinite("gradient", rawGi);
          const gi0 = maximize ? -rawGi : rawGi;
          if (!Number.isFinite(pi)) assertFinite("parameter", pi);

          // Optional L2 weight decay (classic Adam style)
          const gi = weightDecay !== 0 ? gi0 + weightDecay * pi : gi0;

          const mNew = beta1 * (expAvg[i] as number) + (1 - beta1) * gi;
          const vNew = beta2 * (expAvgSq[i] as number) + (1 - beta2) * gi * gi;

          expAvg[i] = mNew;
          expAvgSq[i] = vNew;

          let denomSq = vNew;
          if (maxBuf) {
            const maxV = Math.max(maxBuf[i] as number, vNew);
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
