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

/**
 * Options for the AdamW optimizer.
 *
 * @property lr - Learning rate (step size)
 * @property beta1 - Exponential decay rate for first moment estimates
 * @property beta2 - Exponential decay rate for second moment estimates
 * @property eps - Small constant for numerical stability
 * @property weightDecay - Decoupled weight decay coefficient
 * @property amsgrad - Whether to use the AMSGrad variant
 */
type AdamWOptions = {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  amsgrad: boolean;
  maximize: boolean;
};

/**
 * State maintained per parameter by AdamW.
 *
 * @property step - Number of optimization steps taken
 * @property expAvg - Exponentially weighted average of gradients (first moment)
 * @property expAvgSq - Exponentially weighted average of squared gradients (second moment)
 * @property maxExpAvgSq - Maximum of exponentially weighted average of squared gradients (AMSGrad only)
 */
type AdamWState = {
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
 * AdamW (Adam with decoupled weight decay) optimizer (Loshchilov and Hutter, 2019).
 *
 * AdamW applies weight decay directly to the parameters instead of adding it to
 * the gradient, so the decay is not rescaled by the adaptive denominator. The update
 * follows `torch.optim.AdamW`:
 *
 * ```
 * theta *= 1 - lr * weightDecay
 * m = beta1 * m + (1 - beta1) * g
 * v = beta2 * v + (1 - beta2) * g^2
 * theta -= (lr / (1 - beta1^t)) * m / (sqrt(v / (1 - beta2^t)) + eps)
 * ```
 *
 * @example
 * ```ts
 * import { AdamW } from 'deepbox/optim';
 *
 * const optimizer = new AdamW(model.parameters(), {
 *   lr: 0.001,
 *   weightDecay: 0.01,  // Typical value for AdamW
 *   beta1: 0.9,
 *   beta2: 0.999
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
export class AdamW extends Optimizer<AdamWOptions, AdamWState> {
  /**
   * Create a new AdamW optimizer.
   *
   * @param params - Iterable of parameters or parameter groups to optimize
   * @param options - Optimization options
   * @param options.lr - Learning rate (default: 0.001)
   * @param options.beta1 - First moment decay rate (default: 0.9)
   * @param options.beta2 - Second moment decay rate (default: 0.999)
   * @param options.eps - Numerical stability constant (default: 1e-8)
   * @param options.weightDecay - Decoupled weight decay coefficient (default: 0.01)
   * @param options.amsgrad - Enable AMSGrad variant (default: false)
   * @param options.maximize - Maximize the objective instead of minimizing it (default: false)
   * @throws {InvalidParameterError} If a parameter is invalid
   */
  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<AdamWOptions>>,
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
    // Set default values for all options
    const defaults = {
      lr: options.lr ?? 0.001,
      beta1: options.beta1 ?? 0.9,
      beta2: options.beta2 ?? 0.999,
      eps: options.eps ?? 1e-8,
      weightDecay: options.weightDecay ?? 0.01, // Higher default than Adam
      amsgrad: options.amsgrad ?? false,
      maximize: options.maximize ?? false,
    };

    super(params, defaults);
  }

  protected override validateOptions(options: Readonly<AdamWOptions>): void {
    assertFiniteNonNegative("learning rate", options.lr);
    assertInRange("beta1", options.beta1, 0, 1);
    assertInRange("beta2", options.beta2, 0, 1);
    assertFinitePositive("epsilon", options.eps);
    assertFiniteNonNegative("weight_decay value", options.weightDecay);
  }

  protected isState(state: Record<string, unknown>): state is AdamWState {
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
   * Implements the AdamW update rule with decoupled weight decay.
   *
   * A parameter whose gradient is `null` (it took no part in the loss) is skipped, as in PyTorch.
   *
   * @param closure - Optional function that re-evaluates the model and returns the loss
   * @returns The value returned by `closure`, or undefined when no closure is given
   * @throws {InvalidParameterError} If a gradient or parameter value is not finite
   */
  step(closure?: () => number): number | undefined {
    let loss: number | undefined;

    // Evaluate closure if provided (for algorithms like LBFGS)
    if (closure) {
      loss = closure();
    }

    this.prepareStep("AdamW");
    this.countStep();

    // Update each parameter group
    for (const group of this.paramGroups) {
      const { lr, beta1, beta2, eps, weightDecay, amsgrad, maximize } = group.options;

      // Re-validate hyperparameters (they might have been changed)

      // Update each parameter in the group
      for (const param of this.trainableParams(group)) {
        // Device path: keep the entire AdamW update resident on the accelerator,
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
          const mPrev = dstate.expAvgTensor;
          const vPrev = dstate.expAvgSqTensor;
          const m = mPrev
            ? add(mulScalar(mPrev, beta1), mulScalar(g, 1 - beta1))
            : mulScalar(g, 1 - beta1);
          const v = vPrev
            ? add(mulScalar(vPrev, beta2), mulScalar(square(g), 1 - beta2))
            : mulScalar(square(g), 1 - beta2);
          dstate.expAvgTensor = m;
          dstate.expAvgSqTensor = v;
          let denomSq = v;
          if (amsgrad) {
            const maxPrev = dstate.maxExpAvgSqTensor;
            denomSq = maxPrev ? deviceMaxTensor(maxPrev, v) : v;
            dstate.maxExpAvgSqTensor = denomSq;
          }
          const biasCorrection1 = 1 - beta1 ** t;
          const biasCorrection2 = 1 - beta2 ** t;
          const stepSize = lr / biasCorrection1;
          // denom = sqrt(denomSq / bc2) + eps
          const denom = addScalar(sqrt(mulScalar(denomSq, 1 / biasCorrection2)), eps);
          // Decoupled weight decay: theta -= stepSize*(m/denom) - lr*wd*theta
          let updated = sub(param.tensor, mulScalar(div(m, denom), stepSize));
          if (weightDecay !== 0) {
            updated = sub(updated, mulScalar(param.tensor, lr * weightDecay));
          }
          replaceParamStorage(param, "tensor", updated);
          continue;
        }

        // Get gradient and validate
        const {
          grad,
          gradOffset,
          param: pData,
          paramOffset: pOff,
        } = assertHasGradFloat(param, "AdamW");
        const size = param.tensor.size;

        // Get or initialize optimizer state for this parameter
        const existing = this.state.get(param);
        const state =
          existing ??
          (() => {
            const next: AdamWState = {
              step: 0,
              expAvg: new Float64Array(size), // First moment
              expAvgSq: new Float64Array(size), // Second moment
            };
            this.state.set(param, next);
            return next;
          })();

        // Validate state buffer sizes
        assertBufferSize(state.expAvg, size, "AdamW expAvg");
        assertBufferSize(state.expAvgSq, size, "AdamW expAvgSq");
        // `amsgrad` may be switched on after state was created (or the state may
        // have been saved without the buffer), so create the maximum lazily.
        let maxBuf: Float64Array | undefined;
        if (amsgrad) {
          maxBuf = state.maxExpAvgSq ??= new Float64Array(size);
          assertBufferSize(maxBuf, size, "AdamW maxExpAvgSq");
        }

        // Increment per-parameter step counter
        state.step += 1;

        // Compute bias correction terms
        const biasCorrection1 = 1 - beta1 ** state.step;
        const biasCorrection2 = 1 - beta2 ** state.step;

        // Compute step size with bias correction
        const stepSize = lr / biasCorrection1;
        const decay = lr * weightDecay;
        const expAvg = state.expAvg;
        const expAvgSq = state.expAvgSq;

        // Update each element of the parameter
        for (let i = 0; i < size; i++) {
          const rawGi = grad[gradOffset + i] as number;
          const pi = pData[pOff + i] as number;
          if (!Number.isFinite(rawGi)) assertFinite("gradient", rawGi);
          const gi = maximize ? -rawGi : rawGi;
          if (!Number.isFinite(pi)) assertFinite("parameter", pi);

          // Update biased first moment estimate: m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
          const mNew = beta1 * (expAvg[i] as number) + (1 - beta1) * gi;

          // Update biased second raw moment estimate: v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
          const vNew = beta2 * (expAvgSq[i] as number) + (1 - beta2) * gi * gi;

          expAvg[i] = mNew;
          expAvgSq[i] = vNew;

          // Determine which second moment to use (AMSGrad or standard)
          let denomSq = vNew;
          if (maxBuf) {
            // AMSGrad: use maximum of all past second moments
            const maxV = Math.max(maxBuf[i] as number, vNew);
            maxBuf[i] = maxV;
            denomSq = maxV;
          }

          // Denominator with bias correction: sqrt(v_hat(t)) + eps
          const denom = Math.sqrt(denomSq / biasCorrection2) + eps;

          // AdamW update: theta(t+1) = theta(t) * (1 - lr * lambda) - stepSize * m(t) / denom
          // The weight decay is applied directly to the parameters (decoupled).
          pData[pOff + i] = pi - stepSize * (mNew / denom) - decay * pi;
        }
      }
    }

    return loss;
  }
}
