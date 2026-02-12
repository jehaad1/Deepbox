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

/**
 * Options for the AdamW optimizer.
 *
 * @property lr - Learning rate (step size)
 * @property beta1 - Exponential decay rate for first moment estimates
 * @property beta2 - Exponential decay rate for second moment estimates
 * @property eps - Small constant for numerical stability
 * @property weightDecay - Weight decay coefficient (L2 penalty)
 * @property amsgrad - Whether to use the AMSGrad variant
 */
type AdamWOptions = {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  amsgrad: boolean;
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
};

/**
 * AdamW (Adam with decoupled Weight decay) optimizer.
 *
 * AdamW fixes the weight decay implementation in Adam by decoupling it from the
 * gradient-based update. This leads to better generalization and is the recommended
 * variant for most applications.
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
  /** Internal counter tracking total number of optimization steps */
  private _stepCount = 0;

  /**
   * Get the total number of optimization steps performed.
   *
   * @returns Number of steps taken
   */
  get stepCount(): number {
    return this._stepCount;
  }

  /**
   * Create a new AdamW optimizer.
   *
   * @param params - Iterable of parameters or parameter groups to optimize
   * @param options - Optimization options
   * @param options.lr - Learning rate (default: 0.001)
   * @param options.beta1 - First moment decay rate (default: 0.9)
   * @param options.beta2 - Second moment decay rate (default: 0.999)
   * @param options.eps - Numerical stability constant (default: 1e-8)
   * @param options.weightDecay - Weight decay coefficient (default: 0.01)
   * @param options.amsgrad - Enable AMSGrad variant (default: false)
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
    };

    super(params, defaults);

    // Validate all hyperparameters
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

  /**
   * Perform a single optimization step (parameter update).
   *
   * Implements the AdamW update rule with decoupled weight decay.
   *
   * @param closure - Optional closure that reevaluates the model and returns the loss
   * @returns Loss value if closure is provided, undefined otherwise
   */
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

  step(closure?: () => number): number | undefined {
    let loss: number | undefined;

    // Evaluate closure if provided (for algorithms like LBFGS)
    if (closure) {
      loss = closure();
    }

    // Increment global step counter
    this._stepCount++;

    // Update each parameter group
    for (const group of this.paramGroups) {
      const { lr, beta1, beta2, eps, weightDecay, amsgrad } = group.options;

      // Re-validate hyperparameters (they might have been changed)
      assertFiniteNonNegative("learning rate", lr);
      assertInRange("beta1", beta1, 0, 1);
      assertInRange("beta2", beta2, 0, 1);
      assertFinitePositive("epsilon", eps);
      assertFiniteNonNegative("weight_decay value", weightDecay);

      // Update each parameter in the group
      for (const param of group.params) {
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
            // Initialize state on first use
            const next = {
              step: 0,
              expAvg: new Float64Array(size), // First moment
              expAvgSq: new Float64Array(size), // Second moment
              ...(amsgrad ? { maxExpAvgSq: new Float64Array(size) } : {}), // AMSGrad buffer
            };
            this.state.set(param, next);
            return next;
          })();

        // Validate state buffer sizes
        assertBufferSize(state.expAvg, size, "AdamW expAvg");
        assertBufferSize(state.expAvgSq, size, "AdamW expAvgSq");
        if (amsgrad && state.maxExpAvgSq) {
          assertBufferSize(state.maxExpAvgSq, size, "AdamW maxExpAvgSq");
        }

        // Increment per-parameter step counter
        state.step += 1;

        // Compute bias correction terms
        const biasCorrection1 = 1 - beta1 ** state.step;
        const biasCorrection2 = 1 - beta2 ** state.step;

        // Compute step size with bias correction
        const stepSize = lr / biasCorrection1;

        // Update each element of the parameter
        for (let i = 0; i < size; i++) {
          // Get current gradient and parameter values
          const gi = safeArrayAccess(grad, gradOffset + i, "AdamW gradient");
          const pi = safeArrayAccess(pData, pOff + i, "AdamW parameter");

          // Validate values are finite
          assertFinite("gradient", gi);
          assertFinite("parameter", pi);

          // Get current moment estimates
          const m = safeArrayAccess(state.expAvg, i, "AdamW expAvg");
          const v = safeArrayAccess(state.expAvgSq, i, "AdamW expAvgSq");

          // Update biased first moment estimate: m(t) = β1 * m(t-1) + (1 - β1) * g(t)
          const mNew = beta1 * m + (1 - beta1) * gi;

          // Update biased second raw moment estimate: v(t) = β2 * v(t-1) + (1 - β2) * g(t)^2
          const vNew = beta2 * v + (1 - beta2) * gi * gi;

          // Store updated moments
          state.expAvg[i] = mNew;
          state.expAvgSq[i] = vNew;

          // Determine which second moment to use (AMSGrad or standard)
          let denomSq = vNew;
          if (amsgrad) {
            const maxBuf = state.maxExpAvgSq;
            if (!maxBuf) {
              throw new DeepboxError("Internal error: AMSGrad enabled but maxExpAvgSq is missing");
            }
            // AMSGrad: use maximum of all past second moments
            const maxV = Math.max(safeArrayAccess(maxBuf, i, "AdamW maxExpAvgSq"), vNew);
            maxBuf[i] = maxV;
            denomSq = maxV;
          }

          // Compute denominator with bias correction: √(v̂(t)) + ε
          const denom = Math.sqrt(denomSq / biasCorrection2) + eps;

          // AdamW update: θ(t+1) = θ(t) - lr * (m̂(t) / denom + λ * θ(t))
          // Note: weight decay is applied directly to parameters (decoupled)
          pData[pOff + i] = pi - stepSize * (mNew / denom) - lr * weightDecay * pi;
        }
      }
    }

    return loss;
  }
}
