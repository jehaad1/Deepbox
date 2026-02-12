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

/**
 * Options for the RMSprop optimizer.
 *
 * @property lr - Learning rate (step size)
 * @property alpha - Smoothing constant for moving average of squared gradients
 * @property eps - Small constant for numerical stability
 * @property weightDecay - Weight decay coefficient (L2 penalty)
 * @property momentum - Momentum factor
 * @property centered - Whether to use centered RMSprop variant
 */
type RMSpropOptions = {
  lr: number;
  alpha: number;
  eps: number;
  weightDecay: number;
  momentum: number;
  centered: boolean;
};

/**
 * State maintained per parameter by RMSprop.
 *
 * @property squareAvg - Exponentially weighted average of squared gradients
 * @property momentumBuffer - Momentum buffer (if momentum > 0)
 * @property gradAvg - Exponentially weighted average of gradients (centered variant only)
 */
type RMSpropState = {
  squareAvg: Float64Array;
  momentumBuffer?: Float64Array;
  gradAvg?: Float64Array;
};

/**
 * RMSprop (Root Mean Square Propagation) optimizer.
 *
 * RMSprop adapts the learning rate for each parameter by dividing by a running
 * average of recent gradient magnitudes. This helps with non-stationary objectives
 * and is particularly effective for RNNs.
 *
 * @example
 * ```ts
 * import { RMSprop } from 'deepbox/optim';
 *
 * const optimizer = new RMSprop(model.parameters(), {
 *   lr: 0.01,
 *   alpha: 0.99,
 *   momentum: 0.9,
 *   centered: true
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
export class RMSprop extends Optimizer<RMSpropOptions, RMSpropState> {
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
   * Create a new RMSprop optimizer.
   *
   * @param params - Iterable of parameters or parameter groups to optimize
   * @param options - Optimization options
   * @param options.lr - Learning rate (default: 0.01)
   * @param options.alpha - Smoothing constant (default: 0.99)
   * @param options.eps - Numerical stability constant (default: 1e-8)
   * @param options.weightDecay - Weight decay coefficient (default: 0)
   * @param options.momentum - Momentum factor (default: 0)
   * @param options.centered - Use centered variant (default: false)
   * @throws {InvalidParameterError} If a parameter is invalid
   */
  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<RMSpropOptions>>,
    options: {
      readonly lr?: number;
      readonly alpha?: number;
      readonly eps?: number;
      readonly weightDecay?: number;
      readonly momentum?: number;
      readonly centered?: boolean;
    } = {}
  ) {
    // Set default values for all options
    const defaults = {
      lr: options.lr ?? 0.01,
      alpha: options.alpha ?? 0.99,
      eps: options.eps ?? 1e-8,
      weightDecay: options.weightDecay ?? 0,
      momentum: options.momentum ?? 0,
      centered: options.centered ?? false,
    };

    super(params, defaults);

    // Validate all hyperparameters
    assertFiniteNonNegative("learning rate", defaults.lr);
    if (!Number.isFinite(defaults.alpha) || defaults.alpha < 0 || defaults.alpha > 1) {
      throw new InvalidParameterError(
        `Invalid alpha: ${defaults.alpha} (must be in range [0, 1])`,
        "alpha",
        defaults.alpha
      );
    }
    assertFinitePositive("epsilon", defaults.eps);
    assertFiniteNonNegative("weight_decay value", defaults.weightDecay);
    assertFiniteNonNegative("momentum value", defaults.momentum);
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

  protected isState(state: Record<string, unknown>): state is RMSpropState {
    if (!(state["squareAvg"] instanceof Float64Array)) return false;
    if (
      state["momentumBuffer"] !== undefined &&
      !(state["momentumBuffer"] instanceof Float64Array)
    ) {
      return false;
    }
    if (state["gradAvg"] !== undefined && !(state["gradAvg"] instanceof Float64Array)) {
      return false;
    }
    return true;
  }

  step(closure?: () => number): number | undefined {
    let loss: number | undefined;

    // Evaluate closure if provided
    if (closure) {
      loss = closure();
    }

    // Increment global step counter
    this._stepCount++;

    // Update each parameter group
    for (const group of this.paramGroups) {
      const { lr, alpha, eps, weightDecay, momentum, centered } = group.options;

      // Re-validate hyperparameters
      assertFiniteNonNegative("learning rate", lr);
      if (!Number.isFinite(alpha) || alpha < 0 || alpha > 1) {
        throw new InvalidParameterError(
          `Invalid alpha: ${alpha} (must be in range [0, 1])`,
          "alpha",
          alpha
        );
      }
      assertFinitePositive("epsilon", eps);
      assertFiniteNonNegative("weight_decay value", weightDecay);
      assertFiniteNonNegative("momentum value", momentum);

      // Update each parameter in the group
      for (const param of group.params) {
        // Get gradient and validate
        const {
          grad: gradData,
          gradOffset: gOff,
          param: pData,
          paramOffset: pOff,
        } = assertHasGradFloat(param, "RMSprop");
        const size = param.tensor.size;

        // Get or initialize optimizer state for this parameter
        let state = this.state.get(param);
        if (!state) {
          state = {
            squareAvg: new Float64Array(size),
          };
          this.state.set(param, state);
        }

        // Initialize momentum buffer if needed
        if (momentum > 0 && !state.momentumBuffer) {
          state.momentumBuffer = new Float64Array(size);
        }

        // Initialize gradient average buffer if centered variant is used
        if (centered && !state.gradAvg) {
          state.gradAvg = new Float64Array(size);
        }

        // Validate state buffer sizes
        assertBufferSize(state.squareAvg, size, "RMSprop squareAvg");
        if (momentum > 0 && state.momentumBuffer) {
          assertBufferSize(state.momentumBuffer, size, "RMSprop momentumBuffer");
        }
        if (centered && state.gradAvg) {
          assertBufferSize(state.gradAvg, size, "RMSprop gradAvg");
        }

        // Update each element of the parameter
        for (let i = 0; i < size; i++) {
          // Get current gradient and parameter values
          const gi = safeArrayAccess(gradData, gOff + i, "RMSprop gradient");
          const pi = safeArrayAccess(pData, pOff + i, "RMSprop parameter");

          // Validate values are finite
          assertFinite("gradient", gi);
          assertFinite("parameter", pi);

          // Apply weight decay to gradient if specified
          let grad = gi;
          if (weightDecay !== 0) {
            grad = grad + weightDecay * pi;
          }

          // Update exponentially weighted average of squared gradients
          // v(t) = α * v(t-1) + (1 - α) * g(t)^2
          const sqAvg = safeArrayAccess(state.squareAvg, i, "RMSprop squareAvg");
          const sqAvgNew = alpha * sqAvg + (1 - alpha) * grad * grad;
          state.squareAvg[i] = sqAvgNew;

          // Compute adaptive learning rate denominator
          let avg = sqAvgNew;

          // Centered variant: subtract squared mean of gradients
          if (centered) {
            const gAvg = state.gradAvg ? safeArrayAccess(state.gradAvg, i, "RMSprop gradAvg") : 0;
            // Update exponentially weighted average of gradients
            const gAvgNew = alpha * gAvg + (1 - alpha) * grad;
            if (state.gradAvg) state.gradAvg[i] = gAvgNew;
            // Use variance instead of second moment
            avg = sqAvgNew - gAvgNew * gAvgNew;
          }

          const denom = centered ? Math.sqrt(Math.max(avg, 0) + eps) : Math.sqrt(avg) + eps;
          const normalizedGrad = grad / denom;

          // Apply momentum if specified
          if (momentum > 0) {
            const buf = state.momentumBuffer
              ? safeArrayAccess(state.momentumBuffer, i, "RMSprop momentumBuffer")
              : 0;
            const bufNew = momentum * buf + normalizedGrad;
            if (state.momentumBuffer) state.momentumBuffer[i] = bufNew;
            // Update parameter with momentum
            pData[pOff + i] = pi - lr * bufNew;
          } else {
            // Update parameter without momentum
            pData[pOff + i] = pi - lr * normalizedGrad;
          }
        }
      }
    }

    return loss;
  }
}
