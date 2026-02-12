import { InvalidParameterError } from "../../core";
import type { GradTensor } from "../../ndarray";
import {
  assertFinite,
  assertFiniteNonNegative,
  assertHasGradFloat,
  safeArrayAccess,
} from "../_internal";
import { Optimizer, type ParamGroup } from "../Optimizer";

type SGDOptions = {
  lr: number;
  momentum: number;
  dampening: number;
  weightDecay: number;
  nesterov: boolean;
};

type SGDState = {
  momentumBuffer?: Float64Array;
};

/**
 * Stochastic Gradient Descent (SGD) optimizer.
 *
 * Implements vanilla SGD with optional momentum, weight decay, and Nesterov acceleration.
 *
 * @example
 * ```ts
 * import { SGD } from 'deepbox/optim';
 * import { Module } from 'deepbox/nn';
 *
 * const model: Module = ...;
 * const optimizer = new SGD(model.parameters(), {
 *   lr: 0.01,
 *   momentum: 0.9,
 *   weightDecay: 5e-4,
 *   nesterov: true
 * });
 *
 * // Training loop
 * for (let epoch = 0; epoch < numEpochs; epoch++) {
 *   for (const [inputs, targets] of dataLoader) {
 *     optimizer.zeroGrad();
 *     const outputs = model.forward(inputs);
 *     const loss = criterion(outputs, targets);
 *     loss.backward();
 *     optimizer.step();
 *   }
 * }
 * ```
 *
 * @category Optimizers
 */
export class SGD extends Optimizer<SGDOptions, SGDState> {
  /** Internal counter tracking total number of optimization steps */
  private _stepCount = 0;

  get stepCount(): number {
    return this._stepCount;
  }
  /**
   * Create a new SGD optimizer.
   *
   * @param params - Iterable of parameters or parameter groups to optimize
   * @param options - Optimization options
   * @param options.lr - Learning rate (default: 0.01)
   * @param options.momentum - Momentum factor (default: 0)
   * @param options.dampening - Dampening for momentum (default: 0)
   * @param options.weightDecay - Weight decay (L2 penalty) (default: 0)
   * @param options.nesterov - Enable Nesterov momentum (default: false)
   */
  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<SGDOptions>>,
    options: {
      readonly lr?: number;
      readonly momentum?: number;
      readonly dampening?: number;
      readonly weightDecay?: number;
      readonly nesterov?: boolean;
    } = {}
  ) {
    const defaults = {
      lr: options.lr ?? 0.01,
      momentum: options.momentum ?? 0,
      dampening: options.dampening ?? 0,
      weightDecay: options.weightDecay ?? 0,
      nesterov: options.nesterov ?? false,
    };

    super(params, defaults);

    // Validate options
    assertFiniteNonNegative("learning rate", defaults.lr);
    assertFiniteNonNegative("momentum value", defaults.momentum);
    assertFiniteNonNegative("dampening", defaults.dampening);
    assertFiniteNonNegative("weight_decay value", defaults.weightDecay);
    if (defaults.nesterov && (defaults.momentum <= 0 || defaults.dampening !== 0)) {
      throw new InvalidParameterError(
        "Nesterov momentum requires a momentum and zero dampening",
        "nesterov",
        {
          momentum: defaults.momentum,
          dampening: defaults.dampening,
          nesterov: defaults.nesterov,
        }
      );
    }
  }

  /**
   * Perform a single optimization step.
   *
   * Implements the SGD update rule with optional momentum and weight decay.
   *
   * @param closure - Optional closure that reevaluates the model and returns the loss
   * @returns Loss value if closure is provided
   */
  protected isState(state: Record<string, unknown>): state is SGDState {
    if (
      state["momentumBuffer"] !== undefined &&
      !(state["momentumBuffer"] instanceof Float64Array)
    ) {
      return false;
    }
    return true;
  }

  step(closure?: () => number): number | undefined {
    let loss: number | undefined;

    // Evaluate loss if closure provided
    if (closure) {
      loss = closure();
    }

    // Increment global step counter
    this._stepCount++;

    // Update each parameter group
    for (const group of this.paramGroups) {
      const { lr, momentum, dampening, weightDecay, nesterov } = group.options;

      assertFiniteNonNegative("learning rate", lr);
      assertFiniteNonNegative("momentum value", momentum);
      assertFiniteNonNegative("dampening", dampening);
      assertFiniteNonNegative("weight_decay value", weightDecay);

      if (nesterov && (momentum <= 0 || dampening !== 0)) {
        throw new InvalidParameterError(
          "Nesterov momentum requires a momentum and zero dampening",
          "nesterov",
          { momentum, dampening, nesterov }
        );
      }

      for (const param of group.params) {
        const {
          grad: gradData,
          gradOffset,
          param: paramData,
          paramOffset,
        } = assertHasGradFloat(param, "SGD");
        const size = param.tensor.size;

        let state = this.state.get(param);
        if (!state) {
          state = {};
          this.state.set(param, state);
        }

        // Momentum buffer is stored densely (one value per element).
        let momentumBuffer: Float64Array | undefined;
        if (momentum !== 0) {
          if (!state.momentumBuffer) {
            state.momentumBuffer = new Float64Array(size);
          }
          momentumBuffer = state.momentumBuffer;
        }

        for (let i = 0; i < size; i++) {
          const gi = safeArrayAccess(gradData, gradOffset + i, "SGD gradient");
          const pi = safeArrayAccess(paramData, paramOffset + i, "SGD parameter");
          assertFinite("gradient", gi);
          assertFinite("parameter", pi);

          // d_p = grad + weightDecay * param
          let d = gi;
          if (weightDecay !== 0) {
            d = d + weightDecay * pi;
          }

          if (momentumBuffer) {
            const bPrev = safeArrayAccess(momentumBuffer, i, "SGD momentum buffer");
            const bNew = momentum * bPrev + (1 - dampening) * d;
            momentumBuffer[i] = bNew;
            d = nesterov ? d + momentum * bNew : bNew;
          }

          // param -= lr * d
          paramData[paramOffset + i] = pi - lr * d;
        }
      }
    }

    return loss;
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
}
