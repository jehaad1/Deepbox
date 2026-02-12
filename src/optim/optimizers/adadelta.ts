import { InvalidParameterError } from "../../core";
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

type AdaDeltaOptions = {
  lr: number;
  readonly rho: number;
  readonly eps: number;
  readonly weightDecay: number;
};

type AdaDeltaState = {
  squareAvg: Float64Array;
  accDelta: Float64Array;
};

/**
 * AdaDelta optimizer.
 *
 * Implements AdaDelta algorithm - an extension of Adagrad that seeks to reduce
 * its aggressive, monotonically decreasing learning rate. AdaDelta adapts learning
 * rates based on a moving window of gradient updates, rather than accumulating all
 * past gradients.
 *
 * @example
 * ```ts
 * import { AdaDelta } from 'deepbox/optim';
 *
 * const optimizer = new AdaDelta(model.parameters(), {
 *   lr: 1.0,
 *   rho: 0.9,
 *   eps: 1e-6
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
export class AdaDelta extends Optimizer<AdaDeltaOptions, AdaDeltaState> {
  private _stepCount = 0;

  get stepCount(): number {
    return this._stepCount;
  }
  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<AdaDeltaOptions>>,
    options: {
      readonly lr?: number;
      readonly rho?: number;
      readonly eps?: number;
      readonly weightDecay?: number;
    } = {}
  ) {
    const defaults = {
      lr: options.lr ?? 1.0,
      rho: options.rho ?? 0.9,
      eps: options.eps ?? 1e-6,
      weightDecay: options.weightDecay ?? 0,
    };

    super(params, defaults);

    // Validate hyperparameters
    assertFiniteNonNegative("learning rate", defaults.lr);
    assertInRange("rho", defaults.rho, 0, 1);
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

  protected isState(state: Record<string, unknown>): state is AdaDeltaState {
    return state["squareAvg"] instanceof Float64Array && state["accDelta"] instanceof Float64Array;
  }

  step(closure?: () => number): number | undefined {
    let loss: number | undefined;

    if (closure) {
      loss = closure();
    }

    // Increment global step counter
    this._stepCount++;

    for (const group of this.paramGroups) {
      const { lr, rho, eps, weightDecay } = group.options;

      // Re-validate hyperparameters
      assertFiniteNonNegative("learning rate", lr);
      assertInRange("rho", rho, 0, 1);
      assertFinitePositive("epsilon", eps);
      assertFiniteNonNegative("weight_decay value", weightDecay);

      for (const param of group.params) {
        const {
          grad: gradData,
          gradOffset: gOff,
          param: pData,
          paramOffset: pOff,
        } = assertHasGradFloat(param, "AdaDelta");
        const size = param.tensor.size;

        // Initialize state if needed
        let state = this.state.get(param);
        if (!state) {
          state = {
            squareAvg: new Float64Array(size),
            accDelta: new Float64Array(size),
          };
          this.state.set(param, state);
        }

        // Validate state buffer sizes
        assertBufferSize(state.squareAvg, size, "AdaDelta squareAvg");
        assertBufferSize(state.accDelta, size, "AdaDelta accDelta");

        for (let i = 0; i < size; i++) {
          const gi0 = safeArrayAccess(gradData, gOff + i, "AdaDelta gradient");
          const pi = safeArrayAccess(pData, pOff + i, "AdaDelta parameter");
          assertFinite("gradient", gi0);
          assertFinite("parameter", pi);

          // Apply weight decay
          const gi = weightDecay !== 0 ? gi0 + weightDecay * pi : gi0;

          // Update square average: E[g²](t) = ρ * E[g²](t-1) + (1 - ρ) * g(t)²
          const sq = safeArrayAccess(state.squareAvg, i, "AdaDelta squareAvg");
          const sqNew = rho * sq + (1 - rho) * gi * gi;
          state.squareAvg[i] = sqNew;

          // Compute RMS[g](t) = √(E[g²](t) + ε)
          const std = Math.sqrt(sqNew + eps);

          // Compute RMS[Δθ](t-1) = √(E[Δθ²](t-1) + ε)
          const accD = safeArrayAccess(state.accDelta, i, "AdaDelta accDelta");
          const rmsUpdate = Math.sqrt(accD + eps);

          // Compute parameter update: Δθ(t) = -RMS[Δθ](t-1) / RMS[g](t) * g(t)
          const delta = (rmsUpdate / std) * gi;

          // Update accumulated delta: E[Δθ²](t) = ρ * E[Δθ²](t-1) + (1 - ρ) * Δθ(t)²
          state.accDelta[i] = rho * accD + (1 - rho) * delta * delta;

          // Update parameter: θ(t+1) = θ(t) - lr * Δθ(t)
          pData[pOff + i] = pi - lr * delta;
        }
      }
    }

    return loss;
  }
}
