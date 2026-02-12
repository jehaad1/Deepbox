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

type NadamOptions = {
  lr: number;
  readonly beta1: number;
  readonly beta2: number;
  readonly eps: number;
  readonly weightDecay: number;
  readonly momentumDecay: number;
};

type NadamState = {
  step: number;
  expAvg: Float64Array;
  expAvgSq: Float64Array;
  muProduct: number;
};

/**
 * Nadam (Nesterov-accelerated Adam) optimizer.
 *
 * Implements Nadam algorithm - combines Adam's adaptive learning rates with
 * Nesterov momentum for potentially faster convergence. Nadam applies Nesterov
 * acceleration to the momentum term, providing a "look-ahead" gradient.
 *
 * @example
 * ```ts
 * import { Nadam } from 'deepbox/optim';
 *
 * const optimizer = new Nadam(model.parameters(), {
 *   lr: 0.002,
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
export class Nadam extends Optimizer<NadamOptions, NadamState> {
  private _stepCount = 0;

  get stepCount(): number {
    return this._stepCount;
  }
  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<NadamOptions>>,
    options: {
      readonly lr?: number;
      readonly beta1?: number;
      readonly beta2?: number;
      readonly eps?: number;
      readonly weightDecay?: number;
      readonly momentumDecay?: number;
    } = {}
  ) {
    const defaults = {
      lr: options.lr ?? 0.002,
      beta1: options.beta1 ?? 0.9,
      beta2: options.beta2 ?? 0.999,
      eps: options.eps ?? 1e-8,
      weightDecay: options.weightDecay ?? 0,
      momentumDecay: options.momentumDecay ?? 0.004,
    };

    super(params, defaults);

    // Validate hyperparameters
    assertFiniteNonNegative("learning rate", defaults.lr);
    assertInRange("beta1", defaults.beta1, 0, 1);
    assertInRange("beta2", defaults.beta2, 0, 1);
    assertFinitePositive("epsilon", defaults.eps);
    assertFiniteNonNegative("weight_decay value", defaults.weightDecay);
    assertFiniteNonNegative("momentum_decay", defaults.momentumDecay);
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

  protected isState(state: Record<string, unknown>): state is NadamState {
    return (
      typeof state["step"] === "number" &&
      state["expAvg"] instanceof Float64Array &&
      state["expAvgSq"] instanceof Float64Array &&
      typeof state["muProduct"] === "number"
    );
  }

  step(closure?: () => number): number | undefined {
    let loss: number | undefined;

    if (closure) {
      loss = closure();
    }

    // Increment global step counter
    this._stepCount++;

    for (const group of this.paramGroups) {
      const { lr, beta1, beta2, eps, weightDecay, momentumDecay } = group.options;

      // Re-validate hyperparameters
      assertFiniteNonNegative("learning rate", lr);
      assertInRange("beta1", beta1, 0, 1);
      assertInRange("beta2", beta2, 0, 1);
      assertFinitePositive("epsilon", eps);
      assertFiniteNonNegative("weight_decay value", weightDecay);
      assertFiniteNonNegative("momentum_decay", momentumDecay);

      for (const param of group.params) {
        const {
          grad: gradData,
          gradOffset: gOff,
          param: pData,
          paramOffset: pOff,
        } = assertHasGradFloat(param, "Nadam");
        const size = param.tensor.size;

        // Initialize state if needed
        let state = this.state.get(param);
        if (!state) {
          state = {
            step: 0,
            expAvg: new Float64Array(size),
            expAvgSq: new Float64Array(size),
            muProduct: 1,
          };
          this.state.set(param, state);
        }

        // Validate state buffer sizes
        assertBufferSize(state.expAvg, size, "Nadam expAvg");
        assertBufferSize(state.expAvgSq, size, "Nadam expAvgSq");

        state.step++;
        const t = state.step;

        const biasCorrection2 = 1 - beta2 ** t;
        const mu = beta1 * (1 - 0.5 * 0.96 ** (t * momentumDecay));
        const muNext = beta1 * (1 - 0.5 * 0.96 ** ((t + 1) * momentumDecay));
        const muProduct = state.muProduct * mu;
        const muProductNext = muProduct * muNext;
        state.muProduct = muProduct;

        for (let i = 0; i < size; i++) {
          const gi0 = safeArrayAccess(gradData, gOff + i, "Nadam gradient");
          const pi = safeArrayAccess(pData, pOff + i, "Nadam parameter");
          assertFinite("gradient", gi0);
          assertFinite("parameter", pi);

          // Apply weight decay
          const gi = weightDecay !== 0 ? gi0 + weightDecay * pi : gi0;

          // Update biased first moment estimate: m(t) = β1 * m(t-1) + (1 - β1) * g(t)
          const m = safeArrayAccess(state.expAvg, i, "Nadam expAvg");
          const mNew = beta1 * m + (1 - beta1) * gi;
          state.expAvg[i] = mNew;

          // Update biased second moment estimate: v(t) = β2 * v(t-1) + (1 - β2) * g(t)²
          const v = safeArrayAccess(state.expAvgSq, i, "Nadam expAvgSq");
          const vNew = beta2 * v + (1 - beta2) * gi * gi;
          state.expAvgSq[i] = vNew;

          const denom = Math.sqrt(vNew / biasCorrection2) + eps;
          const mHatNext = mNew / (1 - muProductNext);
          const gHat = gi / (1 - muProduct);
          const mNesterov = muNext * mHatNext + (1 - mu) * gHat;
          pData[pOff + i] = pi - (lr * mNesterov) / denom;
        }
      }
    }

    return loss;
  }
}
