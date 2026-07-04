/**
 * @see {@link https://deepbox.dev/docs/optim-optimizers | Deepbox documentation}
 */

import { DeviceError, InvalidParameterError } from "../../core";
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

type SparseAdamOptions = {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
};

type SparseAdamState = {
  step: number;
  expAvg: Float64Array;
  expAvgSq: Float64Array;
};

/**
 * SparseAdam optimizer — a variant of Adam designed for sparse gradients.
 *
 * Only updates the moment estimates for gradient indices that are non-zero,
 * making it efficient for parameters with sparse gradient updates such as
 * embedding layers.
 *
 * **Algorithm:**
 * For each parameter element where gradient ≠ 0:
 * ```
 * m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
 * v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
 * m̂_t = m_t / (1 - β₁^t)
 * v̂_t = v_t / (1 - β₂^t)
 * θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
 * ```
 *
 * For indices where gradient = 0, moment estimates and parameters are unchanged.
 *
 * @example
 * ```ts
 * import { SparseAdam } from 'deepbox/optim';
 * import { Embedding } from 'deepbox/nn';
 *
 * const embedding = new Embedding(10000, 128);
 * const optimizer = new SparseAdam(embedding.parameters(), { lr: 0.001 });
 * ```
 *
 * References:
 * - Kingma & Ba, "Adam: A Method for Stochastic Optimization", 2015
 * - PyTorch SparseAdam implementation
 *
 * @category Optimizers
 */
export class SparseAdam extends Optimizer<SparseAdamOptions, SparseAdamState> {
  private _stepCount = 0;

  get stepCount(): number {
    return this._stepCount;
  }

  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<SparseAdamOptions>>,
    options: {
      readonly lr?: number;
      readonly beta1?: number;
      readonly beta2?: number;
      readonly eps?: number;
    } = {}
  ) {
    const defaults = {
      lr: options.lr ?? 0.001,
      beta1: options.beta1 ?? 0.9,
      beta2: options.beta2 ?? 0.999,
      eps: options.eps ?? 1e-8,
    };

    super(params, defaults);

    assertFiniteNonNegative("learning rate", defaults.lr);
    assertInRange("beta1", defaults.beta1, 0, 1);
    assertInRange("beta2", defaults.beta2, 0, 1);
    assertFinitePositive("epsilon", defaults.eps);
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

  protected isState(state: Record<string, unknown>): state is SparseAdamState {
    return (
      typeof state["step"] === "number" &&
      state["expAvg"] instanceof Float64Array &&
      state["expAvgSq"] instanceof Float64Array
    );
  }

  step(closure?: () => number): number | undefined {
    let loss: number | undefined;

    if (closure) {
      loss = closure();
    }

    this._stepCount++;

    for (const group of this.paramGroups) {
      const { lr, beta1, beta2, eps } = group.options;

      assertFiniteNonNegative("learning rate", lr);
      assertInRange("beta1", beta1, 0, 1);
      assertInRange("beta2", beta2, 0, 1);
      assertFinitePositive("epsilon", eps);

      for (const param of group.params) {
        // SparseAdam skips zero-gradient entries, which requires per-index
        // scatter over host-readable gradients. That cannot be expressed with the
        // dense device tensor ops, so device parameters are explicitly rejected.
        if (param.tensor.isDeviceTensor) {
          throw new DeviceError(
            "SparseAdam is not supported on device tensors (it needs sparse/host-readable gradients). Move the parameters to CPU with `.to('cpu')` before optimizing."
          );
        }

        const {
          grad: gradData,
          gradOffset,
          param: paramData,
          paramOffset,
        } = assertHasGradFloat(param, "SparseAdam");
        const size = param.tensor.size;

        const existing = this.state.get(param);
        const state =
          existing ??
          (() => {
            const next: SparseAdamState = {
              step: 0,
              expAvg: new Float64Array(size),
              expAvgSq: new Float64Array(size),
            };
            this.state.set(param, next);
            return next;
          })();

        assertBufferSize(state.expAvg, size, "SparseAdam expAvg");
        assertBufferSize(state.expAvgSq, size, "SparseAdam expAvgSq");

        state.step += 1;

        // Bias correction factors
        const biasCorrection1 = 1 - beta1 ** state.step;
        const biasCorrection2 = 1 - beta2 ** state.step;

        for (let i = 0; i < size; i++) {
          const gi = safeArrayAccess(gradData, gradOffset + i, "SparseAdam gradient");

          // Only update for non-zero gradients (sparse update)
          if (gi === 0) continue;

          assertFinite("gradient", gi);

          const pi = safeArrayAccess(paramData, paramOffset + i, "SparseAdam parameter");
          assertFinite("parameter", pi);

          const m = safeArrayAccess(state.expAvg, i, "SparseAdam expAvg");
          const v = safeArrayAccess(state.expAvgSq, i, "SparseAdam expAvgSq");

          // Update biased first and second moment estimates
          const mNew = beta1 * m + (1 - beta1) * gi;
          const vNew = beta2 * v + (1 - beta2) * gi * gi;

          state.expAvg[i] = mNew;
          state.expAvgSq[i] = vNew;

          // Bias-corrected estimates
          const mHat = mNew / biasCorrection1;
          const vHat = vNew / biasCorrection2;

          // Parameter update
          paramData[paramOffset + i] = pi - (lr * mHat) / (Math.sqrt(vHat) + eps);
        }
      }
    }

    return loss;
  }
}
