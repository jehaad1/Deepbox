/**
 * @see {@link https://deepbox.dev/docs/optim-optimizers | Deepbox documentation}
 */

import { DeviceError } from "../../core";
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
 * Options for the SparseAdam optimizer.
 *
 * @property lr - Learning rate
 * @property beta1 - Decay rate of the first moment estimate
 * @property beta2 - Decay rate of the second moment estimate
 * @property eps - Small constant added to the square root of the second moment
 * @property maximize - Maximize the objective instead of minimizing it
 */
type SparseAdamOptions = {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  maximize: boolean;
};

function validateSparseAdamOptions(options: Readonly<SparseAdamOptions>): void {
  assertFiniteNonNegative("learning rate", options.lr);
  assertInRange("beta1", options.beta1, 0, 1);
  assertInRange("beta2", options.beta2, 0, 1);
  assertFinitePositive("epsilon", options.eps);
}

type SparseAdamState = {
  step: number;
  expAvg: Float64Array;
  expAvgSq: Float64Array;
};

/**
 * SparseAdam optimizer, a variant of Adam designed for sparse gradients.
 *
 * Only updates the moment estimates for gradient indices that are non-zero,
 * making it suited to parameters with sparse gradient updates such as
 * embedding layers. Gradients are dense tensors here, and an exact zero marks an
 * index that was not touched. The update follows PyTorch's `torch.optim.SparseAdam`.
 * Weight decay is not supported.
 *
 * **Algorithm:**
 * For each parameter element where gradient ≠ 0:
 * ```
 * m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
 * v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
 * θ_t = θ_{t-1} - lr * √(1 - β₂^t) / (1 - β₁^t) * m_t / (√v_t + ε)
 * ```
 *
 * For indices where gradient = 0, moment estimates and parameters are unchanged.
 * The step counter `t` advances on every call to `step()`.
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
  /**
   * Create a new SparseAdam optimizer.
   *
   * @param params - Iterable of parameters or parameter groups to optimize
   * @param options - Optimization options
   * @param options.lr - Learning rate (default: 0.001)
   * @param options.beta1 - First moment decay, in [0, 1) (default: 0.9)
   * @param options.beta2 - Second moment decay, in [0, 1) (default: 0.999)
   * @param options.eps - Numerical stability constant (default: 1e-8)
   * @param options.maximize - Maximize the objective instead of minimizing it (default: false)
   * @throws {InvalidParameterError} If a hyperparameter is out of range
   */
  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<SparseAdamOptions>>,
    options: {
      readonly lr?: number;
      readonly beta1?: number;
      readonly beta2?: number;
      readonly eps?: number;
      readonly maximize?: boolean;
    } = {}
  ) {
    const defaults: SparseAdamOptions = {
      lr: options.lr ?? 0.001,
      beta1: options.beta1 ?? 0.9,
      beta2: options.beta2 ?? 0.999,
      eps: options.eps ?? 1e-8,
      maximize: options.maximize ?? false,
    };

    super(params, defaults);
  }

  protected override validateOptions(options: Readonly<SparseAdamOptions>): void {
    validateSparseAdamOptions(options);
  }

  protected isState(state: Record<string, unknown>): state is SparseAdamState {
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
   * @throws {DeviceError} If a parameter lives on a kernel device
   * @throws {InvalidParameterError} If a group option is invalid or a gradient or
   *   parameter value is not finite
   */
  step(closure?: () => number): number | undefined {
    let loss: number | undefined;

    if (closure) {
      loss = closure();
    }

    // SparseAdam skips zero-gradient entries, which requires per-index scatter over
    // host-readable gradients. That cannot be expressed with the dense device tensor ops,
    // so device parameters are rejected up front, before any parameter is modified.
    for (const group of this.paramGroups) {
      for (const param of this.trainableParams(group)) {
        if (param.tensor.isDeviceTensor) {
          throw new DeviceError(
            "SparseAdam is not supported on device tensors (it needs sparse/host-readable gradients). Move the parameters to CPU with `.to('cpu')` before optimizing."
          );
        }
      }
    }

    this.prepareStep("SparseAdam");
    this.countStep();

    for (const group of this.paramGroups) {
      const { lr, beta1, beta2, eps, maximize } = group.options;

      for (const param of this.trainableParams(group)) {
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

        // Bias correction folded into one step size, as PyTorch does:
        // lr * sqrt(1 - beta2^t) / (1 - beta1^t).
        const biasCorrection1 = 1 - beta1 ** state.step;
        const biasCorrection2 = 1 - beta2 ** state.step;
        const stepSize = (lr * Math.sqrt(biasCorrection2)) / biasCorrection1;

        for (let i = 0; i < size; i++) {
          const giRaw = safeArrayAccess(gradData, gradOffset + i, "SparseAdam gradient");

          // Only update for non-zero gradients (sparse update)
          if (giRaw === 0) continue;

          assertFinite("gradient", giRaw);
          const gi = maximize ? -giRaw : giRaw;

          const pi = safeArrayAccess(paramData, paramOffset + i, "SparseAdam parameter");
          assertFinite("parameter", pi);

          const m = safeArrayAccess(state.expAvg, i, "SparseAdam expAvg");
          const v = safeArrayAccess(state.expAvgSq, i, "SparseAdam expAvgSq");

          // Update biased first and second moment estimates
          const mNew = beta1 * m + (1 - beta1) * gi;
          const vNew = beta2 * v + (1 - beta2) * gi * gi;

          state.expAvg[i] = mNew;
          state.expAvgSq[i] = vNew;

          // Parameter update
          paramData[paramOffset + i] = pi - (stepSize * mNew) / (Math.sqrt(vNew) + eps);
        }
      }
    }

    return loss;
  }
}
