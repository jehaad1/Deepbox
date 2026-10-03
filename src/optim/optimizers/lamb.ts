/**
 * @see {@link https://deepbox.dev/docs/optim-optimizers | Deepbox documentation}
 */

import {
  add,
  addScalar,
  div,
  type GradTensor,
  mul,
  mulScalar,
  sqrt,
  square,
  sub,
  sum,
  type Tensor,
  tensor,
  where,
} from "../../ndarray";
import {
  assertBufferSize,
  assertFinite,
  assertFiniteNonNegative,
  assertFinitePositive,
  assertHasGradFloat,
  assertInRange,
  replaceParamStorage,
} from "../_internal";
import { Optimizer, type ParamGroup } from "../Optimizer";

/**
 * Options for the LAMB optimizer.
 *
 * @property lr - Learning rate (step size)
 * @property beta1 - Exponential decay rate for first moment estimates
 * @property beta2 - Exponential decay rate for second moment estimates
 * @property eps - Small constant for numerical stability
 * @property weightDecay - Weight decay coefficient
 */
type LAMBOptions = {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
};

/**
 * State maintained per parameter by LAMB.
 *
 * @property step - Number of optimization steps taken
 * @property expAvg - First moment estimate
 * @property expAvgSq - Second moment estimate
 */
type LAMBState = {
  step: number;
  expAvg: Float64Array;
  expAvgSq: Float64Array;
  /** Device state (used when the parameter lives on a kernel device). */
  expAvgTensor?: Tensor;
  expAvgSqTensor?: Tensor;
  deviceStep?: number;
};

/**
 * LAMB (Layer-wise Adaptive Moments optimizer for Batch training) optimizer.
 *
 * LAMB extends Adam with layer-wise adaptive learning rates, enabling
 * training with very large batch sizes (up to 64K for BERT).
 * It combines the benefits of Adam's per-element adaptivity with
 * LARS-style layer-wise trust ratios. Each parameter tensor is treated as one layer:
 *
 * ```
 * r = m_hat / (sqrt(v_hat) + eps) + weightDecay * theta
 * trust = ||theta|| / ||r||        (1 when either norm is zero)
 * theta -= lr * trust * r
 * ```
 *
 * where `m_hat` and `v_hat` are the bias-corrected Adam moment estimates.
 *
 * Reference: "Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes"
 * (You et al., 2019)
 *
 * @example
 * ```ts
 * import { LAMB } from 'deepbox/optim';
 *
 * const optimizer = new LAMB(model.parameters(), {
 *   lr: 0.001,
 *   beta1: 0.9,
 *   beta2: 0.999,
 *   weightDecay: 0.01,
 * });
 *
 * for (let epoch = 0; epoch < numEpochs; epoch++) {
 *   optimizer.zeroGrad();
 *   // ...
 *   optimizer.step();
 * }
 * ```
 *
 * @category Optimizers
 */
export class LAMB extends Optimizer<LAMBOptions, LAMBState> {
  /**
   * Create a new LAMB optimizer.
   *
   * @param params - Parameters to optimize, or an array of parameter groups with per-group options
   * @param options - Hyperparameters
   * @param options.lr - Learning rate (default: 0.001)
   * @param options.beta1 - Decay rate of the first moment, in [0, 1) (default: 0.9)
   * @param options.beta2 - Decay rate of the second moment, in [0, 1) (default: 0.999)
   * @param options.eps - Term added to the denominator for numerical stability (default: 1e-6)
   * @param options.weightDecay - Decoupled weight decay coefficient (default: 0.01)
   * @throws {InvalidParameterError} If a hyperparameter is out of range
   */
  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<LAMBOptions>>,
    options: {
      readonly lr?: number;
      readonly beta1?: number;
      readonly beta2?: number;
      readonly eps?: number;
      readonly weightDecay?: number;
    } = {}
  ) {
    const defaults: LAMBOptions = {
      lr: options.lr ?? 0.001,
      beta1: options.beta1 ?? 0.9,
      beta2: options.beta2 ?? 0.999,
      eps: options.eps ?? 1e-6,
      weightDecay: options.weightDecay ?? 0.01,
    };

    super(params, defaults);
  }

  protected override validateOptions(options: Readonly<LAMBOptions>): void {
    assertFiniteNonNegative("learning rate", options.lr);
    assertInRange("beta1", options.beta1, 0, 1);
    assertInRange("beta2", options.beta2, 0, 1);
    assertFinitePositive("epsilon", options.eps);
    assertFiniteNonNegative("weight_decay value", options.weightDecay);
  }

  protected isState(state: Record<string, unknown>): state is LAMBState {
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
   * @param closure - Optional function that re-evaluates the model and returns the loss
   * @returns The value returned by `closure`, or undefined when no closure is given
   * @throws {InvalidParameterError} If a gradient or parameter value is not finite
   */
  step(closure?: () => number): number | undefined {
    let loss: number | undefined;
    if (closure) {
      loss = closure();
    }

    this.prepareStep("LAMB");
    this.countStep();

    for (const group of this.paramGroups) {
      const { lr, beta1, beta2, eps, weightDecay } = group.options;

      for (const param of this.trainableParams(group)) {
        // Device path: compose the LAMB update (Adam step + layer-wise trust
        // ratio) from device-dispatched ops. The trust ratio is a device scalar
        // (norms are full reductions), kept resident and broadcast into the update.
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
          const mPrev = dstate.expAvgTensor;
          const vPrev = dstate.expAvgSqTensor;
          const mNew = mPrev
            ? add(mulScalar(mPrev, beta1), mulScalar(g, 1 - beta1))
            : mulScalar(g, 1 - beta1);
          const vNew = vPrev
            ? add(mulScalar(vPrev, beta2), mulScalar(square(g), 1 - beta2))
            : mulScalar(square(g), 1 - beta2);
          dstate.expAvgTensor = mNew;
          dstate.expAvgSqTensor = vNew;
          const mHat = mulScalar(mNew, 1 / biasCorrection1);
          const vHat = mulScalar(vNew, 1 / biasCorrection2);
          // u = mHat / (sqrt(vHat) + eps) + weightDecay * param
          const u = add(
            div(mHat, addScalar(sqrt(vHat), eps)),
            mulScalar(param.tensor, weightDecay)
          );
          const paramNorm = sqrt(sum(square(param.tensor)));
          const updateNorm = sqrt(sum(square(u)));
          // trustRatio = (paramNorm>0 && updateNorm>0) ? paramNorm/updateNorm : 1
          const cond = mul(paramNorm, updateNorm);
          const trustRatio = where(cond, div(paramNorm, updateNorm), tensor(1));
          // theta -= lr * trustRatio * u
          replaceParamStorage(
            param,
            "tensor",
            sub(param.tensor, mulScalar(mul(u, trustRatio), lr))
          );
          continue;
        }

        const {
          grad,
          gradOffset,
          param: pData,
          paramOffset: pOff,
        } = assertHasGradFloat(param, "LAMB");
        const size = param.tensor.size;

        const existing = this.state.get(param);
        const state =
          existing ??
          (() => {
            const next: LAMBState = {
              step: 0,
              expAvg: new Float64Array(size),
              expAvgSq: new Float64Array(size),
            };
            this.state.set(param, next);
            return next;
          })();

        assertBufferSize(state.expAvg, size, "LAMB expAvg");
        assertBufferSize(state.expAvgSq, size, "LAMB expAvgSq");

        state.step += 1;

        const biasCorrection1 = 1 - beta1 ** state.step;
        const biasCorrection2 = 1 - beta2 ** state.step;

        const expAvg = state.expAvg;
        const expAvgSq = state.expAvgSq;

        // First pass: update the moments and accumulate ||theta||^2 and ||r||^2,
        // where r is the Adam direction plus weight decay.
        let paramNormSq = 0;
        let updateNormSq = 0;

        for (let i = 0; i < size; i++) {
          const gi = grad[gradOffset + i] as number;
          const pi = pData[pOff + i] as number;
          if (!Number.isFinite(gi)) assertFinite("gradient", gi);
          if (!Number.isFinite(pi)) assertFinite("parameter", pi);

          // Update moments
          const mNew = beta1 * (expAvg[i] as number) + (1 - beta1) * gi;
          const vNew = beta2 * (expAvgSq[i] as number) + (1 - beta2) * gi * gi;
          expAvg[i] = mNew;
          expAvgSq[i] = vNew;

          // Adam update + weight decay
          const u =
            mNew / biasCorrection1 / (Math.sqrt(vNew / biasCorrection2) + eps) + weightDecay * pi;

          paramNormSq += pi * pi;
          updateNormSq += u * u;
        }

        const paramNorm = Math.sqrt(paramNormSq);
        const updateNorm = Math.sqrt(updateNormSq);

        // LAMB trust ratio
        let trustRatio = 1.0;
        if (paramNorm > 0 && updateNorm > 0) {
          trustRatio = paramNorm / updateNorm;
        }

        // Second pass: recompute r from the stored moments (parameters are unchanged
        // until written here) instead of keeping a temporary buffer.
        const effectiveLr = lr * trustRatio;
        for (let i = 0; i < size; i++) {
          const pi = pData[pOff + i] as number;
          const mHat = (expAvg[i] as number) / biasCorrection1;
          const vHat = (expAvgSq[i] as number) / biasCorrection2;
          const u = mHat / (Math.sqrt(vHat) + eps) + weightDecay * pi;
          pData[pOff + i] = pi - effectiveLr * u;
        }
      }
    }

    return loss;
  }
}
