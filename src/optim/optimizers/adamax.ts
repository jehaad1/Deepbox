/**
 * @see {@link https://deepbox.dev/docs/optim-optimizers | Deepbox documentation}
 */

import {
  abs,
  add,
  addScalar,
  div,
  type GradTensor,
  mulScalar,
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

type AdamaxOptions = {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  maximize: boolean;
};

type AdamaxState = {
  step: number;
  expAvg: Float64Array;
  expInfNorm: Float64Array;
  /** Device state (used when the parameter lives on a kernel device). */
  expAvgTensor?: Tensor;
  expInfNormTensor?: Tensor;
  deviceStep?: number;
};

/**
 * Adamax optimizer, a variant of Adam based on the infinity norm (Kingma and Ba, 2015).
 *
 * Often used for embeddings and sparse gradients. The update follows
 * `torch.optim.Adamax`:
 *
 * ```
 * m = beta1 * m + (1 - beta1) * g
 * u = max(beta2 * u, |g| + eps)
 * theta -= (lr / (1 - beta1^t)) * m / u
 * ```
 *
 * When `weightDecay` is non-zero, `weightDecay * theta` is added to the gradient
 * (L2 penalty) before the update.
 *
 * @example
 * ```ts
 * import { Adamax } from 'deepbox/optim';
 *
 * const optimizer = new Adamax(model.parameters(), { lr: 0.002 });
 * ```
 *
 * @category Optimizers
 */
export class Adamax extends Optimizer<AdamaxOptions, AdamaxState> {
  /**
   * Create a new Adamax optimizer.
   *
   * @param params - Parameters to optimize, or an array of parameter groups with per-group options
   * @param options - Hyperparameters
   * @param options.lr - Learning rate (default: 0.002)
   * @param options.beta1 - Decay rate of the first moment, in [0, 1) (default: 0.9)
   * @param options.beta2 - Decay rate of the infinity norm, in [0, 1) (default: 0.999)
   * @param options.eps - Term added to the infinity norm for numerical stability (default: 1e-8)
   * @param options.weightDecay - L2 penalty coefficient (default: 0)
   * @param options.maximize - Maximize the objective instead of minimizing it (default: false)
   * @throws {InvalidParameterError} If a hyperparameter is out of range
   */
  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<AdamaxOptions>>,
    options: {
      readonly lr?: number;
      readonly beta1?: number;
      readonly beta2?: number;
      readonly eps?: number;
      readonly weightDecay?: number;
      readonly maximize?: boolean;
    } = {}
  ) {
    const defaults = {
      lr: options.lr ?? 0.002,
      beta1: options.beta1 ?? 0.9,
      beta2: options.beta2 ?? 0.999,
      eps: options.eps ?? 1e-8,
      weightDecay: options.weightDecay ?? 0,
      maximize: options.maximize ?? false,
    };

    super(params, defaults);
  }

  protected override validateOptions(options: Readonly<AdamaxOptions>): void {
    assertFiniteNonNegative("learning rate", options.lr);
    assertInRange("beta1", options.beta1, 0, 1);
    assertInRange("beta2", options.beta2, 0, 1);
    assertFinitePositive("epsilon", options.eps);
    assertFiniteNonNegative("weight_decay value", options.weightDecay);
  }

  protected isState(state: Record<string, unknown>): state is AdamaxState {
    return (
      typeof state["step"] === "number" &&
      state["expAvg"] instanceof Float64Array &&
      state["expInfNorm"] instanceof Float64Array
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
    if (closure) loss = closure();

    this.prepareStep("Adamax");
    this.countStep();

    for (const group of this.paramGroups) {
      const { lr, beta1, beta2, eps, weightDecay, maximize } = group.options;

      // Re-validate hyperparameters (they might have been changed)

      for (const param of this.trainableParams(group)) {
        // Device path: compose the Adamax update from device-dispatched ops.
        if (param.tensor.isDeviceTensor) {
          const rawGrad = param.grad;
          if (!rawGrad) continue;
          const g = maximize ? mulScalar(rawGrad, -1) : rawGrad;
          let dstate = this.state.get(param);
          if (!dstate) {
            dstate = { step: 0, expAvg: new Float64Array(0), expInfNorm: new Float64Array(0) };
            this.state.set(param, dstate);
          }
          const t = (dstate.deviceStep ?? 0) + 1;
          dstate.deviceStep = t;
          const biasCorrection1 = 1 - beta1 ** t;
          const stepSize = lr / biasCorrection1;
          const grad = weightDecay !== 0 ? add(g, mulScalar(param.tensor, weightDecay)) : g;
          const mPrev = dstate.expAvgTensor;
          const uPrev = dstate.expInfNormTensor;
          const mNew = mPrev
            ? add(mulScalar(mPrev, beta1), mulScalar(grad, 1 - beta1))
            : mulScalar(grad, 1 - beta1);
          // u(t) = max(beta2 * u(t-1), |g| + eps); first step u=0 -> |g| + eps.
          const absGradEps = addScalar(abs(grad), eps);
          const uNew = uPrev ? deviceMaxTensor(mulScalar(uPrev, beta2), absGradEps) : absGradEps;
          dstate.expAvgTensor = mNew;
          dstate.expInfNormTensor = uNew;
          replaceParamStorage(
            param,
            "tensor",
            sub(param.tensor, mulScalar(div(mNew, uNew), stepSize))
          );
          continue;
        }

        const {
          grad: gradData,
          gradOffset,
          param: paramData,
          paramOffset,
        } = assertHasGradFloat(param, "Adamax");
        const size = param.tensor.size;

        const existing = this.state.get(param);
        const state =
          existing ??
          (() => {
            const next: AdamaxState = {
              step: 0,
              expAvg: new Float64Array(size),
              expInfNorm: new Float64Array(size),
            };
            this.state.set(param, next);
            return next;
          })();

        assertBufferSize(state.expAvg, size, "Adamax expAvg");
        assertBufferSize(state.expInfNorm, size, "Adamax expInfNorm");

        state.step += 1;

        const biasCorrection1 = 1 - beta1 ** state.step;
        const stepSize = lr / biasCorrection1;

        const expAvg = state.expAvg;
        const expInfNorm = state.expInfNorm;

        for (let i = 0; i < size; i++) {
          const rawGi = gradData[gradOffset + i] as number;
          const pi = paramData[paramOffset + i] as number;
          if (!Number.isFinite(rawGi)) assertFinite("gradient", rawGi);
          const gi0 = maximize ? -rawGi : rawGi;
          if (!Number.isFinite(pi)) assertFinite("parameter", pi);

          // L2 weight decay
          const gi = weightDecay !== 0 ? gi0 + weightDecay * pi : gi0;

          const mNew = beta1 * (expAvg[i] as number) + (1 - beta1) * gi;
          // eps is folded into the norm (as torch does) so the denominator is never zero.
          const uNew = Math.max(beta2 * (expInfNorm[i] as number), Math.abs(gi) + eps);

          expAvg[i] = mNew;
          expInfNorm[i] = uNew;

          paramData[paramOffset + i] = pi - (stepSize * mNew) / uNew;
        }
      }
    }

    return loss;
  }
}
