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
  type Tensor,
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

type AdaDeltaOptions = {
  lr: number;
  readonly rho: number;
  readonly eps: number;
  readonly weightDecay: number;
  maximize: boolean;
};

type AdaDeltaState = {
  squareAvg: Float64Array;
  accDelta: Float64Array;
  /** Device state (used when the parameter lives on a kernel device). */
  squareAvgTensor?: Tensor;
  accDeltaTensor?: Tensor;
};

/**
 * AdaDelta optimizer (Zeiler, 2012).
 *
 * AdaDelta is an extension of Adagrad that avoids its monotonically shrinking
 * learning rate. It keeps exponential moving averages of the squared gradients and
 * of the squared parameter updates, and scales each step by the ratio of their
 * root-mean-squares. The update follows `torch.optim.Adadelta`:
 *
 * ```
 * v = rho * v + (1 - rho) * g^2
 * delta = sqrt(acc + eps) / sqrt(v + eps) * g
 * acc = rho * acc + (1 - rho) * delta^2
 * theta -= lr * delta
 * ```
 *
 * When `weightDecay` is non-zero, `weightDecay * theta` is added to the gradient
 * (L2 penalty) before the update.
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
  /**
   * Create a new AdaDelta optimizer.
   *
   * @param params - Parameters to optimize, or an array of parameter groups with per-group options
   * @param options - Hyperparameters
   * @param options.lr - Coefficient applied to the computed update (default: 1.0)
   * @param options.rho - Decay rate of the moving averages, in [0, 1) (default: 0.9)
   * @param options.eps - Term added to the denominators for numerical stability (default: 1e-6)
   * @param options.weightDecay - L2 penalty coefficient (default: 0)
   * @param options.maximize - Maximize the objective instead of minimizing it (default: false)
   * @throws {InvalidParameterError} If a hyperparameter is out of range
   */
  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<AdaDeltaOptions>>,
    options: {
      readonly lr?: number;
      readonly rho?: number;
      readonly eps?: number;
      readonly weightDecay?: number;
      readonly maximize?: boolean;
    } = {}
  ) {
    const defaults = {
      lr: options.lr ?? 1.0,
      rho: options.rho ?? 0.9,
      eps: options.eps ?? 1e-6,
      weightDecay: options.weightDecay ?? 0,
      maximize: options.maximize ?? false,
    };

    super(params, defaults);
  }

  protected override validateOptions(options: Readonly<AdaDeltaOptions>): void {
    assertFiniteNonNegative("learning rate", options.lr);
    assertInRange("rho", options.rho, 0, 1);
    assertFinitePositive("epsilon", options.eps);
    assertFiniteNonNegative("weight_decay value", options.weightDecay);
  }

  protected isState(state: Record<string, unknown>): state is AdaDeltaState {
    return state["squareAvg"] instanceof Float64Array && state["accDelta"] instanceof Float64Array;
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

    this.prepareStep("AdaDelta");
    this.countStep();

    for (const group of this.paramGroups) {
      const { lr, rho, eps, weightDecay, maximize } = group.options;

      // Re-validate hyperparameters

      for (const param of this.trainableParams(group)) {
        // Device path: compose the AdaDelta update from device-dispatched ops.
        if (param.tensor.isDeviceTensor) {
          const rawGrad = param.grad;
          if (!rawGrad) continue;
          const g = maximize ? mulScalar(rawGrad, -1) : rawGrad;
          let dstate = this.state.get(param);
          if (!dstate) {
            dstate = { squareAvg: new Float64Array(0), accDelta: new Float64Array(0) };
            this.state.set(param, dstate);
          }
          const grad = weightDecay !== 0 ? add(g, mulScalar(param.tensor, weightDecay)) : g;
          const sqPrev = dstate.squareAvgTensor;
          // E[g^2](t) = rho * E[g^2](t-1) + (1 - rho) * g^2
          const sqNew = sqPrev
            ? add(mulScalar(sqPrev, rho), mulScalar(square(grad), 1 - rho))
            : mulScalar(square(grad), 1 - rho);
          dstate.squareAvgTensor = sqNew;
          const std = sqrt(addScalar(sqNew, eps));
          // RMS[dx](t-1) = sqrt(E[dx^2](t-1) + eps); first step acc=0 -> sqrt(eps).
          const accPrev = dstate.accDeltaTensor ?? mulScalar(sqNew, 0);
          const rms = sqrt(addScalar(accPrev, eps));
          // delta = (RMS[dx](t-1) / RMS[g](t)) * g
          const delta = mul(div(rms, std), grad);
          // E[dx^2](t) = rho * E[dx^2](t-1) + (1 - rho) * delta^2
          const accNew = dstate.accDeltaTensor
            ? add(mulScalar(accPrev, rho), mulScalar(square(delta), 1 - rho))
            : mulScalar(square(delta), 1 - rho);
          dstate.accDeltaTensor = accNew;
          replaceParamStorage(param, "tensor", sub(param.tensor, mulScalar(delta, lr)));
          continue;
        }

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

        const squareAvg = state.squareAvg;
        const accDelta = state.accDelta;
        for (let i = 0; i < size; i++) {
          const rawGi = gradData[gOff + i] as number;
          const pi = pData[pOff + i] as number;
          if (!Number.isFinite(rawGi)) assertFinite("gradient", rawGi);
          const gi0 = maximize ? -rawGi : rawGi;
          if (!Number.isFinite(pi)) assertFinite("parameter", pi);

          // L2 weight decay
          const gi = weightDecay !== 0 ? gi0 + weightDecay * pi : gi0;

          // E[g^2](t) = rho * E[g^2](t-1) + (1 - rho) * g(t)^2
          const sqNew = rho * (squareAvg[i] as number) + (1 - rho) * gi * gi;
          squareAvg[i] = sqNew;

          // RMS[g](t) = sqrt(E[g^2](t) + eps)
          const std = Math.sqrt(sqNew + eps);

          // RMS[delta](t-1) = sqrt(E[delta^2](t-1) + eps)
          const accD = accDelta[i] as number;
          const rmsUpdate = Math.sqrt(accD + eps);

          // delta(t) = RMS[delta](t-1) / RMS[g](t) * g(t)
          const delta = (rmsUpdate / std) * gi;

          // E[delta^2](t) = rho * E[delta^2](t-1) + (1 - rho) * delta(t)^2
          accDelta[i] = rho * accD + (1 - rho) * delta * delta;

          // theta(t+1) = theta(t) - lr * delta(t)
          pData[pOff + i] = pi - lr * delta;
        }
      }
    }

    return loss;
  }
}
