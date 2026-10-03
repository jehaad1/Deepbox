/**
 * @see {@link https://deepbox.dev/docs/optim-optimizers | Deepbox documentation}
 */

import {
  add,
  addScalar,
  div,
  type GradTensor,
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

type NadamOptions = {
  lr: number;
  readonly beta1: number;
  readonly beta2: number;
  readonly eps: number;
  readonly weightDecay: number;
  readonly momentumDecay: number;
  readonly decoupledWeightDecay: boolean;
  maximize: boolean;
};

type NadamState = {
  step: number;
  expAvg: Float64Array;
  expAvgSq: Float64Array;
  muProduct: number;
  /** Device state (used when the parameter lives on a kernel device). */
  expAvgTensor?: Tensor;
  expAvgSqTensor?: Tensor;
  deviceStep?: number;
  deviceMuProduct?: number;
};

/**
 * Nadam (Nesterov-accelerated Adam) optimizer (Dozat, 2016).
 *
 * Combines Adam's adaptive learning rates with Nesterov momentum: the first
 * moment estimate is replaced by a look-ahead blend of the next momentum
 * estimate and the current gradient. The momentum coefficient follows the
 * schedule `mu_t = beta1 * (1 - 0.5 * 0.96^(t * momentumDecay))` used by
 * `torch.optim.NAdam`.
 *
 * By default `weightDecay` is an L2 penalty added to the gradient. With
 * `decoupledWeightDecay: true` the parameters are instead multiplied by
 * `1 - lr * weightDecay` before the update (NAdamW).
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
  /**
   * Create a new Nadam optimizer.
   *
   * @param params - Parameters to optimize, or an array of parameter groups with per-group options
   * @param options - Hyperparameters
   * @param options.lr - Learning rate (default: 0.002)
   * @param options.beta1 - Decay rate of the first moment, in [0, 1) (default: 0.9)
   * @param options.beta2 - Decay rate of the second moment, in [0, 1) (default: 0.999)
   * @param options.eps - Term added to the denominator for numerical stability (default: 1e-8)
   * @param options.weightDecay - Weight decay coefficient (default: 0)
   * @param options.momentumDecay - Decay of the momentum schedule (default: 0.004)
   * @param options.decoupledWeightDecay - Apply weight decay directly to the parameters
   *   instead of the gradient (default: false)
   * @param options.maximize - Maximize the objective instead of minimizing it (default: false)
   * @throws {InvalidParameterError} If a hyperparameter is out of range
   */
  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<NadamOptions>>,
    options: {
      readonly lr?: number;
      readonly beta1?: number;
      readonly beta2?: number;
      readonly eps?: number;
      readonly weightDecay?: number;
      readonly momentumDecay?: number;
      readonly decoupledWeightDecay?: boolean;
      readonly maximize?: boolean;
    } = {}
  ) {
    const defaults = {
      lr: options.lr ?? 0.002,
      beta1: options.beta1 ?? 0.9,
      beta2: options.beta2 ?? 0.999,
      eps: options.eps ?? 1e-8,
      weightDecay: options.weightDecay ?? 0,
      momentumDecay: options.momentumDecay ?? 0.004,
      decoupledWeightDecay: options.decoupledWeightDecay ?? false,
      maximize: options.maximize ?? false,
    };

    super(params, defaults);
  }

  protected override validateOptions(options: Readonly<NadamOptions>): void {
    assertFiniteNonNegative("learning rate", options.lr);
    assertInRange("beta1", options.beta1, 0, 1);
    assertInRange("beta2", options.beta2, 0, 1);
    assertFinitePositive("epsilon", options.eps);
    assertFiniteNonNegative("weight_decay value", options.weightDecay);
    assertFiniteNonNegative("momentum_decay", options.momentumDecay);
  }

  protected isState(state: Record<string, unknown>): state is NadamState {
    return (
      typeof state["step"] === "number" &&
      state["expAvg"] instanceof Float64Array &&
      state["expAvgSq"] instanceof Float64Array &&
      typeof state["muProduct"] === "number"
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

    this.prepareStep("Nadam");
    this.countStep();

    for (const group of this.paramGroups) {
      const { lr, beta1, beta2, eps, weightDecay, momentumDecay, decoupledWeightDecay, maximize } =
        group.options;

      // Re-validate hyperparameters

      for (const param of this.trainableParams(group)) {
        // Device path: compose the Nadam update from device-dispatched ops.
        if (param.tensor.isDeviceTensor) {
          const rawGrad = param.grad;
          if (!rawGrad) continue;
          const g = maximize ? mulScalar(rawGrad, -1) : rawGrad;
          let dstate = this.state.get(param);
          if (!dstate) {
            dstate = {
              step: 0,
              expAvg: new Float64Array(0),
              expAvgSq: new Float64Array(0),
              muProduct: 1,
            };
            this.state.set(param, dstate);
          }
          const t = (dstate.deviceStep ?? 0) + 1;
          dstate.deviceStep = t;
          const biasCorrection2 = 1 - beta2 ** t;
          const mu = beta1 * (1 - 0.5 * 0.96 ** (t * momentumDecay));
          const muNext = beta1 * (1 - 0.5 * 0.96 ** ((t + 1) * momentumDecay));
          const muProduct = (dstate.deviceMuProduct ?? 1) * mu;
          const muProductNext = muProduct * muNext;
          dstate.deviceMuProduct = muProduct;
          const grad =
            weightDecay !== 0 && !decoupledWeightDecay
              ? add(g, mulScalar(param.tensor, weightDecay))
              : g;
          // Decoupled decay shrinks the parameter before the Adam-style update.
          const base =
            weightDecay !== 0 && decoupledWeightDecay
              ? mulScalar(param.tensor, 1 - lr * weightDecay)
              : param.tensor;
          const mPrev = dstate.expAvgTensor;
          const vPrev = dstate.expAvgSqTensor;
          const mNew = mPrev
            ? add(mulScalar(mPrev, beta1), mulScalar(grad, 1 - beta1))
            : mulScalar(grad, 1 - beta1);
          const vNew = vPrev
            ? add(mulScalar(vPrev, beta2), mulScalar(square(grad), 1 - beta2))
            : mulScalar(square(grad), 1 - beta2);
          dstate.expAvgTensor = mNew;
          dstate.expAvgSqTensor = vNew;
          const denom = addScalar(sqrt(mulScalar(vNew, 1 / biasCorrection2)), eps);
          const mHatNext = mulScalar(mNew, 1 / (1 - muProductNext));
          const gHat = mulScalar(grad, 1 / (1 - muProduct));
          const mNesterov = add(mulScalar(mHatNext, muNext), mulScalar(gHat, 1 - mu));
          replaceParamStorage(param, "tensor", sub(base, mulScalar(div(mNesterov, denom), lr)));
          continue;
        }

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

        const expAvg = state.expAvg;
        const expAvgSq = state.expAvgSq;
        const l2 = weightDecay !== 0 && !decoupledWeightDecay;
        const shrink = weightDecay !== 0 && decoupledWeightDecay ? 1 - lr * weightDecay : 1;

        for (let i = 0; i < size; i++) {
          const rawGi = gradData[gOff + i] as number;
          const p0 = pData[pOff + i] as number;
          if (!Number.isFinite(rawGi)) assertFinite("gradient", rawGi);
          const gi0 = maximize ? -rawGi : rawGi;
          if (!Number.isFinite(p0)) assertFinite("parameter", p0);

          // L2 weight decay goes into the gradient; decoupled decay shrinks the parameter
          const gi = l2 ? gi0 + weightDecay * p0 : gi0;
          const pi = p0 * shrink;

          // Update biased first moment estimate: m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
          const mNew = beta1 * (expAvg[i] as number) + (1 - beta1) * gi;
          expAvg[i] = mNew;

          // Update biased second moment estimate: v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
          const vNew = beta2 * (expAvgSq[i] as number) + (1 - beta2) * gi * gi;
          expAvgSq[i] = vNew;

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
