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
  replaceParamStorage,
} from "../_internal";
import { Optimizer, type ParamGroup } from "../Optimizer";

type AdagradOptions = {
  lr: number;
  eps: number;
  weightDecay: number;
  lrDecay: number;
  initialAccumulatorValue: number;
  maximize: boolean;
};

type AdagradState = {
  step: number;
  sum: Float64Array;
  /** Device state (used when the parameter lives on a kernel device). */
  sumTensor?: Tensor;
  deviceStep?: number;
};

/**
 * Adagrad (Adaptive Gradient Algorithm) optimizer.
 *
 * Adagrad adapts the learning rate for each parameter based on the historical
 * sum of squared gradients. Parameters with larger gradients receive smaller
 * effective learning rates, while parameters with smaller gradients receive
 * larger effective learning rates. The update follows `torch.optim.Adagrad`:
 *
 * ```
 * clr = lr / (1 + (t - 1) * lrDecay)
 * sum += g^2
 * theta -= clr * g / (sqrt(sum) + eps)
 * ```
 *
 * When `weightDecay` is non-zero, `weightDecay * theta` is added to the gradient
 * (L2 penalty) before the update.
 *
 * @example
 * ```ts
 * import { Adagrad } from 'deepbox/optim';
 *
 * const optimizer = new Adagrad(model.parameters(), {
 *   lr: 0.01,
 *   eps: 1e-10
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
export class Adagrad extends Optimizer<AdagradOptions, AdagradState> {
  /**
   * Create a new Adagrad optimizer.
   *
   * @param params - Parameters to optimize, or an array of parameter groups with per-group options
   * @param options - Hyperparameters
   * @param options.lr - Learning rate (default: 0.01)
   * @param options.eps - Term added to the denominator for numerical stability (default: 1e-10)
   * @param options.weightDecay - L2 penalty coefficient (default: 0)
   * @param options.lrDecay - Learning rate decay applied per step (default: 0)
   * @param options.initialAccumulatorValue - Starting value of the squared-gradient sum (default: 0)
   * @param options.maximize - Maximize the objective instead of minimizing it (default: false)
   * @throws {InvalidParameterError} If a hyperparameter is out of range
   */
  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<AdagradOptions>>,
    options: {
      readonly lr?: number;
      readonly eps?: number;
      readonly weightDecay?: number;
      readonly lrDecay?: number;
      readonly initialAccumulatorValue?: number;
      readonly maximize?: boolean;
    } = {}
  ) {
    const defaults = {
      lr: options.lr ?? 0.01,
      eps: options.eps ?? 1e-10,
      weightDecay: options.weightDecay ?? 0,
      lrDecay: options.lrDecay ?? 0,
      initialAccumulatorValue: options.initialAccumulatorValue ?? 0,
      maximize: options.maximize ?? false,
    };

    super(params, defaults);
  }

  protected override validateOptions(options: Readonly<AdagradOptions>): void {
    assertFiniteNonNegative("learning rate", options.lr);
    assertFinitePositive("epsilon", options.eps);
    assertFiniteNonNegative("weight_decay value", options.weightDecay);
    assertFiniteNonNegative("lr_decay", options.lrDecay);
    assertFiniteNonNegative("initial_accumulator_value", options.initialAccumulatorValue);
  }

  protected isState(state: Record<string, unknown>): state is AdagradState {
    return typeof state["step"] === "number" && state["sum"] instanceof Float64Array;
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

    this.prepareStep("Adagrad");
    this.countStep();

    for (const group of this.paramGroups) {
      const { lr, eps, weightDecay, lrDecay, initialAccumulatorValue, maximize } = group.options;

      for (const param of this.trainableParams(group)) {
        // Device path: compose the Adagrad update from device-dispatched ops.
        if (param.tensor.isDeviceTensor) {
          const rawGrad = param.grad;
          if (!rawGrad) continue;
          const g = maximize ? mulScalar(rawGrad, -1) : rawGrad;
          let dstate = this.state.get(param);
          if (!dstate) {
            dstate = { step: 0, sum: new Float64Array(0) };
            this.state.set(param, dstate);
          }
          const t = (dstate.deviceStep ?? 0) + 1;
          dstate.deviceStep = t;
          const clr = lr / (1 + (t - 1) * lrDecay);
          const grad = weightDecay !== 0 ? add(g, mulScalar(param.tensor, weightDecay)) : g;
          const sumPrev = dstate.sumTensor;
          const sumNew = sumPrev
            ? add(sumPrev, square(grad))
            : addScalar(square(grad), initialAccumulatorValue);
          dstate.sumTensor = sumNew;
          const std = addScalar(sqrt(sumNew), eps);
          replaceParamStorage(param, "tensor", sub(param.tensor, mulScalar(div(grad, std), clr)));
          continue;
        }

        const {
          grad: gradData,
          gradOffset: gOff,
          param: pData,
          paramOffset: pOff,
        } = assertHasGradFloat(param, "Adagrad");
        const size = param.tensor.size;

        const existing = this.state.get(param);
        const state =
          existing ??
          (() => {
            const next = {
              step: 0,
              sum: new Float64Array(size).fill(initialAccumulatorValue),
            };
            this.state.set(param, next);
            return next;
          })();

        // Validate state buffer size
        assertBufferSize(state.sum, size, "Adagrad sum");

        state.step += 1;

        const clr = lr / (1 + (state.step - 1) * lrDecay);
        const sum = state.sum;

        for (let i = 0; i < size; i++) {
          const rawGi = gradData[gOff + i] as number;
          const pi = pData[pOff + i] as number;
          if (!Number.isFinite(rawGi)) assertFinite("gradient", rawGi);
          const gi0 = maximize ? -rawGi : rawGi;
          if (!Number.isFinite(pi)) assertFinite("parameter", pi);

          // L2 weight decay
          const gi = weightDecay !== 0 ? gi0 + weightDecay * pi : gi0;

          const sumNew = (sum[i] as number) + gi * gi;
          sum[i] = sumNew;

          const std = Math.sqrt(sumNew) + eps;
          pData[pOff + i] = pi - clr * (gi / std);
        }
      }
    }

    return loss;
  }
}
