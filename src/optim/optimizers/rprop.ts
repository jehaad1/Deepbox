/**
 * @see {@link https://deepbox.dev/docs/optim-optimizers | Deepbox documentation}
 */

import { InvalidParameterError } from "../../core";
import {
  addScalar,
  type GradTensor,
  mul,
  mulScalar,
  neg,
  relu,
  sub,
  type Tensor,
  tensor,
  where,
} from "../../ndarray";
import {
  assertBufferSize,
  assertFinite,
  assertFiniteNonNegative,
  assertHasGradFloat,
  deviceMaxScalar,
  deviceMinScalar,
  deviceSign,
  replaceParamStorage,
  safeArrayAccess,
} from "../_internal";
import { Optimizer, type ParamGroup } from "../Optimizer";

/**
 * Options for the Rprop optimizer.
 *
 * @property lr - Initial step size of every parameter element
 * @property etaMinus - Factor applied to the step size when the gradient changes sign, in (0, 1)
 * @property etaPlus - Factor applied to the step size when the gradient keeps its sign, > 1
 * @property stepMin - Lower bound of the step size
 * @property stepMax - Upper bound of the step size
 * @property maximize - Maximize the objective instead of minimizing it
 */
type RpropOptions = {
  lr: number;
  etaMinus: number;
  etaPlus: number;
  stepMin: number;
  stepMax: number;
  maximize: boolean;
};

function validateRpropOptions(options: Readonly<RpropOptions>): void {
  assertFiniteNonNegative("learning rate", options.lr);
  if (!Number.isFinite(options.etaMinus) || options.etaMinus <= 0 || options.etaMinus >= 1) {
    throw new InvalidParameterError(
      `Invalid etaMinus: ${options.etaMinus} (must be in (0, 1))`,
      "etaMinus",
      options.etaMinus
    );
  }
  if (!Number.isFinite(options.etaPlus) || options.etaPlus <= 1) {
    throw new InvalidParameterError(
      `Invalid etaPlus: ${options.etaPlus} (must be finite and > 1)`,
      "etaPlus",
      options.etaPlus
    );
  }
  assertFiniteNonNegative("stepMin", options.stepMin);
  assertFiniteNonNegative("stepMax", options.stepMax);
  if (options.stepMin > options.stepMax) {
    throw new InvalidParameterError(
      `Invalid stepMin: ${options.stepMin} (must be <= stepMax = ${options.stepMax})`,
      "stepMin",
      options.stepMin
    );
  }
}

type RpropState = {
  prevGrad?: Float64Array;
  stepSizes?: Float64Array;
  /** Device state (used when the parameter lives on a kernel device). */
  prevGradTensor?: Tensor;
  stepSizesTensor?: Tensor;
};

/**
 * Rprop (Resilient Backpropagation) optimizer.
 *
 * Only uses the sign of the gradient, not its magnitude.
 * Step sizes adapt individually per parameter based on sign changes. The
 * update follows PyTorch's `torch.optim.Rprop` (Rprop- with weight reset on a
 * sign change): for every element, with `prev` the previous gradient,
 *
 * ```
 * g * prev > 0:  step = min(step * etaPlus,  stepMax);  param -= sign(g) * step
 * g * prev < 0:  step = max(step * etaMinus, stepMin);  param unchanged, prev = 0
 * otherwise:     param -= sign(g) * step
 * ```
 *
 * The step sizes start at `lr` (clamped to `[stepMin, stepMax]`). Because the
 * learning rate only seeds the step sizes, changing it later (for example with
 * an LR scheduler) does not affect parameters that already have state.
 *
 * @example
 * ```ts
 * import { Rprop } from 'deepbox/optim';
 *
 * const optimizer = new Rprop(model.parameters(), {
 *   lr: 0.01,
 *   etaMinus: 0.5,
 *   etaPlus: 1.2,
 * });
 * ```
 *
 * @category Optimizers
 */
export class Rprop extends Optimizer<RpropOptions, RpropState> {
  /**
   * Create a new Rprop optimizer.
   *
   * @param params - Iterable of parameters or parameter groups to optimize
   * @param options - Optimization options
   * @param options.lr - Initial step size (default: 0.01)
   * @param options.etaMinus - Step size decrease factor, in (0, 1) (default: 0.5)
   * @param options.etaPlus - Step size increase factor, > 1 (default: 1.2)
   * @param options.stepMin - Smallest step size (default: 1e-6)
   * @param options.stepMax - Largest step size (default: 50)
   * @param options.maximize - Maximize the objective instead of minimizing it (default: false)
   * @throws {InvalidParameterError} If a hyperparameter is out of range, or
   *   `stepMin` exceeds `stepMax`
   */
  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<RpropOptions>>,
    options: {
      readonly lr?: number;
      readonly etaMinus?: number;
      readonly etaPlus?: number;
      readonly stepMin?: number;
      readonly stepMax?: number;
      readonly maximize?: boolean;
    } = {}
  ) {
    const defaults: RpropOptions = {
      lr: options.lr ?? 0.01,
      etaMinus: options.etaMinus ?? 0.5,
      etaPlus: options.etaPlus ?? 1.2,
      stepMin: options.stepMin ?? 1e-6,
      stepMax: options.stepMax ?? 50,
      maximize: options.maximize ?? false,
    };

    super(params, defaults);
  }

  protected override validateOptions(options: Readonly<RpropOptions>): void {
    validateRpropOptions(options);
  }

  protected isState(state: Record<string, unknown>): state is RpropState {
    if (state["prevGrad"] !== undefined && !(state["prevGrad"] instanceof Float64Array)) {
      return false;
    }
    if (state["stepSizes"] !== undefined && !(state["stepSizes"] instanceof Float64Array)) {
      return false;
    }
    return true;
  }

  /**
   * Perform a single optimization step.
   *
   * A parameter whose gradient is `null` (it took no part in the loss) is skipped, as in PyTorch.
   *
   * @param closure - Optional closure that reevaluates the model and returns the loss
   * @returns Loss value if a closure is provided
   * @throws {InvalidParameterError} If a group option is invalid or a gradient or
   *   parameter value is not finite
   */
  step(closure?: () => number): number | undefined {
    let loss: number | undefined;
    if (closure) {
      loss = closure();
    }

    this.prepareStep("Rprop");
    this.countStep();

    for (const group of this.paramGroups) {
      const { lr, etaMinus, etaPlus, stepMin, stepMax, maximize } = group.options;
      const initialStep = Math.min(Math.max(lr, stepMin), stepMax);

      for (const param of this.trainableParams(group)) {
        // Device path: compose the sign-based Rprop update from device-dispatched
        // ops. The three-way per-element branch (grad sign product >0 / <0 / ==0)
        // is expressed with masks + where, matching the host loop exactly.
        if (param.tensor.isDeviceTensor) {
          const g = param.grad;
          if (!g) continue;
          let dstate = this.state.get(param);
          if (!dstate) {
            dstate = {};
            this.state.set(param, dstate);
          }
          const one = tensor(1);
          const zero = tensor(0);
          // prevGrad starts at 0; stepSizes start at lr clamped to [stepMin, stepMax].
          const gSigned = maximize ? mulScalar(g, -1) : g;
          const prevT = dstate.prevGradTensor ?? mulScalar(g, 0);
          const stepT = dstate.stepSizesTensor ?? addScalar(mulScalar(g, 0), initialStep);
          const product = mul(gSigned, prevT);
          const posMask = where(relu(product), one, zero); // 1 where product > 0
          const negMask = where(relu(neg(product)), one, zero); // 1 where product < 0
          const stepPos = deviceMinScalar(mulScalar(stepT, etaPlus), stepMax);
          const stepNeg = deviceMaxScalar(mulScalar(stepT, etaMinus), stepMin);
          // product>0 -> stepPos, product<0 -> stepNeg, product==0 -> unchanged.
          const newStep = where(posMask, stepPos, where(negMask, stepNeg, stepT));
          const delta = mul(deviceSign(gSigned), newStep);
          const updated = sub(param.tensor, delta);
          // On product<0 the parameter is unchanged and prevGrad is reset to 0.
          dstate.stepSizesTensor = newStep;
          dstate.prevGradTensor = where(negMask, zero, gSigned);
          replaceParamStorage(param, "tensor", where(negMask, param.tensor, updated));
          continue;
        }

        const {
          grad: gradData,
          gradOffset,
          param: paramData,
          paramOffset,
        } = assertHasGradFloat(param, "Rprop");
        const size = param.tensor.size;

        let state = this.state.get(param);
        if (!state) {
          state = {};
          this.state.set(param, state);
        }

        if (!state.prevGrad) {
          state.prevGrad = new Float64Array(size);
        }
        if (!state.stepSizes) {
          state.stepSizes = new Float64Array(size);
          state.stepSizes.fill(initialStep);
        }

        const prevGrad = state.prevGrad;
        const stepSizes = state.stepSizes;
        assertBufferSize(prevGrad, size, "Rprop prevGrad");
        assertBufferSize(stepSizes, size, "Rprop stepSizes");

        for (let i = 0; i < size; i++) {
          const giRaw = safeArrayAccess(gradData, gradOffset + i, "Rprop gradient");
          const pi = safeArrayAccess(paramData, paramOffset + i, "Rprop parameter");
          assertFinite("gradient", giRaw);
          assertFinite("parameter", pi);
          const gi = maximize ? -giRaw : giRaw;

          const prev = safeArrayAccess(prevGrad, i, "Rprop prevGrad");
          const product = gi * prev;

          let step = safeArrayAccess(stepSizes, i, "Rprop stepSize");

          if (product > 0) {
            step = Math.min(Math.max(step * etaPlus, stepMin), stepMax);
            stepSizes[i] = step;
            paramData[paramOffset + i] = pi - Math.sign(gi) * step;
          } else if (product < 0) {
            step = Math.max(step * etaMinus, stepMin);
            stepSizes[i] = step;
            // Revert gradient to prevent double punishment
            prevGrad[i] = 0;
            continue;
          } else {
            // The step size is clamped in every branch, as PyTorch does.
            step = Math.min(Math.max(step, stepMin), stepMax);
            stepSizes[i] = step;
            paramData[paramOffset + i] = pi - Math.sign(gi) * step;
          }

          prevGrad[i] = gi;
        }
      }
    }

    return loss;
  }
}
