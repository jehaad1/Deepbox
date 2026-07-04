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

type RpropOptions = {
  lr: number;
  etaMinus: number;
  etaPlus: number;
  stepMin: number;
  stepMax: number;
};

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
 * Step sizes adapt individually per parameter based on sign changes.
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
  private _stepCount = 0;

  get stepCount(): number {
    return this._stepCount;
  }

  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<RpropOptions>>,
    options: {
      readonly lr?: number;
      readonly etaMinus?: number;
      readonly etaPlus?: number;
      readonly stepMin?: number;
      readonly stepMax?: number;
    } = {}
  ) {
    const defaults: RpropOptions = {
      lr: options.lr ?? 0.01,
      etaMinus: options.etaMinus ?? 0.5,
      etaPlus: options.etaPlus ?? 1.2,
      stepMin: options.stepMin ?? 1e-6,
      stepMax: options.stepMax ?? 50,
    };

    super(params, defaults);

    assertFiniteNonNegative("learning rate", defaults.lr);
    if (defaults.etaMinus <= 0 || defaults.etaMinus >= 1) {
      throw new InvalidParameterError("etaMinus must be in (0, 1)", "etaMinus", defaults.etaMinus);
    }
    if (defaults.etaPlus <= 1) {
      throw new InvalidParameterError("etaPlus must be > 1", "etaPlus", defaults.etaPlus);
    }
    assertFiniteNonNegative("stepMin", defaults.stepMin);
    assertFiniteNonNegative("stepMax", defaults.stepMax);
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

  step(closure?: () => number): number | undefined {
    let loss: number | undefined;
    if (closure) {
      loss = closure();
    }

    this._stepCount++;

    for (const group of this.paramGroups) {
      const { lr, etaMinus, etaPlus, stepMin, stepMax } = group.options;

      for (const param of group.params) {
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
          // prevGrad starts at 0; stepSizes start at lr.
          const prevT = dstate.prevGradTensor ?? mulScalar(g, 0);
          const stepT = dstate.stepSizesTensor ?? addScalar(mulScalar(g, 0), lr);
          const product = mul(g, prevT);
          const posMask = where(relu(product), one, zero); // 1 where product > 0
          const negMask = where(relu(neg(product)), one, zero); // 1 where product < 0
          const stepPos = deviceMinScalar(mulScalar(stepT, etaPlus), stepMax);
          const stepNeg = deviceMaxScalar(mulScalar(stepT, etaMinus), stepMin);
          // product>0 -> stepPos, product<0 -> stepNeg, product==0 -> unchanged.
          const newStep = where(posMask, stepPos, where(negMask, stepNeg, stepT));
          const delta = mul(deviceSign(g), newStep);
          const updated = sub(param.tensor, delta);
          // On product<0 the parameter is unchanged and prevGrad is reset to 0.
          dstate.stepSizesTensor = newStep;
          dstate.prevGradTensor = where(negMask, zero, g);
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
          state.stepSizes.fill(lr);
        }

        const prevGrad = state.prevGrad;
        const stepSizes = state.stepSizes;

        for (let i = 0; i < size; i++) {
          const gi = safeArrayAccess(gradData, gradOffset + i, "Rprop gradient");
          const pi = safeArrayAccess(paramData, paramOffset + i, "Rprop parameter");
          assertFinite("gradient", gi);
          assertFinite("parameter", pi);

          const prev = safeArrayAccess(prevGrad, i, "Rprop prevGrad");
          const product = gi * prev;

          let step = safeArrayAccess(stepSizes, i, "Rprop stepSize");

          if (product > 0) {
            step = Math.min(step * etaPlus, stepMax);
            stepSizes[i] = step;
            paramData[paramOffset + i] = pi - Math.sign(gi) * step;
          } else if (product < 0) {
            step = Math.max(step * etaMinus, stepMin);
            stepSizes[i] = step;
            // Revert gradient to prevent double punishment
            prevGrad[i] = 0;
            continue;
          } else {
            paramData[paramOffset + i] = pi - Math.sign(gi) * step;
          }

          prevGrad[i] = gi;
        }
      }
    }

    return loss;
  }

  getLearningRate(groupIdx = 0): number {
    const group = this.paramGroups[groupIdx];
    if (!group) {
      throw new InvalidParameterError(`Invalid group index: ${groupIdx}`, "groupIdx", groupIdx);
    }
    return group.options.lr;
  }

  setLearningRate(lr: number): void {
    assertFiniteNonNegative("learning rate", lr);
    for (const group of this.paramGroups) {
      group.options.lr = lr;
    }
  }
}
