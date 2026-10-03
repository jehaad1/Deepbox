/**
 * @see {@link https://deepbox.dev/docs/optim-optimizers | Deepbox documentation}
 */

import { InvalidParameterError } from "../../core";
import { add, type GradTensor, mulScalar, sub, type Tensor } from "../../ndarray";
import {
  assertBufferSize,
  assertFinite,
  assertFiniteNonNegative,
  assertHasGradFloat,
  replaceParamStorage,
  safeArrayAccess,
} from "../_internal";
import { Optimizer, type ParamGroup } from "../Optimizer";

/**
 * Options for the SGD optimizer.
 *
 * @property lr - Learning rate
 * @property momentum - Momentum factor (0 disables momentum)
 * @property dampening - Dampening applied to the gradient in the momentum buffer
 * @property weightDecay - Weight decay coefficient (L2 penalty)
 * @property nesterov - Use Nesterov momentum
 * @property maximize - Maximize the objective instead of minimizing it
 */
type SGDOptions = {
  lr: number;
  momentum: number;
  dampening: number;
  weightDecay: number;
  nesterov: boolean;
  maximize: boolean;
};

type SGDState = {
  momentumBuffer?: Float64Array;
  /** Device momentum buffer (used when the parameter lives on a kernel device). */
  momentumTensor?: Tensor;
};

/**
 * Stochastic Gradient Descent (SGD) optimizer.
 *
 * Implements vanilla SGD with optional momentum, weight decay, and Nesterov acceleration.
 * The update follows PyTorch's `torch.optim.SGD`:
 *
 * ```
 * d = grad + weightDecay * param
 * buf = momentum * buf + (1 - dampening) * d      (buf = d on the first step)
 * d = nesterov ? d + momentum * buf : buf
 * param = param - lr * d
 * ```
 *
 * With `maximize: true` the gradient sign is flipped and the objective is maximized.
 *
 * @example
 * ```ts
 * import { SGD } from 'deepbox/optim';
 * import { Module } from 'deepbox/nn';
 *
 * const model: Module = ...;
 * const optimizer = new SGD(model.parameters(), {
 *   lr: 0.01,
 *   momentum: 0.9,
 *   weightDecay: 5e-4,
 *   nesterov: true
 * });
 *
 * // Training loop
 * for (let epoch = 0; epoch < numEpochs; epoch++) {
 *   for (const [inputs, targets] of dataLoader) {
 *     optimizer.zeroGrad();
 *     const outputs = model.forward(inputs);
 *     const loss = criterion(outputs, targets);
 *     loss.backward();
 *     optimizer.step();
 *     console.log(`epoch ${epoch}: loss ${loss.item()}`);
 *   }
 * }
 * ```
 *
 * @category Optimizers
 */
export class SGD extends Optimizer<SGDOptions, SGDState> {
  /**
   * Create a new SGD optimizer.
   *
   * @param params - Iterable of parameters or parameter groups to optimize
   * @param options - Optimization options
   * @param options.lr - Learning rate (default: 0.01)
   * @param options.momentum - Momentum factor (default: 0)
   * @param options.dampening - Dampening for momentum (default: 0)
   * @param options.weightDecay - Weight decay (L2 penalty) (default: 0)
   * @param options.nesterov - Enable Nesterov momentum (default: false)
   * @param options.maximize - Maximize the objective instead of minimizing it (default: false)
   * @throws {InvalidParameterError} If a hyperparameter is invalid, or `nesterov` is set
   *   without a positive momentum and zero dampening
   */
  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<SGDOptions>>,
    options: {
      readonly lr?: number;
      readonly momentum?: number;
      readonly dampening?: number;
      readonly weightDecay?: number;
      readonly nesterov?: boolean;
      readonly maximize?: boolean;
    } = {}
  ) {
    const defaults = {
      lr: options.lr ?? 0.01,
      momentum: options.momentum ?? 0,
      dampening: options.dampening ?? 0,
      weightDecay: options.weightDecay ?? 0,
      nesterov: options.nesterov ?? false,
      maximize: options.maximize ?? false,
    };

    super(params, defaults);
  }

  protected override validateOptions(options: Readonly<SGDOptions>): void {
    assertFiniteNonNegative("learning rate", options.lr);
    assertFiniteNonNegative("momentum value", options.momentum);
    assertFiniteNonNegative("dampening", options.dampening);
    assertFiniteNonNegative("weight_decay value", options.weightDecay);
    if (options.nesterov && (options.momentum <= 0 || options.dampening !== 0)) {
      throw new InvalidParameterError(
        "Nesterov momentum requires a momentum and zero dampening",
        "nesterov",
        {
          momentum: options.momentum,
          dampening: options.dampening,
          nesterov: options.nesterov,
        }
      );
    }
  }

  protected isState(state: Record<string, unknown>): state is SGDState {
    if (
      state["momentumBuffer"] !== undefined &&
      !(state["momentumBuffer"] instanceof Float64Array)
    ) {
      return false;
    }
    return true;
  }

  /**
   * Perform a single optimization step.
   *
   * Applies the update rule from the class description to every parameter. A
   * parameter whose gradient is `null` (it took no part in the loss) is skipped, as in PyTorch.
   *
   * @param closure - Optional closure that reevaluates the model and returns the loss
   * @returns Loss value if closure is provided
   * @throws {InvalidParameterError} If a group option is invalid or a gradient or
   *   parameter value is not finite
   */
  step(closure?: () => number): number | undefined {
    let loss: number | undefined;

    // Evaluate loss if closure provided
    if (closure) {
      loss = closure();
    }

    this.prepareStep("SGD");
    this.countStep();

    // Update each parameter group
    for (const group of this.paramGroups) {
      const { lr, momentum, dampening, weightDecay, nesterov, maximize } = group.options;

      for (const param of this.trainableParams(group)) {
        let state = this.state.get(param);
        if (!state) {
          state = {};
          this.state.set(param, state);
        }

        // Device path: keep the whole update resident on the accelerator by
        // composing it from device-dispatched tensor ops (no host readback).
        if (param.tensor.isDeviceTensor) {
          const g = param.grad;
          if (!g) continue;
          const signed: Tensor = maximize ? mulScalar(g, -1) : g;
          let d: Tensor =
            weightDecay !== 0 ? add(signed, mulScalar(param.tensor, weightDecay)) : signed;
          if (momentum !== 0) {
            const prev = state.momentumTensor;
            // First step: buf = d_p (no dampening). Later: momentum*buf + (1-dampening)*d_p.
            const buf = prev ? add(mulScalar(prev, momentum), mulScalar(d, 1 - dampening)) : d;
            state.momentumTensor = buf;
            d = nesterov ? add(d, mulScalar(buf, momentum)) : buf;
          }
          replaceParamStorage(param, "tensor", sub(param.tensor, mulScalar(d, lr)));
          continue;
        }

        const {
          grad: gradData,
          gradOffset,
          param: paramData,
          paramOffset,
        } = assertHasGradFloat(param, "SGD");
        const size = param.tensor.size;

        // Momentum buffer is stored densely (one value per element). On the
        // first step PyTorch initializes the buffer to a clone of d_p WITHOUT
        // applying dampening; dampening only affects subsequent updates.
        let momentumBuffer: Float64Array | undefined;
        let bufferInitialized = true;
        if (momentum !== 0) {
          if (!state.momentumBuffer) {
            state.momentumBuffer = new Float64Array(size);
            bufferInitialized = false;
          }
          momentumBuffer = state.momentumBuffer;
          assertBufferSize(momentumBuffer, size, "SGD momentumBuffer");
        }

        for (let i = 0; i < size; i++) {
          const gi = safeArrayAccess(gradData, gradOffset + i, "SGD gradient");
          const pi = safeArrayAccess(paramData, paramOffset + i, "SGD parameter");
          assertFinite("gradient", gi);
          assertFinite("parameter", pi);

          // d_p = grad + weightDecay * param
          let d = maximize ? -gi : gi;
          if (weightDecay !== 0) {
            d = d + weightDecay * pi;
          }

          if (momentumBuffer) {
            // First step: buf = d_p (no dampening). Later: buf = momentum*buf + (1-dampening)*d_p.
            const bNew = bufferInitialized
              ? momentum * safeArrayAccess(momentumBuffer, i, "SGD momentum buffer") +
                (1 - dampening) * d
              : d;
            momentumBuffer[i] = bNew;
            d = nesterov ? d + momentum * bNew : bNew;
          }

          // param -= lr * d
          paramData[paramOffset + i] = pi - lr * d;
        }
      }
    }

    return loss;
  }
}
