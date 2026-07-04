/**
 * @see {@link https://deepbox.dev/docs/optim-optimizers | Deepbox documentation}
 */

import { DeviceError, InvalidParameterError } from "../../core";
import type { GradTensor } from "../../ndarray";
import {
  assertFiniteNonNegative,
  assertFinitePositive,
  assertHasGradFloat,
  safeArrayAccess,
} from "../_internal";
import { Optimizer, type ParamGroup } from "../Optimizer";

type LBFGSOptions = {
  lr: number;
  maxIter: number;
  maxEval: number;
  toleranceGrad: number;
  toleranceChange: number;
  historySize: number;
  lineSearchFn: "strong_wolfe" | null;
};

type LBFGSState = {
  step: number;
};

/**
 * L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) optimizer.
 *
 * A quasi-Newton method that approximates the inverse Hessian using a limited
 * history of past updates. Unlike first-order methods (Adam, SGD), LBFGS can
 * converge much faster for smooth objectives.
 *
 * **Important**: LBFGS requires a closure that reevaluates the model and returns
 * the loss. This is because LBFGS may evaluate the function multiple times per step.
 *
 * @example
 * ```ts
 * import { LBFGS } from 'deepbox/optim';
 *
 * const optimizer = new LBFGS(model.parameters(), { lr: 1 });
 *
 * for (let epoch = 0; epoch < numEpochs; epoch++) {
 *   const closure = () => {
 *     optimizer.zeroGrad();
 *     const output = model.forward(input);
 *     const loss = criterion(output, target);
 *     loss.backward();
 *     return loss.item();
 *   };
 *   optimizer.step(closure);
 * }
 * ```
 *
 * @category Optimizers
 */
export class LBFGS extends Optimizer<LBFGSOptions, LBFGSState> {
  private _stepCount = 0;

  // L-BFGS history buffers (shared across all params, flattened)
  private _sHistory: Float64Array[] = [];
  private _yHistory: Float64Array[] = [];
  private _rhoHistory: number[] = [];
  private _prevFlatGrad: Float64Array | null = null;
  private _prevFlatParams: Float64Array | null = null;

  get stepCount(): number {
    return this._stepCount;
  }

  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<LBFGSOptions>>,
    options: {
      readonly lr?: number;
      readonly maxIter?: number;
      readonly maxEval?: number;
      readonly toleranceGrad?: number;
      readonly toleranceChange?: number;
      readonly historySize?: number;
      readonly lineSearchFn?: "strong_wolfe" | null;
    } = {}
  ) {
    const defaults: LBFGSOptions = {
      lr: options.lr ?? 1,
      maxIter: options.maxIter ?? 20,
      maxEval: options.maxEval ?? 25,
      toleranceGrad: options.toleranceGrad ?? 1e-7,
      toleranceChange: options.toleranceChange ?? 1e-9,
      historySize: options.historySize ?? 10,
      lineSearchFn: options.lineSearchFn ?? null,
    };

    super(params, defaults);

    assertFiniteNonNegative("learning rate", defaults.lr);
    assertFinitePositive("toleranceGrad", defaults.toleranceGrad);
    assertFinitePositive("toleranceChange", defaults.toleranceChange);

    if (defaults.maxIter < 1) {
      throw new InvalidParameterError("maxIter must be >= 1", "maxIter", defaults.maxIter);
    }
    if (defaults.historySize < 1) {
      throw new InvalidParameterError(
        "historySize must be >= 1",
        "historySize",
        defaults.historySize
      );
    }
    if (defaults.lineSearchFn !== null && defaults.lineSearchFn !== "strong_wolfe") {
      throw new InvalidParameterError(
        `lineSearchFn must be "strong_wolfe" or null; received ${String(defaults.lineSearchFn)}`,
        "lineSearchFn",
        defaults.lineSearchFn
      );
    }
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

  protected isState(state: Record<string, unknown>): state is LBFGSState {
    return typeof state["step"] === "number";
  }

  /** Gather all parameter values into a single flat array */
  private flattenParams(): Float64Array {
    const parts: {
      data: Float64Array | Float32Array;
      offset: number;
      size: number;
    }[] = [];
    let totalSize = 0;
    for (const group of this.paramGroups) {
      for (const param of group.params) {
        const { param: pData, paramOffset } = assertHasGradFloat(param, "LBFGS");
        const size = param.tensor.size;
        parts.push({ data: pData, offset: paramOffset, size });
        totalSize += size;
      }
    }
    const flat = new Float64Array(totalSize);
    let idx = 0;
    for (const p of parts) {
      for (let i = 0; i < p.size; i++) {
        flat[idx++] = safeArrayAccess(p.data, p.offset + i, "LBFGS param");
      }
    }
    return flat;
  }

  /** Gather all gradient values into a single flat array */
  private flattenGrads(): Float64Array {
    const parts: {
      data: Float64Array | Float32Array;
      offset: number;
      size: number;
    }[] = [];
    let totalSize = 0;
    for (const group of this.paramGroups) {
      for (const param of group.params) {
        const { grad, gradOffset } = assertHasGradFloat(param, "LBFGS");
        const size = param.tensor.size;
        parts.push({ data: grad, offset: gradOffset, size });
        totalSize += size;
      }
    }
    const flat = new Float64Array(totalSize);
    let idx = 0;
    for (const p of parts) {
      for (let i = 0; i < p.size; i++) {
        flat[idx++] = safeArrayAccess(p.data, p.offset + i, "LBFGS grad");
      }
    }
    return flat;
  }

  /** Set all parameters from a flat array */
  private unflattenParams(flat: Float64Array): void {
    let idx = 0;
    for (const group of this.paramGroups) {
      for (const param of group.params) {
        const { param: pData, paramOffset } = assertHasGradFloat(param, "LBFGS");
        const size = param.tensor.size;
        for (let i = 0; i < size; i++) {
          const fVal = flat[idx]!;
          idx++;
          pData[paramOffset + i] = fVal;
        }
      }
    }
  }

  /** Add flat direction * step to parameters */
  private addDirection(direction: Float64Array, stepSize: number): void {
    let idx = 0;
    for (const group of this.paramGroups) {
      for (const param of group.params) {
        const { param: pData, paramOffset } = assertHasGradFloat(param, "LBFGS");
        const size = param.tensor.size;
        for (let i = 0; i < size; i++) {
          const dVal = direction[idx]!;
          idx++;
          const cur = pData[paramOffset + i] ?? 0;
          pData[paramOffset + i] = cur + stepSize * dVal;
        }
      }
    }
  }

  /** Compute dot product of two flat arrays */
  private static dot(a: Float64Array, b: Float64Array): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += a[i]! * b[i]!;
    }
    return sum;
  }

  /** L-BFGS two-loop recursion to compute search direction */
  private computeDirection(grad: Float64Array): Float64Array {
    const n = grad.length;
    const q = new Float64Array(n);
    for (let i = 0; i < n; i++) q[i] = -grad[i]!;

    const m = this._sHistory.length;
    if (m === 0) return q;

    const alphas: number[] = new Array(m).fill(0);

    // Forward pass
    for (let i = m - 1; i >= 0; i--) {
      const s = this._sHistory[i]!;
      const rho = this._rhoHistory[i]!;
      alphas[i] = rho * LBFGS.dot(s, q);
      const y = this._yHistory[i]!;
      for (let j = 0; j < n; j++) {
        const ai = alphas[i] ?? 0;
        q[j] = (q[j] ?? 0) - ai * y[j]!;
      }
    }

    // Scale by initial Hessian approximation: H0 = (s^T y) / (y^T y)
    const sLast = this._sHistory[m - 1]!;
    const yLast = this._yHistory[m - 1]!;
    const ys = LBFGS.dot(yLast, sLast);
    const yy = LBFGS.dot(yLast, yLast);
    const gamma = yy > 0 ? ys / yy : 1;
    for (let j = 0; j < n; j++) {
      q[j] = (q[j] ?? 0) * gamma;
    }

    // Backward pass
    for (let i = 0; i < m; i++) {
      const y = this._yHistory[i]!;
      const rho = this._rhoHistory[i]!;
      const beta = rho * LBFGS.dot(y, q);
      const s = this._sHistory[i]!;
      for (let j = 0; j < n; j++) {
        q[j] = (q[j] ?? 0) + ((alphas[i] ?? 0) - beta) * s[j]!;
      }
    }

    return q;
  }

  step(closure?: () => number): number | undefined {
    if (!closure) {
      throw new InvalidParameterError(
        "LBFGS requires a closure that reevaluates the model and returns the loss",
        "closure",
        undefined
      );
    }

    // LBFGS flattens parameters/gradients into a single host vector and runs a
    // line search that reads scalar loss/curvature values back — none of which
    // is possible on opaque device memory. Reject device parameters clearly.
    for (const group of this.paramGroups) {
      for (const param of group.params) {
        if (param.tensor.isDeviceTensor) {
          throw new DeviceError(
            "LBFGS is not supported on device tensors (it needs host-readable parameters and scalar line-search values). Move the parameters to CPU with `.to('cpu')` before optimizing."
          );
        }
      }
    }

    const options = this.paramGroups[0]?.options ?? this.defaults;
    const { lr, maxIter, toleranceGrad, toleranceChange, historySize } = options;
    const maxEval = options.maxEval;

    this._stepCount++;
    let nFuncEval = 0;

    // Evaluate initial loss and gradient
    let loss = closure();
    nFuncEval++;
    let flatGrad = this.flattenGrads();

    // Check convergence on gradient
    let gradMaxAbs = 0;
    for (let i = 0; i < flatGrad.length; i++) {
      const absG = Math.abs(flatGrad[i]!);
      if (absG > gradMaxAbs) gradMaxAbs = absG;
    }
    if (gradMaxAbs <= toleranceGrad) return loss;

    // Update history from previous step
    if (this._prevFlatGrad !== null && this._prevFlatParams !== null) {
      const currentParams = this.flattenParams();
      const n = flatGrad.length;
      const s = new Float64Array(n);
      const y = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        s[i] = currentParams[i]! - this._prevFlatParams[i]!;
        y[i] = flatGrad[i]! - this._prevFlatGrad[i]!;
      }
      const ys = LBFGS.dot(y, s);
      if (ys > 1e-10) {
        if (this._sHistory.length >= historySize) {
          this._sHistory.shift();
          this._yHistory.shift();
          this._rhoHistory.shift();
        }
        this._sHistory.push(s);
        this._yHistory.push(y);
        this._rhoHistory.push(1 / ys);
      }
    }

    // L-BFGS iteration
    for (let iter = 0; iter < maxIter; iter++) {
      // Compute search direction
      const direction = this.computeDirection(flatGrad);

      // Check if direction is a descent direction
      const dirDeriv = LBFGS.dot(flatGrad, direction);
      if (dirDeriv > 0) {
        // Not a descent direction, reset to steepest descent
        for (let i = 0; i < direction.length; i++) {
          direction[i] = -flatGrad[i]!;
        }
        this._sHistory.length = 0;
        this._yHistory.length = 0;
        this._rhoHistory.length = 0;
      }

      // Simple backtracking line search
      const prevParams = this.flattenParams();
      const prevGrad = new Float64Array(flatGrad);
      const prevLoss = loss;
      let stepSize = lr;
      let foundBetter = false;

      for (let ls = 0; ls < 10 && nFuncEval < maxEval; ls++) {
        // Restore params and apply direction * stepSize
        this.unflattenParams(prevParams);
        this.addDirection(direction, stepSize);

        loss = closure();
        nFuncEval++;

        if (loss < prevLoss) {
          foundBetter = true;
          break;
        }
        stepSize *= 0.5;
      }

      if (!foundBetter) {
        // Restore previous params
        this.unflattenParams(prevParams);
        loss = prevLoss;
        break;
      }

      flatGrad = this.flattenGrads();

      // Check gradient convergence
      gradMaxAbs = 0;
      for (let i = 0; i < flatGrad.length; i++) {
        const absG = Math.abs(flatGrad[i]!);
        if (absG > gradMaxAbs) gradMaxAbs = absG;
      }
      if (gradMaxAbs <= toleranceGrad) break;

      // Check function value convergence
      if (Math.abs(loss - prevLoss) < toleranceChange) break;

      // Update L-BFGS history
      const currentParams = this.flattenParams();
      const n = flatGrad.length;
      const s = new Float64Array(n);
      const y = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        s[i] = currentParams[i]! - prevParams[i]!;
        y[i] = flatGrad[i]! - prevGrad[i]!;
      }
      const ys = LBFGS.dot(y, s);
      if (ys > 1e-10) {
        if (this._sHistory.length >= historySize) {
          this._sHistory.shift();
          this._yHistory.shift();
          this._rhoHistory.shift();
        }
        this._sHistory.push(s);
        this._yHistory.push(y);
        this._rhoHistory.push(1 / ys);
      }

      if (nFuncEval >= maxEval) break;
    }

    // Save state for next step
    this._prevFlatGrad = this.flattenGrads();
    this._prevFlatParams = this.flattenParams();

    return loss;
  }
}
