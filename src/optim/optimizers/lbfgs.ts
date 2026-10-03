/**
 * @see {@link https://deepbox.dev/docs/optim-optimizers | Deepbox documentation}
 */

import { DataValidationError, DeviceError, InvalidParameterError } from "../../core";
import type { GradTensor } from "../../ndarray";
import {
  assertFinite,
  assertFiniteNonNegative,
  assertFinitePositive,
  assertFloatParam,
  assertHasGradFloat,
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

/** Result of evaluating the objective at a trial point of a line search. */
type LineSearchPoint = {
  /** Loss at the trial point (Infinity when the loss or gradient is not finite). */
  f: number;
  /** Flat gradient at the trial point. */
  g: Float64Array;
};

/** Sufficient decrease (Armijo) constant of the strong Wolfe conditions. */
const WOLFE_C1 = 1e-4;
/** Curvature constant of the strong Wolfe conditions. */
const WOLFE_C2 = 0.9;
/** Maximum number of objective evaluations in one strong Wolfe line search. */
const WOLFE_MAX_LS = 25;
/** Maximum number of step halvings in the backtracking line search. */
const BACKTRACK_MAX_HALVINGS = 10;
/** Curvature pairs with `s^T y` at or below this value are discarded. */
const CURVATURE_EPS = 1e-10;

function dot(a: Float64Array, b: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += (a[i] as number) * (b[i] as number);
  }
  return sum;
}

/** True for an integer >= 1 or Infinity. */
function isCount(value: number): boolean {
  return value === Number.POSITIVE_INFINITY || (Number.isInteger(value) && value >= 1);
}

function maxAbs(a: Float64Array): number {
  let max = 0;
  for (let i = 0; i < a.length; i++) {
    const v = Math.abs(a[i] as number);
    if (v > max) max = v;
  }
  return max;
}

function sumAbs(a: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.abs(a[i] as number);
  }
  return sum;
}

function isFiniteArray(a: Float64Array): boolean {
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(a[i])) return false;
  }
  return true;
}

/**
 * Minimizer of the cubic that interpolates `(x1, f1, g1)` and `(x2, f2, g2)`,
 * clamped to `bounds` (default: the interval between `x1` and `x2`). Falls back to
 * the midpoint of the bounds when the cubic has no real minimizer or the inputs are
 * not finite.
 */
function cubicInterpolate(
  x1: number,
  f1: number,
  g1: number,
  x2: number,
  f2: number,
  g2: number,
  bounds?: readonly [number, number]
): number {
  const [lo, hi] = bounds ?? (x1 <= x2 ? [x1, x2] : [x2, x1]);
  const d1 = g1 + g2 - (3 * (f1 - f2)) / (x1 - x2);
  const d2Square = d1 * d1 - g1 * g2;
  if (Number.isFinite(d1) && d2Square >= 0) {
    const d2 = Math.sqrt(d2Square);
    const minPos =
      x1 <= x2
        ? x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        : x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2));
    if (Number.isFinite(minPos)) {
      return Math.min(Math.max(minPos, lo), hi);
    }
  }
  return (lo + hi) / 2;
}

/**
 * Line search satisfying the strong Wolfe conditions (bracketing followed by a
 * zoom phase with cubic interpolation, as in Nocedal and Wright, Algorithms 3.5
 * and 3.6, and `torch.optim.LBFGS`).
 *
 * @param evaluate - Evaluates the loss and flat gradient at `x + t * d`
 * @param t - Initial step length
 * @param d - Search direction
 * @param f - Loss at `t = 0`
 * @param g - Flat gradient at `t = 0`
 * @param gtd - Directional derivative `g . d` at `t = 0` (must be negative)
 * @param toleranceChange - Stop when the bracket is narrower than this (in parameter units)
 * @returns The accepted step length with the loss and gradient there
 */
function strongWolfe(
  evaluate: (t: number) => LineSearchPoint,
  t0: number,
  d: Float64Array,
  f: number,
  g: Float64Array,
  gtd: number,
  toleranceChange: number
): { f: number; g: Float64Array; t: number } {
  const dNorm = maxAbs(d);
  let t = t0;
  let { f: fNew, g: gNew } = evaluate(t);
  let gtdNew = dot(gNew, d);

  let tPrev = 0;
  let fPrev = f;
  let gPrev = g;
  let gtdPrev = gtd;
  let done = false;
  let lsIter = 0;

  let bracket: number[] = [];
  let bracketF: number[] = [];
  let bracketG: Float64Array[] = [];
  let bracketGtd: number[] = [];

  // Bracketing phase: find an interval that contains a point satisfying the conditions.
  while (lsIter < WOLFE_MAX_LS) {
    if (fNew > f + WOLFE_C1 * t * gtd || (lsIter > 1 && fNew >= fPrev)) {
      bracket = [tPrev, t];
      bracketF = [fPrev, fNew];
      bracketG = [gPrev, gNew];
      bracketGtd = [gtdPrev, gtdNew];
      break;
    }
    if (Math.abs(gtdNew) <= -WOLFE_C2 * gtd) {
      bracket = [t];
      bracketF = [fNew];
      bracketG = [gNew];
      done = true;
      break;
    }
    if (gtdNew >= 0) {
      bracket = [tPrev, t];
      bracketF = [fPrev, fNew];
      bracketG = [gPrev, gNew];
      bracketGtd = [gtdPrev, gtdNew];
      break;
    }

    // Extrapolate with a cubic, keeping the step within [t + 0.01 (t - tPrev), 10 t].
    const minStep = t + 0.01 * (t - tPrev);
    const maxStep = t * 10;
    const previous = t;
    t = cubicInterpolate(tPrev, fPrev, gtdPrev, t, fNew, gtdNew, [minStep, maxStep]);

    tPrev = previous;
    fPrev = fNew;
    gPrev = gNew;
    gtdPrev = gtdNew;
    ({ f: fNew, g: gNew } = evaluate(t));
    gtdNew = dot(gNew, d);
    lsIter++;
  }

  // The evaluation budget ran out while still bracketing: keep the last point.
  if (lsIter === WOLFE_MAX_LS) {
    bracket = [0, t];
    bracketF = [f, fNew];
    bracketG = [g, gNew];
    bracketGtd = [gtd, gtdNew];
  }

  // Zoom phase: shrink the bracket until a point satisfies the conditions.
  let insufficientProgress = false;
  let lowPos = (bracketF[0] as number) <= (bracketF[bracketF.length - 1] as number) ? 0 : 1;
  let highPos = 1 - lowPos;
  while (!done && lsIter < WOLFE_MAX_LS) {
    const b0 = bracket[0] as number;
    const b1 = bracket[1] as number;
    if (Math.abs(b1 - b0) * dNorm < toleranceChange) break;

    t = cubicInterpolate(
      b0,
      bracketF[0] as number,
      bracketGtd[0] as number,
      b1,
      bracketF[1] as number,
      bracketGtd[1] as number
    );

    // Require sufficient progress: keep the trial away from the bracket ends.
    const bMax = Math.max(b0, b1);
    const bMin = Math.min(b0, b1);
    const margin = 0.1 * (bMax - bMin);
    if (Math.min(bMax - t, t - bMin) < margin) {
      if (insufficientProgress || t >= bMax || t <= bMin) {
        t = Math.abs(t - bMax) < Math.abs(t - bMin) ? bMax - margin : bMin + margin;
        insufficientProgress = false;
      } else {
        insufficientProgress = true;
      }
    } else {
      insufficientProgress = false;
    }

    ({ f: fNew, g: gNew } = evaluate(t));
    gtdNew = dot(gNew, d);
    lsIter++;

    if (fNew > f + WOLFE_C1 * t * gtd || fNew >= (bracketF[lowPos] as number)) {
      // Armijo condition violated or not lower than the best point: t becomes the high end.
      bracket[highPos] = t;
      bracketF[highPos] = fNew;
      bracketG[highPos] = gNew;
      bracketGtd[highPos] = gtdNew;
      lowPos = (bracketF[0] as number) <= (bracketF[1] as number) ? 0 : 1;
      highPos = 1 - lowPos;
    } else {
      if (Math.abs(gtdNew) <= -WOLFE_C2 * gtd) {
        done = true;
      } else if (gtdNew * ((bracket[highPos] as number) - (bracket[lowPos] as number)) >= 0) {
        // The old high end becomes the new low end.
        bracket[highPos] = bracket[lowPos] as number;
        bracketF[highPos] = bracketF[lowPos] as number;
        bracketG[highPos] = bracketG[lowPos] as Float64Array;
        bracketGtd[highPos] = bracketGtd[lowPos] as number;
      }
      bracket[lowPos] = t;
      bracketF[lowPos] = fNew;
      bracketG[lowPos] = gNew;
      bracketGtd[lowPos] = gtdNew;
    }
  }

  return {
    f: bracketF[lowPos] as number,
    g: bracketG[lowPos] as Float64Array,
    t: bracket[lowPos] as number,
  };
}

/**
 * L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) optimizer.
 *
 * A quasi-Newton method that approximates the inverse Hessian using a limited
 * history of past updates. Unlike first-order methods (Adam, SGD), LBFGS can
 * converge much faster for smooth, deterministic objectives.
 *
 * **Important**: LBFGS requires a closure that reevaluates the model and returns
 * the loss. This is because LBFGS may evaluate the function multiple times per step.
 * All parameters are treated as one flat vector, so it uses a single set of
 * hyperparameters: parameter groups with differing options are rejected.
 *
 * Each `step(closure)` runs up to `maxIter` L-BFGS iterations. The step length is
 * chosen by one of two line searches:
 * - `lineSearchFn: null` (default): the initial step `lr` is halved (at most 10 times)
 *   until the loss decreases. The first step is scaled by `min(1, 1 / ||g||_1)`.
 *   `maxEval` is checked between iterations, so one line search can exceed it by up to
 *   nine evaluations.
 * - `lineSearchFn: "strong_wolfe"`: a line search satisfying the strong Wolfe
 *   conditions, as in `torch.optim.LBFGS`. This is the better choice for
 *   non-quadratic objectives.
 *
 * LBFGS only supports float32/float64 parameters on the host; device tensors are
 * rejected with a {@link DeviceError}.
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
  // L-BFGS history buffers (shared across all params, flattened)
  private _sHistory: Float64Array[] = [];
  private _yHistory: Float64Array[] = [];
  private _rhoHistory: number[] = [];
  // Parameters and gradient at the start of the previous iteration.
  private _prevFlatGrad: Float64Array | null = null;
  private _prevFlatParams: Float64Array | null = null;

  /**
   * Create a new LBFGS optimizer.
   *
   * @param params - Parameters to optimize (a single parameter group)
   * @param options - Hyperparameters
   * @param options.lr - Step length scale (default: 1)
   * @param options.maxIter - Maximum L-BFGS iterations per `step()` call; an integer >= 1 or
   *   `Infinity` (default: 20)
   * @param options.maxEval - Maximum closure evaluations per `step()` call
   *   (default: `floor(maxIter * 5 / 4)`)
   * @param options.toleranceGrad - Stop when the largest gradient magnitude is at most this
   *   (default: 1e-7)
   * @param options.toleranceChange - Stop when the loss, or the parameter update, changes by
   *   less than this (default: 1e-9)
   * @param options.historySize - Number of curvature pairs kept (default: 10)
   * @param options.lineSearchFn - `"strong_wolfe"` or `null` for backtracking (default: null)
   * @throws {InvalidParameterError} If a hyperparameter is out of range
   */
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
    const maxIter = options.maxIter ?? 20;
    const defaults: LBFGSOptions = {
      lr: options.lr ?? 1,
      maxIter,
      // Same default as PyTorch: scale the evaluation budget with the iteration budget.
      maxEval: options.maxEval ?? Math.floor((maxIter * 5) / 4),
      toleranceGrad: options.toleranceGrad ?? 1e-7,
      toleranceChange: options.toleranceChange ?? 1e-9,
      historySize: options.historySize ?? 10,
      lineSearchFn: options.lineSearchFn ?? null,
    };

    super(params, defaults);
  }

  protected override validateOptions(options: Readonly<LBFGSOptions>): void {
    assertFiniteNonNegative("learning rate", options.lr);
    assertFinitePositive("toleranceGrad", options.toleranceGrad);
    assertFinitePositive("toleranceChange", options.toleranceChange);

    // `Infinity` is accepted for the two budgets: it means "run until converged".
    if (!isCount(options.maxIter)) {
      throw new InvalidParameterError(
        "maxIter must be an integer >= 1 (or Infinity)",
        "maxIter",
        options.maxIter
      );
    }
    if (!isCount(options.maxEval)) {
      throw new InvalidParameterError(
        "maxEval must be an integer >= 1 (or Infinity)",
        "maxEval",
        options.maxEval
      );
    }
    if (!Number.isInteger(options.historySize) || options.historySize < 1) {
      throw new InvalidParameterError(
        "historySize must be an integer >= 1",
        "historySize",
        options.historySize
      );
    }
    if (options.lineSearchFn !== null && options.lineSearchFn !== "strong_wolfe") {
      throw new InvalidParameterError(
        `lineSearchFn must be "strong_wolfe" or null; received ${String(options.lineSearchFn)}`,
        "lineSearchFn",
        options.lineSearchFn
      );
    }
  }

  protected isState(state: Record<string, unknown>): state is LBFGSState {
    return typeof state["step"] === "number";
  }

  /**
   * Snapshot of the optimizer, including the L-BFGS curvature history so that a
   * resumed run continues with the same Hessian approximation.
   */
  override stateDict() {
    const copy = (a: Float64Array | null) => (a === null ? null : new Float64Array(a));
    return {
      ...super.stateDict(),
      lbfgs: {
        stepCount: this.stepCount,
        sHistory: this._sHistory.map((a) => new Float64Array(a)),
        yHistory: this._yHistory.map((a) => new Float64Array(a)),
        rhoHistory: [...this._rhoHistory],
        prevFlatGrad: copy(this._prevFlatGrad),
        prevFlatParams: copy(this._prevFlatParams),
      },
    };
  }

  /**
   * Restore optimizer state saved by {@link LBFGS.stateDict}. A state dictionary without
   * L-BFGS history (for example one written by an older version) clears the history.
   *
   * @param stateDict - State dictionary previously returned by `stateDict()`
   * @throws {DataValidationError} If the dictionary is malformed or its buffers do not
   *   match the optimized parameters
   */
  override loadStateDict(stateDict: Record<string, unknown>): void {
    if (typeof stateDict !== "object" || stateDict === null) {
      super.loadStateDict(stateDict); // throws DataValidationError
      return;
    }

    // Parse the L-BFGS history first so that a malformed entry leaves the optimizer untouched.
    let history: {
      stepCount: number;
      s: Float64Array[];
      y: Float64Array[];
      rho: number[];
      prevGrad: Float64Array | null;
      prevParams: Float64Array | null;
    } | null = null;

    if (Object.hasOwn(stateDict, "lbfgs")) {
      const raw = stateDict["lbfgs"];
      if (typeof raw !== "object" || raw === null) {
        throw new DataValidationError("lbfgs must be an object");
      }
      const record: Record<string, unknown> = { ...raw };
      let total = 0;
      for (const group of this.paramGroups) {
        for (const param of this.requiresGradParams(group)) total += param.tensor.size;
      }
      const flat = (value: unknown, name: string): Float64Array => {
        if (!(value instanceof Float64Array) || value.length !== total) {
          throw new DataValidationError(`lbfgs.${name} must be a Float64Array of length ${total}`);
        }
        return new Float64Array(value);
      };
      const flatList = (value: unknown, name: string): Float64Array[] => {
        if (!Array.isArray(value)) {
          throw new DataValidationError(`lbfgs.${name} must be an array`);
        }
        return value.map((entry) => flat(entry, name));
      };
      const optionalFlat = (value: unknown, name: string): Float64Array | null =>
        value === null || value === undefined ? null : flat(value, name);

      const rho = record["rhoHistory"];
      if (!Array.isArray(rho) || !rho.every((v) => typeof v === "number")) {
        throw new DataValidationError("lbfgs.rhoHistory must be an array of numbers");
      }
      const s = flatList(record["sHistory"], "sHistory");
      const y = flatList(record["yHistory"], "yHistory");
      if (s.length !== y.length || s.length !== rho.length) {
        throw new DataValidationError("lbfgs history arrays must have equal length");
      }
      const stepCount = record["stepCount"];
      if (typeof stepCount !== "number" || !Number.isInteger(stepCount) || stepCount < 0) {
        throw new DataValidationError("lbfgs.stepCount must be a non-negative integer");
      }
      const prevGrad = optionalFlat(record["prevFlatGrad"], "prevFlatGrad");
      const prevParams = optionalFlat(record["prevFlatParams"], "prevFlatParams");
      history = { stepCount, s, y, rho: [...rho], prevGrad, prevParams };
    }

    super.loadStateDict(stateDict);

    this._sHistory = history?.s ?? [];
    this._yHistory = history?.y ?? [];
    this._rhoHistory = history?.rho ?? [];
    this._prevFlatGrad = history?.prevGrad ?? null;
    this._prevFlatParams = history?.prevParams ?? null;
    // A dictionary written before the base class stored the step count only has it here.
    if (history && !Object.hasOwn(stateDict, "stepCount")) this.restoreStepCount(history.stepCount);
  }

  /** Options shared by all parameter groups; LBFGS cannot honor per-group options. */
  private resolveOptions(): LBFGSOptions {
    const first = this.paramGroups[0]?.options ?? this.defaults;
    for (let g = 1; g < this.paramGroups.length; g++) {
      const options = this.paramGroups[g]?.options;
      if (!options) continue;
      for (const key of Object.keys(first) as Array<keyof LBFGSOptions>) {
        if (options[key] !== first[key]) {
          throw new InvalidParameterError(
            `LBFGS does not support per-group options (group ${g} differs in "${key}")`,
            key,
            options[key]
          );
        }
      }
    }
    this.validateOptions(first);
    return first;
  }

  /** Gather all parameter values into a single flat array. */
  private flattenParams(): Float64Array {
    let total = 0;
    for (const group of this.paramGroups) {
      for (const param of this.requiresGradParams(group)) total += param.tensor.size;
    }
    const flat = new Float64Array(total);
    let idx = 0;
    for (const group of this.paramGroups) {
      for (const param of this.requiresGradParams(group)) {
        const { param: pData, paramOffset } = assertFloatParam(param, "LBFGS");
        const size = param.tensor.size;
        for (let i = 0; i < size; i++) {
          const v = pData[paramOffset + i] as number;
          if (!Number.isFinite(v)) assertFinite("parameter", v);
          flat[idx++] = v;
        }
      }
    }
    return flat;
  }

  /**
   * Gather all gradient values into a single flat array. With `strict`, a non-finite
   * gradient is an error; otherwise it is returned as is (used at line-search trial
   * points, where an overflow just means the step was too long).
   */
  private flattenGrads(strict: boolean): Float64Array {
    let total = 0;
    for (const group of this.paramGroups) {
      for (const param of this.requiresGradParams(group)) total += param.tensor.size;
    }
    const flat = new Float64Array(total);
    let idx = 0;
    for (const group of this.paramGroups) {
      for (const param of this.requiresGradParams(group)) {
        const size = param.tensor.size;
        // A parameter that received no gradient contributes zeros (as in PyTorch).
        if (param.grad === null) {
          idx += size;
          continue;
        }
        const { grad, gradOffset } = assertHasGradFloat(param, "LBFGS");
        for (let i = 0; i < size; i++) {
          const v = grad[gradOffset + i] as number;
          if (strict && !Number.isFinite(v)) assertFinite("gradient", v);
          flat[idx++] = v;
        }
      }
    }
    return flat;
  }

  /** Write `x + t * d` into the parameters (`d` omitted: write `x` itself). */
  private writeParams(x: Float64Array, d?: Float64Array, t = 0): void {
    let idx = 0;
    for (const group of this.paramGroups) {
      for (const param of this.requiresGradParams(group)) {
        const { param: pData, paramOffset } = assertFloatParam(param, "LBFGS");
        const size = param.tensor.size;
        for (let i = 0; i < size; i++) {
          const base = x[idx] as number;
          pData[paramOffset + i] = d ? base + t * (d[idx] as number) : base;
          idx++;
        }
      }
    }
  }

  /** L-BFGS two-loop recursion: returns the search direction `-H g`. */
  private computeDirection(grad: Float64Array): Float64Array {
    const n = grad.length;
    const q = new Float64Array(n);
    for (let i = 0; i < n; i++) q[i] = -(grad[i] as number);

    const m = this._sHistory.length;
    if (m === 0) return q;

    const alphas = new Float64Array(m);

    for (let i = m - 1; i >= 0; i--) {
      const s = this._sHistory[i] as Float64Array;
      const y = this._yHistory[i] as Float64Array;
      const alpha = (this._rhoHistory[i] as number) * dot(s, q);
      alphas[i] = alpha;
      for (let j = 0; j < n; j++) {
        q[j] = (q[j] as number) - alpha * (y[j] as number);
      }
    }

    // Scale by the initial Hessian approximation H0 = (s^T y) / (y^T y) of the newest pair.
    const sLast = this._sHistory[m - 1] as Float64Array;
    const yLast = this._yHistory[m - 1] as Float64Array;
    const yy = dot(yLast, yLast);
    const gamma = yy > 0 ? dot(yLast, sLast) / yy : 1;
    for (let j = 0; j < n; j++) {
      q[j] = (q[j] as number) * gamma;
    }

    for (let i = 0; i < m; i++) {
      const s = this._sHistory[i] as Float64Array;
      const y = this._yHistory[i] as Float64Array;
      const beta = (this._rhoHistory[i] as number) * dot(y, q);
      const coeff = (alphas[i] as number) - beta;
      for (let j = 0; j < n; j++) {
        q[j] = (q[j] as number) + coeff * (s[j] as number);
      }
    }

    return q;
  }

  private clearHistory(): void {
    this._sHistory.length = 0;
    this._yHistory.length = 0;
    this._rhoHistory.length = 0;
  }

  /** Record the curvature pair `(s, y)` when it satisfies the curvature condition. */
  private pushCurvaturePair(s: Float64Array, y: Float64Array, historySize: number): void {
    const ys = dot(y, s);
    if (!(ys > CURVATURE_EPS)) return;
    while (this._sHistory.length >= historySize) {
      this._sHistory.shift();
      this._yHistory.shift();
      this._rhoHistory.shift();
    }
    this._sHistory.push(s);
    this._yHistory.push(y);
    this._rhoHistory.push(1 / ys);
  }

  /**
   * Run up to `maxIter` L-BFGS iterations.
   *
   * A parameter that has no gradient after the closure counts as a zero gradient.
   *
   * @param closure - Function that zeroes the gradients, recomputes the loss, calls
   *   `backward()` and returns the loss. It may be called several times per step.
   * @returns The loss at the final parameters. The `.grad` buffers hold the gradient of the
   *   last point the closure was evaluated at, which after a line search can be a rejected
   *   trial point rather than the final parameters.
   * @throws {InvalidParameterError} If no closure is given, the options are invalid or
   *   differ between parameter groups, or a gradient or parameter is not finite
   * @throws {DeviceError} If a parameter lives on a device
   */
  step(closure?: () => number): number | undefined {
    if (!closure) {
      throw new InvalidParameterError(
        "LBFGS requires a closure that reevaluates the model and returns the loss",
        "closure",
        undefined
      );
    }

    // LBFGS flattens parameters/gradients into a single host vector and runs a
    // line search that reads scalar loss/curvature values back, none of which
    // is possible on opaque device memory. Reject device parameters clearly.
    for (const group of this.paramGroups) {
      for (const param of this.requiresGradParams(group)) {
        if (param.tensor.isDeviceTensor) {
          throw new DeviceError(
            "LBFGS is not supported on device tensors (it needs host-readable parameters and scalar line-search values). Move the parameters to CPU with `.to('cpu')` before optimizing."
          );
        }
      }
    }

    const { lr, maxIter, maxEval, toleranceGrad, toleranceChange, historySize, lineSearchFn } =
      this.resolveOptions();

    this.countStep();
    let nFuncEval = 0;

    // Evaluate initial loss and gradient
    let loss = closure();
    nFuncEval++;
    let flatGrad = this.flattenGrads(true);

    if (maxAbs(flatGrad) <= toleranceGrad) return loss;

    let flatParams = this.flattenParams();

    for (let iter = 0; iter < maxIter; iter++) {
      // Curvature pair from the previous iteration (possibly from the previous step() call).
      if (this._prevFlatGrad !== null && this._prevFlatParams !== null) {
        const n = flatGrad.length;
        if (this._prevFlatGrad.length !== n || this._prevFlatParams.length !== n) {
          // The set of trainable parameters changed (for example a layer was frozen).
          this.clearHistory();
          this._prevFlatGrad = null;
          this._prevFlatParams = null;
        }
      }
      if (this._prevFlatGrad !== null && this._prevFlatParams !== null) {
        const n = flatGrad.length;
        const s = new Float64Array(n);
        const y = new Float64Array(n);
        for (let i = 0; i < n; i++) {
          s[i] = (flatParams[i] as number) - (this._prevFlatParams[i] as number);
          y[i] = (flatGrad[i] as number) - (this._prevFlatGrad[i] as number);
        }
        this.pushCurvaturePair(s, y, historySize);
      }
      this._prevFlatGrad = flatGrad;
      this._prevFlatParams = flatParams;

      // Search direction; fall back to steepest descent if it is not a descent direction.
      let direction = this.computeDirection(flatGrad);
      let gtd = dot(flatGrad, direction);
      if (!(gtd < 0)) {
        this.clearHistory();
        direction = this.computeDirection(flatGrad);
        gtd = dot(flatGrad, direction);
      }
      // The directional derivative is too small to make progress.
      if (gtd > -toleranceChange) break;

      // Initial step length: scale the very first (steepest descent) step so that it is
      // not larger than the gradient's l1 norm allows.
      let t = this._sHistory.length === 0 ? Math.min(1, 1 / sumAbs(flatGrad)) * lr : lr;

      const prevLoss = loss;
      const x = flatParams;
      let evalsThisIteration = 0;

      if (lineSearchFn === "strong_wolfe") {
        const evaluate = (step: number): LineSearchPoint => {
          this.writeParams(x, direction, step);
          const f = closure();
          evalsThisIteration++;
          const g = this.flattenGrads(false);
          return { f: Number.isFinite(f) && isFiniteArray(g) ? f : Number.POSITIVE_INFINITY, g };
        };
        const result = strongWolfe(evaluate, t, direction, loss, flatGrad, gtd, toleranceChange);
        loss = result.f;
        flatGrad = result.g;
        t = result.t;
        this.writeParams(x, direction, t);
      } else {
        // Backtracking: halve the step until the loss decreases.
        let accepted = false;
        let trialLoss = loss;
        for (let ls = 0; ls < BACKTRACK_MAX_HALVINGS; ls++) {
          this.writeParams(x, direction, t);
          trialLoss = closure();
          evalsThisIteration++;
          if (trialLoss < loss) {
            accepted = true;
            break;
          }
          t *= 0.5;
        }

        if (!accepted) {
          this.writeParams(x);
          nFuncEval += evalsThisIteration;
          if (this._sHistory.length > 0 && nFuncEval < maxEval) {
            // The quasi-Newton direction failed: retry from steepest descent.
            this.clearHistory();
            continue;
          }
          break;
        }
        loss = trialLoss;
        flatGrad = this.flattenGrads(true);
      }

      nFuncEval += evalsThisIteration;
      // Parameters as actually stored (float32 parameters round the update).
      flatParams = this.flattenParams();

      if (iter === maxIter - 1) break;
      if (nFuncEval >= maxEval) break;
      if (maxAbs(flatGrad) <= toleranceGrad) break;
      if (maxAbs(direction) * Math.abs(t) <= toleranceChange) break;
      if (Math.abs(loss - prevLoss) < toleranceChange) break;
    }

    return loss;
  }
}
