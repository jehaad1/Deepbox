/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import {
  DataValidationError,
  DeepboxError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
  warn,
} from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import {
  assertContiguous,
  toFloat64View,
  validateFitInputs,
  validatePredictInputs,
} from "../_validation";
import type { Classifier } from "../base";

type Penalty = "l1" | "l2" | "none";
type Solver = "lbfgs" | "liblinear" | "saga";
type MultiClass = "ovr" | "auto" | "multinomial";
type ClassWeight = "balanced" | Record<number, number>;

interface LogisticOptions {
  penalty?: Penalty;
  tol?: number;
  C?: number;
  fitIntercept?: boolean;
  maxIter?: number;
  learningRate?: number;
  multiClass?: MultiClass;
  classWeight?: ClassWeight;
  solver?: Solver;
}

/** Number of L-BFGS correction pairs kept. */
const LBFGS_HISTORY = 10;

/** Numerically stable logistic function. */
function sigmoid(z: number): number {
  if (z >= 0) {
    return 1 / (1 + Math.exp(-z));
  }
  const ez = Math.exp(z);
  return ez / (1 + ez);
}

/** Numerically stable `log(1 + exp(z))`. */
function softplus(z: number): number {
  return z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z));
}

/** Objective evaluator: writes the gradient into `grad` and returns the value. */
type Objective = (theta: Float64Array, grad: Float64Array) => number;

function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] as number) * (b[i] as number);
  return s;
}

function maxAbs(a: Float64Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const v = Math.abs(a[i] as number);
    if (v > m) m = v;
  }
  return m;
}

interface SolveResult {
  theta: Float64Array;
  nIter: number;
  converged: boolean;
}

/**
 * Limited-memory BFGS with an Armijo backtracking line search.
 *
 * Stops when the largest gradient entry is at most `tol`, or when the
 * relative decrease of the objective falls to `64 * eps` (the same rules as
 * scikit-learn's `lbfgs` solver).
 */
function lbfgs(
  objective: Objective,
  theta0: Float64Array,
  maxIter: number,
  tol: number
): SolveResult {
  const dim = theta0.length;
  let x = Float64Array.from(theta0);
  let g = new Float64Array(dim);
  let f = objective(x, g);
  const sHist: Float64Array[] = [];
  const yHist: Float64Array[] = [];
  const rhoHist: number[] = [];
  const alphaBuf = new Float64Array(LBFGS_HISTORY);
  const ftol = 64 * Number.EPSILON;

  let nIter = 0;
  let converged = false;

  for (let iter = 0; iter < maxIter; iter++) {
    if (maxAbs(g) <= tol) {
      converged = true;
      break;
    }
    nIter = iter + 1;

    // Two-loop recursion for d = -H g.
    const d = Float64Array.from(g);
    const h = sHist.length;
    for (let i = h - 1; i >= 0; i--) {
      const si = sHist[i] as Float64Array;
      const yi = yHist[i] as Float64Array;
      const a = (rhoHist[i] as number) * dot(si, d);
      alphaBuf[i] = a;
      for (let k = 0; k < dim; k++) d[k] = (d[k] as number) - a * (yi[k] as number);
    }
    if (h > 0) {
      const sLast = sHist[h - 1] as Float64Array;
      const yLast = yHist[h - 1] as Float64Array;
      const gamma = dot(sLast, yLast) / dot(yLast, yLast);
      for (let k = 0; k < dim; k++) d[k] = (d[k] as number) * gamma;
    }
    for (let i = 0; i < h; i++) {
      const si = sHist[i] as Float64Array;
      const yi = yHist[i] as Float64Array;
      const b = (rhoHist[i] as number) * dot(yi, d);
      const a = alphaBuf[i] as number;
      for (let k = 0; k < dim; k++) d[k] = (d[k] as number) + si[k]! * (a - b);
    }
    for (let k = 0; k < dim; k++) d[k] = -(d[k] as number);

    let slope = dot(g, d);
    if (!(slope < 0)) {
      // Not a descent direction (numerical breakdown): restart from steepest descent.
      sHist.length = 0;
      yHist.length = 0;
      rhoHist.length = 0;
      for (let k = 0; k < dim; k++) d[k] = -(g[k] as number);
      slope = dot(g, d);
    }

    let step = sHist.length === 0 ? Math.min(1, 1 / Math.sqrt(-slope)) : 1;
    const xNew = new Float64Array(dim);
    const gNew = new Float64Array(dim);
    let fNew = f;
    let accepted = false;
    for (let ls = 0; ls < 40; ls++) {
      for (let k = 0; k < dim; k++) xNew[k] = (x[k] as number) + step * (d[k] as number);
      fNew = objective(xNew, gNew);
      if (Number.isFinite(fNew) && fNew <= f + 1e-4 * step * slope) {
        accepted = true;
        break;
      }
      step *= 0.5;
    }
    if (!accepted) {
      if (sHist.length > 0) {
        // Drop the curvature model and retry with a gradient step.
        sHist.length = 0;
        yHist.length = 0;
        rhoHist.length = 0;
        continue;
      }
      break;
    }

    const s = new Float64Array(dim);
    const yv = new Float64Array(dim);
    for (let k = 0; k < dim; k++) {
      s[k] = (xNew[k] as number) - (x[k] as number);
      yv[k] = (gNew[k] as number) - (g[k] as number);
    }
    const sy = dot(s, yv);
    if (sy > 1e-10 * Math.sqrt(dot(s, s) * dot(yv, yv))) {
      if (sHist.length === LBFGS_HISTORY) {
        sHist.shift();
        yHist.shift();
        rhoHist.shift();
      }
      sHist.push(s);
      yHist.push(yv);
      rhoHist.push(1 / sy);
    }

    const decrease = f - fNew;
    x = Float64Array.from(xNew);
    g = gNew;
    f = fNew;
    if (decrease <= ftol * Math.max(Math.abs(f), Math.abs(f + decrease), 1)) {
      converged = true;
      break;
    }
  }
  if (!converged && maxAbs(g) <= tol) converged = true;
  return { theta: x, nIter, converged };
}

/**
 * Accelerated proximal gradient (FISTA) with backtracking and function-value
 * restarts, for `smooth(theta) + l1 * sum_{j < nPenalized} |theta_j|`.
 *
 * Stops when the largest entry of the proximal gradient mapping is at most `tol`.
 */
function proximalGradient(
  objective: Objective,
  theta0: Float64Array,
  nPenalized: number,
  l1: number,
  initialStep: number,
  maxIter: number,
  tol: number
): SolveResult {
  const dim = theta0.length;
  const l1Norm = (v: Float64Array): number => {
    let s = 0;
    for (let j = 0; j < nPenalized; j++) s += Math.abs(v[j] as number);
    return s * l1;
  };
  const prox = (point: Float64Array, grad: Float64Array, L: number, out: Float64Array): void => {
    const thr = l1 / L;
    for (let j = 0; j < dim; j++) {
      const v = (point[j] as number) - (grad[j] as number) / L;
      if (j < nPenalized) {
        out[j] = v > thr ? v - thr : v < -thr ? v + thr : 0;
      } else {
        out[j] = v;
      }
    }
  };

  let x = Float64Array.from(theta0);
  let yv = Float64Array.from(theta0);
  const gy = new Float64Array(dim);
  const gx = new Float64Array(dim);
  let xNew = new Float64Array(dim);
  const mapping = new Float64Array(dim);
  let L = 1 / initialStep;
  let momentum = 1;
  let shrinkL = true;
  let objX = objective(x, gx) + l1Norm(x);
  let nIter = 0;
  let converged = false;

  for (let iter = 0; iter < maxIter; iter++) {
    // Optimality of the current iterate: proximal gradient mapping at x.
    prox(x, gx, L, mapping);
    let res = 0;
    for (let j = 0; j < dim; j++) {
      res = Math.max(res, Math.abs(L * ((x[j] as number) - (mapping[j] as number))));
    }
    if (res <= tol) {
      converged = true;
      break;
    }
    nIter = iter + 1;

    const fy = objective(yv, gy);
    // Try a smaller Lipschitz estimate only while the objective still drops
    // clearly. Near the optimum the sufficient-decrease test passes for
    // almost any L because of rounding, and a tiny L makes the iterates wander.
    let Lk = shrinkL ? Math.max(L / 2, 1e-12) : L;
    let accepted = false;
    let fNew = fy;
    for (let bt = 0; bt < 60; bt++) {
      prox(yv, gy, Lk, xNew);
      let lin = 0;
      let quad = 0;
      for (let j = 0; j < dim; j++) {
        const dj = (xNew[j] as number) - (yv[j] as number);
        lin += (gy[j] as number) * dj;
        quad += dj * dj;
      }
      fNew = objective(xNew, gx);
      if (Number.isFinite(fNew) && fNew <= fy + lin + 0.5 * Lk * quad + 1e-12 * Math.abs(fy)) {
        accepted = true;
        break;
      }
      Lk *= 2;
    }
    if (!accepted) break;
    L = Lk;

    const objNew = fNew + l1Norm(xNew);
    shrinkL = objX - objNew > 1e-10 * Math.max(1, Math.abs(objX));
    if (objNew > objX && momentum > 1) {
      // Momentum overshot: restart from the best point (gx must match x again).
      // A plain proximal step (momentum === 1) is always accepted; otherwise
      // rounding noise near the optimum could trigger restarts forever.
      momentum = 1;
      yv = Float64Array.from(x);
      objective(x, gx);
      continue;
    }
    const nextMomentum = (1 + Math.sqrt(1 + 4 * momentum * momentum)) / 2;
    const beta = (momentum - 1) / nextMomentum;
    const prev = x;
    x = xNew;
    xNew = new Float64Array(dim);
    yv = new Float64Array(dim);
    for (let j = 0; j < dim; j++) {
      yv[j] = (x[j] as number) + beta * ((x[j] as number) - (prev[j] as number));
    }
    momentum = nextMomentum;
    objX = objNew;
  }
  if (!converged) {
    objective(x, gx);
    prox(x, gx, L, mapping);
    let res = 0;
    for (let j = 0; j < dim; j++) {
      res = Math.max(res, Math.abs(L * ((x[j] as number) - (mapping[j] as number))));
    }
    if (res <= tol) converged = true;
  }
  return { theta: x, nIter, converged };
}

/**
 * Logistic Regression (Binary and Multiclass Classification).
 *
 * Minimizes the regularized negative log-likelihood. With sample weights
 * `s_i` (from `classWeight`) the objective is, up to a constant factor,
 *
 *   C * sum_i s_i * logloss_i + penalty(w)
 *
 * where `penalty` is `0.5 * ||w||_2^2` (`'l2'`), `||w||_1` (`'l1'`) or zero
 * (`'none'`). This is scikit-learn's objective; the intercept is never
 * penalized. L2 and unpenalized problems are solved with L-BFGS, L1 problems
 * with an accelerated proximal gradient method. The `solver` option is
 * accepted for scikit-learn compatibility and checked against the penalty by
 * the constructor, but every solver minimizes the same objective and reaches
 * the same solution. (scikit-learn's `liblinear` also penalizes the intercept,
 * so its results differ slightly; here the intercept is never penalized.)
 *
 * With more than two classes, `multiClass: 'multinomial'` fits a softmax model
 * and `'ovr'` fits one binary model per class and normalizes their
 * probabilities. `'auto'` (the default) uses `'multinomial'`, except with
 * `solver: 'liblinear'`, which uses `'ovr'`. Binary problems use a single
 * logistic model in every mode. In a one-vs-rest fit, `classWeight: 'balanced'`
 * balances each binary problem (class k against the rest), while a weight
 * dictionary applies the weight of each sample's original class.
 *
 * @example
 * ```ts
 * import { LogisticRegression } from 'deepbox/ml';
 *
 * // Binary classification
 * const model = new LogisticRegression({ C: 1.0, maxIter: 100 });
 * model.fit(X_train, y_train);
 *
 * const predictions = model.predict(X_test);
 * const probabilities = model.predictProba(X_test);
 *
 * // Multiclass classification
 * const multiModel = new LogisticRegression({ multiClass: 'multinomial' });
 * multiModel.fit(X_train_multi, y_train_multi);
 * ```
 *
 * @category Linear Models
 * @implements {Classifier}
 */
export class LogisticRegression implements Classifier {
  private options: LogisticOptions;

  private coef_?: Tensor; // Shape (n_features,) for binary, (n_classes, n_features) for multiclass
  private coefArray_?: Float64Array;
  private intercept_?: number | number[]; // Scalar for binary, Array for multiclass
  private nFeaturesIn_?: number;
  private classes_?: Tensor;
  private nIter_ = 0;
  private fitted = false;
  private multiclass_ = false;
  private softmax_ = false;

  /**
   * Create a new Logistic Regression classifier.
   *
   * @param options - Configuration options
   * @param options.penalty - Regularization type: 'l1', 'l2', or 'none' (default: 'l2')
   * @param options.C - Inverse regularization strength (default: 1.0). Must be > 0; `Infinity` disables regularization.
   * @param options.tol - Convergence tolerance (default: 1e-4). L-BFGS stops when the largest gradient entry of the mean loss is below it; the L1 solver uses the proximal gradient mapping.
   * @param options.maxIter - Maximum number of solver iterations (default: 1000)
   * @param options.fitIntercept - Whether to fit intercept (default: true)
   * @param options.learningRate - Initial step size of the L1 proximal gradient solver (default: 0.1). It is reduced automatically when too large and is ignored by L-BFGS.
   * @param options.multiClass - Multiclass strategy: 'ovr' (one-vs-rest), 'multinomial' (softmax) or 'auto' (default: 'auto')
   * @param options.classWeight - Class weights: 'balanced' or {classLabel: weight} (default: undefined = equal weights)
   * @param options.solver - 'lbfgs' (l2/none only), 'liblinear' (l1/l2, one-vs-rest only), 'saga' (l1/l2/none) (default: 'lbfgs')
   * @throws {InvalidParameterError} If an option is invalid or the solver does not support the penalty
   */
  constructor(
    options: {
      readonly penalty?: Penalty;
      readonly tol?: number;
      readonly C?: number;
      readonly fitIntercept?: boolean;
      readonly maxIter?: number;
      readonly learningRate?: number;
      readonly multiClass?: MultiClass;
      readonly classWeight?: ClassWeight;
      readonly solver?: Solver;
    } = {}
  ) {
    this.options = { ...options };
    if (options.classWeight !== undefined && typeof options.classWeight === "object") {
      this.options.classWeight = { ...options.classWeight };
    }

    const penalty = this.options.penalty ?? "l2";
    if (penalty !== "l1" && penalty !== "l2" && penalty !== "none") {
      throw new InvalidParameterError(
        `penalty must be 'l1', 'l2', or 'none'; received ${String(penalty)}`,
        "penalty",
        penalty
      );
    }
    this.options.penalty = penalty;

    const solver = this.options.solver ?? "lbfgs";
    if (solver !== "lbfgs" && solver !== "liblinear" && solver !== "saga") {
      throw new InvalidParameterError(
        `solver must be 'lbfgs', 'liblinear', or 'saga'; received ${String(solver)}`,
        "solver",
        solver
      );
    }
    this.options.solver = solver;

    // Validate solver/penalty compatibility
    LogisticRegression.assertSolverPenalty(solver, penalty);

    const multiClass = this.options.multiClass ?? "auto";
    if (multiClass !== "ovr" && multiClass !== "auto" && multiClass !== "multinomial") {
      throw new InvalidParameterError(
        `multiClass must be 'ovr', 'multinomial' or 'auto'; received ${String(multiClass)}`,
        "multiClass",
        multiClass
      );
    }
    this.options.multiClass = multiClass;

    const C = this.options.C;
    if (C !== undefined && !(C > 0)) {
      throw new InvalidParameterError(`C must be > 0; received ${C}`, "C", C);
    }
    if (
      this.options.maxIter !== undefined &&
      (!Number.isFinite(this.options.maxIter) || this.options.maxIter <= 0)
    ) {
      throw new InvalidParameterError(
        `maxIter must be a positive finite number; received ${this.options.maxIter}`,
        "maxIter",
        this.options.maxIter
      );
    }
    if (
      this.options.tol !== undefined &&
      (!Number.isFinite(this.options.tol) || this.options.tol < 0)
    ) {
      throw new InvalidParameterError(
        `tol must be a finite number >= 0; received ${this.options.tol}`,
        "tol",
        this.options.tol
      );
    }
    if (
      this.options.learningRate !== undefined &&
      (!Number.isFinite(this.options.learningRate) || this.options.learningRate <= 0)
    ) {
      throw new InvalidParameterError(
        `learningRate must be a positive finite number; received ${this.options.learningRate}`,
        "learningRate",
        this.options.learningRate
      );
    }
    if (this.options.fitIntercept !== undefined && typeof this.options.fitIntercept !== "boolean") {
      throw new InvalidParameterError(
        `fitIntercept must be a boolean; received ${String(this.options.fitIntercept)}`,
        "fitIntercept",
        this.options.fitIntercept
      );
    }
    if (this.options.classWeight !== undefined) {
      LogisticRegression.assertClassWeight(this.options.classWeight);
    }
  }

  private static assertSolverPenalty(solver: Solver, penalty: Penalty): void {
    if (solver === "lbfgs" && penalty === "l1") {
      throw new InvalidParameterError(
        "solver='lbfgs' does not support penalty='l1'; use 'liblinear' or 'saga'",
        "solver",
        solver
      );
    }
  }

  private static assertClassWeight(cw: unknown): void {
    if (cw === "balanced") return;
    if (typeof cw !== "object" || cw === null || Array.isArray(cw)) {
      throw new InvalidParameterError(
        "classWeight must be 'balanced' or an object mapping class labels to weights",
        "classWeight",
        cw
      );
    }
    for (const [label, weight] of Object.entries(cw)) {
      if (!Number.isFinite(Number(label))) {
        throw new InvalidParameterError(
          `classWeight keys must be numeric class labels; received '${label}'`,
          "classWeight",
          cw
        );
      }
      if (typeof weight !== "number" || !Number.isFinite(weight) || weight < 0) {
        throw new InvalidParameterError(
          `classWeight values must be finite numbers >= 0; received ${String(weight)} for class ${label}`,
          "classWeight",
          cw
        );
      }
    }
  }

  private ensureFitted(): void {
    if (!this.fitted || !this.coef_) {
      throw new NotFittedError("LogisticRegression must be fitted before using this method");
    }
  }

  /**
   * Build the objective (mean weighted log loss plus the L2 term) for `nOut`
   * logits. `nOut = 1` is the binary logistic model with targets in {0, 1}
   * (`labels` holds 0/1); otherwise a softmax model over `nOut` classes
   * (`labels` holds class indices).
   *
   * Parameter layout: `nOut * n` weights (row-major per output), followed by
   * `nOut` intercepts when `fitIntercept` is true.
   */
  private buildObjective(
    xData: Float64Array,
    labels: Int32Array,
    sw: Float64Array | undefined,
    swSum: number,
    m: number,
    n: number,
    nOut: number,
    fitIntercept: boolean,
    l2: number
  ): Objective {
    const nW = nOut * n;
    const logits = new Float64Array(nOut);
    return (theta, grad) => {
      grad.fill(0);
      let loss = 0;
      for (let i = 0; i < m; i++) {
        const base = i * n;
        const wi = sw ? (sw[i] as number) : 1;
        if (wi === 0) continue;
        if (nOut === 1) {
          let z = fitIntercept ? (theta[nW] as number) : 0;
          for (let j = 0; j < n; j++) z += (theta[j] as number) * (xData[base + j] as number);
          const yi = labels[i] as number;
          loss += wi * (softplus(z) - yi * z);
          const e = wi * (sigmoid(z) - yi);
          for (let j = 0; j < n; j++) {
            grad[j] = (grad[j] as number) + e * (xData[base + j] as number);
          }
          if (fitIntercept) grad[nW] = (grad[nW] as number) + e;
        } else {
          const yi = labels[i] as number;
          let maxLogit = -Infinity;
          let trueLogit = 0;
          for (let k = 0; k < nOut; k++) {
            let z = fitIntercept ? (theta[nW + k] as number) : 0;
            const wBase = k * n;
            for (let j = 0; j < n; j++) {
              z += (theta[wBase + j] as number) * (xData[base + j] as number);
            }
            logits[k] = z;
            if (k === yi) trueLogit = z;
            if (z > maxLogit) maxLogit = z;
          }
          let sumExp = 0;
          for (let k = 0; k < nOut; k++) {
            const ez = Math.exp((logits[k] as number) - maxLogit);
            logits[k] = ez;
            sumExp += ez;
          }
          loss += wi * (maxLogit + Math.log(sumExp) - trueLogit);
          for (let k = 0; k < nOut; k++) {
            const p = (logits[k] as number) / sumExp;
            const e = wi * (p - (k === yi ? 1 : 0));
            const wBase = k * n;
            for (let j = 0; j < n; j++) {
              grad[wBase + j] = (grad[wBase + j] as number) + e * (xData[base + j] as number);
            }
            if (fitIntercept) grad[nW + k] = (grad[nW + k] as number) + e;
          }
        }
      }
      const invS = 1 / swSum;
      let value = loss * invS;
      for (let k = 0; k < grad.length; k++) grad[k] = (grad[k] as number) * invS;
      if (l2 > 0) {
        let sq = 0;
        for (let j = 0; j < nW; j++) {
          const wj = theta[j] as number;
          sq += wj * wj;
          grad[j] = (grad[j] as number) + l2 * wj;
        }
        value += 0.5 * l2 * sq;
      }
      return value;
    };
  }

  /**
   * Fit logistic regression model.
   *
   * Binary problems fit one logistic model for `classes[1]` against
   * `classes[0]`. With more than two classes, `multiClass` selects a softmax
   * model or one-vs-rest. Class labels can be any finite numbers.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target labels of shape (n_samples,)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D or y is not 1D
   * @throws {ShapeError} If X and y have different number of samples
   * @throws {DataValidationError} If X or y contain NaN/Inf values
   * @throws {DataValidationError} If X or y are empty, or all sample weights are zero
   * @throws {InvalidParameterError} If `multiClass='multinomial'` is combined with `solver='liblinear'` and there are more than two classes
   */
  fit(X: Tensor, y: Tensor): this {
    // Validate inputs (dimensions, empty data, NaN/Inf)
    validateFitInputs(X, y);

    const penalty = this.options.penalty ?? "l2";
    const solver = this.options.solver ?? "lbfgs";
    const multiClass = this.options.multiClass ?? "auto";
    const C = this.options.C ?? 1.0;
    if (!(C > 0)) {
      throw new InvalidParameterError(`C must be > 0; received ${C}`, "C", C);
    }
    const maxIter = Math.max(1, Math.floor(this.options.maxIter ?? 1000));
    const tol = this.options.tol ?? 1e-4;
    const lr = this.options.learningRate ?? 0.1;
    const fitIntercept = this.options.fitIntercept ?? true;

    const m = X.shape[0] ?? 0;
    const n = X.shape[1] ?? 0;
    const xData = toFloat64View(X);
    const yData = toFloat64View(y);

    // Identify unique classes and map every sample to its class index.
    const uniqueClasses = [...new Set(yData)].sort((a, b) => a - b);
    const nClasses = uniqueClasses.length;
    const classIndex = new Map<number, number>();
    uniqueClasses.forEach((label, idx) => {
      classIndex.set(label, idx);
    });
    const yIdx = new Int32Array(m);
    for (let i = 0; i < m; i++) {
      const idx = classIndex.get(yData[i] as number) as number;
      yIdx[i] = idx;
    }

    const cw = this.options.classWeight;
    /**
     * Sample weights from `classWeight` for a problem whose samples fall into
     * `groups` (class indices, or 0/1 for a one-vs-rest problem). 'balanced'
     * is `n_samples / (n_groups * n_samples_in_group)`; a dictionary maps the
     * original class labels (`labels`) to weights, with missing classes at 1.
     */
    const weightsFor = (
      groups: Int32Array,
      nGroups: number,
      labels: readonly number[]
    ): { sw: Float64Array | undefined; swSum: number } => {
      if (cw === undefined) return { sw: undefined, swSum: m };
      const counts = new Float64Array(nGroups);
      for (let i = 0; i < m; i++) {
        const g = groups[i] as number;
        counts[g] = (counts[g] as number) + 1;
      }
      const perGroup = new Float64Array(nGroups);
      for (let g = 0; g < nGroups; g++) {
        perGroup[g] =
          cw === "balanced"
            ? m / (nGroups * (counts[g] as number))
            : (cw[labels[g] as number] ?? 1);
      }
      const sw = new Float64Array(m);
      let swSum = 0;
      for (let i = 0; i < m; i++) {
        const wi = perGroup[groups[i] as number] as number;
        sw[i] = wi;
        swSum += wi;
      }
      if (!(swSum > 0)) {
        throw new DataValidationError("classWeight gives every training sample a weight of zero");
      }
      return { sw, swSum };
    };

    const useSoftmax =
      nClasses > 2 &&
      (multiClass === "multinomial" || (multiClass === "auto" && solver !== "liblinear"));
    if (nClasses > 2 && multiClass === "multinomial" && solver === "liblinear") {
      throw new InvalidParameterError(
        "solver='liblinear' does not support multiClass='multinomial'; use 'ovr', 'lbfgs' or 'saga'",
        "multiClass",
        multiClass
      );
    }

    const runSolver = (
      labels: Int32Array,
      nOut: number,
      weights: { sw: Float64Array | undefined; swSum: number }
    ): SolveResult => {
      // sklearn minimizes C * sum(w_i * loss_i) + penalty; dividing by
      // C * sum(w_i) gives the mean-loss form used here.
      const reg = penalty === "none" ? 0 : 1 / (C * weights.swSum);
      const l2 = penalty === "l2" ? reg : 0;
      const l1 = penalty === "l1" ? reg : 0;
      const objective = this.buildObjective(
        xData,
        labels,
        weights.sw,
        weights.swSum,
        m,
        n,
        nOut,
        fitIntercept,
        l2
      );
      const dim = nOut * n + (fitIntercept ? nOut : 0);
      const theta0 = new Float64Array(dim);
      const result =
        l1 > 0
          ? proximalGradient(objective, theta0, nOut * n, l1, lr, maxIter, tol)
          : lbfgs(objective, theta0, maxIter, tol);
      for (let k = 0; k < result.theta.length; k++) {
        if (!Number.isFinite(result.theta[k])) {
          throw new DataValidationError(
            "LogisticRegression solver produced non-finite parameters; scale the features or increase C regularization"
          );
        }
      }
      return result;
    };

    let converged = true;
    let nIter = 0;
    let coefArray: Float64Array;
    let coefTensor: Tensor;
    let intercept: number | number[];
    let multiclass: boolean;

    if (nClasses === 1) {
      // Degenerate problem: nothing to separate.
      multiclass = false;
      weightsFor(yIdx, 1, uniqueClasses); // rejects an all-zero weight vector
      coefArray = new Float64Array(n);
      coefTensor = tensor(coefArray, { dtype: "float64" });
      intercept = 0;
    } else if (nClasses === 2) {
      multiclass = false;
      const res = runSolver(yIdx, 1, weightsFor(yIdx, 2, uniqueClasses));
      converged = res.converged;
      nIter = res.nIter;
      coefArray = res.theta.slice(0, n);
      coefTensor = tensor(coefArray, { dtype: "float64" });
      intercept = fitIntercept ? (res.theta[n] as number) : 0;
    } else if (useSoftmax) {
      multiclass = true;
      const res = runSolver(yIdx, nClasses, weightsFor(yIdx, nClasses, uniqueClasses));
      converged = res.converged;
      nIter = res.nIter;
      coefArray = res.theta.slice(0, nClasses * n);
      coefTensor = tensor(coefArray, { dtype: "float64" }).reshape([nClasses, n]);
      const b: number[] = [];
      for (let k = 0; k < nClasses; k++) {
        b.push(fitIntercept ? (res.theta[nClasses * n + k] as number) : 0);
      }
      intercept = b;
    } else {
      // One-vs-rest
      multiclass = true;
      coefArray = new Float64Array(nClasses * n);
      const b: number[] = [];
      const yBin = new Int32Array(m);
      for (let k = 0; k < nClasses; k++) {
        for (let i = 0; i < m; i++) yBin[i] = yIdx[i] === k ? 1 : 0;
        // 'balanced' balances each binary problem; a dictionary weights the original classes.
        const weights =
          cw === "balanced"
            ? weightsFor(yBin, 2, [0, 1])
            : weightsFor(yIdx, nClasses, uniqueClasses);
        const res = runSolver(yBin, 1, weights);
        converged = converged && res.converged;
        nIter = Math.max(nIter, res.nIter);
        coefArray.set(res.theta.subarray(0, n), k * n);
        b.push(fitIntercept ? (res.theta[n] as number) : 0);
      }
      coefTensor = tensor(coefArray, { dtype: "float64" }).reshape([nClasses, n]);
      intercept = b;
    }

    if (!converged) {
      warn(
        `Solver did not converge within maxIter=${maxIter} iterations; ` +
          "increase maxIter, loosen tol, or scale the features",
        "ConvergenceWarning",
        "LogisticRegression"
      );
    }

    // Replace fitted state only after the whole fit succeeded.
    this.nFeaturesIn_ = n;
    this.classes_ = tensor(uniqueClasses, { dtype: "float64" });
    this.multiclass_ = multiclass;
    this.softmax_ = multiclass && useSoftmax;
    this.coefArray_ = coefArray;
    this.coef_ = coefTensor;
    this.intercept_ = intercept;
    this.nIter_ = nIter;
    this.fitted = true;
    return this;
  }

  /**
   * Class labels seen during `fit`, sorted ascending.
   */
  get classes(): Tensor | undefined {
    return this.classes_;
  }

  /**
   * Get the model coefficients (weights).
   *
   * @returns Coefficient tensor of shape (n_features,) for binary problems or
   * (n_classes, n_features) for more than two classes (dtype float64)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get coef(): Tensor {
    this.ensureFitted();
    const coef = this.coef_;
    if (!coef) {
      throw new DeepboxError("Internal error: coef_ is missing after ensureFitted() ");
    }
    return coef;
  }

  /**
   * Get the intercept (bias term).
   *
   * @returns Intercept value (scalar for binary, array for multiclass)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get intercept(): number | number[] {
    this.ensureFitted();
    if (this.intercept_ === undefined) {
      return 0;
    }
    return this.intercept_;
  }

  /**
   * Number of solver iterations of the last fit (the maximum over the binary
   * problems for one-vs-rest).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nIter(): number {
    this.ensureFitted();
    return this.nIter_;
  }

  /**
   * Raw scores `X @ coef + intercept` for each sample.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Shape (n_samples,) for binary problems (positive means `classes[1]`),
   * (n_samples, n_classes) for more than two classes
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  decisionFunction(X: Tensor): Tensor {
    const scores = this.scores(X);
    const m = X.shape[0] ?? 0;
    if (this.multiclass_) {
      const nClasses = this.classes_?.size ?? 0;
      return tensor(scores, { dtype: "float64" }).reshape([m, nClasses]);
    }
    return tensor(scores, { dtype: "float64" });
  }

  /** Raw scores, flat: (m) for binary / single class, (m * nClasses) for multiclass. */
  private scores(X: Tensor): Float64Array {
    this.ensureFitted();
    const coef = this.coefArray_;
    if (!coef) {
      throw new DeepboxError("Internal error: coef_ is missing after ensureFitted()");
    }
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "LogisticRegression");

    const m = X.shape[0] ?? 0;
    const n = X.shape[1] ?? 0;
    const xData = toFloat64View(X);
    const interceptValue = this.intercept_;

    if (this.multiclass_) {
      const nClasses = this.classes_?.size ?? 0;
      if (!Array.isArray(interceptValue)) {
        throw new DeepboxError("Internal error: intercept_ must be an array for multiclass");
      }
      const out = new Float64Array(m * nClasses);
      for (let i = 0; i < m; i++) {
        const base = i * n;
        for (let k = 0; k < nClasses; k++) {
          let z = interceptValue[k] ?? 0;
          const cBase = k * n;
          for (let j = 0; j < n; j++) {
            z += (xData[base + j] as number) * (coef[cBase + j] as number);
          }
          out[i * nClasses + k] = z;
        }
      }
      return out;
    }

    if (Array.isArray(interceptValue) || typeof interceptValue !== "number") {
      throw new DeepboxError("Internal error: intercept_ must be a number for binary case");
    }
    const out = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      const base = i * n;
      let z = interceptValue;
      for (let j = 0; j < n; j++) z += (xData[base + j] as number) * (coef[j] as number);
      out[i] = z;
    }
    return out;
  }

  /**
   * Predict class labels.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted labels of shape (n_samples,), dtype float64
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    const scores = this.scores(X);
    const m = X.shape[0] ?? 0;
    const classes = this.classes_;
    if (!classes) {
      throw new NotFittedError("Model not fitted (classes_ missing)");
    }
    const labels = toFloat64View(classes);
    const pred = new Float64Array(m);

    if (this.multiclass_) {
      const nClasses = labels.length;
      for (let i = 0; i < m; i++) {
        let best = -Infinity;
        let bestIdx = 0;
        for (let k = 0; k < nClasses; k++) {
          const s = scores[i * nClasses + k] as number;
          if (s > best) {
            best = s;
            bestIdx = k;
          }
        }
        pred[i] = labels[bestIdx] as number;
      }
    } else if (labels.length === 1) {
      pred.fill(labels[0] as number);
    } else {
      const cls0 = labels[0] as number;
      const cls1 = labels[1] as number;
      for (let i = 0; i < m; i++) pred[i] = (scores[i] as number) > 0 ? cls1 : cls0;
    }
    return tensor(pred, { dtype: "float64" });
  }

  /**
   * Predict class probabilities for samples.
   *
   * Binary problems use the logistic function, softmax models use softmax,
   * and one-vs-rest models normalize the per-class logistic probabilities to
   * sum to one.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Probabilities of shape (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predictProba(X: Tensor): Tensor {
    const scores = this.scores(X);
    const m = X.shape[0] ?? 0;

    if (this.multiclass_) {
      const nClasses = this.classes_?.size ?? 0;
      const proba = new Float64Array(m * nClasses);
      for (let i = 0; i < m; i++) {
        const base = i * nClasses;
        if (this.softmax_) {
          let maxScore = -Infinity;
          for (let k = 0; k < nClasses; k++)
            maxScore = Math.max(maxScore, scores[base + k] as number);
          let sum = 0;
          for (let k = 0; k < nClasses; k++) {
            const e = Math.exp((scores[base + k] as number) - maxScore);
            proba[base + k] = e;
            sum += e;
          }
          for (let k = 0; k < nClasses; k++) proba[base + k] = (proba[base + k] as number) / sum;
        } else {
          // One-vs-rest: logistic probability per class, normalized to sum to 1.
          let sum = 0;
          for (let k = 0; k < nClasses; k++) {
            const p = sigmoid(scores[base + k] as number);
            proba[base + k] = p;
            sum += p;
          }
          for (let k = 0; k < nClasses; k++) {
            proba[base + k] = sum > 0 ? (proba[base + k] as number) / sum : 1 / nClasses;
          }
        }
      }
      return tensor(proba, { dtype: "float64" }).reshape([m, nClasses]);
    }

    if ((this.classes_?.size ?? 0) === 1) {
      return tensor(new Float64Array(m).fill(1), { dtype: "float64" }).reshape([m, 1]);
    }
    const proba = new Float64Array(m * 2);
    for (let i = 0; i < m; i++) {
      const z = scores[i] as number;
      proba[i * 2] = sigmoid(-z);
      proba[i * 2 + 1] = sigmoid(z);
    }
    return tensor(proba, { dtype: "float64" }).reshape([m, 2]);
  }

  /**
   * Return the mean accuracy on the given test data and labels.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True labels of shape (n_samples,)
   * @returns Accuracy score in range [0, 1]
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional or sample counts mismatch
   * @throws {DataValidationError} If y is empty or contains NaN/Inf values
   */
  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    assertContiguous(y, "y");
    if (y.size === 0) {
      throw new DataValidationError("y must contain at least one sample");
    }
    for (let i = 0; i < y.size; i++) {
      const val = y.data[y.offset + i] ?? 0;
      if (typeof val === "number" && !Number.isFinite(val)) {
        throw new DataValidationError("y contains non-finite values (NaN or Inf)");
      }
    }
    const pred = this.predict(X);
    if (pred.size !== y.size) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${pred.size}, y=${y.size}`
      );
    }
    const predData = toFloat64View(pred);
    const yData = toFloat64View(y);
    let correct = 0;
    for (let i = 0; i < yData.length; i++) {
      if (predData[i] === yData[i]) correct++;
    }
    return correct / yData.length;
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * Returns every hyperparameter with its effective value (`classWeight` is
   * included only when set), so the result can be passed to the constructor.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    const params: Record<string, unknown> = {
      penalty: this.options.penalty ?? "l2",
      tol: this.options.tol ?? 1e-4,
      C: this.options.C ?? 1.0,
      fitIntercept: this.options.fitIntercept ?? true,
      maxIter: this.options.maxIter ?? 1000,
      learningRate: this.options.learningRate ?? 0.1,
      multiClass: this.options.multiClass ?? "auto",
      solver: this.options.solver ?? "lbfgs",
    };
    const cw = this.options.classWeight;
    if (cw !== undefined) {
      params["classWeight"] = typeof cw === "string" ? cw : { ...cw };
    }
    return params;
  }

  /**
   * Set the parameters of this estimator.
   *
   * Changing parameters does not alter a fitted model; call `fit` again. Unlike
   * the constructor, `setParams` does not reject `penalty: 'l1'` with
   * `solver: 'lbfgs'`; the L1 problem is then solved with the proximal gradient
   * method.
   *
   * @param params - Parameters to set (maxIter, tol, C, learningRate, penalty, fitIntercept, solver, multiClass, classWeight)
   * @returns this
   * @throws {InvalidParameterError} If any parameter value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "maxIter":
          if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
            throw new InvalidParameterError(
              "maxIter must be a positive finite number",
              "maxIter",
              value
            );
          }
          this.options.maxIter = value;
          break;
        case "tol":
          if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
            throw new InvalidParameterError("tol must be a finite number >= 0", "tol", value);
          }
          this.options.tol = value;
          break;
        case "C":
          if (typeof value !== "number" || Number.isNaN(value) || value <= 0) {
            throw new InvalidParameterError("C must be a positive number", "C", value);
          }
          this.options.C = value;
          break;
        case "learningRate":
          if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
            throw new InvalidParameterError(
              "learningRate must be a positive finite number",
              "learningRate",
              value
            );
          }
          this.options.learningRate = value;
          break;
        case "fitIntercept":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError(
              "fitIntercept must be a boolean",
              "fitIntercept",
              value
            );
          }
          this.options.fitIntercept = value;
          break;
        case "penalty":
          if (value !== "none" && value !== "l1" && value !== "l2") {
            throw new InvalidParameterError(
              `penalty must be 'l1', 'l2', or 'none'; received ${String(value)}`,
              "penalty",
              value
            );
          }
          this.options.penalty = value;
          break;
        case "solver":
          if (value !== "lbfgs" && value !== "liblinear" && value !== "saga") {
            throw new InvalidParameterError(
              `solver must be 'lbfgs', 'liblinear', or 'saga'; received ${String(value)}`,
              "solver",
              value
            );
          }
          this.options.solver = value;
          break;
        case "multiClass":
          if (value !== "ovr" && value !== "auto" && value !== "multinomial") {
            throw new InvalidParameterError(
              `multiClass must be 'ovr', 'multinomial' or 'auto'; received ${String(value)}`,
              "multiClass",
              value
            );
          }
          this.options.multiClass = value;
          break;
        case "classWeight":
          if (value === undefined) {
            delete this.options.classWeight;
          } else {
            LogisticRegression.assertClassWeight(value);
            this.options.classWeight =
              value === "balanced" ? "balanced" : { ...(value as Record<number, number>) };
          }
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  /**
   * Create an unfitted copy of this estimator with the same parameters.
   *
   * @returns A new LogisticRegression instance
   */
  clone(): LogisticRegression {
    return new LogisticRegression(
      this.getParams() as {
        penalty?: Penalty;
        tol?: number;
        C?: number;
        fitIntercept?: boolean;
        maxIter?: number;
        learningRate?: number;
        multiClass?: MultiClass;
        classWeight?: ClassWeight;
        solver?: Solver;
      }
    );
  }
}
