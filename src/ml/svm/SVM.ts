/**
 * Linear support vector machines: {@link LinearSVC} and {@link LinearSVR}.
 *
 * Both solve the L2-regularized problem in the dual with coordinate descent
 * (Hsieh et al. 2008, Ho and Lin 2012), the algorithm behind LIBLINEAR and scikit-learn's
 * `LinearSVC` and `LinearSVR`. The intercept is handled like LIBLINEAR does, as an extra
 * constant feature `interceptScaling` that is regularized together with the weights.
 *
 * @module ml/svm/SVM
 * @see {@link https://deepbox.dev/docs/ml-svm | Deepbox SVM}
 */

import { InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __random, __randomBelow } from "../../random/random";
import { toFloat64View, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier, Regressor } from "../base";
import {
  accuracyOf,
  type ClassWeightOption,
  copyClassWeight,
  encodeLabels,
  labelsToTensor,
  mergeParams,
  optionOr,
  parseC,
  parseClassWeight,
  parseEpsilon,
  parseMaxIter,
  parseTol,
  r2Of,
  readSampleWeight,
  resolveClassWeights,
  stableSigmoid,
  warnNotConverged,
} from "./KernelSVM";

/** Loss of {@link LinearSVC}: `"hinge"` or `"squaredHinge"`. */
export type LinearSVCLoss = "hinge" | "squaredHinge";

/** Loss of {@link LinearSVR}: `"epsilonInsensitive"` or `"squaredEpsilonInsensitive"`. */
export type LinearSVRLoss = "epsilonInsensitive" | "squaredEpsilonInsensitive";

function parseFitIntercept(value: unknown): boolean {
  if (typeof value !== "boolean") {
    throw new InvalidParameterError("fitIntercept must be a boolean", "fitIntercept", value);
  }
  return value;
}

function parseInterceptScaling(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new InvalidParameterError(
      "interceptScaling must be positive and finite",
      "interceptScaling",
      value
    );
  }
  return value;
}

function parseRandomState(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new InvalidParameterError("randomState must be a finite number", "randomState", value);
  }
  return value;
}

function parseSvcLoss(value: unknown): LinearSVCLoss {
  if (value !== "hinge" && value !== "squaredHinge") {
    throw new InvalidParameterError('loss must be "hinge" or "squaredHinge"', "loss", value);
  }
  return value;
}

function parseSvrLoss(value: unknown): LinearSVRLoss {
  if (value !== "epsilonInsensitive" && value !== "squaredEpsilonInsensitive") {
    throw new InvalidParameterError(
      'loss must be "epsilonInsensitive" or "squaredEpsilonInsensitive"',
      "loss",
      value
    );
  }
  return value;
}

/** Integer seeds keep their low 32 bits; fractional ones use their IEEE bits (0.5 and 0.7 differ). */
function seedToUint32(seed: number): number {
  if (Number.isInteger(seed)) return seed >>> 0;
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, seed);
  return (view.getUint32(0) ^ view.getUint32(4)) >>> 0;
}

/** Uniform [0, 1) generator: the global seeded one, or mulberry32 seeded with `seed`. */
function makeRng(seed: number | undefined): () => number {
  if (seed === undefined) return __random;
  let s = seedToUint32(seed);
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** In-place Fisher-Yates shuffle. */
function shuffle(order: Int32Array, rng: () => number): void {
  for (let i = order.length - 1; i > 0; i--) {
    const j = __randomBelow(rng, i + 1);
    const tmp = order[i] as number;
    order[i] = order[j] as number;
    order[j] = tmp;
  }
}

type LinearSolution = {
  readonly w: Float64Array;
  readonly bias: number;
  readonly iterations: number;
  readonly converged: boolean;
};

/**
 * Dual coordinate descent for the L2-regularized L1-loss and L2-loss SVM.
 *
 * Minimizes `0.5 ||w||^2 + sum_i C_i loss(y_i, w.x_i)` where the intercept, when `scale > 0`,
 * is the weight of an extra feature that always equals `scale`.
 */
function solveLinearSvc(
  X: Float64Array,
  n: number,
  d: number,
  y: Int8Array,
  C: Float64Array,
  squared: boolean,
  scale: number,
  maxIter: number,
  tol: number,
  rng: () => number
): LinearSolution {
  const w = new Float64Array(d);
  let wb = 0;
  const alpha = new Float64Array(n);
  const diag = new Float64Array(n);
  const upper = new Float64Array(n);
  const QD = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const ci = C[i] as number;
    diag[i] = squared ? 0.5 / ci : 0;
    upper[i] = squared ? Number.POSITIVE_INFINITY : ci;
    let sq = scale * scale;
    for (let k = 0; k < d; k++) {
      const v = X[i * d + k] as number;
      sq += v * v;
    }
    QD[i] = (diag[i] as number) + sq;
  }

  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  let converged = false;
  let iter = 0;
  while (iter < maxIter) {
    shuffle(order, rng);
    let pgMax = Number.NEGATIVE_INFINITY;
    let pgMin = Number.POSITIVE_INFINITY;
    for (let s = 0; s < n; s++) {
      const i = order[s] as number;
      if ((C[i] as number) === 0) continue;
      const yi = y[i] as number;
      const base = i * d;
      let wx = wb * scale;
      for (let k = 0; k < d; k++) wx += (w[k] as number) * (X[base + k] as number);
      const ai = alpha[i] as number;
      const G = yi * wx - 1 + ai * (diag[i] as number);
      let PG: number;
      if (ai === 0) PG = Math.min(G, 0);
      else if (ai >= (upper[i] as number)) PG = Math.max(G, 0);
      else PG = G;
      if (PG > pgMax) pgMax = PG;
      if (PG < pgMin) pgMin = PG;
      if (Math.abs(PG) > 1e-12) {
        const next = Math.min(Math.max(ai - G / (QD[i] as number), 0), upper[i] as number);
        alpha[i] = next;
        const step = (next - ai) * yi;
        for (let k = 0; k < d; k++) w[k] = (w[k] as number) + step * (X[base + k] as number);
        wb += step * scale;
      }
    }
    iter++;
    if (pgMax - pgMin <= tol) {
      converged = true;
      break;
    }
  }
  return { w, bias: wb * scale, iterations: iter, converged };
}

/**
 * Dual coordinate descent for L2-regularized epsilon-insensitive regression
 * (LIBLINEAR `solve_l2r_l1l2_svr`).
 */
function solveLinearSvr(
  X: Float64Array,
  n: number,
  d: number,
  y: Float64Array,
  C: number,
  epsilon: number,
  squared: boolean,
  scale: number,
  maxIter: number,
  tol: number,
  rng: () => number
): LinearSolution {
  const w = new Float64Array(d);
  let wb = 0;
  const beta = new Float64Array(n);
  const lambda = squared ? 0.5 / C : 0;
  const upper = squared ? Number.POSITIVE_INFINITY : C;
  const QD = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sq = scale * scale;
    for (let k = 0; k < d; k++) {
      const v = X[i * d + k] as number;
      sq += v * v;
    }
    QD[i] = sq;
  }

  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  let converged = false;
  let iter = 0;
  let gnormInit = 0;
  while (iter < maxIter) {
    shuffle(order, rng);
    let gmaxNew = 0;
    for (let s = 0; s < n; s++) {
      const i = order[s] as number;
      const base = i * d;
      const bi = beta[i] as number;
      const H = (QD[i] as number) + lambda;
      let wx = wb * scale;
      for (let k = 0; k < d; k++) wx += (w[k] as number) * (X[base + k] as number);
      const G = -(y[i] as number) + lambda * bi + wx;
      const Gp = G + epsilon;
      const Gn = G - epsilon;
      let violation = 0;
      if (bi === 0) {
        if (Gp < 0) violation = -Gp;
        else if (Gn > 0) violation = Gn;
      } else if (bi >= upper) {
        if (Gp > 0) violation = Gp;
      } else if (bi <= -upper) {
        if (Gn < 0) violation = -Gn;
      } else if (bi > 0) {
        violation = Math.abs(Gp);
      } else {
        violation = Math.abs(Gn);
      }
      if (violation > gmaxNew) gmaxNew = violation;

      let step: number;
      if (Gp < H * bi) step = -Gp / H;
      else if (Gn > H * bi) step = -Gn / H;
      else step = -bi;
      if (Math.abs(step) < 1e-12) continue;
      const next = Math.min(Math.max(bi + step, -upper), upper);
      step = next - bi;
      if (step !== 0) {
        beta[i] = next;
        for (let k = 0; k < d; k++) w[k] = (w[k] as number) + step * (X[base + k] as number);
        wb += step * scale;
      }
    }
    if (iter === 0) gnormInit = gmaxNew;
    iter++;
    if (gmaxNew <= tol * gnormInit) {
      converged = true;
      break;
    }
  }
  return { w, bias: wb * scale, iterations: iter, converged };
}

// ---------------------------------------------------------------------------
// LinearSVC
// ---------------------------------------------------------------------------

/** Constructor options of {@link LinearSVC}. */
export type LinearSVCOptions = {
  /** Penalty of margin violations, must be positive (default: 1.0). */
  readonly C?: number;
  /** Loss function (default: "hinge"). scikit-learn defaults to the squared hinge. */
  readonly loss?: LinearSVCLoss;
  /** Fit an intercept (default: true). */
  readonly fitIntercept?: boolean;
  /**
   * Value of the constant feature that carries the intercept (default: 1). The intercept is
   * regularized, so larger values weaken the penalty on it.
   */
  readonly interceptScaling?: number;
  /** Maximum number of passes over the data (default: 1000). */
  readonly maxIter?: number;
  /** Stopping tolerance on the projected-gradient range (default: 1e-4). */
  readonly tol?: number;
  /** Per-class multiplier of `C`: "balanced" or a map from class label to weight. */
  readonly classWeight?: ClassWeightOption;
  /**
   * Seed of the coordinate order. When omitted the global generator is used, so `setSeed`
   * makes fits reproducible.
   */
  readonly randomState?: number;
};

type LinearSvcConfig = {
  C: number;
  loss: LinearSVCLoss;
  fitIntercept: boolean;
  interceptScaling: number;
  maxIter: number;
  tol: number;
  classWeight: ClassWeightOption | undefined;
  randomState: number | undefined;
};

const LINEAR_SVC_KEYS = [
  "C",
  "loss",
  "fitIntercept",
  "interceptScaling",
  "maxIter",
  "tol",
  "classWeight",
  "randomState",
] as const;

function normalizeLinearSvcConfig(o: Record<string, unknown>): LinearSvcConfig {
  return {
    C: optionOr(o, "C", 1.0, parseC),
    loss: optionOr<LinearSVCLoss>(o, "loss", "hinge", parseSvcLoss),
    fitIntercept: optionOr(o, "fitIntercept", true, parseFitIntercept),
    interceptScaling: optionOr(o, "interceptScaling", 1, parseInterceptScaling),
    maxIter: optionOr(o, "maxIter", 1000, parseMaxIter),
    tol: optionOr(o, "tol", 1e-4, parseTol),
    classWeight: o["classWeight"] === undefined ? undefined : parseClassWeight(o["classWeight"]),
    randomState: o["randomState"] === undefined ? undefined : parseRandomState(o["randomState"]),
  };
}

type LinearSvcModel = {
  readonly labels: Float64Array;
  readonly nFeatures: number;
  /** One weight vector for two classes, one per class otherwise (row-major). */
  readonly weights: Float64Array;
  readonly biases: Float64Array;
  readonly nModels: number;
  readonly nIter: number;
};

/**
 * Support Vector Machine classifier with a linear kernel.
 *
 * Minimizes `0.5 ||w||^2 + C * sum_i loss(y_i, w.x_i + b)` with dual coordinate descent
 * (the LIBLINEAR algorithm). More than two classes are handled one-vs-rest.
 *
 * `predictProba` returns a monotone squashing of the decision values, not calibrated
 * probabilities.
 *
 * @example
 * ```ts
 * import { LinearSVC } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [2, 3], [3, 1], [4, 2]]);
 * const y = tensor([0, 0, 1, 1]);
 *
 * const svm = new LinearSVC({ C: 1.0 });
 * svm.fit(X, y);
 * const predictions = svm.predict(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-svm | Deepbox SVM}
 */
export class LinearSVC implements Classifier {
  private cfg: LinearSvcConfig;
  private model_: LinearSvcModel | undefined;

  /**
   * @param options - Hyperparameters, see {@link LinearSVCOptions}
   * @throws {InvalidParameterError} If an option is out of range
   */
  constructor(options: LinearSVCOptions = {}) {
    this.cfg = normalizeLinearSvcConfig(options as Record<string, unknown>);
  }

  private get fitted(): LinearSvcModel {
    if (this.model_ === undefined) {
      throw new NotFittedError("LinearSVC must be fitted before prediction");
    }
    return this.model_;
  }

  /**
   * Fit the classifier.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Class labels of shape (n_samples,), at least two distinct values
   * @param sampleWeight - Optional per-sample multipliers of `C`, shape (n_samples,)
   * @returns this
   * @throws {ShapeError} If X is not 2D, y is not 1D or the sample counts differ
   * @throws {DataValidationError} If X or y contain NaN/Inf
   * @throws {InvalidParameterError} If y has fewer than 2 classes
   */
  // biome-ignore lint/suspicious/noConfusingVoidType: `void` keeps this compatible with Estimator.fit(X, y, params)
  fit(X: Tensor, y: Tensor, sampleWeightArg?: Tensor | void): this {
    const sampleWeight = sampleWeightArg as Tensor | undefined;
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const Xf = toFloat64View(X);
    const { labels, index } = encodeLabels(toFloat64View(y));
    const K = labels.length;
    if (K < 2) {
      throw new InvalidParameterError("LinearSVC requires at least 2 classes", "y", K);
    }
    const sw = readSampleWeight(sampleWeight, n);
    const cw = resolveClassWeights(this.cfg.classWeight, labels, index);
    const scale = this.cfg.fitIntercept ? this.cfg.interceptScaling : 0;
    const rng = makeRng(this.cfg.randomState);
    const squared = this.cfg.loss === "squaredHinge";

    const nModels = K === 2 ? 1 : K;
    const weights = new Float64Array(nModels * d);
    const biases = new Float64Array(nModels);
    let converged = true;
    let nIter = 0;
    const ySub = new Int8Array(n);
    const C = new Float64Array(n);
    for (let m = 0; m < nModels; m++) {
      for (let i = 0; i < n; i++) {
        const c = index[i] as number;
        const swi = sw ? (sw[i] as number) : 1;
        if (K === 2) {
          ySub[i] = c === 1 ? 1 : -1;
          C[i] = this.cfg.C * (cw[c] as number) * swi;
        } else {
          // One-vs-rest, as in LIBLINEAR: only the positive class carries its class weight.
          ySub[i] = c === m ? 1 : -1;
          C[i] = this.cfg.C * (c === m ? (cw[c] as number) : 1) * swi;
        }
      }
      const sol = solveLinearSvc(
        Xf,
        n,
        d,
        ySub,
        C,
        squared,
        scale,
        this.cfg.maxIter,
        this.cfg.tol,
        rng
      );
      converged = converged && sol.converged;
      nIter = Math.max(nIter, sol.iterations);
      weights.set(sol.w, m * d);
      biases[m] = sol.bias;
    }
    if (!converged) warnNotConverged("LinearSVC", this.cfg.maxIter);

    this.model_ = { labels, nFeatures: d, weights, biases, nModels, nIter };
    return this;
  }

  /** Raw scores, row-major `(n, nModels)`. */
  private scores(X: Tensor): { scores: Float64Array; n: number } {
    const model = this.fitted;
    validatePredictInputs(X, model.nFeatures, "LinearSVC");
    const n = X.shape[0] ?? 0;
    const d = model.nFeatures;
    const Xf = toFloat64View(X);
    const out = new Float64Array(n * model.nModels);
    for (let i = 0; i < n; i++) {
      for (let m = 0; m < model.nModels; m++) {
        let s = model.biases[m] as number;
        for (let k = 0; k < d; k++) {
          s += (model.weights[m * d + k] as number) * (Xf[i * d + k] as number);
        }
        out[i * model.nModels + m] = s;
      }
    }
    return { scores: out, n };
  }

  /**
   * Predict class labels.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Labels of shape (n_samples,): int32 for integer classes, float64 otherwise
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    const { scores, n } = this.scores(X);
    const model = this.fitted;
    const out = new Float64Array(n);
    if (model.nModels === 1) {
      for (let i = 0; i < n; i++) {
        out[i] = model.labels[(scores[i] as number) > 0 ? 1 : 0] as number;
      }
    } else {
      const K = model.nModels;
      for (let i = 0; i < n; i++) {
        let best = 0;
        for (let c = 1; c < K; c++) {
          if ((scores[i * K + c] as number) > (scores[i * K + best] as number)) best = c;
        }
        out[i] = model.labels[best] as number;
      }
    }
    return labelsToTensor(out);
  }

  /**
   * Signed distances `X @ coef.T + intercept` to the separating hyperplanes.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Shape (n_samples,) for two classes (positive means `classes[1]`), otherwise
   * (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   */
  decisionFunction(X: Tensor): Tensor {
    const { scores, n } = this.scores(X);
    const model = this.fitted;
    if (model.nModels === 1) return tensor(scores, { dtype: "float64" });
    return tensor(scores, { dtype: "float64" }).reshape([n, model.nModels]);
  }

  /**
   * Class scores squashed into rows that sum to one: the logistic function of the decision
   * value for two classes, the per-class logistic values normalized to sum to one otherwise.
   * These are not calibrated probabilities.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Probability-like scores of shape (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predictProba(X: Tensor): Tensor {
    const { scores, n } = this.scores(X);
    const model = this.fitted;
    if (model.nModels === 1) {
      const out = new Float64Array(n * 2);
      for (let i = 0; i < n; i++) {
        const p1 = stableSigmoid(scores[i] as number);
        out[i * 2] = 1 - p1;
        out[i * 2 + 1] = p1;
      }
      return tensor(out, { dtype: "float64" }).reshape([n, 2]);
    }
    const K = model.nModels;
    const out = new Float64Array(n * K);
    for (let i = 0; i < n; i++) {
      let total = 0;
      for (let c = 0; c < K; c++) {
        const p = stableSigmoid(scores[i * K + c] as number);
        out[i * K + c] = p;
        total += p;
      }
      for (let c = 0; c < K; c++) {
        out[i * K + c] = total > 0 ? (out[i * K + c] as number) / total : 1 / K;
      }
    }
    return tensor(out, { dtype: "float64" }).reshape([n, K]);
  }

  /**
   * Mean accuracy on the given data.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True labels of shape (n_samples,)
   * @returns Accuracy in [0, 1]
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional or the sample counts differ
   * @throws {DataValidationError} If y is empty or contains NaN/Inf
   */
  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    return accuracyOf(this.predict(X), y);
  }

  /** Sorted class labels seen during `fit`, or `undefined` before fitting. */
  get classes(): Tensor | undefined {
    return this.model_ === undefined ? undefined : labelsToTensor(this.model_.labels);
  }

  /**
   * Weight vectors, shape (1, n_features) for two classes (positive means `classes[1]`) and
   * (n_classes, n_features) otherwise.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get coef(): Tensor {
    const model = this.fitted;
    return tensor(Float64Array.from(model.weights), { dtype: "float64" }).reshape([
      model.nModels,
      model.nFeatures,
    ]);
  }

  /**
   * Intercepts, shape (1,) for two classes and (n_classes,) otherwise.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get intercept(): Tensor {
    return tensor(Float64Array.from(this.fitted.biases), { dtype: "float64" });
  }

  /**
   * Number of passes over the data used by the slowest binary problem.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nIter(): number {
    return this.fitted.nIter;
  }

  /**
   * Get hyperparameters, including `classWeight` and `randomState` only when set.
   *
   * @returns Object that can be passed back to the constructor
   */
  getParams(): Record<string, unknown> {
    const { classWeight, randomState, ...rest } = this.cfg;
    const out: Record<string, unknown> = { ...rest };
    if (classWeight !== undefined) out["classWeight"] = copyClassWeight(classWeight);
    if (randomState !== undefined) out["randomState"] = randomState;
    return out;
  }

  /**
   * Set hyperparameters. The call is atomic: if any value is invalid nothing changes.
   *
   * @param params - Parameters to set
   * @throws {InvalidParameterError} If a parameter is unknown or its value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    this.cfg = normalizeLinearSvcConfig(
      mergeParams(this.getParams(), params, LINEAR_SVC_KEYS, ["classWeight", "randomState"])
    );
    return this;
  }

  /** Create an unfitted copy with the same hyperparameters. */
  clone(): LinearSVC {
    return new LinearSVC(this.getParams() as LinearSVCOptions);
  }
}

// ---------------------------------------------------------------------------
// LinearSVR
// ---------------------------------------------------------------------------

/** Constructor options of {@link LinearSVR}. */
export type LinearSVROptions = {
  /** Penalty of errors outside the epsilon tube, must be positive (default: 1.0). */
  readonly C?: number;
  /**
   * Half-width of the tube in which errors are not penalized (default: 0.1). scikit-learn
   * defaults to 0.
   */
  readonly epsilon?: number;
  /** Loss function (default: "epsilonInsensitive"). */
  readonly loss?: LinearSVRLoss;
  /** Fit an intercept (default: true). */
  readonly fitIntercept?: boolean;
  /**
   * Value of the constant feature that carries the intercept (default: 1). The intercept is
   * regularized, so larger values weaken the penalty on it.
   */
  readonly interceptScaling?: number;
  /** Maximum number of passes over the data (default: 1000). */
  readonly maxIter?: number;
  /** Stopping tolerance relative to the violation of the first pass (default: 1e-4). */
  readonly tol?: number;
  /**
   * Seed of the coordinate order. When omitted the global generator is used, so `setSeed`
   * makes fits reproducible.
   */
  readonly randomState?: number;
};

type LinearSvrConfig = {
  C: number;
  epsilon: number;
  loss: LinearSVRLoss;
  fitIntercept: boolean;
  interceptScaling: number;
  maxIter: number;
  tol: number;
  randomState: number | undefined;
};

const LINEAR_SVR_KEYS = [
  "C",
  "epsilon",
  "loss",
  "fitIntercept",
  "interceptScaling",
  "maxIter",
  "tol",
  "randomState",
] as const;

function parseSvrMaxIter(value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new InvalidParameterError("maxIter must be positive (an integer >= 1)", "maxIter", value);
  }
  return value;
}

function normalizeLinearSvrConfig(o: Record<string, unknown>): LinearSvrConfig {
  return {
    C: optionOr(o, "C", 1.0, parseC),
    epsilon: optionOr(o, "epsilon", 0.1, parseEpsilon),
    loss: optionOr<LinearSVRLoss>(o, "loss", "epsilonInsensitive", parseSvrLoss),
    fitIntercept: optionOr(o, "fitIntercept", true, parseFitIntercept),
    interceptScaling: optionOr(o, "interceptScaling", 1, parseInterceptScaling),
    maxIter: optionOr(o, "maxIter", 1000, parseSvrMaxIter),
    tol: optionOr(o, "tol", 1e-4, parseTol),
    randomState: o["randomState"] === undefined ? undefined : parseRandomState(o["randomState"]),
  };
}

/**
 * Support Vector Regression with a linear kernel.
 *
 * Minimizes `0.5 ||w||^2 + C * sum_i loss(|y_i - w.x_i - b| - epsilon)` with dual coordinate
 * descent (the LIBLINEAR algorithm).
 *
 * @example
 * ```ts
 * import { LinearSVR } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1], [2], [3], [4]]);
 * const y = tensor([1.5, 2.5, 3.5, 4.5]);
 *
 * const svr = new LinearSVR({ C: 1.0, epsilon: 0.1 });
 * svr.fit(X, y);
 * const predictions = svr.predict(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-svm | Deepbox SVM}
 */
export class LinearSVR implements Regressor {
  private cfg: LinearSvrConfig;
  private weights_: Float64Array | undefined;
  private bias_ = 0;
  private nFeatures_ = 0;
  private nIter_ = 0;

  /**
   * @param options - Hyperparameters, see {@link LinearSVROptions}
   * @throws {InvalidParameterError} If an option is out of range
   */
  constructor(options: LinearSVROptions = {}) {
    this.cfg = normalizeLinearSvrConfig(options as Record<string, unknown>);
  }

  private get weights(): Float64Array {
    if (this.weights_ === undefined) {
      throw new NotFittedError("LinearSVR must be fitted before prediction");
    }
    return this.weights_;
  }

  /**
   * Fit the regressor.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Targets of shape (n_samples,)
   * @returns this
   * @throws {ShapeError} If X is not 2D, y is not 1D or the sample counts differ
   * @throws {DataValidationError} If X or y contain NaN/Inf
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;
    const sol = solveLinearSvr(
      toFloat64View(X),
      n,
      d,
      toFloat64View(y),
      this.cfg.C,
      this.cfg.epsilon,
      this.cfg.loss === "squaredEpsilonInsensitive",
      this.cfg.fitIntercept ? this.cfg.interceptScaling : 0,
      this.cfg.maxIter,
      this.cfg.tol,
      makeRng(this.cfg.randomState)
    );
    if (!sol.converged) warnNotConverged("LinearSVR", this.cfg.maxIter);

    this.weights_ = sol.w;
    this.bias_ = sol.bias;
    this.nFeatures_ = d;
    this.nIter_ = sol.iterations;
    return this;
  }

  /**
   * Predict target values.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predictions of shape (n_samples,), dtype float64
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    const w = this.weights;
    validatePredictInputs(X, this.nFeatures_, "LinearSVR");
    const n = X.shape[0] ?? 0;
    const d = this.nFeatures_;
    const Xf = toFloat64View(X);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = this.bias_;
      for (let k = 0; k < d; k++) s += (w[k] as number) * (Xf[i * d + k] as number);
      out[i] = s;
    }
    return tensor(out, { dtype: "float64" });
  }

  /**
   * Coefficient of determination R^2 on the given data.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True targets of shape (n_samples,)
   * @returns R^2 (1 is perfect, can be negative); a constant y scores 1 or 0
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional or the sample counts differ
   * @throws {DataValidationError} If y is empty or contains NaN/Inf
   */
  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    return r2Of(this.predict(X), y);
  }

  /**
   * Weight vector, shape (n_features,).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get coef(): Tensor {
    return tensor(Float64Array.from(this.weights), { dtype: "float64" });
  }

  /**
   * Intercept, shape (1,).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get intercept(): Tensor {
    void this.weights;
    return tensor(Float64Array.of(this.bias_), { dtype: "float64" });
  }

  /**
   * Number of passes over the data used by the last fit.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nIter(): number {
    void this.weights;
    return this.nIter_;
  }

  /**
   * Get hyperparameters, including `randomState` only when set.
   *
   * @returns Object that can be passed back to the constructor
   */
  getParams(): Record<string, unknown> {
    const { randomState, ...rest } = this.cfg;
    const out: Record<string, unknown> = { ...rest };
    if (randomState !== undefined) out["randomState"] = randomState;
    return out;
  }

  /**
   * Set hyperparameters. The call is atomic: if any value is invalid nothing changes.
   *
   * @param params - Parameters to set
   * @throws {InvalidParameterError} If a parameter is unknown or its value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    this.cfg = normalizeLinearSvrConfig(
      mergeParams(this.getParams(), params, LINEAR_SVR_KEYS, ["randomState"])
    );
    return this;
  }

  /** Create an unfitted copy with the same hyperparameters. */
  clone(): LinearSVR {
    return new LinearSVR(this.getParams() as LinearSVROptions);
  }
}
