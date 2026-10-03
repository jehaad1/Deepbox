/**
 * Stochastic Gradient Descent (SGD) linear models.
 *
 * SGDClassifier and SGDRegressor implement regularized linear models
 * with SGD optimization. Processes one sample at a time, making them
 * suitable for large-scale datasets. The update rules, learning rate
 * schedules, stopping rule and penalty handling follow scikit-learn's
 * `SGDClassifier` and `SGDRegressor`.
 *
 * @module ml/linear/SGD
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import {
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
  warn,
} from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __randomBelow } from "../../random/random";
import {
  assertContiguous,
  toFloat64View,
  validateFitInputs,
  validatePredictInputs,
} from "../_validation";
import type { Classifier, Regressor } from "../base";
import { createTreeRng } from "../tree/DecisionTree";
import { r2ScoreOf } from "./LinearRegression";

type Loss = "hinge" | "log_loss" | "modified_huber" | "squared_hinge";
type RegressionLoss = "squared_error" | "huber" | "epsilon_insensitive";
type Penalty = "l1" | "l2" | "elasticnet" | "none";
type LearningRateSchedule = "constant" | "optimal" | "invscaling" | "adaptive";
type ClassWeight = "balanced" | Record<number, number>;

interface SGDBaseOptions {
  readonly penalty?: Penalty;
  readonly alpha?: number;
  readonly l1Ratio?: number;
  readonly fitIntercept?: boolean;
  readonly maxIter?: number;
  readonly tol?: number;
  readonly shuffle?: boolean;
  readonly randomState?: number;
  readonly learningRate?: LearningRateSchedule;
  readonly eta0?: number;
  readonly powerT?: number;
  readonly nIterNoChange?: number;
}

const CLASSIFIER_LOSSES: readonly Loss[] = ["hinge", "log_loss", "modified_huber", "squared_hinge"];
const REGRESSION_LOSSES: readonly RegressionLoss[] = [
  "squared_error",
  "huber",
  "epsilon_insensitive",
];
const PENALTIES: readonly Penalty[] = ["l1", "l2", "elasticnet", "none"];
const SCHEDULES: readonly LearningRateSchedule[] = [
  "constant",
  "optimal",
  "invscaling",
  "adaptive",
];

/** Gradients are clipped to this magnitude, as in scikit-learn. */
const MAX_DLOSS = 1e12;

/** Hyperparameters shared by the classifier and the regressor. */
interface SGDParams {
  loss: string;
  penalty: Penalty;
  alpha: number;
  l1Ratio: number;
  fitIntercept: boolean;
  maxIter: number;
  tol: number;
  shuffle: boolean;
  randomState: number | undefined;
  learningRate: LearningRateSchedule;
  eta0: number;
  powerT: number;
  nIterNoChange: number;
  epsilon: number;
}

/** Loss value and derivative with respect to the prediction. */
interface LossFn {
  loss(p: number, y: number): number;
  dloss(p: number, y: number): number;
}

function classificationLoss(loss: Loss): LossFn {
  switch (loss) {
    case "hinge":
      return {
        loss: (p, y) => Math.max(0, 1 - y * p),
        dloss: (p, y) => (y * p <= 1 ? -y : 0),
      };
    case "squared_hinge":
      return {
        loss: (p, y) => {
          const h = Math.max(0, 1 - y * p);
          return h * h;
        },
        dloss: (p, y) => {
          const z = y * p;
          return z <= 1 ? -2 * y * (1 - z) : 0;
        },
      };
    case "modified_huber":
      return {
        loss: (p, y) => {
          const z = y * p;
          if (z >= 1) return 0;
          if (z >= -1) return (1 - z) * (1 - z);
          return -4 * z;
        },
        dloss: (p, y) => {
          const z = y * p;
          if (z >= 1) return 0;
          if (z >= -1) return -2 * y * (1 - z);
          return -4 * y;
        },
      };
    case "log_loss":
      return {
        // log(1 + exp(-z)) without overflow
        loss: (p, y) => {
          const z = y * p;
          return z > 0 ? Math.log1p(Math.exp(-z)) : -z + Math.log1p(Math.exp(z));
        },
        dloss: (p, y) => {
          const z = y * p;
          return z > 0 ? (-y * Math.exp(-z)) / (1 + Math.exp(-z)) : -y / (1 + Math.exp(z));
        },
      };
  }
}

function regressionLoss(loss: RegressionLoss, epsilon: number): LossFn {
  switch (loss) {
    case "squared_error":
      return {
        loss: (p, y) => 0.5 * (p - y) * (p - y),
        dloss: (p, y) => p - y,
      };
    case "huber":
      return {
        loss: (p, y) => {
          const r = Math.abs(p - y);
          return r <= epsilon ? 0.5 * r * r : epsilon * r - 0.5 * epsilon * epsilon;
        },
        dloss: (p, y) => {
          const r = p - y;
          return Math.abs(r) <= epsilon ? r : epsilon * Math.sign(r);
        },
      };
    case "epsilon_insensitive":
      return {
        loss: (p, y) => Math.max(0, Math.abs(p - y) - epsilon),
        dloss: (p, y) => {
          const r = p - y;
          return Math.abs(r) <= epsilon ? 0 : Math.sign(r);
        },
      };
  }
}

function shuffleIndices(indices: Int32Array, rng: () => number): void {
  for (let i = indices.length - 1; i > 0; i--) {
    const j = __randomBelow(rng, i + 1);
    const tmp = indices[i] as number;
    indices[i] = indices[j] as number;
    indices[j] = tmp;
  }
}

function defaultParams(
  options: SGDBaseOptions & { readonly epsilon?: number },
  loss: string,
  learningRate: LearningRateSchedule,
  powerT: number
): SGDParams {
  return {
    loss,
    penalty: options.penalty ?? "l2",
    alpha: options.alpha ?? 0.0001,
    l1Ratio: options.l1Ratio ?? 0.15,
    fitIntercept: options.fitIntercept ?? true,
    maxIter: options.maxIter ?? 1000,
    tol: options.tol ?? 1e-3,
    shuffle: options.shuffle ?? true,
    randomState: options.randomState,
    learningRate: options.learningRate ?? learningRate,
    eta0: options.eta0 ?? 0.01,
    powerT: options.powerT ?? powerT,
    nIterNoChange: options.nIterNoChange ?? 5,
    epsilon: options.epsilon ?? 0.1,
  };
}

function validateParams(p: SGDParams, losses: readonly string[]): void {
  if (!losses.includes(p.loss)) {
    throw new InvalidParameterError(
      `loss must be one of ${losses.map((l) => `'${l}'`).join(", ")}; received ${String(p.loss)}`,
      "loss",
      p.loss
    );
  }
  if (!PENALTIES.includes(p.penalty)) {
    throw new InvalidParameterError(
      `penalty must be one of 'l1', 'l2', 'elasticnet', 'none'; received ${String(p.penalty)}`,
      "penalty",
      p.penalty
    );
  }
  if (typeof p.alpha !== "number" || !Number.isFinite(p.alpha) || p.alpha < 0) {
    throw new InvalidParameterError(
      `alpha must be a finite number >= 0; received ${String(p.alpha)}`,
      "alpha",
      p.alpha
    );
  }
  if (typeof p.l1Ratio !== "number" || !(p.l1Ratio >= 0 && p.l1Ratio <= 1)) {
    throw new InvalidParameterError(
      `l1Ratio must be in [0, 1]; received ${String(p.l1Ratio)}`,
      "l1Ratio",
      p.l1Ratio
    );
  }
  if (typeof p.fitIntercept !== "boolean") {
    throw new InvalidParameterError(
      `fitIntercept must be a boolean; received ${String(p.fitIntercept)}`,
      "fitIntercept",
      p.fitIntercept
    );
  }
  if (!Number.isInteger(p.maxIter) || p.maxIter < 1) {
    throw new InvalidParameterError("maxIter must be a positive integer", "maxIter", p.maxIter);
  }
  if (typeof p.tol !== "number" || !Number.isFinite(p.tol) || p.tol < 0) {
    throw new InvalidParameterError(
      `tol must be a finite number >= 0; received ${String(p.tol)}`,
      "tol",
      p.tol
    );
  }
  if (typeof p.shuffle !== "boolean") {
    throw new InvalidParameterError(
      `shuffle must be a boolean; received ${String(p.shuffle)}`,
      "shuffle",
      p.shuffle
    );
  }
  if (p.randomState !== undefined && !Number.isFinite(p.randomState)) {
    throw new InvalidParameterError(
      `randomState must be a finite number; received ${String(p.randomState)}`,
      "randomState",
      p.randomState
    );
  }
  if (!SCHEDULES.includes(p.learningRate)) {
    throw new InvalidParameterError(
      `learningRate must be one of 'constant', 'optimal', 'invscaling', 'adaptive'; received ${String(p.learningRate)}`,
      "learningRate",
      p.learningRate
    );
  }
  if (typeof p.eta0 !== "number" || !Number.isFinite(p.eta0) || p.eta0 < 0) {
    throw new InvalidParameterError(
      `eta0 must be a finite number >= 0; received ${String(p.eta0)}`,
      "eta0",
      p.eta0
    );
  }
  if (p.learningRate !== "optimal" && !(p.eta0 > 0)) {
    throw new InvalidParameterError(
      `eta0 must be > 0 when learningRate is '${p.learningRate}'; received ${p.eta0}`,
      "eta0",
      p.eta0
    );
  }
  if (p.learningRate === "optimal" && !(p.alpha > 0)) {
    throw new InvalidParameterError("learningRate 'optimal' requires alpha > 0", "alpha", p.alpha);
  }
  if (typeof p.powerT !== "number" || !Number.isFinite(p.powerT)) {
    throw new InvalidParameterError(
      `powerT must be a finite number; received ${String(p.powerT)}`,
      "powerT",
      p.powerT
    );
  }
  if (!Number.isInteger(p.nIterNoChange) || p.nIterNoChange < 1) {
    throw new InvalidParameterError(
      `nIterNoChange must be a positive integer; received ${String(p.nIterNoChange)}`,
      "nIterNoChange",
      p.nIterNoChange
    );
  }
  if (typeof p.epsilon !== "number" || !Number.isFinite(p.epsilon) || p.epsilon < 0) {
    throw new InvalidParameterError(
      `epsilon must be a finite number >= 0; received ${String(p.epsilon)}`,
      "epsilon",
      p.epsilon
    );
  }
}

/** Apply `values` to `p` after checking the types of the fields shared by both estimators. */
function applySharedParams(
  p: SGDParams,
  params: Record<string, unknown>,
  extraKeys: readonly string[]
): SGDParams {
  const next: SGDParams = { ...p };
  const numberKeys = [
    "alpha",
    "l1Ratio",
    "maxIter",
    "tol",
    "eta0",
    "powerT",
    "nIterNoChange",
    "epsilon",
  ];
  for (const [key, value] of Object.entries(params)) {
    if (numberKeys.includes(key) && extraKeys.includes(key)) {
      if (typeof value !== "number") {
        throw new InvalidParameterError(
          `${key} must be a number; received ${String(value)}`,
          key,
          value
        );
      }
      (next as unknown as Record<string, unknown>)[key] = value;
    } else if (key === "fitIntercept" || key === "shuffle") {
      if (typeof value !== "boolean") {
        throw new InvalidParameterError(
          `${key} must be a boolean; received ${String(value)}`,
          key,
          value
        );
      }
      next[key] = value;
    } else if (key === "loss" || key === "penalty" || key === "learningRate") {
      if (typeof value !== "string") {
        throw new InvalidParameterError(
          `${key} must be a string; received ${String(value)}`,
          key,
          value
        );
      }
      (next as unknown as Record<string, unknown>)[key] = value;
    } else if (key === "randomState") {
      if (value !== undefined && typeof value !== "number") {
        throw new InvalidParameterError(
          `randomState must be a number or undefined; received ${String(value)}`,
          "randomState",
          value
        );
      }
      next.randomState = value as number | undefined;
    } else if (!extraKeys.includes(key)) {
      throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
    }
  }
  return next;
}

/** State of one binary linear model trained by SGD. */
interface SGDState {
  w: Float64Array;
  b: number;
  /** 1-based update counter, as in scikit-learn. */
  t: number;
  /** Current step size for the 'adaptive' schedule. */
  eta: number;
  /** Cumulative L1 penalty actually applied to each weight. */
  q: Float64Array;
  /** Cumulative L1 penalty each weight could have received. */
  u: number;
  nIter: number;
}

function newState(nFeatures: number, eta0: number): SGDState {
  return {
    w: new Float64Array(nFeatures),
    b: 0,
    t: 1,
    eta: eta0,
    q: new Float64Array(nFeatures),
    u: 0,
    nIter: 0,
  };
}

/** Initial offset of the 'optimal' schedule: `1 / (eta_init * alpha)`. */
function optimalInit(p: SGDParams, lossFn: LossFn): number {
  const typw = Math.sqrt(1 / Math.sqrt(p.alpha));
  const etaInit = typw / Math.max(1, lossFn.dloss(-typw, 1));
  return 1 / (etaInit * p.alpha);
}

function stepSize(p: SGDParams, state: SGDState, init: number): number {
  switch (p.learningRate) {
    case "constant":
      return p.eta0;
    case "optimal":
      return 1 / (p.alpha * (init + state.t - 1));
    case "invscaling":
      return p.eta0 / state.t ** p.powerT;
    case "adaptive":
      return state.eta;
  }
}

/**
 * One pass over the samples in `order`. Returns the summed training objective:
 * for every sample the (unweighted) loss plus the penalty term, both evaluated
 * at the weights before that sample's update, as in scikit-learn.
 *
 * Per sample: compute the prediction, take the loss gradient step, shrink the
 * weights for the L2 part of the penalty, and apply the cumulative-penalty
 * truncation (Tsuruoka et al., 2009) for the L1 part.
 */
function sgdEpoch(
  state: SGDState,
  p: SGDParams,
  lossFn: LossFn,
  init: number,
  x: Float64Array,
  y: Float64Array,
  sw: Float64Array | undefined,
  order: Int32Array,
  nFeatures: number
): number {
  const { w, q } = state;
  const l1Ratio = p.penalty === "l1" ? 1 : p.penalty === "elasticnet" ? p.l1Ratio : 0;
  const hasL2 = (p.penalty === "l2" || p.penalty === "elasticnet") && p.alpha > 0;
  const hasL1 = (p.penalty === "l1" || p.penalty === "elasticnet") && p.alpha > 0;
  const penalized = p.penalty !== "none";
  let sumObjective = 0;

  for (let s = 0; s < order.length; s++) {
    const idx = order[s] as number;
    const eta = stepSize(p, state, init);
    const base = idx * nFeatures;
    const yi = y[idx] as number;
    const wi = sw ? (sw[idx] as number) : 1;

    let pred = p.fitIntercept ? state.b : 0;
    let sqNorm = 0;
    let l1Norm = 0;
    for (let j = 0; j < nFeatures; j++) {
      const wj = w[j] as number;
      pred += wj * (x[base + j] as number);
      if (penalized) {
        sqNorm += wj * wj;
        l1Norm += Math.abs(wj);
      }
    }

    sumObjective += lossFn.loss(pred, yi);
    if (penalized) sumObjective += p.alpha * ((1 - l1Ratio) * 0.5 * sqNorm + l1Ratio * l1Norm);
    let dloss = lossFn.dloss(pred, yi);
    if (dloss > MAX_DLOSS) dloss = MAX_DLOSS;
    else if (dloss < -MAX_DLOSS) dloss = -MAX_DLOSS;

    const shrink = hasL2 ? Math.max(0, 1 - (1 - l1Ratio) * eta * p.alpha) : 1;
    const step = eta * dloss * wi;
    for (let j = 0; j < nFeatures; j++) {
      w[j] = (w[j] as number) * shrink - step * (x[base + j] as number);
    }
    if (p.fitIntercept) state.b -= step;

    if (hasL1) {
      state.u += l1Ratio * eta * p.alpha;
      const u = state.u;
      for (let j = 0; j < nFeatures; j++) {
        const z = w[j] as number;
        const qj = q[j] as number;
        if (z > 0) w[j] = Math.max(0, z - (u + qj));
        else if (z < 0) w[j] = Math.min(0, z + (u - qj));
        q[j] = qj + ((w[j] as number) - z);
      }
    }
    state.t++;
  }
  return sumObjective;
}

/**
 * Train one binary model from scratch for up to `maxIter` epochs.
 *
 * Stops when the mean training objective (loss plus penalty) has not improved
 * by more than `tol` for `nIterNoChange` consecutive epochs. With the 'adaptive' schedule the step
 * size is divided by 5 instead, until it drops to 1e-6.
 *
 * @returns The state and whether the stopping rule fired before `maxIter`
 */
function trainBinary(
  p: SGDParams,
  lossFn: LossFn,
  x: Float64Array,
  y: Float64Array,
  sw: Float64Array | undefined,
  nSamples: number,
  nFeatures: number,
  rng: () => number
): { state: SGDState; converged: boolean } {
  const state = newState(nFeatures, p.eta0);
  const init = p.learningRate === "optimal" ? optimalInit(p, lossFn) : 0;
  const order = new Int32Array(nSamples);
  for (let i = 0; i < nSamples; i++) order[i] = i;

  let best = Infinity;
  let noImprovement = 0;
  let converged = false;

  for (let epoch = 0; epoch < p.maxIter; epoch++) {
    if (p.shuffle) shuffleIndices(order, rng);
    const sumObjective = sgdEpoch(state, p, lossFn, init, x, y, sw, order, nFeatures);
    state.nIter = epoch + 1;
    assertFiniteState(state, sumObjective, epoch + 1);

    const objective = sumObjective / nSamples;
    if (objective > best - p.tol) noImprovement++;
    else noImprovement = 0;
    if (objective < best) best = objective;

    if (noImprovement >= p.nIterNoChange) {
      if (p.learningRate === "adaptive" && state.eta > 1e-6) {
        state.eta /= 5;
        noImprovement = 0;
      } else {
        converged = true;
        break;
      }
    }
  }
  return { state, converged };
}

function assertFiniteState(state: SGDState, sumLoss: number, epoch: number): void {
  let finite = Number.isFinite(sumLoss) && Number.isFinite(state.b);
  for (let j = 0; finite && j < state.w.length; j++) finite = Number.isFinite(state.w[j]);
  if (!finite) {
    throw new DataValidationError(
      `Floating-point overflow at epoch ${epoch}. Scale the input features (for example with StandardScaler) or lower eta0.`
    );
  }
}

function warnNoConvergence(maxIter: number, source: string): void {
  warn(
    `Maximum number of iterations (${maxIter}) reached before the objective stopped improving; ` +
      "increase maxIter or loosen tol",
    "ConvergenceWarning",
    source
  );
}

/** Check that `y` is a finite 1-D contiguous tensor and return it as float64. */
function checkTargets(y: Tensor): Float64Array {
  if (y.ndim !== 1) throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
  assertContiguous(y, "y");
  if (y.size === 0) throw new DataValidationError("y must contain at least one sample");
  const data = toFloat64View(y);
  for (let i = 0; i < data.length; i++) {
    if (!Number.isFinite(data[i])) {
      throw new DataValidationError("y contains non-finite values (NaN or Inf)");
    }
  }
  return data;
}

function assertClassWeight(cw: unknown): void {
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

/** Weight of each sample's own class, for the single binary problem of a two-class model. */
function perSampleWeights(
  yData: Float64Array,
  classes: ArrayLike<number>,
  perClass: Float64Array | undefined
): Float64Array | undefined {
  if (!perClass) return undefined;
  const sw = new Float64Array(yData.length);
  for (let i = 0; i < yData.length; i++) {
    sw[i] = yData[i] === classes[0] ? (perClass[0] as number) : (perClass[1] as number);
  }
  return sw;
}

/**
 * Weights for the one-vs-rest problem of class `k`: samples of class `k` carry
 * that class's weight and every other sample weighs 1 (scikit-learn's rule).
 */
function ovrWeights(
  yData: Float64Array,
  classes: ArrayLike<number>,
  perClass: Float64Array | undefined,
  k: number
): Float64Array | undefined {
  if (!perClass) return undefined;
  const sw = new Float64Array(yData.length);
  const label = classes[k] as number;
  for (let i = 0; i < yData.length; i++) sw[i] = yData[i] === label ? (perClass[k] as number) : 1;
  return sw;
}

/**
 * SGD Classifier: linear classifiers trained with stochastic gradient descent.
 *
 * Supports multiple loss functions:
 * - `"hinge"`: Linear SVM (default)
 * - `"log_loss"`: Logistic regression
 * - `"modified_huber"`: Smoothed hinge loss with probability estimates
 * - `"squared_hinge"`: Squared hinge loss
 *
 * Models with more than two classes are trained one-vs-rest. The step size
 * follows `learningRate` ('optimal' by default, which needs `alpha > 0`).
 * Feature scaling matters for SGD; standardize the inputs first.
 *
 * @example
 * ```ts
 * import { SGDClassifier } from 'deepbox/ml';
 *
 * const clf = new SGDClassifier({ loss: 'log_loss', alpha: 0.0001 });
 * clf.fit(X_train, y_train);
 * const preds = clf.predict(X_test);
 * ```
 *
 * @category Linear Models
 * @implements {Classifier}
 */
export class SGDClassifier implements Classifier {
  private p: SGDParams;
  private classWeight_: ClassWeight | undefined;

  /** Loss used by the last fit, so that `setParams` cannot change what `predictProba` computes. */
  private fitLoss_: string | undefined;
  private states_?: SGDState[]; // one state for binary, one per class (OvR) otherwise
  private classes_?: Tensor;
  private nFeaturesIn_ = 0;
  private fitted = false;
  private rng_?: () => number;

  /**
   * Create a new SGD classifier.
   *
   * @param options - Configuration options
   * @param options.loss - 'hinge' (default), 'log_loss', 'modified_huber' or 'squared_hinge'
   * @param options.penalty - 'l2' (default), 'l1', 'elasticnet' or 'none'
   * @param options.alpha - Regularization strength (default: 0.0001)
   * @param options.l1Ratio - Elastic net mixing parameter in [0, 1] (default: 0.15)
   * @param options.fitIntercept - Whether to fit an intercept (default: true)
   * @param options.maxIter - Maximum number of epochs (default: 1000)
   * @param options.tol - Stop when the mean training objective improves by less than `tol` for `nIterNoChange` epochs (default: 1e-3)
   * @param options.nIterNoChange - Epochs without improvement before stopping (default: 5)
   * @param options.shuffle - Whether to shuffle the samples every epoch (default: true)
   * @param options.randomState - Seed for shuffling; omit for a random run
   * @param options.learningRate - 'optimal' (default), 'constant', 'invscaling' or 'adaptive'
   * @param options.eta0 - Initial step size for 'constant', 'invscaling' and 'adaptive' (default: 0.01)
   * @param options.powerT - Exponent of the 'invscaling' schedule (default: 0.5)
   * @param options.classWeight - 'balanced' or {classLabel: weight} (default: equal weights)
   * @throws {InvalidParameterError} If any option is invalid
   */
  constructor(
    options: SGDBaseOptions & {
      readonly loss?: Loss;
      readonly classWeight?: ClassWeight;
    } = {}
  ) {
    this.p = defaultParams(options, options.loss ?? "hinge", "optimal", 0.5);
    validateParams(this.p, CLASSIFIER_LOSSES);
    if (options.classWeight !== undefined) {
      assertClassWeight(options.classWeight);
      this.classWeight_ =
        options.classWeight === "balanced" ? "balanced" : { ...options.classWeight };
    }
  }

  private get isMulti(): boolean {
    return (this.states_?.length ?? 0) > 1 && (this.classes_?.size ?? 0) > 2;
  }

  /**
   * Fit the classifier. Any previous fit is discarded.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target labels of shape (n_samples,)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D, y is not 1D, or their sample counts differ
   * @throws {DataValidationError} If the data is empty, contains NaN/Inf, or training overflows
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    validateParams(this.p, CLASSIFIER_LOSSES);

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const x = toFloat64View(X);
    const yData = toFloat64View(y);

    const uniqueClasses = [...new Set(yData)].sort((a, b) => a - b);
    const nClasses = uniqueClasses.length;
    const perClass = this.classWeights(yData, uniqueClasses);
    const lossFn = classificationLoss(this.p.loss as Loss);

    const states: SGDState[] = [];
    let allConverged = true;
    if (nClasses === 2) {
      const yBinary = new Float64Array(nSamples);
      const positive = uniqueClasses[1] as number;
      for (let i = 0; i < nSamples; i++) yBinary[i] = yData[i] === positive ? 1 : -1;
      const res = trainBinary(
        this.p,
        lossFn,
        x,
        yBinary,
        perSampleWeights(yData, uniqueClasses, perClass),
        nSamples,
        nFeatures,
        createTreeRng(this.p.randomState)
      );
      states.push(res.state);
      allConverged = res.converged;
    } else if (nClasses > 2) {
      for (let k = 0; k < nClasses; k++) {
        const cls = uniqueClasses[k] as number;
        const yBinary = new Float64Array(nSamples);
        for (let i = 0; i < nSamples; i++) yBinary[i] = yData[i] === cls ? 1 : -1;
        const res = trainBinary(
          this.p,
          lossFn,
          x,
          yBinary,
          ovrWeights(yData, uniqueClasses, perClass, k),
          nSamples,
          nFeatures,
          createTreeRng(this.p.randomState)
        );
        states.push(res.state);
        allConverged = allConverged && res.converged;
      }
    }
    if (!allConverged) warnNoConvergence(this.p.maxIter, "SGDClassifier");

    this.nFeaturesIn_ = nFeatures;
    this.fitLoss_ = this.p.loss;
    this.classes_ = tensor(uniqueClasses, { dtype: "float64" });
    this.states_ = states;
    this.rng_ = createTreeRng(this.p.randomState);
    this.fitted = true;
    return this;
  }

  /**
   * Run one epoch of SGD on a batch, continuing from the current weights.
   *
   * The first call fixes the class labels: pass `classes` (all labels that
   * will ever appear), or omit it to use the labels found in `y`. Later calls
   * must only contain labels from that set. Calling `partialFit` after `fit`
   * continues from the fitted model. The step size schedule keeps counting
   * updates across calls.
   *
   * @param X - Batch of shape (n_samples, n_features)
   * @param y - Batch labels of shape (n_samples,)
   * @param classes - All class labels (first call only)
   * @returns this
   * @throws {DataValidationError} If a label is not among the known classes
   * @throws {InvalidParameterError} If `classes` is given after the first call and differs from the known classes
   */
  partialFit(X: Tensor, y: Tensor, classes?: Tensor | readonly number[]): this {
    validateFitInputs(X, y);
    validateParams(this.p, CLASSIFIER_LOSSES);
    if (this.classWeight_ === "balanced") {
      throw new InvalidParameterError(
        "classWeight 'balanced' is not supported by partialFit; pass explicit class weights",
        "classWeight",
        this.classWeight_
      );
    }
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const x = toFloat64View(X);
    const yData = toFloat64View(y);
    const lossFn = classificationLoss(this.p.loss as Loss);

    if (!this.fitted || !this.states_ || !this.classes_) {
      let labels: number[];
      if (classes === undefined) {
        labels = [...new Set(yData)];
      } else {
        const raw =
          classes instanceof Float64Array || Array.isArray(classes)
            ? Array.from(classes as ArrayLike<number>)
            : Array.from(toFloat64View(classes as Tensor));
        labels = [...new Set(raw)];
        for (const v of yData) {
          if (!labels.includes(v)) {
            throw new DataValidationError(`y contains label ${v} that is not in classes`);
          }
        }
      }
      labels.sort((a, b) => a - b);
      for (const v of labels) {
        if (!Number.isFinite(v)) {
          throw new InvalidParameterError(
            "classes must contain finite numbers",
            "classes",
            classes
          );
        }
      }
      const nStates = labels.length <= 1 ? 0 : labels.length === 2 ? 1 : labels.length;
      const states: SGDState[] = [];
      for (let k = 0; k < nStates; k++) states.push(newState(nFeatures, this.p.eta0));
      this.classes_ = tensor(labels, { dtype: "float64" });
      this.states_ = states;
      this.nFeaturesIn_ = nFeatures;
      this.rng_ = createTreeRng(this.p.randomState);
      this.fitted = true;
    } else {
      if (nFeatures !== this.nFeaturesIn_) {
        throw new ShapeError(
          `X has ${nFeatures} features but SGDClassifier was fitted with ${this.nFeaturesIn_} features`
        );
      }
      if (classes !== undefined) {
        const known = toFloat64View(this.classes_);
        const given = (
          classes instanceof Float64Array || Array.isArray(classes)
            ? Array.from(classes as ArrayLike<number>)
            : Array.from(toFloat64View(classes as Tensor))
        ).sort((a, b) => a - b);
        const same = given.length === known.length && given.every((v, i) => v === known[i]);
        if (!same) {
          throw new InvalidParameterError(
            "classes differs from the classes seen on the first call",
            "classes",
            classes
          );
        }
      }
    }

    const known = toFloat64View(this.classes_ as Tensor);
    const knownSet = new Set(known);
    for (const v of yData) {
      if (!knownSet.has(v)) {
        throw new DataValidationError(`y contains label ${v} that is not among the known classes`);
      }
    }

    const states = this.states_ as SGDState[];
    const perClass = this.classWeights(yData, known);
    const rng = this.rng_ ?? createTreeRng(this.p.randomState);
    const order = new Int32Array(nSamples);
    for (let i = 0; i < nSamples; i++) order[i] = i;
    const init = this.p.learningRate === "optimal" ? optimalInit(this.p, lossFn) : 0;
    this.fitLoss_ = this.p.loss;

    for (let k = 0; k < states.length; k++) {
      const positive = known.length === 2 ? (known[1] as number) : (known[k] as number);
      const yBinary = new Float64Array(nSamples);
      for (let i = 0; i < nSamples; i++) yBinary[i] = yData[i] === positive ? 1 : -1;
      if (this.p.shuffle) shuffleIndices(order, rng);
      const state = states[k] as SGDState;
      const sw =
        known.length === 2
          ? perSampleWeights(yData, known, perClass)
          : ovrWeights(yData, known, perClass, k);
      const sumLoss = sgdEpoch(state, this.p, lossFn, init, x, yBinary, sw, order, nFeatures);
      state.nIter++;
      assertFiniteState(state, sumLoss, state.nIter);
    }
    return this;
  }

  /**
   * Weight of every class in `classes` from `classWeight`, or undefined for
   * equal weights. 'balanced' uses `n_samples / (n_classes * count)`; classes
   * missing from a dictionary weigh 1.
   */
  private classWeights(yData: Float64Array, classes: ArrayLike<number>): Float64Array | undefined {
    const cw = this.classWeight_;
    if (cw === undefined) return undefined;
    const nClasses = classes.length;
    const out = new Float64Array(nClasses);
    for (let k = 0; k < nClasses; k++) {
      const label = classes[k] as number;
      if (cw === "balanced") {
        let count = 0;
        for (let i = 0; i < yData.length; i++) if (yData[i] === label) count++;
        out[k] = yData.length / (nClasses * count);
      } else {
        out[k] = cw[label] ?? 1;
      }
    }
    return out;
  }

  /** Raw scores, flat: (m) for binary / single class, (m * nClasses) for one-vs-rest. */
  private scores(X: Tensor, method: string): Float64Array {
    if (!this.fitted || !this.states_) {
      throw new NotFittedError(`SGDClassifier must be fitted before ${method}`);
    }
    validatePredictInputs(X, this.nFeaturesIn_, "SGDClassifier");
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = this.nFeaturesIn_;
    const x = toFloat64View(X);
    const states = this.states_;
    const nStates = Math.max(states.length, 1);
    const out = new Float64Array(nSamples * nStates);
    for (let k = 0; k < states.length; k++) {
      const { w, b } = states[k] as SGDState;
      for (let i = 0; i < nSamples; i++) {
        let s = b;
        const base = i * nFeatures;
        for (let j = 0; j < nFeatures; j++) s += (w[j] as number) * (x[base + j] as number);
        out[i * states.length + k] = s;
      }
    }
    return out;
  }

  /**
   * Signed distance of each sample to the separating hyperplane(s).
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Shape (n_samples,) for two classes (positive means `classes[1]`),
   * (n_samples, n_classes) for more than two classes
   * @throws {NotFittedError} If the model has not been fitted
   */
  decisionFunction(X: Tensor): Tensor {
    const scores = this.scores(X, "decisionFunction");
    const nSamples = X.shape[0] ?? 0;
    if (this.isMulti) {
      return tensor(scores, { dtype: "float64" }).reshape([nSamples, this.classes_?.size ?? 0]);
    }
    return tensor(scores, { dtype: "float64" });
  }

  /**
   * Predict class labels.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted labels of shape (n_samples,), dtype float64
   * @throws {NotFittedError} If the model has not been fitted
   */
  predict(X: Tensor): Tensor {
    const scores = this.scores(X, "predict");
    const nSamples = X.shape[0] ?? 0;
    const labels = toFloat64View(this.classes_ as Tensor);
    const result = new Float64Array(nSamples);

    if (this.isMulti) {
      const nClasses = labels.length;
      for (let i = 0; i < nSamples; i++) {
        let best = -Infinity;
        let bestClass = 0;
        for (let k = 0; k < nClasses; k++) {
          const s = scores[i * nClasses + k] as number;
          if (s > best) {
            best = s;
            bestClass = k;
          }
        }
        result[i] = labels[bestClass] as number;
      }
    } else if (labels.length === 1) {
      result.fill(labels[0] as number);
    } else {
      for (let i = 0; i < nSamples; i++) {
        result[i] = (scores[i] as number) > 0 ? (labels[1] as number) : (labels[0] as number);
      }
    }
    return tensor(result, { dtype: "float64" });
  }

  /**
   * Predict class probabilities.
   *
   * Available for `loss: 'log_loss'` (logistic function) and
   * `loss: 'modified_huber'` (`(clip(score, -1, 1) + 1) / 2`). With more than
   * two classes the one-vs-rest probabilities are normalized to sum to one.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Probabilities of shape (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {DataValidationError} If the loss does not provide probabilities
   */
  predictProba(X: Tensor): Tensor {
    if (!this.fitted) throw new NotFittedError("SGDClassifier must be fitted before predictProba");
    const loss = this.fitLoss_ ?? this.p.loss;
    if (loss !== "log_loss" && loss !== "modified_huber") {
      throw new DataValidationError(
        "predictProba is only available for loss='log_loss' or 'modified_huber'"
      );
    }
    const scores = this.scores(X, "predictProba");
    const nSamples = X.shape[0] ?? 0;
    const nClassesTotal = this.classes_?.size ?? 0;
    const toProb = (s: number): number =>
      loss === "log_loss"
        ? s >= 0
          ? 1 / (1 + Math.exp(-s))
          : Math.exp(s) / (1 + Math.exp(s))
        : (Math.min(1, Math.max(-1, s)) + 1) / 2;

    if (this.isMulti) {
      const nClasses = nClassesTotal;
      const proba = new Float64Array(nSamples * nClasses);
      for (let i = 0; i < nSamples; i++) {
        let sum = 0;
        for (let k = 0; k < nClasses; k++) {
          const p = toProb(scores[i * nClasses + k] as number);
          proba[i * nClasses + k] = p;
          sum += p;
        }
        for (let k = 0; k < nClasses; k++) {
          proba[i * nClasses + k] =
            sum > 0 ? (proba[i * nClasses + k] as number) / sum : 1 / nClasses;
        }
      }
      return tensor(proba, { dtype: "float64" }).reshape([nSamples, nClasses]);
    }
    if (nClassesTotal <= 1) {
      return tensor(new Float64Array(nSamples).fill(1), { dtype: "float64" }).reshape([
        nSamples,
        1,
      ]);
    }
    const proba = new Float64Array(nSamples * 2);
    for (let i = 0; i < nSamples; i++) {
      const p1 = toProb(scores[i] as number);
      proba[i * 2] = 1 - p1;
      proba[i * 2 + 1] = p1;
    }
    return tensor(proba, { dtype: "float64" }).reshape([nSamples, 2]);
  }

  /**
   * Mean accuracy on the given data and labels.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True labels of shape (n_samples,)
   * @returns Accuracy in [0, 1]
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1D or its length differs from the number of samples
   * @throws {DataValidationError} If y is empty or contains NaN/Inf
   */
  score(X: Tensor, y: Tensor): number {
    const yData = checkTargets(y);
    const pred = toFloat64View(this.predict(X));
    if (pred.length !== yData.length) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${pred.length}, y=${yData.length}`
      );
    }
    let correct = 0;
    for (let i = 0; i < yData.length; i++) if (pred[i] === yData[i]) correct++;
    return correct / yData.length;
  }

  /** Class labels seen during fit, sorted ascending. */
  get classes(): Tensor | undefined {
    return this.classes_;
  }

  /**
   * Fitted coefficients as a flat array: `n_features` values for two classes,
   * `n_classes * n_features` values (row-major, one row per class) otherwise.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get coef(): Float64Array {
    if (!this.fitted || !this.states_) {
      throw new NotFittedError("SGDClassifier must be fitted to access coef");
    }
    const states = this.states_;
    if (states.length <= 1) return states[0]?.w ?? new Float64Array(this.nFeaturesIn_);
    const out = new Float64Array(states.length * this.nFeaturesIn_);
    states.forEach((s, k) => {
      out.set(s.w, k * this.nFeaturesIn_);
    });
    return out;
  }

  /**
   * Fitted coefficients as a float64 tensor: shape (n_features,) for two
   * classes, (n_classes, n_features) for more.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get coefTensor(): Tensor {
    const flat = this.coef;
    if (this.isMulti) {
      return tensor(flat, { dtype: "float64" }).reshape([
        this.states_?.length ?? 0,
        this.nFeaturesIn_,
      ]);
    }
    return tensor(flat, { dtype: "float64" });
  }

  /**
   * Fitted intercept of a binary model (0 when `fitIntercept` is false).
   *
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {DataValidationError} If the model has more than two classes; use `interceptArray`
   */
  get intercept(): number {
    if (!this.fitted || !this.states_) {
      throw new NotFittedError("SGDClassifier must be fitted to access intercept");
    }
    if (this.isMulti) {
      throw new DataValidationError(
        "A multiclass model has one intercept per class; use interceptArray"
      );
    }
    return this.states_[0]?.b ?? 0;
  }

  /**
   * Fitted intercepts, one per one-vs-rest model (a single value for two classes).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get interceptArray(): number[] {
    if (!this.fitted || !this.states_) {
      throw new NotFittedError("SGDClassifier must be fitted to access interceptArray");
    }
    if (this.states_.length === 0) return [0];
    return this.states_.map((s) => s.b);
  }

  /**
   * Number of epochs run by the last fit (the maximum over the one-vs-rest models).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nIter(): number {
    if (!this.fitted || !this.states_) {
      throw new NotFittedError("SGDClassifier must be fitted to access nIter");
    }
    return this.states_.reduce((m, s) => Math.max(m, s.nIter), 0);
  }

  /**
   * Get the hyperparameters of this estimator.
   */
  getParams(): Record<string, unknown> {
    const params: Record<string, unknown> = {
      loss: this.p.loss,
      penalty: this.p.penalty,
      alpha: this.p.alpha,
      l1Ratio: this.p.l1Ratio,
      fitIntercept: this.p.fitIntercept,
      maxIter: this.p.maxIter,
      tol: this.p.tol,
      shuffle: this.p.shuffle,
      randomState: this.p.randomState,
      learningRate: this.p.learningRate,
      eta0: this.p.eta0,
      powerT: this.p.powerT,
      nIterNoChange: this.p.nIterNoChange,
    };
    const cw = this.classWeight_;
    if (cw !== undefined) params["classWeight"] = typeof cw === "string" ? cw : { ...cw };
    return params;
  }

  /**
   * Set hyperparameters. All values are validated before any is applied; the
   * fitted model is unchanged until `fit` is called again.
   *
   * @param params - Parameters to set
   * @returns this
   * @throws {InvalidParameterError} If a name is unknown or a value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    const { classWeight, ...rest } = params;
    const next = applySharedParams(this.p, rest, [
      "loss",
      "penalty",
      "alpha",
      "l1Ratio",
      "fitIntercept",
      "maxIter",
      "tol",
      "shuffle",
      "randomState",
      "learningRate",
      "eta0",
      "powerT",
      "nIterNoChange",
    ]);
    validateParams(next, CLASSIFIER_LOSSES);
    if ("classWeight" in params && classWeight !== undefined) assertClassWeight(classWeight);
    this.p = next;
    if ("classWeight" in params) {
      this.classWeight_ =
        classWeight === undefined
          ? undefined
          : classWeight === "balanced"
            ? "balanced"
            : { ...(classWeight as Record<number, number>) };
    }
    return this;
  }

  /**
   * Create an unfitted copy of this estimator with the same parameters.
   */
  clone(): SGDClassifier {
    return new SGDClassifier(
      this.getParams() as SGDBaseOptions & { loss?: Loss; classWeight?: ClassWeight }
    );
  }
}

/**
 * SGD Regressor: linear regressor trained with stochastic gradient descent.
 *
 * Supports multiple loss functions:
 * - `"squared_error"`: Ordinary least squares (default)
 * - `"huber"`: Huber loss (less sensitive to outliers)
 * - `"epsilon_insensitive"`: Support vector regression loss
 *
 * The default schedule is 'invscaling' with `eta0 = 0.01`. Feature and target
 * scaling matter for SGD; standardize the inputs first.
 *
 * @example
 * ```ts
 * import { SGDRegressor } from 'deepbox/ml';
 *
 * const reg = new SGDRegressor({ loss: 'huber', alpha: 0.0001 });
 * reg.fit(X_train, y_train);
 * const preds = reg.predict(X_test);
 * ```
 *
 * @category Linear Models
 * @implements {Regressor}
 */
export class SGDRegressor implements Regressor {
  private p: SGDParams;

  private state_?: SGDState;
  private nFeaturesIn_ = 0;
  private fitted = false;
  private rng_?: () => number;

  /**
   * Create a new SGD regressor.
   *
   * @param options - Configuration options
   * @param options.loss - 'squared_error' (default), 'huber' or 'epsilon_insensitive'
   * @param options.epsilon - Threshold of the 'huber' and 'epsilon_insensitive' losses (default: 0.1)
   * @param options.penalty - 'l2' (default), 'l1', 'elasticnet' or 'none'
   * @param options.alpha - Regularization strength (default: 0.0001)
   * @param options.l1Ratio - Elastic net mixing parameter in [0, 1] (default: 0.15)
   * @param options.fitIntercept - Whether to fit an intercept (default: true)
   * @param options.maxIter - Maximum number of epochs (default: 1000)
   * @param options.tol - Stop when the mean training objective improves by less than `tol` for `nIterNoChange` epochs (default: 1e-3)
   * @param options.nIterNoChange - Epochs without improvement before stopping (default: 5)
   * @param options.shuffle - Whether to shuffle the samples every epoch (default: true)
   * @param options.randomState - Seed for shuffling; omit for a random run
   * @param options.learningRate - 'invscaling' (default), 'constant', 'optimal' or 'adaptive'
   * @param options.eta0 - Initial step size (default: 0.01)
   * @param options.powerT - Exponent of the 'invscaling' schedule (default: 0.25)
   * @throws {InvalidParameterError} If any option is invalid
   */
  constructor(
    options: SGDBaseOptions & {
      readonly loss?: RegressionLoss;
      readonly epsilon?: number;
    } = {}
  ) {
    this.p = defaultParams(options, options.loss ?? "squared_error", "invscaling", 0.25);
    validateParams(this.p, REGRESSION_LOSSES);
  }

  /**
   * Fit the regressor. Any previous fit is discarded.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target values of shape (n_samples,)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D, y is not 1D, or their sample counts differ
   * @throws {DataValidationError} If the data is empty, contains NaN/Inf, or training overflows
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    validateParams(this.p, REGRESSION_LOSSES);

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const x = toFloat64View(X);
    const yData = toFloat64View(y);
    const lossFn = regressionLoss(this.p.loss as RegressionLoss, this.p.epsilon);

    const res = trainBinary(
      this.p,
      lossFn,
      x,
      yData,
      undefined,
      nSamples,
      nFeatures,
      createTreeRng(this.p.randomState)
    );
    if (!res.converged) warnNoConvergence(this.p.maxIter, "SGDRegressor");

    this.nFeaturesIn_ = nFeatures;
    this.state_ = res.state;
    this.rng_ = createTreeRng(this.p.randomState);
    this.fitted = true;
    return this;
  }

  /**
   * Run one epoch of SGD on a batch, continuing from the current weights.
   *
   * The first call starts from zero weights. Calling `partialFit` after `fit`
   * continues from the fitted model. The step size schedule keeps counting
   * updates across calls.
   *
   * @param X - Batch of shape (n_samples, n_features)
   * @param y - Batch targets of shape (n_samples,)
   * @returns this
   * @throws {ShapeError} If the feature count differs from the previous call
   */
  partialFit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    validateParams(this.p, REGRESSION_LOSSES);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    if (this.fitted && nFeatures !== this.nFeaturesIn_) {
      throw new ShapeError(
        `X has ${nFeatures} features but SGDRegressor was fitted with ${this.nFeaturesIn_} features`
      );
    }
    const x = toFloat64View(X);
    const yData = toFloat64View(y);
    const lossFn = regressionLoss(this.p.loss as RegressionLoss, this.p.epsilon);

    if (!this.fitted || !this.state_) {
      this.state_ = newState(nFeatures, this.p.eta0);
      this.nFeaturesIn_ = nFeatures;
      this.rng_ = createTreeRng(this.p.randomState);
    }
    const state = this.state_;
    const rng = this.rng_ ?? createTreeRng(this.p.randomState);
    const order = new Int32Array(nSamples);
    for (let i = 0; i < nSamples; i++) order[i] = i;
    if (this.p.shuffle) shuffleIndices(order, rng);
    const init = this.p.learningRate === "optimal" ? optimalInit(this.p, lossFn) : 0;
    const sumLoss = sgdEpoch(state, this.p, lossFn, init, x, yData, undefined, order, nFeatures);
    state.nIter++;
    assertFiniteState(state, sumLoss, state.nIter);
    this.fitted = true;
    return this;
  }

  /**
   * Predict target values.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predictions of shape (n_samples,), dtype float64
   * @throws {NotFittedError} If the model has not been fitted
   */
  predict(X: Tensor): Tensor {
    const state = this.state_;
    if (!this.fitted || !state) {
      throw new NotFittedError("SGDRegressor must be fitted before predict");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "SGDRegressor");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = this.nFeaturesIn_;
    const x = toFloat64View(X);
    const w = state.w;
    const intercept = state.b;
    const result = new Float64Array(nSamples);

    for (let i = 0; i < nSamples; i++) {
      let pred = intercept;
      const base = i * nFeatures;
      for (let j = 0; j < nFeatures; j++) pred += (w[j] as number) * (x[base + j] as number);
      result[i] = pred;
    }
    return tensor(result, { dtype: "float64" });
  }

  /**
   * Coefficient of determination R² of the prediction.
   *
   * A constant y gives 1 when the predictions are exact and 0 otherwise.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True values of shape (n_samples,)
   * @returns R² score (can be negative)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1D or its length differs from the number of samples
   * @throws {DataValidationError} If y is empty or contains NaN/Inf
   */
  score(X: Tensor, y: Tensor): number {
    return r2ScoreOf(y, () => this.predict(X));
  }

  /**
   * Fitted coefficients, one per feature.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get coef(): Float64Array {
    if (!this.fitted || !this.state_) {
      throw new NotFittedError("SGDRegressor must be fitted to access coef");
    }
    return this.state_.w;
  }

  /**
   * Fitted coefficients as a float64 tensor of shape (n_features,).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get coefTensor(): Tensor {
    return tensor(this.coef, { dtype: "float64" });
  }

  /**
   * Fitted intercept (0 when `fitIntercept` is false).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get intercept(): number {
    if (!this.fitted || !this.state_) {
      throw new NotFittedError("SGDRegressor must be fitted to access intercept");
    }
    return this.state_.b;
  }

  /**
   * Number of epochs run by the last fit.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nIter(): number {
    if (!this.fitted || !this.state_) {
      throw new NotFittedError("SGDRegressor must be fitted to access nIter");
    }
    return this.state_.nIter;
  }

  /**
   * Get the hyperparameters of this estimator.
   */
  getParams(): Record<string, unknown> {
    return {
      loss: this.p.loss,
      penalty: this.p.penalty,
      alpha: this.p.alpha,
      l1Ratio: this.p.l1Ratio,
      fitIntercept: this.p.fitIntercept,
      maxIter: this.p.maxIter,
      tol: this.p.tol,
      shuffle: this.p.shuffle,
      randomState: this.p.randomState,
      learningRate: this.p.learningRate,
      eta0: this.p.eta0,
      powerT: this.p.powerT,
      nIterNoChange: this.p.nIterNoChange,
      epsilon: this.p.epsilon,
    };
  }

  /**
   * Set hyperparameters. All values are validated before any is applied; the
   * fitted model is unchanged until `fit` is called again.
   *
   * @param params - Parameters to set
   * @returns this
   * @throws {InvalidParameterError} If a name is unknown or a value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    const next = applySharedParams(this.p, params, [
      "loss",
      "penalty",
      "alpha",
      "l1Ratio",
      "fitIntercept",
      "maxIter",
      "tol",
      "shuffle",
      "randomState",
      "learningRate",
      "eta0",
      "powerT",
      "nIterNoChange",
      "epsilon",
    ]);
    validateParams(next, REGRESSION_LOSSES);
    this.p = next;
    return this;
  }

  /**
   * Create an unfitted copy of this estimator with the same parameters.
   */
  clone(): SGDRegressor {
    return new SGDRegressor(
      this.getParams() as SGDBaseOptions & { loss?: RegressionLoss; epsilon?: number }
    );
  }
}
