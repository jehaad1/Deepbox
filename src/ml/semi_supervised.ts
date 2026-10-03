/**
 * Semi-supervised learning: Label Propagation, Label Spreading and Self-Training.
 *
 * Label Propagation and Label Spreading spread labels from labeled to unlabeled
 * samples over a similarity graph (RBF kernel). Unlabeled samples must have
 * label = -1.
 *
 * @module ml/semi_supervised
 * @see {@link https://deepbox.dev/docs/ml-advanced | Deepbox Semi-Supervised Learning}
 */

import {
  DataValidationError,
  InvalidParameterError,
  MemoryError,
  NotFittedError,
  ShapeError,
  warn,
} from "../core";
import { type Tensor, tensor } from "../ndarray";
import { cloneEstimator } from "./_internal";
import { toFloat64View, validateFitInputs, validatePredictInputs } from "./_validation";
import type { Classifier } from "./base";

/** Label that marks an unlabeled sample. */
const UNLABELED = -1;

/** Largest affinity matrix (number of entries) the graph models will allocate. */
const MAX_GRAPH_ENTRIES = 2 ** 28;

/** Fraction of correct predictions, with the same target checks for every classifier here. */
function accuracy(pred: Tensor, y: Tensor): number {
  if (y.ndim !== 1) {
    throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
  }
  const truth = toFloat64View(y);
  if (truth.length === 0) {
    throw new DataValidationError("y must contain at least one sample");
  }
  const predicted = toFloat64View(pred);
  if (predicted.length !== truth.length) {
    throw new ShapeError(
      `X and y must have the same number of samples; got X=${predicted.length}, y=${truth.length}`
    );
  }
  let correct = 0;
  for (let i = 0; i < truth.length; i++) {
    if (predicted[i] === truth[i]) correct++;
  }
  return correct / truth.length;
}

function checkMaxIter(value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new InvalidParameterError("maxIter must be an integer >= 1", "maxIter", value);
  }
  return value;
}

function checkTol(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new InvalidParameterError("tol must be a finite number >= 0", "tol", value);
  }
  return value;
}

function checkGamma(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new InvalidParameterError("gamma must be a finite number > 0", "gamma", value);
  }
  return value;
}

function checkAlpha(value: unknown): number {
  if (typeof value !== "number" || !(value >= 0 && value <= 1)) {
    throw new InvalidParameterError("alpha must be in [0, 1]", "alpha", value);
  }
  return value;
}

/**
 * Shared implementation of {@link LabelPropagation} and {@link LabelSpreading}.
 *
 * Both models build a dense RBF affinity matrix over the training samples and iterate a
 * linear update on the label distributions, exactly like scikit-learn's
 * `LabelPropagation` / `LabelSpreading`.
 */
abstract class GraphLabelModel {
  protected maxIter: number;
  protected tol: number;
  protected gamma: number;

  private classes_: number[] = [];
  private labelDistributions_?: Float64Array;
  private xTrain_?: Float64Array;
  private nTrainSamples_ = 0;
  private nFeaturesIn_ = 0;
  private nIter_ = 0;
  private fitted = false;

  protected constructor(maxIter: number, tol: number, gamma: number) {
    this.maxIter = checkMaxIter(maxIter);
    this.tol = checkTol(tol);
    this.gamma = checkGamma(gamma);
  }

  protected abstract readonly modelName: string;

  /**
   * Run the graph algorithm.
   *
   * @param alpha - Clamping factor of Label Spreading, or `null` for Label Propagation
   */
  protected fitGraph(X: Tensor, y: Tensor, alpha: number | null): void {
    this.fitted = false;
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const nF = X.shape[1] ?? 0;
    if (n * n > MAX_GRAPH_ENTRIES) {
      throw new MemoryError(
        `${this.modelName} builds a dense ${n} x ${n} affinity matrix, which is too large`,
        { requestedBytes: n * n * 8 }
      );
    }

    const xTrain = Float64Array.from(toFloat64View(X));
    const labels = toFloat64View(y);

    const classes = [...new Set(labels)].filter((c) => c !== UNLABELED).sort((a, b) => a - b);
    const nClasses = classes.length;
    if (nClasses === 0) {
      throw new DataValidationError(
        `${this.modelName} needs at least one labeled sample; every label in y is ${UNLABELED}`
      );
    }
    const classIndex = new Map<number, number>();
    classes.forEach((c, i) => {
      classIndex.set(c, i);
    });
    const isLabeled = new Uint8Array(n);
    for (let i = 0; i < n; i++) isLabeled[i] = (labels[i] as number) !== UNLABELED ? 1 : 0;

    // RBF affinity W_ij = exp(-gamma * ||x_i - x_j||^2); the diagonal is 1.
    const W = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      W[i * n + i] = 1;
      for (let j = i + 1; j < n; j++) {
        let sq = 0;
        for (let f = 0; f < nF; f++) {
          const diff = (xTrain[i * nF + f] as number) - (xTrain[j * nF + f] as number);
          sq += diff * diff;
        }
        const w = Math.exp(-this.gamma * sq);
        W[i * n + j] = w;
        W[j * n + i] = w;
      }
    }

    if (alpha === null) {
      // Propagation: row-normalized transition matrix D^-1 W (the diagonal stays in).
      for (let i = 0; i < n; i++) {
        let rowSum = 0;
        for (let j = 0; j < n; j++) rowSum += W[i * n + j] as number;
        for (let j = 0; j < n; j++) W[i * n + j] = (W[i * n + j] as number) / rowSum;
      }
    } else {
      // Spreading: symmetric normalization D^-1/2 W D^-1/2 with a zero diagonal.
      const invSqrtDeg = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        // The diagonal (W_ii = 1) is not part of the graph; summing only the off-diagonal
        // entries keeps tiny affinities of isolated samples from being lost to rounding.
        let rowSum = 0;
        for (let j = 0; j < n; j++) {
          if (j !== i) rowSum += W[i * n + j] as number;
        }
        invSqrtDeg[i] = rowSum > 0 ? 1 / Math.sqrt(rowSum) : 0;
      }
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          W[i * n + j] =
            i === j
              ? 0
              : (invSqrtDeg[i] as number) * (W[i * n + j] as number) * (invSqrtDeg[j] as number);
        }
      }
    }

    // Distributions start as one-hot rows for labeled samples and zeros otherwise.
    let Y = new Float64Array(n * nClasses);
    const yStatic = new Float64Array(n * nClasses);
    for (let i = 0; i < n; i++) {
      if (isLabeled[i]) {
        const c = classIndex.get(labels[i] as number) as number;
        Y[i * nClasses + c] = 1;
        yStatic[i * nClasses + c] = alpha === null ? 1 : 1 - alpha;
      }
    }

    let previous = new Float64Array(n * nClasses);
    let next = new Float64Array(n * nClasses);
    let nIter = this.maxIter;
    let converged = false;
    for (let iter = 0; iter < this.maxIter; iter++) {
      let change = 0;
      for (let k = 0; k < Y.length; k++)
        change += Math.abs((Y[k] as number) - (previous[k] as number));
      if (change < this.tol) {
        nIter = iter;
        converged = true;
        break;
      }

      previous = Y;
      next.fill(0);
      for (let i = 0; i < n; i++) {
        const rowOut = i * nClasses;
        for (let j = 0; j < n; j++) {
          const t = W[i * n + j] as number;
          if (t === 0) continue;
          const rowIn = j * nClasses;
          for (let c = 0; c < nClasses; c++) {
            next[rowOut + c] = (next[rowOut + c] as number) + t * (Y[rowIn + c] as number);
          }
        }
      }

      if (alpha === null) {
        for (let i = 0; i < n; i++) {
          if (isLabeled[i]) {
            // Hard clamping: labeled samples keep their original label.
            for (let c = 0; c < nClasses; c++) {
              next[i * nClasses + c] = yStatic[i * nClasses + c] as number;
            }
          } else {
            let rowSum = 0;
            for (let c = 0; c < nClasses; c++) rowSum += next[i * nClasses + c] as number;
            if (rowSum > 0) {
              for (let c = 0; c < nClasses; c++) {
                next[i * nClasses + c] = (next[i * nClasses + c] as number) / rowSum;
              }
            }
          }
        }
      } else {
        for (let k = 0; k < next.length; k++) {
          next[k] = alpha * (next[k] as number) + (yStatic[k] as number);
        }
      }

      const recycled = previous;
      Y = next;
      next = recycled;
    }
    if (!converged) {
      warn(
        `maxIter=${this.maxIter} was reached without convergence.`,
        "ConvergenceWarning",
        this.modelName
      );
    }

    // Final distributions are row-normalized (rows without any mass stay zero).
    for (let i = 0; i < n; i++) {
      let rowSum = 0;
      for (let c = 0; c < nClasses; c++) rowSum += Y[i * nClasses + c] as number;
      if (rowSum > 0) {
        for (let c = 0; c < nClasses; c++) {
          Y[i * nClasses + c] = (Y[i * nClasses + c] as number) / rowSum;
        }
      }
    }

    this.xTrain_ = xTrain;
    this.classes_ = classes;
    this.labelDistributions_ = Y;
    this.nTrainSamples_ = n;
    this.nFeaturesIn_ = nF;
    this.nIter_ = nIter;
    this.fitted = true;
  }

  /**
   * Predict the most probable class of each sample.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted labels of shape (n_samples,), float64 (ties go to the smaller label)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong dimensions or feature count
   */
  predict(X: Tensor): Tensor {
    const proba = toFloat64View(this.predictProba(X));
    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classes_.length;
    const labels = new Float64Array(nSamples);

    for (let i = 0; i < nSamples; i++) {
      let bestC = 0;
      for (let c = 1; c < nClasses; c++) {
        if ((proba[i * nClasses + c] as number) > (proba[i * nClasses + bestC] as number))
          bestC = c;
      }
      labels[i] = this.classes_[bestC] as number;
    }
    return tensor(labels);
  }

  /**
   * Estimate class probabilities of new samples.
   *
   * Each sample gets the kernel-weighted average of the learned label distributions of the
   * training samples, normalized to sum to 1. The kernel weights are normalized in a
   * numerically stable way, so samples far from all training data still get a meaningful
   * result (dominated by their nearest training samples) instead of 0/0. If every nearby
   * training sample carries no label information the row is uniform.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Probabilities of shape (n_samples, n_classes), float64; columns follow `classes`
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong dimensions or feature count
   */
  predictProba(X: Tensor): Tensor {
    if (!this.fitted || !this.xTrain_ || !this.labelDistributions_) {
      throw new NotFittedError(`${this.modelName} must be fitted before predict`);
    }
    validatePredictInputs(X, this.nFeaturesIn_, this.modelName);
    const nTest = X.shape[0] ?? 0;
    const nF = this.nFeaturesIn_;
    const nTrain = this.nTrainSamples_;
    const nClasses = this.classes_.length;
    const rows = toFloat64View(X);
    const train = this.xTrain_;
    const dist = this.labelDistributions_;

    const result = new Float64Array(nTest * nClasses);
    const logWeights = new Float64Array(nTrain);
    for (let i = 0; i < nTest; i++) {
      let maxLog = Number.NEGATIVE_INFINITY;
      for (let j = 0; j < nTrain; j++) {
        let sq = 0;
        for (let f = 0; f < nF; f++) {
          const diff = (rows[i * nF + f] as number) - (train[j * nF + f] as number);
          sq += diff * diff;
        }
        const lw = -this.gamma * sq;
        logWeights[j] = lw;
        if (lw > maxLog) maxLog = lw;
      }

      // exp(logw - max) cannot underflow for the nearest training sample.
      let total = 0;
      for (let j = 0; j < nTrain; j++) {
        const w = Math.exp((logWeights[j] as number) - maxLog);
        logWeights[j] = w;
        const base = j * nClasses;
        for (let c = 0; c < nClasses; c++) {
          result[i * nClasses + c] =
            (result[i * nClasses + c] as number) + w * (dist[base + c] as number);
        }
      }
      for (let c = 0; c < nClasses; c++) total += result[i * nClasses + c] as number;
      for (let c = 0; c < nClasses; c++) {
        result[i * nClasses + c] =
          total > 0 ? (result[i * nClasses + c] as number) / total : 1 / nClasses;
      }
    }

    return tensor(result).reshape([nTest, nClasses]);
  }

  /**
   * Return the mean accuracy on the given test data and labels.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True labels of shape (n_samples,)
   * @returns Accuracy in [0, 1]
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-D or its length differs from the number of samples
   */
  score(X: Tensor, y: Tensor): number {
    return accuracy(this.predict(X), y);
  }

  /**
   * Sorted labels seen during fit, without the unlabeled marker -1.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get classes(): Tensor {
    if (!this.fitted) {
      throw new NotFittedError(`${this.modelName} must be fitted to access classes`);
    }
    return tensor(Float64Array.from(this.classes_));
  }

  /**
   * Learned label distribution of every training sample, shape (n_samples, n_classes).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get labelDistributions(): Tensor {
    if (!this.fitted || !this.labelDistributions_) {
      throw new NotFittedError(`${this.modelName} must be fitted to access labelDistributions`);
    }
    return tensor(Float64Array.from(this.labelDistributions_)).reshape([
      this.nTrainSamples_,
      this.classes_.length,
    ]);
  }

  /**
   * Label assigned to every training sample (the most probable class of its distribution).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get transduction(): Tensor {
    if (!this.fitted || !this.labelDistributions_) {
      throw new NotFittedError(`${this.modelName} must be fitted to access transduction`);
    }
    const nClasses = this.classes_.length;
    const out = new Float64Array(this.nTrainSamples_);
    for (let i = 0; i < out.length; i++) {
      let best = 0;
      for (let c = 1; c < nClasses; c++) {
        if (
          (this.labelDistributions_[i * nClasses + c] as number) >
          (this.labelDistributions_[i * nClasses + best] as number)
        ) {
          best = c;
        }
      }
      out[i] = this.classes_[best] as number;
    }
    return tensor(out);
  }

  /**
   * Number of iterations run during fit (`maxIter` if it did not converge).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nIter(): number {
    if (!this.fitted) {
      throw new NotFittedError(`${this.modelName} must be fitted to access nIter`);
    }
    return this.nIter_;
  }
}

/**
 * Label Propagation algorithm.
 *
 * Builds a fully connected graph over the training samples with RBF similarities
 * `exp(-gamma * ||x - x'||^2)` and repeatedly replaces each unlabeled sample's label
 * distribution by the similarity-weighted average of all samples' distributions. Labeled
 * samples are clamped to their original label. Follows scikit-learn's `LabelPropagation`
 * (RBF kernel).
 *
 * Unlabeled samples must have label -1.
 *
 * @example
 * ```ts
 * import { LabelPropagation } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0,0],[1,0],[0,1],[1,1],[5,5],[6,5]]);
 * const y = tensor([0, -1, -1, -1, 1, -1]); // -1 = unlabeled
 * const lp = new LabelPropagation();
 * lp.fit(X, y);
 * console.log(lp.predict(X));
 * ```
 */
export class LabelPropagation extends GraphLabelModel implements Classifier {
  protected readonly modelName = "LabelPropagation";

  /**
   * @param options.maxIter - Maximum number of iterations (default: 1000)
   * @param options.tol - Convergence tolerance on the summed absolute change of the label
   *   distributions (default: 1e-3)
   * @param options.gamma - RBF kernel coefficient, > 0 (default: 20)
   */
  constructor(
    options: {
      readonly maxIter?: number;
      readonly tol?: number;
      readonly gamma?: number;
    } = {}
  ) {
    super(options.maxIter ?? 1000, options.tol ?? 1e-3, options.gamma ?? 20);
  }

  /**
   * Propagate the known labels to all training samples.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @param y - Labels of shape (n_samples,), with -1 for unlabeled samples
   * @returns this
   * @throws {ShapeError} If X is not 2D, y is not 1D or their sample counts differ
   * @throws {DataValidationError} If X or y contain NaN/Inf or no sample is labeled
   */
  fit(X: Tensor, y: Tensor): this {
    this.fitGraph(X, y, null);
    return this;
  }

  getParams(): Record<string, unknown> {
    return { maxIter: this.maxIter, tol: this.tol, gamma: this.gamma };
  }

  /**
   * Set hyperparameters. Refit the model afterwards for them to take effect.
   *
   * @throws {InvalidParameterError} If a value is invalid or a key is unknown
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "maxIter":
          this.maxIter = checkMaxIter(value);
          break;
        case "tol":
          this.tol = checkTol(value);
          break;
        case "gamma":
          this.gamma = checkGamma(value);
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  /**
   * Create an unfitted copy with the same hyperparameters.
   *
   * @returns A new LabelPropagation
   */
  clone(): LabelPropagation {
    return new LabelPropagation({ maxIter: this.maxIter, tol: this.tol, gamma: this.gamma });
  }
}

/**
 * Label Spreading algorithm.
 *
 * Similar to {@link LabelPropagation} but uses the symmetrically normalized graph
 * `S = D^-1/2 W D^-1/2` (zero diagonal) and soft clamping. Each iteration computes
 * `Y = alpha * S Y + (1 - alpha) * Y0`, where `Y0` holds the original labels. `alpha` is the
 * share of a sample's distribution that comes from its neighbors: at `alpha = 0` nothing is
 * propagated and unlabeled samples stay without a distribution; larger values let the
 * graph dominate and the original labels fade (at `alpha = 1` they are ignored entirely).
 * Follows scikit-learn's `LabelSpreading` (RBF kernel).
 *
 * Unlabeled samples must have label -1.
 *
 * @example
 * ```ts
 * import { LabelSpreading } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0,0],[1,0],[0,1],[1,1],[5,5],[6,5]]);
 * const y = tensor([0, -1, -1, -1, 1, -1]); // -1 = unlabeled
 * const ls = new LabelSpreading({ alpha: 0.2 });
 * ls.fit(X, y);
 * console.log(ls.predict(X));
 * ```
 */
export class LabelSpreading extends GraphLabelModel implements Classifier {
  protected readonly modelName = "LabelSpreading";
  private alpha: number;

  /**
   * @param options.maxIter - Maximum number of iterations (default: 30)
   * @param options.tol - Convergence tolerance on the summed absolute change of the label
   *   distributions (default: 1e-3)
   * @param options.gamma - RBF kernel coefficient, > 0 (default: 20)
   * @param options.alpha - Share of the neighbor information in each update, in [0, 1]
   *   (default: 0.2)
   */
  constructor(
    options: {
      readonly maxIter?: number;
      readonly tol?: number;
      readonly gamma?: number;
      readonly alpha?: number;
    } = {}
  ) {
    super(options.maxIter ?? 30, options.tol ?? 1e-3, options.gamma ?? 20);
    this.alpha = checkAlpha(options.alpha ?? 0.2);
  }

  /**
   * Spread the known labels to all training samples.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @param y - Labels of shape (n_samples,), with -1 for unlabeled samples
   * @returns this
   * @throws {ShapeError} If X is not 2D, y is not 1D or their sample counts differ
   * @throws {DataValidationError} If X or y contain NaN/Inf or no sample is labeled
   */
  fit(X: Tensor, y: Tensor): this {
    this.fitGraph(X, y, this.alpha);
    return this;
  }

  getParams(): Record<string, unknown> {
    return {
      maxIter: this.maxIter,
      tol: this.tol,
      gamma: this.gamma,
      alpha: this.alpha,
    };
  }

  /**
   * Set hyperparameters. Refit the model afterwards for them to take effect.
   *
   * @throws {InvalidParameterError} If a value is invalid or a key is unknown
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "maxIter":
          this.maxIter = checkMaxIter(value);
          break;
        case "tol":
          this.tol = checkTol(value);
          break;
        case "gamma":
          this.gamma = checkGamma(value);
          break;
        case "alpha":
          this.alpha = checkAlpha(value);
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  /**
   * Create an unfitted copy with the same hyperparameters.
   *
   * @returns A new LabelSpreading
   */
  clone(): LabelSpreading {
    return new LabelSpreading({
      maxIter: this.maxIter,
      tol: this.tol,
      gamma: this.gamma,
      alpha: this.alpha,
    });
  }
}

/** Why {@link SelfTrainingClassifier} stopped adding labels. */
export type SelfTrainingTermination = "max_iter" | "no_change" | "all_labeled";

function checkThreshold(value: unknown): number {
  if (typeof value !== "number" || !(value >= 0 && value <= 1)) {
    throw new InvalidParameterError("threshold must be in [0, 1]", "threshold", value);
  }
  return value;
}

function checkCriterion(value: unknown): "threshold" | "kBest" {
  if (value !== "threshold" && value !== "kBest") {
    throw new InvalidParameterError(
      `criterion must be "threshold" or "kBest"; received ${String(value)}`,
      "criterion",
      value
    );
  }
  return value;
}

function checkKBest(value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new InvalidParameterError("kBest must be an integer >= 1", "kBest", value);
  }
  return value;
}

/** Build a float64 matrix from the given rows of a row-major matrix. */
function takeRows(data: Float64Array, nF: number, rows: readonly number[]): Tensor {
  const out = new Float64Array(rows.length * nF);
  for (let r = 0; r < rows.length; r++) {
    const base = (rows[r] as number) * nF;
    for (let f = 0; f < nF; f++) out[r * nF + f] = data[base + f] as number;
  }
  return tensor(out).reshape([rows.length, nF]);
}

/**
 * Self-Training Classifier.
 *
 * A semi-supervised meta-estimator that iteratively labels unlabeled
 * data using predictions from a supervised base classifier. In each
 * iteration, the classifier is fit on all currently-labeled data, then
 * the most confident predictions on unlabeled data are added to the labeled set:
 * with `criterion: "threshold"` (default) those whose top class probability is
 * strictly above `threshold`, with `criterion: "kBest"` the `kBest` most confident ones.
 * Training stops when every sample is labeled, nothing was added in an iteration, or
 * `maxIter` iterations ran. The final estimator is fit on the original labels plus all
 * pseudo-labels.
 *
 * Unlabeled samples should have label = -1. The base estimator is cloned, so the instance
 * passed in is not modified (use `estimator` after fitting to inspect the fitted copy).
 *
 * @example
 * ```ts
 * import { SelfTrainingClassifier } from 'deepbox/ml';
 * import { LogisticRegression } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0,0],[1,0],[0,1],[1,1],[5,5],[6,5],[5,6],[6,6]]);
 * const y = tensor([0, -1, -1, -1, 1, -1, -1, -1]); // -1 = unlabeled
 * const base = new LogisticRegression();
 * const st = new SelfTrainingClassifier({ baseEstimator: base });
 * st.fit(X, y);
 * st.predict(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-advanced | Deepbox Semi-Supervised Learning}
 *
 * References:
 * - Yarowsky, D. (1995). Unsupervised word sense disambiguation rivaling supervised methods.
 * - Triguero, I., et al. (2015). Self-labeled techniques for semi-supervised learning.
 */
export class SelfTrainingClassifier implements Classifier {
  private baseEstimator: Classifier;
  private threshold: number;
  private maxIter: number;
  private criterion: "threshold" | "kBest";
  private kBest: number;
  private verbose: boolean;

  private estimator_?: Classifier;
  private classes_: number[] = [];
  private nFeaturesIn_ = 0;
  private nIter_ = 0;
  private labeledIter_?: Int32Array;
  private transduction_?: Float64Array;
  private termination_?: SelfTrainingTermination;
  private fitted = false;

  /**
   * @param options.baseEstimator - Classifier with `predictProba` (it is cloned for fitting)
   * @param options.threshold - Minimum top-class probability, exclusive, for adding a
   *   pseudo-label with `criterion: "threshold"`; in [0, 1] (default: 0.75)
   * @param options.maxIter - Maximum number of self-training iterations (default: 10)
   * @param options.criterion - "threshold" (default) or "kBest"
   * @param options.kBest - Number of pseudo-labels added per iteration with `criterion: "kBest"`
   *   (default: 10)
   * @param options.verbose - Warn when an iteration adds no labels (default: false)
   */
  constructor(options: {
    readonly baseEstimator: Classifier;
    readonly threshold?: number;
    readonly maxIter?: number;
    readonly criterion?: "threshold" | "kBest";
    readonly kBest?: number;
    readonly verbose?: boolean;
  }) {
    SelfTrainingClassifier.checkBase(options.baseEstimator);
    this.baseEstimator = options.baseEstimator;
    this.threshold = checkThreshold(options.threshold ?? 0.75);
    this.maxIter = checkMaxIter(options.maxIter ?? 10);
    this.criterion = checkCriterion(options.criterion ?? "threshold");
    this.kBest = checkKBest(options.kBest ?? 10);
    this.verbose = options.verbose ?? false;
  }

  private static checkBase(value: unknown): void {
    const base = value as Partial<Classifier> | null | undefined;
    if (
      typeof base !== "object" ||
      base === null ||
      typeof base.fit !== "function" ||
      typeof base.predict !== "function" ||
      typeof base.getParams !== "function"
    ) {
      throw new InvalidParameterError(
        "baseEstimator must be a classifier with fit(), predict() and getParams()",
        "baseEstimator",
        value
      );
    }
  }

  /**
   * Fit the base classifier, adding confident pseudo-labels over several rounds.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @param y - Labels of shape (n_samples,), with -1 for unlabeled samples
   * @returns this
   * @throws {ShapeError} If X is not 2D, y is not 1D or their sample counts differ
   * @throws {DataValidationError} If X or y contain NaN/Inf
   * @throws {InvalidParameterError} If the labeled data holds fewer than 2 classes
   */
  fit(X: Tensor, y: Tensor): this {
    this.fitted = false;
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const nF = X.shape[1] ?? 0;
    const xData = Float64Array.from(toFloat64View(X));
    const labels = Float64Array.from(toFloat64View(y));

    const classes = [...new Set(labels)].filter((c) => c !== UNLABELED).sort((a, b) => a - b);
    if (classes.length < 2) {
      throw new InvalidParameterError(
        "SelfTrainingClassifier requires at least 2 classes in the labeled data",
        "y",
        classes.length
      );
    }

    const hasLabel = new Uint8Array(n);
    const labeledIter = new Int32Array(n).fill(-1);
    for (let i = 0; i < n; i++) {
      if ((labels[i] as number) !== UNLABELED) {
        hasLabel[i] = 1;
        labeledIter[i] = 0;
      }
    }
    let nLabeled = hasLabel.reduce((a, b) => a + b, 0);
    if (nLabeled === n) {
      warn("y contains no unlabeled samples", "UserWarning", "SelfTrainingClassifier.fit");
    }

    const estimator = this.cloneBase();
    let nIter = 0;
    let termination: SelfTrainingTermination | undefined;

    while (nLabeled < n && nIter < this.maxIter) {
      nIter++;
      const labeledRows: number[] = [];
      const unlabeledRows: number[] = [];
      for (let i = 0; i < n; i++) (hasLabel[i] ? labeledRows : unlabeledRows).push(i);

      estimator.fit(
        takeRows(xData, nF, labeledRows),
        tensor(Float64Array.from(labeledRows, (i) => labels[i] as number))
      );

      const { bestClass, bestProba } = this.confidentPredictions(
        estimator,
        takeRows(xData, nF, unlabeledRows),
        unlabeledRows.length,
        classes
      );

      const selected = this.select(bestProba);
      for (const u of selected) {
        const row = unlabeledRows[u] as number;
        labels[row] = bestClass[u] as number;
        hasLabel[row] = 1;
        labeledIter[row] = nIter;
      }
      nLabeled += selected.length;

      if (selected.length === 0) {
        termination = "no_change";
        if (this.verbose) {
          warn("SelfTrainingClassifier stopped: no new labels assigned", "ConvergenceWarning");
        }
        break;
      }
    }

    if (nIter === this.maxIter && termination === undefined) termination = "max_iter";
    if (nLabeled === n) termination = "all_labeled";

    const finalRows: number[] = [];
    for (let i = 0; i < n; i++) if (hasLabel[i]) finalRows.push(i);
    estimator.fit(
      takeRows(xData, nF, finalRows),
      tensor(Float64Array.from(finalRows, (i) => labels[i] as number))
    );

    this.estimator_ = estimator;
    this.classes_ = classes;
    this.nFeaturesIn_ = nF;
    this.nIter_ = nIter;
    this.labeledIter_ = labeledIter;
    this.transduction_ = labels;
    this.termination_ = termination ?? "all_labeled";
    this.fitted = true;
    return this;
  }

  /**
   * Predict with the final base estimator.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong dimensions or feature count
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.estimator_) {
      throw new NotFittedError("SelfTrainingClassifier must be fitted before predict");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "SelfTrainingClassifier");
    return this.estimator_.predict(X);
  }

  /**
   * Predict class probabilities with the final base estimator.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Probabilities of shape (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {InvalidParameterError} If the base estimator has no `predictProba`
   * @throws {ShapeError} If X has the wrong dimensions or feature count
   */
  predictProba(X: Tensor): Tensor {
    if (!this.fitted || !this.estimator_) {
      throw new NotFittedError("SelfTrainingClassifier must be fitted before predictProba");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "SelfTrainingClassifier");
    if (typeof this.estimator_.predictProba !== "function") {
      throw new InvalidParameterError(
        "Base estimator does not support predictProba",
        "baseEstimator",
        this.baseEstimator
      );
    }
    return this.estimator_.predictProba(X);
  }

  /**
   * Return the mean accuracy on the given test data and labels.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True labels of shape (n_samples,)
   * @returns Accuracy in [0, 1]
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-D or its length differs from the number of samples
   */
  score(X: Tensor, y: Tensor): number {
    return accuracy(this.predict(X), y);
  }

  /**
   * Sorted class labels of the labeled training data.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get classes(): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("SelfTrainingClassifier must be fitted to access classes");
    }
    return tensor(Float64Array.from(this.classes_));
  }

  /**
   * The fitted copy of the base estimator.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get estimator(): Classifier {
    if (!this.fitted || !this.estimator_) {
      throw new NotFittedError("SelfTrainingClassifier must be fitted to access estimator");
    }
    return this.estimator_;
  }

  /** Number of self-training iterations that ran during the last fit. */
  get nIterations(): number {
    return this.nIter_;
  }

  /**
   * Why the last fit stopped: "all_labeled", "no_change" or "max_iter".
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get terminationCondition(): SelfTrainingTermination {
    if (!this.fitted || this.termination_ === undefined) {
      throw new NotFittedError(
        "SelfTrainingClassifier must be fitted to access terminationCondition"
      );
    }
    return this.termination_;
  }

  /**
   * Label of every training sample after self-training; -1 where no label was assigned.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get transduction(): Tensor {
    if (!this.fitted || !this.transduction_) {
      throw new NotFittedError("SelfTrainingClassifier must be fitted to access transduction");
    }
    return tensor(Float64Array.from(this.transduction_));
  }

  /**
   * Iteration in which each training sample received its label: 0 for samples that were
   * labeled from the start, `k` for samples labeled in iteration `k`, and -1 for samples that
   * never got a label.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get labeledIterations(): Int32Array {
    if (!this.fitted || !this.labeledIter_) {
      throw new NotFittedError("SelfTrainingClassifier must be fitted to access labeledIterations");
    }
    return Int32Array.from(this.labeledIter_);
  }

  /**
   * Per-sample iteration record in the original encoding: -1 for samples labeled from the
   * start, -2 for samples that never got a label, otherwise the zero-based iteration.
   *
   * @deprecated The name is misleading (it holds iteration numbers, not labels). Use
   *   {@link labeledIterations} for the iteration of each sample and {@link transduction}
   *   for the labels.
   */
  get transductionLabels(): Int32Array | undefined {
    if (!this.labeledIter_) return undefined;
    return Int32Array.from(this.labeledIter_, (k) => (k === 0 ? -1 : k === -1 ? -2 : k - 1));
  }

  getParams(): Record<string, unknown> {
    return {
      baseEstimator: this.baseEstimator,
      threshold: this.threshold,
      maxIter: this.maxIter,
      criterion: this.criterion,
      kBest: this.kBest,
      verbose: this.verbose,
    };
  }

  /**
   * Set hyperparameters. Parameters of the base estimator can be set as
   * `baseEstimator__param`. Refit the model afterwards for them to take effect.
   *
   * @throws {InvalidParameterError} If a value is invalid or a key is unknown
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "baseEstimator":
          SelfTrainingClassifier.checkBase(value);
          this.baseEstimator = value as Classifier;
          break;
        case "threshold":
          this.threshold = checkThreshold(value);
          break;
        case "maxIter":
          this.maxIter = checkMaxIter(value);
          break;
        case "criterion":
          this.criterion = checkCriterion(value);
          break;
        case "kBest":
          this.kBest = checkKBest(value);
          break;
        case "verbose":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("verbose must be a boolean", "verbose", value);
          }
          this.verbose = value;
          break;
        default:
          if (key.startsWith("baseEstimator__")) {
            this.baseEstimator.setParams({ [key.slice("baseEstimator__".length)]: value });
          } else {
            throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
          }
      }
    }
    return this;
  }

  /**
   * Create an unfitted copy with a cloned base estimator.
   *
   * @returns A new SelfTrainingClassifier
   */
  clone(): SelfTrainingClassifier {
    return new SelfTrainingClassifier({
      baseEstimator: this.cloneBase(),
      threshold: this.threshold,
      maxIter: this.maxIter,
      criterion: this.criterion,
      kBest: this.kBest,
      verbose: this.verbose,
    });
  }

  /**
   * Predict the unlabeled rows and report each row's most likely class and its confidence.
   * Estimators without `predictProba` are treated as fully confident.
   */
  private confidentPredictions(
    estimator: Classifier,
    xUnlabeled: Tensor,
    nUnlabeled: number,
    classes: readonly number[]
  ): { bestClass: Float64Array; bestProba: Float64Array } {
    const bestClass = new Float64Array(nUnlabeled);
    const bestProba = new Float64Array(nUnlabeled);

    if (typeof estimator.predictProba !== "function") {
      const preds = toFloat64View(estimator.predict(xUnlabeled));
      for (let u = 0; u < nUnlabeled; u++) {
        bestClass[u] = preds[u] as number;
        bestProba[u] = 1;
      }
      return { bestClass, bestProba };
    }

    const proba = estimator.predictProba(xUnlabeled);
    const nCols = proba.ndim === 2 ? (proba.shape[1] ?? 0) : 0;
    // Columns are ordered like the estimator's own classes when it reports them.
    const reported = estimator.classes ? Array.from(toFloat64View(estimator.classes)) : undefined;
    const columnLabels = reported && reported.length === nCols ? reported : [...classes];
    if (columnLabels.length !== nCols || proba.shape[0] !== nUnlabeled) {
      throw new DataValidationError(
        `Base estimator predictProba returned shape [${proba.shape.join(", ")}]; ` +
          `expected [${nUnlabeled}, ${classes.length}]`
      );
    }
    const values = toFloat64View(proba);
    for (let u = 0; u < nUnlabeled; u++) {
      let best = 0;
      for (let c = 1; c < nCols; c++) {
        if ((values[u * nCols + c] as number) > (values[u * nCols + best] as number)) best = c;
      }
      bestClass[u] = columnLabels[best] as number;
      bestProba[u] = values[u * nCols + best] as number;
    }
    return { bestClass, bestProba };
  }

  /** Indices (into the unlabeled rows) of the pseudo-labels to accept. */
  private select(bestProba: Float64Array): number[] {
    const picked: number[] = [];
    if (this.criterion === "threshold") {
      for (let u = 0; u < bestProba.length; u++) {
        if ((bestProba[u] as number) > this.threshold) picked.push(u);
      }
      return picked;
    }
    const order = Array.from(bestProba.keys()).sort(
      (a, b) => (bestProba[b] as number) - (bestProba[a] as number) || a - b
    );
    return order.slice(0, Math.min(this.kBest, order.length));
  }

  /** Fresh unfitted copy of the base estimator, so fitting never touches the caller's instance. */
  private cloneBase(): Classifier {
    return cloneEstimator(this.baseEstimator, "SelfTrainingClassifier");
  }
}
