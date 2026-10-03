/**
 * @see {@link https://deepbox.dev/docs/ml-ensemble | Deepbox Ensemble Methods}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import {
  assertContiguous,
  toFloat64View,
  validateFitInputs,
  validatePredictInputs,
} from "../_validation";
import type { Classifier, Regressor } from "../base";
import { createTreeRng, DecisionTreeClassifier, DecisionTreeRegressor } from "../tree/DecisionTree";
import { predictRegressionTree } from "./_tree";

function floatMatrix(data: Float64Array, rows: number, cols: number): Tensor {
  return tensor(data, { dtype: "float64" }).reshape([rows, cols]);
}

function checkNEstimators(value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new InvalidParameterError("nEstimators must be an integer >= 1", "nEstimators", value);
  }
  return value;
}

function checkLearningRate(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new InvalidParameterError(
      "learningRate must be a finite number > 0",
      "learningRate",
      value
    );
  }
  return value;
}

function checkMaxDepth(value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new InvalidParameterError("maxDepth must be an integer >= 1", "maxDepth", value);
  }
  return value;
}

function checkRandomState(value: unknown): number | undefined {
  if (value !== undefined && (typeof value !== "number" || !Number.isFinite(value))) {
    throw new InvalidParameterError("randomState must be a finite number", "randomState", value);
  }
  return value;
}

function checkRegressionLoss(value: unknown): "linear" | "square" | "exponential" {
  if (value !== "linear" && value !== "square" && value !== "exponential") {
    throw new InvalidParameterError(
      `loss must be "linear", "square", or "exponential"`,
      "loss",
      value
    );
  }
  return value;
}

/** Throw unless every label is an integer that fits in int32 (the dtype of `predict`). */
function assertIntegerLabels(y: Float64Array, who: string): void {
  for (let i = 0; i < y.length; i++) {
    const v = y[i] as number;
    if (!Number.isInteger(v) || v < -2147483648 || v > 2147483647) {
      throw new DataValidationError(
        `${who} requires integer class labels in the int32 range; y[${i}] = ${v}`
      );
    }
  }
}

/** Shared `score` input checks; returns the targets as a flat array. */
function checkScoreTarget(y: Tensor, nPredicted: number): Float64Array {
  if (y.ndim !== 1) {
    throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
  }
  assertContiguous(y, "y");
  const yv = toFloat64View(y);
  for (let i = 0; i < yv.length; i++) {
    if (!Number.isFinite(yv[i])) {
      throw new DataValidationError("y contains non-finite values (NaN or Inf)");
    }
  }
  if (yv.length !== nPredicted) {
    throw new ShapeError(
      `X and y must have the same number of samples; got X=${nPredicted}, y=${yv.length}`
    );
  }
  if (yv.length === 0) {
    throw new DataValidationError("y must contain at least one sample");
  }
  return yv;
}

/**
 * Draw `n` row indices with probability proportional to `weights` (inverse-CDF
 * sampling with replacement). Rows with zero weight are never drawn.
 */
function weightedIndices(weights: Float64Array, rng: () => number): Int32Array {
  const n = weights.length;
  const cum = new Float64Array(n);
  let total = 0;
  for (let i = 0; i < n; i++) {
    total += weights[i] as number;
    cum[i] = total;
  }
  const out = new Int32Array(n);
  for (let s = 0; s < n; s++) {
    const r = rng() * total;
    let lo = 0;
    let hi = n - 1;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if ((cum[mid] as number) <= r) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    out[s] = lo;
  }
  return out;
}

/** Copy the rows `indices` of the row-major matrix `x` into a new float64 matrix tensor. */
function gatherRows(x: Float64Array, nFeatures: number, indices: Int32Array): Tensor {
  const out = new Float64Array(indices.length * nFeatures);
  for (let r = 0; r < indices.length; r++) {
    const src = (indices[r] as number) * nFeatures;
    out.set(x.subarray(src, src + nFeatures), r * nFeatures);
  }
  return floatMatrix(out, indices.length, nFeatures);
}

function gatherValues(v: Float64Array, indices: Int32Array): Tensor {
  const out = new Float64Array(indices.length);
  for (let r = 0; r < indices.length; r++) out[r] = v[indices[r] as number] as number;
  return tensor(out, { dtype: "float64" });
}

/**
 * Normalize `w` in place so it sums to one. Returns false, leaving `w` untouched, when the
 * total is zero or not finite (every weight underflowed), so boosting can stop cleanly.
 */
function normalizeWeights(w: Float64Array): boolean {
  let sum = 0;
  for (let i = 0; i < w.length; i++) sum += w[i] as number;
  if (!(sum > 0) || !Number.isFinite(sum)) return false;
  for (let i = 0; i < w.length; i++) w[i] = (w[i] as number) / sum;
  return true;
}

/** Weighted mean of the per-tree normalized feature importances. */
function weightedImportances(
  trees: ReadonlyArray<{ readonly featureImportances: Tensor }>,
  weights: readonly number[],
  nFeatures: number
): Tensor {
  const avg = new Float64Array(nFeatures);
  let totalWeight = 0;
  for (let m = 0; m < trees.length; m++) {
    const w = weights[m] ?? 0;
    totalWeight += w;
    const imp = toFloat64View((trees[m] as { featureImportances: Tensor }).featureImportances);
    for (let j = 0; j < nFeatures; j++) {
      avg[j] = (avg[j] as number) + w * (imp[j] as number);
    }
  }
  let total = 0;
  if (totalWeight > 0) {
    for (let j = 0; j < nFeatures; j++) {
      avg[j] = (avg[j] as number) / totalWeight;
      total += avg[j] as number;
    }
  }
  if (total > 0) {
    for (let j = 0; j < nFeatures; j++) avg[j] = (avg[j] as number) / total;
  }
  return tensor(avg);
}

/** Options of {@link AdaBoostClassifier}. */
export type AdaBoostClassifierOptions = {
  readonly nEstimators?: number;
  readonly learningRate?: number;
  readonly maxDepth?: number;
  readonly randomState?: number;
};

/** Options of {@link AdaBoostRegressor}. */
export type AdaBoostRegressorOptions = AdaBoostClassifierOptions & {
  readonly loss?: "linear" | "square" | "exponential";
};

function definedOptions<T>(params: Record<string, unknown>): T {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) out[key] = value;
  }
  return out as T;
}

/**
 * AdaBoost Classifier (Adaptive Boosting).
 *
 * Fits a sequence of weak classifiers (decision stumps by default) on
 * re-weighted versions of the data, then combines them via weighted majority vote.
 *
 * **Algorithm**: SAMME (Stagewise Additive Modeling using a Multi-class Exponential loss).
 * The decision tree does not take sample weights, so each round trains the tree on a
 * bootstrap sample drawn with probability proportional to the current weights, while the
 * weighted error and the weight update use the full training set.
 *
 * Fitting stops early when a round has zero weighted error (that tree gets weight 1) or
 * when a tree is no better than random guessing. Class labels must be integers.
 *
 * @example
 * ```ts
 * import { AdaBoostClassifier } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const clf = new AdaBoostClassifier({ nEstimators: 50, randomState: 0 });
 * clf.fit(X_train, y_train);
 * const predictions = clf.predict(X_test);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-ensemble | Deepbox Ensemble Methods}
 */
export class AdaBoostClassifier implements Classifier {
  private nEstimators: number;
  private learningRate: number;
  private maxDepth: number;
  private randomState: number | undefined;

  private estimators: DecisionTreeClassifier[] = [];
  private alphas: number[] = [];
  private classLabels: number[] = [];
  private nFeatures = 0;
  private fitted = false;

  /**
   * @param options.nEstimators - Maximum number of boosting rounds (default: 50)
   * @param options.learningRate - Shrinks the weight of each tree, must be > 0 (default: 1.0)
   * @param options.maxDepth - Depth of each tree; 1 gives decision stumps (default: 1)
   * @param options.randomState - Seed for the weighted bootstrap draws. Without it the global
   *   Deepbox generator is used, so `setSeed` also makes the fit reproducible.
   * @throws {InvalidParameterError} If an option is out of range
   */
  constructor(options: AdaBoostClassifierOptions = {}) {
    this.nEstimators = checkNEstimators(options.nEstimators ?? 50);
    this.learningRate = checkLearningRate(options.learningRate ?? 1.0);
    this.maxDepth = checkMaxDepth(options.maxDepth ?? 1);
    this.randomState = checkRandomState(options.randomState);
  }

  /**
   * Fit the ensemble. Any previously fitted trees are discarded.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Integer class labels of shape (n_samples,), with at least two classes
   * @returns this
   * @throws {ShapeError} If X is not 2D, y is not 1D or the sample counts differ
   * @throws {DataValidationError} If X or y contain NaN/Inf or y has non-integer labels
   * @throws {InvalidParameterError} If y contains a single class
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const xv = toFloat64View(X);
    const yv = toFloat64View(y);
    assertIntegerLabels(yv, "AdaBoostClassifier");

    const classLabels = [...new Set(yv)].sort((a, b) => a - b);
    const nClasses = classLabels.length;
    if (nClasses < 2) {
      throw new InvalidParameterError(
        "AdaBoostClassifier requires at least 2 classes",
        "y",
        nClasses
      );
    }

    const rng = createTreeRng(this.randomState);
    const weights = new Float64Array(nSamples).fill(1 / nSamples);
    const estimators: DecisionTreeClassifier[] = [];
    const estimatorWeights: number[] = [];

    for (let m = 0; m < this.nEstimators; m++) {
      const idx = weightedIndices(weights, rng);
      const tree = new DecisionTreeClassifier({
        maxDepth: this.maxDepth,
        minSamplesSplit: 2,
        minSamplesLeaf: 1,
      });
      tree.fit(gatherRows(xv, nFeatures, idx), gatherValues(yv, idx));

      const pred = toFloat64View(tree.predict(X));
      let error = 0;
      for (let i = 0; i < nSamples; i++) {
        if (pred[i] !== yv[i]) error += weights[i] as number;
      }

      if (error <= 0) {
        // A perfect round: nothing left to re-weight, so stop with this tree at weight 1.
        estimators.push(tree);
        estimatorWeights.push(1.0);
        break;
      }
      if (error >= 1 - 1 / nClasses) {
        // No better than random guessing: drop the tree, but never end up with none.
        if (estimators.length === 0) {
          estimators.push(tree);
          estimatorWeights.push(1.0);
        }
        break;
      }

      const alpha = this.learningRate * (Math.log((1 - error) / error) + Math.log(nClasses - 1));
      estimators.push(tree);
      estimatorWeights.push(alpha);

      // Multiply misclassified weights by exp(alpha). Scaling the correct ones by
      // exp(-alpha) instead gives the same weights after normalization and cannot overflow.
      const keep = Math.exp(-alpha);
      for (let i = 0; i < nSamples; i++) {
        if (pred[i] === yv[i]) weights[i] = (weights[i] as number) * keep;
      }
      if (!normalizeWeights(weights)) break;
    }

    this.estimators = estimators;
    this.alphas = estimatorWeights;
    this.classLabels = classLabels;
    this.nFeatures = nFeatures;
    this.fitted = true;
    return this;
  }

  private checkFitted(what: string): void {
    if (!this.fitted) {
      throw new NotFittedError(`AdaBoostClassifier must be fitted before ${what}`);
    }
  }

  /** Weighted votes per class, flattened row-major to (n, nClasses). */
  private votes(X: Tensor): Float64Array {
    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classLabels.length;
    const classIndex = new Map<number, number>();
    this.classLabels.forEach((label, idx) => {
      classIndex.set(label, idx);
    });
    const votes = new Float64Array(nSamples * nClasses);
    for (let m = 0; m < this.estimators.length; m++) {
      const pred = toFloat64View((this.estimators[m] as DecisionTreeClassifier).predict(X));
      const w = this.alphas[m] as number;
      for (let i = 0; i < nSamples; i++) {
        const c = classIndex.get(pred[i] as number);
        if (c !== undefined) {
          const at = i * nClasses + c;
          votes[at] = (votes[at] as number) + w;
        }
      }
    }
    return votes;
  }

  /**
   * Predict class labels by weighted majority vote (ties go to the smallest label).
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Int32 labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong rank or feature count
   */
  predict(X: Tensor): Tensor {
    this.checkFitted("prediction");
    validatePredictInputs(X, this.nFeatures, "AdaBoostClassifier");
    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classLabels.length;
    const votes = this.votes(X);
    const out = new Int32Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      let best = 0;
      let bestScore = -Infinity;
      for (let c = 0; c < nClasses; c++) {
        const s = votes[i * nClasses + c] as number;
        if (s > bestScore) {
          bestScore = s;
          best = c;
        }
      }
      out[i] = this.classLabels[best] as number;
    }
    return tensor(out, { dtype: "int32" });
  }

  /**
   * Decision function of the SAMME ensemble, following scikit-learn.
   *
   * For class k the score is `sum_m w_m * (pred_m == k ? 1 : -1/(K-1)) / sum_m w_m`, in
   * [-1, 1]. With two classes the result is the single score of the second class minus
   * that of the first, in [-2, 2] (positive means `classes[1]`).
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Shape (n_samples,) for two classes, otherwise (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong rank or feature count
   */
  decisionFunction(X: Tensor): Tensor {
    this.checkFitted("computing the decision function");
    validatePredictInputs(X, this.nFeatures, "AdaBoostClassifier");
    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classLabels.length;
    const votes = this.votes(X);
    let totalWeight = 0;
    for (const w of this.alphas) totalWeight += w;

    const dec = new Float64Array(nSamples * nClasses);
    const scale = nClasses / (nClasses - 1);
    for (let i = 0; i < dec.length; i++) {
      dec[i] = ((votes[i] as number) * scale - totalWeight / (nClasses - 1)) / totalWeight;
    }
    if (nClasses === 2) {
      const out = new Float64Array(nSamples);
      for (let i = 0; i < nSamples; i++) {
        out[i] = (dec[2 * i + 1] as number) - (dec[2 * i] as number);
      }
      return tensor(out);
    }
    return floatMatrix(dec, nSamples, nClasses);
  }

  /**
   * Predict class probabilities, following scikit-learn: the softmax of the
   * decision function divided by `K - 1` (for two classes, `sigmoid(decision)`).
   * Because the weights are normalized, the probabilities stay close to uniform even for
   * a confident ensemble; they are a ranking score, not a calibrated estimate.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Probabilities of shape (n_samples, n_classes), columns ordered like `classes`
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong rank or feature count
   */
  predictProba(X: Tensor): Tensor {
    this.checkFitted("prediction");
    validatePredictInputs(X, this.nFeatures, "AdaBoostClassifier");
    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classLabels.length;
    const dec = toFloat64View(this.decisionFunction(X));
    const out = new Float64Array(nSamples * nClasses);

    if (nClasses === 2) {
      for (let i = 0; i < nSamples; i++) {
        const d = dec[i] as number;
        // softmax([-d, d] / 2) written as two sigmoids so that neither side loses precision.
        out[2 * i] = 1 / (1 + Math.exp(d));
        out[2 * i + 1] = 1 / (1 + Math.exp(-d));
      }
    } else {
      const denom = nClasses - 1;
      for (let i = 0; i < nSamples; i++) {
        let max = -Infinity;
        for (let c = 0; c < nClasses; c++) {
          max = Math.max(max, (dec[i * nClasses + c] as number) / denom);
        }
        let sum = 0;
        for (let c = 0; c < nClasses; c++) {
          const e = Math.exp((dec[i * nClasses + c] as number) / denom - max);
          out[i * nClasses + c] = e;
          sum += e;
        }
        for (let c = 0; c < nClasses; c++) {
          out[i * nClasses + c] = (out[i * nClasses + c] as number) / sum;
        }
      }
    }
    return floatMatrix(out, nSamples, nClasses);
  }

  /**
   * Mean accuracy on the given data.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True labels of shape (n_samples,)
   * @returns Accuracy in [0, 1]
   * @throws {ShapeError} If y is not 1D or its length differs from the number of rows in X
   * @throws {DataValidationError} If y is empty or contains NaN/Inf
   */
  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    const predictions = this.predict(X);
    const yv = checkScoreTarget(y, predictions.size);
    const pv = toFloat64View(predictions);
    let correct = 0;
    for (let i = 0; i < yv.length; i++) {
      if (pv[i] === yv[i]) correct++;
    }
    return correct / yv.length;
  }

  /** Sorted class labels seen during fit, or `undefined` before fitting. */
  get classes(): Tensor | undefined {
    if (!this.fitted || this.classLabels.length === 0) return undefined;
    return tensor(this.classLabels, { dtype: "int32" });
  }

  /** Number of trees kept by the last fit (can be below `nEstimators` after an early stop). */
  get nEstimatorsFitted(): number {
    return this.estimators.length;
  }

  /**
   * Weight of each fitted tree in the vote.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get estimatorWeights(): Tensor {
    this.checkFitted("accessing the estimator weights");
    return tensor(Float64Array.from(this.alphas));
  }

  /**
   * Feature importances: the mean of the trees' importances weighted by the tree
   * weights, normalized to sum to one.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get featureImportances(): Tensor {
    if (!this.fitted || this.estimators.length === 0 || this.nFeatures === 0) {
      throw new NotFittedError("AdaBoostClassifier must be fitted to access feature_importances_");
    }
    return weightedImportances(this.estimators, this.alphas, this.nFeatures);
  }

  getParams(): Record<string, unknown> {
    return {
      nEstimators: this.nEstimators,
      learningRate: this.learningRate,
      maxDepth: this.maxDepth,
      randomState: this.randomState,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nEstimators":
          this.nEstimators = checkNEstimators(value);
          break;
        case "learningRate":
          this.learningRate = checkLearningRate(value);
          break;
        case "maxDepth":
          this.maxDepth = checkMaxDepth(value);
          break;
        case "randomState":
          this.randomState = checkRandomState(value);
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  /** Create an unfitted copy with the same hyperparameters. */
  clone(): AdaBoostClassifier {
    return new AdaBoostClassifier(definedOptions<AdaBoostClassifierOptions>(this.getParams()));
  }
}

/**
 * AdaBoost Regressor (Adaptive Boosting for Regression).
 *
 * Uses the AdaBoost.R2 algorithm with decision tree regressors as base estimators and
 * predicts the weighted median of the trees' predictions. As in the classifier, each
 * tree is trained on a bootstrap sample drawn with probability proportional to the
 * current sample weights.
 *
 * @example
 * ```ts
 * import { AdaBoostRegressor } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const reg = new AdaBoostRegressor({ nEstimators: 50, randomState: 0 });
 * reg.fit(X_train, y_train);
 * const predictions = reg.predict(X_test);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-ensemble | Deepbox Ensemble Methods}
 */
export class AdaBoostRegressor implements Regressor {
  private nEstimators: number;
  private learningRate: number;
  private maxDepth: number;
  private loss: "linear" | "square" | "exponential";
  private randomState: number | undefined;

  private estimators: DecisionTreeRegressor[] = [];
  private alphas: number[] = [];
  private nFeatures = 0;
  private fitted = false;

  /**
   * @param options.nEstimators - Maximum number of boosting rounds (default: 50)
   * @param options.learningRate - Shrinks the weight of each tree, must be > 0 (default: 1.0)
   * @param options.maxDepth - Depth of each tree (default: 3)
   * @param options.loss - Loss applied to the normalized absolute errors:
   *   `"linear"`, `"square"` or `"exponential"` (default: "linear")
   * @param options.randomState - Seed for the weighted bootstrap draws. Without it the global
   *   Deepbox generator is used, so `setSeed` also makes the fit reproducible.
   * @throws {InvalidParameterError} If an option is out of range
   */
  constructor(options: AdaBoostRegressorOptions = {}) {
    this.nEstimators = checkNEstimators(options.nEstimators ?? 50);
    this.learningRate = checkLearningRate(options.learningRate ?? 1.0);
    this.maxDepth = checkMaxDepth(options.maxDepth ?? 3);
    this.loss = checkRegressionLoss(options.loss ?? "linear");
    this.randomState = checkRandomState(options.randomState);
  }

  /**
   * Fit the ensemble. Any previously fitted trees are discarded.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Targets of shape (n_samples,)
   * @returns this
   * @throws {ShapeError} If X is not 2D, y is not 1D or the sample counts differ
   * @throws {DataValidationError} If X or y contain NaN/Inf
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const xv = toFloat64View(X);
    const yv = toFloat64View(y);

    const rng = createTreeRng(this.randomState);
    const weights = new Float64Array(nSamples).fill(1 / nSamples);
    const losses = new Float64Array(nSamples);
    const estimators: DecisionTreeRegressor[] = [];
    const estimatorWeights: number[] = [];

    for (let m = 0; m < this.nEstimators; m++) {
      const idx = weightedIndices(weights, rng);
      const tree = new DecisionTreeRegressor({
        maxDepth: this.maxDepth,
        minSamplesSplit: 2,
        minSamplesLeaf: 1,
      });
      tree.fit(gatherRows(xv, nFeatures, idx), gatherValues(yv, idx));

      const pred = predictRegressionTree(tree, xv, nSamples, nFeatures);
      let maxError = 0;
      for (let i = 0; i < nSamples; i++) {
        const err = Math.abs((yv[i] as number) - (pred[i] as number));
        losses[i] = err;
        if (err > maxError) maxError = err;
      }

      if (maxError === 0) {
        // A perfect fit on every training row.
        estimators.push(tree);
        estimatorWeights.push(1.0);
        break;
      }

      let avgLoss = 0;
      for (let i = 0; i < nSamples; i++) {
        const e = (losses[i] as number) / maxError;
        const loss = this.loss === "linear" ? e : this.loss === "square" ? e * e : 1 - Math.exp(-e);
        losses[i] = loss;
        avgLoss += (weights[i] as number) * loss;
      }

      if (avgLoss <= 0) {
        // Every row that carries weight is fitted exactly.
        estimators.push(tree);
        estimatorWeights.push(1.0);
        break;
      }
      if (avgLoss >= 0.5) {
        // Worse than the R2 threshold: drop the tree, but never end up with none.
        if (estimators.length === 0) {
          estimators.push(tree);
          estimatorWeights.push(1.0);
        }
        break;
      }

      const beta = avgLoss / (1 - avgLoss);
      estimators.push(tree);
      estimatorWeights.push(this.learningRate * Math.log(1 / beta));

      // w_i <- w_i * beta^((1 - loss_i) * learningRate), as in scikit-learn.
      for (let i = 0; i < nSamples; i++) {
        weights[i] =
          (weights[i] as number) * beta ** ((1 - (losses[i] as number)) * this.learningRate);
      }
      if (!normalizeWeights(weights)) break;
    }

    this.estimators = estimators;
    this.alphas = estimatorWeights;
    this.nFeatures = nFeatures;
    this.fitted = true;
    return this;
  }

  /**
   * Predict targets as the weighted median of the trees' predictions.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predictions of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong rank or feature count
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("AdaBoostRegressor must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "AdaBoostRegressor");

    const nSamples = X.shape[0] ?? 0;
    const xv = toFloat64View(X);
    const nTrees = this.estimators.length;
    const all = new Float64Array(nTrees * nSamples);
    for (let m = 0; m < nTrees; m++) {
      all.set(
        predictRegressionTree(
          this.estimators[m] as DecisionTreeRegressor,
          xv,
          nSamples,
          this.nFeatures
        ),
        m * nSamples
      );
    }
    let totalWeight = 0;
    for (const w of this.alphas) totalWeight += w;
    const half = totalWeight / 2;

    const out = new Float64Array(nSamples);
    const order = new Array<number>(nTrees);
    for (let i = 0; i < nSamples; i++) {
      for (let m = 0; m < nTrees; m++) order[m] = m;
      order.sort((a, b) => (all[a * nSamples + i] as number) - (all[b * nSamples + i] as number));
      let cum = 0;
      let median = all[(order[0] as number) * nSamples + i] as number;
      for (let k = 0; k < nTrees; k++) {
        const m = order[k] as number;
        cum += this.alphas[m] as number;
        if (cum >= half) {
          median = all[m * nSamples + i] as number;
          break;
        }
      }
      out[i] = median;
    }
    return tensor(out);
  }

  /**
   * Coefficient of determination R^2 on the given data.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True targets of shape (n_samples,)
   * @returns R^2 (1.0 is perfect, it can be negative)
   * @throws {ShapeError} If y is not 1D or its length differs from the number of rows in X
   * @throws {DataValidationError} If y is empty or contains NaN/Inf
   */
  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    const predictions = this.predict(X);
    const yv = checkScoreTarget(y, predictions.size);
    const pv = toFloat64View(predictions);
    let mean = 0;
    for (let i = 0; i < yv.length; i++) mean += yv[i] as number;
    mean /= yv.length;
    let ssRes = 0;
    let ssTot = 0;
    for (let i = 0; i < yv.length; i++) {
      const yi = yv[i] as number;
      ssRes += (yi - (pv[i] as number)) ** 2;
      ssTot += (yi - mean) ** 2;
    }
    return ssTot === 0 ? (ssRes === 0 ? 1.0 : 0.0) : 1 - ssRes / ssTot;
  }

  /** Number of trees kept by the last fit (can be below `nEstimators` after an early stop). */
  get nEstimatorsFitted(): number {
    return this.estimators.length;
  }

  /**
   * Weight of each fitted tree in the weighted median.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get estimatorWeights(): Tensor {
    if (!this.fitted) {
      throw new NotFittedError(
        "AdaBoostRegressor must be fitted before accessing the estimator weights"
      );
    }
    return tensor(Float64Array.from(this.alphas));
  }

  /**
   * Feature importances: the mean of the trees' importances weighted by the tree
   * weights, normalized to sum to one.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get featureImportances(): Tensor {
    if (!this.fitted || this.estimators.length === 0 || this.nFeatures === 0) {
      throw new NotFittedError("AdaBoostRegressor must be fitted to access feature_importances_");
    }
    return weightedImportances(this.estimators, this.alphas, this.nFeatures);
  }

  getParams(): Record<string, unknown> {
    return {
      nEstimators: this.nEstimators,
      learningRate: this.learningRate,
      maxDepth: this.maxDepth,
      loss: this.loss,
      randomState: this.randomState,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nEstimators":
          this.nEstimators = checkNEstimators(value);
          break;
        case "learningRate":
          this.learningRate = checkLearningRate(value);
          break;
        case "maxDepth":
          this.maxDepth = checkMaxDepth(value);
          break;
        case "loss":
          this.loss = checkRegressionLoss(value);
          break;
        case "randomState":
          this.randomState = checkRandomState(value);
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  /** Create an unfitted copy with the same hyperparameters. */
  clone(): AdaBoostRegressor {
    return new AdaBoostRegressor(definedOptions<AdaBoostRegressorOptions>(this.getParams()));
  }
}
