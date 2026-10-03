/**
 * @see {@link https://deepbox.dev/docs/ml-ensemble | Deepbox Ensemble Methods}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __randomBelow } from "../../random/random";
import {
  assertContiguous,
  toFloat64View,
  validateFitInputs,
  validatePredictInputs,
} from "../_validation";
import type { Classifier, Regressor } from "../base";
import { createTreeRng, DecisionTreeClassifier, DecisionTreeRegressor } from "../tree/DecisionTree";
import { predictRegressionTree } from "./_tree";

/** Largest depth handed to the base trees when `maxDepth` is `Infinity`. */
const UNLIMITED_DEPTH = Number.MAX_SAFE_INTEGER;

/** Options shared by {@link BaggingClassifier} and {@link BaggingRegressor}. */
export type BaggingOptions = {
  readonly nEstimators?: number;
  readonly maxSamples?: number;
  readonly maxFeatures?: number;
  readonly bootstrap?: boolean;
  readonly maxDepth?: number;
  readonly randomState?: number;
};

function floatMatrix(data: Float64Array, rows: number, cols: number): Tensor {
  return tensor(data, { dtype: "float64" }).reshape([rows, cols]);
}

function checkNEstimators(value: unknown): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new InvalidParameterError("nEstimators must be an integer >= 1", "nEstimators", value);
  }
  return value;
}

function checkFraction(name: "maxSamples" | "maxFeatures", value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0 || value > 1) {
    throw new InvalidParameterError(`${name} must be in (0, 1]`, name, value);
  }
  return value;
}

function checkBootstrap(value: unknown): boolean {
  if (typeof value !== "boolean") {
    throw new InvalidParameterError("bootstrap must be a boolean", "bootstrap", value);
  }
  return value;
}

function checkMaxDepth(value: unknown): number {
  if (
    typeof value !== "number" ||
    (value !== Infinity && (!Number.isInteger(value) || value < 1))
  ) {
    throw new InvalidParameterError(
      "maxDepth must be an integer >= 1 or Infinity",
      "maxDepth",
      value
    );
  }
  return value;
}

function checkRandomState(value: unknown): number | undefined {
  if (value !== undefined && (typeof value !== "number" || !Number.isFinite(value))) {
    throw new InvalidParameterError("randomState must be a finite number", "randomState", value);
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

/** Shared `score` input checks; returns the labels as a flat array. */
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

function accuracy(pred: Tensor, y: Float64Array): number {
  const pv = toFloat64View(pred);
  let correct = 0;
  for (let i = 0; i < y.length; i++) {
    if (pv[i] === y[i]) correct++;
  }
  return correct / y.length;
}

function r2(pred: Tensor, y: Float64Array): number {
  const pv = toFloat64View(pred);
  let mean = 0;
  for (let i = 0; i < y.length; i++) mean += y[i] as number;
  mean /= y.length;
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < y.length; i++) {
    const yi = y[i] as number;
    ssRes += (yi - (pv[i] as number)) ** 2;
    ssTot += (yi - mean) ** 2;
  }
  return ssTot === 0 ? (ssRes === 0 ? 1.0 : 0.0) : 1 - ssRes / ssTot;
}

/**
 * Draw the feature subset and the row sample for one ensemble member.
 *
 * Features are always drawn without replacement. Rows are drawn with replacement
 * when `bootstrap` is true and without replacement otherwise.
 */
function drawSubsets(
  rng: () => number,
  nSamples: number,
  nFeatures: number,
  nSamplesDraw: number,
  nFeaturesDraw: number,
  bootstrap: boolean
): { rows: Int32Array; features: Int32Array } {
  const features = new Int32Array(nFeatures);
  for (let i = 0; i < nFeatures; i++) features[i] = i;
  if (nFeaturesDraw < nFeatures) {
    for (let i = 0; i < nFeaturesDraw; i++) {
      const j = i + __randomBelow(rng, nFeatures - i);
      const tmp = features[i] as number;
      features[i] = features[j] as number;
      features[j] = tmp;
    }
  }
  const chosenFeatures = features.slice(0, nFeaturesDraw).sort();

  const rows = new Int32Array(nSamplesDraw);
  if (bootstrap) {
    for (let s = 0; s < nSamplesDraw; s++) rows[s] = __randomBelow(rng, nSamples);
  } else {
    const all = new Int32Array(nSamples);
    for (let i = 0; i < nSamples; i++) all[i] = i;
    for (let i = 0; i < nSamplesDraw; i++) {
      const j = i + __randomBelow(rng, nSamples - i);
      const tmp = all[i] as number;
      all[i] = all[j] as number;
      all[j] = tmp;
    }
    rows.set(all.subarray(0, nSamplesDraw));
  }
  return { rows, features: chosenFeatures };
}

/** Gather `rows` x `features` of the row-major matrix `x` (with `nFeatures` columns). */
function gather(
  x: Float64Array,
  nFeatures: number,
  rows: Int32Array | null,
  nRows: number,
  features: Int32Array
): Float64Array {
  const k = features.length;
  const out = new Float64Array(nRows * k);
  for (let r = 0; r < nRows; r++) {
    const src = (rows === null ? r : (rows[r] as number)) * nFeatures;
    const dst = r * k;
    for (let c = 0; c < k; c++) out[dst + c] = x[src + (features[c] as number)] as number;
  }
  return out;
}

/** Tree options for a base learner; `Infinity` maps to a depth no tree can reach. */
function treeOptions(maxDepth: number): {
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
} {
  return {
    maxDepth: Number.isFinite(maxDepth) ? maxDepth : UNLIMITED_DEPTH,
    minSamplesSplit: 2,
    minSamplesLeaf: 1,
  };
}

function definedOptions(params: Record<string, unknown>): BaggingOptions {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) out[key] = value;
  }
  return out as BaggingOptions;
}

/**
 * Bagging Classifier (Bootstrap Aggregating).
 *
 * Trains decision tree classifiers on random subsets of the rows and features and
 * averages their predicted class probabilities. `predict` returns the class with the
 * largest averaged probability, as scikit-learn's `BaggingClassifier` does.
 *
 * Class labels must be integers. Fitting again discards the previous ensemble.
 *
 * @example
 * ```ts
 * import { BaggingClassifier } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const clf = new BaggingClassifier({ nEstimators: 10, randomState: 0 });
 * clf.fit(X_train, y_train);
 * const predictions = clf.predict(X_test);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-ensemble | Deepbox Ensemble Methods}
 */
export class BaggingClassifier implements Classifier {
  private nEstimators: number;
  private maxSamples: number;
  private maxFeatures: number;
  private bootstrap: boolean;
  private maxDepth: number;
  private randomState: number | undefined;

  private estimators: DecisionTreeClassifier[] = [];
  private featureIndices: Int32Array[] = [];
  private classLabels: number[] = [];
  private nFeatures = 0;
  private fitted = false;

  /**
   * The number of rows and features drawn is `max(1, floor(fraction * n))`, as in scikit-learn.
   *
   * @param options.nEstimators - Number of base estimators (default: 10)
   * @param options.maxSamples - Fraction of rows drawn for each estimator, in (0, 1] (default: 1.0)
   * @param options.maxFeatures - Fraction of features drawn for each estimator, in (0, 1] (default: 1.0)
   * @param options.bootstrap - Draw rows with replacement (default: true)
   * @param options.maxDepth - Maximum depth of each tree; `Infinity` grows full trees (default: Infinity)
   * @param options.randomState - Seed for the row and feature draws. Without it the global
   *   Deepbox generator is used, so `setSeed` also makes the fit reproducible.
   * @throws {InvalidParameterError} If an option is out of range
   */
  constructor(options: BaggingOptions = {}) {
    this.nEstimators = checkNEstimators(options.nEstimators ?? 10);
    this.maxSamples = checkFraction("maxSamples", options.maxSamples ?? 1.0);
    this.maxFeatures = checkFraction("maxFeatures", options.maxFeatures ?? 1.0);
    this.bootstrap = checkBootstrap(options.bootstrap ?? true);
    this.maxDepth = checkMaxDepth(options.maxDepth ?? Infinity);
    this.randomState = checkRandomState(options.randomState);
  }

  /**
   * Fit the ensemble.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Integer class labels of shape (n_samples,)
   * @returns this
   * @throws {ShapeError} If X is not 2D, y is not 1D or the sample counts differ
   * @throws {DataValidationError} If X or y contain NaN/Inf or y has non-integer labels
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const xv = toFloat64View(X);
    const yv = toFloat64View(y);
    assertIntegerLabels(yv, "BaggingClassifier");
    const classLabels = [...new Set(yv)].sort((a, b) => a - b);

    const nSamplesDraw = Math.max(1, Math.floor(this.maxSamples * nSamples));
    const nFeaturesDraw = Math.max(1, Math.floor(this.maxFeatures * nFeatures));
    const rng = createTreeRng(this.randomState);

    const estimators: DecisionTreeClassifier[] = [];
    const featureIndices: Int32Array[] = [];
    for (let m = 0; m < this.nEstimators; m++) {
      const { rows, features } = drawSubsets(
        rng,
        nSamples,
        nFeatures,
        nSamplesDraw,
        nFeaturesDraw,
        this.bootstrap
      );
      const xSub = gather(xv, nFeatures, rows, nSamplesDraw, features);
      const ySub = new Float64Array(nSamplesDraw);
      for (let r = 0; r < nSamplesDraw; r++) ySub[r] = yv[rows[r] as number] as number;

      const tree = new DecisionTreeClassifier(treeOptions(this.maxDepth));
      tree.fit(
        floatMatrix(xSub, nSamplesDraw, features.length),
        tensor(ySub, { dtype: "float64" })
      );
      estimators.push(tree);
      featureIndices.push(features);
    }

    this.estimators = estimators;
    this.featureIndices = featureIndices;
    this.classLabels = classLabels;
    this.nFeatures = nFeatures;
    this.fitted = true;
    return this;
  }

  private checkFitted(): void {
    if (!this.fitted) {
      throw new NotFittedError("BaggingClassifier must be fitted before prediction");
    }
  }

  /** Average of the trees' class probabilities, flattened row-major to (n, nClasses). */
  private averageProba(X: Tensor): Float64Array {
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = this.nFeatures;
    const nClasses = this.classLabels.length;
    const xv = toFloat64View(X);
    const classIndex = new Map<number, number>();
    this.classLabels.forEach((label, idx) => {
      classIndex.set(label, idx);
    });

    const sum = new Float64Array(nSamples * nClasses);
    for (let m = 0; m < this.estimators.length; m++) {
      const tree = this.estimators[m] as DecisionTreeClassifier;
      const features = this.featureIndices[m] as Int32Array;
      const xSub = floatMatrix(
        gather(xv, nFeatures, null, nSamples, features),
        nSamples,
        features.length
      );
      const proba = toFloat64View(tree.predictProba(xSub));
      const treeClasses = tree.classes;
      if (!treeClasses) continue;
      const labels = toFloat64View(treeClasses);
      const k = labels.length;
      for (let j = 0; j < k; j++) {
        const col = classIndex.get(labels[j] as number);
        if (col === undefined) continue;
        for (let i = 0; i < nSamples; i++) {
          const at = i * nClasses + col;
          sum[at] = (sum[at] as number) + (proba[i * k + j] as number);
        }
      }
    }
    const inv = 1 / this.estimators.length;
    for (let i = 0; i < sum.length; i++) sum[i] = (sum[i] as number) * inv;
    return sum;
  }

  /**
   * Predict class labels: the class with the largest averaged tree probability
   * (ties go to the smallest label).
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Int32 labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong rank or feature count
   */
  predict(X: Tensor): Tensor {
    this.checkFitted();
    validatePredictInputs(X, this.nFeatures, "BaggingClassifier");
    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classLabels.length;
    const proba = this.averageProba(X);
    const out = new Int32Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      let best = 0;
      let bestP = -Infinity;
      for (let c = 0; c < nClasses; c++) {
        const p = proba[i * nClasses + c] as number;
        if (p > bestP) {
          bestP = p;
          best = c;
        }
      }
      out[i] = this.classLabels[best] as number;
    }
    return tensor(out, { dtype: "int32" });
  }

  /**
   * Predict class probabilities: the mean of the trees' `predictProba`, with columns
   * ordered like {@link BaggingClassifier.classes}.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Probabilities of shape (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong rank or feature count
   */
  predictProba(X: Tensor): Tensor {
    this.checkFitted();
    validatePredictInputs(X, this.nFeatures, "BaggingClassifier");
    const nSamples = X.shape[0] ?? 0;
    return floatMatrix(this.averageProba(X), nSamples, this.classLabels.length);
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
    return accuracy(predictions, checkScoreTarget(y, predictions.size));
  }

  /** Sorted class labels seen during fit, or `undefined` before fitting. */
  get classes(): Tensor | undefined {
    if (!this.fitted) return undefined;
    return tensor(this.classLabels, { dtype: "int32" });
  }

  getParams(): Record<string, unknown> {
    return {
      nEstimators: this.nEstimators,
      maxSamples: this.maxSamples,
      maxFeatures: this.maxFeatures,
      bootstrap: this.bootstrap,
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
        case "maxSamples":
          this.maxSamples = checkFraction("maxSamples", value);
          break;
        case "maxFeatures":
          this.maxFeatures = checkFraction("maxFeatures", value);
          break;
        case "bootstrap":
          this.bootstrap = checkBootstrap(value);
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
  clone(): BaggingClassifier {
    return new BaggingClassifier(definedOptions(this.getParams()));
  }
}

/**
 * Bagging Regressor (Bootstrap Aggregating for Regression).
 *
 * Trains decision tree regressors on random subsets of the rows and features and
 * averages their predictions. Fitting again discards the previous ensemble.
 *
 * @example
 * ```ts
 * import { BaggingRegressor } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const reg = new BaggingRegressor({ nEstimators: 10, randomState: 0 });
 * reg.fit(X_train, y_train);
 * const predictions = reg.predict(X_test);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-ensemble | Deepbox Ensemble Methods}
 */
export class BaggingRegressor implements Regressor {
  private nEstimators: number;
  private maxSamples: number;
  private maxFeatures: number;
  private bootstrap: boolean;
  private maxDepth: number;
  private randomState: number | undefined;

  private estimators: DecisionTreeRegressor[] = [];
  private featureIndices: Int32Array[] = [];
  private nFeatures = 0;
  private fitted = false;

  /**
   * @param options - Same options as {@link BaggingClassifier}
   * @throws {InvalidParameterError} If an option is out of range
   */
  constructor(options: BaggingOptions = {}) {
    this.nEstimators = checkNEstimators(options.nEstimators ?? 10);
    this.maxSamples = checkFraction("maxSamples", options.maxSamples ?? 1.0);
    this.maxFeatures = checkFraction("maxFeatures", options.maxFeatures ?? 1.0);
    this.bootstrap = checkBootstrap(options.bootstrap ?? true);
    this.maxDepth = checkMaxDepth(options.maxDepth ?? Infinity);
    this.randomState = checkRandomState(options.randomState);
  }

  /**
   * Fit the ensemble.
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

    const nSamplesDraw = Math.max(1, Math.floor(this.maxSamples * nSamples));
    const nFeaturesDraw = Math.max(1, Math.floor(this.maxFeatures * nFeatures));
    const rng = createTreeRng(this.randomState);

    const estimators: DecisionTreeRegressor[] = [];
    const featureIndices: Int32Array[] = [];
    for (let m = 0; m < this.nEstimators; m++) {
      const { rows, features } = drawSubsets(
        rng,
        nSamples,
        nFeatures,
        nSamplesDraw,
        nFeaturesDraw,
        this.bootstrap
      );
      const xSub = gather(xv, nFeatures, rows, nSamplesDraw, features);
      const ySub = new Float64Array(nSamplesDraw);
      for (let r = 0; r < nSamplesDraw; r++) ySub[r] = yv[rows[r] as number] as number;

      const tree = new DecisionTreeRegressor(treeOptions(this.maxDepth));
      tree.fit(
        floatMatrix(xSub, nSamplesDraw, features.length),
        tensor(ySub, { dtype: "float64" })
      );
      estimators.push(tree);
      featureIndices.push(features);
    }

    this.estimators = estimators;
    this.featureIndices = featureIndices;
    this.nFeatures = nFeatures;
    this.fitted = true;
    return this;
  }

  /**
   * Predict targets as the mean of the trees' predictions.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predictions of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong rank or feature count
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("BaggingRegressor must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "BaggingRegressor");

    const nSamples = X.shape[0] ?? 0;
    const xv = toFloat64View(X);
    const sums = new Float64Array(nSamples);

    for (let m = 0; m < this.estimators.length; m++) {
      const tree = this.estimators[m] as DecisionTreeRegressor;
      const features = this.featureIndices[m] as Int32Array;
      const preds = predictRegressionTree(
        tree,
        gather(xv, this.nFeatures, null, nSamples, features),
        nSamples,
        features.length
      );
      for (let i = 0; i < nSamples; i++) {
        sums[i] = (sums[i] as number) + (preds[i] as number);
      }
    }

    const inv = 1 / this.estimators.length;
    for (let i = 0; i < nSamples; i++) sums[i] = (sums[i] as number) * inv;
    return tensor(sums);
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
    return r2(predictions, checkScoreTarget(y, predictions.size));
  }

  getParams(): Record<string, unknown> {
    return {
      nEstimators: this.nEstimators,
      maxSamples: this.maxSamples,
      maxFeatures: this.maxFeatures,
      bootstrap: this.bootstrap,
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
        case "maxSamples":
          this.maxSamples = checkFraction("maxSamples", value);
          break;
        case "maxFeatures":
          this.maxFeatures = checkFraction("maxFeatures", value);
          break;
        case "bootstrap":
          this.bootstrap = checkBootstrap(value);
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
  clone(): BaggingRegressor {
    return new BaggingRegressor(definedOptions(this.getParams()));
  }
}
