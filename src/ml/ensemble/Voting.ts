/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
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

/**
 * Options for {@link VotingClassifier}.
 */
export type VotingClassifierOptions = {
  /** Classifier instances to combine. They are fitted in place by `fit`. */
  readonly estimators: readonly Classifier[];
  /** `"hard"` takes a weighted majority vote, `"soft"` averages predicted probabilities. */
  readonly voting?: "hard" | "soft";
  /** One non-negative weight per estimator, at least one of them positive. */
  readonly weights?: readonly number[];
};

/**
 * Options for {@link VotingRegressor}.
 */
export type VotingRegressorOptions = {
  /** Regressor instances to combine. They are fitted in place by `fit`. */
  readonly estimators: readonly Regressor[];
  /** One non-negative weight per estimator, at least one of them positive. */
  readonly weights?: readonly number[];
};

/**
 * Validate a weight vector and return a private copy of it.
 *
 * @throws {InvalidParameterError} If `weights` is not a numeric array of the right length,
 * holds a negative or non-finite entry, or sums to zero
 */
function normalizeWeights(raw: unknown, nEstimators: number): number[] {
  if (!Array.isArray(raw) || raw.some((w) => typeof w !== "number")) {
    throw new InvalidParameterError("weights must be an array of numbers", "weights", raw);
  }
  const weights = raw as number[];
  if (weights.length !== nEstimators) {
    throw new InvalidParameterError(
      `weights length must match estimators length; got ${weights.length} weights for ${nEstimators} estimators`,
      "weights",
      weights.length
    );
  }
  let total = 0;
  for (const w of weights) {
    if (!Number.isFinite(w) || w < 0) {
      throw new InvalidParameterError("weights must be finite and >= 0", "weights", weights);
    }
    total += w;
  }
  if (!(total > 0)) {
    throw new InvalidParameterError("weights must have at least one positive entry", "weights", [
      ...weights,
    ]);
  }
  return [...weights];
}

/**
 * Read `size` values of a prediction tensor, checking that it has the expected length.
 */
function readPredictions(pred: Tensor, size: number, owner: string, index: number): Float64Array {
  if (pred.size !== size) {
    throw new ShapeError(
      `${owner}: estimator ${index} returned ${pred.size} predictions for ${size} samples`
    );
  }
  return toFloat64View(pred);
}

/**
 * Check a target vector passed to `score`.
 */
function checkScoreTarget(y: Tensor): void {
  if (y.ndim !== 1) {
    throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
  }
  assertContiguous(y, "y");
  if (y.size === 0) {
    throw new DataValidationError("y must contain at least one sample");
  }
  for (let i = 0; i < y.size; i++) {
    // Number() also accepts the BigInt values of int64 tensors, which Number.isFinite rejects.
    if (!Number.isFinite(Number(y.data[y.offset + i] ?? 0))) {
      throw new DataValidationError("y contains non-finite values (NaN or Inf)");
    }
  }
}

/**
 * Build a label tensor: `int32` when every class label is an integer that fits,
 * otherwise `float64` so fractional labels are not truncated.
 */
function labelTensor(values: ArrayLike<number>, integerLabels: boolean): Tensor {
  if (integerLabels) return tensor(Int32Array.from(values));
  return tensor(Float64Array.from(values));
}

function areInt32Labels(labels: readonly number[]): boolean {
  return labels.every((v) => Number.isInteger(v) && v >= -2147483648 && v <= 2147483647);
}

/**
 * Clone an estimator through `clone()` or, when it has none, through its constructor
 * and `getParams()`.
 */
function cloneEstimator<T extends { getParams(): Record<string, unknown>; clone?(): unknown }>(
  est: T,
  owner: string
): T {
  if (typeof est.clone === "function") return est.clone() as T;
  const Ctor = est.constructor as new (params: Record<string, unknown>) => T;
  try {
    return new Ctor(est.getParams());
  } catch (cause) {
    throw new InvalidParameterError(
      `${owner}.clone() cannot clone a base estimator; implement clone() on it or make its constructor accept the result of getParams()`,
      "estimators",
      est,
      { cause }
    );
  }
}

/**
 * Voting Classifier.
 *
 * Combines multiple classifiers via weighted majority voting (hard) or
 * weighted averaged probability (soft) voting. Hard voting resolves ties in
 * favor of the smallest class label, like scikit-learn.
 *
 * The estimators passed to the constructor are fitted in place (they are not
 * cloned). Call {@link VotingClassifier.clone} to get an unfitted copy.
 *
 * @example
 * ```ts
 * import { VotingClassifier, DecisionTreeClassifier, LogisticRegression } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0], [1, 1], [8, 8], [9, 9]]);
 * const y = tensor([0, 0, 1, 1]);
 * const clf = new VotingClassifier({
 *   estimators: [
 *     new DecisionTreeClassifier({ maxDepth: 3 }),
 *     new LogisticRegression(),
 *   ],
 *   voting: 'hard',
 * });
 * clf.fit(X, y);
 * const predictions = clf.predict(X);
 * ```
 */
export class VotingClassifier implements Classifier {
  private readonly estimators: Classifier[];
  private voting: "hard" | "soft";
  private weights: number[];

  private classLabels: number[] = [];
  private integerLabels = true;
  private nFeatures = 0;
  private fitted = false;

  /**
   * @param options.estimators - Array of classifier instances
   * @param options.voting - Voting strategy: 'hard' (majority) or 'soft' (averaged probabilities) (default: 'hard')
   * @param options.weights - Per-estimator weights (default: equal weights)
   * @throws {InvalidParameterError} If there are no estimators, `voting` is unknown, or `weights` is invalid
   */
  constructor(options: VotingClassifierOptions) {
    if (!Array.isArray(options.estimators) || options.estimators.length === 0) {
      throw new InvalidParameterError(
        "VotingClassifier requires at least one estimator",
        "estimators",
        options.estimators
      );
    }
    this.estimators = [...options.estimators];
    const voting = options.voting ?? "hard";
    if (voting !== "hard" && voting !== "soft") {
      throw new InvalidParameterError(`voting must be "hard" or "soft"`, "voting", voting);
    }
    this.voting = voting;
    this.weights =
      options.weights === undefined
        ? new Array<number>(this.estimators.length).fill(1)
        : normalizeWeights(options.weights, this.estimators.length);
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    this.fitted = false;
    if (this.voting === "soft") {
      for (const est of this.estimators) {
        if (typeof est.predictProba !== "function") {
          throw new InvalidParameterError(
            'voting="soft" requires every estimator to implement predictProba',
            "voting",
            this.voting
          );
        }
      }
    }
    this.nFeatures = X.shape[1] ?? 0;

    const nSamples = X.shape[0] ?? 0;
    const yData = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      yData[i] = Number(y.data[y.offset + i]);
    }
    this.classLabels = [...new Set(yData)].sort((a, b) => a - b);
    this.integerLabels = areInt32Labels(this.classLabels);

    for (const est of this.estimators) {
      est.fit(X, y);
    }

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("VotingClassifier must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "VotingClassifier");

    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classLabels.length;
    const predictions = new Float64Array(nSamples);

    if (this.voting === "soft") {
      const proba = this.averagedProba(X, nSamples);
      for (let i = 0; i < nSamples; i++) {
        let bestC = 0;
        let bestP = Number.NEGATIVE_INFINITY;
        for (let c = 0; c < nClasses; c++) {
          const p = proba[i * nClasses + c] ?? 0;
          if (p > bestP) {
            bestP = p;
            bestC = c;
          }
        }
        predictions[i] = this.classLabels[bestC] ?? 0;
      }
      return labelTensor(predictions, this.integerLabels);
    }

    // Hard voting: weighted majority vote, ties go to the smallest label.
    const allPreds: Float64Array[] = [];
    for (let m = 0; m < this.estimators.length; m++) {
      allPreds.push(
        readPredictions(this.estimators[m]!.predict(X), nSamples, "VotingClassifier", m)
      );
    }

    const votes = new Map<number, number>();
    for (let i = 0; i < nSamples; i++) {
      votes.clear();
      for (let m = 0; m < this.estimators.length; m++) {
        const label = allPreds[m]![i] ?? 0;
        votes.set(label, (votes.get(label) ?? 0) + (this.weights[m] ?? 0));
      }
      let bestLabel = 0;
      let bestVote = Number.NEGATIVE_INFINITY;
      for (const [label, vote] of votes) {
        if (vote > bestVote || (vote === bestVote && label < bestLabel)) {
          bestVote = vote;
          bestLabel = label;
        }
      }
      predictions[i] = bestLabel;
    }

    return labelTensor(predictions, this.integerLabels);
  }

  /**
   * Weighted average of the estimators' class probabilities.
   *
   * Works for both voting modes. Columns follow {@link VotingClassifier.classes}.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Probabilities of shape (n_samples, n_classes)
   */
  predictProba(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("VotingClassifier must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "VotingClassifier");

    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classLabels.length;
    const avg = this.averagedProba(X, nSamples);
    return tensor(avg, { dtype: "float64" }).reshape([nSamples, nClasses]);
  }

  /**
   * Weighted mean of every estimator's probabilities, mapped onto the global class order.
   * Returns a row-major (n_samples, n_classes) array.
   */
  private averagedProba(X: Tensor, nSamples: number): Float64Array {
    const nClasses = this.classLabels.length;
    const avg = new Float64Array(nSamples * nClasses);
    const labelIndex = new Map<number, number>();
    for (let c = 0; c < nClasses; c++) labelIndex.set(this.classLabels[c] ?? 0, c);

    let totalWeight = 0;
    for (let m = 0; m < this.estimators.length; m++) {
      const w = this.weights[m] ?? 0;
      totalWeight += w;
      if (w === 0) continue;
      const est = this.estimators[m]!;
      if (typeof est.predictProba !== "function") {
        throw new InvalidParameterError(
          `VotingClassifier: estimator ${m} does not implement predictProba`,
          "estimators",
          est
        );
      }
      const proba = est.predictProba(X);
      if (proba.ndim !== 2 || proba.shape[0] !== nSamples) {
        throw new ShapeError(
          `VotingClassifier: estimator ${m} returned probabilities of shape [${proba.shape.join(", ")}]; expected [${nSamples}, n_classes]`
        );
      }
      const estNClasses = proba.shape[1] ?? 0;
      const values = toFloat64View(proba);

      // Map estimator columns to global class indices.
      const columnMap = new Int32Array(estNClasses).fill(-1);
      const estClasses = est.classes;
      if (estClasses) {
        const labels = toFloat64View(estClasses);
        if (labels.length !== estNClasses) {
          throw new ShapeError(
            `VotingClassifier: estimator ${m} has ${labels.length} classes but returned ${estNClasses} probability columns`
          );
        }
        for (let c = 0; c < estNClasses; c++) {
          columnMap[c] = labelIndex.get(labels[c] ?? 0) ?? -1;
        }
      } else {
        if (estNClasses !== nClasses) {
          throw new ShapeError(
            `VotingClassifier: estimator ${m} returned ${estNClasses} probability columns for ${nClasses} classes and does not expose classes`
          );
        }
        for (let c = 0; c < estNClasses; c++) columnMap[c] = c;
      }

      for (let i = 0; i < nSamples; i++) {
        for (let c = 0; c < estNClasses; c++) {
          const globalC = columnMap[c] ?? -1;
          if (globalC >= 0) {
            avg[i * nClasses + globalC] =
              (avg[i * nClasses + globalC] ?? 0) + w * (values[i * estNClasses + c] ?? 0);
          }
        }
      }
    }

    for (let i = 0; i < avg.length; i++) {
      avg[i] = (avg[i] ?? 0) / totalWeight;
    }
    return avg;
  }

  /**
   * Mean accuracy of `predict(X)` against `y`.
   *
   * @param X - Test samples
   * @param y - True labels of shape (n_samples,)
   * @returns Fraction of correctly classified samples
   * @throws {ShapeError} If `y` is not 1-D or its length differs from the number of samples in `X`
   * @throws {DataValidationError} If `y` is empty or holds NaN or Infinity
   */
  score(X: Tensor, y: Tensor): number {
    checkScoreTarget(y);
    const predictions = this.predict(X);
    if (predictions.size !== y.size) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${predictions.size}, y=${y.size}`
      );
    }
    let correct = 0;
    for (let i = 0; i < y.size; i++) {
      if (Number(predictions.data[predictions.offset + i]) === Number(y.data[y.offset + i])) {
        correct++;
      }
    }
    return correct / y.size;
  }

  /** Sorted class labels seen during `fit`, or `undefined` before fitting. */
  get classes(): Tensor | undefined {
    if (!this.fitted) return undefined;
    return labelTensor(this.classLabels, this.integerLabels);
  }

  getParams(): Record<string, unknown> {
    return {
      estimators: [...this.estimators],
      voting: this.voting,
      weights: [...this.weights],
      nEstimators: this.estimators.length,
    };
  }

  /**
   * Update `voting` and/or `weights`. The change applies to later `predict` calls;
   * the estimators are not refitted.
   *
   * @throws {InvalidParameterError} On an unknown key or an invalid value
   */
  setParams(params: Record<string, unknown>): this {
    // Validate everything first so a bad entry leaves the model untouched.
    let voting = this.voting;
    let weights = this.weights;
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "voting":
          if (value !== "hard" && value !== "soft") {
            throw new InvalidParameterError(`voting must be "hard" or "soft"`, "voting", value);
          }
          voting = value;
          break;
        case "weights":
          weights = normalizeWeights(value, this.estimators.length);
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    this.voting = voting;
    this.weights = weights;
    return this;
  }

  /**
   * Create an unfitted copy with the same parameters and cloned estimators.
   */
  clone(): VotingClassifier {
    return new VotingClassifier({
      estimators: this.estimators.map((e) => cloneEstimator(e, "VotingClassifier")),
      voting: this.voting,
      weights: this.weights,
    });
  }
}

/**
 * Voting Regressor.
 *
 * Combines multiple regressors by averaging their predictions,
 * optionally with per-estimator weights.
 *
 * The estimators passed to the constructor are fitted in place (they are not
 * cloned). Call {@link VotingRegressor.clone} to get an unfitted copy.
 *
 * @example
 * ```ts
 * import { VotingRegressor, DecisionTreeRegressor, LinearRegression } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1], [2], [3], [4], [5]]);
 * const y = tensor([1.1, 1.9, 3.2, 3.9, 5.1]);
 * const reg = new VotingRegressor({
 *   estimators: [
 *     new DecisionTreeRegressor({ maxDepth: 3 }),
 *     new LinearRegression(),
 *   ],
 * });
 * reg.fit(X, y);
 * const predictions = reg.predict(X);
 * ```
 */
export class VotingRegressor implements Regressor {
  private readonly estimators: Regressor[];
  private weights: number[];

  private nFeatures = 0;
  private fitted = false;

  /**
   * @param options.estimators - Array of regressor instances
   * @param options.weights - Per-estimator weights (default: equal weights)
   * @throws {InvalidParameterError} If there are no estimators or `weights` is invalid
   */
  constructor(options: VotingRegressorOptions) {
    if (!Array.isArray(options.estimators) || options.estimators.length === 0) {
      throw new InvalidParameterError(
        "VotingRegressor requires at least one estimator",
        "estimators",
        options.estimators
      );
    }
    this.estimators = [...options.estimators];
    this.weights =
      options.weights === undefined
        ? new Array<number>(this.estimators.length).fill(1)
        : normalizeWeights(options.weights, this.estimators.length);
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    this.fitted = false;
    this.nFeatures = X.shape[1] ?? 0;

    for (const est of this.estimators) {
      est.fit(X, y);
    }

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("VotingRegressor must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "VotingRegressor");

    const nSamples = X.shape[0] ?? 0;
    const sums = new Float64Array(nSamples);
    let totalWeight = 0;

    for (let m = 0; m < this.estimators.length; m++) {
      const w = this.weights[m] ?? 0;
      totalWeight += w;
      const pred = readPredictions(this.estimators[m]!.predict(X), nSamples, "VotingRegressor", m);
      if (w === 0) continue;
      for (let i = 0; i < nSamples; i++) {
        sums[i] = (sums[i] ?? 0) + w * (pred[i] ?? 0);
      }
    }

    for (let i = 0; i < nSamples; i++) {
      sums[i] = (sums[i] ?? 0) / totalWeight;
    }
    return tensor(sums, { dtype: "float64" });
  }

  /**
   * Coefficient of determination R^2 of `predict(X)` against `y`.
   *
   * A constant target gives 1 for a perfect fit and 0 otherwise, like scikit-learn.
   *
   * @throws {ShapeError} If `y` is not 1-D or its length differs from the number of samples in `X`
   * @throws {DataValidationError} If `y` is empty or holds NaN or Infinity
   */
  score(X: Tensor, y: Tensor): number {
    checkScoreTarget(y);
    const predictions = this.predict(X);
    if (predictions.size !== y.size) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${predictions.size}, y=${y.size}`
      );
    }
    let ssRes = 0;
    let ssTot = 0;
    let yMean = 0;
    for (let i = 0; i < y.size; i++) {
      yMean += Number(y.data[y.offset + i]);
    }
    yMean /= y.size;
    for (let i = 0; i < y.size; i++) {
      const yVal = Number(y.data[y.offset + i]);
      const pVal = Number(predictions.data[predictions.offset + i]);
      ssRes += (yVal - pVal) ** 2;
      ssTot += (yVal - yMean) ** 2;
    }
    return ssTot === 0 ? (ssRes === 0 ? 1.0 : 0.0) : 1 - ssRes / ssTot;
  }

  getParams(): Record<string, unknown> {
    return {
      estimators: [...this.estimators],
      weights: [...this.weights],
      nEstimators: this.estimators.length,
    };
  }

  /**
   * Update `weights`. The change applies to later `predict` calls; the estimators
   * are not refitted.
   *
   * @throws {InvalidParameterError} On an unknown key or invalid weights
   */
  setParams(params: Record<string, unknown>): this {
    let weights = this.weights;
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "weights":
          weights = normalizeWeights(value, this.estimators.length);
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    this.weights = weights;
    return this;
  }

  /**
   * Create an unfitted copy with the same parameters and cloned estimators.
   */
  clone(): VotingRegressor {
    return new VotingRegressor({
      estimators: this.estimators.map((e) => cloneEstimator(e, "VotingRegressor")),
      weights: this.weights,
    });
  }
}
