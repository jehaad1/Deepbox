/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier, Regressor } from "../base";

/**
 * Voting Classifier.
 *
 * Combines multiple classifiers via majority voting (hard) or
 * averaged probability (soft) voting.
 *
 * @example
 * ```ts
 * import { VotingClassifier, DecisionTreeClassifier, LogisticRegression } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
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
  private nFeatures = 0;
  private fitted = false;

  /**
   * @param options.estimators - Array of classifier instances
   * @param options.voting - Voting strategy: 'hard' (majority) or 'soft' (averaged probabilities) (default: 'hard')
   * @param options.weights - Per-estimator weights (default: equal weights)
   */
  constructor(options: {
    readonly estimators: Classifier[];
    readonly voting?: "hard" | "soft";
    readonly weights?: number[];
  }) {
    if (!options.estimators || options.estimators.length === 0) {
      throw new InvalidParameterError(
        "VotingClassifier requires at least one estimator",
        "estimators",
        options.estimators
      );
    }
    this.estimators = options.estimators;
    this.voting = options.voting ?? "hard";
    this.weights = options.weights ?? new Array<number>(this.estimators.length).fill(1);

    if (this.weights.length !== this.estimators.length) {
      throw new InvalidParameterError(
        "weights length must match estimators length",
        "weights",
        this.weights.length
      );
    }
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    this.nFeatures = X.shape[1] ?? 0;

    const yData: number[] = [];
    for (let i = 0; i < (X.shape[0] ?? 0); i++) {
      yData.push(Number(y.data[y.offset + i]));
    }
    this.classLabels = [...new Set(yData)].sort((a, b) => a - b);

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

    if (this.voting === "soft") {
      // Average probabilities
      const proba = this.predictProba(X);
      const predictions: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        let bestC = 0;
        let bestP = -1;
        for (let c = 0; c < nClasses; c++) {
          const p = Number(proba.data[proba.offset + i * nClasses + c]);
          if (p > bestP) {
            bestP = p;
            bestC = c;
          }
        }
        predictions.push(this.classLabels[bestC] ?? 0);
      }
      return tensor(predictions, { dtype: "int32" });
    }

    // Hard voting: majority vote
    const allPreds: number[][] = [];
    for (const est of this.estimators) {
      const pred = est.predict(X);
      const p: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        p.push(Number(pred.data[pred.offset + i]));
      }
      allPreds.push(p);
    }

    const predictions: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      const votes = new Map<number, number>();
      for (let m = 0; m < this.estimators.length; m++) {
        const pred = allPreds[m]![i] ?? 0;
        const w = this.weights[m] ?? 1;
        votes.set(pred, (votes.get(pred) ?? 0) + w);
      }
      let bestLabel = 0;
      let bestVote = -1;
      for (const [label, vote] of votes) {
        if (vote > bestVote) {
          bestVote = vote;
          bestLabel = label;
        }
      }
      predictions.push(bestLabel);
    }

    return tensor(predictions, { dtype: "int32" });
  }

  predictProba(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("VotingClassifier must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "VotingClassifier");

    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classLabels.length;

    // Average weighted probabilities from all estimators
    const avgProba: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
      avgProba.push(new Array<number>(nClasses).fill(0));
    }

    let totalWeight = 0;
    for (let m = 0; m < this.estimators.length; m++) {
      const w = this.weights[m] ?? 1;
      totalWeight += w;
      const proba = this.estimators[m]!.predictProba(X);
      const estNClasses = proba.shape[1] ?? 0;

      // Map estimator classes to global class indices
      const estClasses = this.estimators[m]!.classes;
      const classMap = new Map<number, number>();
      if (estClasses) {
        for (let c = 0; c < estNClasses; c++) {
          const label = Number(estClasses.data[estClasses.offset + c]);
          const globalIdx = this.classLabels.indexOf(label);
          if (globalIdx >= 0) classMap.set(c, globalIdx);
        }
      } else {
        // Assume same class ordering
        for (let c = 0; c < Math.min(estNClasses, nClasses); c++) {
          classMap.set(c, c);
        }
      }

      for (let i = 0; i < nSamples; i++) {
        for (let c = 0; c < estNClasses; c++) {
          const globalC = classMap.get(c);
          if (globalC !== undefined) {
            const row = avgProba[i]!;
            row[globalC] =
              (row[globalC] ?? 0) + w * Number(proba.data[proba.offset + i * estNClasses + c] ?? 0);
          }
        }
      }
    }

    // Normalize
    for (let i = 0; i < nSamples; i++) {
      const row = avgProba[i]!;
      for (let c = 0; c < nClasses; c++) {
        row[c] = (row[c] ?? 0) / totalWeight;
      }
    }

    return tensor(avgProba);
  }

  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    assertContiguous(y, "y");
    for (let i = 0; i < y.size; i++) {
      if (!Number.isFinite(y.data[y.offset + i] ?? 0)) {
        throw new DataValidationError("y contains non-finite values (NaN or Inf)");
      }
    }
    const predictions = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.size; i++) {
      if (Number(predictions.data[predictions.offset + i]) === Number(y.data[y.offset + i])) {
        correct++;
      }
    }
    return correct / y.size;
  }

  get classes(): Tensor | undefined {
    if (!this.fitted) return undefined;
    return tensor(this.classLabels, { dtype: "int32" });
  }

  getParams(): Record<string, unknown> {
    return {
      voting: this.voting,
      weights: this.weights,
      nEstimators: this.estimators.length,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "voting":
          if (value !== "hard" && value !== "soft") {
            throw new InvalidParameterError(`voting must be "hard" or "soft"`, "voting", value);
          }
          this.voting = value;
          break;
        case "weights":
          if (!Array.isArray(value) || value.some((w) => typeof w !== "number")) {
            throw new InvalidParameterError(
              "weights must be an array of numbers",
              "weights",
              value
            );
          }
          this.weights = value.filter((w): w is number => typeof w === "number");
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}

/**
 * Voting Regressor.
 *
 * Combines multiple regressors by averaging their predictions,
 * optionally with per-estimator weights.
 *
 * @example
 * ```ts
 * import { VotingRegressor, DecisionTreeRegressor, LinearRegression } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
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
   */
  constructor(options: {
    readonly estimators: Regressor[];
    readonly weights?: number[];
  }) {
    if (!options.estimators || options.estimators.length === 0) {
      throw new InvalidParameterError(
        "VotingRegressor requires at least one estimator",
        "estimators",
        options.estimators
      );
    }
    this.estimators = options.estimators;
    this.weights = options.weights ?? new Array<number>(this.estimators.length).fill(1);

    if (this.weights.length !== this.estimators.length) {
      throw new InvalidParameterError(
        "weights length must match estimators length",
        "weights",
        this.weights.length
      );
    }
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
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
      const w = this.weights[m] ?? 1;
      totalWeight += w;
      const pred = this.estimators[m]!.predict(X);
      for (let i = 0; i < nSamples; i++) {
        sums[i] = (sums[i] ?? 0) + w * Number(pred.data[pred.offset + i]);
      }
    }

    const result: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      result.push((sums[i] ?? 0) / totalWeight);
    }
    return tensor(result);
  }

  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    assertContiguous(y, "y");
    for (let i = 0; i < y.size; i++) {
      if (!Number.isFinite(y.data[y.offset + i] ?? 0)) {
        throw new DataValidationError("y contains non-finite values (NaN or Inf)");
      }
    }
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
    return { weights: this.weights, nEstimators: this.estimators.length };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "weights":
          if (!Array.isArray(value) || value.some((w) => typeof w !== "number")) {
            throw new InvalidParameterError(
              "weights must be an array of numbers",
              "weights",
              value
            );
          }
          this.weights = value as number[];
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
