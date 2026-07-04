/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier } from "../base";

/**
 * Multinomial Naive Bayes classifier.
 *
 * Suitable for classification with discrete features (e.g., word counts
 * for text classification). The multinomial distribution normally requires
 * integer feature counts, but fractional counts (e.g., TF-IDF) work in practice.
 *
 * **Algorithm**:
 * 1. For each class, compute the log probability of each feature
 *    using smoothed maximum likelihood estimates.
 * 2. For prediction, compute the log posterior for each class and
 *    select the class with the highest posterior.
 *
 * @example
 * ```ts
 * import { MultinomialNB } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * // Word count features for text classification
 * const X = tensor([[3, 0, 1], [0, 2, 1], [1, 0, 3], [0, 3, 0]]);
 * const y = tensor([0, 1, 0, 1]);
 *
 * const clf = new MultinomialNB();
 * clf.fit(X, y);
 * const predictions = clf.predict(tensor([[2, 0, 1]]));
 * ```
 */
export class MultinomialNB implements Classifier {
  private alpha: number;
  private fitPrior: boolean;

  private classes_?: number[];
  private classLogPrior_?: number[];
  private featureLogProb_?: number[][]; // [nClasses][nFeatures]
  private nFeaturesIn_?: number;
  private fitted = false;

  /**
   * @param options.alpha - Additive (Laplace/Lidstone) smoothing (default: 1.0)
   * @param options.fitPrior - Whether to learn class prior probabilities (default: true)
   */
  constructor(
    options: {
      readonly alpha?: number;
      readonly fitPrior?: boolean;
    } = {}
  ) {
    this.alpha = options.alpha ?? 1.0;
    this.fitPrior = options.fitPrior ?? true;
    if (!Number.isFinite(this.alpha) || this.alpha < 0) {
      throw new InvalidParameterError("alpha must be a finite number >= 0", "alpha", this.alpha);
    }
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nFeatures;

    // Validate non-negative features
    for (let i = 0; i < X.size; i++) {
      const v = Number(X.data[X.offset + i] ?? 0);
      if (v < 0) {
        throw new DataValidationError("MultinomialNB requires non-negative feature values");
      }
    }

    // Get unique classes
    const classSet = new Set<number>();
    for (let i = 0; i < nSamples; i++) {
      classSet.add(Number(y.data[y.offset + i]));
    }
    this.classes_ = Array.from(classSet).sort((a, b) => a - b);
    const nClasses = this.classes_.length;

    // Compute class counts and feature counts per class
    const classCounts = new Array<number>(nClasses).fill(0);
    const featureCounts: number[][] = [];
    for (let c = 0; c < nClasses; c++) {
      featureCounts.push(new Array<number>(nFeatures).fill(0));
    }

    const classIndex = new Map<number, number>();
    for (let c = 0; c < nClasses; c++) {
      classIndex.set(this.classes_[c] ?? 0, c);
    }

    for (let i = 0; i < nSamples; i++) {
      const label = Number(y.data[y.offset + i]);
      const ci = classIndex.get(label) ?? 0;
      classCounts[ci] = (classCounts[ci] ?? 0) + 1;
      const rowBase = X.offset + i * nFeatures;
      const row = featureCounts[ci];
      if (row) {
        for (let j = 0; j < nFeatures; j++) {
          row[j] = (row[j] ?? 0) + Number(X.data[rowBase + j] ?? 0);
        }
      }
    }

    // Log prior
    this.classLogPrior_ = new Array<number>(nClasses);
    if (this.fitPrior) {
      for (let c = 0; c < nClasses; c++) {
        this.classLogPrior_[c] = Math.log((classCounts[c] ?? 0) / nSamples);
      }
    } else {
      const uniformLog = Math.log(1 / nClasses);
      for (let c = 0; c < nClasses; c++) {
        this.classLogPrior_[c] = uniformLog;
      }
    }

    // Feature log probabilities with Laplace smoothing
    this.featureLogProb_ = [];
    for (let c = 0; c < nClasses; c++) {
      const row = featureCounts[c] ?? [];
      let totalCount = 0;
      for (let j = 0; j < nFeatures; j++) {
        totalCount += row[j] ?? 0;
      }
      const smoothedTotal = totalCount + this.alpha * nFeatures;
      const logProbs = new Array<number>(nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        logProbs[j] = Math.log(((row[j] ?? 0) + this.alpha) / smoothedTotal);
      }
      this.featureLogProb_.push(logProbs);
    }

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("MultinomialNB must be fitted before prediction");
    }
    const proba = this.predictProba(X);
    const nSamples = proba.shape[0] ?? 0;
    const nClasses = proba.shape[1] ?? 0;
    const predictions: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      let maxP = -1;
      let maxC = 0;
      for (let j = 0; j < nClasses; j++) {
        const p = Number(proba.data[proba.offset + i * nClasses + j]);
        if (p > maxP) {
          maxP = p;
          maxC = this.classes_?.[j] ?? 0;
        }
      }
      predictions.push(maxC);
    }
    return tensor(predictions, { dtype: "int32" });
  }

  predictProba(X: Tensor): Tensor {
    if (!this.fitted || !this.classes_ || !this.classLogPrior_ || !this.featureLogProb_) {
      throw new NotFittedError("MultinomialNB must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "MultinomialNB");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const nClasses = this.classes_.length;
    const probabilities: number[][] = [];

    for (let i = 0; i < nSamples; i++) {
      const logProbs: number[] = [];
      for (let c = 0; c < nClasses; c++) {
        let logP = this.classLogPrior_[c] ?? 0;
        const flp = this.featureLogProb_[c] ?? [];
        for (let j = 0; j < nFeatures; j++) {
          const xVal = Number(X.data[X.offset + i * nFeatures + j] ?? 0);
          logP += xVal * (flp[j] ?? 0);
        }
        logProbs.push(logP);
      }
      // Log-sum-exp
      const maxLP = Math.max(...logProbs);
      const exps = logProbs.map((lp) => Math.exp(lp - maxLP));
      const sumExps = exps.reduce((a, b) => a + b, 0);
      probabilities.push(exps.map((e) => e / sumExps));
    }

    return tensor(probabilities);
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
    const yPred = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.size; i++) {
      if (Number(y.data[y.offset + i]) === Number(yPred.data[yPred.offset + i])) {
        correct++;
      }
    }
    return correct / y.size;
  }

  get classes(): Tensor | undefined {
    if (!this.fitted || !this.classes_) return undefined;
    return tensor(this.classes_, { dtype: "int32" });
  }

  getParams(): Record<string, unknown> {
    return { alpha: this.alpha, fitPrior: this.fitPrior };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "alpha":
          if (typeof value !== "number" || value < 0) {
            throw new InvalidParameterError(
              `alpha must be a non-negative number; got ${String(value)}`,
              "alpha",
              value
            );
          }
          this.alpha = value;
          break;
        case "fitPrior":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError(
              `fitPrior must be a boolean; got ${String(value)}`,
              "fitPrior",
              value
            );
          }
          this.fitPrior = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
