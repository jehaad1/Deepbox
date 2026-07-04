/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier } from "../base";

/**
 * Bernoulli Naive Bayes classifier.
 *
 * Like MultinomialNB, this classifier is suitable for discrete data.
 * The difference is that while MultinomialNB works with occurrence counts,
 * BernoulliNB is designed for binary/boolean features. It binarizes input
 * using the `binarize` threshold.
 *
 * @example
 * ```ts
 * import { BernoulliNB } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0]]);
 * const y = tensor([0, 1, 0, 1]);
 *
 * const clf = new BernoulliNB();
 * clf.fit(X, y);
 * const predictions = clf.predict(tensor([[1, 0, 1]]));
 * ```
 */
export class BernoulliNB implements Classifier {
  private alpha: number;
  private binarize: number | null;
  private fitPrior: boolean;

  private classes_?: number[];
  private classLogPrior_?: number[];
  private featureLogProb_?: number[][]; // log P(x_j=1 | y=c)
  private featureLogCompProb_?: number[][]; // log P(x_j=0 | y=c)
  private nFeaturesIn_?: number;
  private fitted = false;

  /**
   * @param options.alpha - Additive smoothing (default: 1.0)
   * @param options.binarize - Threshold for binarizing features (default: 0.0). Set to null to assume already binary.
   * @param options.fitPrior - Whether to learn class prior probabilities (default: true)
   */
  constructor(
    options: {
      readonly alpha?: number;
      readonly binarize?: number | null;
      readonly fitPrior?: boolean;
    } = {}
  ) {
    this.alpha = options.alpha ?? 1.0;
    this.binarize = options.binarize === undefined ? 0.0 : options.binarize;
    this.fitPrior = options.fitPrior ?? true;
    if (!Number.isFinite(this.alpha) || this.alpha < 0) {
      throw new InvalidParameterError("alpha must be a finite number >= 0", "alpha", this.alpha);
    }
  }

  private binarizeValue(x: number): number {
    if (this.binarize === null) return x;
    return x > this.binarize ? 1 : 0;
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nFeatures;

    const classSet = new Set<number>();
    for (let i = 0; i < nSamples; i++) {
      classSet.add(Number(y.data[y.offset + i]));
    }
    this.classes_ = Array.from(classSet).sort((a, b) => a - b);
    const nClasses = this.classes_.length;

    const classIndex = new Map<number, number>();
    for (let c = 0; c < nClasses; c++) {
      classIndex.set(this.classes_[c] ?? 0, c);
    }

    const classCounts = new Array<number>(nClasses).fill(0);
    // featureCounts[c][j] = number of samples in class c where feature j is 1
    const featureCounts: number[][] = [];
    for (let c = 0; c < nClasses; c++) {
      featureCounts.push(new Array<number>(nFeatures).fill(0));
    }

    for (let i = 0; i < nSamples; i++) {
      const label = Number(y.data[y.offset + i]);
      const ci = classIndex.get(label) ?? 0;
      classCounts[ci] = (classCounts[ci] ?? 0) + 1;
      const rowBase = X.offset + i * nFeatures;
      const row = featureCounts[ci];
      if (row) {
        for (let j = 0; j < nFeatures; j++) {
          const xVal = this.binarizeValue(Number(X.data[rowBase + j] ?? 0));
          if (xVal > 0) {
            row[j] = (row[j] ?? 0) + 1;
          }
        }
      }
    }

    // Class log priors
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

    // Feature log probabilities with smoothing
    this.featureLogProb_ = [];
    this.featureLogCompProb_ = [];
    for (let c = 0; c < nClasses; c++) {
      const row = featureCounts[c] ?? [];
      const nc = classCounts[c] ?? 0;
      const smoothedTotal = nc + 2 * this.alpha;
      const logProbs = new Array<number>(nFeatures);
      const logCompProbs = new Array<number>(nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        const p = ((row[j] ?? 0) + this.alpha) / smoothedTotal;
        logProbs[j] = Math.log(p);
        logCompProbs[j] = Math.log(1 - p);
      }
      this.featureLogProb_.push(logProbs);
      this.featureLogCompProb_.push(logCompProbs);
    }

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("BernoulliNB must be fitted before prediction");
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
    if (
      !this.fitted ||
      !this.classes_ ||
      !this.classLogPrior_ ||
      !this.featureLogProb_ ||
      !this.featureLogCompProb_
    ) {
      throw new NotFittedError("BernoulliNB must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "BernoulliNB");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const nClasses = this.classes_.length;
    const probabilities: number[][] = [];

    for (let i = 0; i < nSamples; i++) {
      const logProbs: number[] = [];
      for (let c = 0; c < nClasses; c++) {
        let logP = this.classLogPrior_[c] ?? 0;
        const flp = this.featureLogProb_[c] ?? [];
        const flcp = this.featureLogCompProb_[c] ?? [];
        for (let j = 0; j < nFeatures; j++) {
          const xVal = this.binarizeValue(Number(X.data[X.offset + i * nFeatures + j] ?? 0));
          if (xVal > 0) {
            logP += flp[j] ?? 0;
          } else {
            logP += flcp[j] ?? 0;
          }
        }
        logProbs.push(logP);
      }
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
    return {
      alpha: this.alpha,
      binarize: this.binarize,
      fitPrior: this.fitPrior,
    };
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
        case "binarize":
          if (value !== null && typeof value !== "number") {
            throw new InvalidParameterError(
              `binarize must be a number or null; got ${String(value)}`,
              "binarize",
              value
            );
          }
          this.binarize = value;
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
