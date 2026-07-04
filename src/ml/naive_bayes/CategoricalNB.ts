/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier } from "../base";

/**
 * Categorical Naive Bayes classifier.
 *
 * Suitable for classification with discrete/categorical features.
 * Each feature is assumed to be generated from a categorical distribution.
 *
 * Unlike MultinomialNB (which models feature counts) or BernoulliNB (binary),
 * CategoricalNB models each feature as having its own set of categories.
 *
 * **Algorithm**:
 * 1. For each feature and each class, compute the probability of each category
 *    using smoothed maximum likelihood estimates.
 * 2. For prediction, compute the log posterior for each class.
 *
 * @example
 * ```ts
 * import { CategoricalNB } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * // Categorical features (e.g., color=0/1/2, size=0/1)
 * const X = tensor([[0, 0], [1, 1], [2, 0], [0, 1], [1, 0], [2, 1]]);
 * const y = tensor([0, 0, 0, 1, 1, 1]);
 *
 * const clf = new CategoricalNB();
 * clf.fit(X, y);
 * const predictions = clf.predict(tensor([[1, 0]]));
 * ```
 */
export class CategoricalNB implements Classifier {
  private alpha: number;
  private fitPrior: boolean;

  private classes_?: number[];
  private classLogPrior_?: number[];
  // featureLogProb_[classIdx][featureIdx] = Map<category, logProb>
  private featureLogProb_?: Array<Array<Map<number, number>>>;
  private nCategories_?: number[]; // number of categories per feature
  private nFeaturesIn_?: number;
  private fitted = false;

  /**
   * @param options.alpha - Additive (Laplace) smoothing (default: 1.0)
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

    // Extract classes
    const classSet = new Set<number>();
    for (let i = 0; i < nSamples; i++) {
      classSet.add(Number(y.data[y.offset + i] ?? 0));
    }
    this.classes_ = Array.from(classSet).sort((a, b) => a - b);
    const nClasses = this.classes_.length;

    const classIndex = new Map<number, number>();
    for (let c = 0; c < nClasses; c++) {
      classIndex.set(this.classes_[c] ?? 0, c);
    }

    // Count class occurrences
    const classCounts = new Float64Array(nClasses);
    for (let i = 0; i < nSamples; i++) {
      const ci = classIndex.get(Number(y.data[y.offset + i] ?? 0)) ?? 0;
      classCounts[ci] = (classCounts[ci] ?? 0) + 1;
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

    // Discover categories for each feature
    const featureCategories: Set<number>[] = [];
    for (let j = 0; j < nFeatures; j++) {
      featureCategories.push(new Set<number>());
    }
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        const cat = featureCategories[j];
        if (cat) {
          cat.add(Number(X.data[X.offset + i * nFeatures + j] ?? 0));
        }
      }
    }

    this.nCategories_ = featureCategories.map((s) => s.size);

    // Count feature-category occurrences per class
    // counts[c][j] = Map<category, count>
    const counts: Array<Array<Map<number, number>>> = [];
    for (let c = 0; c < nClasses; c++) {
      const classFeatures: Array<Map<number, number>> = [];
      for (let j = 0; j < nFeatures; j++) {
        classFeatures.push(new Map<number, number>());
      }
      counts.push(classFeatures);
    }

    for (let i = 0; i < nSamples; i++) {
      const ci = classIndex.get(Number(y.data[y.offset + i] ?? 0)) ?? 0;
      const classFeats = counts[ci];
      if (!classFeats) continue;
      for (let j = 0; j < nFeatures; j++) {
        const val = Number(X.data[X.offset + i * nFeatures + j] ?? 0);
        const featMap = classFeats[j];
        if (featMap) {
          featMap.set(val, (featMap.get(val) ?? 0) + 1);
        }
      }
    }

    // Compute log probabilities with smoothing
    this.featureLogProb_ = [];
    for (let c = 0; c < nClasses; c++) {
      const classFeats: Array<Map<number, number>> = [];
      const nc = classCounts[c] ?? 0;
      for (let j = 0; j < nFeatures; j++) {
        const logProbMap = new Map<number, number>();
        const catCount = this.nCategories_[j] ?? 1;
        const denominator = nc + this.alpha * catCount;
        const catSet = featureCategories[j];
        if (catSet) {
          for (const cat of catSet) {
            const count = counts[c]?.[j]?.get(cat) ?? 0;
            logProbMap.set(cat, Math.log((count + this.alpha) / denominator));
          }
        }
        // Store default log prob for unseen categories
        logProbMap.set(-Infinity, Math.log(this.alpha / denominator));
        classFeats.push(logProbMap);
      }
      this.featureLogProb_.push(classFeats);
    }

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("CategoricalNB must be fitted before prediction");
    }
    const proba = this.predictProba(X);
    const nSamples = proba.shape[0] ?? 0;
    const nClasses = proba.shape[1] ?? 0;
    const predictions: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      let maxP = -1;
      let maxC = 0;
      for (let j = 0; j < nClasses; j++) {
        const p = Number(proba.data[proba.offset + i * nClasses + j] ?? 0);
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
      throw new NotFittedError("CategoricalNB must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "CategoricalNB");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const nClasses = this.classes_.length;
    const probabilities: number[][] = [];

    for (let i = 0; i < nSamples; i++) {
      const logProbs: number[] = [];
      for (let c = 0; c < nClasses; c++) {
        let logP = this.classLogPrior_[c] ?? 0;
        const classFeats = this.featureLogProb_[c];
        if (classFeats) {
          for (let j = 0; j < nFeatures; j++) {
            const val = Number(X.data[X.offset + i * nFeatures + j] ?? 0);
            const featMap = classFeats[j];
            if (featMap) {
              logP += featMap.get(val) ?? featMap.get(-Infinity) ?? 0;
            }
          }
        }
        logProbs.push(logP);
      }
      // Softmax
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
      if (Number(y.data[y.offset + i] ?? 0) === Number(yPred.data[yPred.offset + i] ?? 0)) {
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
      fitPrior: this.fitPrior,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "alpha":
          if (typeof value !== "number" || value < 0) {
            throw new InvalidParameterError("alpha must be a non-negative number", "alpha", value);
          }
          this.alpha = value;
          break;
        case "fitPrior":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("fitPrior must be a boolean", "fitPrior", value);
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
