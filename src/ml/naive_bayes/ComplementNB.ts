/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier } from "../base";

/**
 * Complement Naive Bayes classifier.
 *
 * An adaptation of the standard Multinomial Naive Bayes that is particularly
 * suited for imbalanced datasets. Instead of computing the likelihood from
 * samples of a class, it uses the complement (all samples NOT in the class).
 *
 * **Algorithm**:
 * 1. For each class c, compute feature weights using the complement set (all classes except c).
 * 2. Optionally normalize weights per class so they sum to 1.
 * 3. Predict the class that assigns the highest weight to the sample.
 *
 * Reference: Rennie et al., "Tackling the Poor Assumptions of Naive Bayes Text Classifiers", ICML 2003.
 *
 * @example
 * ```ts
 * import { ComplementNB } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[3, 0, 1], [0, 2, 1], [1, 0, 3], [0, 3, 0]]);
 * const y = tensor([0, 1, 0, 1]);
 *
 * const clf = new ComplementNB();
 * clf.fit(X, y);
 * const predictions = clf.predict(tensor([[2, 0, 1]]));
 * ```
 */
export class ComplementNB implements Classifier {
  private alpha: number;
  private fitPrior: boolean;
  private norm: boolean;

  private classes_?: number[];
  private classLogPrior_?: number[];
  private featureLogProb_?: number[][]; // complement weights [nClasses][nFeatures]
  private nFeaturesIn_?: number;
  private fitted = false;

  /**
   * @param options.alpha - Additive smoothing (default: 1.0)
   * @param options.fitPrior - Whether to learn class prior probabilities (default: true)
   * @param options.norm - Whether to L2-normalize complement weights (default: false)
   */
  constructor(
    options: {
      readonly alpha?: number;
      readonly fitPrior?: boolean;
      readonly norm?: boolean;
    } = {}
  ) {
    this.alpha = options.alpha ?? 1.0;
    this.fitPrior = options.fitPrior ?? true;
    this.norm = options.norm ?? false;
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
        throw new DataValidationError("ComplementNB requires non-negative feature values");
      }
    }

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

    // Compute per-class feature sums
    const classCounts = new Array<number>(nClasses).fill(0);
    const featureSums: number[][] = [];
    for (let c = 0; c < nClasses; c++) {
      featureSums.push(new Array<number>(nFeatures).fill(0));
    }

    for (let i = 0; i < nSamples; i++) {
      const label = Number(y.data[y.offset + i]);
      const ci = classIndex.get(label) ?? 0;
      classCounts[ci] = (classCounts[ci] ?? 0) + 1;
      const rowBase = X.offset + i * nFeatures;
      const row = featureSums[ci];
      if (row) {
        for (let j = 0; j < nFeatures; j++) {
          row[j] = (row[j] ?? 0) + Number(X.data[rowBase + j] ?? 0);
        }
      }
    }

    // Total feature sums across all classes
    const totalFeatureSums = new Array<number>(nFeatures).fill(0);
    for (let c = 0; c < nClasses; c++) {
      const row = featureSums[c] ?? [];
      for (let j = 0; j < nFeatures; j++) {
        totalFeatureSums[j] = (totalFeatureSums[j] ?? 0) + (row[j] ?? 0);
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

    // Complement feature weights
    // For class c, use features from all OTHER classes
    this.featureLogProb_ = [];
    for (let c = 0; c < nClasses; c++) {
      const complementSums = new Array<number>(nFeatures);
      let complementTotal = 0;
      for (let j = 0; j < nFeatures; j++) {
        const cs = (totalFeatureSums[j] ?? 0) - ((featureSums[c] ?? [])[j] ?? 0);
        complementSums[j] = cs;
        complementTotal += cs;
      }

      const smoothedTotal = complementTotal + this.alpha * nFeatures;
      const weights = new Array<number>(nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        weights[j] = Math.log(((complementSums[j] ?? 0) + this.alpha) / smoothedTotal);
      }

      // Optionally L2-normalize
      if (this.norm) {
        let l2 = 0;
        for (let j = 0; j < nFeatures; j++) {
          l2 += (weights[j] ?? 0) * (weights[j] ?? 0);
        }
        l2 = Math.sqrt(l2);
        if (l2 > 0) {
          for (let j = 0; j < nFeatures; j++) {
            weights[j] = (weights[j] ?? 0) / l2;
          }
        }
      }

      this.featureLogProb_.push(weights);
    }

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("ComplementNB must be fitted before prediction");
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
      throw new NotFittedError("ComplementNB must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "ComplementNB");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const nClasses = this.classes_.length;
    const probabilities: number[][] = [];

    for (let i = 0; i < nSamples; i++) {
      const logProbs: number[] = [];
      for (let c = 0; c < nClasses; c++) {
        // ComplementNB decision score = -sum(x_j * complement_weight_j).
        // scikit-learn only adds the class log-prior in the degenerate
        // single-class case; with >1 class the prior is excluded (the
        // complement formulation already accounts for class balance).
        let logP = nClasses === 1 ? (this.classLogPrior_[c] ?? 0) : 0;
        const flp = this.featureLogProb_[c] ?? [];
        for (let j = 0; j < nFeatures; j++) {
          const xVal = Number(X.data[X.offset + i * nFeatures + j] ?? 0);
          logP -= xVal * (flp[j] ?? 0);
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
    return { alpha: this.alpha, fitPrior: this.fitPrior, norm: this.norm };
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
        case "norm":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError(
              `norm must be a boolean; got ${String(value)}`,
              "norm",
              value
            );
          }
          this.norm = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
