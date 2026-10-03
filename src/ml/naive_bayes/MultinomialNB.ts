/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError } from "../../core";
import type { Tensor } from "../../ndarray";
import { toFloat64View, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier } from "../base";
import {
  nbAccuracy,
  nbArgmaxRows,
  nbClassesTensor,
  nbClassLogPrior,
  nbEncodeLabels,
  nbLabelTensor,
  nbLogSoftmaxRows,
  nbMatrixTensor,
  nbValidateAlpha,
  nbValidateBoolean,
  nbValidatePriors,
  nbValidateScoreTarget,
} from "./index";

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
  private classPrior: number[] | null;

  private classes_?: number[];
  private classLogPrior_?: Float64Array;
  private featureLogProb_?: Float64Array; // [nClasses * nFeatures]
  private nFeaturesIn_?: number;
  private fitted = false;

  /**
   * @param options.alpha - Additive (Laplace/Lidstone) smoothing (default: 1.0)
   * @param options.fitPrior - Whether to learn class prior probabilities (default: true). When false a uniform prior is used.
   * @param options.classPrior - Fixed class prior probabilities in the order of the sorted class labels. Overrides `fitPrior`.
   * @throws {InvalidParameterError} If `alpha` is not a finite number >= 0 or an option has the wrong type
   */
  constructor(
    options: {
      readonly alpha?: number;
      readonly fitPrior?: boolean;
      readonly classPrior?: readonly number[] | null;
    } = {}
  ) {
    this.alpha = nbValidateAlpha(options.alpha ?? 1.0);
    this.fitPrior = nbValidateBoolean("fitPrior", options.fitPrior ?? true);
    this.classPrior = nbValidatePriors("classPrior", options.classPrior);
  }

  /**
   * Fit the classifier on non-negative feature counts.
   *
   * @param X - Training counts of shape (n_samples, n_features)
   * @param y - Class labels of shape (n_samples,)
   * @returns this
   * @throws {DataValidationError} If X contains negative values, or a class has no counts while `alpha` is 0
   * @throws {ShapeError} If X is not 2-D, y is not 1-D or their lengths differ
   * @throws {InvalidParameterError} If `classPrior` does not have one entry per class
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const xv = toFloat64View(X);

    for (let i = 0; i < xv.length; i++) {
      if ((xv[i] as number) < 0) {
        throw new DataValidationError("MultinomialNB requires non-negative feature values");
      }
    }

    const { classes, labelIndex, counts } = nbEncodeLabels(y);
    const nClasses = classes.length;

    const featureCounts = new Float64Array(nClasses * nFeatures);
    for (let i = 0; i < nSamples; i++) {
      const base = (labelIndex[i] as number) * nFeatures;
      const row = i * nFeatures;
      for (let j = 0; j < nFeatures; j++) {
        featureCounts[base + j] = (featureCounts[base + j] as number) + (xv[row + j] as number);
      }
    }

    const classLogPrior = nbClassLogPrior(
      counts,
      nSamples,
      this.fitPrior,
      this.classPrior,
      "classPrior"
    );

    const featureLogProb = new Float64Array(nClasses * nFeatures);
    for (let c = 0; c < nClasses; c++) {
      const base = c * nFeatures;
      let total = 0;
      for (let j = 0; j < nFeatures; j++) total += featureCounts[base + j] as number;
      const smoothedTotal = total + this.alpha * nFeatures;
      if (!(smoothedTotal > 0)) {
        throw new DataValidationError(
          `Class ${String(classes[c])} has no feature counts and alpha is 0; use alpha > 0`
        );
      }
      for (let j = 0; j < nFeatures; j++) {
        featureLogProb[base + j] = Math.log(
          ((featureCounts[base + j] as number) + this.alpha) / smoothedTotal
        );
      }
    }

    this.classes_ = classes;
    this.classLogPrior_ = classLogPrior;
    this.featureLogProb_ = featureLogProb;
    this.nFeaturesIn_ = nFeatures;
    this.fitted = true;
    return this;
  }

  private jointLogLikelihood(X: Tensor): Float64Array {
    if (
      !this.fitted ||
      !this.classes_ ||
      !this.classLogPrior_ ||
      !this.featureLogProb_ ||
      this.nFeaturesIn_ === undefined
    ) {
      throw new NotFittedError("MultinomialNB must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "MultinomialNB");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = this.nFeaturesIn_;
    const nClasses = this.classes_.length;
    const xv = toFloat64View(X);
    const flp = this.featureLogProb_;
    const jll = new Float64Array(nSamples * nClasses);

    for (let i = 0; i < nSamples; i++) {
      const row = i * nFeatures;
      for (let c = 0; c < nClasses; c++) {
        const base = c * nFeatures;
        let logP = this.classLogPrior_[c] as number;
        for (let j = 0; j < nFeatures; j++) {
          const x = xv[row + j] as number;
          // Skipping x = 0 keeps 0 * -Infinity (alpha = 0) from turning into NaN.
          if (x !== 0) logP += x * (flp[base + j] as number);
        }
        jll[i * nClasses + c] = logP;
      }
    }
    return jll;
  }

  /**
   * Predict class labels.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong rank or feature count
   */
  predict(X: Tensor): Tensor {
    const jll = this.jointLogLikelihood(X);
    const classes = this.classes_ as number[];
    const nSamples = X.shape[0] ?? 0;
    const best = nbArgmaxRows(jll, nSamples, classes.length);
    const labels = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) labels[i] = classes[best[i] as number] as number;
    return nbLabelTensor(labels, classes);
  }

  /**
   * Predict class probabilities.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Probabilities of shape (n_samples, n_classes); columns follow `classes`
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong rank or feature count
   */
  predictProba(X: Tensor): Tensor {
    const jll = this.jointLogLikelihood(X);
    const nSamples = X.shape[0] ?? 0;
    const nClasses = (this.classes_ as number[]).length;
    const proba = nbLogSoftmaxRows(jll, nSamples, nClasses);
    for (let i = 0; i < proba.length; i++) proba[i] = Math.exp(proba[i] as number);
    return nbMatrixTensor(proba, nSamples, nClasses);
  }

  /**
   * Predict log class probabilities.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Log probabilities of shape (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong rank or feature count
   */
  predictLogProba(X: Tensor): Tensor {
    const jll = this.jointLogLikelihood(X);
    const nSamples = X.shape[0] ?? 0;
    const nClasses = (this.classes_ as number[]).length;
    return nbMatrixTensor(nbLogSoftmaxRows(jll, nSamples, nClasses), nSamples, nClasses);
  }

  /**
   * Mean accuracy on the given data and labels.
   *
   * @param X - Test samples
   * @param y - True labels of shape (n_samples,)
   * @returns Accuracy in [0, 1]
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-D or its length differs from the number of samples
   * @throws {DataValidationError} If y is empty or contains NaN/Inf
   */
  score(X: Tensor, y: Tensor): number {
    nbValidateScoreTarget(y);
    return nbAccuracy(this.predict(X), y);
  }

  /** Sorted class labels seen during `fit`, or `undefined` before fitting. */
  get classes(): Tensor | undefined {
    if (!this.fitted || !this.classes_) return undefined;
    return nbClassesTensor(this.classes_);
  }

  /** Hyperparameters of this estimator. */
  getParams(): Record<string, unknown> {
    return {
      alpha: this.alpha,
      fitPrior: this.fitPrior,
      // Only reported when set, so the default parameter object stays unchanged.
      ...(this.classPrior === null ? {} : { classPrior: [...this.classPrior] }),
    };
  }

  /**
   * Update hyperparameters. The model must be refitted for changes to take effect.
   *
   * @throws {InvalidParameterError} If a value is invalid or the key is unknown
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "alpha":
          this.alpha = nbValidateAlpha(value);
          break;
        case "fitPrior":
          this.fitPrior = nbValidateBoolean("fitPrior", value);
          break;
        case "classPrior":
          this.classPrior = nbValidatePriors("classPrior", value);
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
   * @returns A new MultinomialNB
   */
  clone(): MultinomialNB {
    return new MultinomialNB(this.getParams() as ConstructorParameters<typeof MultinomialNB>[0]);
  }
}
