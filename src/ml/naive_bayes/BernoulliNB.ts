/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../../core";
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

function validateBinarize(value: unknown): number | null {
  if (value === null) return null;
  if (typeof value !== "number" || Number.isNaN(value)) {
    throw new InvalidParameterError(
      `binarize must be a number or null; got ${String(value)}`,
      "binarize",
      value
    );
  }
  return value;
}

/**
 * Bernoulli Naive Bayes classifier.
 *
 * Like MultinomialNB, this classifier is suitable for discrete data.
 * The difference is that while MultinomialNB works with occurrence counts,
 * BernoulliNB is designed for binary/boolean features. A feature counts as
 * present when its value is strictly greater than the `binarize` threshold. With
 * `binarize: null` the threshold is 0, so 0/1 data is used as is. Absent features
 * contribute `log(1 - p)` to the score, which distinguishes this model from
 * MultinomialNB.
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
  private classPrior: number[] | null;

  private classes_?: number[];
  private classLogPrior_?: Float64Array;
  private featureLogProb_?: Float64Array; // log P(x_j=1 | y=c), [nClasses * nFeatures]
  private featureLogCompProb_?: Float64Array; // log P(x_j=0 | y=c), [nClasses * nFeatures]
  private nFeaturesIn_?: number;
  private fitted = false;

  /**
   * @param options.alpha - Additive smoothing (default: 1.0)
   * @param options.binarize - Threshold for binarizing features (default: 0.0): values greater than the threshold become 1, others 0. Set to null for data that is already 0/1 (values above 0 count as 1).
   * @param options.fitPrior - Whether to learn class prior probabilities (default: true). When false a uniform prior is used.
   * @param options.classPrior - Fixed class prior probabilities in the order of the sorted class labels. Overrides `fitPrior`.
   * @throws {InvalidParameterError} If an option is invalid
   */
  constructor(
    options: {
      readonly alpha?: number;
      readonly binarize?: number | null;
      readonly fitPrior?: boolean;
      readonly classPrior?: readonly number[] | null;
    } = {}
  ) {
    this.alpha = nbValidateAlpha(options.alpha ?? 1.0);
    this.binarize = validateBinarize(options.binarize === undefined ? 0.0 : options.binarize);
    this.fitPrior = nbValidateBoolean("fitPrior", options.fitPrior ?? true);
    this.classPrior = nbValidatePriors("classPrior", options.classPrior);
  }

  /**
   * Fit the classifier.
   *
   * @param X - Training samples of shape (n_samples, n_features)
   * @param y - Class labels of shape (n_samples,)
   * @returns this
   * @throws {ShapeError} If X is not 2-D, y is not 1-D or their lengths differ
   * @throws {InvalidParameterError} If `classPrior` does not have one entry per class
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const xv = toFloat64View(X);
    const threshold = this.binarize ?? 0;

    const { classes, labelIndex, counts } = nbEncodeLabels(y);
    const nClasses = classes.length;

    // featureCounts[c * nFeatures + j] = samples of class c in which feature j is present
    const featureCounts = new Float64Array(nClasses * nFeatures);
    for (let i = 0; i < nSamples; i++) {
      const base = (labelIndex[i] as number) * nFeatures;
      const row = i * nFeatures;
      for (let j = 0; j < nFeatures; j++) {
        if ((xv[row + j] as number) > threshold) {
          featureCounts[base + j] = (featureCounts[base + j] as number) + 1;
        }
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
    const featureLogCompProb = new Float64Array(nClasses * nFeatures);
    for (let c = 0; c < nClasses; c++) {
      const nc = counts[c] as number;
      const smoothedTotal = nc + 2 * this.alpha;
      for (let j = 0; j < nFeatures; j++) {
        const present = featureCounts[c * nFeatures + j] as number;
        // Both terms are computed directly so that 1 - p does not lose precision.
        featureLogProb[c * nFeatures + j] = Math.log((present + this.alpha) / smoothedTotal);
        featureLogCompProb[c * nFeatures + j] = Math.log(
          (nc - present + this.alpha) / smoothedTotal
        );
      }
    }

    this.classes_ = classes;
    this.classLogPrior_ = classLogPrior;
    this.featureLogProb_ = featureLogProb;
    this.featureLogCompProb_ = featureLogCompProb;
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
      !this.featureLogCompProb_ ||
      this.nFeaturesIn_ === undefined
    ) {
      throw new NotFittedError("BernoulliNB must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "BernoulliNB");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = this.nFeaturesIn_;
    const nClasses = this.classes_.length;
    const xv = toFloat64View(X);
    const threshold = this.binarize ?? 0;
    const flp = this.featureLogProb_;
    const flcp = this.featureLogCompProb_;
    const jll = new Float64Array(nSamples * nClasses);

    for (let i = 0; i < nSamples; i++) {
      const row = i * nFeatures;
      for (let c = 0; c < nClasses; c++) {
        const base = c * nFeatures;
        let logP = this.classLogPrior_[c] as number;
        for (let j = 0; j < nFeatures; j++) {
          logP +=
            (xv[row + j] as number) > threshold
              ? (flp[base + j] as number)
              : (flcp[base + j] as number);
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
      binarize: this.binarize,
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
        case "binarize":
          this.binarize = validateBinarize(value);
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
   * @returns A new BernoulliNB
   */
  clone(): BernoulliNB {
    return new BernoulliNB(this.getParams() as ConstructorParameters<typeof BernoulliNB>[0]);
  }
}
