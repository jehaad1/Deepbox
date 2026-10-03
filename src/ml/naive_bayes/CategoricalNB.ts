/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
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

type MinCategories = number | number[] | null;

function validateMinCategories(value: unknown): MinCategories {
  if (value === null || value === undefined) return null;
  const check = (v: unknown): number => {
    if (typeof v !== "number" || !Number.isInteger(v) || v < 1) {
      throw new InvalidParameterError(
        `minCategories must be null, a positive integer or an array of positive integers; got ${String(v)}`,
        "minCategories",
        value
      );
    }
    return v;
  };
  if (Array.isArray(value)) {
    if (value.length === 0) {
      throw new InvalidParameterError(
        "minCategories must not be an empty array",
        "minCategories",
        value
      );
    }
    return (value as unknown[]).map(check);
  }
  return check(value);
}

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
 * **Categories**: a feature's categories are the distinct values seen in training. When
 * every training value of a feature is a non-negative integer (ordinal codes, as
 * scikit-learn requires), the categories are `0..max`, so codes that never appear still
 * count towards the smoothing denominator; `minCategories` raises that count further. A
 * category not seen for a class, or not seen at all, gets probability
 * `alpha / (count_c + alpha * n_categories)`.
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
  private classPrior: number[] | null;
  private minCategories: MinCategories;

  private classes_?: number[];
  private classLogPrior_?: Float64Array;
  // Per feature: category value -> column in featureLogProb_[j].
  private categoryIndex_?: Array<Map<number, number>>;
  // Per feature: log probabilities, [nClasses * (nObserved + 1)]; the last column of a
  // class holds the value used for categories that were not observed with that class.
  private featureLogProb_?: Float64Array[];
  private nCategories_?: number[];
  private nFeaturesIn_?: number;
  private fitted = false;

  /**
   * @param options.alpha - Additive (Laplace) smoothing (default: 1.0)
   * @param options.fitPrior - Whether to learn class prior probabilities (default: true). When false a uniform prior is used.
   * @param options.classPrior - Fixed class prior probabilities in the order of the sorted class labels. Overrides `fitPrior`.
   * @param options.minCategories - Minimum number of categories, as one integer for all features or one per feature. Use it when some categories are absent from the training data (default: null).
   * @throws {InvalidParameterError} If an option is invalid
   */
  constructor(
    options: {
      readonly alpha?: number;
      readonly fitPrior?: boolean;
      readonly classPrior?: readonly number[] | null;
      readonly minCategories?: number | readonly number[] | null;
    } = {}
  ) {
    this.alpha = nbValidateAlpha(options.alpha ?? 1.0);
    this.fitPrior = nbValidateBoolean("fitPrior", options.fitPrior ?? true);
    this.classPrior = nbValidatePriors("classPrior", options.classPrior);
    this.minCategories = validateMinCategories(options.minCategories);
  }

  /**
   * Fit the classifier.
   *
   * @param X - Categorical features of shape (n_samples, n_features)
   * @param y - Class labels of shape (n_samples,)
   * @returns this
   * @throws {ShapeError} If X is not 2-D, y is not 1-D or their lengths differ
   * @throws {InvalidParameterError} If `classPrior` does not have one entry per class, or `minCategories` is an array whose length differs from the number of features
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const xv = toFloat64View(X);

    if (Array.isArray(this.minCategories) && this.minCategories.length !== nFeatures) {
      throw new InvalidParameterError(
        `minCategories must have one entry per feature; got ${this.minCategories.length} entries for ${nFeatures} features`,
        "minCategories",
        this.minCategories
      );
    }

    const { classes, labelIndex, counts } = nbEncodeLabels(y);
    const nClasses = classes.length;
    const classLogPrior = nbClassLogPrior(
      counts,
      nSamples,
      this.fitPrior,
      this.classPrior,
      "classPrior"
    );

    const categoryIndex: Array<Map<number, number>> = [];
    const featureLogProb: Float64Array[] = [];
    const nCategories: number[] = [];

    for (let j = 0; j < nFeatures; j++) {
      // Distinct values in ascending order; flag ordinal (non-negative integer) codes.
      const distinct = new Set<number>();
      let ordinal = true;
      let maxValue = 0;
      for (let i = 0; i < nSamples; i++) {
        const v = (xv[i * nFeatures + j] as number) + 0;
        distinct.add(v);
        if (!Number.isInteger(v) || v < 0) ordinal = false;
        else if (v > maxValue) maxValue = v;
      }
      const values = Array.from(distinct).sort((a, b) => a - b);
      const index = new Map<number, number>();
      for (let k = 0; k < values.length; k++) index.set(values[k] as number, k);

      const minCat = Array.isArray(this.minCategories)
        ? (this.minCategories[j] as number)
        : (this.minCategories ?? 0);
      const nCat = Math.max(values.length, ordinal ? maxValue + 1 : 0, minCat);
      const width = values.length + 1;

      const table = new Float64Array(nClasses * width);
      for (let i = 0; i < nSamples; i++) {
        const col = index.get((xv[i * nFeatures + j] as number) + 0) as number;
        const slot = (labelIndex[i] as number) * width + col;
        table[slot] = (table[slot] as number) + 1;
      }
      for (let c = 0; c < nClasses; c++) {
        const denominator = (counts[c] as number) + this.alpha * nCat;
        for (let k = 0; k < values.length; k++) {
          table[c * width + k] = Math.log(
            ((table[c * width + k] as number) + this.alpha) / denominator
          );
        }
        table[c * width + values.length] = Math.log(this.alpha / denominator);
      }

      categoryIndex.push(index);
      featureLogProb.push(table);
      nCategories.push(nCat);
    }

    this.classes_ = classes;
    this.classLogPrior_ = classLogPrior;
    this.categoryIndex_ = categoryIndex;
    this.featureLogProb_ = featureLogProb;
    this.nCategories_ = nCategories;
    this.nFeaturesIn_ = nFeatures;
    this.fitted = true;
    return this;
  }

  private jointLogLikelihood(X: Tensor): Float64Array {
    if (
      !this.fitted ||
      !this.classes_ ||
      !this.classLogPrior_ ||
      !this.categoryIndex_ ||
      !this.featureLogProb_ ||
      this.nFeaturesIn_ === undefined
    ) {
      throw new NotFittedError("CategoricalNB must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "CategoricalNB");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = this.nFeaturesIn_;
    const nClasses = this.classes_.length;
    const xv = toFloat64View(X);
    const jll = new Float64Array(nSamples * nClasses);

    for (let i = 0; i < nSamples; i++) {
      const out = i * nClasses;
      for (let c = 0; c < nClasses; c++) jll[out + c] = this.classLogPrior_[c] as number;
      for (let j = 0; j < nFeatures; j++) {
        const index = this.categoryIndex_[j] as Map<number, number>;
        const table = this.featureLogProb_[j] as Float64Array;
        const width = index.size + 1;
        // Categories never seen in training use the last column of each class.
        const col = index.get((xv[i * nFeatures + j] as number) + 0) ?? width - 1;
        for (let c = 0; c < nClasses; c++) {
          jll[out + c] = (jll[out + c] as number) + (table[c * width + col] as number);
        }
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

  /**
   * Number of categories used for smoothing in each feature, as a 1-D tensor.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nCategories(): Tensor {
    if (!this.fitted || !this.nCategories_) {
      throw new NotFittedError("CategoricalNB must be fitted to access nCategories");
    }
    return tensor(Int32Array.from(this.nCategories_));
  }

  /** Hyperparameters of this estimator. */
  getParams(): Record<string, unknown> {
    return {
      alpha: this.alpha,
      fitPrior: this.fitPrior,
      // Only reported when set, so the default parameter object stays unchanged.
      ...(this.classPrior === null ? {} : { classPrior: [...this.classPrior] }),
      ...(this.minCategories === null
        ? {}
        : {
            minCategories: Array.isArray(this.minCategories)
              ? [...this.minCategories]
              : this.minCategories,
          }),
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
        case "minCategories":
          this.minCategories = validateMinCategories(value);
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
   * @returns A new CategoricalNB
   */
  clone(): CategoricalNB {
    return new CategoricalNB(this.getParams() as ConstructorParameters<typeof CategoricalNB>[0]);
  }
}
