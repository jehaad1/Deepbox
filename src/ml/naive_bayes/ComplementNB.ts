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
 * Complement Naive Bayes classifier.
 *
 * An adaptation of the standard Multinomial Naive Bayes that is particularly
 * suited for imbalanced datasets. Instead of computing the likelihood from
 * samples of a class, it uses the complement (all samples NOT in the class).
 *
 * **Algorithm** (as in scikit-learn):
 * 1. For each class c, count every feature over the samples of all other classes and
 *    smooth the counts with `alpha`.
 * 2. Take `w_cj = -log(theta_cj)`, where `theta_cj` is the smoothed complement frequency
 *    of feature j. With `norm: true` the weights of each class are divided by their sum
 *    (the L1 norm, since all weights are positive).
 * 3. Score class c as `sum_j x_j * w_cj` and predict the class with the highest score.
 *
 * The class prior is only used when there is a single class, as in scikit-learn.
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
  private classPrior: number[] | null;

  private classes_?: number[];
  private classLogPrior_?: Float64Array;
  private featureWeights_?: Float64Array; // complement weights, [nClasses * nFeatures]
  private nFeaturesIn_?: number;
  private fitted = false;

  /**
   * @param options.alpha - Additive smoothing (default: 1.0)
   * @param options.fitPrior - Whether to learn class prior probabilities (default: true). Only affects the single-class case.
   * @param options.norm - Whether to divide each class's weights by their sum (default: false)
   * @param options.classPrior - Fixed class prior probabilities in the order of the sorted class labels. Only affects the single-class case.
   * @throws {InvalidParameterError} If an option is invalid
   */
  constructor(
    options: {
      readonly alpha?: number;
      readonly fitPrior?: boolean;
      readonly norm?: boolean;
      readonly classPrior?: readonly number[] | null;
    } = {}
  ) {
    this.alpha = nbValidateAlpha(options.alpha ?? 1.0);
    this.fitPrior = nbValidateBoolean("fitPrior", options.fitPrior ?? true);
    this.norm = nbValidateBoolean("norm", options.norm ?? false);
    this.classPrior = nbValidatePriors("classPrior", options.classPrior);
  }

  /**
   * Fit the classifier on non-negative feature counts.
   *
   * @param X - Training counts of shape (n_samples, n_features)
   * @param y - Class labels of shape (n_samples,)
   * @returns this
   * @throws {DataValidationError} If X contains negative values, or `alpha` is 0 and a class complement has no counts (or, with `norm`, a feature is missing from it)
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
        throw new DataValidationError("ComplementNB requires non-negative feature values");
      }
    }

    const { classes, labelIndex, counts } = nbEncodeLabels(y);
    const nClasses = classes.length;

    const featureSums = new Float64Array(nClasses * nFeatures);
    const totalSums = new Float64Array(nFeatures);
    for (let i = 0; i < nSamples; i++) {
      const base = (labelIndex[i] as number) * nFeatures;
      const row = i * nFeatures;
      for (let j = 0; j < nFeatures; j++) {
        const v = xv[row + j] as number;
        featureSums[base + j] = (featureSums[base + j] as number) + v;
        totalSums[j] = (totalSums[j] as number) + v;
      }
    }

    const classLogPrior = nbClassLogPrior(
      counts,
      nSamples,
      this.fitPrior,
      this.classPrior,
      "classPrior"
    );

    // Complement counts: everything except the samples of class c.
    const weights = new Float64Array(nClasses * nFeatures);
    for (let c = 0; c < nClasses; c++) {
      const base = c * nFeatures;
      let complementTotal = 0;
      for (let j = 0; j < nFeatures; j++) {
        const cs = (totalSums[j] as number) - (featureSums[base + j] as number);
        weights[base + j] = cs;
        complementTotal += cs;
      }
      const smoothedTotal = complementTotal + this.alpha * nFeatures;
      if (!(smoothedTotal > 0)) {
        throw new DataValidationError(
          `The samples outside class ${String(classes[c])} have no feature counts and alpha is 0; use alpha > 0`
        );
      }
      let logSum = 0;
      for (let j = 0; j < nFeatures; j++) {
        const logTheta = Math.log(((weights[base + j] as number) + this.alpha) / smoothedTotal);
        weights[base + j] = logTheta;
        logSum += logTheta;
      }
      if (this.norm) {
        if (!Number.isFinite(logSum)) {
          throw new DataValidationError(
            `Cannot normalize the weights of class ${String(classes[c])}: use alpha > 0 so every feature has a finite weight`
          );
        }
        // logSum is 0 only when every weight is 0 (a single feature): nothing to scale.
        if (logSum !== 0) {
          for (let j = 0; j < nFeatures; j++) {
            weights[base + j] = (weights[base + j] as number) / logSum;
          }
        }
      } else {
        for (let j = 0; j < nFeatures; j++) {
          weights[base + j] = -(weights[base + j] as number);
        }
      }
    }

    this.classes_ = classes;
    this.classLogPrior_ = classLogPrior;
    this.featureWeights_ = weights;
    this.nFeaturesIn_ = nFeatures;
    this.fitted = true;
    return this;
  }

  private jointLogLikelihood(X: Tensor): Float64Array {
    if (
      !this.fitted ||
      !this.classes_ ||
      !this.classLogPrior_ ||
      !this.featureWeights_ ||
      this.nFeaturesIn_ === undefined
    ) {
      throw new NotFittedError("ComplementNB must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "ComplementNB");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = this.nFeaturesIn_;
    const nClasses = this.classes_.length;
    const xv = toFloat64View(X);
    const w = this.featureWeights_;
    const jll = new Float64Array(nSamples * nClasses);

    for (let i = 0; i < nSamples; i++) {
      const row = i * nFeatures;
      for (let c = 0; c < nClasses; c++) {
        const base = c * nFeatures;
        // The prior is only added in the degenerate single-class case, like scikit-learn:
        // the complement formulation already accounts for class balance.
        let score = nClasses === 1 ? (this.classLogPrior_[c] as number) : 0;
        for (let j = 0; j < nFeatures; j++) {
          const x = xv[row + j] as number;
          // Skipping x = 0 keeps 0 * Infinity (alpha = 0) from turning into NaN.
          if (x !== 0) score += x * (w[base + j] as number);
        }
        jll[i * nClasses + c] = score;
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
   * Predict class probabilities (softmax of the complement scores).
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
      norm: this.norm,
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
        case "norm":
          this.norm = nbValidateBoolean("norm", value);
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
   * @returns A new ComplementNB
   */
  clone(): ComplementNB {
    return new ComplementNB(this.getParams() as ConstructorParameters<typeof ComplementNB>[0]);
  }
}
