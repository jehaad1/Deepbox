import {
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  NotImplementedError,
  ShapeError,
} from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier } from "../base";

/**
 * Gaussian Naive Bayes classifier.
 *
 * Implements the Gaussian Naive Bayes algorithm for classification.
 * Assumes features follow a Gaussian (normal) distribution.
 *
 * **Algorithm**:
 * 1. Calculate mean and variance for each feature per class
 * 2. For prediction, calculate likelihood using Gaussian PDF
 * 3. Apply Bayes' theorem to get posterior probabilities
 * 4. Predict class with highest posterior probability
 *
 * **Time Complexity**:
 * - Training: O(n * d) where n=samples, d=features
 * - Prediction: O(k * d) per sample where k=classes
 *
 * @example
 * ```ts
 * import { GaussianNB } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [2, 3], [3, 4], [4, 5]]);
 * const y = tensor([0, 0, 1, 1]);
 *
 * const nb = new GaussianNB();
 * nb.fit(X, y);
 *
 * const predictions = nb.predict(tensor([[2.5, 3.5]]));
 * ```
 *
 * @see {@link https://en.wikipedia.org/wiki/Naive_Bayes_classifier | Wikipedia: Naive Bayes}
 * @see {@link https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html | scikit-learn GaussianNB}
 */
export class GaussianNB implements Classifier {
  private readonly varSmoothing: number;

  private classes_?: number[];
  private classPrior_?: Map<number, number>;
  private theta_?: Map<number, number[]>; // Mean for each class and feature
  private var_?: Map<number, number[]>; // Variance for each class and feature
  private nFeaturesIn_?: number;
  private fitted = false;

  /**
   * Create a new Gaussian Naive Bayes classifier.
   *
   * @param options - Configuration options
   * @param options.varSmoothing - Portion of largest variance added to variances for stability (default: 1e-9)
   */
  constructor(
    options: {
      readonly varSmoothing?: number;
    } = {}
  ) {
    this.varSmoothing = options.varSmoothing ?? 1e-9;
    if (!Number.isFinite(this.varSmoothing) || this.varSmoothing < 0) {
      throw new InvalidParameterError(
        "varSmoothing must be a finite number >= 0",
        "varSmoothing",
        this.varSmoothing
      );
    }
  }

  /**
   * Fit Gaussian Naive Bayes classifier from the training set.
   *
   * Computes per-class mean, variance, and prior probabilities.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target class labels of shape (n_samples,)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D or y is not 1D
   * @throws {ShapeError} If X and y have different number of samples
   * @throws {DataValidationError} If X or y contain NaN/Inf values
   * @throws {DataValidationError} If zero variance encountered with varSmoothing=0
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nFeatures;

    // Get unique classes
    const classSet = new Set<number>();
    for (let i = 0; i < y.size; i++) {
      classSet.add(Number(y.data[y.offset + i]));
    }
    this.classes_ = Array.from(classSet).sort((a, b) => a - b);

    // Calculate class priors
    this.classPrior_ = new Map();
    for (const cls of this.classes_) {
      let count = 0;
      for (let i = 0; i < nSamples; i++) {
        if (Number(y.data[y.offset + i]) === cls) {
          count++;
        }
      }
      this.classPrior_.set(cls, count / nSamples);
    }

    // Calculate mean and variance for each class and feature
    this.theta_ = new Map();
    this.var_ = new Map();

    for (const cls of this.classes_) {
      const classSamples: number[][] = [];

      for (let i = 0; i < nSamples; i++) {
        if (Number(y.data[y.offset + i]) === cls) {
          const sample: number[] = [];
          for (let j = 0; j < nFeatures; j++) {
            sample.push(Number(X.data[X.offset + i * nFeatures + j]));
          }
          classSamples.push(sample);
        }
      }

      const means: number[] = [];
      const variances: number[] = [];

      for (let j = 0; j < nFeatures; j++) {
        // Calculate mean
        let sum = 0;
        for (const sample of classSamples) {
          sum += sample[j] ?? 0;
        }
        const mean = sum / classSamples.length;
        means.push(mean);

        // Calculate variance
        let varSum = 0;
        for (const sample of classSamples) {
          const diff = (sample[j] ?? 0) - mean;
          varSum += diff * diff;
        }
        const variance = varSum / classSamples.length;
        if (variance === 0 && this.varSmoothing === 0) {
          throw new DataValidationError(
            "Zero variance encountered with varSmoothing=0; set varSmoothing > 0 to avoid degenerate Gaussians"
          );
        }
        variances.push(variance);
      }

      this.theta_.set(cls, means);
      this.var_.set(cls, variances);
    }

    this.fitted = true;
    return this;
  }

  /**
   * Predict class labels for samples in X.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted class labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("GaussianNB must be fitted before prediction");
    }

    const proba = this.predictProba(X);
    const nSamples = proba.shape[0] ?? 0;
    const nClasses = proba.shape[1] ?? 0;
    const predictions: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      let maxProb = -1;
      let maxClass = 0;

      for (let j = 0; j < nClasses; j++) {
        const prob = Number(proba.data[proba.offset + i * nClasses + j]);
        if (prob > maxProb) {
          maxProb = prob;
          maxClass = this.classes_?.[j] ?? 0;
        }
      }

      predictions.push(maxClass);
    }

    return tensor(predictions, { dtype: "int32" });
  }

  /**
   * Predict class probabilities for samples in X.
   *
   * Uses Bayes' theorem with Gaussian class-conditional likelihoods.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Class probability matrix of shape (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predictProba(X: Tensor): Tensor {
    if (!this.fitted || !this.classes_ || !this.classPrior_ || !this.theta_ || !this.var_) {
      throw new NotFittedError("GaussianNB must be fitted before prediction");
    }

    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "GaussianNB");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    const probabilities: number[][] = [];

    for (let i = 0; i < nSamples; i++) {
      const logProbs: number[] = [];

      for (const cls of this.classes_) {
        const prior = this.classPrior_.get(cls) ?? 0;
        const means = this.theta_.get(cls) ?? [];
        const variances = this.var_.get(cls) ?? [];

        let logProb = Math.log(prior);

        for (let j = 0; j < nFeatures; j++) {
          const x = Number(X.data[X.offset + i * nFeatures + j]);
          const mean = means[j] ?? 0;
          const variance = (variances[j] ?? 0) + this.varSmoothing;

          // Gaussian PDF in log space
          logProb -= 0.5 * Math.log(2 * Math.PI * variance);
          logProb -= (x - mean) ** 2 / (2 * variance);
        }

        logProbs.push(logProb);
      }

      // Convert log probabilities to probabilities using log-sum-exp trick
      const maxLogProb = Math.max(...logProbs);
      const expProbs = logProbs.map((lp) => Math.exp(lp - maxLogProb));
      const sumExpProbs = expProbs.reduce((a, b) => a + b, 0);
      const probs = expProbs.map((ep) => ep / sumExpProbs);

      probabilities.push(probs);
    }

    return tensor(probabilities);
  }

  /**
   * Return the mean accuracy on the given test data and labels.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True labels of shape (n_samples,)
   * @returns Accuracy score in range [0, 1]
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional or sample counts mismatch
   * @throws {DataValidationError} If y contains NaN/Inf values
   */
  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    assertContiguous(y, "y");
    for (let i = 0; i < y.size; i++) {
      const val = y.data[y.offset + i] ?? 0;
      if (!Number.isFinite(val)) {
        throw new DataValidationError("y contains non-finite values (NaN or Inf)");
      }
    }
    const yPred = this.predict(X);
    if (yPred.size !== y.size) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${yPred.size}, y=${y.size}`
      );
    }
    let correct = 0;

    for (let i = 0; i < y.size; i++) {
      if (Number(y.data[y.offset + i]) === Number(yPred.data[yPred.offset + i])) {
        correct++;
      }
    }

    return correct / y.size;
  }

  /**
   * Get the unique class labels discovered during fitting.
   *
   * @returns Tensor of class labels or undefined if not fitted
   */
  get classes(): Tensor | undefined {
    if (!this.fitted || !this.classes_) {
      return undefined;
    }
    return tensor(this.classes_, { dtype: "int32" });
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      varSmoothing: this.varSmoothing,
    };
  }

  /**
   * Set the parameters of this estimator.
   *
   * @param _params - Parameters to set
   * @throws {NotImplementedError} Always — parameters cannot be changed after construction
   */
  setParams(_params: Record<string, unknown>): this {
    throw new NotImplementedError("GaussianNB does not support setParams after construction");
  }
}
