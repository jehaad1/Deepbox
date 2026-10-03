import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import {
  assertContiguous,
  toFloat64View,
  validateFitInputs,
  validatePredictInputs,
} from "../_validation";
import type { Classifier } from "../base";

// ---------------------------------------------------------------------------
// Helpers shared by the Naive Bayes classifiers in this directory.
// They are internal: `deepbox/ml` only re-exports the estimator classes.
// ---------------------------------------------------------------------------

/**
 * Class labels found in `y`, with the class index of every sample.
 *
 * @internal
 */
export type NbClassInfo = {
  /** Sorted unique labels. */
  readonly classes: number[];
  /** Index into `classes` for every sample. */
  readonly labelIndex: Int32Array;
  /** Number of samples per class. */
  readonly counts: Float64Array;
};

/**
 * Collect the sorted unique labels of a validated 1-D target tensor.
 *
 * @internal
 */
export function nbEncodeLabels(y: Tensor): NbClassInfo {
  const values = toFloat64View(y);
  const unique = new Set<number>();
  for (let i = 0; i < values.length; i++) {
    // `+ 0` turns -0 into 0 so both spellings fall into one class.
    unique.add((values[i] as number) + 0);
  }
  const classes = Array.from(unique).sort((a, b) => a - b);
  const index = new Map<number, number>();
  for (let c = 0; c < classes.length; c++) index.set(classes[c] as number, c);
  const labelIndex = new Int32Array(values.length);
  const counts = new Float64Array(classes.length);
  for (let i = 0; i < values.length; i++) {
    const c = index.get((values[i] as number) + 0) as number;
    labelIndex[i] = c;
    counts[c] = (counts[c] as number) + 1;
  }
  return { classes, labelIndex, counts };
}

/**
 * Validate the smoothing parameter shared by the discrete Naive Bayes models.
 *
 * @internal
 */
export function nbValidateAlpha(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new InvalidParameterError(
      `alpha must be a finite number >= 0; got ${String(value)}`,
      "alpha",
      value
    );
  }
  return value;
}

/**
 * Validate a boolean option.
 *
 * @internal
 */
export function nbValidateBoolean(name: string, value: unknown): boolean {
  if (typeof value !== "boolean") {
    throw new InvalidParameterError(`${name} must be a boolean; got ${String(value)}`, name, value);
  }
  return value;
}

/**
 * Validate an optional vector of class prior probabilities.
 *
 * @returns A copy of the priors, or `null` when none were given
 * @internal
 */
export function nbValidatePriors(name: string, value: unknown): number[] | null {
  if (value === null || value === undefined) return null;
  if (!Array.isArray(value) || value.length === 0) {
    throw new InvalidParameterError(
      `${name} must be null or a non-empty array of numbers; got ${String(value)}`,
      name,
      value
    );
  }
  const out: number[] = [];
  for (const p of value as unknown[]) {
    if (typeof p !== "number" || !Number.isFinite(p) || p < 0) {
      throw new InvalidParameterError(
        `${name} must contain finite numbers >= 0; got ${String(p)}`,
        name,
        value
      );
    }
    out.push(p);
  }
  return out;
}

/**
 * Log prior of every class: explicit priors, empirical frequencies, or uniform.
 *
 * @internal
 */
export function nbClassLogPrior(
  counts: Float64Array,
  nSamples: number,
  fitPrior: boolean,
  priors: readonly number[] | null,
  priorName: string
): Float64Array {
  const nClasses = counts.length;
  const out = new Float64Array(nClasses);
  if (priors !== null) {
    if (priors.length !== nClasses) {
      throw new InvalidParameterError(
        `${priorName} must have one entry per class; got ${priors.length} entries for ${nClasses} classes`,
        priorName,
        priors
      );
    }
    for (let c = 0; c < nClasses; c++) out[c] = Math.log(priors[c] as number);
  } else if (fitPrior) {
    for (let c = 0; c < nClasses; c++) out[c] = Math.log((counts[c] as number) / nSamples);
  } else {
    out.fill(-Math.log(nClasses));
  }
  return out;
}

/**
 * Build a 1-D label tensor. Labels that are all 32-bit integers produce an `int32`
 * tensor (the dtype used by the other classifiers); any other label set keeps full
 * precision in a `float64` tensor so non-integer labels are not truncated.
 *
 * @internal
 */
export function nbLabelTensor(values: ArrayLike<number>, classes: readonly number[]): Tensor {
  const integral = classes.every((c) => Number.isInteger(c) && c >= -2147483648 && c <= 2147483647);
  if (integral) return tensor(Int32Array.from(values as ArrayLike<number>));
  return tensor(Float64Array.from(values as ArrayLike<number>));
}

/**
 * Tensor of class labels, see {@link nbLabelTensor} for the dtype rule.
 *
 * @internal
 */
export function nbClassesTensor(classes: readonly number[]): Tensor {
  return nbLabelTensor(classes, classes);
}

/**
 * Index of the largest entry of every row (the first one on ties).
 *
 * @internal
 */
export function nbArgmaxRows(jll: Float64Array, nSamples: number, nClasses: number): Int32Array {
  const out = new Int32Array(nSamples);
  for (let i = 0; i < nSamples; i++) {
    const base = i * nClasses;
    let best = 0;
    let bestValue = jll[base] as number;
    for (let c = 1; c < nClasses; c++) {
      const v = jll[base + c] as number;
      if (v > bestValue) {
        bestValue = v;
        best = c;
      }
    }
    out[i] = best;
  }
  return out;
}

/**
 * Row-wise log-softmax of joint log likelihoods, computed with the log-sum-exp
 * shift. Rows where every class has probability zero (all `-Infinity`) become
 * uniform; rows with `+Infinity` entries share the probability among those classes.
 *
 * @internal
 */
export function nbLogSoftmaxRows(
  jll: Float64Array,
  nSamples: number,
  nClasses: number
): Float64Array {
  const out = new Float64Array(nSamples * nClasses);
  for (let i = 0; i < nSamples; i++) {
    const base = i * nClasses;
    let max = -Infinity;
    for (let c = 0; c < nClasses; c++) {
      const v = jll[base + c] as number;
      if (v > max || Number.isNaN(v)) max = v;
    }
    if (max === -Infinity) {
      out.fill(-Math.log(nClasses), base, base + nClasses);
      continue;
    }
    if (max === Infinity) {
      let nInf = 0;
      for (let c = 0; c < nClasses; c++) if (jll[base + c] === Infinity) nInf++;
      for (let c = 0; c < nClasses; c++) {
        out[base + c] = jll[base + c] === Infinity ? -Math.log(nInf) : -Infinity;
      }
      continue;
    }
    let sum = 0;
    for (let c = 0; c < nClasses; c++) sum += Math.exp((jll[base + c] as number) - max);
    const lse = max + Math.log(sum);
    for (let c = 0; c < nClasses; c++) out[base + c] = (jll[base + c] as number) - lse;
  }
  return out;
}

/**
 * Wrap a flat row-major buffer as an `(nSamples, nClasses)` float64 tensor.
 *
 * @internal
 */
export function nbMatrixTensor(values: Float64Array, nSamples: number, nClasses: number): Tensor {
  return tensor(values).reshape([nSamples, nClasses]);
}

/**
 * Check the target given to `score` and compare it with the predictions.
 *
 * @internal
 */
export function nbAccuracy(yPred: Tensor, y: Tensor): number {
  const n = y.size;
  if (yPred.size !== n) {
    throw new ShapeError(
      `X and y must have the same number of samples; got X=${yPred.size}, y=${n}`
    );
  }
  let correct = 0;
  for (let i = 0; i < n; i++) {
    if (Number(y.data[y.offset + i]) === Number(yPred.data[yPred.offset + i])) correct++;
  }
  return correct / n;
}

/**
 * Validate the target tensor passed to `score`.
 *
 * @internal
 */
export function nbValidateScoreTarget(y: Tensor): void {
  if (y.ndim !== 1) {
    throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
  }
  assertContiguous(y, "y");
  if (y.size === 0) {
    throw new DataValidationError("y must contain at least one sample");
  }
  for (let i = 0; i < y.size; i++) {
    // Number() so that int64 (BigInt) targets are accepted.
    if (!Number.isFinite(Number(y.data[y.offset + i]))) {
      throw new DataValidationError("y contains non-finite values (NaN or Inf)");
    }
  }
}

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
 * Like scikit-learn, `varSmoothing` is a fraction of the largest per-feature variance
 * of the whole training set; that value is added to every class variance. If every
 * feature is constant, `varSmoothing` itself is added.
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
 * @see {@link https://deepbox.dev/docs/ml-naive-bayes | Deepbox Naive Bayes}
 */
export class GaussianNB implements Classifier {
  private varSmoothing: number;
  private priors: number[] | null;

  private classes_?: number[];
  private classLogPrior_?: Float64Array;
  private theta_?: Float64Array; // mean, [nClasses * nFeatures]
  private var_?: Float64Array; // variance without smoothing, [nClasses * nFeatures]
  private epsilon_ = 0;
  private nFeaturesIn_?: number;
  private fitted = false;

  /**
   * Create a new Gaussian Naive Bayes classifier.
   *
   * @param options - Configuration options
   * @param options.varSmoothing - Fraction of the largest feature variance added to all variances for stability (default: 1e-9)
   * @param options.priors - Fixed class prior probabilities, in the order of the sorted class labels. When omitted they are estimated from the training data.
   * @throws {InvalidParameterError} If `varSmoothing` is not a finite number >= 0 or `priors` is invalid
   */
  constructor(
    options: {
      readonly varSmoothing?: number;
      readonly priors?: readonly number[] | null;
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
    this.priors = nbValidatePriors("priors", options.priors);
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
   * @throws {DataValidationError} If a smoothed variance is zero (constant feature with `varSmoothing=0`, or all features constant)
   * @throws {InvalidParameterError} If `priors` does not have one entry per class
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const xv = toFloat64View(X);
    const { classes, labelIndex, counts } = nbEncodeLabels(y);
    const nClasses = classes.length;

    // Per-class means (first pass).
    const theta = new Float64Array(nClasses * nFeatures);
    for (let i = 0; i < nSamples; i++) {
      const base = (labelIndex[i] as number) * nFeatures;
      const row = i * nFeatures;
      for (let j = 0; j < nFeatures; j++) {
        theta[base + j] = (theta[base + j] as number) + (xv[row + j] as number);
      }
    }
    for (let c = 0; c < nClasses; c++) {
      const nc = counts[c] as number;
      for (let j = 0; j < nFeatures; j++) {
        theta[c * nFeatures + j] = (theta[c * nFeatures + j] as number) / nc;
      }
    }

    // Per-class variances (second pass, population variance).
    const variance = new Float64Array(nClasses * nFeatures);
    for (let i = 0; i < nSamples; i++) {
      const base = (labelIndex[i] as number) * nFeatures;
      const row = i * nFeatures;
      for (let j = 0; j < nFeatures; j++) {
        const d = (xv[row + j] as number) - (theta[base + j] as number);
        variance[base + j] = (variance[base + j] as number) + d * d;
      }
    }
    for (let c = 0; c < nClasses; c++) {
      const nc = counts[c] as number;
      for (let j = 0; j < nFeatures; j++) {
        variance[c * nFeatures + j] = (variance[c * nFeatures + j] as number) / nc;
      }
    }

    // epsilon = varSmoothing * largest per-feature variance of the full training set.
    let maxVar = 0;
    for (let j = 0; j < nFeatures; j++) {
      let mean = 0;
      for (let i = 0; i < nSamples; i++) mean += xv[i * nFeatures + j] as number;
      mean /= nSamples;
      let acc = 0;
      for (let i = 0; i < nSamples; i++) {
        const d = (xv[i * nFeatures + j] as number) - mean;
        acc += d * d;
      }
      maxVar = Math.max(maxVar, acc / nSamples);
    }
    // With all features constant there is no variance to scale, so the absolute value is used.
    const epsilon = maxVar > 0 ? this.varSmoothing * maxVar : this.varSmoothing;

    for (let k = 0; k < variance.length; k++) {
      if ((variance[k] as number) + epsilon <= 0) {
        throw new DataValidationError(
          "Zero variance encountered; increase varSmoothing or remove constant features"
        );
      }
    }

    const classLogPrior = nbClassLogPrior(counts, nSamples, true, this.priors, "priors");

    this.classes_ = classes;
    this.classLogPrior_ = classLogPrior;
    this.theta_ = theta;
    this.var_ = variance;
    this.epsilon_ = epsilon;
    this.nFeaturesIn_ = nFeatures;
    this.fitted = true;
    return this;
  }

  private jointLogLikelihood(X: Tensor): Float64Array {
    if (
      !this.fitted ||
      !this.classes_ ||
      !this.classLogPrior_ ||
      !this.theta_ ||
      !this.var_ ||
      this.nFeaturesIn_ === undefined
    ) {
      throw new NotFittedError("GaussianNB must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "GaussianNB");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = this.nFeaturesIn_;
    const nClasses = this.classes_.length;
    const xv = toFloat64View(X);
    const theta = this.theta_;
    const eps = this.epsilon_;

    // Per class constant: log prior - 0.5 * sum(log(2 pi var)).
    const constant = new Float64Array(nClasses);
    const invVar = new Float64Array(nClasses * nFeatures);
    for (let c = 0; c < nClasses; c++) {
      let acc = this.classLogPrior_[c] as number;
      for (let j = 0; j < nFeatures; j++) {
        const v = (this.var_[c * nFeatures + j] as number) + eps;
        acc -= 0.5 * Math.log(2 * Math.PI * v);
        invVar[c * nFeatures + j] = 1 / v;
      }
      constant[c] = acc;
    }

    const jll = new Float64Array(nSamples * nClasses);
    for (let i = 0; i < nSamples; i++) {
      const row = i * nFeatures;
      for (let c = 0; c < nClasses; c++) {
        const base = c * nFeatures;
        let quad = 0;
        for (let j = 0; j < nFeatures; j++) {
          const d = (xv[row + j] as number) - (theta[base + j] as number);
          quad += d * d * (invVar[base + j] as number);
        }
        jll[i * nClasses + c] = (constant[c] as number) - 0.5 * quad;
      }
    }
    return jll;
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
    const jll = this.jointLogLikelihood(X);
    const classes = this.classes_ as number[];
    const nSamples = X.shape[0] ?? 0;
    const best = nbArgmaxRows(jll, nSamples, classes.length);
    const labels = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) labels[i] = classes[best[i] as number] as number;
    return nbLabelTensor(labels, classes);
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
    const jll = this.jointLogLikelihood(X);
    const nSamples = X.shape[0] ?? 0;
    const nClasses = (this.classes_ as number[]).length;
    const proba = nbLogSoftmaxRows(jll, nSamples, nClasses);
    for (let i = 0; i < proba.length; i++) proba[i] = Math.exp(proba[i] as number);
    return nbMatrixTensor(proba, nSamples, nClasses);
  }

  /**
   * Predict log class probabilities for samples in X.
   *
   * More accurate than `log(predictProba(X))` when probabilities are tiny.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Log probabilities of shape (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predictLogProba(X: Tensor): Tensor {
    const jll = this.jointLogLikelihood(X);
    const nSamples = X.shape[0] ?? 0;
    const nClasses = (this.classes_ as number[]).length;
    return nbMatrixTensor(nbLogSoftmaxRows(jll, nSamples, nClasses), nSamples, nClasses);
  }

  /**
   * Return the mean accuracy on the given test data and labels.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True labels of shape (n_samples,)
   * @returns Accuracy score in range [0, 1]
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional or sample counts mismatch
   * @throws {DataValidationError} If y is empty or contains NaN/Inf values
   */
  score(X: Tensor, y: Tensor): number {
    nbValidateScoreTarget(y);
    return nbAccuracy(this.predict(X), y);
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
    return nbClassesTensor(this.classes_);
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      varSmoothing: this.varSmoothing,
      ...(this.priors === null ? {} : { priors: [...this.priors] }),
    };
  }

  /**
   * Set the parameters of this estimator.
   *
   * @param params - Parameters to set
   * @throws {InvalidParameterError} If a parameter value is invalid or unknown
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "varSmoothing":
          if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
            throw new InvalidParameterError(
              `varSmoothing must be a finite number >= 0; got ${String(value)}`,
              "varSmoothing",
              value
            );
          }
          this.varSmoothing = value;
          break;
        case "priors":
          this.priors = nbValidatePriors("priors", value);
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
   * @returns A new GaussianNB
   */
  clone(): GaussianNB {
    return new GaussianNB(this.getParams() as ConstructorParameters<typeof GaussianNB>[0]);
  }
}
