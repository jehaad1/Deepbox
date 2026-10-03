/**
 * Discriminant Analysis classifiers.
 *
 * Implements Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA).
 *
 * LDA projects data to a lower-dimensional space while maximizing class separability,
 * and can also be used as a classifier. QDA fits a separate covariance matrix per class,
 * allowing for quadratic decision boundaries.
 *
 * @see {@link https://deepbox.dev/docs/ml-advanced | Deepbox Discriminant Analysis}
 */

import {
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
  warn,
} from "../core";
import { type Tensor, tensor } from "../ndarray";
import { jacobiEigenSymmetric } from "./_internal";
import {
  assertContiguous,
  toFloat64View,
  validateFitInputs,
  validatePredictInputs,
} from "./_validation";
import type { Classifier, Transformer } from "./base";

// ---- Shared helpers ----

type FloatDType = "float32" | "float64";

function floatDTypeOf(X: Tensor): FloatDType {
  return X.dtype === "float32" ? "float32" : "float64";
}

function makeTensor(data: Float64Array, shape: number[], dtype: FloatDType): Tensor {
  const flat = dtype === "float64" ? tensor(data) : tensor(Float32Array.from(data));
  return flat.reshape(shape);
}

/** Labels as int32 when they are all integers in range, otherwise float64. */
function labelTensor(values: ArrayLike<number>): Tensor {
  let allInt32 = true;
  for (let i = 0; i < values.length; i++) {
    const v = values[i] as number;
    if (!Number.isInteger(v) || v < -2147483648 || v > 2147483647) {
      allInt32 = false;
      break;
    }
  }
  return allInt32
    ? tensor(Int32Array.from(values as ArrayLike<number>))
    : tensor(Float64Array.from(values as ArrayLike<number>));
}

/** Check the shape-independent properties of a priors array. */
function checkPriorValues(priors: readonly number[]): void {
  if (!Array.isArray(priors)) {
    throw new InvalidParameterError("priors must be an array of numbers", "priors", priors);
  }
  let sum = 0;
  for (const p of priors) {
    if (typeof p !== "number" || !Number.isFinite(p) || p < 0) {
      throw new InvalidParameterError("priors must be finite and non-negative", "priors", priors);
    }
    sum += p;
  }
  if (Math.abs(sum - 1) > 1e-6) {
    throw new InvalidParameterError("priors must sum to 1", "priors", priors);
  }
}

function resolvePriors(
  priors: readonly number[] | undefined,
  counts: Float64Array,
  nSamples: number
): Float64Array {
  const nClasses = counts.length;
  const out = new Float64Array(nClasses);
  if (priors) {
    if (priors.length !== nClasses) {
      throw new InvalidParameterError(
        `priors must have length ${nClasses}; got ${priors.length}`,
        "priors",
        priors
      );
    }
    checkPriorValues(priors);
    for (let c = 0; c < nClasses; c++) out[c] = priors[c] ?? 0;
  } else {
    for (let c = 0; c < nClasses; c++) out[c] = (counts[c] ?? 0) / nSamples;
  }
  return out;
}

/** Sorted unique labels of `y` and the class index of every sample. */
function encodeLabels(y: Tensor): { classes: number[]; index: Int32Array } {
  const values = toFloat64View(y);
  const classes = Array.from(new Set(values)).sort((a, b) => a - b);
  const lookup = new Map<number, number>();
  classes.forEach((label, i) => {
    lookup.set(label, i);
  });
  const index = new Int32Array(values.length);
  for (let i = 0; i < values.length; i++) index[i] = lookup.get(values[i] ?? 0) ?? 0;
  return { classes, index };
}

/** Row-wise log-softmax of an (n x K) matrix of unnormalized log posteriors. */
function logSoftmaxRows(scores: Float64Array, n: number, K: number): Float64Array {
  const out = new Float64Array(n * K);
  for (let i = 0; i < n; i++) {
    let max = Number.NEGATIVE_INFINITY;
    for (let c = 0; c < K; c++) max = Math.max(max, scores[i * K + c] ?? Number.NEGATIVE_INFINITY);
    if (!Number.isFinite(max)) {
      for (let c = 0; c < K; c++) out[i * K + c] = -Math.log(K);
      continue;
    }
    let sum = 0;
    for (let c = 0; c < K; c++)
      sum += Math.exp((scores[i * K + c] ?? Number.NEGATIVE_INFINITY) - max);
    const logSum = max + Math.log(sum);
    for (let c = 0; c < K; c++)
      out[i * K + c] = (scores[i * K + c] ?? Number.NEGATIVE_INFINITY) - logSum;
  }
  return out;
}

function argmaxRows(scores: Float64Array, n: number, K: number): Int32Array {
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    let best = 0;
    let bestVal = Number.NEGATIVE_INFINITY;
    for (let c = 0; c < K; c++) {
      const v = scores[i * K + c] ?? Number.NEGATIVE_INFINITY;
      if (v > bestVal) {
        bestVal = v;
        best = c;
      }
    }
    out[i] = best;
  }
  return out;
}

function accuracy(predicted: Tensor, y: Tensor): number {
  if (y.ndim !== 1) {
    throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
  }
  assertContiguous(y, "y");
  const truth = toFloat64View(y);
  const pred = toFloat64View(predicted);
  if (truth.length !== pred.length) {
    throw new ShapeError(
      `X and y must have the same number of samples; got ${pred.length} and ${truth.length}`
    );
  }
  if (truth.length === 0) {
    throw new DataValidationError("y must have at least one sample");
  }
  let correct = 0;
  for (let i = 0; i < truth.length; i++) if (truth[i] === pred[i]) correct++;
  return correct / truth.length;
}

/**
 * Ledoit-Wolf shrinkage intensity for the rows of `Z` (n x f), assumed centered.
 * Same estimator as scikit-learn's `ledoit_wolf_shrinkage`.
 */
function ledoitWolfShrinkage(Z: Float64Array, n: number, f: number): number {
  if (f === 1) return 0;
  const empTrace = new Float64Array(f);
  let betaSum = 0;
  for (let i = 0; i < n; i++) {
    let rowSq = 0;
    for (let j = 0; j < f; j++) {
      const z2 = (Z[i * f + j] ?? 0) ** 2;
      empTrace[j] = (empTrace[j] ?? 0) + z2;
      rowSq += z2;
    }
    betaSum += rowSq * rowSq;
  }
  let traceSum = 0;
  for (let j = 0; j < f; j++) {
    empTrace[j] = (empTrace[j] ?? 0) / n;
    traceSum += empTrace[j] ?? 0;
  }
  const mu = traceSum / f;

  // delta_ = sum of squared entries of Z^T Z, divided by n^2
  const gram = new Float64Array(f * f);
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < f; p++) {
      const zp = Z[i * f + p] ?? 0;
      if (zp === 0) continue;
      for (let q = p; q < f; q++) {
        gram[p * f + q] = (gram[p * f + q] ?? 0) + zp * (Z[i * f + q] ?? 0);
      }
    }
  }
  let deltaSum = 0;
  for (let p = 0; p < f; p++) {
    for (let q = p; q < f; q++) {
      const g = gram[p * f + q] ?? 0;
      deltaSum += (p === q ? 1 : 2) * g * g;
    }
  }
  const delta0 = deltaSum / (n * n);
  const beta0 = (betaSum / n - delta0) / (f * n);
  const delta = (delta0 - 2 * mu * traceSum + f * mu * mu) / f;
  const beta = Math.min(beta0, delta);
  if (!(beta > 0) || !(delta > 0)) return 0;
  return Math.min(1, beta / delta);
}

/**
 * Linear Discriminant Analysis (LDA).
 *
 * A classifier with a linear decision boundary, generated by fitting class
 * conditional densities to the data and using Bayes' rule. The model fits
 * a Gaussian density to each class, assuming all classes share the same
 * covariance matrix.
 *
 * Can also be used for supervised dimensionality reduction by projecting
 * data to the most discriminant directions.
 *
 * **Algorithm** (same as scikit-learn's default `svd` solver):
 * 1. Compute class means and the pooled within-class covariance (divided by n - n_classes)
 * 2. Whiten the pooled covariance; directions with a numerically zero variance are dropped
 * 3. Take the eigenvectors of the between-class scatter in the whitened space as
 *    discriminant directions, so projected data has identity within-class covariance
 * 4. Classify with the log posterior under the shared covariance
 *
 * @example
 * ```ts
 * import { LinearDiscriminantAnalysis } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [2, 1], [3, 3], [7, 8], [8, 7], [9, 9]]);
 * const y = tensor([0, 0, 0, 1, 1, 1]);
 *
 * const lda = new LinearDiscriminantAnalysis();
 * lda.fit(X, y);
 * const predictions = lda.predict(tensor([[2, 3], [6, 7]]));
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-advanced | Deepbox Discriminant Analysis}
 */
export class LinearDiscriminantAnalysis implements Classifier, Transformer {
  private nComponents: number | undefined;
  private shrinkage: number | "auto" | undefined;
  private priors: readonly number[] | undefined;

  private classes_?: number[];
  private means_?: Float64Array; // (K, f)
  private xbar_?: Float64Array; // prior-weighted overall mean (f)
  private priors_?: Float64Array;
  private whitener_?: Float64Array; // (f, rank): maps x to whitened coordinates
  private projMeans_?: Float64Array; // (K, rank): class means in whitened coordinates
  private rank_ = 0;
  private scalings_?: Float64Array; // (f, nComp)
  private nFeaturesIn_ = 0;
  private nComponentsActual_ = 0;
  private explainedVarianceRatio_?: Float64Array;
  private outDType_: FloatDType = "float64";
  private fitted = false;

  /**
   * @param options.nComponents - Number of components for dimensionality reduction (default: min(n_classes - 1, n_features))
   * @param options.shrinkage - Shrinkage of the pooled covariance. A number in [0, 1] blends the covariance with mean-variance times identity; "auto" estimates the intensity with the Ledoit-Wolf formula on standardized data and shrinks toward the diagonal. (default: undefined = no shrinkage)
   * @param options.priors - Prior probabilities of classes, ordered by sorted class label. If not given, class proportions are used.
   */
  constructor(
    options: {
      readonly nComponents?: number;
      readonly shrinkage?: number | "auto";
      readonly priors?: readonly number[];
    } = {}
  ) {
    if (options.nComponents !== undefined) {
      this.nComponents = LinearDiscriminantAnalysis.checkNComponents(options.nComponents);
    }
    if (options.shrinkage !== undefined) {
      this.shrinkage = LinearDiscriminantAnalysis.checkShrinkage(options.shrinkage);
    }
    if (options.priors !== undefined) {
      checkPriorValues(options.priors);
      this.priors = options.priors;
    }
  }

  private static checkNComponents(value: unknown): number {
    if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
      throw new InvalidParameterError("nComponents must be an integer >= 1", "nComponents", value);
    }
    return value;
  }

  private static checkShrinkage(value: unknown): number | "auto" {
    if (value === "auto") return value;
    if (typeof value !== "number" || !(value >= 0 && value <= 1)) {
      throw new InvalidParameterError(
        "shrinkage must be 'auto' or a number in [0, 1]",
        "shrinkage",
        value
      );
    }
    return value;
  }

  /**
   * Fit the model.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Class labels of shape (n_samples,)
   * @returns this
   * @throws {DataValidationError} If there are fewer than 2 classes or not more samples than classes
   * @throws {InvalidParameterError} If nComponents or priors are inconsistent with the data
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const f = X.shape[1] ?? 0;
    const data = toFloat64View(X);

    const { classes, index } = encodeLabels(y);
    const K = classes.length;
    if (K < 2) {
      throw new DataValidationError("LDA requires at least 2 classes");
    }
    if (nSamples <= K) {
      throw new DataValidationError(
        `LDA requires more samples than classes; got n_samples=${nSamples} and n_classes=${K}`
      );
    }

    const maxComponents = Math.min(K - 1, f);
    const nComp = this.nComponents ?? maxComponents;
    if (nComp > maxComponents) {
      throw new InvalidParameterError(
        `nComponents=${nComp} must be <= min(n_classes - 1, n_features) = ${maxComponents}`,
        "nComponents",
        nComp
      );
    }

    // Class counts and means
    const counts = new Float64Array(K);
    const means = new Float64Array(K * f);
    for (let i = 0; i < nSamples; i++) {
      const c = index[i] ?? 0;
      counts[c] = (counts[c] ?? 0) + 1;
      for (let j = 0; j < f; j++) {
        means[c * f + j] = (means[c * f + j] ?? 0) + (data[i * f + j] ?? 0);
      }
    }
    for (let c = 0; c < K; c++) {
      for (let j = 0; j < f; j++) means[c * f + j] = (means[c * f + j] ?? 0) / (counts[c] ?? 1);
    }

    const priors = resolvePriors(this.priors, counts, nSamples);

    // Prior-weighted overall mean (the centre of the projection, as in scikit-learn).
    const xbar = new Float64Array(f);
    for (let c = 0; c < K; c++) {
      for (let j = 0; j < f; j++) {
        xbar[j] = (xbar[j] ?? 0) + (priors[c] ?? 0) * (means[c * f + j] ?? 0);
      }
    }

    // Within-class centered data and pooled covariance Sigma = Sw / (n - K)
    const centered = new Float64Array(nSamples * f);
    const sigma = new Float64Array(f * f);
    for (let i = 0; i < nSamples; i++) {
      const c = index[i] ?? 0;
      for (let j = 0; j < f; j++) {
        centered[i * f + j] = (data[i * f + j] ?? 0) - (means[c * f + j] ?? 0);
      }
    }
    for (let i = 0; i < nSamples; i++) {
      for (let p = 0; p < f; p++) {
        const dp = centered[i * f + p] ?? 0;
        if (dp === 0) continue;
        for (let q = p; q < f; q++) {
          sigma[p * f + q] = (sigma[p * f + q] ?? 0) + dp * (centered[i * f + q] ?? 0);
        }
      }
    }
    const denom = nSamples - K;
    for (let p = 0; p < f; p++) {
      for (let q = p; q < f; q++) {
        const v = (sigma[p * f + q] ?? 0) / denom;
        sigma[p * f + q] = v;
        sigma[q * f + p] = v;
      }
    }

    // Shrinkage. A number blends Sigma with mu * I (mu = mean variance), as scikit-learn's
    // shrunk_covariance does. "auto" estimates the intensity with Ledoit-Wolf on standardized
    // data, where the target is the identity, i.e. diag(Sigma) in the original scale.
    if (this.shrinkage === "auto") {
      const std = new Float64Array(f);
      for (let j = 0; j < f; j++) {
        const s = Math.sqrt(sigma[j * f + j] ?? 0);
        std[j] = s > 0 ? s : 1;
      }
      const Z = new Float64Array(nSamples * f);
      for (let i = 0; i < nSamples; i++) {
        for (let j = 0; j < f; j++) Z[i * f + j] = (centered[i * f + j] ?? 0) / (std[j] ?? 1);
      }
      const delta = ledoitWolfShrinkage(Z, nSamples, f);
      for (let p = 0; p < f; p++) {
        for (let q = 0; q < f; q++) {
          if (p !== q) sigma[p * f + q] = (sigma[p * f + q] ?? 0) * (1 - delta);
        }
      }
    } else if (typeof this.shrinkage === "number" && this.shrinkage > 0) {
      const s = this.shrinkage;
      let trace = 0;
      for (let j = 0; j < f; j++) trace += sigma[j * f + j] ?? 0;
      const mu = trace / f;
      for (let i = 0; i < f * f; i++) sigma[i] = (sigma[i] ?? 0) * (1 - s);
      for (let j = 0; j < f; j++) sigma[j * f + j] = (sigma[j * f + j] ?? 0) + s * mu;
    }

    // Whitening of the pooled covariance: Sigma = V diag(lambda) V^T
    const eig = jacobiEigenSymmetric(sigma, f);
    const lambdaMax = eig.values[0] ?? 0;
    if (!(lambdaMax > 0)) {
      throw new DataValidationError(
        "Within-class covariance is zero: every sample equals its class mean"
      );
    }
    let rank = 0;
    while (rank < f && (eig.values[rank] ?? 0) > lambdaMax * 1e-10) rank++;
    if (rank < f) {
      warn(
        `Variables are collinear: the within-class covariance has rank ${rank} < n_features=${f}.`,
        "UserWarning",
        "LinearDiscriminantAnalysis"
      );
    }
    const whitener = new Float64Array(f * rank);
    for (let j = 0; j < f; j++) {
      for (let r = 0; r < rank; r++) {
        whitener[j * rank + r] = (eig.vectors[j * f + r] ?? 0) / Math.sqrt(eig.values[r] ?? 1);
      }
    }

    // Class means in whitened coordinates
    const projMeans = new Float64Array(K * rank);
    for (let c = 0; c < K; c++) {
      for (let r = 0; r < rank; r++) {
        let s = 0;
        for (let j = 0; j < f; j++) s += (means[c * f + j] ?? 0) * (whitener[j * rank + r] ?? 0);
        projMeans[c * rank + r] = s;
      }
    }

    // Between-class scatter in whitened coordinates: sum_k n * pi_k / (K - 1) b_k b_k^T
    const projXbar = new Float64Array(rank);
    for (let r = 0; r < rank; r++) {
      let s = 0;
      for (let j = 0; j < f; j++) s += (xbar[j] ?? 0) * (whitener[j * rank + r] ?? 0);
      projXbar[r] = s;
    }
    const between = new Float64Array(rank * rank);
    for (let c = 0; c < K; c++) {
      const w = (nSamples * (priors[c] ?? 0)) / (K - 1);
      for (let p = 0; p < rank; p++) {
        const bp = (projMeans[c * rank + p] ?? 0) - (projXbar[p] ?? 0);
        for (let q = 0; q < rank; q++) {
          const bq = (projMeans[c * rank + q] ?? 0) - (projXbar[q] ?? 0);
          between[p * rank + q] = (between[p * rank + q] ?? 0) + w * bp * bq;
        }
      }
    }
    const betweenEig = jacobiEigenSymmetric(between, rank);

    // Discriminant directions: scalings = whitener @ U[:, :nComp]
    const scalings = new Float64Array(f * nComp);
    for (let k = 0; k < Math.min(nComp, rank); k++) {
      for (let j = 0; j < f; j++) {
        let s = 0;
        for (let r = 0; r < rank; r++) {
          s += (whitener[j * rank + r] ?? 0) * (betweenEig.vectors[r * rank + k] ?? 0);
        }
        scalings[j * nComp + k] = s;
      }
      // Fix the sign: the entry of largest magnitude is positive.
      let best = 0;
      let bestAbs = -1;
      for (let j = 0; j < f; j++) {
        const a = Math.abs(scalings[j * nComp + k] ?? 0);
        if (a > bestAbs) {
          bestAbs = a;
          best = j;
        }
      }
      if ((scalings[best * nComp + k] ?? 0) < 0) {
        for (let j = 0; j < f; j++) scalings[j * nComp + k] = -(scalings[j * nComp + k] ?? 0);
      }
    }

    // Explained variance ratio: eigenvalue share among all discriminant directions
    let eigSum = 0;
    for (let r = 0; r < rank; r++) eigSum += Math.max(betweenEig.values[r] ?? 0, 0);
    const ratio = new Float64Array(nComp);
    if (eigSum > 0) {
      for (let k = 0; k < Math.min(nComp, rank); k++) {
        ratio[k] = Math.max(betweenEig.values[k] ?? 0, 0) / eigSum;
      }
    }

    this.classes_ = classes;
    this.means_ = means;
    this.xbar_ = xbar;
    this.priors_ = priors;
    this.whitener_ = whitener;
    this.projMeans_ = projMeans;
    this.rank_ = rank;
    this.scalings_ = scalings;
    this.nFeaturesIn_ = f;
    this.nComponentsActual_ = nComp;
    this.explainedVarianceRatio_ = ratio;
    this.outDType_ = floatDTypeOf(X);
    this.fitted = true;
    return this;
  }

  /**
   * Predict class labels.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted labels of shape (n_samples,): int32 when all classes are integers, float64 otherwise
   * @throws {NotFittedError} If the model has not been fitted
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("LinearDiscriminantAnalysis must be fitted before prediction");
    }
    const classes = this.classes_ ?? [];
    const scores = this.computeLogPosteriors(X);
    const best = argmaxRows(scores, X.shape[0] ?? 0, classes.length);
    return labelTensor(Array.from(best, (idx) => classes[idx] ?? 0));
  }

  /**
   * Posterior class probabilities.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Probabilities of shape (n_samples, n_classes); columns follow the sorted class labels
   * @throws {NotFittedError} If the model has not been fitted
   */
  predictProba(X: Tensor): Tensor {
    const logProba = this.predictLogProbaArray(X);
    for (let i = 0; i < logProba.length; i++) logProba[i] = Math.exp(logProba[i] ?? 0);
    return makeTensor(logProba, [X.shape[0] ?? 0, this.classes_?.length ?? 0], floatDTypeOf(X));
  }

  /**
   * Log of the posterior class probabilities.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Log probabilities of shape (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   */
  predictLogProba(X: Tensor): Tensor {
    const logProba = this.predictLogProbaArray(X);
    return makeTensor(logProba, [X.shape[0] ?? 0, this.classes_?.length ?? 0], floatDTypeOf(X));
  }

  private predictLogProbaArray(X: Tensor): Float64Array {
    if (!this.fitted) {
      throw new NotFittedError("LinearDiscriminantAnalysis must be fitted before prediction");
    }
    const K = this.classes_?.length ?? 0;
    return logSoftmaxRows(this.computeLogPosteriors(X), X.shape[0] ?? 0, K);
  }

  /**
   * Project data to the most discriminant directions: `(X - xbar) @ scalings`.
   * The projected within-class covariance is the identity.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Projected data of shape (n_samples, n_components)
   * @throws {NotFittedError} If the model has not been fitted
   */
  transform(X: Tensor): Tensor {
    if (!this.fitted || !this.scalings_ || !this.xbar_) {
      throw new NotFittedError("LinearDiscriminantAnalysis must be fitted before transform");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "LinearDiscriminantAnalysis");

    const nSamples = X.shape[0] ?? 0;
    const f = this.nFeaturesIn_;
    const nComp = this.nComponentsActual_;
    const data = toFloat64View(X);
    const out = new Float64Array(nSamples * nComp);
    for (let i = 0; i < nSamples; i++) {
      for (let k = 0; k < nComp; k++) {
        let sum = 0;
        for (let j = 0; j < f; j++) {
          sum +=
            ((data[i * f + j] ?? 0) - (this.xbar_[j] ?? 0)) * (this.scalings_[j * nComp + k] ?? 0);
        }
        out[i * nComp + k] = sum;
      }
    }
    return makeTensor(out, [nSamples, nComp], floatDTypeOf(X));
  }

  /**
   * Fit the model and project the training data.
   *
   * @param X - Training data
   * @param y - Class labels (required)
   * @returns Projected data of shape (n_samples, n_components)
   * @throws {InvalidParameterError} If y is missing
   */
  fitTransform(X: Tensor, y?: Tensor): Tensor {
    if (!y) {
      throw new InvalidParameterError("LDA.fitTransform requires y (supervised)", "y", undefined);
    }
    this.fit(X, y);
    return this.transform(X);
  }

  /**
   * Mean accuracy on the given data and labels.
   *
   * @param X - Test samples
   * @param y - True labels of shape (n_samples,)
   * @returns Fraction of correctly classified samples
   */
  score(X: Tensor, y: Tensor): number {
    return accuracy(this.predict(X), y);
  }

  /** Class labels seen during fit (sorted), or `undefined` before fitting. */
  get classes(): Tensor | undefined {
    if (!this.fitted || !this.classes_) return undefined;
    return labelTensor(this.classes_);
  }

  /**
   * Share of the between-class variance carried by each kept discriminant direction,
   * relative to all directions (so the values may sum to less than 1 when `nComponents`
   * is smaller than min(n_classes - 1, n_features)). `undefined` before fitting.
   */
  get explainedVarianceRatio(): Float64Array | undefined {
    return this.explainedVarianceRatio_
      ? Float64Array.from(this.explainedVarianceRatio_)
      : undefined;
  }

  /**
   * Per-class feature means, shape (n_classes, n_features).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get classMeans(): Tensor {
    if (!this.fitted || !this.means_) {
      throw new NotFittedError("LinearDiscriminantAnalysis must be fitted to access classMeans");
    }
    return makeTensor(
      Float64Array.from(this.means_),
      [this.classes_?.length ?? 0, this.nFeaturesIn_],
      this.outDType_
    );
  }

  /**
   * Class prior probabilities, shape (n_classes,).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get classPriors(): Tensor {
    if (!this.fitted || !this.priors_) {
      throw new NotFittedError("LinearDiscriminantAnalysis must be fitted to access classPriors");
    }
    return makeTensor(Float64Array.from(this.priors_), [this.priors_.length], this.outDType_);
  }

  /**
   * Projection matrix used by `transform`, shape (n_features, n_components).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get scalings(): Tensor {
    if (!this.fitted || !this.scalings_) {
      throw new NotFittedError("LinearDiscriminantAnalysis must be fitted to access scalings");
    }
    return makeTensor(
      Float64Array.from(this.scalings_),
      [this.nFeaturesIn_, this.nComponentsActual_],
      this.outDType_
    );
  }

  /**
   * Number of features seen during fit.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nFeaturesIn(): number {
    if (!this.fitted) {
      throw new NotFittedError("LinearDiscriminantAnalysis must be fitted to access nFeaturesIn");
    }
    return this.nFeaturesIn_;
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      nComponents: this.nComponents,
      shrinkage: this.shrinkage,
      priors: this.priors,
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
        case "nComponents":
          this.nComponents =
            value === undefined ? undefined : LinearDiscriminantAnalysis.checkNComponents(value);
          break;
        case "shrinkage":
          this.shrinkage =
            value === undefined ? undefined : LinearDiscriminantAnalysis.checkShrinkage(value);
          break;
        case "priors":
          if (value !== undefined) checkPriorValues(value as readonly number[]);
          this.priors = value as readonly number[] | undefined;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  // ---- Private helpers ----

  /** Unnormalized log posteriors log(pi_k) - 0.5 * mahalanobis_k, shape (n, K) flat. */
  private computeLogPosteriors(X: Tensor): Float64Array {
    if (!this.whitener_ || !this.projMeans_ || !this.priors_ || !this.classes_) {
      throw new NotFittedError("LinearDiscriminantAnalysis must be fitted");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "LinearDiscriminantAnalysis");

    const nSamples = X.shape[0] ?? 0;
    const f = this.nFeaturesIn_;
    const K = this.classes_.length;
    const rank = this.rank_;
    const data = toFloat64View(X);
    const z = new Float64Array(rank);
    const out = new Float64Array(nSamples * K);

    for (let i = 0; i < nSamples; i++) {
      for (let r = 0; r < rank; r++) {
        let s = 0;
        for (let j = 0; j < f; j++)
          s += (data[i * f + j] ?? 0) * (this.whitener_[j * rank + r] ?? 0);
        z[r] = s;
      }
      for (let c = 0; c < K; c++) {
        let mahal = 0;
        for (let r = 0; r < rank; r++) {
          const d = (z[r] ?? 0) - (this.projMeans_[c * rank + r] ?? 0);
          mahal += d * d;
        }
        out[i * K + c] = Math.log(this.priors_[c] ?? 0) - 0.5 * mahal;
      }
    }
    return out;
  }
}

/**
 * Quadratic Discriminant Analysis (QDA).
 *
 * A classifier with a quadratic decision boundary, generated by fitting class
 * conditional densities to the data and using Bayes' rule. Unlike LDA, QDA fits
 * a separate covariance matrix per class, allowing for more flexible (quadratic)
 * decision boundaries.
 *
 * Each class needs at least 2 samples (or `regParam > 0`). Class covariances are
 * eigen-decomposed; eigenvalues below 1e-12 of the largest one are raised to that floor so that
 * collinear features give finite, consistent scores (a warning is issued).
 *
 * @example
 * ```ts
 * import { QuadraticDiscriminantAnalysis } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [2, 1], [3, 3], [7, 8], [8, 7], [9, 9]]);
 * const y = tensor([0, 0, 0, 1, 1, 1]);
 *
 * const qda = new QuadraticDiscriminantAnalysis();
 * qda.fit(X, y);
 * const predictions = qda.predict(tensor([[2, 3], [6, 7]]));
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-advanced | Deepbox Discriminant Analysis}
 */
export class QuadraticDiscriminantAnalysis implements Classifier {
  private regParam: number;
  private priors: readonly number[] | undefined;

  private classes_?: number[];
  private means_?: Float64Array; // (K, f)
  private rotations_?: Float64Array[]; // per class (f, f): V diag(1 / sqrt(lambda))
  private logDets_?: Float64Array;
  private priors_?: Float64Array;
  private nFeaturesIn_ = 0;
  private outDType_: FloatDType = "float64";
  private fitted = false;

  /**
   * @param options.regParam - Value added to the diagonal of each class covariance (default: 0). This
   *   is an additive ridge and is not bounded. scikit-learn's `reg_param` instead blends
   *   `(1 - r) * Sigma + r * I` with `r` in [0, 1], so the two are not interchangeable.
   * @param options.priors - Prior probabilities of classes, ordered by sorted class label
   */
  constructor(
    options: {
      readonly regParam?: number;
      readonly priors?: readonly number[];
    } = {}
  ) {
    this.regParam = QuadraticDiscriminantAnalysis.checkRegParam(options.regParam ?? 0);
    if (options.priors !== undefined) {
      checkPriorValues(options.priors);
      this.priors = options.priors;
    }
  }

  private static checkRegParam(value: unknown): number {
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
      throw new InvalidParameterError("regParam must be a finite number >= 0", "regParam", value);
    }
    return value;
  }

  /**
   * Fit the model.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Class labels of shape (n_samples,)
   * @returns this
   * @throws {DataValidationError} If there are fewer than 2 classes, or a class has a single
   *   sample while `regParam` is 0
   * @throws {InvalidParameterError} If priors are inconsistent with the data
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const f = X.shape[1] ?? 0;
    const data = toFloat64View(X);

    const { classes, index } = encodeLabels(y);
    const K = classes.length;
    if (K < 2) {
      throw new DataValidationError("QDA requires at least 2 classes");
    }

    const counts = new Float64Array(K);
    for (let i = 0; i < nSamples; i++) {
      const c = index[i] ?? 0;
      counts[c] = (counts[c] ?? 0) + 1;
    }
    if (this.regParam === 0) {
      for (let c = 0; c < K; c++) {
        if ((counts[c] ?? 0) < 2) {
          throw new DataValidationError(
            `Class ${classes[c]} has only 1 sample; QDA needs at least 2 samples per class to estimate a covariance (or set regParam > 0)`
          );
        }
      }
    }
    const priors = resolvePriors(this.priors, counts, nSamples);

    const means = new Float64Array(K * f);
    for (let i = 0; i < nSamples; i++) {
      const c = index[i] ?? 0;
      for (let j = 0; j < f; j++) {
        means[c * f + j] = (means[c * f + j] ?? 0) + (data[i * f + j] ?? 0);
      }
    }
    for (let c = 0; c < K; c++) {
      for (let j = 0; j < f; j++) means[c * f + j] = (means[c * f + j] ?? 0) / (counts[c] ?? 1);
    }

    // Per-class covariance (n_c - 1 denominator) and its eigen-decomposition
    const covs: Float64Array[] = Array.from({ length: K }, () => new Float64Array(f * f));
    for (let i = 0; i < nSamples; i++) {
      const c = index[i] ?? 0;
      const cov = covs[c];
      if (!cov) continue;
      for (let p = 0; p < f; p++) {
        const dp = (data[i * f + p] ?? 0) - (means[c * f + p] ?? 0);
        if (dp === 0) continue;
        for (let q = p; q < f; q++) {
          cov[p * f + q] =
            (cov[p * f + q] ?? 0) + dp * ((data[i * f + q] ?? 0) - (means[c * f + q] ?? 0));
        }
      }
    }

    const rotations: Float64Array[] = [];
    const logDets = new Float64Array(K);
    let collinear = false;
    for (let c = 0; c < K; c++) {
      const cov = covs[c] ?? new Float64Array(f * f);
      const denom = Math.max((counts[c] ?? 1) - 1, 1);
      for (let p = 0; p < f; p++) {
        for (let q = p; q < f; q++) {
          const v = (cov[p * f + q] ?? 0) / denom;
          cov[p * f + q] = v;
          cov[q * f + p] = v;
        }
        cov[p * f + p] = (cov[p * f + p] ?? 0) + this.regParam;
      }
      const eig = jacobiEigenSymmetric(cov, f);
      const top = eig.values[0] ?? 0;
      const floor = top > 0 ? top * 1e-12 : 1e-12;
      const rotation = new Float64Array(f * f);
      let logDet = 0;
      for (let r = 0; r < f; r++) {
        const lam = eig.values[r] ?? 0;
        if (lam < floor) collinear = true;
        const clipped = Math.max(lam, floor);
        logDet += Math.log(clipped);
        const inv = 1 / Math.sqrt(clipped);
        for (let j = 0; j < f; j++) rotation[j * f + r] = (eig.vectors[j * f + r] ?? 0) * inv;
      }
      rotations.push(rotation);
      logDets[c] = logDet;
    }
    if (collinear) {
      warn(
        "Variables are collinear: a class covariance is singular, so its small eigenvalues were raised to a floor. Consider setting regParam > 0.",
        "UserWarning",
        "QuadraticDiscriminantAnalysis"
      );
    }

    this.classes_ = classes;
    this.means_ = means;
    this.rotations_ = rotations;
    this.logDets_ = logDets;
    this.priors_ = priors;
    this.nFeaturesIn_ = f;
    this.outDType_ = floatDTypeOf(X);
    this.fitted = true;
    return this;
  }

  /**
   * Predict class labels.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted labels of shape (n_samples,): int32 when all classes are integers, float64 otherwise
   * @throws {NotFittedError} If the model has not been fitted
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("QuadraticDiscriminantAnalysis must be fitted before prediction");
    }
    const classes = this.classes_ ?? [];
    const scores = this.computeLogPosteriors(X);
    const best = argmaxRows(scores, X.shape[0] ?? 0, classes.length);
    return labelTensor(Array.from(best, (idx) => classes[idx] ?? 0));
  }

  /**
   * Posterior class probabilities.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Probabilities of shape (n_samples, n_classes); columns follow the sorted class labels
   * @throws {NotFittedError} If the model has not been fitted
   */
  predictProba(X: Tensor): Tensor {
    const logProba = this.predictLogProbaArray(X);
    for (let i = 0; i < logProba.length; i++) logProba[i] = Math.exp(logProba[i] ?? 0);
    return makeTensor(logProba, [X.shape[0] ?? 0, this.classes_?.length ?? 0], floatDTypeOf(X));
  }

  /**
   * Log of the posterior class probabilities.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Log probabilities of shape (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   */
  predictLogProba(X: Tensor): Tensor {
    const logProba = this.predictLogProbaArray(X);
    return makeTensor(logProba, [X.shape[0] ?? 0, this.classes_?.length ?? 0], floatDTypeOf(X));
  }

  private predictLogProbaArray(X: Tensor): Float64Array {
    if (!this.fitted) {
      throw new NotFittedError("QuadraticDiscriminantAnalysis must be fitted before prediction");
    }
    const K = this.classes_?.length ?? 0;
    return logSoftmaxRows(this.computeLogPosteriors(X), X.shape[0] ?? 0, K);
  }

  /**
   * Mean accuracy on the given data and labels.
   *
   * @param X - Test samples
   * @param y - True labels of shape (n_samples,)
   * @returns Fraction of correctly classified samples
   */
  score(X: Tensor, y: Tensor): number {
    return accuracy(this.predict(X), y);
  }

  /** Class labels seen during fit (sorted), or `undefined` before fitting. */
  get classes(): Tensor | undefined {
    if (!this.fitted || !this.classes_) return undefined;
    return labelTensor(this.classes_);
  }

  /**
   * Per-class feature means, shape (n_classes, n_features).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get classMeans(): Tensor {
    if (!this.fitted || !this.means_) {
      throw new NotFittedError("QuadraticDiscriminantAnalysis must be fitted to access classMeans");
    }
    return makeTensor(
      Float64Array.from(this.means_),
      [this.classes_?.length ?? 0, this.nFeaturesIn_],
      this.outDType_
    );
  }

  /**
   * Class prior probabilities, shape (n_classes,).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get classPriors(): Tensor {
    if (!this.fitted || !this.priors_) {
      throw new NotFittedError(
        "QuadraticDiscriminantAnalysis must be fitted to access classPriors"
      );
    }
    return makeTensor(Float64Array.from(this.priors_), [this.priors_.length], this.outDType_);
  }

  /**
   * Number of features seen during fit.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nFeaturesIn(): number {
    if (!this.fitted) {
      throw new NotFittedError(
        "QuadraticDiscriminantAnalysis must be fitted to access nFeaturesIn"
      );
    }
    return this.nFeaturesIn_;
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      regParam: this.regParam,
      priors: this.priors,
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
        case "regParam":
          this.regParam = QuadraticDiscriminantAnalysis.checkRegParam(value);
          break;
        case "priors":
          if (value !== undefined) checkPriorValues(value as readonly number[]);
          this.priors = value as readonly number[] | undefined;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  // ---- Private helpers ----

  /** Unnormalized log posteriors log(pi_k) - 0.5 * (log|S_k| + mahalanobis_k), flat (n, K). */
  private computeLogPosteriors(X: Tensor): Float64Array {
    if (!this.rotations_ || !this.means_ || !this.priors_ || !this.logDets_ || !this.classes_) {
      throw new NotFittedError("QuadraticDiscriminantAnalysis must be fitted");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "QuadraticDiscriminantAnalysis");

    const nSamples = X.shape[0] ?? 0;
    const f = this.nFeaturesIn_;
    const K = this.classes_.length;
    const data = toFloat64View(X);
    const diff = new Float64Array(f);
    const out = new Float64Array(nSamples * K);

    for (let i = 0; i < nSamples; i++) {
      for (let c = 0; c < K; c++) {
        const rotation = this.rotations_[c];
        for (let j = 0; j < f; j++)
          diff[j] = (data[i * f + j] ?? 0) - (this.means_[c * f + j] ?? 0);
        let mahal = 0;
        for (let r = 0; r < f; r++) {
          let s = 0;
          for (let j = 0; j < f; j++) s += (diff[j] ?? 0) * (rotation?.[j * f + r] ?? 0);
          mahal += s * s;
        }
        out[i * K + c] = Math.log(this.priors_[c] ?? 0) - 0.5 * ((this.logDets_[c] ?? 0) + mahal);
      }
    }
    return out;
  }
}
