/**
 * Matrix decomposition and dimensionality reduction estimators.
 *
 * Provides `PCA`, `TruncatedSVD`, `NMF`, `FastICA` and `LatentDirichletAllocation`.
 * All estimators follow the usual `fit` / `transform` / `fitTransform` contract and
 * keep their fitted state in internal arrays; the public getters return fresh tensors.
 *
 * @see {@link https://deepbox.dev/docs/ml-decomposition | Deepbox Dimensionality Reduction}
 */

import {
  ConvergenceError,
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
  warn,
} from "../../core";
import { svd } from "../../linalg";
import { type Tensor, tensor } from "../../ndarray";
import { Generator } from "../../random/Generator";
import { __random } from "../../random/random";
import { digamma, jacobiEigenSymmetric } from "../_internal";
import {
  assertContiguous,
  toFloat64View,
  validatePredictInputs,
  validateUnsupervisedFitInputs,
} from "../_validation";
import type { Transformer } from "../base";

export { digamma, jacobiEigenSymmetric } from "../_internal";

// ---------------------------------------------------------------------------
// Shared helpers (module private unless marked @internal)
// ---------------------------------------------------------------------------

type FloatDType = "float32" | "float64";

/** Floating point dtype used for the outputs of an estimator fitted on `X`. */
function floatDTypeOf(X: Tensor): FloatDType {
  return X.dtype === "float32" ? "float32" : "float64";
}

/**
 * Wrap a flat row-major buffer in a tensor of the given shape. The tensor takes
 * ownership of `data` when the dtype is float64, so callers pass a fresh array.
 */
function makeTensor(data: Float64Array, shape: number[], dtype: FloatDType): Tensor {
  const flat = dtype === "float64" ? tensor(data) : tensor(Float32Array.from(data));
  return flat.reshape(shape);
}

/**
 * Create a random generator. A given `randomState` always yields the same stream;
 * without one the generator is seeded from the global random stream, so `setSeed`
 * still makes results reproducible.
 */
function createGenerator(randomState: number | undefined): Generator {
  return new Generator(randomState ?? Math.floor(__random() * 4294967296));
}

function validateRandomStateValue(value: unknown): void {
  if (value !== undefined && (typeof value !== "number" || !Number.isFinite(value))) {
    throw new InvalidParameterError("randomState must be a finite number", "randomState", value);
  }
}

function validateTolValue(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new InvalidParameterError("tol must be a finite number >= 0", "tol", value);
  }
  return value;
}

function validatePositiveInt(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new InvalidParameterError(`${name} must be an integer >= 1`, name, value);
  }
  return value;
}

/** Column means of a row-major (n x f) matrix with a second pass to cancel rounding error. */
function columnMeans(data: Float64Array, n: number, f: number): Float64Array {
  const mean = new Float64Array(f);
  for (let i = 0; i < n; i++) {
    const base = i * f;
    for (let j = 0; j < f; j++) mean[j] = (mean[j] ?? 0) + (data[base + j] ?? 0);
  }
  for (let j = 0; j < f; j++) mean[j] = (mean[j] ?? 0) / n;
  const corr = new Float64Array(f);
  for (let i = 0; i < n; i++) {
    const base = i * f;
    for (let j = 0; j < f; j++) corr[j] = (corr[j] ?? 0) + ((data[base + j] ?? 0) - (mean[j] ?? 0));
  }
  for (let j = 0; j < f; j++) mean[j] = (mean[j] ?? 0) + (corr[j] ?? 0) / n;
  return mean;
}

/**
 * Flip the sign of each row of `Vt` so that its entry of largest magnitude is
 * positive. This is scikit-learn's `svd_flip(..., u_based_decision=False)` and
 * makes component signs independent of the SVD routine.
 */
function flipRowSigns(Vt: Float64Array, rows: number, cols: number): void {
  for (let i = 0; i < rows; i++) {
    let best = 0;
    let bestAbs = -1;
    for (let j = 0; j < cols; j++) {
      const a = Math.abs(Vt[i * cols + j] ?? 0);
      if (a > bestAbs) {
        bestAbs = a;
        best = j;
      }
    }
    if ((Vt[i * cols + best] ?? 0) < 0) {
      for (let j = 0; j < cols; j++) Vt[i * cols + j] = -(Vt[i * cols + j] ?? 0);
    }
  }
}

/** Singular value decomposition of a row-major (m x n) matrix, returning `s` and the rows of `Vt`. */
function economySvd(
  data: Float64Array,
  m: number,
  n: number
): { s: Float64Array; Vt: Float64Array; rank: number } {
  const [, S, Vt] = svd(tensor(Float64Array.from(data)).reshape([m, n]), false);
  const s = Float64Array.from(toFloat64View(S));
  return { s, Vt: Float64Array.from(toFloat64View(Vt)), rank: s.length };
}

/** C = A (m x k) @ B (k x n) for row-major buffers. */
function matmul(A: Float64Array, m: number, k: number, B: Float64Array, n: number): Float64Array {
  const C = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    const cBase = i * n;
    for (let l = 0; l < k; l++) {
      const a = A[i * k + l] ?? 0;
      if (a === 0) continue;
      const bBase = l * n;
      for (let j = 0; j < n; j++) C[cBase + j] = (C[cBase + j] ?? 0) + a * (B[bBase + j] ?? 0);
    }
  }
  return C;
}

/** C = A^T (n x m) @ B (m x p) where A is (m x n); both row-major. */
function matmulAtB(
  A: Float64Array,
  m: number,
  n: number,
  B: Float64Array,
  p: number
): Float64Array {
  const C = new Float64Array(n * p);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      const a = A[i * n + j] ?? 0;
      if (a === 0) continue;
      const cBase = j * p;
      const bBase = i * p;
      for (let l = 0; l < p; l++) C[cBase + l] = (C[cBase + l] ?? 0) + a * (B[bBase + l] ?? 0);
    }
  }
  return C;
}

/**
 * Orthonormalize the columns of the row-major (rows x cols) matrix `M` in place using
 * modified Gram-Schmidt with one re-orthogonalization pass. Columns that are numerically
 * dependent on the previous ones are set to zero.
 */
function orthonormalizeColumns(M: Float64Array, rows: number, cols: number): void {
  for (let j = 0; j < cols; j++) {
    let initial = 0;
    for (let i = 0; i < rows; i++) initial += (M[i * cols + j] ?? 0) ** 2;
    initial = Math.sqrt(initial);
    for (let pass = 0; pass < 2; pass++) {
      for (let prev = 0; prev < j; prev++) {
        let dot = 0;
        for (let i = 0; i < rows; i++) dot += (M[i * cols + prev] ?? 0) * (M[i * cols + j] ?? 0);
        if (dot === 0) continue;
        for (let i = 0; i < rows; i++) {
          M[i * cols + j] = (M[i * cols + j] ?? 0) - dot * (M[i * cols + prev] ?? 0);
        }
      }
    }
    let norm = 0;
    for (let i = 0; i < rows; i++) norm += (M[i * cols + j] ?? 0) ** 2;
    norm = Math.sqrt(norm);
    if (norm > initial * 1e-12 && norm > 0) {
      for (let i = 0; i < rows; i++) M[i * cols + j] = (M[i * cols + j] ?? 0) / norm;
    } else {
      for (let i = 0; i < rows; i++) M[i * cols + j] = 0;
    }
  }
}

function assertAllNonNegative(X: Tensor, message: string): void {
  const data = toFloat64View(X);
  for (let i = 0; i < data.length; i++) {
    if ((data[i] ?? 0) < 0) throw new DataValidationError(message);
  }
}

function assertFinite(values: Float64Array, message: string): void {
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) throw new DataValidationError(message);
  }
}

/**
 * Validate a (n_samples, nColumns) input for `inverseTransform` and return it as a flat view.
 */
function readLatentInput(X: Tensor, nColumns: number, what: string): Float64Array {
  if (X.ndim !== 2) {
    throw new ShapeError(`X must be 2-dimensional; got ndim=${X.ndim}`);
  }
  assertContiguous(X, "X");
  const data = toFloat64View(X);
  const cols = X.shape[1] ?? 0;
  if (cols !== nColumns) {
    throw new ShapeError(`X must have ${nColumns} ${what}; got ${cols}`);
  }
  assertFinite(data, "X contains non-finite values (NaN or Inf)");
  return data;
}

// ---------------------------------------------------------------------------
// PCA
// ---------------------------------------------------------------------------

/**
 * Principal Component Analysis (PCA).
 *
 * Linear dimensionality reduction using Singular Value Decomposition (SVD)
 * to project data to a lower dimensional space.
 *
 * **Algorithm**:
 * 1. Center the data by subtracting the column means
 * 2. Compute SVD: X = U * Σ * V^T (exact, or randomized with power iterations)
 * 3. Principal components are the rows of V^T, signed so that the entry of
 *    largest magnitude in each component is positive (same convention as scikit-learn)
 * 4. Transform data by projecting onto the principal components
 *
 * **Time Complexity**: O(min(n*d^2, d*n^2)) for the exact solver, where n=samples, d=features
 *
 * @example
 * ```ts
 * import { PCA } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0]]);
 * const pca = new PCA({ nComponents: 1 });
 * pca.fit(X);
 *
 * const XTransformed = pca.transform(X);
 * console.log('Explained variance ratio:', pca.explainedVarianceRatio);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-decomposition | Deepbox Dimensionality Reduction}
 */
export class PCA implements Transformer {
  private nComponents: number | undefined;
  private whiten: boolean;
  private svdSolver: "auto" | "full" | "randomized";
  private nOversamples: number;
  private randomState: number | undefined;

  private components_?: Float64Array; // (k, f)
  private explainedVariance_?: Float64Array;
  private explainedVarianceRatio_?: Float64Array;
  private singularValues_?: Float64Array;
  private mean_?: Float64Array;
  private noiseVariance_ = 0;
  private nComponentsFit_ = 0;
  private nFeaturesIn_ = 0;
  private outDType_: FloatDType = "float64";
  private fitted = false;

  /**
   * Create a new PCA model.
   *
   * @param options - Configuration options
   * @param options.nComponents - Number of components to keep (default: min(n_samples, n_features)).
   *   A number in (0, 1) keeps the smallest number of components whose cumulative explained
   *   variance ratio exceeds that fraction (exact solver only).
   * @param options.whiten - Scale the projected components to unit variance (default: false)
   * @param options.svdSolver - SVD solver: 'auto', 'full', or 'randomized' (default: 'auto')
   * @param options.nOversamples - Additional random vectors for randomized SVD (default: 10)
   * @param options.randomState - Random seed for randomized SVD
   */
  constructor(
    options: {
      readonly nComponents?: number;
      readonly whiten?: boolean;
      readonly svdSolver?: "auto" | "full" | "randomized";
      readonly nOversamples?: number;
      readonly randomState?: number;
    } = {}
  ) {
    if (options.nComponents !== undefined) {
      this.nComponents = PCA.checkNComponents(options.nComponents);
    }
    this.whiten = options.whiten ?? false;
    this.svdSolver = options.svdSolver ?? "auto";
    this.nOversamples = options.nOversamples ?? 10;
    if (options.randomState !== undefined) {
      validateRandomStateValue(options.randomState);
      this.randomState = options.randomState;
    }

    if (typeof this.whiten !== "boolean") {
      throw new InvalidParameterError("whiten must be a boolean", "whiten", this.whiten);
    }
    if (this.svdSolver !== "auto" && this.svdSolver !== "full" && this.svdSolver !== "randomized") {
      throw new InvalidParameterError(
        "svdSolver must be 'auto', 'full', or 'randomized'",
        "svdSolver",
        this.svdSolver
      );
    }
    if (!Number.isInteger(this.nOversamples) || this.nOversamples < 0) {
      throw new InvalidParameterError(
        "nOversamples must be an integer >= 0",
        "nOversamples",
        this.nOversamples
      );
    }
  }

  private static checkNComponents(value: unknown): number {
    if (
      typeof value !== "number" ||
      !((Number.isInteger(value) && value >= 1) || (value > 0 && value < 1))
    ) {
      throw new InvalidParameterError(
        "nComponents must be an integer >= 1 or a variance fraction in (0, 1)",
        "nComponents",
        value
      );
    }
    return value;
  }

  /**
   * Randomized SVD using the Halko-Martinsson-Tropp algorithm with power iterations.
   * Computes an approximate truncated SVD of a row-major (m x n) matrix.
   *
   * @returns The top `k` singular values and the matching rows of V^T
   */
  private randomizedSvd(
    X: Float64Array,
    m: number,
    n: number,
    k: number
  ): { s: Float64Array; Vt: Float64Array } {
    const maxRank = Math.min(m, n);
    const p = Math.min(k + this.nOversamples, maxRank);
    const generator = createGenerator(this.randomState);

    // Range finder: Y = X @ Omega with Omega ~ N(0, 1) of shape (n, p).
    const omega = generator.normalArray(0, 1, n * p);
    let Y = matmul(X, m, n, omega, p);

    // Power iterations sharpen the spectrum; re-orthonormalize every half step.
    const nIter = k < 0.1 * maxRank ? 7 : 4;
    for (let it = 0; it < nIter; it++) {
      orthonormalizeColumns(Y, m, p);
      const Z = matmulAtB(X, m, n, Y, p); // (n, p)
      orthonormalizeColumns(Z, n, p);
      Y = matmul(X, m, n, Z, p);
    }
    orthonormalizeColumns(Y, m, p); // Q, shape (m, p)

    // B = Q^T X has shape (p, n); its SVD gives the singular values and V^T of X.
    const B = matmulAtB(Y, m, p, X, n);
    const { s, Vt, rank } = economySvd(B, p, n);
    const kk = Math.min(k, rank);
    return { s: s.slice(0, kk), Vt: Vt.slice(0, kk * n) };
  }

  /**
   * Fit PCA on training data.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Ignored (exists for compatibility)
   * @returns this
   * @throws {DataValidationError} If X has fewer than 2 samples or contains non-finite values
   * @throws {InvalidParameterError} If nComponents exceeds min(n_samples, n_features)
   */
  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    if (nSamples < 2) {
      throw new DataValidationError("X must have at least 2 samples for PCA");
    }

    const maxComponents = Math.min(nSamples, nFeatures);
    const isFraction = this.nComponents !== undefined && this.nComponents < 1;
    if (isFraction && this.svdSolver === "randomized") {
      throw new InvalidParameterError(
        "A fractional nComponents requires svdSolver 'full' or 'auto'",
        "svdSolver",
        this.svdSolver
      );
    }
    let nComp = isFraction ? maxComponents : (this.nComponents ?? maxComponents);
    if (nComp > maxComponents) {
      throw new InvalidParameterError(
        `nComponents=${nComp} must be <= min(n_samples, n_features)=${maxComponents}`,
        "nComponents",
        nComp
      );
    }

    // Center the data (copy: the input tensor is never modified).
    const raw = toFloat64View(X);
    const mean = columnMeans(raw, nSamples, nFeatures);
    const Xc = new Float64Array(nSamples * nFeatures);
    for (let i = 0; i < nSamples; i++) {
      const base = i * nFeatures;
      for (let j = 0; j < nFeatures; j++) Xc[base + j] = (raw[base + j] ?? 0) - (mean[j] ?? 0);
    }

    // Total variance is the trace of the covariance, taken from the centered data (the
    // randomized solver only returns k singular values, so they cannot give the total).
    let sumSq = 0;
    for (let i = 0; i < Xc.length; i++) sumSq += (Xc[i] ?? 0) ** 2;
    const totalVariance = sumSq / (nSamples - 1);

    let solver = this.svdSolver;
    if (solver === "auto") {
      solver =
        !isFraction && Math.max(nSamples, nFeatures) > 500 && nComp < 0.8 * maxComponents
          ? "randomized"
          : "full";
    }

    let s: Float64Array;
    let Vt: Float64Array;
    if (solver === "randomized") {
      ({ s, Vt } = this.randomizedSvd(Xc, nSamples, nFeatures, nComp));
    } else {
      ({ s, Vt } = economySvd(Xc, nSamples, nFeatures));
    }
    flipRowSigns(Vt, s.length, nFeatures);

    if (isFraction) {
      // Smallest k whose cumulative explained variance ratio exceeds the requested fraction.
      const fraction = this.nComponents as number;
      let cumulative = 0;
      nComp = s.length;
      for (let i = 0; i < s.length; i++) {
        cumulative += totalVariance > 0 ? (s[i] ?? 0) ** 2 / (nSamples - 1) / totalVariance : 0;
        if (cumulative > fraction) {
          nComp = i + 1;
          break;
        }
      }
    }

    const explainedVariance = new Float64Array(nComp);
    const explainedVarianceRatio = new Float64Array(nComp);
    const singularValues = new Float64Array(nComp);
    let explainedSum = 0;
    for (let i = 0; i < nComp; i++) {
      const sv = s[i] ?? 0;
      singularValues[i] = sv;
      const ev = (sv * sv) / (nSamples - 1);
      explainedVariance[i] = ev;
      explainedVarianceRatio[i] = totalVariance > 0 ? ev / totalVariance : 0;
      explainedSum += ev;
    }
    const noise =
      nComp < maxComponents
        ? Math.max(0, totalVariance - explainedSum) / (maxComponents - nComp)
        : 0;

    this.components_ = Vt.slice(0, nComp * nFeatures);
    this.explainedVariance_ = explainedVariance;
    this.explainedVarianceRatio_ = explainedVarianceRatio;
    this.singularValues_ = singularValues;
    this.mean_ = mean;
    this.noiseVariance_ = noise;
    this.nComponentsFit_ = nComp;
    this.nFeaturesIn_ = nFeatures;
    this.outDType_ = floatDTypeOf(X);
    this.fitted = true;
    return this;
  }

  /**
   * Whitening scale of component `c`: the standard deviation of its projection.
   * Components whose variance is numerically zero are mapped to 0 instead of being amplified.
   */
  private whitenFactors(): { scale: Float64Array; inverse: Float64Array } {
    const ev = this.explainedVariance_ ?? new Float64Array(0);
    const k = ev.length;
    const scale = new Float64Array(k);
    const inverse = new Float64Array(k);
    const top = ev[0] ?? 0;
    for (let c = 0; c < k; c++) {
      const v = ev[c] ?? 0;
      scale[c] = Math.sqrt(v);
      inverse[c] = v > top * 1e-20 && v > 0 ? 1 / Math.sqrt(v) : 0;
    }
    return { scale, inverse };
  }

  /**
   * Transform data to principal component space.
   *
   * @param X - Data of shape (n_samples, n_features)
   * @returns Transformed data of shape (n_samples, n_components)
   * @throws {NotFittedError} If the model has not been fitted
   */
  transform(X: Tensor): Tensor {
    if (!this.fitted || !this.components_ || !this.mean_) {
      throw new NotFittedError("PCA must be fitted before transform");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "PCA");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = this.nFeaturesIn_;
    const k = this.nComponentsFit_;
    const data = toFloat64View(X);
    const comps = this.components_;
    const mean = this.mean_;
    const inverse = this.whiten ? this.whitenFactors().inverse : undefined;

    const out = new Float64Array(nSamples * k);
    for (let i = 0; i < nSamples; i++) {
      const base = i * nFeatures;
      for (let c = 0; c < k; c++) {
        const cBase = c * nFeatures;
        let sum = 0;
        for (let j = 0; j < nFeatures; j++) {
          sum += ((data[base + j] ?? 0) - (mean[j] ?? 0)) * (comps[cBase + j] ?? 0);
        }
        out[i * k + c] = inverse ? sum * (inverse[c] ?? 0) : sum;
      }
    }
    return makeTensor(out, [nSamples, k], floatDTypeOf(X));
  }

  /**
   * Fit and transform in one step.
   *
   * @param X - Training data
   * @param y - Ignored (exists for compatibility)
   * @returns Transformed data
   */
  fitTransform(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    return this.transform(X);
  }

  /**
   * Transform data back to original space.
   *
   * @param X - Transformed data of shape (n_samples, n_components)
   * @returns Reconstructed data of shape (n_samples, n_features)
   * @throws {NotFittedError} If the model has not been fitted
   */
  inverseTransform(X: Tensor): Tensor {
    if (!this.fitted || !this.components_ || !this.mean_) {
      throw new NotFittedError("PCA must be fitted before inverse transform");
    }
    const k = this.nComponentsFit_;
    const nFeatures = this.nFeaturesIn_;
    const data = readLatentInput(X, k, "components");
    const nSamples = X.shape[0] ?? 0;
    const comps = this.components_;
    const mean = this.mean_;
    // Undo whitening by restoring the original component scale.
    const scale = this.whiten ? this.whitenFactors().scale : undefined;

    const out = new Float64Array(nSamples * nFeatures);
    for (let i = 0; i < nSamples; i++) {
      for (let c = 0; c < k; c++) {
        const xv = (data[i * k + c] ?? 0) * (scale ? (scale[c] ?? 0) : 1);
        if (xv === 0) continue;
        const cBase = c * nFeatures;
        const oBase = i * nFeatures;
        for (let j = 0; j < nFeatures; j++) {
          out[oBase + j] = (out[oBase + j] ?? 0) + xv * (comps[cBase + j] ?? 0);
        }
      }
      for (let j = 0; j < nFeatures; j++) {
        out[i * nFeatures + j] = (out[i * nFeatures + j] ?? 0) + (mean[j] ?? 0);
      }
    }
    return makeTensor(out, [nSamples, nFeatures], floatDTypeOf(X));
  }

  /**
   * Principal axes in feature space, shape (n_components, n_features).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get components(): Tensor {
    if (!this.fitted || !this.components_) {
      throw new NotFittedError("PCA must be fitted to access components");
    }
    return makeTensor(
      Float64Array.from(this.components_),
      [this.nComponentsFit_, this.nFeaturesIn_],
      this.outDType_
    );
  }

  /**
   * Variance explained by each component, shape (n_components,).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get explainedVariance(): Tensor {
    if (!this.fitted || !this.explainedVariance_) {
      throw new NotFittedError("PCA must be fitted to access explained variance");
    }
    return makeTensor(
      Float64Array.from(this.explainedVariance_),
      [this.nComponentsFit_],
      this.outDType_
    );
  }

  /**
   * Fraction of the total variance explained by each component, shape (n_components,).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get explainedVarianceRatio(): Tensor {
    if (!this.fitted || !this.explainedVarianceRatio_) {
      throw new NotFittedError("PCA must be fitted to access explained variance ratio");
    }
    return makeTensor(
      Float64Array.from(this.explainedVarianceRatio_),
      [this.nComponentsFit_],
      this.outDType_
    );
  }

  /**
   * Singular values of the centered training data for the kept components, shape (n_components,).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get singularValues(): Tensor {
    if (!this.fitted || !this.singularValues_) {
      throw new NotFittedError("PCA must be fitted to access singular values");
    }
    return makeTensor(
      Float64Array.from(this.singularValues_),
      [this.nComponentsFit_],
      this.outDType_
    );
  }

  /**
   * Per-feature mean of the training data, shape (n_features,).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get mean(): Tensor {
    if (!this.fitted || !this.mean_) {
      throw new NotFittedError("PCA must be fitted to access the mean");
    }
    return makeTensor(Float64Array.from(this.mean_), [this.nFeaturesIn_], this.outDType_);
  }

  /**
   * Average variance of the discarded components (0 when all components are kept).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get noiseVariance(): number {
    if (!this.fitted) {
      throw new NotFittedError("PCA must be fitted to access noise variance");
    }
    return this.noiseVariance_;
  }

  /**
   * Number of features seen during fit.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nFeaturesIn(): number {
    if (!this.fitted) {
      throw new NotFittedError("PCA must be fitted to access nFeaturesIn");
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
      whiten: this.whiten,
      svdSolver: this.svdSolver,
      nOversamples: this.nOversamples,
      randomState: this.randomState,
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
          this.nComponents = value === undefined ? undefined : PCA.checkNComponents(value);
          break;
        case "whiten":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("whiten must be a boolean", "whiten", value);
          }
          this.whiten = value;
          break;
        case "svdSolver":
          if (value !== "auto" && value !== "full" && value !== "randomized") {
            throw new InvalidParameterError(
              "svdSolver must be 'auto', 'full', or 'randomized'",
              "svdSolver",
              value
            );
          }
          this.svdSolver = value;
          break;
        case "nOversamples":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
            throw new InvalidParameterError(
              "nOversamples must be an integer >= 0",
              "nOversamples",
              value
            );
          }
          this.nOversamples = value;
          break;
        case "randomState":
          validateRandomStateValue(value);
          this.randomState = value as number | undefined;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}

// ---------------------------------------------------------------------------
// TruncatedSVD
// ---------------------------------------------------------------------------

/**
 * Truncated SVD (aka LSA: Latent Semantic Analysis).
 *
 * Unlike PCA, TruncatedSVD does **not** center the data before computing SVD.
 * This makes it suitable for sparse data (e.g., TF-IDF matrices from text),
 * where centering would destroy sparsity.
 *
 * **Algorithm**:
 * 1. Compute SVD of X directly: X ≈ U * Σ * V^T (truncated to nComponents)
 * 2. Components are rows of V^T (signed so the largest-magnitude entry is positive)
 * 3. Transform: X_new = X @ V = U * Σ
 *
 * `explainedVariance` is the variance of each transformed column (population variance,
 * ddof = 0) and `explainedVarianceRatio` divides it by the total variance of the columns
 * of X, as in scikit-learn.
 *
 * @example
 * ```ts
 * import { TruncatedSVD } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 0, 2, 0], [0, 3, 0, 1], [4, 0, 1, 0], [0, 2, 0, 5]]);
 * const tsvd = new TruncatedSVD({ nComponents: 2 });
 * const XReduced = tsvd.fitTransform(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-decomposition | Deepbox Dimensionality Reduction}
 */
export class TruncatedSVD implements Transformer {
  private nComponents: number;

  private components_?: Float64Array; // (k, f)
  private explainedVariance_?: Float64Array;
  private explainedVarianceRatio_?: Float64Array;
  private singularValues_?: Float64Array;
  private nComponentsFit_ = 0;
  private nFeaturesIn_ = 0;
  private outDType_: FloatDType = "float64";
  private fitted = false;

  /**
   * @param options.nComponents - Number of singular vectors to keep (default: 2)
   */
  constructor(
    options: {
      readonly nComponents?: number;
    } = {}
  ) {
    this.nComponents = validatePositiveInt(options.nComponents ?? 2, "nComponents");
  }

  /**
   * Fit the model on X.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Ignored (exists for compatibility)
   * @returns this
   * @throws {InvalidParameterError} If nComponents exceeds min(n_samples, n_features)
   */
  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const k = this.nComponents;
    const maxComponents = Math.min(nSamples, nFeatures);
    if (k > maxComponents) {
      throw new InvalidParameterError(
        `nComponents=${k} must be <= min(n_samples, n_features)=${maxComponents}`,
        "nComponents",
        k
      );
    }

    // SVD without centering
    const data = toFloat64View(X);
    const { s, Vt } = economySvd(data, nSamples, nFeatures);
    flipRowSigns(Vt, s.length, nFeatures);
    const components = Vt.slice(0, k * nFeatures);
    const singularValues = s.slice(0, k);

    // Variance of the transformed columns (ddof = 0) and of the original columns.
    const projected = new Float64Array(nSamples * k);
    for (let i = 0; i < nSamples; i++) {
      for (let c = 0; c < k; c++) {
        let sum = 0;
        for (let j = 0; j < nFeatures; j++) {
          sum += (data[i * nFeatures + j] ?? 0) * (components[c * nFeatures + j] ?? 0);
        }
        projected[i * k + c] = sum;
      }
    }
    const explainedVariance = new Float64Array(k);
    const projMean = columnMeans(projected, nSamples, k);
    for (let i = 0; i < nSamples; i++) {
      for (let c = 0; c < k; c++) {
        const d = (projected[i * k + c] ?? 0) - (projMean[c] ?? 0);
        explainedVariance[c] = (explainedVariance[c] ?? 0) + d * d;
      }
    }
    for (let c = 0; c < k; c++) explainedVariance[c] = (explainedVariance[c] ?? 0) / nSamples;

    const colMean = columnMeans(data, nSamples, nFeatures);
    let totalVar = 0;
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        const d = (data[i * nFeatures + j] ?? 0) - (colMean[j] ?? 0);
        totalVar += d * d;
      }
    }
    totalVar /= nSamples;
    const ratio = new Float64Array(k);
    if (totalVar > 0) {
      for (let c = 0; c < k; c++) ratio[c] = (explainedVariance[c] ?? 0) / totalVar;
    }

    this.components_ = components;
    this.singularValues_ = singularValues;
    this.explainedVariance_ = explainedVariance;
    this.explainedVarianceRatio_ = ratio;
    this.nComponentsFit_ = k;
    this.nFeaturesIn_ = nFeatures;
    this.outDType_ = floatDTypeOf(X);
    this.fitted = true;
    return this;
  }

  /**
   * Project X onto the fitted right singular vectors.
   *
   * @param X - Data of shape (n_samples, n_features)
   * @returns Reduced data of shape (n_samples, n_components)
   * @throws {NotFittedError} If the model has not been fitted
   */
  transform(X: Tensor): Tensor {
    if (!this.fitted || !this.components_) {
      throw new NotFittedError("TruncatedSVD must be fitted before transform");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "TruncatedSVD");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = this.nFeaturesIn_;
    const k = this.nComponentsFit_;
    const data = toFloat64View(X);
    const comps = this.components_;

    // X_new = X @ V (components^T)
    const out = new Float64Array(nSamples * k);
    for (let i = 0; i < nSamples; i++) {
      for (let c = 0; c < k; c++) {
        let sum = 0;
        for (let j = 0; j < nFeatures; j++) {
          sum += (data[i * nFeatures + j] ?? 0) * (comps[c * nFeatures + j] ?? 0);
        }
        out[i * k + c] = sum;
      }
    }
    return makeTensor(out, [nSamples, k], floatDTypeOf(X));
  }

  /**
   * Fit the model and return the reduced data.
   *
   * @param X - Training data
   * @param y - Ignored (exists for compatibility)
   * @returns Reduced data of shape (n_samples, n_components)
   */
  fitTransform(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    return this.transform(X);
  }

  /**
   * Map reduced data back to the original feature space.
   *
   * @param X - Reduced data of shape (n_samples, n_components)
   * @returns Approximate reconstruction of shape (n_samples, n_features)
   * @throws {NotFittedError} If the model has not been fitted
   */
  inverseTransform(X: Tensor): Tensor {
    if (!this.fitted || !this.components_) {
      throw new NotFittedError("TruncatedSVD must be fitted before inverse transform");
    }
    const k = this.nComponentsFit_;
    const nFeatures = this.nFeaturesIn_;
    const data = readLatentInput(X, k, "components");
    const nSamples = X.shape[0] ?? 0;

    // X_reconstructed = X_reduced @ components
    const out = matmul(data, nSamples, k, this.components_, nFeatures);
    return makeTensor(out, [nSamples, nFeatures], floatDTypeOf(X));
  }

  /**
   * Right singular vectors, shape (n_components, n_features).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get components(): Tensor {
    if (!this.fitted || !this.components_) {
      throw new NotFittedError("TruncatedSVD must be fitted to access components");
    }
    return makeTensor(
      Float64Array.from(this.components_),
      [this.nComponentsFit_, this.nFeaturesIn_],
      this.outDType_
    );
  }

  /**
   * Variance of each transformed column (ddof = 0), shape (n_components,).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get explainedVariance(): Tensor {
    if (!this.fitted || !this.explainedVariance_) {
      throw new NotFittedError("TruncatedSVD must be fitted to access explained variance");
    }
    return makeTensor(
      Float64Array.from(this.explainedVariance_),
      [this.nComponentsFit_],
      this.outDType_
    );
  }

  /**
   * Fraction of the total column variance of X explained by each component.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get explainedVarianceRatio(): Tensor {
    if (!this.fitted || !this.explainedVarianceRatio_) {
      throw new NotFittedError("TruncatedSVD must be fitted to access explained variance ratio");
    }
    return makeTensor(
      Float64Array.from(this.explainedVarianceRatio_),
      [this.nComponentsFit_],
      this.outDType_
    );
  }

  /**
   * Singular values of X for the kept components, shape (n_components,).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get singularValues(): Tensor {
    if (!this.fitted || !this.singularValues_) {
      throw new NotFittedError("TruncatedSVD must be fitted to access singular values");
    }
    return makeTensor(
      Float64Array.from(this.singularValues_),
      [this.nComponentsFit_],
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
      throw new NotFittedError("TruncatedSVD must be fitted to access nFeaturesIn");
    }
    return this.nFeaturesIn_;
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return { nComponents: this.nComponents };
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
          this.nComponents = validatePositiveInt(value, "nComponents");
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}

// ---------------------------------------------------------------------------
// NMF
// ---------------------------------------------------------------------------

type NmfInit = "random" | "nndsvd" | "nndsvda";

function validateNmfInit(value: unknown): NmfInit {
  if (value !== "random" && value !== "nndsvd" && value !== "nndsvda") {
    throw new InvalidParameterError("init must be 'random', 'nndsvd', or 'nndsvda'", "init", value);
  }
  return value;
}

/**
 * Non-negative Matrix Factorization (NMF).
 *
 * Factorizes a non-negative matrix X ≈ W * H where:
 * - W is the transformed data (n_samples × n_components)
 * - H is the components matrix (n_components × n_features)
 * - Both W and H are non-negative
 *
 * Uses multiplicative update rules (Lee & Seung, 2001) that minimize the squared
 * Frobenius norm. With `init: "random"` the factors start from non-negative random
 * values scaled to the data (sqrt(mean(X) / nComponents)); `"nndsvd"` and `"nndsvda"`
 * use a deterministic SVD-based start (Boutsidis & Gallopoulos, 2008). Plain `"nndsvd"`
 * produces exact zeros, which multiplicative updates never change, so `"nndsvda"` is
 * usually the better choice.
 *
 * Useful for topic modeling, recommendation systems, and signal separation.
 *
 * @example
 * ```ts
 * import { NMF } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 0, 2], [0, 3, 1], [2, 1, 0], [1, 2, 3]]);
 * const nmf = new NMF({ nComponents: 2, randomState: 0 });
 * const W = nmf.fitTransform(X);  // X must be non-negative
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-decomposition | Deepbox Dimensionality Reduction}
 */
export class NMF implements Transformer {
  private nComponents: number;
  private maxIter: number;
  private tol: number;
  private randomState: number | undefined;
  private init: NmfInit;

  private H_?: Float64Array; // (nComponentsFit x nFeatures) row-major
  private nComponentsFit_ = 0;
  private nFeaturesIn_ = 0;
  private nIter_ = 0;
  private reconstructionErr_ = 0;
  private outDType_: FloatDType = "float64";
  private fitted = false;

  /**
   * @param options.nComponents - Number of components (default: 2)
   * @param options.maxIter - Maximum number of update iterations (default: 200)
   * @param options.tol - Stop when the relative change of the squared error is below this (default: 1e-4)
   * @param options.randomState - Seed for the random initialization
   * @param options.init - Initialization: "random" (default), "nndsvd" or "nndsvda"
   */
  constructor(
    options: {
      readonly nComponents?: number;
      readonly maxIter?: number;
      readonly tol?: number;
      readonly randomState?: number;
      readonly init?: NmfInit;
    } = {}
  ) {
    this.nComponents = validatePositiveInt(options.nComponents ?? 2, "nComponents");
    this.maxIter = validatePositiveInt(options.maxIter ?? 200, "maxIter");
    this.tol = validateTolValue(options.tol ?? 1e-4);
    this.init = validateNmfInit(options.init ?? "random");
    if (options.randomState !== undefined) {
      validateRandomStateValue(options.randomState);
      this.randomState = options.randomState;
    }
  }

  /** Squared Frobenius norm of X - W @ H. */
  private static squaredError(
    X: Float64Array,
    W: Float64Array,
    H: Float64Array,
    n: number,
    f: number,
    k: number
  ): number {
    let cost = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < f; j++) {
        let wh = 0;
        for (let c = 0; c < k; c++) wh += (W[i * k + c] ?? 0) * (H[c * f + j] ?? 0);
        const diff = (X[i * f + j] ?? 0) - wh;
        cost += diff * diff;
      }
    }
    return cost;
  }

  /** Non-negative double SVD initialization (NNDSVD). */
  private nndsvdInit(
    X: Float64Array,
    n: number,
    f: number,
    k: number,
    fillZeros: boolean,
    mean: number
  ): { W: Float64Array; H: Float64Array } {
    const [U, S, Vt] = svd(tensor(Float64Array.from(X)).reshape([n, f]), false);
    const u = toFloat64View(U);
    const s = toFloat64View(S);
    const vt = toFloat64View(Vt);
    const r = s.length;
    const W = new Float64Array(n * k);
    const H = new Float64Array(k * f);

    const s0 = Math.sqrt(s[0] ?? 0);
    for (let i = 0; i < n; i++) W[i * k] = s0 * Math.abs(u[i * r] ?? 0);
    for (let j = 0; j < f; j++) H[j] = s0 * Math.abs(vt[j] ?? 0);

    for (let c = 1; c < k; c++) {
      let xpn = 0;
      let xnn = 0;
      let ypn = 0;
      let ynn = 0;
      for (let i = 0; i < n; i++) {
        const x = u[i * r + c] ?? 0;
        if (x > 0) xpn += x * x;
        else xnn += x * x;
      }
      for (let j = 0; j < f; j++) {
        const y = vt[c * f + j] ?? 0;
        if (y > 0) ypn += y * y;
        else ynn += y * y;
      }
      xpn = Math.sqrt(xpn);
      xnn = Math.sqrt(xnn);
      ypn = Math.sqrt(ypn);
      ynn = Math.sqrt(ynn);
      const mp = xpn * ypn;
      const mn = xnn * ynn;
      const usePositive = mp > mn;
      const xNorm = usePositive ? xpn : xnn;
      const yNorm = usePositive ? ypn : ynn;
      const sigma = usePositive ? mp : mn;
      if (xNorm === 0 || yNorm === 0) continue;
      const lbd = Math.sqrt((s[c] ?? 0) * sigma);
      for (let i = 0; i < n; i++) {
        const x = u[i * r + c] ?? 0;
        const part = usePositive ? Math.max(x, 0) : Math.max(-x, 0);
        W[i * k + c] = (lbd * part) / xNorm;
      }
      for (let j = 0; j < f; j++) {
        const y = vt[c * f + j] ?? 0;
        const part = usePositive ? Math.max(y, 0) : Math.max(-y, 0);
        H[c * f + j] = (lbd * part) / yNorm;
      }
    }

    for (let i = 0; i < W.length; i++) {
      if ((W[i] ?? 0) < 1e-6) W[i] = fillZeros ? mean : 0;
    }
    for (let i = 0; i < H.length; i++) {
      if ((H[i] ?? 0) < 1e-6) H[i] = fillZeros ? mean : 0;
    }
    return { W, H };
  }

  /** Fit the factorization and return W. Fitted state is only replaced on success. */
  private fitInternal(X: Tensor): Float64Array {
    validateUnsupervisedFitInputs(X);
    assertAllNonNegative(X, "NMF requires all values in X to be non-negative");

    const n = X.shape[0] ?? 0;
    const f = X.shape[1] ?? 0;
    const k = this.nComponents;
    const Xd = toFloat64View(X);

    let total = 0;
    for (let i = 0; i < Xd.length; i++) total += Xd[i] ?? 0;
    const mean = total / Xd.length;

    let W: Float64Array;
    let H: Float64Array;
    if (this.init === "random") {
      const rng = createGenerator(this.randomState);
      const scale = Math.sqrt(mean / k);
      W = rng.randomArray(n * k);
      H = rng.randomArray(k * f);
      for (let i = 0; i < W.length; i++) W[i] = (W[i] ?? 0) * scale + 1e-12;
      for (let i = 0; i < H.length; i++) H[i] = (H[i] ?? 0) * scale + 1e-12;
    } else {
      if (k > Math.min(n, f)) {
        throw new InvalidParameterError(
          `init='${this.init}' requires nComponents <= min(n_samples, n_features)=${Math.min(n, f)}; got ${k}`,
          "init",
          this.init
        );
      }
      ({ W, H } = this.nndsvdInit(Xd, n, f, k, this.init === "nndsvda", mean));
    }

    const eps = 1e-12;
    let prevCost = Number.POSITIVE_INFINITY;
    let cost = 0;
    let iterations = 0;
    let converged = false;

    for (let iter = 0; iter < this.maxIter; iter++) {
      // Update W: W *= (X @ H^T) / (W @ (H @ H^T))
      const HHt = new Float64Array(k * k);
      for (let a = 0; a < k; a++) {
        for (let b = a; b < k; b++) {
          let sum = 0;
          for (let j = 0; j < f; j++) sum += (H[a * f + j] ?? 0) * (H[b * f + j] ?? 0);
          HHt[a * k + b] = sum;
          HHt[b * k + a] = sum;
        }
      }
      // The denominator of every entry of a row uses the row's values from before the update.
      const wRow = new Float64Array(k);
      for (let i = 0; i < n; i++) {
        for (let c = 0; c < k; c++) {
          let num = 0;
          for (let j = 0; j < f; j++) num += (Xd[i * f + j] ?? 0) * (H[c * f + j] ?? 0);
          let den = 0;
          for (let b = 0; b < k; b++) den += (W[i * k + b] ?? 0) * (HHt[b * k + c] ?? 0);
          wRow[c] = (W[i * k + c] ?? 0) * (num / (den + eps));
        }
        for (let c = 0; c < k; c++) W[i * k + c] = wRow[c] ?? 0;
      }

      // Update H: H *= (W^T @ X) / ((W^T @ W) @ H)
      const WtX = matmulAtB(W, n, k, Xd, f);
      const WtW = matmulAtB(W, n, k, W, k);
      const WtWH = matmul(WtW, k, k, H, f);
      for (let i = 0; i < k * f; i++) {
        H[i] = (H[i] ?? 0) * ((WtX[i] ?? 0) / ((WtWH[i] ?? 0) + eps));
      }

      cost = NMF.squaredError(Xd, W, H, n, f, k);
      iterations = iter + 1;
      if (cost === 0 || Math.abs(prevCost - cost) / (prevCost + eps) < this.tol) {
        converged = true;
        break;
      }
      prevCost = cost;
    }

    if (!converged) {
      warn(
        `NMF did not converge within maxIter=${this.maxIter} iterations; increase maxIter or tol.`,
        "ConvergenceWarning",
        "NMF"
      );
    }

    this.H_ = H;
    this.nComponentsFit_ = k;
    this.nFeaturesIn_ = f;
    this.nIter_ = iterations;
    this.reconstructionErr_ = Math.sqrt(cost);
    this.outDType_ = floatDTypeOf(X);
    this.fitted = true;
    return W;
  }

  /**
   * Fit the model to a non-negative matrix.
   *
   * @param X - Non-negative data of shape (n_samples, n_features)
   * @param y - Ignored (exists for compatibility)
   * @returns this
   * @throws {DataValidationError} If X contains negative or non-finite values
   */
  fit(X: Tensor, _y?: Tensor): this {
    this.fitInternal(X);
    return this;
  }

  /**
   * Compute W for new data with H held fixed.
   *
   * W starts from the constant sqrt(mean(X) / nComponents) and is refined with
   * multiplicative updates, so the result is deterministic.
   *
   * @param X - Non-negative data of shape (n_samples, n_features)
   * @returns W of shape (n_samples, n_components)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {DataValidationError} If X contains negative or non-finite values
   */
  transform(X: Tensor): Tensor {
    if (!this.fitted || !this.H_) {
      throw new NotFittedError("NMF must be fitted before transform");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "NMF");
    assertAllNonNegative(X, "NMF requires all values in X to be non-negative");

    const n = X.shape[0] ?? 0;
    const f = this.nFeaturesIn_;
    const k = this.nComponentsFit_;
    const H = this.H_;
    const Xd = toFloat64View(X);

    let total = 0;
    for (let i = 0; i < Xd.length; i++) total += Xd[i] ?? 0;
    const start = Xd.length > 0 ? Math.sqrt(total / Xd.length / k) : 0;
    const W = new Float64Array(n * k).fill(start);

    const eps = 1e-12;
    const HHt = new Float64Array(k * k);
    for (let a = 0; a < k; a++) {
      for (let b = a; b < k; b++) {
        let sum = 0;
        for (let j = 0; j < f; j++) sum += (H[a * f + j] ?? 0) * (H[b * f + j] ?? 0);
        HHt[a * k + b] = sum;
        HHt[b * k + a] = sum;
      }
    }
    // Numerator X @ H^T does not depend on W.
    const XHt = new Float64Array(n * k);
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < k; c++) {
        let num = 0;
        for (let j = 0; j < f; j++) num += (Xd[i * f + j] ?? 0) * (H[c * f + j] ?? 0);
        XHt[i * k + c] = num;
      }
    }

    const stopTol = Math.min(this.tol, 1e-6);
    const maxIter = Math.max(this.maxIter, 100);
    const row = new Float64Array(k);
    for (let iter = 0; iter < maxIter; iter++) {
      let change = 0;
      let magnitude = 0;
      for (let i = 0; i < n; i++) {
        for (let c = 0; c < k; c++) {
          let den = 0;
          for (let b = 0; b < k; b++) den += (W[i * k + b] ?? 0) * (HHt[b * k + c] ?? 0);
          row[c] = (W[i * k + c] ?? 0) * ((XHt[i * k + c] ?? 0) / (den + eps));
        }
        for (let c = 0; c < k; c++) {
          const nv = row[c] ?? 0;
          change += Math.abs(nv - (W[i * k + c] ?? 0));
          magnitude += nv;
          W[i * k + c] = nv;
        }
      }
      if (change <= stopTol * (magnitude + eps)) break;
    }

    return makeTensor(W, [n, k], floatDTypeOf(X));
  }

  /**
   * Fit the model and return W from the fit.
   *
   * @param X - Non-negative data of shape (n_samples, n_features)
   * @param y - Ignored (exists for compatibility)
   * @returns W of shape (n_samples, n_components)
   */
  fitTransform(X: Tensor, _y?: Tensor): Tensor {
    const W = this.fitInternal(X);
    return makeTensor(W, [X.shape[0] ?? 0, this.nComponentsFit_], floatDTypeOf(X));
  }

  /**
   * Reconstruct data as W @ H.
   *
   * @param X - W matrix of shape (n_samples, n_components)
   * @returns Reconstructed data of shape (n_samples, n_features)
   * @throws {NotFittedError} If the model has not been fitted
   */
  inverseTransform(X: Tensor): Tensor {
    if (!this.fitted || !this.H_) {
      throw new NotFittedError("NMF must be fitted before inverse transform");
    }
    const k = this.nComponentsFit_;
    const data = readLatentInput(X, k, "components");
    const n = X.shape[0] ?? 0;
    const out = matmul(data, n, k, this.H_, this.nFeaturesIn_);
    return makeTensor(out, [n, this.nFeaturesIn_], floatDTypeOf(X));
  }

  /**
   * Factorization matrix H, shape (n_components, n_features).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get components(): Tensor {
    if (!this.fitted || !this.H_) {
      throw new NotFittedError("NMF must be fitted to access components");
    }
    return makeTensor(
      Float64Array.from(this.H_),
      [this.nComponentsFit_, this.nFeaturesIn_],
      this.outDType_
    );
  }

  /** Number of iterations run by the last fit. */
  get nIter(): number {
    return this.nIter_;
  }

  /**
   * Frobenius norm of X - W @ H for the training data of the last fit.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get reconstructionErr(): number {
    if (!this.fitted) {
      throw new NotFittedError("NMF must be fitted to access reconstruction error");
    }
    return this.reconstructionErr_;
  }

  /**
   * Number of features seen during fit.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nFeaturesIn(): number {
    if (!this.fitted) {
      throw new NotFittedError("NMF must be fitted to access nFeaturesIn");
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
      maxIter: this.maxIter,
      tol: this.tol,
      randomState: this.randomState,
      init: this.init,
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
          this.nComponents = validatePositiveInt(value, "nComponents");
          break;
        case "maxIter":
          this.maxIter = validatePositiveInt(value, "maxIter");
          break;
        case "tol":
          this.tol = validateTolValue(value);
          break;
        case "randomState":
          validateRandomStateValue(value);
          this.randomState = value as number | undefined;
          break;
        case "init":
          this.init = validateNmfInit(value);
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}

// ---------------------------------------------------------------------------
// FastICA
// ---------------------------------------------------------------------------

type IcaFun = "logcosh" | "exp" | "cube";

function validateIcaFun(value: unknown): IcaFun {
  if (value !== "logcosh" && value !== "exp" && value !== "cube") {
    throw new InvalidParameterError(`fun must be "logcosh", "exp", or "cube"`, "fun", value);
  }
  return value;
}

/**
 * FastICA: Independent Component Analysis using the fast fixed-point algorithm.
 *
 * Separates a multivariate signal into additive, independent non-Gaussian
 * components. Uses negentropy maximization with the logcosh or exp
 * contrast functions (or the kurtosis-based cube function), symmetric
 * decorrelation (parallel algorithm) and, by default, whitening to unit variance.
 *
 * After fitting, `components` holds the unmixing matrix W·K of shape
 * (n_components, n_features) so that `transform(X) = (X - mean) @ componentsᵀ`, and
 * `mixingMatrix` is its pseudo-inverse of shape (n_features, n_components).
 * Independent components are only defined up to sign, scale and order.
 *
 * @example
 * ```ts
 * import { FastICA } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 5], [3, 1], [5, 4], [7, 3], [9, 8], [2, 6]]);
 * const ica = new FastICA({ nComponents: 2, randomState: 0 });
 * const S = ica.fitTransform(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-decomposition | Deepbox Dimensionality Reduction}
 */
export class FastICA implements Transformer {
  private nComponents: number;
  private maxIter: number;
  private tol: number;
  private fun: IcaFun;
  private whiten: boolean;
  private randomState: number | undefined;

  private mean_?: Float64Array;
  private components_?: Float64Array; // unmixing matrix in feature space (k x f)
  private mixing_?: Float64Array; // pseudo-inverse of components_ (f x k)
  private nFeaturesIn_ = 0;
  private nIter_ = 0;
  // Actual component count after clamping to min(nComponents, nFeatures, nSamples).
  private kActual_ = 0;
  private outDType_: FloatDType = "float64";
  private fitted = false;

  /**
   * @param options.nComponents - Number of components to extract (default: 2). Clamped to
   *   min(n_samples, n_features) with a warning when larger.
   * @param options.maxIter - Maximum number of fixed-point iterations (default: 200)
   * @param options.tol - Convergence tolerance on 1 - |<w_new, w_old>| (default: 1e-4)
   * @param options.fun - Contrast function: "logcosh" (default), "exp" or "cube"
   * @param options.whiten - Whiten the data to unit variance before iterating (default: true)
   * @param options.randomState - Seed for the random initial unmixing matrix
   */
  constructor(
    options: {
      readonly nComponents?: number;
      readonly maxIter?: number;
      readonly tol?: number;
      readonly fun?: IcaFun;
      readonly whiten?: boolean;
      readonly randomState?: number;
    } = {}
  ) {
    this.nComponents = validatePositiveInt(options.nComponents ?? 2, "nComponents");
    this.maxIter = validatePositiveInt(options.maxIter ?? 200, "maxIter");
    this.tol = validateTolValue(options.tol ?? 1e-4);
    this.fun = validateIcaFun(options.fun ?? "logcosh");
    this.whiten = options.whiten ?? true;
    if (typeof this.whiten !== "boolean") {
      throw new InvalidParameterError("whiten must be a boolean", "whiten", this.whiten);
    }
    if (options.randomState !== undefined) {
      validateRandomStateValue(options.randomState);
      this.randomState = options.randomState;
    }
  }

  /**
   * Fit the model: estimate the unmixing matrix.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Ignored (exists for compatibility)
   * @returns this
   * @throws {DataValidationError} If X has fewer than 2 samples or is rank deficient
   *   for the requested number of components
   * @throws {ConvergenceError} If the iteration diverges to non-finite values
   */
  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    if (nSamples < 2) {
      throw new DataValidationError("X must have at least 2 samples for FastICA");
    }
    const maxK = Math.min(nSamples, nFeatures);
    const k = Math.min(this.nComponents, maxK);
    if (this.nComponents > maxK) {
      warn(
        `nComponents=${this.nComponents} exceeds min(n_samples, n_features)=${maxK}; using ${k} components.`,
        "UserWarning",
        "FastICA"
      );
    }

    // Center the data
    const raw = toFloat64View(X);
    const mean = columnMeans(raw, nSamples, nFeatures);
    const data = new Float64Array(nSamples * nFeatures);
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        data[i * nFeatures + j] = (raw[i * nFeatures + j] ?? 0) - (mean[j] ?? 0);
      }
    }

    // Whiten using the SVD of the centered data. K = sqrt(n) * diag(1/S[:k]) @ Vt[:k]
    // gives Z = X K^T unit variance and identity covariance (scikit-learn's convention).
    // Without the sqrt(n) factor Z would have variance 1/n, w^T z would stay in the linear
    // regime of the nonlinearity and the fixed point would never move.
    let Z: Float64Array;
    let dim: number;
    let whiteningVt: Float64Array | undefined; // rows of Vt kept (k x f)
    let whiteningS: Float64Array | undefined;
    let whitening: Float64Array | undefined; // K (k x f)
    const sqrtN = Math.sqrt(nSamples);
    if (this.whiten) {
      const { s, Vt } = economySvd(data, nSamples, nFeatures);
      const sTop = s[0] ?? 0;
      const minS = sTop * Math.max(nSamples, nFeatures) * Number.EPSILON;
      if (!((s[k - 1] ?? 0) > minS)) {
        throw new DataValidationError(
          `After centering, X has rank below nComponents=${k}; reduce nComponents or remove collinear features`
        );
      }
      whiteningVt = Vt.slice(0, k * nFeatures);
      whiteningS = s.slice(0, k);
      whitening = new Float64Array(k * nFeatures);
      for (let c = 0; c < k; c++) {
        const factor = sqrtN / (s[c] ?? 1);
        for (let j = 0; j < nFeatures; j++) {
          whitening[c * nFeatures + j] = (Vt[c * nFeatures + j] ?? 0) * factor;
        }
      }
      dim = k;
      Z = new Float64Array(nSamples * k);
      for (let i = 0; i < nSamples; i++) {
        for (let c = 0; c < k; c++) {
          let sum = 0;
          for (let j = 0; j < nFeatures; j++) {
            sum += (data[i * nFeatures + j] ?? 0) * (whitening[c * nFeatures + j] ?? 0);
          }
          Z[i * k + c] = sum;
        }
      }
    } else {
      dim = nFeatures;
      Z = data;
    }

    // Random initial unmixing matrix W (k x dim), orthogonalized.
    const generator = createGenerator(this.randomState);
    const W = generator.normalArray(0, 1, k * dim);
    FastICA.symmetricDecorrelation(W, k, dim);

    const gpMean = new Float64Array(k);
    const wx = new Float64Array(k);
    let converged = false;
    let iterations = 0;
    for (let iter = 0; iter < this.maxIter; iter++) {
      // Wnew = E[z g(Wz)^T]^T - diag(E[g'(Wz)]) W
      const Wnew = new Float64Array(k * dim);
      gpMean.fill(0);
      for (let i = 0; i < nSamples; i++) {
        const zBase = i * dim;
        for (let c = 0; c < k; c++) {
          let sum = 0;
          for (let d = 0; d < dim; d++) sum += (W[c * dim + d] ?? 0) * (Z[zBase + d] ?? 0);
          wx[c] = sum;
        }
        for (let c = 0; c < k; c++) {
          const u = wx[c] ?? 0;
          let g: number;
          let gp: number;
          if (this.fun === "logcosh") {
            const t = Math.tanh(u);
            g = t;
            gp = 1 - t * t;
          } else if (this.fun === "exp") {
            const e = Math.exp(-0.5 * u * u);
            g = u * e;
            gp = (1 - u * u) * e;
          } else {
            g = u * u * u;
            gp = 3 * u * u;
          }
          gpMean[c] = (gpMean[c] ?? 0) + gp;
          const wBase = c * dim;
          for (let d = 0; d < dim; d++) {
            Wnew[wBase + d] = (Wnew[wBase + d] ?? 0) + (Z[zBase + d] ?? 0) * g;
          }
        }
      }
      for (let c = 0; c < k; c++) {
        const meanGp = (gpMean[c] ?? 0) / nSamples;
        for (let d = 0; d < dim; d++) {
          Wnew[c * dim + d] = (Wnew[c * dim + d] ?? 0) / nSamples - meanGp * (W[c * dim + d] ?? 0);
        }
      }
      FastICA.symmetricDecorrelation(Wnew, k, dim);

      let maxChange = 0;
      let finite = true;
      for (let c = 0; c < k; c++) {
        let dot = 0;
        for (let d = 0; d < dim; d++) dot += (Wnew[c * dim + d] ?? 0) * (W[c * dim + d] ?? 0);
        const change = Math.abs(Math.abs(dot) - 1);
        if (!Number.isFinite(change)) finite = false;
        else if (change > maxChange) maxChange = change;
      }
      W.set(Wnew);
      iterations = iter + 1;
      if (!finite) {
        throw new ConvergenceError(
          `FastICA diverged to non-finite values with fun="${this.fun}"; try fun="logcosh" or standardize X`,
          { iterations: iter + 1, tolerance: this.tol }
        );
      }
      if (maxChange < this.tol) {
        converged = true;
        break;
      }
    }
    if (!converged) {
      warn(
        `FastICA did not converge within maxIter=${this.maxIter} iterations; increase maxIter or tol.`,
        "ConvergenceWarning",
        "FastICA"
      );
    }

    // Unmixing matrix in feature space and its pseudo-inverse.
    const components = new Float64Array(k * nFeatures);
    const mixing = new Float64Array(nFeatures * k);
    if (whitening && whiteningVt && whiteningS) {
      // components = W @ K ; pinv(W K) = K^+ W^T with K^+ = Vt^T diag(S) / sqrt(n) (W is orthogonal).
      for (let c = 0; c < k; c++) {
        for (let j = 0; j < nFeatures; j++) {
          let sum = 0;
          for (let m = 0; m < k; m++) {
            sum += (W[c * k + m] ?? 0) * (whitening[m * nFeatures + j] ?? 0);
          }
          components[c * nFeatures + j] = sum;
        }
      }
      for (let j = 0; j < nFeatures; j++) {
        for (let c = 0; c < k; c++) {
          let sum = 0;
          for (let m = 0; m < k; m++) {
            sum +=
              (whiteningVt[m * nFeatures + j] ?? 0) *
              (((whiteningS[m] ?? 0) / sqrtN) * (W[c * k + m] ?? 0));
          }
          mixing[j * k + c] = sum;
        }
      }
    } else {
      // Rows of W are orthonormal, so pinv(W) = W^T.
      components.set(W);
      for (let c = 0; c < k; c++) {
        for (let j = 0; j < nFeatures; j++) mixing[j * k + c] = W[c * nFeatures + j] ?? 0;
      }
    }

    this.mean_ = mean;
    this.components_ = components;
    this.mixing_ = mixing;
    this.nFeaturesIn_ = nFeatures;
    this.nIter_ = iterations;
    this.kActual_ = k;
    this.outDType_ = floatDTypeOf(X);
    this.fitted = true;
    return this;
  }

  /**
   * Recover the sources: `(X - mean) @ componentsᵀ`.
   *
   * @param X - Data of shape (n_samples, n_features)
   * @returns Sources of shape (n_samples, n_components)
   * @throws {NotFittedError} If the model has not been fitted
   */
  transform(X: Tensor): Tensor {
    if (!this.fitted || !this.mean_ || !this.components_) {
      throw new NotFittedError("FastICA must be fitted before transform");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "FastICA");
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = this.nFeaturesIn_;
    const k = this.kActual_;
    const data = toFloat64View(X);
    const mean = this.mean_;
    const comps = this.components_;

    const out = new Float64Array(nSamples * k);
    for (let i = 0; i < nSamples; i++) {
      for (let c = 0; c < k; c++) {
        let sum = 0;
        for (let j = 0; j < nFeatures; j++) {
          sum +=
            ((data[i * nFeatures + j] ?? 0) - (mean[j] ?? 0)) * (comps[c * nFeatures + j] ?? 0);
        }
        out[i * k + c] = sum;
      }
    }
    return makeTensor(out, [nSamples, k], floatDTypeOf(X));
  }

  /**
   * Fit the model and return the recovered sources.
   *
   * @param X - Training data
   * @param y - Ignored (exists for compatibility)
   * @returns Sources of shape (n_samples, n_components)
   */
  fitTransform(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    return this.transform(X);
  }

  /**
   * Map sources back to the feature space: `S @ mixingMatrixᵀ + mean`.
   *
   * @param X - Sources of shape (n_samples, n_components)
   * @returns Reconstructed data of shape (n_samples, n_features)
   * @throws {NotFittedError} If the model has not been fitted
   */
  inverseTransform(X: Tensor): Tensor {
    if (!this.fitted || !this.mixing_ || !this.mean_) {
      throw new NotFittedError("FastICA must be fitted before inverseTransform");
    }
    const k = this.kActual_;
    const nFeatures = this.nFeaturesIn_;
    const data = readLatentInput(X, k, "components");
    const nSamples = X.shape[0] ?? 0;
    const A = this.mixing_;
    const mean = this.mean_;

    const out = new Float64Array(nSamples * nFeatures);
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        let sum = mean[j] ?? 0;
        for (let c = 0; c < k; c++) sum += (data[i * k + c] ?? 0) * (A[j * k + c] ?? 0);
        out[i * nFeatures + j] = sum;
      }
    }
    return makeTensor(out, [nSamples, nFeatures], floatDTypeOf(X));
  }

  /**
   * Unmixing matrix in feature space, shape (n_components, n_features).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get components(): Tensor {
    if (!this.fitted || !this.components_) {
      throw new NotFittedError("FastICA must be fitted to access components");
    }
    return makeTensor(
      Float64Array.from(this.components_),
      [this.kActual_, this.nFeaturesIn_],
      this.outDType_
    );
  }

  /**
   * Mixing matrix (pseudo-inverse of `components`), shape (n_features, n_components).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get mixingMatrix(): Tensor {
    if (!this.fitted || !this.mixing_) {
      throw new NotFittedError("FastICA must be fitted to access mixing matrix");
    }
    return makeTensor(
      Float64Array.from(this.mixing_),
      [this.nFeaturesIn_, this.kActual_],
      this.outDType_
    );
  }

  /**
   * Per-feature mean of the training data, shape (n_features,).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get mean(): Tensor {
    if (!this.fitted || !this.mean_) {
      throw new NotFittedError("FastICA must be fitted to access the mean");
    }
    return makeTensor(Float64Array.from(this.mean_), [this.nFeaturesIn_], this.outDType_);
  }

  /** Number of fixed-point iterations run by the last fit. */
  get nIter(): number {
    return this.nIter_;
  }

  /**
   * Number of features seen during fit.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nFeaturesIn(): number {
    if (!this.fitted) {
      throw new NotFittedError("FastICA must be fitted to access nFeaturesIn");
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
      maxIter: this.maxIter,
      tol: this.tol,
      fun: this.fun,
      whiten: this.whiten,
      randomState: this.randomState,
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
          this.nComponents = validatePositiveInt(value, "nComponents");
          break;
        case "maxIter":
          this.maxIter = validatePositiveInt(value, "maxIter");
          break;
        case "tol":
          this.tol = validateTolValue(value);
          break;
        case "fun":
          this.fun = validateIcaFun(value);
          break;
        case "whiten":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("whiten must be a boolean", "whiten", value);
          }
          this.whiten = value;
          break;
        case "randomState":
          validateRandomStateValue(value);
          this.randomState = value as number | undefined;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  /**
   * Symmetric decorrelation in place: W <- (W W^T)^(-1/2) W for a row-major (rows x cols) W.
   */
  private static symmetricDecorrelation(W: Float64Array, rows: number, cols: number): void {
    const WWT = new Float64Array(rows * rows);
    for (let i = 0; i < rows; i++) {
      for (let j = i; j < rows; j++) {
        let s = 0;
        for (let d = 0; d < cols; d++) s += (W[i * cols + d] ?? 0) * (W[j * cols + d] ?? 0);
        WWT[i * rows + j] = s;
        WWT[j * rows + i] = s;
      }
    }
    const { values, vectors } = jacobiEigenSymmetric(WWT, rows);
    const floor = Math.max((values[0] ?? 0) * 1e-14, 1e-300);

    // invSqrt = V diag(1/sqrt(lambda)) V^T
    const invSqrt = new Float64Array(rows * rows);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < rows; j++) {
        let s = 0;
        for (let d = 0; d < rows; d++) {
          s +=
            (vectors[i * rows + d] ?? 0) *
            (vectors[j * rows + d] ?? 0) *
            (1 / Math.sqrt(Math.max(values[d] ?? 0, floor)));
        }
        invSqrt[i * rows + j] = s;
      }
    }
    W.set(matmul(invSqrt, rows, rows, W, cols));
  }
}

// ---------------------------------------------------------------------------
// LatentDirichletAllocation
// ---------------------------------------------------------------------------

/**
 * Latent Dirichlet Allocation (LDA) for topic modeling.
 *
 * Decomposes a document-term matrix into document-topic and topic-term
 * distributions using batch variational Bayes inference. This is the topic
 * model, not Linear Discriminant Analysis (see `LinearDiscriminantAnalysis`).
 *
 * Input X should be a non-negative matrix (e.g., from CountVectorizer).
 *
 * @example
 * ```ts
 * import { LatentDirichletAllocation } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[3, 0, 1], [0, 2, 4], [1, 1, 1]]);
 * const lda = new LatentDirichletAllocation({ nComponents: 2, randomState: 0 });
 * const docTopics = lda.fitTransform(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-decomposition | Deepbox Dimensionality Reduction}
 */
export class LatentDirichletAllocation implements Transformer {
  private nComponents: number;
  private maxIter: number;
  private tol: number;
  private docTopicPrior: number | undefined;
  private topicWordPrior: number | undefined;
  private randomState: number | undefined;

  private components_?: Float64Array; // lambda (K x V)
  private nComponentsFit_ = 0;
  private docTopicPriorFit_ = 0;
  private nFeaturesIn_ = 0;
  private nIter_ = 0;
  private outDType_: FloatDType = "float64";
  private fitted = false;

  /**
   * @param options.nComponents - Number of topics (default: 10)
   * @param options.maxIter - Maximum number of passes over the corpus (default: 10)
   * @param options.tol - Stop when the largest change of a topic-word parameter is below this (default: 1e-3)
   * @param options.docTopicPrior - Dirichlet prior on document-topic weights (default: 1 / nComponents)
   * @param options.topicWordPrior - Dirichlet prior on topic-word weights (default: 1 / nComponents)
   * @param options.randomState - Seed for the random initialization
   */
  constructor(
    options: {
      readonly nComponents?: number;
      readonly maxIter?: number;
      readonly tol?: number;
      readonly docTopicPrior?: number;
      readonly topicWordPrior?: number;
      readonly randomState?: number;
    } = {}
  ) {
    this.nComponents = validatePositiveInt(options.nComponents ?? 10, "nComponents");
    this.maxIter = validatePositiveInt(options.maxIter ?? 10, "maxIter");
    this.tol = validateTolValue(options.tol ?? 1e-3);
    if (options.docTopicPrior !== undefined) {
      this.docTopicPrior = LatentDirichletAllocation.checkPrior(
        options.docTopicPrior,
        "docTopicPrior"
      );
    }
    if (options.topicWordPrior !== undefined) {
      this.topicWordPrior = LatentDirichletAllocation.checkPrior(
        options.topicWordPrior,
        "topicWordPrior"
      );
    }
    if (options.randomState !== undefined) {
      validateRandomStateValue(options.randomState);
      this.randomState = options.randomState;
    }
  }

  private static checkPrior(value: unknown, name: string): number {
    if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
      throw new InvalidParameterError(`${name} must be a finite number > 0`, name, value);
    }
    return value;
  }

  /** E[log beta] for each topic: digamma(lambda_kv) - digamma(sum_v lambda_kv). */
  private static expectedLogBeta(lambda: Float64Array, K: number, V: number): Float64Array {
    const out = new Float64Array(K * V);
    for (let k = 0; k < K; k++) {
      let sumK = 0;
      for (let v = 0; v < V; v++) sumK += lambda[k * V + v] ?? 0;
      const digSumK = digamma(sumK);
      for (let v = 0; v < V; v++) out[k * V + v] = digamma(lambda[k * V + v] ?? 0) - digSumK;
    }
    return out;
  }

  /**
   * Variational inference of the topic weights gamma of one document.
   *
   * `counts` holds the document's word counts at `offset .. offset + V`. When `sstats`
   * is given, the expected topic-word counts of the document are added to it.
   */
  private static inferDocument(
    counts: Float64Array,
    offset: number,
    V: number,
    eLogBeta: Float64Array,
    K: number,
    alpha: number,
    gamma: Float64Array,
    sstats: Float64Array | undefined
  ): void {
    const words: number[] = [];
    for (let v = 0; v < V; v++) if ((counts[offset + v] ?? 0) !== 0) words.push(v);

    const eLogTheta = new Float64Array(K);
    const phi = new Float64Array(K);
    const gammaNew = new Float64Array(K);

    // Fills `phi` with the responsibilities of word `v` given the current eLogTheta.
    const responsibilities = (v: number): void => {
      let maxLp = Number.NEGATIVE_INFINITY;
      for (let k = 0; k < K; k++) {
        const lp = (eLogTheta[k] ?? 0) + (eLogBeta[k * V + v] ?? 0);
        phi[k] = lp;
        if (lp > maxLp) maxLp = lp;
      }
      let sum = 0;
      for (let k = 0; k < K; k++) {
        const e = Math.exp((phi[k] ?? 0) - maxLp);
        phi[k] = e;
        sum += e;
      }
      for (let k = 0; k < K; k++) phi[k] = sum > 0 ? (phi[k] ?? 0) / sum : 1 / K;
    };

    const updateExpectedLogTheta = (): void => {
      let sumGamma = 0;
      for (let k = 0; k < K; k++) sumGamma += gamma[k] ?? 0;
      const digSum = digamma(sumGamma);
      for (let k = 0; k < K; k++) eLogTheta[k] = digamma(gamma[k] ?? 0) - digSum;
    };

    for (let inner = 0; inner < 20; inner++) {
      updateExpectedLogTheta();
      gammaNew.fill(alpha);
      for (const v of words) {
        const wc = counts[offset + v] ?? 0;
        responsibilities(v);
        for (let k = 0; k < K; k++) gammaNew[k] = (gammaNew[k] ?? 0) + wc * (phi[k] ?? 0);
      }
      let change = 0;
      for (let k = 0; k < K; k++) {
        change += Math.abs((gammaNew[k] ?? 0) - (gamma[k] ?? 0));
        gamma[k] = gammaNew[k] ?? 0;
      }
      if (change < 1e-3) break;
    }

    if (sstats) {
      updateExpectedLogTheta();
      for (const v of words) {
        const wc = counts[offset + v] ?? 0;
        responsibilities(v);
        for (let k = 0; k < K; k++) {
          sstats[k * V + v] = (sstats[k * V + v] ?? 0) + wc * (phi[k] ?? 0);
        }
      }
    }
  }

  /**
   * Fit the topic model with batch variational Bayes.
   *
   * @param X - Non-negative document-term matrix of shape (n_samples, n_features)
   * @param y - Ignored (exists for compatibility)
   * @returns this
   * @throws {DataValidationError} If X contains negative or non-finite values
   */
  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    assertAllNonNegative(X, "LDA requires all values in X to be non-negative");
    const nSamples = X.shape[0] ?? 0;
    const V = X.shape[1] ?? 0;
    const K = this.nComponents;
    const alpha = this.docTopicPrior ?? 1 / K;
    const eta = this.topicWordPrior ?? 1 / K;
    const rng = createGenerator(this.randomState);
    const data = toFloat64View(X);

    // Initialize topic-word parameters lambda (K x V)
    const lambda = rng.randomArray(K * V);
    for (let i = 0; i < lambda.length; i++) lambda[i] = eta + (lambda[i] ?? 0) * 0.1;

    let iterations = 0;
    for (let iter = 0; iter < this.maxIter; iter++) {
      const eLogBeta = LatentDirichletAllocation.expectedLogBeta(lambda, K, V);
      const sstats = new Float64Array(K * V);
      const gamma = new Float64Array(K);

      // E-step: per-document variational inference
      for (let d = 0; d < nSamples; d++) {
        for (let k = 0; k < K; k++) gamma[k] = alpha + rng.random() * 0.01;
        LatentDirichletAllocation.inferDocument(data, d * V, V, eLogBeta, K, alpha, gamma, sstats);
      }

      // M-step: lambda = eta + expected topic-word counts
      let maxDiff = 0;
      for (let i = 0; i < K * V; i++) {
        const next = eta + (sstats[i] ?? 0);
        const diff = Math.abs(next - (lambda[i] ?? 0));
        if (diff > maxDiff) maxDiff = diff;
        lambda[i] = next;
      }
      iterations = iter + 1;
      if (maxDiff < this.tol) break;
    }

    this.components_ = lambda;
    this.nComponentsFit_ = K;
    this.docTopicPriorFit_ = alpha;
    this.nFeaturesIn_ = V;
    this.nIter_ = iterations;
    this.outDType_ = floatDTypeOf(X);
    this.fitted = true;
    return this;
  }

  /**
   * Infer normalized document-topic distributions for X.
   *
   * @param X - Non-negative document-term matrix of shape (n_samples, n_features)
   * @returns Topic proportions of shape (n_samples, n_components); each row sums to 1
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {DataValidationError} If X contains negative or non-finite values
   */
  transform(X: Tensor): Tensor {
    if (!this.fitted || !this.components_) {
      throw new NotFittedError("LatentDirichletAllocation must be fitted before transform");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "LatentDirichletAllocation");
    assertAllNonNegative(X, "LDA requires all values in X to be non-negative");
    const nSamples = X.shape[0] ?? 0;
    const V = this.nFeaturesIn_;
    const K = this.nComponentsFit_;
    const alpha = this.docTopicPriorFit_;
    const data = toFloat64View(X);
    const eLogBeta = LatentDirichletAllocation.expectedLogBeta(this.components_, K, V);

    const result = new Float64Array(nSamples * K);
    const gamma = new Float64Array(K);
    for (let d = 0; d < nSamples; d++) {
      gamma.fill(alpha + 1);
      LatentDirichletAllocation.inferDocument(data, d * V, V, eLogBeta, K, alpha, gamma, undefined);
      let sumG = 0;
      for (let k = 0; k < K; k++) sumG += gamma[k] ?? 0;
      for (let k = 0; k < K; k++) result[d * K + k] = sumG > 0 ? (gamma[k] ?? 0) / sumG : 1 / K;
    }
    return makeTensor(result, [nSamples, K], floatDTypeOf(X));
  }

  /**
   * Fit the model and return the document-topic distributions of X.
   *
   * @param X - Non-negative document-term matrix
   * @param y - Ignored (exists for compatibility)
   * @returns Topic proportions of shape (n_samples, n_components)
   */
  fitTransform(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    return this.transform(X);
  }

  /**
   * Transform topic distributions back to approximate word distributions.
   *
   * Reconstructs word distributions by multiplying the topic-document
   * distribution by the normalized topic-word matrix (components).
   *
   * @param X - Topic distributions of shape (n_samples, n_components)
   * @returns Reconstructed word distributions of shape (n_samples, n_features)
   * @throws {NotFittedError} If the model has not been fitted
   */
  inverseTransform(X: Tensor): Tensor {
    if (!this.fitted || !this.components_) {
      throw new NotFittedError("LatentDirichletAllocation must be fitted before inverseTransform");
    }
    const K = this.nComponentsFit_;
    const V = this.nFeaturesIn_;
    const data = readLatentInput(X, K, "topics");
    const nSamples = X.shape[0] ?? 0;

    // Normalize components_ rows to get topic-word probabilities
    const beta = new Float64Array(K * V);
    for (let k = 0; k < K; k++) {
      let rowSum = 0;
      for (let v = 0; v < V; v++) rowSum += this.components_[k * V + v] ?? 0;
      for (let v = 0; v < V; v++) {
        beta[k * V + v] = rowSum > 0 ? (this.components_[k * V + v] ?? 0) / rowSum : 0;
      }
    }
    return makeTensor(matmul(data, nSamples, K, beta, V), [nSamples, V], floatDTypeOf(X));
  }

  /**
   * Topic-word variational parameters (lambda), shape (n_components, n_features).
   * Normalize each row to get a topic-word distribution.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get components(): Tensor {
    if (!this.fitted || !this.components_) {
      throw new NotFittedError("LatentDirichletAllocation must be fitted to access components");
    }
    return makeTensor(
      Float64Array.from(this.components_),
      [this.nComponentsFit_, this.nFeaturesIn_],
      this.outDType_
    );
  }

  /** Number of passes over the corpus run by the last fit. */
  get nIter(): number {
    return this.nIter_;
  }

  /**
   * Number of features (vocabulary size) seen during fit.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nFeaturesIn(): number {
    if (!this.fitted) {
      throw new NotFittedError("LatentDirichletAllocation must be fitted to access nFeaturesIn");
    }
    return this.nFeaturesIn_;
  }

  /**
   * Get hyperparameters for this estimator. Unset priors are reported as their
   * effective value, 1 / nComponents.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      nComponents: this.nComponents,
      maxIter: this.maxIter,
      tol: this.tol,
      docTopicPrior: this.docTopicPrior ?? 1 / this.nComponents,
      topicWordPrior: this.topicWordPrior ?? 1 / this.nComponents,
      randomState: this.randomState,
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
          this.nComponents = validatePositiveInt(value, "nComponents");
          break;
        case "maxIter":
          this.maxIter = validatePositiveInt(value, "maxIter");
          break;
        case "tol":
          this.tol = validateTolValue(value);
          break;
        case "docTopicPrior":
          this.docTopicPrior =
            value === undefined
              ? undefined
              : LatentDirichletAllocation.checkPrior(value, "docTopicPrior");
          break;
        case "topicWordPrior":
          this.topicWordPrior =
            value === undefined
              ? undefined
              : LatentDirichletAllocation.checkPrior(value, "topicWordPrior");
          break;
        case "randomState":
          validateRandomStateValue(value);
          this.randomState = value as number | undefined;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
