import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { svd } from "../../linalg";
import { mean, type Tensor, tensor } from "../../ndarray";
import { Generator } from "../../random/Generator";
import { __random } from "../../random/random";
import {
  assertContiguous,
  validatePredictInputs,
  validateUnsupervisedFitInputs,
} from "../_validation";
import type { Transformer } from "../base";

/**
 * Principal Component Analysis (PCA).
 *
 * Linear dimensionality reduction using Singular Value Decomposition (SVD)
 * to project data to a lower dimensional space.
 *
 * **Algorithm**:
 * 1. Center the data by subtracting the mean
 * 2. Compute SVD: X = U * Σ * V^T
 * 3. Principal components are columns of V
 * 4. Transform data by projecting onto principal components
 *
 * **Time Complexity**: O(min(n*d^2, d*n^2)) where n=samples, d=features
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
 * @see {@link https://deepbox.dev/docs/ml-decomposition | Deepbox Dimensionality Reduction}
 */
export class PCA implements Transformer {
  private nComponents: number | undefined;
  private whiten: boolean;
  private svdSolver: "auto" | "full" | "randomized";
  private nOversamples: number;
  private randomState: number | undefined;

  private components_?: Tensor;
  private explainedVariance_?: Tensor;
  private explainedVarianceRatio_?: Tensor;
  private mean_?: Tensor;
  private nComponentsActual_?: number;
  private nFeaturesIn_?: number;
  private fitted = false;

  /**
   * Create a new PCA model.
   *
   * @param options - Configuration options
   * @param options.nComponents - Number of components to keep (default: min(n_samples, n_features))
   * @param options.whiten - Whether to whiten the data (default: false)
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
      this.nComponents = options.nComponents;
    }
    this.whiten = options.whiten ?? false;
    this.svdSolver = options.svdSolver ?? "auto";
    this.nOversamples = options.nOversamples ?? 10;
    if (options.randomState !== undefined) {
      this.randomState = options.randomState;
    }

    if (this.nComponents !== undefined) {
      if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
        throw new InvalidParameterError(
          "nComponents must be an integer >= 1",
          "nComponents",
          this.nComponents
        );
      }
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

  /**
   * Randomized SVD using the Halko-Martinsson-Tropp algorithm.
   * Computes an approximate truncated SVD.
   *
   * @param X - Matrix of shape (m, n)
   * @param k - Number of singular values/vectors to compute
   * @param nOversamples - Additional random vectors for accuracy
   * @returns [U, s, Vt] approximate truncated SVD
   */
  private randomizedSvd(X: Tensor, k: number, nOversamples: number): [Tensor, Tensor, Tensor] {
    const m = X.shape[0] ?? 0;
    const n = X.shape[1] ?? 0;
    const p = Math.min(k + nOversamples, n);

    // Create PCG-based RNG for reproducible random projections
    const rng = this.randomState !== undefined ? new Generator(this.randomState) : null;

    const nextRandom = (): number => {
      if (rng) return rng.random();
      return __random();
    };

    // Step 1: Generate random Gaussian matrix Omega of shape (n, p)
    const omega = new Float64Array(n * p);
    for (let i = 0; i < n * p; i++) {
      // Box-Muller transform for Gaussian
      const u1 = Math.max(1e-15, nextRandom());
      const u2 = nextRandom();
      omega[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    // Step 2: Form Y = X @ Omega, shape (m, p)
    const Y = new Float64Array(m * p);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < p; j++) {
        let s = 0;
        for (let l = 0; l < n; l++) {
          s += Number(X.data[X.offset + i * n + l]) * (omega[l * p + j] ?? 0);
        }
        Y[i * p + j] = s;
      }
    }

    // Step 3: QR factorization of Y to get orthonormal basis Q, shape (m, p)
    // Using modified Gram-Schmidt
    const Q = new Float64Array(m * p);
    Q.set(Y);
    for (let j = 0; j < p; j++) {
      // Normalize column j
      let norm = 0;
      for (let i = 0; i < m; i++) {
        norm += (Q[i * p + j] ?? 0) * (Q[i * p + j] ?? 0);
      }
      norm = Math.sqrt(norm);
      if (norm > 1e-15) {
        for (let i = 0; i < m; i++) {
          Q[i * p + j] = (Q[i * p + j] ?? 0) / norm;
        }
      }
      // Orthogonalize remaining columns
      for (let jj = j + 1; jj < p; jj++) {
        let dot = 0;
        for (let i = 0; i < m; i++) {
          dot += (Q[i * p + j] ?? 0) * (Q[i * p + jj] ?? 0);
        }
        for (let i = 0; i < m; i++) {
          Q[i * p + jj] = (Q[i * p + jj] ?? 0) - dot * (Q[i * p + j] ?? 0);
        }
      }
    }

    // Step 4: Form B = Q^T @ X, shape (p, n)
    const B = new Float64Array(p * n);
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let l = 0; l < m; l++) {
          s += (Q[l * p + i] ?? 0) * Number(X.data[X.offset + l * n + j]);
        }
        B[i * n + j] = s;
      }
    }

    // Step 5: Compute SVD of small matrix B
    const BTensor = tensor(Array.from(B)).reshape([p, n]);
    const [Ub, sb, Vtb] = svd(BTensor, false);

    // Step 6: U = Q @ Ub, truncated to k columns
    const kActual = Math.min(k, p, Math.min(m, n));
    const Ufull = new Float64Array(m * kActual);
    const UbSize = Ub.shape[1] ?? 0;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < kActual; j++) {
        let s = 0;
        const ubCols = Math.min(p, UbSize);
        for (let l = 0; l < ubCols; l++) {
          s += (Q[i * p + l] ?? 0) * Number(Ub.data[Ub.offset + l * UbSize + j] ?? 0);
        }
        Ufull[i * kActual + j] = s;
      }
    }

    // Extract truncated s and Vt
    const sData = new Float64Array(kActual);
    for (let i = 0; i < kActual; i++) {
      sData[i] = Number(sb.data[sb.offset + i] ?? 0);
    }

    const VtData = new Float64Array(kActual * n);
    const vtCols = Vtb.shape[1] ?? n;
    for (let i = 0; i < kActual; i++) {
      for (let j = 0; j < n; j++) {
        VtData[i * n + j] = Number(Vtb.data[Vtb.offset + i * vtCols + j] ?? 0);
      }
    }

    return [
      tensor(Array.from(Ufull)).reshape([m, kActual]),
      tensor(Array.from(sData)),
      tensor(Array.from(VtData)).reshape([kActual, n]),
    ];
  }

  /**
   * Fit PCA on training data.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Ignored (exists for compatibility)
   * @returns this
   */
  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nFeatures;

    if (nSamples < 2) {
      throw new DataValidationError("X must have at least 2 samples for PCA");
    }

    // Determine number of components
    const nComponentsActual = this.nComponents ?? Math.min(nSamples, nFeatures);
    if (nComponentsActual > Math.min(nSamples, nFeatures)) {
      throw new InvalidParameterError(
        `nComponents=${nComponentsActual} must be <= min(n_samples, n_features)=${Math.min(nSamples, nFeatures)}`,
        "nComponents",
        nComponentsActual
      );
    }

    // Center the data
    const meanVec = mean(X, 0);
    this.mean_ = meanVec;

    const XCentered = this.centerData(X, meanVec);

    // Determine solver
    let useSolver = this.svdSolver;
    if (useSolver === "auto") {
      // Use randomized for large matrices when requesting few components
      if (
        nComponentsActual < Math.min(nSamples, nFeatures) &&
        Math.max(nSamples, nFeatures) > 500
      ) {
        useSolver = "randomized";
      } else {
        useSolver = "full";
      }
    }

    // Compute SVD
    let s: Tensor;
    let Vt: Tensor;
    if (useSolver === "randomized") {
      const [, sR, VtR] = this.randomizedSvd(XCentered, nComponentsActual, this.nOversamples);
      s = sR;
      Vt = VtR;
    } else {
      const [, sF, VtF] = svd(XCentered, false);
      s = sF;
      Vt = VtF;
    }

    // Extract components (rows of Vt are principal components)
    const components: number[][] = [];
    for (let i = 0; i < nComponentsActual; i++) {
      const component: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        component.push(Number(Vt.data[Vt.offset + i * nFeatures + j]));
      }
      components.push(component);
    }
    this.components_ = tensor(components);

    // Compute explained variance
    const explainedVariance: number[] = [];
    for (let i = 0; i < nComponentsActual; i++) {
      const sv = Number(s.data[s.offset + i]);
      explainedVariance.push((sv * sv) / (nSamples - 1));
    }
    this.explainedVariance_ = tensor(explainedVariance);

    // Compute explained variance ratio. Total variance is the trace of the
    // covariance (sum of all feature variances), computed from the CENTERED
    // data — NOT from the singular values, because the randomized solver
    // returns only k truncated singular values, which would make the ratio
    // spuriously sum to 1.
    let totalVariance = 0;
    for (let j = 0; j < nFeatures; j++) {
      let colSumSq = 0;
      for (let i = 0; i < nSamples; i++) {
        const v = Number(XCentered.data[XCentered.offset + i * nFeatures + j]);
        colSumSq += v * v;
      }
      totalVariance += colSumSq / (nSamples - 1);
    }
    const explainedVarianceRatio =
      totalVariance === 0
        ? explainedVariance.map(() => 0)
        : explainedVariance.map((v) => v / totalVariance);
    this.explainedVarianceRatio_ = tensor(explainedVarianceRatio);

    this.nComponentsActual_ = nComponentsActual;
    this.fitted = true;

    return this;
  }

  /**
   * Transform data to principal component space.
   *
   * @param X - Data of shape (n_samples, n_features)
   * @returns Transformed data of shape (n_samples, n_components)
   */
  transform(X: Tensor): Tensor {
    if (!this.fitted || !this.components_ || !this.mean_) {
      throw new NotFittedError("PCA must be fitted before transform");
    }

    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "PCA");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const nComponents = this.nComponentsActual_ ?? 0;

    // Center the data
    const XCentered = this.centerData(X, this.mean_);

    // Project onto principal components: X_transformed = X_centered @ components.T
    const transformed: number[][] = [];
    const varianceEps = 1e-12;
    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let k = 0; k < nComponents; k++) {
        let sum = 0;
        for (let j = 0; j < nFeatures; j++) {
          sum +=
            Number(XCentered.data[XCentered.offset + i * nFeatures + j]) *
            Number(this.components_.data[this.components_.offset + k * nFeatures + j]);
        }
        // If whitening is enabled, scale each component to unit variance.
        if (this.whiten) {
          const variance = Number(
            this.explainedVariance_?.data[this.explainedVariance_.offset + k] ?? 0
          );
          row.push(sum / Math.sqrt(variance + varianceEps));
        } else {
          row.push(sum);
        }
      }
      transformed.push(row);
    }

    return tensor(transformed);
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
   */
  inverseTransform(X: Tensor): Tensor {
    if (!this.fitted || !this.components_ || !this.mean_) {
      throw new NotFittedError("PCA must be fitted before inverse transform");
    }

    if (X.ndim !== 2) {
      throw new ShapeError(`X must be 2-dimensional; got ndim=${X.ndim}`);
    }
    assertContiguous(X, "X");

    const nSamples = X.shape[0] ?? 0;
    const nComponents = this.nComponentsActual_ ?? 0;
    const nFeatures = this.components_.shape[1] ?? 0;
    if ((X.shape[1] ?? 0) !== nComponents) {
      throw new ShapeError(
        `X must have ${nComponents} components; got ${(X.shape[1] ?? 0).toString()}`
      );
    }

    for (let i = 0; i < X.size; i++) {
      const val = X.data[X.offset + i] ?? 0;
      if (!Number.isFinite(val)) {
        throw new DataValidationError("X contains non-finite values (NaN or Inf)");
      }
    }

    // Reconstruct: X_reconstructed = X_transformed @ components
    const reconstructed: number[][] = [];
    const varianceEps = 1e-12;
    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        let sum = 0;
        for (let k = 0; k < nComponents; k++) {
          const xVal = Number(X.data[X.offset + i * nComponents + k]);
          const variance = Number(
            this.explainedVariance_?.data[this.explainedVariance_.offset + k] ?? 0
          );
          // Undo whitening by restoring the original component scale.
          const scaled = this.whiten ? xVal * Math.sqrt(variance + varianceEps) : xVal;
          sum +=
            scaled * Number(this.components_.data[this.components_.offset + k * nFeatures + j]);
        }
        // Add back the mean
        sum += Number(this.mean_.data[this.mean_.offset + j]);
        row.push(sum);
      }
      reconstructed.push(row);
    }

    return tensor(reconstructed);
  }

  /**
   * Center data by subtracting mean.
   */
  private centerData(X: Tensor, meanVec: Tensor): Tensor {
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const centered: number[][] = [];

    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        const val = Number(X.data[X.offset + i * nFeatures + j]);
        const meanVal = Number(meanVec.data[meanVec.offset + j]);
        row.push(val - meanVal);
      }
      centered.push(row);
    }

    return tensor(centered);
  }

  /**
   * Get principal components.
   */
  get components(): Tensor {
    if (!this.fitted || !this.components_) {
      throw new NotFittedError("PCA must be fitted to access components");
    }
    return this.components_;
  }

  /**
   * Get explained variance.
   */
  get explainedVariance(): Tensor {
    if (!this.fitted || !this.explainedVariance_) {
      throw new NotFittedError("PCA must be fitted to access explained variance");
    }
    return this.explainedVariance_;
  }

  /**
   * Get explained variance ratio.
   */
  get explainedVarianceRatio(): Tensor {
    if (!this.fitted || !this.explainedVarianceRatio_) {
      throw new NotFittedError("PCA must be fitted to access explained variance ratio");
    }
    return this.explainedVarianceRatio_;
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
          if (
            value !== undefined &&
            (typeof value !== "number" || !Number.isInteger(value) || value < 1)
          ) {
            throw new InvalidParameterError(
              "nComponents must be an integer >= 1 or undefined",
              "nComponents",
              value
            );
          }
          this.nComponents = value;
          break;
        case "whiten":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("whiten must be a boolean", "whiten", value);
          }
          this.whiten = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}

/**
 * Truncated SVD (aka LSA — Latent Semantic Analysis).
 *
 * Unlike PCA, TruncatedSVD does **not** center the data before computing SVD.
 * This makes it suitable for sparse data (e.g., TF-IDF matrices from text),
 * where centering would destroy sparsity.
 *
 * **Algorithm**:
 * 1. Compute SVD of X directly: X ≈ U * Σ * V^T (truncated to nComponents)
 * 2. Components are rows of V^T
 * 3. Transform: X_new = X @ V = U * Σ
 *
 * @example
 * ```ts
 * import { TruncatedSVD } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const tsvd = new TruncatedSVD({ nComponents: 2 });
 * const X_reduced = tsvd.fitTransform(X_tfidf);
 * ```
 */
export class TruncatedSVD implements Transformer {
  private nComponents: number;

  private components_?: Tensor;
  private explainedVariance_?: Tensor;
  private explainedVarianceRatio_?: Tensor;
  private singularValues_?: Tensor;
  private nFeaturesIn_?: number;
  private fitted = false;

  constructor(
    options: {
      readonly nComponents?: number;
    } = {}
  ) {
    this.nComponents = options.nComponents ?? 2;
    if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
      throw new InvalidParameterError(
        "nComponents must be an integer >= 1",
        "nComponents",
        this.nComponents
      );
    }
  }

  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nFeatures;

    const maxComponents = Math.min(nSamples, nFeatures);
    if (this.nComponents > maxComponents) {
      throw new InvalidParameterError(
        `nComponents=${this.nComponents} must be <= min(n_samples, n_features)=${maxComponents}`,
        "nComponents",
        this.nComponents
      );
    }

    // SVD without centering
    const [_U, s, Vt] = svd(X, false);

    // Extract components (top nComponents rows of Vt)
    const components: number[][] = [];
    for (let i = 0; i < this.nComponents; i++) {
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        row.push(Number(Vt.data[Vt.offset + i * nFeatures + j]));
      }
      components.push(row);
    }
    this.components_ = tensor(components);

    // Singular values
    const svals: number[] = [];
    for (let i = 0; i < this.nComponents; i++) {
      svals.push(Number(s.data[s.offset + i]));
    }
    this.singularValues_ = tensor(svals);

    // Explained variance = s^2 / (n_samples - 1)
    const explVar: number[] = svals.map((sv) => (sv * sv) / (nSamples - 1));
    this.explainedVariance_ = tensor(explVar);

    // Total variance from all singular values
    let totalVar = 0;
    for (let i = 0; i < s.size; i++) {
      const sv = Number(s.data[s.offset + i]);
      totalVar += (sv * sv) / (nSamples - 1);
    }
    this.explainedVarianceRatio_ = tensor(
      totalVar === 0 ? explVar.map(() => 0) : explVar.map((v) => v / totalVar)
    );

    this.fitted = true;
    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.fitted || !this.components_) {
      throw new NotFittedError("TruncatedSVD must be fitted before transform");
    }
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "TruncatedSVD");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    // X_new = X @ V (components_.T)
    const result: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let k = 0; k < this.nComponents; k++) {
        let sum = 0;
        for (let j = 0; j < nFeatures; j++) {
          sum +=
            Number(X.data[X.offset + i * nFeatures + j]) *
            Number(this.components_.data[this.components_.offset + k * nFeatures + j]);
        }
        row.push(sum);
      }
      result.push(row);
    }
    return tensor(result);
  }

  fitTransform(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    return this.transform(X);
  }

  inverseTransform(X: Tensor): Tensor {
    if (!this.fitted || !this.components_) {
      throw new NotFittedError("TruncatedSVD must be fitted before inverse transform");
    }
    if (X.ndim !== 2) {
      throw new ShapeError(`X must be 2-dimensional; got ndim=${X.ndim}`);
    }
    assertContiguous(X, "X");

    const nSamples = X.shape[0] ?? 0;
    const nComponents = this.nComponents;
    const nFeatures = this.components_.shape[1] ?? 0;
    if ((X.shape[1] ?? 0) !== nComponents) {
      throw new ShapeError(`X must have ${nComponents} components; got ${X.shape[1] ?? 0}`);
    }

    for (let i = 0; i < X.size; i++) {
      const val = X.data[X.offset + i] ?? 0;
      if (!Number.isFinite(val)) {
        throw new DataValidationError("X contains non-finite values (NaN or Inf)");
      }
    }

    // Reconstruct: X_reconstructed = X_reduced @ components
    const result: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        let sum = 0;
        for (let k = 0; k < nComponents; k++) {
          sum +=
            Number(X.data[X.offset + i * nComponents + k]) *
            Number(this.components_.data[this.components_.offset + k * nFeatures + j]);
        }
        row.push(sum);
      }
      result.push(row);
    }
    return tensor(result);
  }

  get components(): Tensor {
    if (!this.fitted || !this.components_) {
      throw new NotFittedError("TruncatedSVD must be fitted to access components");
    }
    return this.components_;
  }

  get explainedVariance(): Tensor {
    if (!this.fitted || !this.explainedVariance_) {
      throw new NotFittedError("TruncatedSVD must be fitted to access explained variance");
    }
    return this.explainedVariance_;
  }

  get explainedVarianceRatio(): Tensor {
    if (!this.fitted || !this.explainedVarianceRatio_) {
      throw new NotFittedError("TruncatedSVD must be fitted to access explained variance ratio");
    }
    return this.explainedVarianceRatio_;
  }

  get singularValues(): Tensor {
    if (!this.fitted || !this.singularValues_) {
      throw new NotFittedError("TruncatedSVD must be fitted to access singular values");
    }
    return this.singularValues_;
  }

  getParams(): Record<string, unknown> {
    return { nComponents: this.nComponents };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nComponents":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError(
              "nComponents must be an integer >= 1",
              "nComponents",
              value
            );
          }
          this.nComponents = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}

/**
 * Non-negative Matrix Factorization (NMF).
 *
 * Factorizes a non-negative matrix X ≈ W * H where:
 * - W is the transformed data (n_samples × n_components)
 * - H is the components matrix (n_components × n_features)
 * - Both W and H are non-negative
 *
 * Uses multiplicative update rules (Lee & Seung, 2001).
 *
 * Useful for topic modeling, recommendation systems, and signal separation.
 *
 * @example
 * ```ts
 * import { NMF } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const nmf = new NMF({ nComponents: 3 });
 * const W = nmf.fitTransform(X);  // X must be non-negative
 * ```
 */
export class NMF implements Transformer {
  private nComponents: number;
  private maxIter: number;
  private tol: number;
  private randomState: number | undefined;

  private H_?: Float64Array; // (nComponents x nFeatures) row-major
  private nFeaturesIn_?: number;
  private nIter_ = 0;
  private fitted = false;

  constructor(
    options: {
      readonly nComponents?: number;
      readonly maxIter?: number;
      readonly tol?: number;
      readonly randomState?: number;
    } = {}
  ) {
    this.nComponents = options.nComponents ?? 2;
    this.maxIter = options.maxIter ?? 200;
    this.tol = options.tol ?? 1e-4;
    if (options.randomState !== undefined) this.randomState = options.randomState;

    if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
      throw new InvalidParameterError(
        "nComponents must be an integer >= 1",
        "nComponents",
        this.nComponents
      );
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new InvalidParameterError("maxIter must be an integer >= 1", "maxIter", this.maxIter);
    }
  }

  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    this.validateNonNegative(X);

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nFeatures;
    const k = this.nComponents;

    // Initialize W and H with small positive random values
    const rng = this.createRng();
    const W = new Float64Array(nSamples * k);
    const H = new Float64Array(k * nFeatures);
    for (let i = 0; i < W.length; i++) W[i] = Math.abs(rng()) * 0.1 + 1e-6;
    for (let i = 0; i < H.length; i++) H[i] = Math.abs(rng()) * 0.1 + 1e-6;

    // Extract X data
    const Xd = new Float64Array(nSamples * nFeatures);
    for (let i = 0; i < nSamples * nFeatures; i++) {
      Xd[i] = Number(X.data[X.offset + i]);
    }

    const eps = 1e-12;
    let prevCost = Infinity;

    for (let iter = 0; iter < this.maxIter; iter++) {
      // Update W: W *= (X @ H^T) / (W @ H @ H^T)
      // Numerator: X @ H^T → (nSamples x nFeatures) @ (nFeatures x k) = (nSamples x k)
      const XHt = new Float64Array(nSamples * k);
      for (let i = 0; i < nSamples; i++) {
        for (let j = 0; j < k; j++) {
          let sum = 0;
          for (let f = 0; f < nFeatures; f++) {
            sum += (Xd[i * nFeatures + f] ?? 0) * (H[j * nFeatures + f] ?? 0);
          }
          XHt[i * k + j] = sum;
        }
      }

      // Denominator: W @ H @ H^T = (W @ H) @ H^T
      // First: WH = W @ H → (nSamples x nFeatures)
      // Then: WH @ H^T → (nSamples x k)
      const WHHt = new Float64Array(nSamples * k);
      for (let i = 0; i < nSamples; i++) {
        // Compute WH[i,f] for all f, then multiply by H^T
        for (let j = 0; j < k; j++) {
          let sum = 0;
          for (let f = 0; f < nFeatures; f++) {
            // WH[i,f] = sum_c W[i,c] * H[c,f]
            let wh_if = 0;
            for (let c = 0; c < k; c++) {
              wh_if += (W[i * k + c] ?? 0) * (H[c * nFeatures + f] ?? 0);
            }
            sum += wh_if * (H[j * nFeatures + f] ?? 0);
          }
          WHHt[i * k + j] = sum;
        }
      }

      for (let i = 0; i < nSamples * k; i++) {
        W[i] = (W[i] ?? 0) * ((XHt[i] ?? 0) / ((WHHt[i] ?? 0) + eps));
      }

      // Update H: H *= (W^T @ X) / (W^T @ W @ H)
      // Numerator: W^T @ X → (k x nSamples) @ (nSamples x nFeatures) = (k x nFeatures)
      const WtX = new Float64Array(k * nFeatures);
      for (let i = 0; i < k; i++) {
        for (let j = 0; j < nFeatures; j++) {
          let sum = 0;
          for (let s = 0; s < nSamples; s++) {
            sum += (W[s * k + i] ?? 0) * (Xd[s * nFeatures + j] ?? 0);
          }
          WtX[i * nFeatures + j] = sum;
        }
      }

      // Denominator: W^T @ W @ H
      // W^T @ W → (k x k)
      const WtW = new Float64Array(k * k);
      for (let i = 0; i < k; i++) {
        for (let j = 0; j < k; j++) {
          let sum = 0;
          for (let s = 0; s < nSamples; s++) {
            sum += (W[s * k + i] ?? 0) * (W[s * k + j] ?? 0);
          }
          WtW[i * k + j] = sum;
        }
      }
      // WtW @ H → (k x nFeatures)
      const WtWH = new Float64Array(k * nFeatures);
      for (let i = 0; i < k; i++) {
        for (let j = 0; j < nFeatures; j++) {
          let sum = 0;
          for (let c = 0; c < k; c++) {
            sum += (WtW[i * k + c] ?? 0) * (H[c * nFeatures + j] ?? 0);
          }
          WtWH[i * nFeatures + j] = sum;
        }
      }

      for (let i = 0; i < k * nFeatures; i++) {
        H[i] = (H[i] ?? 0) * ((WtX[i] ?? 0) / ((WtWH[i] ?? 0) + eps));
      }

      // Check convergence: Frobenius norm of (X - W@H)
      let cost = 0;
      for (let i = 0; i < nSamples; i++) {
        for (let j = 0; j < nFeatures; j++) {
          let wh = 0;
          for (let c = 0; c < k; c++) {
            wh += (W[i * k + c] ?? 0) * (H[c * nFeatures + j] ?? 0);
          }
          const diff = (Xd[i * nFeatures + j] ?? 0) - wh;
          cost += diff * diff;
        }
      }

      this.nIter_ = iter + 1;
      if (Math.abs(prevCost - cost) / (prevCost + eps) < this.tol) break;
      prevCost = cost;
    }

    this.H_ = H;
    this.fitted = true;
    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.fitted || !this.H_) {
      throw new NotFittedError("NMF must be fitted before transform");
    }
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "NMF");
    this.validateNonNegative(X);

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const k = this.nComponents;
    const H = this.H_;

    // Solve for W given fixed H using multiplicative updates
    const rng = this.createRng();
    const W = new Float64Array(nSamples * k);
    for (let i = 0; i < W.length; i++) W[i] = Math.abs(rng()) * 0.1 + 1e-6;

    const Xd = new Float64Array(nSamples * nFeatures);
    for (let i = 0; i < nSamples * nFeatures; i++) {
      Xd[i] = Number(X.data[X.offset + i]);
    }

    const eps = 1e-12;
    // HHt = H @ H^T → (k x k)
    const HHt = new Float64Array(k * k);
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) {
        let sum = 0;
        for (let f = 0; f < nFeatures; f++) {
          sum += (H[i * nFeatures + f] ?? 0) * (H[j * nFeatures + f] ?? 0);
        }
        HHt[i * k + j] = sum;
      }
    }

    for (let iter = 0; iter < 100; iter++) {
      // Numerator: X @ H^T
      for (let i = 0; i < nSamples; i++) {
        for (let j = 0; j < k; j++) {
          let num = 0;
          for (let f = 0; f < nFeatures; f++) {
            num += (Xd[i * nFeatures + f] ?? 0) * (H[j * nFeatures + f] ?? 0);
          }
          // Denominator: W @ HHt
          let den = 0;
          for (let c = 0; c < k; c++) {
            den += (W[i * k + c] ?? 0) * (HHt[c * k + j] ?? 0);
          }
          W[i * k + j] = (W[i * k + j] ?? 0) * (num / (den + eps));
        }
      }
    }

    const result: number[] = Array.from(W);
    return tensor(result).reshape([nSamples, k]);
  }

  fitTransform(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    return this.transform(X);
  }

  get components(): Tensor {
    if (!this.fitted || !this.H_) {
      throw new NotFittedError("NMF must be fitted to access components");
    }
    const k = this.nComponents;
    const nF = this.nFeaturesIn_ ?? 0;
    const data: number[] = Array.from(this.H_);
    return tensor(data).reshape([k, nF]);
  }

  get nIter(): number {
    return this.nIter_;
  }

  getParams(): Record<string, unknown> {
    return {
      nComponents: this.nComponents,
      maxIter: this.maxIter,
      tol: this.tol,
      randomState: this.randomState,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nComponents":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError(
              "nComponents must be an integer >= 1",
              "nComponents",
              value
            );
          }
          this.nComponents = value;
          break;
        case "maxIter":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("maxIter must be an integer >= 1", "maxIter", value);
          }
          this.maxIter = value;
          break;
        case "tol":
          if (typeof value !== "number" || value < 0) {
            throw new InvalidParameterError("tol must be >= 0", "tol", value);
          }
          this.tol = value;
          break;
        case "randomState":
          if (value !== undefined && (typeof value !== "number" || !Number.isFinite(value))) {
            throw new InvalidParameterError(
              "randomState must be a finite number",
              "randomState",
              value
            );
          }
          this.randomState = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  private validateNonNegative(X: Tensor): void {
    for (let i = 0; i < X.size; i++) {
      const val = Number(X.data[X.offset + i]);
      if (val < 0) {
        throw new DataValidationError("NMF requires all values in X to be non-negative");
      }
    }
  }

  private createRng(): () => number {
    if (this.randomState === undefined) return __random;
    let s = this.randomState;
    return () => {
      s = (s * 9301 + 49297) % 233280;
      return s / 233280;
    };
  }
}

/**
 * FastICA — Independent Component Analysis using the fast fixed-point algorithm.
 *
 * Separates a multivariate signal into additive, independent non-Gaussian
 * components. Uses negentropy maximization with the logcosh or exp
 * contrast functions.
 *
 * @example
 * ```ts
 * import { FastICA } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [3, 4], [5, 6], [7, 8]]);
 * const ica = new FastICA({ nComponents: 2 });
 * const S = ica.fitTransform(X);
 * ```
 */
export class FastICA implements Transformer {
  private nComponents: number;
  private maxIter: number;
  private tol: number;
  private fun: "logcosh" | "exp" | "cube";
  private whiten: boolean;
  private randomState: number | undefined;

  private mean_?: Float64Array;
  private whitening_?: Float64Array; // whitening matrix (nComponents x nFeatures)
  private unmixing_?: Float64Array; // unmixing matrix W (nComponents x nComponents)
  private mixing_?: Float64Array; // mixing matrix A = pinv(W) (nComponents x nComponents)
  private nFeaturesIn_ = 0;
  private nIter_ = 0;
  private fitted = false;
  // Actual component count after clamping to min(nComponents, nFeatures);
  // getters use this (not the requested nComponents) to avoid reshape errors
  // when nComponents > nFeatures.
  private kActual_ = 0;

  constructor(
    options: {
      readonly nComponents?: number;
      readonly maxIter?: number;
      readonly tol?: number;
      readonly fun?: "logcosh" | "exp" | "cube";
      readonly whiten?: boolean;
      readonly randomState?: number;
    } = {}
  ) {
    this.nComponents = options.nComponents ?? 2;
    this.maxIter = options.maxIter ?? 200;
    this.tol = options.tol ?? 1e-4;
    this.fun = options.fun ?? "logcosh";
    this.whiten = options.whiten ?? true;
    if (options.randomState !== undefined) this.randomState = options.randomState;

    if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
      throw new InvalidParameterError("nComponents must be >= 1", "nComponents", this.nComponents);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new InvalidParameterError("maxIter must be >= 1", "maxIter", this.maxIter);
    }
  }

  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nFeatures;
    const k = Math.min(this.nComponents, nFeatures);
    this.kActual_ = k;

    // Extract & center data
    const data = new Float64Array(nSamples * nFeatures);
    this.mean_ = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) {
      let s = 0;
      for (let i = 0; i < nSamples; i++) {
        s += Number(X.data[X.offset + i * nFeatures + j]);
      }
      this.mean_[j] = s / nSamples;
    }
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        data[i * nFeatures + j] =
          Number(X.data[X.offset + i * nFeatures + j]) - (this.mean_[j] ?? 0);
      }
    }

    // Whiten using SVD
    let Z: Float64Array; // whitened data (nSamples x k)
    if (this.whiten) {
      const dataTensor = tensor(Array.from(data)).reshape([nSamples, nFeatures]);
      // Reduced SVD: whitening only needs S and Vt (the right singular
      // vectors), not the full n×n U — building U for a tall matrix is O(n²)
      // memory/time and dominated FastICA runtime (~30s for 500 samples).
      const [, S, Vt] = svd(dataTensor, false);

      // K = sqrt(n) * diag(1/S[:k]) @ Vt[:k, :]  (whitening matrix: k x nFeatures).
      // The sqrt(n) factor (scikit-learn's convention) makes the whitened data
      // Z = X·Kᵀ have UNIT variance. Without it Z has variance 1/n, so wᵀz lies
      // in the linear regime of the nonlinearity and the FastICA fixed-point
      // never moves — it "converges" at iteration 1 to a random rotation.
      const sqrtN = Math.sqrt(nSamples);
      this.whitening_ = new Float64Array(k * nFeatures);
      for (let c = 0; c < k; c++) {
        const sVal = Math.max(Number(S.data[S.offset + c]), 1e-12);
        for (let j = 0; j < nFeatures; j++) {
          this.whitening_[c * nFeatures + j] =
            (Number(Vt.data[Vt.offset + c * nFeatures + j]) / sVal) * sqrtN;
        }
      }

      // Z = X_centered @ K^T  (nSamples x k)
      Z = new Float64Array(nSamples * k);
      for (let i = 0; i < nSamples; i++) {
        for (let c = 0; c < k; c++) {
          let s = 0;
          for (let j = 0; j < nFeatures; j++) {
            s += (data[i * nFeatures + j] ?? 0) * (this.whitening_[c * nFeatures + j] ?? 0);
          }
          Z[i * k + c] = s;
        }
      }
    } else {
      Z = new Float64Array(data);
    }

    // FastICA fixed-point iteration
    const dim = this.whiten ? k : nFeatures;
    const rng = this.createRng();

    // Initialize W randomly (k x dim)
    const W = new Float64Array(k * dim);
    for (let i = 0; i < k * dim; i++) {
      W[i] = rng() - 0.5;
    }

    // Orthogonalize W using symmetric decorrelation
    this.symmetricDecorrelation(W, k, dim);

    for (let iter = 0; iter < this.maxIter; iter++) {
      const Wnew = new Float64Array(k * dim);
      let maxChange = 0;

      for (let c = 0; c < k; c++) {
        // Compute w^T z for all samples
        const wx = new Float64Array(nSamples);
        for (let i = 0; i < nSamples; i++) {
          let s = 0;
          for (let d = 0; d < dim; d++) {
            s += (W[c * dim + d] ?? 0) * (Z[i * dim + d] ?? 0);
          }
          wx[i] = s;
        }

        // Apply nonlinearity g and g'
        const gx = new Float64Array(nSamples);
        const gpx = new Float64Array(nSamples);
        this.applyNonlinearity(wx, gx, gpx, nSamples);

        // Update: w_new = E[z * g(w^T z)] - E[g'(w^T z)] * w
        const meanGp = gpx.reduce((a, b) => a + b, 0) / nSamples;
        for (let d = 0; d < dim; d++) {
          let eZg = 0;
          for (let i = 0; i < nSamples; i++) {
            eZg += (Z[i * dim + d] ?? 0) * (gx[i] ?? 0);
          }
          eZg /= nSamples;
          Wnew[c * dim + d] = eZg - meanGp * (W[c * dim + d] ?? 0);
        }
      }

      // Symmetric decorrelation of Wnew
      this.symmetricDecorrelation(Wnew, k, dim);

      // Check convergence (max abs change in dot products)
      for (let c = 0; c < k; c++) {
        let dotProd = 0;
        for (let d = 0; d < dim; d++) {
          dotProd += (Wnew[c * dim + d] ?? 0) * (W[c * dim + d] ?? 0);
        }
        const change = 1 - Math.abs(dotProd);
        if (change > maxChange) maxChange = change;
      }

      // Copy Wnew -> W
      for (let i = 0; i < k * dim; i++) {
        W[i] = Wnew[i] ?? 0;
      }

      this.nIter_ = iter + 1;
      if (maxChange < this.tol) break;
    }

    this.unmixing_ = W;

    // Compute mixing matrix (pseudo-inverse of W)
    // For square W: A = W^{-1}; for non-square: A = pinv(W)
    this.mixing_ = this.pseudoInverse(W, k, dim);

    this.fitted = true;
    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.fitted || !this.mean_) {
      throw new NotFittedError("FastICA must be fitted before transform");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "FastICA");
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const k = Math.min(this.nComponents, nFeatures);

    // Center
    const centered = new Float64Array(nSamples * nFeatures);
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        centered[i * nFeatures + j] =
          Number(X.data[X.offset + i * nFeatures + j]) - (this.mean_[j] ?? 0);
      }
    }

    let Z: Float64Array;
    if (this.whiten && this.whitening_) {
      Z = new Float64Array(nSamples * k);
      for (let i = 0; i < nSamples; i++) {
        for (let c = 0; c < k; c++) {
          let s = 0;
          for (let j = 0; j < nFeatures; j++) {
            s += (centered[i * nFeatures + j] ?? 0) * (this.whitening_[c * nFeatures + j] ?? 0);
          }
          Z[i * k + c] = s;
        }
      }
    } else {
      Z = centered;
    }

    // Apply unmixing: S = Z @ W^T
    const dim = this.whiten ? k : nFeatures;
    const W = this.unmixing_!;
    const result = new Float64Array(nSamples * k);
    for (let i = 0; i < nSamples; i++) {
      for (let c = 0; c < k; c++) {
        let s = 0;
        for (let d = 0; d < dim; d++) {
          s += (Z[i * dim + d] ?? 0) * (W[c * dim + d] ?? 0);
        }
        result[i * k + c] = s;
      }
    }

    return tensor(Array.from(result)).reshape([nSamples, k]);
  }

  fitTransform(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    return this.transform(X);
  }

  inverseTransform(X: Tensor): Tensor {
    if (!this.fitted || !this.mixing_ || !this.mean_) {
      throw new NotFittedError("FastICA must be fitted before inverseTransform");
    }
    const nSamples = X.shape[0] ?? 0;
    const k = Math.min(this.nComponents, this.nFeaturesIn_);
    const nFeatures = this.nFeaturesIn_;
    const dim = this.whiten ? k : nFeatures;

    // Reconstruct: X_centered = S @ A^T, then un-whiten and add mean
    // S @ A^T gives back the whitened space, then K^{-1} maps to original
    // Simplified: use mixing_ (pseudo-inverse of unmixing)
    const A = this.mixing_!;

    // Z_recon = S @ A^T
    const Zrecon = new Float64Array(nSamples * dim);
    for (let i = 0; i < nSamples; i++) {
      for (let d = 0; d < dim; d++) {
        let s = 0;
        for (let c = 0; c < k; c++) {
          s += Number(X.data[X.offset + i * k + c]) * (A[d * k + c] ?? 0);
        }
        Zrecon[i * dim + d] = s;
      }
    }

    // Un-whiten if needed
    const result = new Float64Array(nSamples * nFeatures);
    if (this.whiten && this.whitening_) {
      // K is k x nFeatures, so K^+ (pseudoinverse) is nFeatures x k
      const Kpinv = this.pseudoInverse(this.whitening_, k, nFeatures);
      for (let i = 0; i < nSamples; i++) {
        for (let j = 0; j < nFeatures; j++) {
          let s = 0;
          for (let c = 0; c < k; c++) {
            s += (Zrecon[i * k + c] ?? 0) * (Kpinv[j * k + c] ?? 0);
          }
          result[i * nFeatures + j] = s + (this.mean_[j] ?? 0);
        }
      }
    } else {
      for (let i = 0; i < nSamples; i++) {
        for (let j = 0; j < nFeatures; j++) {
          result[i * nFeatures + j] = (Zrecon[i * nFeatures + j] ?? 0) + (this.mean_[j] ?? 0);
        }
      }
    }

    return tensor(Array.from(result)).reshape([nSamples, nFeatures]);
  }

  get components(): Tensor {
    if (!this.fitted || !this.unmixing_) {
      throw new NotFittedError("FastICA must be fitted to access components");
    }
    const k = this.kActual_;
    const dim = this.whiten ? k : this.nFeaturesIn_;
    return tensor(Array.from(this.unmixing_)).reshape([k, dim]);
  }

  get mixingMatrix(): Tensor {
    if (!this.fitted || !this.mixing_) {
      throw new NotFittedError("FastICA must be fitted to access mixing matrix");
    }
    const k = this.kActual_;
    const dim = this.whiten ? k : this.nFeaturesIn_;
    return tensor(Array.from(this.mixing_)).reshape([dim, k]);
  }

  get nIter(): number {
    return this.nIter_;
  }

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

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nComponents":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError(
              "nComponents must be an integer >= 1",
              "nComponents",
              value
            );
          }
          this.nComponents = value;
          break;
        case "maxIter":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("maxIter must be an integer >= 1", "maxIter", value);
          }
          this.maxIter = value;
          break;
        case "tol":
          if (typeof value !== "number" || value < 0) {
            throw new InvalidParameterError("tol must be >= 0", "tol", value);
          }
          this.tol = value;
          break;
        case "fun":
          if (value !== "logcosh" && value !== "exp" && value !== "cube") {
            throw new InvalidParameterError(
              `fun must be "logcosh", "exp", or "cube"`,
              "fun",
              value
            );
          }
          this.fun = value;
          break;
        case "whiten":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("whiten must be a boolean", "whiten", value);
          }
          this.whiten = value;
          break;
        case "randomState":
          if (value !== undefined && (typeof value !== "number" || !Number.isFinite(value))) {
            throw new InvalidParameterError(
              "randomState must be a finite number",
              "randomState",
              value
            );
          }
          this.randomState = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  private applyNonlinearity(
    wx: Float64Array,
    gx: Float64Array,
    gpx: Float64Array,
    n: number
  ): void {
    if (this.fun === "logcosh") {
      for (let i = 0; i < n; i++) {
        const t = Math.tanh(wx[i] ?? 0);
        gx[i] = t;
        gpx[i] = 1 - t * t;
      }
    } else if (this.fun === "exp") {
      for (let i = 0; i < n; i++) {
        const u = wx[i] ?? 0;
        const e = Math.exp(-0.5 * u * u);
        gx[i] = u * e;
        gpx[i] = (1 - u * u) * e;
      }
    } else {
      // cube
      for (let i = 0; i < n; i++) {
        const u = wx[i] ?? 0;
        gx[i] = u * u * u;
        gpx[i] = 3 * u * u;
      }
    }
  }

  private symmetricDecorrelation(W: Float64Array, rows: number, cols: number): void {
    // W = W @ (W @ W^T)^{-1/2}
    // Compute WWT = W @ W^T
    const WWT = new Float64Array(rows * rows);
    for (let i = 0; i < rows; i++) {
      for (let j = i; j < rows; j++) {
        let s = 0;
        for (let d = 0; d < cols; d++) {
          s += (W[i * cols + d] ?? 0) * (W[j * cols + d] ?? 0);
        }
        WWT[i * rows + j] = s;
        WWT[j * rows + i] = s;
      }
    }

    // Eigen-decompose WWT (small symmetric matrix)
    const eigenvals = new Float64Array(rows);
    const eigenvecs = new Float64Array(rows * rows);
    for (let i = 0; i < rows; i++) eigenvecs[i * rows + i] = 1;

    // Jacobi iteration for small symmetric matrix
    const A = new Float64Array(WWT);
    const V = new Float64Array(eigenvecs);
    for (let sweep = 0; sweep < 100; sweep++) {
      let off = 0;
      for (let i = 0; i < rows; i++) {
        for (let j = i + 1; j < rows; j++) {
          off += Math.abs(A[i * rows + j] ?? 0);
        }
      }
      if (off < 1e-12) break;

      for (let p = 0; p < rows; p++) {
        for (let q = p + 1; q < rows; q++) {
          const apq = A[p * rows + q] ?? 0;
          if (Math.abs(apq) < 1e-15) continue;
          const theta = 0.5 * Math.atan2(2 * apq, (A[p * rows + p] ?? 0) - (A[q * rows + q] ?? 0));
          const c = Math.cos(theta);
          const s = Math.sin(theta);

          for (let i = 0; i < rows; i++) {
            const aip = A[i * rows + p] ?? 0;
            const aiq = A[i * rows + q] ?? 0;
            A[i * rows + p] = c * aip + s * aiq;
            A[i * rows + q] = -s * aip + c * aiq;
          }
          for (let j = 0; j < rows; j++) {
            const apj = A[p * rows + j] ?? 0;
            const aqj = A[q * rows + j] ?? 0;
            A[p * rows + j] = c * apj + s * aqj;
            A[q * rows + j] = -s * apj + c * aqj;
          }
          for (let i = 0; i < rows; i++) {
            const vip = V[i * rows + p] ?? 0;
            const viq = V[i * rows + q] ?? 0;
            V[i * rows + p] = c * vip + s * viq;
            V[i * rows + q] = -s * vip + c * viq;
          }
        }
      }
    }

    // eigenvals = diag(A), compute D^{-1/2}
    for (let i = 0; i < rows; i++) {
      eigenvals[i] = Math.max(A[i * rows + i] ?? 0, 1e-12);
    }

    // Compute (WWT)^{-1/2} = V @ diag(1/sqrt(eigenvals)) @ V^T
    const invSqrt = new Float64Array(rows * rows);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < rows; j++) {
        let s = 0;
        for (let d = 0; d < rows; d++) {
          s += (V[i * rows + d] ?? 0) * (1 / Math.sqrt(eigenvals[d] ?? 1)) * (V[j * rows + d] ?? 0);
        }
        invSqrt[i * rows + j] = s;
      }
    }

    // W_new = invSqrt @ W
    const Wnew = new Float64Array(rows * cols);
    for (let i = 0; i < rows; i++) {
      for (let d = 0; d < cols; d++) {
        let s = 0;
        for (let j = 0; j < rows; j++) {
          s += (invSqrt[i * rows + j] ?? 0) * (W[j * cols + d] ?? 0);
        }
        Wnew[i * cols + d] = s;
      }
    }

    for (let i = 0; i < rows * cols; i++) {
      W[i] = Wnew[i] ?? 0;
    }
  }

  private pseudoInverse(M: Float64Array, rows: number, cols: number): Float64Array {
    // Compute M^+ = (M^T M)^{-1} M^T for rows <= cols (tall pseudo-inverse)
    // or M^T (M M^T)^{-1} for rows > cols
    if (rows <= cols) {
      // M^+ = M^T (M M^T)^{-1}
      const MMT = new Float64Array(rows * rows);
      for (let i = 0; i < rows; i++) {
        for (let j = i; j < rows; j++) {
          let s = 0;
          for (let d = 0; d < cols; d++) {
            s += (M[i * cols + d] ?? 0) * (M[j * cols + d] ?? 0);
          }
          MMT[i * rows + j] = s;
          MMT[j * rows + i] = s;
        }
      }
      const MMTinv = this.invertSmall(MMT, rows);
      // Result is cols x rows
      const result = new Float64Array(cols * rows);
      for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
          let s = 0;
          for (let k = 0; k < rows; k++) {
            s += (M[k * cols + i] ?? 0) * (MMTinv[k * rows + j] ?? 0);
          }
          result[i * rows + j] = s;
        }
      }
      return result;
    } else {
      // M^+ = (M^T M)^{-1} M^T
      const MTM = new Float64Array(cols * cols);
      for (let i = 0; i < cols; i++) {
        for (let j = i; j < cols; j++) {
          let s = 0;
          for (let d = 0; d < rows; d++) {
            s += (M[d * cols + i] ?? 0) * (M[d * cols + j] ?? 0);
          }
          MTM[i * cols + j] = s;
          MTM[j * cols + i] = s;
        }
      }
      const MTMinv = this.invertSmall(MTM, cols);
      const result = new Float64Array(cols * rows);
      for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
          let s = 0;
          for (let k = 0; k < cols; k++) {
            s += (MTMinv[i * cols + k] ?? 0) * (M[j * cols + k] ?? 0);
          }
          result[i * rows + j] = s;
        }
      }
      return result;
    }
  }

  private invertSmall(A: Float64Array, n: number): Float64Array {
    const aug = new Float64Array(n * 2 * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) aug[i * 2 * n + j] = A[i * n + j] ?? 0;
      aug[i * 2 * n + n + i] = 1;
    }
    for (let col = 0; col < n; col++) {
      let maxVal = Math.abs(aug[col * 2 * n + col] ?? 0);
      let maxRow = col;
      for (let row = col + 1; row < n; row++) {
        const val = Math.abs(aug[row * 2 * n + col] ?? 0);
        if (val > maxVal) {
          maxVal = val;
          maxRow = row;
        }
      }
      if (maxRow !== col) {
        for (let j = 0; j < 2 * n; j++) {
          const tmp = aug[col * 2 * n + j] ?? 0;
          aug[col * 2 * n + j] = aug[maxRow * 2 * n + j] ?? 0;
          aug[maxRow * 2 * n + j] = tmp;
        }
      }
      const pivot = aug[col * 2 * n + col] ?? 1;
      if (Math.abs(pivot) < 1e-20) continue;
      for (let j = 0; j < 2 * n; j++) aug[col * 2 * n + j] = (aug[col * 2 * n + j] ?? 0) / pivot;
      for (let row = 0; row < n; row++) {
        if (row === col) continue;
        const factor = aug[row * 2 * n + col] ?? 0;
        for (let j = 0; j < 2 * n; j++) {
          aug[row * 2 * n + j] = (aug[row * 2 * n + j] ?? 0) - factor * (aug[col * 2 * n + j] ?? 0);
        }
      }
    }
    const inv = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) inv[i * n + j] = aug[i * 2 * n + n + j] ?? 0;
    }
    return inv;
  }

  private createRng(): () => number {
    if (this.randomState === undefined) return __random;
    let s = this.randomState;
    return () => {
      s = (s * 9301 + 49297) % 233280;
      return s / 233280;
    };
  }
}

/**
 * Latent Dirichlet Allocation (LDA) for topic modeling.
 *
 * Decomposes a document-term matrix into document-topic and topic-term
 * distributions using online variational Bayes inference.
 *
 * Input X should be a non-negative matrix (e.g., from CountVectorizer).
 *
 * @example
 * ```ts
 * import { LatentDirichletAllocation } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[3, 0, 1], [0, 2, 4], [1, 1, 1]]);
 * const lda = new LatentDirichletAllocation({ nComponents: 2 });
 * const docTopics = lda.fitTransform(X);
 * ```
 */
export class LatentDirichletAllocation implements Transformer {
  private nComponents: number;
  private maxIter: number;
  private tol: number;
  private docTopicPrior: number;
  private topicWordPrior: number;
  private randomState: number | undefined;

  private components_?: Float64Array;
  private nFeaturesIn_ = 0;
  private nIter_ = 0;
  private fitted = false;

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
    this.nComponents = options.nComponents ?? 10;
    this.maxIter = options.maxIter ?? 10;
    this.tol = options.tol ?? 1e-3;
    this.docTopicPrior = options.docTopicPrior ?? 1 / this.nComponents;
    this.topicWordPrior = options.topicWordPrior ?? 1 / this.nComponents;
    if (options.randomState !== undefined) this.randomState = options.randomState;

    if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
      throw new InvalidParameterError("nComponents must be >= 1", "nComponents", this.nComponents);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new InvalidParameterError("maxIter must be >= 1", "maxIter", this.maxIter);
    }
  }

  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    this.ldaValidateNonNeg(X);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nFeatures;
    const K = this.nComponents;
    const alpha = this.docTopicPrior;
    const eta = this.topicWordPrior;
    const rng = this.createLdaRng();

    // Initialize topic-word distribution (lambda): K x nFeatures
    const lambda = new Float64Array(K * nFeatures);
    for (let i = 0; i < K * nFeatures; i++) {
      lambda[i] = eta + rng() * 0.1;
    }

    const data = new Float64Array(nSamples * nFeatures);
    for (let i = 0; i < nSamples * nFeatures; i++) {
      data[i] = Number(X.data[X.offset + i]);
    }

    for (let iter = 0; iter < this.maxIter; iter++) {
      const lambdaOld = new Float64Array(lambda);

      // E[log(beta)]
      const eLogBeta = new Float64Array(K * nFeatures);
      for (let k = 0; k < K; k++) {
        let sumK = 0;
        for (let v = 0; v < nFeatures; v++) sumK += lambda[k * nFeatures + v] ?? 0;
        const digSumK = this.ldaDigamma(sumK);
        for (let v = 0; v < nFeatures; v++) {
          eLogBeta[k * nFeatures + v] = this.ldaDigamma(lambda[k * nFeatures + v] ?? 0) - digSumK;
        }
      }

      // E-step: per-document variational inference
      const gammaAll = new Float64Array(nSamples * K);

      for (let d = 0; d < nSamples; d++) {
        const gamma = new Float64Array(K);
        for (let k = 0; k < K; k++) gamma[k] = alpha + rng() * 0.01;

        for (let innerIter = 0; innerIter < 20; innerIter++) {
          let sumGamma = 0;
          for (let k = 0; k < K; k++) sumGamma += gamma[k] ?? 0;
          const digSumG = this.ldaDigamma(sumGamma);
          const eLogTheta = new Float64Array(K);
          for (let k = 0; k < K; k++) {
            eLogTheta[k] = this.ldaDigamma(gamma[k] ?? 0) - digSumG;
          }

          const gammaNew = new Float64Array(K).fill(alpha);
          for (let v = 0; v < nFeatures; v++) {
            const wc = data[d * nFeatures + v] ?? 0;
            if (wc === 0) continue;

            const logPhi = new Float64Array(K);
            let maxLP = -Infinity;
            for (let k = 0; k < K; k++) {
              const lp = (eLogTheta[k] ?? 0) + (eLogBeta[k * nFeatures + v] ?? 0);
              logPhi[k] = lp;
              if (lp > maxLP) maxLP = lp;
            }
            let sP = 0;
            for (let k = 0; k < K; k++) sP += Math.exp((logPhi[k] ?? 0) - maxLP);
            if (sP > 0) {
              for (let k = 0; k < K; k++) {
                gammaNew[k] = (gammaNew[k] ?? 0) + (wc * Math.exp((logPhi[k] ?? 0) - maxLP)) / sP;
              }
            }
          }

          let change = 0;
          for (let k = 0; k < K; k++) {
            const gNew = gammaNew[k] ?? 0;
            const gOld = gamma[k] ?? 0;
            change += Math.abs(gNew - gOld);
            gamma[k] = gNew;
          }
          if (change < 1e-3) break;
        }

        for (let k = 0; k < K; k++) gammaAll[d * K + k] = gamma[k] ?? 0;
      }

      // M-step: update lambda
      for (let k = 0; k < K; k++) {
        for (let v = 0; v < nFeatures; v++) {
          let sumPhiW = eta;
          for (let d = 0; d < nSamples; d++) {
            const wc = data[d * nFeatures + v] ?? 0;
            if (wc === 0) continue;

            let sumGD = 0;
            for (let kk = 0; kk < K; kk++) sumGD += gammaAll[d * K + kk] ?? 0;
            const digSumD = this.ldaDigamma(sumGD);

            const logPhi = new Float64Array(K);
            let maxLP = -Infinity;
            for (let kk = 0; kk < K; kk++) {
              const elt = this.ldaDigamma(gammaAll[d * K + kk] ?? 0) - digSumD;
              const lpVal = elt + (eLogBeta[kk * nFeatures + v] ?? 0);
              logPhi[kk] = lpVal;
              if (lpVal > maxLP) maxLP = lpVal;
            }
            let sP = 0;
            for (let kk = 0; kk < K; kk++) sP += Math.exp((logPhi[kk] ?? 0) - maxLP);
            const phiDVK = sP > 0 ? Math.exp((logPhi[k] ?? 0) - maxLP) / sP : 1 / K;
            sumPhiW += wc * phiDVK;
          }
          lambda[k * nFeatures + v] = sumPhiW;
        }
      }

      let maxDiff = 0;
      for (let i = 0; i < K * nFeatures; i++) {
        const lNew = lambda[i] ?? 0;
        const lOld = lambdaOld[i] ?? 0;
        const diff = Math.abs(lNew - lOld);
        if (diff > maxDiff) maxDiff = diff;
      }
      this.nIter_ = iter + 1;
      if (maxDiff < this.tol) break;
    }

    this.components_ = lambda;
    this.fitted = true;
    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.fitted || !this.components_) {
      throw new NotFittedError("LatentDirichletAllocation must be fitted before transform");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "LatentDirichletAllocation");
    this.ldaValidateNonNeg(X);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const K = this.nComponents;
    const alpha = this.docTopicPrior;
    const lambda = this.components_!;

    const eLogBeta = new Float64Array(K * nFeatures);
    for (let k = 0; k < K; k++) {
      let sumK = 0;
      for (let v = 0; v < nFeatures; v++) sumK += lambda[k * nFeatures + v] ?? 0;
      const digSumK = this.ldaDigamma(sumK);
      for (let v = 0; v < nFeatures; v++) {
        eLogBeta[k * nFeatures + v] = this.ldaDigamma(lambda[k * nFeatures + v] ?? 0) - digSumK;
      }
    }

    const result = new Float64Array(nSamples * K);
    for (let d = 0; d < nSamples; d++) {
      const gamma = new Float64Array(K).fill(alpha + 1);
      for (let innerIter = 0; innerIter < 20; innerIter++) {
        let sumGamma = 0;
        for (let k = 0; k < K; k++) sumGamma += gamma[k] ?? 0;
        const digSum = this.ldaDigamma(sumGamma);
        const eLogTheta = new Float64Array(K);
        for (let k = 0; k < K; k++) eLogTheta[k] = this.ldaDigamma(gamma[k] ?? 0) - digSum;

        const gammaNew = new Float64Array(K).fill(alpha);
        for (let v = 0; v < nFeatures; v++) {
          const wc = Number(X.data[X.offset + d * nFeatures + v]);
          if (wc === 0) continue;
          const logPhi = new Float64Array(K);
          let maxLP = -Infinity;
          for (let k = 0; k < K; k++) {
            const lpv = (eLogTheta[k] ?? 0) + (eLogBeta[k * nFeatures + v] ?? 0);
            logPhi[k] = lpv;
            if (lpv > maxLP) maxLP = lpv;
          }
          let sP = 0;
          for (let k = 0; k < K; k++) sP += Math.exp((logPhi[k] ?? 0) - maxLP);
          if (sP > 0) {
            for (let k = 0; k < K; k++) {
              gammaNew[k] = (gammaNew[k] ?? 0) + (wc * Math.exp((logPhi[k] ?? 0) - maxLP)) / sP;
            }
          }
        }
        let change = 0;
        for (let k = 0; k < K; k++) {
          const gN = gammaNew[k] ?? 0;
          const gO = gamma[k] ?? 0;
          change += Math.abs(gN - gO);
          gamma[k] = gN;
        }
        if (change < 1e-3) break;
      }
      let sumG = 0;
      for (let k = 0; k < K; k++) sumG += gamma[k] ?? 0;
      for (let k = 0; k < K; k++) {
        result[d * K + k] = sumG > 0 ? (gamma[k] ?? 0) / sumG : 1 / K;
      }
    }
    return tensor(Array.from(result)).reshape([nSamples, K]);
  }

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
    const nSamples = X.shape[0] ?? 0;
    const K = this.nComponents;
    const nFeatures = this.nFeaturesIn_;

    // Normalize components_ rows to get topic-word probabilities
    const beta = new Float64Array(K * nFeatures);
    for (let k = 0; k < K; k++) {
      let rowSum = 0;
      for (let v = 0; v < nFeatures; v++) {
        rowSum += this.components_[k * nFeatures + v] ?? 0;
      }
      for (let v = 0; v < nFeatures; v++) {
        beta[k * nFeatures + v] =
          rowSum > 0 ? (this.components_[k * nFeatures + v] ?? 0) / rowSum : 0;
      }
    }

    // Reconstruct: result[d, v] = sum_k X[d, k] * beta[k, v]
    const result = new Float64Array(nSamples * nFeatures);
    for (let d = 0; d < nSamples; d++) {
      for (let k = 0; k < K; k++) {
        const topicWeight = Number(X.data[X.offset + d * K + k]);
        for (let v = 0; v < nFeatures; v++) {
          result[d * nFeatures + v] =
            (result[d * nFeatures + v] ?? 0) + topicWeight * (beta[k * nFeatures + v] ?? 0);
        }
      }
    }

    return tensor(Array.from(result)).reshape([nSamples, nFeatures]);
  }

  get components(): Tensor {
    if (!this.fitted || !this.components_) {
      throw new NotFittedError("LatentDirichletAllocation must be fitted to access components");
    }
    return tensor(Array.from(this.components_)).reshape([this.nComponents, this.nFeaturesIn_]);
  }

  get nIter(): number {
    return this.nIter_;
  }

  getParams(): Record<string, unknown> {
    return {
      nComponents: this.nComponents,
      maxIter: this.maxIter,
      tol: this.tol,
      docTopicPrior: this.docTopicPrior,
      topicWordPrior: this.topicWordPrior,
      randomState: this.randomState,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nComponents":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError(
              "nComponents must be an integer >= 1",
              "nComponents",
              value
            );
          }
          this.nComponents = value;
          break;
        case "maxIter":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("maxIter must be an integer >= 1", "maxIter", value);
          }
          this.maxIter = value;
          break;
        case "tol":
          if (typeof value !== "number" || value < 0) {
            throw new InvalidParameterError("tol must be >= 0", "tol", value);
          }
          this.tol = value;
          break;
        case "docTopicPrior":
          if (typeof value !== "number" || value <= 0) {
            throw new InvalidParameterError("docTopicPrior must be > 0", "docTopicPrior", value);
          }
          this.docTopicPrior = value;
          break;
        case "topicWordPrior":
          if (typeof value !== "number" || value <= 0) {
            throw new InvalidParameterError("topicWordPrior must be > 0", "topicWordPrior", value);
          }
          this.topicWordPrior = value;
          break;
        case "randomState":
          if (value !== undefined && (typeof value !== "number" || !Number.isFinite(value))) {
            throw new InvalidParameterError(
              "randomState must be a finite number",
              "randomState",
              value
            );
          }
          this.randomState = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  private ldaDigamma(x: number): number {
    let result = 0;
    let val = x;
    while (val < 6) {
      if (val <= 0) return -1e10;
      result -= 1 / val;
      val += 1;
    }
    return (
      result +
      Math.log(val) -
      1 / (2 * val) -
      1 / (12 * val * val) +
      1 / (120 * val * val * val * val)
    );
  }

  private ldaValidateNonNeg(X: Tensor): void {
    for (let i = 0; i < X.size; i++) {
      if (Number(X.data[X.offset + i]) < 0) {
        throw new DataValidationError("LDA requires all values in X to be non-negative");
      }
    }
  }

  private createLdaRng(): () => number {
    if (this.randomState === undefined) return __random;
    let s = this.randomState;
    return () => {
      s = (s * 9301 + 49297) % 233280;
      return s / 233280;
    };
  }
}
