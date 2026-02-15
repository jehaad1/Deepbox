import {
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  NotImplementedError,
  ShapeError,
} from "../../core";
import { svd } from "../../linalg";
import { mean, type Tensor, tensor } from "../../ndarray";
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
  private readonly nComponents?: number;
  private readonly whiten: boolean;

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
   */
  constructor(
    options: {
      readonly nComponents?: number;
      readonly whiten?: boolean;
    } = {}
  ) {
    if (options.nComponents !== undefined) {
      this.nComponents = options.nComponents;
    }
    this.whiten = options.whiten ?? false;

    if (this.nComponents !== undefined) {
      if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
        throw new InvalidParameterError(
          "nComponents must be an integer >= 1",
          "nComponents",
          this.nComponents
        );
      }
    }
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

    // Compute SVD
    const [_U, s, Vt] = svd(XCentered, false);

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

    // Compute explained variance ratio
    let totalVariance = 0;
    for (let i = 0; i < s.size; i++) {
      const sv = Number(s.data[s.offset + i]);
      totalVariance += (sv * sv) / (nSamples - 1);
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
   * @param _params - Parameters to set
   * @throws {NotImplementedError} Always — parameters cannot be changed after construction
   */
  setParams(_params: Record<string, unknown>): this {
    throw new NotImplementedError("PCA does not support setParams after construction");
  }
}
