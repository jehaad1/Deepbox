/**
 * Multidimensional Scaling (MDS).
 *
 * Classical (metric) MDS embeds data in a lower-dimensional space
 * by preserving pairwise distances as faithfully as possible.
 *
 * @module ml/manifold/MDS
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { fromDenseMatrix2D } from "../../linalg/_internal";
import type { Tensor } from "../../ndarray";
import { toFloat64View, validateUnsupervisedFitInputs } from "../_validation";
import { classicalMDSDecompose, pairwiseEuclidean } from "./Isomap";

/**
 * Classical (Torgerson) multidimensional scaling.
 *
 * Computes the pairwise Euclidean distances (or takes a precomputed distance
 * matrix), double-centres the squared distances and embeds the points with the
 * leading eigenvectors. This is a closed-form, deterministic solution and not
 * the iterative SMACOF stress minimisation that scikit-learn's `MDS` runs; for
 * Euclidean input it equals PCA of the data.
 *
 * @example
 * ```ts
 * import { MDS } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]]);
 * const embedding = new MDS({ nComponents: 2 }).fitTransform(X);
 *
 * // From a precomputed distance matrix
 * const D = tensor([[0, 3, 4], [3, 0, 5], [4, 5, 0]]);
 * const layout = new MDS({ nComponents: 2, dissimilarity: 'precomputed' }).fitTransform(D);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-manifold | Deepbox Manifold Learning}
 */
export class MDS {
  private nComponents: number;
  private dissimilarity: "euclidean" | "precomputed";
  private embedding_?: Tensor;
  private nFeaturesIn_ = 0;
  private fitted = false;

  /**
   * @param options.nComponents - Dimension of the embedding (default: 2)
   * @param options.dissimilarity - `"euclidean"` computes distances from the samples;
   *   `"precomputed"` expects `fit` to receive a symmetric (n_samples, n_samples)
   *   distance matrix (default: "euclidean")
   */
  constructor(
    options: {
      readonly nComponents?: number;
      readonly dissimilarity?: "euclidean" | "precomputed";
    } = {}
  ) {
    this.nComponents = options.nComponents ?? 2;
    this.dissimilarity = options.dissimilarity ?? "euclidean";
    this.validateParams();
  }

  private validateParams(): void {
    if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
      throw new InvalidParameterError(
        "nComponents must be an integer >= 1",
        "nComponents",
        this.nComponents
      );
    }
    if (this.dissimilarity !== "euclidean" && this.dissimilarity !== "precomputed") {
      throw new InvalidParameterError(
        "dissimilarity must be 'euclidean' or 'precomputed'",
        "dissimilarity",
        this.dissimilarity
      );
    }
  }

  /**
   * Compute the embedding.
   *
   * @param X - Data of shape (n_samples, n_features), or a distance matrix of
   *   shape (n_samples, n_samples) when `dissimilarity` is `"precomputed"`
   * @throws {InvalidParameterError} If `nComponents > n_samples`
   * @throws {ShapeError} If a precomputed matrix is not square
   * @throws {DataValidationError} If a precomputed matrix is not symmetric or has negative entries
   */
  fit(X: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const n = X.shape[0] ?? 0;
    const nF = X.shape[1] ?? 0;
    const flat = toFloat64View(X);

    let dist: Float64Array;
    if (this.dissimilarity === "precomputed") {
      if (n !== nF) {
        throw new ShapeError(
          `A precomputed dissimilarity matrix must be square; got shape [${n}, ${nF}]`
        );
      }
      let maxAbs = 0;
      for (let i = 0; i < flat.length; i++) {
        const v = flat[i] as number;
        if (v < 0) {
          throw new DataValidationError("A precomputed dissimilarity matrix must be non-negative");
        }
        if (v > maxAbs) maxAbs = v;
      }
      const tol = 1e-8 * maxAbs;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          if (Math.abs((flat[i * n + j] as number) - (flat[j * n + i] as number)) > tol) {
            throw new DataValidationError("A precomputed dissimilarity matrix must be symmetric");
          }
        }
      }
      dist = flat;
    } else {
      dist = pairwiseEuclidean(flat, n, nF);
    }

    const { embedding } = classicalMDSDecompose(dist, n, this.nComponents);
    this.embedding_ = fromDenseMatrix2D(n, this.nComponents, embedding);
    this.nFeaturesIn_ = nF;
    this.fitted = true;
    return this;
  }

  /**
   * Fit the model and return the embedding.
   *
   * @returns Embedding of shape (n_samples, nComponents)
   */
  fitTransform(X: Tensor): Tensor {
    this.fit(X);
    return this.embedding;
  }

  /**
   * Embedding of the fitted data, shape (n_samples, nComponents).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get embedding(): Tensor {
    if (!this.fitted || !this.embedding_) {
      throw new NotFittedError("MDS must be fitted before accessing embedding");
    }
    return this.embedding_;
  }

  /**
   * Number of columns of the matrix passed to `fit`.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nFeaturesIn(): number {
    if (!this.fitted) {
      throw new NotFittedError("MDS must be fitted before accessing nFeaturesIn");
    }
    return this.nFeaturesIn_;
  }

  getParams(): Record<string, unknown> {
    return { nComponents: this.nComponents, dissimilarity: this.dissimilarity };
  }

  /**
   * Update hyperparameters. The model must be refitted afterwards.
   *
   * @throws {InvalidParameterError} On an unknown or invalid parameter
   */
  setParams(params: Record<string, unknown>): this {
    let nComponents = this.nComponents;
    let dissimilarity = this.dissimilarity;
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nComponents":
          nComponents = value as number;
          break;
        case "dissimilarity":
          dissimilarity = value as "euclidean" | "precomputed";
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    const prev = [this.nComponents, this.dissimilarity] as const;
    this.nComponents = nComponents;
    this.dissimilarity = dissimilarity;
    try {
      this.validateParams();
    } catch (e) {
      this.nComponents = prev[0];
      this.dissimilarity = prev[1];
      throw e;
    }
    return this;
  }
}
