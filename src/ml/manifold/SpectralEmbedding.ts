/**
 * Spectral Embedding for nonlinear dimensionality reduction.
 *
 * Constructs an affinity graph (RBF kernel) and embeds data using
 * the bottom eigenvectors of the normalized graph Laplacian.
 *
 * @module ml/manifold/SpectralEmbedding
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError } from "../../core";
import { fromDenseMatrix2D } from "../../linalg/_internal";
import { eigh } from "../../linalg/decomposition/eig";
import type { Tensor } from "../../ndarray";
import { toFloat64View, validateUnsupervisedFitInputs } from "../_validation";

/**
 * Laplacian eigenmaps with a fully connected RBF affinity graph.
 *
 * The affinity is `W_ij = exp(-gamma * ||x_i - x_j||^2)` with a zero diagonal.
 * With the degree matrix `D`, the embedding consists of the eigenvectors of the
 * symmetric normalized affinity `D^-1/2 W D^-1/2` that belong to its largest
 * eigenvalues (the smallest eigenvalues of the normalized Laplacian), without
 * the trivial first one, mapped back with `D^-1/2`. This is the construction of
 * scikit-learn's `SpectralEmbedding(affinity="rbf")`.
 *
 * @example
 * ```ts
 * import { SpectralEmbedding } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0], [0, 1], [1, 0], [5, 5], [5, 6], [6, 5]]);
 * const se = new SpectralEmbedding({ nComponents: 2, gamma: 0.5 });
 * const embedding = se.fitTransform(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-manifold | Deepbox Manifold Learning}
 * @see Belkin, Niyogi (2003). "Laplacian Eigenmaps for Dimensionality Reduction and Data Representation"
 */
export class SpectralEmbedding {
  private nComponents: number;
  private gamma: number;
  private embedding_?: Tensor;
  private nFeaturesIn_ = 0;
  private fitted = false;

  /**
   * @param options.nComponents - Dimension of the embedding (default: 2)
   * @param options.gamma - RBF kernel coefficient, must be > 0 (default: 1)
   */
  constructor(
    options: {
      readonly nComponents?: number;
      readonly gamma?: number;
    } = {}
  ) {
    this.nComponents = options.nComponents ?? 2;
    this.gamma = options.gamma ?? 1;
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
    if (!Number.isFinite(this.gamma) || this.gamma <= 0) {
      throw new InvalidParameterError("gamma must be > 0", "gamma", this.gamma);
    }
  }

  /**
   * Compute the embedding.
   *
   * @param X - Data of shape (n_samples, n_features)
   * @throws {InvalidParameterError} If `nComponents >= n_samples`
   * @throws {DataValidationError} If a sample has no affinity to any other sample
   *   (all kernel values underflow to 0), which happens when `gamma` is too large for the data scale
   */
  fit(X: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const n = X.shape[0] ?? 0;
    const nF = X.shape[1] ?? 0;

    if (this.nComponents >= n) {
      throw new InvalidParameterError(
        `nComponents must be < n_samples (${n})`,
        "nComponents",
        this.nComponents
      );
    }

    const flat = toFloat64View(X);

    // RBF affinity with a zero diagonal and the degree of every node.
    const W = new Float64Array(n * n);
    const degree = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const bi = i * nF;
      for (let j = i + 1; j < n; j++) {
        const bj = j * nF;
        let sq = 0;
        for (let f = 0; f < nF; f++) {
          const diff = (flat[bi + f] as number) - (flat[bj + f] as number);
          sq += diff * diff;
        }
        const w = Math.exp(-this.gamma * sq);
        W[i * n + j] = w;
        W[j * n + i] = w;
        degree[i] = (degree[i] as number) + w;
        degree[j] = (degree[j] as number) + w;
      }
    }
    for (let i = 0; i < n; i++) {
      if (!((degree[i] as number) > 0)) {
        throw new DataValidationError(
          `SpectralEmbedding: sample ${i} has zero affinity to every other sample. ` +
            "Decrease gamma or rescale the data so that the kernel values do not underflow."
        );
      }
    }

    // Normalized affinity S = D^-1/2 W D^-1/2 (symmetric by construction).
    const invSqrtDeg = new Float64Array(n);
    for (let i = 0; i < n; i++) invSqrtDeg[i] = 1 / Math.sqrt(degree[i] as number);
    const S = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const s = (invSqrtDeg[i] as number) * (W[i * n + j] as number) * (invSqrtDeg[j] as number);
        S[i * n + j] = s;
        S[j * n + i] = s;
      }
    }

    // eigh returns ascending eigenvalues. The largest eigenvalue belongs to the
    // trivial eigenvector (proportional to sqrt(degree)); components 1..nComponents
    // follow it in descending order.
    const [, vectors] = eigh(fromDenseMatrix2D(n, n, S));
    const vecs = vectors.data as Float64Array;
    const vOff = vectors.offset;
    const vs0 = vectors.strides[0] ?? n;
    const vs1 = vectors.strides[1] ?? 1;

    const out = new Float64Array(n * this.nComponents);
    for (let c = 0; c < this.nComponents; c++) {
      const col = n - 2 - c;
      let maxAbs = 0;
      let sign = 1;
      for (let i = 0; i < n; i++) {
        const v = (vecs[vOff + i * vs0 + col * vs1] as number) * (invSqrtDeg[i] as number);
        if (Math.abs(v) > maxAbs) {
          maxAbs = Math.abs(v);
          sign = v < 0 ? -1 : 1;
        }
      }
      for (let i = 0; i < n; i++) {
        out[i * this.nComponents + c] =
          sign * (vecs[vOff + i * vs0 + col * vs1] as number) * (invSqrtDeg[i] as number);
      }
    }

    this.embedding_ = fromDenseMatrix2D(n, this.nComponents, out);
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
      throw new NotFittedError("SpectralEmbedding must be fitted before accessing embedding");
    }
    return this.embedding_;
  }

  /**
   * Number of features seen during `fit`.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nFeaturesIn(): number {
    if (!this.fitted) {
      throw new NotFittedError("SpectralEmbedding must be fitted before accessing nFeaturesIn");
    }
    return this.nFeaturesIn_;
  }

  getParams(): Record<string, unknown> {
    return { nComponents: this.nComponents, gamma: this.gamma };
  }

  /**
   * Update hyperparameters. The model must be refitted afterwards.
   *
   * @throws {InvalidParameterError} On an unknown or invalid parameter
   */
  setParams(params: Record<string, unknown>): this {
    let nComponents = this.nComponents;
    let gamma = this.gamma;
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nComponents":
          nComponents = value as number;
          break;
        case "gamma":
          gamma = value as number;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    const prev = [this.nComponents, this.gamma] as const;
    this.nComponents = nComponents;
    this.gamma = gamma;
    try {
      this.validateParams();
    } catch (e) {
      this.nComponents = prev[0];
      this.gamma = prev[1];
      throw e;
    }
    return this;
  }
}
