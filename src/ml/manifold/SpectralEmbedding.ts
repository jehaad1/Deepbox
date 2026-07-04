/**
 * Spectral Embedding for nonlinear dimensionality reduction.
 *
 * Constructs an affinity graph (RBF kernel) and embeds data using
 * the bottom eigenvectors of the normalized graph Laplacian.
 *
 * @module ml/manifold/SpectralEmbedding
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __random } from "../../random/random";
import { validateUnsupervisedFitInputs } from "../_validation";

export class SpectralEmbedding {
  private readonly nComponents: number;
  private readonly gamma: number;
  private embedding_?: Tensor;
  private fitted = false;

  constructor(
    options: {
      readonly nComponents?: number;
      readonly gamma?: number;
    } = {}
  ) {
    this.nComponents = options.nComponents ?? 2;
    this.gamma = options.gamma ?? 1;

    if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
      throw new InvalidParameterError(
        "nComponents must be an integer >= 1",
        "nComponents",
        this.nComponents
      );
    }
    if (this.gamma <= 0) {
      throw new InvalidParameterError("gamma must be > 0", "gamma", this.gamma);
    }
  }

  fit(X: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const n = X.shape[0] ?? 0;
    const nF = X.shape[1] ?? 0;

    const flat = new Float64Array(n * nF);
    for (let i = 0; i < n * nF; i++) {
      flat[i] = Number(X.data[X.offset + i]);
    }

    // Compute RBF affinity matrix W
    const W = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let sq = 0;
        for (let f = 0; f < nF; f++) {
          const diff = (flat[i * nF + f] ?? 0) - (flat[j * nF + f] ?? 0);
          sq += diff * diff;
        }
        const w = Math.exp(-this.gamma * sq);
        W[i * n + j] = w;
        W[j * n + i] = w;
      }
    }

    // Compute D^{-1/2}
    const Dinvsqrt = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let rowSum = 0;
      for (let j = 0; j < n; j++) rowSum += W[i * n + j] ?? 0;
      Dinvsqrt[i] = rowSum > 0 ? 1 / Math.sqrt(rowSum) : 0;
    }

    // Normalized affinity: S = D^{-1/2} W D^{-1/2}
    const S = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        S[i * n + j] = (Dinvsqrt[i] ?? 0) * (W[i * n + j] ?? 0) * (Dinvsqrt[j] ?? 0);
      }
    }

    // We need the nComponents+1 largest eigenvectors of S,
    // then discard the first (trivial) one.
    // Use deflation-based power iteration.
    const nEig = this.nComponents + 1;
    const eigvecs = new Float64Array(n * nEig);
    const eigvals = new Float64Array(nEig);
    const Swork = new Float64Array(S);

    for (let comp = 0; comp < nEig; comp++) {
      const v = new Float64Array(n);
      for (let i = 0; i < n; i++) v[i] = __random() - 0.5;

      let norm = 0;
      for (let i = 0; i < n; i++) norm += (v[i] ?? 0) * (v[i] ?? 0);
      norm = Math.sqrt(norm);
      for (let i = 0; i < n; i++) v[i] = (v[i] ?? 0) / norm;

      for (let iter = 0; iter < 300; iter++) {
        const w = new Float64Array(n);
        for (let i = 0; i < n; i++) {
          let s = 0;
          for (let j = 0; j < n; j++) {
            s += (Swork[i * n + j] ?? 0) * (v[j] ?? 0);
          }
          w[i] = s;
        }

        norm = 0;
        for (let i = 0; i < n; i++) norm += (w[i] ?? 0) * (w[i] ?? 0);
        norm = Math.sqrt(norm);
        if (norm < 1e-15) break;

        let lambda = 0;
        for (let i = 0; i < n; i++) lambda += (w[i] ?? 0) * (v[i] ?? 0);

        let diff = 0;
        for (let i = 0; i < n; i++) {
          const newV = (w[i] ?? 0) / norm;
          diff += ((v[i] ?? 0) - newV) ** 2;
          v[i] = newV;
        }
        eigvals[comp] = lambda;
        if (Math.sqrt(diff) < 1e-10) break;
      }

      for (let i = 0; i < n; i++) eigvecs[i * nEig + comp] = v[i] ?? 0;

      // Deflate
      const lambda = eigvals[comp] ?? 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          Swork[i * n + j] = (Swork[i * n + j] ?? 0) - lambda * (v[i] ?? 0) * (v[j] ?? 0);
        }
      }
    }

    // Take eigenvectors 1..nComponents (skip index 0, the trivial eigenvector)
    const result: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let d = 1; d <= this.nComponents; d++) {
        row.push(eigvecs[i * nEig + d] ?? 0);
      }
      result.push(row);
    }

    this.embedding_ = tensor(result);
    this.fitted = true;
    return this;
  }

  fitTransform(X: Tensor): Tensor {
    this.fit(X);
    return this.embedding_!;
  }

  get embedding(): Tensor {
    if (!this.fitted || !this.embedding_) {
      throw new NotFittedError("SpectralEmbedding must be fitted before accessing embedding");
    }
    return this.embedding_;
  }

  getParams(): Record<string, unknown> {
    return { nComponents: this.nComponents, gamma: this.gamma };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}
