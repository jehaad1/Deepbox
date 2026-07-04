/**
 * Multidimensional Scaling (MDS).
 *
 * Classical (metric) MDS embeds data in a lower-dimensional space
 * by preserving pairwise distances as faithfully as possible.
 *
 * @module ml/manifold/MDS
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../../core";
import type { Tensor } from "../../ndarray";
import { validateUnsupervisedFitInputs } from "../_validation";
import { classicalMDS } from "./Isomap";

export class MDS {
  private readonly nComponents: number;
  private embedding_?: Tensor;
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

  fit(X: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const n = X.shape[0] ?? 0;
    const nF = X.shape[1] ?? 0;

    // Compute pairwise Euclidean distances
    const flat = new Float64Array(n * nF);
    for (let i = 0; i < n * nF; i++) {
      flat[i] = Number(X.data[X.offset + i]);
    }

    const dist = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let sq = 0;
        for (let f = 0; f < nF; f++) {
          const diff = (flat[i * nF + f] ?? 0) - (flat[j * nF + f] ?? 0);
          sq += diff * diff;
        }
        const d = Math.sqrt(sq);
        dist[i * n + j] = d;
        dist[j * n + i] = d;
      }
    }

    this.embedding_ = classicalMDS(dist, n, this.nComponents);
    this.fitted = true;
    return this;
  }

  fitTransform(X: Tensor): Tensor {
    this.fit(X);
    return this.embedding_!;
  }

  get embedding(): Tensor {
    if (!this.fitted || !this.embedding_) {
      throw new NotFittedError("MDS must be fitted before accessing embedding");
    }
    return this.embedding_;
  }

  getParams(): Record<string, unknown> {
    return { nComponents: this.nComponents };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}
