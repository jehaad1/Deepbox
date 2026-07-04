/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../core";
import { type Tensor, tensor } from "../ndarray";
import { __random } from "../random/random";
import { validatePredictInputs, validateUnsupervisedFitInputs } from "./_validation";
import type { Transformer } from "./base";

/**
 * Gaussian Random Projection.
 *
 * Reduces dimensionality by projecting data onto a random Gaussian matrix.
 * Each component of the projection matrix is drawn from N(0, 1/n_components).
 *
 * Based on the Johnson-Lindenstrauss lemma: random projections approximately
 * preserve pairwise distances.
 *
 * @example
 * ```ts
 * import { GaussianRandomProjection } from 'deepbox/ml';
 *
 * const proj = new GaussianRandomProjection({ nComponents: 50 });
 * proj.fit(X_train);
 * const X_reduced = proj.transform(X_test);
 * ```
 *
 * @category Random Projection
 * @implements {Transformer}
 */
export class GaussianRandomProjection implements Transformer {
  private options: {
    nComponents: number;
    seed: number | null;
  };

  private components_?: Float64Array;
  private nFeaturesIn_?: number;
  private fitted = false;

  constructor(
    options: {
      readonly nComponents?: number;
      readonly seed?: number | null;
    } = {}
  ) {
    this.options = {
      nComponents: options.nComponents ?? 2,
      seed: options.seed ?? null,
    };
    if (this.options.nComponents < 1 || !Number.isInteger(this.options.nComponents)) {
      throw new InvalidParameterError(
        "nComponents must be a positive integer",
        "nComponents",
        this.options.nComponents
      );
    }
  }

  fit(X: Tensor, _y?: Tensor): GaussianRandomProjection {
    validateUnsupervisedFitInputs(X);
    const nFeatures = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nFeatures;

    const nComp = this.options.nComponents;
    const scale = 1 / Math.sqrt(nComp);

    // Generate random Gaussian projection matrix (nComponents x nFeatures)
    const components = new Float64Array(nComp * nFeatures);
    const rng = this.options.seed !== null ? this.seededRng(this.options.seed) : null;

    for (let i = 0; i < nComp * nFeatures; i++) {
      components[i] = this.randn(rng) * scale;
    }

    this.components_ = components;
    this.fitted = true;
    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.fitted || !this.components_) {
      throw new NotFittedError("GaussianRandomProjection");
    }
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "GaussianRandomProjection");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const nComp = this.options.nComponents;

    // X_new = X @ components^T  (nSamples x nComp)
    const result = new Float64Array(nSamples * nComp);
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nComp; j++) {
        let s = 0;
        for (let f = 0; f < nFeatures; f++) {
          s +=
            Number(X.data[(X.offset ?? 0) + i * nFeatures + f] ?? 0) *
            (this.components_[j * nFeatures + f] ?? 0);
        }
        result[i * nComp + j] = s;
      }
    }

    return tensor(Array.from(result), { dtype: "float64" }).reshape([nSamples, nComp]);
  }

  fitTransform(X: Tensor, y?: Tensor): Tensor {
    this.fit(X, y);
    return this.transform(X);
  }

  getParams(): Record<string, unknown> {
    return { ...this.options };
  }

  setParams(params: Record<string, unknown>): GaussianRandomProjection {
    if (params["nComponents"] !== undefined)
      this.options.nComponents = params["nComponents"] as number;
    if (params["seed"] !== undefined) this.options.seed = params["seed"] as number | null;
    return this;
  }

  /** Box-Muller with optional seeded RNG */
  private randn(rng: (() => number) | null): number {
    const rand = rng ?? __random;
    let u = 0;
    let v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** Simple seeded PRNG (xorshift32) */
  private seededRng(seed: number): () => number {
    let state = seed | 0 || 1;
    return () => {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      return (state >>> 0) / 4294967296;
    };
  }
}
