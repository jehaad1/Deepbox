/**
 * Random projection for dimensionality reduction.
 *
 * @module ml/random_projection
 * @see {@link https://deepbox.dev/docs/ml-decomposition | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError, warn } from "../core";
import { type Tensor, tensor } from "../ndarray";
import { __fillNormal, __SeededRandom, __seedToUint64 } from "../random/random";
import { toFloat64View, validatePredictInputs, validateUnsupervisedFitInputs } from "./_validation";
import type { Transformer } from "./base";

/**
 * Smallest number of random-projection components that keeps pairwise distances of
 * `nSamples` points within a factor `1 +/- eps`, according to the Johnson-Lindenstrauss lemma
 * (the bound used by scikit-learn's `johnson_lindenstrauss_min_dim`).
 *
 * @param nSamples - Number of points (>= 1)
 * @param eps - Allowed distortion, in the open interval (0, 1)
 * @returns Minimum safe number of components
 * @throws {InvalidParameterError} If `nSamples` is not a positive integer or `eps` is outside (0, 1)
 *
 * @example
 * ```ts
 * johnsonLindenstraussMinDim(1e6, 0.5); // 663
 * ```
 */
export function johnsonLindenstraussMinDim(nSamples: number, eps = 0.1): number {
  if (!Number.isInteger(nSamples) || nSamples < 1) {
    throw new InvalidParameterError(
      `nSamples must be a positive integer; received ${nSamples}`,
      "nSamples",
      nSamples
    );
  }
  if (!(eps > 0 && eps < 1)) {
    throw new InvalidParameterError(`eps must be in (0, 1); received ${eps}`, "eps", eps);
  }
  const denominator = (eps * eps) / 2 - (eps * eps * eps) / 3;
  return Math.trunc((4 * Math.log(nSamples)) / denominator);
}

type ProjectionOptions = {
  nComponents: number | "auto";
  eps: number;
  randomState: number | null;
};

function checkNComponents(value: unknown): number | "auto" {
  if (value === "auto") return value;
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new InvalidParameterError(
      'nComponents must be a positive integer or "auto"',
      "nComponents",
      value
    );
  }
  return value;
}

function checkEps(value: unknown): number {
  if (typeof value !== "number" || !(value > 0 && value < 1)) {
    throw new InvalidParameterError(
      `eps must be in (0, 1); received ${String(value)}`,
      "eps",
      value
    );
  }
  return value;
}

function checkRandomState(value: unknown, name: string): number | null {
  if (value === null) return null;
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new InvalidParameterError(
      `${name} must be a finite number or null; received ${String(value)}`,
      name,
      value
    );
  }
  return value;
}

/**
 * Gaussian Random Projection.
 *
 * Reduces dimensionality by projecting data onto a random Gaussian matrix.
 * Each component of the projection matrix is drawn from N(0, 1/n_components).
 *
 * Based on the Johnson-Lindenstrauss lemma: random projections approximately
 * preserve pairwise distances. With `nComponents: "auto"` the target dimension is
 * the Johnson-Lindenstrauss bound for the number of training samples and `eps`.
 *
 * Without `randomState` the matrix is drawn from the global generator, so `setSeed()`
 * makes it reproducible; with `randomState` it comes from a private stream.
 *
 * @example
 * ```ts
 * import { GaussianRandomProjection } from 'deepbox/ml';
 *
 * const proj = new GaussianRandomProjection({ nComponents: 50, randomState: 0 });
 * proj.fit(X_train);
 * const X_reduced = proj.transform(X_test);
 * ```
 *
 * @category Random Projection
 * @implements {Transformer}
 */
export class GaussianRandomProjection implements Transformer {
  private options: ProjectionOptions;

  private components_?: Float64Array;
  private nComponentsFitted_ = 0;
  private nFeaturesIn_ = 0;
  private fitted = false;

  /**
   * @param options.nComponents - Target dimension (positive integer) or "auto" (default: 2)
   * @param options.eps - Allowed distortion for `nComponents: "auto"`, in (0, 1) (default: 0.1)
   * @param options.randomState - Seed of a private random stream (default: use the global generator)
   * @param options.seed - Deprecated alias of `randomState`
   * @throws {InvalidParameterError} If a value is invalid or `seed` and `randomState` disagree
   */
  constructor(
    options: {
      readonly nComponents?: number | "auto";
      readonly eps?: number;
      readonly randomState?: number | null;
      /** @deprecated Use `randomState`. */
      readonly seed?: number | null;
    } = {}
  ) {
    const randomState = options.randomState ?? options.seed ?? null;
    if (
      options.randomState !== undefined &&
      options.seed !== undefined &&
      options.randomState !== options.seed
    ) {
      throw new InvalidParameterError(
        "seed is a deprecated alias of randomState; pass only one of them",
        "seed",
        options.seed
      );
    }
    this.options = {
      nComponents: checkNComponents(options.nComponents ?? 2),
      eps: checkEps(options.eps ?? 0.1),
      randomState: checkRandomState(randomState, "randomState"),
    };
  }

  /**
   * Draw the random projection matrix.
   *
   * @param X - Training data of shape (n_samples, n_features); only its shape is used
   * @param _y - Ignored
   * @returns this
   * @throws {InvalidParameterError} If `nComponents: "auto"` gives a dimension that is not
   *   positive or exceeds the number of features
   * @throws {ShapeError} If X is not 2-D
   */
  fit(X: Tensor, _y?: Tensor): GaussianRandomProjection {
    this.fitted = false;
    validateUnsupervisedFitInputs(X);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    let nComp: number;
    if (this.options.nComponents === "auto") {
      nComp = johnsonLindenstraussMinDim(nSamples, this.options.eps);
      if (nComp <= 0) {
        throw new InvalidParameterError(
          `eps=${this.options.eps} and nSamples=${nSamples} lead to a target dimension of ${nComp}, which is invalid`,
          "eps",
          this.options.eps
        );
      }
      if (nComp > nFeatures) {
        throw new InvalidParameterError(
          `eps=${this.options.eps} and nSamples=${nSamples} lead to a target dimension of ${nComp}, ` +
            `which is larger than the original space with nFeatures=${nFeatures}`,
          "nComponents",
          nComp
        );
      }
    } else {
      nComp = this.options.nComponents;
      if (nComp > nFeatures) {
        warn(
          `nComponents (${nComp}) is greater than the number of features (${nFeatures}); ` +
            "the dimensionality of the problem will not be reduced",
          "UserWarning",
          "GaussianRandomProjection.fit"
        );
      }
    }

    // Matrix of shape (nComponents, nFeatures) with entries ~ N(0, 1 / nComponents).
    const components = new Float64Array(nComp * nFeatures);
    const state = this.options.randomState;
    if (state === null) {
      __fillNormal(components, components.length);
    } else {
      const rng = new __SeededRandom(__seedToUint64(state));
      for (let i = 0; i < components.length; i++) components[i] = rng.nextNormal();
    }
    const scale = 1 / Math.sqrt(nComp);
    for (let i = 0; i < components.length; i++) {
      components[i] = (components[i] as number) * scale;
    }

    this.components_ = components;
    this.nComponentsFitted_ = nComp;
    this.nFeaturesIn_ = nFeatures;
    this.fitted = true;
    return this;
  }

  /**
   * Project `X` onto the random components.
   *
   * @param X - Data of shape (n_samples, n_features)
   * @returns Projected data of shape (n_samples, n_components), float64
   * @throws {NotFittedError} If the transformer has not been fitted
   * @throws {ShapeError} If X has a different number of features than in `fit`
   */
  transform(X: Tensor): Tensor {
    if (!this.fitted || !this.components_) {
      throw new NotFittedError("GaussianRandomProjection must be fitted before transform");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "GaussianRandomProjection");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = this.nFeaturesIn_;
    const nComp = this.nComponentsFitted_;
    const x = toFloat64View(X);
    const components = this.components_;

    // X_new = X @ components^T  (nSamples x nComp)
    const result = new Float64Array(nSamples * nComp);
    for (let i = 0; i < nSamples; i++) {
      const xBase = i * nFeatures;
      for (let j = 0; j < nComp; j++) {
        const cBase = j * nFeatures;
        let s = 0;
        for (let f = 0; f < nFeatures; f++) {
          s += (x[xBase + f] as number) * (components[cBase + f] as number);
        }
        result[i * nComp + j] = s;
      }
    }

    return tensor(result).reshape([nSamples, nComp]);
  }

  /**
   * Fit and project in one call.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Ignored
   * @returns Projected training data
   */
  fitTransform(X: Tensor, y?: Tensor): Tensor {
    this.fit(X, y);
    return this.transform(X);
  }

  /**
   * Projection matrix of shape (n_components, n_features).
   *
   * @throws {NotFittedError} If the transformer has not been fitted
   */
  get components(): Tensor {
    if (!this.fitted || !this.components_) {
      throw new NotFittedError("GaussianRandomProjection must be fitted to access components");
    }
    return tensor(Float64Array.from(this.components_)).reshape([
      this.nComponentsFitted_,
      this.nFeaturesIn_,
    ]);
  }

  getParams(): Record<string, unknown> {
    return {
      nComponents: this.options.nComponents,
      eps: this.options.eps,
      randomState: this.options.randomState,
      seed: this.options.randomState,
    };
  }

  /**
   * Set hyperparameters. Refit the transformer afterwards for them to take effect.
   *
   * @throws {InvalidParameterError} If a value is invalid or a key is unknown
   */
  setParams(params: Record<string, unknown>): GaussianRandomProjection {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nComponents":
          this.options.nComponents = checkNComponents(value);
          break;
        case "eps":
          this.options.eps = checkEps(value);
          break;
        case "randomState":
        case "seed":
          this.options.randomState = checkRandomState(value, key);
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
   * @returns A new GaussianRandomProjection
   */
  clone(): GaussianRandomProjection {
    return new GaussianRandomProjection({
      nComponents: this.options.nComponents,
      eps: this.options.eps,
      randomState: this.options.randomState,
    });
  }
}
