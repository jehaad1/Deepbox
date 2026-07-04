/**
 * Polynomial feature generation for ML pipelines.
 *
 * @module preprocess/polynomial
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../core";
import type { Transformer } from "../ml/base";
import { type Tensor, Tensor as TensorClass } from "../ndarray";
import { assert2D, getShape2D, getStrides2D } from "./_internal";

/**
 * Generate polynomial and interaction features.
 *
 * Generates a new feature matrix consisting of all polynomial combinations
 * of the features with degree less than or equal to the specified degree.
 *
 * For example, if input has features [a, b] and degree=2:
 * - interactionOnly=false: [1, a, b, a², ab, b²]
 * - interactionOnly=true:  [1, a, b, ab]
 *
 * @example
 * ```ts
 * import { PolynomialFeatures } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const poly = new PolynomialFeatures({ degree: 2 });
 * const X = tensor([[1, 2], [3, 4]]);
 * const Xpoly = poly.fitTransform(X);
 * // Xpoly has columns: [1, x1, x2, x1², x1*x2, x2²]
 * ```
 */
export class PolynomialFeatures implements Transformer {
  private readonly degree: number;
  private readonly interactionOnly: boolean;
  private readonly includeBias: boolean;
  private nFeaturesIn = 0;
  private fitted = false;

  constructor(
    options: {
      readonly degree?: number;
      readonly interactionOnly?: boolean;
      readonly includeBias?: boolean;
    } = {}
  ) {
    this.degree = options.degree ?? 2;
    this.interactionOnly = options.interactionOnly ?? false;
    this.includeBias = options.includeBias ?? true;

    if (!Number.isInteger(this.degree) || this.degree < 0) {
      throw new InvalidParameterError(
        "degree must be a non-negative integer",
        "degree",
        this.degree
      );
    }
  }

  fit(X: Tensor): this {
    assert2D(X, "PolynomialFeatures.fit");
    const [, nFeatures] = getShape2D(X);
    this.nFeaturesIn = nFeatures;
    this.fitted = true;
    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("PolynomialFeatures must be fitted before transform");
    }
    assert2D(X, "PolynomialFeatures.transform");
    const [nSamples, nFeatures] = getShape2D(X);

    if (nFeatures !== this.nFeaturesIn) {
      throw new InvalidParameterError(
        `Expected ${this.nFeaturesIn} features, got ${nFeatures}`,
        "X",
        nFeatures
      );
    }

    // Generate combination indices for each output column
    const combos = this.generateCombinations(nFeatures);
    const nOutputFeatures = combos.length;

    const result = new Float64Array(nSamples * nOutputFeatures);
    const [s0, s1] = getStrides2D(X);

    // Densify the input row-major once so the product loop reads monomorphic
    // Float64 values (no per-element Number() coercion) and the output wraps
    // the buffer directly (skips Array.from + nested tensor() revalidation).
    const src = X.data;
    const offset = X.offset;
    const dense = new Float64Array(nSamples * nFeatures);
    let dp = 0;
    for (let i = 0; i < nSamples; i++) {
      const rowOffset = offset + i * s0;
      for (let j = 0; j < nFeatures; j++) dense[dp++] = Number(src[rowOffset + j * s1]);
    }

    for (let i = 0; i < nSamples; i++) {
      const denseBase = i * nFeatures;
      const outBase = i * nOutputFeatures;
      for (let c = 0; c < nOutputFeatures; c++) {
        const combo = combos[c] as number[];
        let val = 1;
        for (let t = 0; t < combo.length; t++) {
          val *= dense[denseBase + (combo[t] as number)] as number;
        }
        result[outBase + c] = val;
      }
    }

    return TensorClass.fromTypedArray({
      data: result,
      shape: [nSamples, nOutputFeatures],
      dtype: "float64",
      device: X.device,
    });
  }

  fitTransform(X: Tensor): Tensor {
    this.fit(X);
    return this.transform(X);
  }

  getParams(): Record<string, unknown> {
    return {
      degree: this.degree,
      interactionOnly: this.interactionOnly,
      includeBias: this.includeBias,
    };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }

  /** Number of output features after transformation */
  get nOutputFeatures(): number {
    if (!this.fitted) {
      throw new NotFittedError("PolynomialFeatures must be fitted first");
    }
    return this.generateCombinations(this.nFeaturesIn).length;
  }

  private generateCombinations(nFeatures: number): number[][] {
    const combos: number[][] = [];

    // Bias term (degree 0)
    if (this.includeBias) {
      combos.push([]);
    }

    // Generate combinations of features for each degree
    for (let d = 1; d <= this.degree; d++) {
      if (this.interactionOnly) {
        // Only interaction terms (each feature appears at most once)
        this.combinationsWithoutRepetition(nFeatures, d, 0, [], combos);
      } else {
        // All polynomial terms (features can repeat)
        this.combinationsWithRepetition(nFeatures, d, 0, [], combos);
      }
    }

    return combos;
  }

  private combinationsWithRepetition(
    n: number,
    k: number,
    start: number,
    current: number[],
    result: number[][]
  ): void {
    if (current.length === k) {
      result.push([...current]);
      return;
    }
    for (let i = start; i < n; i++) {
      current.push(i);
      this.combinationsWithRepetition(n, k, i, current, result);
      current.pop();
    }
  }

  private combinationsWithoutRepetition(
    n: number,
    k: number,
    start: number,
    current: number[],
    result: number[][]
  ): void {
    if (current.length === k) {
      result.push([...current]);
      return;
    }
    for (let i = start; i < n; i++) {
      current.push(i);
      this.combinationsWithoutRepetition(n, k, i + 1, current, result);
      current.pop();
    }
  }
}

/**
 * Binarize data (set feature values to 0 or 1) according to a threshold.
 *
 * Values greater than the threshold map to 1, while values less than or
 * equal to the threshold map to 0.
 *
 * @example
 * ```ts
 * import { Binarizer } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const binarizer = new Binarizer({ threshold: 1.5 });
 * const X = tensor([[1, 2], [3, 0.5]]);
 * const Xb = binarizer.fitTransform(X);
 * // Xb = [[0, 1], [1, 0]]
 * ```
 */
export class Binarizer implements Transformer {
  private readonly threshold: number;
  private fitted = false;

  constructor(options: { readonly threshold?: number } = {}) {
    this.threshold = options.threshold ?? 0;
  }

  fit(_X: Tensor): this {
    this.fitted = true;
    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("Binarizer must be fitted before transform");
    }
    assert2D(X, "Binarizer.transform");
    const [nSamples, nFeatures] = getShape2D(X);
    const result = new Float64Array(nSamples * nFeatures);
    const [s0, s1] = getStrides2D(X);

    const src = X.data;
    const offset = X.offset;
    const threshold = this.threshold;
    let pos = 0;
    for (let i = 0; i < nSamples; i++) {
      const rowBase = offset + i * s0;
      for (let j = 0; j < nFeatures; j++) {
        result[pos++] = Number(src[rowBase + j * s1]) > threshold ? 1 : 0;
      }
    }

    return TensorClass.fromTypedArray({
      data: result,
      shape: [nSamples, nFeatures],
      dtype: "float64",
      device: X.device,
    });
  }

  fitTransform(X: Tensor): Tensor {
    this.fit(X);
    return this.transform(X);
  }

  getParams(): Record<string, unknown> {
    return { threshold: this.threshold };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}

/**
 * Constructs a transformer from an arbitrary callable.
 *
 * Useful for wrapping arbitrary functions in a Pipeline-compatible
 * transformer without subclassing.
 *
 * @example
 * ```ts
 * import { FunctionTransformer } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const log1p = new FunctionTransformer({
 *   func: (X) => {
 *     // apply log1p element-wise
 *     const data = [];
 *     for (let i = 0; i < X.size; i++) data.push(Math.log1p(Number(X.data[X.offset + i])));
 *     return tensor(data).reshape(X.shape);
 *   },
 * });
 * ```
 */
export class FunctionTransformer implements Transformer {
  private readonly func: ((X: Tensor) => Tensor) | undefined;
  private readonly inverseFunc: ((X: Tensor) => Tensor) | undefined;
  private fitted = false;

  constructor(
    options: {
      readonly func?: (X: Tensor) => Tensor;
      readonly inverseFunc?: (X: Tensor) => Tensor;
    } = {}
  ) {
    this.func = options.func;
    this.inverseFunc = options.inverseFunc;
  }

  fit(_X: Tensor): this {
    this.fitted = true;
    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("FunctionTransformer must be fitted before transform");
    }
    if (this.func) {
      return this.func(X);
    }
    return X;
  }

  fitTransform(X: Tensor): Tensor {
    this.fit(X);
    return this.transform(X);
  }

  inverseTransform(X: Tensor): Tensor {
    if (this.inverseFunc) {
      return this.inverseFunc(X);
    }
    return X;
  }

  getParams(): Record<string, unknown> {
    return { func: this.func, inverseFunc: this.inverseFunc };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}
