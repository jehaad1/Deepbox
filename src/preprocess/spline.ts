/**
 * Spline basis feature generation for ML pipelines.
 *
 * @module preprocess/spline
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../core";
import type { Transformer } from "../ml/base";
import { type Tensor, tensor } from "../ndarray";
import { assert2D, getShape2D, getStrides2D } from "./_internal";

/**
 * Generate B-spline basis features for each input feature.
 *
 * Transforms each feature into a set of B-spline basis functions,
 * which can capture nonlinear relationships while remaining linear
 * in the parameters (useful for linear models).
 *
 * @example
 * ```ts
 * import { SplineTransformer } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const st = new SplineTransformer({ nKnots: 5, degree: 3 });
 * const X = tensor([[0], [0.25], [0.5], [0.75], [1.0]]);
 * const Xt = st.fitTransform(X);
 * ```
 */
export class SplineTransformer implements Transformer {
  private readonly nKnots: number;
  private readonly degree: number;
  private readonly extrapolation: "constant" | "linear" | "continue" | "periodic";
  private readonly includeBias: boolean;

  private nFeaturesIn = 0;
  private knots_?: number[][]; // knots per feature (including boundary knots)
  private fitted = false;

  constructor(
    options: {
      readonly nKnots?: number;
      readonly degree?: number;
      readonly extrapolation?: "constant" | "linear" | "continue" | "periodic";
      readonly includeBias?: boolean;
    } = {}
  ) {
    this.nKnots = options.nKnots ?? 5;
    this.degree = options.degree ?? 3;
    this.extrapolation = options.extrapolation ?? "constant";
    this.includeBias = options.includeBias ?? true;

    if (!Number.isInteger(this.nKnots) || this.nKnots < 2) {
      throw new InvalidParameterError("nKnots must be an integer >= 2", "nKnots", this.nKnots);
    }
    if (!Number.isInteger(this.degree) || this.degree < 0) {
      throw new InvalidParameterError(
        "degree must be a non-negative integer",
        "degree",
        this.degree
      );
    }
  }

  fit(X: Tensor, _y?: Tensor): this {
    assert2D(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const [__s0, __s1] = getStrides2D(X);
    this.nFeaturesIn = nFeatures;

    // Compute knots for each feature based on quantiles of training data
    this.knots_ = [];
    for (let f = 0; f < nFeatures; f++) {
      // Extract column values and sort
      const vals: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        vals.push(Number(X.data[X.offset + i * __s0 + f * __s1]));
      }
      vals.sort((a, b) => a - b);

      // Place internal knots at quantiles
      const internalKnots: number[] = [];
      for (let k = 0; k < this.nKnots; k++) {
        const q = k / (this.nKnots - 1);
        const idx = Math.min(Math.floor(q * (nSamples - 1)), nSamples - 1);
        internalKnots.push(vals[idx] ?? 0);
      }

      // Add boundary knots (degree repetitions at each end)
      const knotVector: number[] = [];
      const lo = internalKnots[0] ?? 0;
      const hi = internalKnots[internalKnots.length - 1] ?? 0;
      for (let i = 0; i < this.degree; i++) {
        knotVector.push(lo);
      }
      for (const k of internalKnots) {
        knotVector.push(k);
      }
      for (let i = 0; i < this.degree; i++) {
        knotVector.push(hi);
      }

      this.knots_.push(knotVector);
    }

    this.fitted = true;
    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.fitted || !this.knots_) {
      throw new NotFittedError("SplineTransformer must be fitted before transform");
    }
    assert2D(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const [__s0, __s1] = getStrides2D(X);

    if (nFeatures !== this.nFeaturesIn) {
      throw new InvalidParameterError(
        `Expected ${this.nFeaturesIn} features, got ${nFeatures}`,
        "nFeatures",
        nFeatures
      );
    }

    // Number of basis functions per feature
    const nBasis = this.nKnots + this.degree - 1;
    const startBasis = this.includeBias ? 0 : 1;
    const nBasisOut = nBasis - startBasis;
    const totalFeatures = nFeatures * nBasisOut;

    const result = new Float64Array(nSamples * totalFeatures);

    for (let f = 0; f < nFeatures; f++) {
      const knots = this.knots_[f]!;

      for (let i = 0; i < nSamples; i++) {
        let x = Number(X.data[X.offset + i * __s0 + f * __s1]);

        // Handle extrapolation
        const lo = knots[this.degree] ?? 0;
        const hi = knots[knots.length - 1 - this.degree] ?? 0;
        if (this.extrapolation === "constant") {
          x = Math.max(lo, Math.min(hi, x));
        }

        // Evaluate B-spline basis using de Boor recursion
        const bases = this.evaluateBSpline(x, knots, this.degree, nBasis);

        for (let b = startBasis; b < nBasis; b++) {
          result[i * totalFeatures + f * nBasisOut + (b - startBasis)] = bases[b] ?? 0;
        }
      }
    }

    return tensor(Array.from(result)).reshape([nSamples, totalFeatures]);
  }

  fitTransform(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    return this.transform(X);
  }

  inverseTransform(_X: Tensor): Tensor {
    throw new NotFittedError("SplineTransformer does not support inverseTransform");
  }

  private evaluateBSpline(x: number, knots: number[], degree: number, nBasis: number): number[] {
    const n = knots.length - 1;

    // Initialize 0-th degree basis
    const B: number[][] = [];
    for (let d = 0; d <= degree; d++) {
      B.push(new Array<number>(n).fill(0));
    }

    // Degree 0: indicator functions
    for (let j = 0; j < n; j++) {
      const kj = knots[j] ?? 0;
      const kj1 = knots[j + 1] ?? 0;
      if (kj <= x && x < kj1) {
        B[0]![j] = 1;
      } else if (x === kj1 && j + 1 === n) {
        // Include right endpoint in last interval
        B[0]![j] = 1;
      }
    }

    // Handle edge case: x equals last knot
    if (x >= (knots[n] ?? 0)) {
      // Activate the last valid basis
      for (let j = n - 1; j >= 0; j--) {
        if ((knots[j] ?? 0) < (knots[j + 1] ?? 0)) {
          B[0]![j] = 1;
          break;
        }
      }
    }

    // Recursion for higher degrees
    for (let d = 1; d <= degree; d++) {
      for (let j = 0; j < n - d; j++) {
        const kj = knots[j] ?? 0;
        const kjd = knots[j + d] ?? 0;
        const kj1 = knots[j + 1] ?? 0;
        const kjd1 = knots[j + d + 1] ?? 0;

        let left = 0;
        if (kjd - kj > 1e-15) {
          left = ((x - kj) / (kjd - kj)) * (B[d - 1]![j] ?? 0);
        }

        let right = 0;
        if (kjd1 - kj1 > 1e-15) {
          right = ((kjd1 - x) / (kjd1 - kj1)) * (B[d - 1]![j + 1] ?? 0);
        }

        B[d]![j] = left + right;
      }
    }

    // Extract the basis functions at the target degree
    const result: number[] = [];
    for (let j = 0; j < nBasis; j++) {
      result.push(B[degree]![j] ?? 0);
    }

    return result;
  }

  get nFeaturesOut(): number {
    if (!this.fitted)
      throw new NotFittedError("SplineTransformer must be fitted to get nFeaturesOut");
    const nBasis = this.nKnots + this.degree - 1;
    const startBasis = this.includeBias ? 0 : 1;
    return this.nFeaturesIn * (nBasis - startBasis);
  }

  getParams(): Record<string, unknown> {
    return {
      nKnots: this.nKnots,
      degree: this.degree,
      extrapolation: this.extrapolation,
      includeBias: this.includeBias,
    };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}
