/**
 * Spline basis feature generation for ML pipelines.
 *
 * @module preprocess/spline
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox documentation}
 */

import {
  DataValidationError,
  DTypeError,
  InvalidParameterError,
  NotFittedError,
  NotImplementedError,
} from "../core/errors";
import type { Transformer } from "../ml/base";
import { type Tensor, Tensor as TensorClass } from "../ndarray";
import { assert2D, assertNumericTensor, getShape2D, getStrides2D } from "./_internal";

/**
 * How `transform` treats values outside the range of the fitted knots.
 *
 * - `"constant"`: values are clamped to the boundary (the basis is constant outside)
 * - `"linear"`: the basis is continued linearly, matching the value and slope at the boundary
 * - `"continue"`: the boundary polynomial pieces are continued unchanged
 * - `"periodic"`: the input is wrapped around the fitted range
 * - `"error"`: out-of-range values throw a {@link DataValidationError}
 */
export type SplineExtrapolation = "constant" | "linear" | "continue" | "periodic" | "error";

/**
 * How the knots are placed along each feature.
 *
 * - `"quantile"`: at evenly spaced quantiles of the training data
 * - `"uniform"`: evenly spaced between the training minimum and maximum
 */
export type SplineKnots = "quantile" | "uniform";

const EXTRAPOLATIONS: readonly SplineExtrapolation[] = [
  "constant",
  "linear",
  "continue",
  "periodic",
  "error",
];
const KNOT_MODES: readonly SplineKnots[] = ["quantile", "uniform"];

type SplineOptions = {
  readonly nKnots?: number;
  readonly degree?: number;
  readonly extrapolation?: SplineExtrapolation;
  readonly includeBias?: boolean;
  readonly knots?: SplineKnots;
};

/** Per-feature fitted data. */
type FeatureKnots = {
  /** Knot vector used for evaluation (boundary knots repeated, or periodically extended). */
  readonly full: Float64Array;
  /** The `nKnots` knots that were placed on the data. */
  readonly base: Float64Array;
  /** True when the training feature had a single distinct value; its output columns are zero. */
  readonly constant: boolean;
};

/**
 * Generate B-spline basis features for each input feature.
 *
 * Transforms each feature into a set of B-spline basis functions,
 * which can capture nonlinear relationships while remaining linear
 * in the parameters (useful for linear models).
 *
 * With `nKnots` knots and spline degree `degree` every feature produces
 * `nKnots + degree - 1` basis columns (`nKnots - 1` for periodic splines),
 * one fewer when `includeBias` is false. Columns are grouped by input feature.
 * The knot vector is clamped (boundary knots repeated `degree` extra times),
 * so the basis is the textbook B-spline basis; it spans the same functions as
 * scikit-learn's, but individual columns differ from scikit-learn's. A feature
 * with a single distinct value in the training data has no knot span and gives
 * all-zero columns, as in scikit-learn.
 *
 * @example
 * ```ts
 * import { SplineTransformer } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const st = new SplineTransformer({ nKnots: 5, degree: 3 });
 * const X = tensor([[0], [0.25], [0.5], [0.75], [1.0]]);
 * const Xt = st.fitTransform(X); // shape [5, 7]
 * ```
 */
export class SplineTransformer implements Transformer {
  private nKnots: number;
  private degree: number;
  private extrapolation: SplineExtrapolation;
  private includeBias: boolean;
  private knotMode: SplineKnots;

  private nFeaturesIn = 0;
  private knots_: FeatureKnots[] | undefined;

  /**
   * Creates a new SplineTransformer.
   *
   * @param options - Configuration options
   * @param options.nKnots - Number of knots per feature, including both boundary knots (default: 5)
   * @param options.degree - Degree of the splines (default: 3)
   * @param options.extrapolation - Behaviour outside the fitted range (default: "constant")
   * @param options.includeBias - Keep all basis columns; when false, one redundant column is dropped per feature, the first (the last for periodic splines) (default: true)
   * @param options.knots - Knot placement, "quantile" or "uniform" (default: "quantile")
   * @throws {InvalidParameterError} If an option is invalid, or if periodic extrapolation is
   * requested with `degree >= nKnots`
   */
  constructor(options: SplineOptions = {}) {
    this.nKnots = options.nKnots ?? 5;
    this.degree = options.degree ?? 3;
    this.extrapolation = options.extrapolation ?? "constant";
    this.includeBias = options.includeBias ?? true;
    this.knotMode = options.knots ?? "quantile";
    this.validateParams();
  }

  private validateParams(): void {
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
    if (!EXTRAPOLATIONS.includes(this.extrapolation)) {
      throw new InvalidParameterError(
        `extrapolation must be one of: ${EXTRAPOLATIONS.join(", ")}`,
        "extrapolation",
        this.extrapolation
      );
    }
    if (typeof this.includeBias !== "boolean") {
      throw new InvalidParameterError(
        "includeBias must be a boolean",
        "includeBias",
        this.includeBias
      );
    }
    if (!KNOT_MODES.includes(this.knotMode)) {
      throw new InvalidParameterError(
        `knots must be one of: ${KNOT_MODES.join(", ")}`,
        "knots",
        this.knotMode
      );
    }
    if (this.extrapolation === "periodic" && this.degree >= this.nKnots) {
      throw new InvalidParameterError(
        `periodic extrapolation requires degree < nKnots; got degree=${this.degree}, nKnots=${this.nKnots}`,
        "degree",
        this.degree
      );
    }
  }

  /** Number of basis functions per feature before the bias column is dropped. */
  private get basisPerFeature(): number {
    return this.extrapolation === "periodic" ? this.nKnots - 1 : this.nKnots + this.degree - 1;
  }

  /**
   * Place the knots on the training data of each feature.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (present for API compatibility)
   * @returns this
   * @throws {InvalidParameterError} If X is empty
   * @throws {ShapeError} If X is not 2-D
   * @throws {DTypeError} If X is not real numeric
   * @throws {DataValidationError} If X contains NaN or Infinity
   */
  fit(X: Tensor, _y?: Tensor): this {
    if (X.size === 0) {
      throw new InvalidParameterError("X must contain at least one sample and one feature", "X");
    }
    const { data, nSamples, nFeatures } = readMatrix(X);

    const knots: FeatureKnots[] = [];
    const column = new Float64Array(nSamples);
    for (let f = 0; f < nFeatures; f++) {
      for (let i = 0; i < nSamples; i++) column[i] = data[i * nFeatures + f] as number;
      column.sort();
      const lo = column[0] as number;
      const hi = column[nSamples - 1] as number;
      if (lo === hi) {
        // No knot span exists. Like scikit-learn, the feature yields all-zero columns.
        const flatKnots = new Float64Array(this.nKnots).fill(lo);
        knots.push({ base: flatKnots, full: this.extendKnots(flatKnots), constant: true });
        continue;
      }

      const base = new Float64Array(this.nKnots);
      for (let k = 0; k < this.nKnots; k++) {
        const q = k / (this.nKnots - 1);
        base[k] = this.knotMode === "uniform" ? lo + q * (hi - lo) : quantile(column, q);
      }
      base[0] = lo;
      base[this.nKnots - 1] = hi;
      knots.push({ base, full: this.extendKnots(base), constant: false });
    }

    this.knots_ = knots;
    this.nFeaturesIn = nFeatures;
    return this;
  }

  private extendKnots(base: Float64Array): Float64Array {
    const p = this.degree;
    const n = this.nKnots;
    const full = new Float64Array(n + 2 * p);
    full.set(base, p);
    if (this.extrapolation === "periodic") {
      const period = (base[n - 1] as number) - (base[0] as number);
      for (let k = 0; k < p; k++) {
        full[k] = (base[n - 1 - p + k] as number) - period;
        full[p + n + k] = (base[1 + k] as number) + period;
      }
    } else {
      full.fill(base[0] as number, 0, p);
      full.fill(base[n - 1] as number, p + n, n + 2 * p);
    }
    return full;
  }

  /**
   * Evaluate the spline basis of every feature.
   *
   * @param X - Data of shape (n_samples, n_features)
   * @returns Float64 tensor of shape (n_samples, nFeaturesOut)
   * @throws {NotFittedError} If the transformer has not been fitted
   * @throws {InvalidParameterError} If the number of features differs from the fit
   * @throws {DataValidationError} If X contains NaN or Infinity, or, with
   * `extrapolation: "error"`, values outside the fitted range
   */
  transform(X: Tensor): Tensor {
    if (!this.knots_) {
      throw new NotFittedError("SplineTransformer must be fitted before transform");
    }
    const { data, nSamples, nFeatures } = readMatrix(X);
    if (nFeatures !== this.nFeaturesIn) {
      throw new InvalidParameterError(
        `Expected ${this.nFeaturesIn} features, got ${nFeatures}`,
        "nFeatures",
        nFeatures
      );
    }

    const p = this.degree;
    const periodic = this.extrapolation === "periodic";
    const nPeriodic = this.nKnots - 1;
    // Without the bias column a non-periodic basis drops its first function and
    // a periodic one its last (the columns of both sum to one).
    const dropped = this.includeBias ? 0 : 1;
    const startBasis = periodic ? 0 : dropped;
    const nOut = this.basisPerFeature - dropped;
    const totalFeatures = nFeatures * nOut;
    const result = new Float64Array(nSamples * totalFeatures);

    const N = new Float64Array(p + 1);
    const N1 = new Float64Array(p + 1);
    const left = new Float64Array(p + 1);
    const right = new Float64Array(p + 1);

    for (let f = 0; f < nFeatures; f++) {
      const { full, constant } = this.knots_[f] as FeatureKnots;
      if (constant) {
        // All-zero columns; with extrapolation "error" a differing value still throws.
        if (this.extrapolation === "error") {
          const only = full[p] as number;
          for (let i = 0; i < nSamples; i++) {
            const x = data[i * nFeatures + f] as number;
            if (x !== only) {
              throw new DataValidationError(
                `X contains a value outside the fitted range [${only}, ${only}] at feature ${f}, sample ${i}: ${x}`
              );
            }
          }
        }
        continue;
      }
      const nBasisFull = full.length - p - 1;
      const lo = full[p] as number;
      const hi = full[nBasisFull] as number;

      for (let i = 0; i < nSamples; i++) {
        let x = data[i * nFeatures + f] as number;
        // Linear extrapolation evaluates the basis at the boundary and adds its
        // derivative times the distance moved beyond it.
        let slopeOffset = 0;
        let useSlope = false;
        if (periodic) {
          const period = hi - lo;
          x = lo + ((((x - lo) % period) + period) % period);
        } else if (x < lo || x > hi) {
          if (this.extrapolation === "constant") {
            x = x < lo ? lo : hi;
          } else if (this.extrapolation === "error") {
            throw new DataValidationError(
              `X contains a value outside the fitted range [${lo}, ${hi}] at feature ${f}, sample ${i}: ${x}`
            );
          } else if (this.extrapolation === "linear") {
            const boundary = x < lo ? lo : hi;
            slopeOffset = x - boundary;
            x = boundary;
            useSlope = p > 0;
          }
        }

        const span = findSpan(full, p, nBasisFull, x, lo, hi);
        basisFunctions(full, span, x, p, N, left, right);
        if (useSlope) {
          basisFunctions(full, span, x, p - 1, N1, left, right);
        }

        const rowBase = i * totalFeatures + f * nOut;
        for (let r = 0; r <= p; r++) {
          const j = span - p + r;
          let value = N[r] as number;
          if (useSlope) {
            let d = 0;
            if (j >= span - p + 1) {
              d +=
                (N1[j - span + p - 1] as number) / ((full[j + p] as number) - (full[j] as number));
            }
            if (j <= span - 1) {
              d -=
                (N1[j - span + p] as number) /
                ((full[j + p + 1] as number) - (full[j + 1] as number));
            }
            value += p * d * slopeOffset;
          }
          const column = periodic && j >= nPeriodic ? j - nPeriodic : j;
          if (column >= startBasis && column - startBasis < nOut) {
            const target = rowBase + column - startBasis;
            result[target] = (result[target] as number) + value;
          }
        }
      }
    }

    return TensorClass.fromTypedArray({
      data: result,
      shape: [nSamples, totalFeatures],
      dtype: "float64",
      device: X.device,
    });
  }

  /**
   * Fit to X, then transform it.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (present for API compatibility)
   * @returns Spline features
   */
  fitTransform(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    return this.transform(X);
  }

  /**
   * Spline features cannot be inverted, so this always throws.
   *
   * @throws {NotImplementedError} Always
   */
  inverseTransform(_X: Tensor): Tensor {
    throw new NotImplementedError("SplineTransformer does not support inverseTransform");
  }

  /**
   * Number of output columns.
   *
   * @throws {NotFittedError} If the transformer has not been fitted
   */
  get nFeaturesOut(): number {
    if (!this.knots_) {
      throw new NotFittedError("SplineTransformer must be fitted to get nFeaturesOut");
    }
    const startBasis = this.includeBias ? 0 : 1;
    return this.nFeaturesIn * (this.basisPerFeature - startBasis);
  }

  /** Knots placed on each feature during fitting (one array of `nKnots` values per feature), or `undefined` if unfitted. */
  get knotPositions(): number[][] | undefined {
    return this.knots_?.map((k) => Array.from(k.base));
  }

  /** Constructor options of this transformer. */
  getParams(): Record<string, unknown> {
    return {
      nKnots: this.nKnots,
      degree: this.degree,
      extrapolation: this.extrapolation,
      includeBias: this.includeBias,
      knots: this.knotMode,
    };
  }

  /**
   * Change options. Any change discards the fitted knots.
   *
   * @throws {InvalidParameterError} If the resulting options are invalid; the transformer is left unchanged
   */
  setParams(params: Record<string, unknown>): this {
    const previous = this.getParams() as Required<SplineOptions>;
    const next = {
      nKnots: params["nKnots"] ?? previous.nKnots,
      degree: params["degree"] ?? previous.degree,
      extrapolation: params["extrapolation"] ?? previous.extrapolation,
      includeBias: params["includeBias"] ?? previous.includeBias,
      knots: params["knots"] ?? previous.knots,
    };
    this.nKnots = next.nKnots as number;
    this.degree = next.degree as number;
    this.extrapolation = next.extrapolation as SplineExtrapolation;
    this.includeBias = next.includeBias as boolean;
    this.knotMode = next.knots as SplineKnots;
    try {
      this.validateParams();
    } catch (error) {
      this.nKnots = previous.nKnots;
      this.degree = previous.degree;
      this.extrapolation = previous.extrapolation;
      this.includeBias = previous.includeBias;
      this.knotMode = previous.knots;
      throw error;
    }
    const changed = (Object.keys(next) as (keyof typeof next)[]).some(
      (key) => next[key] !== previous[key]
    );
    if (changed) {
      this.knots_ = undefined;
      this.nFeaturesIn = 0;
    }
    return this;
  }

  /** Create an unfitted transformer with the same options. */
  clone(): SplineTransformer {
    return new SplineTransformer({
      nKnots: this.nKnots,
      degree: this.degree,
      extrapolation: this.extrapolation,
      includeBias: this.includeBias,
      knots: this.knotMode,
    });
  }
}

/** Read a 2-D real numeric tensor into a dense row-major Float64Array, rejecting NaN and Infinity. */
function readMatrix(X: Tensor): { data: Float64Array; nSamples: number; nFeatures: number } {
  assert2D(X, "X");
  assertNumericTensor(X, "X");
  const [nSamples, nFeatures] = getShape2D(X);
  const [s0, s1] = getStrides2D(X);
  const src = X.data;
  if (Array.isArray(src)) {
    throw new DTypeError("X must be numeric");
  }
  const out = new Float64Array(nSamples * nFeatures);
  let pos = 0;
  for (let i = 0; i < nSamples; i++) {
    let idx = X.offset + i * s0;
    for (let j = 0; j < nFeatures; j++) {
      const v = Number(src[idx]);
      idx += s1;
      if (!Number.isFinite(v)) {
        throw new DataValidationError(`X contains NaN or Infinity at index ${pos}`);
      }
      out[pos++] = v;
    }
  }
  return { data: out, nSamples, nFeatures };
}

/** Linear-interpolated quantile `q` in [0, 1] of an ascending sorted array (as `numpy.percentile`). */
function quantile(sorted: Float64Array, q: number): number {
  const n = sorted.length;
  const position = q * (n - 1);
  const lower = Math.floor(position);
  if (lower >= n - 1) return sorted[n - 1] as number;
  const a = sorted[lower] as number;
  const b = sorted[lower + 1] as number;
  const t = position - lower;
  const d = b - a;
  return t >= 0.5 ? b - d * (1 - t) : a + d * t;
}

/**
 * Index `i` of the knot interval [t_i, t_{i+1}) whose polynomial piece is used
 * for `x`, restricted to the interior of the knot vector. Points at or beyond
 * the ends use the last / first non-empty interval, which makes the basis
 * right-continuous at the upper boundary and extendable outside the range.
 */
function findSpan(
  knots: Float64Array,
  degree: number,
  nBasis: number,
  x: number,
  lo: number,
  hi: number
): number {
  let i: number;
  if (x >= hi) {
    i = nBasis - 1;
    while (i > degree && knots[i] === knots[i + 1]) i--;
    return i;
  }
  if (x < lo) {
    i = degree;
    while (i < nBasis - 1 && knots[i] === knots[i + 1]) i++;
    return i;
  }
  let low = degree;
  let high = nBasis - 1;
  while (low < high) {
    const mid = (low + high + 1) >>> 1;
    if ((knots[mid] as number) <= x) low = mid;
    else high = mid - 1;
  }
  return low;
}

/**
 * Values of the `degree + 1` B-spline basis functions of the given degree that
 * are non-zero on knot interval `span` (Cox-de Boor recursion, The NURBS Book
 * algorithm A2.2). `out[r]` is the basis function with index `span - degree + r`.
 */
function basisFunctions(
  knots: Float64Array,
  span: number,
  x: number,
  degree: number,
  out: Float64Array,
  left: Float64Array,
  right: Float64Array
): void {
  out[0] = 1;
  for (let j = 1; j <= degree; j++) {
    left[j] = x - (knots[span + 1 - j] as number);
    right[j] = (knots[span + j] as number) - x;
    let saved = 0;
    for (let r = 0; r < j; r++) {
      const temp = (out[r] as number) / ((right[r + 1] as number) + (left[j - r] as number));
      out[r] = saved + (right[r + 1] as number) * temp;
      saved = (left[j - r] as number) * temp;
    }
    out[j] = saved;
  }
}
