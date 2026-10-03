/**
 * KBinsDiscretizer: bin continuous features into discrete intervals.
 *
 * @module preprocess/discretizer
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../core";
import { type Tensor, Tensor as TensorClass } from "../ndarray";
import { assert2D, assertNumericTensor, getShape2D, getStrides2D } from "./_internal";

type Strategy = "uniform" | "quantile";
type Encode = "ordinal" | "onehot-dense";

/**
 * Bins narrower than this are merged away by the quantile strategy
 * (same threshold as scikit-learn).
 */
const MIN_BIN_WIDTH = 1e-8;

/**
 * Linear interpolation between two sorted neighbours, written the way NumPy
 * does it so that quantile edges agree with `np.percentile` to the last bit.
 */
function lerp(a: number, b: number, t: number): number {
  const diff = b - a;
  return t >= 0.5 ? b - diff * (1 - t) : a + diff * t;
}

function readColumnSorted(
  src: ArrayLike<number | bigint>,
  base: number,
  stride: number,
  out: Float64Array,
  feature: number
): void {
  for (let i = 0; i < out.length; i++) {
    const v = Number(src[base + i * stride]);
    if (!Number.isFinite(v)) {
      throw new DataValidationError(
        `KBinsDiscretizer does not accept NaN or Infinity (feature ${feature}, sample ${i})`
      );
    }
    out[i] = v;
  }
  out.sort();
}

/**
 * Bin continuous data into intervals.
 *
 * Supports uniform (equal-width) and quantile (equal-frequency) strategies.
 * A value that falls exactly on an inner bin edge goes to the upper bin, and
 * values outside the fitted range are clipped into the first or last bin
 * (as in scikit-learn). A constant feature gets a single bin with edges
 * `[-Infinity, Infinity]`, and quantile bins that collapse to zero width are
 * merged, so a feature can end up with fewer bins than requested.
 *
 * @example
 * ```ts
 * import { KBinsDiscretizer } from "deepbox/preprocess";
 * import { tensor } from "deepbox/ndarray";
 *
 * const X = tensor([[0], [1], [2], [3], [4]]);
 * const kbd = new KBinsDiscretizer({ nBins: 2, strategy: "uniform" });
 * kbd.fitTransform(X); // [[0], [0], [1], [1], [1]]
 * kbd.binEdges;        // [[0, 2, 4]]
 * ```
 */
export class KBinsDiscretizer {
  private readonly nBins: number | readonly number[];
  private readonly strategy: Strategy;
  private readonly encode: Encode;
  private binEdges_: number[][] | null = null;

  /**
   * @param options.nBins - Number of bins per feature (default 5), or one count per feature. Must be integers >= 2.
   * @param options.strategy - "quantile" (equal frequency, default) or "uniform" (equal width).
   * @param options.encode - "ordinal" returns one bin index column per feature (default); "onehot-dense" returns a dense one-hot block per feature.
   */
  constructor(
    options: { nBins?: number | readonly number[]; strategy?: Strategy; encode?: Encode } = {}
  ) {
    this.nBins = options.nBins ?? 5;
    this.strategy = options.strategy ?? "quantile";
    this.encode = options.encode ?? "ordinal";
    if (typeof this.nBins !== "number" && !Array.isArray(this.nBins)) {
      throw new InvalidParameterError(
        "nBins must be an integer or an array of integers",
        "nBins",
        this.nBins
      );
    }
    const counts = typeof this.nBins === "number" ? [this.nBins] : this.nBins;
    if (counts.length === 0) {
      throw new InvalidParameterError("nBins must not be an empty array", "nBins", this.nBins);
    }
    for (const n of counts) {
      if (!Number.isInteger(n) || n < 2) {
        throw new InvalidParameterError("nBins must be an integer >= 2", "nBins", n);
      }
    }
    if (this.strategy !== "uniform" && this.strategy !== "quantile") {
      throw new InvalidParameterError(
        "strategy must be 'uniform' or 'quantile'",
        "strategy",
        this.strategy
      );
    }
    if (this.encode !== "ordinal" && this.encode !== "onehot-dense") {
      throw new InvalidParameterError(
        "encode must be 'ordinal' or 'onehot-dense'",
        "encode",
        this.encode
      );
    }
  }

  /** Learned bin edges, one ascending array per feature (a copy). */
  get binEdges(): number[][] {
    if (!this.binEdges_) {
      throw new NotFittedError("KBinsDiscretizer is not fitted yet");
    }
    return this.binEdges_.map((e) => e.slice());
  }

  /** Number of bins actually used per feature after fitting. */
  get nBinsPerFeature(): number[] {
    if (!this.binEdges_) {
      throw new NotFittedError("KBinsDiscretizer is not fitted yet");
    }
    return this.binEdges_.map((e) => e.length - 1);
  }

  /**
   * Learn the bin edges of every feature.
   *
   * A failed fit leaves a previously fitted discretizer unchanged.
   *
   * @param X - Data of shape [nSamples, nFeatures]; values must be finite
   * @returns this
   * @throws {InvalidParameterError} If X is empty or `nBins` has the wrong length
   * @throws {DataValidationError} If X contains NaN or Infinity
   */
  fit(X: Tensor): this {
    assertNumericTensor(X, "KBinsDiscretizer.fit");
    assert2D(X, "KBinsDiscretizer.fit");
    const [nSamples, nFeatures] = getShape2D(X);
    if (nSamples === 0 || nFeatures === 0) {
      throw new InvalidParameterError("Cannot fit KBinsDiscretizer on empty array", "X");
    }
    if (typeof this.nBins !== "number" && this.nBins.length !== nFeatures) {
      throw new InvalidParameterError(
        `nBins has ${this.nBins.length} entries but X has ${nFeatures} features`,
        "nBins",
        this.nBins
      );
    }
    const [s0, s1] = getStrides2D(X);

    const allEdges: number[][] = [];

    // One reusable column buffer; typed-array numeric `.sort()` (comparator-free)
    // keeps the per-feature sort cheap.
    const vals = new Float64Array(nSamples);
    const src = X.data as ArrayLike<number | bigint>;
    const offset = X.offset;
    for (let f = 0; f < nFeatures; f++) {
      readColumnSorted(src, offset + f * s1, s0, vals, f);
      const nBins = typeof this.nBins === "number" ? this.nBins : (this.nBins[f] as number);
      const mn = vals[0] as number;
      const mx = vals[nSamples - 1] as number;

      if (mn === mx) {
        // Constant feature: one bin that covers the whole real line.
        allEdges.push([Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY]);
        continue;
      }

      let edges: number[] = [];
      if (this.strategy === "uniform") {
        // Same construction as np.linspace(min, max, nBins + 1); the last edge is
        // pinned to max so rounding cannot leave the largest value outside.
        const span = mx - mn;
        const step = Number.isFinite(span) ? span / nBins : mx / nBins - mn / nBins;
        for (let b = 0; b < nBins; b++) {
          edges.push(mn + b * step);
        }
        edges.push(mx);
      } else {
        // Percentile levels as in np.linspace(0, 100, nBins + 1) / 100, so the
        // interpolation position is bit-identical to np.percentile's. A value that
        // sits exactly on an edge must land in the same bin as in scikit-learn.
        const levelStep = 100 / nBins;
        for (let b = 0; b <= nBins; b++) {
          const q = b === nBins ? 1 : (b * levelStep) / 100;
          const pos = (nSamples - 1) * q;
          const lo = Math.floor(pos);
          const hi = Math.min(lo + 1, nSamples - 1);
          edges.push(lerp(vals[lo] as number, vals[hi] as number, pos - lo));
        }
        // Drop bins that collapsed (heavy ties) so bin indices stay contiguous.
        // Each edge is compared with its predecessor in the full list, as scikit-learn does.
        const kept: number[] = [edges[0] as number];
        for (let b = 1; b < edges.length; b++) {
          if ((edges[b] as number) - (edges[b - 1] as number) > MIN_BIN_WIDTH) {
            kept.push(edges[b] as number);
          }
        }
        edges = kept.length >= 2 ? kept : [Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY];
      }
      allEdges.push(edges);
    }

    this.binEdges_ = allEdges;
    return this;
  }

  /**
   * Replace every value by the index of its bin (or by a one-hot block when
   * `encode` is "onehot-dense"). Values outside the fitted range go to the
   * first or last bin.
   *
   * @param X - Data with the same number of features as in `fit`; values must be finite
   * @returns Float64 tensor of shape [nSamples, nFeatures] (ordinal) or [nSamples, total bins] (one-hot)
   * @throws {NotFittedError} If the discretizer is not fitted
   * @throws {DataValidationError} If X contains NaN or Infinity
   */
  transform(X: Tensor): Tensor {
    if (!this.binEdges_) {
      throw new NotFittedError("KBinsDiscretizer is not fitted yet");
    }
    assertNumericTensor(X, "KBinsDiscretizer.transform");
    assert2D(X, "KBinsDiscretizer.transform");
    const [nSamples, nFeatures] = getShape2D(X);
    const [s0, s1] = getStrides2D(X);
    const edgesByFeature = this.binEdges_;

    if (nFeatures !== edgesByFeature.length) {
      throw new ShapeError(`Expected ${edgesByFeature.length} features, got ${nFeatures}`);
    }

    const onehot = this.encode === "onehot-dense";
    const colOffsets = new Array<number>(nFeatures);
    let totalCols = 0;
    for (let f = 0; f < nFeatures; f++) {
      colOffsets[f] = totalCols;
      totalCols += onehot ? (edgesByFeature[f] as number[]).length - 1 : 1;
    }

    const out = new Float64Array(nSamples * totalCols);
    const src = X.data as ArrayLike<number | bigint>;
    const offset = X.offset;
    for (let i = 0; i < nSamples; i++) {
      const rowBase = offset + i * s0;
      for (let f = 0; f < nFeatures; f++) {
        const val = Number(src[rowBase + f * s1]);
        if (!Number.isFinite(val)) {
          throw new DataValidationError(
            `KBinsDiscretizer does not accept NaN or Infinity (feature ${f}, sample ${i})`
          );
        }
        const edges = edgesByFeature[f] as number[];
        // bin = number of inner edges <= val (a value on an edge goes up).
        // Inner edges are edges[1 .. length - 2]; binary search for the first one > val.
        let lo = 1;
        let hi = edges.length - 1;
        while (lo < hi) {
          const mid = (lo + hi) >>> 1;
          if ((edges[mid] as number) <= val) lo = mid + 1;
          else hi = mid;
        }
        const bin = lo - 1;
        if (onehot) {
          out[i * totalCols + (colOffsets[f] as number) + bin] = 1;
        } else {
          out[i * totalCols + f] = bin;
        }
      }
    }

    return TensorClass.fromTypedArray({
      data: out,
      shape: [nSamples, totalCols],
      dtype: "float64",
      device: X.device,
    });
  }

  /** Fit to X and return its binned representation. */
  fitTransform(X: Tensor): Tensor {
    return this.fit(X).transform(X);
  }

  /**
   * Map bin indices (or one-hot blocks) back to the center of each bin.
   *
   * The result is only an approximation of the original data. A constant
   * feature has infinite edges, so its bin center is NaN, as in scikit-learn.
   *
   * @param Xt - Output of `transform` (ordinal indices or one-hot columns)
   * @returns Tensor of bin centers with one column per feature
   */
  inverseTransform(Xt: Tensor): Tensor {
    if (!this.binEdges_) {
      throw new NotFittedError("KBinsDiscretizer is not fitted yet");
    }
    assertNumericTensor(Xt, "KBinsDiscretizer.inverseTransform");
    assert2D(Xt, "KBinsDiscretizer.inverseTransform");
    const [nSamples, nCols] = getShape2D(Xt);
    const [s0, s1] = getStrides2D(Xt);
    const edgesByFeature = this.binEdges_;
    const nFeatures = edgesByFeature.length;
    const onehot = this.encode === "onehot-dense";

    let expectedCols = 0;
    for (const e of edgesByFeature) expectedCols += onehot ? e.length - 1 : 1;
    if (nCols !== expectedCols) {
      throw new ShapeError(`Expected ${expectedCols} columns, got ${nCols}`);
    }

    const out = new Float64Array(nSamples * nFeatures);
    const src = Xt.data as ArrayLike<number | bigint>;
    const offset = Xt.offset;
    for (let i = 0; i < nSamples; i++) {
      let col = 0;
      for (let f = 0; f < nFeatures; f++) {
        const edges = edgesByFeature[f] as number[];
        const nb = edges.length - 1;
        let bin: number;
        if (onehot) {
          bin = 0;
          let best = Number.NEGATIVE_INFINITY;
          for (let k = 0; k < nb; k++) {
            const v = Number(src[offset + i * s0 + (col + k) * s1]);
            if (v > best) {
              best = v;
              bin = k;
            }
          }
          col += nb;
          if (best === Number.NEGATIVE_INFINITY) {
            throw new InvalidParameterError(
              `No valid one-hot column found for feature ${f} in sample ${i}`,
              "Xt"
            );
          }
        } else {
          bin = Number(src[offset + i * s0 + col * s1]);
          col += 1;
          if (!Number.isInteger(bin) || bin < 0 || bin >= nb) {
            throw new InvalidParameterError(
              `Invalid bin index ${bin} for feature ${f}. Must be an integer in [0, ${nb - 1}]`,
              "Xt",
              bin
            );
          }
        }
        out[i * nFeatures + f] = ((edges[bin] as number) + (edges[bin + 1] as number)) * 0.5;
      }
    }

    return TensorClass.fromTypedArray({
      data: out,
      shape: [nSamples, nFeatures],
      dtype: "float64",
      device: Xt.device,
    });
  }
}
