/**
 * KBinsDiscretizer: bin continuous features into discrete intervals.
 *
 * @module preprocess/discretizer
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError, ShapeError } from "../core";
import { type Tensor, Tensor as TensorClass } from "../ndarray";
import { assert2D, assertNumericTensor, getShape2D, getStrides2D } from "./_internal";

type Strategy = "uniform" | "quantile";

/**
 * Bin continuous data into intervals.
 *
 * Supports uniform (equal-width) and quantile (equal-frequency) strategies.
 */
export class KBinsDiscretizer {
  private readonly nBins: number;
  private readonly strategy: Strategy;
  private binEdges_: number[][] | null = null;

  constructor(options: { nBins?: number; strategy?: Strategy } = {}) {
    this.nBins = options.nBins ?? 5;
    this.strategy = options.strategy ?? "quantile";
    if (this.nBins < 2) {
      throw new InvalidParameterError("nBins must be >= 2", "nBins", this.nBins);
    }
    if (this.strategy !== "uniform" && this.strategy !== "quantile") {
      throw new InvalidParameterError(
        "strategy must be 'uniform' or 'quantile'",
        "strategy",
        this.strategy
      );
    }
  }

  get binEdges(): number[][] {
    if (!this.binEdges_) {
      throw new NotFittedError("KBinsDiscretizer is not fitted yet");
    }
    return this.binEdges_;
  }

  fit(X: Tensor): this {
    assertNumericTensor(X, "KBinsDiscretizer.fit");
    assert2D(X, "KBinsDiscretizer.fit");
    const [nSamples, nFeatures] = getShape2D(X);
    const [s0, s1] = getStrides2D(X);

    this.binEdges_ = [];

    // One reusable column buffer; typed-array numeric `.sort()` (comparator-
    // free) replaces the per-feature `number[].push()` + `(a,b)=>a-b` sort.
    const vals = new Float64Array(nSamples);
    const src = X.data;
    const offset = X.offset;
    for (let f = 0; f < nFeatures; f++) {
      for (let i = 0; i < nSamples; i++) {
        vals[i] = Number(src[offset + i * s0 + f * s1]);
      }
      vals.sort();

      if (this.strategy === "uniform") {
        const mn = vals[0] as number;
        const mx = vals[nSamples - 1] as number;
        const edges: number[] = [];
        for (let b = 0; b <= this.nBins; b++) {
          edges.push(mn + (b / this.nBins) * (mx - mn));
        }
        this.binEdges_.push(edges);
      } else {
        // quantile
        const edges: number[] = [];
        for (let b = 0; b <= this.nBins; b++) {
          const q = b / this.nBins;
          const pos = q * (nSamples - 1);
          const lo = Math.floor(pos);
          const hi = Math.ceil(pos);
          const frac = pos - lo;
          edges.push((vals[lo] as number) * (1 - frac) + (vals[hi] as number) * frac);
        }
        this.binEdges_.push(edges);
      }
    }

    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.binEdges_) {
      throw new NotFittedError("KBinsDiscretizer is not fitted yet");
    }
    assertNumericTensor(X, "KBinsDiscretizer.transform");
    assert2D(X, "KBinsDiscretizer.transform");
    const [nSamples, nFeatures] = getShape2D(X);
    const [s0, s1] = getStrides2D(X);

    if (nFeatures !== this.binEdges_.length) {
      throw new ShapeError(`Expected ${this.binEdges_.length} features, got ${nFeatures}`);
    }

    const out = new Float64Array(nSamples * nFeatures);
    const src = X.data;
    const offset = X.offset;
    const maxBin = this.nBins - 1;
    const edgesByFeature = this.binEdges_;
    let pos = 0;
    for (let i = 0; i < nSamples; i++) {
      const rowBase = offset + i * s0;
      for (let f = 0; f < nFeatures; f++) {
        const val = Number(src[rowBase + f * s1]);
        const edges = edgesByFeature[f] as number[];
        // Edges are ascending; the bin is the last b with val > edges[b].
        let bin = 0;
        for (let b = 1; b < edges.length; b++) {
          if (val > (edges[b] as number)) bin = b;
          else break;
        }
        out[pos++] = bin < maxBin ? bin : maxBin;
      }
    }

    return TensorClass.fromTypedArray({
      data: out,
      shape: [nSamples, nFeatures],
      dtype: "float64",
      device: X.device,
    });
  }

  fitTransform(X: Tensor): Tensor {
    return this.fit(X).transform(X);
  }
}
