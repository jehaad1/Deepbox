import { describe, expect, it } from "vitest";
import { tensor, zeros } from "../src/ndarray";
import {
  Normalizer,
  QuantileTransformer,
  RobustScaler,
  StandardScaler,
} from "../src/preprocess/scalers";

import { toNumberMatrix } from "./preprocess-test-helpers";

describe("preprocess scalers additional branch coverage", () => {
  it("QuantileTransformer handles single-sample fit/transform/inverse", () => {
    const X = tensor([[5]]);
    const qt = new QuantileTransformer({ nQuantiles: 10, outputDistribution: "normal" });
    qt.fit(X);

    const transformed = qt.transform(X);
    expect(transformed.shape).toEqual([1, 1]);
    const inverted = qt.inverseTransform(transformed);
    expect(inverted.shape).toEqual([1, 1]);
  });

  it("QuantileTransformer handles empty transforms", () => {
    const X = tensor([[1], [2]]);
    const qt = new QuantileTransformer({ outputDistribution: "uniform" });
    qt.fit(X);

    const empty = zeros([0, 1]);
    const transformed = qt.transform(empty);
    expect(transformed.shape).toEqual([0, 1]);
    const inverted = qt.inverseTransform(empty);
    expect(inverted.shape).toEqual([0, 1]);
  });

  it("RobustScaler covers single-sample quantiles with unitVariance", () => {
    const X = tensor([[1, 2]]);
    const scaler = new RobustScaler({ unitVariance: true, quantileRange: [0.1, 99.9] });
    scaler.fit(X);
    const out = scaler.transform(X);
    expect(out.shape).toEqual([1, 2]);
  });

  it("StandardScaler rejects string dtype", () => {
    const scaler = new StandardScaler();
    expect(() => scaler.fit(tensor([["a"]]))).toThrow(/numeric/i);
  });

  it("Normalizer rejects non-2D input", () => {
    const norm = new Normalizer();
    expect(() => norm.transform(tensor([1, 2, 3]))).toThrow(/2D/);
  });

  it("RobustScaler with unitVariance keeps finite outputs", () => {
    const X = tensor([[0], [1], [2]]);
    const scaler = new RobustScaler({ unitVariance: true, quantileRange: [10, 90] });
    const out = toNumberMatrix(scaler.fitTransform(X), "scaled");
    expect(out.length).toBe(3);
  });
});
