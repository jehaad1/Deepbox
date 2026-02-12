import { describe, expect, it } from "vitest";
import { tensor, transpose } from "../src/ndarray";
import {
  MaxAbsScaler,
  MinMaxScaler,
  Normalizer,
  PowerTransformer,
  QuantileTransformer,
  RobustScaler,
  StandardScaler,
} from "../src/preprocess/scalers";
import { toNumberMatrix } from "./preprocess-test-helpers";

describe("deepbox/preprocess - Scalers Extra", () => {
  it("StandardScaler fit/transform/inverse", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const scaler = new StandardScaler();
    const XScaled = scaler.fitTransform(X);
    expect(XScaled.shape).toEqual([3, 2]);
    const XInv = scaler.inverseTransform(XScaled);
    expect(XInv.toArray()).toEqual(X.toArray());
  });

  it("MinMaxScaler scales to feature range", () => {
    const X = tensor([
      [0, 10],
      [5, 20],
      [10, 30],
    ]);
    const scaler = new MinMaxScaler({ featureRange: [-1, 1] });
    const XScaled = scaler.fitTransform(X);
    const arr = toNumberMatrix(XScaled, "XScaled");
    expect(arr[0]?.[0]).toBeCloseTo(-1, 6);
    expect(arr[2]?.[0]).toBeCloseTo(1, 6);
    const XInv = scaler.inverseTransform(XScaled);
    expect(XInv.toArray()).toEqual(X.toArray());
  });

  it("MinMaxScaler clips out-of-range values when clip=true", () => {
    const X = tensor([[0], [10]]);
    const scaler = new MinMaxScaler({ featureRange: [0, 1], clip: true });
    scaler.fit(X);
    const transformed = scaler.transform(tensor([[-5], [5], [15]]));
    const arr = toNumberMatrix(transformed, "transformed");
    expect(arr[0]?.[0]).toBeCloseTo(0, 6);
    expect(arr[1]?.[0]).toBeCloseTo(0.5, 6);
    expect(arr[2]?.[0]).toBeCloseTo(1, 6);
  });

  it("MaxAbsScaler scales by max absolute values", () => {
    const X = tensor([
      [-2, 4],
      [1, -8],
    ]);
    const scaler = new MaxAbsScaler();
    const XScaled = scaler.fitTransform(X);
    const arr = toNumberMatrix(XScaled, "XScaled");
    expect(arr[0]?.[0]).toBeCloseTo(-1, 6);
    expect(arr[1]?.[1]).toBeCloseTo(-1, 6);
    const XInv = scaler.inverseTransform(XScaled);
    expect(XInv.toArray()).toEqual(X.toArray());
  });

  it("RobustScaler handles outliers", () => {
    const X = tensor([[1], [2], [100], [3], [4]]);
    const scaler = new RobustScaler();
    const XScaled = scaler.fitTransform(X);
    expect(XScaled.shape).toEqual([5, 1]);
    const XInv = scaler.inverseTransform(XScaled);
    expect(XInv.toArray()).toEqual(X.toArray());
  });

  it("RobustScaler unitVariance rescales to normal-variance units", () => {
    const X = tensor([[0], [1], [2], [3], [4]]);
    const scaler = new RobustScaler({
      quantileRange: [25, 75],
      unitVariance: true,
    });
    const out = toNumberMatrix(scaler.fitTransform(X), "scaled");
    expect(out[0]?.[0]).toBeCloseTo(-1.3489795, 6);
    expect(out[4]?.[0]).toBeCloseTo(1.3489795, 6);
  });

  it("Normalizer supports l1, l2, and max", () => {
    const X = tensor([
      [3, 4],
      [1, -1],
    ]);
    const l2 = new Normalizer({ norm: "l2" }).transform(X);
    expect(l2.shape).toEqual([2, 2]);
    const l1 = new Normalizer({ norm: "l1" }).transform(X);
    expect(l1.shape).toEqual([2, 2]);
    const max = new Normalizer({ norm: "max" }).transform(X);
    expect(max.shape).toEqual([2, 2]);
  });

  it("QuantileTransformer maps to uniform and normal distributions", () => {
    const X = tensor([[1], [2], [3], [4]]);
    const qtUniform = new QuantileTransformer({
      outputDistribution: "uniform",
    });
    const u = qtUniform.fitTransform(X);
    expect(u.shape).toEqual([4, 1]);

    const qtNormal = new QuantileTransformer({ outputDistribution: "normal" });
    const n = qtNormal.fitTransform(X);
    expect(n.shape).toEqual([4, 1]);
  });

  it("QuantileTransformer subsample is deterministic with randomState", () => {
    const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8]]);
    const qtA = new QuantileTransformer({
      nQuantiles: 4,
      subsample: 4,
      randomState: 7,
    });
    const qtB = new QuantileTransformer({
      nQuantiles: 4,
      subsample: 4,
      randomState: 7,
    });
    const outA = toNumberMatrix(qtA.fitTransform(X), "outA");
    const outB = toNumberMatrix(qtB.fitTransform(X), "outB");
    expect(outA).toEqual(outB);
  });

  it("PowerTransformer supports box-cox and yeo-johnson", () => {
    const X = tensor([[1], [2], [3]]);
    const box = new PowerTransformer({ method: "box-cox" });
    const xb = box.fitTransform(X);
    expect(xb.shape).toEqual([3, 1]);

    const X2 = tensor([[-1], [0], [1]]);
    const yj = new PowerTransformer({ method: "yeo-johnson" });
    const xy = yj.fitTransform(X2);
    expect(xy.shape).toEqual([3, 1]);
  });

  it("PowerTransformer box-cox rejects non-positive values", () => {
    const X = tensor([[0], [1]]);
    const box = new PowerTransformer({ method: "box-cox" });
    expect(() => box.fit(X)).toThrow();
  });

  it("StandardScaler computes std around mean when withMean=false", () => {
    const X = tensor([[1], [3], [5]]);
    const scaler = new StandardScaler({ withMean: false, withStd: true });
    const out = toNumberMatrix(scaler.fitTransform(X), "XScaled");
    const expectedStd = Math.sqrt(8 / 3);
    expect(out[0]?.[0]).toBeCloseTo(1 / expectedStd, 6);
    expect(out[1]?.[0]).toBeCloseTo(3 / expectedStd, 6);
    expect(out[2]?.[0]).toBeCloseTo(5 / expectedStd, 6);
  });

  it("MinMaxScaler maps constant features to minRange", () => {
    const X = tensor([
      [1, 10],
      [1, 20],
      [1, 30],
    ]);
    const scaler = new MinMaxScaler({ featureRange: [-1, 1] });
    const out = toNumberMatrix(scaler.fitTransform(X), "XScaled");
    expect(out[0]?.[0]).toBeCloseTo(-1, 6);
    expect(out[1]?.[0]).toBeCloseTo(-1, 6);
    expect(out[2]?.[0]).toBeCloseTo(-1, 6);
  });

  it("RobustScaler uses interpolated quantiles", () => {
    const X = tensor([[1], [2], [3], [4]]);
    const scaler = new RobustScaler({ quantileRange: [25, 75] });
    const out = toNumberMatrix(scaler.fitTransform(X), "XScaled");
    expect(out[0]?.[0]).toBeCloseTo(-1, 6);
    expect(out[3]?.[0]).toBeCloseTo(1, 6);
  });

  it("QuantileTransformer inverseTransform recovers original values", () => {
    const X = tensor([[1], [2], [3], [4]]);
    const qt = new QuantileTransformer({ outputDistribution: "uniform" });
    const transformed = qt.fitTransform(X);
    const inverted = qt.inverseTransform(transformed);
    const arr = toNumberMatrix(inverted, "inverted");
    expect(arr[0]?.[0]).toBeCloseTo(1, 6);
    expect(arr[1]?.[0]).toBeCloseTo(2, 6);
    expect(arr[2]?.[0]).toBeCloseTo(3, 6);
    expect(arr[3]?.[0]).toBeCloseTo(4, 6);
  });

  it("PowerTransformer inverseTransform recovers original values", () => {
    const X = tensor([[1], [2], [3]]);
    const pt = new PowerTransformer({ method: "box-cox" });
    const transformed = pt.fitTransform(X);
    const inverted = pt.inverseTransform(transformed);
    const arr = toNumberMatrix(inverted, "inverted");
    expect(arr[0]?.[0]).toBeCloseTo(1, 6);
    expect(arr[1]?.[0]).toBeCloseTo(2, 6);
    expect(arr[2]?.[0]).toBeCloseTo(3, 6);
  });

  it("PowerTransformer standardize produces zero-mean unit-variance features", () => {
    const X = tensor([[1], [2], [3], [4]]);
    const pt = new PowerTransformer({
      method: "yeo-johnson",
      standardize: true,
    });
    const transformed = toNumberMatrix(pt.fitTransform(X), "transformed");
    const values = transformed.map((row) => row[0] ?? 0);
    const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
    const variance = values.reduce((sum, v) => sum + (v - mean) * (v - mean), 0) / values.length;
    expect(mean).toBeCloseTo(0, 6);
    expect(Math.sqrt(variance)).toBeCloseTo(1, 6);
  });

  it("StandardScaler handles strided (transposed) tensors", () => {
    const X = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const Xt = transpose(X);
    const scaler = new StandardScaler();
    const out = toNumberMatrix(scaler.fitTransform(Xt), "XScaled");
    expect(out[0]?.[0]).toBeCloseTo(-1.224744871, 6);
    expect(out[2]?.[0]).toBeCloseTo(1.224744871, 6);
    expect(out[0]?.[1]).toBeCloseTo(-1.224744871, 6);
    expect(out[2]?.[1]).toBeCloseTo(1.224744871, 6);
  });
});
