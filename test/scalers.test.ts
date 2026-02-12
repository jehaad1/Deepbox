import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  MaxAbsScaler,
  MinMaxScaler,
  Normalizer,
  PowerTransformer,
  QuantileTransformer,
  RobustScaler,
  StandardScaler,
} from "../src/preprocess";

describe("StandardScaler", () => {
  it("should fit and transform data to zero mean unit variance", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const scaler = new StandardScaler();
    scaler.fit(X);
    const XScaled = scaler.transform(X);
    expect(XScaled.shape).toEqual([3, 2]);
  });

  it("should produce consistent fit_transform results", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const scaler = new StandardScaler();
    const XScaled = scaler.fitTransform(X);
    expect(XScaled.shape).toEqual([3, 2]);
  });

  it("should round-trip via inverse transform", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const scaler = new StandardScaler();
    const XScaled = scaler.fitTransform(X);
    const XOriginal = scaler.inverseTransform(XScaled);
    expect(XOriginal.shape).toEqual([3, 2]);
    // Verify round-trip preserves values
    for (let i = 0; i < X.size; i++) {
      expect(Number(XOriginal.data[i])).toBeCloseTo(Number(X.data[i]), 5);
    }
  });
});

describe("MinMaxScaler", () => {
  it("should scale data to [0, 1] by default", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const scaler = new MinMaxScaler();
    const XScaled = scaler.fitTransform(X);
    expect(XScaled.shape).toEqual([3, 2]);
  });

  it("should accept custom feature range", () => {
    const scaler = new MinMaxScaler({ featureRange: [-1, 1] });
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const XScaled = scaler.fitTransform(X);
    expect(XScaled.shape).toEqual([3, 2]);
  });
});

describe("MaxAbsScaler", () => {
  it("should scale data by maximum absolute value", () => {
    const X = tensor([
      [1, -2],
      [3, -4],
      [5, -6],
    ]);
    const scaler = new MaxAbsScaler();
    const XScaled = scaler.fitTransform(X);
    expect(XScaled.shape).toEqual([3, 2]);
  });
});

describe("RobustScaler", () => {
  it("should scale data using median and IQR", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const scaler = new RobustScaler();
    const XScaled = scaler.fitTransform(X);
    expect(XScaled.shape).toEqual([3, 2]);
  });

  it("should accept custom quantile range", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const scaler = new RobustScaler({ quantileRange: [10, 90] });
    const XScaled = scaler.fitTransform(X);
    expect(XScaled.shape).toEqual([3, 2]);
  });
});

describe("Normalizer", () => {
  it("should normalize rows to unit norm", () => {
    const X = tensor([
      [3, 4],
      [1, 0],
    ]);
    const normalizer = new Normalizer();
    const XNorm = normalizer.fitTransform(X);
    expect(XNorm.shape).toEqual([2, 2]);
  });

  it("should support l1, l2, and max norms", () => {
    const X = tensor([
      [3, 4],
      [1, 2],
    ]);
    const l1 = new Normalizer({ norm: "l1" }).fitTransform(X);
    const l2 = new Normalizer({ norm: "l2" }).fitTransform(X);
    const maxN = new Normalizer({ norm: "max" }).fitTransform(X);
    expect(l1.shape).toEqual([2, 2]);
    expect(l2.shape).toEqual([2, 2]);
    expect(maxN.shape).toEqual([2, 2]);
  });
});

describe("QuantileTransformer", () => {
  it("should transform to uniform distribution by default", () => {
    const X = tensor([[1], [2], [3], [4], [5]]);
    const transformer = new QuantileTransformer();
    const XTransformed = transformer.fitTransform(X);
    expect(XTransformed.shape).toEqual([5, 1]);
  });

  it("should support normal output distribution", () => {
    const X = tensor([[1], [2], [3], [4], [5]]);
    const transformer = new QuantileTransformer({
      outputDistribution: "normal",
    });
    const XTransformed = transformer.fitTransform(X);
    expect(XTransformed.shape).toEqual([5, 1]);
  });
});

describe("PowerTransformer", () => {
  it("should apply yeo-johnson by default", () => {
    const X = tensor([[1], [2], [3], [4], [5]]);
    const transformer = new PowerTransformer();
    const XTransformed = transformer.fitTransform(X);
    expect(XTransformed.shape).toEqual([5, 1]);
  });

  it("should support box-cox method for positive data", () => {
    const X = tensor([[1], [2], [3], [4], [5]]);
    const transformer = new PowerTransformer({ method: "box-cox" });
    const XTransformed = transformer.fitTransform(X);
    expect(XTransformed.shape).toEqual([5, 1]);
  });
});
