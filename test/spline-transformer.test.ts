import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { SplineTransformer } from "../src/preprocess";

describe("SplineTransformer", () => {
  const X = tensor([[0], [0.25], [0.5], [0.75], [1.0]]);

  it("fitTransform produces correct output shape", () => {
    const st = new SplineTransformer({ nKnots: 5, degree: 3 });
    const Xt = st.fitTransform(X);
    // nBasis = nKnots + degree - 1 = 5 + 3 - 1 = 7
    expect(Xt.shape[0]).toBe(5);
    expect(Xt.shape[1]).toBe(7);
  });

  it("includeBias=false reduces output features by 1", () => {
    const st = new SplineTransformer({ nKnots: 5, degree: 3, includeBias: false });
    const Xt = st.fitTransform(X);
    // nBasis = 7 - 1 = 6
    expect(Xt.shape[0]).toBe(5);
    expect(Xt.shape[1]).toBe(6);
  });

  it("basis functions sum to ~1 (partition of unity)", () => {
    const st = new SplineTransformer({ nKnots: 5, degree: 3 });
    const Xt = st.fitTransform(X);
    const nBasis = Xt.shape[1] ?? 0;
    for (let i = 0; i < 5; i++) {
      let sum = 0;
      for (let j = 0; j < nBasis; j++) {
        sum += Number(Xt.data[Xt.offset + i * nBasis + j]);
      }
      expect(sum).toBeCloseTo(1.0, 5);
    }
  });

  it("all basis values are non-negative", () => {
    const st = new SplineTransformer({ nKnots: 5, degree: 3 });
    const Xt = st.fitTransform(X);
    for (let i = 0; i < Xt.size; i++) {
      expect(Number(Xt.data[Xt.offset + i])).toBeGreaterThanOrEqual(-1e-10);
    }
  });

  it("multivariate input works", () => {
    const Xm = tensor([
      [0, 0],
      [0.5, 0.5],
      [1, 1],
    ]);
    const st = new SplineTransformer({ nKnots: 3, degree: 2 });
    const Xt = st.fitTransform(Xm);
    // nBasis per feature = 3 + 2 - 1 = 4, total = 2 * 4 = 8
    expect(Xt.shape[0]).toBe(3);
    expect(Xt.shape[1]).toBe(8);
  });

  it("degree=1 produces linear splines", () => {
    const st = new SplineTransformer({ nKnots: 3, degree: 1 });
    const Xt = st.fitTransform(X);
    // nBasis = 3 + 1 - 1 = 3
    expect(Xt.shape[1]).toBe(3);
  });

  it("degree=0 produces piecewise constant basis", () => {
    const st = new SplineTransformer({ nKnots: 4, degree: 0 });
    const Xt = st.fitTransform(X);
    // nBasis = 4 + 0 - 1 = 3
    expect(Xt.shape[1]).toBe(3);
  });

  it("nFeaturesOut reports correct value", () => {
    const st = new SplineTransformer({ nKnots: 5, degree: 3 });
    st.fit(X);
    expect(st.nFeaturesOut).toBe(7);
  });

  it("nFeaturesOut with includeBias=false", () => {
    const st = new SplineTransformer({ nKnots: 5, degree: 3, includeBias: false });
    st.fit(X);
    expect(st.nFeaturesOut).toBe(6);
  });

  it("throws when not fitted", () => {
    const st = new SplineTransformer();
    expect(() => st.transform(X)).toThrow();
    expect(() => st.nFeaturesOut).toThrow();
  });

  it("throws for mismatched features in transform", () => {
    const st = new SplineTransformer({ nKnots: 3, degree: 2 });
    st.fit(X);
    const X2 = tensor([
      [0, 0],
      [1, 1],
    ]);
    expect(() => st.transform(X2)).toThrow();
  });

  it("throws for invalid nKnots", () => {
    expect(() => new SplineTransformer({ nKnots: 1 })).toThrow();
    expect(() => new SplineTransformer({ nKnots: 0 })).toThrow();
  });

  it("throws for invalid degree", () => {
    expect(() => new SplineTransformer({ degree: -1 })).toThrow();
  });

  it("getParams returns options", () => {
    const st = new SplineTransformer({ nKnots: 6, degree: 2 });
    const params = st.getParams();
    expect(params.nKnots).toBe(6);
    expect(params.degree).toBe(2);
    expect(params.extrapolation).toBe("constant");
    expect(params.includeBias).toBe(true);
  });

  it("inverseTransform throws", () => {
    const st = new SplineTransformer();
    expect(() => st.inverseTransform(X)).toThrow();
  });

  it("extrapolation=constant clamps out-of-range values", () => {
    const Xtrain = tensor([[0], [1], [2], [3], [4]]);
    const st = new SplineTransformer({ nKnots: 3, degree: 2, extrapolation: "constant" });
    st.fit(Xtrain);
    const Xtest = tensor([[-1], [5]]);
    const Xt = st.transform(Xtest);
    // Should not produce NaN
    for (let i = 0; i < Xt.size; i++) {
      expect(Number.isFinite(Number(Xt.data[Xt.offset + i]))).toBe(true);
    }
  });
});
