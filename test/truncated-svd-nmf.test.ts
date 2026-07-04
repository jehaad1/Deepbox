import { describe, expect, it } from "vitest";
import { NMF, PCA, TruncatedSVD } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("TruncatedSVD", () => {
  const X = tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [2, 3, 4, 5],
    [6, 7, 8, 9],
  ]);

  it("fit and transform reduces dimensions", () => {
    const tsvd = new TruncatedSVD({ nComponents: 2 });
    tsvd.fit(X);
    const Xt = tsvd.transform(X);
    expect(Xt.shape).toEqual([5, 2]);
  });

  it("fitTransform convenience", () => {
    const tsvd = new TruncatedSVD({ nComponents: 2 });
    const Xt = tsvd.fitTransform(X);
    expect(Xt.shape).toEqual([5, 2]);
  });

  it("components shape matches (nComponents x nFeatures)", () => {
    const tsvd = new TruncatedSVD({ nComponents: 3 });
    tsvd.fit(X);
    expect(tsvd.components.shape).toEqual([3, 4]);
  });

  it("singular values are positive and sorted descending", () => {
    const tsvd = new TruncatedSVD({ nComponents: 3 });
    tsvd.fit(X);
    const sv = tsvd.singularValues;
    for (let i = 0; i < sv.size; i++) {
      expect(Number(sv.data[sv.offset + i])).toBeGreaterThan(0);
    }
    for (let i = 1; i < sv.size; i++) {
      expect(Number(sv.data[sv.offset + i - 1])).toBeGreaterThanOrEqual(
        Number(sv.data[sv.offset + i])
      );
    }
  });

  it("explained variance ratio sums to <= 1", () => {
    const tsvd = new TruncatedSVD({ nComponents: 3 });
    tsvd.fit(X);
    const evr = tsvd.explainedVarianceRatio;
    let sum = 0;
    for (let i = 0; i < evr.size; i++) {
      const v = Number(evr.data[evr.offset + i]);
      expect(v).toBeGreaterThanOrEqual(0);
      sum += v;
    }
    expect(sum).toBeLessThanOrEqual(1.0 + 1e-10);
  });

  it("explained variance is positive", () => {
    const tsvd = new TruncatedSVD({ nComponents: 2 });
    tsvd.fit(X);
    const ev = tsvd.explainedVariance;
    for (let i = 0; i < ev.size; i++) {
      expect(Number(ev.data[ev.offset + i])).toBeGreaterThan(0);
    }
  });

  it("inverseTransform approximately reconstructs data", () => {
    const tsvd = new TruncatedSVD({ nComponents: 4 });
    tsvd.fit(X);
    const Xt = tsvd.transform(X);
    const Xr = tsvd.inverseTransform(Xt);
    expect(Xr.shape).toEqual([5, 4]);
    // With all components, reconstruction should be near-perfect
    for (let i = 0; i < X.size; i++) {
      expect(Number(Xr.data[Xr.offset + i])).toBeCloseTo(Number(X.data[X.offset + i]), 3);
    }
  });

  it("does NOT center data (unlike PCA)", () => {
    // TruncatedSVD on constant data should still produce non-zero transform
    const Xconst = tensor([
      [5, 5, 5],
      [5, 5, 5],
      [5, 5, 5],
      [5, 5, 6],
    ]);
    const tsvd = new TruncatedSVD({ nComponents: 2 });
    const Xt = tsvd.fitTransform(Xconst);
    // First component should capture the constant offset
    let firstCompSum = 0;
    for (let i = 0; i < 4; i++) {
      firstCompSum += Math.abs(Number(Xt.data[Xt.offset + i * 2]));
    }
    expect(firstCompSum).toBeGreaterThan(0);
  });

  it("throws when not fitted", () => {
    const tsvd = new TruncatedSVD({ nComponents: 2 });
    expect(() => tsvd.transform(X)).toThrow();
    expect(() => tsvd.components).toThrow();
    expect(() => tsvd.explainedVariance).toThrow();
    expect(() => tsvd.explainedVarianceRatio).toThrow();
    expect(() => tsvd.singularValues).toThrow();
  });

  it("throws for invalid nComponents", () => {
    expect(() => new TruncatedSVD({ nComponents: 0 })).toThrow();
    expect(() => new TruncatedSVD({ nComponents: -1 })).toThrow();
    expect(() => new TruncatedSVD({ nComponents: 1.5 })).toThrow();
  });

  it("throws when nComponents > min(n_samples, n_features)", () => {
    const tsvd = new TruncatedSVD({ nComponents: 10 });
    expect(() => tsvd.fit(X)).toThrow();
  });

  it("getParams returns constructor options", () => {
    const tsvd = new TruncatedSVD({ nComponents: 3 });
    expect(tsvd.getParams()).toEqual({ nComponents: 3 });
  });

  it("default nComponents is 2", () => {
    const tsvd = new TruncatedSVD();
    const Xt = tsvd.fitTransform(X);
    expect(Xt.shape[1]).toBe(2);
  });
});

describe("NMF", () => {
  // Non-negative data
  const X = tensor([
    [1, 2, 0, 3],
    [5, 0, 7, 1],
    [0, 3, 4, 2],
    [2, 1, 3, 5],
    [4, 3, 2, 1],
  ]);

  it("fit and transform produces correct shape", () => {
    const nmf = new NMF({ nComponents: 2, randomState: 42 });
    nmf.fit(X);
    const W = nmf.transform(X);
    expect(W.shape).toEqual([5, 2]);
  });

  it("fitTransform convenience", () => {
    const nmf = new NMF({ nComponents: 2, randomState: 42 });
    const W = nmf.fitTransform(X);
    expect(W.shape).toEqual([5, 2]);
  });

  it("components shape matches (nComponents x nFeatures)", () => {
    const nmf = new NMF({ nComponents: 3, randomState: 42 });
    nmf.fit(X);
    expect(nmf.components.shape).toEqual([3, 4]);
  });

  it("W and H are non-negative", () => {
    const nmf = new NMF({ nComponents: 2, randomState: 42 });
    nmf.fit(X);
    const W = nmf.transform(X);
    const H = nmf.components;

    for (let i = 0; i < W.size; i++) {
      expect(Number(W.data[W.offset + i])).toBeGreaterThanOrEqual(0);
    }
    for (let i = 0; i < H.size; i++) {
      expect(Number(H.data[H.offset + i])).toBeGreaterThanOrEqual(0);
    }
  });

  it("reconstruction W @ H approximates X", () => {
    const nmf = new NMF({ nComponents: 4, randomState: 42, maxIter: 500 });
    nmf.fit(X);
    const W = nmf.transform(X);
    const H = nmf.components;

    const nSamples = 5;
    const nFeatures = 4;
    const k = 4;

    // Reconstruct: Xr = W @ H
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        let sum = 0;
        for (let c = 0; c < k; c++) {
          sum +=
            Number(W.data[W.offset + i * k + c]) * Number(H.data[H.offset + c * nFeatures + j]);
        }
        const original = Number(X.data[X.offset + i * nFeatures + j]);
        // Allow some reconstruction error
        expect(sum).toBeCloseTo(original, 0);
      }
    }
  });

  it("nIter tracks iteration count", () => {
    const nmf = new NMF({ nComponents: 2, randomState: 42, maxIter: 10 });
    nmf.fit(X);
    expect(nmf.nIter).toBeGreaterThan(0);
    expect(nmf.nIter).toBeLessThanOrEqual(10);
  });

  it("throws for negative data", () => {
    const Xneg = tensor([
      [1, -2],
      [3, 4],
    ]);
    const nmf = new NMF({ nComponents: 1 });
    expect(() => nmf.fit(Xneg)).toThrow(/non-negative/);
  });

  it("throws when not fitted", () => {
    const nmf = new NMF({ nComponents: 2 });
    expect(() => nmf.transform(X)).toThrow();
    expect(() => nmf.components).toThrow();
  });

  it("throws for invalid nComponents", () => {
    expect(() => new NMF({ nComponents: 0 })).toThrow();
    expect(() => new NMF({ nComponents: -1 })).toThrow();
  });

  it("throws for invalid maxIter", () => {
    expect(() => new NMF({ maxIter: 0 })).toThrow();
  });

  it("getParams returns constructor options", () => {
    const nmf = new NMF({ nComponents: 3, maxIter: 100, tol: 1e-5, randomState: 42 });
    const params = nmf.getParams();
    expect(params.nComponents).toBe(3);
    expect(params.maxIter).toBe(100);
    expect(params.tol).toBe(1e-5);
    expect(params.randomState).toBe(42);
  });

  it("deterministic with randomState", () => {
    const nmf1 = new NMF({ nComponents: 2, randomState: 123 });
    const nmf2 = new NMF({ nComponents: 2, randomState: 123 });
    nmf1.fit(X);
    nmf2.fit(X);

    const H1 = nmf1.components;
    const H2 = nmf2.components;
    for (let i = 0; i < H1.size; i++) {
      expect(Number(H1.data[H1.offset + i])).toBeCloseTo(Number(H2.data[H2.offset + i]), 10);
    }
  });
});

describe("TruncatedSVD vs PCA difference", () => {
  it("TruncatedSVD and PCA give different results on non-centered data", () => {
    const X = tensor([
      [10, 20, 30],
      [11, 22, 33],
      [12, 24, 36],
      [13, 26, 39],
    ]);

    const pca = new PCA({ nComponents: 2 });
    const tsvd = new TruncatedSVD({ nComponents: 2 });

    const XtPCA = pca.fitTransform(X);
    const XtSVD = tsvd.fitTransform(X);

    // Results should differ because PCA centers data, TruncatedSVD doesn't
    let anyDiff = false;
    for (let i = 0; i < XtPCA.size; i++) {
      if (
        Math.abs(Number(XtPCA.data[XtPCA.offset + i]) - Number(XtSVD.data[XtSVD.offset + i])) > 1e-6
      ) {
        anyDiff = true;
        break;
      }
    }
    expect(anyDiff).toBe(true);
  });
});
