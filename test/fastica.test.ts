import { describe, expect, it } from "vitest";
import { FastICA } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("FastICA", () => {
  // Mixed signals (simple linear mixing of independent sources)
  const X = tensor([
    [1, 2],
    [3, 1],
    [5, 4],
    [7, 3],
    [9, 6],
    [2, 5],
    [4, 7],
    [6, 8],
    [8, 2],
    [10, 9],
  ]);

  it("fitTransform produces correct shape", () => {
    const ica = new FastICA({ nComponents: 2, randomState: 42 });
    const S = ica.fitTransform(X);
    expect(S.shape[0]).toBe(10);
    expect(S.shape[1]).toBe(2);
  });

  it("transform produces same shape as fitTransform", () => {
    const ica = new FastICA({ nComponents: 2, randomState: 42 });
    ica.fit(X);
    const S = ica.transform(X);
    expect(S.shape[0]).toBe(10);
    expect(S.shape[1]).toBe(2);
  });

  it("nComponents < nFeatures reduces dimensionality", () => {
    const ica = new FastICA({ nComponents: 1, randomState: 42 });
    const S = ica.fitTransform(X);
    expect(S.shape[0]).toBe(10);
    expect(S.shape[1]).toBe(1);
  });

  it("components are accessible after fit", () => {
    const ica = new FastICA({ nComponents: 2, randomState: 42 });
    ica.fit(X);
    const comp = ica.components;
    expect(comp.shape[0]).toBe(2);
    expect(comp.shape[1]).toBe(2);
  });

  it("mixing matrix is accessible after fit", () => {
    const ica = new FastICA({ nComponents: 2, randomState: 42 });
    ica.fit(X);
    const mix = ica.mixingMatrix;
    expect(mix.shape[0]).toBe(2);
    expect(mix.shape[1]).toBe(2);
  });

  it("inverseTransform approximately reconstructs original", () => {
    const ica = new FastICA({ nComponents: 2, randomState: 42 });
    ica.fit(X);
    const S = ica.transform(X);
    const Xrecon = ica.inverseTransform(S);
    expect(Xrecon.shape[0]).toBe(10);
    expect(Xrecon.shape[1]).toBe(2);
    // Check reconstruction is close to original
    for (let i = 0; i < Xrecon.size; i++) {
      expect(Number(Xrecon.data[Xrecon.offset + i])).toBeCloseTo(Number(X.data[X.offset + i]), 0);
    }
  });

  it("exp contrast function works", () => {
    const ica = new FastICA({ nComponents: 2, fun: "exp", randomState: 42 });
    const S = ica.fitTransform(X);
    expect(S.shape[0]).toBe(10);
    expect(S.shape[1]).toBe(2);
  });

  it("cube contrast function works", () => {
    const ica = new FastICA({ nComponents: 2, fun: "cube", randomState: 42 });
    const S = ica.fitTransform(X);
    expect(S.shape[0]).toBe(10);
    expect(S.shape[1]).toBe(2);
  });

  it("nIter is positive after fit", () => {
    const ica = new FastICA({ nComponents: 2, randomState: 42 });
    ica.fit(X);
    expect(ica.nIter).toBeGreaterThan(0);
  });

  it("deterministic with randomState", () => {
    const ica1 = new FastICA({ nComponents: 2, randomState: 42 });
    const ica2 = new FastICA({ nComponents: 2, randomState: 42 });
    const S1 = ica1.fitTransform(X);
    const S2 = ica2.fitTransform(X);
    for (let i = 0; i < S1.size; i++) {
      expect(Number(S1.data[S1.offset + i])).toBeCloseTo(Number(S2.data[S2.offset + i]), 10);
    }
  });

  it("throws when not fitted", () => {
    const ica = new FastICA({ nComponents: 2 });
    expect(() => ica.transform(X)).toThrow();
    expect(() => ica.components).toThrow();
    expect(() => ica.mixingMatrix).toThrow();
    expect(() => ica.inverseTransform(X)).toThrow();
  });

  it("throws for invalid nComponents", () => {
    expect(() => new FastICA({ nComponents: 0 })).toThrow();
    expect(() => new FastICA({ nComponents: -1 })).toThrow();
  });

  it("throws for invalid maxIter", () => {
    expect(() => new FastICA({ maxIter: 0 })).toThrow();
  });

  it("getParams returns options", () => {
    const ica = new FastICA({ nComponents: 3, fun: "exp", randomState: 42 });
    const params = ica.getParams();
    expect(params.nComponents).toBe(3);
    expect(params.fun).toBe("exp");
    expect(params.randomState).toBe(42);
    expect(params.whiten).toBe(true);
  });

  it("setParams works and rejects unknown params", () => {
    const ica = new FastICA({ nComponents: 2 });
    expect(ica.setParams({})).toBe(ica);
    expect(() => ica.setParams({ unknown: 1 })).toThrow(/Unknown parameter/);
  });
});
