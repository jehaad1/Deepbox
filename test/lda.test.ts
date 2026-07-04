import { describe, expect, it } from "vitest";
import { LatentDirichletAllocation } from "../src/ml";
import { tensor } from "../src/ndarray";

describe("LatentDirichletAllocation", () => {
  // Simple document-term matrix (5 docs, 6 words)
  const X = tensor([
    [3, 0, 1, 0, 0, 2],
    [0, 2, 0, 4, 1, 0],
    [2, 0, 3, 0, 0, 1],
    [0, 1, 0, 3, 2, 0],
    [1, 1, 1, 1, 1, 1],
  ]);

  it("fitTransform produces correct shape", () => {
    const lda = new LatentDirichletAllocation({
      nComponents: 2,
      maxIter: 5,
      randomState: 42,
    });
    const dt = lda.fitTransform(X);
    expect(dt.shape[0]).toBe(5);
    expect(dt.shape[1]).toBe(2);
  });

  it("document-topic proportions sum to ~1", () => {
    const lda = new LatentDirichletAllocation({
      nComponents: 3,
      maxIter: 5,
      randomState: 42,
    });
    const dt = lda.fitTransform(X);
    const K = dt.shape[1] ?? 0;
    for (let i = 0; i < 5; i++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += Number(dt.data[dt.offset + i * K + k]);
      }
      expect(sum).toBeCloseTo(1.0, 3);
    }
  });

  it("topic proportions are non-negative", () => {
    const lda = new LatentDirichletAllocation({
      nComponents: 2,
      maxIter: 5,
      randomState: 42,
    });
    const dt = lda.fitTransform(X);
    for (let i = 0; i < dt.size; i++) {
      expect(Number(dt.data[dt.offset + i])).toBeGreaterThanOrEqual(0);
    }
  });

  it("components are accessible after fit", () => {
    const lda = new LatentDirichletAllocation({
      nComponents: 2,
      maxIter: 5,
      randomState: 42,
    });
    lda.fit(X);
    const comp = lda.components;
    expect(comp.shape[0]).toBe(2);
    expect(comp.shape[1]).toBe(6);
  });

  it("components values are positive (Dirichlet parameters)", () => {
    const lda = new LatentDirichletAllocation({
      nComponents: 2,
      maxIter: 5,
      randomState: 42,
    });
    lda.fit(X);
    const comp = lda.components;
    for (let i = 0; i < comp.size; i++) {
      expect(Number(comp.data[comp.offset + i])).toBeGreaterThan(0);
    }
  });

  it("transform produces same shape on new data", () => {
    const lda = new LatentDirichletAllocation({
      nComponents: 2,
      maxIter: 5,
      randomState: 42,
    });
    lda.fit(X);
    const Xnew = tensor([
      [1, 0, 2, 0, 0, 1],
      [0, 1, 0, 2, 1, 0],
    ]);
    const dt = lda.transform(Xnew);
    expect(dt.shape[0]).toBe(2);
    expect(dt.shape[1]).toBe(2);
  });

  it("nIter is positive after fit", () => {
    const lda = new LatentDirichletAllocation({
      nComponents: 2,
      maxIter: 5,
      randomState: 42,
    });
    lda.fit(X);
    expect(lda.nIter).toBeGreaterThan(0);
  });

  it("deterministic with randomState", () => {
    const lda1 = new LatentDirichletAllocation({
      nComponents: 2,
      maxIter: 5,
      randomState: 42,
    });
    const lda2 = new LatentDirichletAllocation({
      nComponents: 2,
      maxIter: 5,
      randomState: 42,
    });
    const dt1 = lda1.fitTransform(X);
    const dt2 = lda2.fitTransform(X);
    for (let i = 0; i < dt1.size; i++) {
      expect(Number(dt1.data[dt1.offset + i])).toBeCloseTo(Number(dt2.data[dt2.offset + i]), 10);
    }
  });

  it("throws when not fitted", () => {
    const lda = new LatentDirichletAllocation({ nComponents: 2 });
    expect(() => lda.transform(X)).toThrow();
    expect(() => lda.components).toThrow();
  });

  it("throws for negative values", () => {
    const lda = new LatentDirichletAllocation({
      nComponents: 2,
      randomState: 42,
    });
    const Xneg = tensor([
      [1, -1],
      [0, 2],
    ]);
    expect(() => lda.fit(Xneg)).toThrow();
  });

  it("throws for invalid nComponents", () => {
    expect(() => new LatentDirichletAllocation({ nComponents: 0 })).toThrow();
  });

  it("throws for invalid maxIter", () => {
    expect(() => new LatentDirichletAllocation({ maxIter: 0 })).toThrow();
  });

  it("inverseTransform works after fit", () => {
    const lda = new LatentDirichletAllocation({
      nComponents: 2,
      randomState: 42,
    });
    lda.fit(X);
    const topicDist = lda.transform(X);
    const reconstructed = lda.inverseTransform(topicDist);
    expect(reconstructed.shape[0]).toBe(X.shape[0]);
    expect(reconstructed.shape[1]).toBe(X.shape[1]);
  });

  it("getParams returns options", () => {
    const lda = new LatentDirichletAllocation({
      nComponents: 3,
      maxIter: 20,
      randomState: 42,
    });
    const params = lda.getParams();
    expect(params.nComponents).toBe(3);
    expect(params.maxIter).toBe(20);
    expect(params.randomState).toBe(42);
  });

  it("setParams works and rejects unknown params", () => {
    const lda = new LatentDirichletAllocation({ nComponents: 2 });
    expect(lda.setParams({})).toBe(lda);
    expect(() => lda.setParams({ unknown: 1 })).toThrow(/Unknown parameter/);
  });
});
