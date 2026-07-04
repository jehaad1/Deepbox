import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { mutual_info_classif, mutual_info_regression, SelectKBest } from "../src/preprocess";

describe("mutual_info_classif", () => {
  it("returns non-negative scores for each feature", () => {
    const X = tensor([
      [1, 10],
      [2, 20],
      [3, 30],
      [4, 40],
      [5, 50],
      [6, 60],
    ]);
    const y = tensor([0, 0, 0, 1, 1, 1]);
    const scores = mutual_info_classif(X, y);
    expect(scores).toHaveLength(2);
    for (const s of scores) {
      expect(s).toBeGreaterThanOrEqual(0);
      expect(Number.isFinite(s)).toBe(true);
    }
  });

  it("assigns higher MI to informative features", () => {
    // Feature 0 perfectly separates classes; feature 1 is random noise
    const X = tensor([
      [0, 5],
      [0, 3],
      [0, 7],
      [0, 1],
      [1, 4],
      [1, 6],
      [1, 2],
      [1, 8],
    ]);
    const y = tensor([0, 0, 0, 0, 1, 1, 1, 1]);
    const scores = mutual_info_classif(X, y, { nNeighbors: 2 });
    expect(scores).toHaveLength(2);
    // Feature 0 (perfect separator) should have higher MI than feature 1 (noise)
    expect(scores[0]).toBeGreaterThan(scores[1]!);
  });

  it("works with single feature", () => {
    const X = tensor([[1], [2], [3], [4], [5], [6]]);
    const y = tensor([0, 0, 0, 1, 1, 1]);
    const scores = mutual_info_classif(X, y);
    expect(scores).toHaveLength(1);
    expect(scores[0]).toBeGreaterThanOrEqual(0);
  });

  it("returns zero MI for independent features", () => {
    // All same class => no information from any feature
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const y = tensor([0, 0, 0, 0]);
    const scores = mutual_info_classif(X, y, { nNeighbors: 1 });
    expect(scores).toHaveLength(2);
    for (const s of scores) {
      expect(s).toBe(0);
    }
  });

  it("is deterministic with same randomState", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const y = tensor([0, 0, 1, 1]);
    const scores1 = mutual_info_classif(X, y, { randomState: 42 });
    const scores2 = mutual_info_classif(X, y, { randomState: 42 });
    expect(scores1).toEqual(scores2);
  });

  it("respects nNeighbors parameter", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
      [9, 10],
      [11, 12],
    ]);
    const y = tensor([0, 0, 0, 1, 1, 1]);
    const scores1 = mutual_info_classif(X, y, { nNeighbors: 1 });
    const scores2 = mutual_info_classif(X, y, { nNeighbors: 2 });
    // Different k should give different (but both valid) results
    expect(scores1).toHaveLength(2);
    expect(scores2).toHaveLength(2);
    for (const s of [...scores1, ...scores2]) {
      expect(s).toBeGreaterThanOrEqual(0);
    }
  });

  it("can be used as scoreFunc for SelectKBest", () => {
    const X = tensor([
      [0, 5, 10],
      [0, 3, 20],
      [0, 7, 30],
      [1, 4, 40],
      [1, 6, 50],
      [1, 2, 60],
    ]);
    const y = tensor([0, 0, 0, 1, 1, 1]);
    const skb = new SelectKBest({ scoreFunc: mutual_info_classif, k: 1 });
    skb.fit(X, y);
    const support = skb.getSupport();
    expect(support).toHaveLength(3);
    // At least one feature should be selected
    expect(support.filter(Boolean)).toHaveLength(1);
  });

  // Error cases
  it("throws on string dtype X", () => {
    const X = tensor(["a", "b"]).reshape([2, 1]);
    const y = tensor([0, 1]);
    expect(() => mutual_info_classif(X, y)).toThrow();
  });

  it("throws on 1D X", () => {
    const X = tensor([1, 2, 3]);
    const y = tensor([0, 0, 1]);
    expect(() => mutual_info_classif(X, y)).toThrow();
  });

  it("throws on mismatched X and y lengths", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    const y = tensor([0, 0, 1]);
    expect(() => mutual_info_classif(X, y)).toThrow();
  });

  it("throws on invalid nNeighbors", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    const y = tensor([0, 1]);
    expect(() => mutual_info_classif(X, y, { nNeighbors: 0 })).toThrow();
    expect(() => mutual_info_classif(X, y, { nNeighbors: -1 })).toThrow();
    expect(() => mutual_info_classif(X, y, { nNeighbors: 2 })).toThrow(); // >= nSamples
  });
});

describe("mutual_info_regression", () => {
  it("returns non-negative scores for each feature", () => {
    const X = tensor([
      [1, 10],
      [2, 20],
      [3, 30],
      [4, 40],
      [5, 50],
      [6, 60],
    ]);
    const y = tensor([1.1, 2.2, 3.3, 4.4, 5.5, 6.6]);
    const scores = mutual_info_regression(X, y);
    expect(scores).toHaveLength(2);
    for (const s of scores) {
      expect(s).toBeGreaterThanOrEqual(0);
      expect(Number.isFinite(s)).toBe(true);
    }
  });

  it("assigns higher MI to correlated features", () => {
    // Feature 0: perfectly correlated with y; feature 1: constant (no info)
    const X = tensor([
      [1, 5],
      [2, 5],
      [3, 5],
      [4, 5],
      [5, 5],
      [6, 5],
      [7, 5],
      [8, 5],
    ]);
    const y = tensor([1, 2, 3, 4, 5, 6, 7, 8]);
    const scores = mutual_info_regression(X, y, { nNeighbors: 2 });
    expect(scores).toHaveLength(2);
    expect(scores[0]).toBeGreaterThan(scores[1]!);
  });

  it("works with single feature", () => {
    const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]);
    const y = tensor([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);
    const scores = mutual_info_regression(X, y, { nNeighbors: 2 });
    expect(scores).toHaveLength(1);
    expect(scores[0]).toBeGreaterThanOrEqual(0);
  });

  it("is deterministic with same randomState", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const y = tensor([1.0, 2.0, 3.0, 4.0]);
    const scores1 = mutual_info_regression(X, y, { randomState: 42 });
    const scores2 = mutual_info_regression(X, y, { randomState: 42 });
    expect(scores1).toEqual(scores2);
  });

  it("can be used as scoreFunc for SelectKBest", () => {
    const X = tensor([
      [1, 5, 100],
      [2, 5, 200],
      [3, 5, 300],
      [4, 5, 400],
      [5, 5, 500],
      [6, 5, 600],
    ]);
    const y = tensor([1, 2, 3, 4, 5, 6]);
    const skb = new SelectKBest({ scoreFunc: mutual_info_regression, k: 1 });
    skb.fit(X, y);
    const support = skb.getSupport();
    expect(support).toHaveLength(3);
    expect(support.filter(Boolean)).toHaveLength(1);
  });

  // Error cases
  it("throws on string dtype X", () => {
    const X = tensor(["a", "b"]).reshape([2, 1]);
    const y = tensor([1.0, 2.0]);
    expect(() => mutual_info_regression(X, y)).toThrow();
  });

  it("throws on 1D X", () => {
    const X = tensor([1, 2, 3]);
    const y = tensor([1.0, 2.0, 3.0]);
    expect(() => mutual_info_regression(X, y)).toThrow();
  });

  it("throws on mismatched lengths", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    const y = tensor([1.0, 2.0, 3.0]);
    expect(() => mutual_info_regression(X, y)).toThrow();
  });

  it("throws on invalid nNeighbors", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    const y = tensor([1.0, 2.0]);
    expect(() => mutual_info_regression(X, y, { nNeighbors: 0 })).toThrow();
    expect(() => mutual_info_regression(X, y, { nNeighbors: 2 })).toThrow();
  });

  it("throws on string dtype y", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
    ]);
    const y = tensor(["a", "b"]);
    expect(() => mutual_info_regression(X, y)).toThrow();
  });
});
