import { describe, expect, it } from "vitest";
import {
  ndcgScore,
  pairwiseCosine,
  pairwiseEuclidean,
  pairwiseManhattan,
  reciprocalRank,
} from "../src/metrics";
import { tensor } from "../src/ndarray";

describe("pairwiseEuclidean", () => {
  it("computes pairwise Euclidean distances", () => {
    const X = tensor([
      [0, 0],
      [3, 4],
      [1, 0],
    ]);
    const D = pairwiseEuclidean(X);
    expect(D.shape).toEqual([3, 3]);
    const arr = D.toArray() as number[][];
    // d(0,0) = 0
    expect(arr[0]![0]).toBeCloseTo(0, 10);
    // d(0,1) = sqrt(9+16) = 5
    expect(arr[0]![1]).toBeCloseTo(5, 10);
    // d(0,2) = 1
    expect(arr[0]![2]).toBeCloseTo(1, 10);
    // Symmetric
    expect(arr[1]![0]).toBeCloseTo(arr[0]![1]!, 10);
  });

  it("diagonal is zero", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const D = pairwiseEuclidean(X);
    const arr = D.toArray() as number[][];
    for (let i = 0; i < 3; i++) {
      expect(arr[i]![i]).toBeCloseTo(0, 10);
    }
  });
});

describe("pairwiseManhattan", () => {
  it("computes pairwise Manhattan distances", () => {
    const X = tensor([
      [0, 0],
      [3, 4],
      [1, 0],
    ]);
    const D = pairwiseManhattan(X);
    const arr = D.toArray() as number[][];
    // d(0,1) = 3+4 = 7
    expect(arr[0]![1]).toBeCloseTo(7, 10);
    // d(0,2) = 1
    expect(arr[0]![2]).toBeCloseTo(1, 10);
    // Symmetric
    expect(arr[1]![0]).toBeCloseTo(arr[0]![1]!, 10);
  });
});

describe("pairwiseCosine", () => {
  it("identical vectors have distance 0", () => {
    const X = tensor([
      [1, 2],
      [1, 2],
    ]);
    const D = pairwiseCosine(X);
    const arr = D.toArray() as number[][];
    expect(arr[0]![1]).toBeCloseTo(0, 10);
  });

  it("orthogonal vectors have distance 1", () => {
    const X = tensor([
      [1, 0],
      [0, 1],
    ]);
    const D = pairwiseCosine(X);
    const arr = D.toArray() as number[][];
    expect(arr[0]![1]).toBeCloseTo(1, 10);
  });

  it("opposite vectors have distance 2", () => {
    const X = tensor([
      [1, 0],
      [-1, 0],
    ]);
    const D = pairwiseCosine(X);
    const arr = D.toArray() as number[][];
    expect(arr[0]![1]).toBeCloseTo(2, 10);
  });
});

describe("ndcgScore", () => {
  it("perfect ranking returns 1.0", () => {
    const yTrue = tensor([3, 2, 1, 0]);
    const yScore = tensor([3, 2, 1, 0]);
    expect(ndcgScore(yTrue, yScore)).toBeCloseTo(1.0, 10);
  });

  it("reversed ranking returns < 1.0", () => {
    const yTrue = tensor([3, 2, 1, 0]);
    const yScore = tensor([0, 1, 2, 3]);
    expect(ndcgScore(yTrue, yScore)).toBeLessThan(1.0);
  });

  it("supports k parameter", () => {
    const yTrue = tensor([3, 2, 1, 0]);
    const yScore = tensor([3, 2, 1, 0]);
    expect(ndcgScore(yTrue, yScore, 2)).toBeCloseTo(1.0, 10);
  });

  it("returns 0 for empty inputs", () => {
    expect(ndcgScore(tensor([]), tensor([]))).toBe(0);
  });

  it("all-zero relevance returns 0", () => {
    const yTrue = tensor([0, 0, 0]);
    const yScore = tensor([1, 2, 3]);
    expect(ndcgScore(yTrue, yScore)).toBe(0);
  });
});

describe("reciprocalRank", () => {
  it("returns 1.0 when first result is relevant", () => {
    const yTrue = tensor([1, 0, 0]);
    const yScore = tensor([3, 2, 1]);
    expect(reciprocalRank(yTrue, yScore)).toBeCloseTo(1.0, 10);
  });

  it("returns 0.5 when second result is relevant", () => {
    const yTrue = tensor([0, 1, 0]);
    const yScore = tensor([3, 2, 1]);
    expect(reciprocalRank(yTrue, yScore)).toBeCloseTo(0.5, 10);
  });

  it("returns 0 when no relevant results", () => {
    const yTrue = tensor([0, 0, 0]);
    const yScore = tensor([3, 2, 1]);
    expect(reciprocalRank(yTrue, yScore)).toBe(0);
  });

  it("returns 0 for empty input", () => {
    expect(reciprocalRank(tensor([]), tensor([]))).toBe(0);
  });
});
