import { describe, expect, it } from "vitest";
import { coverageError, labelRankingLoss } from "../src/metrics";
import { tensor } from "../src/ndarray";

describe("coverageError", () => {
  it("perfect ranking gives minimum coverage", () => {
    // Labels: sample 0 has label 0, sample 1 has label 1
    const yTrue = tensor([
      [1, 0],
      [0, 1],
    ]);
    const yScore = tensor([
      [1.0, 0.0],
      [0.0, 1.0],
    ]);
    // Perfect ranking: coverage = avg number of true labels = 1
    expect(coverageError(yTrue, yScore)).toBe(1);
  });

  it("worst ranking gives maximum coverage", () => {
    // Label 0 is true but has lowest score
    const yTrue = tensor([[1, 0]]);
    const yScore = tensor([[0.0, 1.0]]);
    // Need to go through rank 2 to find the true label
    expect(coverageError(yTrue, yScore)).toBe(2);
  });

  it("handles multiple true labels", () => {
    // Both labels are true
    const yTrue = tensor([[1, 1]]);
    const yScore = tensor([[0.5, 0.8]]);
    // Max rank of true labels = 2 (need to cover both)
    expect(coverageError(yTrue, yScore)).toBe(2);
  });

  it("handles multiple samples", () => {
    const yTrue = tensor([
      [1, 0, 0],
      [0, 1, 1],
    ]);
    const yScore = tensor([
      [0.9, 0.1, 0.2],
      [0.1, 0.8, 0.9],
    ]);
    const result = coverageError(yTrue, yScore);
    expect(result).toBeGreaterThanOrEqual(1);
    expect(result).toBeLessThanOrEqual(3);
  });

  it("throws on 1D input", () => {
    expect(() => coverageError(tensor([1, 0]), tensor([0.5, 0.5]))).toThrow(/2D/);
  });

  it("throws on shape mismatch", () => {
    expect(() =>
      coverageError(
        tensor([[1, 0]]),
        tensor([
          [0.5, 0.5],
          [0.3, 0.7],
        ])
      )
    ).toThrow(/shape mismatch/);
  });
});

describe("labelRankingLoss", () => {
  it("perfect ranking gives 0 loss", () => {
    const yTrue = tensor([
      [1, 0],
      [0, 1],
    ]);
    const yScore = tensor([
      [1.0, 0.0],
      [0.0, 1.0],
    ]);
    expect(labelRankingLoss(yTrue, yScore)).toBe(0);
  });

  it("worst ranking gives 1 loss", () => {
    // True label has lower score than false label
    const yTrue = tensor([[1, 0]]);
    const yScore = tensor([[0.0, 1.0]]);
    expect(labelRankingLoss(yTrue, yScore)).toBe(1);
  });

  it("partial ordering gives intermediate loss", () => {
    // 2 labels true, 1 false. One true has higher score, one has lower.
    const yTrue = tensor([[1, 1, 0]]);
    const yScore = tensor([[0.9, 0.1, 0.5]]);
    // Pairs: (0,2) correct (0.9 > 0.5), (1,2) incorrect (0.1 <= 0.5)
    // Loss for this sample = 1/2 = 0.5
    expect(labelRankingLoss(yTrue, yScore)).toBeCloseTo(0.5, 5);
  });

  it("handles multiple samples", () => {
    const yTrue = tensor([
      [1, 0, 0],
      [0, 1, 0],
    ]);
    const yScore = tensor([
      [0.9, 0.1, 0.2],
      [0.1, 0.9, 0.2],
    ]);
    // Both perfectly ranked
    expect(labelRankingLoss(yTrue, yScore)).toBe(0);
  });

  it("skips samples with all-positive or all-negative labels", () => {
    // Sample 0 has all labels true → no negative pairs → skip
    const yTrue = tensor([
      [1, 1],
      [1, 0],
    ]);
    const yScore = tensor([
      [0.5, 0.8],
      [0.9, 0.1],
    ]);
    // Only sample 1 contributes: perfectly ranked → 0
    // Total = 0 / 2 = 0
    expect(labelRankingLoss(yTrue, yScore)).toBe(0);
  });

  it("throws on 1D input", () => {
    expect(() => labelRankingLoss(tensor([1, 0]), tensor([0.5, 0.5]))).toThrow(/2D/);
  });

  it("throws on shape mismatch", () => {
    expect(() =>
      labelRankingLoss(
        tensor([[1, 0]]),
        tensor([
          [0.5, 0.5],
          [0.3, 0.7],
        ])
      )
    ).toThrow(/shape mismatch/);
  });
});
