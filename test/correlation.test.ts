import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { corrcoef, cov, kendalltau, pearsonr, spearmanr } from "../src/stats";

describe("Correlation Measures", () => {
  it("should calculate Pearson correlation", () => {
    const x = tensor([1, 2, 3, 4, 5]);
    const y = tensor([2, 4, 6, 8, 10]);
    const [r, p] = pearsonr(x, y);
    expect(r).toBeCloseTo(1, 10);
    expect(p).toBeLessThan(0.01);
  });

  it("should calculate Spearman correlation", () => {
    const x = tensor([1, 2, 3, 4, 5]);
    const y = tensor([5, 6, 7, 8, 9]);
    const [r, p] = spearmanr(x, y);
    expect(r).toBeCloseTo(1, 10);
    expect(p).toBeLessThan(0.01);
  });

  it("should calculate Kendall's tau", () => {
    const x = tensor([1, 2, 3, 4, 5]);
    const y = tensor([1, 3, 2, 5, 4]);
    const [tau, p] = kendalltau(x, y);
    expect(tau).toBeGreaterThan(0);
    expect(tau).toBeLessThanOrEqual(1);
    expect(p).toBeGreaterThan(0);
    expect(p).toBeLessThanOrEqual(1);
  });

  it("should calculate correlation coefficient matrix", () => {
    const x = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const result = corrcoef(x);
    expect(result.shape).toEqual([2, 2]);
    // Diagonal should be 1
    expect(Number(result.data[0])).toBeCloseTo(1, 10);
    expect(Number(result.data[3])).toBeCloseTo(1, 10);
  });

  it("should calculate covariance matrix", () => {
    const x = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const result = cov(x);
    expect(result.shape).toEqual([2, 2]);
    // Symmetric matrix
    expect(Number(result.data[1])).toBeCloseTo(Number(result.data[2]), 10);
  });
});
