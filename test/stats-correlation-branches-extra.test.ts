import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { corrcoef, cov, kendalltau, pearsonr, spearmanr } from "../src/stats";

describe("stats correlation branch coverage extras", () => {
  it("pearsonr handles df=0 and constant input", () => {
    const [r, p] = pearsonr(tensor([1, 2]), tensor([2, 3]));
    expect(r).toBeCloseTo(1, 6);
    expect(Number.isNaN(p)).toBe(true);

    expect(() => pearsonr(tensor([1, 1, 1]), tensor([2, 2, 2]))).toThrow(/constant/);
  });

  it("spearmanr handles df=0", () => {
    const [rho, p] = spearmanr(tensor([1, 2]), tensor([2, 1]));
    expect(rho).toBeCloseTo(-1, 6);
    expect(Number.isNaN(p)).toBe(true);
  });

  it("kendalltau supports minimal input without throwing", () => {
    const [tau, p] = kendalltau(tensor([1, 2]), tensor([2, 1]));
    expect(Number.isFinite(tau)).toBe(true);
    expect(Number.isFinite(p)).toBe(true);
  });

  it("corrcoef handles 1D, 2D, and y input", () => {
    const c1 = corrcoef(tensor([1, 2, 3]));
    expect(c1.shape).toEqual([1, 1]);

    const c2 = corrcoef(tensor([1, 2, 3]), tensor([1, 2, 3]));
    expect(c2.shape).toEqual([2, 2]);

    const data = tensor([
      [1, 1],
      [1, 2],
      [1, 3],
    ]);
    const c3 = corrcoef(data);
    expect(c3.shape).toEqual([2, 2]);
    // Column 0 is constant -> NaN correlations
    expect(Number.isNaN(Number(c3.data[c3.offset + 1]))).toBe(true);
  });

  it("cov validates ddof and mismatched input size", () => {
    expect(() => cov(tensor([1, 2, 3]), undefined, 3)).toThrow(/ddof/i);
    expect(() => cov(tensor([1, 2, 3]), undefined, -1)).toThrow(/ddof/i);
    expect(() => cov(tensor([1, 2]), tensor([1, 2, 3]))).toThrow(/cov/i);
  });
});
