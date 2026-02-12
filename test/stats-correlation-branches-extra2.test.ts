import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { kendalltau, pearsonr, spearmanr } from "../src/stats/correlation";

describe("stats correlation branches extra", () => {
  it("pearsonr returns NaN pvalue for df <= 0", () => {
    const [r, p] = pearsonr(tensor([1, 2]), tensor([1, 2]));
    expect(r).toBeCloseTo(1, 6);
    expect(Number.isNaN(p)).toBe(true);
  });

  it("spearmanr handles df <= 0", () => {
    const [rho, p] = spearmanr(tensor([1, 2]), tensor([2, 1]));
    expect(rho).toBeCloseTo(-1, 6);
    expect(Number.isNaN(p)).toBe(true);
  });

  it("kendalltau handles ties", () => {
    const [tau, p] = kendalltau(tensor([1, 1, 2]), tensor([2, 1, 1]));
    expect(Number.isFinite(tau)).toBe(true);
    expect(p).toBeGreaterThanOrEqual(0);
  });
});
