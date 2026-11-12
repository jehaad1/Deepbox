import { describe, expect, it } from "vitest";
import { catchWarnings } from "../src/core";
import { tensor } from "../src/ndarray";
import { corrcoef, cov, kendalltau, pearsonr, spearmanr } from "../src/stats/correlation";

describe("stats correlation error branches", () => {
  it("throws for insufficient or constant inputs", () => {
    // Constant input gives NaN and a warning (scipy), not an error.
    let constant: [number, number] = [0, 0];
    catchWarnings(() => {
      constant = pearsonr(tensor([1, 1]), tensor([2, 2]));
    });
    expect(constant[0]).toBeNaN();
    expect(() => pearsonr(tensor([1]), tensor([2]))).toThrow(/at least 2/);
    expect(() => spearmanr(tensor([1]), tensor([2]))).toThrow(/at least 2/);
    expect(() => kendalltau(tensor([1]), tensor([2]))).toThrow(/at least 2/);
  });

  it("validates corrcoef and cov dimensions", () => {
    expect(() => corrcoef(tensor([[[1]]]))).toThrow(/1D or 2D/);
    expect(() => cov(tensor([[[1]]]))).toThrow(/1D or 2D/);
  });

  it("validates cov ddof", () => {
    expect(() => cov(tensor([1, 2, 3]), undefined, 3)).toThrow(/ddof/);
  });
});
