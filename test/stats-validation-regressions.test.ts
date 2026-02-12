import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  chisquare,
  f_oneway,
  friedmanchisquare,
  kruskal,
  kstest,
  mannwhitneyu,
  wilcoxon,
} from "../src/stats/tests";

describe("stats validation regressions", () => {
  it("rejects unsupported kstest distribution names", () => {
    expect(() => kstest(tensor([1, 2, 3]), "exponential")).toThrow(/Unsupported distribution/i);
  });

  it("rejects chisquare observed/expected length mismatch", () => {
    expect(() => chisquare(tensor([10, 20, 30]), tensor([15, 45]))).toThrow(/same length/i);
  });

  it("rejects chisquare when sums differ or inputs are negative", () => {
    expect(() => chisquare(tensor([10, 20]), tensor([10, 5]))).toThrow(/sum/i);
    expect(() => chisquare(tensor([-1, 2]), tensor([0.5, 0.5]))).toThrow(/observed/i);
  });

  it("rejects paired-length mismatch in wilcoxon", () => {
    expect(() => wilcoxon(tensor([1, 2, 3]), tensor([1, 2]))).toThrow(/equal length/i);
  });

  it("rejects wilcoxon when all paired differences are zero", () => {
    expect(() => wilcoxon(tensor([1, 2, 3]), tensor([1, 2, 3]))).toThrow(
      /all differences are zero/i
    );
  });

  it("rejects empty samples in mannwhitneyu", () => {
    expect(() => mannwhitneyu(tensor([]), tensor([1, 2]))).toThrow(/non-empty/i);
  });

  it("rejects unequal block lengths in friedmanchisquare", () => {
    expect(() => friedmanchisquare(tensor([1, 2, 3]), tensor([1, 2]), tensor([2, 3, 4]))).toThrow(
      /same length/i
    );
  });

  it("rejects kruskal and friedman when all values are tied", () => {
    expect(() => kruskal(tensor([1, 1]), tensor([1, 1]))).toThrow(/identical|undefined/i);
    expect(() =>
      friedmanchisquare(tensor([1, 1, 1]), tensor([1, 1, 1]), tensor([1, 1, 1]))
    ).toThrow(/identical|undefined/i);
  });

  it("rejects f_oneway when within-group degrees of freedom are invalid", () => {
    expect(() => f_oneway(tensor([1]), tensor([2]))).toThrow(/more than one sample/i);
  });
});
