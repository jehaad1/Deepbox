import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  chisquare,
  kstest,
  normaltest,
  ttest_1samp,
  ttest_ind,
  ttest_rel,
} from "../src/stats/tests";

describe("stats tests branch coverage", () => {
  it("handles t-tests edge cases", () => {
    expect(() => ttest_1samp(tensor([1]), 0)).toThrow(/at least 2/);
    expect(() => ttest_1samp(tensor([1, 1, 1]), 1)).toThrow(/constant input/i);
    const one = ttest_1samp(tensor([1, 2, 3]), 2);
    expect(one.pvalue).toBeGreaterThan(0);

    expect(() => ttest_ind(tensor([1]), tensor([2, 3]))).toThrow(/at least 2/);
    expect(() => ttest_ind(tensor([1, 1]), tensor([1, 1]), true)).toThrow(/constant input/i);
    const ind = ttest_ind(tensor([1, 2, 3]), tensor([2, 3, 4]), false);
    expect(ind.pvalue).toBeGreaterThan(0);

    expect(() => ttest_rel(tensor([1, 2]), tensor([1]))).toThrow(/equal length/i);
    expect(() => ttest_rel(tensor([1]), tensor([1]))).toThrow(/at least 2/);
    expect(() => ttest_rel(tensor([1, 1]), tensor([1, 1]))).toThrow(/constant differences/i);
  });

  it("handles chisquare and kstest branches", () => {
    const obs = tensor([10, 20, 30]);
    const chi = chisquare(obs);
    expect(chi.pvalue).toBeGreaterThan(0);

    const exp = tensor([15, 15, 30]);
    const chiExp = chisquare(obs, exp);
    expect(chiExp.pvalue).toBeGreaterThan(0);

    expect(() => kstest(tensor([]), "norm")).toThrow(/at least one element/i);

    const ksNorm = kstest(tensor([0, 0.5, 1]), "norm");
    expect(ksNorm.pvalue).toBeGreaterThanOrEqual(0);

    const ksFunc = kstest(tensor([0, 1, 2]), (x) => x / 2);
    expect(ksFunc.pvalue).toBeGreaterThanOrEqual(0);

    expect(() => kstest(tensor([0, 1, 2]), "unknown")).toThrow(/Unsupported distribution/i);
  });

  it("handles normaltest edge cases", () => {
    expect(() => normaltest(tensor([1, 2, 3, 4, 5, 6, 7]))).toThrow(/at least 8/);
    expect(() => normaltest(tensor([1, 1, 1, 1, 1, 1, 1, 1]))).toThrow(/constant/i);

    const res = normaltest(tensor([1, 2, 3, 4, 5, 6, 7, 8]));
    expect(res.pvalue).toBeGreaterThanOrEqual(0);
  });
});
