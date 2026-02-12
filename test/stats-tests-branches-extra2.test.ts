import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { bartlett, levene, mannwhitneyu, shapiro, wilcoxon } from "../src/stats/tests";

describe("stats tests additional branch coverage", () => {
  it("covers Mann-Whitney continuity correction branches", () => {
    const equalU = mannwhitneyu(tensor([1, 4]), tensor([2, 3]));
    expect(Number.isFinite(equalU.statistic)).toBe(true);
    expect(Number.isFinite(equalU.pvalue)).toBe(true);

    const lowerU = mannwhitneyu(tensor([1, 2]), tensor([3, 4]));
    expect(Number.isFinite(lowerU.statistic)).toBe(true);
    expect(Number.isFinite(lowerU.pvalue)).toBe(true);
  });

  it("covers Levene median branch for even sample sizes", () => {
    const res = levene("median", tensor([1, 2, 4, 8]), tensor([1, 3, 6, 10]));
    expect(Number.isFinite(res.statistic)).toBe(true);
  });

  it("covers shapiro n=3 and n>11 paths", () => {
    const res3 = shapiro(tensor([1, 2, 3]));
    expect(res3.pvalue).toBeGreaterThanOrEqual(0);

    const res12 = shapiro(tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]));
    expect(res12.pvalue).toBeGreaterThanOrEqual(0);
  });

  it("covers wilcoxon without paired input", () => {
    const res = wilcoxon(tensor([1, 2, 3]));
    expect(Number.isFinite(res.statistic)).toBe(true);
  });

  it("covers bartlett with three groups", () => {
    const res = bartlett(tensor([1, 2, 3]), tensor([2, 4, 6]), tensor([3, 6, 9]));
    expect(Number.isFinite(res.statistic)).toBe(true);
  });
});
