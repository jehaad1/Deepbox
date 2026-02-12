import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  anderson,
  bartlett,
  chisquare,
  f_oneway,
  friedmanchisquare,
  kruskal,
  kstest,
  levene,
  mannwhitneyu,
  normaltest,
  shapiro,
  ttest_1samp,
  ttest_ind,
  ttest_rel,
  wilcoxon,
} from "../src/stats/tests";

describe("stats tests branch coverage", () => {
  it("covers t-tests validations and branches", () => {
    expect(() => ttest_1samp(tensor([1]), 0)).toThrow(/requires at least 2/);
    expect(() => ttest_1samp(tensor([2, 2]), 0)).toThrow(/constant input/);
    const t1 = ttest_1samp(tensor([1, 2, 3, 4]), 2.5);
    expect(Number.isFinite(t1.statistic)).toBe(true);

    expect(() => ttest_ind(tensor([1]), tensor([1, 2]))).toThrow(/at least 2/);
    expect(() => ttest_ind(tensor([1, 1]), tensor([1, 1]), true)).toThrow(/constant input/);
    const t2 = ttest_ind(tensor([1, 2, 3]), tensor([4, 5, 6]), false);
    expect(Number.isFinite(t2.statistic)).toBe(true);

    expect(() => ttest_rel(tensor([1, 2]), tensor([1]))).toThrow(/paired samples/);
    expect(() => ttest_rel(tensor([1]), tensor([2]))).toThrow(/at least 2/);
    expect(() => ttest_rel(tensor([1, 2]), tensor([1, 2]))).toThrow(/constant differences/);
    const t3 = ttest_rel(tensor([1, 2, 3]), tensor([1, 1, 2]));
    expect(Number.isFinite(t3.statistic)).toBe(true);
  });

  it("covers chisquare and kstest branches", () => {
    expect(() => chisquare(tensor([]))).toThrow(/at least one/);
    expect(() => chisquare(tensor([1, -1]))).toThrow(/finite and >= 0/);
    expect(() => chisquare(tensor([1, 2]), tensor([1]))).toThrow(/same length/);
    expect(() => chisquare(tensor([1, 2]), tensor([1, 1]))).toThrow(/sum/);
    const c1 = chisquare(tensor([1, 2, 3]));
    expect(Number.isFinite(c1.statistic)).toBe(true);

    expect(() => kstest(tensor([]), "norm")).toThrow(/at least one/);
    expect(() => kstest(tensor([1, 2, 3]), "bad")).toThrow(/Unsupported/);
    const k1 = kstest(tensor([1, 2, 3, 4]), "norm");
    expect(k1.pvalue).toBeGreaterThanOrEqual(0);
    const k2 = kstest(tensor([1, 2, 3, 4]), (x) => (x <= 0 ? 0 : x >= 1 ? 1 : x));
    expect(k2.pvalue).toBeGreaterThanOrEqual(0);
  });

  it("covers normality tests", () => {
    expect(() => normaltest(tensor([1, 2, 3]))).toThrow(/at least 8/);
    expect(() => normaltest(tensor([2, 2, 2, 2, 2, 2, 2, 2]))).toThrow(/constant input/);
    const n1 = normaltest(tensor([1, 2, 3, 4, 5, 6, 7, 8]));
    expect(Number.isFinite(n1.statistic)).toBe(true);

    expect(() => shapiro(tensor([1, 2]))).toThrow(/between 3 and 5000/);
    expect(() => shapiro(tensor([2, 2, 2]))).toThrow(/identical/);
    const s1 = shapiro(tensor([1, 2, 3, 4, 5]));
    expect(Number.isFinite(s1.statistic)).toBe(true);

    expect(() => anderson(tensor([]))).toThrow(/at least one/);
    const a1 = anderson(tensor([1, 2, 3, 4, 5]));
    expect(a1.critical_values.length).toBeGreaterThan(0);
  });

  it("covers rank-based tests", () => {
    expect(() => mannwhitneyu(tensor([]), tensor([1]))).toThrow(/non-empty/);
    const mw = mannwhitneyu(tensor([1, 1]), tensor([1, 1]));
    expect(Number.isNaN(mw.pvalue)).toBe(true);

    expect(() => wilcoxon(tensor([1, 1]), tensor([1, 1]))).toThrow(/all differences are zero/);
    expect(() => wilcoxon(tensor([1, 2]), tensor([1]))).toThrow(/equal length/);
    const w1 = wilcoxon(tensor([1, 2, 3]), tensor([1, 1, 2]));
    expect(Number.isFinite(w1.statistic)).toBe(true);

    expect(() => kruskal(tensor([1]))).toThrow(/at least 2/);
    expect(() => kruskal(tensor([]), tensor([1]))).toThrow(/non-empty/);
    expect(() => kruskal(tensor([1, 1]), tensor([1, 1]))).toThrow(/identical/);

    expect(() => friedmanchisquare(tensor([1]), tensor([1]))).toThrow(/at least 3/);
    expect(() => friedmanchisquare(tensor([1, 2]), tensor([1]), tensor([1, 2]))).toThrow(
      /same length/
    );
    expect(() => friedmanchisquare(tensor([1, 1]), tensor([1, 1]), tensor([1, 1]))).toThrow(
      /identical/
    );
  });

  it("covers variance tests", () => {
    expect(() => levene("median", tensor([1]))).toThrow(/at least 2 groups/);
    expect(() => levene("median", tensor([1]), tensor([2, 3]))).toThrow(/at least 2 samples/);
    const lv = levene("trimmed", tensor([1, 2, 3, 4]), tensor([2, 4, 6, 8]));
    expect(Number.isFinite(lv.statistic)).toBe(true);

    expect(() => bartlett(tensor([1]))).toThrow(/at least 2 groups/);
    expect(() => bartlett(tensor([1]), tensor([2, 3]))).toThrow(/at least 2 samples/);
    expect(() => bartlett(tensor([1, 1]), tensor([2, 2]))).toThrow(/zero variance/);
    const bt = bartlett(tensor([1, 2, 3]), tensor([2, 3, 4]));
    expect(Number.isFinite(bt.statistic)).toBe(true);

    expect(() => f_oneway(tensor([1]))).toThrow(/at least 2 groups/);
    expect(() => f_oneway(tensor([]), tensor([1]))).toThrow(/non-empty/);
    expect(() => f_oneway(tensor([1]), tensor([2]))).toThrow(/more than one sample/);
    const f1 = f_oneway(tensor([1, 2, 3]), tensor([2, 3, 4]));
    expect(Number.isFinite(f1.statistic)).toBe(true);
  });
});
