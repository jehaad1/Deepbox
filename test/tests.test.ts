import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  anderson,
  chisquare,
  f_oneway,
  friedmanchisquare,
  kruskal,
  kstest,
  mannwhitneyu,
  normaltest,
  shapiro,
  ttest_1samp,
  ttest_ind,
  ttest_rel,
  wilcoxon,
} from "../src/stats";

describe("T-Tests", () => {
  it("should perform one-sample t-test", () => {
    const data = tensor([1, 2, 3, 4, 5]);
    const result = ttest_1samp(data, 3);
    expect(Number.isFinite(result.statistic)).toBe(true);
    // Mean equals popmean, so statistic should be ~0 and p-value > 0.05
    expect(Math.abs(result.statistic)).toBeLessThan(1e-10);
    expect(result.pvalue).toBeGreaterThan(0.05);
  });

  it("should perform independent t-test", () => {
    const a = tensor([1, 2, 3, 4, 5]);
    const b = tensor([2, 3, 4, 5, 6]);
    const result = ttest_ind(a, b);
    expect(Number.isFinite(result.statistic)).toBe(true);
    expect(result.statistic).toBeLessThan(0);
    expect(result.pvalue).toBeGreaterThan(0);
    expect(result.pvalue).toBeLessThanOrEqual(1);
  });

  it("should perform paired t-test", () => {
    const a = tensor([1, 2, 3, 4, 5]);
    const b = tensor([1.1, 2.2, 2.9, 4.1, 5.2]);
    const result = ttest_rel(a, b);
    expect(Number.isFinite(result.statistic)).toBe(true);
    expect(result.pvalue).toBeGreaterThan(0);
    expect(result.pvalue).toBeLessThanOrEqual(1);
  });
});

describe("Chi-Square and KS Tests", () => {
  it("should perform chi-square test", () => {
    const observed = tensor([10, 20, 30]);
    const result = chisquare(observed);
    expect(result.statistic).toBeGreaterThan(0);
    expect(result.pvalue).toBeGreaterThan(0);
    expect(result.pvalue).toBeLessThanOrEqual(1);
  });

  it("should perform Kolmogorov-Smirnov test", () => {
    const data = tensor([1, 2, 3, 4, 5]);
    const result = kstest(data, "norm");
    expect(result.statistic).toBeGreaterThanOrEqual(0);
    expect(result.statistic).toBeLessThanOrEqual(1);
    expect(result.pvalue).toBeGreaterThan(0);
    expect(result.pvalue).toBeLessThanOrEqual(1);
  });
});

describe("Normality Tests", () => {
  it("should perform normality test", () => {
    const data = tensor([1, 2, 3, 4, 5, 6, 7, 8]);
    const result = normaltest(data);
    expect(result.statistic).toBeGreaterThanOrEqual(0);
    expect(result.pvalue).toBeGreaterThan(0);
    expect(result.pvalue).toBeLessThanOrEqual(1);
  });

  it("should perform Shapiro-Wilk test", () => {
    const data = tensor([1, 2, 3, 4, 5]);
    const result = shapiro(data);
    expect(result.statistic).toBeGreaterThan(0);
    expect(result.statistic).toBeLessThanOrEqual(1);
    expect(result.pvalue).toBeGreaterThan(0);
    expect(result.pvalue).toBeLessThanOrEqual(1);
  });

  it("should perform Anderson-Darling test", () => {
    const data = tensor([1, 2, 3, 4, 5]);
    const result = anderson(data);
    expect(result.statistic).toBeGreaterThanOrEqual(0);
    expect(result.critical_values.length).toBe(5);
    expect(result.significance_level.length).toBe(5);
  });
});

describe("Non-Parametric Tests", () => {
  it("should perform Mann-Whitney U test", () => {
    const x = tensor([1, 2, 3, 4, 5]);
    const y = tensor([2, 3, 4, 5, 6]);
    const result = mannwhitneyu(x, y);
    expect(result.statistic).toBeGreaterThanOrEqual(0);
    expect(result.pvalue).toBeGreaterThan(0);
    expect(result.pvalue).toBeLessThanOrEqual(1);
  });

  it("should perform Wilcoxon test", () => {
    const x = tensor([1, 2, 3, 4, 5]);
    const result = wilcoxon(x);
    expect(result.statistic).toBeGreaterThanOrEqual(0);
    expect(result.pvalue).toBeGreaterThan(0);
    expect(result.pvalue).toBeLessThanOrEqual(1);
  });
});

describe("ANOVA Tests", () => {
  it("should perform Kruskal-Wallis test", () => {
    const sample1 = tensor([1, 2, 3]);
    const sample2 = tensor([4, 5, 6]);
    const sample3 = tensor([7, 8, 9]);
    const result = kruskal(sample1, sample2, sample3);
    expect(result.statistic).toBeGreaterThan(0);
    expect(result.pvalue).toBeGreaterThan(0);
    expect(result.pvalue).toBeLessThanOrEqual(1);
  });

  it("should perform Friedman test", () => {
    const sample1 = tensor([1, 2, 3]);
    const sample2 = tensor([1, 3, 2]);
    const sample3 = tensor([2, 1, 3]);
    const result = friedmanchisquare(sample1, sample2, sample3);
    expect(result.statistic).toBeGreaterThanOrEqual(0);
    expect(result.pvalue).toBeGreaterThan(0);
    expect(result.pvalue).toBeLessThanOrEqual(1);
  });

  it("should perform one-way ANOVA", () => {
    const sample1 = tensor([1, 2, 3]);
    const sample2 = tensor([4, 5, 6]);
    const result = f_oneway(sample1, sample2);
    expect(result.statistic).toBeGreaterThan(0);
    expect(result.pvalue).toBeGreaterThan(0);
    expect(result.pvalue).toBeLessThanOrEqual(1);
  });
});
