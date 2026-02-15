import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  anderson,
  bartlett,
  chisquare,
  corrcoef,
  cov,
  f_oneway,
  friedmanchisquare,
  geometricMean,
  harmonicMean,
  kendalltau,
  kruskal,
  kstest,
  kurtosis,
  levene,
  mannwhitneyu,
  mean,
  median,
  mode,
  moment,
  normaltest,
  pearsonr,
  percentile,
  quantile,
  shapiro,
  skewness,
  spearmanr,
  std,
  trimMean,
  ttest_1samp,
  ttest_ind,
  ttest_rel,
  variance,
  wilcoxon,
} from "../src/stats";

describe("consumer API: stats", () => {
  const data = tensor([2, 4, 4, 4, 5, 5, 7, 9]);

  describe("descriptive", () => {
    it("mean, std, variance, median", () => {
      expect(Math.abs(Number(mean(data).at()) - 5)).toBeLessThan(0.01);
      expect(Number(std(data).at())).toBeGreaterThan(0);
      expect(Number(variance(data).at())).toBeGreaterThan(0);
      expect(Number(median(data).at())).toBe(4.5);
    });

    it("mode", () => {
      const m = mode(data);
      expect(m.size).toBeGreaterThanOrEqual(1);
      expect(Number(m.at(0))).toBe(4);
    });

    it("quantile, percentile", () => {
      expect(typeof Number(quantile(data, 0.5).at(0))).toBe("number");
      expect(typeof Number(percentile(data, 50).at(0))).toBe("number");
    });

    it("skewness, kurtosis", () => {
      expect(typeof Number(skewness(data).at())).toBe("number");
      expect(typeof Number(kurtosis(data).at())).toBe("number");
    });

    it("geometricMean, harmonicMean", () => {
      const pos = tensor([1, 2, 3, 4, 5]);
      expect(Number(geometricMean(pos).at())).toBeGreaterThan(0);
      expect(Number(harmonicMean(pos).at())).toBeGreaterThan(0);
    });

    it("moment, trimMean", () => {
      const m = moment(data, 2);
      expect(m.size).toBe(1);
      const tm = trimMean(data, 0.1);
      expect(tm.size).toBe(1);
    });
  });

  describe("correlation", () => {
    const x = tensor([1, 2, 3, 4, 5]);
    const y = tensor([2, 4, 5, 4, 5]);

    it("corrcoef, cov", () => {
      const cc = corrcoef(x, y);
      expect(cc.shape[0]).toBe(2);
      expect(cc.shape[1]).toBe(2);
      const cv = cov(x, y);
      expect(cv.shape[0]).toBe(2);
    });

    it("pearsonr, spearmanr, kendalltau", () => {
      const [pr, pp] = pearsonr(x, y);
      expect(typeof pr).toBe("number");
      expect(typeof pp).toBe("number");
      const [sr, _sp] = spearmanr(x, y);
      expect(typeof sr).toBe("number");
      const [kt, _kp] = kendalltau(x, y);
      expect(typeof kt).toBe("number");
    });
  });

  describe("hypothesis tests", () => {
    const a = tensor([5.1, 4.9, 5.0, 5.2, 4.8, 5.1, 5.0, 4.9]);
    const b = tensor([6.1, 5.8, 6.0, 6.3, 5.7, 6.2, 5.9, 6.0]);

    it("ttest_1samp, ttest_ind, ttest_rel", () => {
      const r1 = ttest_1samp(a, 5.0);
      expect(typeof r1.statistic).toBe("number");
      expect(typeof r1.pvalue).toBe("number");
      const r2 = ttest_ind(a, b);
      expect(typeof r2.statistic).toBe("number");
      const r3 = ttest_rel(a, b);
      expect(typeof r3.statistic).toBe("number");
    });

    it("shapiro, anderson, normaltest", () => {
      const sr = shapiro(a);
      expect(typeof sr.statistic).toBe("number");
      const ar = anderson(a);
      expect(typeof ar.statistic).toBe("number");
      const nr = normaltest(a);
      expect(typeof nr.statistic).toBe("number");
    });

    it("f_oneway, chisquare", () => {
      const fr = f_oneway(a, b);
      expect(typeof fr.statistic).toBe("number");
      const observed = tensor([10, 20, 30]);
      const cr = chisquare(observed);
      expect(typeof cr.statistic).toBe("number");
    });

    it("levene, bartlett", () => {
      const lr = levene("mean", a, b);
      expect(typeof lr.statistic).toBe("number");
      const br = bartlett(a, b);
      expect(typeof br.statistic).toBe("number");
    });

    it("mannwhitneyu, kruskal, wilcoxon, friedmanchisquare", () => {
      const mr = mannwhitneyu(a, b);
      expect(typeof mr.statistic).toBe("number");
      const kr = kruskal(a, b);
      expect(typeof kr.statistic).toBe("number");
      const wr = wilcoxon(a, b);
      expect(typeof wr.statistic).toBe("number");
      const c = tensor([7.1, 6.9, 7.0, 7.2, 6.8, 7.1, 7.0, 6.9]);
      const fcr = friedmanchisquare(a, b, c);
      expect(typeof fcr.statistic).toBe("number");
    });

    it("kstest", () => {
      const kr = kstest(a, "norm");
      expect(typeof kr.statistic).toBe("number");
      expect(typeof kr.pvalue).toBe("number");
    });
  });
});
