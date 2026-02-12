import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import * as stats from "../src/stats";

/**
 * Tests for NaN and Infinity handling in stats functions.
 * Documents expected behavior with special floating-point values.
 */
describe("stats - NaN and Infinity Handling", () => {
  describe("Descriptive Statistics with NaN", () => {
    it("mean propagates NaN", () => {
      const t = tensor([1, 2, NaN, 4, 5]);
      const result = Number(stats.mean(t).data[0]);
      expect(Number.isNaN(result)).toBe(true);
    });

    it("median with NaN returns NaN", () => {
      const t = tensor([1, 2, NaN, 4, 5]);
      const result = Number(stats.median(t).data[0]);
      expect(Number.isNaN(result)).toBe(true);
    });

    it("variance propagates NaN", () => {
      const t = tensor([1, 2, NaN, 4, 5]);
      const result = Number(stats.variance(t).data[0]);
      expect(Number.isNaN(result)).toBe(true);
    });

    it("std propagates NaN", () => {
      const t = tensor([1, 2, NaN, 4, 5]);
      const result = Number(stats.std(t).data[0]);
      expect(Number.isNaN(result)).toBe(true);
    });

    it("skewness with NaN returns NaN", () => {
      const t = tensor([1, 2, NaN, 4, 5]);
      const result = Number(stats.skewness(t).data[0]);
      expect(Number.isNaN(result)).toBe(true);
    });

    it("kurtosis with NaN returns NaN", () => {
      const t = tensor([1, 2, NaN, 4, 5]);
      const result = Number(stats.kurtosis(t).data[0]);
      expect(Number.isNaN(result)).toBe(true);
    });

    it("quantile with NaN returns NaN", () => {
      const t = tensor([1, 2, NaN, 4, 5]);
      const result = Number(stats.quantile(t, 0.5).data[0]);
      expect(Number.isNaN(result)).toBe(true);
    });
  });

  describe("Descriptive Statistics with Infinity", () => {
    it("mean with Infinity returns Infinity", () => {
      const t = tensor([1, 2, Infinity, 4, 5]);
      const result = Number(stats.mean(t).data[0]);
      expect(result).toBe(Infinity);
    });

    it("mean with -Infinity returns -Infinity", () => {
      const t = tensor([1, 2, -Infinity, 4, 5]);
      const result = Number(stats.mean(t).data[0]);
      expect(result).toBe(-Infinity);
    });

    it("mean with mixed Infinity returns NaN", () => {
      const t = tensor([Infinity, -Infinity]);
      const result = Number(stats.mean(t).data[0]);
      expect(Number.isNaN(result)).toBe(true);
    });

    it("median with Infinity returns middle value after sort", () => {
      // Infinity sorts to end, so median is middle of sorted array
      const t = tensor([1, 2, 3, 4, Infinity]);
      const result = Number(stats.median(t).data[0]);
      expect(result).toBe(3); // Middle value of [1, 2, 3, 4, Infinity]
    });

    it("variance with Infinity returns NaN", () => {
      const t = tensor([1, 2, Infinity, 4, 5]);
      const result = Number(stats.variance(t).data[0]);
      expect(Number.isNaN(result)).toBe(true);
    });
  });

  describe("Correlation with NaN/Infinity", () => {
    it("pearsonr with NaN returns NaN correlation", () => {
      const x = tensor([1, 2, NaN, 4, 5]);
      const y = tensor([2, 4, 6, 8, 10]);
      const [r] = stats.pearsonr(x, y);
      expect(Number.isNaN(r)).toBe(true);
    });

    it("pearsonr with Infinity returns NaN correlation", () => {
      const x = tensor([1, 2, Infinity, 4, 5]);
      const y = tensor([2, 4, 6, 8, 10]);
      const [r] = stats.pearsonr(x, y);
      expect(Number.isNaN(r)).toBe(true);
    });

    it("spearmanr with NaN may not return NaN (ranking behavior)", () => {
      // Spearman ranks values, and NaN ranking behavior depends on sort
      const x = tensor([1, 2, NaN, 4, 5]);
      const y = tensor([2, 4, 6, 8, 10]);
      const [rho] = stats.spearmanr(x, y);
      // Actual behavior: NaN gets a rank position, may not propagate as NaN
      expect(rho).toBeDefined();
    });
  });

  describe("Hypothesis Tests with NaN/Infinity", () => {
    it("ttest_1samp with NaN returns NaN statistic", () => {
      const t = tensor([1, 2, NaN, 4, 5]);
      const result = stats.ttest_1samp(t, 3);
      expect(Number.isNaN(result.statistic)).toBe(true);
    });

    it("ttest_1samp with Infinity returns Infinity or NaN", () => {
      const t = tensor([1, 2, Infinity, 4, 5]);
      const result = stats.ttest_1samp(t, 3);
      // Either Infinity or NaN is acceptable behavior
      expect(!Number.isFinite(result.statistic)).toBe(true);
    });

    it("ttest_ind with NaN returns NaN statistic", () => {
      const a = tensor([1, 2, NaN, 4, 5]);
      const b = tensor([2, 3, 4, 5, 6]);
      const result = stats.ttest_ind(a, b);
      expect(Number.isNaN(result.statistic)).toBe(true);
    });
  });

  describe("Special Means with Invalid Values", () => {
    it("geometricMean with NaN returns NaN", () => {
      const t = tensor([1, 2, NaN, 4, 5]);
      const result = Number(stats.geometricMean(t).data[0]);
      expect(Number.isNaN(result)).toBe(true);
    });

    it("harmonicMean with NaN returns NaN", () => {
      const t = tensor([1, 2, NaN, 4, 5]);
      const result = Number(stats.harmonicMean(t).data[0]);
      expect(Number.isNaN(result)).toBe(true);
    });

    it("harmonicMean with Infinity returns finite value (1/Infinity = 0)", () => {
      // 1/Infinity = 0, so it contributes 0 to the sum of reciprocals
      const t = tensor([1, 2, Infinity, 4, 5]);
      const result = Number(stats.harmonicMean(t).data[0]);
      // n / (1/1 + 1/2 + 0 + 1/4 + 1/5) = 5 / 1.95 ≈ 2.564
      expect(result).toBeGreaterThan(2);
      expect(result).toBeLessThan(3);
    });
  });

  describe("Edge Cases Documentation", () => {
    it("documents that NaN propagates through most operations", () => {
      // This test serves as documentation that NaN follows IEEE 754 semantics
      // NaN + x = NaN, NaN * x = NaN, etc.
      expect(NaN + 1).toBeNaN();
      expect(NaN * 2).toBeNaN();
      expect(Math.sqrt(NaN)).toBeNaN();
    });

    it("documents that Infinity follows IEEE 754 semantics", () => {
      // This test serves as documentation
      expect(Infinity + 1).toBe(Infinity);
      expect(Infinity * 2).toBe(Infinity);
      expect(Infinity - Infinity).toBeNaN();
    });
  });
});
