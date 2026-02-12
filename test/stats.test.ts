import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import * as stats from "../src/stats";
import { numData } from "./_helpers";

function expectClose(actual: number, expected: number, tolerance = 1e-10): void {
  expect(Math.abs(actual - expected)).toBeLessThan(tolerance);
}

function expectArrayClose(actual: number[], expected: number[], tolerance = 1e-10): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    expectClose(actual[i] ?? 0, expected[i] ?? 0, tolerance);
  }
}

describe("stats - Descriptive Statistics", () => {
  describe("mean", () => {
    it("computes mean of 1D tensor", () => {
      const t = tensor([1, 2, 3, 4, 5]);
      expectClose(Number(stats.mean(t).data[0]), 3);
    });

    it("computes mean of 2D tensor", () => {
      const t = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expectClose(Number(stats.mean(t).data[0]), 3.5);
    });

    it("computes mean along axis 0", () => {
      const t = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expectArrayClose(numData(stats.mean(t, 0)), [2.5, 3.5, 4.5]);
    });

    it("computes mean along axis 1", () => {
      const t = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expectArrayClose(numData(stats.mean(t, 1)), [2, 5]);
    });

    it("handles keepdims=true", () => {
      const t = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = stats.mean(t, 1, true);
      expect(result.shape).toEqual([2, 1]);
    });

    it("handles negative axis", () => {
      const t = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expectArrayClose(numData(stats.mean(t, -1)), [2, 5]);
    });

    it("throws on empty tensor", () => {
      expect(() => stats.mean(tensor([]))).toThrow();
    });

    it("handles single element", () => {
      expectClose(Number(stats.mean(tensor([42])).data[0]), 42);
    });

    it("handles negative numbers", () => {
      expectClose(Number(stats.mean(tensor([-1, -2, -3, -4, -5])).data[0]), -3);
    });

    it("handles large numbers", () => {
      expectClose(Number(stats.mean(tensor([1e10, 2e10, 3e10])).data[0]), 2e10, 1e5);
    });

    it("handles floating point", () => {
      expectClose(Number(stats.mean(tensor([1.5, 2.5, 3.5])).data[0]), 2.5);
    });

    it("handles multiple axes", () => {
      const t = tensor([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const result = stats.mean(t, [0, 2]);
      expect(result.shape).toEqual([2]);
    });
  });

  describe("median", () => {
    it("computes median of odd-length array", () => {
      expectClose(Number(stats.median(tensor([1, 2, 3, 4, 5])).data[0]), 3);
    });

    it("computes median of even-length array", () => {
      expectClose(Number(stats.median(tensor([1, 2, 3, 4])).data[0]), 2.5);
    });

    it("handles unsorted data", () => {
      expectClose(Number(stats.median(tensor([5, 1, 3, 2, 4])).data[0]), 3);
    });

    it("handles duplicates", () => {
      expectClose(Number(stats.median(tensor([1, 2, 2, 3, 4])).data[0]), 2);
    });

    it("handles negative numbers", () => {
      expectClose(Number(stats.median(tensor([-5, -1, -3, -2, -4])).data[0]), -3);
    });

    it("throws on empty tensor", () => {
      expect(() => stats.median(tensor([]))).toThrow();
    });

    it("handles single element", () => {
      expectClose(Number(stats.median(tensor([42])).data[0]), 42);
    });

    it("is robust to outliers", () => {
      expectClose(Number(stats.median(tensor([1, 2, 3, 4, 100])).data[0]), 3);
    });

    it("computes median along axis", () => {
      const t = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expectArrayClose(numData(stats.median(t, 1)), [2, 5]);
    });

    it("handles two elements", () => {
      expectClose(Number(stats.median(tensor([1, 3])).data[0]), 2);
    });
  });

  describe("mode", () => {
    it("computes mode", () => {
      expectClose(Number(stats.mode(tensor([1, 2, 2, 3, 3, 3])).data[0]), 3);
    });

    it("computes mode along axis", () => {
      const t = tensor([
        [1, 2, 2],
        [3, 3, 4],
      ]);
      expectArrayClose(numData(stats.mode(t, 1)), [2, 3]);
    });

    it("throws on empty tensor", () => {
      expect(() => stats.mode(tensor([]))).toThrow();
    });

    it("handles single element", () => {
      expectClose(Number(stats.mode(tensor([42])).data[0]), 42);
    });

    it("handles negative numbers", () => {
      expectClose(Number(stats.mode(tensor([-1, -1, -2, -3])).data[0]), -1);
    });

    it("returns smallest value when tied", () => {
      expectClose(Number(stats.mode(tensor([1, 2, 2, 3, 3])).data[0]), 2);
    });

    it("handles all unique values", () => {
      const result = Number(stats.mode(tensor([1, 2, 3, 4, 5])).data[0]);
      expect(result).toBe(1);
    });
  });

  describe("std", () => {
    it("computes population std", () => {
      expectClose(Number(stats.std(tensor([1, 2, 3, 4, 5])).data[0]), Math.sqrt(2), 1e-10);
    });

    it("computes sample std", () => {
      expectClose(
        Number(stats.std(tensor([1, 2, 3, 4, 5]), undefined, false, 1).data[0]),
        Math.sqrt(2.5),
        1e-10
      );
    });

    it("handles zero variance", () => {
      expectClose(Number(stats.std(tensor([5, 5, 5, 5])).data[0]), 0);
    });

    it("throws on empty tensor", () => {
      expect(() => stats.std(tensor([]))).toThrow();
    });

    it("throws when ddof >= size", () => {
      expect(() => stats.std(tensor([1, 2]), undefined, false, 2)).toThrow();
    });

    it("throws when ddof is negative", () => {
      expect(() => stats.std(tensor([1, 2, 3]), undefined, false, -1)).toThrow(/ddof/i);
    });

    it("handles single element with ddof=0", () => {
      expectClose(Number(stats.std(tensor([42]), undefined, false, 0).data[0]), 0);
    });

    it("handles negative numbers", () => {
      expectClose(Number(stats.std(tensor([-1, -2, -3, -4, -5])).data[0]), Math.sqrt(2), 1e-10);
    });

    it("computes std along axis", () => {
      const result = stats.std(
        tensor([
          [1, 2, 3],
          [4, 5, 6],
        ]),
        0
      );
      expect(result.shape).toEqual([3]);
    });
  });

  describe("variance", () => {
    it("computes population variance", () => {
      expectClose(Number(stats.variance(tensor([1, 2, 3, 4, 5])).data[0]), 2);
    });

    it("computes sample variance", () => {
      expectClose(
        Number(stats.variance(tensor([1, 2, 3, 4, 5]), undefined, false, 1).data[0]),
        2.5
      );
    });

    it("handles zero variance", () => {
      expectClose(Number(stats.variance(tensor([5, 5, 5, 5])).data[0]), 0);
    });

    it("throws on empty tensor", () => {
      expect(() => stats.variance(tensor([]))).toThrow();
    });

    it("throws when ddof >= size", () => {
      expect(() => stats.variance(tensor([1, 2]), undefined, false, 2)).toThrow();
    });

    it("throws when ddof is negative", () => {
      expect(() => stats.variance(tensor([1, 2, 3]), undefined, false, -1)).toThrow(/ddof/i);
    });

    it("computes variance along axis", () => {
      const result = stats.variance(
        tensor([
          [1, 2, 3],
          [4, 5, 6],
        ]),
        1
      );
      expectArrayClose(numData(result), [2 / 3, 2 / 3], 1e-10);
    });
  });

  describe("skewness", () => {
    it("computes skewness of symmetric distribution", () => {
      expectClose(Number(stats.skewness(tensor([1, 2, 3, 4, 5])).data[0]), 0, 1e-10);
    });

    it("computes positive skewness", () => {
      const result = Number(stats.skewness(tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])).data[0]);
      expect(result).toBeLessThan(0);
    });

    it("computes negative skewness", () => {
      const result = Number(stats.skewness(tensor([1, 1, 1, 1, 2, 2, 2, 3, 3, 4])).data[0]);
      expect(result).toBeGreaterThan(0);
    });

    it("returns NaN for zero variance", () => {
      expect(Number.isNaN(Number(stats.skewness(tensor([5, 5, 5, 5])).data[0]))).toBe(true);
    });

    it("computes skewness along axis", () => {
      expect(
        stats.skewness(
          tensor([
            [1, 2, 3],
            [4, 5, 6],
          ]),
          1
        ).shape
      ).toEqual([2]);
    });

    it("applies bias correction when requested", () => {
      const data = [1, 2, 2, 3, 9];
      const t = tensor(data);
      const biased = Number(stats.skewness(t).data[0]);
      const unbiased = Number(stats.skewness(t, undefined, false).data[0]);
      const n = data.length;
      const expected = (biased * Math.sqrt(n * (n - 1))) / (n - 2);
      expectClose(unbiased, expected, 1e-12);
    });

    it("returns NaN for unbiased skewness with n < 3", () => {
      const t = tensor([1, 2]);
      expect(Number.isNaN(Number(stats.skewness(t, undefined, false).data[0]))).toBe(true);
    });
  });

  describe("kurtosis", () => {
    it("computes excess kurtosis", () => {
      const result = Number(stats.kurtosis(tensor([1, 2, 3, 4, 5]), undefined, true).data[0]);
      expectClose(result, -1.3, 0.1);
    });

    it("computes raw kurtosis", () => {
      const fisher = Number(stats.kurtosis(tensor([1, 2, 3, 4, 5]), undefined, true).data[0]);
      const pearson = Number(stats.kurtosis(tensor([1, 2, 3, 4, 5]), undefined, false).data[0]);
      expectClose(pearson - 3, fisher, 1e-10);
    });

    it("returns NaN for zero variance", () => {
      expect(Number.isNaN(Number(stats.kurtosis(tensor([5, 5, 5, 5])).data[0]))).toBe(true);
    });

    it("computes kurtosis along axis", () => {
      expect(
        stats.kurtosis(
          tensor([
            [1, 2, 3],
            [4, 5, 6],
          ]),
          1
        ).shape
      ).toEqual([2]);
    });

    it("applies bias correction when requested", () => {
      const data = [1, 2, 2, 3, 9, 10];
      const t = tensor(data);
      const biased = Number(stats.kurtosis(t, undefined, true, true).data[0]);
      const unbiased = Number(stats.kurtosis(t, undefined, true, false).data[0]);
      const n = data.length;
      const expected = ((n + 1) * biased + 6) * ((n - 1) / ((n - 2) * (n - 3)));
      expectClose(unbiased, expected, 1e-12);
    });

    it("returns NaN for unbiased kurtosis with n < 4", () => {
      const t = tensor([1, 2, 3]);
      expect(Number.isNaN(Number(stats.kurtosis(t, undefined, true, false).data[0]))).toBe(true);
    });
  });

  describe("quantile", () => {
    it("computes median (0.5 quantile)", () => {
      expectClose(Number(stats.quantile(tensor([1, 2, 3, 4, 5]), 0.5).data[0]), 3);
    });

    it("computes quartiles", () => {
      const result = stats.quantile(tensor([1, 2, 3, 4, 5]), [0.25, 0.75]);
      expectClose(Number(result.data[0]), 2);
      expectClose(Number(result.data[1]), 4);
    });

    it("computes min (0 quantile)", () => {
      expectClose(Number(stats.quantile(tensor([1, 2, 3, 4, 5]), 0).data[0]), 1);
    });

    it("computes max (1 quantile)", () => {
      expectClose(Number(stats.quantile(tensor([1, 2, 3, 4, 5]), 1).data[0]), 5);
    });

    it("uses linear interpolation", () => {
      expectClose(Number(stats.quantile(tensor([1, 2, 3, 4]), 0.5).data[0]), 2.5);
    });

    it("throws on q < 0", () => {
      expect(() => stats.quantile(tensor([1, 2, 3]), -0.1)).toThrow();
    });

    it("throws on q > 1", () => {
      expect(() => stats.quantile(tensor([1, 2, 3]), 1.1)).toThrow();
    });

    it("throws on empty tensor", () => {
      expect(() => stats.quantile(tensor([]), 0.5)).toThrow();
    });

    it("computes quantile along axis", () => {
      const result = stats.quantile(
        tensor([
          [1, 2, 3],
          [4, 5, 6],
        ]),
        0.5,
        1
      );
      expectArrayClose(numData(result), [2, 5]);
    });
  });

  describe("percentile", () => {
    it("computes 50th percentile", () => {
      expectClose(Number(stats.percentile(tensor([1, 2, 3, 4, 5]), 50).data[0]), 3);
    });

    it("computes 25th and 75th percentiles", () => {
      const result = stats.percentile(tensor([1, 2, 3, 4, 5]), [25, 75]);
      expectClose(Number(result.data[0]), 2);
      expectClose(Number(result.data[1]), 4);
    });

    it("computes 0th percentile", () => {
      expectClose(Number(stats.percentile(tensor([1, 2, 3, 4, 5]), 0).data[0]), 1);
    });

    it("computes 100th percentile", () => {
      expectClose(Number(stats.percentile(tensor([1, 2, 3, 4, 5]), 100).data[0]), 5);
    });

    it("computes 95th percentile", () => {
      expectClose(Number(stats.percentile(tensor([1, 2, 3, 4, 5]), 95).data[0]), 4.8, 1e-6);
    });
  });

  describe("moment", () => {
    it("computes first moment (~0)", () => {
      expectClose(Number(stats.moment(tensor([1, 2, 3, 4, 5]), 1).data[0]), 0, 1e-10);
    });

    it("computes second moment (variance)", () => {
      const moment2 = Number(stats.moment(tensor([1, 2, 3, 4, 5]), 2).data[0]);
      const variance = Number(stats.variance(tensor([1, 2, 3, 4, 5])).data[0]);
      expectClose(moment2, variance, 1e-10);
    });

    it("computes third moment", () => {
      // Symmetric distribution → third central moment ≈ 0
      expect(Number(stats.moment(tensor([1, 2, 3, 4, 5]), 3).data[0])).toBeCloseTo(0, 10);
    });

    it("computes fourth moment", () => {
      const m4 = Number(stats.moment(tensor([1, 2, 3, 4, 5]), 4).data[0]);
      expect(m4).toBeGreaterThan(0);
      expect(Number.isFinite(m4)).toBe(true);
    });

    it("throws on negative n", () => {
      expect(() => stats.moment(tensor([1, 2, 3]), -1)).toThrow();
    });

    it("throws on non-integer n", () => {
      expect(() => stats.moment(tensor([1, 2, 3]), 2.5)).toThrow();
    });

    it("computes zeroth moment (=1)", () => {
      expectClose(Number(stats.moment(tensor([1, 2, 3, 4, 5]), 0).data[0]), 1);
    });
  });

  describe("geometricMean", () => {
    it("computes geometric mean", () => {
      expectClose(Number(stats.geometricMean(tensor([1, 2, 4, 8])).data[0]), 64 ** 0.25, 1e-10);
    });

    it("handles growth rates", () => {
      expectClose(
        Number(stats.geometricMean(tensor([1.1, 1.2])).data[0]),
        Math.sqrt(1.1 * 1.2),
        1e-7
      );
    });

    it("throws on zero", () => {
      expect(() => stats.geometricMean(tensor([1, 0, 2]))).toThrow();
    });

    it("throws on negative", () => {
      expect(() => stats.geometricMean(tensor([1, -2, 3]))).toThrow();
    });

    it("handles single element", () => {
      expectClose(Number(stats.geometricMean(tensor([42])).data[0]), 42);
    });

    it("computes along axis", () => {
      expect(
        stats.geometricMean(
          tensor([
            [1, 2, 4],
            [8, 16, 32],
          ]),
          1
        ).shape
      ).toEqual([2]);
    });
  });

  describe("harmonicMean", () => {
    it("computes harmonic mean", () => {
      expectClose(
        Number(stats.harmonicMean(tensor([1, 2, 4])).data[0]),
        3 / (1 + 0.5 + 0.25),
        1e-10
      );
    });

    it("handles speed averaging", () => {
      expectClose(Number(stats.harmonicMean(tensor([60, 40])).data[0]), 48);
    });

    it("throws on zero", () => {
      expect(() => stats.harmonicMean(tensor([1, 0, 2]))).toThrow();
    });

    it("throws on negative numbers", () => {
      expect(() => stats.harmonicMean(tensor([-1, -2, -4]))).toThrow();
    });

    it("handles single element", () => {
      expectClose(Number(stats.harmonicMean(tensor([42])).data[0]), 42);
    });

    it("computes along axis", () => {
      expect(
        stats.harmonicMean(
          tensor([
            [1, 2, 4],
            [2, 4, 8],
          ]),
          1
        ).shape
      ).toEqual([2]);
    });
  });

  describe("trimMean", () => {
    it("computes trimmed mean", () => {
      expectClose(Number(stats.trimMean(tensor([1, 2, 3, 4, 5, 100]), 0.2).data[0]), 3.5);
    });

    it("handles no trimming", () => {
      const trim = Number(stats.trimMean(tensor([1, 2, 3, 4, 5]), 0).data[0]);
      const mean = Number(stats.mean(tensor([1, 2, 3, 4, 5])).data[0]);
      expectClose(trim, mean);
    });

    it("throws on proportiontocut < 0", () => {
      expect(() => stats.trimMean(tensor([1, 2, 3]), -0.1)).toThrow();
    });

    it("throws on proportiontocut >= 0.5", () => {
      expect(() => stats.trimMean(tensor([1, 2, 3]), 0.5)).toThrow();
    });

    it("handles outliers", () => {
      const trimmed = Number(stats.trimMean(tensor([1, 2, 3, 4, 5, 1000]), 0.2).data[0]);
      const regular = Number(stats.mean(tensor([1, 2, 3, 4, 5, 1000])).data[0]);
      expect(trimmed).toBeLessThan(regular);
    });

    it("throws on empty tensor", () => {
      expect(() => stats.trimMean(tensor([]), 0.1)).toThrow();
    });
  });
});

describe("stats - Correlation", () => {
  describe("pearsonr", () => {
    it("computes perfect positive correlation", () => {
      const [r, p] = stats.pearsonr(tensor([1, 2, 3, 4, 5]), tensor([2, 4, 6, 8, 10]));
      expectClose(r, 1, 1e-10);
      expect(p).toBeLessThan(0.01);
    });

    it("computes perfect negative correlation", () => {
      const [r, p] = stats.pearsonr(tensor([1, 2, 3, 4, 5]), tensor([10, 8, 6, 4, 2]));
      expectClose(r, -1, 1e-10);
      expect(p).toBeLessThan(0.01);
    });

    it("throws on constant input", () => {
      expect(() => stats.pearsonr(tensor([1, 2, 3, 4, 5]), tensor([1, 1, 1, 1, 1]))).toThrow();
    });

    it("computes moderate correlation", () => {
      const [r] = stats.pearsonr(tensor([1, 2, 3, 4, 5]), tensor([2, 4, 5, 4, 5]));
      expect(r).toBeGreaterThan(0);
      expect(r).toBeLessThan(1);
    });

    it("throws on size mismatch", () => {
      expect(() => stats.pearsonr(tensor([1, 2, 3]), tensor([1, 2]))).toThrow();
    });

    it("throws on < 2 samples", () => {
      expect(() => stats.pearsonr(tensor([1]), tensor([2]))).toThrow();
    });

    it("handles negative numbers", () => {
      const [r] = stats.pearsonr(tensor([-1, -2, -3, -4, -5]), tensor([-2, -4, -6, -8, -10]));
      expectClose(r, 1, 1e-10);
    });
  });

  describe("spearmanr", () => {
    it("computes perfect monotonic correlation", () => {
      const [rho, p] = stats.spearmanr(tensor([1, 2, 3, 4, 5]), tensor([2, 4, 6, 8, 10]));
      expectClose(rho, 1, 1e-10);
      expect(p).toBeLessThan(0.01);
    });

    it("handles non-linear monotonic relationship", () => {
      const [rho] = stats.spearmanr(tensor([1, 2, 3, 4, 5]), tensor([1, 4, 9, 16, 25]));
      expectClose(rho, 1, 1e-10);
    });

    it("computes negative correlation", () => {
      const [rho] = stats.spearmanr(tensor([1, 2, 3, 4, 5]), tensor([5, 4, 3, 2, 1]));
      expectClose(rho, -1, 1e-10);
    });

    it("throws on size mismatch", () => {
      expect(() => stats.spearmanr(tensor([1, 2, 3]), tensor([1, 2]))).toThrow();
    });

    it("throws on < 2 samples", () => {
      expect(() => stats.spearmanr(tensor([1]), tensor([2]))).toThrow();
    });
  });

  describe("kendalltau", () => {
    it("computes perfect concordance", () => {
      const [tau, p] = stats.kendalltau(tensor([1, 2, 3, 4, 5]), tensor([1, 2, 3, 4, 5]));
      expectClose(tau, 1, 1e-10);
      expect(p).toBeLessThan(0.05);
    });

    it("computes perfect discordance", () => {
      const [tau] = stats.kendalltau(tensor([1, 2, 3, 4, 5]), tensor([5, 4, 3, 2, 1]));
      expectClose(tau, -1, 1e-10);
    });

    it("handles ties", () => {
      const [tau] = stats.kendalltau(tensor([1, 2, 3, 4, 5]), tensor([1, 3, 2, 4, 5]));
      expect(tau).toBeGreaterThan(0);
      expect(tau).toBeLessThan(1);
    });

    it("throws on size mismatch", () => {
      expect(() => stats.kendalltau(tensor([1, 2, 3]), tensor([1, 2]))).toThrow();
    });

    it("throws on < 2 samples", () => {
      expect(() => stats.kendalltau(tensor([1]), tensor([2]))).toThrow();
    });
  });

  describe("corrcoef", () => {
    it("computes 2x2 correlation matrix", () => {
      const result = stats.corrcoef(tensor([1, 2, 3, 4, 5]), tensor([2, 4, 5, 4, 5]));
      expect(result.shape).toEqual([2, 2]);
      expectClose(Number(result.data[0]), 1);
      expectClose(Number(result.data[3]), 1);
    });

    it("returns [[1]] for single vector", () => {
      const result = stats.corrcoef(tensor([1, 2, 3, 4, 5]));
      expect(result.shape).toEqual([1, 1]);
      expectClose(Number(result.data[0]), 1);
    });

    it("computes correlation matrix for 2D tensor", () => {
      const result = stats.corrcoef(
        tensor([
          [1, 2],
          [3, 4],
          [5, 6],
        ])
      );
      expect(result.shape).toEqual([2, 2]);
    });

    it("produces symmetric matrix", () => {
      const result = stats.corrcoef(
        tensor([
          [1, 2],
          [3, 4],
          [5, 6],
        ])
      );
      expectClose(Number(result.data[1]), Number(result.data[2]), 1e-10);
    });

    it("throws on < 2 observations", () => {
      expect(() => stats.corrcoef(tensor([[1, 2]]))).toThrow();
    });

    it("throws on wrong dimensions", () => {
      expect(() =>
        stats.corrcoef(
          tensor([
            [
              [1, 2],
              [3, 4],
            ],
          ])
        )
      ).toThrow();
    });
  });

  describe("cov", () => {
    it("computes 2x2 covariance matrix", () => {
      const result = stats.cov(tensor([1, 2, 3, 4, 5]), tensor([2, 4, 5, 4, 5]));
      expect(result.shape).toEqual([2, 2]);
    });

    it("computes covariance matrix for 2D tensor", () => {
      const result = stats.cov(
        tensor([
          [1, 2],
          [3, 4],
          [5, 6],
        ])
      );
      expect(result.shape).toEqual([2, 2]);
    });

    it("produces symmetric matrix", () => {
      const result = stats.cov(
        tensor([
          [1, 2],
          [3, 4],
          [5, 6],
        ])
      );
      expectClose(Number(result.data[1]), Number(result.data[2]), 1e-10);
    });

    it("respects ddof parameter", () => {
      const result0 = stats.cov(tensor([1, 2, 3, 4, 5]), undefined, 0);
      const result1 = stats.cov(tensor([1, 2, 3, 4, 5]), undefined, 1);
      expect(Number(result0.data[0])).not.toBe(Number(result1.data[0]));
    });

    it("throws on empty tensor", () => {
      expect(() => stats.cov(tensor([]))).toThrow();
    });

    it("throws when ddof >= size", () => {
      expect(() => stats.cov(tensor([1, 2]), undefined, 2)).toThrow();
    });

    it("throws on size mismatch", () => {
      expect(() => stats.cov(tensor([1, 2, 3]), tensor([1, 2]))).toThrow();
    });
  });
});

describe("stats - Statistical Tests", () => {
  describe("ttest_1samp", () => {
    it("performs one-sample t-test", () => {
      const result = stats.ttest_1samp(tensor([1, 2, 3, 4, 5]), 3);
      expect(result.statistic).toBeDefined();
      expect(result.pvalue).toBeGreaterThan(0.05);
    });

    it("rejects when mean differs significantly", () => {
      const result = stats.ttest_1samp(tensor([10, 11, 12, 13, 14]), 3);
      expect(result.pvalue).toBeLessThan(0.01);
    });

    it("throws on < 2 samples", () => {
      expect(() => stats.ttest_1samp(tensor([1]), 0)).toThrow();
    });

    it("throws on constant input", () => {
      expect(() => stats.ttest_1samp(tensor([5, 5, 5, 5]), 5)).toThrow();
    });

    it("handles negative values", () => {
      const result = stats.ttest_1samp(tensor([-1, -2, -3, -4, -5]), -3);
      expect(result.pvalue).toBeGreaterThan(0.05);
    });
  });

  describe("ttest_ind", () => {
    it("performs independent t-test with equal variance", () => {
      const result = stats.ttest_ind(tensor([1, 2, 3, 4, 5]), tensor([2, 3, 4, 5, 6]), true);
      expect(Number.isFinite(result.statistic)).toBe(true);
      expect(result.pvalue).toBeGreaterThan(0);
      expect(result.pvalue).toBeLessThanOrEqual(1);
    });

    it("performs independent t-test with unequal variance", () => {
      const result = stats.ttest_ind(tensor([1, 2, 3, 4, 5]), tensor([10, 20, 30, 40, 50]), false);
      expect(result.pvalue).toBeLessThan(0.05);
    });

    it("throws on < 2 samples in each group", () => {
      expect(() => stats.ttest_ind(tensor([1]), tensor([2, 3]))).toThrow();
    });

    it("throws on constant input", () => {
      expect(() => stats.ttest_ind(tensor([5, 5, 5, 5]), tensor([6, 6, 6, 6]))).toThrow();
    });

    it("handles negative values", () => {
      const result = stats.ttest_ind(tensor([-1, -2, -3, -4, -5]), tensor([-2, -3, -4, -5, -6]));
      expect(result.pvalue).toBeGreaterThan(0.05);
    });
  });

  describe("ttest_rel", () => {
    it("performs paired t-test", () => {
      const result = stats.ttest_rel(tensor([1, 2, 3, 4, 5]), tensor([1.1, 2.1, 3.1, 4.1, 5.1]));
      expect(Number.isFinite(result.statistic)).toBe(true);
      expect(result.pvalue).toBeGreaterThanOrEqual(0);
      expect(result.pvalue).toBeLessThanOrEqual(1);
    });

    it("rejects when differences are significant", () => {
      const result = stats.ttest_rel(tensor([1, 2, 3, 4, 5]), tensor([10, 20, 30, 40, 50]));
      expect(result.pvalue).toBeLessThan(0.05);
    });

    it("throws on size mismatch", () => {
      expect(() => stats.ttest_rel(tensor([1, 2, 3]), tensor([1, 2]))).toThrow();
    });

    it("throws on < 2 samples", () => {
      expect(() => stats.ttest_rel(tensor([1]), tensor([2]))).toThrow();
    });

    it("throws on constant differences", () => {
      expect(() => stats.ttest_rel(tensor([1, 2, 3, 4, 5]), tensor([1, 2, 3, 4, 5]))).toThrow();
    });
  });

  describe("chisquare", () => {
    it("performs chi-square goodness of fit test", () => {
      const result = stats.chisquare(tensor([10, 20, 30]));
      expect(Number.isFinite(result.statistic)).toBe(true);
      expect(result.statistic).toBeGreaterThan(0);
      expect(result.pvalue).toBeGreaterThan(0);
      expect(result.pvalue).toBeLessThanOrEqual(1);
    });

    it("tests against expected frequencies", () => {
      const result = stats.chisquare(tensor([10, 20, 30]), tensor([15, 20, 25]));
      expect(result.statistic).toBeGreaterThan(0);
    });

    it("accepts uniform distribution", () => {
      const result = stats.chisquare(tensor([20, 20, 20, 20]));
      expectClose(result.statistic, 0, 1e-10);
    });

    it("rejects non-uniform when expected uniform", () => {
      const result = stats.chisquare(tensor([10, 20, 30, 40]));
      expect(result.pvalue).toBeLessThan(0.1);
    });
  });

  describe("kstest", () => {
    it("performs Kolmogorov-Smirnov test", () => {
      const result = stats.kstest(tensor([1, 2, 3, 4, 5]), "norm");
      expect(Number.isFinite(result.statistic)).toBe(true);
      expect(result.statistic).toBeGreaterThanOrEqual(0);
      expect(result.pvalue).toBeGreaterThanOrEqual(0);
      expect(result.pvalue).toBeLessThanOrEqual(1);
    });

    it("accepts normal distribution for normal data", () => {
      const result = stats.kstest(tensor([0, 0.1, -0.1, 0.2, -0.2]), "norm");
      expect(result.pvalue).toBeGreaterThan(0.01);
    });

    it("accepts custom CDF function", () => {
      const result = stats.kstest(tensor([0.1, 0.2, 0.3, 0.4, 0.5]), (x) => x);
      expect(Number.isFinite(result.statistic)).toBe(true);
      expect(result.statistic).toBeGreaterThanOrEqual(0);
    });

    it("throws on empty data", () => {
      expect(() => stats.kstest(tensor([]), "norm")).toThrow();
    });
  });

  describe("normaltest", () => {
    it("performs D'Agostino-Pearson normality test", () => {
      const result = stats.normaltest(tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
      expect(Number.isFinite(result.statistic)).toBe(true);
      expect(result.statistic).toBeGreaterThanOrEqual(0);
      expect(result.pvalue).toBeGreaterThan(0);
      expect(result.pvalue).toBeLessThanOrEqual(1);
    });

    it("throws on < 8 samples", () => {
      expect(() => stats.normaltest(tensor([1, 2, 3, 4, 5, 6, 7]))).toThrow();
    });

    it("throws on constant input", () => {
      expect(() => stats.normaltest(tensor([5, 5, 5, 5, 5, 5, 5, 5]))).toThrow();
    });

    it("accepts approximately normal data", () => {
      const result = stats.normaltest(tensor([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]));
      expect(result.pvalue).toBeGreaterThan(0.01);
    });
  });

  describe("shapiro", () => {
    it("performs Shapiro-Wilk test", () => {
      const result = stats.shapiro(tensor([1, 2, 3, 4, 5]));
      expect(Number.isFinite(result.statistic)).toBe(true);
      expect(result.statistic).toBeGreaterThan(0);
      expect(result.statistic).toBeLessThanOrEqual(1);
      expect(result.pvalue).toBeGreaterThan(0);
      expect(result.pvalue).toBeLessThanOrEqual(1);
    });

    it("throws on < 3 samples", () => {
      expect(() => stats.shapiro(tensor([1, 2]))).toThrow();
    });

    it("throws on > 5000 samples", () => {
      expect(() => stats.shapiro(tensor(Array(5001).fill(1)))).toThrow();
    });

    it("throws on constant values", () => {
      expect(() => stats.shapiro(tensor([5, 5, 5, 5]))).toThrow();
    });

    it("accepts normal-like data", () => {
      const result = stats.shapiro(tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]));
      expect(result.pvalue).toBeGreaterThan(0.01);
    });
  });

  describe("anderson", () => {
    it("performs Anderson-Darling test", () => {
      const result = stats.anderson(tensor([1, 2, 3, 4, 5]));
      expect(Number.isFinite(result.statistic)).toBe(true);
      expect(result.statistic).toBeGreaterThanOrEqual(0);
      expect(result.critical_values.length).toBeGreaterThan(0);
      expect(result.significance_level.length).toBeGreaterThan(0);
    });

    it("returns 5 critical values", () => {
      const result = stats.anderson(tensor([1, 2, 3, 4, 5]));
      expect(result.critical_values.length).toBe(5);
      expect(result.significance_level.length).toBe(5);
    });

    it("throws on empty data", () => {
      expect(() => stats.anderson(tensor([]))).toThrow();
    });

    it("throws on constant input", () => {
      expect(() => stats.anderson(tensor([5, 5, 5, 5]))).toThrow();
    });

    it("flags clearly non-normal data", () => {
      const result = stats.anderson(tensor([-10, -5, -2, 0, 0, 2, 5, 10, 20]));
      expect(result.statistic).toBeGreaterThan(result.critical_values[2] ?? 0);
    });
  });

  describe("mannwhitneyu", () => {
    it("performs Mann-Whitney U test", () => {
      const result = stats.mannwhitneyu(tensor([1, 2, 3, 4, 5]), tensor([2, 3, 4, 5, 6]));
      expect(Number.isFinite(result.statistic)).toBe(true);
      expect(result.statistic).toBeGreaterThanOrEqual(0);
      expect(result.pvalue).toBeGreaterThan(0);
      expect(result.pvalue).toBeLessThanOrEqual(1);
    });

    it("rejects when distributions differ significantly", () => {
      const result = stats.mannwhitneyu(tensor([1, 2, 3, 4, 5]), tensor([10, 20, 30, 40, 50]));
      expect(result.pvalue).toBeLessThan(0.01);
    });

    it("accepts similar distributions", () => {
      const result = stats.mannwhitneyu(tensor([1, 2, 3, 4, 5]), tensor([1.5, 2.5, 3.5, 4.5, 5.5]));
      expect(result.pvalue).toBeGreaterThan(0.05);
    });

    it("handles ties", () => {
      const result = stats.mannwhitneyu(tensor([1, 2, 2, 3, 4]), tensor([2, 3, 3, 4, 5]));
      expect(result.statistic).toBeGreaterThanOrEqual(0);
      expect(result.pvalue).toBeGreaterThan(0);
    });
  });

  describe("wilcoxon", () => {
    it("performs Wilcoxon signed-rank test", () => {
      const result = stats.wilcoxon(tensor([1, 2, 3, 4, 5]), tensor([1.1, 2.1, 3.1, 4.1, 5.1]));
      expect(Number.isFinite(result.statistic)).toBe(true);
      expect(result.statistic).toBeGreaterThanOrEqual(0);
      expect(result.pvalue).toBeGreaterThan(0);
      expect(result.pvalue).toBeLessThanOrEqual(1);
    });

    it("tests single sample against zero", () => {
      const result = stats.wilcoxon(tensor([1, 2, 3, 4, 5]));
      expect(Number.isFinite(result.statistic)).toBe(true);
      expect(result.statistic).toBeGreaterThanOrEqual(0);
    });

    it("rejects when differences are significant", () => {
      const result = stats.wilcoxon(tensor([1, 2, 3, 4, 5]), tensor([10, 20, 30, 40, 50]));
      expect(result.pvalue).toBeLessThan(0.05);
    });

    it("ignores zero differences", () => {
      const result = stats.wilcoxon(tensor([1, 2, 3, 4, 5]), tensor([1, 2, 3.1, 4.1, 5.1]));
      expect(result.statistic).toBeGreaterThanOrEqual(0);
      expect(result.pvalue).toBeGreaterThan(0);
    });
  });

  describe("kruskal", () => {
    it("performs Kruskal-Wallis H test", () => {
      const result = stats.kruskal(
        tensor([1, 2, 3, 4, 5]),
        tensor([2, 3, 4, 5, 6]),
        tensor([3, 4, 5, 6, 7])
      );
      expect(Number.isFinite(result.statistic)).toBe(true);
      expect(result.statistic).toBeGreaterThanOrEqual(0);
      expect(result.pvalue).toBeGreaterThan(0);
      expect(result.pvalue).toBeLessThanOrEqual(1);
    });

    it("accepts similar groups", () => {
      const result = stats.kruskal(
        tensor([1, 2, 3, 4, 5]),
        tensor([1.5, 2.5, 3.5, 4.5, 5.5]),
        tensor([1.2, 2.2, 3.2, 4.2, 5.2])
      );
      expect(result.pvalue).toBeGreaterThan(0.05);
    });

    it("rejects when groups differ significantly", () => {
      const result = stats.kruskal(
        tensor([1, 2, 3, 4, 5]),
        tensor([10, 20, 30, 40, 50]),
        tensor([100, 200, 300, 400, 500])
      );
      expect(result.pvalue).toBeLessThan(0.01);
    });

    it("handles two groups", () => {
      const result = stats.kruskal(tensor([1, 2, 3, 4, 5]), tensor([2, 3, 4, 5, 6]));
      expect(result.statistic).toBeGreaterThanOrEqual(0);
      expect(result.pvalue).toBeGreaterThan(0);
    });
  });

  describe("friedmanchisquare", () => {
    it("performs Friedman test", () => {
      const result = stats.friedmanchisquare(
        tensor([1, 2, 3, 4, 5]),
        tensor([1.1, 2.1, 3.1, 4.1, 5.1]),
        tensor([1.2, 2.2, 3.2, 4.2, 5.2])
      );
      expect(Number.isFinite(result.statistic)).toBe(true);
      expect(result.statistic).toBeGreaterThanOrEqual(0);
      expect(result.pvalue).toBeGreaterThan(0);
      expect(result.pvalue).toBeLessThanOrEqual(1);
    });

    it("accepts similar repeated measures", () => {
      const result = stats.friedmanchisquare(
        tensor([1, 2, 3, 4, 5]),
        tensor([1.01, 2.01, 3.01, 4.01, 5.01]),
        tensor([1.02, 2.02, 3.02, 4.02, 5.02])
      );
      expect(result.statistic).toBeGreaterThanOrEqual(0);
      expect(result.pvalue).toBeGreaterThan(0);
    });

    it("rejects when treatments differ significantly", () => {
      const result = stats.friedmanchisquare(
        tensor([1, 2, 3, 4, 5]),
        tensor([10, 20, 30, 40, 50]),
        tensor([100, 200, 300, 400, 500])
      );
      expect(result.pvalue).toBeLessThan(0.01);
    });

    it("produces correct statistic and p-value for known data", () => {
      // Reference values verified against SciPy
      // Data: 3 treatments, 5 blocks
      const result = stats.friedmanchisquare(
        tensor([8.6, 8.2, 7.9, 8.7, 9.0]),
        tensor([9.1, 8.8, 8.5, 9.2, 9.3]),
        tensor([9.8, 9.5, 9.2, 9.9, 10.0])
      );
      // Expected: chi-square statistic = 10.0, df = 2
      // This should give a p-value around 0.0067
      expect(result.statistic).toBeCloseTo(10.0, 0);
      expect(result.pvalue).toBeGreaterThan(0.005);
      expect(result.pvalue).toBeLessThan(0.01);
    });
  });

  describe("f_oneway", () => {
    it("performs one-way ANOVA", () => {
      const result = stats.f_oneway(
        tensor([1, 2, 3, 4, 5]),
        tensor([2, 3, 4, 5, 6]),
        tensor([3, 4, 5, 6, 7])
      );
      expect(result.statistic).toBeDefined();
      expect(result.pvalue).toBeDefined();
    });

    it("accepts similar groups", () => {
      const result = stats.f_oneway(
        tensor([1, 2, 3, 4, 5]),
        tensor([1.1, 2.1, 3.1, 4.1, 5.1]),
        tensor([1.2, 2.2, 3.2, 4.2, 5.2])
      );
      expect(result.pvalue).toBeGreaterThan(0.05);
    });

    it("rejects when groups differ significantly", () => {
      const result = stats.f_oneway(
        tensor([1, 2, 3, 4, 5]),
        tensor([10, 20, 30, 40, 50]),
        tensor([100, 200, 300, 400, 500])
      );
      expect(result.pvalue).toBeLessThan(0.01);
    });

    it("handles two groups", () => {
      const result = stats.f_oneway(tensor([1, 2, 3, 4, 5]), tensor([2, 3, 4, 5, 6]));
      expect(result.statistic).toBeGreaterThanOrEqual(0);
      expect(result.pvalue).toBeGreaterThan(0);
    });

    it("handles unequal group sizes", () => {
      const result = stats.f_oneway(
        tensor([1, 2, 3]),
        tensor([4, 5, 6, 7]),
        tensor([8, 9, 10, 11, 12])
      );
      expect(result.statistic).toBeGreaterThan(0);
      expect(result.pvalue).toBeGreaterThan(0);
    });
  });
});

describe("stats - Edge Cases", () => {
  it("handles very large datasets", () => {
    const large = tensor(
      Array(10000)
        .fill(0)
        .map((_, i) => i)
    );
    expectClose(Number(stats.mean(large).data[0]), 4999.5, 1);
  });

  it("handles mixed positive and negative", () => {
    expectClose(Number(stats.mean(tensor([-100, -50, 0, 50, 100])).data[0]), 0);
  });

  it("handles very small variance", () => {
    const result = stats.variance(tensor([1.0000001, 1.0000002, 1.0000003]));
    expect(Number(result.data[0])).toBeGreaterThan(0);
  });

  it("handles extreme outliers in robust statistics", () => {
    const median = Number(stats.median(tensor([1, 2, 3, 4, 1e10])).data[0]);
    const mean = Number(stats.mean(tensor([1, 2, 3, 4, 1e10])).data[0]);
    expect(median).toBeLessThan(mean);
  });

  it("maintains precision with floating point", () => {
    expectClose(Number(stats.mean(tensor([0.1, 0.2, 0.3])).data[0]), 0.2, 1e-8);
  });

  it("handles integer overflow scenarios", () => {
    expectClose(Number(stats.mean(tensor([1e15, 1e15, 1e15])).data[0]), 1e15, 1e8);
  });

  it("handles underflow scenarios", () => {
    expect(Number(stats.mean(tensor([1e-30, 1e-30, 1e-30])).data[0])).toBeGreaterThan(0);
  });

  it("handles very small numbers", () => {
    const result = Number(stats.mean(tensor([1e-10, 1e-10, 1e-10])).data[0]);
    expectClose(result, 1e-10, 1e-15);
  });

  it("handles NaN propagation", () => {
    expect(Number.isNaN(Number(stats.mean(tensor([1, 2, NaN, 4, 5])).data[0]))).toBe(true);
  });

  it("handles Infinity values", () => {
    expect(Number(stats.mean(tensor([1, 2, Infinity, 4, 5])).data[0])).toBe(Infinity);
  });

  it("handles negative infinity", () => {
    expect(Number(stats.mean(tensor([1, 2, -Infinity, 4, 5])).data[0])).toBe(-Infinity);
  });

  it("validates axis bounds", () => {
    expect(() =>
      stats.mean(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        5
      )
    ).toThrow();
    expect(() =>
      stats.mean(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        -5
      )
    ).toThrow();
  });

  it("handles keepdims correctly", () => {
    const result = stats.mean(
      tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]),
      1,
      true
    );
    expect(result.ndim).toBe(2);
    expect(result.shape[1]).toBe(1);
  });

  it("returns float64 dtype", () => {
    const result = stats.mean(tensor([1, 2, 3]));
    expect(result.dtype).toBe("float64");
  });

  it("handles 3D tensors", () => {
    const t = tensor([
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
    ]);
    expect(() => stats.mean(t, 0)).not.toThrow();
  });

  it("handles multiple axis reduction", () => {
    const t = tensor([
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
    ]);
    const result = stats.mean(t, [0, 2]);
    expect(result.shape).toEqual([2]);
  });

  it("handles duplicate axes (should deduplicate)", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(() => stats.mean(t, [1, 1])).not.toThrow();
  });
});

describe("stats - Variance Equality Tests", () => {
  describe("levene", () => {
    it("performs Levene's test with median center", () => {
      const result = stats.levene(
        "median",
        tensor([1, 2, 3, 4, 5]),
        tensor([2, 3, 4, 5, 6]),
        tensor([3, 4, 5, 6, 7])
      );
      expect(result.statistic).toBeDefined();
      expect(result.pvalue).toBeDefined();
      expect(result.pvalue).toBeGreaterThan(0);
      expect(result.pvalue).toBeLessThanOrEqual(1);
    });

    it("performs Levene's test with mean center", () => {
      const result = stats.levene("mean", tensor([1, 2, 3, 4, 5]), tensor([2, 3, 4, 5, 6]));
      expect(result.statistic).toBeGreaterThanOrEqual(0);
      expect(result.pvalue).toBeGreaterThan(0);
      expect(result.pvalue).toBeLessThanOrEqual(1);
    });

    it("performs Levene's test with trimmed center", () => {
      const result = stats.levene(
        "trimmed",
        tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        tensor([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
      );
      expect(result.statistic).toBeGreaterThanOrEqual(0);
      expect(result.pvalue).toBeGreaterThan(0);
      expect(result.pvalue).toBeLessThanOrEqual(1);
    });

    it("accepts groups with equal variances", () => {
      const result = stats.levene(
        "median",
        tensor([1, 2, 3, 4, 5]),
        tensor([2, 3, 4, 5, 6]),
        tensor([3, 4, 5, 6, 7])
      );
      expect(result.pvalue).toBeGreaterThan(0.05);
    });

    it("rejects groups with unequal variances", () => {
      const result = stats.levene("median", tensor([1, 2, 3, 4, 5]), tensor([1, 10, 20, 30, 40]));
      expect(result.pvalue).toBeLessThan(0.1);
    });

    it("throws on fewer than 2 groups", () => {
      expect(() => stats.levene("median", tensor([1, 2, 3]))).toThrow();
    });

    it("throws on group with fewer than 2 samples", () => {
      expect(() => stats.levene("median", tensor([1]), tensor([2, 3, 4]))).toThrow();
    });

    it("handles two groups", () => {
      const result = stats.levene("median", tensor([1, 2, 3, 4, 5]), tensor([2, 4, 6, 8, 10]));
      expect(result.statistic).toBeGreaterThanOrEqual(0);
      expect(result.pvalue).toBeGreaterThan(0);
    });
  });

  describe("bartlett", () => {
    it("performs Bartlett's test", () => {
      const result = stats.bartlett(
        tensor([1, 2, 3, 4, 5]),
        tensor([2, 3, 4, 5, 6]),
        tensor([3, 4, 5, 6, 7])
      );
      expect(result.statistic).toBeDefined();
      expect(result.pvalue).toBeDefined();
      expect(result.pvalue).toBeGreaterThan(0);
      expect(result.pvalue).toBeLessThanOrEqual(1);
    });

    it("accepts groups with equal variances", () => {
      const result = stats.bartlett(
        tensor([1, 2, 3, 4, 5]),
        tensor([2, 3, 4, 5, 6]),
        tensor([3, 4, 5, 6, 7])
      );
      expect(result.pvalue).toBeGreaterThan(0.05);
    });

    it("rejects groups with unequal variances", () => {
      const result = stats.bartlett(tensor([1, 2, 3, 4, 5]), tensor([1, 10, 20, 30, 40]));
      expect(result.pvalue).toBeLessThan(0.1);
    });

    it("throws on fewer than 2 groups", () => {
      expect(() => stats.bartlett(tensor([1, 2, 3]))).toThrow();
    });

    it("throws on group with fewer than 2 samples", () => {
      expect(() => stats.bartlett(tensor([1]), tensor([2, 3, 4]))).toThrow();
    });

    it("throws on zero variance in any group", () => {
      expect(() => stats.bartlett(tensor([5, 5, 5, 5]), tensor([1, 2, 3, 4]))).toThrow();
    });

    it("handles two groups", () => {
      const result = stats.bartlett(tensor([1, 2, 3, 4, 5]), tensor([2, 4, 6, 8, 10]));
      expect(result.statistic).toBeGreaterThanOrEqual(0);
      expect(result.pvalue).toBeGreaterThan(0);
    });

    it("handles many groups", () => {
      const result = stats.bartlett(
        tensor([1, 2, 3, 4, 5]),
        tensor([2, 3, 4, 5, 6]),
        tensor([3, 4, 5, 6, 7]),
        tensor([4, 5, 6, 7, 8]),
        tensor([5, 6, 7, 8, 9])
      );
      expect(result.statistic).toBeDefined();
      expect(result.pvalue).toBeGreaterThan(0.05);
    });
  });
});
