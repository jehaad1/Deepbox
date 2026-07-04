import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  benjaminiHochberg,
  bonferroni,
  bootstrap,
  cohenD,
  holm,
  iqr,
  partialcorr,
  pointbiserialr,
  sem,
  sidak,
  zscore,
} from "../src/stats";

describe("zscore", () => {
  it("computes z-scores for a 1-D tensor (ddof=0)", () => {
    const t = tensor([1, 2, 3, 4, 5]);
    const z = zscore(t);
    const arr = z.toArray() as number[];
    // mean=3, std=sqrt(2), z = (x-3)/sqrt(2)
    expect(arr[0]).toBeCloseTo((1 - 3) / Math.sqrt(2), 10);
    expect(arr[2]).toBeCloseTo(0, 10);
    expect(arr[4]).toBeCloseTo((5 - 3) / Math.sqrt(2), 10);
  });

  it("computes z-scores with ddof=1", () => {
    const t = tensor([2, 4, 4, 4, 5, 5, 7, 9]);
    const z = zscore(t, 1);
    const arr = z.toArray() as number[];
    // mean=5, std with ddof=1 = sqrt(sum((x-5)^2)/7)
    const m = 5;
    const ss = [2, 4, 4, 4, 5, 5, 7, 9].reduce((s, v) => s + (v - m) ** 2, 0);
    const s = Math.sqrt(ss / 7);
    expect(arr[0]).toBeCloseTo((2 - m) / s, 10);
    expect(arr[7]).toBeCloseTo((9 - m) / s, 10);
  });

  it("preserves shape", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const z = zscore(t);
    expect(z.shape).toEqual([2, 2]);
  });

  it("returns 0 for constant tensor", () => {
    const t = tensor([5, 5, 5]);
    const z = zscore(t);
    expect(z.toArray()).toEqual([0, 0, 0]);
  });

  it("throws on empty tensor", () => {
    expect(() => zscore(tensor([]))).toThrow();
  });
});

describe("cohenD", () => {
  it("computes effect size for two samples", () => {
    const a = [1, 2, 3, 4, 5];
    const b = [3, 4, 5, 6, 7];
    const d = cohenD(a, b);
    // mean_a=3, mean_b=5, var_a=var_b=2.5, pooled_std=sqrt(2.5)
    expect(d).toBeCloseTo(-2 / Math.sqrt(2.5), 10);
  });

  it("returns 0 for identical distributions", () => {
    const a = [1, 2, 3, 4, 5];
    const d = cohenD(a, a);
    expect(d).toBeCloseTo(0, 10);
  });

  it("is negative when a < b and positive when a > b", () => {
    const a = [1, 2, 3];
    const b = [10, 11, 12];
    expect(cohenD(a, b)).toBeLessThan(0);
    expect(cohenD(b, a)).toBeGreaterThan(0);
  });

  it("throws on too-small samples", () => {
    expect(() => cohenD([1], [2, 3])).toThrow();
    expect(() => cohenD([1, 2], [3])).toThrow();
  });
});

describe("bootstrap", () => {
  const meanFn = (s: number[]) => s.reduce((a, b) => a + b, 0) / s.length;

  it("returns estimate equal to the original statistic", () => {
    const data = [1, 2, 3, 4, 5];
    const result = bootstrap(data, meanFn, { seed: 42 });
    expect(result.estimate).toBeCloseTo(3, 10);
  });

  it("returns confidence interval as [lo, hi]", () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const result = bootstrap(data, meanFn, { seed: 42, nResamples: 500 });
    expect(result.ci[0]).toBeLessThanOrEqual(result.estimate);
    expect(result.ci[1]).toBeGreaterThanOrEqual(result.estimate);
  });

  it("produces the requested number of resamples", () => {
    const data = [1, 2, 3];
    const result = bootstrap(data, meanFn, { nResamples: 200, seed: 42 });
    expect(result.samples.length).toBe(200);
  });

  it("is deterministic with seed", () => {
    const data = [1, 2, 3, 4, 5];
    const r1 = bootstrap(data, meanFn, { seed: 123, nResamples: 100 });
    const r2 = bootstrap(data, meanFn, { seed: 123, nResamples: 100 });
    expect(r1.samples).toEqual(r2.samples);
  });

  it("throws on empty data", () => {
    expect(() => bootstrap([], meanFn)).toThrow();
  });

  it("works with custom confidence level", () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const r90 = bootstrap(data, meanFn, { seed: 42, confidenceLevel: 0.9 });
    const r99 = bootstrap(data, meanFn, { seed: 42, confidenceLevel: 0.99 });
    // 99% CI should be wider than 90% CI
    expect(r99.ci[1] - r99.ci[0]).toBeGreaterThanOrEqual(r90.ci[1] - r90.ci[0]);
  });
});

describe("sem", () => {
  it("computes SEM for a 1-D tensor", () => {
    const t = tensor([2, 4, 4, 4, 5, 5, 7, 9]);
    const result = sem(t);
    const arr = result.toArray() as number;
    // mean=5, std(ddof=1)=sqrt(32/7)≈2.138, sem=2.138/sqrt(8)≈0.756
    const m = 5;
    const ss = [2, 4, 4, 4, 5, 5, 7, 9].reduce((s, v) => s + (v - m) ** 2, 0);
    const expected = Math.sqrt(ss / 7) / Math.sqrt(8);
    expect(arr).toBeCloseTo(expected, 10);
  });

  it("works with axis parameter", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const result = sem(t, 0);
    const arr = result.toArray() as number[];
    expect(arr.length).toBe(2);
  });

  it("throws on empty tensor", () => {
    expect(() => sem(tensor([]))).toThrow();
  });
});

describe("iqr", () => {
  it("computes IQR for a 1-D tensor", () => {
    const t = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const result = iqr(t);
    const val = result.toArray() as number;
    // Q1 = 3.25, Q3 = 7.75, IQR = 4.5
    expect(val).toBeCloseTo(4.5, 10);
  });

  it("returns NaN for tensor with NaN", () => {
    const t = tensor([1, 2, NaN, 4]);
    const result = iqr(t);
    const arr = result.toArray();
    const val = Array.isArray(arr) ? arr[0] : arr;
    expect(Number.isNaN(val)).toBe(true);
  });

  it("throws on empty tensor", () => {
    expect(() => iqr(tensor([]))).toThrow();
  });

  it("computes IQR for small array", () => {
    const t = tensor([1, 2, 3, 4]);
    const result = iqr(t);
    // Q1 = 1.75, Q3 = 3.25, IQR = 1.5
    expect(result.toArray() as number).toBeCloseTo(1.5, 10);
  });
});

describe("pointbiserialr", () => {
  it("computes correlation between binary and continuous", () => {
    const x = tensor([0, 1, 1, 0, 1, 0, 1, 0]);
    const y = tensor([10, 20, 18, 12, 22, 11, 19, 13]);
    const [r, p] = pointbiserialr(x, y);
    expect(r).toBeGreaterThan(0); // group 1 has higher scores
    expect(p).toBeLessThan(1);
    expect(p).toBeGreaterThanOrEqual(0);
  });

  it("throws for non-binary input", () => {
    expect(() => pointbiserialr(tensor([0, 1, 2]), tensor([10, 20, 30]))).toThrow(/binary/);
  });

  it("throws for fewer than 2 samples", () => {
    expect(() => pointbiserialr(tensor([0]), tensor([10]))).toThrow();
  });
});

describe("partialcorr", () => {
  it("computes partial correlation with single confounder", () => {
    // x, y both correlated with z
    const x = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const y = tensor([2, 3, 5, 4, 7, 6, 8, 9, 10, 11]);
    const z = tensor([1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
    const [r, p] = partialcorr(x, y, z);
    expect(Math.abs(r)).toBeLessThanOrEqual(1);
    expect(p).toBeGreaterThanOrEqual(0);
  });

  it("computes partial correlation with multiple confounders", () => {
    const x = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const y = tensor([2, 3, 5, 4, 7, 6, 8, 9, 10, 11]);
    const z1 = tensor([1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
    const z2 = tensor([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
    const [r, p] = partialcorr(x, y, [z1, z2]);
    expect(Math.abs(r)).toBeLessThanOrEqual(1);
    expect(p).toBeGreaterThanOrEqual(0);
  });

  it("throws for insufficient samples", () => {
    expect(() => partialcorr(tensor([1, 2, 3]), tensor([4, 5, 6]), tensor([7, 8, 9]))).toThrow();
  });
});

describe("bonferroni", () => {
  it("multiplies p-values by number of tests", () => {
    const result = bonferroni([0.01, 0.04, 0.03, 0.005]);
    expect(result.corrected).toEqual([0.04, 0.16, 0.12, 0.02]);
  });

  it("caps at 1.0", () => {
    const result = bonferroni([0.5, 0.6]);
    expect(result.corrected[0]).toBe(1);
    expect(result.corrected[1]).toBe(1);
  });

  it("rejects correctly at alpha=0.05", () => {
    const result = bonferroni([0.01, 0.04, 0.03, 0.005], 0.05);
    expect(result.rejected).toEqual([true, false, false, true]);
  });

  it("throws on empty pvalues", () => {
    expect(() => bonferroni([])).toThrow();
  });
});

describe("holm", () => {
  it("applies step-down correction", () => {
    const result = holm([0.01, 0.04, 0.03, 0.005], 0.05);
    // Sorted: 0.005(idx3), 0.01(idx0), 0.03(idx2), 0.04(idx1)
    // Adjusted: 0.005*4=0.02, max(0.02, 0.01*3)=0.03, max(0.03, 0.03*2)=0.06, max(0.06, 0.04*1)=0.06
    expect(result.corrected[3]).toBeCloseTo(0.02, 10);
    expect(result.corrected[0]).toBeCloseTo(0.03, 10);
    expect(result.rejected[3]).toBe(true);
    expect(result.rejected[0]).toBe(true);
  });

  it("throws on empty pvalues", () => {
    expect(() => holm([])).toThrow();
  });
});

describe("benjaminiHochberg", () => {
  it("controls FDR", () => {
    const result = benjaminiHochberg([0.01, 0.04, 0.03, 0.005], 0.05);
    // All corrected p-values should be in [0, 1]
    for (const p of result.corrected) {
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }
    // The smallest p-value should always be rejected
    expect(result.rejected[3]).toBe(true);
  });

  it("is less conservative than Bonferroni", () => {
    const pvals = [0.01, 0.03, 0.04, 0.005];
    const bh = benjaminiHochberg(pvals, 0.05);
    const bonf = bonferroni(pvals, 0.05);
    // BH should reject at least as many as Bonferroni
    const bhRejected = bh.rejected.filter(Boolean).length;
    const bonfRejected = bonf.rejected.filter(Boolean).length;
    expect(bhRejected).toBeGreaterThanOrEqual(bonfRejected);
  });
});

describe("sidak", () => {
  it("applies Šidák correction", () => {
    const result = sidak([0.01, 0.04], 0.05);
    // p_corrected = 1 - (1-p)^m
    expect(result.corrected[0]).toBeCloseTo(1 - (1 - 0.01) ** 2, 10);
    expect(result.corrected[1]).toBeCloseTo(1 - (1 - 0.04) ** 2, 10);
  });

  it("is less conservative than Bonferroni", () => {
    const pvals = [0.01, 0.03, 0.04, 0.005];
    const sid = sidak(pvals, 0.05);
    const bonf = bonferroni(pvals, 0.05);
    // Sidak corrected p-values should be <= Bonferroni corrected
    for (let i = 0; i < pvals.length; i++) {
      expect(sid.corrected[i]).toBeLessThanOrEqual(bonf.corrected[i]!);
    }
  });
});
