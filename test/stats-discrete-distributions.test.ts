import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { geom, hypergeom, ks_2samp, median_test, nbinom } from "../src/stats";

// ---------------------------------------------------------------------------
// Geometric distribution
// ---------------------------------------------------------------------------
describe("geom", () => {
  it("should have correct mean and variance", () => {
    const g = geom(0.5);
    expect(g.mean()).toBeCloseTo(2, 10);
    expect(g.variance()).toBeCloseTo(2, 10);
  });

  it("should compute pmf correctly", () => {
    const g = geom(0.3);
    // P(X=1) = 0.3
    expect(g.pmf(1)).toBeCloseTo(0.3, 10);
    // P(X=2) = 0.3 * 0.7 = 0.21
    expect(g.pmf(2)).toBeCloseTo(0.21, 10);
    // P(X=0) = 0 (support starts at 1)
    expect(g.pmf(0)).toBe(0);
  });

  it("should compute cdf correctly", () => {
    const g = geom(0.5);
    expect(g.cdf(1)).toBeCloseTo(0.5, 10);
    expect(g.cdf(2)).toBeCloseTo(0.75, 10);
  });

  it("should compute ppf correctly", () => {
    const g = geom(0.5);
    expect(g.ppf(0.5)).toBe(1);
    expect(g.ppf(0.75)).toBe(2);
  });

  it("should generate random samples", () => {
    const g = geom(0.5);
    const samples = g.rvs(100);
    expect(samples.length).toBe(100);
    expect(samples.every((s) => s >= 1 && Number.isInteger(s))).toBe(true);
  });

  it("should throw for invalid p", () => {
    expect(() => geom(0)).toThrow();
    expect(() => geom(1.1)).toThrow();
    expect(() => geom(-0.1)).toThrow();
  });
});

// ---------------------------------------------------------------------------
// Negative Binomial distribution
// ---------------------------------------------------------------------------
describe("nbinom", () => {
  it("should have correct mean and variance", () => {
    const nb = nbinom(5, 0.5);
    // mean = r*(1-p)/p = 5*0.5/0.5 = 5
    expect(nb.mean()).toBeCloseTo(5, 10);
    // variance = r*(1-p)/p^2 = 5*0.5/0.25 = 10
    expect(nb.variance()).toBeCloseTo(10, 10);
  });

  it("should compute pmf correctly for k=0", () => {
    const nb = nbinom(3, 0.4);
    // P(X=0) = p^r = 0.4^3 = 0.064
    expect(nb.pmf(0)).toBeCloseTo(0.064, 6);
  });

  it("should have pmf summing to ~1", () => {
    const nb = nbinom(2, 0.5);
    let sum = 0;
    for (let k = 0; k < 100; k++) {
      sum += nb.pmf(k);
    }
    expect(sum).toBeCloseTo(1, 4);
  });

  it("should throw for invalid parameters", () => {
    expect(() => nbinom(0, 0.5)).toThrow();
    expect(() => nbinom(2, 0)).toThrow();
    expect(() => nbinom(2, 1.5)).toThrow();
  });
});

// ---------------------------------------------------------------------------
// Hypergeometric distribution
// ---------------------------------------------------------------------------
describe("hypergeom", () => {
  it("should have correct mean", () => {
    // N=52, K=13 (spades), n=5 draws
    const h = hypergeom(52, 13, 5);
    expect(h.mean()).toBeCloseTo((5 * 13) / 52, 10);
  });

  it("should compute pmf correctly", () => {
    // Drawing from urn: N=10, K=4, n=3
    const h = hypergeom(10, 4, 3);
    // P(X=0) = C(4,0)*C(6,3)/C(10,3) = 1*20/120 = 1/6
    expect(h.pmf(0)).toBeCloseTo(1 / 6, 6);
  });

  it("should have pmf summing to 1", () => {
    const h = hypergeom(20, 7, 5);
    let sum = 0;
    for (let k = 0; k <= 5; k++) {
      sum += h.pmf(k);
    }
    expect(sum).toBeCloseTo(1, 6);
  });

  it("should compute cdf correctly", () => {
    const h = hypergeom(10, 4, 3);
    // CDF at max value should be 1
    expect(h.cdf(3)).toBeCloseTo(1, 10);
    // CDF at min-1 should be 0
    expect(h.cdf(-1)).toBe(0);
  });

  it("should throw for invalid parameters", () => {
    expect(() => hypergeom(-1, 4, 3)).toThrow();
    expect(() => hypergeom(10, 11, 3)).toThrow();
    expect(() => hypergeom(10, 4, 11)).toThrow();
  });
});

// ---------------------------------------------------------------------------
// ks_2samp (2-sample Kolmogorov-Smirnov)
// ---------------------------------------------------------------------------
describe("ks_2samp", () => {
  it("should return D=0 for identical samples", () => {
    const a = tensor([1, 2, 3, 4, 5]);
    const result = ks_2samp(a, a);
    expect(result.statistic).toBeCloseTo(0, 10);
  });

  it("should detect different distributions", () => {
    const a = tensor([1, 2, 3, 4, 5]);
    const b = tensor([10, 20, 30, 40, 50]);
    const result = ks_2samp(a, b);
    expect(result.statistic).toBe(1); // completely separated
    expect(result.pvalue).toBeLessThan(0.05);
  });

  it("should return high p-value for similar samples", () => {
    const a = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const b = tensor([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]);
    const result = ks_2samp(a, b);
    expect(result.statistic).toBeLessThan(0.5);
  });

  it("should have statistic in [0, 1]", () => {
    const a = tensor([1, 3, 5, 7]);
    const b = tensor([2, 4, 6, 8]);
    const result = ks_2samp(a, b);
    expect(result.statistic).toBeGreaterThanOrEqual(0);
    expect(result.statistic).toBeLessThanOrEqual(1);
    expect(result.pvalue).toBeGreaterThanOrEqual(0);
    expect(result.pvalue).toBeLessThanOrEqual(1);
  });

  it("should throw on empty samples", () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([] as number[]);
    expect(() => ks_2samp(a, b)).toThrow();
  });
});

// ---------------------------------------------------------------------------
// median_test (Mood's median test)
// ---------------------------------------------------------------------------
describe("median_test", () => {
  it("should return high p-value for similar groups", () => {
    const a = tensor([1, 2, 3, 4, 5]);
    const b = tensor([2, 3, 4, 5, 6]);
    const result = median_test(a, b);
    expect(result.pvalue).toBeGreaterThan(0.05);
  });

  it("should detect different medians", () => {
    const a = tensor([1, 2, 3, 4, 5]);
    const b = tensor([10, 20, 30, 40, 50]);
    const result = median_test(a, b);
    expect(result.pvalue).toBeLessThan(0.05);
  });

  it("should work with 3+ groups", () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([4, 5, 6]);
    const c = tensor([100, 200, 300]);
    const result = median_test(a, b, c);
    expect(result.statistic).toBeGreaterThanOrEqual(0);
    expect(result.pvalue).toBeLessThanOrEqual(1);
  });

  it("should throw with fewer than 2 groups", () => {
    const a = tensor([1, 2, 3]);
    expect(() => median_test(a)).toThrow();
  });
});
