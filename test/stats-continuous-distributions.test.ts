import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  cauchy,
  chi2_contingency,
  fisher_exact,
  fligner,
  laplace,
  lilliefors,
  lognorm,
  pareto,
  runs_test,
  weibull,
} from "../src/stats";

// ─── New distributions ──────────────────────────────────────────────────────

describe("Lognormal distribution", () => {
  const d = lognorm(0, 1);

  it("pdf is positive for x > 0 and zero for x <= 0", () => {
    expect(d.pdf(1)).toBeGreaterThan(0);
    expect(d.pdf(0)).toBe(0);
    expect(d.pdf(-1)).toBe(0);
  });

  it("cdf is monotone and bounded", () => {
    expect(d.cdf(0)).toBe(0);
    expect(d.cdf(1)).toBeCloseTo(0.5, 1);
    expect(d.cdf(100)).toBeGreaterThan(0.99);
  });

  it("ppf inverts cdf", () => {
    for (const p of [0.1, 0.5, 0.9]) {
      expect(d.cdf(d.ppf(p))).toBeCloseTo(p, 4);
    }
  });

  it("mean and variance are correct", () => {
    // mu=0, sigma=1: mean = e^(0.5), variance = (e-1)*e
    expect(d.mean()).toBeCloseTo(Math.exp(0.5), 5);
    expect(d.variance()).toBeCloseTo((Math.E - 1) * Math.E, 5);
  });

  it("rvs returns correct count", () => {
    expect(d.rvs(10).length).toBe(10);
  });

  it("entropy is a number", () => {
    expect(Number.isFinite(d.entropy())).toBe(true);
  });
});

describe("Weibull distribution", () => {
  const d = weibull(2, 1); // shape=2, scale=1 (Rayleigh-like)

  it("pdf and cdf basics", () => {
    expect(d.pdf(-1)).toBe(0);
    expect(d.pdf(0.5)).toBeGreaterThan(0);
    expect(d.cdf(0)).toBe(0);
    expect(d.cdf(10)).toBeGreaterThan(0.99);
  });

  it("ppf inverts cdf", () => {
    for (const p of [0.1, 0.5, 0.9]) {
      expect(d.cdf(d.ppf(p))).toBeCloseTo(p, 5);
    }
  });

  it("mean is correct for shape=2, scale=1", () => {
    // mean = Γ(1 + 1/2) = Γ(1.5) = sqrt(π)/2 ≈ 0.8862
    expect(d.mean()).toBeCloseTo(Math.sqrt(Math.PI) / 2, 4);
  });

  it("throws for invalid parameters", () => {
    expect(() => weibull(0)).toThrow();
    expect(() => weibull(-1)).toThrow();
    expect(() => weibull(1, 0)).toThrow();
  });
});

describe("Pareto distribution", () => {
  const d = pareto(3, 1); // alpha=3, xm=1

  it("pdf and cdf basics", () => {
    expect(d.pdf(0.5)).toBe(0); // below xm
    expect(d.pdf(1)).toBeCloseTo(3, 5); // alpha * xm^alpha / xm^(alpha+1) = 3
    expect(d.cdf(1)).toBeCloseTo(0, 5);
    expect(d.cdf(2)).toBeCloseTo(1 - 0.5 ** 3, 5);
  });

  it("ppf inverts cdf", () => {
    for (const p of [0.1, 0.5, 0.9]) {
      expect(d.cdf(d.ppf(p))).toBeCloseTo(p, 5);
    }
  });

  it("mean is correct for alpha=3", () => {
    // mean = alpha * xm / (alpha - 1) = 3/2 = 1.5
    expect(d.mean()).toBeCloseTo(1.5, 5);
  });

  it("throws for invalid parameters", () => {
    expect(() => pareto(0)).toThrow();
    expect(() => pareto(1, 0)).toThrow();
  });
});

describe("Cauchy distribution", () => {
  const d = cauchy(0, 1);

  it("pdf at x0 is 1/(π*γ)", () => {
    expect(d.pdf(0)).toBeCloseTo(1 / Math.PI, 5);
  });

  it("cdf at x0 is 0.5", () => {
    expect(d.cdf(0)).toBeCloseTo(0.5, 5);
  });

  it("ppf inverts cdf", () => {
    for (const p of [0.1, 0.25, 0.5, 0.75, 0.9]) {
      expect(d.cdf(d.ppf(p))).toBeCloseTo(p, 5);
    }
  });

  it("mean and variance are NaN", () => {
    expect(Number.isNaN(d.mean())).toBe(true);
    expect(Number.isNaN(d.variance())).toBe(true);
  });

  it("throws for invalid scale", () => {
    expect(() => cauchy(0, 0)).toThrow();
    expect(() => cauchy(0, -1)).toThrow();
  });
});

describe("Laplace distribution", () => {
  const d = laplace(0, 1);

  it("pdf at loc is 1/(2*scale)", () => {
    expect(d.pdf(0)).toBeCloseTo(0.5, 5);
  });

  it("cdf at loc is 0.5", () => {
    expect(d.cdf(0)).toBeCloseTo(0.5, 5);
  });

  it("ppf inverts cdf", () => {
    for (const p of [0.1, 0.25, 0.5, 0.75, 0.9]) {
      expect(d.cdf(d.ppf(p))).toBeCloseTo(p, 5);
    }
  });

  it("mean and variance are correct", () => {
    expect(d.mean()).toBe(0);
    expect(d.variance()).toBeCloseTo(2, 5); // 2 * scale^2
  });

  it("sf + cdf = 1", () => {
    for (const x of [-2, -1, 0, 1, 2]) {
      expect(d.cdf(x) + d.sf(x)).toBeCloseTo(1, 10);
    }
  });
});

// ─── chi2_contingency ───────────────────────────────────────────────────────

describe("chi2_contingency", () => {
  it("detects independence in a 2x3 table", () => {
    const result = chi2_contingency([
      [10, 20, 30],
      [6, 9, 17],
    ]);
    expect(result.dof).toBe(2); // (2-1)*(3-1)
    expect(result.statistic).toBeGreaterThanOrEqual(0);
    expect(result.pvalue).toBeGreaterThanOrEqual(0);
    expect(result.pvalue).toBeLessThanOrEqual(1);
    expect(result.expected.length).toBe(2);
    expect(result.expected[0]?.length).toBe(3);
  });

  it("returns low p-value for dependent variables", () => {
    // Strong association
    const result = chi2_contingency([
      [100, 0],
      [0, 100],
    ]);
    expect(result.pvalue).toBeLessThan(0.001);
  });

  it("returns high p-value for independent variables", () => {
    // Roughly independent
    const result = chi2_contingency([
      [50, 50],
      [50, 50],
    ]);
    expect(result.pvalue).toBeGreaterThan(0.9);
  });

  it("throws for invalid table dimensions", () => {
    expect(() => chi2_contingency([[1, 2]])).toThrow(); // only 1 row
    expect(() => chi2_contingency([[1], [2]])).toThrow(); // only 1 col
  });

  it("throws for negative values", () => {
    expect(() =>
      chi2_contingency([
        [1, -1],
        [1, 1],
      ])
    ).toThrow();
  });
});

// ─── fisher_exact ───────────────────────────────────────────────────────────

describe("fisher_exact", () => {
  it("detects strong association", () => {
    // Classic tea tasting example
    const result = fisher_exact([
      [1, 9],
      [11, 3],
    ]);
    expect(result.pvalue).toBeLessThan(0.01);
    expect(result.oddsRatio).toBeLessThan(1);
  });

  it("returns 1 for no association", () => {
    const result = fisher_exact([
      [5, 5],
      [5, 5],
    ]);
    expect(result.pvalue).toBeGreaterThan(0.5);
    expect(result.oddsRatio).toBeCloseTo(1, 5);
  });

  it("supports alternative='less'", () => {
    const result = fisher_exact(
      [
        [1, 9],
        [11, 3],
      ],
      "less"
    );
    expect(result.pvalue).toBeGreaterThanOrEqual(0);
    expect(result.pvalue).toBeLessThanOrEqual(1);
  });

  it("supports alternative='greater'", () => {
    const result = fisher_exact(
      [
        [1, 9],
        [11, 3],
      ],
      "greater"
    );
    expect(result.pvalue).toBeGreaterThanOrEqual(0);
    expect(result.pvalue).toBeLessThanOrEqual(1);
  });

  it("throws for negative values", () => {
    expect(() =>
      fisher_exact([
        [-1, 1],
        [1, 1],
      ])
    ).toThrow();
  });
});

// ─── runs_test ──────────────────────────────────────────────────────────────

describe("runs_test", () => {
  it("detects non-random alternating pattern", () => {
    // Perfectly alternating — too many runs
    const data = tensor([1, 10, 1, 10, 1, 10, 1, 10, 1, 10]);
    const result = runs_test(data);
    expect(result.pvalue).toBeLessThan(0.05);
  });

  it("accepts a random-looking sequence", () => {
    const data = tensor([2, 5, 1, 8, 3, 7, 4, 6, 9, 10]);
    const result = runs_test(data);
    // Should not reject at 0.01 level for such a small sample
    expect(result.pvalue).toBeGreaterThan(0.01);
  });

  it("handles all-same values", () => {
    const result = runs_test(tensor([5, 5, 5, 5]));
    expect(result.pvalue).toBe(1);
  });

  it("throws for too few observations", () => {
    expect(() => runs_test(tensor([1]))).toThrow();
  });
});

// ─── lilliefors ─────────────────────────────────────────────────────────────

describe("lilliefors", () => {
  it("does not reject normally distributed data", () => {
    // Generate pseudo-normal data (not random, but roughly normal)
    const data = tensor([
      -1.5, -1.0, -0.8, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, -0.7, -0.2, 0.2, 0.7,
      -0.4, 0.4, 0.6,
    ]);
    const result = lilliefors(data);
    expect(result.statistic).toBeGreaterThanOrEqual(0);
    // Should not strongly reject
    expect(result.pvalue).toBeGreaterThan(0.01);
  });

  it("rejects highly non-normal data", () => {
    // Uniform-ish data with gaps
    const data = tensor([1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 1, 1, 1, 10, 10, 10, 1, 10, 1, 10]);
    const result = lilliefors(data);
    expect(result.statistic).toBeGreaterThan(0);
  });

  it("throws for too few observations", () => {
    expect(() => lilliefors(tensor([1, 2, 3]))).toThrow();
  });

  it("handles constant data", () => {
    const result = lilliefors(tensor([5, 5, 5, 5, 5]));
    expect(result.pvalue).toBe(1);
  });
});

// ─── fligner ────────────────────────────────────────────────────────────────

describe("fligner (Fligner-Killeen test)", () => {
  it("does not reject equal variance groups", () => {
    const g1 = tensor([1, 2, 3, 4, 5]);
    const g2 = tensor([2, 3, 4, 5, 6]);
    const result = fligner([g1, g2]);
    expect(result.pvalue).toBeGreaterThan(0.05);
  });

  it("rejects very unequal variance groups", () => {
    const g1 = tensor([1, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98]);
    const g2 = tensor([0, 10, -10, 20, -20, 15, -15]);
    const result = fligner([g1, g2]);
    expect(result.pvalue).toBeLessThan(0.05);
  });

  it("works with 3 groups", () => {
    const g1 = tensor([1, 2, 3, 4, 5]);
    const g2 = tensor([2, 3, 4, 5, 6]);
    const g3 = tensor([3, 4, 5, 6, 7]);
    const result = fligner([g1, g2, g3]);
    expect(result.statistic).toBeGreaterThanOrEqual(0);
    expect(result.pvalue).toBeGreaterThanOrEqual(0);
    expect(result.pvalue).toBeLessThanOrEqual(1);
  });

  it("throws for fewer than 2 groups", () => {
    expect(() => fligner([tensor([1, 2, 3])])).toThrow();
  });

  it("throws for group with < 2 observations", () => {
    expect(() => fligner([tensor([1]), tensor([2, 3])])).toThrow();
  });
});
