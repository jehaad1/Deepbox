import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core";
import { poisson } from "../../src/stats";

// scipy 1.18: scipy.stats.poisson(0) is the point mass at 0.
//   pmf(-1, 0, 1, 2, 0.5) = 0, 1, 0, 0, 0
//   cdf(-1, -0.5, 0, 0.5, 1, 5) = 0, 0, 1, 1, 1, 1
//   sf(-1, 0, 1, 5) = 1, 0, 0, 0
//   ppf(0.3) = ppf(0.5) = 0, ppf(1) = inf, isf(0) = inf, isf(0.3) = isf(0.5) = 0
//   mean = var = std = median = 0, rvs are all 0.
// Deepbox returns 0 where scipy returns -1 for ppf(0) and isf(1) (the smallest support point).

describe("poisson with mu = 0", () => {
  const d = poisson(0);

  it("is a point mass at 0", () => {
    expect([-1, 0, 1, 2, 0.5].map((k) => d.pmf(k))).toEqual([0, 1, 0, 0, 0]);
    expect([-1, -0.5, 0, 0.5, 1, 5].map((x) => d.cdf(x))).toEqual([0, 0, 1, 1, 1, 1]);
    expect([-1, 0, 1, 5].map((x) => d.sf(x))).toEqual([1, 0, 0, 0]);
  });

  it("has the quantiles of a point mass", () => {
    expect(d.ppf(0.3)).toBe(0);
    expect(d.ppf(0.5)).toBe(0);
    expect(d.ppf(1)).toBe(Number.POSITIVE_INFINITY);
    expect(d.isf(0)).toBe(Number.POSITIVE_INFINITY);
    expect(d.isf(0.3)).toBe(0);
    expect(d.isf(0.5)).toBe(0);
  });

  it("has zero moments and draws zeros", () => {
    expect(d.mean()).toBe(0);
    expect(d.variance()).toBe(0);
    expect(d.std()).toBe(0);
    expect(d.median()).toBe(0);
    expect(d.rvs(5)).toEqual([0, 0, 0, 0, 0]);
  });

  it("still rejects a negative or non-finite rate", () => {
    for (const mu of [-1, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => poisson(mu)).toThrow(InvalidParameterError);
    }
  });
});
