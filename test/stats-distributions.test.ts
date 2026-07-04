import { describe, expect, it } from "vitest";
import {
  beta,
  binom,
  cauchy,
  chi2,
  expon,
  f,
  gamma,
  laplace,
  lognorm,
  norm,
  pareto,
  poisson,
  t,
  uniform,
  weibull,
} from "../src/stats/distributions";

// Helper: check value is close
function closeTo(actual: number, expected: number, tol = 1e-4) {
  if (Number.isNaN(expected)) {
    expect(actual).toBeNaN();
  } else if (!Number.isFinite(expected)) {
    expect(actual).toBe(expected);
  } else {
    expect(Math.abs(actual - expected)).toBeLessThan(tol);
  }
}

describe("Normal Distribution", () => {
  const d = norm(0, 1);

  it("pdf at standard values", () => {
    closeTo(d.pdf(0), 0.3989, 1e-3);
    closeTo(d.pdf(1), 0.242, 1e-3);
    closeTo(d.pdf(-1), 0.242, 1e-3);
  });

  it("cdf at standard values", () => {
    closeTo(d.cdf(0), 0.5);
    closeTo(d.cdf(1.96), 0.975, 1e-2);
    closeTo(d.cdf(-1.96), 0.025, 1e-2);
  });

  it("ppf inverts cdf", () => {
    closeTo(d.ppf(0.5), 0);
    closeTo(d.ppf(0.975), 1.96, 0.05);
    closeTo(d.ppf(0.025), -1.96, 0.05);
  });

  it("ppf boundary cases", () => {
    expect(d.ppf(0)).toBe(-Infinity);
    expect(d.ppf(1)).toBe(Infinity);
    expect(() => d.ppf(-0.1)).toThrow();
    expect(() => d.ppf(1.1)).toThrow();
  });

  it("sf = 1 - cdf", () => {
    closeTo(d.sf(0), 0.5);
    closeTo(d.sf(1.96), 0.025, 1e-2);
  });

  it("rvs generates correct size", () => {
    expect(d.rvs(10).length).toBe(10);
    expect(d.rvs().length).toBe(1);
  });

  it("mean and variance", () => {
    closeTo(d.mean(), 0);
    closeTo(d.variance(), 1);
  });

  it("entropy", () => {
    closeTo(d.entropy(), 1.4189, 1e-3); // 0.5*ln(2*pi*e)
  });

  it("non-standard normal", () => {
    const d2 = norm(5, 2);
    closeTo(d2.mean(), 5);
    closeTo(d2.variance(), 4);
    closeTo(d2.cdf(5), 0.5);
  });

  it("rejects invalid scale", () => {
    expect(() => norm(0, 0)).toThrow();
    expect(() => norm(0, -1)).toThrow();
  });
});

describe("Student's t Distribution", () => {
  const d = t(10);

  it("pdf at 0 is maximal", () => {
    expect(d.pdf(0)).toBeGreaterThan(d.pdf(1));
  });

  it("cdf at 0 is 0.5", () => {
    closeTo(d.cdf(0), 0.5);
  });

  it("ppf inverts cdf", () => {
    closeTo(d.ppf(0.5), 0, 0.01);
  });

  it("ppf boundary cases", () => {
    expect(d.ppf(0)).toBe(-Infinity);
    expect(d.ppf(1)).toBe(Infinity);
    expect(() => d.ppf(-0.1)).toThrow();
    expect(() => d.ppf(1.1)).toThrow();
  });

  it("sf = 1 - cdf", () => {
    closeTo(d.sf(0), 0.5);
  });

  it("rvs generates correct size", () => {
    expect(d.rvs(5).length).toBe(5);
    expect(d.rvs().length).toBe(1);
  });

  it("mean", () => {
    closeTo(t(10).mean(), 0);
    expect(t(1).mean()).toBeNaN(); // df <= 1
  });

  it("variance", () => {
    closeTo(t(10).variance(), 10 / 8);
    expect(t(2).variance()).toBe(Infinity);
    expect(t(1).variance()).toBeNaN();
  });

  it("entropy is finite", () => {
    expect(Number.isFinite(d.entropy())).toBe(true);
  });

  it("rejects invalid df", () => {
    expect(() => t(0)).toThrow();
    expect(() => t(-1)).toThrow();
  });
});

describe("Chi-squared Distribution", () => {
  const d = chi2(5);

  it("pdf returns 0 for x <= 0", () => {
    expect(d.pdf(0)).toBe(0);
    expect(d.pdf(-1)).toBe(0);
  });

  it("pdf positive for x > 0", () => {
    expect(d.pdf(3)).toBeGreaterThan(0);
  });

  it("cdf monotonically increasing", () => {
    expect(d.cdf(1)).toBeLessThan(d.cdf(5));
    expect(d.cdf(5)).toBeLessThan(d.cdf(10));
  });

  it("ppf inverts cdf", () => {
    const x = 3;
    const p = d.cdf(x);
    closeTo(d.ppf(p), x, 0.01);
  });

  it("ppf boundary cases", () => {
    expect(d.ppf(0)).toBe(0);
    expect(d.ppf(1)).toBe(Infinity);
    expect(() => d.ppf(-0.1)).toThrow();
    expect(() => d.ppf(1.1)).toThrow();
  });

  it("sf = 1 - cdf", () => {
    closeTo(d.sf(5), 1 - d.cdf(5));
  });

  it("rvs", () => {
    expect(d.rvs(10).length).toBe(10);
  });

  it("mean and variance", () => {
    closeTo(d.mean(), 5);
    closeTo(d.variance(), 10);
  });

  it("entropy is finite", () => {
    expect(Number.isFinite(d.entropy())).toBe(true);
  });

  it("rejects invalid df", () => {
    expect(() => chi2(0)).toThrow();
    expect(() => chi2(-1)).toThrow();
  });
});

describe("F Distribution", () => {
  const d = f(5, 10);

  it("pdf returns 0 for x <= 0", () => {
    expect(d.pdf(0)).toBe(0);
    expect(d.pdf(-1)).toBe(0);
  });

  it("pdf positive for x > 0", () => {
    expect(d.pdf(1)).toBeGreaterThan(0);
  });

  it("cdf monotonically increasing", () => {
    expect(d.cdf(0.5)).toBeLessThan(d.cdf(1));
    expect(d.cdf(1)).toBeLessThan(d.cdf(5));
  });

  it("ppf inverts cdf", () => {
    const x = 2;
    const p = d.cdf(x);
    closeTo(d.ppf(p), x, 0.05);
  });

  it("ppf boundary cases", () => {
    expect(d.ppf(0)).toBe(0);
    expect(d.ppf(1)).toBe(Infinity);
    expect(() => d.ppf(-0.1)).toThrow();
    expect(() => d.ppf(1.1)).toThrow();
  });

  it("sf = 1 - cdf", () => {
    closeTo(d.sf(2), 1 - d.cdf(2));
  });

  it("rvs", () => {
    expect(d.rvs(10).length).toBe(10);
  });

  it("mean", () => {
    closeTo(f(5, 10).mean(), 10 / 8); // dfd/(dfd-2) when dfd > 2
    expect(f(5, 2).mean()).toBeNaN(); // dfd <= 2
  });

  it("variance", () => {
    expect(Number.isFinite(f(5, 10).variance())).toBe(true);
    expect(f(5, 4).variance()).toBeNaN(); // dfd <= 4
  });

  it("entropy is finite", () => {
    expect(Number.isFinite(d.entropy())).toBe(true);
  });

  it("rejects invalid params", () => {
    expect(() => f(0, 10)).toThrow();
    expect(() => f(5, 0)).toThrow();
    expect(() => f(-1, 10)).toThrow();
    expect(() => f(5, -1)).toThrow();
  });
});

describe("Uniform Distribution", () => {
  const d = uniform(0, 1);

  it("pdf inside range", () => {
    closeTo(d.pdf(0.5), 1);
    closeTo(d.pdf(0), 1);
    closeTo(d.pdf(1), 1);
  });

  it("pdf outside range", () => {
    expect(d.pdf(-0.1)).toBe(0);
    expect(d.pdf(1.1)).toBe(0);
  });

  it("cdf", () => {
    closeTo(d.cdf(0), 0);
    closeTo(d.cdf(0.5), 0.5);
    closeTo(d.cdf(1), 1);
    expect(d.cdf(-1)).toBe(0);
    expect(d.cdf(2)).toBe(1);
  });

  it("ppf", () => {
    closeTo(d.ppf(0), 0);
    closeTo(d.ppf(0.5), 0.5);
    closeTo(d.ppf(1), 1);
    expect(() => d.ppf(-0.1)).toThrow();
    expect(() => d.ppf(1.1)).toThrow();
  });

  it("sf = 1 - cdf", () => {
    closeTo(d.sf(0.3), 0.7);
  });

  it("rvs in range", () => {
    const samples = d.rvs(100);
    expect(samples.length).toBe(100);
    for (const s of samples) {
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(1);
    }
  });

  it("mean and variance", () => {
    closeTo(d.mean(), 0.5);
    closeTo(d.variance(), 1 / 12);
  });

  it("entropy", () => {
    closeTo(d.entropy(), 0); // ln(1) = 0
  });

  it("non-standard uniform", () => {
    const d2 = uniform(2, 6);
    closeTo(d2.mean(), 4);
    closeTo(d2.variance(), 16 / 12);
    closeTo(d2.pdf(3), 0.25);
  });

  it("rejects invalid params", () => {
    expect(() => uniform(1, 1)).toThrow();
    expect(() => uniform(2, 1)).toThrow();
  });
});

describe("Exponential Distribution", () => {
  const d = expon(2);

  it("pdf", () => {
    expect(d.pdf(-1)).toBe(0);
    closeTo(d.pdf(0), 2);
    expect(d.pdf(1)).toBeGreaterThan(0);
  });

  it("cdf", () => {
    expect(d.cdf(-1)).toBe(0);
    closeTo(d.cdf(0), 0);
    expect(d.cdf(1)).toBeGreaterThan(0);
    expect(d.cdf(1)).toBeLessThan(1);
  });

  it("ppf inverts cdf", () => {
    const x = 0.5;
    const p = d.cdf(x);
    closeTo(d.ppf(p), x, 0.001);
    expect(() => d.ppf(-0.1)).toThrow();
    expect(() => d.ppf(1.1)).toThrow();
  });

  it("sf", () => {
    expect(d.sf(-1)).toBe(1);
    closeTo(d.sf(0), 1);
    expect(d.sf(1)).toBeGreaterThan(0);
  });

  it("rvs", () => {
    expect(d.rvs(10).length).toBe(10);
  });

  it("mean and variance", () => {
    closeTo(d.mean(), 0.5);
    closeTo(d.variance(), 0.25);
  });

  it("entropy", () => {
    closeTo(d.entropy(), 1 - Math.log(2));
  });

  it("rejects invalid rate", () => {
    expect(() => expon(0)).toThrow();
    expect(() => expon(-1)).toThrow();
  });
});

describe("Beta Distribution", () => {
  const d = beta(2, 5);

  it("pdf at boundaries", () => {
    expect(d.pdf(0)).toBe(0);
    expect(d.pdf(1)).toBe(0);
    expect(d.pdf(-0.1)).toBe(0);
    expect(d.pdf(1.1)).toBe(0);
  });

  it("pdf positive inside (0,1)", () => {
    expect(d.pdf(0.2)).toBeGreaterThan(0);
    expect(d.pdf(0.5)).toBeGreaterThan(0);
  });

  it("cdf at boundaries", () => {
    expect(d.cdf(0)).toBe(0);
    expect(d.cdf(-0.1)).toBe(0);
    expect(d.cdf(1)).toBe(1);
    expect(d.cdf(1.1)).toBe(1);
  });

  it("ppf inverts cdf", () => {
    const x = 0.3;
    const p = d.cdf(x);
    closeTo(d.ppf(p), x, 0.01);
  });

  it("ppf boundaries", () => {
    expect(d.ppf(0)).toBe(0);
    expect(d.ppf(1)).toBe(1);
    expect(() => d.ppf(-0.1)).toThrow();
    expect(() => d.ppf(1.1)).toThrow();
  });

  it("sf = 1 - cdf", () => {
    closeTo(d.sf(0.3), 1 - d.cdf(0.3));
  });

  it("rvs in (0,1)", () => {
    const samples = d.rvs(20);
    expect(samples.length).toBe(20);
    for (const s of samples) {
      expect(s).toBeGreaterThan(0);
      expect(s).toBeLessThan(1);
    }
  });

  it("mean", () => {
    closeTo(d.mean(), 2 / 7);
  });

  it("variance", () => {
    closeTo(d.variance(), (2 * 5) / (49 * 8));
  });

  it("entropy is finite", () => {
    expect(Number.isFinite(d.entropy())).toBe(true);
  });

  it("rejects invalid params", () => {
    expect(() => beta(0, 1)).toThrow();
    expect(() => beta(1, 0)).toThrow();
    expect(() => beta(-1, 2)).toThrow();
  });
});

describe("Gamma Distribution", () => {
  const d = gamma(2, 1);

  it("pdf at boundaries", () => {
    expect(d.pdf(0)).toBe(0);
    expect(d.pdf(-1)).toBe(0);
    expect(d.pdf(1)).toBeGreaterThan(0);
  });

  it("cdf monotonically increasing", () => {
    expect(d.cdf(0)).toBe(0);
    expect(d.cdf(-1)).toBe(0);
    expect(d.cdf(1)).toBeLessThan(d.cdf(3));
  });

  it("ppf inverts cdf", () => {
    const x = 2;
    const p = d.cdf(x);
    closeTo(d.ppf(p), x, 0.05);
  });

  it("ppf boundaries", () => {
    expect(d.ppf(0)).toBe(0);
    expect(d.ppf(1)).toBe(Infinity);
    expect(() => d.ppf(-0.1)).toThrow();
    expect(() => d.ppf(1.1)).toThrow();
  });

  it("sf = 1 - cdf", () => {
    closeTo(d.sf(2), 1 - d.cdf(2));
  });

  it("rvs", () => {
    expect(d.rvs(10).length).toBe(10);
  });

  it("mean and variance", () => {
    closeTo(d.mean(), 2);
    closeTo(d.variance(), 2);
  });

  it("entropy is finite", () => {
    expect(Number.isFinite(d.entropy())).toBe(true);
  });

  it("rejects invalid params", () => {
    expect(() => gamma(0, 1)).toThrow();
    expect(() => gamma(2, 0)).toThrow();
    expect(() => gamma(-1, 1)).toThrow();
  });
});

describe("Binomial Distribution", () => {
  const d = binom(10, 0.5);

  it("pmf", () => {
    expect(d.pmf(5)).toBeGreaterThan(0);
    expect(d.pmf(-1)).toBe(0);
    expect(d.pmf(11)).toBe(0);
    expect(d.pmf(0.5)).toBe(0); // non-integer
  });

  it("pmf sums to ~1", () => {
    let sum = 0;
    for (let k = 0; k <= 10; k++) sum += d.pmf(k);
    closeTo(sum, 1, 1e-6);
  });

  it("cdf", () => {
    expect(d.cdf(-1)).toBe(0);
    expect(d.cdf(10)).toBe(1);
    expect(d.cdf(5)).toBeGreaterThan(0);
    expect(d.cdf(5)).toBeLessThanOrEqual(1);
  });

  it("ppf inverts cdf", () => {
    expect(d.ppf(0.5)).toBeGreaterThanOrEqual(0);
    expect(d.ppf(0.5)).toBeLessThanOrEqual(10);
    expect(() => d.ppf(-0.1)).toThrow();
    expect(() => d.ppf(1.1)).toThrow();
  });

  it("sf = 1 - cdf", () => {
    closeTo(d.sf(5), 1 - d.cdf(5));
  });

  it("rvs", () => {
    const samples = d.rvs(20);
    expect(samples.length).toBe(20);
    for (const s of samples) {
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(10);
    }
  });

  it("mean and variance", () => {
    closeTo(d.mean(), 5);
    closeTo(d.variance(), 2.5);
  });

  it("rejects invalid params", () => {
    expect(() => binom(-1, 0.5)).toThrow();
    expect(() => binom(1.5, 0.5)).toThrow();
    expect(() => binom(10, -0.1)).toThrow();
    expect(() => binom(10, 1.1)).toThrow();
  });
});

describe("Poisson Distribution", () => {
  const d = poisson(5);

  it("pmf", () => {
    expect(d.pmf(5)).toBeGreaterThan(0);
    expect(d.pmf(-1)).toBe(0);
    expect(d.pmf(0.5)).toBe(0); // non-integer
  });

  it("cdf monotonically increasing", () => {
    expect(d.cdf(-1)).toBe(0);
    expect(d.cdf(0)).toBeGreaterThan(0);
    expect(d.cdf(3)).toBeLessThan(d.cdf(10));
  });

  it("ppf inverts cdf", () => {
    const p = d.cdf(5);
    expect(d.ppf(p)).toBeGreaterThanOrEqual(5);
    expect(() => d.ppf(-0.1)).toThrow();
    expect(() => d.ppf(1.1)).toThrow();
  });

  it("sf = 1 - cdf", () => {
    closeTo(d.sf(5), 1 - d.cdf(5));
  });

  it("rvs", () => {
    const samples = d.rvs(20);
    expect(samples.length).toBe(20);
    for (const s of samples) {
      expect(s).toBeGreaterThanOrEqual(0);
      expect(Number.isInteger(s)).toBe(true);
    }
  });

  it("mean and variance", () => {
    closeTo(d.mean(), 5);
    closeTo(d.variance(), 5);
  });

  it("rejects invalid mu", () => {
    expect(() => poisson(0)).toThrow();
    expect(() => poisson(-1)).toThrow();
  });
});

describe("Lognormal Distribution", () => {
  const d = lognorm(0, 1);

  it("pdf", () => {
    expect(d.pdf(0)).toBe(0);
    expect(d.pdf(-1)).toBe(0);
    expect(d.pdf(1)).toBeGreaterThan(0);
  });

  it("cdf", () => {
    expect(d.cdf(0)).toBe(0);
    expect(d.cdf(-1)).toBe(0);
    expect(d.cdf(1)).toBeGreaterThan(0);
    expect(d.cdf(1)).toBeLessThan(1);
  });

  it("ppf inverts cdf", () => {
    const x = 1.5;
    const p = d.cdf(x);
    closeTo(d.ppf(p), x, 0.05);
  });

  it("ppf boundaries", () => {
    expect(d.ppf(0)).toBe(0);
    expect(d.ppf(1)).toBe(Infinity);
    expect(() => d.ppf(-0.1)).toThrow();
    expect(() => d.ppf(1.1)).toThrow();
  });

  it("sf = 1 - cdf", () => {
    closeTo(d.sf(1), 1 - d.cdf(1));
  });

  it("rvs", () => {
    const samples = d.rvs(10);
    expect(samples.length).toBe(10);
    for (const s of samples) expect(s).toBeGreaterThan(0);
  });

  it("mean and variance", () => {
    closeTo(d.mean(), Math.exp(0.5));
    closeTo(d.variance(), (Math.E - 1) * Math.E);
  });

  it("entropy is finite", () => {
    expect(Number.isFinite(d.entropy())).toBe(true);
  });

  it("rejects invalid sigma", () => {
    expect(() => lognorm(0, 0)).toThrow();
    expect(() => lognorm(0, -1)).toThrow();
  });
});

describe("Weibull Distribution", () => {
  const d = weibull(2, 1);

  it("pdf", () => {
    expect(d.pdf(-1)).toBe(0);
    expect(d.pdf(0)).toBe(0); // k=2, x=0 -> 0
    expect(d.pdf(0.5)).toBeGreaterThan(0);
  });

  it("pdf at x=0 with k=1", () => {
    const d1 = weibull(1, 1);
    closeTo(d1.pdf(0), 1); // k/lam = 1 when k=1
  });

  it("cdf", () => {
    expect(d.cdf(0)).toBe(0);
    expect(d.cdf(-1)).toBe(0);
    expect(d.cdf(1)).toBeGreaterThan(0);
  });

  it("ppf inverts cdf", () => {
    const x = 0.7;
    const p = d.cdf(x);
    closeTo(d.ppf(p), x, 0.01);
  });

  it("ppf boundaries", () => {
    expect(d.ppf(0)).toBe(0);
    expect(d.ppf(1)).toBe(Infinity);
    expect(() => d.ppf(-0.1)).toThrow();
    expect(() => d.ppf(1.1)).toThrow();
  });

  it("sf = 1 - cdf", () => {
    closeTo(d.sf(0.5), 1 - d.cdf(0.5));
  });

  it("rvs", () => {
    expect(d.rvs(10).length).toBe(10);
  });

  it("mean and variance are finite", () => {
    expect(Number.isFinite(d.mean())).toBe(true);
    expect(Number.isFinite(d.variance())).toBe(true);
  });

  it("entropy is finite", () => {
    expect(Number.isFinite(d.entropy())).toBe(true);
  });

  it("rejects invalid params", () => {
    expect(() => weibull(0, 1)).toThrow();
    expect(() => weibull(2, 0)).toThrow();
    expect(() => weibull(-1, 1)).toThrow();
    expect(() => weibull(2, -1)).toThrow();
  });
});

describe("Pareto Distribution", () => {
  const d = pareto(3, 1);

  it("pdf", () => {
    expect(d.pdf(0.5)).toBe(0); // below xm
    expect(d.pdf(1)).toBeGreaterThan(0);
    expect(d.pdf(2)).toBeGreaterThan(0);
  });

  it("cdf", () => {
    expect(d.cdf(0.5)).toBe(0);
    expect(d.cdf(1)).toBe(0); // exactly at xm: 1 - (1/1)^3 = 0
    expect(d.cdf(2)).toBeGreaterThan(0);
  });

  it("ppf inverts cdf", () => {
    const x = 2;
    const p = d.cdf(x);
    closeTo(d.ppf(p), x, 0.01);
  });

  it("ppf boundaries", () => {
    closeTo(d.ppf(0), 1); // returns xm
    expect(d.ppf(1)).toBe(Infinity);
    expect(() => d.ppf(-0.1)).toThrow();
    expect(() => d.ppf(1.1)).toThrow();
  });

  it("sf = 1 - cdf", () => {
    closeTo(d.sf(2), 1 - d.cdf(2));
  });

  it("rvs", () => {
    const samples = d.rvs(10);
    expect(samples.length).toBe(10);
    for (const s of samples) expect(s).toBeGreaterThanOrEqual(1);
  });

  it("mean", () => {
    closeTo(pareto(3, 1).mean(), 1.5); // alpha*xm/(alpha-1)
    expect(pareto(1, 1).mean()).toBe(Infinity); // alpha <= 1
  });

  it("variance", () => {
    expect(Number.isFinite(pareto(3, 1).variance())).toBe(true);
    expect(pareto(2, 1).variance()).toBe(Infinity); // alpha <= 2
  });

  it("entropy is finite", () => {
    expect(Number.isFinite(d.entropy())).toBe(true);
  });

  it("rejects invalid params", () => {
    expect(() => pareto(0, 1)).toThrow();
    expect(() => pareto(3, 0)).toThrow();
    expect(() => pareto(-1, 1)).toThrow();
    expect(() => pareto(3, -1)).toThrow();
  });
});

describe("Cauchy Distribution", () => {
  const d = cauchy(0, 1);

  it("pdf at 0 is maximal", () => {
    closeTo(d.pdf(0), 1 / Math.PI);
    expect(d.pdf(0)).toBeGreaterThan(d.pdf(1));
  });

  it("cdf at 0 is 0.5", () => {
    closeTo(d.cdf(0), 0.5);
  });

  it("ppf inverts cdf", () => {
    closeTo(d.ppf(0.5), 0, 0.001);
    closeTo(d.ppf(0.75), 1, 0.01);
  });

  it("ppf boundaries", () => {
    expect(d.ppf(0)).toBe(-Infinity);
    expect(d.ppf(1)).toBe(Infinity);
    expect(() => d.ppf(-0.1)).toThrow();
    expect(() => d.ppf(1.1)).toThrow();
  });

  it("sf = 1 - cdf", () => {
    closeTo(d.sf(0), 0.5);
  });

  it("rvs", () => {
    expect(d.rvs(10).length).toBe(10);
  });

  it("mean and variance are NaN", () => {
    expect(d.mean()).toBeNaN();
    expect(d.variance()).toBeNaN();
  });

  it("entropy", () => {
    closeTo(d.entropy(), Math.log(4 * Math.PI));
  });

  it("non-standard cauchy", () => {
    const d2 = cauchy(5, 2);
    closeTo(d2.cdf(5), 0.5);
    closeTo(d2.entropy(), Math.log(4 * Math.PI * 2));
  });

  it("rejects invalid gamma", () => {
    expect(() => cauchy(0, 0)).toThrow();
    expect(() => cauchy(0, -1)).toThrow();
  });
});

describe("Laplace Distribution", () => {
  const d = laplace(0, 1);

  it("pdf", () => {
    closeTo(d.pdf(0), 0.5);
    closeTo(d.pdf(1), 0.5 * Math.exp(-1));
    closeTo(d.pdf(-1), 0.5 * Math.exp(-1));
  });

  it("cdf", () => {
    closeTo(d.cdf(0), 0.5);
    expect(d.cdf(-10)).toBeGreaterThan(0);
    expect(d.cdf(10)).toBeLessThan(1);
  });

  it("ppf inverts cdf", () => {
    closeTo(d.ppf(0.5), 0, 0.001);
    const x = 1.5;
    const p = d.cdf(x);
    closeTo(d.ppf(p), x, 0.01);
  });

  it("ppf for p < 0.5", () => {
    const x = -2;
    const p = d.cdf(x);
    closeTo(d.ppf(p), x, 0.01);
  });

  it("ppf boundaries", () => {
    expect(d.ppf(0)).toBe(-Infinity);
    expect(d.ppf(1)).toBe(Infinity);
    expect(() => d.ppf(-0.1)).toThrow();
    expect(() => d.ppf(1.1)).toThrow();
  });

  it("sf = 1 - cdf", () => {
    closeTo(d.sf(1), 1 - d.cdf(1));
  });

  it("rvs", () => {
    expect(d.rvs(10).length).toBe(10);
  });

  it("mean and variance", () => {
    closeTo(d.mean(), 0);
    closeTo(d.variance(), 2);
  });

  it("entropy", () => {
    closeTo(d.entropy(), 1 + Math.log(2));
  });

  it("non-standard laplace", () => {
    const d2 = laplace(3, 2);
    closeTo(d2.mean(), 3);
    closeTo(d2.variance(), 8);
  });

  it("rejects invalid scale", () => {
    expect(() => laplace(0, 0)).toThrow();
    expect(() => laplace(0, -1)).toThrow();
  });
});
