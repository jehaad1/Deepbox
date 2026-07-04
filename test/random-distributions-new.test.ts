import { beforeEach, describe, expect, it } from "vitest";
import {
  cauchy,
  chi2,
  f_distribution,
  hypergeometric,
  laplace,
  negative_binomial,
  setSeed,
  student_t,
  triangular,
  weibull,
} from "../src/random";

beforeEach(() => {
  setSeed(12345);
});

describe("chi2", () => {
  it("produces correct shape", () => {
    const t = chi2(5, [10, 3]);
    expect(t.shape).toEqual([10, 3]);
  });

  it("produces positive values", () => {
    const t = chi2(3, [100]);
    const arr = t.toArray() as number[];
    for (const v of arr) {
      expect(v).toBeGreaterThan(0);
    }
  });

  it("mean approximates df for large sample", () => {
    const df = 10;
    const t = chi2(df, [5000]);
    const arr = t.toArray() as number[];
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    expect(mean).toBeCloseTo(df, 0);
  });

  it("throws for invalid df", () => {
    expect(() => chi2(0)).toThrow();
    expect(() => chi2(-1)).toThrow();
  });
});

describe("student_t", () => {
  it("produces correct shape", () => {
    const t = student_t(10, [5, 4]);
    expect(t.shape).toEqual([5, 4]);
  });

  it("mean approximates 0 for large sample", () => {
    const t = student_t(30, [5000]);
    const arr = t.toArray() as number[];
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    expect(Math.abs(mean)).toBeLessThan(0.1);
  });

  it("throws for invalid df", () => {
    expect(() => student_t(0)).toThrow();
    expect(() => student_t(-5)).toThrow();
  });
});

describe("f_distribution", () => {
  it("produces correct shape", () => {
    const t = f_distribution(5, 10, [20]);
    expect(t.shape).toEqual([20]);
  });

  it("produces positive values", () => {
    const t = f_distribution(5, 10, [100]);
    const arr = t.toArray() as number[];
    for (const v of arr) {
      expect(v).toBeGreaterThan(0);
    }
  });

  it("mean approximates dfd/(dfd-2) for dfd > 2", () => {
    const dfn = 5;
    const dfd = 20;
    const t = f_distribution(dfn, dfd, [5000]);
    const arr = t.toArray() as number[];
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const expected = dfd / (dfd - 2);
    expect(mean).toBeCloseTo(expected, 0);
  });

  it("throws for invalid params", () => {
    expect(() => f_distribution(0, 10)).toThrow();
    expect(() => f_distribution(5, 0)).toThrow();
  });
});

describe("laplace", () => {
  it("produces correct shape", () => {
    const t = laplace(0, 1, [3, 4]);
    expect(t.shape).toEqual([3, 4]);
  });

  it("mean approximates loc", () => {
    const loc = 5;
    const t = laplace(loc, 2, [5000]);
    const arr = t.toArray() as number[];
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    expect(mean).toBeCloseTo(loc, 0);
  });

  it("throws for invalid scale", () => {
    expect(() => laplace(0, 0)).toThrow();
    expect(() => laplace(0, -1)).toThrow();
  });
});

describe("cauchy", () => {
  it("produces correct shape", () => {
    const t = cauchy(0, 1, [10]);
    expect(t.shape).toEqual([10]);
  });

  it("median approximates loc", () => {
    const loc = 3;
    const t = cauchy(loc, 1, [5000]);
    const arr = (t.toArray() as number[]).sort((a, b) => a - b);
    const median = arr[Math.floor(arr.length / 2)]!;
    expect(Math.abs(median - loc)).toBeLessThan(0.2);
  });

  it("throws for invalid scale", () => {
    expect(() => cauchy(0, 0)).toThrow();
    expect(() => cauchy(0, -1)).toThrow();
  });
});

describe("weibull", () => {
  it("produces correct shape", () => {
    const t = weibull(1.5, 1, [8]);
    expect(t.shape).toEqual([8]);
  });

  it("produces positive values", () => {
    const t = weibull(2, 1, [100]);
    const arr = t.toArray() as number[];
    for (const v of arr) {
      expect(v).toBeGreaterThan(0);
    }
  });

  it("with shape=1 is exponential (mean ≈ scale)", () => {
    const scale = 2;
    const t = weibull(1, scale, [5000]);
    const arr = t.toArray() as number[];
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    expect(mean).toBeCloseTo(scale, 0);
  });

  it("throws for invalid params", () => {
    expect(() => weibull(0, 1)).toThrow();
    expect(() => weibull(1, 0)).toThrow();
  });
});

describe("triangular", () => {
  it("produces correct shape", () => {
    const t = triangular(0, 0.5, 1, [12]);
    expect(t.shape).toEqual([12]);
  });

  it("all values in [left, right]", () => {
    const t = triangular(2, 5, 8, [1000]);
    const arr = t.toArray() as number[];
    for (const v of arr) {
      expect(v).toBeGreaterThanOrEqual(2);
      expect(v).toBeLessThanOrEqual(8);
    }
  });

  it("mean approximates (left+mode+right)/3", () => {
    const left = 0,
      mode = 3,
      right = 6;
    const t = triangular(left, mode, right, [5000]);
    const arr = t.toArray() as number[];
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    expect(mean).toBeCloseTo((left + mode + right) / 3, 0);
  });

  it("throws for left >= right", () => {
    expect(() => triangular(5, 3, 5)).toThrow();
    expect(() => triangular(5, 3, 4)).toThrow();
  });

  it("throws for mode out of range", () => {
    expect(() => triangular(0, -1, 1)).toThrow();
    expect(() => triangular(0, 2, 1)).toThrow();
  });
});

describe("negative_binomial", () => {
  it("produces correct shape", () => {
    const t = negative_binomial(5, 0.5, [10]);
    expect(t.shape).toEqual([10]);
  });

  it("produces non-negative integers", () => {
    const t = negative_binomial(3, 0.4, [100]);
    const arr = t.toArray() as number[];
    for (const v of arr) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(Number.isInteger(v)).toBe(true);
    }
  });

  it("p=1 gives all zeros", () => {
    const t = negative_binomial(5, 1, [20]);
    const arr = t.toArray() as number[];
    for (const v of arr) {
      expect(v).toBe(0);
    }
  });

  it("throws for invalid params", () => {
    expect(() => negative_binomial(0, 0.5)).toThrow();
    expect(() => negative_binomial(5, 0)).toThrow();
    expect(() => negative_binomial(5, 1.5)).toThrow();
  });
});

describe("hypergeometric", () => {
  it("produces correct shape", () => {
    const t = hypergeometric(10, 5, 7, [15]);
    expect(t.shape).toEqual([15]);
  });

  it("values bounded by [max(0, nsample-nbad), min(nsample, ngood)]", () => {
    const ngood = 10,
      nbad = 5,
      nsample = 7;
    const t = hypergeometric(ngood, nbad, nsample, [200]);
    const arr = t.toArray() as number[];
    const lo = Math.max(0, nsample - nbad);
    const hi = Math.min(nsample, ngood);
    for (const v of arr) {
      expect(v).toBeGreaterThanOrEqual(lo);
      expect(v).toBeLessThanOrEqual(hi);
      expect(Number.isInteger(v)).toBe(true);
    }
  });

  it("mean approximates nsample * ngood / (ngood+nbad)", () => {
    const ngood = 50,
      nbad = 50,
      nsample = 20;
    const t = hypergeometric(ngood, nbad, nsample, [5000]);
    const arr = t.toArray() as number[];
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const expected = (nsample * ngood) / (ngood + nbad);
    expect(mean).toBeCloseTo(expected, 0);
  });

  it("throws for nsample > population", () => {
    expect(() => hypergeometric(3, 2, 6)).toThrow();
  });

  it("throws for negative params", () => {
    expect(() => hypergeometric(-1, 5, 3)).toThrow();
    expect(() => hypergeometric(5, -1, 3)).toThrow();
    expect(() => hypergeometric(5, 5, -1)).toThrow();
  });
});
