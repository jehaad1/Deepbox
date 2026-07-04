import { beforeEach, describe, expect, it } from "vitest";
import { pareto, rayleigh, setSeed, vonmises, zipf } from "../src/random";

beforeEach(() => {
  setSeed(42);
});

describe("vonmises", () => {
  it("generates samples with correct shape", () => {
    const t = vonmises(0, 1, [100]);
    expect(t.shape).toEqual([100]);
  });

  it("samples are in [-pi, pi)", () => {
    const t = vonmises(0, 5, [1000]);
    for (let i = 0; i < 1000; i++) {
      const v = t.at(i) as number;
      expect(v).toBeGreaterThanOrEqual(-Math.PI);
      expect(v).toBeLessThan(Math.PI + 0.01); // small epsilon for float
    }
  });

  it("kappa=0 gives uniform-like distribution", () => {
    const t = vonmises(0, 0, [500]);
    let sumPos = 0;
    let sumNeg = 0;
    for (let i = 0; i < 500; i++) {
      const v = t.at(i) as number;
      if (v > 0) sumPos++;
      else sumNeg++;
    }
    // Should be roughly 50/50
    expect(sumPos).toBeGreaterThan(150);
    expect(sumNeg).toBeGreaterThan(150);
  });

  it("throws on invalid kappa", () => {
    expect(() => vonmises(0, -1, [1])).toThrow();
  });

  it("throws on non-finite mu", () => {
    expect(() => vonmises(Infinity, 1, [1])).toThrow();
  });

  it("supports 2D shape", () => {
    const t = vonmises(0, 2, [3, 4]);
    expect(t.shape).toEqual([3, 4]);
  });
});

describe("pareto", () => {
  it("generates samples with correct shape", () => {
    const t = pareto(2, 1, [100]);
    expect(t.shape).toEqual([100]);
  });

  it("all samples >= xm", () => {
    const xm = 2;
    const t = pareto(3, xm, [500]);
    for (let i = 0; i < 500; i++) {
      expect(t.at(i) as number).toBeGreaterThanOrEqual(xm);
    }
  });

  it("mean approximates alpha*xm/(alpha-1) for alpha > 1", () => {
    const alpha = 3;
    const xm = 1;
    const t = pareto(alpha, xm, [10000]);
    let sum = 0;
    for (let i = 0; i < 10000; i++) {
      sum += t.at(i) as number;
    }
    const mean = sum / 10000;
    const expected = (alpha * xm) / (alpha - 1); // 1.5
    expect(mean).toBeGreaterThan(expected * 0.8);
    expect(mean).toBeLessThan(expected * 1.3);
  });

  it("throws on invalid alpha", () => {
    expect(() => pareto(0, 1, [1])).toThrow();
    expect(() => pareto(-1, 1, [1])).toThrow();
  });

  it("throws on invalid xm", () => {
    expect(() => pareto(2, 0, [1])).toThrow();
    expect(() => pareto(2, -1, [1])).toThrow();
  });
});

describe("rayleigh", () => {
  it("generates samples with correct shape", () => {
    const t = rayleigh(1, [100]);
    expect(t.shape).toEqual([100]);
  });

  it("all samples are positive", () => {
    const t = rayleigh(1, [500]);
    for (let i = 0; i < 500; i++) {
      expect(t.at(i) as number).toBeGreaterThan(0);
    }
  });

  it("mean approximates sigma*sqrt(pi/2)", () => {
    const sigma = 2;
    const t = rayleigh(sigma, [10000]);
    let sum = 0;
    for (let i = 0; i < 10000; i++) {
      sum += t.at(i) as number;
    }
    const mean = sum / 10000;
    const expected = sigma * Math.sqrt(Math.PI / 2);
    expect(mean).toBeCloseTo(expected, 0);
  });

  it("throws on invalid sigma", () => {
    expect(() => rayleigh(0, [1])).toThrow();
    expect(() => rayleigh(-1, [1])).toThrow();
  });

  it("supports scalar output", () => {
    const t = rayleigh(1, []);
    expect(t.shape).toEqual([]);
  });
});

describe("zipf", () => {
  it("generates samples with correct shape", () => {
    const t = zipf(2, [100]);
    expect(t.shape).toEqual([100]);
  });

  it("all samples are >= 1", () => {
    const t = zipf(2, [500]);
    for (let i = 0; i < 500; i++) {
      const v = Number(t.at(i));
      expect(v).toBeGreaterThanOrEqual(1);
    }
  });

  it("smaller values are more frequent (power law)", () => {
    const t = zipf(2.5, [5000]);
    let count1 = 0;
    let count10plus = 0;
    for (let i = 0; i < 5000; i++) {
      const v = Number(t.at(i));
      if (v === 1) count1++;
      if (v >= 10) count10plus++;
    }
    // k=1 should be much more frequent than k>=10
    expect(count1).toBeGreaterThan(count10plus);
  });

  it("throws on s <= 1", () => {
    expect(() => zipf(1, [1])).toThrow();
    expect(() => zipf(0.5, [1])).toThrow();
  });

  it("throws on non-finite s", () => {
    expect(() => zipf(Infinity, [1])).toThrow();
  });
});
