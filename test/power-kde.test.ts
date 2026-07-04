import { describe, expect, it } from "vitest";
import { GaussianKDE, gaussian_kde, tTestPower } from "../src/stats";

describe("tTestPower", () => {
  it("solves for power given effectSize, nObs, alpha", () => {
    const result = tTestPower({ effectSize: 0.5, nObs: 64, alpha: 0.05 });
    expect(result.power).toBeGreaterThan(0.7);
    expect(result.power).toBeLessThan(0.95);
    expect(result.effectSize).toBe(0.5);
    expect(result.nObs).toBe(64);
    expect(result.alpha).toBe(0.05);
  });

  it("solves for nObs given effectSize, alpha, power", () => {
    const result = tTestPower({ effectSize: 0.5, alpha: 0.05, power: 0.8 });
    expect(result.nObs).toBeGreaterThanOrEqual(50);
    expect(result.nObs).toBeLessThanOrEqual(80);
    expect(result.power).toBeGreaterThanOrEqual(0.8);
  });

  it("solves for effectSize given nObs, alpha, power", () => {
    const result = tTestPower({ nObs: 64, alpha: 0.05, power: 0.8 });
    expect(result.effectSize).toBeGreaterThan(0.3);
    expect(result.effectSize).toBeLessThan(0.7);
  });

  it("solves for alpha given effectSize, nObs, power", () => {
    const result = tTestPower({ effectSize: 0.5, nObs: 64, power: 0.8 });
    expect(result.alpha).toBeGreaterThan(0.01);
    expect(result.alpha).toBeLessThan(0.2);
  });

  it("larger sample size gives more power", () => {
    const small = tTestPower({ effectSize: 0.5, nObs: 20, alpha: 0.05 });
    const large = tTestPower({ effectSize: 0.5, nObs: 100, alpha: 0.05 });
    expect(large.power).toBeGreaterThan(small.power);
  });

  it("larger effect size gives more power", () => {
    const small = tTestPower({ effectSize: 0.2, nObs: 50, alpha: 0.05 });
    const large = tTestPower({ effectSize: 0.8, nObs: 50, alpha: 0.05 });
    expect(large.power).toBeGreaterThan(small.power);
  });

  it("supports one-tailed alternative", () => {
    const twoTailed = tTestPower({
      effectSize: 0.5,
      nObs: 64,
      alpha: 0.05,
      alternative: 2,
    });
    const oneTailed = tTestPower({
      effectSize: 0.5,
      nObs: 64,
      alpha: 0.05,
      alternative: 1,
    });
    expect(oneTailed.power).toBeGreaterThan(twoTailed.power);
  });

  // Error cases
  it("throws when more than one parameter is missing", () => {
    expect(() => tTestPower({ effectSize: 0.5 })).toThrow();
  });

  it("throws when no parameter is missing", () => {
    expect(() => tTestPower({ effectSize: 0.5, nObs: 64, alpha: 0.05, power: 0.8 })).toThrow();
  });

  it("throws on invalid effectSize", () => {
    expect(() => tTestPower({ effectSize: -1, nObs: 64, alpha: 0.05 })).toThrow();
  });

  it("throws on invalid alpha", () => {
    expect(() => tTestPower({ effectSize: 0.5, nObs: 64, alpha: 0 })).toThrow();
    expect(() => tTestPower({ effectSize: 0.5, nObs: 64, alpha: 1 })).toThrow();
  });

  it("throws on invalid nObs", () => {
    expect(() => tTestPower({ effectSize: 0.5, nObs: 1, alpha: 0.05 })).toThrow();
  });
});

describe("gaussian_kde", () => {
  it("creates a KDE from data", () => {
    const data = [1, 2, 3, 4, 5];
    const kde = gaussian_kde(data);
    expect(kde).toBeInstanceOf(GaussianKDE);
    expect(kde.nData).toBe(5);
    expect(kde.bandwidth).toBeGreaterThan(0);
  });

  it("evaluates density at given points", () => {
    const data = [0, 0, 0, 0, 0]; // all zeros
    const kde = gaussian_kde(data);
    const density = kde.evaluate([0]);
    // Density at the data point should be high
    expect(density[0]).toBeGreaterThan(0);
  });

  it("density is highest near data concentration", () => {
    const data = [1, 1, 1, 1, 1, 10, 10, 10];
    const kde = gaussian_kde(data, { bw_method: 0.5 });
    const density = kde.evaluate([1, 5, 10]);
    // Density at 1 (5 points) should be higher than at 5 (no points)
    expect(density[0]!).toBeGreaterThan(density[1]!);
    // Density at 10 (3 points) should be higher than at 5 (no points)
    expect(density[2]!).toBeGreaterThan(density[1]!);
  });

  it("density integrates to approximately 1", () => {
    const data = [1, 2, 3, 4, 5];
    const kde = gaussian_kde(data);
    // Numerical integration via trapezoidal rule
    const lo = -10;
    const hi = 15;
    const steps = 1000;
    const dx = (hi - lo) / steps;
    const points: number[] = [];
    for (let i = 0; i <= steps; i++) {
      points.push(lo + i * dx);
    }
    const density = kde.evaluate(points);
    let integral = 0;
    for (let i = 0; i < steps; i++) {
      integral += (((density[i] ?? 0) + (density[i + 1] ?? 0)) * dx) / 2;
    }
    expect(integral).toBeCloseTo(1, 1);
  });

  it("supports Scott's rule (default)", () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const kde = gaussian_kde(data, { bw_method: "scott" });
    expect(kde.bandwidth).toBeGreaterThan(0);
  });

  it("supports Silverman's rule", () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const kde = gaussian_kde(data, { bw_method: "silverman" });
    expect(kde.bandwidth).toBeGreaterThan(0);
  });

  it("supports explicit bandwidth", () => {
    const data = [1, 2, 3, 4, 5];
    const kde = gaussian_kde(data, { bw_method: 0.5 });
    expect(kde.bandwidth).toBe(0.5);
  });

  it("logDensity returns finite values", () => {
    const data = [1, 2, 3, 4, 5];
    const kde = gaussian_kde(data);
    const logDensity = kde.logDensity([1, 2, 3]);
    for (let i = 0; i < logDensity.length; i++) {
      expect(Number.isFinite(logDensity[i])).toBe(true);
    }
  });

  it("works with Float64Array input", () => {
    const data = new Float64Array([1, 2, 3, 4, 5]);
    const kde = gaussian_kde(data);
    expect(kde.nData).toBe(5);
  });

  // Error cases
  it("throws on empty data", () => {
    expect(() => gaussian_kde([])).toThrow();
  });

  it("throws on non-positive bandwidth", () => {
    expect(() => gaussian_kde([1, 2, 3], { bw_method: 0 })).toThrow();
    expect(() => gaussian_kde([1, 2, 3], { bw_method: -1 })).toThrow();
  });
});
