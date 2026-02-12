import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  geometricMean,
  harmonicMean,
  kurtosis,
  mean,
  median,
  mode,
  moment,
  percentile,
  quantile,
  skewness,
  std,
  trimMean,
  variance,
} from "../src/stats";

describe("Central Tendency", () => {
  it("should calculate mean", () => {
    const data = tensor([1, 2, 3, 4, 5]);
    const result = mean(data);
    expect(Number(result.data[0])).toBeCloseTo(3, 10);
  });

  it("should calculate median", () => {
    const data = tensor([1, 2, 3, 4, 5]);
    const result = median(data);
    expect(Number(result.data[0])).toBeCloseTo(3, 10);
  });

  it("should calculate mode", () => {
    const data = tensor([1, 2, 2, 3, 3, 3, 4]);
    const result = mode(data);
    expect(Number(result.data[0])).toBe(3);
  });
});

describe("Dispersion Measures", () => {
  it("should calculate standard deviation", () => {
    const data = tensor([1, 2, 3, 4, 5]);
    const result = std(data);
    expect(Number(result.data[0])).toBeCloseTo(Math.sqrt(2), 10);
  });

  it("should calculate variance", () => {
    const data = tensor([1, 2, 3, 4, 5]);
    const result = variance(data);
    expect(Number(result.data[0])).toBeCloseTo(2, 10);
  });
});

describe("Shape Measures", () => {
  it("should calculate skewness", () => {
    const data = tensor([1, 2, 3, 4, 5]);
    const result = skewness(data);
    expect(Number(result.data[0])).toBeCloseTo(0, 10);
  });

  it("should calculate kurtosis", () => {
    const data = tensor([1, 2, 3, 4, 5]);
    const result = kurtosis(data);
    expect(Number.isFinite(Number(result.data[0]))).toBe(true);
  });
});

describe("Quantiles", () => {
  it("should calculate quantile", () => {
    const data = tensor([1, 2, 3, 4, 5]);
    const result = quantile(data, 0.5);
    expect(Number(result.data[0])).toBeCloseTo(3, 10);
  });

  it("should calculate percentile", () => {
    const data = tensor([1, 2, 3, 4, 5]);
    const result = percentile(data, 50);
    expect(Number(result.data[0])).toBeCloseTo(3, 10);
  });
});

describe("Moments", () => {
  it("should calculate nth moment", () => {
    const data = tensor([1, 2, 3, 4, 5]);
    const result = moment(data, 2);
    expect(Number(result.data[0])).toBeCloseTo(2, 10);
  });
});

describe("Alternative Means", () => {
  it("should calculate geometric mean", () => {
    const data = tensor([1, 2, 4, 8]);
    const result = geometricMean(data);
    expect(Number(result.data[0])).toBeCloseTo(64 ** 0.25, 10);
  });

  it("should calculate harmonic mean", () => {
    const data = tensor([1, 2, 4]);
    const result = harmonicMean(data);
    expect(Number(result.data[0])).toBeCloseTo(3 / (1 + 0.5 + 0.25), 10);
  });

  it("should calculate trimmed mean", () => {
    const data = tensor([1, 2, 3, 4, 5, 100]);
    const result = trimMean(data, 0.2);
    expect(Number(result.data[0])).toBeCloseTo(3.5, 10);
  });
});
