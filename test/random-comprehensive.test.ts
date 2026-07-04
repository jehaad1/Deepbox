import { beforeEach, describe, expect, it } from "vitest";
import { DataValidationError, DTypeError, InvalidParameterError } from "../src/core";
import { tensor, transpose } from "../src/ndarray";
import {
  beta,
  binomial,
  choice,
  clearSeed,
  exponential,
  gamma,
  getSeed,
  normal,
  permutation,
  poisson,
  rand,
  randint,
  randn,
  setSeed,
  shuffle,
  uniform,
} from "../src/random";
import { numRawData } from "./_helpers";

function mean(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function variance(arr: number[]): number {
  const m = mean(arr);
  return arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length;
}

function std(arr: number[]): number {
  return Math.sqrt(variance(arr));
}

describe("Random Module - Comprehensive Test Suite", () => {
  beforeEach(() => {
    setSeed(12345);
  });

  describe("Seed Management", () => {
    it("should set and retrieve seed correctly", () => {
      setSeed(42);
      expect(getSeed()).toBe(42);
    });

    it("should handle negative seeds", () => {
      setSeed(-100);
      expect(getSeed()).toBe(-100);
    });

    it("should handle zero seed", () => {
      setSeed(0);
      expect(getSeed()).toBe(0);
    });

    it("should handle large seeds", () => {
      setSeed(2147483647);
      expect(getSeed()).toBe(2147483647);
    });

    it("should handle decimal seeds", () => {
      setSeed(42.7);
      expect(getSeed()).toBe(42.7);
    });

    it("should throw on NaN seed", () => {
      expect(() => setSeed(NaN)).toThrow(InvalidParameterError);
    });

    it("should throw on Infinity seed", () => {
      expect(() => setSeed(Infinity)).toThrow(InvalidParameterError);
    });

    it("should throw on -Infinity seed", () => {
      expect(() => setSeed(-Infinity)).toThrow(InvalidParameterError);
    });

    it("should produce deterministic sequences", () => {
      setSeed(999);
      const seq1 = Array.from({ length: 10 }, () => rand([1]).data[0]);
      setSeed(999);
      const seq2 = Array.from({ length: 10 }, () => rand([1]).data[0]);
      expect(seq1).toEqual(seq2);
    });

    it("should produce different sequences for different seeds", () => {
      setSeed(1);
      const seq1 = Array.from({ length: 10 }, () => rand([1]).data[0]);
      setSeed(2);
      const seq2 = Array.from({ length: 10 }, () => rand([1]).data[0]);
      expect(seq1).not.toEqual(seq2);
    });

    it("should clear seed and revert to secure randomness", () => {
      setSeed(123);
      clearSeed();
      expect(getSeed()).toBeUndefined();
      const x = rand([5]);
      expect(x.size).toBe(5);
    });
  });

  describe("rand() - Uniform [0, 1)", () => {
    it("should generate scalar", () => {
      const x = rand([]);
      expect(x.shape).toEqual([]);
      expect(x.size).toBe(1);
      const val = x.data[0];
      expect(typeof val).toBe("number");
    });

    it("should generate 1D array", () => {
      const x = rand([10]);
      expect(x.shape).toEqual([10]);
      expect(x.size).toBe(10);
    });

    it("should generate 2D array", () => {
      const x = rand([3, 4]);
      expect(x.shape).toEqual([3, 4]);
      expect(x.size).toBe(12);
    });

    it("should generate 3D array", () => {
      const x = rand([2, 3, 4]);
      expect(x.shape).toEqual([2, 3, 4]);
      expect(x.size).toBe(24);
    });

    it("should generate 4D array", () => {
      const x = rand([2, 2, 2, 2]);
      expect(x.shape).toEqual([2, 2, 2, 2]);
      expect(x.size).toBe(16);
    });

    it("should have values in [0, 1)", () => {
      setSeed(42);
      const x = rand([1000]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => v >= 0 && v < 1)).toBe(true);
    });

    it("should have mean close to 0.5", () => {
      setSeed(42);
      const x = rand([10000]);
      const m = mean(numRawData(x.data));
      expect(m).toBeGreaterThan(0.48);
      expect(m).toBeLessThan(0.52);
    });

    it("should have variance close to 1/12", () => {
      setSeed(42);
      const x = rand([10000]);
      const v = variance(numRawData(x.data));
      expect(v).toBeGreaterThan(0.08);
      expect(v).toBeLessThan(0.09);
    });

    it("should support float32 dtype", () => {
      const x = rand([5], { dtype: "float32" });
      expect(x.dtype).toBe("float32");
    });

    it("should support float64 dtype", () => {
      const x = rand([5], { dtype: "float64" });
      expect(x.dtype).toBe("float64");
    });

    it("should throw on non-float dtype", () => {
      expect(() => rand([5], { dtype: "int32" })).toThrow(DTypeError);
    });

    it("should throw on non-array shape", () => {
      // @ts-expect-error - invalid shape should be rejected at runtime
      expect(() => rand(5)).toThrow(DataValidationError);
    });

    it("should handle empty shape", () => {
      const x = rand([]);
      expect(x.size).toBe(1);
    });

    it("should handle large arrays", () => {
      const x = rand([1000, 100]);
      expect(x.size).toBe(100000);
    });
  });

  describe("randn() - Standard Normal", () => {
    it("should generate scalar", () => {
      const x = randn([]);
      expect(x.shape).toEqual([]);
    });

    it("should generate 1D array", () => {
      const x = randn([10]);
      expect(x.shape).toEqual([10]);
    });

    it("should generate 2D array", () => {
      const x = randn([3, 4]);
      expect(x.shape).toEqual([3, 4]);
    });

    it("should generate 3D array", () => {
      const x = randn([2, 3, 4]);
      expect(x.shape).toEqual([2, 3, 4]);
    });

    it("should have mean close to 0", () => {
      setSeed(42);
      const x = randn([10000]);
      const m = mean(numRawData(x.data));
      expect(Math.abs(m)).toBeLessThan(0.05);
    });

    it("should have std close to 1", () => {
      setSeed(42);
      const x = randn([10000]);
      const s = std(numRawData(x.data));
      expect(s).toBeGreaterThan(0.95);
      expect(s).toBeLessThan(1.05);
    });

    it("should support float32 dtype", () => {
      const x = randn([5], { dtype: "float32" });
      expect(x.dtype).toBe("float32");
    });

    it("should support float64 dtype", () => {
      const x = randn([5], { dtype: "float64" });
      expect(x.dtype).toBe("float64");
    });

    it("should throw on non-float dtype", () => {
      expect(() => randn([5], { dtype: "int32" })).toThrow(DTypeError);
    });

    it("should throw on non-array shape", () => {
      // @ts-expect-error - invalid shape should be rejected at runtime
      expect(() => randn(5)).toThrow(DataValidationError);
    });

    it("should be deterministic when seeded", () => {
      setSeed(123);
      const a = randn([5]);
      setSeed(123);
      const b = randn([5]);
      expect(numRawData(a.data)).toEqual(numRawData(b.data));
    });

    it("should handle large arrays", () => {
      const x = randn([1000, 100]);
      expect(x.size).toBe(100000);
    });
  });

  describe("randint() - Random Integers", () => {
    it("should generate integers in range", () => {
      setSeed(42);
      const x = randint(0, 10, [100]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => Number.isInteger(v) && v >= 0 && v < 10)).toBe(true);
    });

    it("should generate scalar", () => {
      const x = randint(0, 10, []);
      expect(x.shape).toEqual([]);
    });

    it("should generate 1D array", () => {
      const x = randint(0, 10, [10]);
      expect(x.shape).toEqual([10]);
    });

    it("should generate 2D array", () => {
      const x = randint(0, 10, [3, 4]);
      expect(x.shape).toEqual([3, 4]);
    });

    it("should handle negative ranges", () => {
      setSeed(42);
      const x = randint(-10, 10, [100]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => v >= -10 && v < 10)).toBe(true);
    });

    it("should handle single value range", () => {
      const x = randint(5, 6, [10]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => v === 5)).toBe(true);
    });

    it("should throw when high <= low", () => {
      expect(() => randint(10, 10, [1])).toThrow(InvalidParameterError);
      expect(() => randint(10, 5, [1])).toThrow(InvalidParameterError);
    });

    it("should throw on non-integer low", () => {
      expect(() => randint(1.5, 10, [1])).toThrow(InvalidParameterError);
    });

    it("should throw on non-integer high", () => {
      expect(() => randint(0, 10.5, [1])).toThrow(InvalidParameterError);
    });

    it("should throw on non-finite low", () => {
      expect(() => randint(NaN, 10, [1])).toThrow(InvalidParameterError);
      expect(() => randint(Infinity, 10, [1])).toThrow(InvalidParameterError);
    });

    it("should throw on non-finite high", () => {
      expect(() => randint(0, NaN, [1])).toThrow(InvalidParameterError);
      expect(() => randint(0, Infinity, [1])).toThrow(InvalidParameterError);
    });

    it("should be deterministic when seeded", () => {
      setSeed(789);
      const a = randint(0, 100, [10]);
      setSeed(789);
      const b = randint(0, 100, [10]);
      expect(numRawData(a.data)).toEqual(numRawData(b.data));
    });

    it("should support int32 dtype", () => {
      const x = randint(0, 10, [5], { dtype: "int32" });
      expect(x.dtype).toBe("int32");
    });

    it("should throw on non-integer dtype", () => {
      expect(() => randint(0, 10, [5], { dtype: "float32" })).toThrow(DTypeError);
    });

    it("should throw when bounds exceed int32 range for int32 output", () => {
      expect(() => randint(-2147483649, 10, [1])).toThrow(InvalidParameterError);
      expect(() => randint(0, 2147483649, [1])).toThrow(InvalidParameterError);
    });
  });

  describe("uniform() - Continuous Uniform", () => {
    it("should generate values in range", () => {
      setSeed(42);
      const x = uniform(-1, 1, [100]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => v >= -1 && v <= 1)).toBe(true);
    });

    it("should generate scalar", () => {
      const x = uniform(0, 1, []);
      expect(x.shape).toEqual([]);
    });

    it("should use default parameters", () => {
      const x = uniform();
      expect(x.shape).toEqual([]);
    });

    it("should generate 1D array", () => {
      const x = uniform(0, 1, [10]);
      expect(x.shape).toEqual([10]);
    });

    it("should generate 2D array", () => {
      const x = uniform(0, 1, [3, 4]);
      expect(x.shape).toEqual([3, 4]);
    });

    it("should have mean close to (low+high)/2", () => {
      setSeed(42);
      const x = uniform(-5, 5, [10000]);
      const m = mean(numRawData(x.data));
      expect(m).toBeGreaterThan(-0.2);
      expect(m).toBeLessThan(0.2);
    });

    it("should throw when high < low", () => {
      expect(() => uniform(1, -1, [1])).toThrow(InvalidParameterError);
    });

    it("should allow high == low", () => {
      const x = uniform(5, 5, [10]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => v === 5)).toBe(true);
    });

    it("should throw on non-finite low", () => {
      expect(() => uniform(NaN, 1, [1])).toThrow(InvalidParameterError);
      expect(() => uniform(Infinity, 1, [1])).toThrow(InvalidParameterError);
    });

    it("should throw on non-finite high", () => {
      expect(() => uniform(0, NaN, [1])).toThrow(InvalidParameterError);
      expect(() => uniform(0, Infinity, [1])).toThrow(InvalidParameterError);
    });

    it("should throw on non-float dtype", () => {
      expect(() => uniform(0, 1, [1], { dtype: "int32" })).toThrow(DTypeError);
    });

    it("should be deterministic when seeded", () => {
      setSeed(456);
      const a = uniform(-10, 10, [10]);
      setSeed(456);
      const b = uniform(-10, 10, [10]);
      expect(numRawData(a.data)).toEqual(numRawData(b.data));
    });
  });

  describe("normal() - Normal Distribution", () => {
    it("should generate values from normal distribution", () => {
      setSeed(42);
      const x = normal(0, 1, [10000]);
      const m = mean(numRawData(x.data));
      const s = std(numRawData(x.data));
      expect(Math.abs(m)).toBeLessThan(0.05);
      expect(s).toBeGreaterThan(0.95);
      expect(s).toBeLessThan(1.05);
    });

    it("should generate scalar", () => {
      const x = normal(0, 1, []);
      expect(x.shape).toEqual([]);
    });

    it("should use default parameters", () => {
      const x = normal();
      expect(x.shape).toEqual([]);
    });

    it("should generate 1D array", () => {
      const x = normal(0, 1, [10]);
      expect(x.shape).toEqual([10]);
    });

    it("should generate 2D array", () => {
      const x = normal(0, 1, [3, 4]);
      expect(x.shape).toEqual([3, 4]);
    });

    it("should respect mean parameter", () => {
      setSeed(42);
      const x = normal(10, 1, [10000]);
      const m = mean(numRawData(x.data));
      expect(m).toBeGreaterThan(9.9);
      expect(m).toBeLessThan(10.1);
    });

    it("should respect std parameter", () => {
      setSeed(42);
      const x = normal(0, 5, [10000]);
      const s = std(numRawData(x.data));
      expect(s).toBeGreaterThan(4.8);
      expect(s).toBeLessThan(5.2);
    });

    it("should throw when std < 0", () => {
      expect(() => normal(0, -1, [1])).toThrow(InvalidParameterError);
    });

    it("should allow std == 0", () => {
      const x = normal(5, 0, [10]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => v === 5)).toBe(true);
    });

    it("should throw on non-finite mean", () => {
      expect(() => normal(NaN, 1, [1])).toThrow(InvalidParameterError);
      expect(() => normal(Infinity, 1, [1])).toThrow(InvalidParameterError);
    });

    it("should throw on non-finite std", () => {
      expect(() => normal(0, NaN, [1])).toThrow(InvalidParameterError);
      expect(() => normal(0, Infinity, [1])).toThrow(InvalidParameterError);
    });

    it("should be deterministic when seeded", () => {
      setSeed(321);
      const a = normal(5, 2, [10]);
      setSeed(321);
      const b = normal(5, 2, [10]);
      expect(numRawData(a.data)).toEqual(numRawData(b.data));
    });
  });

  describe("binomial() - Binomial Distribution", () => {
    it("should generate values in valid range", () => {
      setSeed(42);
      const x = binomial(10, 0.5, [100]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => Number.isInteger(v) && v >= 0 && v <= 10)).toBe(true);
    });

    it("should generate scalar", () => {
      const x = binomial(10, 0.5, []);
      expect(x.shape).toEqual([]);
    });

    it("should use default shape", () => {
      const x = binomial(10, 0.5);
      expect(x.shape).toEqual([]);
    });

    it("should generate 1D array", () => {
      const x = binomial(10, 0.5, [10]);
      expect(x.shape).toEqual([10]);
    });

    it("should have mean close to n*p", () => {
      setSeed(42);
      const n = 100;
      const p = 0.3;
      const x = binomial(n, p, [10000]);
      const m = mean(numRawData(x.data));
      expect(m).toBeGreaterThan(n * p - 1);
      expect(m).toBeLessThan(n * p + 1);
    });

    it("should handle p=0", () => {
      const x = binomial(10, 0, [100]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => v === 0)).toBe(true);
    });

    it("should handle p=1", () => {
      const x = binomial(10, 1, [100]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => v === 10)).toBe(true);
    });

    it("should handle n=0", () => {
      const x = binomial(0, 0.5, [100]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => v === 0)).toBe(true);
    });

    it("should throw when n < 0", () => {
      expect(() => binomial(-1, 0.5, [1])).toThrow(InvalidParameterError);
    });

    it("should throw when n is not integer", () => {
      expect(() => binomial(10.5, 0.5, [1])).toThrow(InvalidParameterError);
    });

    it("should throw when p < 0", () => {
      expect(() => binomial(10, -0.1, [1])).toThrow(InvalidParameterError);
    });

    it("should throw when p > 1", () => {
      expect(() => binomial(10, 1.1, [1])).toThrow(InvalidParameterError);
    });

    it("should throw on non-finite n", () => {
      expect(() => binomial(NaN, 0.5, [1])).toThrow(InvalidParameterError);
      expect(() => binomial(Infinity, 0.5, [1])).toThrow(InvalidParameterError);
    });

    it("should throw on non-finite p", () => {
      expect(() => binomial(10, NaN, [1])).toThrow(InvalidParameterError);
      expect(() => binomial(10, Infinity, [1])).toThrow(InvalidParameterError);
    });

    it("should throw on non-integer dtype", () => {
      expect(() => binomial(10, 0.5, [1], { dtype: "float32" })).toThrow(DTypeError);
    });

    it("should be deterministic when seeded", () => {
      setSeed(654);
      const a = binomial(20, 0.7, [10]);
      setSeed(654);
      const b = binomial(20, 0.7, [10]);
      expect(numRawData(a.data)).toEqual(numRawData(b.data));
    });
  });

  describe("poisson() - Poisson Distribution", () => {
    it("should generate non-negative integers", () => {
      setSeed(42);
      const x = poisson(5, [100]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => Number.isInteger(v) && v >= 0)).toBe(true);
    });

    it("should generate scalar", () => {
      const x = poisson(5, []);
      expect(x.shape).toEqual([]);
    });

    it("should use default shape", () => {
      const x = poisson(5);
      expect(x.shape).toEqual([]);
    });

    it("should generate 1D array", () => {
      const x = poisson(5, [10]);
      expect(x.shape).toEqual([10]);
    });

    it("should have mean close to lambda", () => {
      setSeed(42);
      const lambda = 7;
      const x = poisson(lambda, [10000]);
      const m = mean(numRawData(x.data));
      expect(m).toBeGreaterThan(lambda - 0.3);
      expect(m).toBeLessThan(lambda + 0.3);
    });

    it("should handle lambda=0", () => {
      const x = poisson(0, [100]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => v === 0)).toBe(true);
    });

    it("should handle small lambda", () => {
      setSeed(42);
      const x = poisson(0.1, [1000]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => v >= 0)).toBe(true);
    });

    it("should handle large lambda", () => {
      setSeed(42);
      const x = poisson(50, [1000]);
      const m = mean(numRawData(x.data));
      expect(m).toBeGreaterThan(48);
      expect(m).toBeLessThan(52);
    });

    it("should throw when lambda < 0", () => {
      expect(() => poisson(-1, [1])).toThrow(InvalidParameterError);
    });

    it("should throw on non-finite lambda", () => {
      expect(() => poisson(NaN, [1])).toThrow(InvalidParameterError);
      expect(() => poisson(Infinity, [1])).toThrow(InvalidParameterError);
    });

    it("should throw on non-integer dtype", () => {
      expect(() => poisson(5, [1], { dtype: "float32" })).toThrow(DTypeError);
    });

    it("should be deterministic when seeded", () => {
      setSeed(987);
      const a = poisson(10, [10]);
      setSeed(987);
      const b = poisson(10, [10]);
      expect(numRawData(a.data)).toEqual(numRawData(b.data));
    });
  });

  describe("exponential() - Exponential Distribution", () => {
    it("should generate positive values", () => {
      setSeed(42);
      const x = exponential(1, [100]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => v > 0)).toBe(true);
    });

    it("should generate scalar", () => {
      const x = exponential(1, []);
      expect(x.shape).toEqual([]);
    });

    it("should use default parameters", () => {
      const x = exponential();
      expect(x.shape).toEqual([]);
    });

    it("should generate 1D array", () => {
      const x = exponential(1, [10]);
      expect(x.shape).toEqual([10]);
    });

    it("should have mean close to scale", () => {
      setSeed(42);
      const scale = 3;
      const x = exponential(scale, [10000]);
      const m = mean(numRawData(x.data));
      expect(m).toBeGreaterThan(scale - 0.2);
      expect(m).toBeLessThan(scale + 0.2);
    });

    it("should handle small scale", () => {
      setSeed(42);
      const x = exponential(0.1, [1000]);
      const m = mean(numRawData(x.data));
      expect(m).toBeGreaterThan(0.08);
      expect(m).toBeLessThan(0.12);
    });

    it("should handle large scale", () => {
      setSeed(42);
      const x = exponential(100, [1000]);
      const m = mean(numRawData(x.data));
      expect(m).toBeGreaterThan(90);
      expect(m).toBeLessThan(110);
    });

    it("should throw when scale <= 0", () => {
      expect(() => exponential(0, [1])).toThrow(InvalidParameterError);
      expect(() => exponential(-1, [1])).toThrow(InvalidParameterError);
    });

    it("should throw on non-finite scale", () => {
      expect(() => exponential(NaN, [1])).toThrow(InvalidParameterError);
      expect(() => exponential(Infinity, [1])).toThrow(InvalidParameterError);
    });

    it("should be deterministic when seeded", () => {
      setSeed(111);
      const a = exponential(2, [10]);
      setSeed(111);
      const b = exponential(2, [10]);
      expect(numRawData(a.data)).toEqual(numRawData(b.data));
    });
  });

  describe("gamma() - Gamma Distribution", () => {
    it("should generate positive values", () => {
      setSeed(42);
      const x = gamma(2, 2, [100]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => v > 0)).toBe(true);
    });

    it("should generate scalar", () => {
      const x = gamma(2, 2, []);
      expect(x.shape).toEqual([]);
    });

    it("should use default parameters", () => {
      const x = gamma(2);
      expect(x.shape).toEqual([]);
    });

    it("should generate 1D array", () => {
      const x = gamma(2, 2, [10]);
      expect(x.shape).toEqual([10]);
    });

    it("should have mean close to shape*scale", () => {
      setSeed(42);
      const shape = 3;
      const scale = 2;
      const x = gamma(shape, scale, [10000]);
      const m = mean(numRawData(x.data));
      expect(m).toBeGreaterThan(shape * scale - 0.5);
      expect(m).toBeLessThan(shape * scale + 0.5);
    });

    it("should handle shape < 1", () => {
      setSeed(42);
      const x = gamma(0.5, 1, [1000]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => v > 0)).toBe(true);
    });

    it("should handle shape = 1 (exponential)", () => {
      setSeed(42);
      const x = gamma(1, 2, [10000]);
      const m = mean(numRawData(x.data));
      expect(m).toBeGreaterThan(1.8);
      expect(m).toBeLessThan(2.2);
    });

    it("should handle large shape", () => {
      setSeed(42);
      const x = gamma(100, 1, [1000]);
      const m = mean(numRawData(x.data));
      expect(m).toBeGreaterThan(95);
      expect(m).toBeLessThan(105);
    });

    it("should throw when shape <= 0", () => {
      expect(() => gamma(0, 1, [1])).toThrow(InvalidParameterError);
      expect(() => gamma(-1, 1, [1])).toThrow(InvalidParameterError);
    });

    it("should throw when scale <= 0", () => {
      expect(() => gamma(2, 0, [1])).toThrow(InvalidParameterError);
      expect(() => gamma(2, -1, [1])).toThrow(InvalidParameterError);
    });

    it("should throw on non-finite shape", () => {
      expect(() => gamma(NaN, 1, [1])).toThrow(InvalidParameterError);
      expect(() => gamma(Infinity, 1, [1])).toThrow(InvalidParameterError);
    });

    it("should throw on non-finite scale", () => {
      expect(() => gamma(2, NaN, [1])).toThrow(InvalidParameterError);
      expect(() => gamma(2, Infinity, [1])).toThrow(InvalidParameterError);
    });

    it("should be deterministic when seeded", () => {
      setSeed(222);
      const a = gamma(3, 2, [10]);
      setSeed(222);
      const b = gamma(3, 2, [10]);
      expect(numRawData(a.data)).toEqual(numRawData(b.data));
    });
  });

  describe("beta() - Beta Distribution", () => {
    it("should generate values in (0, 1)", () => {
      setSeed(42);
      const x = beta(2, 5, [100]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => v > 0 && v < 1)).toBe(true);
    });

    it("should generate scalar", () => {
      const x = beta(2, 5, []);
      expect(x.shape).toEqual([]);
    });

    it("should use default shape", () => {
      const x = beta(2, 5);
      expect(x.shape).toEqual([]);
    });

    it("should generate 1D array", () => {
      const x = beta(2, 5, [10]);
      expect(x.shape).toEqual([10]);
    });

    it("should have mean close to alpha/(alpha+beta)", () => {
      setSeed(42);
      const alpha = 2;
      const beta_param = 5;
      const x = beta(alpha, beta_param, [10000]);
      const m = mean(numRawData(x.data));
      const expected = alpha / (alpha + beta_param);
      expect(m).toBeGreaterThan(expected - 0.02);
      expect(m).toBeLessThan(expected + 0.02);
    });

    it("should handle symmetric beta", () => {
      setSeed(42);
      const x = beta(2, 2, [10000]);
      const m = mean(numRawData(x.data));
      expect(m).toBeGreaterThan(0.48);
      expect(m).toBeLessThan(0.52);
    });

    it("should handle alpha < 1", () => {
      setSeed(42);
      const x = beta(0.5, 2, [1000]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => v > 0 && v < 1)).toBe(true);
    });

    it("should handle beta < 1", () => {
      setSeed(42);
      const x = beta(2, 0.5, [1000]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => v >= 0 && v <= 1)).toBe(true);
    });

    it("should throw when alpha <= 0", () => {
      expect(() => beta(0, 1, [1])).toThrow(InvalidParameterError);
      expect(() => beta(-1, 1, [1])).toThrow(InvalidParameterError);
    });

    it("should throw when beta <= 0", () => {
      expect(() => beta(1, 0, [1])).toThrow(InvalidParameterError);
      expect(() => beta(1, -1, [1])).toThrow(InvalidParameterError);
    });

    it("should throw on non-finite alpha", () => {
      expect(() => beta(NaN, 1, [1])).toThrow(InvalidParameterError);
      expect(() => beta(Infinity, 1, [1])).toThrow(InvalidParameterError);
    });

    it("should throw on non-finite beta", () => {
      expect(() => beta(1, NaN, [1])).toThrow(InvalidParameterError);
      expect(() => beta(1, Infinity, [1])).toThrow(InvalidParameterError);
    });

    it("should remain finite for very small shape parameters", () => {
      setSeed(987);
      const x = beta(0.1, 0.2, [1000]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => Number.isFinite(v) && v >= 0 && v <= 1)).toBe(true);
    });

    it("should be deterministic when seeded", () => {
      setSeed(333);
      const a = beta(3, 7, [10]);
      setSeed(333);
      const b = beta(3, 7, [10]);
      expect(numRawData(a.data)).toEqual(numRawData(b.data));
    });
  });

  describe("choice() - Random Sampling", () => {
    it("should sample from integer range", () => {
      setSeed(42);
      const x = choice(10, 5);
      expect(x.shape).toEqual([5]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => v >= 0 && v < 10)).toBe(true);
    });

    it("should return int32 dtype for numeric populations", () => {
      const x = choice(10, 3);
      expect(x.dtype).toBe("int32");
    });

    it("should sample from tensor", () => {
      setSeed(42);
      const t = tensor([1, 2, 3, 4, 5]);
      const x = choice(t, 3);
      expect(x.shape).toEqual([3]);
    });

    it("should sample with replacement by default", () => {
      setSeed(42);
      const x = choice(5, 10, true);
      expect(x.shape).toEqual([10]);
    });

    it("should sample without replacement", () => {
      setSeed(42);
      const x = choice(10, 5, false);
      const arr = numRawData(x.data);
      const unique = new Set(arr);
      expect(unique.size).toBe(5);
    });

    it("should throw when sampling without replacement and size > population", () => {
      expect(() => choice(5, 10, false)).toThrow(InvalidParameterError);
    });

    it("should handle size as shape array", () => {
      setSeed(42);
      const x = choice(10, [2, 3]);
      expect(x.shape).toEqual([2, 3]);
    });

    it("should default to size=1", () => {
      setSeed(42);
      const x = choice(10);
      expect(x.shape).toEqual([1]);
    });

    it("should preserve dtype when sampling from tensor", () => {
      const t = tensor([1, 2, 3, 4], { dtype: "int32" });
      const x = choice(t, 2);
      expect(x.dtype).toBe("int32");
    });

    it("should throw on negative population size", () => {
      expect(() => choice(-1, 1)).toThrow(InvalidParameterError);
    });

    it("should throw on non-integer population size", () => {
      expect(() => choice(10.5, 1)).toThrow(InvalidParameterError);
    });

    it("should throw on non-finite population size", () => {
      expect(() => choice(NaN, 1)).toThrow(InvalidParameterError);
      expect(() => choice(Infinity, 1)).toThrow(InvalidParameterError);
    });

    it("should throw on population sizes that exceed int32 range", () => {
      expect(() => choice(2147483649, 1)).toThrow(InvalidParameterError);
    });

    it("should throw on string tensors", () => {
      const t = tensor(["a", "b", "c"]);
      expect(() => choice(t, 2)).toThrow();
    });

    it("should be deterministic when seeded", () => {
      setSeed(444);
      const a = choice(20, 5);
      setSeed(444);
      const b = choice(20, 5);
      expect(numRawData(a.data)).toEqual(numRawData(b.data));
    });

    it("should handle zero population", () => {
      const x = choice(0, 0);
      expect(x.size).toBe(0);
    });

    it("should handle sampling all elements without replacement", () => {
      setSeed(42);
      const x = choice(5, 5, false);
      const arr = numRawData(x.data).sort((a, b) => a - b);
      expect(arr).toEqual([0, 1, 2, 3, 4]);
    });

    it("should support weighted sampling with replacement", () => {
      setSeed(2024);
      const values = tensor([10, 20, 30], { dtype: "int32" });
      const probs = tensor([0, 0, 1]);
      const samples = choice(values, 6, true, probs);
      expect(numRawData(samples.data)).toEqual([30, 30, 30, 30, 30, 30]);
    });

    it("should support weighted sampling without replacement", () => {
      setSeed(777);
      const values = tensor([0, 1, 2, 3], { dtype: "int32" });
      const probs = tensor([0, 0.7, 0.3, 0]);
      const samples = choice(values, 2, false, probs);
      const sorted = numRawData(samples.data).sort((a, b) => a - b);
      expect(sorted).toEqual([1, 2]);
    });

    it("should validate probability vector for weighted sampling", () => {
      const probsBadLength = tensor([1, 1]);
      const probsNegative = tensor([0.5, -0.5, 1.0]);
      const probsNaN = tensor([0.2, NaN, 0.8]);
      const probs2d = tensor([
        [0.2, 0.8],
        [0.3, 0.7],
      ]);

      expect(() => choice(3, 1, true, probsBadLength)).toThrow(InvalidParameterError);
      expect(() => choice(3, 1, true, probsNegative)).toThrow(InvalidParameterError);
      expect(() => choice(3, 1, true, probsNaN)).toThrow(InvalidParameterError);
      expect(() => choice(3, 1, true, probs2d)).toThrow(InvalidParameterError);
      expect(() => choice(3, 3, false, tensor([1, 0, 0]))).toThrow(InvalidParameterError);
    });
  });

  describe("shuffle() - In-place Shuffling", () => {
    it("should shuffle array in-place", () => {
      setSeed(42);
      const x = tensor([1, 2, 3, 4, 5], { dtype: "int32" });
      const original = numRawData(x.data);
      shuffle(x);
      const shuffled = numRawData(x.data);
      expect(shuffled.sort()).toEqual(original.sort());
    });

    it("should preserve all elements", () => {
      setSeed(42);
      const x = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], { dtype: "int32" });
      shuffle(x);
      const arr = numRawData(x.data).sort((a, b) => a - b);
      expect(arr).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    });

    it("should be deterministic when seeded", () => {
      const a = tensor([1, 2, 3, 4, 5], { dtype: "int32" });
      const b = tensor([1, 2, 3, 4, 5], { dtype: "int32" });
      setSeed(555);
      shuffle(a);
      setSeed(555);
      shuffle(b);
      expect(numRawData(a.data)).toEqual(numRawData(b.data));
    });

    it("should handle single element", () => {
      const x = tensor([42], { dtype: "int32" });
      shuffle(x);
      expect(x.data[0]).toBe(42);
    });

    it("should handle two elements", () => {
      setSeed(42);
      const x = tensor([1, 2], { dtype: "int32" });
      shuffle(x);
      const arr = numRawData(x.data).sort();
      expect(arr).toEqual([1, 2]);
    });

    it("should throw on string tensors", () => {
      const x = tensor(["a", "b", "c"]);
      expect(() => shuffle(x)).toThrow();
    });

    it("should handle float tensors", () => {
      setSeed(42);
      const x = tensor([1.1, 2.2, 3.3, 4.4], { dtype: "float32" });
      shuffle(x);
      const arr = numRawData(x.data).sort();
      expect(arr).toHaveLength(4);
    });

    it("should modify original tensor", () => {
      setSeed(42);
      const x = tensor([1, 2, 3, 4, 5], { dtype: "int32" });
      const before = numRawData(x.data);
      shuffle(x);
      const after = numRawData(x.data);
      expect(before).not.toEqual(after);
    });
  });

  describe("permutation() - Random Permutation", () => {
    it("should generate permutation of integers", () => {
      setSeed(42);
      const x = permutation(10);
      expect(x.shape).toEqual([10]);
      const arr = numRawData(x.data).sort((a, b) => a - b);
      expect(arr).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    });

    it("should generate permutation of tensor", () => {
      setSeed(42);
      const t = tensor([10, 20, 30, 40, 50], { dtype: "int32" });
      const x = permutation(t);
      expect(x.shape).toEqual([5]);
      const arr = numRawData(x.data).sort((a, b) => a - b);
      expect(arr).toEqual([10, 20, 30, 40, 50]);
    });

    it("should not modify original tensor", () => {
      setSeed(42);
      const t = tensor([1, 2, 3, 4, 5], { dtype: "int32" });
      const original = numRawData(t.data);
      permutation(t);
      expect(numRawData(t.data)).toEqual(original);
    });

    it("should be deterministic when seeded", () => {
      setSeed(666);
      const a = permutation(10);
      setSeed(666);
      const b = permutation(10);
      expect(numRawData(a.data)).toEqual(numRawData(b.data));
    });

    it("should handle single element", () => {
      const x = permutation(1);
      expect(numRawData(x.data)).toEqual([0]);
    });

    it("should handle zero elements", () => {
      const x = permutation(0);
      expect(x.size).toBe(0);
    });

    it("should throw on invalid numeric input", () => {
      expect(() => permutation(-1)).toThrow(InvalidParameterError);
      expect(() => permutation(1.5)).toThrow(InvalidParameterError);
      expect(() => permutation(NaN)).toThrow(InvalidParameterError);
    });

    it("should throw on string tensors", () => {
      const t = tensor(["a", "b", "c"]);
      expect(() => permutation(t)).toThrow();
    });

    it("should throw on non-contiguous tensor inputs", () => {
      const t = tensor(
        [
          [1, 2],
          [3, 4],
        ],
        { dtype: "int32" }
      );
      const tT = transpose(t);
      expect(() => permutation(tT)).toThrow(InvalidParameterError);
    });

    it("should handle large permutations", () => {
      setSeed(42);
      const x = permutation(1000);
      const arr = numRawData(x.data).sort((a, b) => a - b);
      expect(arr).toEqual(Array.from({ length: 1000 }, (_, i) => i));
    });

    it("should produce different permutations", () => {
      setSeed(42);
      const a = permutation(10);
      const b = permutation(10);
      expect(numRawData(a.data)).not.toEqual(numRawData(b.data));
    });
  });

  describe("Edge Cases and Stress Tests", () => {
    it("should handle very large arrays efficiently", () => {
      setSeed(42);
      const start = Date.now();
      const x = rand([10000, 100]);
      const elapsed = Date.now() - start;
      expect(x.size).toBe(1000000);
      expect(elapsed).toBeLessThan(5000);
    });

    it("should handle repeated seeding", () => {
      for (let i = 0; i < 100; i++) {
        setSeed(i);
        expect(getSeed()).toBe(i);
      }
    });

    it("should handle alternating seeded and unseeded", () => {
      setSeed(42);
      const a = rand([5]);
      setSeed(42);
      const b = rand([5]);
      expect(numRawData(a.data)).toEqual(numRawData(b.data));
    });

    it("should handle extreme parameter values for normal", () => {
      setSeed(42);
      const x = normal(1e10, 1e-10, [10]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => Math.abs(v - 1e10) < 1)).toBe(true);
    });

    it("should handle extreme parameter values for uniform", () => {
      setSeed(42);
      const x = uniform(-1e10, 1e10, [10]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => v >= -1e10 && v <= 1e10)).toBe(true);
    });

    it("should handle many distributions in sequence", () => {
      setSeed(42);
      rand([10]);
      randn([10]);
      randint(0, 10, [10]);
      uniform(0, 1, [10]);
      normal(0, 1, [10]);
      binomial(10, 0.5, [10]);
      poisson(5, [10]);
      exponential(1, [10]);
      gamma(2, 2, [10]);
      beta(2, 5, [10]);
      expect(true).toBe(true);
    });

    it("should handle scalar operations", () => {
      setSeed(42);
      const a = rand([]);
      const b = randn([]);
      const c = uniform(0, 1, []);
      const d = normal(0, 1, []);
      expect(a.size).toBe(1);
      expect(b.size).toBe(1);
      expect(c.size).toBe(1);
      expect(d.size).toBe(1);
    });

    it("should handle high-dimensional tensors", () => {
      setSeed(42);
      const x = rand([2, 3, 4, 5]);
      expect(x.shape).toEqual([2, 3, 4, 5]);
      expect(x.size).toBe(120);
    });

    it("should handle mixed dtype operations", () => {
      setSeed(42);
      rand([5], { dtype: "float32" });
      rand([5], { dtype: "float64" });
      randint(0, 10, [5], { dtype: "int32" });
      expect(true).toBe(true);
    });
  });

  describe("Additional Edge Cases (P1-5)", () => {
    it("should handle extreme lambda for Poisson (lambda > 100)", () => {
      setSeed(42);
      const x = poisson(150, [1000]);
      const m = mean(numRawData(x.data));
      expect(m).toBeGreaterThan(140);
      expect(m).toBeLessThan(160);
    });

    it("should handle very large lambda for Poisson (lambda = 500)", () => {
      setSeed(42);
      const x = poisson(500, [1000]);
      const m = mean(numRawData(x.data));
      expect(m).toBeGreaterThan(480);
      expect(m).toBeLessThan(520);
    });

    it("should handle very large n for binomial (n = 1000)", () => {
      setSeed(42);
      const x = binomial(1000, 0.5, [100]);
      const m = mean(numRawData(x.data));
      expect(m).toBeGreaterThan(480);
      expect(m).toBeLessThan(520);
    });

    it("should handle beta distribution with small alpha", () => {
      setSeed(42);
      const x = beta(0.5, 2, [1000]);
      const arr = numRawData(x.data);
      const m = mean(arr);
      expect(m).toBeGreaterThan(0.1);
      expect(m).toBeLessThan(0.3);
    });

    it("should handle beta distribution with small beta", () => {
      setSeed(42);
      const x = beta(2, 0.5, [1000]);
      const arr = numRawData(x.data);
      const m = mean(arr);
      expect(m).toBeGreaterThan(0.7);
      expect(m).toBeLessThan(0.9);
    });

    it("should handle gamma distribution with small shape", () => {
      setSeed(42);
      const x = gamma(0.5, 1, [1000]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => v > 0)).toBe(true);
      const m = mean(arr);
      expect(m).toBeGreaterThan(0.4);
      expect(m).toBeLessThan(0.6);
    });

    it("should handle choice with 2D shape parameter", () => {
      setSeed(42);
      const x = choice(10, [2, 3]);
      expect(x.shape).toEqual([2, 3]);
      expect(x.size).toBe(6);
    });

    it("should handle choice with 3D shape parameter", () => {
      setSeed(42);
      const x = choice(10, [2, 2, 2]);
      expect(x.shape).toEqual([2, 2, 2]);
      expect(x.size).toBe(8);
    });

    it("should handle choice with size=0", () => {
      const x = choice(10, 0);
      expect(x.size).toBe(0);
    });

    it("should handle shuffle on empty tensor", () => {
      const x = tensor([], { dtype: "int32" });
      shuffle(x);
      expect(x.size).toBe(0);
    });

    it("should handle permutation with zero elements", () => {
      const x = permutation(0);
      expect(x.size).toBe(0);
      expect(x.shape).toEqual([0]);
    });

    it("should handle large values in uniform", () => {
      setSeed(42);
      const x = uniform(1e6, 1e6 + 1000, [100]);
      const arr = numRawData(x.data);
      const m = mean(arr);
      expect(m).toBeGreaterThan(1e6);
      expect(m).toBeLessThan(1e6 + 1000);
    });

    it("should handle normal distribution with very large std", () => {
      setSeed(42);
      const x = normal(0, 1e6, [100]);
      const s = std(numRawData(x.data));
      expect(s).toBeGreaterThan(9e5);
      expect(s).toBeLessThan(1.1e6);
    });

    it("should handle exponential with very small scale", () => {
      setSeed(42);
      const x = exponential(0.001, [1000]);
      const m = mean(numRawData(x.data));
      expect(m).toBeGreaterThan(0.0008);
      expect(m).toBeLessThan(0.0012);
    });

    it("should handle exponential with very large scale", () => {
      setSeed(42);
      const x = exponential(1000, [1000]);
      const m = mean(numRawData(x.data));
      expect(m).toBeGreaterThan(900);
      expect(m).toBeLessThan(1100);
    });

    it("should handle gamma with large shape parameter", () => {
      setSeed(42);
      const x = gamma(200, 1, [1000]);
      const m = mean(numRawData(x.data));
      expect(m).toBeGreaterThan(190);
      expect(m).toBeLessThan(210);
    });

    it("should handle randint with large range", () => {
      setSeed(42);
      const x = randint(-1000000, 1000000, [1000]);
      const arr = numRawData(x.data);
      expect(arr.every((v) => v >= -1000000 && v < 1000000)).toBe(true);
    });

    it("should handle choice without replacement at boundary", () => {
      setSeed(42);
      const x = choice(10, 10, false);
      const arr = numRawData(x.data).sort((a, b) => a - b);
      expect(arr).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    });

    it("should handle permutation with large n", () => {
      setSeed(42);
      const x = permutation(10000);
      expect(x.size).toBe(10000);
      const arr = numRawData(x.data);
      const unique = new Set(arr);
      expect(unique.size).toBe(10000);
    });
  });
});
