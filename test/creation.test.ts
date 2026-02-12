import { describe, expect, it } from "vitest";
import {
  arange,
  empty,
  eye,
  full,
  linspace,
  logspace,
  ones,
  randn,
  tensor,
  zeros,
} from "../src/ndarray";
import { numData } from "./_helpers";

describe("deepbox/ndarray - Tensor Creation", () => {
  describe("tensor", () => {
    it("should create 1D tensor from array", () => {
      const t = tensor([1, 2, 3]);
      expect(t.shape).toEqual([3]);
    });

    it("should create 2D tensor from nested array", () => {
      const t = tensor([
        [1, 2],
        [3, 4],
      ]);
      expect(t.shape).toEqual([2, 2]);
    });

    it("should create 3D tensor", () => {
      const t = tensor([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      expect(t.shape).toEqual([2, 2, 2]);
    });

    it("should handle empty array", () => {
      expect(() => tensor([])).not.toThrow();
    });

    it("should handle single element", () => {
      const t = tensor([5]);
      expect(t.shape).toEqual([1]);
    });

    it("should handle negative values", () => {
      const t = tensor([-1, -2, -3]);
      expect(t.shape).toEqual([3]);
    });

    it("should handle zero values", () => {
      const t = tensor([0, 0, 0]);
      expect(t.shape).toEqual([3]);
    });

    it("should accept dtype option", () => {
      expect(() => tensor([1, 2, 3], { dtype: "float32" })).not.toThrow();
    });

    it("should accept device option", () => {
      expect(() => tensor([1, 2, 3], { device: "cpu" })).not.toThrow();
    });
  });

  describe("zeros", () => {
    it("should create 1D zeros", () => {
      const t = zeros([5]);
      expect(t.shape).toEqual([5]);
    });

    it("should create 2D zeros", () => {
      const t = zeros([3, 4]);
      expect(t.shape).toEqual([3, 4]);
    });

    it("should create 3D zeros", () => {
      const t = zeros([2, 3, 4]);
      expect(t.shape).toEqual([2, 3, 4]);
    });

    it("should handle single dimension", () => {
      const t = zeros([1]);
      expect(t.shape).toEqual([1]);
    });

    it("should handle empty shape", () => {
      const t = zeros([]);
      expect(t.shape).toEqual([]);
    });
  });

  describe("ones", () => {
    it("should create 1D ones", () => {
      const t = ones([5]);
      expect(t.shape).toEqual([5]);
    });

    it("should create 2D ones", () => {
      const t = ones([3, 4]);
      expect(t.shape).toEqual([3, 4]);
    });

    it("should handle large dimensions", () => {
      const t = ones([100, 100]);
      expect(t.shape).toEqual([100, 100]);
    });
  });

  describe("full", () => {
    it("should create tensor filled with value", () => {
      expect(() => full([3, 3], 5)).not.toThrow();
    });

    it("should handle negative fill value", () => {
      expect(() => full([2, 2], -1)).not.toThrow();
    });

    it("should handle zero fill value", () => {
      expect(() => full([2, 2], 0)).not.toThrow();
    });

    it("should handle large fill value", () => {
      expect(() => full([2, 2], 1e10)).not.toThrow();
    });
  });

  describe("empty", () => {
    it("should create uninitialized tensor", () => {
      expect(() => empty([3, 3])).not.toThrow();
    });

    it("should have correct shape", () => {
      expect(() => empty([2, 3, 4])).not.toThrow();
    });
  });

  describe("arange", () => {
    it("should create range with stop only", () => {
      const t = arange(5);
      expect(t.shape).toEqual([5]);
    });

    it("should create range with start and stop", () => {
      const t = arange(2, 7);
      expect(t.shape).toEqual([5]);
    });

    it("should create range with step", () => {
      expect(() => arange(0, 10, 2)).not.toThrow();
    });

    it("should handle negative step", () => {
      expect(() => arange(10, 0, -1)).not.toThrow();
    });

    it("should handle negative start/stop", () => {
      expect(() => arange(-5, 5)).not.toThrow();
    });

    it("should handle zero in range", () => {
      expect(() => arange(-2, 3)).not.toThrow();
    });
  });

  describe("linspace", () => {
    it("should create evenly spaced values", () => {
      const t = linspace(0, 10, 5);
      expect(t.shape).toEqual([5]);
      expect(numData(t)).toEqual([0, 2.5, 5, 7.5, 10]);
    });

    it("should handle negative range", () => {
      const t = linspace(-5, 5, 11);
      expect(t.shape).toEqual([11]);
      expect(Number(t.data[0])).toBeCloseTo(-5, 10);
      expect(Number(t.data[10])).toBeCloseTo(5, 10);
    });

    it("should handle single point", () => {
      const t = linspace(0, 10, 1);
      expect(t.shape).toEqual([1]);
      expect(Number(t.data[0])).toBeCloseTo(0, 10);
    });

    it("should handle large num", () => {
      const t = linspace(0, 1, 1000);
      expect(t.shape).toEqual([1000]);
      expect(Number(t.data[0])).toBeCloseTo(0, 10);
      expect(Number(t.data[999])).toBeCloseTo(1, 10);
    });

    it("should handle reversed range", () => {
      const t = linspace(10, 0, 5);
      expect(t.shape).toEqual([5]);
      expect(numData(t)).toEqual([10, 7.5, 5, 2.5, 0]);
    });

    it("should return empty tensor for num=0", () => {
      const t = linspace(0, 1, 0);
      expect(t.shape).toEqual([0]);
      expect(t.size).toBe(0);
    });
  });

  describe("logspace", () => {
    it("should create logarithmically spaced values", () => {
      const t = logspace(0, 3, 4);
      expect(t.shape).toEqual([4]);
      expect(numData(t)).toEqual([1, 10, 100, 1000]);
    });

    it("should accept custom base", () => {
      const t = logspace(0, 3, 4, 2);
      expect(t.shape).toEqual([4]);
      expect(numData(t)).toEqual([1, 2, 4, 8]);
    });

    it("should handle negative exponents", () => {
      const t = logspace(-3, 0, 4);
      expect(t.shape).toEqual([4]);
      expect(Number(t.data[0])).toBeCloseTo(0.001, 10);
      expect(Number(t.data[3])).toBeCloseTo(1, 10);
    });
  });

  describe("eye", () => {
    it("should create identity matrix", () => {
      const I = eye(3);
      expect(I.shape).toEqual([3, 3]);
      expect(numData(I)).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    });

    it("should create rectangular identity", () => {
      const A = eye(3, 4);
      expect(A.shape).toEqual([3, 4]);
      expect(numData(A)).toEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]);
    });

    it("should handle offset diagonal", () => {
      const A = eye(3, 3, 1);
      expect(A.shape).toEqual([3, 3]);
      expect(numData(A)).toEqual([0, 1, 0, 0, 0, 1, 0, 0, 0]);
    });

    it("should handle negative offset", () => {
      const A = eye(3, 3, -1);
      expect(A.shape).toEqual([3, 3]);
      expect(numData(A)).toEqual([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    });

    it("should handle single dimension", () => {
      const I = eye(1);
      expect(I.shape).toEqual([1, 1]);
      expect(numData(I)).toEqual([1]);
    });
  });

  describe("randn", () => {
    it("should create normal random tensor", () => {
      const t = randn([3, 3]);
      expect(t.shape).toEqual([3, 3]);
      expect(t.size).toBe(9);
      for (let i = 0; i < t.size; i++) {
        expect(Number.isFinite(Number(t.data[t.offset + i]))).toBe(true);
      }
    });

    it("should handle large dimensions", () => {
      const t = randn([100, 100]);
      expect(t.shape).toEqual([100, 100]);
      expect(t.size).toBe(10000);
    });
  });
});
