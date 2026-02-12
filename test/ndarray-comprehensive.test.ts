import { describe, expect, it } from "vitest";
import {
  abs,
  add,
  addScalar,
  all,
  any,
  clip,
  div,
  equal,
  exp,
  greater,
  isfinite,
  isinf,
  isnan,
  less,
  log,
  logicalAnd,
  logicalNot,
  logicalOr,
  max,
  maximum,
  mean,
  min,
  minimum,
  mul,
  mulScalar,
  neg,
  pow,
  reciprocal,
  sign,
  sqrt,
  square,
  std,
  sub,
  sum,
  tensor,
  variance,
} from "../src/ndarray";
import { numData } from "./_helpers";

describe("deepbox/ndarray - Comprehensive Operations Tests", () => {
  describe("Arithmetic Operations - Edge Cases", () => {
    it("should handle add with broadcasting [1] + [3]", () => {
      const a = tensor([5]);
      const b = tensor([1, 2, 3]);
      const c = add(a, b);
      expect(numData(c)).toEqual([6, 7, 8]);
    });

    it("should handle add with 2D broadcasting", () => {
      const a = tensor([[1], [2], [3]]);
      const b = tensor([[10, 20, 30]]);
      const c = add(a, b);
      expect(c.shape).toEqual([3, 3]);
      expect(numData(c)).toEqual([11, 21, 31, 12, 22, 32, 13, 23, 33]);
    });

    it("should handle sub with negative results", () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([5, 5, 5]);
      const c = sub(a, b);
      expect(numData(c)).toEqual([-4, -3, -2]);
    });

    it("should handle mul with zeros", () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([0, 0, 0]);
      const c = mul(a, b);
      expect(numData(c)).toEqual([0, 0, 0]);
    });

    it("should handle div by small numbers", () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([0.1, 0.1, 0.1]);
      const c = div(a, b);
      expect(numData(c)).toEqual([10, 20, 30]);
    });

    it("should handle div by zero producing Infinity", () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([0, 0, 0]);
      const c = div(a, b);
      expect(Number(c.data[0])).toBe(Infinity);
    });

    it("should handle addScalar with negative scalar", () => {
      const a = tensor([1, 2, 3]);
      const b = addScalar(a, -5);
      expect(numData(b)).toEqual([-4, -3, -2]);
    });

    it("should handle mulScalar with zero", () => {
      const a = tensor([1, 2, 3]);
      const b = mulScalar(a, 0);
      expect(numData(b)).toEqual([0, 0, 0]);
    });

    it("should handle pow with fractional exponents", () => {
      const a = tensor([4, 9, 16]);
      const b = tensor([0.5, 0.5, 0.5]);
      const c = pow(a, b);
      expect(numData(c)).toEqual([2, 3, 4]);
    });

    it("should handle pow with negative base and integer exponent", () => {
      const a = tensor([-2, -3, -4]);
      const b = tensor([2, 2, 2]);
      const c = pow(a, b);
      expect(numData(c)).toEqual([4, 9, 16]);
    });

    it("should handle neg with mixed signs", () => {
      const a = tensor([-1, 0, 1, -5, 5]);
      const b = neg(a);
      const result = numData(b);
      expect(result[0]).toBe(1);
      expect(Math.abs(result[1])).toBe(0);
      expect(result[2]).toBe(-1);
      expect(result[3]).toBe(5);
      expect(result[4]).toBe(-5);
    });

    it("should handle abs with negative numbers", () => {
      const a = tensor([-1, -2, -3, 0, 1, 2]);
      const b = abs(a);
      expect(numData(b)).toEqual([1, 2, 3, 0, 1, 2]);
    });

    it("should handle sign with zeros", () => {
      const a = tensor([-5, -0.1, 0, 0.1, 5]);
      const b = sign(a);
      expect(numData(b)).toEqual([-1, -1, 0, 1, 1]);
    });

    it("should handle reciprocal with large numbers", () => {
      const a = tensor([100, 1000, 10000]);
      const b = reciprocal(a);
      expect(Number(b.data[0])).toBeCloseTo(0.01);
      expect(Number(b.data[1])).toBeCloseTo(0.001);
      expect(Number(b.data[2])).toBeCloseTo(0.0001);
    });

    it("should handle maximum with NaN", () => {
      const a = tensor([1, NaN, 3]);
      const b = tensor([2, 2, 2]);
      const c = maximum(a, b);
      expect(Number.isNaN(Number(c.data[1]))).toBe(true);
    });

    it("should handle minimum with negative infinity", () => {
      const a = tensor([1, -Infinity, 3]);
      const b = tensor([2, 2, 2]);
      const c = minimum(a, b);
      expect(Number(c.data[1])).toBe(-Infinity);
    });

    it("should handle clip with reversed bounds", () => {
      const a = tensor([1, 2, 3, 4, 5]);
      const b = clip(a, 2, 4);
      expect(numData(b)).toEqual([2, 2, 3, 4, 4]);
    });

    it("should handle clip with equal bounds", () => {
      const a = tensor([1, 2, 3]);
      const b = clip(a, 2, 2);
      expect(numData(b)).toEqual([2, 2, 2]);
    });
  });

  describe("Comparison Operations - Edge Cases", () => {
    it("should handle equal with floating point precision", () => {
      const a = tensor([0.1 + 0.2]);
      const b = tensor([0.3]);
      const c = equal(a, b);
      // Note: 0.1 + 0.2 === 0.3 is true in JavaScript due to rounding
      expect(Number(c.data[0])).toBe(1);
    });

    it("should handle greater with equal values", () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([1, 2, 3]);
      const c = greater(a, b);
      expect(numData(c)).toEqual([0, 0, 0]);
    });

    it("should handle less with infinity", () => {
      const a = tensor([1, Infinity, -Infinity]);
      const b = tensor([Infinity, Infinity, -Infinity]);
      const c = less(a, b);
      expect(numData(c)).toEqual([1, 0, 0]);
    });

    it("should handle isnan correctly", () => {
      const a = tensor([1, NaN, 3, 0 / 0]);
      const b = isnan(a);
      expect(numData(b)).toEqual([0, 1, 0, 1]);
    });

    it("should handle isinf correctly", () => {
      const a = tensor([1, Infinity, -Infinity, 1 / 0]);
      const b = isinf(a);
      expect(numData(b)).toEqual([0, 1, 1, 1]);
    });

    it("should handle isfinite with mixed values", () => {
      const a = tensor([1, NaN, Infinity, -Infinity, 0]);
      const b = isfinite(a);
      expect(numData(b)).toEqual([1, 0, 0, 0, 1]);
    });
  });

  describe("Logical Operations - Edge Cases", () => {
    it("should handle logicalAnd with zeros", () => {
      const a = tensor([0, 0, 1, 1]);
      const b = tensor([0, 1, 0, 1]);
      const c = logicalAnd(a, b);
      expect(numData(c)).toEqual([0, 0, 0, 1]);
    });

    it("should handle logicalOr with all zeros", () => {
      const a = tensor([0, 0, 0]);
      const b = tensor([0, 0, 0]);
      const c = logicalOr(a, b);
      expect(numData(c)).toEqual([0, 0, 0]);
    });

    it("should handle logicalNot with mixed values", () => {
      const a = tensor([0, 1, -1, 0.5, 0]);
      const b = logicalNot(a);
      expect(numData(b)).toEqual([1, 0, 0, 0, 1]);
    });
  });

  describe("Math Operations - Edge Cases", () => {
    it("should handle exp with large values", () => {
      const a = tensor([10, 20, 30]);
      const b = exp(a);
      expect(Number(b.data[0])).toBeGreaterThan(20000);
    });

    it("should handle exp with negative values", () => {
      const a = tensor([-1, -2, -3]);
      const b = exp(a);
      expect(Number(b.data[0])).toBeCloseTo(0.3678794411714423);
    });

    it("should handle log with values near zero", () => {
      const a = tensor([0.001, 0.01, 0.1]);
      const b = log(a);
      expect(Number(b.data[0])).toBeCloseTo(-6.907755278982137);
    });

    it("should handle log with zero producing -Infinity", () => {
      const a = tensor([0]);
      const b = log(a);
      expect(Number(b.data[0])).toBe(-Infinity);
    });

    it("should handle sqrt with zero", () => {
      const a = tensor([0, 1, 4, 9]);
      const b = sqrt(a);
      expect(numData(b)).toEqual([0, 1, 2, 3]);
    });

    it("should handle sqrt with negative producing NaN", () => {
      const a = tensor([-1]);
      const b = sqrt(a);
      expect(Number.isNaN(Number(b.data[0]))).toBe(true);
    });

    it("should handle square with negative numbers", () => {
      const a = tensor([-1, -2, -3]);
      const b = square(a);
      expect(numData(b)).toEqual([1, 4, 9]);
    });
  });

  describe("Reduction Operations - Edge Cases", () => {
    it("should handle sum with all zeros", () => {
      const a = tensor([0, 0, 0, 0]);
      const b = sum(a);
      expect(Number(b.data[0])).toBe(0);
    });

    it("should handle sum with negative numbers", () => {
      const a = tensor([-1, -2, -3]);
      const b = sum(a);
      expect(Number(b.data[0])).toBe(-6);
    });

    it("should handle mean with single element", () => {
      const a = tensor([42]);
      const b = mean(a);
      expect(Number(b.data[0])).toBe(42);
    });

    it("should handle mean with negative numbers", () => {
      const a = tensor([-1, -2, -3, -4]);
      const b = mean(a);
      expect(Number(b.data[0])).toBe(-2.5);
    });

    it("should handle max with all equal values", () => {
      const a = tensor([5, 5, 5, 5]);
      const b = max(a);
      expect(Number(b.data[0])).toBe(5);
    });

    it("should handle max with negative numbers", () => {
      const a = tensor([-10, -5, -1, -20]);
      const b = max(a);
      expect(Number(b.data[0])).toBe(-1);
    });

    it("should handle min with positive numbers", () => {
      const a = tensor([10, 5, 1, 20]);
      const b = min(a);
      expect(Number(b.data[0])).toBe(1);
    });

    it("should handle variance with constant values", () => {
      const a = tensor([5, 5, 5, 5]);
      const b = variance(a);
      expect(Number(b.data[0])).toBe(0);
    });

    it("should handle std with constant values", () => {
      const a = tensor([5, 5, 5, 5]);
      const b = std(a);
      expect(Number(b.data[0])).toBe(0);
    });

    it("should handle std with known values", () => {
      const a = tensor([1, 2, 3, 4, 5]);
      const b = std(a);
      expect(Number(b.data[0])).toBeCloseTo(Math.SQRT2);
    });

    it("should handle any with all zeros", () => {
      const a = tensor([0, 0, 0]);
      const b = any(a);
      expect(Number(b.data[0])).toBe(0);
    });

    it("should handle any with one non-zero", () => {
      const a = tensor([0, 0, 1, 0]);
      const b = any(a);
      expect(Number(b.data[0])).toBe(1);
    });

    it("should handle all with all ones", () => {
      const a = tensor([1, 1, 1]);
      const b = all(a);
      expect(Number(b.data[0])).toBe(1);
    });

    it("should handle all with one zero", () => {
      const a = tensor([1, 0, 1]);
      const b = all(a);
      expect(Number(b.data[0])).toBe(0);
    });
  });

  describe("Multi-dimensional Operations", () => {
    it("should handle 2D tensor addition", () => {
      const a = tensor([
        [1, 2],
        [3, 4],
      ]);
      const b = tensor([
        [5, 6],
        [7, 8],
      ]);
      const c = add(a, b);
      expect(numData(c)).toEqual([6, 8, 10, 12]);
    });

    it("should handle 3D tensor multiplication", () => {
      const a = tensor([[[1, 2]], [[3, 4]]]);
      const b = tensor([[[2, 2]], [[2, 2]]]);
      const c = mul(a, b);
      expect(numData(c)).toEqual([2, 4, 6, 8]);
    });

    it("should handle 2D sum reduction", () => {
      const a = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const b = sum(a);
      expect(Number(b.data[0])).toBe(21);
    });

    it("should handle 2D mean reduction", () => {
      const a = tensor([
        [2, 4],
        [6, 8],
      ]);
      const b = mean(a);
      expect(Number(b.data[0])).toBe(5);
    });
  });

  describe("Type Preservation", () => {
    it("should preserve int32 dtype in add", () => {
      const a = tensor([1, 2, 3], { dtype: "int32" });
      const b = tensor([4, 5, 6], { dtype: "int32" });
      const c = add(a, b);
      expect(c.dtype).toBe("int32");
    });

    it("should preserve float32 dtype in mul", () => {
      const a = tensor([1, 2, 3], { dtype: "float32" });
      const b = tensor([2, 2, 2], { dtype: "float32" });
      const c = mul(a, b);
      expect(c.dtype).toBe("float32");
    });

    it("should convert to float64 in exp", () => {
      const a = tensor([1, 2, 3], { dtype: "int32" });
      const b = exp(a);
      expect(b.dtype).toBe("float64");
    });

    it("should convert to float64 in sqrt", () => {
      const a = tensor([1, 4, 9], { dtype: "int32" });
      const b = sqrt(a);
      expect(b.dtype).toBe("float64");
    });
  });

  describe("Large Tensor Operations", () => {
    it("should handle large 1D tensor addition", () => {
      const size = 10000;
      const data1 = Array.from({ length: size }, (_, i) => i);
      const data2 = Array.from({ length: size }, (_, i) => i);
      const a = tensor(data1);
      const b = tensor(data2);
      const c = add(a, b);
      expect(c.size).toBe(size);
      expect(Number(c.data[0])).toBe(0);
      expect(Number(c.data[size - 1])).toBe((size - 1) * 2);
    });

    it("should handle large tensor sum", () => {
      const size = 1000;
      const data = Array.from({ length: size }, () => 1);
      const a = tensor(data);
      const b = sum(a);
      expect(Number(b.data[0])).toBe(size);
    });

    it("should handle large tensor mean", () => {
      const size = 1000;
      const data = Array.from({ length: size }, (_, i) => i);
      const a = tensor(data);
      const b = mean(a);
      expect(Number(b.data[0])).toBe(499.5);
    });
  });

  describe("Special Values", () => {
    it("should handle Infinity in addition", () => {
      const a = tensor([Infinity, 1, 2]);
      const b = tensor([1, 2, 3]);
      const c = add(a, b);
      expect(Number(c.data[0])).toBe(Infinity);
    });

    it("should handle -Infinity in subtraction", () => {
      const a = tensor([-Infinity, 1, 2]);
      const b = tensor([1, 2, 3]);
      const c = sub(a, b);
      expect(Number(c.data[0])).toBe(-Infinity);
    });

    it("should handle NaN propagation in multiplication", () => {
      const a = tensor([NaN, 1, 2]);
      const b = tensor([1, 2, 3]);
      const c = mul(a, b);
      expect(Number.isNaN(Number(c.data[0]))).toBe(true);
    });

    it("should handle Infinity * 0 = NaN", () => {
      const a = tensor([Infinity]);
      const b = tensor([0]);
      const c = mul(a, b);
      expect(Number.isNaN(Number(c.data[0]))).toBe(true);
    });
  });

  describe("Zero-dimensional Tensors", () => {
    it("should handle 0D tensor addition", () => {
      const a = tensor(5);
      const b = tensor(10);
      const c = add(a, b);
      expect(Number(c.data[0])).toBe(15);
    });

    it("should handle 0D tensor multiplication", () => {
      const a = tensor(3);
      const b = tensor(4);
      const c = mul(a, b);
      expect(Number(c.data[0])).toBe(12);
    });

    it("should handle 0D tensor sum", () => {
      const a = tensor(42);
      const b = sum(a);
      expect(Number(b.data[0])).toBe(42);
    });
  });

  describe("Empty and Single Element Tensors", () => {
    it("should handle single element addition", () => {
      const a = tensor([5]);
      const b = tensor([3]);
      const c = add(a, b);
      expect(Number(c.data[0])).toBe(8);
    });

    it("should handle single element sum", () => {
      const a = tensor([42]);
      const b = sum(a);
      expect(Number(b.data[0])).toBe(42);
    });

    it("should handle single element mean", () => {
      const a = tensor([42]);
      const b = mean(a);
      expect(Number(b.data[0])).toBe(42);
    });
  });
});
