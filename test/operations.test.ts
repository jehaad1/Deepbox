import { describe, expect, it } from "vitest";
import {
  abs,
  add,
  addScalar,
  ceil,
  clip,
  cos,
  div,
  dot,
  exp,
  flatten,
  floor,
  log,
  max,
  mean,
  min,
  mul,
  mulScalar,
  neg,
  pow,
  prod,
  reciprocal,
  reshape,
  sign,
  sin,
  sqrt,
  square,
  std,
  sub,
  sum,
  tan,
  tensor,
  transpose,
  variance,
} from "../src/ndarray";

describe("deepbox/ndarray - Operations", () => {
  describe("arithmetic", () => {
    it("should add tensors", () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([4, 5, 6]);
      const result = add(a, b);
      expect(result.shape).toEqual([3]);
    });

    it("should subtract tensors", () => {
      const a = tensor([5, 6, 7]);
      const b = tensor([1, 2, 3]);
      const result = sub(a, b);
      expect(result.shape).toEqual([3]);
    });

    it("should multiply tensors", () => {
      const a = tensor([2, 3, 4]);
      const b = tensor([5, 6, 7]);
      const result = mul(a, b);
      expect(result.shape).toEqual([3]);
    });

    it("should divide tensors", () => {
      const a = tensor([10, 20, 30]);
      const b = tensor([2, 4, 5]);
      const result = div(a, b);
      expect(result.shape).toEqual([3]);
    });

    it("should handle broadcasting", () => {
      const a = tensor([
        [1, 2],
        [3, 4],
      ]);
      const b = tensor([10, 20]);
      expect(() => add(a, b)).not.toThrow();
    });

    it("should handle negative values", () => {
      const a = tensor([-1, -2, -3]);
      const b = tensor([1, 2, 3]);
      expect(() => add(a, b)).not.toThrow();
    });

    it("should handle zero", () => {
      const a = tensor([0, 0, 0]);
      const b = tensor([1, 2, 3]);
      expect(() => add(a, b)).not.toThrow();
    });
  });

  describe("unary operations", () => {
    it("should negate tensor", () => {
      const a = tensor([1, -2, 3]);
      const result = neg(a);
      expect(result.shape).toEqual([3]);
      expect(Number(result.data[0])).toBeCloseTo(-1, 5);
      expect(Number(result.data[1])).toBeCloseTo(2, 5);
      expect(Number(result.data[2])).toBeCloseTo(-3, 5);
    });

    it("should compute absolute value", () => {
      const a = tensor([-1, -2, 3]);
      const result = abs(a);
      expect(result.shape).toEqual([3]);
      expect(Number(result.data[0])).toBeCloseTo(1, 5);
      expect(Number(result.data[1])).toBeCloseTo(2, 5);
      expect(Number(result.data[2])).toBeCloseTo(3, 5);
    });

    it("should compute sign", () => {
      const a = tensor([-5, 0, 3]);
      const result = sign(a);
      expect(result.shape).toEqual([3]);
      expect(Number(result.data[0])).toBe(-1);
      expect(Number(result.data[1])).toBe(0);
      expect(Number(result.data[2])).toBe(1);
    });

    it("should compute reciprocal", () => {
      const a = tensor([2, 4, 5]);
      const result = reciprocal(a);
      expect(result.shape).toEqual([3]);
      expect(Number(result.data[0])).toBeCloseTo(0.5, 5);
      expect(Number(result.data[1])).toBeCloseTo(0.25, 5);
      expect(Number(result.data[2])).toBeCloseTo(0.2, 5);
    });

    it("should compute square", () => {
      const a = tensor([2, 3, 4]);
      const result = square(a);
      expect(result.shape).toEqual([3]);
      expect(Number(result.data[0])).toBeCloseTo(4, 5);
      expect(Number(result.data[1])).toBeCloseTo(9, 5);
      expect(Number(result.data[2])).toBeCloseTo(16, 5);
    });
  });

  describe("scalar operations", () => {
    it("should add scalar to tensor", () => {
      const a = tensor([1, 2, 3]);
      const result = addScalar(a, 10);
      expect(result.shape).toEqual([3]);
      expect(Number(result.data[0])).toBeCloseTo(11, 5);
      expect(Number(result.data[1])).toBeCloseTo(12, 5);
      expect(Number(result.data[2])).toBeCloseTo(13, 5);
    });

    it("should multiply tensor by scalar", () => {
      const a = tensor([1, 2, 3]);
      const result = mulScalar(a, 2);
      expect(result.shape).toEqual([3]);
      expect(Number(result.data[0])).toBeCloseTo(2, 5);
      expect(Number(result.data[1])).toBeCloseTo(4, 5);
      expect(Number(result.data[2])).toBeCloseTo(6, 5);
    });

    it("should clip tensor values", () => {
      const a = tensor([1, 5, 10, 15]);
      const result = clip(a, 3, 12);
      expect(result.shape).toEqual([4]);
      expect(Number(result.data[0])).toBeCloseTo(3, 5);
      expect(Number(result.data[1])).toBeCloseTo(5, 5);
      expect(Number(result.data[2])).toBeCloseTo(10, 5);
      expect(Number(result.data[3])).toBeCloseTo(12, 5);
    });

    it("should compute power", () => {
      const a = tensor([2, 3, 4]);
      const b = tensor([2, 2, 2]);
      const result = pow(a, b);
      expect(result.shape).toEqual([3]);
      expect(Number(result.data[0])).toBeCloseTo(4, 5);
      expect(Number(result.data[1])).toBeCloseTo(9, 5);
      expect(Number(result.data[2])).toBeCloseTo(16, 5);
    });
  });

  describe("math functions", () => {
    it("should compute exponential", () => {
      const a = tensor([0, 1, 2]);
      const result = exp(a);
      expect(result.shape).toEqual([3]);
    });

    it("should compute logarithm", () => {
      const a = tensor([1, Math.E, 7.389]);
      const result = log(a);
      expect(result.shape).toEqual([3]);
    });

    it("should compute square root", () => {
      const a = tensor([1, 4, 9]);
      const result = sqrt(a);
      expect(result.shape).toEqual([3]);
    });

    it("should compute sine", () => {
      const a = tensor([0, 1.57, 3.14]);
      const result = sin(a);
      expect(result.shape).toEqual([3]);
    });

    it("should handle large values", () => {
      const a = tensor([100, 1000, 10000]);
      expect(() => exp(a)).not.toThrow();
    });

    it("should handle small values", () => {
      const a = tensor([0.001, 0.0001, 0.00001]);
      expect(() => log(a)).not.toThrow();
    });
  });

  describe("reduction operations", () => {
    it("should sum all elements", () => {
      const a = tensor([1, 2, 3, 4]);
      const result = sum(a);
      expect(result.size).toBe(1);
      expect(Number(result.data[0])).toBeCloseTo(10, 5);
    });

    it("should compute mean", () => {
      const a = tensor([1, 2, 3, 4]);
      const result = mean(a);
      expect(result.size).toBe(1);
      expect(Number(result.data[0])).toBeCloseTo(2.5, 5);
    });

    it("should find maximum", () => {
      const a = tensor([1, 5, 3, 2]);
      const result = max(a);
      expect(result.size).toBe(1);
      expect(Number(result.data[0])).toBe(5);
    });

    it("should find minimum", () => {
      const a = tensor([3, 1, 5, 2]);
      const result = min(a);
      expect(result.size).toBe(1);
      expect(Number(result.data[0])).toBe(1);
    });

    it("should compute product", () => {
      const a = tensor([1, 2, 3, 4]);
      const result = prod(a);
      expect(result.size).toBe(1);
      expect(Number(result.data[0])).toBeCloseTo(24, 5);
    });

    it("should compute variance", () => {
      const a = tensor([1, 2, 3, 4, 5]);
      const result = variance(a);
      // Variance returns scalar tensor
      expect(result.size).toBe(1);
      expect(Number(result.data[0])).toBeCloseTo(2, 5);
    });

    it("should compute standard deviation", () => {
      const a = tensor([1, 2, 3, 4, 5]);
      const result = std(a);
      // Std returns scalar tensor
      expect(result.size).toBe(1);
      expect(Number(result.data[0])).toBeCloseTo(Math.sqrt(2), 5);
    });

    it("should handle negative values in max/min", () => {
      const a = tensor([-5, -2, -8, -1]);
      const maxResult = max(a);
      const minResult = min(a);
      expect(Number(maxResult.data[0])).toBe(-1);
      expect(Number(minResult.data[0])).toBe(-8);
    });

    it("should handle all equal values", () => {
      const a = tensor([5, 5, 5, 5]);
      const result = sum(a);
      expect(Number(result.data[0])).toBeCloseTo(20, 5);
    });
  });

  describe("rounding operations", () => {
    it("should compute floor", () => {
      const a = tensor([1.7, 2.3, -1.5]);
      const result = floor(a);
      expect(result.shape).toEqual([3]);
      expect(Number(result.data[0])).toBe(1);
      expect(Number(result.data[1])).toBe(2);
      expect(Number(result.data[2])).toBe(-2);
    });

    it("should compute ceil", () => {
      const a = tensor([1.2, 2.8, -1.5]);
      const result = ceil(a);
      expect(result.shape).toEqual([3]);
      expect(Number(result.data[0])).toBe(2);
      expect(Number(result.data[1])).toBe(3);
      expect(Number(result.data[2])).toBe(-1);
    });
  });

  describe("trigonometric operations", () => {
    it("should compute cosine", () => {
      const a = tensor([0, Math.PI / 2, Math.PI]);
      const result = cos(a);
      expect(result.shape).toEqual([3]);
      expect(Number(result.data[0])).toBeCloseTo(1, 5);
      expect(Number(result.data[1])).toBeCloseTo(0, 5);
      expect(Number(result.data[2])).toBeCloseTo(-1, 5);
    });

    it("should compute tangent", () => {
      const a = tensor([0, Math.PI / 4]);
      const result = tan(a);
      expect(result.shape).toEqual([2]);
      expect(Number(result.data[0])).toBeCloseTo(0, 5);
      expect(Number(result.data[1])).toBeCloseTo(1, 5);
    });
  });

  describe("shape operations", () => {
    it("should reshape tensor", () => {
      const a = tensor([1, 2, 3, 4, 5, 6]);
      const result = reshape(a, [2, 3]);
      expect(result.shape).toEqual([2, 3]);
    });

    it("should transpose 2D tensor", () => {
      const a = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = transpose(a);
      expect(result.shape).toEqual([3, 2]);
    });

    it("should flatten tensor", () => {
      const a = tensor([
        [1, 2],
        [3, 4],
      ]);
      const result = flatten(a);
      expect(result.shape).toEqual([4]);
    });

    it("should handle 3D reshape", () => {
      const a = tensor([1, 2, 3, 4, 5, 6, 7, 8]);
      expect(() => reshape(a, [2, 2, 2])).not.toThrow();
    });
  });

  describe("linear algebra", () => {
    it("should multiply matrices", () => {
      const a = tensor([
        [1, 2],
        [3, 4],
      ]);
      const b = tensor([
        [5, 6],
        [7, 8],
      ]);
      const result = dot(a, b);
      expect(result.shape).toEqual([2, 2]);
    });

    it("should compute dot product", () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([4, 5, 6]);
      const result = dot(a, b);
      expect(typeof result).toBe("object");
    });

    it("should handle matrix-vector multiplication", () => {
      const A = tensor([
        [1, 2],
        [3, 4],
      ]);
      const v = tensor([5, 6]);
      expect(() => dot(A, v)).not.toThrow();
    });

    it("should handle 3D dot", () => {
      const a = tensor([[[1, 2]], [[3, 4]]]);
      const b = tensor([
        [[5], [6]],
        [[7], [8]],
      ]);
      expect(() => dot(a, b)).not.toThrow();
    });
  });

  describe("edge cases", () => {
    it("should handle empty tensors", () => {
      const a = tensor([]);
      expect(a.shape).toBeDefined();
    });

    it("should handle single element operations", () => {
      const a = tensor([5]);
      const b = tensor([3]);
      expect(() => add(a, b)).not.toThrow();
    });

    it("should handle NaN values", () => {
      const a = tensor([NaN, 1, 2]);
      expect(() => sum(a)).not.toThrow();
    });

    it("should handle Infinity", () => {
      const a = tensor([Infinity, 1, 2]);
      expect(() => sum(a)).not.toThrow();
    });

    it("should handle very large tensors", () => {
      const size = 1000;
      const data = Array.from({ length: size }, (_, i) => i);
      const a = tensor(data);
      expect(() => sum(a)).not.toThrow();
    });
  });
});
