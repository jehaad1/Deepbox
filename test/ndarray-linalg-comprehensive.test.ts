import { describe, expect, it } from "vitest";
import { dot, reshape, tensor } from "../src/ndarray";
import { matmul } from "../src/ndarray/linalg/basic";
import { numData } from "./_helpers";

describe("deepbox/ndarray - Linear Algebra Comprehensive Tests", () => {
  describe("dot product - 1D vectors", () => {
    it("should compute dot product of simple vectors", () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([4, 5, 6]);
      const c = dot(a, b);
      expect(c.shape).toEqual([]);
      expect(Number(c.data[0])).toBe(32);
    });

    it("should compute dot product with zeros", () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([0, 0, 0]);
      const c = dot(a, b);
      expect(Number(c.data[0])).toBe(0);
    });

    it("should compute dot product with negative values", () => {
      const a = tensor([1, -2, 3]);
      const b = tensor([-1, 2, -3]);
      const c = dot(a, b);
      expect(Number(c.data[0])).toBe(-14);
    });

    it("should compute dot product of unit vectors", () => {
      const a = tensor([1, 0, 0]);
      const b = tensor([0, 1, 0]);
      const c = dot(a, b);
      expect(Number(c.data[0])).toBe(0);
    });

    it("should compute dot product with single element", () => {
      const a = tensor([5]);
      const b = tensor([3]);
      const c = dot(a, b);
      expect(Number(c.data[0])).toBe(15);
    });

    it("should throw on mismatched sizes", () => {
      const a = tensor([1, 2, 3]);
      const b = tensor([1, 2]);
      expect(() => dot(a, b)).toThrow();
    });

    it("should compute dot product with large vectors", () => {
      const size = 1000;
      const data = Array.from({ length: size }, () => 1);
      const a = tensor(data);
      const b = tensor(data);
      const c = dot(a, b);
      expect(Number(c.data[0])).toBe(size);
    });

    it("should preserve dtype for int32", () => {
      const a = tensor([1, 2, 3], { dtype: "int32" });
      const b = tensor([4, 5, 6], { dtype: "int32" });
      const c = dot(a, b);
      expect(c.dtype).toBe("int32");
    });

    it("should handle float32 dtype", () => {
      const a = tensor([1, 2, 3], { dtype: "float32" });
      const b = tensor([4, 5, 6], { dtype: "float32" });
      const c = dot(a, b);
      expect(c.dtype).toBe("float32");
    });
  });

  describe("matrix multiplication - 2D", () => {
    it("should multiply 2x2 matrices", () => {
      const a = tensor([
        [1, 2],
        [3, 4],
      ]);
      const b = tensor([
        [5, 6],
        [7, 8],
      ]);
      const c = matmul(a, b);
      expect(c.shape).toEqual([2, 2]);
      expect(numData(c)).toEqual([19, 22, 43, 50]);
    });

    it("should multiply 2x3 and 3x2 matrices", () => {
      const a = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const b = tensor([
        [7, 8],
        [9, 10],
        [11, 12],
      ]);
      const c = matmul(a, b);
      expect(c.shape).toEqual([2, 2]);
      expect(numData(c)).toEqual([58, 64, 139, 154]);
    });

    it("should multiply with identity matrix", () => {
      const a = tensor([
        [1, 2],
        [3, 4],
      ]);
      const identity = tensor([
        [1, 0],
        [0, 1],
      ]);
      const c = matmul(a, identity);
      expect(numData(c)).toEqual([1, 2, 3, 4]);
    });

    it("should multiply with zero matrix", () => {
      const a = tensor([
        [1, 2],
        [3, 4],
      ]);
      const zeros = tensor([
        [0, 0],
        [0, 0],
      ]);
      const c = matmul(a, zeros);
      expect(numData(c)).toEqual([0, 0, 0, 0]);
    });

    it("should handle 1x1 matrices", () => {
      const a = tensor([[5]]);
      const b = tensor([[3]]);
      const c = matmul(a, b);
      expect(c.shape).toEqual([1, 1]);
      expect(Number(c.data[0])).toBe(15);
    });

    it("should handle rectangular matrices", () => {
      const a = tensor([[1, 2, 3]]);
      const b = tensor([[4], [5], [6]]);
      const c = matmul(a, b);
      expect(c.shape).toEqual([1, 1]);
      expect(Number(c.data[0])).toBe(32);
    });

    it("should multiply compatible shapes (2x2 * 2x3)", () => {
      const a = tensor([
        [1, 2],
        [3, 4],
      ]);
      const b = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const c = matmul(a, b);
      expect(c.shape).toEqual([2, 3]);
      expect(numData(c)).toEqual([9, 12, 15, 19, 26, 33]);
    });

    it("should handle negative values", () => {
      const a = tensor([
        [1, -2],
        [-3, 4],
      ]);
      const b = tensor([
        [-5, 6],
        [7, -8],
      ]);
      const c = matmul(a, b);
      expect(numData(c)).toEqual([-19, 22, 43, -50]);
    });

    it("should preserve int32 dtype", () => {
      const a = tensor(
        [
          [1, 2],
          [3, 4],
        ],
        { dtype: "int32" }
      );
      const b = tensor(
        [
          [5, 6],
          [7, 8],
        ],
        { dtype: "int32" }
      );
      const c = matmul(a, b);
      expect(c.dtype).toBe("int32");
    });

    it("should handle large matrices", () => {
      const size = 50;
      const data = Array.from({ length: size * size }, () => 1);
      const a = reshape(tensor(data), [size, size]);
      const b = reshape(tensor(data), [size, size]);
      const c = matmul(a, b);
      expect(c.shape).toEqual([size, size]);
      expect(Number(c.data[0])).toBe(size);
    });
  });

  describe("dot - matrix-vector multiplication", () => {
    it("should multiply 2x2 matrix with 2D vector", () => {
      const a = tensor([
        [1, 2],
        [3, 4],
      ]);
      const b = tensor([5, 6]);
      const c = dot(a, b);
      expect(c.shape).toEqual([2]);
      expect(numData(c)).toEqual([17, 39]);
    });

    it("should multiply 3x3 matrix with 3D vector", () => {
      const a = tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
      const b = tensor([5, 6, 7]);
      const c = dot(a, b);
      expect(numData(c)).toEqual([5, 6, 7]);
    });

    it("should handle zero vector", () => {
      const a = tensor([
        [1, 2],
        [3, 4],
      ]);
      const b = tensor([0, 0]);
      const c = dot(a, b);
      expect(numData(c)).toEqual([0, 0]);
    });

    it("should throw on incompatible shapes", () => {
      const a = tensor([
        [1, 2],
        [3, 4],
      ]);
      const b = tensor([1, 2, 3]);
      expect(() => dot(a, b)).toThrow();
    });
  });

  describe("dot - batch matrix multiplication (3D)", () => {
    it("should multiply 3D tensors (batch matmul)", () => {
      const a = tensor([[[1, 2]], [[3, 4]]]);
      const b = tensor([
        [[5], [6]],
        [[7], [8]],
      ]);
      const c = dot(a, b);
      expect(c.shape).toEqual([2, 1, 1]);
      expect(numData(c)).toEqual([17, 53]);
    });

    it("should handle batch size 1", () => {
      const a = tensor([
        [
          [1, 2],
          [3, 4],
        ],
      ]);
      const b = tensor([
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const c = dot(a, b);
      expect(c.shape).toEqual([1, 2, 2]);
      expect(numData(c)).toEqual([19, 22, 43, 50]);
    });

    it("should throw on mismatched batch dimensions", () => {
      const a = tensor([[[1, 2]], [[3, 4]]]);
      const b = tensor([[[5], [6]]]);
      expect(() => dot(a, b)).toThrow("batch dimensions");
    });

    it("should handle identity batch multiplication", () => {
      const a = tensor([
        [
          [1, 0],
          [0, 1],
        ],
        [
          [1, 0],
          [0, 1],
        ],
      ]);
      const b = tensor([
        [
          [2, 3],
          [4, 5],
        ],
        [
          [6, 7],
          [8, 9],
        ],
      ]);
      const c = dot(a, b);
      expect(c.shape).toEqual([2, 2, 2]);
      expect(numData(c)).toEqual([2, 3, 4, 5, 6, 7, 8, 9]);
    });
  });

  describe("edge cases and error handling", () => {
    it("should throw on string dtype", () => {
      const a = tensor(["a", "b"]);
      const b = tensor(["c", "d"]);
      expect(() => dot(a, b)).toThrow("string dtype");
    });

    it("should throw on 0D tensors", () => {
      const a = tensor(5);
      const b = tensor(10);
      expect(() => dot(a, b)).toThrow();
    });

    it("should handle very small values", () => {
      const a = tensor([0.0001, 0.0002, 0.0003]);
      const b = tensor([0.0001, 0.0002, 0.0003]);
      const c = dot(a, b);
      expect(Number(c.data[0])).toBeCloseTo(0.00000014);
    });

    it("should handle very large values", () => {
      const a = tensor([1e10, 2e10, 3e10]);
      const b = tensor([1e10, 2e10, 3e10]);
      const c = dot(a, b);
      expect(Number(c.data[0])).toBeGreaterThan(1e20);
    });

    it("should handle mixed positive and negative", () => {
      const a = tensor([1, -2, 3, -4]);
      const b = tensor([-1, 2, -3, 4]);
      const c = dot(a, b);
      expect(Number(c.data[0])).toBe(-30);
    });
  });

  describe("performance and stress tests", () => {
    it("should handle 100x100 matrix multiplication", () => {
      const size = 100;
      const data = Array.from({ length: size * size }, (_, i) => i % 10);
      const a = reshape(tensor(data), [size, size]);
      const b = reshape(tensor(data), [size, size]);
      const c = matmul(a, b);
      expect(c.shape).toEqual([size, size]);
      expect(c.size).toBe(size * size);
    });

    it("should handle long vectors (10000 elements)", () => {
      const size = 10000;
      const data = Array.from({ length: size }, () => 1);
      const a = tensor(data);
      const b = tensor(data);
      const c = dot(a, b);
      expect(Number(c.data[0])).toBe(size);
    });

    it("should handle sparse-like matrices (mostly zeros)", () => {
      const a = tensor([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3],
      ]);
      const b = tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
      const c = matmul(a, b);
      expect(numData(c)).toEqual([1, 0, 0, 0, 2, 0, 0, 0, 3]);
    });
  });

  describe("special matrix operations", () => {
    it("should multiply diagonal matrices", () => {
      const a = tensor([
        [2, 0],
        [0, 3],
      ]);
      const b = tensor([
        [4, 0],
        [0, 5],
      ]);
      const c = matmul(a, b);
      expect(numData(c)).toEqual([8, 0, 0, 15]);
    });

    it("should multiply upper triangular matrices", () => {
      const a = tensor([
        [1, 2],
        [0, 3],
      ]);
      const b = tensor([
        [4, 5],
        [0, 6],
      ]);
      const c = matmul(a, b);
      expect(numData(c)).toEqual([4, 17, 0, 18]);
    });

    it("should multiply lower triangular matrices", () => {
      const a = tensor([
        [1, 0],
        [2, 3],
      ]);
      const b = tensor([
        [4, 0],
        [5, 6],
      ]);
      const c = matmul(a, b);
      expect(numData(c)).toEqual([4, 0, 23, 18]);
    });

    it("should handle symmetric matrices", () => {
      const a = tensor([
        [1, 2],
        [2, 3],
      ]);
      const b = tensor([
        [1, 2],
        [2, 3],
      ]);
      const c = matmul(a, b);
      expect(numData(c)).toEqual([5, 8, 8, 13]);
    });
  });

  describe("numerical stability", () => {
    it("should handle matrices with very different magnitudes", () => {
      const a = tensor([
        [1e-10, 2e-10],
        [3e-10, 4e-10],
      ]);
      const b = tensor([
        [1e10, 2e10],
        [3e10, 4e10],
      ]);
      const c = matmul(a, b);
      expect(c.shape).toEqual([2, 2]);
      const arr = numData(c);
      expect(arr[0]).toBeCloseTo(7, 5);
      expect(arr[1]).toBeCloseTo(10, 5);
      expect(arr[2]).toBeCloseTo(15, 5);
      expect(arr[3]).toBeCloseTo(22, 5);
    });

    it("should handle near-singular matrices", () => {
      const a = tensor([
        [1, 1],
        [1, 1.0001],
      ]);
      const b = tensor([
        [2, 3],
        [4, 5],
      ]);
      const c = matmul(a, b);
      expect(c.shape).toEqual([2, 2]);
      const arr = numData(c);
      expect(arr[0]).toBeCloseTo(6, 6);
      expect(arr[1]).toBeCloseTo(8, 6);
      expect(arr[2]).toBeCloseTo(6.0004, 6);
      expect(arr[3]).toBeCloseTo(8.0005, 6);
    });

    it("should handle orthogonal matrices", () => {
      const sqrt2 = Math.sqrt(2);
      const a = tensor([
        [1 / sqrt2, -1 / sqrt2],
        [1 / sqrt2, 1 / sqrt2],
      ]);
      const b = tensor([
        [1 / sqrt2, 1 / sqrt2],
        [-1 / sqrt2, 1 / sqrt2],
      ]);
      const c = matmul(a, b);
      expect(Number(c.data[0])).toBeCloseTo(1);
      expect(Number(c.data[1])).toBeCloseTo(0, 5);
      expect(Number(c.data[2])).toBeCloseTo(0, 5);
      expect(Number(c.data[3])).toBeCloseTo(1);
    });
  });
});
