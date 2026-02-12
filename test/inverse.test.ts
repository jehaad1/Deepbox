import { describe, expect, it } from "vitest";
import { inv, pinv } from "../src/linalg";
import { allclose, eye, tensor, zeros } from "../src/ndarray";
import { matmul } from "../src/ndarray/linalg/basic";

/**
 * Helper to compute Frobenius norm of difference between two tensors
 */
function frobeniusDiff(A: ReturnType<typeof tensor>, B: ReturnType<typeof tensor>): number {
  let sum = 0;
  for (let i = 0; i < A.size; i++) {
    const diff = Number(A.data[A.offset + i]) - Number(B.data[B.offset + i]);
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function randomMatrix(rows: number, cols: number, seed: number = 0): ReturnType<typeof tensor> {
  const data = new Float64Array(rows * cols);
  // Simple LCG for deterministic random numbers in tests
  let state = seed;
  const next = () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return (state / 4294967296) * 2 - 1; // [-1, 1]
  };

  for (let i = 0; i < data.length; i++) {
    data[i] = next();
  }
  return tensor(Array.from(data)).view([rows, cols]);
}

describe("deepbox/linalg - Inverse and Pseudo-Inverse", () => {
  describe("inv", () => {
    it("validates shapes and empty matrices", () => {
      expect(() => inv(tensor([1, 2, 3]))).toThrow(/2D/);
      expect(() => inv(tensor([[1, 2, 3]]))).toThrow(/square/);

      const empty = zeros([0, 0]);
      const out = inv(empty);
      expect(out.shape).toEqual([0, 0]);
    });

    it("computes inverse correctly for a simple matrix", () => {
      const A = tensor([
        [4, 7],
        [2, 6],
      ]);
      const Ainv = inv(A);
      const I = eye(2);
      expect(allclose(matmul(A, Ainv), I, 1e-8, 1e-8)).toBe(true);
      expect(allclose(matmul(Ainv, A), I, 1e-8, 1e-8)).toBe(true);
    });
  });

  describe("pinv", () => {
    it("validates rcond and handles empty matrices", () => {
      expect(() => pinv(tensor([[1, 2]]), -1)).toThrow(/rcond/);
      const empty = zeros([0, 3]);
      const out = pinv(empty);
      expect(out.shape).toEqual([3, 0]);
    });

    it("throws on non-finite values", () => {
      const A = tensor([
        [1, NaN],
        [2, 3],
      ]);
      expect(() => pinv(A)).toThrow(/non-finite/i);
    });

    it("should satisfy A * pinv(A) * A ≈ A for singular matrix", () => {
      const A = tensor([
        [1, 2],
        [2, 4],
      ]);
      const Ap = pinv(A);
      const reconstructed = matmul(matmul(A, Ap), A);

      for (let i = 0; i < A.size; i++) {
        const v = Number(A.data[A.offset + i]);
        const r = Number(reconstructed.data[reconstructed.offset + i]);
        expect(r).toBeCloseTo(v, 4);
      }
    });

    it("should maintain A * pinv(A) * A ≈ A for random rectangular matrices", () => {
      // 10x5 matrix (tall)
      const A = randomMatrix(10, 5, 555);
      const Ap = pinv(A);
      const reconstructed = matmul(matmul(A, Ap), A);
      expect(frobeniusDiff(reconstructed, A)).toBeLessThan(1e-8);

      // 5x10 matrix (wide)
      const B = randomMatrix(5, 10, 777);
      const Bp = pinv(B);
      const reconstructedB = matmul(matmul(B, Bp), B);
      expect(frobeniusDiff(reconstructedB, B)).toBeLessThan(1e-8);
    });
  });
});
