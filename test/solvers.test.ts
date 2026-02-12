import { describe, expect, it } from "vitest";
import { lstsq, solve, solveTriangular } from "../src/linalg";
import { tensor } from "../src/ndarray";

describe("deepbox/linalg - Solvers", () => {
  describe("solve", () => {
    it("should solve 2x2 system", () => {
      const A = tensor([
        [3, 1],
        [1, 2],
      ]);
      const b = tensor([9, 8]);
      const x = solve(A, b);
      expect(x.shape).toEqual([2]);
      // Verify A @ x = b
      expect(Number(x.data[0])).toBeCloseTo(2, 5);
      expect(Number(x.data[1])).toBeCloseTo(3, 5);
    });

    it("should solve 3x3 system", () => {
      const A = tensor([
        [2, 1, 1],
        [1, 3, 2],
        [1, 0, 0],
      ]);
      const b = tensor([4, 5, 6]);
      const x = solve(A, b);
      expect(x.shape).toEqual([3]);
      // Verify solution x = [6, -9, 1]
      // 2*6 + 1*-9 + 1*1 = 12 - 9 + 1 = 4 (ok)
      // 1*6 + 3*-9 + 2*1 = 6 - 27 + 2 = -19 != 5 ???
      // Wait, let's just check correctness manually or via closeTo
      // 1*x0 = 6 => x0 = 6.
      // x0 + 3x1 + 2x2 = 5 => 6 + 3x1 + 2x2 = 5 => 3x1 + 2x2 = -1
      // 2x0 + x1 + x2 = 4 => 12 + x1 + x2 = 4 => x1 + x2 = -8 => x2 = -8 - x1
      // 3x1 + 2(-8 - x1) = -1 => 3x1 - 16 - 2x1 = -1 => x1 = 15
      // x2 = -8 - 15 = -23
      // Let's check:
      // A = [[2,1,1],[1,3,2],[1,0,0]]
      // b = [4,5,6]
      // Row 3: x0 = 6. Correct.
      expect(Number(x.data[0])).toBeCloseTo(6, 5);
      expect(Number(x.data[1])).toBeCloseTo(15, 5);
      expect(Number(x.data[2])).toBeCloseTo(-23, 5);
    });

    it("should solve system with identity matrix", () => {
      const I = tensor([
        [1, 0],
        [0, 1],
      ]);
      const b = tensor([3, 5]);
      const x = solve(I, b);
      expect(Number(x.data[0])).toBeCloseTo(3, 5);
      expect(Number(x.data[1])).toBeCloseTo(5, 5);
    });

    it("should solve with multiple RHS", () => {
      const A = tensor([
        [2, 0],
        [0, 2],
      ]);
      const B = tensor([
        [2, 4],
        [6, 8],
      ]);
      const X = solve(A, B);
      expect(X.shape).toEqual([2, 2]);
      expect(X.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });

    it("validates solve input shapes", () => {
      expect(() => solve(tensor([1, 2]), tensor([1, 2]))).toThrow(/2D matrix/i);
      expect(() => solve(tensor([[1, 2, 3]]), tensor([1]))).toThrow(/square/i);
      expect(() =>
        solve(
          tensor([
            [1, 2],
            [3, 4],
          ]),
          tensor([[1]])
        )
      ).toThrow(/dimensions do not match/i);
    });

    it("should throw on singular matrix", () => {
      const A = tensor([
        [1, 2],
        [2, 4],
      ]);
      const b = tensor([1, 2]);
      expect(() => solve(A, b)).toThrow();
    });

    it("should throw on non-finite values", () => {
      const A = tensor([
        [1, NaN],
        [2, 3],
      ]);
      const b = tensor([1, 2]);
      expect(() => solve(A, b)).toThrow(/non-finite/i);
    });
  });

  describe("solveTriangular", () => {
    it("should solve lower triangular system", () => {
      const L = tensor([
        [2, 0],
        [3, 4],
      ]);
      const b = tensor([6, 18]);
      const x = solveTriangular(L, b, true);
      // 2x0 = 6 => x0=3
      // 3x0 + 4x1 = 18 => 9 + 4x1 = 18 => 4x1=9 => x1=2.25
      expect(Number(x.data[0])).toBeCloseTo(3, 5);
      expect(Number(x.data[1])).toBeCloseTo(2.25, 5);
    });

    it("should solve upper triangular system", () => {
      const U = tensor([
        [2, 1],
        [0, 4],
      ]);
      const b = tensor([6, 18]); // Changed b to match branch tests if needed, but let's stick to simple
      // 4x1 = 18 => x1 = 4.5
      // 2x0 + x1 = 6 => 2x0 + 4.5 = 6 => 2x0 = 1.5 => x0 = 0.75
      const x = solveTriangular(U, b, false);
      expect(Number(x.data[1])).toBeCloseTo(4.5, 5);
      expect(Number(x.data[0])).toBeCloseTo(0.75, 5);
    });

    it("validates triangular solver inputs", () => {
      const L = tensor([
        [0, 0],
        [1, 2],
      ]);
      expect(() => solveTriangular(L, tensor([1, 2]), true)).toThrow(/singular/i);
    });
  });

  describe("lstsq", () => {
    it("should solve overdetermined system (minimize residual)", () => {
      const A = tensor([
        [1, 1],
        [1, 2],
        [1, 3],
      ]);
      const b = tensor([2, 3, 5]);
      // Fits y = mx + c. points (1,2), (2,3), (3,5).
      // x=[c, m].
      // c + m = 2
      // c + 2m = 3
      // c + 3m = 5
      // Normal equations: A^T A x = A^T b
      // A^T A = [[3, 6], [6, 14]]
      // A^T b = [10, 23]
      // 3c + 6m = 10
      // 6c + 14m = 23
      // 6c + 12m = 20
      // 2m = 3 => m = 1.5
      // 3c + 9 = 10 => 3c = 1 => c = 1/3 = 0.333
      const result = lstsq(A, b);
      expect(result.x.shape).toEqual([2]);
      expect(Number(result.x.data[0])).toBeCloseTo(1 / 3, 5);
      expect(Number(result.x.data[1])).toBeCloseTo(1.5, 5);
      expect(result.rank).toBe(2);
      expect(Number(result.residuals.data[result.residuals.offset])).toBeCloseTo(1 / 6, 5);
    });

    it("should solve underdetermined system (minimize norm)", () => {
      // x + y = 2. Min norm solution is x=1, y=1.
      const A = tensor([[1, 1]]);
      const b = tensor([2]);
      const result = lstsq(A, b);
      expect(Number(result.x.data[0])).toBeCloseTo(1, 5);
      expect(Number(result.x.data[1])).toBeCloseTo(1, 5);
      expect(result.rank).toBe(1);
      expect(Number(result.residuals.data[result.residuals.offset])).toBeCloseTo(0, 8);
    });

    it("should handle rank-deficient matrix", () => {
      const A = tensor([
        [1, 2],
        [2, 4],
        [3, 6],
      ]);
      const b = tensor([1, 2, 3]);
      // Col2 is 2*Col1. Rank 1.
      // x + 2y = 1
      // 2x + 4y = 2
      // 3x + 6y = 3
      // Infinite solutions. Min norm solution?
      // x + 2y = 1. Min norm of (x,y) subject to this.
      // Lagrange: x^2 + y^2 - lambda(x + 2y - 1).
      // 2x = lambda, 2y = 2lambda => y = 2x.
      // x + 2(2x) = 1 => 5x = 1 => x = 0.2. y = 0.4.
      const result = lstsq(A, b);
      expect(Number(result.x.data[0])).toBeCloseTo(0.2, 5);
      expect(Number(result.x.data[1])).toBeCloseTo(0.4, 5);
      expect(result.rank).toBe(1);
      expect(Number(result.residuals.data[result.residuals.offset])).toBeCloseTo(0, 8);
    });

    it("validates input dimensions", () => {
      expect(() => lstsq(tensor([1, 2, 3]), tensor([1, 2]))).toThrow(/2D/);
      expect(() => lstsq(tensor([[1, 2]]), tensor([[[1]]]))).toThrow(/1D or 2D/);
    });

    it("throws on non-finite values", () => {
      const A = tensor([
        [1, NaN],
        [2, 3],
      ]);
      const b = tensor([1, 2]);
      expect(() => lstsq(A, b)).toThrow(/non-finite/i);
    });

    it("respects rcond cutoff for rank", () => {
      const A = tensor([
        [1, 0],
        [0, 1e-12],
      ]);
      const b = tensor([1, 1]);
      const full = lstsq(A, b, 0);
      expect(full.rank).toBe(2);
      const cut = lstsq(A, b, 0.9);
      expect(cut.rank).toBe(1);
    });

    it("should handle 2D right-hand side", () => {
      const A = tensor([
        [1, 0],
        [0, 1],
      ]);
      const B = tensor([
        [1, 2],
        [3, 4],
      ]);
      const result = lstsq(A, B);
      expect(result.x.shape).toEqual([2, 2]);
      expect(Number(result.x.data[0])).toBeCloseTo(1, 5);
      expect(Number(result.x.data[3])).toBeCloseTo(4, 5);
      expect(result.residuals.toArray()).toEqual([0, 0]);
    });
  });
});
