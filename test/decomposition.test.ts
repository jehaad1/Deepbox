import { describe, expect, it } from "vitest";
import { ConvergenceError } from "../src/core";
import { cholesky, eig, eigh, lu, qr, svd } from "../src/linalg";
import { allclose, eye, tensor, transpose, zeros } from "../src/ndarray";
import { matmul } from "../src/ndarray/linalg/basic";

describe("deepbox/linalg - Decompositions", () => {
  describe("svd", () => {
    it("should decompose 2x2 matrix and return correct values", () => {
      const A = tensor([
        [1, 2],
        [3, 4],
      ]);
      const [U, s, Vt] = svd(A);
      expect(U.shape).toEqual([2, 2]);
      expect(s.shape).toEqual([2]);
      expect(Vt.shape).toEqual([2, 2]);

      // Check singular values (approx 5.465, 0.366)
      const sData = s.data;
      if (!(sData instanceof Float64Array)) {
        throw new Error("Expected Float64Array");
      }
      expect(sData[s.offset]).toBeCloseTo(5.4649857, 5);
      expect(sData[s.offset + 1]).toBeCloseTo(0.36596619, 5);
    });

    it("should decompose rectangular matrix (M > N)", () => {
      const A = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const [U, s, Vt] = svd(A, false);
      expect(U.shape).toEqual([3, 2]);
      expect(s.shape).toEqual([2]);
      expect(Vt.shape).toEqual([2, 2]);
      const k = s.size;
      const S = zeros([k, k]);
      for (let i = 0; i < k; i++) {
        S.data[S.offset + i * k + i] = Number(s.data[s.offset + i]);
      }
      const reconstructed = matmul(matmul(U, S), Vt);
      expect(allclose(reconstructed, A, 1e-6, 1e-6)).toBe(true);
    });

    it("should decompose rectangular matrix (M < N)", () => {
      const A = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const [U, s, Vt] = svd(A, false);
      expect(U.shape).toEqual([2, 2]);
      expect(s.shape).toEqual([2]);
      expect(Vt.shape).toEqual([2, 3]);
      const k = s.size;
      const S = zeros([k, k]);
      for (let i = 0; i < k; i++) {
        S.data[S.offset + i * k + i] = Number(s.data[s.offset + i]);
      }
      const reconstructed = matmul(matmul(U, S), Vt);
      expect(allclose(reconstructed, A, 1e-6, 1e-6)).toBe(true);
    });

    it("should return non-negative singular values sorted descending", () => {
      const A = tensor([
        [1, 2],
        [3, 4],
      ]);
      const [_U, s, _Vt] = svd(A);
      const sData = s.data;
      if (!(sData instanceof Float64Array)) {
        throw new Error("Expected Float64Array");
      }
      for (let i = 0; i < s.size; i++) {
        const val = sData[s.offset + i] ?? 0;
        expect(val).toBeGreaterThanOrEqual(0);
        if (i > 0) {
          const prev = sData[s.offset + i - 1] ?? 0;
          expect(prev).toBeGreaterThanOrEqual(val);
        }
      }
    });

    it("should handle identity matrix", () => {
      const I = tensor([
        [1, 0],
        [0, 1],
      ]);
      const [_U, s, _Vt] = svd(I);
      expect(Number(s.data[s.offset])).toBeCloseTo(1, 5);
      expect(Number(s.data[s.offset + 1])).toBeCloseTo(1, 5);
    });

    it("should handle zero matrix", () => {
      const Z = tensor([
        [0, 0],
        [0, 0],
      ]);
      const [_U, s, _Vt] = svd(Z);
      expect(Number(s.data[s.offset])).toBeCloseTo(0, 5);
    });

    it("should handle single element matrix", () => {
      const A = tensor([[5]]);
      const [_U, s, _Vt] = svd(A);
      expect(Number(s.data[s.offset])).toBeCloseTo(5, 5);
    });

    it("handles empty matrices", () => {
      const A = zeros([0, 2]);
      const [U, s, Vt] = svd(A, false);
      expect(U.shape).toEqual([0, 0]);
      expect(s.shape).toEqual([0]);
      expect(Vt.shape).toEqual([0, 2]);
    });

    it("returns full orthonormal U and Vt when fullMatrices=true", () => {
      const A = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const [U, s, Vt] = svd(A, true);
      expect(U.shape).toEqual([3, 3]);
      expect(s.shape).toEqual([2]);
      expect(Vt.shape).toEqual([2, 2]);

      // Check orthogonality of U columns
      const UtU = matmul(transpose(U), U);
      expect(allclose(UtU, eye(3), 1e-10, 1e-10)).toBe(true);
    });

    it("should throw on non-finite values", () => {
      const A = tensor([
        [NaN, 1],
        [2, 3],
      ]);
      expect(() => svd(A)).toThrow(/non-finite/i);
    });
  });

  describe("qr", () => {
    it("should decompose square matrix and return correct values", () => {
      const A = tensor([
        [12, -51, 4],
        [6, 167, -68],
        [-4, 24, -41],
      ]);
      const [Q, R] = qr(A);
      // Q^T Q = I
      const QTQ = matmul(transpose(Q), Q);
      expect(allclose(QTQ, eye(3), 1e-10, 1e-10)).toBe(true);
      // R is upper triangular
      expect(Number(R.data[R.offset + 3])).toBeCloseTo(0, 10); // R[1,0]
      expect(Number(R.data[R.offset + 6])).toBeCloseTo(0, 10); // R[2,0]
      expect(Number(R.data[R.offset + 7])).toBeCloseTo(0, 10); // R[2,1]
      // QR = A
      expect(allclose(matmul(Q, R), A, 1e-10, 1e-10)).toBe(true);
    });

    it("should decompose tall matrix (reduced mode)", () => {
      const A = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const [Q, R] = qr(A, "reduced");
      expect(Q.shape).toEqual([3, 2]);
      expect(R.shape).toEqual([2, 2]);
      expect(allclose(matmul(Q, R), A, 1e-10, 1e-10)).toBe(true);
    });

    it("should decompose tall matrix (complete mode)", () => {
      const A = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const [Q, R] = qr(A, "complete");
      expect(Q.shape).toEqual([3, 3]);
      expect(R.shape).toEqual([3, 2]);
      expect(allclose(matmul(Q, R), A, 1e-10, 1e-10)).toBe(true);
    });

    it("should decompose wide matrix", () => {
      const A = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const [Q, R] = qr(A);
      expect(Q.shape).toEqual([2, 2]);
      expect(R.shape).toEqual([2, 3]);
      expect(allclose(matmul(Q, R), A, 1e-10, 1e-10)).toBe(true);
    });

    it("handles singular matrix", () => {
      const A = tensor([
        [1, 1],
        [1, 1],
      ]);
      const [Q, R] = qr(A);
      expect(allclose(matmul(Q, R), A, 1e-10, 1e-10)).toBe(true);
    });

    it("should throw on non-finite values", () => {
      const A = tensor([
        [1, 2],
        [Infinity, 4],
      ]);
      expect(() => qr(A)).toThrow(/non-finite/i);
    });
  });

  describe("cholesky", () => {
    it("should decompose positive definite matrix", () => {
      const A = tensor([
        [4, 12, -16],
        [12, 37, -43],
        [-16, -43, 98],
      ]);
      const L = cholesky(A);
      expect(L.shape).toEqual([3, 3]);
      // L[0,0] = sqrt(4) = 2
      expect(Number(L.data[L.offset])).toBeCloseTo(2, 5);
      // LL^T = A
      expect(allclose(matmul(L, transpose(L)), A, 1e-10, 1e-10)).toBe(true);
    });

    it("should throw on non-symmetric matrix", () => {
      const A = tensor([
        [1, 2],
        [0, 1],
      ]);
      expect(() => cholesky(A)).toThrow(/symmetric/);
    });

    it("should throw on non-positive-definite matrix", () => {
      const A = tensor([
        [1, 2],
        [2, 1],
      ]);
      expect(() => cholesky(A)).toThrow();
    });
  });

  describe("lu", () => {
    it("should decompose square matrix", () => {
      const A = tensor([
        [2, 1, 1],
        [4, 3, 3],
        [8, 7, 9],
      ]);
      const [P, L, U] = lu(A);
      // PA = LU
      expect(allclose(matmul(P, A), matmul(L, U), 1e-10, 1e-10)).toBe(true);
    });

    it("handles zero pivot columns (rank-deficient)", () => {
      const A = tensor([
        [0, 1],
        [0, 2],
      ]);
      // Rank-deficient matrices are handled gracefully: P @ A = L @ U
      const [P, L, U] = lu(A);
      expect(allclose(matmul(P, A), matmul(L, U), 1e-10, 1e-10)).toBe(true);
    });

    it("handles already upper-triangular without swaps", () => {
      const A = tensor([
        [2, 1],
        [0, 3],
      ]);
      const [_P, L, U] = lu(A);
      expect(Number(L.data[L.offset])).toBeCloseTo(1, 6); // Diagonal 1
      expect(Number(U.data[U.offset])).toBeCloseTo(2, 6); // Diagonal 2
      expect(Number(U.data[U.offset + 3])).toBeCloseTo(3, 6);
    });

    it("handles zero pivot (rank-deficient)", () => {
      const A = tensor([
        [0, 1],
        [0, 1],
      ]);
      // Rank-deficient matrices are handled gracefully: P @ A = L @ U
      const [P, L, U] = lu(A);
      expect(allclose(matmul(P, A), matmul(L, U), 1e-10, 1e-10)).toBe(true);
    });
  });

  describe("eig", () => {
    it("should compute eigenvalues for diagonal matrix", () => {
      const D = tensor([
        [3, 0],
        [0, 5],
      ]);
      const [vals, _vecs] = eig(D);
      // Eigenvalues should be 3 and 5 (order may vary, but sorted usually)
      const vData = vals.data;
      if (!(vData instanceof Float64Array)) {
        throw new Error("Expected Float64Array");
      }
      const v = Array.from(vData).sort((a, b) => a - b);
      expect(Number(v[0])).toBeCloseTo(3, 5);
      expect(Number(v[1])).toBeCloseTo(5, 5);
    });

    it("throws ConvergenceError when maxIter is too small", () => {
      const A = tensor([
        [4, 1],
        [2, 3],
      ]);
      expect(() => eig(A, { maxIter: 1, tol: 1e-12 })).toThrow(ConvergenceError);
    });

    it("should throw on non-square matrix", () => {
      const A = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(() => eig(A)).toThrow();
    });

    it("should throw on non-finite values", () => {
      const A = tensor([
        [1, NaN],
        [2, 3],
      ]);
      expect(() => eig(A)).toThrow(/non-finite/i);
    });
  });

  describe("eigh", () => {
    it("should compute eigenvalues for symmetric matrix", () => {
      const A = tensor([
        [2, 1],
        [1, 2],
      ]);
      const [vals, _vecs] = eigh(A);
      // Eigenvalues of [[2,1],[1,2]] are 1 and 3
      const vData = vals.data;
      if (!(vData instanceof Float64Array)) {
        throw new Error("Expected Float64Array");
      }
      const v = Array.from(vData).sort((a, b) => a - b);
      expect(Number(v[0])).toBeCloseTo(1, 5);
      expect(Number(v[1])).toBeCloseTo(3, 5);
    });

    it("should throw on non-symmetric input", () => {
      const A = tensor([
        [1, 2],
        [0, 1],
      ]);
      expect(() => eigh(A)).toThrow(/symmetric/);
    });

    it("should throw on non-finite values", () => {
      const A = tensor([
        [1, 2],
        [Infinity, 3],
      ]);
      expect(() => eigh(A)).toThrow(/non-finite/i);
    });
  });
});
