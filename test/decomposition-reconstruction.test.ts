import { describe, expect, it } from "vitest";
import { cholesky, eig, eigh, lu, qr, svd } from "../src/linalg";
import { tensor, transpose } from "../src/ndarray";
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

/**
 * Helper to check if matrix is approximately identity
 */
function isApproxIdentity(A: ReturnType<typeof tensor>, tol: number = 1e-10): boolean {
  const n = A.shape[0];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const expected = i === j ? 1 : 0;
      const actual = Number(A.data[A.offset + i * n + j]);
      if (Math.abs(actual - expected) > tol) return false;
    }
  }
  return true;
}

describe("Decomposition Reconstruction Tests", () => {
  describe("SVD reconstruction: A = U * diag(s) * Vt", () => {
    it("should reconstruct 2x2 matrix", () => {
      const A = tensor([
        [1, 2],
        [3, 4],
      ]);
      const [U, s, Vt] = svd(A, true);

      const m = A.shape[0];
      const k = s.size;

      // Build U * diag(s)
      const US = new Float64Array(m * k);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < k; j++) {
          US[i * k + j] = Number(U.data[U.offset + i * k + j]) * Number(s.data[s.offset + j]);
        }
      }
      const UStensor = tensor(Array.from(US)).view([m, k]);
      const reconstructed = matmul(UStensor, Vt);

      expect(frobeniusDiff(A, reconstructed)).toBeLessThan(1e-6);
    });

    it("should reconstruct tall matrix (M > N)", () => {
      const A = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const [U, s, Vt] = svd(A, false);

      const m = A.shape[0];
      const k = s.size;

      const US = new Float64Array(m * k);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < k; j++) {
          US[i * k + j] = Number(U.data[U.offset + i * k + j]) * Number(s.data[s.offset + j]);
        }
      }
      const UStensor = tensor(Array.from(US)).view([m, k]);
      const reconstructed = matmul(UStensor, Vt);

      expect(frobeniusDiff(A, reconstructed)).toBeLessThan(1e-6);
    });

    it("should reconstruct wide matrix (M < N)", () => {
      const A = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const [U, s, Vt] = svd(A, false);

      const m = A.shape[0];
      const k = s.size;

      const US = new Float64Array(m * k);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < k; j++) {
          US[i * k + j] = Number(U.data[U.offset + i * k + j]) * Number(s.data[s.offset + j]);
        }
      }
      const UStensor = tensor(Array.from(US)).view([m, k]);
      const reconstructed = matmul(UStensor, Vt);

      expect(frobeniusDiff(A, reconstructed)).toBeLessThan(1e-6);
    });

    it("should have orthogonal U (U^T * U = I)", () => {
      const A = tensor([
        [1, 2],
        [3, 4],
      ]);
      const [U, _s, _Vt] = svd(A, true);
      const UtU = matmul(transpose(U), U);
      expect(isApproxIdentity(UtU)).toBe(true);
    });

    it("should have orthogonal V (Vt * Vt^T = I)", () => {
      const A = tensor([
        [1, 2],
        [3, 4],
      ]);
      const [_U, _s, Vt] = svd(A, true);
      const VtVtT = matmul(Vt, transpose(Vt));
      expect(isApproxIdentity(VtVtT)).toBe(true);
    });
  });

  describe("QR reconstruction: A = Q * R", () => {
    it("should reconstruct square matrix", () => {
      const A = tensor([
        [12, -51, 4],
        [6, 167, -68],
        [-4, 24, -41],
      ]);
      const [Q, R] = qr(A);
      const reconstructed = matmul(Q, R);
      expect(frobeniusDiff(A, reconstructed)).toBeLessThan(1e-6);
    });

    it("should reconstruct tall matrix", () => {
      const A = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const [Q, R] = qr(A);
      const reconstructed = matmul(Q, R);
      expect(frobeniusDiff(A, reconstructed)).toBeLessThan(1e-6);
    });

    it("should have orthogonal Q (Q^T * Q = I)", () => {
      const A = tensor([
        [12, -51, 4],
        [6, 167, -68],
        [-4, 24, -41],
      ]);
      const [Q, _R] = qr(A);
      const QtQ = matmul(transpose(Q), Q);
      expect(isApproxIdentity(QtQ)).toBe(true);
    });

    it("should have upper triangular R", () => {
      const A = tensor([
        [12, -51, 4],
        [6, 167, -68],
        [-4, 24, -41],
      ]);
      const [_Q, R] = qr(A);
      const n = R.shape[0];
      const m = R.shape[1];
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < Math.min(i, m); j++) {
          expect(Math.abs(Number(R.data[R.offset + i * m + j]))).toBeLessThan(1e-10);
        }
      }
    });

    it("should reconstruct with complete mode", () => {
      const A = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const [Q, R] = qr(A, "complete");
      const reconstructed = matmul(Q, R);
      expect(frobeniusDiff(A, reconstructed)).toBeLessThan(1e-6);
    });
  });

  describe("LU decomposition properties", () => {
    it("should satisfy P * A = L * U", () => {
      const A = tensor([
        [2, 1, 1],
        [4, 3, 3],
        [8, 7, 9],
      ]);
      const [P, L, U] = lu(A);
      // LU decomposition with partial pivoting: P * A = L * U
      const PA = matmul(P, A);
      const LU = matmul(L, U);
      expect(frobeniusDiff(PA, LU)).toBeLessThan(1e-10);
    });

    it("should have permutation matrix P (P * P^T = I)", () => {
      const A = tensor([
        [2, 1, 1],
        [4, 3, 3],
        [8, 7, 9],
      ]);
      const [P, _L, _U] = lu(A);
      const PPt = matmul(P, transpose(P));
      expect(isApproxIdentity(PPt)).toBe(true);
    });

    it("should return correct shapes", () => {
      const A = tensor([
        [2, 1, 1],
        [4, 3, 3],
        [8, 7, 9],
      ]);
      const [P, L, U] = lu(A);
      expect(P.shape).toEqual([3, 3]);
      expect(L.shape).toEqual([3, 3]);
      expect(U.shape).toEqual([3, 3]);
    });
  });

  describe("Cholesky reconstruction: A = L * L^T", () => {
    it("should reconstruct positive definite matrix", () => {
      const A = tensor([
        [4, 12, -16],
        [12, 37, -43],
        [-16, -43, 98],
      ]);
      const L = cholesky(A);
      const reconstructed = matmul(L, transpose(L));
      expect(frobeniusDiff(A, reconstructed)).toBeLessThan(1e-6);
    });

    it("should have lower triangular L", () => {
      const A = tensor([
        [4, 12, -16],
        [12, 37, -43],
        [-16, -43, 98],
      ]);
      const L = cholesky(A);
      const n = L.shape[0];
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          expect(Math.abs(Number(L.data[L.offset + i * n + j]))).toBeLessThan(1e-10);
        }
      }
    });

    it("should have positive diagonal", () => {
      const A = tensor([
        [4, 12, -16],
        [12, 37, -43],
        [-16, -43, 98],
      ]);
      const L = cholesky(A);
      const n = L.shape[0];
      for (let i = 0; i < n; i++) {
        expect(Number(L.data[L.offset + i * n + i])).toBeGreaterThan(0);
      }
    });

    it("should reconstruct identity matrix", () => {
      const I = tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
      const L = cholesky(I);
      const reconstructed = matmul(L, transpose(L));
      expect(frobeniusDiff(I, reconstructed)).toBeLessThan(1e-10);
    });
  });

  describe("Eigendecomposition", () => {
    it("should satisfy A * v = lambda * v for eig", () => {
      const A = tensor([
        [1, 2],
        [2, 1],
      ]);
      const [eigenvalues, eigenvectors] = eig(A);
      const n = A.shape[0];
      const evals = eigenvalues.data;
      const evecs = eigenvectors; // Columns are eigenvectors

      // Check for each eigenvalue/eigenvector pair
      for (let i = 0; i < n; i++) {
        const lambda = Number(evals[eigenvalues.offset + i]);

        // Extract eigenvector column i
        const v_data = new Float64Array(n);
        for (let row = 0; row < n; row++) {
          v_data[row] = Number(evecs.data[evecs.offset + row * n + i]);
        }
        const v = tensor(Array.from(v_data)).view([n, 1]);

        // Calculate A * v
        const Av = matmul(A, v);

        // Calculate lambda * v
        const lambda_v_data = new Float64Array(n);
        for (let k = 0; k < n; k++) lambda_v_data[k] = lambda * v_data[k];
        const lambda_v = tensor(Array.from(lambda_v_data)).view([n, 1]);

        expect(frobeniusDiff(Av, lambda_v)).toBeLessThan(1e-8);
      }
    });

    it("should satisfy A * v = lambda * v for eigh", () => {
      const A = tensor([
        [2, 1],
        [1, 2],
      ]);
      const [eigenvalues, eigenvectors] = eigh(A);
      const n = A.shape[0];
      const evals = eigenvalues.data;
      const evecs = eigenvectors; // Columns are eigenvectors

      for (let i = 0; i < n; i++) {
        const lambda = Number(evals[eigenvalues.offset + i]);

        // Extract eigenvector column i
        const v_data = new Float64Array(n);
        for (let row = 0; row < n; row++) {
          v_data[row] = Number(evecs.data[evecs.offset + row * n + i]);
        }
        const v = tensor(Array.from(v_data)).view([n, 1]);

        const Av = matmul(A, v);

        const lambda_v_data = new Float64Array(n);
        for (let k = 0; k < n; k++) lambda_v_data[k] = lambda * v_data[k];
        const lambda_v = tensor(Array.from(lambda_v_data)).view([n, 1]);

        expect(frobeniusDiff(Av, lambda_v)).toBeLessThan(1e-8);
      }
    });

    it("should have orthogonal eigenvectors for symmetric matrix (eigh)", () => {
      const A = tensor([
        [2, 1],
        [1, 2],
      ]);
      const [_eigenvalues, eigenvectors] = eigh(A);
      const VtV = matmul(transpose(eigenvectors), eigenvectors);
      expect(isApproxIdentity(VtV, 1e-8)).toBe(true);
    });
  });
});
