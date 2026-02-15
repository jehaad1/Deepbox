import { describe, expect, it } from "vitest";
import {
  cholesky,
  cond,
  det,
  eig,
  eigh,
  eigvals,
  eigvalsh,
  inv,
  lu,
  matrixRank,
  norm,
  pinv,
  qr,
  slogdet,
  solve,
  svd,
  trace,
} from "../src/linalg";
import { dot, tensor } from "../src/ndarray";

describe("consumer API: linalg", () => {
  const A = tensor([
    [1, 0, 0],
    [0, 2, 0],
    [0, 0, 3],
  ]);
  const rect = tensor([
    [1, 2],
    [3, 4],
    [5, 6],
  ]);
  const sym = tensor([
    [4, 2],
    [2, 3],
  ]);

  describe("SVD", () => {
    it("full SVD", () => {
      const [U, S, Vt] = svd(A);
      expect(U.shape).toEqual([3, 3]);
      expect(S.size).toBe(3);
      expect(Vt.shape).toEqual([3, 3]);
      expect(S.at(0)).toBe(3);
      expect(S.at(1)).toBe(2);
      expect(S.at(2)).toBe(1);
    });

    it("reduced SVD", () => {
      const [Ur, Sr] = svd(rect, false);
      expect(Ur.shape).toEqual([3, 2]);
      expect(Sr.size).toBe(2);
    });
  });

  describe("QR", () => {
    it("QR decomposition", () => {
      const [Q, R] = qr(A);
      expect(Q.shape[0]).toBe(3);
      expect(R.shape[0]).toBe(3);
      expect(Math.abs(Number(R.at(1, 0)))).toBeLessThan(1e-6);
    });

    it("complete QR", () => {
      const [Qc] = qr(rect, "complete");
      expect(Qc.shape).toEqual([3, 3]);
    });
  });

  describe("LU", () => {
    it("LU decomposition", () => {
      const [P, L, Up] = lu(A);
      expect(P.shape[0]).toBe(3);
      expect(L.shape[0]).toBe(3);
      expect(Up.shape[0]).toBe(3);
      expect(Math.abs(Number(L.at(0, 0)) - 1)).toBeLessThan(1e-6);
    });
  });

  describe("Cholesky", () => {
    it("lower triangular decomposition", () => {
      const Lc = cholesky(sym);
      expect(Lc.shape).toEqual([2, 2]);
      expect(Math.abs(Number(Lc.at(0, 1)))).toBeLessThan(1e-6);
    });
  });

  describe("Eigendecomposition", () => {
    const eigM = tensor([
      [2, 1],
      [1, 2],
    ]);

    it("eig", () => {
      const [eigenvalues, eigenvectors] = eig(eigM);
      expect(eigenvalues.size).toBe(2);
      expect(eigenvectors.shape).toEqual([2, 2]);
    });

    it("eigvals", () => {
      expect(eigvals(eigM).size).toBe(2);
    });

    it("eigh / eigvalsh (symmetric)", () => {
      const [vals, vecs] = eigh(sym);
      expect(vals.size).toBe(2);
      expect(vecs.shape).toEqual([2, 2]);
      expect(eigvalsh(sym).size).toBe(2);
    });
  });

  describe("inv / pinv", () => {
    it("inv produces inverse", () => {
      const M = tensor([
        [1, 2],
        [3, 4],
      ]);
      const invA = inv(M);
      expect(invA.shape).toEqual([2, 2]);
      const product = dot(M, invA);
      expect(Math.abs(Number(product.at(0, 0)) - 1)).toBeLessThan(1e-4);
      expect(Math.abs(Number(product.at(0, 1)))).toBeLessThan(1e-4);
    });

    it("pinv shape", () => {
      const p = pinv(rect);
      expect(p.shape).toEqual([2, 3]);
    });
  });

  describe("det / slogdet / trace / matrixRank / cond", () => {
    it("det", () => {
      expect(
        Math.abs(
          det(
            tensor([
              [1, 2],
              [3, 4],
            ])
          ) - -2
        )
      ).toBeLessThan(0.01);
      expect(
        Math.abs(
          det(
            tensor([
              [1, 0],
              [0, 1],
            ])
          ) - 1
        )
      ).toBeLessThan(1e-6);
    });

    it("slogdet", () => {
      const [signT, logabsdetT] = slogdet(
        tensor([
          [1, 2],
          [3, 4],
        ])
      );
      expect(signT.size).toBe(1);
      expect(logabsdetT.size).toBe(1);
    });

    it("trace", () => {
      expect(trace(A).at(0)).toBe(6);
      expect(
        trace(
          tensor([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ]),
          1
        ).at(0)
      ).toBe(8);
    });

    it("matrixRank", () => {
      expect(matrixRank(A)).toBe(3);
      expect(
        matrixRank(
          tensor([
            [1, 2],
            [2, 4],
          ])
        )
      ).toBe(1);
    });

    it("cond", () => {
      expect(cond(A)).toBeGreaterThanOrEqual(1);
      expect(
        Math.abs(
          cond(
            tensor([
              [1, 0],
              [0, 1],
            ])
          ) - 1
        )
      ).toBeLessThan(1e-6);
    });
  });

  describe("norm", () => {
    it("vector norm", () => {
      const n = norm(tensor([3, 4]));
      const val = typeof n === "number" ? n : Number(n.at());
      expect(Math.abs(val - 5)).toBeLessThan(0.01);
    });
  });

  describe("solve", () => {
    it("diagonal system", () => {
      const x = solve(A, tensor([1, 2, 3]));
      expect(x.size).toBe(3);
      expect(Math.abs(Number(x.at(0)) - 1)).toBeLessThan(1e-4);
      expect(Math.abs(Number(x.at(1)) - 1)).toBeLessThan(1e-4);
      expect(Math.abs(Number(x.at(2)) - 1)).toBeLessThan(1e-4);
    });

    it("2x2 system", () => {
      const x = solve(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([5, 6])
      );
      expect(x.size).toBe(2);
    });
  });
});
