import { describe, expect, it } from "vitest";
import {
  denseToCSR,
  polar,
  schur,
  solve_banded,
  sparseCholeskySolve,
  sparseSolve,
} from "../src/linalg";
import { tensor } from "../src/ndarray";

// ─── Helper: matrix multiply ────────────────────────────────────────────────

function matmul(A: number[][], B: number[][]): number[][] {
  const m = A.length;
  const k = A[0]?.length ?? 0;
  const n = B[0]?.length ?? 0;
  const C: number[][] = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let p = 0; p < k; p++) {
        sum += (A[i]?.[p] ?? 0) * (B[p]?.[j] ?? 0);
      }
      (C[i] as number[])[j] = sum;
    }
  }
  return C;
}

function tensorToArray2D(t: ReturnType<typeof tensor>): number[][] {
  const rows = t.shape[0] ?? 0;
  const cols = t.shape[1] ?? 0;
  const result: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) {
      row.push(Number(t.data[t.offset + i * cols + j]));
    }
    result.push(row);
  }
  return result;
}

function transpose(A: number[][]): number[][] {
  const m = A.length;
  const n = A[0]?.length ?? 0;
  const T: number[][] = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      (T[j] as number[])[i] = A[i]?.[j] ?? 0;
    }
  }
  return T;
}

function frobNorm(A: number[][]): number {
  let sum = 0;
  for (const row of A) {
    for (const v of row) sum += v * v;
  }
  return Math.sqrt(sum);
}

// ─── polar() ────────────────────────────────────────────────────────────────

describe("polar decomposition", () => {
  it("decomposes a 2x2 matrix (right)", () => {
    const A = tensor([
      [1, 2],
      [3, 4],
    ]);
    const [U, P] = polar(A);

    expect(U.shape).toEqual([2, 2]);
    expect(P.shape).toEqual([2, 2]);

    // Verify U * P ≈ A
    const uArr = tensorToArray2D(U);
    const pArr = tensorToArray2D(P);
    const aArr = tensorToArray2D(A);
    const product = matmul(uArr, pArr);
    const diff = frobNorm(product.map((r, i) => r.map((v, j) => v - (aArr[i]?.[j] ?? 0))));
    expect(diff).toBeLessThan(1e-10);

    // Verify U is orthogonal: U^T * U ≈ I
    const utU = matmul(transpose(uArr), uArr);
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        const expected = i === j ? 1 : 0;
        expect(Math.abs((utU[i]?.[j] ?? 0) - expected)).toBeLessThan(1e-10);
      }
    }

    // Verify P is symmetric: P ≈ P^T
    const pT = transpose(pArr);
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        expect(Math.abs((pArr[i]?.[j] ?? 0) - (pT[i]?.[j] ?? 0))).toBeLessThan(1e-10);
      }
    }
  });

  it("decomposes a 3x3 matrix (right)", () => {
    const A = tensor([
      [2, -1, 0],
      [-1, 2, -1],
      [0, -1, 2],
    ]);
    const [U, P] = polar(A);

    expect(U.shape).toEqual([3, 3]);
    expect(P.shape).toEqual([3, 3]);

    // Verify U * P ≈ A
    const product = matmul(tensorToArray2D(U), tensorToArray2D(P));
    const aArr = tensorToArray2D(A);
    const diff = frobNorm(product.map((r, i) => r.map((v, j) => v - (aArr[i]?.[j] ?? 0))));
    expect(diff).toBeLessThan(1e-10);
  });

  it("decomposes with side='left'", () => {
    const A = tensor([
      [1, 2],
      [3, 4],
    ]);
    const [S, U] = polar(A, "left");

    expect(S.shape).toEqual([2, 2]);
    expect(U.shape).toEqual([2, 2]);

    // Verify S * U ≈ A
    const product = matmul(tensorToArray2D(S), tensorToArray2D(U));
    const aArr = tensorToArray2D(A);
    const diff = frobNorm(product.map((r, i) => r.map((v, j) => v - (aArr[i]?.[j] ?? 0))));
    expect(diff).toBeLessThan(1e-10);
  });

  it("throws for non-2D input", () => {
    expect(() => polar(tensor([1, 2, 3]))).toThrow();
  });
});

// ─── schur() ────────────────────────────────────────────────────────────────

describe("Schur decomposition", () => {
  it("decomposes a 2x2 matrix", () => {
    const A = tensor([
      [1, 2],
      [0, 3],
    ]);
    const [T, Q] = schur(A);

    expect(T.shape).toEqual([2, 2]);
    expect(Q.shape).toEqual([2, 2]);

    // Verify Q * T * Q^T ≈ A
    const qArr = tensorToArray2D(Q);
    const tArr = tensorToArray2D(T);
    const aArr = tensorToArray2D(A);
    const product = matmul(matmul(qArr, tArr), transpose(qArr));
    const diff = frobNorm(product.map((r, i) => r.map((v, j) => v - (aArr[i]?.[j] ?? 0))));
    expect(diff).toBeLessThan(1e-10);

    // Verify Q is orthogonal
    const qtQ = matmul(transpose(qArr), qArr);
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        const expected = i === j ? 1 : 0;
        expect(Math.abs((qtQ[i]?.[j] ?? 0) - expected)).toBeLessThan(1e-10);
      }
    }
  });

  it("decomposes a 3x3 symmetric matrix", () => {
    const A = tensor([
      [4, 1, 0],
      [1, 3, 1],
      [0, 1, 2],
    ]);
    const [T, Q] = schur(A);

    // Verify Q * T * Q^T ≈ A
    const product = matmul(
      matmul(tensorToArray2D(Q), tensorToArray2D(T)),
      transpose(tensorToArray2D(Q))
    );
    const aArr = tensorToArray2D(A);
    const diff = frobNorm(product.map((r, i) => r.map((v, j) => v - (aArr[i]?.[j] ?? 0))));
    expect(diff).toBeLessThan(1e-10);

    // For symmetric matrices, T should be approximately diagonal
    // (real eigenvalues on diagonal)
    const tArr = tensorToArray2D(T);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < i; j++) {
        expect(Math.abs(tArr[i]?.[j] ?? 0)).toBeLessThan(1e-10);
      }
    }
  });

  it("decomposes a 4x4 matrix", () => {
    const A = tensor([
      [1, 2, 0, 1],
      [0, 3, 1, 0],
      [1, 0, 2, 1],
      [0, 1, 0, 4],
    ]);
    const [T, Q] = schur(A);

    expect(T.shape).toEqual([4, 4]);
    expect(Q.shape).toEqual([4, 4]);

    // Verify Q * T * Q^T ≈ A
    const product = matmul(
      matmul(tensorToArray2D(Q), tensorToArray2D(T)),
      transpose(tensorToArray2D(Q))
    );
    const aArr = tensorToArray2D(A);
    const diff = frobNorm(product.map((r, i) => r.map((v, j) => v - (aArr[i]?.[j] ?? 0))));
    expect(diff).toBeLessThan(1e-8);
  });

  it("handles 1x1 matrix", () => {
    const [T, Q] = schur(tensor([[5]]));
    expect(Number(T.data[0])).toBeCloseTo(5);
    expect(Number(Q.data[0])).toBeCloseTo(1);
  });

  it("throws for non-square matrix", () => {
    expect(() =>
      schur(
        tensor([
          [1, 2, 3],
          [4, 5, 6],
        ])
      )
    ).toThrow();
  });

  it("throws for non-2D input", () => {
    expect(() => schur(tensor([1, 2, 3]))).toThrow();
  });
});

// ─── solve_banded() ─────────────────────────────────────────────────────────

describe("solve_banded", () => {
  it("solves a tridiagonal system", () => {
    // [2 -1 0; -1 2 -1; 0 -1 2] x = [1; 0; 1]
    // Band storage (l=1, u=1):
    //   row 0 (upper): [0, -1, -1]
    //   row 1 (diag):  [2,  2,  2]
    //   row 2 (lower): [-1, -1,  0]
    const ab = tensor([
      [0, -1, -1],
      [2, 2, 2],
      [-1, -1, 0],
    ]);
    const b = tensor([1, 0, 1]);
    const x = solve_banded([1, 1], ab, b);

    expect(x.shape).toEqual([3]);

    // Verify: A * x ≈ b
    const xArr = [Number(x.data[0]), Number(x.data[1]), Number(x.data[2])];
    const Ax = [
      2 * xArr[0]! - xArr[1]!,
      -xArr[0]! + 2 * xArr[1]! - xArr[2]!,
      -xArr[1]! + 2 * xArr[2]!,
    ];
    expect(Math.abs(Ax[0]! - 1)).toBeLessThan(1e-10);
    expect(Math.abs(Ax[1]! - 0)).toBeLessThan(1e-10);
    expect(Math.abs(Ax[2]! - 1)).toBeLessThan(1e-10);
  });

  it("solves a pentadiagonal system", () => {
    // 5x5 pentadiagonal (l=2, u=2)
    // [3 1 1 0 0; 1 3 1 1 0; 1 1 3 1 1; 0 1 1 3 1; 0 0 1 1 3]
    const ab = tensor([
      [0, 0, 1, 1, 1], // u=2
      [0, 1, 1, 1, 1], // u=1
      [3, 3, 3, 3, 3], // diag
      [1, 1, 1, 1, 0], // l=1
      [1, 1, 1, 0, 0], // l=2
    ]);
    const b = tensor([1, 2, 3, 2, 1]);
    const x = solve_banded([2, 2], ab, b);

    expect(x.shape).toEqual([5]);

    // Solution should be finite
    for (let i = 0; i < 5; i++) {
      expect(Number.isFinite(Number(x.data[i]))).toBe(true);
    }
  });

  it("throws for invalid band widths", () => {
    expect(() => solve_banded([-1, 1], tensor([[1], [2]]), tensor([1]))).toThrow();
  });

  it("throws for wrong ab shape", () => {
    expect(() =>
      solve_banded(
        [1, 1],
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([1, 2])
      )
    ).toThrow();
  });
});

// ─── Sparse solvers ─────────────────────────────────────────────────────────

describe("sparse solvers", () => {
  describe("denseToCSR", () => {
    it("converts dense matrix to CSR", () => {
      const A = tensor([
        [4, 1, 0],
        [1, 3, 1],
        [0, 1, 2],
      ]);
      const csr = denseToCSR(A);
      expect(csr.n).toBe(3);
      // 7 non-zero entries
      expect(csr.values.length).toBe(7);
      expect(csr.colIndices.length).toBe(7);
      expect(csr.rowPointers.length).toBe(4);
    });

    it("throws for non-square matrix", () => {
      expect(() =>
        denseToCSR(
          tensor([
            [1, 2, 3],
            [4, 5, 6],
          ])
        )
      ).toThrow();
    });

    it("throws for non-2D input", () => {
      expect(() => denseToCSR(tensor([1, 2, 3]))).toThrow();
    });
  });

  describe("sparseSolve (LU)", () => {
    it("solves a simple SPD system", () => {
      const A = tensor([
        [4, 1, 0],
        [1, 3, 1],
        [0, 1, 2],
      ]);
      const b = tensor([1, 2, 3]);
      const csr = denseToCSR(A);
      const x = sparseSolve(csr, b);

      expect(x.shape).toEqual([3]);

      // Verify A * x ≈ b
      const xArr = Array.from({ length: 3 }, (_, i) => Number(x.data[i]));
      const Ax = [
        4 * xArr[0]! + 1 * xArr[1]!,
        1 * xArr[0]! + 3 * xArr[1]! + 1 * xArr[2]!,
        1 * xArr[1]! + 2 * xArr[2]!,
      ];
      expect(Math.abs(Ax[0]! - 1)).toBeLessThan(1e-10);
      expect(Math.abs(Ax[1]! - 2)).toBeLessThan(1e-10);
      expect(Math.abs(Ax[2]! - 3)).toBeLessThan(1e-10);
    });

    it("solves a non-symmetric system", () => {
      const A = tensor([
        [3, 1, 0],
        [0, 2, 1],
        [1, 0, 4],
      ]);
      const b = tensor([4, 3, 5]);
      const csr = denseToCSR(A);
      const x = sparseSolve(csr, b);

      // Verify A * x ≈ b
      const xArr = Array.from({ length: 3 }, (_, i) => Number(x.data[i]));
      const Ax = [
        3 * xArr[0]! + 1 * xArr[1]!,
        2 * xArr[1]! + 1 * xArr[2]!,
        1 * xArr[0]! + 4 * xArr[2]!,
      ];
      expect(Math.abs(Ax[0]! - 4)).toBeLessThan(1e-10);
      expect(Math.abs(Ax[1]! - 3)).toBeLessThan(1e-10);
      expect(Math.abs(Ax[2]! - 5)).toBeLessThan(1e-10);
    });

    it("throws for non-1D b", () => {
      const csr = denseToCSR(
        tensor([
          [1, 0],
          [0, 1],
        ])
      );
      expect(() => sparseSolve(csr, tensor([[1], [2]]))).toThrow();
    });
  });

  describe("sparseCholeskySolve", () => {
    it("solves a symmetric positive-definite system", () => {
      const A = tensor([
        [4, 2, 0],
        [2, 5, 1],
        [0, 1, 3],
      ]);
      const b = tensor([1, 2, 3]);
      const csr = denseToCSR(A);
      const x = sparseCholeskySolve(csr, b);

      expect(x.shape).toEqual([3]);

      // Verify A * x ≈ b
      const xArr = Array.from({ length: 3 }, (_, i) => Number(x.data[i]));
      const Ax = [
        4 * xArr[0]! + 2 * xArr[1]!,
        2 * xArr[0]! + 5 * xArr[1]! + 1 * xArr[2]!,
        1 * xArr[1]! + 3 * xArr[2]!,
      ];
      expect(Math.abs(Ax[0]! - 1)).toBeLessThan(1e-10);
      expect(Math.abs(Ax[1]! - 2)).toBeLessThan(1e-10);
      expect(Math.abs(Ax[2]! - 3)).toBeLessThan(1e-10);
    });

    it("throws for non-positive-definite matrix", () => {
      // Indefinite matrix
      const A = tensor([
        [1, 0],
        [0, -1],
      ]);
      const b = tensor([1, 1]);
      const csr = denseToCSR(A);
      expect(() => sparseCholeskySolve(csr, b)).toThrow(/not positive definite/);
    });

    it("throws for mismatched dimensions", () => {
      const csr = denseToCSR(
        tensor([
          [1, 0],
          [0, 1],
        ])
      );
      expect(() => sparseCholeskySolve(csr, tensor([1, 2, 3]))).toThrow();
    });
  });
});
