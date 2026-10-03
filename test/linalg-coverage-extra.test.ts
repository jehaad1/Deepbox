import { describe, expect, it } from "vitest";
import { cond, hessenberg, lyapunov, norm, solve_banded, sylvester } from "../src/linalg";
import { tensor } from "../src/ndarray";

const f64 = { dtype: "float64" as const };

// ---------------------------------------------------------------------------
// solve_banded coverage: multi-RHS, general banded (non-tridiagonal), errors
// ---------------------------------------------------------------------------
describe("solve_banded coverage", () => {
  it("should solve tridiagonal system (Thomas algorithm)", () => {
    // [2 -1 0; -1 2 -1; 0 -1 2] x = [1; 0; 1]
    const ab = tensor(
      [
        [0, -1, -1],
        [2, 2, 2],
        [-1, -1, 0],
      ],
      f64
    );
    const b = tensor([1, 0, 1], f64);
    const x = solve_banded([1, 1], ab, b);
    expect(x.shape).toEqual([3]);
    // Solution: x = [1, 1, 1]
    expect(Number(x.data[0])).toBeCloseTo(1, 6);
    expect(Number(x.data[1])).toBeCloseTo(1, 6);
    expect(Number(x.data[2])).toBeCloseTo(1, 6);
  });

  it("should solve tridiagonal with multiple RHS", () => {
    const ab = tensor(
      [
        [0, -1, -1],
        [2, 2, 2],
        [-1, -1, 0],
      ],
      f64
    );
    // b is (3, 2) with two right-hand sides
    const b = tensor(
      [
        [1, 2],
        [0, 0],
        [1, 2],
      ],
      f64
    );
    const x = solve_banded([1, 1], ab, b);
    expect(x.shape).toEqual([3, 2]);
  });

  it("should solve general banded system (l=2, u=1)", () => {
    // 4x4 matrix with l=2, u=1 (pentadiagonal-like)
    // Band storage: 4 rows (l+u+1=4), 4 cols
    // Row 0 (u=1 upper): [0, a01, a12, a23]
    // Row 1 (diagonal):  [a00, a11, a22, a33]
    // Row 2 (l=1 lower): [a10, a21, a32, 0]
    // Row 3 (l=2 lower): [a20, a31, 0, 0]
    const ab = tensor(
      [
        [0, 1, 0, 0],
        [4, 4, 4, 4],
        [1, 1, 1, 0],
        [0, 0, 0, 0],
      ],
      f64
    );
    const b = tensor([5, 6, 5, 4], f64);
    const x = solve_banded([2, 1], ab, b);
    expect(x.shape).toEqual([4]);
    // Verify Ax ≈ b by manual matrix-vector product
    const xArr = Array.from({ length: 4 }, (_, i) => Number(x.data[i]));
    const A = [
      [4, 1, 0, 0],
      [1, 4, 0, 0],
      [0, 1, 4, 0],
      [0, 0, 1, 4],
    ];
    for (let i = 0; i < 4; i++) {
      let row = 0;
      for (let j = 0; j < 4; j++) {
        row += ((A[i] as number[])[j] ?? 0) * (xArr[j] ?? 0);
      }
      expect(row).toBeCloseTo(Number(b.data[i]), 4);
    }
  });

  it("should solve general banded with multiple RHS", () => {
    const ab = tensor(
      [
        [0, 1, 0, 0],
        [4, 4, 4, 4],
        [1, 1, 1, 0],
        [0, 0, 0, 0],
      ],
      f64
    );
    const bmat = tensor(
      [
        [5, 10],
        [6, 12],
        [5, 10],
        [4, 8],
      ],
      f64
    );
    const x = solve_banded([2, 1], ab, bmat);
    expect(x.shape).toEqual([4, 2]);
  });

  it("should throw on invalid band widths", () => {
    const ab = tensor([[1]], f64);
    const b = tensor([1], f64);
    expect(() => solve_banded([-1, 0], ab, b)).toThrow();
  });

  it("should throw on dimension mismatch", () => {
    const ab = tensor(
      [
        [0, -1],
        [2, 2],
        [-1, 0],
      ],
      f64
    );
    const b = tensor([1, 2, 3], f64);
    expect(() => solve_banded([1, 1], ab, b)).toThrow();
  });

  it("should handle empty system", () => {
    const ab = tensor([] as number[], f64).reshape([1, 0]);
    const b = tensor([] as number[], f64);
    const x = solve_banded([0, 0], ab, b);
    expect(x.size).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// norm coverage: matrix norms (1, inf, nuc, fro), vector norms, axis, cond
// ---------------------------------------------------------------------------
describe("norm coverage", () => {
  it("should compute Frobenius norm of matrix", () => {
    const A = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      f64
    );
    const n = norm(A, "fro");
    expect(n).toBeCloseTo(Math.sqrt(1 + 4 + 9 + 16), 10);
  });

  it("should compute nuclear norm of matrix", () => {
    const A = tensor(
      [
        [1, 0],
        [0, 2],
      ],
      f64
    );
    const n = norm(A, "nuc");
    expect(n).toBeCloseTo(3, 6); // sum of singular values = 1 + 2
  });

  it("should compute 1-norm of matrix (max column sum)", () => {
    const A = tensor(
      [
        [1, -2],
        [3, 4],
      ],
      f64
    );
    const n = norm(A, 1);
    // Col sums: |1|+|3| = 4, |-2|+|4| = 6 → max = 6
    expect(n).toBeCloseTo(6, 10);
  });

  it("should compute inf-norm of matrix (max row sum)", () => {
    const A = tensor(
      [
        [1, -2],
        [3, 4],
      ],
      f64
    );
    const n = norm(A, Infinity);
    // Row sums: |1|+|-2| = 3, |3|+|4| = 7 → max = 7
    expect(n).toBeCloseTo(7, 10);
  });

  it("should compute vector 1-norm", () => {
    const v = tensor([1, -2, 3], f64);
    const n = norm(v, 1);
    expect(n).toBeCloseTo(6, 10);
  });

  it("should compute vector inf-norm", () => {
    const v = tensor([1, -5, 3], f64);
    const n = norm(v, Infinity);
    expect(n).toBeCloseTo(5, 10);
  });

  it("should compute vector 0-norm (count of nonzeros)", () => {
    const v = tensor([1, 0, 3, 0, 5], f64);
    const n = norm(v, 0);
    expect(n).toBeCloseTo(3, 10);
  });

  it("should compute vector -inf norm (minimum absolute value)", () => {
    const v = tensor([1, -2, 3], f64);
    const n = norm(v, -Infinity);
    expect(n).toBeCloseTo(1, 10);
  });

  it("should compute norm along axis", () => {
    const A = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      f64
    );
    // L2 norm along axis=1 (rows)
    const rowNorms = norm(A, 2, 1);
    expect(rowNorms).toBeDefined();
  });

  it("should compute norm along axis with keepdims", () => {
    const A = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      f64
    );
    const result = norm(A, 2, 1, true);
    expect(result).toBeDefined();
  });

  it("should throw nuc for vector", () => {
    const v = tensor([1, 2, 3], f64);
    expect(() => norm(v, "nuc")).toThrow();
  });

  it("should compute matrix -1 norm (min column sum)", () => {
    const A = tensor(
      [
        [1, -2],
        [3, 4],
      ],
      f64
    );
    const n = norm(A, -1);
    // Col sums: |1|+|3|=4, |-2|+|4|=6 → min = 4
    expect(n).toBeCloseTo(4, 10);
  });

  it("should compute matrix -inf norm (min row sum)", () => {
    const A = tensor(
      [
        [1, -2],
        [3, 4],
      ],
      f64
    );
    const n = norm(A, -Infinity);
    // Row sums: |1|+|-2|=3, |3|+|4|=7 → min = 3
    expect(n).toBeCloseTo(3, 10);
  });

  it("should compute matrix 2-norm (largest singular value)", () => {
    const A = tensor(
      [
        [1, 0],
        [0, 2],
      ],
      f64
    );
    const n = norm(A, 2);
    expect(n).toBeCloseTo(2, 6);
  });

  it("should compute matrix -2 norm (smallest singular value)", () => {
    const A = tensor(
      [
        [1, 0],
        [0, 2],
      ],
      f64
    );
    const n = norm(A, -2);
    expect(n).toBeCloseTo(1, 6);
  });

  it("should compute vector p-norm for arbitrary p", () => {
    const v = tensor([1, 2, 3], f64);
    const n = norm(v, 3);
    // (1^3 + 2^3 + 3^3)^(1/3) = (1+8+27)^(1/3) = 36^(1/3)
    expect(n).toBeCloseTo(36 ** (1 / 3), 6);
  });

  it("should compute scalar norm", () => {
    const s = tensor([5], f64).reshape([]);
    const n = norm(s);
    expect(n).toBeCloseTo(5, 10);
  });

  it("should compute norm with default ord for 2D (fro)", () => {
    const A = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      f64
    );
    const n = norm(A);
    expect(n).toBeCloseTo(Math.sqrt(30), 6);
  });

  it("should compute axis-based matrix norm with 2 axes (fro)", () => {
    const A = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      f64
    );
    const n = norm(A, "fro", [0, 1]);
    expect(n).toBeCloseTo(Math.sqrt(30), 6);
  });

  it("should compute axis-based matrix norm with 2 axes (1-norm)", () => {
    const A = tensor(
      [
        [1, -2],
        [3, 4],
      ],
      f64
    );
    const n = norm(A, 1, [0, 1]);
    expect(n).toBeCloseTo(6, 10);
  });

  it("should compute axis-based matrix norm with 2 axes (-1)", () => {
    const A = tensor(
      [
        [1, -2],
        [3, 4],
      ],
      f64
    );
    const n = norm(A, -1, [0, 1]);
    expect(n).toBeCloseTo(4, 10);
  });

  it("should compute axis-based matrix norm with 2 axes (inf)", () => {
    const A = tensor(
      [
        [1, -2],
        [3, 4],
      ],
      f64
    );
    const n = norm(A, Infinity, [0, 1]);
    expect(n).toBeCloseTo(7, 10);
  });

  it("should compute axis-based matrix norm with 2 axes (-inf)", () => {
    const A = tensor(
      [
        [1, -2],
        [3, 4],
      ],
      f64
    );
    const n = norm(A, -Infinity, [0, 1]);
    expect(n).toBeCloseTo(3, 10);
  });

  it("should compute axis-based matrix norm with 2 axes (2-norm)", () => {
    const A = tensor(
      [
        [1, 0],
        [0, 2],
      ],
      f64
    );
    const n = norm(A, 2, [0, 1]);
    expect(n).toBeCloseTo(2, 6);
  });

  it("should compute axis-based matrix norm with 2 axes (-2 norm)", () => {
    const A = tensor(
      [
        [1, 0],
        [0, 2],
      ],
      f64
    );
    const n = norm(A, -2, [0, 1]);
    expect(n).toBeCloseTo(1, 6);
  });

  it("should compute axis-based matrix norm with 2 axes and keepdims", () => {
    const A = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      f64
    );
    const result = norm(A, "fro", [0, 1], true);
    expect(result).toBeDefined();
  });

  it("should compute vector norm along axis=0", () => {
    const A = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      f64
    );
    const n = norm(A, 1, 0);
    expect(n).toBeDefined();
  });

  it("should compute vector norm along axis=0 with keepdims", () => {
    const A = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      f64
    );
    const n = norm(A, 1, 0, true);
    expect(n).toBeDefined();
  });

  it("should compute axis-based vector norms (0, inf, -inf, p)", () => {
    const A = tensor(
      [
        [1, 0],
        [3, 4],
      ],
      f64
    );
    expect(norm(A, 0, 1)).toBeDefined();
    expect(norm(A, Infinity, 1)).toBeDefined();
    expect(norm(A, -Infinity, 1)).toBeDefined();
    expect(norm(A, 3, 1)).toBeDefined();
  });
});

describe("cond coverage", () => {
  it("should compute condition number of well-conditioned matrix", () => {
    const A = tensor(
      [
        [1, 0],
        [0, 1],
      ],
      f64
    );
    const c = cond(A);
    expect(c).toBeCloseTo(1, 6);
  });

  it("should compute condition number of ill-conditioned matrix", () => {
    const A = tensor(
      [
        [1, 0],
        [0, 1e-10],
      ],
      f64
    );
    const c = cond(A);
    expect(c).toBeGreaterThan(1e9);
  });

  it("should compute Frobenius condition number", () => {
    const A = tensor(
      [
        [2, 0],
        [0, 1],
      ],
      f64
    );
    const c = cond(A, "fro");
    // cond_fro = ||A||_F * ||A^-1||_F = sqrt(4+1) * sqrt(1/4+1) = sqrt(5)*sqrt(5/4)
    expect(c).toBeGreaterThan(1);
  });

  it("should return Infinity for singular matrix", () => {
    const A = tensor(
      [
        [1, 0],
        [0, 0],
      ],
      f64
    );
    const c = cond(A);
    expect(c).toBe(Infinity);
  });
});

// ---------------------------------------------------------------------------
// sylvester/lyapunov coverage: larger matrices, 3x3 non-diagonal
// ---------------------------------------------------------------------------
describe("sylvester coverage", () => {
  it("should solve 3x3 non-diagonal Sylvester equation", () => {
    const A = tensor(
      [
        [1, 2, 0],
        [0, 3, 1],
        [0, 0, 5],
      ],
      f64
    );
    const B = tensor(
      [
        [2, 1, 0],
        [0, 4, 1],
        [0, 0, 6],
      ],
      f64
    );
    const C = tensor(
      [
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
      ],
      f64
    );
    const X = sylvester(A, B, C);
    expect(X.shape).toEqual([3, 3]);

    // Verify AX + XB ≈ C
    const n = 3;
    const a = toArr(A, n);
    const b = toArr(B, n);
    const c = toArr(C, n);
    const x = toArr(X, n);
    const ax = matmul(a, x, n);
    const xb = matmul(x, b, n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        expect((ax[i * n + j] ?? 0) + (xb[i * n + j] ?? 0)).toBeCloseTo(c[i * n + j] ?? 0, 4);
      }
    }
  });

  it("should solve with asymmetric A having complex eigenvalue structure", () => {
    // This matrix has complex eigenvalues → 2x2 blocks in Schur form
    const A = tensor(
      [
        [0, -1],
        [1, 0],
      ],
      f64
    ); // eigenvalues: ±i
    const B = tensor(
      [
        [0, -2],
        [2, 0],
      ],
      f64
    ); // eigenvalues: ±2i
    const C = tensor(
      [
        [1, 0],
        [0, 1],
      ],
      f64
    );
    const X = sylvester(A, B, C);
    expect(X.shape).toEqual([2, 2]);

    const a = toArr(A, 2);
    const b = toArr(B, 2);
    const c = toArr(C, 2);
    const x = toArr(X, 2);
    const ax = matmul(a, x, 2);
    const xb = matmul(x, b, 2);
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        expect((ax[i * 2 + j] ?? 0) + (xb[i * 2 + j] ?? 0)).toBeCloseTo(c[i * 2 + j] ?? 0, 4);
      }
    }
  });

  it("should solve Lyapunov with matrix having complex eigenvalues", () => {
    const A = tensor(
      [
        [-1, -2],
        [2, -1],
      ],
      f64
    ); // eigenvalues: -1 ± 2i (stable)
    const Q = tensor(
      [
        [5, 0],
        [0, 5],
      ],
      f64
    );
    const X = lyapunov(A, Q);
    expect(X.shape).toEqual([2, 2]);

    // Verify AX + XA^T ≈ Q
    const a = toArr(A, 2);
    const aT = [a[0] ?? 0, a[2] ?? 0, a[1] ?? 0, a[3] ?? 0];
    const q = toArr(Q, 2);
    const x = toArr(X, 2);
    const ax = matmul(a, x, 2);
    const xat = matmul(x, aT, 2);
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        expect((ax[i * 2 + j] ?? 0) + (xat[i * 2 + j] ?? 0)).toBeCloseTo(q[i * 2 + j] ?? 0, 4);
      }
    }
  });
});

// ---------------------------------------------------------------------------
// hessenberg coverage: larger matrix
// ---------------------------------------------------------------------------
describe("hessenberg coverage", () => {
  it("should decompose 5x5 matrix correctly", () => {
    const A = tensor(
      [
        [2, 1, 3, 0, 1],
        [1, 3, 1, 2, 0],
        [3, 1, 2, 1, 1],
        [0, 2, 1, 4, 2],
        [1, 0, 1, 2, 3],
      ],
      f64
    );
    const [H, Q] = hessenberg(A);
    expect(H.shape).toEqual([5, 5]);
    expect(Q.shape).toEqual([5, 5]);

    // Verify H is upper Hessenberg
    for (let i = 2; i < 5; i++) {
      for (let j = 0; j < i - 1; j++) {
        expect(Math.abs(Number(H.data[i * 5 + j]))).toBeLessThan(1e-10);
      }
    }

    // Verify Q * H * Q^T ≈ A
    const n = 5;
    const h = toArr(H, n);
    const q = toArr(Q, n);
    const qT = transposeArr(q, n);
    const qh = matmul(q, h, n);
    const qhqt = matmul(qh, qT, n);
    const aArr = toArr(A, n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        expect(qhqt[i * n + j] ?? 0).toBeCloseTo(aArr[i * n + j] ?? 0, 6);
      }
    }
  });

  it("should handle 0x0 matrix", () => {
    const A = tensor([] as number[], f64).reshape([0, 0]);
    const [H, Q] = hessenberg(A);
    expect(H.size).toBe(0);
    expect(Q.size).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
// biome-ignore lint/suspicious/noExplicitAny: test helper
function toArr(t: any, n: number): number[] {
  const arr: number[] = [];
  for (let i = 0; i < n * n; i++) {
    arr.push(Number(t.data[i] ?? 0));
  }
  return arr;
}

function matmul(a: number[], b: number[], n: number): number[] {
  const c = new Array(n * n).fill(0) as number[];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) {
        s += (a[i * n + k] ?? 0) * (b[k * n + j] ?? 0);
      }
      c[i * n + j] = s;
    }
  }
  return c;
}

function transposeArr(a: number[], n: number): number[] {
  const t = new Array(n * n).fill(0) as number[];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      t[j * n + i] = a[i * n + j] ?? 0;
    }
  }
  return t;
}
