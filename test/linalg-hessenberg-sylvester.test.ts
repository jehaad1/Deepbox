import { describe, expect, it } from "vitest";
import { hessenberg, lyapunov, sylvester } from "../src/linalg";
import { tensor } from "../src/ndarray";

const f64 = { dtype: "float64" as const };

// Helper: multiply two square matrices (n x n)
function matmul(A: number[][], B: number[][]): number[][] {
  const n = A.length;
  const m = B[0]?.length ?? 0;
  const k = B.length;
  const C: number[][] = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let p = 0; p < k; p++) {
        s += (A[i]?.[p] ?? 0) * (B[p]?.[j] ?? 0);
      }
      (C[i] as number[])[j] = s;
    }
  }
  return C;
}

// Helper: transpose
function transpose(A: number[][]): number[][] {
  const n = A.length;
  const m = A[0]?.length ?? 0;
  const T: number[][] = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      (T[j] as number[])[i] = A[i]?.[j] ?? 0;
    }
  }
  return T;
}

// Helper: tensor to 2D number array
// biome-ignore lint/suspicious/noExplicitAny: test helper
function toArray(t: any): number[][] {
  const rows = (t.shape[0] ?? 0) as number;
  const cols = (t.shape[1] ?? 0) as number;
  const result: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) {
      row.push(Number(t.data[i * cols + j]));
    }
    result.push(row);
  }
  return result;
}

describe("hessenberg", () => {
  it("should decompose a 3x3 matrix", () => {
    const A = tensor(
      [
        [2, 1, 3],
        [1, 3, 1],
        [3, 1, 2],
      ],
      f64
    );
    const [H, Q] = hessenberg(A);
    expect(H.shape).toEqual([3, 3]);
    expect(Q.shape).toEqual([3, 3]);

    // Verify Q * H * Q^T ≈ A
    const hArr = toArray(H);
    const qArr = toArray(Q);
    const qT = transpose(qArr);
    const qh = matmul(qArr, hArr);
    const qhqt = matmul(qh, qT);
    const aArr = toArray(A);

    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect((qhqt[i] as number[])[j]).toBeCloseTo((aArr[i] as number[])[j] ?? 0, 8);
      }
    }
  });

  it("should produce upper Hessenberg form (zeros below first sub-diagonal)", () => {
    const A = tensor(
      [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
      ],
      f64
    );
    const [H] = hessenberg(A);
    const hArr = toArray(H);

    // Check that entries below the first sub-diagonal are zero
    for (let i = 2; i < 4; i++) {
      for (let j = 0; j < i - 1; j++) {
        expect(Math.abs((hArr[i] as number[])[j] ?? 0)).toBeLessThan(1e-10);
      }
    }
  });

  it("should return identity Q for diagonal matrix", () => {
    const A = tensor(
      [
        [3, 0],
        [0, 5],
      ],
      f64
    );
    const [H, Q] = hessenberg(A);
    const hArr = toArray(H);
    const qArr = toArray(Q);

    // H should be close to A
    expect((hArr[0] as number[])[0]).toBeCloseTo(3, 10);
    expect((hArr[1] as number[])[1]).toBeCloseTo(5, 10);

    // Q should be close to identity
    expect((qArr[0] as number[])[0]).toBeCloseTo(1, 10);
    expect((qArr[1] as number[])[1]).toBeCloseTo(1, 10);
  });

  it("should handle 1x1 matrix", () => {
    const A = tensor([[7]], f64);
    const [H, Q] = hessenberg(A);
    expect(Number(H.data[0])).toBeCloseTo(7, 10);
    expect(Number(Q.data[0])).toBeCloseTo(1, 10);
  });

  it("should throw on non-square matrix", () => {
    const A = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      f64
    );
    expect(() => hessenberg(A)).toThrow();
  });

  it("should throw on non-2D input", () => {
    const A = tensor([1, 2, 3], f64);
    expect(() => hessenberg(A)).toThrow();
  });
});

describe("sylvester", () => {
  it("should solve AX + XB = C for diagonal A and B", () => {
    const A = tensor(
      [
        [1, 0],
        [0, 2],
      ],
      f64
    );
    const B = tensor(
      [
        [3, 0],
        [0, 4],
      ],
      f64
    );
    const C = tensor(
      [
        [4, 5],
        [6, 12],
      ],
      f64
    );
    const X = sylvester(A, B, C);
    expect(X.shape).toEqual([2, 2]);

    // Verify: A*X + X*B ≈ C
    const aArr = toArray(A);
    const bArr = toArray(B);
    const cArr = toArray(C);
    const xArr = toArray(X);

    const ax = matmul(aArr, xArr);
    const xb = matmul(xArr, bArr);

    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        const lhs = ((ax[i] as number[])[j] ?? 0) + ((xb[i] as number[])[j] ?? 0);
        expect(lhs).toBeCloseTo((cArr[i] as number[])[j] ?? 0, 6);
      }
    }
  });

  it("should solve for non-diagonal matrices", () => {
    const A = tensor(
      [
        [1, 2],
        [0, 3],
      ],
      f64
    );
    const B = tensor(
      [
        [4, 1],
        [0, 5],
      ],
      f64
    );
    const C = tensor(
      [
        [10, 15],
        [12, 24],
      ],
      f64
    );
    const X = sylvester(A, B, C);

    const aArr = toArray(A);
    const bArr = toArray(B);
    const cArr = toArray(C);
    const xArr = toArray(X);
    const ax = matmul(aArr, xArr);
    const xb = matmul(xArr, bArr);

    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        const lhs = ((ax[i] as number[])[j] ?? 0) + ((xb[i] as number[])[j] ?? 0);
        expect(lhs).toBeCloseTo((cArr[i] as number[])[j] ?? 0, 6);
      }
    }
  });

  it("should throw on shape mismatch", () => {
    const A = tensor(
      [
        [1, 0],
        [0, 2],
      ],
      f64
    );
    const B = tensor([[3]], f64);
    const C = tensor(
      [
        [4, 5],
        [6, 7],
      ],
      f64
    );
    expect(() => sylvester(A, B, C)).toThrow();
  });

  it("should throw on non-square A", () => {
    const A = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      f64
    );
    const B = tensor([[1]], f64);
    const C = tensor([[1], [2]], f64);
    expect(() => sylvester(A, B, C)).toThrow();
  });
});

describe("lyapunov", () => {
  it("should solve AX + XA^T = Q for stable diagonal A", () => {
    const A = tensor(
      [
        [-1, 0],
        [0, -2],
      ],
      f64
    );
    const Q = tensor(
      [
        [2, 0],
        [0, 8],
      ],
      f64
    );
    const X = lyapunov(A, Q);
    expect(X.shape).toEqual([2, 2]);

    // Verify: A*X + X*A^T ≈ Q
    const aArr = toArray(A);
    const aT = transpose(aArr);
    const qArr = toArray(Q);
    const xArr = toArray(X);
    const ax = matmul(aArr, xArr);
    const xat = matmul(xArr, aT);

    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        const lhs = ((ax[i] as number[])[j] ?? 0) + ((xat[i] as number[])[j] ?? 0);
        expect(lhs).toBeCloseTo((qArr[i] as number[])[j] ?? 0, 6);
      }
    }
  });

  it("should solve for non-diagonal A", () => {
    const A = tensor(
      [
        [-2, 1],
        [0, -3],
      ],
      f64
    );
    const Q = tensor(
      [
        [5, 1],
        [1, 9],
      ],
      f64
    );
    const X = lyapunov(A, Q);

    const aArr = toArray(A);
    const aT = transpose(aArr);
    const qArr = toArray(Q);
    const xArr = toArray(X);
    const ax = matmul(aArr, xArr);
    const xat = matmul(xArr, aT);

    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        const lhs = ((ax[i] as number[])[j] ?? 0) + ((xat[i] as number[])[j] ?? 0);
        expect(lhs).toBeCloseTo((qArr[i] as number[])[j] ?? 0, 6);
      }
    }
  });

  it("should throw on non-square A", () => {
    const A = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      f64
    );
    const Q = tensor(
      [
        [1, 2],
        [3, 4],
      ],
      f64
    );
    expect(() => lyapunov(A, Q)).toThrow();
  });

  it("should throw on dimension mismatch", () => {
    const A = tensor(
      [
        [1, 0],
        [0, 2],
      ],
      f64
    );
    const Q = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ],
      f64
    );
    expect(() => lyapunov(A, Q)).toThrow();
  });
});
