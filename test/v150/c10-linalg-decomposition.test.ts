/**
 * Regression tests for the linalg decompositions (v1.5.0 review).
 *
 * Reference values come from NumPy 2.4 / SciPy 1.17 (`numpy.linalg.qr`, `numpy.linalg.eigvals`,
 * `scipy.linalg.hessenberg`, `scipy.linalg.polar`, `scipy.linalg.schur`, `scipy.linalg.lu`).
 */
import { describe, expect, it } from "vitest";
import { ConvergenceError, DataValidationError, InvalidParameterError } from "../../src/core";
import {
  cholesky,
  eig,
  eigh,
  eigvals,
  eigvalsh,
  hessenberg,
  lu,
  pinv,
  polar,
  qr,
  schur,
  svd,
  svdvals,
} from "../../src/linalg";
import type { Tensor } from "../../src/ndarray";
import { tensor } from "../../src/ndarray";

type Mat = number[][];

/** Float64 tensor from nested rows. */
function mat(rows: Mat): Tensor {
  return tensor(rows, { dtype: "float64" });
}

/** Dense row-major copy of a 2D tensor as nested arrays. */
function toRows(t: Tensor): Mat {
  const [r = 0, c = 0] = t.shape;
  const data = t.data as Float64Array;
  const out: Mat = [];
  for (let i = 0; i < r; i++) {
    const row: number[] = [];
    for (let j = 0; j < c; j++) row.push(data[t.offset + i * c + j] as number);
    out.push(row);
  }
  return out;
}

function toArray(t: Tensor): number[] {
  return Array.from(t.data as Float64Array);
}

function matmulRows(a: Mat, b: Mat): Mat {
  const m = a.length;
  const k = b.length;
  const n = b[0]?.length ?? 0;
  const out: Mat = [];
  for (let i = 0; i < m; i++) {
    const row = new Array<number>(n).fill(0);
    for (let p = 0; p < k; p++) {
      const aip = (a[i] as number[])[p] as number;
      for (let j = 0; j < n; j++) {
        row[j] = (row[j] as number) + aip * ((b[p] as number[])[j] as number);
      }
    }
    out.push(row);
  }
  return out;
}

function transposeRows(a: Mat): Mat {
  const m = a.length;
  const n = a[0]?.length ?? 0;
  const out: Mat = [];
  for (let j = 0; j < n; j++) {
    const row: number[] = [];
    for (let i = 0; i < m; i++) row.push((a[i] as number[])[j] as number);
    out.push(row);
  }
  return out;
}

function maxAbsDiff(a: Mat, b: Mat): number {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < (a[i] as number[]).length; j++) {
      worst = Math.max(
        worst,
        Math.abs(((a[i] as number[])[j] as number) - ((b[i] as number[])[j] as number))
      );
    }
  }
  return worst;
}

function maxAbs(a: Mat): number {
  let worst = 0;
  for (const row of a) for (const v of row) worst = Math.max(worst, Math.abs(v));
  return worst;
}

function identity(n: number): Mat {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
  );
}

function scaled(a: Mat, f: number): Mat {
  return a.map((row) => row.map((v) => v * f));
}

/** Deterministic pseudo-random matrix in [-0.5, 0.5). */
function randomMat(m: number, n: number, seed: number): Mat {
  let s = seed;
  const next = (): number => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296 - 0.5;
  };
  return Array.from({ length: m }, () => Array.from({ length: n }, next));
}

describe("hessenberg", () => {
  it("matches scipy.linalg.hessenberg for a 3x3 matrix", () => {
    const A = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 10],
    ];
    const [H, Q] = hessenberg(mat(A));
    const Hs = toRows(H);
    const Qs = toRows(Q);
    const refH = [
      [1.0, -3.597007303087045, -0.24806946917841663],
      [-8.06225774829855, 14.8, 2.3999999999999995],
      [0.0, 0.3999999999999978, 0.19999999999999837],
    ];
    const refQ = [
      [1, 0, 0],
      [0, -0.49613893835683376, -0.8682431421244592],
      [0, -0.8682431421244592, 0.4961389383568338],
    ];
    expect(maxAbsDiff(Hs, refH)).toBeLessThan(1e-12);
    expect(maxAbsDiff(Qs, refQ)).toBeLessThan(1e-12);
  });

  it("reduces matrices whose entries are all tiny (no absolute thresholds)", () => {
    const base = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 10],
    ];
    for (const f of [1e-20, 1e-100, 1e150]) {
      const A = scaled(base, f);
      const [H, Q] = hessenberg(mat(A));
      const Hs = toRows(H);
      const Qs = toRows(Q);
      expect(Hs[2]?.[0]).toBe(0);
      // Q is orthogonal and Q H Q^T reproduces A.
      expect(maxAbsDiff(matmulRows(transposeRows(Qs), Qs), identity(3))).toBeLessThan(1e-14);
      const rec = matmulRows(matmulRows(Qs, Hs), transposeRows(Qs));
      expect(maxAbsDiff(rec, A) / maxAbs(A)).toBeLessThan(1e-14);
      // The scaled problem has the same Hessenberg form up to the scale.
      expect(Math.abs((Hs[1] as number[])[0] as number) / f).toBeCloseTo(8.06225774829855, 10);
    }
  });

  it("keeps an already Hessenberg matrix unchanged", () => {
    const A = [
      [1, 2, 3],
      [4, 5, 6],
      [0, 7, 8],
    ];
    const [H, Q] = hessenberg(mat(A));
    expect(maxAbsDiff(toRows(H), A)).toBe(0);
    expect(maxAbsDiff(toRows(Q), identity(3))).toBe(0);
  });
});

describe("qr", () => {
  it("uses the LAPACK/NumPy sign convention (3x3)", () => {
    const [Q, R] = qr(
      mat([
        [12, -51, 4],
        [6, 167, -68],
        [-4, 24, -41],
      ])
    );
    const refQ = [
      [-0.8571428571428572, 0.3942857142857143, 0.33142857142857146],
      [-0.4285714285714286, -0.9028571428571428, -0.034285714285714246],
      [0.28571428571428575, -0.1714285714285714, 0.9428571428571428],
    ];
    const refR = [
      [-14, -21, 14],
      [0, -175, 70],
      [0, 0, -35],
    ];
    expect(maxAbsDiff(toRows(Q), refQ)).toBeLessThan(1e-12);
    expect(maxAbsDiff(toRows(R), refR)).toBeLessThan(1e-12);
  });

  it("does not flip signs of an identity matrix", () => {
    const [Q, R] = qr(mat(identity(3)));
    expect(maxAbsDiff(toRows(Q), identity(3))).toBe(0);
    expect(maxAbsDiff(toRows(R), identity(3))).toBe(0);
  });

  it("matches numpy for a tall matrix in reduced and complete mode", () => {
    const A = [
      [2, -1, 0],
      [1, 3, 4],
      [0, 5, 1],
      [2, 2, 2],
    ];
    const refR = [
      [-3, -1.666666666666667, -2.666666666666667],
      [0, -6.018490028422596, -2.7507822522545],
      [0, 0, 2.514375844930617],
    ];
    const refQ = [
      [-0.6666666666666667, 0.3507708912270839, -0.32329432261359214],
      [-0.33333333333333337, -0.40615576878925513, 0.792986074335226],
      [0, -0.830773163432567, -0.5111710233022456],
      [-0.6666666666666667, -0.1476930068324563, -0.07319871455402092],
    ];
    const [Q, R] = qr(mat(A), "reduced");
    expect(Q.shape).toEqual([4, 3]);
    expect(R.shape).toEqual([3, 3]);
    expect(maxAbsDiff(toRows(Q), refQ)).toBeLessThan(1e-12);
    expect(maxAbsDiff(toRows(R), refR)).toBeLessThan(1e-12);

    const [Qc, Rc] = qr(mat(A), "complete");
    expect(Qc.shape).toEqual([4, 4]);
    expect(Rc.shape).toEqual([4, 3]);
    const lastCol = toRows(Qc).map((row) => row[3] as number);
    expect(lastCol[0]).toBeCloseTo(-0.5727094537277445, 12);
    expect(lastCol[3]).toBeCloseTo(0.7269004605005986, 12);
    expect(maxAbsDiff(matmulRows(transposeRows(toRows(Qc)), toRows(Qc)), identity(4))).toBeLessThan(
      1e-14
    );
    expect(maxAbsDiff(matmulRows(toRows(Qc), toRows(Rc)), A)).toBeLessThan(1e-13);
  });

  it("factors matrices with tiny entries (no skipped reflections)", () => {
    const base = [
      [12, -51, 4],
      [6, 167, -68],
      [-4, 24, -41],
    ];
    for (const f of [1e-20, 1e-150]) {
      const A = scaled(base, f);
      const [Q, R] = qr(mat(A));
      const Rs = toRows(R);
      const Qs = toRows(Q);
      for (let i = 1; i < 3; i++) {
        for (let j = 0; j < i; j++) expect(Rs[i]?.[j]).toBe(0);
      }
      expect(Rs[0]?.[0] as number).toBeCloseTo(-14 * f, -Math.round(Math.log10(f)) + 12);
      expect(maxAbsDiff(matmulRows(Qs, Rs), A) / maxAbs(A)).toBeLessThan(1e-14);
      expect(maxAbsDiff(matmulRows(transposeRows(Qs), Qs), identity(3))).toBeLessThan(1e-14);
    }
  });

  it("does not overflow for entries near 1e200", () => {
    const A = [
      [1e200, 2e200],
      [3e200, 4e200],
    ];
    const [Q, R] = qr(mat(A));
    const Rs = toRows(R);
    expect(Number.isFinite(Rs[0]?.[0])).toBe(true);
    expect(Rs[1]?.[0]).toBe(0);
    expect(maxAbsDiff(matmulRows(toRows(Q), Rs), A) / 1e200).toBeLessThan(1e-14);
    // 3.1622776601683795e200 = sqrt(10) * 1e200
    expect(Math.abs((Rs[0] as number[])[0] as number) / 1e200).toBeCloseTo(Math.sqrt(10), 12);
  });

  it("builds only K columns of Q for a tall matrix in reduced mode", () => {
    const A = randomMat(500, 4, 11);
    const [Q, R] = qr(mat(A));
    expect(Q.shape).toEqual([500, 4]);
    expect(R.shape).toEqual([4, 4]);
    expect(maxAbsDiff(matmulRows(transposeRows(toRows(Q)), toRows(Q)), identity(4))).toBeLessThan(
      1e-14
    );
    expect(maxAbsDiff(matmulRows(toRows(Q), toRows(R)), A)).toBeLessThan(1e-13);
  });

  it("handles rank-deficient and zero columns with an upper triangular R", () => {
    const A = [
      [1, 0, 2],
      [2, 0, 4],
      [3, 0, 6],
    ];
    const [Q, R] = qr(mat(A));
    const Rs = toRows(R);
    for (let i = 1; i < 3; i++) {
      for (let j = 0; j < i; j++) expect(Rs[i]?.[j]).toBe(0);
    }
    expect(maxAbsDiff(matmulRows(toRows(Q), Rs), A)).toBeLessThan(1e-13);
  });

  it("rejects an unknown mode", () => {
    expect(() => qr(mat([[1]]), "full" as unknown as "reduced")).toThrow(InvalidParameterError);
  });
});

describe("eig / eigvals", () => {
  const B = [
    [4.75, -1.5, 0.75, -0.75, 1.0],
    [-0.25, 1.5, 0.75, -0.75, 1.0],
    [-1.0, 0.0, 3.0, 1.0, 0.0],
    [-1.75, -0.5, 0.25, 3.75, 3.0],
    [3.75, -1.5, 0.75, -0.75, 2.0],
  ];

  function residual(A: Mat, w: Tensor, V: Tensor): number {
    const n = A.length;
    const wd = toArray(w);
    const Vs = toRows(V);
    let worst = 0;
    for (let k = 0; k < n; k++) {
      let norm = 0;
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) {
          s += ((A[i] as number[])[j] as number) * ((Vs[j] as number[])[k] as number);
        }
        worst = Math.max(
          worst,
          Math.abs(s - (wd[k] as number) * ((Vs[i] as number[])[k] as number))
        );
        norm += ((Vs[i] as number[])[k] as number) ** 2;
      }
      expect(norm).toBeCloseTo(1, 12);
    }
    return worst;
  }

  it("solves a non-symmetric matrix with real spectrum (numpy reference 1..5)", () => {
    const [w, V] = eig(mat(B));
    const sorted = toArray(w).sort((a, b) => a - b);
    [1, 2, 3, 4, 5].forEach((expected, i) => {
      expect(sorted[i]).toBeCloseTo(expected, 12);
    });
    expect(residual(B, w, V)).toBeLessThan(1e-13);
    const only = toArray(eigvals(mat(B))).sort((a, b) => a - b);
    only.forEach((v, i) => {
      expect(v).toBeCloseTo(sorted[i] as number, 14);
    });
  });

  it("does not treat a tiny non-symmetric matrix as symmetric", () => {
    // Entries around 1e-20: an absolute symmetry tolerance of 1e-10 accepted any such matrix.
    const C = [
      [1e-20, 2e-20],
      [3e-20, 1e-20],
    ];
    const w = toArray(eigvals(mat(C))).sort((a, b) => a - b);
    expect(w[0]).toBeCloseTo(-1.449489742783178e-20, 30);
    expect(w[1]).toBeCloseTo(3.449489742783178e-20, 30);
    expect(() => eigh(mat(C))).toThrow(DataValidationError);
    expect(() => eigvalsh(mat(C))).toThrow(DataValidationError);
  });

  it("scales eigenvalues correctly for entries near 1e-200", () => {
    const base = [
      [1, 2, 0],
      [3, 1, 0],
      [0, 1, 5],
    ];
    const w = toArray(eigvals(mat(scaled(base, 1e-200)))).sort((a, b) => a - b);
    // numpy: eigenvalues of base are 1 +- sqrt(6) and 5
    const sqrt6 = Math.sqrt(6);
    expect((w[0] as number) / 1e-200).toBeCloseTo(1 - sqrt6, 10);
    expect((w[1] as number) / 1e-200).toBeCloseTo(1 + sqrt6, 10);
    expect((w[2] as number) / 1e-200).toBeCloseTo(5, 10);
  });

  it("balances badly scaled matrices (numpy: 2 and ~0)", () => {
    const G = [
      [1, 1e8],
      [1e-8, 1],
    ];
    const [w, V] = eig(mat(G));
    const sorted = toArray(w).sort((a, b) => a - b);
    expect(Math.abs(sorted[0] as number)).toBeLessThan(1e-12);
    expect(sorted[1]).toBeCloseTo(2, 12);
    expect(residual(G, w, V)).toBeLessThan(1e-9);
  });

  it("handles triangular, repeated and defective eigenvalues", () => {
    const M = [
      [4, 1, 2],
      [0, 3, 1],
      [0, 0, -2],
    ];
    const [w, V] = eig(mat(M));
    expect(toArray(w)).toEqual([4, 3, -2]);
    expect(residual(M, w, V)).toBeLessThan(1e-14);

    const J = [
      [2, 1, 0],
      [0, 2, 1],
      [0, 0, 2],
    ];
    const [wj, Vj] = eig(mat(J));
    expect(toArray(wj)).toEqual([2, 2, 2]);
    expect(residual(J, wj, Vj)).toBeLessThan(1e-12);
  });

  it("returns eigenvalues in ascending order for symmetric input", () => {
    const S = [
      [2, 1, 0],
      [1, 3, 1],
      [0, 1, 4],
    ];
    const [w] = eig(mat(S));
    const v = toArray(w);
    expect(v).toEqual([...v].sort((a, b) => a - b));
  });

  it("still reports complex eigenvalues", () => {
    expect(() =>
      eig(
        mat([
          [1, 2, 0],
          [-3, 1, 1],
          [0, 0, 5],
        ])
      )
    ).toThrow(InvalidParameterError);
    expect(() =>
      eigvals(
        mat([
          [0, -1],
          [1, 0],
        ])
      )
    ).toThrow(/complex eigenvalues/);
  });

  it("applies maxIter per eigenvalue and validates options", () => {
    expect(() => eig(mat(B), { maxIter: 1 })).toThrow(ConvergenceError);
    expect(() => eig(mat(B), { maxIter: 0 })).toThrow(InvalidParameterError);
    expect(() => eig(mat(B), { maxIter: 2.5 })).toThrow(InvalidParameterError);
    expect(() => eig(mat(B), { tol: -1 })).toThrow(InvalidParameterError);
    expect(() => eigvals(mat(B), { tol: Number.NaN })).toThrow(InvalidParameterError);
    // The 2x2 case needs no sweeps at all.
    const [w] = eig(
      mat([
        [4, 1],
        [2, 3],
      ]),
      { maxIter: 1 }
    );
    expect(toArray(w).sort((a, b) => a - b)).toEqual([2, 5]);
  });
});

describe("eigh / eigvalsh", () => {
  it("accepts rounding-level asymmetry in matrices with large entries", () => {
    // Off-diagonal entries differ by ~1e-3, far below 1e-10 relative to the 1e12 scale.
    const S = [
      [1e12, 2e12],
      [2e12 + 1e-3, 1e12],
    ];
    const w = toArray(eigvalsh(mat(S)));
    expect(w[0]).toBeCloseTo(-999999999999.9999, -2);
    expect((w[1] as number) / 3e12).toBeCloseTo(1, 12);
    const [wh] = eigh(mat(S));
    expect(toArray(wh)[1]).toBeCloseTo(w[1] as number, -3);
  });

  it("symmetrizes the input instead of reading only the lower triangle", () => {
    // Asymmetry of 1e-12 around entries of size 1: eigenvalues of (A + A^T) / 2.
    const S = [
      [2, 1 + 2e-12],
      [1, 2],
    ];
    const w = toArray(eigvalsh(mat(S)));
    const mid = 1 + 1e-12;
    expect(w[0]).toBeCloseTo(2 - mid, 14);
    expect(w[1]).toBeCloseTo(2 + mid, 14);
  });

  it("agrees between eigh, eigvalsh and eigvals", () => {
    const M = randomMat(12, 12, 5);
    const S = M.map((row, i) => row.map((v, j) => v + (M[j] as number[])[i]!));
    const [w, V] = eigh(mat(S));
    const wv = toArray(eigvalsh(mat(S)));
    const wg = toArray(eigvals(mat(S)));
    toArray(w).forEach((v, i) => {
      expect(v).toBeCloseTo(wv[i] as number, 12);
      expect(v).toBeCloseTo(wg[i] as number, 12);
    });
    const Vs = toRows(V);
    expect(maxAbsDiff(matmulRows(transposeRows(Vs), Vs), identity(12))).toBeLessThan(1e-13);
  });

  it("works for 1x1 and diagonal inputs", () => {
    expect(toArray(eigvalsh(mat([[7]])))).toEqual([7]);
    const [w, V] = eigh(
      mat([
        [3, 0],
        [0, 1],
      ])
    );
    expect(toArray(w)).toEqual([1, 3]);
    expect(Math.abs((toRows(V)[0] as number[])[1] as number)).toBe(1);
  });
});

describe("schur", () => {
  it("returns a triangular T for real eigenvalues (scipy reference)", () => {
    const A = [
      [1, 2],
      [3, 4],
    ];
    const [T, Q] = schur(mat(A));
    const Ts = toRows(T);
    expect(Ts[1]?.[0]).toBe(0);
    expect(Ts[0]?.[0]).toBeCloseTo(-0.3722813232690143, 12);
    expect(Ts[1]?.[1]).toBeCloseTo(5.372281323269014, 12);
    expect(Math.abs(Ts[0]?.[1] as number)).toBeCloseTo(1, 12);
    const Qs = toRows(Q);
    expect(maxAbsDiff(matmulRows(matmulRows(Qs, Ts), transposeRows(Qs)), A)).toBeLessThan(1e-13);
  });

  it("standardizes complex 2x2 blocks (equal diagonal, opposite off-diagonal signs)", () => {
    // Eigenvalues 1 +- i*sqrt(6) and 5.
    const A = [
      [1, 2, 0],
      [-3, 1, 1],
      [0, 0, 5],
    ];
    const [T, Q] = schur(mat(A));
    const Ts = toRows(T);
    const a = Ts[0]?.[0] as number;
    const b = Ts[0]?.[1] as number;
    const c = Ts[1]?.[0] as number;
    const d = Ts[1]?.[1] as number;
    expect(a).toBeCloseTo(d, 13);
    expect(b * c).toBeLessThan(0);
    expect(a).toBeCloseTo(1, 13);
    expect(Math.sqrt(-b * c)).toBeCloseTo(Math.sqrt(6), 12);
    expect(Ts[2]?.[2]).toBeCloseTo(5, 13);
    expect(Ts[2]?.[0]).toBe(0);
    expect(Ts[2]?.[1]).toBe(0);
    const Qs = toRows(Q);
    expect(maxAbsDiff(matmulRows(matmulRows(Qs, Ts), transposeRows(Qs)), A)).toBeLessThan(1e-13);
  });

  it("converges and reconstructs for permutation, rotation and random matrices", () => {
    const cyc = [
      [0, 1, 0, 0, 0],
      [0, 0, 1, 0, 0],
      [0, 0, 0, 1, 0],
      [0, 0, 0, 0, 1],
      [1, 0, 0, 0, 0],
    ];
    const inputs: Mat[] = [
      cyc,
      [
        [0, -1],
        [1, 0],
      ],
      randomMat(30, 30, 1),
      randomMat(60, 60, 2),
      scaled(randomMat(8, 8, 3), 1e-200),
      scaled(randomMat(8, 8, 4), 1e200),
    ];
    for (const A of inputs) {
      const n = A.length;
      const [T, Q] = schur(mat(A));
      const Ts = toRows(T);
      const Qs = toRows(Q);
      const rec = matmulRows(matmulRows(Qs, Ts), transposeRows(Qs));
      expect(maxAbsDiff(rec, A) / maxAbs(A)).toBeLessThan(1e-13);
      expect(maxAbsDiff(matmulRows(transposeRows(Qs), Qs), identity(n))).toBeLessThan(1e-13);
      for (let i = 2; i < n; i++) {
        for (let j = 0; j < i - 1; j++) expect(Ts[i]?.[j]).toBe(0);
      }
      // No two adjacent non-zero sub-diagonal entries: every 2x2 block is isolated.
      for (let i = 1; i + 1 < n; i++) {
        expect(Ts[i]?.[i - 1] !== 0 && Ts[i + 1]?.[i] !== 0).toBe(false);
      }
    }
  });

  it("handles zero and 1x1 matrices", () => {
    const [T0, Q0] = schur(
      mat([
        [0, 0],
        [0, 0],
      ])
    );
    expect(toArray(T0)).toEqual([0, 0, 0, 0]);
    expect(toArray(Q0)).toEqual([1, 0, 0, 1]);
    const [T1, Q1] = schur(mat([[3]]));
    expect(toArray(T1)).toEqual([3]);
    expect(toArray(Q1)).toEqual([1]);
  });
});

describe("svd / svdvals", () => {
  it("matches numpy singular values for a 3x2 matrix", () => {
    const A = [
      [2, 1],
      [1, 3],
      [0, 1],
    ];
    const s = toArray(svdvals(mat(A)));
    expect(s[0]).toBeCloseTo(3.7189987758596126, 12);
    expect(s[1]).toBeCloseTo(1.472768856662409, 12);
    const [, s2] = svd(mat(A));
    expect(toArray(s2)[0]).toBeCloseTo(3.7189987758596126, 12);
  });

  it("completes U to an orthonormal basis even when the first unit vector is nearly in the span", () => {
    // The single left singular vector is almost e_0, so orthogonalizing e_0 against it
    // cancels catastrophically unless the best candidate is chosen and re-orthogonalized.
    const A = [[1], [1e-9], [0]];
    const [U, s] = svd(mat(A));
    const Us = toRows(U);
    expect(U.shape).toEqual([3, 3]);
    expect(maxAbsDiff(matmulRows(transposeRows(Us), Us), identity(3))).toBeLessThan(1e-14);
    expect(toArray(s)[0]).toBeCloseTo(1, 12);
  });

  it("returns orthonormal full factors for tall and wide matrices", () => {
    for (const [m, n] of [
      [60, 3],
      [3, 60],
      [7, 7],
    ] as const) {
      const A = randomMat(m, n, m * 31 + n);
      const [U, s, Vt] = svd(mat(A), true);
      const Us = toRows(U);
      const Vs = toRows(Vt);
      expect(U.shape).toEqual([m, m]);
      expect(Vt.shape).toEqual([n, n]);
      expect(maxAbsDiff(matmulRows(transposeRows(Us), Us), identity(m))).toBeLessThan(1e-13);
      expect(maxAbsDiff(matmulRows(Vs, transposeRows(Vs)), identity(n))).toBeLessThan(1e-13);
      const k = Math.min(m, n);
      const sd = toArray(s);
      const Uk = Us.map((row) => row.slice(0, k).map((v, j) => v * (sd[j] as number)));
      const rec = matmulRows(Uk, Vs.slice(0, k));
      expect(maxAbsDiff(rec, A)).toBeLessThan(1e-13);
    }
  });

  it("agrees between the vectors and values-only paths and handles extreme scales", () => {
    const base = randomMat(9, 5, 17);
    for (const f of [1, 1e-150, 1e150]) {
      const A = scaled(base, f);
      const full = toArray(svd(mat(A), false)[1]);
      const vals = toArray(svdvals(mat(A)));
      full.forEach((v, i) => {
        expect(v / f).toBeCloseTo((vals[i] as number) / f, 12);
      });
    }
  });

  it("handles rank-deficient and zero matrices", () => {
    const [, s] = svd(
      mat([
        [1, 2],
        [2, 4],
        [3, 6],
      ])
    );
    expect(toArray(s)[0]).toBeCloseTo(Math.sqrt(70), 12);
    expect(Math.abs(toArray(s)[1] as number)).toBeLessThan(1e-14);
    const [U0, s0] = svd(
      mat([
        [0, 0],
        [0, 0],
      ])
    );
    expect(toArray(s0)).toEqual([0, 0]);
    expect(maxAbsDiff(matmulRows(transposeRows(toRows(U0)), toRows(U0)), identity(2))).toBeLessThan(
      1e-14
    );
  });
});

describe("cholesky", () => {
  const A = [
    [4, 12, -16],
    [12, 37, -43],
    [-16, -43, 98],
  ];

  it("matches numpy and supports the upper factor", () => {
    const L = toRows(cholesky(mat(A)));
    expect(
      maxAbsDiff(L, [
        [2, 0, 0],
        [6, 1, 0],
        [-8, 5, 3],
      ])
    ).toBeLessThan(1e-14);
    const U = toRows(cholesky(mat(A), { upper: true }));
    expect(maxAbsDiff(U, transposeRows(L))).toBe(0);
    expect(maxAbsDiff(matmulRows(transposeRows(U), U), A)).toBeLessThan(1e-12);
  });

  it("accepts rounding-level asymmetry relative to the matrix scale", () => {
    const big = A.map((row) => row.map((v) => v * 1e9));
    big[0]![1] = (big[0]![1] as number) + 1e-3;
    expect(() => cholesky(mat(big))).not.toThrow();
  });

  it("rejects clearly asymmetric matrices even when entries are tiny", () => {
    expect(() =>
      cholesky(
        mat([
          [1e-20, 2e-20],
          [3e-20, 1e-20],
        ])
      )
    ).toThrow(/symmetric/);
  });

  it("factors matrices with tiny but well-conditioned entries", () => {
    const L = toRows(
      cholesky(
        mat([
          [4e-20, 2e-20],
          [2e-20, 5e-20],
        ])
      )
    );
    expect((L[0] as number[])[0] as number).toBeCloseTo(2e-10, 22);
    expect((L[1] as number[])[0] as number).toBeCloseTo(1e-10, 22);
    expect((L[1] as number[])[1] as number).toBeCloseTo(2e-10, 22);
  });

  it("names the failing leading minor", () => {
    expect(() =>
      cholesky(
        mat([
          [1, 2],
          [2, 1],
        ])
      )
    ).toThrow(/leading minor of order 2/);
    expect(() => cholesky(mat([[-1]]))).toThrow(/not positive definite/);
  });
});

describe("lu", () => {
  it("matches scipy for a 4x4 matrix (P @ A = L @ U convention)", () => {
    const A = [
      [2, 1, 1, 0],
      [4, 3, 3, 1],
      [8, 7, 9, 5],
      [6, 7, 9, 8],
    ];
    const [P, L, U] = lu(mat(A));
    const refL = [
      [1, 0, 0, 0],
      [0.75, 1, 0, 0],
      [0.5, -0.2857142857142857, 1, 0],
      [0.25, -0.42857142857142855, 0.3333333333333333, 1],
    ];
    const refU = [
      [8, 7, 9, 5],
      [0, 1.75, 2.25, 4.25],
      [0, 0, -0.8571428571428572, -0.2857142857142858],
      [0, 0, 0, 0.6666666666666665],
    ];
    expect(maxAbsDiff(toRows(L), refL)).toBeLessThan(1e-14);
    expect(maxAbsDiff(toRows(U), refU)).toBeLessThan(1e-14);
    // scipy returns p with A = p @ l @ u; here P is its transpose.
    const refP = transposeRows([
      [0, 0, 0, 1],
      [0, 0, 1, 0],
      [1, 0, 0, 0],
      [0, 1, 0, 0],
    ]);
    expect(maxAbsDiff(toRows(P), refP)).toBe(0);
  });

  it("factors wide, tall and rank-deficient matrices", () => {
    const inputs: Mat[] = [
      randomMat(3, 6, 1),
      randomMat(6, 3, 2),
      [
        [0, 1, 2],
        [0, 2, 4],
        [0, 3, 6],
      ],
      [
        [1, 2],
        [2, 4],
        [3, 6],
      ],
    ];
    for (const A of inputs) {
      const [P, L, U] = lu(mat(A));
      const k = Math.min(A.length, A[0]?.length ?? 0);
      expect(L.shape).toEqual([A.length, k]);
      expect(U.shape).toEqual([k, A[0]?.length ?? 0]);
      const PA = matmulRows(toRows(P), A);
      expect(maxAbsDiff(PA, matmulRows(toRows(L), toRows(U)))).toBeLessThan(1e-13);
      const Ls = toRows(L);
      for (let i = 0; i < k; i++) expect(Ls[i]?.[i]).toBe(1);
    }
  });
});

describe("polar", () => {
  const A = [
    [1, 2],
    [3, 4],
  ];

  it("matches scipy.linalg.polar for side='right' and 'left'", () => {
    const [U, P] = polar(mat(A));
    expect(
      maxAbsDiff(toRows(U), [
        [-0.5144957554275263, 0.8574929257125441],
        [0.8574929257125443, 0.5144957554275262],
      ])
    ).toBeLessThan(1e-12);
    expect(
      maxAbsDiff(toRows(P), [
        [2.0579830217101063, 2.400980191995124],
        [2.400980191995124, 3.7729688731351945],
      ])
    ).toBeLessThan(1e-12);
    const [S, UL] = polar(mat(A), "left");
    expect(
      maxAbsDiff(toRows(S), [
        [1.200490095997561, 1.8864844365675963],
        [1.8864844365675963, 4.630461798847736],
      ])
    ).toBeLessThan(1e-12);
    expect(maxAbsDiff(toRows(UL), toRows(U))).toBeLessThan(1e-12);
  });

  it("returns exactly symmetric positive semi-definite factors", () => {
    const M = randomMat(9, 5, 21);
    const [, P] = polar(mat(M));
    const Ps = toRows(P);
    expect(maxAbsDiff(Ps, transposeRows(Ps))).toBe(0);
    const [S] = polar(mat(M), "left");
    const Ss = toRows(S);
    expect(maxAbsDiff(Ss, transposeRows(Ss))).toBe(0);
  });

  it("handles wide matrices (orthonormal rows) like scipy", () => {
    const W = [
      [1, 2, 3],
      [4, 5, 6],
    ];
    const [U, P] = polar(mat(W));
    expect(U.shape).toEqual([2, 3]);
    expect(P.shape).toEqual([3, 3]);
    expect(
      maxAbsDiff(toRows(P), [
        [2.2491922571417873, 2.3781464513085, 2.5071006454752127],
        [2.3781464513085, 3.059020585463927, 3.739894719619354],
        [2.5071006454752127, 3.739894719619354, 4.972688793763495],
      ])
    ).toBeLessThan(1e-12);
    expect(maxAbsDiff(matmulRows(toRows(U), toRows(P)), W)).toBeLessThan(1e-12);
    const Us = toRows(U);
    expect(maxAbsDiff(matmulRows(Us, transposeRows(Us)), identity(2))).toBeLessThan(1e-13);
  });

  it("rejects an unknown side", () => {
    expect(() => polar(mat(A), "up" as unknown as "left")).toThrow(InvalidParameterError);
  });
});

describe("pinv", () => {
  it("matches numpy for a rank-one matrix", () => {
    const P = toRows(
      pinv(
        mat([
          [1, 2],
          [2, 4],
          [3, 6],
        ])
      )
    );
    expect(
      maxAbsDiff(P, [
        [0.014285714285714287, 0.028571428571428574, 0.042857142857142864],
        [0.028571428571428574, 0.05714285714285715, 0.08571428571428573],
      ])
    ).toBeLessThan(1e-14);
  });
});

describe("graded and extreme-scale inputs", () => {
  const ones = (n: number): Mat => Array.from({ length: n }, () => new Array<number>(n).fill(1));

  it("hessenberg keeps Q orthogonal when the reduced matrix underflows (all-ones 30x30)", () => {
    // Householder reflections of the all-ones matrix leave a trailing block at the
    // underflow limit; normalizing by the (denormal) norm used to destroy orthogonality.
    const A = ones(30);
    const [H, Q] = hessenberg(mat(A));
    const Qs = toRows(Q);
    expect(maxAbsDiff(matmulRows(transposeRows(Qs), Qs), identity(30))).toBeLessThan(1e-14);
    expect(maxAbsDiff(matmulRows(matmulRows(Qs, toRows(H)), transposeRows(Qs)), A)).toBeLessThan(
      1e-13
    );
  });

  it("schur converges for the all-ones 30x30 matrix", () => {
    const A = ones(30);
    const [T, Q] = schur(mat(A));
    const Ts = toRows(T);
    const Qs = toRows(Q);
    expect(maxAbsDiff(matmulRows(matmulRows(Qs, Ts), transposeRows(Qs)), A)).toBeLessThan(1e-12);
    expect(maxAbsDiff(matmulRows(transposeRows(Qs), Qs), identity(30))).toBeLessThan(1e-13);
    const diag = Ts.map((row, i) => row[i] as number).sort((a, b) => b - a);
    expect(diag[0]).toBeCloseTo(30, 12);
    expect(Math.abs(diag[1] as number)).toBeLessThan(1e-13);
  });

  it("eig handles a non-symmetric rank-one matrix (repeated zero eigenvalue)", () => {
    // numpy returns complex pairs with imaginary parts around 1e-16 here; they are
    // rounding noise around the real eigenvalue 0 and must not be reported as complex.
    for (const n of [10, 30]) {
      const u = randomMat(n, 1, 7 + n).map((r) => (r[0] as number) + 0.6);
      const v = randomMat(n, 1, 9 + n).map((r) => (r[0] as number) + 0.6);
      const A = u.map((a) => v.map((b) => a * b));
      const trace = u.reduce((s, a, i) => s + a * (v[i] as number), 0);
      const w = toArray(eigvals(mat(A))).sort((a, b) => b - a);
      expect(w[0]).toBeCloseTo(trace, 12);
      for (const x of w.slice(1)) expect(Math.abs(x)).toBeLessThan(1e-13);
      const [w2, V] = eig(mat(A));
      const Vs = toRows(V);
      const wd = toArray(w2);
      let worst = 0;
      for (let k = 0; k < n; k++) {
        for (let i = 0; i < n; i++) {
          let s = 0;
          for (let j = 0; j < n; j++) {
            s += ((A[i] as number[])[j] as number) * ((Vs[j] as number[])[k] as number);
          }
          worst = Math.max(
            worst,
            Math.abs(s - (wd[k] as number) * ((Vs[i] as number[])[k] as number))
          );
        }
      }
      expect(worst).toBeLessThan(1e-12);
    }
  });

  it("eigh and eigvalsh scale extreme symmetric matrices exactly", () => {
    const M = randomMat(6, 6, 41);
    const S = M.map((row, i) => row.map((v, j) => v + (M[j] as number[])[i]!));
    const ref = toArray(eigvalsh(mat(S)));
    for (const f of [1e-200, 1e-310, 1e300]) {
      const A = scaled(S, f);
      const w = toArray(eigvalsh(mat(A)));
      const [wh, V] = eigh(mat(A));
      w.forEach((x, i) => {
        expect(x / f).toBeCloseTo(ref[i] as number, 10);
        expect((toArray(wh)[i] as number) / f).toBeCloseTo(ref[i] as number, 10);
      });
      const Vs = toRows(V);
      expect(maxAbsDiff(matmulRows(transposeRows(Vs), Vs), identity(6))).toBeLessThan(1e-14);
    }
  });

  it("qr, hessenberg and schur stay orthogonal for denormal entries", () => {
    const A = scaled(randomMat(6, 6, 3), 1e-320);
    const [Q] = qr(mat(A));
    const [, Qh] = hessenberg(mat(A));
    const [, Z] = schur(mat(A));
    for (const X of [Q, Qh, Z]) {
      const Xs = toRows(X);
      expect(maxAbsDiff(matmulRows(transposeRows(Xs), Xs), identity(6))).toBeLessThan(1e-13);
    }
  });

  it("qr and svdvals stay finite for entries near the largest double", () => {
    // 2 ** 1024 overflows, so the power-of-two prescale must round down.
    const A = [
      [1.5e308, 1e307],
      [2e307, -1.2e308],
    ];
    const [, R] = qr(mat(A));
    const Rs = toRows(R);
    expect(Rs.flat().every(Number.isFinite)).toBe(true);
    // numpy on A / 1e300, rescaled: sqrt(1.5^2 + 0.2^2) = 1.51327...
    expect(Math.abs(Rs[0]?.[0] as number) / 1e308).toBeCloseTo(1.5132745950421556, 12);
    const s = toArray(svdvals(mat(A)));
    expect((s[0] as number) / 1e308).toBeCloseTo(1.5164216537290316, 12);
    expect((s[1] as number) / 1e308).toBeCloseTo(1.200193887712193, 12);
    const w = toArray(eigvals(mat(A))).sort((a, b) => a - b);
    expect((w[0] as number) / 1e308).toBeCloseTo(-1.207387196049823, 12);
    expect((w[1] as number) / 1e308).toBeCloseTo(1.5073871960498226, 12);
  });
});
