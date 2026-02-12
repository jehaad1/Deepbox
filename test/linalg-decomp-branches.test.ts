import { describe, expect, it } from "vitest";
import { cholesky } from "../src/linalg/decomposition/cholesky";
import { eig, eigh, eigvals, eigvalsh } from "../src/linalg/decomposition/eig";
import { lu } from "../src/linalg/decomposition/lu";
import { tensor, zeros } from "../src/ndarray";
import { toNum2D, toNumArr } from "./_helpers";

function matMul2(a: number[][], b: number[][]): number[][] {
  return [
    [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
    [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
  ];
}

describe("linalg decomposition branches", () => {
  it("handles LU decomposition branches", () => {
    expect(() => lu(tensor([1, 2]))).toThrow(/2D matrix/i);

    const A = tensor([
      [0, 2],
      [1, 2],
    ]);
    const [P, L, U] = lu(A);
    const p = toNum2D(P.toArray());
    const l = toNum2D(L.toArray());
    const u = toNum2D(U.toArray());

    const pa = matMul2(p, toNum2D(A.toArray()));
    const luRecon = matMul2(l, u);
    expect(pa[0][0]).toBeCloseTo(luRecon[0][0], 6);
    expect(pa[0][1]).toBeCloseTo(luRecon[0][1], 6);
    expect(pa[1][0]).toBeCloseTo(luRecon[1][0], 6);
    expect(pa[1][1]).toBeCloseTo(luRecon[1][1], 6);

    // Rank-deficient (all-zero) matrix is handled gracefully
    const Z = tensor([
      [0, 0],
      [0, 0],
    ]);
    const [Pz, Lz, Uz] = lu(Z);
    const pz = toNum2D(Pz.toArray());
    const lz = toNum2D(Lz.toArray());
    const uz = toNum2D(Uz.toArray());
    const paz = matMul2(pz, toNum2D(Z.toArray()));
    const luzRecon = matMul2(lz, uz);
    expect(paz[0][0]).toBeCloseTo(luzRecon[0][0], 6);
    expect(paz[0][1]).toBeCloseTo(luzRecon[0][1], 6);
    expect(paz[1][0]).toBeCloseTo(luzRecon[1][0], 6);
    expect(paz[1][1]).toBeCloseTo(luzRecon[1][1], 6);

    const bad = tensor([
      [NaN, 0],
      [0, 1],
    ]);
    expect(() => lu(bad)).toThrow(/non-finite/i);
  });

  it("handles eig/eigh branches", () => {
    expect(() => eig(tensor([1, 2]))).toThrow(/2D matrix/i);
    expect(() => eig(tensor([[1, 2, 3]]))).toThrow(/square matrix/i);

    const sym = tensor([
      [2, 1],
      [1, 2],
    ]);
    const [vals, vecs] = eig(sym);
    expect(vals.shape).toEqual([2]);
    expect(vecs.shape).toEqual([2, 2]);
    const valsArr = toNumArr(vals.toArray());
    const vecArr = toNum2D(vecs.toArray());
    for (let i = 0; i < 2; i++) {
      const lambda = valsArr[i] ?? 0;
      const v0 = vecArr[0]?.[i] ?? 0;
      const v1 = vecArr[1]?.[i] ?? 0;
      const av0 = 2 * v0 + 1 * v1;
      const av1 = 1 * v0 + 2 * v1;
      expect(av0).toBeCloseTo(lambda * v0, 5);
      expect(av1).toBeCloseTo(lambda * v1, 5);
    }

    const valsOnly = eigvals(sym);
    expect(valsOnly.shape).toEqual([2]);
    const sorted = toNumArr(valsOnly.toArray())
      .slice()
      .sort((a, b) => a - b);
    expect(sorted[0]).toBeCloseTo(1, 5);
    expect(sorted[1]).toBeCloseTo(3, 5);

    const [svals, svecs] = eigh(sym);
    expect(svals.shape).toEqual([2]);
    expect(svecs.shape).toEqual([2, 2]);

    expect(eigvalsh(sym).shape).toEqual([2]);

    const nonsym = tensor([
      [1, 2],
      [0, 3],
    ]);
    const [nvals, nvecs] = eig(nonsym);
    expect(nvals.shape).toEqual([2]);
    expect(nvecs.shape).toEqual([2, 2]);
    const nvalsArr = toNumArr(nvals.toArray());
    const nvecArr = toNum2D(nvecs.toArray());
    for (let i = 0; i < 2; i++) {
      const lambda = nvalsArr[i] ?? 0;
      const v0 = nvecArr[0]?.[i] ?? 0;
      const v1 = nvecArr[1]?.[i] ?? 0;
      const av0 = 1 * v0 + 2 * v1;
      const av1 = 0 * v0 + 3 * v1;
      expect(av0).toBeCloseTo(lambda * v0, 5);
      expect(av1).toBeCloseTo(lambda * v1, 5);
    }

    const rotation = tensor([
      [0, -1],
      [1, 0],
    ]);
    expect(() => eig(rotation)).toThrow(/complex eigenvalues/i);
    expect(() => eigvals(rotation)).toThrow(/complex eigenvalues/i);

    const badSym = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => eigh(badSym)).toThrow(/symmetric/i);
  });

  it("handles cholesky branches", () => {
    expect(() => cholesky(tensor([1, 2]))).toThrow(/2D matrix/i);
    expect(() => cholesky(tensor([[1, 2, 3]]))).toThrow(/square/i);

    const empty = zeros([0, 0]);
    const Lempty = cholesky(empty);
    expect(Lempty.shape).toEqual([0, 0]);

    const sym = tensor([
      [4, 12],
      [12, 37],
    ]);
    const L = cholesky(sym);
    expect(L.shape).toEqual([2, 2]);
    const lArr = toNum2D(L.toArray());
    const lt = [
      [lArr[0]?.[0] ?? 0, lArr[1]?.[0] ?? 0],
      [lArr[0]?.[1] ?? 0, lArr[1]?.[1] ?? 0],
    ];
    const recon = matMul2(lArr, lt);
    const sArr = toNum2D(sym.toArray());
    expect(recon[0][0]).toBeCloseTo(sArr[0]?.[0] ?? 0, 6);
    expect(recon[0][1]).toBeCloseTo(sArr[0]?.[1] ?? 0, 6);
    expect(recon[1][0]).toBeCloseTo(sArr[1]?.[0] ?? 0, 6);
    expect(recon[1][1]).toBeCloseTo(sArr[1]?.[1] ?? 0, 6);

    const nonSym = tensor([
      [1, 2],
      [0, 1],
    ]);
    expect(() => cholesky(nonSym)).toThrow(/symmetric/i);

    const nonPos = tensor([
      [1, 2],
      [2, 1],
    ]);
    expect(() => cholesky(nonPos)).toThrow(/positive definite/i);
  });
});
