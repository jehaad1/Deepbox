import { describe, expect, it } from "vitest";
import { cholesky, eig, lu, qr, svd } from "../src/linalg";
import { allclose, tensor } from "../src/ndarray";
import { matmul } from "../src/ndarray/linalg/basic";
import { toNum2D, toNumArr } from "./_helpers";

describe("deepbox/linalg - decomposition extra branches", () => {
  it("lu throws on non-finite values", () => {
    const a = tensor([
      [NaN, 1],
      [2, 3],
    ]);
    expect(() => lu(a)).toThrow(/non-finite/i);
  });

  it("qr complete handles dependent columns (norm==0 branch)", () => {
    const a = tensor([[0], [1], [0]]);
    const [Q, R] = qr(a, "complete");

    expect(Q.shape).toEqual([3, 3]);
    expect(R.shape).toEqual([3, 1]);

    const reconstructed = matmul(Q, R);
    expect(allclose(reconstructed, a, 1e-8, 1e-8)).toBe(true);
  });

  it("svd handles rank-deficient matrices (sigma == 0 branch)", () => {
    const a = tensor([
      [1, 2],
      [2, 4],
    ]);
    const [U, s, Vt] = svd(a, false);
    expect(U.shape).toEqual([2, 2]);
    expect(s.shape).toEqual([2]);
    expect(Vt.shape).toEqual([2, 2]);

    const sVals = toNumArr(s.toArray());
    expect(sVals[1]).toBeCloseTo(0, 12);

    const u = toNum2D(U.toArray());
    const col0 = [u[0]?.[0] ?? 0, u[1]?.[0] ?? 0];
    const col1 = [u[0]?.[1] ?? 0, u[1]?.[1] ?? 0];
    const dot01 = col0[0] * col1[0] + col0[1] * col1[1];
    const norm0 = Math.hypot(col0[0], col0[1]);
    const norm1 = Math.hypot(col1[0], col1[1]);
    expect(Math.abs(dot01)).toBeLessThan(1e-6);
    expect(norm0).toBeCloseTo(1, 6);
    expect(norm1).toBeCloseTo(1, 6);
  });

  it("svd full matrices handles orthonormal completion", () => {
    const a = tensor([[0], [1], [0]]);
    const [U, s, Vt] = svd(a, true);
    expect(U.shape).toEqual([3, 3]);
    expect(s.shape).toEqual([1]);
    expect(Vt.shape).toEqual([1, 1]);

    const u = toNum2D(U.toArray());
    const vt = toNum2D(Vt.toArray());
    const sigma = toNumArr(s.toArray())[0] ?? 0;
    const recon: number[][] = Array.from({ length: 3 }, () => [0]);
    for (let i = 0; i < 3; i++) {
      recon[i][0] = (u[i]?.[0] ?? 0) * sigma * (vt[0]?.[0] ?? 0);
    }
    const aArr = toNum2D(a.toArray());
    for (let i = 0; i < 3; i++) {
      expect(recon[i]?.[0]).toBeCloseTo(aArr[i]?.[0] ?? 0, 6);
    }
  });

  it("eig non-symmetric path handles zero column QR (norm==0 branch)", () => {
    const a = tensor([
      [1, 0],
      [2, 0],
    ]);
    const [vals, vecs] = eig(a);
    expect(vals.shape).toEqual([2]);
    expect(vecs.shape).toEqual([2, 2]);

    const valsArr = toNumArr(vals.toArray());
    const vecArr = toNum2D(vecs.toArray());
    for (let i = 0; i < 2; i++) {
      const lambda = valsArr[i] ?? 0;
      const v0 = vecArr[0]?.[i] ?? 0;
      const v1 = vecArr[1]?.[i] ?? 0;
      const av0 = 1 * v0 + 0 * v1;
      const av1 = 2 * v0 + 0 * v1;
      expect(av0).toBeCloseTo(lambda * v0, 5);
      expect(av1).toBeCloseTo(lambda * v1, 5);
    }
  });

  it("cholesky rejects non-symmetric and non-positive-definite matrices", () => {
    const nonsym = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => cholesky(nonsym)).toThrow(/symmetric/);

    const notPosDef = tensor([
      [1, 2],
      [2, 1],
    ]);
    expect(() => cholesky(notPosDef)).toThrow(/positive definite/);
  });

  it("cholesky rejects matrices with non-finite values", () => {
    const withNaN = tensor([
      [4, NaN],
      [NaN, 9],
    ]);
    expect(() => cholesky(withNaN)).toThrow(/non-finite/i);

    const withInfinity = tensor([
      [4, Infinity],
      [Infinity, 9],
    ]);
    expect(() => cholesky(withInfinity)).toThrow(/non-finite/i);
  });
});
