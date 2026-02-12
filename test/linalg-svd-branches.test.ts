import { describe, expect, it } from "vitest";
import { svd } from "../src/linalg/decomposition/svd";
import { allclose, tensor, transpose, zeros } from "../src/ndarray";
import { matmul } from "../src/ndarray/linalg/basic";
import { toNum2D, toNumArr } from "./_helpers";

function matMul(a: number[][], b: number[][]): number[][] {
  const rows = a.length;
  const cols = b[0]?.length ?? 0;
  const inner = b.length;
  const out: number[][] = Array.from({ length: rows }, () => new Array<number>(cols).fill(0));
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      let sum = 0;
      for (let k = 0; k < inner; k++) {
        sum += (a[i]?.[k] ?? 0) * (b[k]?.[j] ?? 0);
      }
      out[i][j] = sum;
    }
  }
  return out;
}

describe("svd branch coverage", () => {
  it("handles empty matrices", () => {
    const A = zeros([0, 2]);
    const [U, s, Vt] = svd(A, false);
    expect(U.shape).toEqual([0, 0]);
    expect(s.shape).toEqual([0]);
    expect(Vt.shape).toEqual([0, 2]);
  });

  it("handles rank-deficient matrices (zero singular values)", () => {
    const A = tensor([
      [1, 0],
      [0, 0],
    ]);
    const [U, s, Vt] = svd(A, false);
    expect(U.shape).toEqual([2, 2]);
    const sArr1 = toNumArr(s.toArray());
    expect(sArr1.length).toBe(2);
    expect(Vt.shape).toEqual([2, 2]);
    expect(sArr1[1]).toBeCloseTo(0, 6);
  });

  it("computes full matrices and orthonormal complements", () => {
    const A = tensor([[0], [2]]);
    const [U, s, Vt] = svd(A, true);
    expect(U.shape).toEqual([2, 2]);
    expect(s.shape).toEqual([1]);
    expect(Vt.shape).toEqual([1, 1]);

    const u = toNum2D(U.toArray());
    const v = toNum2D(Vt.toArray());
    const sigma = toNumArr(s.toArray())[0] ?? 0;
    const US = u.map((row) => [row[0] * sigma]);
    const reconstructed = matMul(US, v);
    expect(reconstructed[0][0]).toBeCloseTo(0, 6);
    expect(reconstructed[1][0]).toBeCloseTo(2, 6);
  });

  it("keeps U orthonormal for rank-deficient inputs", () => {
    const A = tensor([
      [1, 0],
      [2, 0],
    ]);
    const [U, _s, _Vt] = svd(A, false);
    const UtU = toNum2D(matmul(transpose(U), U).toArray());
    for (let i = 0; i < UtU.length; i++) {
      const row = UtU[i];
      expect(row).toBeDefined();
      if (!row) continue;
      for (let j = 0; j < row.length; j++) {
        const val = row[j];
        expect(val).toBeDefined();
        if (val === undefined) continue;
        const expected = i === j ? 1 : 0;
        expect(val).toBeCloseTo(expected, 6);
      }
    }
  });

  it("handles wide matrices (m < n) in compact form", () => {
    const A = tensor([
      [1, 0, 0],
      [0, 2, 0],
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
    expect(allclose(reconstructed, A, 1e-8, 1e-8)).toBe(true);
  });

  it("recovers small singular values for ill-conditioned inputs", () => {
    const A = tensor([
      [1, 0],
      [0, 1e-8],
    ]);
    const [_U, s, _Vt] = svd(A, false);
    const sArr = toNumArr(s.toArray());
    expect(sArr[0]).toBeGreaterThan(0.9);
    const ratio = (sArr[1] ?? 0) / 1e-8;
    expect(ratio).toBeGreaterThan(0.1);
    expect(ratio).toBeLessThan(10);
  });
});
