import { describe, expect, it } from "vitest";
import { eig, eigh, eigvals, eigvalsh, lu, qr } from "../src/linalg";
import { allclose, tensor, transpose } from "../src/ndarray";
import { matmul } from "../src/ndarray/linalg/basic";
import { toNum2D, toNumArr } from "./_helpers";

describe("linalg eig/lu/qr extra branches", () => {
  it("eig routes symmetric matrices to eigh", () => {
    const a = tensor([
      [2, 1],
      [1, 2],
    ]);
    const [vals1] = eig(a);
    const [vals2] = eigh(a);
    expect(vals1.toArray()).toEqual(vals2.toArray());
    expect(eigvals(a).toArray()).toEqual(vals2.toArray());
    expect(eigvalsh(a).toArray()).toEqual(vals2.toArray());
  });

  it("eigh rejects non-symmetric input", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => eigh(a)).toThrow(/symmetric/i);
  });

  it("eig handles non-symmetric matrices and returns square eigenvectors", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
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
      const av0 = 1 * v0 + 2 * v1;
      const av1 = 3 * v0 + 4 * v1;
      expect(av0).toBeCloseTo(lambda * v0, 5);
      expect(av1).toBeCloseTo(lambda * v1, 5);
    }
  });

  it("lu handles non-finite input and pivot-zero branch", () => {
    const bad = tensor([
      [NaN, 1],
      [2, 3],
    ]);
    expect(() => lu(bad)).toThrow(/non-finite/i);

    // Rank-deficient matrices are handled gracefully (no throw)
    const pivotZero = tensor([
      [0, 0],
      [0, 1],
    ]);
    const [P, L, U] = lu(pivotZero);
    expect(allclose(matmul(P, pivotZero), matmul(L, U), 1e-10, 1e-10)).toBe(true);
  });

  it("qr handles linearly dependent columns (zero norm branch)", () => {
    const a = tensor([
      [1, 2],
      [2, 4],
    ]);
    const [Q, R] = qr(a);
    expect(Q.shape).toEqual([2, 2]);
    expect(R.shape).toEqual([2, 2]);
    const reconstructed = matmul(Q, R);
    expect(allclose(reconstructed, a, 1e-8, 1e-8)).toBe(true);
  });

  it("qr handles non-contiguous input", () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const t = transpose(a);
    const [Q, R] = qr(t);
    expect(Q.shape[0]).toBe(3);
    expect(R.shape[1]).toBe(2);
    const reconstructed = matmul(Q, R);
    expect(allclose(reconstructed, t, 1e-8, 1e-8)).toBe(true);
  });
});
