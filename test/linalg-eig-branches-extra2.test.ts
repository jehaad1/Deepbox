import { describe, expect, it } from "vitest";
import { eig } from "../src/linalg/decomposition/eig";
import { tensor } from "../src/ndarray";
import { toNum2D, toNumArr } from "./_helpers";

describe("linalg eig branches extra 2", () => {
  it("handles nonsymmetric matrix with zero column", () => {
    const A = tensor([
      [1, 0],
      [2, 0],
    ]);
    const [vals, vecs] = eig(A);
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

  it("returns eigenvectors that satisfy A*v ≈ λ*v for nonsymmetric matrices", () => {
    const A = tensor([
      [2, 1],
      [0, 3],
    ]);
    const [vals, vecs] = eig(A);
    const valsArr = toNumArr(vals.toArray());
    const vecArr = toNum2D(vecs.toArray());
    for (let i = 0; i < valsArr.length; i++) {
      const lambda = valsArr[i];
      const v0 = vecArr[0]?.[i];
      const v1 = vecArr[1]?.[i];
      expect(v0).toBeDefined();
      expect(v1).toBeDefined();
      if (v0 === undefined || v1 === undefined) continue;
      const av0 = 2 * v0 + 1 * v1;
      const av1 = 0 * v0 + 3 * v1;
      expect(av0).toBeCloseTo(lambda * v0, 5);
      expect(av1).toBeCloseTo(lambda * v1, 5);
    }
  });
});
