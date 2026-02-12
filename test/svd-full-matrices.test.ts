import { describe, expect, it } from "vitest";
import { svd } from "../src/linalg";
import { tensor } from "../src/ndarray";
import { toNum2D, toNumArr } from "./_helpers";

function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] ?? 0) * (b[i] ?? 0);
  return sum;
}

describe("deepbox/linalg - SVD fullMatrices", () => {
  it("returns full orthonormal U and Vt when fullMatrices=true", () => {
    const A = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const [U, s, Vt] = svd(A, true);

    expect(U.shape).toEqual([3, 3]);
    expect(s.shape).toEqual([2]);
    expect(Vt.shape).toEqual([2, 2]);

    const Uarr = toNum2D(U.toArray());
    const col0 = [Uarr[0]?.[0] ?? 0, Uarr[1]?.[0] ?? 0, Uarr[2]?.[0] ?? 0];
    const col1 = [Uarr[0]?.[1] ?? 0, Uarr[1]?.[1] ?? 0, Uarr[2]?.[1] ?? 0];
    const col2 = [Uarr[0]?.[2] ?? 0, Uarr[1]?.[2] ?? 0, Uarr[2]?.[2] ?? 0];
    expect(dot(col0, col1)).toBeCloseTo(0, 5);
    expect(dot(col0, col2)).toBeCloseTo(0, 5);
    expect(dot(col1, col2)).toBeCloseTo(0, 5);
  });

  it("returns full Vt for wide matrices", () => {
    const A = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const [U, s, Vt] = svd(A, true);
    expect(U.shape).toEqual([2, 2]);
    expect(s.shape).toEqual([2]);
    expect(Vt.shape).toEqual([3, 3]);

    const u = toNum2D(U.toArray());
    const vt = toNum2D(Vt.toArray());
    const sArr = toNumArr(s.toArray());
    const k = sArr.length;
    const recon: number[][] = Array.from({ length: 2 }, () => new Array<number>(3).fill(0));
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 3; j++) {
        let sum = 0;
        for (let r = 0; r < k; r++) {
          sum += (u[i]?.[r] ?? 0) * (sArr[r] ?? 0) * (vt[r]?.[j] ?? 0);
        }
        recon[i][j] = sum;
      }
    }
    const aArr = toNum2D(A.toArray());
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 3; j++) {
        expect(recon[i]?.[j]).toBeCloseTo(aArr[i]?.[j] ?? 0, 6);
      }
    }
  });
});
