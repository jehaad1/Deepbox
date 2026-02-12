import { describe, expect, it } from "vitest";
import { lu } from "../src/linalg/decomposition/lu";
import { allclose, tensor } from "../src/ndarray";
import { matmul } from "../src/ndarray/linalg/basic";
import { toNum2D } from "./_helpers";

describe("linalg LU branches extra", () => {
  it("handles zero pivot columns", () => {
    const A = tensor([
      [0, 1],
      [0, 2],
    ]);
    // Rank-deficient matrices are handled gracefully: P @ A = L @ U
    const [P, L, U] = lu(A);
    expect(allclose(matmul(P, A), matmul(L, U), 1e-10, 1e-10)).toBe(true);
  });

  it("handles already upper-triangular without swaps", () => {
    const A = tensor([
      [2, 1],
      [0, 3],
    ]);
    const [_P, L, U] = lu(A);
    expect(toNum2D(L.toArray())[0][0]).toBeCloseTo(1, 6);
    expect(toNum2D(U.toArray())[0][0]).toBeCloseTo(2, 6);
  });

  it("returns an upper-triangular U", () => {
    const A = tensor([
      [2, 1, 1],
      [4, 3, 3],
      [8, 7, 9],
    ]);
    const [_P, _L, U] = lu(A);
    const u = toNum2D(U.toArray());
    for (let i = 0; i < u.length; i++) {
      const row = u[i];
      expect(row).toBeDefined();
      if (!row) continue;
      for (let j = 0; j < Math.min(i, row.length); j++) {
        const val = row[j];
        expect(val).toBeDefined();
        if (val === undefined) continue;
        expect(Math.abs(val)).toBeLessThan(1e-10);
      }
    }
  });
});
