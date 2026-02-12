import { describe, expect, it } from "vitest";
import { det, matrixRank, slogdet, trace } from "../src/linalg";
import { tensor, zeros } from "../src/ndarray";

describe("linalg properties extra branches", () => {
  it("det/slogdet validate inputs and handle singular matrices", () => {
    expect(() => det(tensor([1, 2, 3]))).toThrow(/2-D/);
    expect(() => det(tensor([[1, 2, 3]]))).toThrow(/square/);

    const singular = tensor([
      [1, 2],
      [2, 4],
    ]);
    expect(det(singular)).toBe(0);

    const [sign, logdet] = slogdet(singular);
    expect(sign.toArray()).toEqual([0]);
    expect(logdet.toArray()).toEqual([-Infinity]);
  });

  it("trace handles positive and negative offsets", () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    expect(trace(a, 0).toArray()).toEqual([15]);
    expect(trace(a, 1).toArray()).toEqual([8]);
    expect(trace(a, -1).toArray()).toEqual([12]);
  });

  it("matrixRank validates tol and handles empty matrices", () => {
    expect(() => matrixRank(tensor([1, 2, 3]))).toThrow(/2-D/);
    expect(() => matrixRank(tensor([[1]]), -1)).toThrow(/non-negative/);

    const empty = zeros([0, 3]);
    expect(matrixRank(empty)).toBe(0);
  });
});
