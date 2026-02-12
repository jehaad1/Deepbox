import { describe, expect, it } from "vitest";
import { lstsq } from "../src/linalg/solvers/lstsq";
import { tensor } from "../src/ndarray";

describe("lstsq branch coverage extras", () => {
  it("validates input dimensions", () => {
    expect(() => lstsq(tensor([1, 2, 3]), tensor([1, 2]))).toThrow(/2D/);
    expect(() => lstsq(tensor([[1, 2]]), tensor([[[1]]]))).toThrow(/1D or 2D/);
    expect(() => lstsq(tensor([[1, 2]]), tensor([1, 2]))).toThrow(/do not match/i);
  });

  it("handles 2D right-hand side", () => {
    const A = tensor([
      [1, 0],
      [0, 1],
    ]);
    const B = tensor([
      [1, 2],
      [3, 4],
    ]);
    const result = lstsq(A, B);
    expect(result.x.shape).toEqual([2, 2]);
    expect(result.residuals.shape).toEqual([2]);
    expect(result.rank).toBe(2);
    expect(result.x.toArray()).toEqual([
      [1, 2],
      [3, 4],
    ]);
    expect(result.residuals.toArray()).toEqual([0, 0]);
  });
});
