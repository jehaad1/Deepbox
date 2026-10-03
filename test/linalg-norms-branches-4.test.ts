import { describe, expect, it } from "vitest";
import { cond } from "../src/linalg";
import { tensor, zeros } from "../src/ndarray";

describe("linalg norms branches 4", () => {
  it("cond validates inputs and handles fro errors", () => {
    expect(() => cond(tensor([1, 2]))).toThrow(/2D/);
    expect(() => cond(tensor([[1, 2]]), 3)).toThrow(/Unsupported norm order/);

    const bad = tensor([
      [1, NaN],
      [2, 3],
    ]);
    expect(() => cond(bad, "fro")).toThrow(/non-finite/i);
  });

  it("returns Infinity for empty matrices and a very large value for singular ones", () => {
    const empty = zeros([0, 0]);
    expect(cond(empty)).toBe(Infinity);

    // A numerically singular matrix yields a smallest singular value on the
    // order of machine epsilon (matching NumPy/LAPACK, which return a very
    // large finite condition number rather than exactly Infinity).
    const singular = tensor([
      [1, 2],
      [2, 4],
    ]);
    expect(cond(singular)).toBeGreaterThan(1e12);
  });
});
