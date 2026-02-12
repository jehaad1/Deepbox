import { describe, expect, it } from "vitest";
import { cond } from "../src/linalg";
import { tensor, zeros } from "../src/ndarray";

describe("linalg norms cond extra branches", () => {
  it("cond validates inputs and handles fro errors", () => {
    expect(() => cond(tensor([1, 2]))).toThrow(/2D/);
    expect(() => cond(tensor([[1, 2]]), 1)).toThrow(/Only 2-norm/);

    const bad = tensor([
      [1, NaN],
      [2, 3],
    ]);
    expect(() => cond(bad, "fro")).toThrow(/non-finite/i);
  });

  it("cond returns Infinity for empty and singular matrices", () => {
    const empty = zeros([0, 0]);
    expect(cond(empty)).toBe(Infinity);

    const singular = tensor([
      [1, 2],
      [2, 4],
    ]);
    expect(cond(singular)).toBe(Infinity);
  });
});
