import { describe, expect, it } from "vitest";
import { norm } from "../src/linalg/norms";
import { tensor } from "../src/ndarray";

describe("linalg norms branches extra 3", () => {
  it("handles vector norms across orders", () => {
    const v = tensor([3, -4, 0]);
    expect(norm(v, 0)).toBe(2);
    expect(norm(v, 1)).toBe(7);
    expect(norm(v, 2)).toBe(5);
    expect(norm(v, Number.POSITIVE_INFINITY)).toBe(4);
    expect(norm(v, Number.NEGATIVE_INFINITY)).toBe(0);
    expect(norm(v, "fro")).toBeCloseTo(5, 6);
    expect(norm(v, 3)).toBeCloseTo((27 + 64) ** (1 / 3), 6);
  });

  it("handles matrix norms with axis specified", () => {
    const m = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const a0 = norm(m, 0, 0);
    if (typeof a0 === "number") throw new Error("Expected Tensor");
    expect(a0.toArray()).toEqual([2, 2, 2]);

    const a1 = norm(m, 1, 1);
    if (typeof a1 === "number") throw new Error("Expected Tensor");
    expect(a1.toArray()).toEqual([6, 15]);
  });
});
