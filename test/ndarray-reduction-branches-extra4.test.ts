import { describe, expect, it } from "vitest";
import { any, max, min, prod, tensor } from "../src/ndarray";

describe("ndarray reduction branches extra 4", () => {
  it("prod handles axis undefined and keepdims", () => {
    const t = tensor([
      [2, 3],
      [4, 5],
    ]);
    const p = prod(t, undefined, true);
    expect(p.shape).toEqual([1, 1]);
    expect(p.toArray()).toEqual([[120]]);
  });

  it("min/max reject string dtype", () => {
    const s = tensor(["a", "b"]);
    expect(() => min(s)).toThrow(/string/i);
    expect(() => max(s)).toThrow(/string/i);
  });

  it("any handles keepdims false with axis", () => {
    const t = tensor([
      [0, 1],
      [0, 0],
    ]);
    const out = any(t, 1);
    expect(out.toArray()).toEqual([1, 0]);
  });
});
