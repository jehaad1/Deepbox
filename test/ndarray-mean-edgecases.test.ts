import { describe, expect, it } from "vitest";
import { mean, tensor } from "../src/ndarray";

describe("deepbox/ndarray - mean edge cases", () => {
  it("returns float64 for integer inputs", () => {
    const t32 = tensor([1, 2, 3], { dtype: "int32" });
    const m32 = mean(t32);
    expect(m32.dtype).toBe("float64");
    expect(m32.toArray()).toBe(2);

    const t64 = tensor([1, 2], { dtype: "int64" });
    const m64 = mean(t64);
    expect(m64.dtype).toBe("float64");
    expect(m64.toArray()).toBe(1.5);
  });

  it("keeps dims when requested", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const m = mean(t, 0, true);
    expect(m.shape).toEqual([1, 2]);
    expect(m.toArray()).toEqual([[2, 3]]);
  });

  it("returns NaN for empty full reductions", () => {
    const empty = tensor([], { dtype: "float32" });
    const m = mean(empty);
    expect(m.shape).toEqual([]);
    expect(m.dtype).toBe("float64");
    expect(Number.isNaN(Number(m.toArray()))).toBe(true);
  });

  it("returns NaN for empty axis reductions", () => {
    const emptyAxis = tensor([[], []], { dtype: "float32" });

    const m1 = mean(emptyAxis, 1);
    expect(m1.shape).toEqual([2]);
    expect(m1.toArray()).toEqual([NaN, NaN]);

    const m2 = mean(emptyAxis, 1, true);
    expect(m2.shape).toEqual([2, 1]);
    expect(m2.toArray()).toEqual([[NaN], [NaN]]);
  });
});
