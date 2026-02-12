import { describe, expect, it } from "vitest";
import { std, tensor, variance, zeros } from "../src/ndarray";

describe("ndarray variance/std branches", () => {
  it("validates variance inputs and ddof", () => {
    expect(() => variance(tensor(["a"]))).toThrow(/string/i);
    expect(() => variance(zeros([0]))).toThrow(/at least one element/i);
    expect(() => variance(tensor([1, 2]), undefined, false, 2)).toThrow(/ddof/i);

    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => variance(t, 0, false, 2)).toThrow(/ddof/i);
  });

  it("computes variance with keepdims and bigint", () => {
    const t = tensor(new BigInt64Array([1n, 3n, 5n, 7n]), { dtype: "int64" }).reshape([2, 2]);
    const v = variance(t, 1, true);
    expect(v.shape).toEqual([2, 1]);
    expect(v.toArray()).toEqual([[1], [1]]);
  });

  it("computes std from variance", () => {
    const t = tensor([1, 2, 3]);
    const s = std(t);
    expect(Number(s.toArray())).toBeCloseTo(Math.sqrt(2 / 3), 6);
  });
});
