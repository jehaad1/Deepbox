import { describe, expect, it } from "vitest";
import { tensor, zeros } from "../src/ndarray";
import { prod, sum } from "../src/ndarray/ops/reduction";

describe("ndarray reduction overflow and edge branches", () => {
  it("throws on int64 sum overflow", () => {
    const big = new Array(2000).fill(9_000_000_000_000_000);
    const t = tensor(big, { dtype: "int64" });
    expect(() => sum(t)).toThrow(/int64 sum overflow/i);
  });

  it("throws on int64 prod overflow and handles empty prod", () => {
    const t = tensor(new BigInt64Array([9_223_372_036_854_775_807n, 2n]), {
      dtype: "int64",
    });
    expect(() => prod(t)).toThrow(/int64 prod overflow/i);

    const empty = zeros([0]);
    expect(prod(empty).toArray()).toBe(1);
  });

  it("handles keepdims on sum", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const out = sum(t, 1, true);
    expect(out.shape).toEqual([2, 1]);
    expect(out.toArray()).toEqual([[3], [7]]);
  });
});
