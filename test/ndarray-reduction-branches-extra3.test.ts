import { describe, expect, it } from "vitest";
import { diff, max, min, tensor, zeros } from "../src/ndarray";

describe("ndarray reduction branches extra 3", () => {
  it("handles diff edge cases", () => {
    const t = tensor([1, 2, 4]);
    const d0 = diff(t, 0);
    expect(d0.toArray()).toEqual([1, 2, 4]);

    const short = tensor([5]);
    const d1 = diff(short, 1);
    expect(d1.toArray()).toEqual([]);

    expect(() => diff(tensor(["a"]))).toThrow(/string/i);
    expect(() => diff(t, -1)).toThrow(/n must be/);
  });

  it("handles min/max with bigint and keepdims", () => {
    const t = tensor(new BigInt64Array([3n, 1n, 4n, 2n]), { dtype: "int64" }).reshape([2, 2]);
    const mn = min(t, 1, true);
    const mx = max(t, 0, true);
    expect(mn.shape).toEqual([2, 1]);
    expect(mn.toArray()).toEqual([[1n], [2n]]);
    expect(mx.shape).toEqual([1, 2]);
    expect(mx.toArray()).toEqual([[4n, 2n]]);
  });

  it("min/max reject empty axis", () => {
    const t = zeros([0, 2]);
    expect(() => min(t, 0)).toThrow(/at least one element/i);
    expect(() => max(t, 0)).toThrow(/at least one element/i);
  });
});
