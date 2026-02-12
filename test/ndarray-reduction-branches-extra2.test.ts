import { describe, expect, it } from "vitest";
import { all, any, diff, tensor } from "../src/ndarray";

describe("ndarray reduction branches extra", () => {
  it("handles diff for int64", () => {
    const t = tensor(new BigInt64Array([1n, 4n, 9n, 16n]), { dtype: "int64" });
    const d = diff(t, 2);
    expect(d.toArray()).toEqual([2n, 2n]);
  });

  it("handles any/all with keepdims and bigint", () => {
    const t = tensor(new BigInt64Array([0n, 1n, 0n, 2n]), { dtype: "int64" }).reshape([2, 2]);
    const any0 = any(t, 0, true);
    expect(any0.shape).toEqual([1, 2]);
    expect(any0.toArray()).toEqual([[0, 1]]);

    const all1 = all(t, 1);
    expect(all1.toArray()).toEqual([0, 0]);
  });

  it("rejects any/all on string dtype", () => {
    const s = tensor(["a", "b"]);
    expect(() => any(s)).toThrow(/string/i);
    expect(() => all(s)).toThrow(/string/i);
  });
});
