import { describe, expect, it } from "vitest";
import { prod, sum, tensor, zeros } from "../src/ndarray";
import { Tensor } from "../src/ndarray/tensor/Tensor";

const INT64_MAX = (1n << 63n) - 1n;

describe("ndarray reduction branches", () => {
  it("validates sum axis errors and int64 overflow", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => sum(t, 2)).toThrow(/out of bounds/i);

    const empty = zeros([0, 2]);
    const reduced = sum(empty, 0);
    expect(reduced.shape).toEqual([2]);
    expect(reduced.dtype).toBe("float64");
    expect(reduced.toArray()).toEqual([0, 0]);

    const reducedKeep = sum(empty, 0, true);
    expect(reducedKeep.shape).toEqual([1, 2]);
    expect(reducedKeep.dtype).toBe("float64");
    expect(reducedKeep.toArray()).toEqual([[0, 0]]);

    const big = tensor(new BigInt64Array([INT64_MAX, 1n]), { dtype: "int64" });
    expect(() => sum(big)).toThrow(/overflow/i);
  });

  it("returns additive identity for empty-axis int64 sum", () => {
    const emptyInt64 = Tensor.fromTypedArray({
      data: new BigInt64Array(0),
      shape: [0, 2],
      dtype: "int64",
      device: "cpu",
    });

    const reduced = sum(emptyInt64, 0);
    expect(reduced.shape).toEqual([2]);
    expect(reduced.dtype).toBe("int64");
    expect(reduced.toArray()).toEqual([0n, 0n]);
  });

  it("handles sum keepdims with int64 axis", () => {
    const data = tensor(new BigInt64Array([2n, 3n, 5n, 7n]), { dtype: "int64" }).reshape([2, 2]);
    const out = sum(data, 1, true);
    expect(out.shape).toEqual([2, 1]);
    expect(out.toArray()).toEqual([[5n], [12n]]);
  });

  it("validates prod dtype and overflow", () => {
    expect(() => prod(tensor(["a", "b"]))).toThrow(/string/i);

    const big = tensor(new BigInt64Array([INT64_MAX, 2n]), { dtype: "int64" });
    expect(() => prod(big)).toThrow(/overflow/i);

    const t = tensor([
      [2, 3],
      [4, 5],
    ]);
    const p = prod(t, [0], true);
    expect(p.shape).toEqual([1, 2]);
    expect(p.toArray()).toEqual([[8, 15]]);
  });

  it("rejects duplicate axes for prod", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => prod(t, [0, 0])).toThrow(/duplicate axis/i);
  });
});
