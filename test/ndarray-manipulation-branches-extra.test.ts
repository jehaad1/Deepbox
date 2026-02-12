import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { concatenate, repeat, split, stack, tile } from "../src/ndarray/ops/manipulation";
import { Tensor } from "../src/ndarray/tensor/Tensor";

describe("ndarray manipulation branch coverage extras", () => {
  it("concatenate covers error paths and dtype branches", () => {
    expect(() => concatenate([])).toThrow(/at least one/i);

    const a = tensor([1, 2]);
    const b = tensor([3, 4], { dtype: "int32" });
    expect(() => concatenate([a, b])).toThrow(/dtype/i);

    const c = tensor([
      [1, 2],
      [3, 4],
    ]);
    const d = tensor([
      [5, 6, 7],
      [8, 9, 10],
    ]);
    expect(() => concatenate([c, d], 0)).toThrow(/shapes must match/i);
    expect(() => concatenate([a, b], 2)).toThrow(/out of bounds/i);

    const s = Tensor.fromStringArray({
      data: ["a", "b"],
      shape: [2],
      device: "cpu",
    });
    const sOut = concatenate([s]);
    expect(sOut.shape).toEqual([2]);
    expect(String(sOut.data[0])).toBe("a");

    const big = Tensor.fromTypedArray({
      data: new BigInt64Array([1n, 2n]),
      shape: [2],
      dtype: "int64",
      device: "cpu",
    });
    const bigOut = concatenate([big, big], 0);
    expect(bigOut.shape).toEqual([4]);
  });

  it("stack covers axis normalization and dtype branches", () => {
    expect(() => stack([])).toThrow(/at least one/i);

    const a = tensor([1, 2]);
    const b = tensor([3, 4]);
    const out = stack([a, b], -1);
    expect(out.shape).toEqual([2, 2]);

    const bad = tensor([[1, 2]]);
    expect(() => stack([a, bad] as never)).toThrow(/ndim/i);

    const badDtype = tensor([1, 2], { dtype: "int32" });
    expect(() => stack([a, badDtype])).toThrow(/dtype/i);

    expect(() => stack([a, b], 3)).toThrow(/out of bounds/i);

    const s = Tensor.fromStringArray({
      data: ["x", "y"],
      shape: [2],
      device: "cpu",
    });
    const sOut = stack([s, s], 0);
    expect(sOut.shape).toEqual([2, 2]);
    expect(sOut.toArray()).toEqual([
      ["x", "y"],
      ["x", "y"],
    ]);

    const big = Tensor.fromTypedArray({
      data: new BigInt64Array([1n, 2n]),
      shape: [2],
      dtype: "int64",
      device: "cpu",
    });
    const bigOut = stack([big, big], 0);
    expect(bigOut.shape).toEqual([2, 2]);
  });

  it("split covers axis errors and string dtype path", () => {
    const t = tensor([1, 2, 3, 4]);
    expect(() => split(t, 3, 1)).toThrow(/out of bounds/i);
    expect(() => split(t, 3)).toThrow(/not divisible/i);

    const parts = split(t, [2]);
    expect(parts.length).toBe(2);
    expect(parts[0]?.shape).toEqual([2]);
    expect(parts[1]?.shape).toEqual([2]);

    const s = Tensor.fromStringArray({
      data: ["a", "b", "c", "d"],
      shape: [4],
      device: "cpu",
    });
    const sParts = split(s, [1, 3]);
    expect(sParts.length).toBe(3);
    expect(sParts[0]?.toArray()).toEqual(["a"]);
    expect(sParts[1]?.toArray()).toEqual(["b", "c"]);
    expect(sParts[2]?.toArray()).toEqual(["d"]);
  });

  it("tile covers reps validation and dtype branches", () => {
    const t = tensor([1, 2]);
    expect(() => tile(t, [])).toThrow(/at least one/i);

    const out = tile(t, [2, 3]);
    expect(out.shape).toEqual([2, 6]);

    const s = Tensor.fromStringArray({
      data: ["x", "y"],
      shape: [2],
      device: "cpu",
    });
    expect(tile(s, [2]).toArray()).toEqual(["x", "y", "x", "y"]);

    const big = Tensor.fromTypedArray({
      data: new BigInt64Array([1n, 2n]),
      shape: [2],
      dtype: "int64",
      device: "cpu",
    });
    const bigOut = tile(big, [2]);
    expect(bigOut.shape).toEqual([4]);
  });

  it("repeat covers flatten, axis, and dtype branches", () => {
    const t = tensor([1, 2]);
    const out = repeat(t, 3);
    expect(out.shape).toEqual([6]);

    const t2 = tensor([
      [1, 2],
      [3, 4],
    ]);
    const outAxis = repeat(t2, 2, -1);
    expect(outAxis.shape).toEqual([2, 4]);
    expect(() => repeat(t2, 2, 3)).toThrow(/out of bounds/i);

    const s = Tensor.fromStringArray({
      data: ["a", "b"],
      shape: [2],
      device: "cpu",
    });
    expect(repeat(s, 2).toArray()).toEqual(["a", "a", "b", "b"]);

    const big = Tensor.fromTypedArray({
      data: new BigInt64Array([1n, 2n]),
      shape: [2],
      dtype: "int64",
      device: "cpu",
    });
    const bigOut = repeat(big, 2);
    expect(bigOut.shape).toEqual([4]);
  });
});
