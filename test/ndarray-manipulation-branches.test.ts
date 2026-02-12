import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { concatenate, repeat, split, stack, tile } from "../src/ndarray/ops/manipulation";

describe("ndarray manipulation branches", () => {
  it("concatenate validates inputs and handles strings", () => {
    expect(() => concatenate([])).toThrow(/at least one/);

    const a = tensor([[1, 2]]);
    const b = tensor([[3, 4]]);
    const c = concatenate([a, b], 0);
    expect(c.toArray()).toEqual([
      [1, 2],
      [3, 4],
    ]);

    expect(() => concatenate([a, tensor([1, 2])], 0)).toThrow(/same ndim/i);
    expect(() => concatenate([a, tensor([[1, 2]], { dtype: "int32" })], 0)).toThrow(/same dtype/i);
    expect(() => concatenate([a, tensor([[1, 2, 3]])], 0)).toThrow(/shapes must match/i);
    expect(() => concatenate([a, b], 2)).toThrow(/out of bounds/i);

    const s = tensor(["a", "b"]);
    const s2 = concatenate([s], 0);
    expect(s2.toArray()).toEqual(["a", "b"]);

    const s3 = tensor(["c"]);
    const sOut = concatenate([s, s3], 0);
    expect(sOut.toArray()).toEqual(["a", "b", "c"]);
  });

  it("stack validates inputs and supports negative axis", () => {
    const a = tensor([1, 2]);
    const b = tensor([3, 4]);
    const s = stack([a, b], -1);
    expect(s.shape).toEqual([2, 2]);

    expect(() => stack([])).toThrow(/at least one/);
    expect(() => stack([a, tensor([[1, 2]])], 0)).toThrow(/same ndim/i);
    expect(() => stack([a, tensor([1, 2], { dtype: "int32" })], 0)).toThrow(/same dtype/i);
    expect(() => stack([a, tensor([1, 2, 3])], 0)).toThrow(/same shape/i);
    expect(() => stack([a, b], 3)).toThrow(/out of bounds/i);
  });

  it("splits tensors and validates divisions", () => {
    const t = tensor([1, 2, 3, 4]);
    const parts = split(t, 2);
    expect(parts.length).toBe(2);
    expect(parts[0]?.toArray()).toEqual([1, 2]);

    const parts2 = split(t, [1, 3]);
    expect(parts2.length).toBe(3);

    expect(() => split(t, 3)).toThrow(/not divisible/i);
    expect(() => split(t, 2, 2)).toThrow(/out of bounds/i);
  });

  it("tiles and repeats along axes", () => {
    const t = tensor([[1, 2]]);
    const tiled = tile(t, [2, 2]);
    expect(tiled.shape).toEqual([2, 4]);

    expect(() => tile(t, [])).toThrow(/at least one element/i);

    const r1 = repeat(tensor([1, 2]), 2);
    expect(r1.toArray()).toEqual([1, 1, 2, 2]);

    const r2 = repeat(tensor([[1, 2]]), 3, 1);
    expect(r2.shape).toEqual([1, 6]);

    expect(() => repeat(tensor([1, 2]), 2, 2)).toThrow(/out of bounds/i);
  });

  it("handles string repeats and tiles", () => {
    const s = tensor(["x", "y"]);
    expect(repeat(s, 2).toArray()).toEqual(["x", "x", "y", "y"]);
    expect(tile(s, [2]).toArray()).toEqual(["x", "y", "x", "y"]);
  });
});
