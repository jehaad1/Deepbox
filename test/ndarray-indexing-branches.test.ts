import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { gather, slice } from "../src/ndarray/tensor/indexing";

describe("ndarray indexing branch coverage", () => {
  it("slices with numbers, negative indices, and steps", () => {
    const t2d = tensor([
      [1, 2],
      [3, 4],
    ]);
    const scalar = slice(t2d, -1, 0);
    expect(scalar.shape).toEqual([]);
    expect(scalar.toArray()).toBe(3);

    const t1d = tensor([1, 2, 3, 4]);
    const stepped = slice(t1d, { start: 0, end: 4, step: 2 });
    expect(stepped.toArray()).toEqual([1, 3]);
  });

  it("throws for too many indices and invalid steps", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(() => slice(t, 0, 0, 0)).toThrow(/Too many indices/i);
    expect(() => slice(t, { start: 0, end: 2, step: 0 })).toThrow(/step/i);
    expect(() => slice(t, { start: 0, end: 2, step: 0.5 })).toThrow(/step/i);
  });

  it("slices string tensors with steps", () => {
    const ts = tensor([
      ["a", "b", "c"],
      ["d", "e", "f"],
    ]);
    const out = slice(ts, 1, { start: 0, end: 3, step: 2 });
    expect(out.shape).toEqual([2]);
    expect(out.toArray()).toEqual(["d", "f"]);
  });

  it("gathers numeric and string tensors with validation", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const idx = tensor([1, 0]);
    const out = gather(t, idx, 0);
    expect(out.shape).toEqual([2, 2]);
    expect(out.toArray()).toEqual([
      [3, 4],
      [1, 2],
    ]);

    const ts = tensor([
      ["a", "b"],
      ["c", "d"],
    ]);
    const strOut = gather(ts, idx, 0);
    expect(strOut.toArray()).toEqual([
      ["c", "d"],
      ["a", "b"],
    ]);

    expect(() => gather(t, tensor(["0"]), 0)).toThrow(/numeric indices/i);
    expect(() => gather(t, idx, 2)).toThrow(/axis 2/i);
    expect(() => gather(t, tensor([2]), 0)).toThrow(/out of bounds/i);
  });
});
