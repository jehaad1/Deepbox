import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { concatenate, repeat, split, stack, tile } from "../src/ndarray/ops/manipulation";

describe("ndarray manipulation branches extra 2", () => {
  it("concatenate returns copy for single tensor", () => {
    const t = tensor([1, 2, 3]);
    const out = concatenate([t]);
    expect(out.toArray()).toEqual([1, 2, 3]);
  });

  it("split supports indices array and negative axis", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const parts = split(t, [1, 2], -1);
    expect(parts.length).toBe(3);
    expect(parts[0]?.shape).toEqual([2, 1]);
  });

  it("repeat supports negative axis and tile with longer reps", () => {
    const t = tensor([[1, 2]]);
    const r = repeat(t, 2, -1);
    expect(r.shape).toEqual([1, 4]);

    const tiled = tile(t, [2, 2, 1]);
    expect(tiled.shape).toEqual([2, 2, 2]);
  });

  it("stack supports axis at end", () => {
    const a = tensor([1, 2]);
    const b = tensor([3, 4]);
    const s = stack([a, b], 1);
    expect(s.shape).toEqual([2, 2]);
  });
});
