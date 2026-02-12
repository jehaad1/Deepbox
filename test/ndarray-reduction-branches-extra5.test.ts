import { describe, expect, it } from "vitest";
import { all, any, diff, tensor } from "../src/ndarray";

describe("ndarray reduction extra branches", () => {
  it("any/all with keepdims and multi-axis", () => {
    const t = tensor([
      [0, 1],
      [0, 0],
    ]);
    const a = any(t, [0, 1], true);
    expect(a.shape).toEqual([1, 1]);
    expect(a.toArray()).toEqual([[1]]);

    const b = all(t, [0, 1], true);
    expect(b.shape).toEqual([1, 1]);
    expect(b.toArray()).toEqual([[0]]);
  });

  it("diff numeric on axis 0 for 2D", () => {
    const t = tensor([
      [1, 2],
      [4, 6],
      [9, 12],
    ]);
    const d = diff(t, 1, 0);
    expect(d.shape).toEqual([2, 2]);
    expect(d.toArray()).toEqual([
      [3, 4],
      [5, 6],
    ]);
  });
});
