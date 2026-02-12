import { describe, expect, it } from "vitest";
import { tensor, zeros } from "../src/ndarray";
import { median, trimMean } from "../src/stats/descriptive";

describe("stats descriptive branches extra", () => {
  it("computes median with keepdims and axis", () => {
    const t = tensor([
      [1, 3, 2],
      [4, 6, 5],
    ]);
    const m = median(t, 1, true);
    expect(m.shape).toEqual([2, 1]);
    expect(m.toArray()).toEqual([[2], [5]]);
  });

  it("throws median on empty axis", () => {
    const t = zeros([0, 2]);
    expect(() => median(t, 0)).toThrow(/empty axis/i);
  });

  it("computes trimMean along axis and validates proportion", () => {
    const t = tensor([
      [1, 2, 100],
      [3, 4, 200],
    ]);
    const tm = trimMean(t, 1 / 3, 1);
    expect(tm.shape).toEqual([2]);
    expect(tm.toArray()).toEqual([2, 4]);

    expect(() => trimMean(t, 0.5)).toThrow(/proportiontocut/i);
  });
});
