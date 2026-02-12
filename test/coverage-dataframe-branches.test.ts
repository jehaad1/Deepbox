import { describe, expect, it } from "vitest";
import { DataFrame } from "../src/dataframe/DataFrame";

describe("dataframe branch coverage", () => {
  it("constructor validation branches", () => {
    expect(() => new DataFrame({ a: [1], b: [2] }, { columns: ["a", "a"] })).toThrow(
      /duplicate column name/i
    );
    expect(() => new DataFrame({ a: [1], b: [2] }, { columns: ["a", "c"] })).toThrow(/not found/i);
    expect(() => new DataFrame({ a: [1], b: [2, 3] })).toThrow(/length/i);
    // @ts-expect-error Testing invalid input type
    expect(() => new DataFrame({ a: 1 })).toThrow(/array/i);
    expect(() => new DataFrame({ a: [1] }, { index: [0, 1] })).toThrow(/index length/i);
    expect(() => new DataFrame({ a: [1, 2] }, { index: ["x", "x"] })).toThrow(
      /duplicate index label/i
    );
    expect(
      // @ts-expect-error Testing undefined index label
      () => new DataFrame({ a: [1, 2] }, { index: [0, undefined] })
    ).toThrow(/undefined/i);
  });

  it("basic accessors and selection errors", () => {
    const df = new DataFrame({ a: [1, 2], b: [3, 4] }, { index: ["r1", "r2"] });
    expect(() => df.get("c")).toThrow(/not found/i);
    expect(() => df.loc("r3")).toThrow(/not found/i);
    expect(() => df.iloc(5)).toThrow(/out of bounds/i);
    expect(() => df.select(["a", "c"])).toThrow(/not found/i);
    expect(() => df.sort("c")).toThrow(/not found/i);
  });

  it("join branches and overlaps", () => {
    const left = new DataFrame({
      id: [1, 2],
      value: ["a", "b"],
      overlap: [10, 20],
    });
    const right = new DataFrame({
      id: [2, 3],
      overlap: [200, 300],
      other: ["x", "y"],
    });

    expect(() => left.join(right, "missing")).toThrow(/join column/i);
    expect(() => left.join(new DataFrame({ id: [1] }), "id2")).toThrow(/join column/i);

    const inner = left.join(right, "id", "inner");
    const leftJoin = left.join(right, "id", "left");
    const rightJoin = left.join(right, "id", "right");
    const outer = left.join(right, "id", "outer");

    expect(inner.shape[0]).toBe(1);
    expect(leftJoin.shape[0]).toBe(2);
    expect(rightJoin.shape[0]).toBe(2);
    expect(outer.shape[0]).toBe(3);
    expect(outer.columns).toContain("overlap_left");
    expect(outer.columns).toContain("overlap_right");
    expect(outer.toArray()).toEqual([
      [1, "a", 10, null, null],
      [2, "b", 20, 200, "x"],
      [3, null, null, 300, "y"],
    ]);
  });

  it("groupBy agg and numeric validation", () => {
    const df = new DataFrame({
      group: ["A", "A", "B", "B"],
      value: [1, 2, 3, 4],
      note: ["x", null, "y", "z"],
    });

    const grouped = df.groupBy(["group"]);
    const counted = grouped.count();
    expect(counted.shape[0]).toBe(2);
    expect(counted.toArray()).toEqual([
      ["A", 2, 1],
      ["B", 2, 2],
    ]);

    const agg = grouped.agg({
      value: ["sum", "mean", "min", "max", "median", "std", "var"],
      note: ["count", "first", "last"],
    });
    expect(agg.shape[0]).toBe(2);
    expect(agg.get("value_sum").data).toEqual([3, 7]);
    expect(agg.get("value_mean").data).toEqual([1.5, 3.5]);
    expect(agg.get("value_min").data).toEqual([1, 3]);
    expect(agg.get("value_max").data).toEqual([2, 4]);
    expect(agg.get("value_median").data).toEqual([1.5, 3.5]);
    expect(agg.get("value_std").data[0]).toBeCloseTo(Math.SQRT1_2, 4);
    expect(agg.get("value_std").data[1]).toBeCloseTo(Math.SQRT1_2, 4);
    expect(agg.get("value_var").data).toEqual([0.5, 0.5]);
    expect(agg.get("note_count").data).toEqual([1, 2]);
    expect(agg.get("note_first").data).toEqual(["x", "y"]);
    expect(agg.get("note_last").data).toEqual([null, "z"]);

    // @ts-expect-error Testing unsupported aggregation
    expect(() => grouped.agg({ value: "unsupported" })).toThrow(/unsupported/i);

    const mixed = new DataFrame({
      group: ["A", "B"],
      value: [1, 2],
      note: ["x", "y"],
    });
    // Testing invalid aggregation on non-numeric column
    expect(() => mixed.groupBy("group").agg({ note: "sum" })).toThrow(/numbers/i);
  });
});
