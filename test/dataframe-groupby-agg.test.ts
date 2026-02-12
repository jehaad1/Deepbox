import { describe, expect, it } from "vitest";

import { DataFrame } from "../src/dataframe/DataFrame";

describe("DataFrameGroupBy.agg/count", () => {
  it("aggregates multiple functions per column with stable group order", () => {
    const df = new DataFrame({
      cat: ["A", "A", "B", "B", "B"],
      val: [1, 2, 3, 4, 5],
      label: ["x", "y", "z", "w", "q"],
    });

    const result = df.groupBy("cat").agg({
      val: ["sum", "mean", "min", "max"],
      label: "first",
    });

    expect(result.columns).toEqual(["cat", "val_sum", "val_mean", "val_min", "val_max", "label"]);
    expect(result.toArray()).toEqual([
      ["A", 3, 1.5, 1, 2, "x"],
      ["B", 12, 4, 3, 5, "z"],
    ]);
  });

  it("supports count/last and multiple group keys", () => {
    const df = new DataFrame({
      cat: ["A", "A", "A", "B", "B"],
      sub: [1, 1, 2, 1, 1],
      val: [10, 20, 30, 40, 50],
    });

    const result = df.groupBy(["cat", "sub"]).agg({
      val: ["count", "last"],
    });

    expect(result.columns).toEqual(["cat", "sub", "val_count", "val_last"]);
    expect(result.toArray()).toEqual([
      ["A", 1, 2, 20],
      ["A", 2, 1, 30],
      ["B", 1, 2, 50],
    ]);
  });

  it("counts only non-null values for agg count", () => {
    const df = new DataFrame({
      cat: ["A", "A", "B", "B"],
      val: [10, null, undefined, 40],
    });

    const result = df.groupBy("cat").agg({ val: "count" });
    expect(result.columns).toEqual(["cat", "val"]);
    expect(result.toArray()).toEqual([
      ["A", 1],
      ["B", 1],
    ]);
  });

  it("supports median/std/var aggregations", () => {
    const df = new DataFrame({
      cat: ["A", "A", "B", "B"],
      val: [1, 3, 2, 4],
    });

    const result = df.groupBy("cat").agg({
      val: ["median", "std", "var"],
    });

    expect(result.columns).toEqual(["cat", "val_median", "val_std", "val_var"]);
    const rows = result.toArray();

    // Explicitly check shape and types instead of casting
    expect(rows[0]?.[0]).toBe("A");
    expect(rows[0]?.[1]).toBe(2);
    expect(rows[1]?.[0]).toBe("B");
    expect(rows[1]?.[1]).toBe(3);
    expect(result.get("val_std").data[0]).toBeCloseTo(Math.sqrt(2), 6);
    expect(result.get("val_std").data[1]).toBeCloseTo(Math.sqrt(2), 6);
    expect(result.get("val_var").data).toEqual([2, 2]);
  });

  it("count() returns per-column non-null counts", () => {
    const df = new DataFrame({
      cat: ["A", "A", "B"],
      val: [1, 2, 3],
      tag: ["x", "y", "z"],
    });

    const result = df.groupBy("cat").count();
    expect(result.columns).toEqual(["cat", "val", "tag"]);
    expect(result.toArray()).toEqual([
      ["A", 2, 2],
      ["B", 1, 1],
    ]);
  });

  it("throws for numeric aggregations on non-numeric values", () => {
    const df = new DataFrame({
      key: ["A", "A", "B"],
      val: ["a", "b", "c"],
    });
    const grouped = df.groupBy("key");

    // Testing invalid aggregation on non-numeric column
    expect(() => grouped.agg({ val: "sum" })).toThrow(/numbers/i);
    expect(() => grouped.agg({ val: "mean" })).toThrow(/numbers/i);
    expect(() => grouped.agg({ val: "median" })).toThrow(/numbers/i);
    expect(() => grouped.agg({ val: "min" })).toThrow(/numbers/i);
    expect(() => grouped.agg({ val: "max" })).toThrow(/numbers/i);
    expect(() => grouped.agg({ val: "std" })).toThrow(/numbers/i);
    expect(() => grouped.agg({ val: "var" })).toThrow(/numbers/i);
  });
});
