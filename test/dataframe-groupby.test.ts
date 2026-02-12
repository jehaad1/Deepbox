import { describe, expect, it } from "vitest";
import { DataFrame } from "../src/dataframe";

describe("DataFrame groupby tests", () => {
  it("should throw when grouping by missing column", () => {
    const df = new DataFrame({ a: [1, 2], b: [3, 4] });
    expect(() => df.groupBy("missing")).toThrow(/not found/i);
  });

  it("should count non-null values correctly", () => {
    const df = new DataFrame({
      a: ["x", "x", "y", "y"],
      b: [1, 2, null, 4],
    });
    const result = df.groupBy("a").count();
    expect(result.columns).toEqual(["a", "b"]);
    expect(result.toArray()).toEqual([
      ["x", 2],
      ["y", 1],
    ]);
  });

  it("agg should count non-null values correctly", () => {
    const df = new DataFrame({
      a: ["x", "x", "y", "y"],
      b: [1, 2, null, 4],
    });
    const result = df.groupBy("a").agg({ b: "count" });
    expect(result.columns).toEqual(["a", "b"]);
    expect(result.toArray()).toEqual([
      ["x", 2],
      ["y", 1],
    ]);
  });

  it("should calculate sum, mean, min, max", () => {
    const df = new DataFrame({
      group: ["A", "A", "B", "B"],
      val: [1, 3, 2, 4],
    });
    const grouped = df.groupBy("group");

    expect(grouped.sum().toArray()).toEqual([
      ["A", 4],
      ["B", 6],
    ]);
    expect(grouped.mean().toArray()).toEqual([
      ["A", 2],
      ["B", 3],
    ]);
    expect(grouped.min().toArray()).toEqual([
      ["A", 1],
      ["B", 2],
    ]);
    expect(grouped.max().toArray()).toEqual([
      ["A", 3],
      ["B", 4],
    ]);
  });

  it("should handle nulls in aggregations", () => {
    const df = new DataFrame({
      group: ["A", "A", "B", "B"],
      val: [1, null, 2, 4],
    });
    const grouped = df.groupBy("group");

    // A: [1, null] -> sum=1, count=1, mean=1
    // B: [2, 4] -> sum=6, count=2, mean=3
    expect(grouped.sum().get("val").data).toEqual([1, 6]);
    expect(grouped.count().get("val").data).toEqual([1, 2]);
    expect(grouped.mean().get("val").data).toEqual([1, 3]);
  });
});
