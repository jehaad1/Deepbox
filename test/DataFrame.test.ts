import { describe, expect, it } from "vitest";
import { DataFrame } from "../src/dataframe";

describe("DataFrame", () => {
  it("should create DataFrame from object", () => {
    const df = new DataFrame({
      name: ["Alice", "Bob", "Charlie"],
      age: [25, 30, 35],
      city: ["NYC", "LA", "Chicago"],
    });

    expect(df.shape).toEqual([3, 3]);
    expect(df.columns).toEqual(["name", "age", "city"]);
    expect(df.index).toEqual([0, 1, 2]);
    expect(df.toArray()).toEqual([
      ["Alice", 25, "NYC"],
      ["Bob", 30, "LA"],
      ["Charlie", 35, "Chicago"],
    ]);
  });

  it("should get column as Series", () => {
    const df = new DataFrame({
      name: ["Alice", "Bob"],
      age: [25, 30],
    });

    const ages = df.get("age");
    expect(ages.data).toEqual([25, 30]);
  });

  it("should select subset of columns", () => {
    const df = new DataFrame({
      a: [1, 2, 3],
      b: [4, 5, 6],
      c: [7, 8, 9],
    });

    const subset = df.select(["a", "c"]);
    expect(subset.columns).toEqual(["a", "c"]);
    expect(subset.shape).toEqual([3, 2]);
    expect(subset.toArray()).toEqual([
      [1, 7],
      [2, 8],
      [3, 9],
    ]);
  });

  it("should filter rows", () => {
    const df = new DataFrame({
      name: ["Alice", "Bob", "Charlie"],
      age: [25, 30, 35],
    });

    const filtered = df.filter((row: Record<string, unknown>) => {
      const age = row.age;
      return typeof age === "number" && age >= 30;
    });
    expect(filtered.shape).toEqual([2, 2]);
    expect(filtered.get("name").data).toEqual(["Bob", "Charlie"]);
  });

  it("should head and tail", () => {
    const df = new DataFrame({ x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] });
    const head = df.head(3);
    const tail = df.tail(3);
    expect(head.shape).toEqual([3, 1]);
    expect(head.get("x").data).toEqual([1, 2, 3]);
    expect(tail.shape).toEqual([3, 1]);
    expect(tail.get("x").data).toEqual([8, 9, 10]);
  });

  it("should describe statistics", () => {
    const df = new DataFrame({
      age: [25, 30, 35, 40],
      salary: [50000, 60000, 70000, 80000],
    });
    const stats = df.describe();
    expect(stats.index).toEqual(["count", "mean", "std", "min", "25%", "50%", "75%", "max"]);
    expect(stats.columns).toEqual(["age", "salary"]);
    // Check mean
    expect(stats.get("age").get("mean")).toBe(32.5);
  });

  it("should group by and aggregate", () => {
    const df = new DataFrame({
      category: ["A", "B", "A", "B"],
      value: [1, 2, 3, 4],
    });
    const grouped = df.groupBy("category");
    const result = grouped.agg({ value: "sum" });
    expect(result.columns).toEqual(["category", "value"]);
    expect(result.toArray()).toEqual([
      ["A", 4],
      ["B", 6],
    ]);
  });

  it("should support multiple aggregations per column", () => {
    const df = new DataFrame({
      category: ["A", "B", "A", "B"],
      value: [1, 2, 3, 4],
    });

    const result = df.groupBy("category").agg({ value: ["sum", "mean", "min", "max"] });
    expect(result.columns).toEqual([
      "category",
      "value_sum",
      "value_mean",
      "value_min",
      "value_max",
    ]);
    expect(result.toArray()).toEqual([
      ["A", 4, 2, 1, 3],
      ["B", 6, 3, 2, 4],
    ]);
  });

  it("should throw for mismatched column lengths", () => {
    expect(() => new DataFrame({ a: [1, 2, 3], b: [4, 5] })).toThrow(/must match row count/);
  });

  it("should throw for index length mismatch", () => {
    expect(() => new DataFrame({ a: [1, 2, 3] }, { index: ["r1", "r2"] })).toThrow(
      /Index length \(\d+\) must match row count \(\d+\)/
    );
  });

  it("should throw for duplicate index labels", () => {
    expect(() => new DataFrame({ a: [1, 2] }, { index: ["dup", "dup"] })).toThrow(
      /Duplicate index label/
    );
  });

  it("should enforce concat axis=0 column equality", () => {
    const df1 = new DataFrame({ a: [1, 2], b: [3, 4] });
    const df2 = new DataFrame({ a: [5, 6] });
    expect(() => df1.concat(df2, 0)).toThrow(/missing column/);
  });

  it("should align indices for concat axis=1 with mismatched rows", () => {
    const df1 = new DataFrame({ a: [1, 2] });
    const df2 = new DataFrame({ b: [3, 4, 5] });
    const result = df1.concat(df2, 1);
    expect(result.shape).toEqual([3, 2]);
    expect(result.toArray()).toEqual([
      [1, 3],
      [2, 4],
      [null, 5],
    ]);
  });

  it("should apply function across rows with axis=1", () => {
    const df = new DataFrame({ a: [1, 2], b: [3, 4] });
    const out = df.apply(
      (row) =>
        row.map((x) => {
          if (typeof x !== "number") throw new Error("Expected number");
          return x * 2;
        }),
      1
    );
    expect(out.toArray()).toEqual([
      [2, 6],
      [4, 8],
    ]);
  });
});
