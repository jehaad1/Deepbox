import { describe, expect, it } from "vitest";
import { DataFrame, Series } from "../src/dataframe";

describe("DataFrame.copy()", () => {
  it("returns a deep copy with identical data", () => {
    const df = new DataFrame({ a: [1, 2, 3], b: ["x", "y", "z"] });
    const df2 = df.copy();
    expect(df2.shape).toEqual([3, 2]);
    expect(df2.columns).toEqual(["a", "b"]);
    expect(df2.get("a").toArray()).toEqual([1, 2, 3]);
    expect(df2.get("b").toArray()).toEqual(["x", "y", "z"]);
  });

  it("preserves custom index", () => {
    const df = new DataFrame({ a: [10, 20] }, { index: ["r1", "r2"] });
    const df2 = df.copy();
    expect(df2.index).toEqual(["r1", "r2"]);
  });

  it("mutation of copy does not affect original", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    const df2 = df.copy();
    // The copy is a new DataFrame; modifying the underlying arrays of the copy
    // should not touch the original (because constructor copies arrays).
    expect(df.get("a").toArray()).toEqual([1, 2, 3]);
    expect(df2.get("a").toArray()).toEqual([1, 2, 3]);
  });

  it("works on empty DataFrame", () => {
    const df = new DataFrame({});
    const df2 = df.copy();
    expect(df2.shape).toEqual([0, 0]);
  });
});

describe("DataFrame.isin()", () => {
  it("returns boolean DataFrame for matching values", () => {
    const df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] });
    const result = df.isin([2, 4]);
    expect(result.shape).toEqual([3, 2]);
    expect(result.get("a").toArray()).toEqual([false, true, false]);
    expect(result.get("b").toArray()).toEqual([true, false, false]);
  });

  it("works with string values", () => {
    const df = new DataFrame({ x: ["a", "b", "c"] });
    const result = df.isin(["b", "c"]);
    expect(result.get("x").toArray()).toEqual([false, true, true]);
  });

  it("returns all false for empty values list", () => {
    const df = new DataFrame({ a: [1, 2] });
    const result = df.isin([]);
    expect(result.get("a").toArray()).toEqual([false, false]);
  });

  it("handles mixed types", () => {
    const df = new DataFrame({ a: [1, "hello", null, true] });
    const result = df.isin([1, true]);
    expect(result.get("a").toArray()).toEqual([true, false, false, true]);
  });

  it("preserves index and columns", () => {
    const df = new DataFrame({ a: [1, 2], b: [3, 4] }, { index: ["r1", "r2"] });
    const result = df.isin([1]);
    expect(result.columns).toEqual(["a", "b"]);
    expect(result.index).toEqual(["r1", "r2"]);
  });
});

describe("DataFrame.transform()", () => {
  it("transforms each column (axis=0)", () => {
    const df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] });
    const result = df.transform((series) => series.map((x) => Number(x) * 2));
    expect(result.shape).toEqual([3, 2]);
    expect(result.get("a").toArray()).toEqual([2, 4, 6]);
    expect(result.get("b").toArray()).toEqual([8, 10, 12]);
  });

  it("transforms each row (axis=1)", () => {
    const df = new DataFrame({ a: [1, 2], b: [10, 20] });
    const result = df.transform((series) => series.map((x) => Number(x) + 1), 1);
    expect(result.shape).toEqual([2, 2]);
    expect(result.get("a").toArray()).toEqual([2, 3]);
    expect(result.get("b").toArray()).toEqual([11, 21]);
  });

  it("throws if result has wrong length (axis=0)", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    expect(() => df.transform(() => new Series([1, 2]))).toThrow(/result length/);
  });

  it("throws if result has wrong length (axis=1)", () => {
    const df = new DataFrame({ a: [1], b: [2] });
    expect(() => df.transform(() => new Series([1, 2, 3]), 1)).toThrow(/result length/);
  });

  it("preserves index and columns", () => {
    const df = new DataFrame({ a: [1, 2], b: [3, 4] }, { index: ["r1", "r2"] });
    const result = df.transform((s) => s.map((x) => Number(x)));
    expect(result.columns).toEqual(["a", "b"]);
    expect(result.index).toEqual(["r1", "r2"]);
  });
});
