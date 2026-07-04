import { describe, expect, it } from "vitest";
import { DataFrame } from "../src/dataframe";

describe("DataFrame.eval()", () => {
  it("creates a new column from arithmetic expression", () => {
    const df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] });
    const result = df.eval("c = a + b");
    expect(result.columns).toContain("c");
    expect(result.get("c").toArray()).toEqual([5, 7, 9]);
  });

  it("supports subtraction", () => {
    const df = new DataFrame({ a: [10, 20, 30], b: [1, 2, 3] });
    const result = df.eval("c = a - b");
    expect(result.get("c").toArray()).toEqual([9, 18, 27]);
  });

  it("supports multiplication", () => {
    const df = new DataFrame({ a: [2, 3, 4], b: [5, 6, 7] });
    const result = df.eval("c = a * b");
    expect(result.get("c").toArray()).toEqual([10, 18, 28]);
  });

  it("supports division", () => {
    const df = new DataFrame({ a: [10, 20, 30], b: [2, 4, 5] });
    const result = df.eval("c = a / b");
    expect(result.get("c").toArray()).toEqual([5, 5, 6]);
  });

  it("supports power operator", () => {
    const df = new DataFrame({ a: [2, 3, 4] });
    const result = df.eval("c = a ** 2");
    expect(result.get("c").toArray()).toEqual([4, 9, 16]);
  });

  it("preserves original columns when adding new one", () => {
    const df = new DataFrame({ a: [1, 2], b: [3, 4] });
    const result = df.eval("c = a + b");
    expect(result.columns).toEqual(["a", "b", "c"]);
    expect(result.get("a").toArray()).toEqual([1, 2]);
    expect(result.get("b").toArray()).toEqual([3, 4]);
  });

  it("overwrites existing column", () => {
    const df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] });
    const result = df.eval("a = a + b");
    expect(result.get("a").toArray()).toEqual([5, 7, 9]);
  });

  it("filters rows with > comparison", () => {
    const df = new DataFrame({ a: [1, 2, 3, 4, 5] });
    const result = df.eval("a > 3");
    expect(result.shape).toEqual([2, 1]);
    expect(result.get("a").toArray()).toEqual([4, 5]);
  });

  it("filters rows with < comparison", () => {
    const df = new DataFrame({ a: [1, 2, 3, 4, 5] });
    const result = df.eval("a < 3");
    expect(result.shape).toEqual([2, 1]);
    expect(result.get("a").toArray()).toEqual([1, 2]);
  });

  it("filters with expression involving multiple columns", () => {
    const df = new DataFrame({ a: [1, 2, 3], b: [10, 20, 30] });
    const result = df.eval("a + b > 15");
    expect(result.shape[0]).toBe(2);
  });

  it("handles constants in expression", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    const result = df.eval("c = a + 10");
    expect(result.get("c").toArray()).toEqual([11, 12, 13]);
  });

  it("handles parenthesized expressions", () => {
    const df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] });
    const result = df.eval("c = (a + b) * 2");
    expect(result.get("c").toArray()).toEqual([10, 14, 18]);
  });

  it("throws for empty expression", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    expect(() => df.eval("")).toThrow();
  });
});

describe("DataFrame.crosstab()", () => {
  it("computes cross-tabulation counts", () => {
    const df = new DataFrame({
      gender: ["M", "F", "M", "F", "M"],
      handed: ["R", "R", "L", "R", "R"],
    });
    const ct = df.crosstab("gender", "handed");
    // M: R=2, L=1; F: R=2, L=0
    expect(ct.shape[0]).toBe(2); // 2 genders
    expect(ct.columns.length).toBe(2); // R and L
  });

  it("correct counts for simple case", () => {
    const df = new DataFrame({
      a: ["x", "x", "y", "y"],
      b: ["1", "2", "1", "1"],
    });
    const ct = df.crosstab("a", "b");
    // x: 1->1, 2->1; y: 1->2, 2->0
    expect(ct.shape).toEqual([2, 2]);

    // Row x
    const row0 = ct.iloc(0);
    const xCounts = Object.values(row0);
    // x has: one "1" and one "2"
    expect(xCounts).toContain(1);

    // Row y
    const row1 = ct.iloc(1);
    const yCounts = Object.values(row1);
    // y has: two "1" and zero "2"
    expect(yCounts).toContain(2);
    expect(yCounts).toContain(0);
  });

  it("throws for missing row column", () => {
    const df = new DataFrame({ a: [1, 2], b: [3, 4] });
    expect(() => df.crosstab("missing", "b")).toThrow();
  });

  it("throws for missing col column", () => {
    const df = new DataFrame({ a: [1, 2], b: [3, 4] });
    expect(() => df.crosstab("a", "missing")).toThrow();
  });

  it("works with numeric values", () => {
    const df = new DataFrame({
      row: [1, 1, 2, 2, 2],
      col: [10, 20, 10, 10, 20],
    });
    const ct = df.crosstab("row", "col");
    expect(ct.shape[0]).toBe(2); // row values: 1, 2
    expect(ct.columns.length).toBe(2); // col values: 10, 20
  });

  it("single unique combination", () => {
    const df = new DataFrame({
      a: ["x", "x", "x"],
      b: ["y", "y", "y"],
    });
    const ct = df.crosstab("a", "b");
    expect(ct.shape).toEqual([1, 1]);
    expect(ct.get("y").toArray()).toEqual([3]);
  });
});
