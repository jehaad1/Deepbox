import { describe, expect, it } from "vitest";
import { DataFrame, Series } from "../src/dataframe";

const metricsIndex = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"];

describe("DataFrame extra branch coverage", () => {
  it("describe handles empty and non-numeric frames", () => {
    const empty = new DataFrame({});
    const emptyDesc = empty.describe();
    expect(emptyDesc.shape).toEqual([8, 0]);
    expect(emptyDesc.index).toEqual(metricsIndex);

    const text = new DataFrame({ a: ["x", "y"] });
    const textDesc = text.describe();
    expect(textDesc.shape).toEqual([8, 0]);
    expect(textDesc.index).toEqual(metricsIndex);
  });

  it("describe handles single-value numeric columns", () => {
    const df = new DataFrame({ a: [5] });
    const desc = df.describe();
    expect(desc.shape).toEqual([8, 1]);
    expect(desc.index).toEqual(metricsIndex);
    const stats = desc.get("a");
    expect(stats.get("count")).toBe(1);
    expect(stats.get("mean")).toBe(5);
    expect(Number.isNaN(Number(stats.get("std")))).toBe(true);
    expect(stats.get("min")).toBe(5);
    expect(stats.get("max")).toBe(5);
  });

  it("apply handles axis=1 and validates return type/labels", () => {
    const df = new DataFrame({ a: [1, 2], b: [3, 4] });

    const doubled = df.apply(
      (row) =>
        new Series(
          row.data.map((value) => Number(value) * 2),
          { index: Array.from(row.index) }
        ),
      1
    );
    expect(doubled.toArray()).toEqual([
      [2, 6],
      [4, 8],
    ]);

    // @ts-expect-error Testing invalid return type from apply
    expect(() => df.apply(() => 123, 1)).toThrow(/must return a Series/);

    const ambiguous = new DataFrame({ a: [1], b: [2] });
    expect(() => ambiguous.apply(() => new Series([10, 20], { index: [1, "1"] }), 1)).toThrow(
      /ambiguous/i
    );
  });

  it("rename supports mapper functions for columns and index", () => {
    const df = new DataFrame({ a: [1, 2], b: [3, 4] });

    const renamedCols = df.rename((name) => name.toUpperCase(), 1);
    expect(renamedCols.columns).toEqual(["A", "B"]);

    const renamedIdx = df.rename((name) => `row-${name}`, 0);
    expect(renamedIdx.index).toEqual(["row-0", "row-1"]);
  });

  it("duplicated supports keep options and subset", () => {
    const df = new DataFrame({ a: [1, 1, 2], b: [3, 4, 4] });

    expect(df.duplicated(["a"], "first").toArray()).toEqual([false, true, false]);
    expect(df.duplicated(["a"], "last").toArray()).toEqual([true, false, false]);
    expect(df.duplicated(["a"], false).toArray()).toEqual([true, true, false]);
  });

  it("fillna and dropna handle missing values", () => {
    const dfFill = new DataFrame({ a: [1, null, 3], b: [undefined, 5, 6] });
    const filled = dfFill.fillna(0);
    expect(filled.toArray()).toEqual([
      [1, 0],
      [0, 5],
      [3, 6],
    ]);

    const dfDrop = new DataFrame({ a: [1, null, 3], b: [4, 5, undefined] });
    const dropped = dfDrop.dropna();
    expect(dropped.toArray()).toEqual([[1, 4]]);
  });

  it("toTensor rejects non-numeric columns", () => {
    const df = new DataFrame({ a: [1, 2], b: ["x", "y"] });
    expect(() => df.toTensor()).toThrow(/numeric/i);
  });
});
