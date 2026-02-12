import { describe, expect, it } from "vitest";
import { DataFrame } from "../src/dataframe/DataFrame";

describe("DataFrame groupby min/max", () => {
  it("computes groupby min and max", () => {
    const df = new DataFrame({
      group: ["a", "a", "b"],
      value: [3, 1, 5],
      other: [10, 7, 2],
    });

    const grouped = df.groupBy("group");
    const minDf = grouped.min();
    const maxDf = grouped.max();

    expect(minDf.toArray()).toEqual([
      ["a", 1, 7],
      ["b", 5, 2],
    ]);

    expect(maxDf.toArray()).toEqual([
      ["a", 3, 10],
      ["b", 5, 2],
    ]);
  });

  it("throws for non-numeric columns in sum aggregation", () => {
    const df = new DataFrame({
      group: ["a", "a"],
      value: ["x", "y"],
    });
    const grouped = df.groupBy("group");
    expect(() => grouped.agg({ value: "sum" })).toThrow(/numbers/i);
  });
});
