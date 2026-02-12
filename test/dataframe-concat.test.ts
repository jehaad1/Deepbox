import { describe, expect, it } from "vitest";
import { DataFrame } from "../src/dataframe";

describe("DataFrame concat tests", () => {
  it("should concat two DataFrames with default index without error", () => {
    const df1 = new DataFrame({ a: [1, 2], b: [3, 4] });
    const df2 = new DataFrame({ a: [5, 6], b: [7, 8] });
    const result = df1.concat(df2);
    expect(result.shape).toEqual([4, 2]);
    expect(result.toArray()).toEqual([
      [1, 3],
      [2, 4],
      [5, 7],
      [6, 8],
    ]);
  });

  it("should align non-default indices when concatenating on axis=1", () => {
    const df1 = new DataFrame({ x: [1, 2] }, { index: ["b", "a"] });
    const df2 = new DataFrame({ y: [10, 20] }, { index: ["a", "c"] });
    const result = df1.concat(df2, 1);
    expect(result.columns).toEqual(["x", "y"]);
    expect(result.index).toEqual(["b", "a", "c"]);
    expect(result.toArray()).toEqual([
      [1, null],
      [2, 10],
      [null, 20],
    ]);
  });
});
