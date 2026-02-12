import { describe, expect, it } from "vitest";
import { DataValidationError } from "../src/core";
import { DataFrame } from "../src/dataframe";

describe("DataFrame extra tests", () => {
  it("should skip null index/column values during pivot", () => {
    const df = new DataFrame({ i: ["A", null, "B"], c: ["x", "y", null], v: [1, 2, 3] });
    const pivoted = df.pivot("i", "c", "v");
    expect(pivoted.columns).toEqual(["x", "y"]);
    expect(pivoted.index).toEqual(["A", "B"]);
    expect(pivoted.toArray()).toEqual([
      [1, null],
      [null, null],
    ]);
  });

  it("should throw an error for duplicate pivot entries", () => {
    const df = new DataFrame({
      i: [1, 2, 1],
      c: [1, 1, 1],
      v: [10, 20, 30],
    });
    expect(() => df.pivot("i", "c", "v")).toThrow(DataValidationError);
  });
});
