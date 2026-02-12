import { describe, expect, it } from "vitest";
import { DataValidationError } from "../src/core";
import { DataFrame } from "../src/dataframe";

// Type guards for strict testing
const isNullableNumber = (val: unknown): val is number | null =>
  typeof val === "number" || val === null;

describe("DataFrame Extra Features (Strict)", () => {
  describe("pivot()", () => {
    it("should pivot a simple DataFrame correctly and verify values", () => {
      const df = new DataFrame({
        i: ["B", "A", "B"],
        c: ["y", "x", "x"],
        v: [2, 1, 3],
      });
      const pivoted = df.pivot("i", "c", "v");

      // Order follows first appearance
      expect(pivoted.columns).toEqual(["y", "x"]);
      expect(pivoted.index).toEqual(["B", "A"]);

      // Strict value checks using get() with type guard
      const yCol = pivoted.get("y", isNullableNumber);
      const xCol = pivoted.get("x", isNullableNumber);

      expect(yCol.data).toEqual([2, null]);
      expect(xCol.data).toEqual([3, 1]);
    });

    it("should throw an error for duplicate pivot entries", () => {
      const df = new DataFrame({
        i: ["A", "A"],
        c: ["x", "x"],
        v: [null, 2],
      });
      expect(() => df.pivot("i", "c", "v")).toThrow(DataValidationError);
    });
  });

  describe("melt()", () => {
    it("should melt wide to long format and verify exact values", () => {
      const df = new DataFrame({
        id: ["A", "B"],
        x: [1, 2],
        y: [3, 4],
      });

      const melted = df.melt(["id"], ["x", "y"], "variable", "value");

      // Check shape and columns
      expect(melted.shape).toEqual([4, 3]);
      expect(melted.columns).toEqual(["id", "variable", "value"]);

      // Verify specific rows
      // We expect 4 rows: (A, x, 1), (A, y, 3), (B, x, 2), (B, y, 4)
      // Note: Implementation details determine order.
      // Based on reading melt():
      // Outer loop: index (row 0, row 1)
      //   Middle loop: valueVars (x, y)
      //     Inner loop: idVars (id)
      // Row 0:
      //   Var x: id=A, var=x, val=1 -> Push to arrays
      //   Var y: id=A, var=y, val=3 -> Push to arrays
      // Row 1:
      //   Var x: id=B, var=x, val=2
      //   Var y: id=B, var=y, val=4
      //
      // Result order in arrays:
      // id: [A, A, B, B]
      // variable: [x, y, x, y]
      // value: [1, 3, 2, 4]

      const ids = melted.get("id");
      const vars = melted.get("variable");
      const vals = melted.get("value");

      expect(ids.data).toEqual(["A", "A", "B", "B"]);
      expect(vars.data).toEqual(["x", "y", "x", "y"]);
      expect(vals.data).toEqual([1, 3, 2, 4]);
    });
  });

  describe("join() overlap handling", () => {
    it("should handle overlapping columns in join with suffixes", () => {
      const left = new DataFrame({ id: [1, 2], x: [10, 20] });
      const right = new DataFrame({ id: [1, 2], x: [30, 40] });
      const result = left.join(right, "id");

      expect(result.columns).toEqual(["id", "x_left", "x_right"]);

      const idCol = result.get("id");
      const xLeft = result.get("x_left");
      const xRight = result.get("x_right");

      expect(idCol.data).toEqual([1, 2]);
      expect(xLeft.data).toEqual([10, 20]);
      expect(xRight.data).toEqual([30, 40]);
    });
  });
});
