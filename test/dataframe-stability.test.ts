import { describe, expect, it } from "vitest";
import { DataFrame } from "../src/dataframe";

describe("DataFrame Sort Stability", () => {
  it("should maintain relative order of equal elements (stable sort)", () => {
    // Create data with duplicate keys but distinct values in other columns
    const df = new DataFrame({
      key: [1, 1, 1, 2, 2],
      val: ["a", "b", "c", "d", "e"],
      id: [0, 1, 2, 3, 4], // Original order
    });

    // Sort by 'key'
    const sorted = df.sort("key");

    // Check that for key=1, the relative order of 'val'/'id' is preserved
    // Expected: rows 0, 1, 2 should remain in that order
    const ids = sorted.get("id").data;
    expect(ids).toEqual([0, 1, 2, 3, 4]);

    const vals = sorted.get("val").data;
    expect(vals).toEqual(["a", "b", "c", "d", "e"]);
  });

  it("should maintain stability with multiple sort columns", () => {
    const df = new DataFrame({
      group: [1, 1, 1, 1],
      subgroup: [2, 2, 1, 1],
      id: [0, 1, 2, 3],
    });

    // Sort by group (all equal) then subgroup
    // Rows 2,3 (subgroup 1) should come before 0,1 (subgroup 2)
    // Within subgroup 1, row 2 should come before 3
    // Within subgroup 2, row 0 should come before 1
    const sorted = df.sort(["group", "subgroup"]);

    const ids = sorted.get("id").data;
    expect(ids).toEqual([2, 3, 0, 1]);
  });
});
