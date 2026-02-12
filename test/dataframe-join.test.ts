import { describe, expect, it } from "vitest";
import { DataFrame } from "../src/dataframe";

describe("DataFrame join tests", () => {
  it("should match NaN join keys and suffix overlapping columns", () => {
    const left = new DataFrame({ id: [NaN, 1], x: [10, 20] });
    const right = new DataFrame({ id: [NaN, 2], x: [30, 40] });
    const result = left.join(right, "id");
    expect(result.columns).toEqual(["id", "x_left", "x_right"]);

    const rows = result.toArray();
    expect(rows).toHaveLength(1);
    expect(Number.isNaN(Number(rows[0]?.[0]))).toBe(true);
    expect(rows[0]?.[1]).toBe(10);
    expect(rows[0]?.[2]).toBe(30);
  });
});
