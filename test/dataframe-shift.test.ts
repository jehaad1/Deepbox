import { describe, expect, it } from "vitest";
import { DataFrame } from "../src/dataframe";

describe("DataFrame shift tests", () => {
  it("should handle shift with periods > rowCount", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    const result = df.shift(5);
    expect(result.shape).toEqual([3, 1]);
    expect(result.toArray()).toEqual([[null], [null], [null]]);
  });

  it("should handle shift with negative periods > rowCount", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    const result = df.shift(-5);
    expect(result.shape).toEqual([3, 1]);
    expect(result.toArray()).toEqual([[null], [null], [null]]);
  });
});
