import { describe, expect, it } from "vitest";
import { Series } from "../src/dataframe/Series";

describe("Series min/max errors", () => {
  it("throws error for non-numeric data", () => {
    const s = new Series(["a", "b"]);
    expect(() => s.min()).toThrow(/numeric/);
    expect(() => s.max()).toThrow(/numeric/);
  });

  it("throws error for mixed numeric data", () => {
    const s = new Series([1, "a", 2]);
    expect(() => s.min()).toThrow(/numeric/);
    expect(() => s.max()).toThrow(/numeric/);
  });
});
