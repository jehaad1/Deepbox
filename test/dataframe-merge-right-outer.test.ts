import { describe, expect, it } from "vitest";
import { DataFrame } from "../src/dataframe";

describe("DataFrame merge right/outer branches", () => {
  it("includes unmatched right rows for right joins", () => {
    const left = new DataFrame({ id: [1, 2], val: ["a", "b"] });
    const right = new DataFrame({ id: [2, 3], score: [10, 30] });
    const merged = left.merge(right, { on: "id", how: "right" });

    expect(merged.columns).toEqual(["id", "val", "score"]);
    expect(merged.toArray()).toEqual([
      [2, "b", 10],
      [3, null, 30],
    ]);
  });

  it("includes unmatched rows on both sides for outer joins", () => {
    const left = new DataFrame({ id: [1, 2], val: ["a", "b"] });
    const right = new DataFrame({ id: [2, 3], score: [10, 30] });
    const merged = left.merge(right, { on: "id", how: "outer" });

    expect(merged.columns).toEqual(["id", "val", "score"]);
    expect(merged.toArray()).toEqual([
      [1, "a", null],
      [2, "b", 10],
      [3, null, 30],
    ]);
  });
});
