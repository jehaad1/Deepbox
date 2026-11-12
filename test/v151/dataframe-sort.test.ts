import { describe, expect, it } from "vitest";
import { DataFrame, Series } from "../../src/dataframe/index";

// Mixed-type columns have no pandas equivalent (pandas raises TypeError); Deepbox defines a
// deterministic type rank: numbers, then strings, then booleans, then dates, then other values.
describe("sorting a mixed-type column uses a type rank (DF-21)", () => {
  const vals: unknown[] = [3, "b", 1, "a", true, null, Number.NaN, 2, "10", "9", false, "B"];

  it("ascending puts numbers, strings, booleans, then missing values", () => {
    const out = new DataFrame({ v: vals }).sort("v").getColumnData("v");
    expect(out.slice(0, 10)).toEqual([1, 2, 3, "10", "9", "B", "a", "b", false, true]);
    expect(out.slice(10).map((v) => (v === null ? "null" : String(v)))).toEqual(["null", "NaN"]);
  });

  it("descending reverses the present values and keeps missing values last", () => {
    const asc = new DataFrame({ v: vals }).sort("v").getColumnData("v").slice(0, 10);
    const desc = new DataFrame({ v: vals }).sort("v", false).getColumnData("v");
    expect(desc.slice(0, 10)).toEqual([...asc].reverse());
    expect(desc.slice(10).map((v) => (v === null ? "null" : String(v)))).toEqual(["null", "NaN"]);
  });

  it("does not depend on the input order", () => {
    const reference = JSON.stringify(new DataFrame({ v: vals }).sort("v").getColumnData("v"));
    let seed = 7;
    for (let t = 0; t < 50; t++) {
      const shuffled = [...vals];
      for (let i = shuffled.length - 1; i > 0; i--) {
        seed = (seed * 1664525 + 1013904223) % 4294967296;
        const j = seed % (i + 1);
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }
      expect(JSON.stringify(new DataFrame({ v: shuffled }).sort("v").getColumnData("v"))).toBe(
        reference
      );
    }
  });

  it("Series.sort follows the same rank", () => {
    const s = new Series<unknown>([true, "a", 2, new Date(5), null, 1]);
    expect(s.sort().data.slice(0, 5)).toEqual([1, 2, "a", true, new Date(5)]);
    expect(s.sort(false).data.slice(0, 5)).toEqual([new Date(5), true, "a", 2, 1]);
  });

  it("groupBy with sort uses the same order for mixed keys", () => {
    const df = new DataFrame({ k: ["b", 2, true, "a", 1], v: [1, 2, 3, 4, 5] });
    expect(df.groupBy("k", { sort: true }).sum().getColumnData("k")).toEqual([
      1,
      2,
      "a",
      "b",
      true,
    ]);
  });
});
