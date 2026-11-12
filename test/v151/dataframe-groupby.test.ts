import { describe, expect, it } from "vitest";
import { DataFrame, Series } from "../../src/dataframe/index";

// pandas 3.1: pd.DataFrame({'k': [], 'v': []}).groupby('k').sum().columns -> ['v']
//             (the key becomes the index; deepbox keeps it as the first column)
describe("groupBy on an empty DataFrame keeps the value columns (DF-10)", () => {
  const empty = new DataFrame({ k: [] as string[], v: [] as number[], w: [] as number[] });

  it("keeps every non-key column for numeric aggregations", () => {
    for (const result of [
      empty.groupBy("k").sum(),
      empty.groupBy("k").mean(),
      empty.groupBy("k").min(),
      empty.groupBy("k").median(),
    ]) {
      expect(result.columns).toEqual(["k", "v", "w"]);
      expect(result.shape).toEqual([0, 3]);
    }
  });

  it("matches the populated case when grouping by two keys", () => {
    const out = new DataFrame({ a: [] as number[], b: [] as number[], v: [] as number[] })
      .groupBy(["a", "b"])
      .sum();
    expect(out.columns).toEqual(["a", "b", "v"]);
  });
});

// pandas 3.1: pd.Series(['a', None, np.nan, 'a', 'b', np.nan]).value_counts(dropna=False)
//   -> NaN: 3, a: 2, b: 1 (None and NaN are one missing key)
describe("valueCounts counts every missing value as one key when dropna is false (DF-20)", () => {
  it("Series merges null, undefined and NaN", () => {
    const s = new Series<string | number | null | undefined>([
      "a",
      null,
      Number.NaN,
      "a",
      "b",
      undefined,
    ]);
    const counts = s.valueCounts(false);
    expect(counts.data).toEqual([3, 2, 1]);
    expect(counts.index).toEqual(["null", "a", "b"]);
    expect(s.valueCounts({ dropna: false, normalize: true }).data).toEqual([0.5, 1 / 3, 1 / 6]);
  });

  it("Series with dropna true is unchanged", () => {
    const s = new Series<string | null>(["a", null, "a", "b"]);
    expect(s.valueCounts().data).toEqual([2, 1]);
  });

  it("DataFrame merges null and NaN within a column", () => {
    const df = new DataFrame({ a: ["x", null, Number.NaN, "x", undefined] });
    const out = df.valueCounts("a");
    expect(out.get("count").toArray()).toEqual([3, 2]);
  });

  it("DataFrame keeps different value combinations apart", () => {
    const df = new DataFrame({ a: [null, Number.NaN, null], b: [1, 1, 2] });
    const out = df.valueCounts("a", "b");
    expect(out.get("count").toArray()).toEqual([2, 1]);
  });
});
