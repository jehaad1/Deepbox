import { describe, expect, it } from "vitest";
import { DataFrame } from "../../src/dataframe/index";

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
