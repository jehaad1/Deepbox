import { describe, expect, it } from "vitest";
import { DataFrame } from "../../src/dataframe/index";

// pandas 3.1: df = pd.DataFrame({'a': [3., 1, nan, 4, 1, 5, 9, 2, 6]})
//   df.expanding().median().a -> [3, 2, 2, 3, 2, 3, 3.5, 3, 3.5]
//   df.expanding(min_periods=3).median().a -> [nan, nan, nan, 3, 2, 3, 3.5, 3, 3.5]
describe("Expanding.median (DF-8)", () => {
  const df = new DataFrame({
    a: [3, 1, Number.NaN, 4, 1, 5, 9, 2, 6],
    s: ["x", "y", "z", "w", "v", "u", "t", "r", "q"],
  });

  it("matches pandas with the default minPeriods", () => {
    expect(df.expanding().median().get("a").toArray()).toEqual([3, 2, 2, 3, 2, 3, 3.5, 3, 3.5]);
  });

  it("honors minPeriods", () => {
    expect(df.expanding(3).median().get("a").toArray()).toEqual([
      null,
      null,
      null,
      3,
      2,
      3,
      3.5,
      3,
      3.5,
    ]);
  });

  it("gives null for columns without numeric observations", () => {
    expect(df.expanding().median().get("s").toArray()).toEqual(new Array(9).fill(null));
  });
});
