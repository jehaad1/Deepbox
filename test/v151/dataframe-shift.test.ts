import { describe, expect, it } from "vitest";
import { DataFrame } from "../../src/dataframe/index";

const col = (df: DataFrame, name = "a"): unknown[] => df.get(name).toArray();

// Expected values from pandas 3.1:
//   df = pd.DataFrame({'a': [0., 5, 0, 0, -2, 4]})
//   df.pct_change().a -> [nan, inf, -1.0, nan, -inf, -3.0]
describe("pctChange with a zero previous value (DF-7)", () => {
  it("returns +-Infinity for x/0 and NaN for 0/0 like pandas", () => {
    const df = new DataFrame({ a: [0, 5, 0, 0, -2, 4] });
    const out = col(df.pctChange());
    expect(out[0]).toBeNull();
    expect(out[1]).toBe(Infinity);
    expect(out[2]).toBe(-1);
    expect(out[3]).toBeNaN();
    expect(out[4]).toBe(-Infinity);
    expect(out[5]).toBe(-3);
  });
});

// pandas: df.diff(-1).a -> [-5, 5, 0, 2, -6, nan]; df.diff(-2).a -> [0, 5, 2, -4, nan, nan]
//         pd.DataFrame({'a': [1., 2, 4, 8]}).pct_change(-1).a -> [-0.5, -0.5, -0.5, nan]
describe("diff and pctChange with negative periods (DF-19)", () => {
  it("diff(-1) subtracts the next row", () => {
    const df = new DataFrame({ a: [0, 5, 0, 0, -2, 4] });
    expect(col(df.diff(-1))).toEqual([-5, 5, 0, 2, -6, null]);
    expect(col(df.diff(-2))).toEqual([0, 5, 2, -4, null, null]);
  });

  it("pctChange(-1) compares with the next row", () => {
    const df = new DataFrame({ a: [1, 2, 4, 8] });
    expect(col(df.pctChange(-1))).toEqual([-0.5, -0.5, -0.5, null]);
  });

  it("still rejects non-integer periods", () => {
    const df = new DataFrame({ a: [1, 2] });
    expect(() => df.diff(1.5)).toThrow(/periods/);
    expect(() => df.pctChange(Number.NaN)).toThrow(/periods/);
  });
});
