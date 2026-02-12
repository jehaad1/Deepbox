import { describe, expect, it } from "vitest";
import { DataFrame, Series } from "../src/dataframe";
import { tensor } from "../src/ndarray";

// Type guards
const isNumber = (val: unknown): val is number => typeof val === "number";
const isNullableNumber = (val: unknown): val is number | null =>
  typeof val === "number" || val === null;

describe("DataFrame - Coverage Extras (Strict)", () => {
  it("handles selection, filtering, and sorting with value verification", () => {
    const df = new DataFrame(
      {
        a: [3, 1, 2],
        b: ["c", "a", "b"],
        flag: [true, false, true],
      },
      { index: ["r1", "r2", "r3"] }
    );

    expect(df.head(2).shape).toEqual([2, 3]);
    expect(df.tail(1).index).toEqual(["r3"]);

    const filtered = df.filter((row) => {
      const val = row["a"];
      return typeof val === "number" && val > 1;
    });
    expect(filtered.shape[0]).toBe(2);
    expect(filtered.index).toEqual(["r1", "r3"]);
    expect(filtered.get("a", isNumber).data).toEqual([3, 2]);

    const selected = df.select(["b", "a"]);
    expect(selected.columns).toEqual(["b", "a"]);

    const dropped = df.drop(["flag"]);
    expect(dropped.columns).toEqual(["a", "b"]);

    const sortedNum = df.sort("a");
    // Verify sorted values
    expect(sortedNum.get("a", isNumber).data).toEqual([1, 2, 3]);
    // Verify index followed
    expect(sortedNum.index).toEqual(["r2", "r3", "r1"]);

    const sortedStrDesc = df.sort(["b"], false);
    expect(sortedStrDesc.get("b").data).toEqual(["c", "b", "a"]);

    const sortedFallback = df.sort("flag");
    // false < true
    expect(sortedFallback.get("flag").data).toEqual([false, true, true]);
  });

  it("handles concat and missing value operations with strict checks", () => {
    const left = new DataFrame({ a: [1, 2], b: [null, 4] }, { index: [0, 1] });
    const right = new DataFrame({ a: [3], b: [5] }, { index: [2] });

    const stacked = left.concat(right, 0);
    expect(stacked.shape).toEqual([3, 2]);
    expect(stacked.get("a", isNumber).data).toEqual([1, 2, 3]);

    const wide = left.concat(new DataFrame({ c: [9, 10] }, { index: [0, 1] }), 1);
    expect(wide.columns).toEqual(["a", "b", "c"]);
    expect(wide.get("c", isNumber).data).toEqual([9, 10]);

    expect(() => left.concat(new DataFrame({ a: [1], c: [2] }), 0)).toThrow(/missing column/);

    const filled = left.fillna(0);
    expect(filled.get("b", isNumber).data).toEqual([0, 4]);

    const dropped = left.dropna();
    expect(dropped.shape).toEqual([1, 2]);
    expect(dropped.index).toEqual([1]);
  });

  it("handles describe, corr, cov, and apply with value verification", () => {
    const df = new DataFrame({
      a: [1, 2, 3],
      b: [2, 4, 6], // Perfect correlation with a
      c: ["x", "y", "z"],
    });

    const desc = df.describe();
    expect(desc.shape).toEqual([8, 2]); // Only numeric columns
    expect(desc.get("a", isNumber).loc("mean")).toBe(2);
    expect(desc.get("b", isNumber).loc("max")).toBe(6);

    const corr = df.corr();
    expect(corr.shape).toEqual([2, 2]);
    // a and b are perfectly correlated
    expect(corr.get("a", isNumber).loc("b")).toBeCloseTo(1, 6);

    const cov = df.cov();
    expect(cov.shape).toEqual([2, 2]);
    // cov(a, b) = cov(a, 2a) = 2 * var(a). var(1,2,3) = 1. So cov = 2.
    expect(cov.get("a", isNumber).loc("b")).toBeCloseTo(2, 6);

    const dfNumeric = df.select(["a", "b"]);
    const appliedCols = dfNumeric.apply(
      (s) =>
        new Series(
          s.data.map((v) => {
            if (!isNumber(v)) throw new Error("Expected number");
            return v * 2;
          })
        ),
      0
    );
    expect(appliedCols.get("a", isNumber).data).toEqual([2, 4, 6]);

    const appliedRows = dfNumeric.apply(
      (row) =>
        new Series(
          row.data.map((v) => (typeof v === "number" ? v + 1 : v)),
          { index: [...row.index] }
        ),
      1
    );
    expect(appliedRows.get("a", isNumber).data).toEqual([2, 3, 4]);

    // @ts-expect-error Testing invalid return type
    expect(() => df.apply(() => 123, 1)).toThrow(/must return a Series/);
  });

  it("handles tensor conversions and json round-trip", () => {
    const df = new DataFrame({ a: [1, 2], b: [3, 4] });
    const t = df.toTensor();
    expect(t.shape).toEqual([2, 2]);
    // Check flattened data
    expect(t.flatten().toArray()).toEqual([1, 3, 2, 4]);

    // toTensor flattens row by row.
    // Row 0: 1, 3. Row 1: 2, 4.
    // Flat: 1, 3, 2, 4.

    expect(() => new DataFrame({ a: [1], b: ["x"] }).toTensor()).toThrow(/Non-numeric/);

    const roundTrip = DataFrame.fromTensor(
      tensor([
        [1, 2],
        [3, 4],
      ]),
      ["c1", "c2"]
    );
    expect(roundTrip.columns).toEqual(["c1", "c2"]);
    expect(roundTrip.get("c1", isNumber).data).toEqual([1, 3]);

    const json = df.toJsonString();
    const parsed = DataFrame.fromJsonString(json);
    expect(parsed.toArray()).toEqual(df.toArray());
  });

  it("handles ranking, shift, diffs, pct change, and cumulative ops", () => {
    const df = new DataFrame({ a: [3, 1, 2, 1], b: [10, 0, 5, 5] });

    const rankAvg = df.rank();
    // 1, 1, 2, 3 -> ranks 1.5, 1.5, 3, 4.
    // Original: 3 (4), 1 (1.5), 2 (3), 1 (1.5)
    expect(rankAvg.get("a", isNullableNumber).data).toEqual([4, 1.5, 3, 1.5]);

    const rankDense = df.rank("dense", false);
    // Descending: 3, 2, 1, 1.
    // 3 (1), 2 (2), 1 (3), 1 (3).
    expect(rankDense.get("a", isNullableNumber).data).toEqual([1, 3, 2, 3]);

    const shifted = df.shift(1);
    expect(shifted.get("a", isNullableNumber).data).toEqual([null, 3, 1, 2]);

    const shiftedNeg = df.shift(-1, 0);
    expect(shiftedNeg.get("a", isNumber).data).toEqual([1, 2, 1, 0]);

    const diffed = df.diff(2);
    // 3, 1, 2, 1
    // [null, null, 2-3=-1, 1-1=0]
    expect(diffed.get("a", isNullableNumber).data).toEqual([null, null, -1, 0]);

    const pct = df.pct_change();
    // 10, 0, 5, 5
    // [null, (0-10)/10=-1, (5-0)/0=Infinity?, (5-5)/5=0]
    // JS division by zero is Infinity.
    const bPct = pct.get("b", isNullableNumber).data;
    expect(bPct[0]).toBeNull();
    expect(bPct[1]).toBe(-1);
    // previous is 0, code says: if previous !== 0 push (c-p)/p else null
    expect(bPct[2]).toBeNull();
    expect(bPct[3]).toBe(0);

    const cumsum = df.cumsum();
    expect(cumsum.get("a", isNumber).data).toEqual([3, 4, 6, 7]);

    const cumprod = df.cumprod();
    expect(cumprod.get("a", isNumber).data).toEqual([3, 3, 6, 6]);

    const cummax = df.cummax();
    expect(cummax.get("a", isNumber).data).toEqual([3, 3, 3, 3]);

    const cummin = df.cummin();
    expect(cummin.get("a", isNumber).data).toEqual([3, 1, 1, 1]);
  });

  it("handles rolling window correctly", () => {
    const df = new DataFrame({
      value: [1, 2, 3, 4, 5],
    });

    const rolled = df.rolling(3);
    const result = rolled.get("value", isNullableNumber).data;
    // Window 3.
    // i=0: [1] (len 1 < 3) -> null (code says if i < window - 1 push null)
    // i=1: [1, 2] -> null
    // i=2: [1, 2, 3] -> 2
    // i=3: [2, 3, 4] -> 3
    // i=4: [3, 4, 5] -> 4
    expect(result).toEqual([null, null, 2, 3, 4]);
  });
});
