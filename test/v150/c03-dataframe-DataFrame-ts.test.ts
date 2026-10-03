import { describe, expect, it } from "vitest";
import { DataValidationError, InvalidParameterError } from "../../src/core/errors/index";
import { DataFrame } from "../../src/dataframe";
import { tensor, transpose } from "../../src/ndarray";

const col = (df: DataFrame, name: string): unknown[] => [...df.getColumnData(name)];

const expectClose = (actual: unknown[], expected: (number | null)[], digits = 10): void => {
  expect(actual).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const want = expected[i];
    const got = actual[i];
    if (want === null) {
      expect(got).toBeNull();
    } else {
      expect(typeof got).toBe("number");
      expect(got as number).toBeCloseTo(want, digits);
    }
  }
};

describe("c03 DataFrame: construction, selection, sorting", () => {
  it("tail(n) with n above the row count returns every row", () => {
    const df = new DataFrame({ a: [1, 2, 3, 4, 5] });
    expect(col(df.tail(7), "a")).toEqual([1, 2, 3, 4, 5]);
    expect(df.tail(7).index).toEqual([0, 1, 2, 3, 4]);
    expect(col(df.tail(0), "a")).toEqual([]);
    expect(col(df.tail(2), "a")).toEqual([4, 5]);
  });

  it("iloc rejects non-integer positions", () => {
    const df = new DataFrame({ a: [1, 2] });
    expect(() => df.iloc(0.5)).toThrow(InvalidParameterError);
    expect(() => df.iloc(Number.NaN)).toThrow(InvalidParameterError);
  });

  it("constructor reports non-array columns and rejects inherited keys", () => {
    expect(
      () => new DataFrame({ a: [1], b: "x" as unknown as unknown[] }, { columns: ["a", "b"] })
    ).toThrow(/must be an array/);
    expect(() => new DataFrame({ a: [1] }, { columns: ["toString"] })).toThrow(DataValidationError);
    expect(() => new DataFrame({ a: [1] }, { index: [null as unknown as number] })).toThrow(
      /must be a string or number/
    );
  });

  it("sort keeps tie-breaking when keys are equal infinities", () => {
    const df = new DataFrame({ a: [Infinity, Infinity, 1], b: [2, 1, 0] });
    const sorted = df.sort(["a", "b"]);
    expect(col(sorted, "a")).toEqual([1, Infinity, Infinity]);
    expect(col(sorted, "b")).toEqual([0, 1, 2]);
    const desc = new DataFrame({ a: [-Infinity, -Infinity, 1], b: [1, 2, 3] }).sort(
      ["a", "b"],
      false
    );
    expect(col(desc, "b")).toEqual([3, 2, 1]);
  });

  it("sort accepts one direction per column and puts missing values last", () => {
    const df = new DataFrame({
      g: ["x", "y", "x", "y", "x"],
      v: [1, 5, 3, null, Number.NaN],
    });
    const sorted = df.sort(["g", "v"], [true, false]);
    expect(col(sorted, "g")).toEqual(["x", "x", "x", "y", "y"]);
    expect(col(sorted, "v")).toEqual([3, 1, Number.NaN, 5, null]);
    expect(() => df.sort(["g", "v"], [true])).toThrow(InvalidParameterError);
  });

  it("sort orders dates, bigints and booleans by value", () => {
    const d = (s: string) => new Date(s);
    const dates = new DataFrame({ t: [d("2024-03-01"), d("2023-01-01"), d("2024-01-01")] });
    expect(col(dates.sort("t"), "t").map((x) => (x as Date).getFullYear())).toEqual([
      2023, 2024, 2024,
    ]);
    expect(col(dates.sort("t"), "t")[1]).toEqual(d("2024-01-01"));
    const big = new DataFrame({ n: [10n, 9n, 100n] });
    expect(col(big.sort("n"), "n")).toEqual([9n, 10n, 100n]);
    const flags = new DataFrame({ f: [true, false, true, false] });
    expect(col(flags.sort("f"), "f")).toEqual([false, false, true, true]);
  });

  it("select and drop validate their column lists", () => {
    const df = new DataFrame({ a: [1], b: [2] });
    expect(() => df.select(["a", "a"])).toThrow(DataValidationError);
    expect(() => df.select("a" as unknown as string[])).toThrow(InvalidParameterError);
    expect(df.drop(["a"]).columns).toEqual(["b"]);
  });
});

describe("c03 DataFrame: join and merge", () => {
  const left = new DataFrame({ id: [1, 2, 3, 2], v: ["a", "b", "c", "d"] });
  const right = new DataFrame({ id: [2, 3, 4, 2], w: [10, 20, 30, 40] });

  it("matches pandas row order for inner, left and right joins", () => {
    // Reference: pandas 3.0 DataFrame.merge on "id".
    expect(left.join(right, "id", "inner").toArray()).toEqual([
      [2, "b", 10],
      [2, "b", 40],
      [3, "c", 20],
      [2, "d", 10],
      [2, "d", 40],
    ]);
    expect(left.join(right, "id", "left").toArray()).toEqual([
      [1, "a", null],
      [2, "b", 10],
      [2, "b", 40],
      [3, "c", 20],
      [2, "d", 10],
      [2, "d", 40],
    ]);
    // A right join follows the right frame's row order.
    expect(left.join(right, "id", "right").toArray()).toEqual([
      [2, "b", 10],
      [2, "d", 10],
      [3, "c", 20],
      [4, null, 30],
      [2, "b", 40],
      [2, "d", 40],
    ]);
  });

  it("outer join lists left-driven rows, then unmatched right rows", () => {
    expect(left.merge(right, { on: "id", how: "outer" }).toArray()).toEqual([
      [1, "a", null],
      [2, "b", 10],
      [2, "b", 40],
      [3, "c", 20],
      [2, "d", 10],
      [2, "d", 40],
      [4, null, 30],
    ]);
  });

  it("merge with different key names suffixes every shared column", () => {
    // pandas: columns ['k', 'a_x', 'b', 'j', 'a_y', 'c']
    const l = new DataFrame({ k: [1, 2], a: [1, 2], b: [3, 4] });
    const r = new DataFrame({ j: [2, 3], a: [5, 6], c: [7, 8] });
    const out = l.merge(r, { left_on: "k", right_on: "j", how: "outer" });
    expect(out.columns).toEqual(["k", "a_x", "b", "j", "a_y", "c"]);
    expect(out.toArray()).toEqual([
      [1, 1, 3, null, null, null],
      [2, 2, 4, 2, 5, 7],
      [null, null, null, 3, 6, 8],
    ]);
  });

  it("merge suffixes a left column that shares its name with the right key", () => {
    const l = new DataFrame({ a: [1, 2], b: [3, 4] });
    const r = new DataFrame({ b: [1, 2], c: [5, 6] });
    const out = l.merge(r, { left_on: "a", right_on: "b" });
    expect(out.columns).toEqual(["a", "b_x", "b_y", "c"]);
    expect(out.toArray()).toEqual([
      [1, 3, 1, 5],
      [2, 4, 2, 6],
    ]);
  });

  it("merge keeps generated suffix names unique", () => {
    const l = new DataFrame({ id: [1], a: [1], a_x: [2] });
    const r = new DataFrame({ id: [1], a: [3] });
    const out = l.merge(r, { on: "id" });
    expect(new Set(out.columns).size).toBe(out.columns.length);
    expect(out.columns).toHaveLength(4);
  });

  it("null keys never match, NaN keys do", () => {
    const l = new DataFrame({ k: [null, Number.NaN, 1], v: [1, 2, 3] });
    const r = new DataFrame({ k: [null, Number.NaN, 1], w: [4, 5, 6] });
    expect(l.join(r, "k").toArray()).toEqual([
      [Number.NaN, 2, 5],
      [1, 3, 6],
    ]);
  });

  it("does not modify its inputs", () => {
    const before = left.toArray();
    left.join(right, "id", "outer");
    expect(left.toArray()).toEqual(before);
  });
});

describe("c03 DataFrame: missing values", () => {
  it("fillna, isnull and notnull agree that NaN is missing", () => {
    const df = new DataFrame({ a: [1, Number.NaN, null, 4] });
    expect(col(df.isnull(), "a")).toEqual([false, true, true, false]);
    expect(col(df.notnull(), "a")).toEqual([true, false, false, true]);
    expect(col(df.fillna(0), "a")).toEqual([1, 0, 0, 4]);
  });

  it("dropna supports how, subset and thresh", () => {
    const df = new DataFrame({
      a: [1, null, null, 4],
      b: [1, 2, null, Number.NaN],
      c: [1, 2, 3, 4],
    });
    expect(df.dropna().index).toEqual([0]);
    expect(df.dropna({ how: "all" }).index).toEqual([0, 1, 2, 3]);
    expect(df.dropna({ subset: ["a"] }).index).toEqual([0, 3]);
    expect(df.dropna({ subset: ["a", "b"], how: "all" }).index).toEqual([0, 1, 3]);
    expect(df.dropna({ thresh: 2 }).index).toEqual([0, 1, 3]);
    expect(() => df.dropna({ subset: ["zzz"] })).toThrow(InvalidParameterError);
    expect(() => df.dropna({ thresh: -1 })).toThrow(InvalidParameterError);
    expect(() => df.dropna({ how: "some" as "any" })).toThrow(InvalidParameterError);
  });

  it("nunique ignores missing values like pandas", () => {
    const df = new DataFrame({ a: [1, null, Number.NaN, 1, 2], b: ["x", "x", "y", null, "y"] });
    expect(df.nunique().data).toEqual([2, 2]);
    const dates = new DataFrame({ d: [new Date(5), new Date(5), new Date(6)] });
    expect(dates.nunique().data).toEqual([2]);
  });
});

describe("c03 DataFrame: numeric helpers", () => {
  it("round uses half-to-even like NumPy", () => {
    // numpy: np.round([0.5, 1.5, 2.5, -2.5, 2.675], 0) -> [0, 2, 2, -2, 3]
    const df = new DataFrame({ a: [0.5, 1.5, 2.5, -2.5, 2.675] });
    expect(col(df.round(), "a")).toEqual([0, 2, 2, -2, 3]);
    // numpy: np.round([2.675, 1.005, 0.125, 0.375], 2) -> [2.68, 1.0, 0.12, 0.38]
    const f = new DataFrame({ a: [2.675, 1.005, 0.125, 0.375] });
    expect(col(f.round(2), "a")).toEqual([2.68, 1, 0.12, 0.38]);
    // numpy: np.round([1234.5678, 15, 25, 35, -15], -1) -> [1230, 20, 20, 40, -20]
    const neg = new DataFrame({ a: [1234.5678, 15, 25, 35, -15] });
    expect(col(neg.round(-1), "a")).toEqual([1230, 20, 20, 40, -20]);
    expect(col(new DataFrame({ a: [1e308, Infinity, Number.NaN] }).round(2), "a")).toEqual([
      1e308,
      Infinity,
      Number.NaN,
    ]);
    expect(() => df.round(1.5)).toThrow(InvalidParameterError);
  });

  it("cumulative functions skip NaN instead of poisoning the total", () => {
    // pandas: Series([1, 2, 3, 4, 5.5, 2, nan, 7, 1.5]).cumsum()
    const df = new DataFrame({ a: [1, 2, 3, 4, 5.5, 2, Number.NaN, 7, 1.5] });
    const sum = col(df.cumsum(), "a");
    expect(sum.slice(0, 6)).toEqual([1, 3, 6, 10, 15.5, 17.5]);
    expect(sum[6]).toBeNaN();
    expect(sum.slice(7)).toEqual([24.5, 26]);
    const max = col(df.cummax(), "a");
    expect(max[5]).toBe(5.5);
    expect(max[6]).toBeNaN();
    expect(max[7]).toBe(7);
    expect(col(new DataFrame({ a: [2, Number.NaN, 3] }).cumprod(), "a")).toEqual([
      2,
      Number.NaN,
      6,
    ]);
    expect(col(new DataFrame({ a: [3, Number.NaN, 1] }).cummin(), "a")).toEqual([3, Number.NaN, 1]);
    expect(col(new DataFrame({ a: [-Infinity, -Infinity] }).cummax(), "a")).toEqual([
      -Infinity,
      -Infinity,
    ]);
  });

  it("rank handles infinities and validates the method", () => {
    const df = new DataFrame({ a: [3, 1, 2, 1, Infinity, Number.NaN, -Infinity] });
    // pandas: rank() -> [5, 2.5, 4, 2.5, 6, nan, 1]
    expect(col(df.rank(), "a")).toEqual([5, 2.5, 4, 2.5, 6, null, 1]);
    // pandas: rank(method="dense", ascending=False) -> [2, 4, 3, 4, 1, nan, 5]
    expect(col(df.rank("dense", false), "a")).toEqual([2, 4, 3, 4, 1, null, 5]);
    // pandas: rank(method="first") -> [5, 2, 4, 3, 6, nan, 1]
    expect(col(df.rank("first"), "a")).toEqual([5, 2, 4, 3, 6, null, 1]);
    expect(() => df.rank("median" as "average")).toThrow(InvalidParameterError);
  });

  it("quantile and describe match NumPy's linear interpolation", () => {
    const df = new DataFrame({ a: [1, 2, 4, 10, 11.5] });
    // numpy.quantile([1, 2, 4, 10, 11.5], [0.1, 0.25, 0.3, 0.5, 0.9])
    const expected = [1.4, 2, 2.4, 4, 10.9];
    [0.1, 0.25, 0.3, 0.5, 0.9].forEach((q, i) => {
      expect(df.quantile(q).data[0]).toBeCloseTo(expected[i] as number, 12);
    });
    const d = col(df.describe(), "a") as number[];
    expect(d[0]).toBe(5);
    expect(d[1]).toBeCloseTo(5.7, 12);
    expect(d[2]).toBeCloseTo(4.764451699828638, 12);
    expect(d.slice(3)).toEqual([1, 2, 4, 10, 11.5]);
  });

  it("describe and quantile keep infinite values", () => {
    const df = new DataFrame({ a: [1, 2, Infinity] });
    const d = col(df.describe(), "a") as number[];
    expect(d[0]).toBe(3);
    expect(d[1]).toBe(Infinity);
    expect(d[3]).toBe(1);
    expect(d[7]).toBe(Infinity);
    expect(df.quantile(1).data[0]).toBe(Infinity);
    expect(df.quantile(0.5).data[0]).toBe(2);
  });

  it("clip rejects NaN bounds", () => {
    const df = new DataFrame({ a: [1, 5] });
    expect(() => df.clip(Number.NaN, 3)).toThrow(InvalidParameterError);
    expect(col(df.clip(2, 4), "a")).toEqual([2, 4]);
  });

  it("shift(0) copes with very large columns", () => {
    const n = 300_000;
    const df = new DataFrame({ a: Array.from({ length: n }, (_, i) => i) });
    expect(df.shift(0).shape).toEqual([n, 1]);
  });

  it("idxmax and idxmin handle infinities and NaN", () => {
    const df = new DataFrame(
      { a: [-Infinity, -Infinity], b: [Infinity, Infinity], c: [Number.NaN, 2], d: ["x", "y"] },
      { index: ["p", "q"] }
    );
    expect(df.idxmax().data).toEqual(["p", "p", "q", null]);
    expect(df.idxmin().data).toEqual(["p", "p", "q", null]);
  });

  it("nlargest and nsmallest skip NaN and keep ties in order", () => {
    const df = new DataFrame({
      a: [3, Number.NaN, 3, 1, Infinity, null],
      b: [0, 1, 2, 3, 4, 5],
    });
    expect(col(df.nlargest(3, "a"), "b")).toEqual([4, 0, 2]);
    expect(col(df.nsmallest(2, "a"), "b")).toEqual([3, 0]);
    expect(df.nlargest(10, "a").shape[0]).toBe(4);
    expect(() => df.nlargest(-1, "a")).toThrow(InvalidParameterError);
    expect(() => new DataFrame({ s: ["a", "b"] }).nlargest(1, "s")).toThrow(DataValidationError);
  });

  it("between supports one-sided inclusion", () => {
    const df = new DataFrame({ a: [1, 2, 3, 4, Number.NaN] });
    expect(col(df.between("a", 2, 4), "a")).toEqual([2, 3, 4]);
    expect(col(df.between("a", 2, 4, false), "a")).toEqual([3]);
    expect(col(df.between("a", 2, 4, "left"), "a")).toEqual([2, 3]);
    expect(col(df.between("a", 2, 4, "right"), "a")).toEqual([3, 4]);
    expect(() => df.between("a", 2, 4, "x" as "both")).toThrow(InvalidParameterError);
  });
});

describe("c03 DataFrame: element-wise helpers", () => {
  it("applymap passes only the value to the callback", () => {
    const df = new DataFrame({ a: ["1", "2", "3"] });
    expect(col(df.applymap(parseInt as unknown as (value: unknown) => unknown), "a")).toEqual([
      1, 2, 3,
    ]);
  });

  it("rename ignores inherited property names and validates the axis", () => {
    const df = new DataFrame({ toString: [1], constructor: [2] });
    expect(df.rename({ a: "x" }).columns).toEqual(["toString", "constructor"]);
    expect(df.rename({ toString: "t" }).columns).toEqual(["t", "constructor"]);
    expect(df.rename({ 0: "zero" }, "index").index).toEqual(["zero"]);
    expect(() => df.rename({}, 2)).toThrow();
  });

  it("explode turns an empty list into one null row", () => {
    const df = new DataFrame({ id: [1, 2, 3], v: [[1, 2], [], 7] });
    const out = df.explode("v");
    expect(col(out, "id")).toEqual([1, 1, 2, 3]);
    expect(col(out, "v")).toEqual([1, 2, null, 7]);
  });

  it("getDummies ignores missing values and sorts numeric categories numerically", () => {
    const df = new DataFrame({ c: ["b", null, "a", "b", Number.NaN], n: [10, 2, 10, 33, 2] });
    const out = df.getDummies(["c", "n"]);
    expect(out.columns).toEqual(["c_a", "c_b", "n_2", "n_10", "n_33"]);
    expect(col(out, "c_a")).toEqual([0, 0, 1, 0, 0]);
    expect(col(out, "c_b")).toEqual([1, 0, 0, 1, 0]);
    const withNa = df.getDummies("c", { dummyNa: true });
    expect(col(withNa, "c_nan")).toEqual([0, 1, 0, 0, 1]);
    expect(() => df.getDummies("zzz")).toThrow(InvalidParameterError);
    expect(df.getDummies("c", { dropFirst: true }).columns).toEqual(["n", "c_b"]);
  });

  it("assign broadcasts scalars and checks array lengths", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    const out = df.assign({ k: 7, flag: true, sq: (row) => (row.a as number) ** 2 });
    expect(col(out, "k")).toEqual([7, 7, 7]);
    expect(col(out, "flag")).toEqual([true, true, true]);
    expect(col(out, "sq")).toEqual([1, 4, 9]);
    expect(() => df.assign({ bad: [1, 2] })).toThrow(DataValidationError);
  });

  it("where and mask reject a boolean array of the wrong length", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    expect(col(df.where([true, false, true]), "a")).toEqual([1, null, 3]);
    expect(col(df.mask([true, false, true], 0), "a")).toEqual([0, 2, 0]);
    expect(() => df.where([true])).toThrow(InvalidParameterError);
    expect(() => df.mask([true, true, true, true])).toThrow(InvalidParameterError);
  });

  it("astype keeps missing values missing and parses boolean strings", () => {
    const df = new DataFrame({
      n: ["1", null, "", " 2.5 ", "x"],
      s: [1, null, Number.NaN, 2, 3],
      b: ["true", "FALSE", null, 0, 2],
    });
    const nums = col(df.astype("n", "number"), "n");
    expect(nums[0]).toBe(1);
    expect(nums[1]).toBeNaN();
    expect(nums[2]).toBeNaN();
    expect(nums[3]).toBe(2.5);
    expect(nums[4]).toBeNaN();
    expect(col(df.astype("s", "string"), "s")).toEqual(["1", null, null, "2", "3"]);
    expect(col(df.astype("b", "boolean"), "b")).toEqual([true, false, null, false, true]);
    const both = df.astype({ s: "string", n: "number" });
    expect(col(both, "s")[0]).toBe("1");
    expect(() => df.astype("n", "date" as "number")).toThrow(InvalidParameterError);
  });

  it("any and all accept axis aliases and reject unknown axes", () => {
    const df = new DataFrame({ a: [0, 1], b: [0, 0] });
    expect(df.any("columns")).toEqual([false, true]);
    expect(df.all(1)).toEqual([false, false]);
    expect((df.any() as { data: readonly boolean[] }).data).toEqual([true, false]);
    expect(() => df.any(2)).toThrow();
  });

  it("value_counts separates values that stringify alike", () => {
    const df = new DataFrame({ a: [1, "1", 1, "1", null, "null"] });
    const out = df.value_counts("a");
    expect(out.shape[0]).toBe(4);
    expect(col(out, "count").reduce((x, y) => (x as number) + (y as number), 0)).toBe(6);
  });
});

describe("c03 DataFrame: reshaping", () => {
  it("melt defaults value_vars to every other column", () => {
    const df = new DataFrame({ id: ["a", "b"], x: [1, 2], y: [3, 4] });
    const out = df.melt(["id"]);
    expect(out.columns).toEqual(["id", "variable", "value"]);
    expect(out.toArray()).toEqual([
      ["a", "x", 1],
      ["a", "y", 3],
      ["b", "x", 2],
      ["b", "y", 4],
    ]);
  });

  it("stack rejects duplicate and clashing names", () => {
    const df = new DataFrame({ id: [1], x: [2], y: [3] });
    expect(() => df.stack({ columns: ["x", "x"] })).toThrow(DataValidationError);
    expect(() => df.stack({ columns: ["x", "y"], varName: "id" })).toThrow(/must not conflict/);
    expect(df.stack({ columns: ["x", "y"] }).toArray()).toEqual([
      [1, "x", 2],
      [1, "y", 3],
    ]);
  });

  it("unstack throws on duplicate (index, column) pairs", () => {
    const df = new DataFrame({ i: ["a", "a"], c: ["x", "x"], v: [1, 2] });
    expect(() => df.unstack({ index: "i", column: "c", value: "v" })).toThrow(
      /Duplicate unstack entry/
    );
  });

  it("pivot_table sorts labels, skips missing keys and supports median", () => {
    const df = new DataFrame({
      r: ["b", "a", "b", null, "a", "b"],
      c: ["y", "x", "x", "x", "y", "y"],
      v: [1, 2, 3, 99, 4, 5],
    });
    const out = df.pivot_table({ index: "r", columns: "c", values: "v", aggFunc: "median" });
    expect(out.index).toEqual(["a", "b"]);
    expect(out.columns).toEqual(["x", "y"]);
    expect(out.toArray()).toEqual([
      [2, 4],
      [3, 3],
    ]);
    expect(() =>
      df.pivot_table({ index: "r", columns: "c", values: "v", aggFunc: "x" as "mean" })
    ).toThrow(InvalidParameterError);
  });

  it("pivot_table keeps number and string row labels apart and sorts numbers numerically", () => {
    const df = new DataFrame({ r: [10, 9, 10], c: ["a", "a", "a"], v: [1, 2, 3] });
    const out = df.pivot_table({ index: "r", columns: "c", values: "v", aggFunc: "sum" });
    expect(out.index).toEqual([9, 10]);
    expect(col(out, "a")).toEqual([2, 4]);
  });

  it("pivot_table min and max survive very large groups", () => {
    const n = 250_000;
    const df = new DataFrame({
      r: new Array(n).fill("a"),
      c: new Array(n).fill("x"),
      v: Array.from({ length: n }, (_, i) => i),
    });
    expect(
      col(df.pivot_table({ index: "r", columns: "c", values: "v", aggFunc: "max" }), "x")
    ).toEqual([n - 1]);
    expect(
      col(df.pivot_table({ index: "r", columns: "c", values: "v", aggFunc: "min" }), "x")
    ).toEqual([0]);
  });

  it("crosstab sorts labels and ignores rows with a missing value", () => {
    const df = new DataFrame({
      gender: ["M", "F", "M", "F", "M", null],
      handed: ["R", "R", "L", "R", "R", "L"],
    });
    const ct = df.crosstab("gender", "handed");
    expect(ct.index).toEqual(["F", "M"]);
    expect(ct.columns).toEqual(["L", "R"]);
    expect(ct.toArray()).toEqual([
      [0, 2],
      [1, 2],
    ]);
  });

  it("crosstab keeps 1 and '1' in separate rows", () => {
    const df = new DataFrame({ a: [1, "1"], b: ["x", "x"] });
    const ct = df.crosstab("a", "b");
    expect(ct.shape).toEqual([2, 1]);
  });
});

describe("c03 DataFrame: tensors", () => {
  it("toTensor defaults to float32 and can keep float64 precision", () => {
    const df = new DataFrame({ a: [16777217, 0.1], b: [1, null] });
    const f32 = df.toTensor();
    expect(f32.dtype).toBe("float32");
    expect(f32.shape).toEqual([2, 2]);
    const f64 = df.toTensor({ dtype: "float64" });
    expect(f64.dtype).toBe("float64");
    const arr = f64.toArray() as number[][];
    expect(arr[0]).toEqual([16777217, 1]);
    expect(arr[1]?.[0]).toBe(0.1);
    expect(arr[1]?.[1]).toBeNaN();
    expect(() => df.toTensor({ dtype: "int32" as "float32" })).toThrow(InvalidParameterError);
  });

  it("toTensor handles empty frames", () => {
    expect(new DataFrame({ a: [] }).toTensor().shape).toEqual([0, 1]);
    expect(new DataFrame({}, { index: [0, 1] }).toTensor().shape).toEqual([2, 0]);
  });

  it("fromTensor reads transposed and sliced views in logical order", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const tt = transpose(t);
    const df = DataFrame.fromTensor(tt, ["x", "y"]);
    expect(df.toArray()).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
    const colView = t.slice({ start: 0, end: 2 }, { start: 1, end: 2 });
    expect(DataFrame.fromTensor(colView).toArray()).toEqual([[2], [5]]);
  });

  it("fromTensor accepts 1D tensors and rejects other ranks", () => {
    expect(DataFrame.fromTensor(tensor([1, 2, 3]), ["v"]).toArray()).toEqual([[1], [2], [3]]);
    expect(DataFrame.fromTensor(tensor(["a", "b"])).toArray()).toEqual([["a"], ["b"]]);
    expect(() => DataFrame.fromTensor(tensor(5))).toThrow(DataValidationError);
    expect(() => DataFrame.fromTensor(tensor([[1, 2]]), ["only"])).toThrow(DataValidationError);
  });
});

describe("c03 DataFrame: CSV and JSON", () => {
  it("names empty header cells and parses NaN literals", () => {
    const df = DataFrame.fromCsvString(",b\n1,NaN\n2,nan\n");
    expect(df.columns).toEqual(["Unnamed: 0", "b"]);
    expect(col(df, "b")).toEqual([Number.NaN, Number.NaN]);
    expect(col(df, "Unnamed: 0")).toEqual([1, 2]);
  });

  it("round-trips NaN, null, dates and single empty columns", () => {
    const df = new DataFrame({
      n: [1, Number.NaN, null, 4],
      d: [new Date("2024-01-02T03:04:05.000Z"), null, null, null],
    });
    const csv = df.toCsvString();
    expect(csv.split("\n")[1]).toBe("1,2024-01-02T03:04:05.000Z");
    const back = DataFrame.fromCsvString(csv);
    expect(col(back, "n")).toEqual([1, Number.NaN, null, 4]);

    const single = new DataFrame({ a: [null, "x", null] });
    const text = single.toCsvString();
    expect(text).toBe('a\n""\nx\n""');
    const reread = DataFrame.fromCsvString(text);
    expect(reread.shape).toEqual([3, 1]);
    expect(col(reread, "a")).toEqual([null, "x", null]);
  });

  it("keeps a quoted empty last field without a trailing newline", () => {
    const df = DataFrame.fromCsvString('a\n1\n""');
    expect(col(df, "a")).toEqual([1, null]);
  });

  it("validates delimiter, quote and skipRows", () => {
    expect(() => DataFrame.fromCsvString("a,b\n1,2", { delimiter: ";;" })).toThrow(
      InvalidParameterError
    );
    expect(() => DataFrame.fromCsvString("a,b\n1,2", { delimiter: "" })).toThrow(
      InvalidParameterError
    );
    expect(() => DataFrame.fromCsvString("a,b\n1,2", { delimiter: '"' })).toThrow(
      InvalidParameterError
    );
    expect(() => DataFrame.fromCsvString("a,b\n1,2", { skipRows: -1 })).toThrow(
      InvalidParameterError
    );
    expect(() => new DataFrame({ a: [1] }).toCsvString({ quoteChar: "ab" })).toThrow(
      InvalidParameterError
    );
  });

  it("writes arrays as JSON and escapes embedded quotes", () => {
    const df = new DataFrame({ a: [[1, 2]], b: ['say "hi", ok'] });
    expect(df.toCsvString()).toBe('a,b\n"[1,2]","say ""hi"", ok"');
  });

  it("fromJsonString handles columns named like Object.prototype members", () => {
    const df = new DataFrame({ a: [1, 2] }, { index: ["r1", "r2"] });
    expect(DataFrame.fromJsonString(df.toJsonString()).toArray()).toEqual([[1], [2]]);
    const proto = '{"columns":["__proto__"],"index":[0],"data":{"__proto__":[5]}}';
    const parsed = DataFrame.fromJsonString(proto);
    expect(parsed.columns).toEqual(["__proto__"]);
    expect(parsed.toArray()).toEqual([[5]]);
    expect(() =>
      DataFrame.fromJsonString('{"columns":["toString"],"index":[0],"data":{}}')
    ).toThrow(/Missing data for column 'toString'/);
  });
});

describe("c03 DataFrame: duplicates", () => {
  it("validates keep and honours subset", () => {
    const df = new DataFrame({ a: [1, 1, 2, 1], b: [1, 1, 2, 9] });
    expect(df.drop_duplicates().index).toEqual([0, 2, 3]);
    expect(df.drop_duplicates(undefined, "last").index).toEqual([1, 2, 3]);
    expect(df.drop_duplicates(undefined, false).index).toEqual([2, 3]);
    expect(df.drop_duplicates(["a"]).index).toEqual([0, 2]);
    expect(df.duplicated(["a"], false).data).toEqual([true, true, false, true]);
    expect(df.duplicated().data).toEqual([false, true, false, false]);
    expect(() => df.drop_duplicates(undefined, "middle" as "first")).toThrow(InvalidParameterError);
    expect(() => df.duplicated(["zzz"])).toThrow(DataValidationError);
  });
});

describe("c03 DataFrame: query", () => {
  const df = new DataFrame({
    a: [1, 2, 3, 4],
    b: [1, 0, 1, 0],
    c: [1, 1, 0, 0],
    name: ["Tom and Jerry", "Ann", "or else", "Bob"],
    "full name": ["x", "y", "x", "y"],
  });

  it("gives 'and' precedence over 'or'", () => {
    // a == 1 or (b == 1 and c == 0) -> rows 0 and 2
    expect(col(df.query("a == 1 or b == 1 and c == 0"), "a")).toEqual([1, 3]);
    expect(col(df.query("b == 1 and c == 0 or a == 4"), "a")).toEqual([3, 4]);
  });

  it("does not split connectors inside quoted strings", () => {
    expect(col(df.query("name == 'Tom and Jerry'"), "a")).toEqual([1]);
    expect(col(df.query('name == "or else"'), "a")).toEqual([3]);
  });

  it("accepts symbolic connectors and backticked names", () => {
    expect(col(df.query("a > 1 && b == 0"), "a")).toEqual([2, 4]);
    expect(col(df.query("a == 1 | a == 4"), "a")).toEqual([1, 4]);
    expect(col(df.query("`full name` == 'x' and a > 1"), "a")).toEqual([3]);
  });

  it("compares strings and matches missing values explicitly", () => {
    // Strings compare by code unit: "Ann" < "B" <= "Bob", and lowercase sorts after uppercase.
    expect(col(df.query("name >= 'B'"), "a")).toEqual([1, 3, 4]);
    const gaps = new DataFrame({ a: [1, null, Number.NaN, 4] });
    expect(col(gaps.query("a == null"), "a")).toEqual([null, Number.NaN]);
    expect(col(gaps.query("a != null"), "a")).toEqual([1, 4]);
  });

  it("reports parse errors and unknown columns", () => {
    expect(() => df.query("a >")).toThrow(InvalidParameterError);
    expect(() => df.query("zzz > 1")).toThrow(/not found/);
    expect(() => df.query("name == 'oops")).toThrow(/Unterminated quote/);
  });
});

describe("c03 DataFrame: eval", () => {
  const df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6], f: [true, false, true] });

  it("follows Python precedence for unary minus and power", () => {
    // python: -a ** 2 == -(a ** 2); 2 ** 3 ** 2 == 512
    expect(col(df.eval("r = -a ** 2"), "r")).toEqual([-1, -4, -9]);
    expect(col(df.eval("r = 2 ** 3 ** 2 + a * 0"), "r")).toEqual([512, 512, 512]);
    expect(col(df.eval("r = 2 ** -1 + a * 0"), "r")).toEqual([0.5, 0.5, 0.5]);
  });

  it("supports modulo with Python semantics", () => {
    // python: -7 % 3 == 2, 7 % -3 == -2
    const m = new DataFrame({ a: [-7, 7, 8], b: [3, -3, 3] });
    expect(col(m.eval("r = a % b"), "r")).toEqual([2, -2, 2]);
  });

  it("supports logic operators and booleans as numbers", () => {
    expect(col(df.eval("a > 1 and b < 6"), "a")).toEqual([2]);
    expect(col(df.eval("a == 1 or a == 3"), "a")).toEqual([1, 3]);
    expect(col(df.eval("not (a > 1)"), "a")).toEqual([1]);
    expect(col(df.eval("s = f + 1"), "s")).toEqual([2, 1, 2]);
    expect(col(df.eval("e = 1e1 + .5 + a * 0"), "e")).toEqual([10.5, 10.5, 10.5]);
  });

  it("rejects unknown names, stray characters and unbalanced input", () => {
    expect(() => df.eval("zzz + 1")).toThrow(/unknown name 'zzz'/);
    expect(() => df.eval("a @ b")).toThrow(/unexpected character '@'/);
    expect(() => df.eval("(a + b")).toThrow(/missing closing parenthesis/);
    expect(() => df.eval("a + b )")).toThrow(/unexpected '\)'/);
    expect(() => df.eval("a + ")).toThrow(/unexpected end/);
    expect(() => df.eval("a b")).toThrow(/unexpected 'b'/);
    expect(() => df.eval("   ")).toThrow(InvalidParameterError);
  });

  it("does not mistake == for an assignment", () => {
    expect(col(df.eval("a == 2"), "a")).toEqual([2]);
    expect(df.eval("a == 2").columns).toEqual(["a", "b", "f"]);
  });
});

describe("c03 DataFrame: interpolate", () => {
  it("matches pandas for linear interpolation", () => {
    // pandas: Series([nan, 1, nan, nan, 4, nan, nan]).interpolate()
    //   -> [nan, 1, 2, 3, 4, 4, 4]
    const df = new DataFrame({ a: [null, 1, null, Number.NaN, 4, null, null] });
    expect(col(df.interpolate(), "a")).toEqual([null, 1, 2, 3, 4, 4, 4]);
  });

  it("matches pandas for nearest interpolation", () => {
    // pandas: same series, method="nearest" -> [nan, 1, 1, 4, 4, nan, nan]
    const df = new DataFrame({ a: [null, 1, null, Number.NaN, 4, null, null] });
    expect(col(df.interpolate("nearest"), "a")).toEqual([null, 1, 1, 4, 4, null, null]);
    // pandas: [10, nan, nan, nan, 50] -> [10, 10, 10, 50, 50]
    const t = new DataFrame({ a: [10, null, null, null, 50] });
    expect(col(t.interpolate("nearest"), "a")).toEqual([10, 10, 10, 50, 50]);
  });

  it("is linear in the row position, uses original anchors and is fast", () => {
    const n = 100_000;
    const data: (number | null)[] = new Array(n).fill(null);
    data[0] = 0;
    data[n - 1] = n - 1;
    const out = col(new DataFrame({ a: data }).interpolate(), "a") as number[];
    expect(out[50_000]).toBeCloseTo(50_000, 6);
    expect(out[n - 2]).toBeCloseTo(n - 2, 6);
  });

  it("leaves strings alone and validates the method", () => {
    const df = new DataFrame({ a: ["x", null, "y"] });
    expect(col(df.interpolate(), "a")).toEqual(["x", null, "y"]);
    expect(() => df.interpolate("cubic" as "linear")).toThrow(InvalidParameterError);
  });
});

describe("c03 DataFrame: ewm", () => {
  const values = [1, 2, 3, 4, 5.5, 2, Number.NaN, 7, 1.5];
  const df = new DataFrame({ a: values });

  // Reference values from pandas 3.0:
  //   s.ewm(span=4, adjust=..., ignore_na=...).mean() and .var(bias=...)
  // span=4 means alpha=0.4. (pandas 3.0 returns a result for alpha=0.5 and adjust=False
  // that does not follow its documented weights, so that alpha is not used as a reference.)
  it("mean matches pandas for every adjust / ignoreNa combination", () => {
    expectClose(
      col(df.ewm({ span: 4 }).mean(), "a"),
      [1, 1.4, 2.04, 2.824, 3.8944, 3.13664, 3.13664, 5.169987368421053, 3.7019924210526316]
    );
    expectClose(
      col(df.ewm({ span: 4, ignoreNa: true }).mean(), "a"),
      [1, 1.4, 2.04, 2.824, 3.8944, 3.13664, 3.13664, 4.681984, 3.4091904]
    );
    expectClose(
      col(df.ewm({ span: 4, adjust: true }).mean(), "a"),
      [
        1, 1.625, 2.326530612244898, 3.0955882352941173, 4.138445523941707, 3.241205692803437,
        3.241205692803437, 5.264227698285305, 3.4842875404311364,
      ]
    );
    expectClose(
      col(df.ewm({ span: 4, adjust: true, ignoreNa: true }).mean(), "a"),
      [
        1, 1.625, 2.326530612244898, 3.0955882352941173, 4.138445523941707, 3.241205692803437,
        3.241205692803437, 4.788024440991336, 3.4503468171971345,
      ]
    );
  });

  it("mean weights a value after a gap by alpha, not by the weight the old value lost", () => {
    // pandas: Series([1, nan, nan, 4, 7]).ewm(alpha=0.3, adjust=False).mean()
    const gap = new DataFrame({ a: [1, Number.NaN, Number.NaN, 4, 7] });
    expectClose(
      col(gap.ewm({ alpha: 0.3 }).mean(), "a"),
      [1, 1, 1, 2.3996889580093317, 3.7797822706065323]
    );
  });

  it("var matches pandas for bias and adjust settings", () => {
    // Index 0 is null here (pandas gives 0.0 or NaN for a single observation).
    expectClose(col(df.ewm({ span: 4 }).var(), "a"), [
      null,
      0.24,
      0.7584000000000001,
      1.377024,
      2.54484864,
      2.3882095104000003,
      2.3882095104000003,
      4.8523085051036015,
      6.143898851310981,
    ]);
    expectClose(col(df.ewm({ span: 4, bias: false }).var(), "a"), [
      null,
      0.5,
      1.161764705882353,
      1.925886143931257,
      3.451096692217964,
      3.203650597285799,
      3.2036505972857996,
      7.287061854568949,
      8.536550701094198,
    ]);
    expectClose(col(df.ewm({ span: 4, adjust: true, bias: false }).var(), "a"), [
      null,
      0.4999999999999999,
      0.9591836734693878,
      1.4993997599039617,
      2.816304037229049,
      3.1297569230402273,
      3.1297569230402273,
      6.996185146764312,
      8.733548802468666,
    ]);
    expectClose(col(df.ewm({ span: 4, adjust: true, bias: false, ignoreNa: true }).var(), "a"), [
      null,
      0.4999999999999999,
      0.9591836734693878,
      1.4993997599039617,
      2.816304037229049,
      3.1297569230402273,
      3.1297569230402273,
      6.4679256968606795,
      7.324898781651524,
    ]);
    // pandas: s.ewm(span=4, adjust=True).std(bias=True)
    const std = col(df.ewm({ span: 4, adjust: true }).std(), "a") as (number | null)[];
    expect(std[1]).toBeCloseTo(0.4841229182759271, 12);
    expect(std[8]).toBeCloseTo(2.4363380314659, 12);
  });

  it("accepts com and halflife and validates every parameter", () => {
    const a = col(df.ewm({ com: 1 }).mean(), "a");
    const b = col(df.ewm({ alpha: 0.5 }).mean(), "a");
    expect(a).toEqual(b);
    const h = col(df.ewm({ halflife: 1 }).mean(), "a");
    expect(h).toEqual(col(df.ewm({ alpha: 0.5 }).mean(), "a"));
    expect(() => df.ewm({ span: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => df.ewm({ span: 0.5 })).toThrow(InvalidParameterError);
    expect(() => df.ewm({ alpha: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => df.ewm({ alpha: 1.5 })).toThrow(InvalidParameterError);
    expect(() => df.ewm({ com: -1 })).toThrow(InvalidParameterError);
    expect(() => df.ewm({ halflife: 0 })).toThrow(InvalidParameterError);
    expect(() => df.ewm({})).toThrow(InvalidParameterError);
  });
});

describe("c03 DataFrame: expanding and rolling", () => {
  const df = new DataFrame({ a: [1, 2, 3, 4, 5.5, 2, Number.NaN, 7, 1.5] });

  it("expanding std and mean match pandas", () => {
    // pandas: Series.expanding().std() and Series.expanding(2).mean()
    expectClose(col(df.expanding().std(), "a"), [
      null,
      Math.SQRT1_2,
      1,
      1.2909944487358056,
      1.746424919657298,
      1.6253204812179862,
      1.6253204812179862,
      2.140872096444188,
      2.1044171232366047,
    ]);
    expectClose(col(df.expanding(2).mean(), "a"), [
      null,
      1.5,
      2,
      2.5,
      3.1,
      2.9166666666666665,
      2.9166666666666665,
      3.5,
      3.25,
    ]);
  });

  it("expanding validates minPeriods and runs in linear time", () => {
    expect(() => df.expanding(-1)).toThrow(InvalidParameterError);
    expect(col(df.expanding(0).sum(), "a")[0]).toBe(1);
    expect(() => df.expanding(1.5)).toThrow(InvalidParameterError);
    const n = 200_000;
    const big = new DataFrame({ a: Array.from({ length: n }, (_, i) => i % 7) });
    const std = col(big.expanding().std(), "a");
    expect(std[n - 1]).toBeCloseTo(2.0, 3);
  });

  it("expanding min and max skip missing values", () => {
    const d = new DataFrame({ a: [3, Number.NaN, 1, 5] });
    expect(col(d.expanding().min(), "a")).toEqual([3, 3, 1, 1]);
    expect(col(d.expanding().max(), "a")).toEqual([3, 3, 3, 5]);
    expect(col(d.expanding(2).sum(), "a")).toEqual([null, null, 4, 9]);
  });

  it("rolling mean survives values of very different magnitude", () => {
    const d = new DataFrame({ a: [1e16, 1, 1, 1, 1] });
    const mean = col(d.rolling(3).mean(), "a");
    expect(mean[3]).toBe(1);
    expect(mean[4]).toBe(1);
    const sum = col(d.rolling(3).sum(), "a");
    expect(sum[3]).toBe(3);
  });

  it("rolling median, min and max handle large windows", () => {
    const d = new DataFrame({ a: [5, 1, 4, 2, 3] });
    expect(col(d.rolling(3).median(), "a")).toEqual([null, null, 4, 2, 3]);
    expect(col(d.rolling(2).median(), "a")).toEqual([null, 3, 2.5, 3, 2.5]);
    const n = 150_000;
    const big = new DataFrame({ a: Array.from({ length: n }, (_, i) => i) });
    expect(col(big.rolling(n).max(), "a")[n - 1]).toBe(n - 1);
    expect(col(big.rolling(n).min(), "a")[n - 1]).toBe(0);
  });

  it("rolling std matches pandas", () => {
    // pandas: Series([5, 1, 4, 2, 3]).rolling(3).std()
    const d = new DataFrame({ a: [5, 1, 4, 2, 3] });
    expectClose(col(d.rolling(3).std(), "a"), [
      null,
      null,
      2.0816659994661326,
      1.5275252316519468,
      1,
    ]);
  });
});

describe("c03 DataFrame: groupBy", () => {
  it("keeps infinities in aggregations", () => {
    const df = new DataFrame({ g: ["a", "a", "b"], v: [1, Infinity, -Infinity] });
    const out = df.groupBy("g").agg({ v: ["sum", "max", "min", "mean"] });
    expect(col(out, "v_sum")).toEqual([Infinity, -Infinity]);
    expect(col(out, "v_max")).toEqual([Infinity, -Infinity]);
    expect(col(out, "v_min")).toEqual([1, -Infinity]);
    expect(col(out, "v_mean")).toEqual([Infinity, -Infinity]);
  });

  it("sums with compensation", () => {
    const df = new DataFrame({ g: ["a", "a", "a", "a"], v: [1e16, 1, 1, -1e16] });
    expect(col(df.groupBy("g").sum(), "v")).toEqual([2]);
  });

  it("validates the grouping columns and the aggregation request", () => {
    const df = new DataFrame({ g: ["a"], v: [1] });
    expect(() => df.groupBy([])).toThrow(InvalidParameterError);
    expect(() => df.groupBy("zzz")).toThrow(InvalidParameterError);
    expect(() => df.groupBy(["g", "g"])).toThrow(DataValidationError);
    const empty = new DataFrame({ g: [], v: [] });
    expect(() => empty.groupBy("g").agg({ missing: "sum" })).toThrow(InvalidParameterError);
    expect(() => empty.groupBy("g").agg({ v: "mode" as "sum" })).toThrow(/Unsupported/);
  });

  it("count skips NaN and offers first, last and size", () => {
    const df = new DataFrame({
      g: ["a", "b", "a", "a"],
      v: [Number.NaN, 2, 3, null],
      s: ["p", "q", "r", "t"],
    });
    const grouped = df.groupBy("g");
    expect(col(grouped.count(), "v")).toEqual([1, 1]);
    expect(col(grouped.first(), "s")).toEqual(["p", "q"]);
    expect(col(grouped.last(), "s")).toEqual(["t", "q"]);
    const size = grouped.size();
    expect(size.columns).toEqual(["g", "size"]);
    expect(col(size, "size")).toEqual([3, 1]);
    expect(grouped.ngroups).toBe(2);
  });

  it("median matches NumPy and std needs two values", () => {
    const df = new DataFrame({ g: ["a", "a", "a", "a", "b"], v: [1, 2, 4, 10, 7] });
    const out = df.groupBy("g").agg({ v: ["median", "std", "var"] });
    expect(col(out, "v_median")).toEqual([3, 7]);
    expect(col(out, "v_std")[0] as number).toBeCloseTo(4.031128874149275, 12);
    expect(col(out, "v_std")[1]).toBeNaN();
    expect(col(out, "v_var")[0] as number).toBeCloseTo(16.25, 12);
  });
});

describe("c03 DataFrame: copies and aliasing", () => {
  it("copy() does not share data or labels with the original", () => {
    const df = new DataFrame({ a: [1, 2] }, { index: ["x", "y"] });
    const copy = df.copy();
    expect(copy.toArray()).toEqual(df.toArray());
    expect(copy.index).toEqual(["x", "y"]);
    const derived = df.fillna(0);
    expect(derived.getColumnData("a")).not.toBe(df.getColumnData("a"));
  });

  it("toString validates maxRows and aligns columns", () => {
    const df = new DataFrame({ a: [1, 2], b: [3, 4] });
    expect(df.toString()).toBe("   a  b\n0  1  3\n1  2  4");
    expect(() => df.toString(-1)).toThrow(InvalidParameterError);
    const big = new DataFrame({ a: Array.from({ length: 30 }, (_, i) => i) });
    expect(big.toString(4).split("\n")).toHaveLength(6);
  });
});

describe("c03 DataFrame: review fixes", () => {
  it("derived frames keep a column named like an Object.prototype member", () => {
    const json = '{"columns":["__proto__","b"],"index":[0,1],"data":{"__proto__":[5,6],"b":[1,2]}}';
    const df = DataFrame.fromJsonString(json);
    expect(df.head(1).columns).toEqual(["__proto__", "b"]);
    expect(df.sort("b", false).toArray()).toEqual([
      [6, 2],
      [5, 1],
    ]);
    expect(df.fillna(0).toArray()).toEqual([
      [5, 1],
      [6, 2],
    ]);
    expect(df.describe().columns).toEqual(["__proto__", "b"]);
    expect(Object.keys(df.loc(0))).toEqual(["__proto__", "b"]);
    const tricky = new DataFrame({ toString: [1, 2], constructor: [3, 4] });
    expect(tricky.select(["constructor"]).toArray()).toEqual([[3], [4]]);
    expect(tricky.sort("toString", false).toArray()).toEqual([
      [2, 4],
      [1, 3],
    ]);
  });

  it("round with a very negative decimals value gives zero, not NaN", () => {
    // numpy: np.round([123.4, -5.0, 0.0], -400) -> [0, -0, 0]
    expect(col(new DataFrame({ a: [123.4, -5, 0] }).round(-400), "a")).toEqual([0, -0, 0]);
  });

  it("eval modulo keeps Python semantics for tiny and infinite operands", () => {
    // python: -1e-20 % 1 == 1.0, 5 % float("inf") == 5.0, -5 % float("inf") == inf
    const m = new DataFrame({ a: [-1e-20, 5, -5], b: [1, Infinity, Infinity] });
    expect(col(m.eval("r = a % b"), "r")).toEqual([1, 5, Infinity]);
  });

  it("query rejects a dangling connector instead of returning no rows", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    expect(() => df.query("a > 1 or")).toThrow(InvalidParameterError);
    expect(() => df.query("a > 1 and")).toThrow(InvalidParameterError);
    expect(() => df.query("or a > 1")).toThrow(InvalidParameterError);
    expect(col(df.query("a > 1 AND a < 3"), "a")).toEqual([2]);
  });

  it("ewm mean agrees with pandas at several alphas after a gap", () => {
    // pandas: Series([1, nan, 7]).ewm(alpha=a, adjust=False).mean()[2]
    const gap = new DataFrame({ a: [1, Number.NaN, 7] });
    const expected: [number, number][] = [
      [0.1, 1.6593406593406594],
      [0.3, 3.278481012658228],
      [0.6, 5.736842105263158],
      [0.9, 6.934065934065933],
    ];
    for (const [alpha, want] of expected) {
      expect(col(gap.ewm({ alpha }).mean(), "a")[2] as number).toBeCloseTo(want, 12);
    }
  });
});
