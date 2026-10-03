import { describe, expect, it } from "vitest";
import {
  DataFrame,
  DataFrameGroupBy,
  EWM,
  Expanding,
  type GroupByOptions,
  type ParquetReadResult,
  Rolling,
  readParquet,
  readXlsx,
  Series,
  writeParquet,
  type XlsxReadResult,
} from "../../src/dataframe";
import { createKey } from "../../src/dataframe/utils";

const column = (df: DataFrame, name: string): unknown[] => [...df.getColumnData(name)];

describe("wave2 dataframe: exports and IO result types", () => {
  it("exports the window and groupby classes returned by the DataFrame methods", () => {
    const df = new DataFrame({ a: [1, 2, 3], g: ["x", "y", "x"] });
    expect(df.rolling(2)).toBeInstanceOf(Rolling);
    expect(df.expanding()).toBeInstanceOf(Expanding);
    expect(df.ewm({ alpha: 0.5 })).toBeInstanceOf(EWM);
    expect(df.groupBy("g")).toBeInstanceOf(DataFrameGroupBy);
  });

  it("names the IO result types and options", () => {
    const parquet: ParquetReadResult = readParquet(writeParquet(["a"], [{ a: 1 }]));
    expect(parquet.columns).toEqual(["a"]);
    const xlsx: XlsxReadResult = readXlsx(new DataFrame({ a: [1] }).toXlsx());
    expect(xlsx.data[0]?.["a"]).toBe(1);
    const options: GroupByOptions = { sort: true, dropna: true };
    expect(options.sort).toBe(true);
  });
});

describe("wave2 dataframe: Parquet and XLSX wrappers", () => {
  it("handles empty tables and rejects invalid buffers", () => {
    const empty = new DataFrame({ a: [], b: [] });
    const back = DataFrame.fromParquet(empty.toParquet());
    expect(back.columns).toEqual(["a", "b"]);
    expect(back.shape).toEqual([0, 2]);
    expect(() => DataFrame.fromParquet(new Uint8Array(3))).toThrow(/too short/);
    expect(() => DataFrame.fromParquet(new Uint8Array(40))).toThrow(/PAR1/);
  });

  it("round trips numbers, text, dates, bigints and nulls through Parquet", () => {
    const df = new DataFrame({
      n: [1, 2.5, null],
      s: ["a", null, "c"],
      d: [new Date(0), new Date(86_400_000), null],
      big: [2n ** 60n, 5n, 7n],
      flag: [true, false, true],
    });
    const back = DataFrame.fromParquet(df.toParquet());
    expect(back.columns).toEqual(df.columns);
    expect(column(back, "n")).toEqual([1, 2.5, null]);
    expect(column(back, "s")).toEqual(["a", null, "c"]);
    expect(column(back, "d")).toEqual([new Date(0), new Date(86_400_000), null]);
    expect(column(back, "big")).toEqual([2n ** 60n, 5, 7]);
    expect(column(back, "flag")).toEqual([true, false, true]);
    expect(back.index).toEqual([0, 1, 2]);
  });

  it("selects and orders columns when reading Parquet", () => {
    const df = new DataFrame({ a: [1, 2], b: [3, 4], c: [5, 6] });
    const back = DataFrame.fromParquet(df.toParquet(), { columns: ["c", "a"] });
    expect(back.columns).toEqual(["c", "a"]);
    expect(column(back, "c")).toEqual([5, 6]);
  });

  it("keeps a __proto__ column through the wrappers", () => {
    const data = Object.create(null) as Record<string, unknown[]>;
    Object.defineProperty(data, "__proto__", { value: [1, 2], enumerable: true });
    data["x"] = ["u", "v"];
    const df = new DataFrame(data);
    expect(column(DataFrame.fromParquet(df.toParquet()), "__proto__")).toEqual([1, 2]);
    expect(column(DataFrame.fromXlsx(df.toXlsx()), "__proto__")).toEqual([1, 2]);
  });

  it("round trips through XLSX and passes the options on", () => {
    const df = new DataFrame({ a: [1, 2, 3], b: ["x", null, "z"], c: [true, false, true] });
    const back = DataFrame.fromXlsx(df.toXlsx({ sheetName: "Data" }), { sheet: "Data" });
    expect(back.columns).toEqual(["a", "b", "c"]);
    expect(column(back, "a")).toEqual([1, 2, 3]);
    expect(column(back, "b")).toEqual(["x", null, "z"]);
    expect(column(back, "c")).toEqual([true, false, true]);
    expect(() => df.toXlsx({ sheetName: "bad/name" })).toThrow();
    const noHeader = DataFrame.fromXlsx(df.toXlsx(), { header: false });
    expect(noHeader.columns).toEqual(["Column0", "Column1", "Column2"]);
    expect(noHeader.shape[0]).toBe(4);
  });

  it("works with the values Parquet can return in the other DataFrame methods", () => {
    const df = DataFrame.fromParquet(
      new DataFrame({
        d: [new Date(1000), new Date(0)],
        big: [2n ** 60n, 5n],
        k: ["a", "b"],
      }).toParquet()
    );
    expect(column(df.sort("d"), "k")).toEqual(["b", "a"]);
    expect(column(df.sort("big"), "k")).toEqual(["b", "a"]);
    expect(df.groupBy("d").size().shape[0]).toBe(2);
    // The mixed bigint/number column still reports statistics for its plain numbers.
    expect(df.describe().columns).toEqual(["big"]);
    expect(() => df.toCsvString()).not.toThrow();
    // JSON has no bigint: safe values become numbers, larger ones decimal strings.
    const json = JSON.parse(df.toJsonString()) as { data: Record<string, unknown[]> };
    expect(json.data["big"]).toEqual(["1152921504606846976", 5]);
    expect(json.data["d"]).toEqual(["1970-01-01T00:00:01.000Z", "1970-01-01T00:00:00.000Z"]);
  });

  it("prints dates as ISO 8601 text in toString", () => {
    const df = new DataFrame({ d: [new Date(0), new Date(Number.NaN)] });
    expect(df.toString()).toContain("1970-01-01T00:00:00.000Z");
    expect(df.toString()).toContain("Invalid Date");
    expect(new Series([new Date(0)]).toString()).toContain("1970-01-01T00:00:00.000Z");
  });
});

describe("wave2 dataframe: Series", () => {
  it("toTensor honours dtype and keeps float64 values exact", () => {
    const s = new Series([16_777_217, 0.1 + 0.2, null]);
    const wide = s.toTensor({ dtype: "float64" });
    expect(wide.dtype).toBe("float64");
    expect(Array.from(wide.data as Float64Array)).toEqual([16_777_217, 0.1 + 0.2, Number.NaN]);
    const narrow = s.toTensor();
    expect(narrow.dtype).toBe("float32");
    expect((narrow.data as Float32Array)[0]).toBe(16_777_216);
    expect(() => s.toTensor({ dtype: "int32" as unknown as "float32" })).toThrow(/dtype/);
    expect(() => new Series(["a"]).toTensor({ dtype: "float64" })).toThrow(/numeric/);
  });

  it("has isnull and notnull that treat NaN, null, undefined and invalid dates as missing", () => {
    const s = new Series<unknown>([1, null, Number.NaN, undefined, new Date(Number.NaN), "x"], {
      name: "v",
      index: ["a", "b", "c", "d", "e", "f"],
    });
    expect(s.isnull().toArray()).toEqual([false, true, true, true, true, false]);
    expect(s.notnull().toArray()).toEqual([true, false, false, false, false, true]);
    expect(s.isnull().index).toEqual(["a", "b", "c", "d", "e", "f"]);
    expect(s.isnull().name).toBe("v");
  });

  it("sorts strings by code point, not by locale", () => {
    const s = new Series(["b", "B", "a", "A", "é", "z"]);
    expect(s.sort().toArray()).toEqual(["A", "B", "a", "b", "z", "é"]);
    expect(s.sort(false).toArray()).toEqual(["é", "z", "b", "a", "B", "A"]);
  });
});

describe("wave2 dataframe: sorting and sums agree between Series and DataFrame", () => {
  it("DataFrame.sort orders strings by code point like Series.sort", () => {
    const values = ["b", "B", "a", "A", "é", "z"];
    const df = new DataFrame({ s: values, i: values.map((_, k) => k) });
    expect(column(df.sort("s"), "s")).toEqual(new Series(values).sort().toArray());
    expect(column(df.sort("s", false), "s")).toEqual(new Series(values).sort(false).toArray());
  });

  it("groupby var, std, sum and mean give the same bits as the Series methods", () => {
    const vals = [1e8 + 0.1, 1e8 + 0.2, 1e8 + 0.3, 1e8 + 0.7, 1e8 + 0.9, 1e8 + 0.35, 1e8 + 0.45];
    const df = new DataFrame({ g: vals.map(() => "a"), v: vals });
    const s = new Series(vals);
    const agg = df.groupBy("g").agg({ v: ["var", "std", "sum", "mean"] });
    expect(column(agg, "v_var")[0]).toBe(s.var());
    expect(column(agg, "v_std")[0]).toBe(s.std());
    expect(column(agg, "v_sum")[0]).toBe(s.sum());
    expect(column(agg, "v_mean")[0]).toBe(s.mean());
    // The exact value (computed with rational arithmetic) is 0.07988095431810335.
    expect(s.var()).toBeCloseTo(0.07988095431810335, 16);
  });

  it("describe reports the same mean and std as Series", () => {
    const vals = [1e8 + 0.1, 1e8 + 0.2, 1e8 + 0.3, 1e8 + 0.7, 1e8 + 0.9, 1e8 + 0.35, 1e8 + 0.45];
    const d = column(new DataFrame({ v: vals }).describe(), "v");
    const s = new Series(vals);
    expect(d[1]).toBe(s.mean());
    expect(d[2]).toBe(s.std());
  });
});

describe("wave2 dataframe: groupBy options", () => {
  const df = new DataFrame({
    k: ["b", null, "a", "b", Number.NaN, "a", "c"],
    j: [2, 1, 1, 1, 1, 2, 1],
    v: [1, 2, 3, 4, 5, 6, 7],
  });

  it("keeps first-appearance order and missing keys by default", () => {
    expect(column(df.groupBy("k").sum(), "k")).toEqual(["b", null, "a", Number.NaN, "c"]);
  });

  it("sort orders keys ascending with missing keys last", () => {
    const out = df.groupBy("k", { sort: true }).sum();
    expect(column(out, "k")).toEqual(["a", "b", "c", null, Number.NaN]);
    expect(column(out, "v")).toEqual([9, 5, 7, 2, 5]);
  });

  it("dropna removes groups with a missing key and matches pandas with sort", () => {
    const out = df.groupBy("k", { sort: true, dropna: true }).sum();
    expect(column(out, "k")).toEqual(["a", "b", "c"]);
    expect(column(out, "v")).toEqual([9, 5, 7]);
    expect(df.groupBy("k", { dropna: true }).ngroups).toBe(3);
    expect(column(df.groupBy("k", { dropna: true }).size(), "size")).toEqual([2, 2, 1]);
  });

  it("sorts multi-column keys column by column", () => {
    const out = df.groupBy(["k", "j"], { sort: true, dropna: true }).size();
    expect(column(out, "k")).toEqual(["a", "a", "b", "b", "c"]);
    expect(column(out, "j")).toEqual([1, 2, 1, 2, 1]);
  });

  it("sorts numeric keys numerically and strings by code point", () => {
    const nums = new DataFrame({ k: [10, 9, 100], v: [1, 2, 3] });
    expect(column(nums.groupBy("k", { sort: true }).sum(), "k")).toEqual([9, 10, 100]);
    const text = new DataFrame({ k: ["b", "B", "a"], v: [1, 2, 3] });
    expect(column(text.groupBy("k", { sort: true }).sum(), "k")).toEqual(["B", "a", "b"]);
  });

  it("rejects options that are not booleans", () => {
    expect(() => df.groupBy("k", { sort: 1 as unknown as boolean })).toThrow(/sort/);
    expect(() => df.groupBy("k", { dropna: "yes" as unknown as boolean })).toThrow(/dropna/);
  });

  it("documented aggregation semantics hold", () => {
    const g = new DataFrame({ k: ["a", "a", "b", "b"], v: [null, Number.NaN, 1, 3] }).groupBy("k");
    const one = (f: "sum" | "mean" | "median" | "min" | "max" | "std" | "var" | "count") =>
      column(g.agg({ v: f }), "v");
    expect(one("sum")).toEqual([0, 4]);
    expect(one("count")).toEqual([0, 2]);
    expect(one("mean")[0]).toBeNaN();
    expect(one("std")[1]).toBeCloseTo(Math.SQRT2, 15);
    expect(column(g.agg({ v: "first" }), "v")).toEqual([null, 1]);
    expect(column(g.agg({ v: "last" }), "v")).toEqual([Number.NaN, 3]);
  });
});

describe("wave2 dataframe: window functions", () => {
  it("treats infinities as missing observations like pandas", () => {
    const df = new DataFrame({ a: [1, Number.POSITIVE_INFINITY, 3, 4] });
    expect(column(df.rolling(2).sum(), "a")).toEqual([null, null, null, 7]);
    expect(column(df.expanding().sum(), "a")).toEqual([1, 1, 4, 8]);
    expect(column(df.expanding().min(), "a")).toEqual([1, 1, 1, 1]);
    const ewm = column(df.ewm({ alpha: 0.5, adjust: true }).mean(), "a");
    // pandas: [1.0, 1.0, 2.6, 3.4615384615384617]
    expect(ewm[0]).toBe(1);
    expect(ewm[1]).toBe(1);
    expect(ewm[2] as number).toBeCloseTo(2.6, 12);
    expect(ewm[3] as number).toBeCloseTo(3.4615384615384617, 12);
  });

  it("follows the documented EWM weights after a gap at alpha 0.5", () => {
    const df = new DataFrame({ a: [1, Number.NaN, 7] });
    const mean = (alpha: number): unknown[] =>
      column(df.ewm({ alpha, adjust: false, ignoreNa: false }).mean(), "a");
    // Documented weights: (1 - a)^2 for the first value and a for the last one.
    expect(mean(0.5)[2]).toBe(5);
    expect(mean(0.3)[2] as number).toBeCloseTo(3.278481012658228, 12);
    expect(mean(0.51)[2] as number).toBeCloseTo(5.079456072523664, 12);
  });
});

describe("wave2 dataframe: plot and style accessors", () => {
  it("rejects unknown columns instead of returning empty data", () => {
    const df = new DataFrame({ a: [1, 2], b: [3, 4] });
    expect(() => df.plot.line({ x: "a", y: "missing" })).toThrow(/not found/);
    expect(() => df.style.format("missing", String)).toThrow(/not found/);
    expect(() => df.plot.line({ x: "a", y: "b" })).not.toThrow();
  });

  it("shows the index labels as row headers in styled HTML", () => {
    const df = new DataFrame({ a: [1, 2] }, { index: ["first", "<second>"] });
    const html = df.style.toHTML();
    expect(html).toContain("<th>first</th>");
    expect(html).toContain("<th>&lt;second&gt;</th>");
    expect(new DataFrame({ a: [1, 2] }).style.toHTML()).toContain("<th>1</th>");
  });
});

describe("wave2 dataframe: keys and labels", () => {
  it("createKey keeps composite keys apart", () => {
    expect(createKey(["a,s1:b"])).not.toBe(createKey(["a", "b"]));
    expect(createKey([1, "1"])).not.toBe(createKey(["1", 1]));
  });

  it("join and groupBy do not merge distinct composite keys", () => {
    const df = new DataFrame({ x: ["a,s1:b", "a"], y: ["c", "b,c"], v: [1, 2] });
    expect(df.groupBy(["x", "y"]).ngroups).toBe(2);
    expect(df.drop_duplicates(["x", "y"]).shape[0]).toBe(2);
  });

  it("iloc keeps a __proto__ column", () => {
    const data = Object.create(null) as Record<string, unknown[]>;
    Object.defineProperty(data, "__proto__", { value: [7, 8], enumerable: true });
    const df = new DataFrame(data);
    const row = df.iloc(1);
    expect(Object.keys(row)).toEqual(["__proto__"]);
    expect(Object.getOwnPropertyDescriptor(row, "__proto__")?.value).toBe(8);
  });
});
