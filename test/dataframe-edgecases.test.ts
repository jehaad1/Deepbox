import { describe, expect, it } from "vitest";
import { DataFrame, Series } from "../src/dataframe";

describe("DataFrame Edge Cases", () => {
  describe("Empty DataFrames", () => {
    it("should create empty DataFrame", () => {
      const empty = new DataFrame({});
      expect(empty.shape).toEqual([0, 0]);
      expect(empty.columns).toEqual([]);
      expect(empty.index).toEqual([]);
    });

    it("should handle describe() on empty DataFrame", () => {
      const empty = new DataFrame({});
      const desc = empty.describe();
      expect(desc.shape[0]).toBe(8); // 8 metrics rows
      expect(desc.shape[1]).toBe(0); // 0 data columns
    });

    it("should handle operations on empty DataFrame", () => {
      const empty = new DataFrame({});
      expect(empty.head().shape).toEqual([0, 0]);
      expect(empty.tail().shape).toEqual([0, 0]);
      expect(empty.toArray()).toEqual([]);
    });
  });

  describe("Single Row/Column", () => {
    it("should handle single row DataFrame", () => {
      const single = new DataFrame({ a: [1], b: [2] });
      expect(single.shape).toEqual([1, 2]);
      expect(single.iloc(0)).toEqual({ a: 1, b: 2 });
    });

    it("should compute statistics for single row", () => {
      const single = new DataFrame({ a: [5] });
      const desc = single.describe();
      const stats = desc.get("a");
      expect(stats.data[0]).toBe(1); // count
      expect(stats.data[1]).toBe(5); // mean
      expect(stats.data[2]).toBeNaN(); // std (n=1 should be NaN)
      expect(stats.data[3]).toBe(5); // min
      expect(stats.data[7]).toBe(5); // max
    });

    it("should handle single column DataFrame", () => {
      const single = new DataFrame({ x: [1, 2, 3] });
      expect(single.shape).toEqual([3, 1]);
      expect(single.columns).toEqual(["x"]);
    });
  });

  describe("All Null Values", () => {
    it("should handle DataFrame with all null values", () => {
      const allNull = new DataFrame({ a: [null, null, null] });
      expect(allNull.shape).toEqual([3, 1]);
      expect(allNull.index).toEqual([0, 1, 2]);
      expect(allNull.toArray()).toEqual([[null], [null], [null]]);
    });

    it("should skip non-numeric columns in describe()", () => {
      const allNull = new DataFrame({
        a: [null, null, null],
        b: ["x", "y", "z"],
      });
      const desc = allNull.describe();
      // When no numeric columns, returns DataFrame with 8 metric rows but 0 data columns
      expect(desc.shape).toEqual([8, 0]);
    });

    it("should handle dropna() with all nulls", () => {
      const allNull = new DataFrame({ a: [null, null, null] });
      const dropped = allNull.dropna();
      expect(dropped.shape).toEqual([0, 1]);
    });

    it("should handle fillna() with all nulls", () => {
      const allNull = new DataFrame({ a: [null, null, null] });
      const filled = allNull.fillna(0);
      expect(filled.toArray()).toEqual([[0], [0], [0]]);
    });
  });

  describe("Describe With Missing Values", () => {
    it("should ignore null/NaN values in numeric describe", () => {
      const df = new DataFrame({ a: [1, null, 3, NaN], b: [2, 2, 2, 2] });
      const desc = df.describe();
      const aStats = desc.get("a");
      const bStats = desc.get("b");

      expect(aStats.get("count")).toBe(2);
      expect(aStats.get("mean")).toBe(2);
      expect(aStats.get("min")).toBe(1);
      expect(aStats.get("max")).toBe(3);

      expect(bStats.get("count")).toBe(4);
      expect(bStats.get("mean")).toBe(2);
      expect(bStats.get("min")).toBe(2);
      expect(bStats.get("max")).toBe(2);
    });
  });

  describe("Mixed Types", () => {
    it("should handle mixed type columns", () => {
      const mixed = new DataFrame({
        id: [1, 2, 3],
        name: ["Alice", "Bob", "Charlie"],
        score: [85.5, 92.0, 78.5],
        active: [true, false, true],
      });
      expect(mixed.shape).toEqual([3, 4]);
      expect(mixed.columns).toEqual(["id", "name", "score", "active"]);
      expect(mixed.toArray()).toEqual([
        [1, "Alice", 85.5, true],
        [2, "Bob", 92, false],
        [3, "Charlie", 78.5, true],
      ]);
    });

    it("should only describe numeric columns", () => {
      const mixed = new DataFrame({
        name: ["Alice", "Bob"],
        age: [25, 30],
        city: ["NYC", "LA"],
      });
      const desc = mixed.describe();
      expect(desc.columns).toEqual(["age"]); // Only numeric column
    });
  });

  describe("Large Values", () => {
    it("should handle large safe integers", () => {
      const large = new DataFrame({
        big: [Number.MAX_SAFE_INTEGER - 100, 100],
      });
      expect(large.get("big").sum()).toBe(Number.MAX_SAFE_INTEGER);
    });

    it("should handle very small numbers", () => {
      const small = new DataFrame({
        tiny: [Number.MIN_VALUE, Number.MIN_VALUE * 2],
      });
      expect(small.get("tiny").sum()).toBe(Number.MIN_VALUE * 3);
    });
  });

  describe("Special Values", () => {
    it("should handle NaN values", () => {
      const withNaN = new DataFrame({ a: [1, NaN, 3] });
      expect(withNaN.shape).toEqual([3, 1]);
      // NaN is not equal to itself, so we check differently
      const data = withNaN.get("a").data;
      expect(data[0]).toBe(1);
      expect(Number.isNaN(data[1])).toBe(true);
      expect(data[2]).toBe(3);
    });

    it("should handle Infinity values", () => {
      const withInf = new DataFrame({ a: [1, Infinity, -Infinity] });
      expect(withInf.shape).toEqual([3, 1]);
      expect(withInf.get("a").data[1]).toBe(Infinity);
      expect(withInf.get("a").data[2]).toBe(-Infinity);
    });
  });

  describe("Duplicate Handling", () => {
    it("should drop duplicates with keep='first'", () => {
      const df = new DataFrame({ a: [1, 1, 2], b: [null, null, 3] });
      const result = df.drop_duplicates();
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1, null],
        [2, 3],
      ]);
    });

    it("should drop duplicates with keep='last'", () => {
      const df = new DataFrame({ a: [1, 1, 2], b: [3, 4, 5] });
      const result = df.drop_duplicates(["a"], "last");
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1, 4],
        [2, 5],
      ]);
    });

    it("should drop duplicates with keep=false", () => {
      const df = new DataFrame({
        a: [1, 1, 2, 2, 3],
        b: [10, 11, 20, 21, 30],
      });
      const result = df.drop_duplicates(["a"], false);
      expect(result.shape).toEqual([1, 2]);
      expect(result.toArray()).toEqual([[3, 30]]);
    });

    it("should identify duplicates correctly", () => {
      const df = new DataFrame({ a: [1, 2, 3], b: [NaN, NaN, 4] });
      const dups = df.duplicated(["b"]);
      expect(dups.data).toEqual([false, true, false]);
    });
  });

  describe("Error Cases", () => {
    it("should throw on invalid JSON in fromJsonString()", () => {
      expect(() => DataFrame.fromJsonString("invalid json")).toThrow(/Failed to parse JSON/);
    });

    it("should throw on missing columns field in fromJsonString()", () => {
      expect(() => DataFrame.fromJsonString('{"index": [], "data": {}}')).toThrow(
        /missing or invalid "columns" field/
      );
    });

    it("should throw on missing index field in fromJsonString()", () => {
      expect(() => DataFrame.fromJsonString('{"columns": [], "data": {}}')).toThrow(
        /missing or invalid "index" field/
      );
    });

    it("should throw on missing data field in fromJsonString()", () => {
      expect(() => DataFrame.fromJsonString('{"columns": [], "index": []}')).toThrow(
        /missing or invalid "data" field/
      );
    });

    it("should throw on non-object JSON in fromJsonString()", () => {
      expect(() => DataFrame.fromJsonString("[]")).toThrow(/expected object \(not array\)/);
    });

    it("should throw on unexpected data columns in fromJsonString()", () => {
      const json = '{"columns":["a"],"index":[0],"data":{"a":[1],"b":[2]}}';
      expect(() => DataFrame.fromJsonString(json)).toThrow(/Unexpected data column 'b'/);
    });

    it("should throw on non-array data in fromJsonString()", () => {
      const json = '{"columns":["a"],"index":[0],"data":{"a":1}}';
      expect(() => DataFrame.fromJsonString(json)).toThrow(/expected array/);
    });

    it("should throw on invalid index type in fromJsonString()", () => {
      const json = '{"columns":["a"],"index":[{"x":1}],"data":{"a":[1]}}';
      expect(() => DataFrame.fromJsonString(json)).toThrow(/invalid "index" field/);
    });

    it("should throw on data length mismatch in fromJsonString()", () => {
      const json = '{"columns":["a","b"],"index":[0,1],"data":{"a":[1,2],"b":[3]}}';
      expect(() => DataFrame.fromJsonString(json)).toThrow(/length/i);
    });

    it("should throw on index length mismatch in fromJsonString()", () => {
      const json = '{"columns":["a"],"index":[0,1,2],"data":{"a":[1,2]}}';
      expect(() => DataFrame.fromJsonString(json)).toThrow(/Index length/i);
    });

    it("should throw on out of bounds iloc()", () => {
      const df = new DataFrame({ a: [1, 2, 3] });
      expect(() => df.iloc(5)).toThrow(/out of bounds/);
      expect(() => df.iloc(-1)).toThrow(/out of bounds/);
    });

    it("should throw on non-existent column in get()", () => {
      const df = new DataFrame({ a: [1, 2, 3] });
      expect(() => df.get("nonexistent")).toThrow(/not found/);
    });

    it("should throw on non-existent column in select()", () => {
      const df = new DataFrame({ a: [1, 2, 3] });
      expect(() => df.select(["nonexistent"])).toThrow(/not found/);
    });

    it("should throw on sample size larger than DataFrame", () => {
      const df = new DataFrame({ a: [1, 2, 3] });
      expect(() => df.sample(5)).toThrow(/must be between/);
    });

    it("should throw on negative sample size", () => {
      const df = new DataFrame({ a: [1, 2, 3] });
      expect(() => df.sample(-1)).toThrow(/must be between/);
    });

    it("should throw on quantile out of range", () => {
      const df = new DataFrame({ a: [1, 2, 3] });
      expect(() => df.quantile(1.5)).toThrow(/between 0 and 1/);
      expect(() => df.quantile(-0.5)).toThrow(/between 0 and 1/);
    });
  });

  describe("Determinism", () => {
    it("should produce same results with same random_state", () => {
      const df = new DataFrame({ a: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] });
      const sample1 = df.sample(5, 42);
      const sample2 = df.sample(5, 42);
      expect(sample1.toArray()).toEqual(sample2.toArray());
    });

    it("should produce deterministic samples for different random_state values", () => {
      const df = new DataFrame({ a: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] });
      const sample42 = df.sample(5, 42);
      const sample43 = df.sample(5, 43);
      const original = new Set(df.get("a").data);
      const assertSample = (sample: DataFrame) => {
        const data = sample.get("a").data;
        expect(data.length).toBe(5);
        expect(new Set(data).size).toBe(5);
        data.forEach((val) => {
          expect(original.has(val)).toBe(true);
        });
      };
      assertSample(sample42);
      assertSample(sample43);
      expect(sample42.toArray()).not.toEqual(sample43.toArray());
    });
  });

  describe("Sample Boundaries", () => {
    it("should return empty DataFrame for n=0", () => {
      const df = new DataFrame({ a: [1, 2, 3] });
      const result = df.sample(0, 1);
      expect(result.shape).toEqual([0, 1]);
      expect(result.columns).toEqual(["a"]);
      expect(result.index).toEqual([]);
      expect(result.toArray()).toEqual([]);
    });

    it("should return all rows for n=rowCount", () => {
      const df = new DataFrame({ a: [3, 1, 2] });
      const result = df.sample(3, 1);
      const sortedResult = [...result.get("a").data].map(Number).sort((a, b) => a - b);
      expect(sortedResult).toEqual([1, 2, 3]);
    });
  });

  describe("Exact Value Assertions", () => {
    it("should compute exact statistics for known data", () => {
      const df = new DataFrame({ a: [1, 2, 3, 4, 5] });
      const desc = df.describe();
      const stats = desc.get("a");

      expect(stats.data[0]).toBe(5); // count
      expect(stats.data[1]).toBe(3); // mean
      expect(stats.data[2]).toBeCloseTo(1.5811, 4); // sample std
      expect(stats.data[3]).toBe(1); // min
      expect(stats.data[4]).toBe(2); // 25%
      expect(stats.data[5]).toBe(3); // 50% (median)
      expect(stats.data[6]).toBe(4); // 75%
      expect(stats.data[7]).toBe(5); // max
    });

    it("should compute exact correlation", () => {
      const df = new DataFrame({
        a: [1, 2, 3, 4, 5],
        b: [2, 4, 6, 8, 10],
      });
      const corr = df.corr();
      // Perfect positive correlation
      expect(corr.loc("a")["b"]).toBeCloseTo(1.0, 10);
      expect(corr.loc("b")["a"]).toBeCloseTo(1.0, 10);
    });

    it("should compute exact covariance", () => {
      const df = new DataFrame({
        a: [1, 2, 3],
        b: [4, 5, 6],
      });
      const cov = df.cov();
      // Covariance should be 1.0 for this data
      expect(cov.loc("a")["a"]).toBeCloseTo(1.0, 10);
      expect(cov.loc("a")["b"]).toBeCloseTo(1.0, 10);
    });
  });
});

describe("Series Edge Cases", () => {
  it("should handle empty Series", () => {
    const empty = new Series([]);
    expect(empty.length).toBe(0);
    expect(empty.data).toEqual([]);
  });

  it("should throw for sum() of empty Series", () => {
    const empty = new Series([]);
    expect(() => empty.sum()).toThrow(/Cannot get sum of empty Series/);
  });

  it("should throw for mean() of empty Series", () => {
    const empty = new Series([]);
    expect(() => empty.mean()).toThrow(/Cannot get mean of empty Series/);
  });

  it("should throw for median() of empty Series", () => {
    const empty = new Series([]);
    expect(() => empty.median()).toThrow(/Cannot get median of empty Series/);
  });

  it("should return NaN for std() of single element", () => {
    const single = new Series([5]);
    expect(single.std()).toBeNaN();
  });

  it("should return NaN for var() of single element", () => {
    const single = new Series([5]);
    expect(single.var()).toBeNaN();
  });

  it("should compute exact median for even length", () => {
    const s = new Series([1, 2, 3, 4]);
    expect(s.median()).toBe(2.5); // Average of 2 and 3
  });

  it("should compute exact median for odd length", () => {
    const s = new Series([1, 2, 3, 4, 5]);
    expect(s.median()).toBe(3);
  });

  it("should throw on undefined index label", () => {
    // @ts-expect-error - undefined index label should be rejected at runtime
    expect(() => new Series([1, 2], { index: [1, undefined] })).toThrow(/cannot be undefined/);
  });

  it("should throw on duplicate index labels", () => {
    expect(() => new Series([1, 2], { index: ["a", "a"] })).toThrow(/Duplicate index label/);
  });

  it("should convert to Tensor correctly", () => {
    const s = new Series([1, 2, 3, 4]);
    const t = s.toTensor();
    expect(t.shape).toEqual([4]);
    expect([...t.data]).toEqual([1, 2, 3, 4]);
  });

  it("should throw on toTensor() with non-numeric data", () => {
    const s = new Series(["a", "b", "c"]);
    expect(() => s.toTensor()).toThrow(/only works on numeric data/);
  });
});

describe("DataFrame rank() dense method", () => {
  it("should compute dense ranks correctly", () => {
    const df = new DataFrame({ a: [3, 1, 2, 1] });
    const result = df.rank("dense");
    // Dense ranking: 1, 1, 2, 1 -> ranks should be 3, 1, 2, 1
    // Values sorted: 1, 1, 2, 3 -> dense ranks: 1, 1, 2, 3
    expect(result.get("a").data).toEqual([3, 1, 2, 1]);
  });

  it("should compute dense ranks with gaps correctly", () => {
    const df = new DataFrame({ a: [10, 30, 20, 30, 10] });
    const result = df.rank("dense");
    // Values: 10, 30, 20, 30, 10
    // Sorted unique: 10, 20, 30 -> dense ranks: 1, 2, 3
    // Result: 1, 3, 2, 3, 1
    expect(result.get("a").data).toEqual([1, 3, 2, 3, 1]);
  });

  it("should compute dense ranks descending", () => {
    const df = new DataFrame({ a: [1, 2, 3, 2] });
    const result = df.rank("dense", false);
    // Descending: 3, 2, 2, 1 -> dense ranks: 1, 2, 2, 3
    // Original positions: 1->3, 2->2, 3->1, 2->2
    expect(result.get("a").data).toEqual([3, 2, 1, 2]);
  });
});
