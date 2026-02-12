import { describe, expect, it } from "vitest";
import { DataFrame } from "../src/dataframe";
import { Series } from "../src/dataframe/Series";

// Type guards
const isNumber = (val: unknown): val is number => typeof val === "number";

describe("DataFrame - Extended Tests", () => {
  describe("drop_duplicates()", () => {
    it("should remove duplicate rows keeping first", () => {
      const df = new DataFrame({ a: [1, 1, 2], b: [3, 3, 4] });
      const result = df.drop_duplicates();
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1, 3],
        [2, 4],
      ]);
    });

    it("should remove duplicates keeping last", () => {
      const df = new DataFrame({ a: [1, 1, 2], b: [3, 4, 5] });
      const result = df.drop_duplicates(["a"], "last");
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1, 4],
        [2, 5],
      ]);
    });

    it("should remove all duplicates when keep=false", () => {
      const df = new DataFrame({ a: [1, 1, 2], b: [3, 3, 4] });
      const result = df.drop_duplicates(undefined, false);
      expect(result.shape).toEqual([1, 2]);
      expect(result.toArray()).toEqual([[2, 4]]);
    });

    it("should consider subset of columns", () => {
      const df = new DataFrame({ a: [1, 1, 2], b: [3, 4, 5] });
      const result = df.drop_duplicates(["a"]);
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1, 3],
        [2, 5], // First occurrence of a=1 kept (1,3), then a=2 (2,5)
      ]);
    });

    it("should throw for non-existent subset column", () => {
      const df = new DataFrame({ a: [1, 2] });
      expect(() => df.drop_duplicates(["b"])).toThrow(/Column 'b' not found/);
    });
  });

  describe("duplicated()", () => {
    it("should mark duplicates", () => {
      const df = new DataFrame({ a: [1, 1, 2], b: [3, 3, 4] });
      const result = df.duplicated();
      expect(result.data).toEqual([false, true, false]);
    });

    it("should mark duplicates with keep=last", () => {
      const df = new DataFrame({ a: [1, 1, 2] });
      const result = df.duplicated(undefined, "last");
      expect(result.data).toEqual([true, false, false]);
    });

    it("should mark all duplicates when keep=false", () => {
      const df = new DataFrame({ a: [1, 1, 2] });
      const result = df.duplicated(undefined, false);
      expect(result.data).toEqual([true, true, false]);
    });
  });

  describe("rename()", () => {
    it("should rename columns with object mapper", () => {
      const df = new DataFrame({ a: [1, 2], b: [3, 4] });
      const result = df.rename({ a: "x", b: "y" }, 1);
      expect(result.columns).toEqual(["x", "y"]);
      expect(result.toArray()).toEqual([
        [1, 3],
        [2, 4],
      ]);
    });

    it("should rename columns with function", () => {
      const df = new DataFrame({ a: [1, 2], b: [3, 4] });
      const result = df.rename((name) => name.toUpperCase(), 1);
      expect(result.columns).toEqual(["A", "B"]);
      expect(result.toArray()).toEqual([
        [1, 3],
        [2, 4],
      ]);
    });

    it("should rename index with object mapper", () => {
      const df = new DataFrame({ a: [1, 2] }, { index: ["x", "y"] });
      const result = df.rename({ x: "row1", y: "row2" }, 0);
      expect(result.index).toEqual(["row1", "row2"]);
      expect(result.toArray()).toEqual([[1], [2]]);
    });

    it("should keep unmapped columns unchanged", () => {
      const df = new DataFrame({ a: [1, 2], b: [3, 4] });
      const result = df.rename({ a: "x" }, 1);
      expect(result.columns).toEqual(["x", "b"]);
      expect(result.toArray()).toEqual([
        [1, 3],
        [2, 4],
      ]);
    });

    it("should throw when rename creates duplicate columns", () => {
      const df = new DataFrame({ a: [1, 2], b: [3, 4] });
      expect(() => df.rename({ a: "x", b: "x" }, 1)).toThrow(/Duplicate column name/i);
    });
  });

  describe("reset_index()", () => {
    it("should reset index to default", () => {
      const df = new DataFrame({ a: [1, 2] }, { index: ["x", "y"] });
      const result = df.reset_index();
      expect(result.index).toEqual([0, 1]);
      expect(result.columns).toEqual(["index", "a"]);
    });

    it("should drop old index when drop=true", () => {
      const df = new DataFrame({ a: [1, 2] }, { index: ["x", "y"] });
      const result = df.reset_index(true);
      expect(result.columns).toEqual(["a"]);
    });

    it("should avoid index column name collisions", () => {
      const df = new DataFrame({ index: [10, 11], a: [1, 2] }, { index: ["x", "y"] });
      const result = df.reset_index();
      expect(result.columns).toEqual(["index_1", "index", "a"]);
      expect(result.get("index_1").data).toEqual(["x", "y"]);
    });
  });

  describe("set_index()", () => {
    it("should set column as index", () => {
      const df = new DataFrame({ id: ["a", "b", "c"], value: [1, 2, 3] });
      const result = df.set_index("id");
      expect(result.index).toEqual(["a", "b", "c"]);
      expect(result.columns).toEqual(["value"]);
    });

    it("should keep column when drop=false", () => {
      const df = new DataFrame({ id: ["a", "b"], value: [1, 2] });
      const result = df.set_index("id", false);
      expect(result.columns).toEqual(["id", "value"]);
    });

    it("should throw for non-existent column", () => {
      const df = new DataFrame({ a: [1, 2] });
      expect(() => df.set_index("b")).toThrow(/Column 'b' not found/);
    });
  });

  describe("isnull() and notnull()", () => {
    it("should identify null values", () => {
      const df = new DataFrame({ a: [1, null, 3], b: [4, 5, undefined] });
      const result = df.isnull();
      expect(result.toArray()).toEqual([
        [false, false],
        [true, false],
        [false, true],
      ]);
    });

    it("should identify non-null values", () => {
      const df = new DataFrame({ a: [1, null, 3] });
      const result = df.notnull();
      expect(result.get("a").data).toEqual([true, false, true]);
    });
  });

  describe("replace()", () => {
    it("should replace single value", () => {
      const df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] });
      const result = df.replace(2, 99);
      expect(result.get("a").data).toEqual([1, 99, 3]);
    });

    it("should replace multiple values", () => {
      const df = new DataFrame({ a: [1, 2, 3] });
      const result = df.replace([1, 2], 0);
      expect(result.get("a").data).toEqual([0, 0, 3]);
    });
  });

  describe("clip()", () => {
    it("should clip values to range", () => {
      const df = new DataFrame({ a: [1, 5, 10], b: [2, 8, 15] });
      const result = df.clip(3, 9);
      expect(result.toArray()).toEqual([
        [3, 3],
        [5, 8],
        [9, 9],
      ]);
    });

    it("should clip with only lower bound", () => {
      const df = new DataFrame({ a: [1, 5, 10] });
      const result = df.clip(5, undefined);
      expect(result.get("a").data).toEqual([5, 5, 10]);
    });

    it("should clip with only upper bound", () => {
      const df = new DataFrame({ a: [1, 5, 10] });
      const result = df.clip(undefined, 5);
      expect(result.get("a").data).toEqual([1, 5, 5]);
    });
  });

  describe("sample()", () => {
    it("should sample n distinct rows without replacement", () => {
      const df = new DataFrame({ a: [1, 2, 3, 4, 5] });
      const result = df.sample(3, 7);
      expect(result.shape).toEqual([3, 1]);

      // Verify sampled rows exist in original
      const originalData = df.get("a").data;
      const sampledData = result.get("a").data;

      if (!originalData.every(isNumber)) throw new Error("Original data must be numeric");
      if (!sampledData.every(isNumber)) throw new Error("Sampled data must be numeric");

      const originalValues = new Set(originalData);

      expect(sampledData.length).toBe(3);
      expect(new Set(sampledData).size).toBe(3);
      expect(result.index.every((idx) => df.index.includes(idx))).toBe(true);
      sampledData.forEach((val) => {
        expect(originalValues.has(val)).toBe(true);
      });
    });

    it("should throw for invalid sample size", () => {
      const df = new DataFrame({ a: [1, 2, 3] });
      expect(() => df.sample(5)).toThrow(/Sample size 5 must be between 0 and 3/);
    });

    it("should be reproducible with random_state", () => {
      const df = new DataFrame({ a: [1, 2, 3, 4, 5] });
      const result1 = df.sample(3, 42);
      const result2 = df.sample(3, 42);
      expect(result1.toArray()).toEqual(result2.toArray());
    });
  });

  describe("quantile()", () => {
    it("should compute median (0.5 quantile)", () => {
      const df = new DataFrame({ a: [1, 2, 3, 4, 5] });
      const result = df.quantile(0.5);
      expect(result.data[0]).toBe(3);
    });

    it("should compute 25th percentile", () => {
      const df = new DataFrame({ a: [1, 2, 3, 4, 5] });
      const result = df.quantile(0.25);
      expect(result.data[0]).toBe(2);
    });

    it("should throw for invalid quantile", () => {
      const df = new DataFrame({ a: [1, 2, 3] });
      expect(() => df.quantile(1.5)).toThrow(/between 0 and 1/);
    });
  });

  describe("rank()", () => {
    it("should rank values with average method", () => {
      const df = new DataFrame({ a: [3, 1, 2, 1] });
      const result = df.rank();
      expect(result.get("a").data).toEqual([4, 1.5, 3, 1.5]);
    });

    it("should rank with min method", () => {
      const df = new DataFrame({ a: [3, 1, 2, 1] });
      const result = df.rank("min");
      expect(result.get("a").data).toEqual([4, 1, 3, 1]);
    });

    it("should rank with max method", () => {
      const df = new DataFrame({ a: [3, 1, 2, 1] });
      const result = df.rank("max");
      expect(result.get("a").data).toEqual([4, 2, 3, 2]);
    });

    it("should rank descending", () => {
      const df = new DataFrame({ a: [1, 2, 3] });
      const result = df.rank("average", false);
      expect(result.get("a").data).toEqual([3, 2, 1]);
    });
  });

  describe("diff()", () => {
    it("should compute differences", () => {
      const df = new DataFrame({ a: [1, 3, 6, 10] });
      const result = df.diff();
      expect(result.get("a").data).toEqual([null, 2, 3, 4]);
    });

    it("should compute differences with periods=2", () => {
      const df = new DataFrame({ a: [1, 3, 6, 10] });
      const result = df.diff(2);
      expect(result.get("a").data).toEqual([null, null, 5, 7]);
    });
  });

  describe("pct_change()", () => {
    it("should compute percentage changes", () => {
      const df = new DataFrame({ a: [100, 110, 121] });
      const result = df.pct_change();
      expect(result.get("a").data[0]).toBe(null);
      expect(result.get("a").data[1]).toBeCloseTo(0.1, 5);
      expect(result.get("a").data[2]).toBeCloseTo(0.1, 5);
    });
  });

  describe("cumsum(), cumprod(), cummax(), cummin()", () => {
    it("should compute cumulative sum", () => {
      const df = new DataFrame({ a: [1, 2, 3] });
      const result = df.cumsum();
      expect(result.get("a").data).toEqual([1, 3, 6]);
    });

    it("should compute cumulative product", () => {
      const df = new DataFrame({ a: [2, 3, 4] });
      const result = df.cumprod();
      expect(result.get("a").data).toEqual([2, 6, 24]);
    });

    it("should compute cumulative maximum", () => {
      const df = new DataFrame({ a: [3, 1, 5, 2] });
      const result = df.cummax();
      expect(result.get("a").data).toEqual([3, 3, 5, 5]);
    });

    it("should compute cumulative minimum", () => {
      const df = new DataFrame({ a: [3, 1, 5, 2] });
      const result = df.cummin();
      expect(result.get("a").data).toEqual([3, 1, 1, 1]);
    });
  });

  describe("shift()", () => {
    it("should shift data down", () => {
      const df = new DataFrame({ a: [1, 2, 3, 4] });
      const result = df.shift(1);
      expect(result.get("a").data).toEqual([null, 1, 2, 3]);
    });

    it("should shift data up", () => {
      const df = new DataFrame({ a: [1, 2, 3, 4] });
      const result = df.shift(-1);
      expect(result.get("a").data).toEqual([2, 3, 4, null]);
    });

    it("should use custom fill value", () => {
      const df = new DataFrame({ a: [1, 2, 3] });
      const result = df.shift(1, 0);
      expect(result.get("a").data).toEqual([0, 1, 2]);
    });
  });

  describe("concat()", () => {
    it("should concatenate on axis=0", () => {
      const df1 = new DataFrame({ a: [1, 2], b: [3, 4] }, { index: ["a", "b"] });
      const df2 = new DataFrame({ a: [5, 6], b: [7, 8] }, { index: ["b", "c"] });
      const result = df1.concat(df2, 0);
      expect(result.shape).toEqual([4, 2]);
      expect(result.get("a").data).toEqual([1, 2, 5, 6]);
      expect(result.get("b").data).toEqual([3, 4, 7, 8]);
      expect(result.index).toEqual([0, 1, 2, 3]);
    });

    it("should concatenate on axis=1", () => {
      const df1 = new DataFrame({ x: [1, 2] }, { index: ["b", "a"] });
      const df2 = new DataFrame({ x: [3, 4] }, { index: ["a", "c"] });
      const result = df1.concat(df2, 1);
      expect(result.shape).toEqual([3, 2]);
      expect(result.columns).toEqual(["x_left", "x_right"]);
      expect(result.index).toEqual(["b", "a", "c"]);
      expect(result.toArray()).toEqual([
        [1, null],
        [2, 3],
        [null, 4],
      ]);
    });

    it("should throw for axis=0 with mismatched columns", () => {
      const df1 = new DataFrame({ a: [1, 2], b: [3, 4] });
      const df2 = new DataFrame({ a: [5, 6] });
      expect(() => df1.concat(df2, 0)).toThrow(/missing column/);
    });

    it("should align indices for axis=1 with mismatched rows", () => {
      const df1 = new DataFrame({ a: [1, 2] });
      const df2 = new DataFrame({ b: [3, 4, 5] });
      const result = df1.concat(df2, 1);
      expect(result.shape).toEqual([3, 2]);
      expect(result.iloc(2)["a"]).toBeNull();
      expect(result.iloc(2)["b"]).toBe(5);
    });
  });

  describe("join()", () => {
    it("should perform inner join", () => {
      const df1 = new DataFrame({
        id: [1, 2, 3],
        name: ["Alice", "Bob", "Charlie"],
      });
      const df2 = new DataFrame({ id: [1, 2, 4], value: [10, 20, 30] });
      const result = df1.join(df2, "id", "inner");
      expect(result.shape[0]).toBe(2);
      expect(result.columns).toEqual(["id", "name", "value"]);
      expect(result.toArray()).toEqual([
        [1, "Alice", 10],
        [2, "Bob", 20],
      ]);
    });

    it("should perform left join", () => {
      const df1 = new DataFrame({
        id: [1, 2, 3],
        name: ["Alice", "Bob", "Charlie"],
      });
      const df2 = new DataFrame({ id: [1, 2], value: [10, 20] });
      const result = df1.join(df2, "id", "left");
      expect(result.shape[0]).toBe(3);
      expect(result.columns).toEqual(["id", "name", "value"]);
      expect(result.toArray()).toEqual([
        [1, "Alice", 10],
        [2, "Bob", 20],
        [3, "Charlie", null],
      ]);
    });

    it("should throw for missing join column", () => {
      const df1 = new DataFrame({ id: [1, 2] });
      const df2 = new DataFrame({ value: [10, 20] });
      expect(() => df1.join(df2, "id", "inner")).toThrow(/not found in right DataFrame/);
    });
  });

  describe("merge()", () => {
    it("should merge with same column names", () => {
      const df1 = new DataFrame({ id: [1, 2], name: ["Alice", "Bob"] });
      const df2 = new DataFrame({ id: [1, 2], value: [10, 20] });
      const result = df1.merge(df2, { on: "id" });
      expect(result.columns).toContain("name");
      expect(result.columns).toContain("value");
      expect(result.toArray()).toEqual([
        [1, "Alice", 10],
        [2, "Bob", 20],
      ]);
    });

    it("should merge with different column names", () => {
      const df1 = new DataFrame({ emp_id: [1, 2], name: ["Alice", "Bob"] });
      const df2 = new DataFrame({
        employee_id: [1, 2],
        salary: [50000, 60000],
      });
      const result = df1.merge(df2, {
        left_on: "emp_id",
        right_on: "employee_id",
      });
      expect(result.shape[0]).toBe(2);
      expect(result.toArray()).toEqual([
        [1, "Alice", 1, 50000],
        [2, "Bob", 2, 60000],
      ]);
    });

    it("should throw for conflicting options", () => {
      const df1 = new DataFrame({ id: [1, 2] });
      const df2 = new DataFrame({ id: [1, 2] });
      expect(() => df1.merge(df2, { on: "id", left_on: "id" })).toThrow();
    });
  });

  describe("fillna()", () => {
    it("should fill null values", () => {
      const df = new DataFrame({ a: [1, null, 3], b: [4, 5, undefined] });
      const result = df.fillna(0);
      expect(result.toArray()).toEqual([
        [1, 4],
        [0, 5],
        [3, 0],
      ]);
    });
  });

  describe("dropna()", () => {
    it("should drop rows with null values", () => {
      const df = new DataFrame({ a: [1, null, 3], b: [4, 5, 6] });
      const result = df.dropna();
      expect(result.shape).toEqual([2, 2]);
      expect(result.index).toEqual([0, 2]);
      expect(result.toArray()).toEqual([
        [1, 4],
        [3, 6],
      ]);
    });
  });

  describe("describe()", () => {
    it("should generate descriptive statistics", () => {
      const df = new DataFrame({
        age: [25, 30, 35, 40],
        salary: [50000, 60000, 70000, 80000],
      });
      const stats = df.describe();
      expect(stats.columns).toContain("age");
      expect(stats.columns).toContain("salary");
      expect(stats.get("age").get("mean")).toBe(32.5);
      expect(stats.get("salary").get("min")).toBe(50000);
      expect(stats.get("salary").get("max")).toBe(80000);
    });
  });

  describe("corr() and cov()", () => {
    it("should compute correlation matrix", () => {
      const df = new DataFrame({ a: [1, 2, 3], b: [2, 4, 6] });
      const corr = df.corr();
      expect(corr.shape).toEqual([2, 2]);
      expect(corr.index).toEqual(["a", "b"]);
      expect(corr.columns).toEqual(["a", "b"]);

      // Perfect correlation
      expect(corr.loc("a")?.["b"]).toBeCloseTo(1.0);
      expect(corr.loc("b")?.["a"]).toBeCloseTo(1.0);
    });

    it("should compute covariance matrix", () => {
      const df = new DataFrame({ a: [1, 2, 3], b: [2, 4, 6] });
      const cov = df.cov();
      expect(cov.shape).toEqual([2, 2]);

      // Covariance of x, 2x should be var(x)*2
      // var(1,2,3) = 1.0 (population) or 1.0 (sample)?
      // Pandas default is sample (n-1). sum((x-mean)^2)/(n-1)
      // mean=2. (1-2)^2 + (2-2)^2 + (3-2)^2 = 1+0+1 = 2. 2/2 = 1.
      // cov(a,a) = 1.
      // cov(a,b) = cov(a, 2a) = 2*var(a) = 2.
      expect(cov.loc("a")?.["a"]).toBeCloseTo(1.0);
      expect(cov.loc("a")?.["b"]).toBeCloseTo(2.0);
    });
  });

  describe("apply()", () => {
    it("should apply function to columns", () => {
      const df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] });
      const result = df.apply(
        (series) =>
          series.map((x) => {
            if (typeof x !== "number") throw new Error("Expected number");
            return x * 2;
          }),
        0
      );
      expect(result.get("a").data).toEqual([2, 4, 6]);
    });

    it("should throw when column apply returns wrong length", () => {
      const df = new DataFrame({ a: [1, 2], b: [3, 4] });
      const shortSeries = new Series([1], { index: [0] });
      expect(() => df.apply(() => shortSeries, 0)).toThrow(/Index length/);
    });

    it("should apply function to rows", () => {
      const df = new DataFrame({ a: [1, 2], b: [3, 4] });
      const result = df.apply(
        (row) =>
          row.map((x) => {
            if (typeof x !== "number") throw new Error("Expected number");
            return x * 2;
          }),
        1
      );
      expect(result.toArray()).toEqual([
        [2, 6],
        [4, 8],
      ]);
    });

    it("should handle mixed return types in apply", () => {
      const df = new DataFrame({ a: [1, 2], b: [3, 4] });
      // Returns numbers for 'a' (doubled) and strings for 'b'
      const result = df.apply((row) => {
        const aVal = row.get("a");
        const bVal = row.get("b");
        if (!isNumber(aVal) || !isNumber(bVal)) {
          throw new Error("Expected numbers");
        }
        return new Series([aVal * 2, `val_${bVal}`], { index: ["a", "b"] });
      }, 1);

      expect(result.get("a").data).toEqual([2, 4]);
      expect(result.get("b").data).toEqual(["val_3", "val_4"]);
    });

    it("should align apply results with missing labels", () => {
      const df = new DataFrame({ a: [1, 2], b: [3, 4] });
      const result = df.apply((row) => {
        const aVal = row.get("a");
        const bVal = row.get("b");
        if (!isNumber(aVal) || !isNumber(bVal)) {
          throw new Error("Expected numbers");
        }
        if (aVal === 1) {
          return new Series([aVal], { index: ["a"] });
        }
        return new Series([aVal, bVal], { index: ["a", "b"] });
      }, 1);

      expect(result.columns).toEqual(["a", "b"]);
      expect(result.get("a").data).toEqual([1, 2]);
      expect(result.get("b").data).toEqual([null, 4]);
    });
  });

  describe("toArray() and toTensor()", () => {
    it("should convert to 2D array", () => {
      const df = new DataFrame({ a: [1, 2], b: [3, 4] });
      const arr = df.toArray();
      expect(arr).toEqual([
        [1, 3],
        [2, 4],
      ]);
    });

    it("should convert to tensor", () => {
      const df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] });
      const tensor = df.toTensor();
      expect(tensor.shape).toEqual([3, 2]);
      const data = tensor.data;
      if (Array.isArray(data) || data instanceof BigInt64Array) throw new Error("unexpected type");
      expect(Array.from(data)).toEqual([1, 4, 2, 5, 3, 6]);
    });
  });

  describe("CSV Operations", () => {
    it("should parse CSV string", () => {
      const csv = "name,age\nAlice,25\nBob,30";
      const df = DataFrame.fromCsvString(csv);
      expect(df.shape).toEqual([2, 2]);
      expect(df.columns).toEqual(["name", "age"]);
      expect(df.toArray()).toEqual([
        ["Alice", 25],
        ["Bob", 30],
      ]);
    });

    it("should convert to CSV string", () => {
      const df = new DataFrame({ name: ["Alice", "Bob"], age: [25, 30] });
      const csv = df.toCsvString();
      expect(csv).toBe("name,age\nAlice,25\nBob,30");
    });
  });

  describe("JSON Operations", () => {
    it("should convert to JSON string", () => {
      const df = new DataFrame({ a: [1, 2], b: [3, 4] });
      const json = df.toJsonString();
      const parsed = JSON.parse(json) as {
        columns: string[];
        index: number[];
        data: Record<string, unknown[]>;
      };
      expect(parsed.columns).toEqual(["a", "b"]);
      expect(parsed.index).toEqual([0, 1]);
      expect(parsed.data).toEqual({ a: [1, 2], b: [3, 4] });
    });

    it("should create from JSON string", () => {
      const json = '{"columns":["a","b"],"index":[0,1],"data":{"a":[1,2],"b":[3,4]}}';
      const df = DataFrame.fromJsonString(json);
      expect(df.shape).toEqual([2, 2]);
      expect(df.index).toEqual([0, 1]);
      expect(df.toArray()).toEqual([
        [1, 3],
        [2, 4],
      ]);
    });

    it("should round-trip JSON with non-default index and mixed types", () => {
      const original = new DataFrame(
        { name: ["Alice", "Bob"], active: [true, null] },
        { index: ["row1", "row2"] }
      );
      const json = original.toJsonString();
      const parsed = DataFrame.fromJsonString(json);

      expect(parsed.index).toEqual(["row1", "row2"]);
      expect(parsed.columns).toEqual(["name", "active"]);
      expect(parsed.toArray()).toEqual([
        ["Alice", true],
        ["Bob", null],
      ]);
    });
  });
});
