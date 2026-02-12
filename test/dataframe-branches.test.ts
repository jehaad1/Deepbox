import { promises as fs } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { DataFrame } from "../src/dataframe";
import { tensor } from "../src/ndarray";

describe("DataFrame - Branch Coverage", () => {
  it("validates constructor inputs", () => {
    expect(() => new DataFrame({ a: [1] }, { columns: ["b"] })).toThrow(/not found/);
    expect(() => new DataFrame({ a: [1, 2] }, { index: ["x"] })).toThrow(/Index length/);
    expect(() => new DataFrame({ a: [1, 2] }, { index: ["x", "x"] })).toThrow(/Duplicate index/);
    expect(() => new DataFrame({ a: [1], b: [1, 2] })).toThrow(/length/);
    expect(() => new DataFrame({ a: [1], b: [2] }, { columns: ["a", "a"] })).toThrow(
      /Duplicate column name/
    );
  });

  it("validates loc/iloc and column access", () => {
    const df = new DataFrame({ a: [1, 2] }, { index: ["r1", "r2"] });
    expect(() => df.get("missing")).toThrow(/not found/);
    expect(() => df.loc("nope")).toThrow(/not found/);
    expect(() => df.iloc(5)).toThrow(/out of bounds/);
  });

  it("validates select/drop/sort errors", () => {
    const df = new DataFrame({ a: [2, 1], b: ["x", "y"] });
    expect(() => df.select(["c"])).toThrow(/not found/);
    expect(() => df.drop(["c"])).toThrow(/not found/);

    expect(() => df.sort("c")).toThrow(/not found/);
    expect(df.sort("a", false).toArray()[0][0]).toBe(2);
  });

  it("validates merge configuration errors", () => {
    const left = new DataFrame({ a: [1], b: [2] });
    const right = new DataFrame({ a: [1], c: [3] });

    expect(() => left.merge(right, { on: "a", left_on: "a", right_on: "a" })).toThrow(
      /Cannot specify/
    );
    expect(() => left.merge(right, { left_on: "a" })).toThrow(/Must specify/);
    expect(() => left.merge(right, { left_on: "missing", right_on: "a" })).toThrow(/not found/);
    expect(() => left.merge(right, { left_on: "a", right_on: "missing" })).toThrow(/not found/);
  });

  it("validates concat axis errors", () => {
    const a = new DataFrame({ x: [1, 2] });
    const b = new DataFrame({ y: [1, 2, 3] });
    const result = a.concat(b, 1);
    expect(result.shape).toEqual([3, 2]);
    expect(result.iloc(2)["x"]).toBeNull();
    expect(result.iloc(2)["y"]).toBe(3);

    const c = new DataFrame({ x: [1], y: [2] });
    expect(() => a.concat(c, 0)).toThrow(/extra column/);
  });

  it("covers toCsvString options and fromCsvString errors", () => {
    const df = new DataFrame({ a: [1, 2], b: ["x", "y"] }, { index: ["i1", "i2"] });
    const csv = df.toCsvString({ includeIndex: true, header: true, delimiter: ";" });
    expect(csv).toBe("index;a;b\ni1;1;x\ni2;2;y");

    const noHeader = df.toCsvString({ header: false });
    expect(noHeader).toBe("1,x\n2,y");

    expect(() => DataFrame.fromCsvString("", { hasHeader: false })).toThrow(/no data rows/);
  });

  it("covers JSON and CSV file IO", async () => {
    const tempDir = await fs.mkdtemp(join(tmpdir(), "dataframe-test-"));
    try {
      const df = new DataFrame({ a: [1, 2], b: [3, 4] });

      const jsonPath = join(tempDir, `df-${Date.now()}.json`);
      await df.toJson(jsonPath);
      const fromJson = await DataFrame.readJson(jsonPath);
      expect(fromJson.toArray()).toEqual(df.toArray());

      const csvPath = join(tempDir, `df-${Date.now()}.csv`);
      await df.toCsv(csvPath);
      const fromCsv = await DataFrame.readCsv(csvPath);
      expect(fromCsv.toArray()).toEqual(df.toArray());
    } finally {
      await fs.rm(tempDir, { recursive: true, force: true });
    }
  });

  it("covers pivot/melt error branches", () => {
    const df = new DataFrame({ a: [1, 2], b: [3, 4] });
    expect(() => df.pivot("missing", "b", "a")).toThrow(/not found/);
    expect(() => df.pivot("a", "missing", "b")).toThrow(/not found/);
    expect(() => df.pivot("a", "b", "missing")).toThrow(/not found/);

    expect(() => df.melt(["missing"], ["a"])).toThrow(/not found/);
    expect(() => df.melt(["a"], ["missing"])).toThrow(/not found/);
  });

  it("covers tensor conversion fromTensor errors", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    const df = DataFrame.fromTensor(t, ["c1", "c2"]);
    expect(df.columns).toEqual(["c1", "c2"]);

    expect(() => DataFrame.fromTensor(t, ["onlyOne"])).toThrow(/Column count/);
    expect(() => DataFrame.fromTensor(t, ["c1", "c2", "c3"])).toThrow(/Column count/);

    const t1d = tensor([1, 2, 3]);
    expect(() => DataFrame.fromTensor(t1d, ["a", "b"])).toThrow(/Expected exactly 1 column/);
  });
});
