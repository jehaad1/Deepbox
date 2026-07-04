import { describe, expect, it } from "vitest";
import { parseCSV } from "../src/datasets";

describe("parseCSV", () => {
  it("parses a simple CSV with header", () => {
    const csv = `a,b,target\n1,2,0\n3,4,1\n5,6,0`;
    const ds = parseCSV(csv);

    expect(ds.data.shape).toEqual([3, 2]);
    expect(ds.target.shape).toEqual([3]);
    expect(ds.featureNames).toEqual(["a", "b"]);
    expect(ds.targetName).toBe("target");
  });

  it("uses last column as target by default", () => {
    const csv = `f1,f2,f3,y\n1,2,3,10\n4,5,6,20`;
    const ds = parseCSV(csv);

    expect(ds.data.shape).toEqual([2, 3]);
    expect(ds.target.shape).toEqual([2]);
    expect(Number(ds.target.data[ds.target.offset])).toBe(10);
    expect(Number(ds.target.data[ds.target.offset + 1])).toBe(20);
  });

  it("supports custom target column", () => {
    const csv = `y,f1,f2\n0,1,2\n1,3,4`;
    const ds = parseCSV(csv, { targetColumn: 0 });

    expect(ds.data.shape).toEqual([2, 2]);
    expect(ds.targetName).toBe("y");
    expect(ds.featureNames).toEqual(["f1", "f2"]);
  });

  it("handles CSV without header", () => {
    const csv = `1,2,3\n4,5,6`;
    const ds = parseCSV(csv, { header: false });

    expect(ds.data.shape).toEqual([2, 2]);
    expect(ds.target.shape).toEqual([2]);
    expect(ds.featureNames).toEqual(["feature_0", "feature_1"]);
    expect(ds.targetName).toBe("feature_2");
  });

  it("handles custom separator", () => {
    const csv = `a;b;c\n1;2;3\n4;5;6`;
    const ds = parseCSV(csv, { separator: ";" });

    expect(ds.data.shape).toEqual([2, 2]);
    expect(ds.target.shape).toEqual([2]);
  });

  it("throws on empty CSV", () => {
    expect(() => parseCSV("")).toThrow(/empty/i);
  });

  it("throws on non-numeric data", () => {
    const csv = `a,b\nhello,1\n2,3`;
    expect(() => parseCSV(csv)).toThrow(/non-numeric/i);
  });

  it("throws on invalid target column", () => {
    const csv = `a,b\n1,2\n3,4`;
    expect(() => parseCSV(csv, { targetColumn: 5 })).toThrow(/targetColumn/);
  });

  it("includes description", () => {
    const csv = `a,b\n1,2\n3,4`;
    const ds = parseCSV(csv);
    expect(ds.description).toContain("2 samples");
  });

  it("handles Windows-style line endings", () => {
    const csv = "a,b,c\r\n1,2,3\r\n4,5,6\r\n";
    const ds = parseCSV(csv);
    expect(ds.data.shape).toEqual([2, 2]);
  });
});
