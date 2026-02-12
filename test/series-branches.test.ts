import { describe, expect, it } from "vitest";
import { Series } from "../src/dataframe";

describe("Series - Branch Coverage", () => {
  it("validates constructor errors", () => {
    expect(() => new Series([1, 2], { index: ["a"] })).toThrow(/Index length/);
    expect(() => new Series([1, 2], { index: ["a", "a"] })).toThrow(/Duplicate index/);
  });

  it("validates loc/iloc and missing labels", () => {
    const s = new Series([10, 20], { index: ["x", "y"] });
    expect(s.loc("z")).toBeUndefined();
    expect(() => s.iloc(5)).toThrow(/Position 5 is out of bounds/);
  });

  it("covers sort and unique branches", () => {
    const s = new Series([3, 1, 2], { index: ["a", "b", "c"] });
    const asc = s.sort();
    expect(asc.data[0]).toBe(1);
    const desc = s.sort(false);
    expect(desc.data[0]).toBe(3);

    const u = new Series([1, 1, 2, 3]).unique();
    expect(u).toEqual([1, 2, 3]);
  });

  it("covers valueCounts and numeric stats errors", () => {
    const counts = new Series(["a", "b", "a"]).valueCounts();
    expect(counts.data).toEqual([2, 1]);

    const bad = new Series([{ a: 1 }]);
    // valueCounts should throw for objects
    expect(() => bad.valueCounts()).toThrow(/only supports Series/);

    const empty = new Series<number>([]);
    expect(() => empty.sum()).toThrow(/empty/);
    expect(() => empty.mean()).toThrow(/empty/);
    expect(() => empty.median()).toThrow(/empty/);

    const nonNumeric = new Series(["x", "y"]);
    expect(() => nonNumeric.sum()).toThrow(/numeric/);
    expect(() => nonNumeric.mean()).toThrow(/numeric/);
    expect(() => nonNumeric.median()).toThrow(/numeric/);
  });
});
