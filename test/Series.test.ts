import { describe, expect, it } from "vitest";
import { Series } from "../src/dataframe";

describe("Series", () => {
  it("should create Series from array", () => {
    const s = new Series([1, 2, 3, 4, 5]);
    expect(s.length).toBe(5);
  });

  it("should create Series with custom index", () => {
    const s = new Series([1, 2, 3], {
      index: ["a", "b", "c"],
      name: "numbers",
    });

    expect(s.name).toBe("numbers");
    expect(s.index).toEqual(["a", "b", "c"]);
  });

  it("should get values by index", () => {
    const s = new Series([10, 20, 30]);
    const value = s.iloc(1);
    expect(value).toBe(20);
  });

  it("should filter values", () => {
    const s = new Series([1, 2, 3, 4, 5]);
    const filtered = s.filter((x: number) => x > 3);
    expect(filtered.length).toBe(2);
    expect(filtered.data).toEqual([4, 5]);
  });

  it("should map over values", () => {
    const s = new Series([1, 2, 3]);
    const doubled = s.map((x: number) => x * 2);
    expect(doubled.length).toBe(3);
    expect(doubled.data).toEqual([2, 4, 6]);
  });

  it("should compute statistics", () => {
    const s = new Series([1, 2, 3, 4, 5]);
    expect(s.sum()).toBe(15);
    expect(s.std()).toBeCloseTo(1.5811, 3); // sqrt(2.5)
  });

  it("should get unique values", () => {
    const s = new Series([1, 2, 2, 3, 3, 3]);
    const unique = s.unique();
    expect(unique.length).toBe(3);
    expect(unique).toEqual([1, 2, 3]);
  });

  it("should count values", () => {
    const s = new Series(["a", "b", "a", "c", "b", "a"]);
    const counts = s.valueCounts();
    expect(counts.get("a")).toBe(3);
    expect(counts.get("b")).toBe(2);
    expect(counts.get("c")).toBe(1);
  });

  it("should sort values", () => {
    const s = new Series([3, 1, 4, 1, 5, 9, 2, 6]);
    const sorted = s.sort();
    expect(sorted.data).toEqual([1, 1, 2, 3, 4, 5, 6, 9]);
  });
});
