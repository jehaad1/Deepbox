import { describe, expect, it } from "vitest";
import { DataFrame } from "../src/dataframe";

describe("DataFrame.applymap()", () => {
  it("applies function to every element", () => {
    const df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] });
    const result = df.applymap((v) => (v as number) * 2);
    expect(result.get("a").data).toEqual([2, 4, 6]);
    expect(result.get("b").data).toEqual([8, 10, 12]);
  });

  it("preserves index and columns", () => {
    const df = new DataFrame({ x: [10] }, { index: ["row0"] });
    const result = df.applymap((v) => (v as number) + 1);
    expect(result.columns).toEqual(["x"]);
    expect(result.index).toEqual(["row0"]);
    expect(result.get("x").data).toEqual([11]);
  });

  it("works with string transformation", () => {
    const df = new DataFrame({ name: ["alice", "bob"] });
    const result = df.applymap((v) => (v as string).toUpperCase());
    expect(result.get("name").data).toEqual(["ALICE", "BOB"]);
  });
});

describe("DataFrame.pipe()", () => {
  it("passes DataFrame through a function", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    const result = df.pipe((d) => d.shape);
    expect(result).toEqual([3, 1]);
  });

  it("passes extra arguments", () => {
    const df = new DataFrame({ a: [1, 2, 3] });
    const result = df.pipe((d, n) => d.head(n as number), 2);
    expect(result.shape).toEqual([2, 1]);
  });

  it("supports chaining", () => {
    const df = new DataFrame({ a: [1, 2, 3], b: [4, 5, 6] });
    const cols = df.pipe((d) => d.columns);
    expect(cols).toEqual(["a", "b"]);
  });
});

describe("DataFrame.explode()", () => {
  it("explodes an array column into rows", () => {
    const df = new DataFrame({
      id: [1, 2],
      tags: [["a", "b"], ["c"]],
    });
    const result = df.explode("tags");
    expect(result.shape).toEqual([3, 2]);
    expect(result.get("tags").data).toEqual(["a", "b", "c"]);
    expect(result.get("id").data).toEqual([1, 1, 2]);
  });

  it("preserves non-array values as-is", () => {
    const df = new DataFrame({
      x: [1, 2],
      y: ["single", ["a", "b"]],
    });
    const result = df.explode("y");
    expect(result.shape).toEqual([3, 2]);
    expect(result.get("y").data).toEqual(["single", "a", "b"]);
    expect(result.get("x").data).toEqual([1, 2, 2]);
  });

  it("uses numeric index after explosion", () => {
    const df = new DataFrame({ vals: [[1, 2], [3]] }, { index: ["a", "b"] });
    const result = df.explode("vals");
    expect(result.index).toEqual([0, 1, 2]);
    expect(result.get("vals").data).toEqual([1, 2, 3]);
  });

  it("throws on unknown column", () => {
    const df = new DataFrame({ a: [1] });
    expect(() => df.explode("z")).toThrow("not found");
  });
});

describe("DataFrame.getDummies()", () => {
  it("creates dummy columns for a categorical column", () => {
    const df = new DataFrame({
      color: ["red", "blue", "red", "green"],
      value: [1, 2, 3, 4],
    });
    const result = df.getDummies("color");
    expect(result.columns).toContain("value");
    expect(result.columns).toContain("color_blue");
    expect(result.columns).toContain("color_green");
    expect(result.columns).toContain("color_red");
    expect(result.get("color_red").data).toEqual([1, 0, 1, 0]);
    expect(result.get("color_blue").data).toEqual([0, 1, 0, 0]);
    expect(result.get("color_green").data).toEqual([0, 0, 0, 1]);
  });

  it("drops first category with dropFirst option", () => {
    const df = new DataFrame({ animal: ["cat", "dog", "cat"] });
    const result = df.getDummies("animal", { dropFirst: true });
    // Categories sorted: cat, dog → drop "cat", keep "dog"
    expect(result.columns).toEqual(["animal_dog"]);
    expect(result.get("animal_dog").data).toEqual([0, 1, 0]);
  });

  it("uses custom prefix", () => {
    const df = new DataFrame({ status: ["A", "B", "A"] });
    const result = df.getDummies("status", { prefix: "s" });
    expect(result.columns).toContain("s_A");
    expect(result.columns).toContain("s_B");
  });

  it("auto-detects string columns when no columns specified", () => {
    const df = new DataFrame({
      num: [1, 2, 3],
      cat: ["a", "b", "a"],
    });
    const result = df.getDummies();
    expect(result.columns).toContain("num");
    expect(result.columns).toContain("cat_a");
    expect(result.columns).toContain("cat_b");
  });

  it("handles multiple columns", () => {
    const df = new DataFrame({
      color: ["red", "blue"],
      size: ["S", "L"],
    });
    const result = df.getDummies(["color", "size"]);
    expect(result.columns).toContain("color_blue");
    expect(result.columns).toContain("color_red");
    expect(result.columns).toContain("size_L");
    expect(result.columns).toContain("size_S");
  });

  it("preserves index", () => {
    const df = new DataFrame({ cat: ["a", "b"] }, { index: ["r1", "r2"] });
    const result = df.getDummies("cat");
    expect(result.index).toEqual(["r1", "r2"]);
  });
});
