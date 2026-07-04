import { describe, expect, it } from "vitest";
import { Categorical, DataFrame, MultiIndex } from "../src/dataframe";

// ─── stack() ────────────────────────────────────────────────────────────────

describe("DataFrame.stack()", () => {
  it("stacks all columns by default", () => {
    const df = new DataFrame({
      a: [1, 2],
      b: [3, 4],
    });
    const stacked = df.stack();
    // 2 rows × 2 columns = 4 output rows
    expect(stacked.shape).toEqual([4, 2]);
    expect(stacked.columns).toEqual(["variable", "value"]);
  });

  it("stacks specified columns, keeping others as id", () => {
    const df = new DataFrame({
      city: ["NYC", "LA"],
      pop_2020: [8.3, 3.9],
      pop_2021: [8.4, 4.0],
    });
    const stacked = df.stack({
      columns: ["pop_2020", "pop_2021"],
      varName: "year",
      valueName: "population",
    });
    expect(stacked.shape).toEqual([4, 3]);
    expect(stacked.columns).toEqual(["city", "year", "population"]);

    // Check values
    const cities = stacked.get("city").data;
    expect(cities).toEqual(["NYC", "NYC", "LA", "LA"]);
    const years = stacked.get("year").data;
    expect(years).toEqual(["pop_2020", "pop_2021", "pop_2020", "pop_2021"]);
  });

  it("throws for non-existent column", () => {
    const df = new DataFrame({ a: [1] });
    expect(() => df.stack({ columns: ["nonexistent"] })).toThrow();
  });

  it("throws when varName equals valueName", () => {
    const df = new DataFrame({ a: [1] });
    expect(() => df.stack({ varName: "x", valueName: "x" })).toThrow();
  });
});

// ─── unstack() ──────────────────────────────────────────────────────────────

describe("DataFrame.unstack()", () => {
  it("unstacks long format to wide format", () => {
    const df = new DataFrame({
      city: ["NYC", "NYC", "LA", "LA"],
      year: ["2020", "2021", "2020", "2021"],
      pop: [8.3, 8.4, 3.9, 4.0],
    });
    const wide = df.unstack({
      index: "city",
      column: "year",
      value: "pop",
    });
    expect(wide.shape).toEqual([2, 2]);
    expect(wide.columns).toEqual(["2020", "2021"]);
    expect(wide.index).toEqual(["NYC", "LA"]);

    // Check values
    const col2020 = wide.get("2020").data;
    expect(col2020).toEqual([8.3, 3.9]);
    const col2021 = wide.get("2021").data;
    expect(col2021).toEqual([8.4, 4.0]);
  });

  it("throws for non-existent column", () => {
    const df = new DataFrame({ a: [1] });
    expect(() => df.unstack({ index: "a", column: "missing", value: "a" })).toThrow();
  });

  it("fills missing combinations with null", () => {
    const df = new DataFrame({
      key: ["A", "B"],
      var: ["x", "y"],
      val: [1, 2],
    });
    const wide = df.unstack({ index: "key", column: "var", value: "val" });
    expect(wide.shape).toEqual([2, 2]);
    // A only has x, B only has y
    const xCol = wide.get("x").data;
    const yCol = wide.get("y").data;
    expect(xCol[0]).toBe(1);
    expect(xCol[1]).toBeNull();
    expect(yCol[0]).toBeNull();
    expect(yCol[1]).toBe(2);
  });
});

// ─── stack/unstack roundtrip ────────────────────────────────────────────────

describe("stack/unstack roundtrip", () => {
  it("stack then unstack recovers original data", () => {
    const df = new DataFrame({
      city: ["NYC", "LA"],
      pop_2020: [8.3, 3.9],
      pop_2021: [8.4, 4.0],
    });
    const stacked = df.stack({
      columns: ["pop_2020", "pop_2021"],
      varName: "year",
      valueName: "pop",
    });
    const unstacked = stacked.unstack({
      index: "city",
      column: "year",
      value: "pop",
    });
    expect(unstacked.shape).toEqual([2, 2]);
    expect(unstacked.get("pop_2020").data).toEqual([8.3, 3.9]);
    expect(unstacked.get("pop_2021").data).toEqual([8.4, 4.0]);
  });
});

// ─── Categorical ────────────────────────────────────────────────────────────

describe("Categorical", () => {
  it("creates from array with inferred categories", () => {
    const cat = Categorical.from(["red", "blue", "red", "green", "blue"]);
    expect(cat.categories).toEqual(["blue", "green", "red"]);
    expect(cat.length).toBe(5);
    expect(cat.nCategories).toBe(3);
  });

  it("creates with explicit categories", () => {
    const cat = Categorical.from(["M", "S", "L"], {
      categories: ["S", "M", "L", "XL"],
    });
    expect(cat.categories).toEqual(["S", "M", "L", "XL"]);
    expect(cat.nCategories).toBe(4);
  });

  it("handles null/undefined as missing (-1 code)", () => {
    const cat = Categorical.from(["a", null, "b", undefined, "a"]);
    expect(cat.codes[1]).toBe(-1);
    expect(cat.codes[3]).toBe(-1);
    expect(cat.get(1)).toBeNull();
  });

  it("get() returns correct values", () => {
    const cat = Categorical.from(["x", "y", "z"]);
    expect(cat.get(0)).toBe("x");
    expect(cat.get(1)).toBe("y");
    expect(cat.get(2)).toBe("z");
  });

  it("get() throws for out-of-bounds index", () => {
    const cat = Categorical.from(["a"]);
    expect(() => cat.get(5)).toThrow();
    expect(() => cat.get(-1)).toThrow();
  });

  it("toArray() roundtrips correctly", () => {
    const values = ["red", "blue", null, "green"];
    const cat = Categorical.from(values);
    expect(cat.toArray()).toEqual(["red", "blue", null, "green"]);
  });

  it("valueCounts() returns correct counts", () => {
    const cat = Categorical.from(["a", "b", "a", "c", "a", "b"]);
    const counts = cat.valueCounts();
    expect(counts.get("a")).toBe(3);
    expect(counts.get("b")).toBe(2);
    expect(counts.get("c")).toBe(1);
  });

  it("ordered flag works", () => {
    const cat = Categorical.from(["low", "high", "mid"], {
      categories: ["low", "mid", "high"],
      ordered: true,
    });
    expect(cat.ordered).toBe(true);
    expect(cat.compare("low", "high")).toBeLessThan(0);
    expect(cat.compare("high", "low")).toBeGreaterThan(0);
    expect(cat.compare("mid", "mid")).toBe(0);
  });

  it("compare() throws for unordered categorical", () => {
    const cat = Categorical.from(["a", "b"]);
    expect(() => cat.compare("a", "b")).toThrow(/ordered/);
  });

  it("addCategories() adds new categories", () => {
    const cat = Categorical.from(["a", "b"]);
    const extended = cat.addCategories(["c", "d"]);
    expect(extended.categories).toEqual(["a", "b", "c", "d"]);
    expect(extended.toArray()).toEqual(["a", "b"]); // data unchanged
  });

  it("addCategories() throws for duplicate", () => {
    const cat = Categorical.from(["a", "b"]);
    expect(() => cat.addCategories(["a"])).toThrow(/already exists/);
  });

  it("removeCategories() removes and sets missing", () => {
    const cat = Categorical.from(["a", "b", "c", "a"]);
    const reduced = cat.removeCategories(["b"]);
    expect(reduced.categories).toEqual(["a", "c"]);
    expect(reduced.toArray()).toEqual(["a", null, "c", "a"]);
  });

  it("reorderCategories() changes order", () => {
    const cat = Categorical.from(["a", "b", "c"], {
      categories: ["a", "b", "c"],
      ordered: true,
    });
    const reordered = cat.reorderCategories(["c", "b", "a"]);
    expect(reordered.categories).toEqual(["c", "b", "a"]);
    expect(reordered.toArray()).toEqual(["a", "b", "c"]); // values same
    // But codes changed
    expect(reordered.codes[0]).toBe(2); // 'a' is at index 2
    expect(reordered.codes[1]).toBe(1); // 'b' is at index 1
    expect(reordered.codes[2]).toBe(0); // 'c' is at index 0
  });

  it("renameCategories() renames", () => {
    const cat = Categorical.from(["a", "b"]);
    const renamed = cat.renameCategories({ a: "alpha", b: "beta" });
    expect(renamed.categories).toEqual(["alpha", "beta"]);
    expect(renamed.toArray()).toEqual(["alpha", "beta"]);
  });

  it("renameCategories() throws for duplicate result", () => {
    const cat = Categorical.from(["a", "b"]);
    expect(() => cat.renameCategories({ a: "b" })).toThrow(/duplicate/);
  });

  it("sort() sorts by category order", () => {
    const cat = Categorical.from(["c", "a", "b"], {
      categories: ["a", "b", "c"],
    });
    const sorted = cat.sort();
    expect(sorted.toArray()).toEqual(["a", "b", "c"]);
  });

  it("sort() descending", () => {
    const cat = Categorical.from(["c", "a", "b"], {
      categories: ["a", "b", "c"],
    });
    const sorted = cat.sort(false);
    expect(sorted.toArray()).toEqual(["c", "b", "a"]);
  });

  it("sort() puts missing at end", () => {
    const cat = Categorical.from(["b", null, "a"]);
    const sorted = cat.sort();
    expect(sorted.toArray()).toEqual(["a", "b", null]);
  });

  it("setOrdered() changes ordering flag", () => {
    const cat = Categorical.from(["a", "b"]);
    expect(cat.ordered).toBe(false);
    const ordered = cat.setOrdered(true);
    expect(ordered.ordered).toBe(true);
  });

  it("memoryUsage() returns positive number", () => {
    const cat = Categorical.from(["a", "b", "c"]);
    expect(cat.memoryUsage()).toBeGreaterThan(0);
  });

  it("toString() returns meaningful string", () => {
    const cat = Categorical.from(["a", "b", "c"]);
    const str = cat.toString();
    expect(str).toContain("Categorical");
    expect(str).toContain("a");
  });

  it("throws for value not in explicit categories", () => {
    expect(() => Categorical.from(["a", "b", "x"], { categories: ["a", "b"] })).toThrow(
      /not in the category list/
    );
  });

  it("throws for duplicate explicit categories", () => {
    expect(() => Categorical.from(["a"], { categories: ["a", "a"] })).toThrow(/Duplicate/);
  });
});

// ─── MultiIndex ─────────────────────────────────────────────────────────────

describe("MultiIndex", () => {
  it("creates from arrays", () => {
    const mi = MultiIndex.fromArrays(
      [
        ["US", "US", "UK", "UK"],
        [2020, 2021, 2020, 2021],
      ],
      ["country", "year"]
    );
    expect(mi.length).toBe(4);
    expect(mi.nlevels).toBe(2);
    expect(mi.names).toEqual(["country", "year"]);
  });

  it("creates from tuples", () => {
    const mi = MultiIndex.fromTuples(
      [
        ["a", 1],
        ["a", 2],
        ["b", 1],
      ],
      ["letter", "number"]
    );
    expect(mi.length).toBe(3);
    expect(mi.get(0)).toEqual(["a", 1]);
    expect(mi.get(2)).toEqual(["b", 1]);
  });

  it("creates from product (Cartesian)", () => {
    const mi = MultiIndex.fromProduct(
      [
        ["a", "b"],
        [1, 2],
      ],
      ["letter", "num"]
    );
    expect(mi.length).toBe(4);
    expect(mi.get(0)).toEqual(["a", 1]);
    expect(mi.get(1)).toEqual(["a", 2]);
    expect(mi.get(2)).toEqual(["b", 1]);
    expect(mi.get(3)).toEqual(["b", 2]);
  });

  it("getLevel() returns level values", () => {
    const mi = MultiIndex.fromArrays(
      [
        ["US", "UK"],
        [2020, 2021],
      ],
      ["country", "year"]
    );
    expect(mi.getLevel("country")).toEqual(["US", "UK"]);
    expect(mi.getLevel(1)).toEqual([2020, 2021]);
  });

  it("getLevelValues() returns unique values", () => {
    const mi = MultiIndex.fromArrays(
      [
        ["US", "US", "UK"],
        [2020, 2021, 2020],
      ],
      ["country", "year"]
    );
    expect(mi.getLevelValues("country")).toEqual(["US", "UK"]);
    expect(mi.getLevelValues("year")).toEqual([2020, 2021]);
  });

  it("getPosition() finds tuple position", () => {
    const mi = MultiIndex.fromTuples(
      [
        ["a", 1],
        ["b", 2],
      ],
      ["x", "y"]
    );
    expect(mi.getPosition(["a", 1])).toBe(0);
    expect(mi.getPosition(["b", 2])).toBe(1);
    expect(mi.getPosition(["c", 3])).toBe(-1);
  });

  it("select() filters by partial labels", () => {
    const mi = MultiIndex.fromArrays(
      [
        ["US", "US", "UK", "UK"],
        [2020, 2021, 2020, 2021],
      ],
      ["country", "year"]
    );
    expect(mi.select({ country: "US" })).toEqual([0, 1]);
    expect(mi.select({ year: 2021 })).toEqual([1, 3]);
    expect(mi.select({ country: "UK", year: 2020 })).toEqual([2]);
  });

  it("droplevel() removes a level", () => {
    const mi = MultiIndex.fromArrays(
      [
        ["US", "UK"],
        [2020, 2021],
      ],
      ["country", "year"]
    );
    const dropped = mi.droplevel("country");
    expect(dropped.nlevels).toBe(1);
    expect(dropped.names).toEqual(["year"]);
  });

  it("droplevel() throws when only one level", () => {
    const mi = MultiIndex.fromArrays([[1, 2]], ["x"]);
    expect(() => mi.droplevel(0)).toThrow(/last level/);
  });

  it("swaplevel() swaps two levels", () => {
    const mi = MultiIndex.fromArrays(
      [
        ["US", "UK"],
        [2020, 2021],
      ],
      ["country", "year"]
    );
    const swapped = mi.swaplevel("country", "year");
    expect(swapped.names).toEqual(["year", "country"]);
    expect(swapped.get(0)).toEqual([2020, "US"]);
  });

  it("toFlatIndex() creates string labels", () => {
    const mi = MultiIndex.fromTuples(
      [
        ["a", 1],
        ["b", 2],
      ],
      ["x", "y"]
    );
    const flat = mi.toFlatIndex();
    expect(flat).toEqual(["(a, 1)", "(b, 2)"]);
  });

  it("toString() returns meaningful string", () => {
    const mi = MultiIndex.fromArrays(
      [
        ["US", "UK"],
        [2020, 2021],
      ],
      ["country", "year"]
    );
    const str = mi.toString();
    expect(str).toContain("MultiIndex");
    expect(str).toContain("country");
  });

  it("throws for mismatched level lengths", () => {
    expect(() => MultiIndex.fromArrays([["a", "b"], [1]])).toThrow();
  });

  it("throws for empty arrays", () => {
    expect(() => MultiIndex.fromArrays([])).toThrow();
  });

  it("get() throws for out-of-bounds", () => {
    const mi = MultiIndex.fromArrays([[1, 2]], ["x"]);
    expect(() => mi.get(5)).toThrow();
  });

  it("defaults level names when not provided", () => {
    const mi = MultiIndex.fromArrays([
      ["a", "b"],
      [1, 2],
    ]);
    expect(mi.names).toEqual(["level_0", "level_1"]);
  });
});
