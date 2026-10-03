import { describe, expect, it } from "vitest";
import { DataValidationError, InvalidParameterError } from "../../src/core/errors/index";
import { Categorical } from "../../src/dataframe/Categorical";
import * as root from "../../src/index";

// Reference values come from pandas 3.0 (pd.Categorical).

describe("Categorical: inference", () => {
  it("sorts inferred categories by Unicode code point like pandas", () => {
    // JS default sort would put U+1F600 (surrogate pair 0xD83D...) before U+FFEE.
    const cat = Categorical.from(["\u{1F600}", "￮", "a"]);
    expect(cat.categories.map((c) => c.codePointAt(0))).toEqual([97, 65518, 128512]);
  });

  it("keeps plain ASCII and BMP inference in code point order", () => {
    // Python: sorted(["b", "B", "a", "é", "", "ab"]) == ["", "B", "a", "ab", "b", "é"]
    const cat = Categorical.from(["b", "B", "a", "é", "", "ab"]);
    expect(cat.categories).toEqual(["", "B", "a", "ab", "b", "é"]);
  });

  it("orders astral characters by code point against BMP and lone surrogates", () => {
    // Python code point order: U+0061 < U+10000 < U+1F600; lone U+D800 sorts below U+10000.
    const cat = Categorical.from(["\u{1F600}", "\u{10000}", "a", "\ud800"]);
    expect(cat.categories.map((c) => c.codePointAt(0))).toEqual([97, 0xd800, 0x10000, 0x1f600]);
  });

  it("rejects non-string values instead of storing them", () => {
    const bad = [1, "a"] as unknown as string[];
    expect(() => Categorical.from(bad)).toThrow(DataValidationError);
    expect(() => Categorical.from(bad, { categories: ["a"] })).toThrow(DataValidationError);
  });

  it("rejects non-string explicit categories", () => {
    const bad = ["a", 2] as unknown as string[];
    expect(() => Categorical.from(["a"], { categories: bad })).toThrow(DataValidationError);
  });

  it("freezes the category list", () => {
    const cat = Categorical.from(["a", "b"]);
    expect(Object.isFrozen(cat.categories)).toBe(true);
  });

  it("does not alias the caller's category array", () => {
    const cats = ["a", "b"];
    const cat = Categorical.from(["a"], { categories: cats });
    cats.push("c");
    expect(cat.categories).toEqual(["a", "b"]);
  });
});

describe("Categorical.get", () => {
  it("throws for non-integer indices instead of returning null", () => {
    const cat = Categorical.from(["a", "b"]);
    expect(() => cat.get(0.5)).toThrow(InvalidParameterError);
    expect(() => cat.get(Number.NaN)).toThrow(InvalidParameterError);
    expect(() => cat.get(Number.POSITIVE_INFINITY)).toThrow(InvalidParameterError);
    expect(cat.get(1)).toBe("b");
  });
});

describe("Categorical.removeCategories", () => {
  it("throws for unknown categories (pandas raises ValueError)", () => {
    const cat = Categorical.from(["a", "b"]);
    expect(() => cat.removeCategories(["z"])).toThrow(DataValidationError);
    expect(() => cat.removeCategories(["z"])).toThrow(/not found/);
  });

  it("matches pandas codes after removal", () => {
    const cat = Categorical.from(["a", "b", "c", "a", null], { categories: ["a", "b", "c"] });
    const out = cat.removeCategories(["a"]);
    expect(Array.from(out.codes)).toEqual([-1, 0, 1, -1, -1]);
    expect(out.categories).toEqual(["b", "c"]);
  });

  it("does not mutate the original", () => {
    const cat = Categorical.from(["a", "b"]);
    cat.removeCategories(["a"]);
    expect(cat.toArray()).toEqual(["a", "b"]);
  });
});

describe("Categorical.reorderCategories", () => {
  it("rejects duplicates that would silently drop a category", () => {
    const cat = Categorical.from(["a", "b", "c"]);
    expect(() => cat.reorderCategories(["a", "a", "b"])).toThrow(DataValidationError);
    expect(() => cat.reorderCategories(["a", "a", "b"])).toThrow(/Duplicate/);
  });

  it("rejects unknown categories and wrong length", () => {
    const cat = Categorical.from(["a", "b", "c"]);
    expect(() => cat.reorderCategories(["a", "b", "z"])).toThrow(DataValidationError);
    expect(() => cat.reorderCategories(["a", "b"])).toThrow(InvalidParameterError);
  });

  it("supports the ordered option and matches pandas codes", () => {
    const cat = Categorical.from(["a", "b", "c", "a", null], { categories: ["a", "b", "c"] });
    const out = cat.reorderCategories(["c", "b", "a"], { ordered: true });
    expect(Array.from(out.codes)).toEqual([2, 1, 0, 2, -1]);
    expect(out.ordered).toBe(true);
    expect(cat.reorderCategories(["c", "b", "a"]).ordered).toBe(false);
  });
});

describe("Categorical.setCategories", () => {
  it("matches pandas set_categories", () => {
    const cat = Categorical.from(["a", "b", "c", "a", null], { categories: ["a", "b", "c"] });
    const out = cat.setCategories(["b", "c", "d"]);
    expect(Array.from(out.codes)).toEqual([-1, 0, 1, -1, -1]);
    expect(out.categories).toEqual(["b", "c", "d"]);
  });

  it("validates the new list", () => {
    const cat = Categorical.from(["a"]);
    expect(() => cat.setCategories(["a", "a"])).toThrow(DataValidationError);
  });
});

describe("Categorical.removeUnusedCategories / min / max", () => {
  const cat = Categorical.from(["a", "c"], { categories: ["a", "b", "c"], ordered: true });

  it("drops unused categories", () => {
    const out = cat.removeUnusedCategories();
    expect(out.categories).toEqual(["a", "c"]);
    expect(out.toArray()).toEqual(["a", "c"]);
    expect(out.ordered).toBe(true);
  });

  it("min and max follow category order", () => {
    expect(cat.min()).toBe("a");
    expect(cat.max()).toBe("c");
  });

  it("min and max return null when everything is missing", () => {
    const empty = Categorical.from([null], { categories: ["a"], ordered: true });
    expect(empty.min()).toBeNull();
    expect(empty.max()).toBeNull();
  });

  it("min and max require ordered categoricals", () => {
    expect(() => Categorical.from(["a"]).min()).toThrow(/ordered/);
    expect(() => Categorical.from(["a"]).max()).toThrow(/ordered/);
  });
});

describe("Categorical.renameCategories", () => {
  it("does not pick up inherited Object.prototype properties", () => {
    const cat = Categorical.from(["constructor", "toString", "x"]);
    const out = cat.renameCategories({ x: "y" });
    expect(out.categories).toEqual(["constructor", "toString", "y"]);
    expect(out.toArray()).toEqual(["constructor", "toString", "y"]);
  });

  it("accepts a function like pandas", () => {
    const cat = Categorical.from(["a", "b", "a", null]);
    const out = cat.renameCategories((c) => c.toUpperCase());
    expect(out.toArray()).toEqual(["A", "B", "A", null]);
  });

  it("accepts a Map and ignores unknown keys", () => {
    const cat = Categorical.from(["a", "b"]);
    const out = cat.renameCategories(
      new Map([
        ["a", "x"],
        ["zzz", "q"],
      ])
    );
    expect(out.categories).toEqual(["x", "b"]);
  });

  it("rejects duplicate results from a function", () => {
    const cat = Categorical.from(["a", "b"]);
    expect(() => cat.renameCategories(() => "same")).toThrow(/duplicate/);
  });
});

describe("Categorical.sort", () => {
  it("matches pandas descending order with missing last", () => {
    const cat = Categorical.from(["c", "a", null, "b", "a"], { categories: ["a", "b", "c"] });
    expect(Array.from(cat.sort(false).codes)).toEqual([2, 1, 0, 0, -1]);
    expect(Array.from(cat.sort().codes)).toEqual([0, 0, 1, 2, -1]);
  });

  it("handles empty and all-missing input", () => {
    expect(Categorical.from([]).sort().length).toBe(0);
    expect(Categorical.from([null, null]).sort().toArray()).toEqual([null, null]);
  });

  it("keeps unused categories and the ordered flag", () => {
    const cat = Categorical.from(["b", "a"], { categories: ["a", "b", "z"], ordered: true });
    const out = cat.sort();
    expect(out.categories).toEqual(["a", "b", "z"]);
    expect(out.ordered).toBe(true);
    expect(cat.toArray()).toEqual(["b", "a"]);
  });
});

describe("Categorical misc", () => {
  it("isNull marks missing values", () => {
    expect(Categorical.from(["a", null, undefined]).isNull()).toEqual([false, true, true]);
  });

  it("setOrdered keeps categories and codes", () => {
    const cat = Categorical.from(["b", "a", null]).setOrdered(true);
    expect(cat.ordered).toBe(true);
    expect(cat.compare("a", "b")).toBeLessThan(0);
    expect(cat.toArray()).toEqual(["b", "a", null]);
  });

  it("asOrdered / asUnordered", () => {
    const cat = Categorical.from(["a"]);
    expect(cat.asOrdered().ordered).toBe(true);
    expect(cat.asOrdered().asUnordered().ordered).toBe(false);
  });

  it("valueCounts includes unused categories in category order", () => {
    const cat = Categorical.from(["b", "b", null], { categories: ["a", "b"] });
    expect([...cat.valueCounts()]).toEqual([
      ["a", 0],
      ["b", 2],
    ]);
  });

  it("toString elides the middle of long arrays and shows NaN for missing", () => {
    const vals = Array.from({ length: 12 }, (_, i) => (i === 11 ? null : `v${i}`));
    const s = Categorical.from(vals).toString();
    expect(s.startsWith("Categorical([v0, v1, v2, v3, v4, ..., v10, NaN])")).toBe(true);
  });

  it("root entry point exposes every module namespace", () => {
    expect(Object.keys(root).sort()).toEqual(
      expect.arrayContaining([
        "core",
        "dataframe",
        "datasets",
        "linalg",
        "metrics",
        "ml",
        "ndarray",
        "nn",
        "optim",
        "plot",
        "preprocess",
        "random",
        "stats",
      ])
    );
  });
});
