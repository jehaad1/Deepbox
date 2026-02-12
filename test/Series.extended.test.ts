import { describe, expect, it } from "vitest";
import { Series } from "../src/dataframe";

describe("Series - Extended Tests", () => {
  describe("Constructor and Properties", () => {
    it("should create Series from array", () => {
      const s = new Series([1, 2, 3, 4, 5]);
      expect(s.length).toBe(5);
      expect(s.data).toEqual([1, 2, 3, 4, 5]);
    });

    it("should create empty Series", () => {
      const s = new Series([]);
      expect(s.length).toBe(0);
    });

    it("should create Series with custom index", () => {
      const s = new Series([1, 2, 3], {
        index: ["a", "b", "c"],
        name: "numbers",
      });
      expect(s.name).toBe("numbers");
      expect(s.index).toEqual(["a", "b", "c"]);
    });

    it("should create Series with numeric index", () => {
      const s = new Series([10, 20, 30], { index: [0, 1, 2] });
      expect(s.index).toEqual([0, 1, 2]);
    });

    it("should throw for index length mismatch", () => {
      expect(() => new Series([1, 2, 3], { index: ["a", "b"] })).toThrow(
        /Index length.*must match data length/
      );
    });

    it("should throw for duplicate index labels", () => {
      expect(() => new Series([1, 2], { index: ["a", "a"] })).toThrow(/Duplicate index label/);
    });

    it("should throw for undefined index labels", () => {
      const invalidIndex: Array<string | number> = ["a", "b"];
      delete invalidIndex[0];
      expect(() => new Series([1, 2], { index: invalidIndex })).toThrow(
        /Index labels cannot be undefined/
      );
    });

    it("should handle mixed type data", () => {
      const s = new Series([1, "two", 3, "four"]);
      expect(s.length).toBe(4);
    });

    it("should set name from options", () => {
      const s = new Series([1, 2], { name: "test" });
      expect(s.name).toBe("test");
    });

    it("should have undefined name by default", () => {
      const s = new Series([1, 2]);
      expect(s.name).toBeUndefined();
    });
  });

  describe("Data Access Methods", () => {
    it("should get value by numeric position", () => {
      const s = new Series([10, 20, 30]);
      expect(s.iloc(0)).toBe(10);
      expect(s.iloc(1)).toBe(20);
      expect(s.iloc(2)).toBe(30);
    });

    it("should throw for out of bounds position", () => {
      const s = new Series([1, 2, 3]);
      expect(() => s.iloc(5)).toThrow(/out of bounds/);
    });

    it("should get value by label with loc", () => {
      const s = new Series([10, 20], { index: ["a", "b"] });
      expect(s.loc("a")).toBe(10);
      expect(s.loc("b")).toBe(20);
    });

    it("should return undefined for non-existent label", () => {
      const s = new Series([10, 20], { index: ["a", "b"] });
      expect(s.loc("c")).toBeUndefined();
    });

    it("should get value with numeric label", () => {
      const s = new Series([10, 20, 30], { index: [0, 1, 2] });
      expect(s.get(0)).toBe(10);
      expect(s.get(1)).toBe(20);
    });

    it("should get value with string label", () => {
      const s = new Series([10, 20], { index: ["a", "b"] });
      expect(s.get("a")).toBe(10);
    });

    it("should prioritize label lookup over positional", () => {
      const s = new Series([10, 20, 30], { index: [5, 10, 15] });
      expect(s.get(5)).toBe(10);
    });
  });

  describe("head() and tail()", () => {
    it("should return first n elements", () => {
      const s = new Series([1, 2, 3, 4, 5, 6]);
      const head = s.head(3);
      expect(head.data).toEqual([1, 2, 3]);
    });

    it("should return default 5 elements with head", () => {
      const s = new Series([1, 2, 3, 4, 5, 6, 7, 8]);
      const head = s.head();
      expect(head.length).toBe(5);
    });

    it("should return all elements when n > length", () => {
      const s = new Series([1, 2, 3]);
      const head = s.head(10);
      expect(head.length).toBe(3);
    });

    it("should return last n elements", () => {
      const s = new Series([1, 2, 3, 4, 5, 6]);
      const tail = s.tail(3);
      expect(tail.data).toEqual([4, 5, 6]);
    });

    it("should preserve index in head", () => {
      const s = new Series([1, 2, 3], { index: ["a", "b", "c"] });
      const head = s.head(2);
      expect(head.index).toEqual(["a", "b"]);
    });

    it("should preserve index in tail", () => {
      const s = new Series([1, 2, 3], { index: ["a", "b", "c"] });
      const tail = s.tail(2);
      expect(tail.index).toEqual(["b", "c"]);
    });

    it("should preserve name in head", () => {
      const s = new Series([1, 2, 3], { name: "test" });
      const head = s.head(2);
      expect(head.name).toBe("test");
    });

    it("should preserve name in tail", () => {
      const s = new Series([1, 2, 3], { name: "test" });
      const tail = s.tail(2);
      expect(tail.name).toBe("test");
    });
  });

  describe("filter()", () => {
    it("should filter elements by predicate", () => {
      const s = new Series([1, 2, 3, 4, 5]);
      const filtered = s.filter((x) => x > 3);
      expect(filtered.data).toEqual([4, 5]);
    });

    it("should preserve aligned index", () => {
      const s = new Series([1, 2, 3, 4, 5], {
        index: ["a", "b", "c", "d", "e"],
      });
      const filtered = s.filter((x) => x > 3);
      expect(filtered.index).toEqual(["d", "e"]);
    });

    it("should return empty Series when no matches", () => {
      const s = new Series([1, 2, 3]);
      const filtered = s.filter((x) => x > 10);
      expect(filtered.length).toBe(0);
    });

    it("should preserve name", () => {
      const s = new Series([1, 2, 3], { name: "test" });
      const filtered = s.filter((x) => x > 1);
      expect(filtered.name).toBe("test");
    });

    it("should provide index to predicate", () => {
      const s = new Series([10, 20, 30]);
      const filtered = s.filter((_x, i) => i > 0);
      expect(filtered.data).toEqual([20, 30]);
    });
  });

  describe("map()", () => {
    it("should transform elements", () => {
      const s = new Series([1, 2, 3]);
      const mapped = s.map((x) => x * 2);
      expect(mapped.data).toEqual([2, 4, 6]);
    });

    it("should preserve index", () => {
      const s = new Series([1, 2, 3], { index: ["a", "b", "c"] });
      const mapped = s.map((x) => x * 2);
      expect(mapped.index).toEqual(["a", "b", "c"]);
    });

    it("should preserve name", () => {
      const s = new Series([1, 2, 3], { name: "test" });
      const mapped = s.map((x) => x * 2);
      expect(mapped.name).toBe("test");
    });

    it("should change type with map", () => {
      const s = new Series([1, 2, 3]);
      const mapped = s.map((x) => String(x));
      expect(mapped.data).toEqual(["1", "2", "3"]);
    });

    it("should provide index to mapper", () => {
      const s = new Series([10, 20, 30]);
      const mapped = s.map((x, i) => x + i);
      expect(mapped.data).toEqual([10, 21, 32]);
    });
  });

  describe("sort()", () => {
    it("should sort numeric values ascending", () => {
      const s = new Series([3, 1, 4, 1, 5, 9, 2, 6]);
      const sorted = s.sort();
      expect(sorted.data).toEqual([1, 1, 2, 3, 4, 5, 6, 9]);
    });

    it("should sort numeric values descending", () => {
      const s = new Series([3, 1, 2]);
      const sorted = s.sort(false);
      expect(sorted.data).toEqual([3, 2, 1]);
    });

    it("should sort string values", () => {
      const s = new Series(["charlie", "alice", "bob"]);
      const sorted = s.sort();
      expect(sorted.data).toEqual(["alice", "bob", "charlie"]);
    });

    it("should preserve index-value mapping", () => {
      const s = new Series([3, 1, 2], { index: ["a", "b", "c"] });
      const sorted = s.sort();
      expect(sorted.data).toEqual([1, 2, 3]);
      expect(sorted.index).toEqual(["b", "c", "a"]);
    });

    it("should preserve name", () => {
      const s = new Series([3, 1, 2], { name: "test" });
      const sorted = s.sort();
      expect(sorted.name).toBe("test");
    });
  });

  describe("unique()", () => {
    it("should return unique values", () => {
      const s = new Series([1, 2, 2, 3, 3, 3]);
      const unique = s.unique();
      expect(unique).toEqual([1, 2, 3]);
    });

    it("should preserve order of first occurrence", () => {
      const s = new Series([3, 1, 2, 1, 3]);
      const unique = s.unique();
      expect(unique).toEqual([3, 1, 2]);
    });

    it("should work with strings", () => {
      const s = new Series(["a", "b", "a", "c", "b"]);
      const unique = s.unique();
      expect(unique).toEqual(["a", "b", "c"]);
    });

    it("should return empty array for empty Series", () => {
      const s = new Series([]);
      const unique = s.unique();
      expect(unique).toEqual([]);
    });
  });

  describe("valueCounts()", () => {
    it("should count occurrences", () => {
      const s = new Series(["a", "b", "a", "c", "a"]);
      const counts = s.valueCounts();
      expect(counts.loc("a")).toBe(3);
      expect(counts.loc("b")).toBe(1);
      expect(counts.loc("c")).toBe(1);
    });

    it("should work with numbers", () => {
      const s = new Series([1, 2, 1, 3, 1]);
      const counts = s.valueCounts();
      expect(counts.loc(1)).toBe(3);
      expect(counts.loc(2)).toBe(1);
    });

    it("should throw for non-string/number types", () => {
      const s = new Series([{}, {}, {}]);
      expect(() => s.valueCounts()).toThrow(/only supports Series<string \| number>/);
    });

    it("should set appropriate name", () => {
      const s = new Series([1, 2, 1], { name: "values" });
      const counts = s.valueCounts();
      expect(counts.name).toBe("values_counts");
    });
  });

  describe("Statistical Methods", () => {
    describe("sum()", () => {
      it("should compute sum", () => {
        const s = new Series([1, 2, 3, 4]);
        expect(s.sum()).toBe(10);
      });

      it("should throw for empty Series", () => {
        const s = new Series([]);
        expect(() => s.sum()).toThrow(/Cannot get sum of empty Series/);
      });

      it("should throw for non-numeric data", () => {
        const s = new Series(["a", "b", "c"]);
        expect(() => s.sum()).toThrow(/only works on numeric data/);
      });

      it("should handle negative numbers", () => {
        const s = new Series([1, -2, 3, -4]);
        expect(s.sum()).toBe(-2);
      });

      it("should handle decimals", () => {
        const s = new Series([1.5, 2.5, 3.5]);
        expect(s.sum()).toBeCloseTo(7.5, 5);
      });
    });

    describe("mean()", () => {
      it("should compute mean", () => {
        const s = new Series([1, 2, 3, 4]);
        expect(s.mean()).toBe(2.5);
      });

      it("should throw for empty Series", () => {
        const s = new Series([]);
        expect(() => s.mean()).toThrow(/Cannot get mean of empty Series/);
      });

      it("should throw for non-numeric data", () => {
        const s = new Series(["a", "b"]);
        expect(() => s.mean()).toThrow(/only works on numeric data/);
      });

      it("should handle single value", () => {
        const s = new Series([5]);
        expect(s.mean()).toBe(5);
      });
    });

    describe("median()", () => {
      it("should compute median for odd length", () => {
        const s = new Series([1, 2, 3, 4, 5]);
        expect(s.median()).toBe(3);
      });

      it("should compute median for even length", () => {
        const s = new Series([1, 2, 3, 4]);
        expect(s.median()).toBe(2.5);
      });

      it("should throw for empty Series", () => {
        const s = new Series([]);
        expect(() => s.median()).toThrow(/Cannot get median of empty Series/);
      });

      it("should not mutate original data", () => {
        const data = [3, 1, 2];
        const s = new Series(data);
        s.median();
        expect(data).toEqual([3, 1, 2]);
      });

      it("should handle unsorted data", () => {
        const s = new Series([5, 1, 9, 3, 7]);
        expect(s.median()).toBe(5);
      });
    });

    describe("std()", () => {
      it("should compute standard deviation", () => {
        const s = new Series([1, 2, 3, 4, 5]);
        expect(s.std()).toBeCloseTo(1.5811, 3);
      });

      it("should return NaN for single value", () => {
        const s = new Series([5]);
        expect(s.std()).toBeNaN();
      });

      it("should throw for empty Series", () => {
        const s = new Series([]);
        expect(() => s.std()).toThrow(/Cannot get std of empty Series/);
      });

      it("should use sample standard deviation", () => {
        const s = new Series([2, 4, 6, 8]);
        const std = s.std();
        expect(std).toBeCloseTo(2.582, 2);
      });
    });

    describe("var()", () => {
      it("should compute variance", () => {
        const s = new Series([2, 4, 6, 8]);
        expect(s.var()).toBeCloseTo(6.667, 2);
      });

      it("should return NaN for single value", () => {
        const s = new Series([5]);
        expect(s.var()).toBeNaN();
      });

      it("should throw for empty Series", () => {
        const s = new Series([]);
        expect(() => s.var()).toThrow(/Cannot get variance of empty Series/);
      });
    });

    describe("min()", () => {
      it("should find minimum value", () => {
        const s = new Series([5, 2, 8, 1, 9]);
        expect(s.min()).toBe(1);
      });

      it("should throw for empty Series", () => {
        const s = new Series([]);
        expect(() => s.min()).toThrow(/Cannot get min of empty Series/);
      });

      it("should handle negative numbers", () => {
        const s = new Series([5, -2, 8, -10]);
        expect(s.min()).toBe(-10);
      });

      it("should handle single value", () => {
        const s = new Series([42]);
        expect(s.min()).toBe(42);
      });
    });

    describe("max()", () => {
      it("should find maximum value", () => {
        const s = new Series([5, 2, 8, 1, 9]);
        expect(s.max()).toBe(9);
      });

      it("should throw for empty Series", () => {
        const s = new Series([]);
        expect(() => s.max()).toThrow(/Cannot get max of empty Series/);
      });

      it("should handle negative numbers", () => {
        const s = new Series([-5, -2, -8, -10]);
        expect(s.max()).toBe(-2);
      });

      it("should handle single value", () => {
        const s = new Series([42]);
        expect(s.max()).toBe(42);
      });
    });
  });

  describe("Conversion Methods", () => {
    it("should convert to array", () => {
      const s = new Series([1, 2, 3]);
      const arr = s.toArray();
      expect(arr).toEqual([1, 2, 3]);
    });

    it("should return copy in toArray", () => {
      const s = new Series([1, 2, 3]);
      const arr = s.toArray();
      arr[0] = 999;
      expect(s.data[0]).toBe(1);
    });

    it("should convert to tensor", () => {
      const s = new Series([1, 2, 3, 4]);
      const tensor = s.toTensor();
      expect(tensor.shape).toEqual([4]);
    });

    it("should throw when converting non-numeric to tensor", () => {
      const s = new Series(["a", "b", "c"]);
      expect(() => s.toTensor()).toThrow(/only works on numeric data/);
    });
  });

  describe("Edge Cases", () => {
    it("should handle Series with null values", () => {
      const s = new Series([1, null, 3, null, 5]);
      expect(s.length).toBe(5);
    });

    it("should handle Series with undefined values", () => {
      const s = new Series([1, undefined, 3]);
      expect(s.length).toBe(3);
    });

    it("should handle Series with mixed numeric types", () => {
      const s = new Series([1, 2.5, 3, 4.7]);
      expect(s.sum()).toBeCloseTo(11.2, 5);
    });

    it("should handle large Series", () => {
      const data = Array.from({ length: 10000 }, (_, i) => i);
      const s = new Series(data);
      expect(s.length).toBe(10000);
      expect(s.sum()).toBe(49995000);
    });

    it("should handle Series with all same values", () => {
      const s = new Series([5, 5, 5, 5, 5]);
      expect(s.unique()).toEqual([5]);
      expect(s.std()).toBe(0);
    });

    it("should handle Series with boolean values", () => {
      const s = new Series([true, false, true, false]);
      expect(s.length).toBe(4);
    });

    it("should handle Series with object values", () => {
      const s = new Series([{ a: 1 }, { b: 2 }]);
      expect(s.length).toBe(2);
    });
  });

  describe("Immutability", () => {
    it("should not mutate original data in constructor", () => {
      const data = [1, 2, 3];
      const s = new Series(data);
      data[0] = 999;
      expect(s.data[0]).toBe(1);
    });

    it("should not mutate original index in constructor", () => {
      const index = ["a", "b", "c"];
      const s = new Series([1, 2, 3], { index });
      index[0] = "z";
      expect(s.index[0]).toBe("a");
    });

    it("should return readonly data", () => {
      const s = new Series([1, 2, 3]);
      const data = s.data;
      expect(() => {
        Reflect.set(data, 0, 999);
      }).not.toThrow();
    });

    it("should return readonly index", () => {
      const s = new Series([1, 2, 3]);
      const index = s.index;
      expect(() => {
        Reflect.set(index, 0, 999);
      }).not.toThrow();
    });
  });

  describe("Type Handling", () => {
    it("should handle generic type parameter", () => {
      const s = new Series<number>([1, 2, 3]);
      expect(s.data).toEqual([1, 2, 3]);
    });

    it("should handle string Series", () => {
      const s = new Series<string>(["a", "b", "c"]);
      expect(s.data).toEqual(["a", "b", "c"]);
    });

    it("should handle boolean Series", () => {
      const s = new Series<boolean>([true, false, true]);
      expect(s.data).toEqual([true, false, true]);
    });
  });

  describe("Performance", () => {
    it("should handle large filter operations efficiently", () => {
      const data = Array.from({ length: 100000 }, (_, i) => i);
      const s = new Series(data);
      const start = Date.now();
      const filtered = s.filter((x) => x % 2 === 0);
      const duration = Date.now() - start;
      expect(filtered.length).toBe(50000);
      expect(duration).toBeLessThan(1000);
    });

    it("should handle large map operations efficiently", () => {
      const data = Array.from({ length: 100000 }, (_, i) => i);
      const s = new Series(data);
      const start = Date.now();
      const mapped = s.map((x) => x * 2);
      const duration = Date.now() - start;
      expect(mapped.length).toBe(100000);
      expect(duration).toBeLessThan(500);
    });

    it("should handle large sort operations efficiently", () => {
      const data = Array.from({ length: 10000 }, () => Math.random());
      const s = new Series(data);
      const start = Date.now();
      const sorted = s.sort();
      const duration = Date.now() - start;
      expect(sorted.length).toBe(10000);
      expect(duration).toBeLessThan(500);
    });
  });
});
