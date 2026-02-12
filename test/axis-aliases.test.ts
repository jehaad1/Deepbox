import { describe, expect, it } from "vitest";
import { type Axis, normalizeAxes, normalizeAxis } from "../src/core";
import { DataFrame } from "../src/dataframe";
import { norm } from "../src/linalg/norms";
import { trace } from "../src/linalg/properties";
import { tensor } from "../src/ndarray";
import { sum } from "../src/ndarray/ops/reduction";
import { normalizeAxes as statsNormalizeAxes } from "../src/stats/_internal";
import { toNumArr } from "./_helpers";

describe("Axis Aliases", () => {
  describe("Core normalizeAxis", () => {
    it("should handle string aliases", () => {
      expect(normalizeAxis("rows", 2)).toBe(0);
      expect(normalizeAxis("index", 2)).toBe(0);
      expect(normalizeAxis("columns", 2)).toBe(1);
    });

    it("should throw on invalid aliases", () => {
      expect(() => normalizeAxis("foo" as string as Axis, 2)).toThrow(/Invalid axis/);
    });
  });

  describe("Core normalizeAxes", () => {
    it("should handle string aliases", () => {
      expect(normalizeAxes(["rows", "columns"], 2)).toEqual([0, 1]);
    });
  });

  describe("DataFrame.concat", () => {
    it("should support 'rows' alias (vertical stack)", () => {
      const df1 = new DataFrame({ a: [1], b: [2] });
      const df2 = new DataFrame({ a: [3], b: [4] });
      const res = df1.concat(df2, "rows");
      expect(res.shape).toEqual([2, 2]);
      // Use get() and data property
      expect(res.get("a").data).toEqual([1, 3]);
    });

    it("should support 'columns' alias (horizontal stack)", () => {
      const df1 = new DataFrame({ a: [1] });
      const df2 = new DataFrame({ b: [2] });
      const res = df1.concat(df2, "columns");
      expect(res.shape).toEqual([1, 2]);
      expect(res.columns).toEqual(["a", "b"]);
    });
  });

  describe("ndarray reduction", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]); // 2x2

    it("should support 'rows' (axis 0)", () => {
      // sum over rows -> reduce rows -> result is [1+3, 2+4] = [4, 6]
      const s = sum(t, "rows");
      expect(s.toArray()).toEqual([4, 6]);
    });

    it("should support 'columns' (axis 1)", () => {
      // sum over columns -> reduce columns -> result is [1+2, 3+4] = [3, 7]
      const s = sum(t, "columns");
      expect(s.toArray()).toEqual([3, 7]);
    });

    it("should support 'index' alias for axis 0", () => {
      const s = sum(t, "index");
      expect(s.toArray()).toEqual([4, 6]);
    });
  });

  describe("linalg operations", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);

    it("should support aliases in norm", () => {
      // axis=0 (rows) -> vector norms of columns -> [sqrt(1+9), sqrt(4+16)] = [sqrt(10), sqrt(20)]
      const n = norm(t, 2, "rows");
      if (typeof n === "number") throw new Error("Expected Tensor");
      const arr = toNumArr(n.toArray());
      expect(arr[0]).toBeCloseTo(Math.sqrt(10));
      expect(arr[1]).toBeCloseTo(Math.sqrt(20));
    });

    it("should support aliases in trace", () => {
      // trace(t, 0, "rows", "columns") -> main diagonal
      const tr = trace(t, 0, "rows", "columns");
      // 1 + 4 = 5
      // Access underlying data directly for scalar
      expect(Number(tr.data[0])).toBe(5);
    });
  });

  describe("Stats internal", () => {
    it("should normalize string axes correctly", () => {
      const axes = statsNormalizeAxes(["rows", "columns"], 2);
      expect(axes).toEqual([0, 1]);
    });
  });
});
