import { describe, expect, it } from "vitest";
import { DataFrame, Series } from "../src/dataframe";
import { tensor } from "../src/ndarray";

describe("Audit Fixes Verification", () => {
  describe("Series Statistics (n=1)", () => {
    it("should return NaN for std() when n=1", () => {
      const s = new Series([10]);
      expect(s.std()).toBeNaN();
    });

    it("should return NaN for var() when n=1", () => {
      const s = new Series([10]);
      expect(s.var()).toBeNaN();
    });

    it("should still work for n > 1", () => {
      const s = new Series([10, 20]);
      expect(s.std()).not.toBeNaN();
      expect(s.var()).not.toBeNaN();
    });
  });

  describe("DataFrame.fromTensor with Strings", () => {
    it("should allow creating DataFrame from string tensor (simulated array storage)", () => {
      const stringTensor = tensor(["a", "b", "c"], { dtype: "string" });
      const df = DataFrame.fromTensor(stringTensor, ["chars"]);

      expect(df.shape).toEqual([3, 1]);
      expect(df.columns).toEqual(["chars"]);
      expect(df.get("chars").toArray()).toEqual(["a", "b", "c"]);
    });

    it("should allow creating 2D DataFrame from string tensor", () => {
      const stringTensor2D = tensor(
        [
          ["a", "b"],
          ["c", "d"],
        ],
        { dtype: "string" }
      );
      const df = DataFrame.fromTensor(stringTensor2D, ["col1", "col2"]);

      expect(df.shape).toEqual([2, 2]);
      expect(df.iloc(0)).toEqual({ col1: "a", col2: "b" });
      expect(df.iloc(1)).toEqual({ col1: "c", col2: "d" });
    });
  });
});
