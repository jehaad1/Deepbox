import { describe, expect, it } from "vitest";
import { cond, norm } from "../src/linalg/norms";
import { tensor, zeros } from "../src/ndarray";
import { transpose } from "../src/ndarray/tensor/shape";

describe("deepbox/linalg - Norms and Condition Numbers", () => {
  describe("norm", () => {
    it("computes vector norms", () => {
      const v = tensor([1, -2, 3]);
      expect(norm(v)).toBeCloseTo(3.741657, 5); // L2
      expect(norm(v, 1)).toBe(6);
      expect(norm(v, Infinity)).toBe(3);
    });

    it("computes matrix norms", () => {
      const m = tensor([
        [1, 2],
        [3, 4],
      ]);
      expect(norm(m, "fro")).toBeCloseTo(5.477225, 5);
      expect(norm(m, "nuc")).toBeCloseTo(5.8309519, 5);
    });

    it("rejects non-finite values in norm", () => {
      const v = tensor([1, NaN, 2]);
      expect(() => norm(v, 2)).toThrow(/non-finite/i);
      const m = tensor([
        [1, 2],
        [3, Infinity],
      ]);
      expect(() => norm(m, "fro")).toThrow(/non-finite/i);
    });

    it("rejects non-finite values in non-contiguous views", () => {
      const m = tensor([
        [1, 2],
        [3, NaN],
      ]);
      const mt = transpose(m);
      expect(() => norm(mt, "fro")).toThrow(/non-finite/i);
    });
  });

  describe("cond", () => {
    it("validates cond inputs", () => {
      expect(() => cond(tensor([1, 2]))).toThrow(/2D matrix/i);
      expect(() =>
        cond(
          tensor([
            [1, 2],
            [3, 4],
          ]),
          1
        )
      ).toThrow(/Only 2-norm/i);
    });

    it("handles empty matrices for cond", () => {
      const empty = zeros([0, 0]);
      expect(cond(empty)).toBe(Infinity);
    });

    it("does not mask non-finite inputs in cond", () => {
      const bad = tensor([
        [1, NaN],
        [2, 3],
      ]);
      expect(() => cond(bad)).toThrow(/non-finite/i);
    });

    it("computes condition number", () => {
      const m = tensor([
        [1, 2],
        [3, 4],
      ]);
      expect(cond(m)).toBeCloseTo(14.933034, 5);
      const I = tensor([
        [1, 0],
        [0, 1],
      ]);
      expect(cond(I)).toBeCloseTo(1, 8);
    });
  });
});
