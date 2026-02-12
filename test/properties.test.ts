import { describe, expect, it } from "vitest";
import { det, matrixRank, slogdet, trace } from "../src/linalg/properties";
import { tensor, zeros } from "../src/ndarray";

describe("deepbox/linalg - Matrix Properties", () => {
  describe("det & slogdet", () => {
    it("computes determinant correctly", () => {
      const m = tensor([
        [1, 2],
        [3, 4],
      ]);
      expect(det(m)).toBeCloseTo(-2, 6);
    });

    it("handles det/slogdet edge cases", () => {
      const empty = zeros([0, 0]);
      expect(det(empty)).toBe(1);

      const singular = tensor([
        [1, 2],
        [2, 4],
      ]);
      const [sign, logdet] = slogdet(singular);
      expect(sign.toArray()).toEqual([0]);
      expect(logdet.toArray()).toEqual([-Infinity]);
    });
  });

  describe("trace", () => {
    it("computes trace correctly", () => {
      const m = tensor([
        [1, 2],
        [3, 4],
      ]);
      expect(trace(m).toArray()).toEqual([5]);
    });

    it("covers trace offsets", () => {
      const m = tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      expect(trace(m, 1).toArray()).toEqual([8]); // 2 + 6
      expect(trace(m, -1).toArray()).toEqual([12]); // 4 + 8
    });

    it("handles batched trace", () => {
      const batched = tensor([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      expect(trace(batched, 0, 1, 2).toArray()).toEqual([5, 13]);
    });

    it("validates input dimension", () => {
      expect(() => trace(tensor([1, 2]))).toThrow(/at least 2D/i);
    });
  });

  describe("matrixRank", () => {
    it("computes rank correctly", () => {
      const m = tensor([
        [1, 2],
        [3, 4],
      ]);
      expect(matrixRank(m)).toBe(2);
    });

    it("validates rank inputs", () => {
      const m = tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const rank = matrixRank(m, 1e-12);
      expect(rank).toBe(2);
      expect(() => matrixRank(tensor([1, 2]))).toThrow(/2-D/);
    });
  });
});
