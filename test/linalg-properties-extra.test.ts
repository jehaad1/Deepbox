import { describe, expect, it } from "vitest";
import { det, matrixRank, slogdet, trace } from "../src/linalg/properties";
import { tensor, zeros } from "../src/ndarray";

describe("linalg properties extra branches", () => {
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

  it("covers trace offsets and matrixRank validation", () => {
    const m = tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    expect(trace(m, 1).toArray()).toEqual([8]);
    expect(trace(m, -1).toArray()).toEqual([12]);

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

    const rank = matrixRank(m, 1e-12);
    expect(rank).toBe(2);
    expect(() => matrixRank(tensor([1, 2]))).toThrow(/2-D/);
  });
});
