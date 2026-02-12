import { describe, expect, it } from "vitest";
import { gather, slice, tensor, transpose } from "../src/ndarray";

describe("deepbox/ndarray - indexing edge cases", () => {
  it("slice() should handle negative steps correctly", () => {
    const t = tensor([0, 1, 2, 3, 4]);
    const reversed = slice(t, { step: -1 });
    expect(reversed.toArray()).toEqual([4, 3, 2, 1, 0]);

    const mid = slice(t, { start: 4, end: 1, step: -1 });
    expect(mid.toArray()).toEqual([4, 3, 2]);
  });

  it("gather() should work on non-contiguous tensors", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    const tT = transpose(t);
    const indices = tensor([0, 2]);
    const gathered = gather(tT, indices, 1);
    expect(gathered.toArray()).toEqual([
      [1, 7],
      [2, 8],
      [3, 9],
    ]);
  });

  it("gather() should reject non-integer indices", () => {
    const t = tensor([10, 20, 30]);
    const indices = tensor([0.5, 1.1]);
    expect(() => gather(t, indices, 0)).toThrow(/not an integer/);
  });
});
