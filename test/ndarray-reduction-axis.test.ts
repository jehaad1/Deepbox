import { describe, expect, it } from "vitest";
import { max, min, prod, tensor, transpose } from "../src/ndarray";

describe("deepbox/ndarray - reduction axes", () => {
  it("prod() should reduce along axes with keepdims", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const p0 = prod(t, 0);
    expect(p0.toArray()).toEqual([4, 10, 18]);

    const p1 = prod(t, 1, true);
    expect(p1.shape).toEqual([2, 1]);
    expect(p1.toArray()).toEqual([[6], [120]]);
  });

  it("prod() should handle empty reductions as identity", () => {
    const t = tensor([], { dtype: "float32" });
    const p = prod(t, 0);
    expect(p.toArray()).toEqual(1);
  });

  it("min/max should support axis arrays and non-contiguous tensors", () => {
    const t = tensor([
      [3, 1, 4],
      [2, 9, 0],
    ]);
    const minAxis0 = min(t, 0);
    expect(minAxis0.toArray()).toEqual([2, 1, 0]);

    const maxAxis1 = max(t, 1);
    expect(maxAxis1.toArray()).toEqual([4, 9]);

    const minAll = min(t, [0, 1]);
    expect(minAll.toArray()).toEqual(0);

    const tT = transpose(t);
    const maxT = max(tT, 1);
    expect(maxT.toArray()).toEqual([3, 9, 4]);
  });

  it("min/max should throw on empty reduced axes", () => {
    const t = tensor([], { dtype: "float32" });
    expect(() => min(t, 0)).toThrow(/requires at least one element/);
    expect(() => max(t, 0)).toThrow(/requires at least one element/);
  });
});
