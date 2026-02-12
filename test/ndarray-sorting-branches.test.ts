import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { argsort, sort } from "../src/ndarray/ops/sorting";
import { Tensor } from "../src/ndarray/tensor/Tensor";

describe("ndarray sorting branch coverage", () => {
  it("sorts 1D numeric tensors", () => {
    const t = tensor([3, 1, 2]);
    expect(sort(t).toArray()).toEqual([1, 2, 3]);
    expect(argsort(t).toArray()).toEqual([1, 2, 0]);
  });

  it("sorts and argsorts strided 1D views", () => {
    const data = new Float32Array([3, 0, 1, 0, 2, 0, 4]);
    const t = Tensor.fromTypedArray({
      data,
      shape: [4],
      strides: [2],
      dtype: "float32",
      device: "cpu",
    });
    expect(t.toArray()).toEqual([3, 1, 2, 4]);
    expect(sort(t).toArray()).toEqual([1, 2, 3, 4]);
    expect(argsort(t).toArray()).toEqual([1, 2, 0, 3]);
  });

  it("throws for unsupported sort cases", () => {
    const s = tensor(["b", "a"]);
    expect(() => sort(s)).toThrow(/string/i);
    expect(() => argsort(s)).toThrow(/string/i);

    // N-D sort is now supported
    const t2d = tensor([
      [3, 1],
      [4, 2],
    ]);
    expect(sort(t2d).toArray()).toEqual([
      [1, 3],
      [2, 4],
    ]);
    expect(argsort(t2d).toArray()).toEqual([
      [1, 0],
      [1, 0],
    ]);
    // axis out of bounds still throws
    expect(() => sort(tensor([1, 2, 3]), 1)).toThrow(/out of bounds/);
    expect(() => argsort(tensor([1, 2, 3]), 1)).toThrow(/out of bounds/);
  });
});
