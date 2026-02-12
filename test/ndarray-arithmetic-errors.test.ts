import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { add, clip } from "../src/ndarray/ops/arithmetic";
import { Tensor } from "../src/ndarray/tensor/Tensor";

describe("ndarray arithmetic error branches", () => {
  it("throws on dtype mismatch and shape mismatch", () => {
    const a = tensor([1, 2], { dtype: "float32" });
    const b = tensor([1, 2], { dtype: "int32" });
    expect(() => add(a, b)).toThrow(/DType mismatch/i);

    const c = tensor([1, 2]);
    const d = tensor([1, 2, 3]);
    expect(() => add(c, d)).toThrow(/broadcast/i);
  });

  it("rejects invalid broadcasting for zero-sized dimensions", () => {
    const zeroRows = Tensor.fromTypedArray({
      data: new Float32Array(0),
      shape: [0, 3],
      dtype: "float32",
      device: "cpu",
    });
    const twoRows = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      { dtype: "float32" }
    );

    expect(() => add(zeroRows, twoRows)).toThrow(/broadcast/i);
  });

  it("throws when clip min > max", () => {
    const t = tensor([1, 2, 3]);
    expect(() => clip(t, 5, 1)).toThrow(/min.*<= max/i);
  });
});
