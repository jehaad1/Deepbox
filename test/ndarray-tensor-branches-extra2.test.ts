import { describe, expect, it } from "vitest";
import { ShapeError } from "../src/core";
import { tensor } from "../src/ndarray";
import { dtypeToTypedArrayCtor, Tensor } from "../src/ndarray/tensor/Tensor";

describe("Tensor branch coverage - extra", () => {
  it("validates fromTypedArray offsets and lengths", () => {
    const data = new Float32Array([1, 2, 3]);
    expect(() =>
      Tensor.fromTypedArray({ data, shape: [2], dtype: "float32", device: "cpu", offset: -1 })
    ).toThrow(/offset must be/);

    expect(() =>
      Tensor.fromTypedArray({ data, shape: [4], dtype: "float32", device: "cpu" })
    ).toThrow(ShapeError);
  });

  it("validates fromStringArray offsets and lengths", () => {
    const data = ["a", "b"];
    expect(() => Tensor.fromStringArray({ data, shape: [2], device: "cpu", offset: -1 })).toThrow(
      /offset must be/
    );
    expect(() => Tensor.fromStringArray({ data, shape: [3], device: "cpu" })).toThrow(/too small/);
  });

  it("view/reshape handle string tensors", () => {
    const s = tensor(["a", "b", "c", "d"]);
    const view = s.view([2, 2]);
    expect(view.shape).toEqual([2, 2]);
    expect(view.toArray()).toEqual([
      ["a", "b"],
      ["c", "d"],
    ]);

    const reshaped = s.reshape([2, 2]);
    expect(reshaped.toArray()).toEqual([
      ["a", "b"],
      ["c", "d"],
    ]);
  });

  it("at supports negative indexing and bounds checks", () => {
    const t = tensor([10, 20, 30]);
    expect(t.at(-1)).toBe(30);
    expect(() => t.at(3)).toThrow(/out of bounds/);
  });

  it("dtypeToTypedArrayCtor rejects string dtype", () => {
    expect(() => dtypeToTypedArrayCtor("string")).toThrow(/string dtype/);
  });
});
