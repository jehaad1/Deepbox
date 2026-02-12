import { describe, expect, it } from "vitest";
import { ShapeError } from "../src/core";
import { add } from "../src/ndarray/ops/arithmetic";
import { sum } from "../src/ndarray/ops/reduction";
import { Tensor } from "../src/ndarray/tensor/Tensor";

describe("NDArray extra tests", () => {
  it("should throw ShapeError for invalid broadcasting with zero dimension", () => {
    const a = Tensor.fromTypedArray({
      data: new Float32Array(),
      shape: [0, 2],
      dtype: "float32",
      device: "cpu",
    });
    const b = Tensor.fromTypedArray({
      data: new Float32Array([1, 2, 3, 4]),
      shape: [2, 2],
      dtype: "float32",
      device: "cpu",
    });
    expect(() => add(a, b)).toThrow(ShapeError);
  });

  it("should return additive identity for sum over empty axis", () => {
    const a = Tensor.fromTypedArray({
      data: new Float32Array(),
      shape: [2, 0, 3],
      dtype: "float32",
      device: "cpu",
    });
    const result = sum(a, 1);
    expect(result.shape).toEqual([2, 3]);
    expect(result.toArray()).toEqual([
      [0, 0, 0],
      [0, 0, 0],
    ]);
  });

  it("should return additive identity for sum over empty axis with keepdims=true", () => {
    const a = Tensor.fromTypedArray({
      data: new Float32Array(),
      shape: [2, 0, 3],
      dtype: "float32",
      device: "cpu",
    });
    const result = sum(a, 1, true);
    expect(result.shape).toEqual([2, 1, 3]);
    const expected = [[[0, 0, 0]], [[0, 0, 0]]];
    expect(result.toArray()).toEqual(expected);
  });
});
