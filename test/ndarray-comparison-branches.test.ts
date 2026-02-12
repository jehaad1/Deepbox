import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  allclose,
  arrayEqual,
  equal,
  greater,
  isclose,
  isfinite,
  isinf,
  isnan,
  less,
  notEqual,
} from "../src/ndarray/ops/comparison";
import { Tensor } from "../src/ndarray/tensor/Tensor";

describe("deepbox/ndarray - Comparison Branches", () => {
  it("covers BigInt comparisons", () => {
    const a = tensor([1, 2], { dtype: "int64" });
    const b = tensor([2, 1], { dtype: "int64" });
    expect(equal(a, b).toArray()).toEqual([0, 0]);
    expect(notEqual(a, b).toArray()).toEqual([1, 1]);
    expect(greater(a, b).toArray()).toEqual([0, 1]);
    expect(less(a, b).toArray()).toEqual([1, 0]);
  });

  it("covers isclose/allclose error branches", () => {
    const a = tensor([1, 2], { dtype: "int64" });
    const b = tensor([1, 2], { dtype: "int64" });
    expect(() => isclose(a, b)).toThrow(/BigInt/);
    expect(() => allclose(a, b)).toThrow(/BigInt/);
  });

  it("covers arrayEqual and finite checks", () => {
    const x = tensor([1, 2, 3]);
    const y = tensor([1, 2, 3]);
    expect(arrayEqual(x, y)).toBe(true);

    const nanInf = tensor([NaN, Infinity, -Infinity, 1]);
    expect(isnan(nanInf).toArray()).toEqual([1, 0, 0, 0]);
    expect(isinf(nanInf).toArray()).toEqual([0, 1, 1, 0]);
    expect(isfinite(nanInf).toArray()).toEqual([0, 0, 0, 1]);
  });

  it("broadcasts zero-length dimensions correctly", () => {
    const a = Tensor.fromTypedArray({
      data: new Float32Array(0),
      shape: [0, 3],
      dtype: "float32",
      device: "cpu",
    });
    const b = tensor([1, 2, 3], { dtype: "float32" });
    const out = equal(a, b);
    expect(out.shape).toEqual([0, 3]);
    expect(out.size).toBe(0);
  });
});
