import { describe, expect, it } from "vitest";
import {
  fromDenseMatrix2D,
  fromDenseVector1D,
  luFactorSquare,
  luSolveInPlace,
  toDenseMatrix2D,
  toDenseVector1D,
} from "../src/linalg/_internal";
import { tensor } from "../src/ndarray";
import { Tensor } from "../src/ndarray/tensor";
import { transpose } from "../src/ndarray/tensor/shape";

describe("linalg internal branch coverage", () => {
  it("handles matrix views and dense conversions", () => {
    const m = tensor([
      [1, 2],
      [3, 4],
    ]);

    const dense = toDenseMatrix2D(m);
    expect(Array.from(dense.data)).toEqual([1, 2, 3, 4]);

    const mt = transpose(m);
    const denseT = toDenseMatrix2D(mt);
    expect(Array.from(denseT.data)).toEqual([1, 3, 2, 4]);

    expect(() => toDenseVector1D(m)).toThrow(/1D/);
  });

  it("rejects non-finite and string values in dense conversions", () => {
    const bad = tensor([
      [1, NaN],
      [2, 3],
    ]);
    expect(() => toDenseMatrix2D(bad)).toThrow(/non-finite/i);
    const badT = transpose(bad);
    expect(() => toDenseMatrix2D(badT)).toThrow(/non-finite/i);
    const vec = tensor([1, Infinity]);
    expect(() => toDenseVector1D(vec)).toThrow(/non-finite/i);

    const str = Tensor.fromStringArray({ data: ["a", "b"], shape: [2] });
    expect(() => toDenseVector1D(str)).toThrow(/string/i);
    const str2d = Tensor.fromStringArray({
      data: ["a", "b", "c", "d"],
      shape: [2, 2],
    });
    expect(() => toDenseMatrix2D(str2d)).toThrow(/string/i);
  });

  it("covers LU factorization and solver error paths", () => {
    const singular = new Float64Array([1, 2, 2, 4]);
    expect(() => luFactorSquare(singular, 2)).toThrow(/singular/i);

    const needsPivot = new Float64Array([0, 1, 1, 1]);
    const fact = luFactorSquare(needsPivot, 2);
    expect(fact.pivSign).toBe(-1);

    const lu = new Float64Array([0, 1, 0, 0]);
    const b = new Float64Array([1, 2]);
    expect(() => luSolveInPlace(lu, new Int32Array([0, 1]), 2, b, 1)).toThrow(/singular/i);
  });

  it("creates tensors from dense buffers", () => {
    const t2 = fromDenseMatrix2D(1, 2, new Float64Array([5, 6]));
    expect(t2.shape).toEqual([1, 2]);
    const t1 = fromDenseVector1D(new Float64Array([7, 8]));
    expect(t1.shape).toEqual([2]);
  });
});
