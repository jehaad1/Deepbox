import { describe, expect, it } from "vitest";
import {
  asMatrix2D,
  luFactorSquare,
  luSolveInPlace,
  toDenseMatrix2D,
  toDenseVector1D,
} from "../src/linalg/_internal";
import { tensor, transpose } from "../src/ndarray";

describe("linalg internal extra branches", () => {
  it("asMatrix2D rejects non-2D and string tensors", () => {
    expect(() => asMatrix2D(tensor([1, 2, 3]))).toThrow(/2D/);
    expect(() => asMatrix2D(tensor(["a", "b"]))).toThrow(/String/);
  });

  it("toDenseMatrix2D handles non-contiguous layouts", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const t = transpose(a);
    const dense = toDenseMatrix2D(t);
    expect(dense.rows).toBe(2);
    expect(dense.cols).toBe(2);
    expect(Array.from(dense.data)).toEqual([1, 3, 2, 4]);
  });

  it("toDenseVector1D rejects non-finite values", () => {
    expect(() => toDenseVector1D(tensor([1, NaN]))).toThrow(/non-finite/i);
  });

  it("luFactorSquare handles row swaps and rejects singular matrices", () => {
    const { lu, piv, pivSign } = luFactorSquare(new Float64Array([0, 1, 2, 3]), 2);
    expect(pivSign).toBe(-1);
    expect(Array.from(piv)).toEqual([1, 0]);
    expect(lu[0]).toBe(2);

    expect(() => luFactorSquare(new Float64Array([0, 0, 0, 0]), 2)).toThrow(/singular/i);
  });

  it("luSolveInPlace throws on zero diagonal", () => {
    const lu = new Float64Array([0, 0, 0, 0]);
    const piv = new Int32Array([0, 1]);
    const b = new Float64Array([1, 2]);
    expect(() => luSolveInPlace(lu, piv, 2, b, 1)).toThrow(/singular/i);
  });
});
