import { describe, expect, it } from "vitest";
import { inv, solve } from "../../src/linalg";
import {
  asMatrix2D,
  assertFiniteTensor,
  fromDenseMatrix2D,
  luFactorSquare,
  luSolveInPlace,
  toDenseMatrix2D,
} from "../../src/linalg/_internal";
import { tensor } from "../../src/ndarray";
import { Tensor } from "../../src/ndarray/tensor";
import { transpose } from "../../src/ndarray/tensor/shape";

const f64 = { dtype: "float64" } as const;

describe("v1.5.0 linalg/_internal", () => {
  it("LU handles tiny pivots without overflowing 1/pivot", () => {
    // numpy.linalg.solve([[1e-310, 1], [1, 1]], [1, 2]) -> [1, 1]
    const a = new Float64Array([1e-310, 1, 1, 1]);
    const { lu, piv } = luFactorSquare(a, 2);
    const b = new Float64Array([1, 2]);
    luSolveInPlace(lu, piv, 2, b, 1);
    expect(b[0]).toBeCloseTo(1, 12);
    expect(b[1]).toBeCloseTo(1, 12);

    // Whole system in the subnormal range: A = 1e-310 * [[2, 1], [1, 3]], b = 1e-310 * [1, 2]
    const s = 1e-310;
    const x = solve(
      tensor(
        [
          [2 * s, 1 * s],
          [1 * s, 3 * s],
        ],
        f64
      ),
      tensor([1 * s, 2 * s], f64)
    );
    // [[2,1],[1,3]] x = [1,2] -> x = [0.2, 0.6]
    const xd = Array.from(x.data as Float64Array);
    expect(xd[0]).toBeCloseTo(0.2, 6);
    expect(xd[1]).toBeCloseTo(0.6, 6);
  });

  it("inverse of a matrix with huge entries stays accurate", () => {
    const big = 1e300;
    const r = inv(
      tensor(
        [
          [2 * big, 1 * big],
          [1 * big, 3 * big],
        ],
        f64
      )
    );
    const d = Array.from(r.data as Float64Array);
    // inv([[2,1],[1,3]]) = [[0.6,-0.2],[-0.2,0.4]] / 1e300
    expect(d[0]! * big).toBeCloseTo(0.6, 12);
    expect(d[1]! * big).toBeCloseTo(-0.2, 12);
    expect(d[3]! * big).toBeCloseTo(0.4, 12);
  });

  it("reports overflow separately from singularity", () => {
    expect(() => luFactorSquare(new Float64Array([1, 2, 2, 4]), 2)).toThrow(/singular/i);
    expect(() => luFactorSquare(new Float64Array([Number.NaN, 1, 1, 1]), 2)).toThrow(/non-finite/i);
    // The second pivot is 1e308 - (-1e308) = Infinity (numpy.linalg.det gives inf here).
    expect(() => luFactorSquare(new Float64Array([1e308, -1e308, 1e308, 1e308]), 2)).toThrow(
      /overflowed/i
    );
  });

  it("validates buffer sizes of the LU helpers", () => {
    expect(() => luFactorSquare(new Float64Array(3), 2)).toThrow(/does not match/);
    expect(() => luFactorSquare(new Float64Array(4), -1)).toThrow(/non-negative integer/);
    expect(() => luFactorSquare(new Float64Array(4), 1.5)).toThrow(/non-negative integer/);
    const { lu, piv } = luFactorSquare(new Float64Array([2, 0, 0, 2]), 2);
    expect(() => luSolveInPlace(lu, piv, 2, new Float64Array(3), 1)).toThrow(/luSolveInPlace/);
    expect(() => luSolveInPlace(lu, new Int32Array(1), 2, new Float64Array(2), 1)).toThrow(
      /luSolveInPlace/
    );
    expect(() => luSolveInPlace(lu, piv, 2, new Float64Array(2), 0.5)).toThrow(/nrhs/);
  });

  it("empty system round-trips", () => {
    const { lu, piv } = luFactorSquare(new Float64Array(0), 0);
    expect(lu.length).toBe(0);
    expect(piv.length).toBe(0);
    luSolveInPlace(lu, piv, 0, new Float64Array(0), 3);
  });

  it("fromDenseMatrix2D rejects mismatched data length", () => {
    expect(() => fromDenseMatrix2D(2, 2, new Float64Array(3))).toThrow(/does not match shape/);
  });

  it("asMatrix2D flags single-row and single-column views as contiguous", () => {
    const base = new Float64Array([0, 0, 0, 4, 5, 6, 7, 8, 9]);
    const mk = (shape: [number, number], strides: number[], offset: number) =>
      Tensor.fromTypedArray({
        data: base,
        shape,
        dtype: "float64",
        device: "cpu",
        strides,
        offset,
      });
    // 1xN view whose (irrelevant) row stride is not N, e.g. a row of a wider matrix.
    const row = mk([1, 3], [9, 1], 3);
    expect(asMatrix2D(row).isRowMajorContiguous).toBe(true);
    expect(Array.from(toDenseMatrix2D(row).data)).toEqual([4, 5, 6]);
    // Nx1 view with an irrelevant column stride.
    const col = mk([3, 1], [1, 7], 3);
    expect(asMatrix2D(col).isRowMajorContiguous).toBe(true);
    expect(Array.from(toDenseMatrix2D(col).data)).toEqual([4, 5, 6]);
    // A genuinely strided view is not contiguous but still converts correctly.
    const t = transpose(
      tensor(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        f64
      )
    );
    expect(asMatrix2D(t).isRowMajorContiguous).toBe(false);
    expect(Array.from(toDenseMatrix2D(t).data)).toEqual([1, 4, 2, 5, 3, 6]);
  });

  it("assertFiniteTensor scans views with a non-zero offset", () => {
    const data = new Float64Array([Number.NaN, 1, 2, Number.POSITIVE_INFINITY]);
    const view = (offset: number, n: number) =>
      Tensor.fromTypedArray({ data, shape: [n], dtype: "float64", device: "cpu", offset });
    expect(() => assertFiniteTensor(view(1, 2), "x")).not.toThrow();
    expect(() => assertFiniteTensor(view(0, 2), "x")).toThrow(/non-finite/);
    expect(() => assertFiniteTensor(view(2, 2), "x")).toThrow(/non-finite/);
  });
});
