import { describe, expect, it } from "vitest";
import { cond, inv, norm, pinv } from "../src/linalg";
import { lstsq } from "../src/linalg/solvers/lstsq";
import { tensor } from "../src/ndarray";
import { Tensor } from "../src/ndarray/tensor";

describe("linalg extra branches", () => {
  it("inv validates shape and handles empty matrices", () => {
    expect(() => inv(tensor([1, 2, 3]))).toThrow(/2D/);
    expect(() => inv(tensor([[1, 2, 3]]))).toThrow(/square/);

    const nonFinite = tensor([
      [1, NaN],
      [2, 3],
    ]);
    expect(() => inv(nonFinite)).toThrow(/non-finite/i);

    const empty = Tensor.fromTypedArray({
      data: new Float64Array(0),
      shape: [0, 0],
      dtype: "float64",
      device: "cpu",
    });
    const invEmpty = inv(empty);
    expect(invEmpty.shape).toEqual([0, 0]);
  });

  it("pinv validates ndim and handles zero-size matrices", () => {
    expect(() => pinv(tensor([1, 2, 3]))).toThrow(/2D/);
    expect(() =>
      pinv(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        -1
      )
    ).toThrow(/rcond/);

    const nonFinite = tensor([
      [1, 2],
      [Infinity, 4],
    ]);
    expect(() => pinv(nonFinite)).toThrow(/non-finite/i);

    const empty = Tensor.fromTypedArray({
      data: new Float64Array(0),
      shape: [0, 2],
      dtype: "float64",
      device: "cpu",
    });
    const out = pinv(empty);
    expect(out.shape).toEqual([2, 0]);
  });

  it("pinv respects rcond cutoff", () => {
    const A = tensor([
      [1, 0],
      [0, 0],
    ]);
    const Ainv = pinv(A, 1);
    for (let i = 0; i < Ainv.size; i++) {
      expect(Number(Ainv.data[Ainv.offset + i])).toBe(0);
    }
  });

  it("lstsq validates rcond", () => {
    const A = tensor([
      [1, 2],
      [3, 4],
    ]);
    const b = tensor([1, 0]);
    expect(() => lstsq(A, b, -0.1)).toThrow(/rcond/);
  });

  it("computes vector norms across orders", () => {
    const v = tensor([3, -4, 0]);
    expect(norm(v, 1)).toBe(7);
    expect(norm(v, Number.POSITIVE_INFINITY)).toBe(4);
    expect(Number(norm(v, 3))).toBeCloseTo((3 ** 3 + 4 ** 3) ** (1 / 3), 6);
    expect(Number(norm(v, 2))).toBeCloseTo(5, 6);
    const bad = tensor([1, NaN, 2]);
    expect(() => norm(bad, 2)).toThrow(/non-finite/i);
  });

  it("computes matrix Frobenius and axis-based norms", () => {
    const m = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(Number(norm(m, "fro"))).toBeCloseTo(Math.sqrt(30), 6);
    const n10 = norm(m, 1, 0);
    if (typeof n10 === "number") throw new Error("Expected Tensor");
    expect(n10.toArray()).toEqual([4, 6]);
    expect(Number(norm(m, 1, [0, 1]))).toBe(6);
    expect(Number(norm(m, 2, [0, 1]))).toBeCloseTo(Math.sqrt(15 + Math.sqrt(221)), 6);
    const bad = tensor([
      [1, 2],
      [3, Infinity],
    ]);
    expect(() => norm(bad, "fro")).toThrow(/non-finite/i);
  });

  it("cond handles frobenius, empty, and singular matrices", () => {
    const A = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(cond(A, "fro")).toBeGreaterThan(0);
    expect(() => cond(A, 1)).toThrow(/Only 2-norm/);

    const bad = tensor([
      [1, 2],
      [NaN, 4],
    ]);
    expect(() => cond(bad)).toThrow(/non-finite/i);

    const empty = Tensor.fromTypedArray({
      data: new Float64Array(0),
      shape: [0, 0],
      dtype: "float64",
      device: "cpu",
    });
    expect(cond(empty)).toBe(Infinity);

    const singular = tensor([
      [1, 0],
      [0, 0],
    ]);
    expect(cond(singular)).toBe(Infinity);
  });
});
