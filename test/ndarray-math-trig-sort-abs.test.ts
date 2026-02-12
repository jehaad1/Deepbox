import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { abs } from "../src/ndarray/ops/abs";
import {
  cbrt,
  ceil,
  exp,
  exp2,
  expm1,
  floor,
  log,
  log1p,
  log2,
  log10,
  round,
  rsqrt,
  sqrt,
  trunc,
} from "../src/ndarray/ops/math";
import { argsort, sort } from "../src/ndarray/ops/sorting";
import {
  acos,
  acosh,
  asin,
  asinh,
  atan,
  atan2,
  atanh,
  cos,
  cosh,
  sin,
  sinh,
  tan,
  tanh,
} from "../src/ndarray/ops/trigonometry";
import { transpose } from "../src/ndarray/tensor/shape";
import { toNum2D } from "./_helpers";

describe("deepbox/ndarray - Math, Trig, Sorting, Abs", () => {
  it("computes math ops", () => {
    const t = tensor([1, 2, 3]);
    expect(exp(t).shape).toEqual([3]);
    expect(log(t).shape).toEqual([3]);
    expect(sqrt(t).shape).toEqual([3]);
    expect(rsqrt(t).shape).toEqual([3]);
    expect(cbrt(t).shape).toEqual([3]);
    expect(expm1(t).shape).toEqual([3]);
    expect(exp2(t).shape).toEqual([3]);
    expect(log1p(t).shape).toEqual([3]);
    expect(log2(t).shape).toEqual([3]);
    expect(log10(t).shape).toEqual([3]);
    expect(floor(t).shape).toEqual([3]);
    expect(ceil(t).shape).toEqual([3]);
    expect(round(t).shape).toEqual([3]);
    expect(trunc(t).shape).toEqual([3]);
  });

  it("handles strided views for unary math ops", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const tT = transpose(t);
    const out = exp(tT);
    const expected = toNum2D(tT.toArray()).map((row) => row.map((v) => Math.exp(v)));
    expect(out.toArray()).toEqual(expected);
  });

  it("handles BigInt inputs for abs and exp", () => {
    const t = tensor([1, -2, 3], { dtype: "int64" });
    const absOut = abs(t);
    expect(absOut.dtype).toBe("int64");
    expect(absOut.toArray()).toEqual([1n, 2n, 3n]);
    const expOut = exp(t);
    expect(expOut.dtype).toBe("float64");
    const logOut = log(tensor([1, 2], { dtype: "int64" }));
    expect(logOut.dtype).toBe("float64");
    const sqrtOut = sqrt(tensor([1, 4], { dtype: "int64" }));
    expect(sqrtOut.dtype).toBe("float64");
  });

  it("computes trigonometric functions", () => {
    const t = tensor([0, Math.PI / 2, Math.PI]);
    expect(sin(t).shape).toEqual([3]);
    expect(cos(t).shape).toEqual([3]);
    expect(tan(t).shape).toEqual([3]);
    expect(asin(tensor([0, 1, -1])).shape).toEqual([3]);
    expect(acos(tensor([0, 1, -1])).shape).toEqual([3]);
    expect(atan(t).shape).toEqual([3]);
    expect(sinh(t).shape).toEqual([3]);
    expect(cosh(t).shape).toEqual([3]);
    expect(tanh(t).shape).toEqual([3]);
    expect(asinh(t).shape).toEqual([3]);
    expect(acosh(tensor([1, 2, 3])).shape).toEqual([3]);
    expect(atanh(tensor([0, 0.5, -0.5])).shape).toEqual([3]);
    const sinInt = sin(tensor([0, 1], { dtype: "int64" }));
    expect(sinInt.shape).toEqual([2]);
  });

  it("computes atan2 and validates size", () => {
    const y = tensor([1, 0]);
    const x = tensor([0, 1]);
    const out = atan2(y, x);
    expect(out.shape).toEqual([2]);
    expect(() => atan2(tensor([1, 2]), tensor([1, 2, 3]))).toThrow();
  });

  it("sorts and argsorts 1D tensors", () => {
    const t = tensor([3, 1, 2]);
    const sorted = sort(t);
    expect(sorted.toArray()).toEqual([1, 2, 3]);
    const sortedDesc = sort(t, -1, true);
    expect(sortedDesc.toArray()).toEqual([3, 2, 1]);

    const idx = argsort(t);
    expect(idx.toArray()).toEqual([1, 2, 0]);

    const big = tensor([3, 1, 2], { dtype: "int64" });
    expect(sort(big).toArray()).toEqual([1n, 2n, 3n]);
  });

  it("validates sort input", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    // N-D sort is now supported — sort along last axis by default
    expect(sort(t).toArray()).toEqual([
      [1, 2],
      [3, 4],
    ]);
    expect(argsort(t).toArray()).toEqual([
      [0, 1],
      [0, 1],
    ]);
    // axis=1 on a 1D tensor is still out of bounds
    const v = tensor([3, 2, 1]);
    expect(() => sort(v, 1)).toThrow();
    expect(() => argsort(v, 1)).toThrow();
  });

  it("throws on string dtype for math and abs", () => {
    const t = tensor(["a", "b"]);
    expect(() => abs(t)).toThrow();
    expect(() => exp(t)).toThrow();
    expect(() => sin(t)).toThrow();
    expect(() => log(t)).toThrow();
  });
});
