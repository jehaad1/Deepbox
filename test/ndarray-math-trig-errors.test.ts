import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { add, div, mul, sub } from "../src/ndarray/ops/arithmetic";
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
  square,
  trunc,
} from "../src/ndarray/ops/math";
import {
  acos,
  asin,
  atan,
  atan2,
  cos,
  cosh,
  sin,
  sinh,
  tan,
  tanh,
} from "../src/ndarray/ops/trigonometry";

describe("ndarray math/trig error branches", () => {
  it("throws for string dtype in math ops", () => {
    const s = tensor(["a", "b"]);
    const funcs = [
      exp,
      log,
      sqrt,
      square,
      rsqrt,
      cbrt,
      expm1,
      exp2,
      log1p,
      log2,
      log10,
      floor,
      ceil,
      round,
      trunc,
    ];
    for (const fn of funcs) {
      expect(() => fn(s)).toThrow(/string dtype/i);
    }
  });

  it("throws for string dtype in trig ops", () => {
    const s = tensor(["a"]);
    const funcs = [sin, cos, tan, asin, acos, atan, sinh, cosh, tanh];
    for (const fn of funcs) {
      expect(() => fn(s)).toThrow(/string dtype/i);
    }
    expect(() => atan2(s, s)).toThrow(/string dtype/i);
  });

  it("covers BigInt arithmetic branches", () => {
    const a = tensor([1, 2], { dtype: "int64" });
    const b = tensor([3, 4], { dtype: "int64" });
    expect(add(a, b).toArray()).toEqual([4n, 6n]);
    expect(sub(b, a).toArray()).toEqual([2n, 2n]);
    expect(mul(a, b).toArray()).toEqual([3n, 8n]);
    const divOut = div(b, a);
    expect(divOut.dtype).toBe("float64");
    expect(divOut.toArray()).toEqual([3, 2]);
  });
});
