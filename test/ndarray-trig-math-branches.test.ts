import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
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
import { toNumArr } from "./_helpers";

describe("deepbox/ndarray - Trig & Math Branches", () => {
  it("covers BigInt paths", () => {
    const t = tensor([1, 2], { dtype: "int64" });
    expect(sin(t).dtype).toBe("float64");
    expect(cos(t).dtype).toBe("float64");
    expect(tan(t).dtype).toBe("float64");
    expect(asin(tensor([0, 1], { dtype: "int64" })).dtype).toBe("float64");
    expect(acos(tensor([0, 1], { dtype: "int64" })).dtype).toBe("float64");
    expect(atan(t).dtype).toBe("float64");
    expect(sinh(t).dtype).toBe("float64");
    expect(cosh(t).dtype).toBe("float64");
    expect(tanh(t).dtype).toBe("float64");
    expect(asinh(t).dtype).toBe("float64");
    expect(acosh(tensor([1, 2], { dtype: "int64" })).dtype).toBe("float64");
    expect(atanh(tensor([0, 1], { dtype: "int64" })).dtype).toBe("float64");

    expect(exp(t).dtype).toBe("float64");
    expect(log(t).dtype).toBe("float64");
    expect(sqrt(t).dtype).toBe("float64");
    expect(rsqrt(t).dtype).toBe("float64");
    expect(cbrt(t).dtype).toBe("float64");
    expect(expm1(t).dtype).toBe("float64");
    expect(exp2(t).dtype).toBe("float64");
    expect(log1p(t).dtype).toBe("float64");
    expect(log2(t).dtype).toBe("float64");
    expect(log10(t).dtype).toBe("float64");
    expect(floor(t).dtype).toBe("float64");
    expect(ceil(t).dtype).toBe("float64");
    expect(round(t).dtype).toBe("float64");
    expect(trunc(t).dtype).toBe("float64");
    expect(square(t).dtype).toBe("float64");
  });

  it("throws on string dtype and bad atan2 sizes", () => {
    const s = tensor(["a", "b"]);
    expect(() => sin(s)).toThrow();
    expect(() => exp(s)).toThrow();

    expect(() => atan2(tensor([1, 2]), tensor([1, 2, 3]))).toThrow();
  });

  it("promotes integer/bool log outputs to float64", () => {
    const intLog = log(tensor([1, 2], { dtype: "int32" }));
    expect(intLog.dtype).toBe("float64");
    expect(toNumArr(intLog.toArray())[0]).toBeCloseTo(0, 6);

    const boolLog = log(tensor([1, 1], { dtype: "bool" }));
    expect(boolLog.dtype).toBe("float64");
    expect(toNumArr(boolLog.toArray())[0]).toBeCloseTo(0, 6);
  });
});
