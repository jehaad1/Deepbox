import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  allclose,
  equal,
  greater,
  greaterEqual,
  isclose,
  isfinite,
  isinf,
  isnan,
  less,
  lessEqual,
  notEqual,
} from "../src/ndarray/ops/comparison";
import { Tensor } from "../src/ndarray/tensor/Tensor";

describe("ndarray comparison error branches", () => {
  it("throws for string dtype comparisons", () => {
    const s = tensor(["a", "b"]);
    expect(() => equal(s, s)).toThrow(/string dtype/i);
    expect(() => notEqual(s, s)).toThrow(/string dtype/i);
    expect(() => greater(s, s)).toThrow(/string dtype/i);
    expect(() => greaterEqual(s, s)).toThrow(/string dtype/i);
    expect(() => less(s, s)).toThrow(/string dtype/i);
    expect(() => lessEqual(s, s)).toThrow(/string dtype/i);
    expect(() => isclose(s, s)).toThrow(/string dtype/i);
    expect(() => allclose(s, s)).toThrow(/string dtype/i);
    expect(() => isnan(s)).toThrow(/string dtype/i);
    expect(() => isinf(s)).toThrow(/string dtype/i);
    expect(() => isfinite(s)).toThrow(/string dtype/i);
  });

  it("throws for invalid shapes and unsupported BigInt isclose/allclose", () => {
    const a = tensor([1, 2]);
    const b = tensor([1, 2, 3]);
    expect(() => equal(a, b)).toThrow(/broadcast/i);

    const big = tensor([1, 2], { dtype: "int64" });
    expect(() => isclose(big, big)).toThrow(/BigInt/i);
    expect(() => allclose(big, big)).toThrow(/BigInt/i);
  });

  it("rejects invalid zero-dimension broadcasting in comparisons", () => {
    const zeroRows = Tensor.fromTypedArray({
      data: new Float32Array(0),
      shape: [0, 3],
      dtype: "float32",
      device: "cpu",
    });
    const twoRows = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      { dtype: "float32" }
    );

    expect(() => equal(zeroRows, twoRows)).toThrow(/broadcast/i);
  });
});
