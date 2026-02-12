import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { abs as absOp } from "../src/ndarray/ops/abs";
import { relu, sigmoid } from "../src/ndarray/ops/activation";
import { add, clip, div, floorDiv, mod, pow, reciprocal } from "../src/ndarray/ops/arithmetic";
import { allclose, equal, isclose } from "../src/ndarray/ops/comparison";
import { exp, log, square } from "../src/ndarray/ops/math";
import { atan2, sin } from "../src/ndarray/ops/trigonometry";
import { Tensor } from "../src/ndarray/tensor/Tensor";

describe("deepbox/ndarray - Ops Branch Coverage", () => {
  it("covers abs dtype branches", () => {
    const f64 = tensor([-1.5, 2.5], { dtype: "float64" });
    const i32 = tensor([-1, 2], { dtype: "int32" });
    const u8 = tensor([1, 2, 3], { dtype: "uint8" });
    const big = tensor([-5, 3], { dtype: "int64" });

    const absF64 = absOp(f64);
    const absI32 = absOp(i32);
    const absU8 = absOp(u8);
    const absBig = absOp(big);

    expect(absF64.dtype).toBe("float64");
    expect(absI32.dtype).toBe("int32");
    expect(absU8.dtype).toBe("uint8");
    expect(absBig.dtype).toBe("int64");
    expect(absBig.toArray()).toEqual([5n, 3n]);
  });

  it("throws on string dtype in abs", () => {
    const s = tensor(["a", "b"]);
    expect(() => absOp(s)).toThrow();
  });

  it("covers trig BigInt path and range errors", () => {
    const big = tensor([0, 1], { dtype: "int64" });
    const out = sin(big);
    expect(out.dtype).toBe("float64");

    const tooBigSin = Tensor.fromTypedArray({
      data: new BigInt64Array([BigInt("9007199254740993")]),
      shape: [1],
      dtype: "int64",
      device: "cpu",
    });
    expect(() => sin(tooBigSin)).toThrow(/too large/i);

    const atanOut = atan2(tensor([1, 2]), tensor([1]));
    expect(atanOut.shape).toEqual([2]);
  });

  it("covers math BigInt path and range errors", () => {
    const big = tensor([2, 3], { dtype: "int64" });
    expect(square(big).dtype).toBe("float64");

    const tooBigExp = Tensor.fromTypedArray({
      data: new BigInt64Array([BigInt("9007199254740993")]),
      shape: [1],
      dtype: "int64",
      device: "cpu",
    });
    expect(() => exp(tooBigExp)).toThrow(/too large/i);

    const logOut = log(tensor([1, 2], { dtype: "int64" }));
    expect(logOut.dtype).toBe("float64");
  });

  it("covers activation error and BigInt paths", () => {
    const s = tensor(["x"]);
    expect(() => relu(s)).toThrow();

    const big = tensor([-1, 0, 1], { dtype: "int64" });
    const out = sigmoid(big);
    expect(out.dtype).toBe("float64");
  });

  it("covers arithmetic branches and clip errors", () => {
    const a = tensor([[1, 2]], { dtype: "int32" });
    const b = tensor([[1], [2]], { dtype: "int32" });
    const out = add(a, b);
    expect(out.shape).toEqual([2, 2]);

    const big = tensor([4, 2], { dtype: "int64" });
    const bigScalar = tensor(2, { dtype: "int64" });
    const divOut = div(big, bigScalar);
    expect(divOut.dtype).toBe("float64");
    expect(divOut.toArray()).toEqual([2, 1]);
    expect(floorDiv(big, bigScalar).toArray()).toEqual([2n, 1n]);
    expect(mod(big, bigScalar).toArray()).toEqual([0n, 0n]);
    expect(pow(big, bigScalar).toArray()).toEqual([16n, 4n]);

    const rec = reciprocal(tensor([0, 1], { dtype: "int64" }));
    expect(rec.dtype).toBe("float64");
    expect(rec.toArray()).toEqual([Infinity, 1]);

    expect(() => clip(tensor([1, 2, 3]), 5, 2)).toThrow(/min/);
  });

  it("covers comparison BigInt and error paths", () => {
    const a = tensor([1, 2], { dtype: "int64" });
    const b = tensor([1, 3], { dtype: "int32" });
    expect(equal(a, b).toArray()).toEqual([1, 0]);

    expect(() => isclose(a, b)).toThrow(/BigInt/);
    expect(allclose(tensor([1, 2]), tensor([1]))).toBe(false);
  });
});
