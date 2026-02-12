import { describe, expect, it } from "vitest";
import { tensor, transpose } from "../src/ndarray";
import {
  add,
  addScalar,
  clip,
  div,
  floorDiv,
  maximum,
  minimum,
  mod,
  mul,
  mulScalar,
  neg,
  pow,
  reciprocal,
  sign,
  sub,
} from "../src/ndarray/ops/arithmetic";
import { Tensor } from "../src/ndarray/tensor/Tensor";

describe("deepbox/ndarray - Arithmetic Branches", () => {
  it("covers broadcast shape cases", () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const b = tensor([10, 20, 30]);
    const c = add(a, b);
    expect(c.shape).toEqual([2, 3]);
    expect(sub(a, b).shape).toEqual([2, 3]);
    expect(mul(a, b).shape).toEqual([2, 3]);
    expect(div(a, b).shape).toEqual([2, 3]);
    expect(maximum(a, b).shape).toEqual([2, 3]);
    expect(minimum(a, b).shape).toEqual([2, 3]);

    const d = add(tensor([[1], [2]]), tensor([[3, 4, 5]]));
    expect(d.shape).toEqual([2, 3]);

    const at = transpose(a);
    const e = add(at, tensor([[100], [200], [300]]));
    expect(e.shape).toEqual([3, 2]);
    expect(sub(at, tensor([[100], [200], [300]])).shape).toEqual([3, 2]);
  });

  it("covers BigInt branches", () => {
    const a = tensor([5, 6], { dtype: "int64" });
    const b = tensor([2, 3], { dtype: "int64" });
    expect(add(a, b).toArray()).toEqual([7n, 9n]);
    expect(sub(a, b).toArray()).toEqual([3n, 3n]);
    expect(mul(a, b).toArray()).toEqual([10n, 18n]);
    const divOut = div(a, b);
    expect(divOut.dtype).toBe("float64");
    expect(divOut.toArray()).toEqual([2.5, 2]);
    expect(floorDiv(a, b).toArray()).toEqual([2n, 2n]);
    expect(mod(a, b).toArray()).toEqual([1n, 0n]);
    expect(pow(a, b).toArray()).toEqual([25n, 216n]);

    const rec = reciprocal(tensor([0, 2], { dtype: "int64" }));
    expect(rec.dtype).toBe("float64");
    expect(rec.toArray()).toEqual([Infinity, 0.5]);
    expect(sign(tensor([-1, 0, 2], { dtype: "int64" })).toArray()).toEqual([-1n, 0n, 1n]);
  });

  it("covers scalar helpers and errors", () => {
    const t = tensor([1, -2, 3], { dtype: "int32" });
    expect(addScalar(t, 2).toArray()).toEqual([3, 0, 5]);
    expect(mulScalar(t, 2).toArray()).toEqual([2, -4, 6]);
    expect(neg(t).toArray()).toEqual([-1, 2, -3]);
    expect(maximum(tensor([1, 5]), tensor([2, 3])).toArray()).toEqual([2, 5]);
    expect(minimum(tensor([1, 5]), tensor([2, 3])).toArray()).toEqual([1, 3]);

    expect(() => clip(tensor([1, 2, 3]), 2, 1)).toThrow(/min/);
  });

  it("broadcasts zero-length dimensions correctly", () => {
    const a = Tensor.fromTypedArray({
      data: new Float32Array(0),
      shape: [0, 3],
      dtype: "float32",
      device: "cpu",
    });
    const b = tensor([1, 2, 3], { dtype: "float32" });
    const out = add(a, b);
    expect(out.shape).toEqual([0, 3]);
    expect(out.size).toBe(0);
  });

  it("promotes integer division and reciprocal to float outputs", () => {
    const a = tensor([1, 2], { dtype: "int32" });
    const b = tensor([2, 2], { dtype: "int32" });
    const out = div(a, b);
    expect(out.dtype).toBe("float64");
    expect(out.toArray()).toEqual([0.5, 1]);

    const rec = reciprocal(tensor([2, 4], { dtype: "uint8" }));
    expect(rec.dtype).toBe("float64");
    expect(rec.toArray()).toEqual([0.5, 0.25]);
  });
});
