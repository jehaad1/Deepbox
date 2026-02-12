import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  abs,
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

describe("ndarray arithmetic branch coverage extras", () => {
  it("covers BigInt paths with scalar and vector operands", () => {
    const bigVec = Tensor.fromTypedArray({
      data: new BigInt64Array([2n, -3n, 4n]),
      shape: [3],
      dtype: "int64",
      device: "cpu",
    });
    const bigScalar = Tensor.fromTypedArray({
      data: new BigInt64Array([2n]),
      shape: [],
      dtype: "int64",
      device: "cpu",
    });

    expect(add(bigScalar, bigVec).shape).toEqual([3]);
    expect(sub(bigVec, bigScalar).shape).toEqual([3]);
    expect(mul(bigScalar, bigVec).shape).toEqual([3]);
    expect(div(bigVec, bigScalar).shape).toEqual([3]);
    expect(floorDiv(bigScalar, bigVec).shape).toEqual([3]);
    expect(mod(bigVec, bigScalar).shape).toEqual([3]);
    expect(pow(bigVec, bigScalar).shape).toEqual([3]);

    const bigVec2 = Tensor.fromTypedArray({
      data: new BigInt64Array([1n, 1n, 1n]),
      shape: [3],
      dtype: "int64",
      device: "cpu",
    });
    expect(add(bigVec, bigVec2).shape).toEqual([3]);
  });

  it("covers BigInt scalar ops and sign/reciprocal branches", () => {
    const bigVec = Tensor.fromTypedArray({
      data: new BigInt64Array([-2n, 0n, 3n]),
      shape: [3],
      dtype: "int64",
      device: "cpu",
    });

    const negOut = neg(bigVec);
    expect(negOut.shape).toEqual([3]);
    const absOut = abs(bigVec);
    expect(absOut.shape).toEqual([3]);
    const signOut = sign(bigVec);
    expect(signOut.shape).toEqual([3]);
    const recOut = reciprocal(bigVec);
    expect(recOut.shape).toEqual([3]);

    const addOut = addScalar(bigVec, 2);
    expect(addOut.shape).toEqual([3]);
    const mulOut = mulScalar(bigVec, 2);
    expect(mulOut.shape).toEqual([3]);
  });

  it("covers numeric scalar paths and min/max/clip", () => {
    const a = tensor([1, 3, 5]);
    const b = tensor(2);
    expect(add(a, b).shape).toEqual([3]);
    expect(sub(b, a).shape).toEqual([3]);
    expect(mul(a, b).shape).toEqual([3]);
    expect(div(a, b).shape).toEqual([3]);
    expect(floorDiv(a, b).shape).toEqual([3]);
    expect(mod(a, b).shape).toEqual([3]);
    expect(pow(a, b).shape).toEqual([3]);

    const maxOut = maximum(a, tensor([2, 2, 2]));
    expect(maxOut.shape).toEqual([3]);
    const minOut = minimum(a, tensor([2, 4, 6]));
    expect(minOut.shape).toEqual([3]);

    const clipped = clip(a, 2, 4);
    expect(clipped.shape).toEqual([3]);
  });
});
