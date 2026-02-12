import { describe, expect, it } from "vitest";
import { reshape, tensor } from "../src/ndarray";
import {
  add,
  clip,
  div,
  floorDiv,
  maximum,
  minimum,
  mod,
  pow,
  reciprocal,
  sign,
} from "../src/ndarray/ops/arithmetic";
import {
  allclose,
  arrayEqual,
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
import { prod, sum, variance } from "../src/ndarray/ops/reduction";
import { Tensor } from "../src/ndarray/tensor/Tensor";

describe("ndarray ops additional branch coverage", () => {
  it("covers arithmetic BigInt and negative exponent branches", () => {
    const bigA = Tensor.fromTypedArray({
      data: new BigInt64Array([2n, 4n]),
      shape: [2],
      dtype: "int64",
      device: "cpu",
    });
    const bigNegExp = Tensor.fromTypedArray({
      data: new BigInt64Array([-1n, 2n]),
      shape: [2],
      dtype: "int64",
      device: "cpu",
    });
    const powFloat = pow(bigA, bigNegExp);
    expect(powFloat.dtype).toBe("float64");

    const bigExp = Tensor.fromTypedArray({
      data: new BigInt64Array([2n, 1n]),
      shape: [2],
      dtype: "int64",
      device: "cpu",
    });
    const powInt = pow(bigA, bigExp);
    expect(powInt.dtype).toBe("int64");

    const divOut = div(bigA, bigExp);
    expect(divOut.dtype).toBe("float64");

    const negBig = Tensor.fromTypedArray({
      data: new BigInt64Array([-3n, 3n]),
      shape: [2],
      dtype: "int64",
      device: "cpu",
    });
    const twoBig = Tensor.fromTypedArray({
      data: new BigInt64Array([2n, 2n]),
      shape: [2],
      dtype: "int64",
      device: "cpu",
    });
    const floored = floorDiv(negBig, twoBig).toArray() as bigint[];
    expect(floored[0]).toBe(-2n);
    expect(floored[1]).toBe(1n);

    const modded = mod(negBig, twoBig).toArray() as bigint[];
    expect(modded[0]).toBe(1n);

    const emptyInt = Tensor.fromTypedArray({
      data: new Int32Array(0),
      shape: [0],
      dtype: "int32",
      device: "cpu",
    });
    const emptyPow = pow(emptyInt, emptyInt);
    expect(emptyPow.shape).toEqual([0]);
  });

  it("covers arithmetic scalar/broadcast and clip branches", () => {
    const a = tensor([1, 2, 3]);
    const b = tensor(2);
    expect(add(a, b).shape).toEqual([3]);

    const clippedLow = clip(a, 2, undefined);
    const clippedHigh = clip(a, undefined, 2);
    expect(clippedLow.shape).toEqual([3]);
    expect(clippedHigh.shape).toEqual([3]);
    expect(() => clip(a, 3, 2)).toThrow(/min.*max/i);

    const bigA = Tensor.fromTypedArray({
      data: new BigInt64Array([2n, -1n, 4n]),
      shape: [3],
      dtype: "int64",
      device: "cpu",
    });
    const bigB = Tensor.fromTypedArray({
      data: new BigInt64Array([1n, 2n, 3n]),
      shape: [3],
      dtype: "int64",
      device: "cpu",
    });
    expect(maximum(bigA, bigB).shape).toEqual([3]);
    expect(minimum(bigA, bigB).shape).toEqual([3]);

    const signed = sign(bigA).toArray() as bigint[];
    expect(signed).toEqual([1n, -1n, 1n]);

    const rec = reciprocal(bigA);
    expect(rec.dtype).toBe("float64");
  });

  it("covers comparison paths including mixed BigInt and tolerance helpers", () => {
    const bigA = Tensor.fromTypedArray({
      data: new BigInt64Array([1n, 2n]),
      shape: [2],
      dtype: "int64",
      device: "cpu",
    });
    const bigB = Tensor.fromTypedArray({
      data: new BigInt64Array([1n, 3n]),
      shape: [2],
      dtype: "int64",
      device: "cpu",
    });
    expect(equal(bigA, bigB).toArray()).toEqual([1, 0]);
    expect(notEqual(bigA, bigB).toArray()).toEqual([0, 1]);
    expect(greater(bigB, bigA).toArray()).toEqual([0, 1]);
    expect(lessEqual(bigA, bigB).toArray()).toEqual([1, 1]);

    const numA = tensor([1, 2]);
    expect(equal(bigA, numA).shape).toEqual([2]);
    expect(greaterEqual(numA, tensor(1)).shape).toEqual([2]);
    expect(less(numA, tensor(3)).shape).toEqual([2]);

    expect(isclose(tensor([1.0, 1.001]), tensor([1.0, 1.0]), 1e-2, 1e-6).shape).toEqual([2]);
    expect(() => isclose(bigA, bigB)).toThrow(/BigInt/);

    expect(allclose(tensor([1, 2]), tensor([1, 2]))).toBe(true);
    expect(allclose(tensor([1, 2]), tensor([1, 2, 3]))).toBe(false);

    expect(arrayEqual(tensor([1, 2]), tensor([1, 2, 3]))).toBe(false);
    expect(arrayEqual(tensor([1, 2]), tensor([1, 3]))).toBe(false);

    const withNan = tensor([Number.NaN, Number.POSITIVE_INFINITY, 1]);
    expect(isnan(withNan).toArray()).toEqual([1, 0, 0]);
    expect(isinf(withNan).toArray()).toEqual([0, 1, 0]);
    expect(isfinite(withNan).toArray()).toEqual([0, 0, 1]);
  });

  it("covers reduction axis and ddof branches", () => {
    const big = Tensor.fromTypedArray({
      data: new BigInt64Array([1n, 2n, 3n, 4n]),
      shape: [2, 2],
      dtype: "int64",
      device: "cpu",
    });
    const s = sum(big, 0, true);
    expect(s.shape).toEqual([1, 2]);

    expect(() => sum(tensor([[1]]), 3)).toThrow(/axis/);

    const empty = reshape(tensor([]), [0, 2]);
    const sumEmpty = sum(empty, 0);
    expect(sumEmpty.shape).toEqual([2]);

    expect(() =>
      prod(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        [0, 0] as unknown as number[]
      )
    ).toThrow(/duplicate axis/);

    expect(() => variance(tensor([1, 2, 3]), 0, false, -1)).toThrow(/ddof/);
    expect(() => variance(tensor([[1, 2]]), 0, false, 2)).toThrow(/ddof/);
    expect(() => variance(empty, 0)).toThrow(/at least one element/);
  });
});
