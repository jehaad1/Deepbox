import { describe, expect, it } from "vitest";
import { DTypeError } from "../../src/core";
import { add, addScalar, full, mulScalar, tensor } from "../../src/ndarray";

// Reference: numpy wraps out-of-range values and has undefined casts for NaN/Infinity. This
// library throws a DTypeError for every integer dtype, as it already did for int64.

describe("float to narrow integer conversion", () => {
  it("tensor() rejects NaN, Infinity and out-of-range values", () => {
    expect(() => tensor([Number.NaN], { dtype: "int32" })).toThrow(DTypeError);
    expect(() => tensor([Number.POSITIVE_INFINITY], { dtype: "int32" })).toThrow(DTypeError);
    expect(() => tensor([3e9], { dtype: "int32" })).toThrow(DTypeError);
    expect(() => tensor([-2147483649], { dtype: "int32" })).toThrow(DTypeError);
    expect(() => tensor([256], { dtype: "uint8" })).toThrow(DTypeError);
    expect(() => tensor([-1], { dtype: "uint8" })).toThrow(DTypeError);
    expect(() => tensor([Number.NaN], { dtype: "uint8" })).toThrow(DTypeError);
  });

  it("tensor() keeps truncating in-range fractions", () => {
    expect(tensor([1.9, -1.9], { dtype: "int32" }).toArray()).toEqual([1, -1]);
    expect(tensor([255.9, -0.5], { dtype: "uint8" }).toArray()).toEqual([255, 0]);
    expect(tensor([2147483647, -2147483648], { dtype: "int32" }).toArray()).toEqual([
      2147483647, -2147483648,
    ]);
  });

  it("full() rejects NaN and out-of-range values", () => {
    expect(() => full([2], Number.NaN, { dtype: "int32" })).toThrow(DTypeError);
    expect(() => full([2], 3e9, { dtype: "int32" })).toThrow(DTypeError);
    expect(() => full([2], 300, { dtype: "uint8" })).toThrow(DTypeError);
    expect(full([2], 7.8, { dtype: "int32" }).toArray()).toEqual([7, 7]);
  });

  it("astype() rejects NaN, Infinity and out-of-range values for every integer dtype", () => {
    const nan = tensor([Number.NaN], { dtype: "float64" });
    const big = tensor([3e9], { dtype: "float64" });
    expect(() => nan.astype("int32")).toThrow(DTypeError);
    expect(() => nan.astype("uint8")).toThrow(DTypeError);
    expect(() => nan.astype("int64")).toThrow(DTypeError);
    expect(() => big.astype("int32")).toThrow(DTypeError);
    expect(() => big.astype("uint8")).toThrow(DTypeError);
    expect(() => tensor([Number.NEGATIVE_INFINITY]).astype("int32")).toThrow(DTypeError);
  });

  it("astype() checks narrowing between integer dtypes", () => {
    expect(() => tensor([300], { dtype: "int32" }).astype("uint8")).toThrow(DTypeError);
    expect(() => tensor([-1], { dtype: "int32" }).astype("uint8")).toThrow(DTypeError);
    expect(() => tensor([2147483648], { dtype: "int64" }).astype("int32")).toThrow(DTypeError);
    expect(tensor([200], { dtype: "int32" }).astype("uint8").toArray()).toEqual([200]);
    expect(tensor([-5], { dtype: "int64" }).astype("int32").toArray()).toEqual([-5]);
  });

  it("astype() keeps in-range truncation and the bool rule (NumPy: nonzero, NaN included, is true)", () => {
    expect(tensor([1.7, -2.3], { dtype: "float64" }).astype("int32").toArray()).toEqual([1, -2]);
    expect(tensor([0, 2, Number.NaN], { dtype: "float64" }).astype("bool").toArray()).toEqual([
      0, 1, 1,
    ]);
  });

  it("astype() checks non-contiguous views", () => {
    const x = tensor(
      [
        [1, 3e9],
        [2, 4],
      ],
      { dtype: "float64" }
    );
    expect(() => x.astype("int32")).toThrow(DTypeError);
    expect(x.slice({ start: 0, end: 2 }, 0).astype("int32").toArray()).toEqual([1, 2]);
  });
});

describe("string conversion", () => {
  it("parses decimal integers only", () => {
    expect(tensor(["7", " -12 ", "+3"]).astype("int32").toArray()).toEqual([7, -12, 3]);
    expect(tensor(["9007199254740993"]).astype("int64").toArray()).toEqual([9007199254740993n]);
    for (const dtype of ["int32", "uint8", "int64"] as const) {
      expect(() => tensor(["abc"]).astype(dtype)).toThrow(DTypeError);
      expect(() => tensor([""]).astype(dtype)).toThrow(DTypeError);
      expect(() => tensor(["0x10"]).astype(dtype)).toThrow(DTypeError);
      expect(() => tensor(["1.5"]).astype(dtype)).toThrow(DTypeError);
    }
    expect(() => tensor(["3000000000"]).astype("int32")).toThrow(DTypeError);
    expect(() => tensor(["-1"]).astype("uint8")).toThrow(DTypeError);
  });

  it("converts to bool from true/false (any case) and 1/0 only", () => {
    expect(tensor(["true", "FALSE", "True", "1", "0"]).astype("bool").toArray()).toEqual([
      1, 0, 1, 1, 0,
    ]);
    expect(() => tensor(["x"]).astype("bool")).toThrow(DTypeError);
    expect(() => tensor([""]).astype("bool")).toThrow(DTypeError);
    expect(() => tensor(["2"]).astype("bool")).toThrow(DTypeError);
  });

  it("still parses floats with Number()", () => {
    expect(tensor(["1.5", "1e3"]).astype("float64").toArray()).toEqual([1.5, 1000]);
  });
});

describe("integer scalar arithmetic", () => {
  it("wraps the sum like add() and rejects scalars outside the dtype", () => {
    const x = tensor([2147483647], { dtype: "int32" });
    // add(int32, int32) wraps on overflow, as numpy does; addScalar follows it.
    const viaAdd = add(x, tensor([1], { dtype: "int32" })).toArray();
    expect(addScalar(x, 1).toArray()).toEqual(viaAdd);
    expect(() => addScalar(tensor([5], { dtype: "int32" }), 3e9)).toThrow(DTypeError);
    expect(() => mulScalar(tensor([5], { dtype: "int32" }), 3e9)).toThrow(DTypeError);
    expect(() => addScalar(tensor([5], { dtype: "uint8" }), 300)).toThrow(DTypeError);
    expect(() => addScalar(tensor([5], { dtype: "uint8" }), -1)).toThrow(DTypeError);
    expect(addScalar(tensor([255], { dtype: "uint8" }), 1).toArray()).toEqual([0]);
  });
});
