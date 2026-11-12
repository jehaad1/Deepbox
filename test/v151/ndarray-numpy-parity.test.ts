import { describe, expect, it } from "vitest";
import { InvalidParameterError, ShapeError } from "../../src/core";
import {
  all,
  any,
  arange,
  diff,
  expandDims,
  intersect1d,
  max,
  min,
  prod,
  repeat,
  rot90,
  setdiff1d,
  sum,
  tensor,
  union1d,
} from "../../src/ndarray";

// Expected values come from numpy 2.x (np.rot90, np.union1d, np.intersect1d, np.setdiff1d,
// np.expand_dims, np.sum on a 0-d array, np.repeat with an array, np.diff on bool).

const cube = () => arange(12, undefined, 1, { dtype: "float64" }).reshape([2, 3, 2]);

describe("rot90 with axes", () => {
  it("rotates in the plane of the given axes", () => {
    expect(rot90(cube(), 1, [1, 2]).toArray()).toEqual([
      [
        [1, 3, 5],
        [0, 2, 4],
      ],
      [
        [7, 9, 11],
        [6, 8, 10],
      ],
    ]);
    expect(rot90(cube(), 3, [0, 2]).toArray()).toEqual([
      [
        [6, 0],
        [8, 2],
        [10, 4],
      ],
      [
        [7, 1],
        [9, 3],
        [11, 5],
      ],
    ]);
    expect(rot90(cube(), 2, [2, 0]).toArray()).toEqual([
      [
        [7, 6],
        [9, 8],
        [11, 10],
      ],
      [
        [1, 0],
        [3, 2],
        [5, 4],
      ],
    ]);
    expect(rot90(cube(), -1, [-1, 0]).toArray()).toEqual([
      [
        [1, 7],
        [3, 9],
        [5, 11],
      ],
      [
        [0, 6],
        [2, 8],
        [4, 10],
      ],
    ]);
  });

  it("defaults to the first two axes", () => {
    expect(rot90(cube()).toArray()).toEqual([
      [
        [4, 5],
        [10, 11],
      ],
      [
        [2, 3],
        [8, 9],
      ],
      [
        [0, 1],
        [6, 7],
      ],
    ]);
  });

  it("rejects equal or out-of-range axes and 1-D input", () => {
    expect(() => rot90(cube(), 1, [0, 0])).toThrow(InvalidParameterError);
    expect(() => rot90(cube(), 1, [0, 3])).toThrow(InvalidParameterError);
    expect(() => rot90(arange(3))).toThrow(ShapeError);
  });
});

describe("set operations flatten N-D input", () => {
  it("matches numpy", () => {
    const a = tensor(
      [
        [3, 1],
        [2, 1],
      ],
      { dtype: "float64" }
    );
    expect(union1d(a, tensor([5, 1], { dtype: "float64" })).toArray()).toEqual([1, 2, 3, 5]);
    expect(intersect1d(a, tensor([[1, 2]], { dtype: "float64" })).toArray()).toEqual([1, 2]);
    expect(setdiff1d(a, tensor([2], { dtype: "float64" })).toArray()).toEqual([1, 3]);
  });
});

describe("expandDims with several axes", () => {
  it("inserts every axis (numpy.expand_dims(x, (0, 3)).shape == (1, 2, 3, 1))", () => {
    const x = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(expandDims(x, [0, 3]).shape).toEqual([1, 2, 3, 1]);
    expect(expandDims(x, [-1, 0]).shape).toEqual([1, 2, 3, 1]);
    expect(expandDims(x, [1, 2]).shape).toEqual([2, 1, 1, 3]);
    expect(expandDims(x, 0).shape).toEqual([1, 2, 3]);
  });

  it("rejects a repeated axis", () => {
    expect(() => expandDims(tensor([1, 2]), [1, 1])).toThrow(InvalidParameterError);
    expect(() => expandDims(tensor([1, 2]), [0, 3])).toThrow(InvalidParameterError);
  });
});

describe("axis 0 and -1 on 0-d reductions", () => {
  it("is accepted by sum, prod, max, min, all and any like numpy", () => {
    const x = tensor(1.5, { dtype: "float64" });
    expect(sum(x, 0).item()).toBe(1.5);
    expect(sum(x, -1).item()).toBe(1.5);
    expect(prod(x, 0).item()).toBe(1.5);
    expect(max(x, -1).item()).toBe(1.5);
    expect(min(x, 0).item()).toBe(1.5);
    expect(all(x, 0).item()).toBe(1);
    expect(any(x, -1).item()).toBe(1);
    expect(sum(x, 0, true).shape).toEqual([]);
  });

  it("still rejects other axes", () => {
    expect(() => sum(tensor(1.5), 1)).toThrow(InvalidParameterError);
  });
});

describe("repeat with a tensor of counts", () => {
  it("repeats each element by its count", () => {
    expect(repeat(tensor([1, 2, 3]), tensor([1, 0, 2], { dtype: "int32" })).toArray()).toEqual([
      1, 3, 3,
    ]);
    expect(
      repeat(
        tensor([
          [1, 2],
          [3, 4],
        ]),
        tensor([1, 2], { dtype: "int32" }),
        0
      ).toArray()
    ).toEqual([
      [1, 2],
      [3, 4],
      [3, 4],
    ]);
  });

  it("rejects a tensor of counts with more than one dimension", () => {
    expect(() => repeat(tensor([1, 2]), tensor([[1, 2]], { dtype: "int32" }))).toThrow(
      InvalidParameterError
    );
  });
});

describe("diff on bool", () => {
  it("uses XOR and keeps bool, like numpy.diff", () => {
    const b = tensor([true, false, false, true]);
    const d = diff(b);
    expect(d.dtype).toBe("bool");
    expect(d.toArray()).toEqual([1, 0, 1]);
    expect(diff(b, 2).toArray()).toEqual([1, 1]);
    expect(
      diff(
        tensor([
          [true, false],
          [false, false],
        ]),
        1,
        0
      ).toArray()
    ).toEqual([[1, 0]]);
  });
});
